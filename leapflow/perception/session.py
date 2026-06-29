"""Perception Session — lifecycle management for visual perception.

Session-scoped: created at learn-session start, collects interaction signals
during recording, then runs offline extraction at session end.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from leapflow.perception.config import PerceptionConfig
from leapflow.perception.extraction.pipeline import OfflineExtractionPipeline
from leapflow.perception.signals import SignalBuffer
from leapflow.perception.storage.frame_store import FrameStore, LocalFrameStore
from leapflow.perception.types import (
    ChannelStatus,
    InteractionSignal,
    Keyframe,
    VisualAction,
)

from leapflow.domain.trajectory import RecordingMode

if TYPE_CHECKING:
    from leapflow.causal import CausalGraph
    from leapflow.domain.events import SystemEvent
    from leapflow.llm.base import LLMProvider
    from leapflow.platform.protocol import HostRpc
    from leapflow.recording.attention import RecordingContext
    from leapflow.signal_fusion.cross_app import CrossAppContextTracker
    from leapflow.signal_fusion.types import AppTransitionEvent

logger = logging.getLogger(__name__)


class PerceptionSession:
    """Manages the full lifecycle of visual perception for one learn session.

    Phases:
    1. Start: initializes frame store and signal buffers
    2. Online: receives events, extracts signals, feeds causal pipeline
    3. End: runs offline extraction pipeline on captured keyframes

    Integration pattern:
    - Subscribe ``on_system_event`` to the EventBus for automatic event handling.
    - Lifecycle is managed by the DemonstrationRecorder (start_session/end_session)
      or directly by the caller.
    - After stop(), call extract() to run the offline VLM extraction pipeline.
    """

    def __init__(
        self,
        config: PerceptionConfig,
        rpc: "HostRpc",
        vlm: Optional["LLMProvider"] = None,
        frame_store: Optional[FrameStore] = None,
        recording_context: Optional["RecordingContext"] = None,
    ) -> None:
        self._config = config
        self._rpc = rpc
        self._vlm = vlm
        self._recording_context = recording_context

        self._frame_store = frame_store or LocalFrameStore(config.frame_cache_dir)

        self._session_id: Optional[str] = None
        self._active = False
        self._current_app: str = ""
        self._keyframes: List[Keyframe] = []
        self._extracted_actions: List[VisualAction] = []
        self._start_time: float = 0.0
        self._channel_status = ChannelStatus()
        self._cross_app_tracker: Optional["CrossAppContextTracker"] = None
        self._app_transitions: List["AppTransitionEvent"] = []
        self._health_probe = ChannelHealthProbe()
        self._last_ui_event_time: float = 0.0
        self._last_focus_event_time: float = 0.0
        self._recording_mode = RecordingMode.DEFAULT

        self._signal_buffer = SignalBuffer()
        self._signal_channels = config.signal_channels

        from leapflow.causal import CausalFusionPipeline, CausalGraph, build_default_registry
        causal_registry = build_default_registry()
        if config.signal_channels:
            always_available = frozenset({"visual_change"})
            for ch in causal_registry.channels:
                causal_registry.set_available(
                    ch, ch in config.signal_channels or ch in always_available
                )
        vlm_verifier = None
        try:
            from leapflow.config import get_settings
            _s = get_settings()
            if _s.causal_tier3_enabled:
                from leapflow.causal.inference import VLMVerifier
                vlm_verifier = VLMVerifier(
                    confidence_threshold=_s.causal_tier3_confidence_threshold,
                )
        except Exception:
            pass
        self._causal_pipeline = CausalFusionPipeline(
            registry=causal_registry,
            vlm_verifier=vlm_verifier,
        )
        self._causal_graph = CausalGraph()

    @property
    def active(self) -> bool:
        return self._active

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def frame_count(self) -> int:
        return len(self._keyframes)

    @property
    def extracted_actions(self) -> List[VisualAction]:
        """Actions extracted during the last offline extraction pass."""
        return self._extracted_actions

    @property
    def channel_status(self) -> ChannelStatus:
        return self._channel_status

    @property
    def app_transitions(self) -> List["AppTransitionEvent"]:
        return list(self._app_transitions)

    @property
    def keyframes(self) -> List[Keyframe]:
        return list(self._keyframes)

    @property
    def causal_graph(self) -> "CausalGraph":
        """The session's accumulated CausalGraph (causal propagation chain)."""
        return self._causal_graph

    @property
    def causal_pipeline(self) -> "CausalFusionPipeline":
        """Read-only access to the causal fusion pipeline."""
        return self._causal_pipeline

    def set_cross_app_tracker(self, tracker: "CrossAppContextTracker") -> None:
        """Inject cross-app tracker for app transition awareness."""
        self._cross_app_tracker = tracker

    def set_recording_mode(self, mode: RecordingMode) -> None:
        """Set the recording mode, enabling mode-specific behaviors."""
        self._recording_mode = mode

    # ── Lifecycle ──

    async def start(self, session_id: str) -> None:
        """Begin a new perception session."""
        self._session_id = session_id
        self._active = True
        self._start_time = time.time()
        self._keyframes.clear()
        self._extracted_actions.clear()
        self._app_transitions.clear()
        self._last_ui_event_time = 0.0
        self._last_focus_event_time = 0.0
        self._signal_buffer.clear()
        from leapflow.causal import CausalGraph
        self._causal_graph = CausalGraph()
        logger.info("Perception session started: %s (mode=%s)", session_id, self._recording_mode.value)

    async def stop(self) -> List[Keyframe]:
        """Stop the session and return captured keyframes."""
        self._active = False
        logger.info(
            "Perception session stopped: %s (%d frames captured)",
            self._session_id, len(self._keyframes),
        )
        return self._keyframes

    async def extract(
        self,
        *,
        progress: "Optional[Callable[[str, int, int], None]]" = None,
    ) -> List[VisualAction]:
        """Run offline extraction pipeline on captured keyframes.

        Should be called after stop(). Returns extracted visual actions.
        Stores results in ``self.extracted_actions`` for downstream consumption.
        """
        if not self._keyframes:
            return []

        pipeline = OfflineExtractionPipeline(vlm=self._vlm, config=self._config)
        self._extracted_actions = await pipeline.extract(
            self._keyframes, progress=progress,
        )
        return self._extracted_actions

    # ── Online Phase: Event Handling ──

    def on_system_event(self, event: "SystemEvent") -> None:
        """EventBus subscriber callback — matches ``EventCallback`` signature.

        Accepts the normalized SystemEvent from the EventBus and dispatches
        asynchronous event processing. This is the primary integration point
        for wiring perception into the platform event flow.
        """
        if not self._active or not self._session_id:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._handle_event(event.event_type, event.payload))
        except RuntimeError:
            pass

    async def on_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Direct async event handler (alternative to EventBus subscription).

        Can be called directly when the caller has event_type/payload rather
        than a full SystemEvent. Also used internally by on_system_event.
        """
        await self._handle_event(event_type, payload)

    def _is_relevant_event(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """Lightweight attention gate: skip events from unfocused/system apps.

        Shares the recorder's RecordingContext to align perception with
        structural recording scope. Always allows focus_change events through
        (needed to track app transitions).
        """
        if self._recording_context is None:
            return True
        if event_type == "app.focus_change":
            return True

        app = str(payload.get("app_bundle_id", "")) or str(payload.get("bundle_id", ""))
        if not app:
            app = self._current_app

        if not app:
            return True

        if self._recording_context.focused_apps and app not in self._recording_context.focused_apps:
            return False

        if ".inputmethod." in app:
            return False

        return True

    async def _handle_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Core event processing: extract signals and feed the causal pipeline."""
        if not self._active or not self._session_id:
            return

        now = time.time()

        if event_type.startswith("ui."):
            self._last_ui_event_time = now
        elif event_type == "app.focus_change":
            self._last_focus_event_time = now

        prev_app = self._current_app
        if event_type == "app.focus_change":
            self._current_app = payload.get("bundle_id", "")

        if not self._is_relevant_event(event_type, payload):
            return

        signal = self._extract_signal(event_type, payload, prev_app, now)
        if signal:
            self._signal_buffer.record(signal)
            self._causal_pipeline.fuse(signals=[signal], graph=self._causal_graph)

    def update_channel_status(self, status: ChannelStatus) -> None:
        """Update perception channel availability."""
        self._channel_status = status

    # ── Lifecycle helpers for recorder integration ──

    async def start_session(self, trajectory_id: str) -> None:
        """Start a perception session (recorder lifecycle integration)."""
        await self.start(trajectory_id)

    async def end_session(self) -> None:
        """End perception session (recorder lifecycle integration)."""
        await self.stop()

    async def probe_channel_health(self) -> ChannelStatus:
        """Proactively probe channel health and update status."""
        now = time.time()
        status = await self._health_probe.probe(
            self._rpc,
            last_ui_event_time=self._last_ui_event_time,
            last_focus_event_time=self._last_focus_event_time,
            now=now,
        )
        self.update_channel_status(status)
        return status

    def get_timed_visual_actions(self) -> List[tuple]:
        """Return extracted visual actions with their timestamp ranges.

        Each entry is (start_ts, end_ts, VisualAction). Timestamps are derived
        from the keyframes referenced by each action's frame_ref_a/b fields.
        """
        if not self._extracted_actions or not self._keyframes:
            return []

        ref_to_ts: Dict[str, float] = {kf.ref: kf.timestamp for kf in self._keyframes}
        result: List[tuple] = []
        for va in self._extracted_actions:
            ts_a = ref_to_ts.get(va.frame_ref_a, 0.0)
            ts_b = ref_to_ts.get(va.frame_ref_b, ts_a)
            if ts_a > 0:
                result.append((ts_a, ts_b, va))
        return result

    # ── Signal Fusion ──

    def _extract_signal(
        self, event_type: str, payload: Dict[str, Any], prev_app: str, now: float,
    ) -> Optional[InteractionSignal]:
        """Extract an InteractionSignal if the event's channel is enabled.

        Privacy-aware: app_switch signals are always allowed (no sensitive data),
        but position/content signals are suppressed for privacy-sensitive apps.
        """
        if not self._signal_channels:
            return None
        if not self._recording_mode.needs_visual_polling:
            return None

        if event_type == "app.focus_change" and "app_switch" in self._signal_channels:
            new_app = payload.get("bundle_id", "")
            return InteractionSignal(
                timestamp=now,
                signal_type="app_switch",
                app=new_app,
                detail=f"{prev_app} -> {new_app}",
            )

        if self._current_app in self._config.privacy_sensitive_apps:
            return None

        if event_type == "ui.action":
            sub = payload.get("sub_type", "")

            if sub == "click" and "click" in self._signal_channels:
                return InteractionSignal(
                    timestamp=now,
                    signal_type="click",
                    app=payload.get("app_bundle_id", "") or self._current_app,
                    position=(
                        int(payload.get("mouse_x", 0)),
                        int(payload.get("mouse_y", 0)),
                    ),
                )

            if sub == "scroll" and "scroll" in self._signal_channels:
                return InteractionSignal(
                    timestamp=now,
                    signal_type="scroll",
                    app=payload.get("app_bundle_id", "") or self._current_app,
                    position=(
                        int(payload.get("mouse_x", 0)),
                        int(payload.get("mouse_y", 0)),
                    ),
                    detail=f"dy={payload.get('delta_y', 0)}",
                )

            if sub == "shortcut" and "keyboard" in self._signal_channels:
                modifiers = payload.get("modifiers", [])
                char = payload.get("char", "")
                combo = "+".join(modifiers + ([char] if char else []))
                return InteractionSignal(
                    timestamp=now,
                    signal_type="keyboard",
                    app=payload.get("app_bundle_id", "") or self._current_app,
                    detail=combo,
                )

            if sub == "type" and "keyboard" in self._signal_channels:
                text = str(payload.get("text", ""))[:50]
                return InteractionSignal(
                    timestamp=now,
                    signal_type="keyboard",
                    app=payload.get("app_bundle_id", "") or self._current_app,
                    detail=f"type:{text}",
                )

            if sub == "drag" and "drag" in self._signal_channels:
                return InteractionSignal(
                    timestamp=now,
                    signal_type="drag",
                    app=payload.get("app_bundle_id", "") or self._current_app,
                    position=(
                        int(payload.get("start_x", 0)),
                        int(payload.get("start_y", 0)),
                    ),
                    end_position=(
                        int(payload.get("end_x", 0)),
                        int(payload.get("end_y", 0)),
                    ),
                )

        if event_type == "clipboard.change":
            if "clipboard_content" in self._signal_channels:
                text = str(payload.get("text", ""))[:200]
                return InteractionSignal(
                    timestamp=now,
                    signal_type="clipboard",
                    detail=f"content:{text}",
                )
            elif "clipboard" in self._signal_channels:
                return InteractionSignal(
                    timestamp=now,
                    signal_type="clipboard",
                    detail=payload.get("change_type", "change"),
                )

        return None

    @staticmethod
    def _extract_cursor(payload: Dict[str, Any]) -> Optional[tuple]:
        """Extract cursor position from event payload if available."""
        pos = payload.get("position")
        if pos and isinstance(pos, dict):
            return (int(pos.get("x", 0)), int(pos.get("y", 0)))
        return None


class ChannelHealthProbe:
    """Proactive channel health detection based on recent event timing."""

    def __init__(self, stale_threshold_s: float = 30.0) -> None:
        self._stale_threshold = stale_threshold_s

    async def probe(
        self,
        rpc: "HostRpc",
        *,
        last_ui_event_time: float = 0.0,
        last_focus_event_time: float = 0.0,
        now: float = 0.0,
    ) -> ChannelStatus:
        now = now or time.time()
        status = ChannelStatus()

        status.ui_events_available = (
            last_ui_event_time > 0
            and (now - last_ui_event_time) < self._stale_threshold
        )

        status.app_focus_available = (
            last_focus_event_time > 0
            and (now - last_focus_event_time) < self._stale_threshold
        )

        return status
