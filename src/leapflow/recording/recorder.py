"""Demonstration recorder — converts EventBus stream into trajectories.

Plugs into the existing EventBus.subscribe() mechanism as a zero-intrusion
observer.  Records normalized SystemEvents as TrajectorySteps and persists
them to TrajectoryStore.

State machine: IDLE → RECORDING ⇄ PAUSED → IDLE
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING, Dict, Optional, Sequence

from leapflow.platform.client import fire_and_forget
from leapflow.recording.attention import (
    AttentionFilter,
    FilterResult,
    FilterVerdict,
    RecordingContext,
    SurpriseAnnotator,
)
from leapflow.storage.trajectory_store import TrajectoryStore
from leapflow.domain.trajectory import (
    ActionType,
    RawAction,
    RecordingMode,
    RecordingState,
    SnapshotLevel,
    StateSnapshot,
    Trajectory,
    TrajectoryStep,
    action_type_from_event,
)
from leapflow.platform.protocol import HostRpc
from leapflow.domain.events import SystemEvent

if TYPE_CHECKING:
    from leapflow.perception.session import PerceptionSession

logger = logging.getLogger(__name__)

_CLIPBOARD_TRUNCATE_DEFAULT = 1024
_TEXT_CAPTURE_MAX_LENGTH_DEFAULT = 500


class DemonstrationRecorder:
    """EventBus subscriber that builds trajectories from SystemEvents.

    Usage::

        recorder = DemonstrationRecorder(store)
        event_bus.subscribe(recorder.on_event)  # zero-intrusion hookup
        recorder.start()                        # begin recording
        ...
        trajectory = recorder.stop()            # finalize and persist
    """

    def __init__(
        self,
        store: TrajectoryStore,
        *,
        rpc: Optional[HostRpc] = None,
        perception_session: Optional["PerceptionSession"] = None,
        user_id: str = "default",
        auto_flush_interval: int = 10,
        attention_filters: "Sequence[AttentionFilter] | None" = None,
        surprise_annotator: Optional[SurpriseAnnotator] = None,
        text_capture_enabled: bool = True,
        text_capture_exclude_apps: Sequence[str] = (),
        text_capture_secure_roles: Sequence[str] = ("AXSecureTextField",),
        text_capture_max_length: int = _TEXT_CAPTURE_MAX_LENGTH_DEFAULT,
        clipboard_max_length: int = _CLIPBOARD_TRUNCATE_DEFAULT,
        recording_mode: RecordingMode = RecordingMode.DEFAULT,
    ) -> None:
        self._store = store
        self._rpc = rpc
        self._perception_session = perception_session
        self._user_id = user_id
        self._flush_interval = auto_flush_interval

        # Perception depth controls
        self._text_capture_enabled = text_capture_enabled
        self._text_capture_exclude_apps = frozenset(text_capture_exclude_apps)
        self._text_capture_secure_roles = frozenset(text_capture_secure_roles)
        self._text_capture_max_length = text_capture_max_length
        self._clipboard_max_length = clipboard_max_length

        self._recording_mode = recording_mode

        self._state = RecordingState.IDLE
        self._trajectory: Optional[Trajectory] = None
        self._last_focused_app: str = ""
        self._last_clipboard: str = ""
        self._unflushed: int = 0
        self._last_ax_tree: Optional[dict] = None
        self._last_ax_tree_digest: str = ""

        self._control_input_active: bool = False
        self._self_host_app: str = ""
        self._visual_degraded: bool = False

        self._attention_filters: list[AttentionFilter] = list(attention_filters) if attention_filters else []
        self._surprise_annotator = surprise_annotator
        self._attention_context = RecordingContext()

    @property
    def state(self) -> RecordingState:
        return self._state

    @property
    def current_trajectory(self) -> Optional[Trajectory]:
        return self._trajectory

    @property
    def attention_context(self) -> RecordingContext:
        return self._attention_context

    @property
    def recording_mode(self) -> RecordingMode:
        return self._recording_mode

    @property
    def visual_degraded(self) -> bool:
        return self._visual_degraded

    def set_visual_degraded(self, degraded: bool) -> None:
        """Enable/disable visual degradation fallback.

        When degraded=True and the recording mode normally skips structural
        events (e.g. vision_only), the recorder resumes structural event
        recording as a fallback so the trajectory isn't empty.
        """
        if degraded != self._visual_degraded:
            self._visual_degraded = degraded
            label = "degraded" if degraded else "restored"
            logger.warning("Visual channel %s — structural events %s",
                           label, "enabled" if degraded else "per recording mode")

    @property
    def event_stats(self) -> Dict[str, int]:
        """Count of recorded raw actions, keyed by action_type.value."""
        if self._trajectory is None:
            return {}
        counts: Dict[str, int] = {}
        for step in self._trajectory.steps:
            key = step.action.action_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    # ── State machine transitions ──

    def start(self, *, user_id: Optional[str] = None) -> str:
        """Begin a new recording session.  Returns the trajectory ID."""
        if self._state == RecordingState.RECORDING:
            assert self._trajectory is not None
            return self._trajectory.trajectory_id

        traj = Trajectory(
            user_id=user_id or self._user_id,
            start_time=time.time(),
        )
        self._trajectory = traj
        self._state = RecordingState.RECORDING
        self._unflushed = 0
        self._self_host_app = self._last_focused_app
        self._attention_context.reset()
        self._attention_context.self_host_app = self._self_host_app
        if self._perception_session:
            fire_and_forget(self._perception_session.start_session(traj.trajectory_id))
        logger.info("recording.started id=%s", traj.trajectory_id)
        return traj.trajectory_id

    def resume_from(self, trajectory: Trajectory) -> str:
        """Resume recording on an existing trajectory. Returns trajectory ID.

        New events are appended starting from len(trajectory.steps).
        """
        if self._state == RecordingState.RECORDING:
            assert self._trajectory is not None
            return self._trajectory.trajectory_id

        self._trajectory = trajectory
        self._state = RecordingState.RECORDING
        self._unflushed = 0
        self._self_host_app = self._last_focused_app
        if self._perception_session:
            fire_and_forget(self._perception_session.start_session(trajectory.trajectory_id))
        logger.info(
            "recording.resumed_from id=%s existing_steps=%d",
            trajectory.trajectory_id, trajectory.step_count,
        )
        return trajectory.trajectory_id

    def pause(self) -> None:
        """Temporarily pause recording (events are ignored until resume)."""
        if self._state != RecordingState.RECORDING:
            return
        self._state = RecordingState.PAUSED
        self._flush()
        logger.info("recording.paused")

    def resume(self) -> None:
        """Resume a paused recording."""
        if self._state != RecordingState.PAUSED:
            return
        self._state = RecordingState.RECORDING
        logger.info("recording.resumed")

    def stop(self) -> Optional[Trajectory]:
        """Stop recording, finalize and persist the trajectory."""
        if self._state == RecordingState.IDLE:
            return None

        traj = self._trajectory
        if traj is None:
            self._state = RecordingState.IDLE
            return None

        self._control_input_active = False
        self._trim_trailing_control_events()
        traj.end_time = time.time()
        self._flush()
        if self._perception_session:
            fire_and_forget(self._perception_session.end_session())
        self._store.finalize_trajectory(traj)
        self._state = RecordingState.IDLE

        logger.info(
            "recording.stopped id=%s steps=%d duration=%.1fs",
            traj.trajectory_id,
            traj.step_count,
            traj.duration,
        )
        result = traj
        self._trajectory = None
        return result

    # ── EventBus callback ──

    def mark_skip(self, n: int = 1) -> int:
        """Mark the last *n* steps as user-skipped noise. Returns count marked."""
        if not self._trajectory or n <= 0:
            return 0
        marked = 0
        for step in self._trajectory.steps[-n:]:
            noise = step.action.params.setdefault("_noise", [])
            if not any(s.get("signal_type") == "user_skip" for s in noise):
                noise.append({
                    "signal_type": "user_skip",
                    "confidence": 1.0,
                    "related_step": -1,
                })
                marked += 1
        return marked

    # ── Control input suppression ──

    def mark_control_input_start(self) -> None:
        """Mark the start of a control input window.

        Events received while active are silently discarded — they represent
        user interaction with the recording tool's own CLI, not the demonstration.
        """
        self._control_input_active = True

    def end_control_input(self) -> None:
        """Mark the end of a control input window."""
        self._control_input_active = False

    def on_event(self, event: SystemEvent) -> None:
        """EventBus subscriber callback — the only public ingest point."""
        self._update_context(event)
        if self._recording_mode.skip_structural_events and not self._visual_degraded:
            return
        if self._state != RecordingState.RECORDING:
            return
        if self._control_input_active:
            return

        for filt in self._attention_filters:
            result = filt.evaluate(event, self._attention_context)
            if result.verdict == FilterVerdict.REJECT:
                return
            if result.verdict == FilterVerdict.ANNOTATE_NOISE:
                self._record_event(event, attention_noise=result)
                return

        self._record_event(event)

    # ── Internal recording logic ──

    def _record_event(
        self, event: SystemEvent, *, attention_noise: Optional[FilterResult] = None,
    ) -> None:
        traj = self._trajectory
        if traj is None:
            return

        action = self._event_to_action(event)
        if action.action_type == ActionType.UNKNOWN:
            return

        state = self._capture_state(event)

        # Perceptual field: strip sensitive content based on perception level
        if attention_noise and "perception_field:" in (attention_noise.reason or ""):
            if "perception_field:opaque" in attention_noise.reason:
                action = self._strip_to_skeleton(action)
            elif "perception_field:structural" in attention_noise.reason:
                action = self._strip_text_content(action)

        # Detect noise signals
        noise_signals = self._detect_noise_signals(event, action)
        if noise_signals:
            action.params["_noise"] = noise_signals

        # Attach attention filter noise annotation
        if attention_noise is not None:
            noise_list = action.params.setdefault("_noise", [])
            noise_list.append({
                "signal_type": "attention_filter",
                "confidence": attention_noise.confidence,
                "related_step": -1,
                "reason": attention_noise.reason,
            })

        # Surprise annotation (post-filter, never rejects)
        if self._surprise_annotator is not None:
            surprise = self._surprise_annotator.annotate(event, self._attention_context)
            if surprise is not None:
                noise_list = action.params.setdefault("_noise", [])
                noise_list.append(surprise)

        step = TrajectoryStep(state=state, action=action)
        traj.steps.append(step)

        self._unflushed += 1
        if self._unflushed >= self._flush_interval:
            self._flush()

    def _event_to_action(self, event: SystemEvent) -> RawAction:
        """Map a SystemEvent to a RawAction."""
        sub_action = str(event.payload.get("action", ""))
        action_type = action_type_from_event(event.event_type, sub_action)

        target = ""
        target_label = ""
        target_role = ""
        app_bundle_id = ""
        app_name = ""
        params = dict(event.payload)

        if event.event_type == "fs.change":
            target = str(event.payload.get("path", event.source))

        elif event.event_type == "clipboard.change":
            text = str(event.payload.get("text", ""))
            params = {"text_length": len(text)}
            if len(text) <= self._clipboard_max_length:
                params["text_preview"] = text
            else:
                params["text_preview"] = text[:self._clipboard_max_length]

        elif event.event_type == "app.focus_change":
            app_bundle_id = str(event.payload.get("bundle_id", event.source))
            app_name = str(event.payload.get("app_name", ""))
            params = {}
            window_title = event.payload.get("window_title")
            if window_title:
                params["window_title"] = str(window_title)

        elif event.event_type == "chat.interaction":
            target = str(event.payload.get("tool_name", ""))
            target_label = str(event.payload.get("content", ""))[:200]
            target_role = sub_action
            app_bundle_id = "leapflow.engine"
            app_name = "LeapFlow"
            params = {
                k: v for k, v in event.payload.items()
                if k not in ("action",) and isinstance(v, (str, int, float, bool, list))
            }

        elif event.event_type == "ui.action":
            target = str(event.payload.get("node_id", ""))
            target_label = str(event.payload.get("label", ""))
            target_role = str(event.payload.get("role", ""))
            app_bundle_id = str(event.payload.get("app_bundle_id", ""))
            sub = str(event.payload.get("sub_type", ""))
            if sub:
                action_type = action_type_from_event("ui.action", sub)
            params = {}
            if sub == "click":
                params["mouse_x"] = event.payload.get("mouse_x", 0)
                params["mouse_y"] = event.payload.get("mouse_y", 0)
            elif sub in ("type", "shortcut"):
                params["key_code"] = event.payload.get("key_code", 0)
                params["modifiers"] = event.payload.get("modifiers", [])
                char = event.payload.get("char")
                if char and self._should_capture_text(app_bundle_id or self._last_focused_app, event):
                    params["char"] = str(char)[:self._text_capture_max_length]
            elif sub == "scroll":
                params["delta_x"] = event.payload.get("delta_x", 0)
                params["delta_y"] = event.payload.get("delta_y", 0)
                params["mouse_x"] = event.payload.get("mouse_x", 0)
                params["mouse_y"] = event.payload.get("mouse_y", 0)
            elif sub == "drag":
                params["start_x"] = event.payload.get("start_x", 0)
                params["start_y"] = event.payload.get("start_y", 0)
                params["end_x"] = event.payload.get("end_x", 0)
                params["end_y"] = event.payload.get("end_y", 0)
                params["end_role"] = event.payload.get("end_role", "")
                params["end_label"] = event.payload.get("end_label", "")
                params["cross_app"] = event.payload.get("cross_app", False)
                params["start_app"] = event.payload.get("start_app", "")
                params["end_app"] = event.payload.get("end_app", "")

        if not app_bundle_id:
            app_bundle_id = self._last_focused_app

        return RawAction(
            timestamp=event.timestamp,
            action_type=action_type,
            target=target,
            target_label=target_label,
            target_role=target_role,
            params=params,
            app_bundle_id=app_bundle_id,
            app_name=app_name,
        )

    _TEXT_FIELDS = frozenset({"char", "text_preview", "text", "window_title", "text_content"})

    def _strip_text_content(self, action: RawAction) -> RawAction:
        """Strip text content from an action, preserving only structural information."""
        stripped_params = {k: v for k, v in action.params.items() if k not in self._TEXT_FIELDS}
        return RawAction(
            timestamp=action.timestamp,
            action_type=action.action_type,
            target=action.target,
            target_label="",
            target_role=action.target_role,
            params=stripped_params,
            app_bundle_id=action.app_bundle_id,
            app_name=action.app_name,
        )

    def _strip_to_skeleton(self, action: RawAction) -> RawAction:
        """Strip action to skeleton: only timestamp + app (for OPAQUE level)."""
        return RawAction(
            timestamp=action.timestamp,
            action_type=action.action_type,
            target="",
            target_label="",
            target_role="",
            params={},
            app_bundle_id=action.app_bundle_id,
            app_name=action.app_name,
        )

    def _capture_state(self, event: SystemEvent) -> StateSnapshot:
        """Build a state snapshot using event-driven level strategy.

        Snapshot level is determined by event importance:
        - FULL: app focus changes (complete AX tree + clipboard)
        - LIGHT: UI actions (focused element + digest)
        - MINIMAL: file/clipboard events (basic app info only)
        """
        level = self._snapshot_level_for_event(event)

        if level == SnapshotLevel.FULL:
            snapshot = self._capture_full(event)
        elif level == SnapshotLevel.LIGHT:
            snapshot = self._capture_light(event)
        else:
            snapshot = self._capture_minimal(event)

        return snapshot

    # ── Snapshot level strategy ──

    def _snapshot_level_for_event(self, event: SystemEvent) -> SnapshotLevel:
        """Determine snapshot fidelity based on event type."""
        if event.event_type in ("app.focus_change",):
            return SnapshotLevel.FULL
        elif event.event_type in ("ui.action",):
            return SnapshotLevel.LIGHT
        return SnapshotLevel.MINIMAL

    def _capture_full(self, event: SystemEvent) -> StateSnapshot:
        """FULL snapshot: complete AX tree + clipboard + visual frame."""
        clipboard = self._last_clipboard
        if event.event_type == "clipboard.change":
            clipboard = str(event.payload.get("text", ""))[:self._clipboard_max_length]

        # Fetch AX tree for current focused app (sync from cache or event payload)
        current_tree = self._resolve_ax_tree(event)
        tree_snapshot = self._compute_tree_diff(current_tree, self._last_ax_tree)
        if current_tree is not None:
            self._last_ax_tree = current_tree

        return StateSnapshot(
            timestamp=event.timestamp,
            focused_app=self._last_focused_app,
            ax_tree_digest=self._compute_digest(event),
            ax_focused_element=self._extract_focused_element(event),
            clipboard_text=clipboard if clipboard else None,
            ax_tree_snapshot=tree_snapshot,
            snapshot_level=SnapshotLevel.FULL.value,
        )

    def _capture_light(self, event: SystemEvent) -> StateSnapshot:
        """LIGHT snapshot: focused element + AX digest (no full tree)."""
        clipboard = self._last_clipboard
        if event.event_type == "clipboard.change":
            clipboard = str(event.payload.get("text", ""))[:self._clipboard_max_length]

        return StateSnapshot(
            timestamp=event.timestamp,
            focused_app=self._last_focused_app,
            ax_tree_digest=self._compute_digest(event),
            ax_focused_element=self._extract_focused_element(event),
            clipboard_text=clipboard if clipboard else None,
            snapshot_level=SnapshotLevel.LIGHT.value,
        )

    def _capture_minimal(self, event: SystemEvent) -> StateSnapshot:
        """MINIMAL snapshot: only focused app info."""
        return StateSnapshot(
            timestamp=event.timestamp,
            focused_app=self._last_focused_app,
            ax_tree_digest=self._compute_digest(event),
            snapshot_level=SnapshotLevel.MINIMAL.value,
        )

    # ── AX tree helpers ──

    def _resolve_ax_tree(self, event: SystemEvent) -> Optional[dict]:
        """Resolve current AX tree from event payload or cached state.

        In production, the AX tree is pre-fetched via ``refresh_ax_tree()``
        and cached. The event payload may also carry inline tree data from
        the host.
        """
        # Prefer inline tree from event payload (pushed by platform)
        if "ax_tree" in event.payload:
            return dict(event.payload["ax_tree"])
        return self._last_ax_tree

    def _compute_tree_diff(self, current_tree: Optional[dict], last_tree: Optional[dict]) -> Optional[dict]:
        """Return the full tree if structure changed, else None (saves storage).

        Comparison uses a content digest so that structurally identical trees
        are detected without deep comparison.
        """
        if current_tree is None:
            return None
        current_digest = self._tree_digest(current_tree)
        if last_tree is not None and current_digest == self._last_ax_tree_digest:
            return None  # No change — skip storing duplicate
        self._last_ax_tree_digest = current_digest
        return current_tree

    @staticmethod
    def _tree_digest(tree: dict) -> str:
        """Compute a stable content digest for an AX tree dict."""
        raw = json.dumps(tree, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode(), usedforsecurity=False).hexdigest()[:16]

    @staticmethod
    def _extract_focused_element(event: SystemEvent) -> Optional[dict]:
        """Extract the focused UI element from event payload, if present."""
        focused = event.payload.get("focused_element")
        if isinstance(focused, dict):
            return focused
        # Fallback: construct from UI action metadata
        if event.event_type == "ui.action":
            node_id = event.payload.get("node_id")
            if node_id:
                return {
                    "node_id": str(node_id),
                    "role": str(event.payload.get("role", "")),
                    "label": str(event.payload.get("label", "")),
                }
        return None

    async def refresh_ax_tree(self, app_id: Optional[str] = None) -> Optional[dict]:
        """Async: fetch and cache the full AX tree via RPC.

        Call this from an async context (e.g. event loop) to pre-populate
        the AX tree cache before the next FULL snapshot is captured.
        Returns the fetched tree or None if RPC is unavailable.
        """
        if self._rpc is None:
            return None
        target = app_id or self._last_focused_app
        if not target:
            return None
        try:
            result = await self._rpc.call("ax.tree", {"app_id": target})
            if isinstance(result, dict):
                self._last_ax_tree = result
                self._last_ax_tree_digest = self._tree_digest(result)
                return result
        except Exception:
            logger.debug("ax.tree RPC failed for %s", target, exc_info=True)
        return None

    def _should_capture_text(self, app_bundle_id: str, event: SystemEvent) -> bool:
        """Determine whether keystroke text should be recorded for this event."""
        if not self._text_capture_enabled:
            return False
        if app_bundle_id and app_bundle_id in self._text_capture_exclude_apps:
            return False
        role = str(event.payload.get("role", ""))
        if role and role in self._text_capture_secure_roles:
            return False
        return True

    def retract_context(self, app_bundle_id: str, context_pattern: str) -> int:
        """Mark steps matching an app+context as retracted (noise annotation).

        Does not physically delete steps — preserves step indices and audit trail.
        Matches context from: params.window_title, perception_field noise reason,
        or state.focused_app (fallback).
        Returns the number of steps marked.
        """
        from fnmatch import fnmatch

        if not self._trajectory:
            return 0

        retracted = 0
        for step in self._trajectory.steps:
            if step.action.app_bundle_id != app_bundle_id:
                continue
            if self._step_matches_context(step, context_pattern):
                noise = step.action.params.setdefault("_noise", [])
                if not any(n.get("signal_type") == "user_retract" for n in noise):
                    noise.append({
                        "signal_type": "user_retract",
                        "confidence": 1.0,
                        "related_step": -1,
                    })
                    retracted += 1
        return retracted

    @staticmethod
    def _step_matches_context(step: "TrajectoryStep", context_pattern: str) -> bool:
        """Check if a step's context matches the pattern.

        Looks at multiple sources since STRUCTURAL/OPAQUE stripping removes
        window_title from params. Falls back to the perception_field reason
        annotation which preserves the context value.
        """
        from fnmatch import fnmatch

        wt = step.action.params.get("window_title", "")
        if wt and fnmatch(wt.lower(), context_pattern.lower()):
            return True

        for noise_entry in step.action.params.get("_noise", []):
            reason = noise_entry.get("reason", "")
            if "perception_field:" in reason:
                ctx_value = reason.split(":", 2)[-1] if reason.count(":") >= 2 else ""
                if ctx_value and fnmatch(ctx_value.lower(), context_pattern.lower()):
                    return True

        fallback = step.state.focused_app
        if fallback and fnmatch(fallback.lower(), context_pattern.lower()):
            return True

        return False

    def _update_context(self, event: SystemEvent) -> None:
        """Maintain running context for state snapshots and attention filters."""
        if event.event_type == "app.focus_change":
            bundle_id = str(event.payload.get("bundle_id", event.source))
            self._last_focused_app = bundle_id
            self._attention_context.update_focus(bundle_id)
            window_title = event.payload.get("window_title")
            if window_title:
                self._attention_context.last_window_title = str(window_title)
        elif event.event_type == "ui.action":
            window_title = event.payload.get("window_title")
            if window_title:
                self._attention_context.last_window_title = str(window_title)
        elif event.event_type == "clipboard.change":
            self._last_clipboard = str(event.payload.get("text", ""))[:self._clipboard_max_length]
        elif event.event_type == "fs.change":
            path = str(event.payload.get("path", event.source))
            self._attention_context.observe_fs_path(path)

    def _flush(self) -> None:
        """Persist buffered steps to the store."""
        traj = self._trajectory
        if traj is None or self._unflushed == 0:
            return
        start = traj.step_count - self._unflushed
        for i in range(start, traj.step_count):
            self._store.append_step(traj.trajectory_id, i, traj.steps[i])
        self._unflushed = 0

    def _trim_trailing_control_events(self) -> None:
        """Remove trailing UI input events from the self-host app (safety net).

        Catches residual control-command keystrokes that may have been recorded
        before the proactive suppression flag took effect.
        """
        traj = self._trajectory
        if not traj or not self._self_host_app:
            return
        _INPUT_TYPES = (ActionType.UI_TYPE, ActionType.UI_SHORTCUT)
        steps = traj.steps
        trimmed = 0
        while steps:
            action = steps[-1].action
            if (action.app_bundle_id == self._self_host_app
                    and action.action_type in _INPUT_TYPES):
                steps.pop()
                trimmed += 1
            else:
                break
        if trimmed:
            self._unflushed = max(0, self._unflushed - trimmed)
            logger.debug("recording.trim_control_tail trimmed=%d", trimmed)

    def _detect_noise_signals(self, event: SystemEvent, action: RawAction) -> list:
        """Detect noise signals for the current event. Returns list of NoiseSignal dicts."""
        signals = []

        # 1. Undo detection: Cmd+Z (key_code=6, modifiers=["command"])
        if (event.event_type == "ui.action"
            and event.payload.get("sub_type") == "shortcut"
            and event.payload.get("key_code") == 6
            and "command" in (event.payload.get("modifiers") or [])):
            if "shift" not in (event.payload.get("modifiers") or []):
                signals.append({"signal_type": "undo", "confidence": 0.95, "related_step": self._current_step_idx() - 1})
            else:
                signals.append({"signal_type": "redo", "confidence": 0.95, "related_step": -1})

        # 2. Rapid switch: app.focus_change back to previous app within 3s
        if event.event_type == "app.focus_change":
            if self._trajectory and len(self._trajectory.steps) >= 2:
                recent_steps = self._trajectory.steps[-3:]
                for step in recent_steps:
                    if (step.action.action_type == ActionType.APP_SWITCH
                        and step.action.app_bundle_id == event.payload.get("bundle_id", "")
                        and (event.timestamp - step.action.timestamp) < 3.0):
                        signals.append({"signal_type": "rapid_switch", "confidence": 0.8, "related_step": len(self._trajectory.steps) - 1})
                        break

        # 3. Repeated: consecutive identical action_type + target
        if self._trajectory and self._trajectory.steps:
            last_action = self._trajectory.steps[-1].action
            if (last_action.action_type == action.action_type
                and last_action.target == action.target
                and action.target):
                signals.append({"signal_type": "repeated", "confidence": 0.7, "related_step": len(self._trajectory.steps) - 1})

        # 4. Idle scroll: consecutive scroll events (check via a counter)
        if action.action_type == ActionType.UI_SCROLL:
            consecutive_scrolls = self._count_trailing_scrolls()
            if consecutive_scrolls >= 3:
                signals.append({"signal_type": "idle_scroll", "confidence": 0.6, "related_step": -1})

        return signals

    def _current_step_idx(self) -> int:
        """Get the current step index."""
        if self._trajectory:
            return len(self._trajectory.steps)
        return 0

    def _count_trailing_scrolls(self) -> int:
        """Count consecutive scroll events at the end of trajectory."""
        if not self._trajectory:
            return 0
        count = 0
        for step in reversed(self._trajectory.steps[-10:]):
            if step.action.action_type == ActionType.UI_SCROLL:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _compute_digest(event: SystemEvent) -> str:
        """Content-based digest for UI state change detection.

        Hashes the payload content (not the timestamp) so that identical
        UI states produce identical digests, enabling change detection.
        """
        payload_key = json.dumps(event.payload, sort_keys=True, ensure_ascii=False)
        key = f"{event.event_type}:{event.source}:{payload_key}"
        return hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:12]
