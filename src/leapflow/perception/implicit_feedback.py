"""Implicit feedback signal detection — identifies user struggle signals.

Detects patterns that indicate the user is "stuck" or struggling:
- Prolonged inactivity (no input events for extended period)
- Rapid undo/redo sequences (repeated Cmd+Z / Ctrl+Z)
- Frequent application switching (context fragmentation)
- Repeated failed attempts (same action retried)

These signals trigger proactive Copilot assistance via EventBus.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from leapflow.domain.event_types import (
    ImplicitFeedbackType,
    NormalizedEventType,
    UIActionSubType,
    UNDO_SHORTCUTS,
)

if TYPE_CHECKING:
    from leapflow.platform.event_bus import EventBus

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ImplicitFeedbackEvent:
    """An implicit feedback signal detected from user behavior."""

    signal_type: str  # "inactivity" | "undo_storm" | "app_thrashing" | "retry_failure"
    confidence: float  # 0.0-1.0 how confident we are this is a real struggle
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ImplicitFeedbackConfig:
    """Configuration for implicit feedback detection thresholds."""

    # Inactivity detection
    inactivity_threshold_s: float = 120.0  # 2min of silence = potential stuck
    inactivity_check_interval_s: float = 30.0

    # Undo storm detection
    undo_window_s: float = 10.0  # Time window to count undos
    undo_threshold: int = 4  # Undos in window to trigger

    # App thrashing detection
    app_switch_window_s: float = 30.0  # Time window
    app_switch_threshold: int = 6  # Switches in window to trigger

    # Retry detection
    retry_window_s: float = 60.0
    retry_threshold: int = 3  # Same action repeated N times


class ImplicitFeedbackObserver:
    """Detects implicit feedback signals from user behavior patterns.

    Subscribes to EventBus events and analyzes temporal patterns
    to identify when the user might be struggling.

    Implements Observer protocol (start/stop/running).
    """

    def __init__(
        self,
        bus: "EventBus",
        config: Optional[ImplicitFeedbackConfig] = None,
    ) -> None:
        self._bus = bus
        self._config = config or ImplicitFeedbackConfig()
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Event tracking buffers
        self._last_event_ts: float = time.time()
        self._undo_timestamps: List[float] = []
        self._app_switch_timestamps: List[float] = []
        self._recent_actions: List[tuple] = []  # (ts, action_type)

    async def start(self) -> None:
        """Start monitoring for implicit feedback signals."""
        if self._running:
            return
        self._running = True
        self._last_event_ts = time.time()
        # Subscribe to EventBus for relevant events
        self._bus.subscribe(self._on_event)
        # Start inactivity monitor
        self._monitor_task = asyncio.create_task(self._inactivity_monitor())
        logger.info("ImplicitFeedbackObserver started")

    async def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("ImplicitFeedbackObserver stopped")

    @property
    def running(self) -> bool:
        return self._running

    def _on_event(self, event: Any) -> None:
        """Process incoming SystemEvent to detect patterns."""
        if not self._running:
            return
        now = time.time()
        self._last_event_ts = now

        event_type = getattr(event, "event_type", "")
        payload = getattr(event, "payload", {})

        if self._is_undo_event(event_type, payload):
            self._undo_timestamps.append(now)
            self._check_undo_storm(now)

        if event_type == NormalizedEventType.APP_FOCUS_CHANGE:
            self._app_switch_timestamps.append(now)
            self._check_app_thrashing(now)

        action_key = self._build_action_key(event_type, payload)
        self._recent_actions.append((now, action_key))
        self._check_retry_pattern(now)

    def _is_undo_event(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """Detect undo/redo operations from normalized ui.action events."""
        if event_type != NormalizedEventType.UI_ACTION:
            return False
        sub_type = payload.get("sub_type", "")
        if sub_type != UIActionSubType.SHORTCUT:
            return False
        modifiers = payload.get("modifiers", [])
        key_code = payload.get("key_code", 0)
        char = payload.get("char", "").lower()
        if char == "z" and any(m in modifiers for m in ("cmd", "meta", "ctrl", "control")):
            return True
        if char == "y" and any(m in modifiers for m in ("ctrl", "control")):
            return True
        shortcut = payload.get("shortcut", "").lower()
        return shortcut in UNDO_SHORTCUTS

    @staticmethod
    def _build_action_key(event_type: str, payload: Dict[str, Any]) -> str:
        """Build a stable key for retry detection from normalized events."""
        if event_type == NormalizedEventType.UI_ACTION:
            sub_type = payload.get("sub_type", "")
            label = payload.get("label", "")
            node_id = payload.get("node_id", "")
            return f"{event_type}:{sub_type}:{label or node_id}"
        return f"{event_type}:{payload.get('source', '')}"

    def _check_undo_storm(self, now: float) -> None:
        """Check if undo frequency exceeds threshold."""
        window = self._config.undo_window_s
        self._undo_timestamps = [t for t in self._undo_timestamps if now - t <= window]

        if len(self._undo_timestamps) >= self._config.undo_threshold:
            self._emit_signal(ImplicitFeedbackEvent(
                signal_type="undo_storm",
                confidence=min(0.9, 0.5 + 0.1 * len(self._undo_timestamps)),
                context={"undo_count": len(self._undo_timestamps), "window_s": window},
            ))
            self._undo_timestamps.clear()  # Reset to avoid repeated triggers

    def _check_app_thrashing(self, now: float) -> None:
        """Check if app switching frequency indicates context fragmentation."""
        window = self._config.app_switch_window_s
        self._app_switch_timestamps = [
            t for t in self._app_switch_timestamps if now - t <= window
        ]

        if len(self._app_switch_timestamps) >= self._config.app_switch_threshold:
            self._emit_signal(ImplicitFeedbackEvent(
                signal_type="app_thrashing",
                confidence=min(0.8, 0.4 + 0.1 * len(self._app_switch_timestamps)),
                context={"switch_count": len(self._app_switch_timestamps), "window_s": window},
            ))
            self._app_switch_timestamps.clear()

    def _check_retry_pattern(self, now: float) -> None:
        """Check if the same action is being retried repeatedly."""
        window = self._config.retry_window_s
        self._recent_actions = [(t, a) for t, a in self._recent_actions if now - t <= window]

        if len(self._recent_actions) < self._config.retry_threshold:
            return

        # Count action frequencies
        action_counts = Counter(a for _, a in self._recent_actions)
        for action, count in action_counts.most_common(1):
            if count >= self._config.retry_threshold:
                self._emit_signal(ImplicitFeedbackEvent(
                    signal_type="retry_failure",
                    confidence=min(0.85, 0.5 + 0.1 * count),
                    context={"action": action, "retry_count": count},
                ))
                # Remove these to avoid re-triggering
                self._recent_actions = [
                    (t, a) for t, a in self._recent_actions if a != action
                ]
                break

    async def _inactivity_monitor(self) -> None:
        """Periodic check for prolonged inactivity."""
        while self._running:
            try:
                await asyncio.sleep(self._config.inactivity_check_interval_s)
                if not self._running:
                    break

                elapsed = time.time() - self._last_event_ts
                if elapsed >= self._config.inactivity_threshold_s:
                    confidence = min(
                        0.7,
                        0.3 + 0.1 * (elapsed / self._config.inactivity_threshold_s),
                    )
                    self._emit_signal(ImplicitFeedbackEvent(
                        signal_type="inactivity",
                        confidence=confidence,
                        context={"idle_seconds": round(elapsed, 1)},
                    ))
                    # Extend threshold to avoid repeated triggers
                    self._last_event_ts = time.time()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("inactivity_monitor iteration failed", exc_info=True)

    def _emit_signal(self, event: ImplicitFeedbackEvent) -> None:
        """Emit an implicit feedback event via EventBus."""
        logger.info(
            "ImplicitFeedback: %s (confidence=%.2f) %s",
            event.signal_type, event.confidence, event.context,
        )
        event_type = f"{ImplicitFeedbackType.PREFIX}.{event.signal_type}"
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._bus.handle_event(
                    event_type,
                    {
                        "confidence": event.confidence,
                        "signal_type": event.signal_type,
                        **event.context,
                    },
                )
            )
        except RuntimeError:
            pass
        except Exception:
            logger.debug("Failed to emit implicit feedback event", exc_info=True)
