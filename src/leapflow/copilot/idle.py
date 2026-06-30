"""Idle (pause) detection for the Workflow Copilot.

Identifies natural user pauses between operations and triggers predictive
suggestion logic when the pause exceeds an adaptive threshold.

SRP: Only detects idle state — no prediction, no rendering.
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from collections import deque
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from leapflow.copilot.config import CopilotConfig

logger = logging.getLogger(__name__)


class IdleDetector:
    """Idle gap detector — infers user pause state from event timestamp stream.

    Design principles:
    - Adaptive threshold: dynamically adjusts pause detection based on user tempo
    - Zero-delay notification: triggers callback immediately once threshold is met
    - Cancellation: new events instantly cancel the current idle state
    """

    # EMA window for adaptive threshold
    _EMA_WINDOW: int = 20

    def __init__(
        self,
        config: CopilotConfig,
        on_idle: Callable[[int], Awaitable[None]],
    ) -> None:
        """Initialise idle detector.

        Args:
            config: Copilot configuration (min_idle_ms, max_idle_ms).
            on_idle: Async callback invoked with idle duration in ms when
                     the user pauses long enough.
        """
        self._config = config
        self._on_idle = on_idle

        # Timing state
        self._last_event_ts: float = 0.0
        self._idle_since: Optional[float] = None
        self._idle_notified: bool = False

        # Pending async check task
        self._pending_task: Optional[asyncio.Task[None]] = None

        # Adaptive threshold — EMA of recent inter-event intervals
        self._interval_history: deque[float] = deque(maxlen=self._EMA_WINDOW)
        self._ema_interval_ms: float = float(config.min_idle_ms)

    # ── Public API ────────────────────────────────────────────────────────

    def on_event_timestamp(self, ts: float) -> None:
        """Record a new event timestamp and reset idle tracking.

        Called on every incoming signal/event. Cancels any pending idle
        callback and resets the idle state.
        """
        if self._last_event_ts > 0.0:
            interval_ms = (ts - self._last_event_ts) * 1000
            if interval_ms > 0:
                self._interval_history.append(interval_ms)
                self._update_ema(interval_ms)

        self._last_event_ts = ts
        self._idle_since = None
        self._idle_notified = False

        # Cancel any pending idle check
        self._cancel_pending()

        # Schedule a new idle check
        self._schedule_idle_check()

    @property
    def is_idle(self) -> bool:
        """Whether the user is currently in an idle (paused) state."""
        if self._last_event_ts <= 0.0:
            return False
        elapsed_ms = (_time.time() - self._last_event_ts) * 1000
        return elapsed_ms >= self._effective_threshold_ms

    @property
    def idle_duration_ms(self) -> int:
        """Current idle duration in milliseconds. 0 if not idle."""
        if not self.is_idle:
            return 0
        return int((_time.time() - self._last_event_ts) * 1000)

    @property
    def effective_threshold_ms(self) -> float:
        """Current adaptive idle threshold (for observability)."""
        return self._effective_threshold_ms

    # ── Internal ──────────────────────────────────────────────────────────

    @property
    def _effective_threshold_ms(self) -> float:
        """Adaptive threshold: clamp EMA-based value within config bounds."""
        # Use 1.5x the EMA interval as threshold, clamped to config range
        adaptive = self._ema_interval_ms * 1.5
        return max(
            float(self._config.min_idle_ms),
            min(adaptive, float(self._config.max_idle_ms)),
        )

    def _update_ema(self, interval_ms: float) -> None:
        """Update EMA of inter-event intervals."""
        alpha = 2.0 / (len(self._interval_history) + 1)
        self._ema_interval_ms = alpha * interval_ms + (1 - alpha) * self._ema_interval_ms

    def _cancel_pending(self) -> None:
        """Cancel pending idle check task if any."""
        if self._pending_task is not None and not self._pending_task.done():
            self._pending_task.cancel()
            self._pending_task = None

    def _schedule_idle_check(self) -> None:
        """Schedule an async idle check after the effective threshold."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop — skip scheduling (e.g. sync tests)
            return

        self._pending_task = loop.create_task(self._idle_check_coro())

    async def _idle_check_coro(self) -> None:
        """Wait for threshold duration then check if still idle."""
        threshold_s = self._effective_threshold_ms / 1000.0
        try:
            await asyncio.sleep(threshold_s)
        except asyncio.CancelledError:
            return

        # After sleeping, verify we are still idle (no new events arrived)
        if self._idle_notified:
            return

        elapsed_ms = int((_time.time() - self._last_event_ts) * 1000)
        if elapsed_ms < self._config.min_idle_ms:
            return
        if elapsed_ms > self._config.max_idle_ms:
            # User likely left — do not notify
            return

        self._idle_notified = True
        self._idle_since = self._last_event_ts

        try:
            await self._on_idle(elapsed_ms)
        except Exception:
            logger.exception("Error in on_idle callback")
