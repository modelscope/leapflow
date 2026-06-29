"""Signal buffer — accumulates interaction signals between frame stores.

Signals are lightweight temporal anchors (click coords, app switches, clipboard
events) that help the VLM pipeline disambiguate what happened between frames.
The buffer is drained each time a keyframe is stored, attaching accumulated
signals to that frame.
"""

from __future__ import annotations

import logging
import threading
from typing import List

from leapflow.perception.types import InteractionSignal

logger = logging.getLogger(__name__)

_MAX_SIGNALS_PER_INTERVAL = 50


class SignalBuffer:
    """Thread-safe bounded buffer for interaction signals between polling frames."""

    __slots__ = ("_signals", "_lock", "_capacity")

    def __init__(self, capacity: int = _MAX_SIGNALS_PER_INTERVAL) -> None:
        self._signals: List[InteractionSignal] = []
        self._lock = threading.Lock()
        self._capacity = capacity

    def record(self, signal: InteractionSignal) -> None:
        """Append a signal to the buffer (dropped if at capacity)."""
        with self._lock:
            if len(self._signals) < self._capacity:
                self._signals.append(signal)

    def drain(self) -> List[InteractionSignal]:
        """Return all buffered signals and clear the buffer."""
        with self._lock:
            result = self._signals
            self._signals = []
            return result

    def clear(self) -> None:
        """Discard all buffered signals."""
        with self._lock:
            self._signals.clear()

    @property
    def count(self) -> int:
        """Current number of buffered signals (non-locking snapshot)."""
        return len(self._signals)
