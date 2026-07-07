"""Async event reorder buffer for correcting cross-source arrival inversions.

Events produced by different observer threads (CGEvent tap, app focus,
FS watcher) may arrive at the Python side out of causal order.  This buffer
holds incoming events for a brief settle window, then flushes them sorted by their
monotonic origin timestamp — ensuring downstream consumers see events in true
temporal order regardless of dispatch latency differences.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

EmitFn = Callable[[str, Dict[str, Any]], Awaitable[None]]


class EventReorderBuffer:
    """Buffers events for a settle window then emits them in timestamp order.

    Design:
    - First event in a batch starts a timer (settle window).
    - Subsequent events within the window accumulate without restarting the timer.
    - When the timer fires, all buffered events are flushed sorted by mono_ts.
    - Events without a mono_ts are assigned arrival time (preserving arrival order
      relative to each other, but placed after all timestamped events in the window).
    """

    def __init__(self, settle_s: float, emit: EmitFn) -> None:
        self._settle_s = settle_s
        self._emit = emit
        self._buffer: List[Tuple[float, str, Dict[str, Any]]] = []
        self._flush_task: Optional[asyncio.Task[None]] = None
        self._arrival_counter: int = 0

    async def submit(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Submit an event into the reorder buffer."""
        ts = payload.get("_mono_ts")
        if ts is None:
            self._arrival_counter += 1
            ts = float("inf") - 1.0 / self._arrival_counter
        self._buffer.append((ts, event_type, payload))
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._delayed_flush())

    async def drain(self) -> None:
        """Flush all buffered events immediately (used on recording stop)."""
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._flush()

    @property
    def pending_count(self) -> int:
        return len(self._buffer)

    async def _delayed_flush(self) -> None:
        await asyncio.sleep(self._settle_s)
        await self._flush()

    async def _flush(self) -> None:
        if not self._buffer:
            return
        batch = sorted(
            self._buffer,
            key=lambda x: (x[0], -(x[2].get("priority", 3))),
        )
        self._buffer.clear()
        for _, event_type, payload in batch:
            await self._emit(event_type, payload)
