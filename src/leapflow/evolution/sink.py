"""LedgerEvolutionSink: accept traces on the hot side, persist on the cold side.

The probe's contract is "accept and return", so ``record`` only appends to a bounded
deque. Persistence happens when someone calls :meth:`flush` -- the daemon's monitor
cycle, or process exit -- which keeps a file write out of the plugin registry's
version bump and the trust ledger's level transition.

The buffer is bounded and drops *oldest* on overflow. That is the right direction
for this data: a burst means the framework is churning, and the newest traces
describe where it ended up.
"""

from __future__ import annotations

import atexit
import logging
from collections import deque
from typing import Any

from leapflow.domain.evolution_trace import EvolutionTrace

logger = logging.getLogger(__name__)

#: Bounded so a runaway producer cannot grow memory between flushes.
DEFAULT_BUFFER_SIZE = 512


class LedgerEvolutionSink:
    """Buffering :class:`~leapflow.domain.evolution_trace.EvolutionTraceSink`.

    Satisfies the Protocol structurally; no inheritance, so a test can substitute
    anything with a ``record`` method.
    """

    def __init__(
        self,
        *,
        store: Any = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        publish: Any = None,
    ) -> None:
        self._store = store
        self._buffer: deque[EvolutionTrace] = deque(maxlen=max(1, int(buffer_size)))
        # Optional callable invoked per trace after buffering, for event
        # re-publication. Kept as a plain callable so this module needs no
        # dependency on the event bus.
        self._publish = publish
        self._dropped = 0
        self._recorded = 0

    # ── sink side (must stay O(1) and never raise) ─────────────────────────

    def record(self, trace: EvolutionTrace) -> None:
        """Buffer one trace. Called from probe sites, so it does no I/O."""
        if len(self._buffer) == self._buffer.maxlen:
            # Counted rather than silently discarded: a non-zero drop count means
            # the panel is showing an incomplete history, which a reader must be
            # able to find out.
            self._dropped += 1
        self._buffer.append(trace)
        self._recorded += 1
        if self._publish is not None:
            try:
                self._publish(trace)
            except Exception:  # noqa: BLE001 - publication is best-effort
                logger.debug("evolution sink: publish failed", exc_info=True)

    # ── cold side ─────────────────────────────────────────────────────────

    def flush(self) -> int:
        """Persist and clear the buffer. Returns the number of traces written.

        Drains before writing so a store failure cannot cause the same traces to
        be retried forever; they are lost, and ``dropped`` records that they were.
        """
        if not self._buffer:
            return 0
        pending = list(self._buffer)
        self._buffer.clear()
        if self._store is None:
            return 0
        try:
            return int(self._store.append(trace.to_dict() for trace in pending))
        except Exception:  # noqa: BLE001 - a lost trace must not break the cycle
            logger.debug("evolution sink: flush failed", exc_info=True)
            self._dropped += len(pending)
            return 0

    def pending(self) -> tuple[EvolutionTrace, ...]:
        """Buffered traces not yet flushed, for a reader that wants live state."""
        return tuple(self._buffer)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "recorded": self._recorded,
            "buffered": len(self._buffer),
            "dropped": self._dropped,
        }

    def register_atexit(self) -> None:
        """Flush on interpreter exit, so a clean shutdown loses nothing."""
        atexit.register(self._flush_quietly)

    def _flush_quietly(self) -> None:
        try:
            self.flush()
        except Exception:  # noqa: BLE001 - exit-time best effort
            logger.debug("evolution sink: exit flush failed", exc_info=True)


__all__ = ["DEFAULT_BUFFER_SIZE", "LedgerEvolutionSink"]
