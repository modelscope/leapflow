"""Turn admission control for bounded concurrent execution (Stage 3, P3-4).

``TurnAdmission`` bounds how many agent turns run concurrently (up to N) while
still allowing exclusive maintenance operations (host restart, re-entry
dispatch) to run with no turn in flight. It is a readers/writer discipline built
on a semaphore:

* ``turn_slot()`` acquires one of N slots — up to N turns run concurrently.
* ``exclusive()`` drains all N slots, so it runs alone and blocks new turns until
  it finishes; concurrent exclusive ops are serialized (no drain deadlock).

``N = 1`` reduces to a plain mutex — exactly the daemon's pre-P3-4 behavior.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, List


class TurnAdmission:
    """Bounded concurrent turns with exclusive maintenance windows."""

    def __init__(self, max_concurrent_turns: int) -> None:
        self._n = max(1, int(max_concurrent_turns))
        self._sem = asyncio.Semaphore(self._n)
        self._exclusive_lock = asyncio.Lock()

    @property
    def max_concurrent(self) -> int:
        return self._n

    def locked(self) -> bool:
        """True when no turn slot is free (all N slots in use)."""
        return self._sem.locked()

    @asynccontextmanager
    async def turn_slot(self) -> AsyncIterator[None]:
        """Acquire one turn slot (blocks when all N are in use)."""
        await self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()

    @asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        """Run with no turn in flight (drains all N slots; blocks new turns)."""
        async with self._exclusive_lock:  # serialize exclusive ops → no drain deadlock
            acquired = 0
            try:
                for _ in range(self._n):
                    await self._sem.acquire()
                    acquired += 1
                yield
            finally:
                for _ in range(acquired):
                    self._sem.release()

    def exclusive_gate(self) -> "_ExclusiveGate":
        """Return a reusable ``async with gate:`` handle for exclusive access.

        Lets callers that hold a stored lock-like object (e.g. the re-entry
        service) keep ``async with self._engine_lock:`` unchanged while getting
        exclusive semantics.
        """
        return _ExclusiveGate(self)


class _ExclusiveGate:
    """Reusable async context manager adapting ``TurnAdmission.exclusive()``."""

    def __init__(self, admission: TurnAdmission) -> None:
        self._admission = admission
        self._stack: List[Any] = []

    async def __aenter__(self) -> None:
        cm = self._admission.exclusive()
        self._stack.append(cm)
        await cm.__aenter__()
        return None

    async def __aexit__(self, *exc: Any) -> None:
        cm = self._stack.pop()
        await cm.__aexit__(*exc)
