# Copyright (c) Alibaba, Inc. and its affiliates.
"""Turn admission control for bounded concurrent execution (Stage 3, P3-4).

``TurnAdmission`` bounds how many agent turns run concurrently (up to N) while
still allowing exclusive maintenance operations (host restart, re-entry
dispatch) to run with no turn in flight. It is a readers/writer discipline built
on a semaphore:

* ``turn_slot()`` acquires one of N slots — up to N turns run concurrently.
* ``exclusive()`` drains all N slots, so it runs alone and blocks new turns until
  it finishes; concurrent exclusive ops are serialized (no drain deadlock).
* ``parked_for_human_decision()`` hands a slot back while a turn blocks on a
  human (an approval prompt), then re-acquires it once the human answers.

``N = 1`` reduces to a plain mutex — exactly the daemon's pre-P3-4 behavior.

A slot bounds *compute* concurrency, not human think-time. Approval prompts have
no deadline, so a turn waiting on one must not keep its slot: N unanswered
prompts would otherwise stop every other workspace from starting a turn and
block exclusive maintenance (config reload, daemon stop) indefinitely.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, AsyncIterator, List, Optional

# Published by ``turn_slot()`` so code arbitrarily deep inside a turn (the
# approval coordinator) can park without being handed an admission reference.
# Only ever *read* across task boundaries; the value is a mutable holder, so no
# ContextVar is set or reset off the owning coroutine.
_current_slot: ContextVar[Optional["_SlotHolder"]] = ContextVar(
    "leapflow_turn_slot", default=None,
)


class _SlotHolder:
    """Tracks whether the owning turn currently holds its admission slot.

    ``turn_slot()`` releases on exit only when the slot is actually held. That
    matters because ``parked_for_human_decision()`` temporarily gives the slot
    back: without this flag, a cancellation while parked would let ``turn_slot()``
    release a slot it no longer owned, permanently inflating the semaphore's
    capacity past ``max_concurrent_turns``.
    """

    def __init__(self, admission: "TurnAdmission") -> None:
        self._admission = admission
        self.held = True

    def release(self) -> None:
        if not self.held:
            return
        self.held = False
        self._admission._release_slot()

    async def reacquire(self) -> None:
        if self.held:
            return
        await self._admission._acquire_slot()
        self.held = True


@asynccontextmanager
async def parked_for_human_decision() -> AsyncIterator[None]:
    """Release this turn's admission slot while it waits on a human.

    A no-op outside a turn slot (in-process CLI, unit tests), so callers can wrap
    a human wait unconditionally.

    Re-acquisition can queue behind other turns or an exclusive window; that is
    intended — the turn resumes under the same admission discipline it started
    under. If re-acquisition is cancelled, the slot stays released and the owning
    ``turn_slot()`` will not double-release it.
    """
    holder = _current_slot.get()
    if holder is None:
        yield
        return
    admission = holder._admission
    holder.release()
    admission._parked_turns += 1
    try:
        yield
    finally:
        admission._parked_turns = max(0, admission._parked_turns - 1)
        await holder.reacquire()


class TurnAdmission:
    """Bounded concurrent turns with exclusive maintenance windows."""

    def __init__(self, max_concurrent_turns: int) -> None:
        self._n = max(1, int(max_concurrent_turns))
        self._sem = asyncio.Semaphore(self._n)
        self._exclusive_lock = asyncio.Lock()
        # Runtime metrics for TUI/daemon visibility. ``_slots_in_use`` counts
        # both active turns and exclusive maintenance drains; ``_active_turns``
        # counts user agent turns only. All mutations happen on the daemon event
        # loop, so no extra thread lock is needed.
        self._slots_in_use = 0
        self._active_turns = 0
        self._waiting_turns = 0
        self._parked_turns = 0

    @property
    def max_concurrent(self) -> int:
        return self._n

    def locked(self) -> bool:
        """True when no turn slot is free (all N slots in use)."""
        return self._sem.locked()

    def snapshot(self) -> dict[str, int | bool]:
        """Return a structured runtime snapshot for status/UI reporting."""
        available = max(0, self._n - self._slots_in_use)
        return {
            "max_concurrent": self._n,
            "active": max(0, self._active_turns),
            "waiting": max(0, self._waiting_turns),
            "parked": max(0, self._parked_turns),
            "available": available,
            "slots_in_use": max(0, self._slots_in_use),
            "locked": self.locked(),
        }

    async def _acquire_slot(self) -> None:
        """Acquire one slot, maintaining the waiting/in-use counters."""
        queued = self.locked()
        if queued:
            self._waiting_turns += 1
        try:
            await self._sem.acquire()
        finally:
            if queued:
                self._waiting_turns = max(0, self._waiting_turns - 1)
        self._slots_in_use += 1
        self._active_turns += 1

    def _release_slot(self) -> None:
        self._active_turns = max(0, self._active_turns - 1)
        self._slots_in_use = max(0, self._slots_in_use - 1)
        self._sem.release()

    @asynccontextmanager
    async def turn_slot(self) -> AsyncIterator[None]:
        """Acquire one turn slot (blocks when all N are in use)."""
        await self._acquire_slot()
        holder = _SlotHolder(self)
        token = _current_slot.set(holder)
        try:
            yield
        finally:
            _current_slot.reset(token)
            # Only release what is still held: a parked turn cancelled before it
            # re-acquired no longer owns a slot.
            holder.release()

    @asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        """Run with no turn in flight (drains all N slots; blocks new turns)."""
        async with self._exclusive_lock:  # serialize exclusive ops → no drain deadlock
            acquired = 0
            try:
                for _ in range(self._n):
                    await self._sem.acquire()
                    self._slots_in_use += 1
                    acquired += 1
                yield
            finally:
                for _ in range(acquired):
                    self._slots_in_use = max(0, self._slots_in_use - 1)
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
