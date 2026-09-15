# Copyright (c) Alibaba, Inc. and its affiliates.
"""Parking a turn's admission slot while it waits on a human decision.

Approval prompts have no deadline, so a turn blocked on one must hand its slot
back. Otherwise ``max_concurrent_turns`` unanswered prompts stop every other
workspace from starting a turn and block exclusive maintenance (config reload,
daemon stop) for as long as nobody answers.
"""
from __future__ import annotations

import asyncio

import pytest

from leapflow.daemon.turn_admission import TurnAdmission, parked_for_human_decision


@pytest.mark.asyncio
async def test_parking_frees_the_slot_for_another_workspace() -> None:
    """A parked turn must not occupy capacity: N=1 still admits a second turn."""
    adm = TurnAdmission(1)
    parked = asyncio.Event()
    answered = asyncio.Event()
    second_ran = asyncio.Event()

    async def waits_for_human() -> None:
        async with adm.turn_slot():
            async with parked_for_human_decision():
                parked.set()
                await answered.wait()

    async def other_workspace() -> None:
        async with adm.turn_slot():
            second_ran.set()

    first = asyncio.create_task(waits_for_human())
    await parked.wait()
    assert adm.locked() is False  # the slot was handed back

    second = asyncio.create_task(other_workspace())
    await asyncio.wait_for(second_ran.wait(), timeout=1.0)
    await second

    answered.set()
    await asyncio.wait_for(first, timeout=1.0)
    assert adm.snapshot()["available"] == 1


@pytest.mark.asyncio
async def test_parking_lets_exclusive_maintenance_proceed() -> None:
    """`daemon stop` / config reload must not wait on an unanswered prompt."""
    adm = TurnAdmission(1)
    parked = asyncio.Event()
    answered = asyncio.Event()
    maintenance_ran = asyncio.Event()

    async def waits_for_human() -> None:
        async with adm.turn_slot():
            async with parked_for_human_decision():
                parked.set()
                await answered.wait()

    async def maintenance() -> None:
        async with adm.exclusive():
            maintenance_ran.set()

    turn = asyncio.create_task(waits_for_human())
    await parked.wait()

    window = asyncio.create_task(maintenance())
    await asyncio.wait_for(maintenance_ran.wait(), timeout=1.0)
    await window

    answered.set()
    await asyncio.wait_for(turn, timeout=1.0)


@pytest.mark.asyncio
async def test_unparking_reacquires_and_stays_bounded() -> None:
    """After the human answers, the turn is admitted again under the same cap."""
    adm = TurnAdmission(1)
    parked = asyncio.Event()
    answered = asyncio.Event()
    resumed = asyncio.Event()

    async def waits_for_human() -> None:
        async with adm.turn_slot():
            async with parked_for_human_decision():
                parked.set()
                await answered.wait()
            # Back inside the slot.
            assert adm.locked() is True
            resumed.set()

    turn = asyncio.create_task(waits_for_human())
    await parked.wait()
    answered.set()
    await asyncio.wait_for(resumed.wait(), timeout=1.0)
    await turn

    snapshot = adm.snapshot()
    assert snapshot["available"] == 1
    assert snapshot["active"] == 0
    assert snapshot["parked"] == 0


@pytest.mark.asyncio
async def test_cancelling_a_parked_turn_does_not_inflate_capacity() -> None:
    """A cancelled park must not let ``turn_slot()`` release a slot it lost.

    ``parked_for_human_decision()`` gives the slot back, so if the turn is
    cancelled while parked, the enclosing ``turn_slot()`` must not release
    again — a double release would permanently raise the semaphore's capacity
    above ``max_concurrent_turns`` and silently break bounded concurrency.
    """
    adm = TurnAdmission(1)
    parked = asyncio.Event()

    async def waits_forever() -> None:
        async with adm.turn_slot():
            async with parked_for_human_decision():
                parked.set()
                await asyncio.Event().wait()  # never answered

    turn = asyncio.create_task(waits_forever())
    await parked.wait()
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    # Capacity must still be exactly 1: two turns may never run at once.
    concurrent = 0
    peak = 0

    async def worker() -> None:
        nonlocal concurrent, peak
        async with adm.turn_slot():
            concurrent += 1
            peak = max(peak, concurrent)
            await asyncio.sleep(0.01)
            concurrent -= 1

    await asyncio.gather(worker(), worker())
    assert peak == 1
    assert adm.snapshot()["available"] == 1


@pytest.mark.asyncio
async def test_parking_outside_a_turn_slot_is_a_noop() -> None:
    """In-process CLI and unit tests park without an admission in scope."""
    async with parked_for_human_decision():
        pass  # must not raise


@pytest.mark.asyncio
async def test_snapshot_reports_parked_turns() -> None:
    adm = TurnAdmission(2)
    parked = asyncio.Event()
    answered = asyncio.Event()

    async def waits_for_human() -> None:
        async with adm.turn_slot():
            async with parked_for_human_decision():
                parked.set()
                await answered.wait()

    turn = asyncio.create_task(waits_for_human())
    await parked.wait()

    snapshot = adm.snapshot()
    assert snapshot["parked"] == 1
    # A parked turn is not consuming compute, so it is not reported active.
    assert snapshot["active"] == 0
    assert snapshot["available"] == 2

    answered.set()
    await asyncio.wait_for(turn, timeout=1.0)
    assert adm.snapshot()["parked"] == 0


@pytest.mark.asyncio
async def test_coordinator_parks_the_slot_while_a_prompt_is_pending() -> None:
    """The approval wait itself must park, not just the helper in isolation.

    Guards the wiring: the tests above would still pass if
    ``ApprovalCoordinator.request_approval`` stopped wrapping its wait in
    ``parked_for_human_decision()``, and the daemon would silently go back to
    holding a slot for the whole of the user's think-time.
    """
    from leapflow.daemon.approval_coordinator import ApprovalCoordinator
    from leapflow.daemon.protocol import StreamChunk
    from leapflow.security.approval import ApprovalRequest

    adm = TurnAdmission(1)
    coordinator = ApprovalCoordinator()
    queue: asyncio.Queue[StreamChunk] = asyncio.Queue()
    request = ApprovalRequest(category="shell.command", detail="rm -rf build")
    decision: list[str] = []

    async def turn() -> None:
        async with adm.turn_slot():
            decision.append(
                await coordinator.request_approval(request, (queue, "req-1"))
            )

    task = asyncio.create_task(turn())
    # The prompt reaching the client is the point at which the wait begins.
    chunk = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert chunk.event_type == "approval_request"
    await asyncio.sleep(0)  # let the coordinator reach its parked await

    assert adm.locked() is False, "a pending prompt must not hold a turn slot"
    assert adm.snapshot()["parked"] == 1

    pending_id = chunk.metadata["approval"]["pending_id"]
    await coordinator.resolve(pending_id, "allow_once")
    await asyncio.wait_for(task, timeout=1.0)

    assert decision == ["allow_once"]
    assert adm.snapshot()["available"] == 1
    assert adm.snapshot()["parked"] == 0
