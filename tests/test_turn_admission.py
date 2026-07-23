"""Tests for TurnAdmission bounded concurrency (Stage 3, P3-4)."""
from __future__ import annotations

import asyncio

import pytest

from leapflow.daemon.turn_admission import TurnAdmission


@pytest.mark.asyncio
async def test_n1_is_strict_mutex() -> None:
    adm = TurnAdmission(1)
    order: list[str] = []

    async def worker(tag: str, hold: float) -> None:
        async with adm.turn_slot():
            order.append(f"{tag}-in")
            await asyncio.sleep(hold)
            order.append(f"{tag}-out")

    await asyncio.gather(worker("A", 0.04), worker("B", 0.01))
    # N=1 serializes: one turn fully completes before the other starts.
    assert order in (
        ["A-in", "A-out", "B-in", "B-out"],
        ["B-in", "B-out", "A-in", "A-out"],
    )


@pytest.mark.asyncio
async def test_n2_allows_two_concurrent_turns() -> None:
    adm = TurnAdmission(2)
    running: list[str] = []

    async def worker(tag: str) -> None:
        async with adm.turn_slot():
            running.append(tag)
            await asyncio.sleep(0.05)

    t1 = asyncio.create_task(worker("A"))
    t2 = asyncio.create_task(worker("B"))
    await asyncio.sleep(0.02)
    assert set(running) == {"A", "B"}   # both turns run concurrently
    assert adm.locked() is True         # both slots in use
    await asyncio.gather(t1, t2)
    assert adm.locked() is False


@pytest.mark.asyncio
async def test_exclusive_waits_for_turns_and_blocks_new_ones() -> None:
    adm = TurnAdmission(2)
    events: list[str] = []
    turn_holding = asyncio.Event()
    release_turn = asyncio.Event()

    async def turn() -> None:
        async with adm.turn_slot():
            turn_holding.set()
            events.append("turn-in")
            await release_turn.wait()
            events.append("turn-out")

    async def maintenance() -> None:
        await turn_holding.wait()
        async with adm.exclusive():
            events.append("exclusive-in")

    t = asyncio.create_task(turn())
    m = asyncio.create_task(maintenance())
    await turn_holding.wait()
    await asyncio.sleep(0.02)
    # Exclusive cannot enter while a turn holds a slot.
    assert "exclusive-in" not in events
    release_turn.set()
    await asyncio.gather(t, m)
    assert events == ["turn-in", "turn-out", "exclusive-in"]


@pytest.mark.asyncio
async def test_exclusive_gate_drains_all_slots() -> None:
    adm = TurnAdmission(2)
    gate = adm.exclusive_gate()
    async with gate:
        assert adm.locked() is True     # exclusive drained every slot
    assert adm.locked() is False
