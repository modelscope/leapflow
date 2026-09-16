# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for EventReorderBuffer capacity hard limit (Task #4)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

import pytest

from leapflow.platform.reorder_buffer import EventReorderBuffer


@pytest.mark.asyncio
async def test_forced_flush_on_capacity_exceeded() -> None:
    """Submitting max_buffer_size events triggers an immediate forced flush."""
    emitted: List[Tuple[str, Dict[str, Any]]] = []

    async def capture(event_type: str, payload: Dict[str, Any]) -> None:
        emitted.append((event_type, payload))

    max_size = 5
    buf = EventReorderBuffer(settle_s=10.0, emit=capture, max_buffer_size=max_size)

    # Submit exactly max_buffer_size events (should trigger flush on the 5th)
    for i in range(max_size):
        await buf.submit("test.event", {"_mono_ts": float(i), "seq": i})

    # Flush should have happened immediately — buffer drained
    assert buf.pending_count == 0
    assert len(emitted) == max_size
    # Events should be emitted in sorted timestamp order
    seqs = [e[1]["seq"] for e in emitted]
    assert seqs == list(range(max_size))


@pytest.mark.asyncio
async def test_forced_flush_on_capacity_plus_one() -> None:
    """Submitting max_buffer_size + 1 events triggers flush at max_buffer_size."""
    emitted: List[Tuple[str, Dict[str, Any]]] = []

    async def capture(event_type: str, payload: Dict[str, Any]) -> None:
        emitted.append((event_type, payload))

    max_size = 3
    buf = EventReorderBuffer(settle_s=10.0, emit=capture, max_buffer_size=max_size)

    # Submit max_size + 1 events
    for i in range(max_size + 1):
        await buf.submit("test.event", {"_mono_ts": float(i), "seq": i})

    # First max_size events flushed, the extra one remains buffered
    assert len(emitted) == max_size
    assert buf.pending_count == 1

    # Drain to collect the remaining event
    await buf.drain()
    assert len(emitted) == max_size + 1
    assert buf.pending_count == 0


@pytest.mark.asyncio
async def test_normal_behavior_below_capacity() -> None:
    """Events below capacity accumulate and flush only on settle timeout."""
    emitted: List[Tuple[str, Dict[str, Any]]] = []

    async def capture(event_type: str, payload: Dict[str, Any]) -> None:
        emitted.append((event_type, payload))

    max_size = 100
    buf = EventReorderBuffer(settle_s=0.01, emit=capture, max_buffer_size=max_size)

    # Submit far fewer events than capacity
    for i in range(5):
        await buf.submit("test.event", {"_mono_ts": float(i), "seq": i})

    # Events should still be buffered (settle timer not yet fired)
    assert buf.pending_count == 5
    assert len(emitted) == 0

    # Wait for settle window to fire the delayed flush
    await asyncio.sleep(0.05)
    assert buf.pending_count == 0
    assert len(emitted) == 5


@pytest.mark.asyncio
async def test_default_max_buffer_size() -> None:
    """Default max_buffer_size is 5000."""
    emitted: List[Tuple[str, Dict[str, Any]]] = []

    async def capture(event_type: str, payload: Dict[str, Any]) -> None:
        emitted.append((event_type, payload))

    buf = EventReorderBuffer(settle_s=1.0, emit=capture)
    assert buf._max_buffer_size == 5000
