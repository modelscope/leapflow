# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for SignalBuffer overflow observability (dropped_count tracking)."""
from __future__ import annotations

from leapflow.perception.signals import SignalBuffer
from leapflow.perception.types import InteractionSignal


def _make_signal(label: str = "click") -> InteractionSignal:
    return InteractionSignal(timestamp=0.0, signal_type=label)


def test_dropped_count_increments_on_overflow() -> None:
    buf = SignalBuffer(capacity=3)
    for i in range(4):
        buf.record(_make_signal(f"s{i}"))

    assert buf.count == 3
    assert buf.dropped_count == 1


def test_dropped_count_accumulates_across_drains() -> None:
    buf = SignalBuffer(capacity=2)
    # Fill and overflow
    buf.record(_make_signal("a"))
    buf.record(_make_signal("b"))
    buf.record(_make_signal("c"))  # dropped
    assert buf.dropped_count == 1

    # Drain frees capacity but does NOT reset drop counter
    drained = buf.drain()
    assert len(drained) == 2
    assert buf.dropped_count == 1

    # Fill and overflow again
    buf.record(_make_signal("d"))
    buf.record(_make_signal("e"))
    buf.record(_make_signal("f"))  # dropped
    assert buf.dropped_count == 2


def test_clear_resets_dropped_count() -> None:
    buf = SignalBuffer(capacity=1)
    buf.record(_make_signal("x"))
    buf.record(_make_signal("y"))  # dropped
    assert buf.dropped_count == 1

    buf.clear()
    assert buf.dropped_count == 0
    assert buf.count == 0


def test_no_drops_when_under_capacity() -> None:
    buf = SignalBuffer(capacity=10)
    for i in range(10):
        buf.record(_make_signal(f"s{i}"))

    assert buf.count == 10
    assert buf.dropped_count == 0
