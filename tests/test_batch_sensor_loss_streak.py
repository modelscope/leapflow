# Copyright (c) Alibaba, Inc. and its affiliates.
"""Batch sensor-loss streaks count consecutive samples per channel, not readings.

Regression: one batch carries one sample of every channel, and a single summed
counter read one stale poll of a six-joint arm as "three consecutive degraded
samples". The declared ``halt`` then damped-e-stopped a stationary Piper (it
sagged onto the table), and the healthy sibling reading in the same batch cleared
the latch milliseconds later -- leaving the halt as the only trace.
"""

from __future__ import annotations

from typing import Any

import pytest

from leapflow.hardware.context import DegradationPolicy
from leapflow.hardware.stream import BatchStreamCoordinator
from leapflow.hardware.transport import Quality, Reading
from benchmarks.native.harness import make_channel, make_context

_JOINTS = tuple(f"joint.j{i}.position" for i in range(1, 7))


class _Registry:
    def __init__(self) -> None:
        self.cleared: list[str] = []

    def clear_degradation(self, device_id: str, *, recovery: str) -> bool:
        self.cleared.append(recovery)
        return True


class _Coordinator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, action: str, **kwargs: Any) -> Any:
        self.calls.append({"action": action, **kwargs})

        class _Outcome:
            completed = True

        _Outcome.action = action
        return _Outcome()


def _coordinator() -> tuple[BatchStreamCoordinator, _Registry, _Coordinator]:
    channels = tuple(make_channel(cid, writable=False) for cid in _JOINTS)
    context = make_context(channels=channels, degradation=DegradationPolicy())
    registry = _Registry()
    stream = BatchStreamCoordinator(
        registry, context, channels, degradation_policy=DegradationPolicy()
    )
    executed = _Coordinator()
    stream._degradation_coordinator = executed  # type: ignore[assignment]
    return stream, registry, executed


async def _batch(stream: BatchStreamCoordinator, stale: set[str]) -> None:
    for cid in _JOINTS:
        quality = Quality.STALE.value if cid in stale else Quality.OK.value
        await stream._track_sensor_quality(Reading("arm", cid, 0.0, quality=quality))


@pytest.mark.asyncio
async def test_one_stale_batch_does_not_trigger_sensor_loss() -> None:
    stream, registry, executed = _coordinator()

    await _batch(stream, stale={"joint.j1.position", "joint.j2.position", "joint.j3.position"})
    await _batch(stream, stale=set())

    assert executed.calls == []
    assert stream.degradation_triggered is False
    assert registry.cleared == []


@pytest.mark.asyncio
async def test_three_consecutive_stale_batches_of_one_channel_trigger_once() -> None:
    stream, _, executed = _coordinator()

    for _ in range(4):
        await _batch(stream, stale={"joint.j4.position"})

    assert [call["action"] for call in executed.calls] == ["halt"]
    assert executed.calls[0]["failed_channel_id"] == "joint.j4.position"
    assert stream.degradation_triggered is True


@pytest.mark.asyncio
async def test_healthy_sibling_does_not_clear_an_ongoing_loss() -> None:
    stream, registry, _ = _coordinator()
    for _ in range(3):
        await _batch(stream, stale={"joint.j2.position"})

    # Every other joint in these batches read OK; the loss is still in progress.
    assert stream.degradation_triggered is True
    assert registry.cleared == []

    await _batch(stream, stale=set())

    assert stream.degradation_triggered is False
    assert registry.cleared == ["healthy_batch_sensor_read"]
