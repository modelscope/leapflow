# Copyright (c) Alibaba, Inc. and its affiliates.
"""Batch sensor-loss streaks count consecutive samples per channel, not readings.

Regression: one batch carries one sample of every channel, and a single summed
counter read one stale poll of a six-joint arm as "three consecutive degraded
samples". The declared ``halt`` then damped-e-stopped a stationary Piper (it
sagged onto the table), and the healthy sibling reading in the same batch cleared
the latch milliseconds later -- leaving the halt as the only trace.

Supervisory actions are also opt-in now: with the default settings a triggered
policy is reported to the board and nothing is sent to the device.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.hardware.context import DegradationPolicy
from leapflow.hardware.degradation import DEGRADATION_SUPPRESSED
from leapflow.hardware.registry import HardwareSettings
from leapflow.hardware.stream import BatchStreamCoordinator, classify_hardware_event
from leapflow.hardware.transport import Quality, Reading
from benchmarks.native.harness import make_channel, make_context

_JOINTS = tuple(f"joint.j{i}.position" for i in range(1, 7))


class _Registry:
    def __init__(self, *, enabled: bool, debounce_s: float = 0.0) -> None:
        self.settings = SimpleNamespace(
            degradation_actions_enabled=enabled, sensor_loss_debounce_s=debounce_s
        )
        self.cleared: list[str] = []
        self.events: list[Any] = []
        self.halts: list[str] = []

    def clear_degradation(self, device_id: str, *, recovery: str) -> bool:
        self.cleared.append(recovery)
        return True

    def publish_event(self, event: Any) -> None:
        self.events.append(event)

    async def halt(self, device_id: str) -> Any:
        self.halts.append(device_id)
        return None


class _Coordinator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, action: str, **kwargs: Any) -> Any:
        self.calls.append({"action": action, **kwargs})

        class _Outcome:
            completed = True

        _Outcome.action = action
        return _Outcome()


def _stream(registry: _Registry) -> BatchStreamCoordinator:
    channels = tuple(make_channel(cid, writable=False) for cid in _JOINTS)
    context = make_context(channels=channels, degradation=DegradationPolicy())
    return BatchStreamCoordinator(registry, context, channels, degradation_policy=DegradationPolicy())


def _coordinator(
    *, debounce_s: float = 0.0
) -> tuple[BatchStreamCoordinator, _Registry, _Coordinator]:
    registry = _Registry(enabled=True, debounce_s=debounce_s)
    stream = _stream(registry)
    executed = _Coordinator()
    stream._degradation_coordinator = executed  # type: ignore[assignment]
    return stream, registry, executed


async def _batch(stream: BatchStreamCoordinator, stale: set[str]) -> None:
    for cid in _JOINTS:
        quality = Quality.STALE.value if cid in stale else Quality.OK.value
        await stream._track_sensor_quality(Reading("arm", cid, 0.0, quality=quality))


def test_degradation_actions_are_off_by_default() -> None:
    assert HardwareSettings().degradation_actions_enabled is False
    assert HardwareSettings.from_settings(SimpleNamespace()).degradation_actions_enabled is False


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
async def test_a_brief_loss_inside_the_debounce_window_does_not_trigger() -> None:
    stream, _, executed = _coordinator(debounce_s=60.0)

    for _ in range(10):
        await _batch(stream, stale={"joint.j4.position"})

    assert executed.calls == []
    assert stream.degradation_triggered is False


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


@pytest.mark.asyncio
async def test_disabled_actions_report_once_and_never_halt() -> None:
    registry = _Registry(enabled=False)
    stream = _stream(registry)

    for _ in range(6):
        await _batch(stream, stale={"joint.j4.position"})

    assert registry.halts == []
    assert [event.kind for event in registry.events] == [DEGRADATION_SUPPRESSED]
    event = registry.events[0]
    assert event.channel_id == "joint.j4.position"
    assert event.value == "halt"
    assert stream.last_degradation_outcome.suppressed is True
    assert stream.last_degradation_outcome.latched is False
    # A withheld trigger blocks nothing on the write path.
    assert stream.degradation_triggered is False


def test_suppressed_events_are_records_not_reflexes() -> None:
    # REFLEX would hand the alert policy the very stop that was just withheld.
    assert classify_hardware_event(DEGRADATION_SUPPRESSED) == "record"


@pytest.mark.asyncio
async def test_health_monitor_offline_halt_is_withheld_when_actions_disabled() -> None:
    from leapflow.hardware.health_monitor import HardwareHealthMonitor

    registry = _Registry(enabled=False)
    emitted: list[str] = []
    bus = SimpleNamespace(emit=lambda topic, payload: emitted.append(topic))
    monitor = HardwareHealthMonitor(registry, event_bus=bus)

    await monitor._execute_halt("arm", reason="device offline (probe failed)")

    assert registry.halts == []
    assert [event.kind for event in registry.events] == [DEGRADATION_SUPPRESSED]
    assert emitted == ["hw.health.device.halt_suppressed"]
