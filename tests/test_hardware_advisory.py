"""Advisory limits: monitor-only breaches, reported with exponential backoff."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from leapflow.hardware.advisory import (
    ADVISORY_CLEARED,
    ADVISORY_EXCEEDED,
    AdvisoryLimit,
    AdvisoryTracker,
    AdvisoryWatch,
    active_advisories,
    load_advisory_limits,
    resolve_field,
)
from leapflow.hardware.observability.digest import _event_severity
from leapflow.hardware.stream import HardwareEventClass, classify_hardware_event


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _telemetry(temps=(30.0,) * 6, *, flag_motor: int | None = None) -> dict:
    return {"motors": [
        {"motor": i + 1, "driver_temperature_c": t, "voltage_v": 24.0,
         "faults": {"motor_overheating": flag_motor == i + 1, "voltage_too_low": False}}
        for i, t in enumerate(temps)]}


TEMP = AdvisoryLimit.from_mapping({
    "device_id": "arm", "channel_id": "arm.telemetry", "field": "motors.*.driver_temperature_c",
    "label": "driver temperature", "max": 55, "unit": "C", "hysteresis": 3,
    "report_after_s": 60, "report_max_s": 240,
})
FLAGS = AdvisoryLimit.from_mapping({
    "device_id": "arm", "channel_id": "arm.telemetry", "field": "motors.*.faults.*",
    "label": "firmware advisory flags", "flag": True,
})


def test_resolve_field_wildcards_lists_and_mappings():
    rows = resolve_field(_telemetry(flag_motor=2), "motors.*.faults.*")
    assert ("motors.1.faults.motor_overheating", True) in rows
    assert len(rows) == 12
    assert resolve_field({"a": 1}, "") == [("", {"a": 1})]
    assert resolve_field({"a": 1}, "b.c") == []


def test_breach_reports_at_once_then_backs_off_then_clears_once():
    clock = _Clock()
    tracker = AdvisoryTracker(clock=clock)
    hot = _telemetry((30, 30, 58, 30, 30, 30))

    first = tracker.observe(TEMP, hot)
    assert [e.kind for e in first] == [ADVISORY_EXCEEDED]
    assert first[0].value == 58 and "motors.2.driver_temperature_c" in first[0].detail

    reported = []
    for _ in range(120):  # ten minutes of 5 s polls
        clock.now += 5
        reported.extend((clock.now, e) for e in tracker.observe(TEMP, hot))
    gaps = [b[0] - a[0] for a, b in zip([(1000.0, None), *reported], reported)]
    assert gaps[:3] == [60, 120, 240], "doubling from report_after_s"
    assert all(gap == 240 for gap in gaps[3:]), "capped at report_max_s"
    assert "poll(s) folded" in reported[0][1].detail

    # Inside the hysteresis band: still breached-in-spirit, no recovery chatter.
    clock.now += 5
    assert tracker.observe(TEMP, _telemetry((30, 30, 54, 30, 30, 30))) == []
    clock.now += 5
    cleared = tracker.observe(TEMP, _telemetry())
    assert [e.kind for e in cleared] == [ADVISORY_CLEARED]
    clock.now += 5
    assert tracker.observe(TEMP, _telemetry()) == []
    clock.now += 5
    again = tracker.observe(TEMP, hot)
    assert [e.kind for e in again] == [ADVISORY_EXCEEDED], "a new breach is immediate again"


def test_flag_limits_and_unknown_values():
    tracker = AdvisoryTracker(clock=_Clock())
    assert tracker.observe(FLAGS, _telemetry()) == []
    events = tracker.observe(FLAGS, _telemetry(flag_motor=4))
    assert events[0].kind == ADVISORY_EXCEEDED and "motors.3.faults.motor_overheating" in events[0].detail
    assert tracker.observe(FLAGS, {"unrelated": 1}) == [], "no match is not evidence of recovery"
    assert [e.kind for e in tracker.observe(FLAGS, _telemetry())] == [ADVISORY_CLEARED]


def test_advisories_are_record_class_and_notable_never_alert():
    assert classify_hardware_event(ADVISORY_EXCEEDED) == HardwareEventClass.RECORD
    assert classify_hardware_event(ADVISORY_CLEARED) == HardwareEventClass.RECORD
    assert _event_severity(ADVISORY_EXCEEDED) == "notable"
    assert _event_severity(ADVISORY_CLEARED) == "info"


def test_active_advisories_follow_the_newest_event_per_limit():
    rows = [
        {"kind": ADVISORY_CLEARED, "device_id": "arm", "channel_id": "t", "quantity": "flags"},
        {"kind": ADVISORY_EXCEEDED, "device_id": "arm", "channel_id": "t", "quantity": "temp"},
        {"kind": ADVISORY_EXCEEDED, "device_id": "arm", "channel_id": "t", "quantity": "flags"},
        {"kind": "settled", "device_id": "arm", "channel_id": "t", "quantity": ""},
    ]
    assert active_advisories(rows) == ["arm.temp"]


def test_producer_marks_an_active_advisory_notable_and_a_cleared_one_info():
    from leapflow.hardware.observability.producer import _assess
    from leapflow.monitor.types import Severity

    def digest(*kinds):
        events = tuple({"kind": k, "device_id": "arm", "channel_id": "t", "quantity": "temp"}
                       for k in kinds)
        return SimpleNamespace(events=events, series=(), storage={}, sampling=(), devices=(1,))

    severity, headline = _assess(digest(ADVISORY_EXCEEDED))
    assert severity == Severity.NOTABLE and "arm.temp" in headline
    assert _assess(digest(ADVISORY_CLEARED, ADVISORY_EXCEEDED))[0] == Severity.INFO


@pytest.mark.parametrize("raw,count", [
    ("", 0), ("not json", 0), ('{"a": 1}', 0),
    (json.dumps([{"device_id": "arm", "channel_id": "c"}]), 0),
    (json.dumps([{"device_id": "arm", "channel_id": "c", "max": 1}, 3]), 1),
    ([{"device_id": "arm", "channel_id": "c", "flag": True}], 1),
])
def test_load_is_tolerant(raw, count):
    assert len(load_advisory_limits(SimpleNamespace(hardware_advisory_limits=raw))) == count


async def test_watch_reads_each_channel_once_and_skips_failures():
    calls: list[tuple[str, str]] = []
    value = {"payload": _telemetry((60,) * 6)}

    class _Registry:
        async def read(self, device_id, channel_id):
            calls.append((device_id, channel_id))
            if device_id == "gone":
                raise ConnectionError("offline")
            return SimpleNamespace(value=value["payload"], quality="ok")

    gone = AdvisoryLimit(device_id="gone", channel_id="arm.telemetry", max_value=1)
    watch = AdvisoryWatch(_Registry(), [TEMP, FLAGS, gone], clock=_Clock())
    events = await watch.poll()
    assert calls == [("arm", "arm.telemetry"), ("gone", "arm.telemetry")]
    assert [e.quantity for e in events] == ["driver temperature"]

    value["payload"] = _telemetry()
    stale = SimpleNamespace(value=_telemetry(), quality="stale")

    class _Stale(_Registry):
        async def read(self, device_id, channel_id):
            return stale

    assert await AdvisoryWatch(_Stale(), [TEMP]).poll() == []
