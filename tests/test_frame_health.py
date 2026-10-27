"""Frame-pump health: hub stats become stale / settled / sample-loss transitions."""

from __future__ import annotations

from typing import Any

import pytest

from leapflow.hardware.health_monitor import DeviceHealthRecord, HardwareHealthMonitor
from leapflow.hardware.stream import EventKind, HardwareEventClass
from leapflow.hardware.transport import TransportStatus
from leapflow.hardware.vision import FRAME_HUB_CHANNEL, FrameHealthTracker


def _stats(age: float | None, *, sequence: int = 10, fps: float = 30.0, **extra: Any) -> dict:
    return {"running": True, "sequence": sequence, "fps": fps, "frame_age_ms": age, **extra}


def test_only_transitions_are_reported() -> None:
    tracker = FrameHealthTracker(clock=lambda: 123.0)
    assert tracker.observe("cam", _stats(30.0)) == []
    (stale,) = tracker.observe("cam", _stats(2500.0))
    assert stale.kind == EventKind.STALE and stale.channel_id == FRAME_HUB_CHANNEL
    assert stale.value == 2500.0 and stale.unit == "ms" and stale.observed_at == 123.0
    assert stale.resolved_event_class == HardwareEventClass.DELIBERATIVE
    assert tracker.observe("cam", _stats(9000.0)) == [], "still stale: no repeat"
    (settled,) = tracker.observe("cam", _stats(20.0, sequence=400))
    assert settled.kind == EventKind.SETTLED
    assert tracker.observe("cam", _stats(25.0, sequence=430)) == []


def test_the_stale_threshold_scales_with_the_stream_rate() -> None:
    tracker = FrameHealthTracker(stale_floor_ms=1000, stale_frames=15)
    assert tracker.stale_after_ms(30.0) == 1000.0
    assert tracker.stale_after_ms(5.0) == pytest.approx(3000.0)
    tracker.observe("cam", _stats(10.0, fps=5.0))
    assert tracker.observe("cam", _stats(2000.0, fps=5.0)) == [], "2 s is normal at 5 fps"


def test_capture_failures_are_reported_as_sample_loss() -> None:
    tracker = FrameHealthTracker()
    tracker.observe("cam", _stats(10.0, failures=2))
    (loss,) = tracker.observe("cam", _stats(10.0, failures=5, last_error="usb reset"))
    assert loss.kind == EventKind.SAMPLE_LOSS and loss.value == 3
    assert loss.detail == "usb reset"
    assert loss.resolved_event_class == HardwareEventClass.RECORD


def test_a_pump_without_any_frame_is_stale_only_after_a_whole_poll() -> None:
    tracker = FrameHealthTracker()
    assert tracker.observe("cam", _stats(None, sequence=0)) == []
    (stale,) = tracker.observe("cam", _stats(None, sequence=0))
    assert stale.kind == EventKind.STALE and "no frame yet" in stale.detail


def test_a_closed_or_restarted_pump_starts_over() -> None:
    tracker = FrameHealthTracker()
    tracker.observe("cam", _stats(10.0, sequence=50))
    tracker.observe("cam", _stats(5000.0, sequence=50))
    assert tracker.observe("cam", {"running": False}) == [], "closed is idle, not broken"
    assert tracker.observe("cam", _stats(10.0, sequence=1)) == [], "fresh pump, no settle"
    assert tracker.observe("cam", None) == []


class _Camera:
    def __init__(self) -> None:
        self.hub: dict[str, Any] = _stats(10.0)

    async def probe(self) -> TransportStatus:
        return TransportStatus(connected=True, metadata={"frame_hub": dict(self.hub)})


class _Registry:
    def __init__(self) -> None:
        self.camera = _Camera()
        self.published: list[Any] = []

    async def transport(self, device_id: str) -> _Camera:
        return self.camera

    def publish_event(self, event: Any) -> None:
        self.published.append(event)


async def test_the_monitor_publishes_pump_transitions_on_the_signal_path() -> None:
    registry = _Registry()
    monitor = HardwareHealthMonitor(registry)
    monitor._records["cam"] = DeviceHealthRecord(device_id="cam")
    await monitor._probe_device("cam")
    registry.camera.hub = _stats(4000.0)
    await monitor._probe_device("cam")
    await monitor._probe_device("cam")
    assert [(e.kind, e.device_id) for e in registry.published] == [(EventKind.STALE, "cam")]
    assert monitor.device_health("cam")["last_probe_ok"] is True
