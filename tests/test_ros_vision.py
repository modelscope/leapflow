"""ROS 2 image adapters on both sides of a frame hub, with rclpy replaced by fakes.

Owner mode ``ros``: one subscribed topic becomes a frame producer and pump. Owner
mode ``leapflow``: a sidecar republishes each *new* frame of a channel as a
``CompressedImage``. Neither needs a ROS distribution to be tested.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from leapflow.hardware.transport import FrameReading, TransportError
from leapflow.hardware.vision import FrameProducer, FrameTopicBridge, RosImageProducer, TopicHubs
from leapflow.hardware.vision.health import FrameHealthTracker, fold_pump_stats
from leapflow.hardware.vision.ros import compressed_image, ros_image_publisher
from leapflow.hardware.vision.ros_bridge import default_topic, serve_bridge


class _Header:
    def __init__(self) -> None:
        self.stamp = SimpleNamespace(sec=0, nanosec=0)
        self.frame_id = ""


class _CompressedImage:
    """The attribute shape of ``sensor_msgs/msg/CompressedImage``."""

    def __init__(self) -> None:
        self.header = _Header()
        self.format = ""
        self.data = b""


def _frame(sequence: int, *, observed_at: float = 1700000000.25, media_type: str = "image/jpeg") -> FrameReading:
    return FrameReading(
        device_id="cam",
        channel_id="color.preview",
        data=b"\xff\xd8jpeg" + bytes([sequence % 256]),
        media_type=media_type,
        width=4,
        height=3,
        observed_at=observed_at,
        sequence=sequence,
    )


# ── Owner mode ros: topic -> producer -> hub ──


def test_a_topic_producer_hands_over_each_message_once() -> None:
    producer = RosImageProducer("color")

    assert isinstance(producer, FrameProducer)
    assert producer.capabilities().planes == ("color",)
    assert producer.grab(0.01) is None  # silence is not a failure
    producer.offer("m1")
    producer.offer("m2")  # a newer message replaces one nobody grabbed
    assert producer.grab(0.01) == "m2"
    assert producer.grab(0.01) is None


def test_a_waiting_grab_wakes_on_offer_and_on_close() -> None:
    producer = RosImageProducer("color")
    threading.Timer(0.02, producer.offer, args=("late",)).start()
    started = time.monotonic()

    assert producer.grab(2.0) == "late"
    assert time.monotonic() - started < 1.0
    threading.Timer(0.02, producer.close).start()
    assert producer.grab(2.0) is None
    producer.offer("after close")
    assert producer.grab(0.01) is None


def test_topic_hubs_pump_each_topic_and_stop_once() -> None:
    hubs = TopicHubs("node")
    hubs.offer("color", "c1")
    hubs.offer("depth", "d1")

    color = hubs.hub("color").wait(timeout_s=2.0)
    depth = hubs.hub("depth").wait(timeout_s=2.0)

    assert (color.native, depth.native) == ("c1", "d1")
    assert hubs.stats()["pumps"] == 2
    hubs.stop()
    hubs.stop()
    assert hubs.stats() == {}
    with pytest.raises(TransportError) as error:
        hubs.hub("color")
    assert error.value.failure_code == "transport_not_open"


def test_folded_pump_health_is_as_stale_as_the_stalest_topic() -> None:
    fresh = {"running": True, "sequence": 90, "fps": 30.0, "frame_age_ms": 20.0, "failures": 0}
    stuck = {"running": True, "sequence": 4, "fps": 5.0, "frame_age_ms": 9000.0, "failures": 2,
             "last_error": "decode failed"}
    idle = {"running": False, "sequence": 7}

    folded = fold_pump_stats([fresh, stuck, idle])

    assert folded == {
        "running": True,
        "sequence": 94,
        "fps": 5.0,
        "frame_age_ms": 9000.0,
        "failures": 2,
        "last_error": "decode failed",
        "pumps": 2,
    }
    assert fold_pump_stats([idle]) == {}
    silent = fold_pump_stats([fresh, {"running": True, "sequence": 0, "frame_age_ms": None}])
    assert silent["frame_age_ms"] is None  # a topic that never delivered is the stalest
    events = FrameHealthTracker().observe("cam", folded)
    assert [event.kind for event in events] == ["stale"]


# ── Owner mode leapflow: channel -> bridge -> topic ──


def test_a_frame_becomes_a_compressed_image_stamped_with_its_capture_time() -> None:
    message = compressed_image(_CompressedImage, _frame(3), frame_id="camera_color_optical_frame")

    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (1700000000, 250_000_000)
    assert message.header.frame_id == "camera_color_optical_frame"
    assert message.format == "jpeg"
    assert message.data == _frame(3).data
    assert compressed_image(_CompressedImage, _frame(3)).header.frame_id == "color_preview"


def test_a_raw_media_type_is_refused_as_a_compressed_image() -> None:
    with pytest.raises(TransportError) as error:
        compressed_image(_CompressedImage, _frame(1, media_type="image/x-raw"))

    assert error.value.failure_code == "ros2_frame_unsupported"


class _Source:
    """A frame channel whose sequence advances only when the test says so."""

    def __init__(self, *, honour_bound: bool = True) -> None:
        self.sequence = 1
        self.calls: list[dict[str, Any]] = []
        self.failures: list[TransportError] = []
        self._honour = honour_bound

    async def read_frame(self, **kwargs: Any) -> FrameReading:
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        bound = kwargs.get("min_sequence", 0)
        if self._honour and bound > self.sequence:
            self.sequence = bound  # "the next frame arrived"
        frame = _frame(self.sequence)
        if not self._honour:
            self.sequence += len(self.calls) % 2  # repeats every other read
        return frame


def test_the_bridge_asks_for_newer_frames_and_publishes_each_once() -> None:
    source = _Source()
    published: list[int] = []

    async def read_frame(*, max_width: int, quality: int, fps: float, min_sequence: int = 0) -> FrameReading:
        return await source.read_frame(max_width=max_width, quality=quality, fps=fps, min_sequence=min_sequence)

    bridge = FrameTopicBridge(read_frame, lambda frame: published.append(frame.sequence), fps=0.0)
    stats = asyncio.run(bridge.run(limit=3))

    assert published == [1, 2, 3]
    assert [call["min_sequence"] for call in source.calls] == [0, 2, 3]
    assert (stats.published, stats.repeated, stats.errors) == (3, 0, 0)


def test_a_reader_without_freshness_bounds_is_deduplicated_by_sequence() -> None:
    source = _Source(honour_bound=False)
    published: list[int] = []

    async def read_frame(*, max_width: int, quality: int, fps: float) -> FrameReading:
        return await source.read_frame(max_width=max_width)

    bridge = FrameTopicBridge(read_frame, lambda frame: published.append(frame.sequence), fps=0.0)
    stats = asyncio.run(bridge.run(limit=3))

    assert published == [1, 2, 3]
    assert stats.repeated == 1
    assert all("min_sequence" not in call for call in source.calls)


def test_a_failed_read_is_counted_and_the_next_read_is_unbounded() -> None:
    """After a failure the source may have restarted at a lower sequence."""
    source = _Source()
    published: list[int] = []

    async def read_frame(*, min_sequence: int = 0, **_: Any) -> FrameReading:
        return await source.read_frame(min_sequence=min_sequence)

    bridge = FrameTopicBridge(
        read_frame, lambda frame: published.append(frame.sequence), fps=0.0, retry_s=0.0
    )
    asyncio.run(bridge.run(limit=1))
    source.failures.append(TransportError("restarting", failure_code="frame_timeout"))
    source.sequence = 1  # the node came back numbering from one
    stats = asyncio.run(bridge.run(limit=2))

    assert [call["min_sequence"] for call in source.calls] == [0, 2, 0, 2]
    assert published == [1, 2]
    assert stats.repeated == 1 and stats.errors == 1
    assert stats.last_error.startswith("frame_timeout")


def test_a_publish_failure_is_counted_without_stopping_the_bridge() -> None:
    source = _Source()
    attempts: list[int] = []

    def flaky_publish(frame: FrameReading) -> None:
        attempts.append(frame.sequence)
        if len(attempts) == 1:
            raise RuntimeError("rmw publish failed")

    async def read_frame(*, min_sequence: int = 0, **_: Any) -> FrameReading:
        return await source.read_frame(min_sequence=min_sequence)

    bridge = FrameTopicBridge(read_frame, flaky_publish, fps=0.0, retry_s=0.0)
    stats = asyncio.run(bridge.run(limit=1))

    assert attempts == [1, 1]  # the unpublished frame is offered again
    assert (stats.published, stats.errors) == (1, 1)


def test_the_bridge_stops_promptly_when_asked() -> None:
    async def never(**_: Any) -> FrameReading:
        raise TransportError("node offline", failure_code="lhp_connect_failed")

    async def scenario() -> Any:
        stop = asyncio.Event()
        bridge = FrameTopicBridge(never, lambda frame: None, retry_s=30.0)
        asyncio.get_running_loop().call_later(0.05, stop.set)
        started = time.monotonic()
        stats = await bridge.run(stop)
        return stats, time.monotonic() - started

    stats, elapsed = asyncio.run(scenario())

    assert elapsed < 2.0
    assert stats.errors == 1 and stats.published == 0


@pytest.fixture
def fake_sensor_msgs(monkeypatch: pytest.MonkeyPatch) -> type:
    package = ModuleType("sensor_msgs")
    messages = ModuleType("sensor_msgs.msg")
    messages.CompressedImage = _CompressedImage  # type: ignore[attr-defined]
    package.msg = messages  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sensor_msgs", package)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", messages)
    return _CompressedImage


class _Node:
    def __init__(self) -> None:
        self.publishers: list[tuple[Any, str, Any]] = []
        self.sent: list[Any] = []

    def create_publisher(self, msg_type: Any, topic: str, qos: Any) -> Any:
        self.publishers.append((msg_type, topic, qos))
        return SimpleNamespace(publish=self.sent.append)


def test_the_sidecar_publishes_a_channel_onto_its_default_topic(fake_sensor_msgs: type) -> None:
    class _Transport:
        def __init__(self) -> None:
            self.sequence = 0
            self.channels: list[str] = []

        async def read_frame(
            self, channel_id: str, *, max_width: int = 0, quality: int = 0, fps: float = 0.0,
            min_sequence: int = 0,
        ) -> FrameReading:
            self.channels.append(channel_id)
            self.sequence = max(self.sequence + 1, min_sequence)
            return _frame(self.sequence)

    node = _Node()
    topic = default_topic("sensor_realsense_435if", "color.preview")
    stats = asyncio.run(
        serve_bridge(_Transport(), node, channel_id="color.preview", topic=topic, fps=0.0, count=2)
    )

    assert topic == "/leapflow/sensor_realsense_435if/color/preview/compressed"
    assert node.publishers == [(fake_sensor_msgs, topic, 1)]
    assert [message.format for message in node.sent] == ["jpeg", "jpeg"]
    assert stats.published == 2 and stats.last_sequence == 2


def test_without_a_ros_distribution_the_publisher_is_a_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sensor_msgs", None)

    with pytest.raises(TransportError) as error:
        ros_image_publisher(_Node(), "/t")

    assert error.value.failure_code == "ros2_unavailable"
