# Copyright (c) Alibaba, Inc. and its affiliates.
"""ROS 2 image topics on either side of a frame hub.

One physical camera has exactly one opener; ROS is an adapter on whichever side
that opener is not:

* owner mode ``ros`` -- a ROS driver (realsense-ros, an Orbbec or ZED wrapper) owns
  the device. ``RosImageProducer`` turns one subscribed image topic into a
  :class:`~leapflow.hardware.vision.frames.FrameProducer`, and ``TopicHubs`` gives
  every such topic its own :class:`~leapflow.hardware.vision.hub.FrameHub`. The ROS
  transport then serves those topics with the same contract as a camera LeapFlow
  opens itself: freshness bounds, per-size JPEG views shared by all viewers, and pump
  stats the health tracker can judge.
* owner mode ``leapflow`` -- LeapFlow's frame hub owns the device. ``FrameTopicBridge``
  reads a frame channel (normally over LHP, so consent, holds and disconnect release
  all still apply) and republishes each *new* frame as
  ``sensor_msgs/msg/CompressedImage``. It runs as a sidecar (``ros_bridge``) in
  whichever interpreter carries the ROS distribution's rclpy, because rclpy is built
  for that distribution's system Python, not for LeapFlow's environment.

Only previews cross the bridge: LHP carries bounded JPEG, so raw depth for ROS
consumers stays a ROS driver's job (owner mode ``ros``). Nothing here imports rclpy;
message types and publishers are injected.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from leapflow.hardware.transport import FrameReading, TransportError
from leapflow.hardware.vision.frames import FrameCapabilities, accepts_keyword
from leapflow.hardware.vision.health import fold_pump_stats
from leapflow.hardware.vision.hub import FrameHub

logger = logging.getLogger(__name__)

COMPRESSED_FORMATS = {"image/jpeg": "jpeg", "image/png": "png"}
"""``FrameReading.media_type`` -> ``CompressedImage.format`` (image_transport names)."""

DEFAULT_BRIDGE_FPS = 10.0
"""Republish cadence: a preview topic for tools such as RViz or Foxglove, not a control loop."""


class RosImageProducer:
    """One image topic as a frame producer: the subscription offers, the pump grabs."""

    def __init__(self, plane: str) -> None:
        self._plane = str(plane or "image")
        self._cond = threading.Condition()
        self._message: Any = None
        self._fresh = False
        self._closed = False

    def capabilities(self) -> FrameCapabilities:
        # The publisher owns rate, alignment and sync; one topic is one plane.
        return FrameCapabilities(planes=(self._plane,))

    def offer(self, message: Any) -> None:
        """Hand over the newest message. Called from the ROS executor; never blocks on readers."""
        with self._cond:
            if self._closed:
                return
            self._message, self._fresh = message, True
            self._cond.notify_all()

    def grab(self, timeout_s: float) -> Any:
        """The newest message not yet grabbed, or None if none arrived in time.

        A silent topic is not a capture failure: the hub reports it as a frame that
        keeps aging, which is what the health tracker turns into a stale signal.
        """
        with self._cond:
            arrived = self._cond.wait_for(
                lambda: self._fresh or self._closed, max(0.0, float(timeout_s))
            )
            if not arrived or self._closed:
                return None
            self._fresh = False
            return self._message

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._message = None
            self._cond.notify_all()


class TopicHubs:
    """One frame hub per subscribed image topic, started by the topic's first message.

    Thread-safe: messages arrive on the ROS executor thread, reads on the event loop.
    Stopped once; a reopened transport builds a new set.
    """

    def __init__(self, name: str = "ros") -> None:
        self._name = str(name or "ros")
        self._lock = threading.Lock()
        self._hubs: dict[str, tuple[RosImageProducer, FrameHub]] = {}
        self._closed = False

    def offer(self, channel_id: str, message: Any) -> None:
        source = self._source(channel_id)
        if source is not None:
            source[0].offer(message)

    def hub(self, channel_id: str) -> FrameHub:
        source = self._source(channel_id)
        if source is None:
            raise TransportError(
                f"{self._name}: image topics are closed", failure_code="transport_not_open"
            )
        return source[1]

    def stats(self) -> dict[str, Any]:
        """Every topic's pump folded into one ``FrameHub.stats()``-shaped snapshot."""
        with self._lock:
            hubs = [hub for _, hub in self._hubs.values()]
        return fold_pump_stats([hub.stats() for hub in hubs])

    def stop(self) -> None:
        """Stop every pump (blocking). Idempotent."""
        with self._lock:
            self._closed = True
            sources, self._hubs = list(self._hubs.values()), {}
        for producer, hub in sources:
            producer.close()
            hub.stop()

    def _source(self, channel_id: str) -> tuple[RosImageProducer, FrameHub] | None:
        with self._lock:
            if self._closed:
                return None
            source = self._hubs.get(channel_id)
            if source is None:
                producer = RosImageProducer(channel_id)
                hub = FrameHub(producer, name=f"{self._name}:{channel_id}").start()
                source = self._hubs[channel_id] = (producer, hub)
            return source


def compressed_image(message_type: Any, frame: FrameReading, *, frame_id: str = "") -> Any:
    """Build a ``sensor_msgs/msg/CompressedImage`` carrying one encoded frame.

    The header stamp is the frame's capture wall time, so ROS consumers align it
    with other topics by when the scene was seen, not when it was relayed.
    """
    image_format = COMPRESSED_FORMATS.get(frame.media_type)
    if image_format is None:
        raise TransportError(
            f"cannot publish {frame.media_type!r} as a compressed image",
            failure_code="ros2_frame_unsupported",
        )
    message = message_type()
    stamp = max(0.0, float(frame.observed_at))
    seconds = int(stamp)
    message.header.stamp.sec = seconds
    message.header.stamp.nanosec = min(999_999_999, int(round((stamp - seconds) * 1e9)))
    message.header.frame_id = frame_id or frame.channel_id.replace(".", "_")
    message.format = image_format
    message.data = bytes(frame.data)
    return message


def ros_image_publisher(
    node: Any, topic: str, *, frame_id: str = "", depth: int = 1
) -> Callable[[FrameReading], None]:
    """A ``publish(frame)`` callable for *topic* on an rclpy *node*.

    Reliable QoS of depth one: it matches both reliable and best-effort
    subscribers, and a viewer that falls behind gets the newest frame, not a queue.
    """
    try:
        from sensor_msgs.msg import CompressedImage
    except ImportError as exc:
        raise TransportError(
            "sensor_msgs is not importable; source the ROS 2 distribution first",
            failure_code="ros2_unavailable",
        ) from exc
    publisher = node.create_publisher(CompressedImage, topic, max(1, int(depth)))

    def publish(frame: FrameReading) -> None:
        publisher.publish(compressed_image(CompressedImage, frame, frame_id=frame_id))

    return publish


@dataclass
class BridgeStats:
    """What a bridge did: frames published, repeats skipped, failures seen."""

    published: int = 0
    repeated: int = 0
    errors: int = 0
    last_sequence: int = 0
    last_error: str = ""


class FrameTopicBridge:
    """Republish the new frames of one frame channel through ``publish``.

    ``read_frame`` is a bound ``FrameTransport.read_frame`` for the channel. When it
    honours ``min_sequence`` each read waits for a frame newer than the last one
    published, so a slow publisher never sends the same frame twice; a reader that
    ignores the bound is deduplicated by sequence instead.
    """

    def __init__(
        self,
        read_frame: Callable[..., Awaitable[FrameReading]],
        publish: Callable[[FrameReading], None],
        *,
        max_width: int = 0,
        quality: int = 0,
        fps: float = DEFAULT_BRIDGE_FPS,
        retry_s: float = 1.0,
    ) -> None:
        self._read = read_frame
        self._publish = publish
        self._max_width = max(0, int(max_width))
        self._quality = max(0, int(quality))
        self._fps = max(0.0, float(fps))
        self._retry_s = max(0.0, float(retry_s))
        self._fresh = accepts_keyword(read_frame, "min_sequence")
        # After a failed read the source may have restarted with a lower sequence, so
        # the next read takes its newest frame unbounded; the repeat check still holds.
        self._unbounded = False
        self.stats = BridgeStats()

    async def run(self, stop: asyncio.Event | None = None, *, limit: int = 0) -> BridgeStats:
        """Bridge until *stop* is set or *limit* frames were published (0: unbounded)."""
        stop = stop or asyncio.Event()
        period = 1.0 / self._fps if self._fps > 0 else 0.0
        while not stop.is_set() and not (limit and self.stats.published >= limit):
            started = time.monotonic()
            if await self._step():
                await _pause(stop, period - (time.monotonic() - started))
            else:
                await _pause(stop, self._retry_s)
        return self.stats

    async def _step(self) -> bool:
        """Read and publish one frame; False after a failure (the caller backs off)."""
        stats = self.stats
        kwargs: dict[str, Any] = {
            "max_width": self._max_width,
            "quality": self._quality,
            "fps": self._fps,
        }
        if self._fresh and stats.last_sequence and not self._unbounded:
            kwargs["min_sequence"] = stats.last_sequence + 1
        try:
            frame = await self._read(**kwargs)
        except TransportError as exc:
            self._unbounded = True
            return self._failed(f"{exc.failure_code or 'read_failed'}: {exc}")
        self._unbounded = False
        if frame.sequence and frame.sequence == stats.last_sequence:
            stats.repeated += 1
            return True
        try:
            self._publish(frame)
        except Exception as exc:  # noqa: BLE001 - a middleware failure is counted, not fatal
            return self._failed(f"publish_failed: {exc}")
        stats.published += 1
        stats.last_sequence = frame.sequence
        return True

    def _failed(self, reason: str) -> bool:
        stats = self.stats
        stats.errors += 1
        stats.last_error = reason
        logger.warning("frame topic bridge: %s", reason)
        return False


async def _pause(stop: asyncio.Event, seconds: float) -> None:
    if seconds <= 0:
        await asyncio.sleep(0)
        return
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


__all__ = [
    "COMPRESSED_FORMATS",
    "DEFAULT_BRIDGE_FPS",
    "BridgeStats",
    "FrameTopicBridge",
    "RosImageProducer",
    "TopicHubs",
    "compressed_image",
    "ros_image_publisher",
]
