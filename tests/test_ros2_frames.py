"""ROS 2 image-topic previews, without a ROS installation.

A ROS camera driver holds its V4L2 nodes exclusively, so the only non-disruptive view
of such a camera is the stream that driver publishes. These tests pin the two halves
of that path: pixel-format conversion and the transport's frame contract.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from PIL import Image

from leapflow.hardware.context import HardwareContext
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.transport import FrameTransport, TransportError
from leapflow.hardware.transports import ros2 as ros2_module
from leapflow.hardware.transports.ros2 import ROS2Transport, build_transport
from leapflow.hardware.ros2_image import (
    COMPRESSED_IMAGE_TYPE,
    RAW_IMAGE_TYPE,
    ImageConversionError,
    encode_preview,
    frame_message_type,
)

_EXAMPLE = Path(__file__).parents[1] / "src" / "leapflow" / "hardware" / "examples" / "realsense_ros2.yaml"
_COLOR = "camera.color.preview"
_DEPTH = "camera.depth.preview"


def _context() -> HardwareContext:
    return HardwareContext.from_mapping(yaml.safe_load(_EXAMPLE.read_text(encoding="utf-8")))


def _jpeg(width: int, height: int, color: tuple[int, int, int] = (200, 40, 40)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height), color).save(output, format="JPEG")
    return output.getvalue()


def _stamp() -> SimpleNamespace:
    return SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=500_000_000), frame_id="camera_color_optical_frame")


def _compressed(data: bytes) -> SimpleNamespace:
    return SimpleNamespace(header=_stamp(), format="rgb8; jpeg compressed bgr8", data=data)


def _raw(encoding: str, width: int, height: int, data: bytes, *, step: int, big: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        header=_stamp(), encoding=encoding, width=width, height=height,
        step=step, is_bigendian=big, data=data,
    )


def _open_transport(**messages: Any) -> ROS2Transport:
    """A transport whose subscriptions already exist, as ``open()`` would leave them."""
    context = _context()
    transport = build_transport(context.transport.config)
    transport._context = context  # noqa: SLF001 - simulate a completed open()
    transport._node = object()  # noqa: SLF001
    transport._connected = True  # noqa: SLF001
    for channel_id, msg in messages.items():
        transport._frame_channels.add(channel_id)  # noqa: SLF001
        if msg is not None:
            transport._msg_callback(channel_id, msg)  # noqa: SLF001
    return transport


# ── Declaration ──


def test_the_ros2_camera_example_is_a_valid_passive_frame_declaration() -> None:
    context = _context()

    assert context.transport.kind == "ros2"
    assert {channel.channel_id for channel in context.channels} == {_COLOR, _DEPTH}
    assert all(channel.is_media and channel.is_readable for channel in context.channels)
    assert all(channel.privacy == "environment" for channel in context.channels)
    assert all(not channel.is_writable for channel in context.channels)


def test_the_ros2_camera_example_passes_registry_admission() -> None:
    """Parsing is not admission: the node registry rejects ids that are not [a-z0-9_]+."""

    class _Example:
        kind = "static"

        def discover(self) -> tuple[HardwareContext, ...]:
            return (_context(),)

    report = HardwareRegistry(HardwareSettings(enabled=True), providers=[_Example()]).load()

    assert report.rejected == ()
    assert report.admitted == ("realsense_ros2",)


def test_image_topic_types_follow_the_image_transport_convention() -> None:
    assert frame_message_type("/camera/camera/color/image_raw/compressed") == COMPRESSED_IMAGE_TYPE
    assert frame_message_type("/camera/camera/color/image_raw") == RAW_IMAGE_TYPE


def test_frame_channels_subscribe_with_their_image_type_instead_of_a_string_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An undiscovered topic used to be subscribed as std_msgs/String and stay silent."""
    created: list[tuple[str, str, Any]] = []

    class _Node:
        def create_subscription(self, msg_type: str, topic: str, callback: Any, *, qos_profile: Any) -> str:
            created.append((msg_type, topic, qos_profile))
            return f"sub:{topic}"

        def get_topic_names_and_types(self) -> list[Any]:
            raise AssertionError("frame channels must not depend on graph discovery")

    monkeypatch.setattr(ros2_module, "_import_msg_type", lambda type_string: type_string)
    context = _context()
    transport = build_transport(context.transport.config)
    transport._node = _Node()  # noqa: SLF001

    transport._setup_channels(context)  # noqa: SLF001

    assert {(msg_type, topic) for msg_type, topic, _ in created} == {
        (COMPRESSED_IMAGE_TYPE, "/camera/camera/color/image_raw/compressed"),
        (COMPRESSED_IMAGE_TYPE, "/camera/camera/depth/image_rect_raw/compressed"),
    }
    assert transport._frame_channels == {_COLOR, _DEPTH}  # noqa: SLF001
    assert not transport._publishers  # noqa: SLF001 - a camera preview never publishes


def test_a_declared_topic_type_overrides_inference() -> None:
    transport = build_transport({"topics": {"cam": "/cam/image"}, "topic_types": {"cam": COMPRESSED_IMAGE_TYPE}})

    assert transport._topic_types == {"cam": COMPRESSED_IMAGE_TYPE}  # noqa: SLF001


# ── Conversion ──


def test_a_jpeg_that_already_fits_is_passed_through_byte_for_byte() -> None:
    data = _jpeg(64, 48)

    preview = encode_preview(_compressed(data), max_width=640)

    assert preview.data == data
    assert (preview.width, preview.height) == (64, 48)
    assert preview.metadata["passthrough"] is True
    assert preview.metadata["source_stamp"] == pytest.approx(12.5)
    assert preview.metadata["frame_id"] == "camera_color_optical_frame"


def test_a_wide_jpeg_is_downscaled_to_the_requested_width() -> None:
    preview = encode_preview(_compressed(_jpeg(1280, 720)), max_width=320, quality=60)

    assert preview.data.startswith(b"\xff\xd8")
    assert (preview.width, preview.height) == (320, 180)
    assert preview.metadata["passthrough"] is False
    assert preview.metadata["source_width"] == 1280


def test_raw_bgr8_honours_row_padding_and_channel_order() -> None:
    # 8x8 pure red in BGR order, each row padded by four bytes. A solid block, because
    # JPEG chroma subsampling would blend neighbouring pixels of a tiny image.
    row = bytes((0, 0, 255)) * 8 + bytes(4)
    msg = _raw("bgr8", 8, 8, row * 8, step=len(row))

    preview = encode_preview(msg)
    red, green, blue = Image.open(io.BytesIO(preview.data)).convert("RGB").getpixel((4, 4))

    assert (preview.width, preview.height) == (8, 8)
    assert red > 200 and green < 60 and blue < 60  # red survives the BGR swap


def test_raw_16bit_depth_is_stretched_into_a_visible_grayscale_preview() -> None:
    near, far = (100).to_bytes(2, "little"), (4000).to_bytes(2, "little")
    msg = _raw("16UC1", 2, 1, near + far, step=4)

    preview = encode_preview(msg)
    image = Image.open(io.BytesIO(preview.data))

    assert image.mode == "L"
    assert image.getpixel((1, 0)) > 200  # the farthest point maps near white


def test_an_unsupported_encoding_is_a_typed_conversion_error() -> None:
    with pytest.raises(ImageConversionError, match="yuv422"):
        encode_preview(_raw("yuv422", 2, 1, bytes(4), step=4))


# ── Transport contract ──


def test_the_ros2_transport_is_a_frame_transport() -> None:
    assert isinstance(ROS2Transport(), FrameTransport)


def test_read_frame_returns_the_latest_image_with_its_topic_and_age() -> None:
    data = _jpeg(64, 48)
    transport = _open_transport(**{_COLOR: _compressed(data)})

    frame = asyncio.run(transport.read_frame(_COLOR, max_width=640, quality=70, fps=5.0))

    assert frame.device_id == "realsense_ros2"
    assert frame.channel_id == _COLOR
    assert frame.data == data
    assert frame.quality == "ok"
    assert frame.metadata["topic"] == "/camera/camera/color/image_raw/compressed"
    assert frame.metadata["age_s"] < 1.0


def test_an_old_image_is_reported_stale_rather_than_live(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _open_transport(**{_COLOR: _compressed(_jpeg(8, 8))})
    monkeypatch.setattr(ros2_module, "_STALE_THRESHOLD_S", 0.0)  # every image is now "old"

    frame = asyncio.run(transport.read_frame(_COLOR))

    assert frame.quality == "stale"


def test_frame_sequence_counts_messages_and_each_size_is_encoded_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two viewers of one message see one sequence number and share one encode."""
    calls: list[int] = []
    real_encode = ros2_module.ros2_image.encode_preview

    def counting_encode(msg: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["max_width"])
        return real_encode(msg, **kwargs)

    monkeypatch.setattr(ros2_module.ros2_image, "encode_preview", counting_encode)
    transport = _open_transport(**{_COLOR: _compressed(_jpeg(64, 48))})

    async def scenario() -> list[Any]:
        return [await transport.read_frame(_COLOR, max_width=32) for _ in range(3)]

    frames = asyncio.run(scenario())

    assert {frame.sequence for frame in frames} == {1}
    assert calls == [32]
    asyncio.run(transport.close())


def test_min_sequence_waits_for_the_next_message_on_the_topic() -> None:
    transport = _open_transport(**{_COLOR: _compressed(_jpeg(8, 8))})

    async def scenario() -> tuple[Any, Any]:
        first = await transport.read_frame(_COLOR)
        asyncio.get_running_loop().call_later(
            0.05, transport._msg_callback, _COLOR, _compressed(_jpeg(8, 8))  # noqa: SLF001
        )
        second = await transport.read_frame(_COLOR, min_sequence=first.sequence + 1)
        return first, second

    first, second = asyncio.run(scenario())

    assert second.sequence == first.sequence + 1
    assert second.monotonic_at > first.monotonic_at
    asyncio.run(transport.close())


def test_probe_reports_image_topic_pump_health_until_close() -> None:
    transport = _open_transport(**{_COLOR: _compressed(_jpeg(8, 8))})
    asyncio.run(transport.read_frame(_COLOR))

    status = asyncio.run(transport.probe())
    pump = status.metadata["frame_hub"]

    assert pump["running"] is True and pump["pumps"] == 1
    assert pump["sequence"] == 1 and pump["frame_age_ms"] is not None
    asyncio.run(transport.close())
    transport._node, transport._connected = object(), True  # noqa: SLF001 - probe a reopen
    assert asyncio.run(transport.probe()).metadata["frame_hub"] == {}


def test_the_ros2_transport_reads_frames_outside_the_device_lock() -> None:
    assert ROS2Transport.frame_reads_concurrent is True


def test_a_silent_topic_is_a_typed_unavailable_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros2_module, "_FIRST_FRAME_TIMEOUT_S", 0.1)
    transport = _open_transport(**{_DEPTH: None})

    with pytest.raises(TransportError) as error:
        asyncio.run(transport.read_frame(_DEPTH))

    assert error.value.failure_code == "ros2_frame_unavailable"
    assert "/camera/camera/depth/image_rect_raw/compressed" in str(error.value)


def test_read_frame_refuses_a_channel_that_is_not_an_image_subscription() -> None:
    transport = _open_transport()

    with pytest.raises(TransportError) as error:
        asyncio.run(transport.read_frame(_COLOR))

    assert error.value.failure_code == "unknown_channel"


def test_plain_read_of_a_frame_channel_never_returns_pixels() -> None:
    """``read()`` values are persisted and sampled; a JPEG there would fill the disk."""
    data = _jpeg(64, 48)
    transport = _open_transport(**{_COLOR: _compressed(data)})
    transport._subscribers[_COLOR] = object()  # noqa: SLF001

    reading = asyncio.run(transport.read(_COLOR))

    assert reading.value == {
        "preview_channel": _COLOR,
        "topic": "/camera/camera/color/image_raw/compressed",
        "width": 0,
        "height": 0,
        "encoding": "rgb8; jpeg compressed bgr8",
        "bytes": len(data),
    }
