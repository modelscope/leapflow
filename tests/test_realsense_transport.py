"""RealSense transport contract tests without a physical camera or Intel SDK."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest
import yaml

from leapflow.hardware.context import HardwareContext
from leapflow.hardware.transport import TransportError
from leapflow.hardware.transports.realsense import RealSenseTransport, _meets_usb_minimum


_EXAMPLE = (
    Path(__file__).parents[1]
    / "src"
    / "leapflow"
    / "hardware"
    / "examples"
    / "realsense_435if.yaml"
)


def _context() -> HardwareContext:
    payload = yaml.safe_load(_EXAMPLE.read_text(encoding="utf-8"))
    payload["transport"]["config"]["serial_number"] = "123456789"
    return HardwareContext.from_mapping(payload)


def test_realsense_example_is_a_valid_declarative_context() -> None:
    context = _context()

    assert context.transport.kind == "realsense"
    assert context.transport.config["serial_number"] == "123456789"
    assert {channel.channel_id for channel in context.channels} == {
        "camera.color.preview",
        "camera.depth.preview",
        "camera.infrared_left.preview",
        "camera.infrared_right.preview",
        "camera.metadata",
    }
    assert all(channel.privacy == "environment" for channel in context.channels)


def test_realsense_requires_an_immutable_serial_without_importing_sdk() -> None:
    transport = RealSenseTransport({})

    status = asyncio.run(transport.open(_context()))

    assert not status.connected
    assert "serial_number" in status.detail


def test_realsense_missing_sdk_is_a_typed_open_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_sdk(name: str) -> object:
        assert name == "pyrealsense2"
        raise ImportError("not installed")

    monkeypatch.setattr(importlib, "import_module", missing_sdk)
    transport = RealSenseTransport({"serial_number": "123456789"})

    with pytest.raises(TransportError, match="pyrealsense2") as error:
        asyncio.run(transport.open(_context()))

    assert error.value.failure_code == "realsense_sdk_missing"


class _Frame:
    def get_width(self) -> int:
        return 2

    def get_height(self) -> int:
        return 1

    def get_data(self) -> bytes:
        return bytes((0, 0, 255, 0, 255, 0))  # BGR: red then green

    def get_frame_number(self) -> int:
        return 17

    def get_timestamp(self) -> float:
        return 123.0


class _Frames:
    def get_color_frame(self) -> _Frame:
        return _Frame()


class _Pipeline:
    def wait_for_frames(self) -> _Frames:
        return _Frames()


def test_realsense_encodes_bgr_sdk_data_as_a_bounded_jpeg_preview() -> None:
    transport = RealSenseTransport({"serial_number": "123456789"})
    transport._context = _context()
    transport._pipeline = _Pipeline()
    transport._connected = True
    transport._active_profile = {"color": {"width": 2, "height": 1, "fps": 30}}
    transport._depth_scale = 0.001

    frame = asyncio.run(transport.read_frame("camera.color.preview", quality=75))

    assert frame.media_type == "image/jpeg"
    assert frame.data.startswith(b"\xff\xd8")
    assert frame.width == 2
    assert frame.height == 1
    assert frame.metadata["sdk_frame_number"] == 17
    assert frame.metadata["depth_scale_m"] == 0.001


@pytest.mark.parametrize(
    ("actual", "minimum", "expected"),
    [
        ("3.2", "3.0", True),
        ("USB 3.1", "3.0", True),
        ("2.1", "3.0", False),
        ("", "3.0", False),
    ],
)
def test_usb_minimum_is_checked_without_vendor_string_assumptions(
    actual: str, minimum: str, expected: bool
) -> None:
    assert _meets_usb_minimum(actual, minimum) is expected
