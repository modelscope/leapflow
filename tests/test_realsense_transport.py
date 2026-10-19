"""RealSense transport contract tests without a physical camera or Intel SDK."""

from __future__ import annotations

import asyncio
import importlib
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from leapflow.hardware.context import HardwareContext
from leapflow.hardware.drivers import realsense as realsense_module
from leapflow.hardware.drivers.realsense import RealSenseTransport, _meets_usb_minimum
from leapflow.hardware.rgbd import FrameSource, RgbdFrame
from leapflow.hardware.transport import TransportError

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


# ── RGB-D capture through a fake SDK ─────────────────────────────────────────


class _Names:
    def __init__(self, *names: str) -> None:
        for name in names:
            setattr(self, name, name)


class _ColorSensor:
    def __init__(self) -> None:
        self.ranges = {
            "exposure": (1.0, 10000.0),
            "enable_auto_exposure": (0.0, 1.0),
            "white_balance": (2800.0, 6500.0),
            "enable_auto_white_balance": (0.0, 1.0),
        }
        self.applied: list[tuple[str, float]] = []

    def supports(self, key: str) -> bool:
        return key in self.ranges

    def get_option_range(self, key: str) -> SimpleNamespace:
        low, high = self.ranges[key]
        return SimpleNamespace(min=low, max=high)

    def set_option(self, key: str, value: float) -> None:
        self.applied.append((key, value))


class _RawFrame:
    def __init__(self, width: int, height: int, data: bytes) -> None:
        self._width, self._height, self._data = width, height, data

    def __bool__(self) -> bool:
        return True

    def get_width(self) -> int:
        return self._width

    def get_height(self) -> int:
        return self._height

    def get_data(self) -> bytes:
        return self._data

    def get_frame_number(self) -> int:
        return 42

    def get_timestamp(self) -> float:
        return 9.0


class _RgbdFrames:
    def __init__(self, depth_width: int = 2) -> None:
        self._depth_width = depth_width

    def get_color_frame(self) -> _RawFrame:
        return _RawFrame(2, 1, bytes((0, 0, 255, 0, 255, 0)))

    def get_depth_frame(self) -> _RawFrame:
        units = array("H", [500] * self._depth_width).tobytes()
        return _RawFrame(self._depth_width, 1, units)


class _Stream:
    def as_video_stream_profile(self) -> "_Stream":
        return self

    def get_intrinsics(self) -> SimpleNamespace:
        return SimpleNamespace(
            width=2,
            height=1,
            fx=600.0,
            fy=600.0,
            ppx=1.0,
            ppy=0.5,
            model="distortion.inverse_brown_conrady",
            coeffs=[0.0] * 5,
        )

    def get_extrinsics_to(self, other: "_Stream") -> SimpleNamespace:
        return SimpleNamespace(rotation=[1, 0, 0, 0, 1, 0, 0, 0, 1], translation=[0.015, 0, 0])


class _FakeSdk:
    """Just enough of ``pyrealsense2`` to drive the open and RGB-D paths."""

    def __init__(self, *, depth_width: int = 2) -> None:
        self.camera_info = _Names("name", "serial_number", "usb_type_descriptor")
        self.stream = _Names("color", "depth", "infrared")
        self.format = _Names("bgr8", "z16", "y8")
        self.option = _Names(
            "exposure",
            "enable_auto_exposure",
            "white_balance",
            "enable_auto_white_balance",
            "gain",
        )
        self.sensor = _ColorSensor()
        self.stopped = False
        self._depth_width = depth_width
        info = {
            "name": "Intel RealSense D435IF",
            "serial_number": "123456789",
            "usb_type_descriptor": "3.2",
        }
        self.device = SimpleNamespace(
            supports=lambda key: key in info,
            get_info=lambda key: info[key],
            first_color_sensor=lambda: self.sensor,
            first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: 0.001),
        )

    def context(self) -> SimpleNamespace:
        return SimpleNamespace(query_devices=lambda: [self.device])

    def config(self) -> SimpleNamespace:
        return SimpleNamespace(enable_device=lambda serial: None, enable_stream=lambda *a: None)

    def pipeline(self) -> SimpleNamespace:
        profile = SimpleNamespace(get_device=lambda: self.device, get_stream=lambda s: _Stream())

        def stop() -> None:
            self.stopped = True

        return SimpleNamespace(
            start=lambda config: profile,
            stop=stop,
            wait_for_frames=lambda: _RgbdFrames(self._depth_width),
        )

    def align(self, stream: str) -> SimpleNamespace:
        return SimpleNamespace(process=lambda frames: frames)

    def colorizer(self) -> object:
        return object()


def _open_with(
    monkeypatch: pytest.MonkeyPatch, sdk: _FakeSdk, **config: object
) -> RealSenseTransport:
    monkeypatch.setattr(realsense_module, "_load_sdk", lambda: sdk)
    transport = RealSenseTransport({"serial_number": "123456789", **config})
    asyncio.run(transport.open(_context()))
    return transport


def test_capture_rgbd_returns_aligned_raw_frame_with_factory_intrinsics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _open_with(monkeypatch, _FakeSdk())

    frame = asyncio.run(transport.capture_rgbd())

    assert isinstance(transport, FrameSource)
    assert isinstance(frame, RgbdFrame)
    assert (frame.width, frame.height) == (2, 1)
    assert frame.color == bytes((0, 0, 255, 0, 255, 0))
    assert frame.depth_mm(0, 0) == pytest.approx(500.0)
    assert frame.intrinsics.model == "inverse_brown_conrady"
    assert frame.intrinsics.fx == 600.0
    assert frame.sequence == 1
    metadata = asyncio.run(transport.read("camera.metadata")).value
    assert metadata["color_intrinsics"]["ppx"] == 1.0
    assert metadata["depth_to_color_extrinsics"]["translation_m"] == [0.015, 0.0, 0.0]


def test_capture_rgbd_refuses_when_depth_is_not_aligned(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _open_with(monkeypatch, _FakeSdk(), align_depth_to_color=False)

    with pytest.raises(TransportError) as error:
        asyncio.run(transport.capture_rgbd())

    assert error.value.failure_code == "realsense_not_aligned"


def test_capture_rgbd_refuses_a_depth_image_of_another_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _open_with(monkeypatch, _FakeSdk(depth_width=3))

    with pytest.raises(TransportError) as error:
        asyncio.run(transport.capture_rgbd())

    assert error.value.failure_code == "realsense_not_aligned"


def test_capture_rgbd_requires_an_open_pipeline() -> None:
    transport = RealSenseTransport({"serial_number": "123456789"})

    with pytest.raises(TransportError) as error:
        asyncio.run(transport.capture_rgbd())

    assert error.value.failure_code == "transport_not_open"


def test_color_options_turn_the_auto_loop_off_before_the_manual_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk()
    transport = _open_with(monkeypatch, sdk, color_options={"exposure": 166, "white_balance": 4600})

    assert sdk.sensor.applied == [
        ("enable_auto_exposure", 0.0),
        ("enable_auto_white_balance", 0.0),
        ("exposure", 166.0),
        ("white_balance", 4600.0),
    ]
    status = asyncio.run(transport.probe())
    assert status.metadata["active_profile"]["color_options"]["exposure"] == 166.0


@pytest.mark.parametrize(
    "options",
    [{"exposure": 0.0}, {"gain": 64}, {"focus": 1}, {"exposure": "bright"}],
    ids=["out_of_range", "unsupported", "unknown", "not_a_number"],
)
def test_invalid_color_options_refuse_the_open_and_release_the_camera(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, object]
) -> None:
    sdk = _FakeSdk()
    monkeypatch.setattr(realsense_module, "_load_sdk", lambda: sdk)
    transport = RealSenseTransport({"serial_number": "123456789", "color_options": options})

    with pytest.raises(TransportError) as error:
        asyncio.run(transport.open(_context()))

    assert error.value.failure_code == "realsense_option_invalid"
    assert sdk.stopped
    assert not asyncio.run(transport.probe()).connected
