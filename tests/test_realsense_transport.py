"""RealSense transport contract tests without a physical camera or Intel SDK."""

from __future__ import annotations

import asyncio
import importlib
import threading
import time
from array import array
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

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
_FRAME_INTERVAL_S = 0.005
"""Fake SDKs block like a camera does; an instant one would spin the frame pump."""
_OPENED: list[RealSenseTransport] = []


@pytest.fixture(autouse=True)
def _close_opened_transports() -> Iterator[None]:
    """Every opened transport runs a frame pump thread; stop it after each test."""
    yield
    while _OPENED:
        asyncio.run(_OPENED.pop().close())


def _context(example: str = "realsense_435if.yaml") -> HardwareContext:
    payload = yaml.safe_load(_EXAMPLE.with_name(example).read_text(encoding="utf-8"))
    payload["transport"]["config"]["serial_number"] = "123456789"
    return HardwareContext.from_mapping(payload)


@pytest.mark.parametrize("example", ["realsense_435if.yaml", "realsense_d405.yaml"])
def test_realsense_example_is_a_valid_declarative_context(example: str) -> None:
    context = _context(example)

    assert context.transport.kind == "realsense"
    assert context.transport.config["serial_number"] == "123456789"
    expected = {
        "camera.color.preview", "camera.depth.preview",
        "camera.infrared_left.preview", "camera.infrared_right.preview", "camera.metadata",
    }
    expected.update({"camera.rgbd.quality", "camera.rgbd.geometry"})
    if example == "realsense_d405.yaml":
        assert context.device_id == "sensor_realsense_d405"
        assert context.transport.config["expected_product"] == "D405"
        for stream in ("color", "depth", "infrared"):
            assert context.transport.config[stream] == {"width": 640, "height": 480, "fps": 30}
    assert {channel.channel_id for channel in context.channels} == expected
    assert all(channel.privacy == "environment" for channel in context.channels)
    assert all(channel.direction == "read" for channel in context.channels)


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
    def wait_for_frames(self, timeout_ms: int = 5000) -> _Frames:
        time.sleep(_FRAME_INTERVAL_S)
        return _Frames()


def test_realsense_encodes_bgr_sdk_data_as_a_bounded_jpeg_preview() -> None:
    transport = RealSenseTransport({"serial_number": "123456789"})
    _OPENED.append(transport)
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
    assert frame.metadata["session_id"] == transport._metadata()["session_id"]
    assert frame.metadata["session_id"]


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
        self.framesets = 0
        self.aligned = 0
        self.frame_interval_s = _FRAME_INTERVAL_S
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

        def wait_for_frames(timeout_ms: int = 5000) -> _RgbdFrames:
            time.sleep(self.frame_interval_s)
            self.framesets += 1
            return _RgbdFrames(self._depth_width)

        return SimpleNamespace(start=lambda config: profile, stop=stop, wait_for_frames=wait_for_frames)

    def align(self, stream: str) -> SimpleNamespace:
        def process(frames: object) -> object:
            self.aligned += 1
            return frames

        return SimpleNamespace(process=process)

    def colorizer(self) -> object:
        return object()


def _open_with(
    monkeypatch: pytest.MonkeyPatch, sdk: _FakeSdk, **config: object
) -> RealSenseTransport:
    monkeypatch.setattr(realsense_module, "_load_sdk", lambda: sdk)
    transport = RealSenseTransport({"serial_number": "123456789", **config})
    _OPENED.append(transport)
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
    assert frame.sequence >= 1
    metadata = asyncio.run(transport.read("camera.metadata")).value
    assert metadata["color_intrinsics"]["ppx"] == 1.0
    assert metadata["depth_to_color_extrinsics"]["translation_m"] == [0.015, 0.0, 0.0]


def test_preview_and_raw_capture_share_a_session_that_changes_on_reopen(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = _FakeSdk()
    transport = _open_with(monkeypatch, sdk)
    try:
        preview = asyncio.run(transport.read_frame("camera.color.preview"))
        raw = asyncio.run(transport.capture_rgbd())
        metadata = asyncio.run(transport.read("camera.metadata")).value
        session = preview.metadata["session_id"]
        assert session and session == raw.metadata["session_id"] == metadata["session_id"]
        asyncio.run(transport.close())
        asyncio.run(transport.open(_context()))
        reopened = asyncio.run(transport.read_frame("camera.color.preview"))
        assert reopened.metadata["session_id"] != session
        assert reopened.metadata["session_id"] == asyncio.run(transport.read("camera.metadata")).value["session_id"]
    finally:
        asyncio.run(transport.close())


def test_d405_uses_its_own_scale_and_raw_stereo_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = _FakeSdk()
    info = {"name": "RealSense D405", "serial_number": "123456789", "usb_type_descriptor": "3.2"}
    sdk.device.get_info = lambda key: info[key]
    sdk.option.emitter_enabled = "emitter_enabled"
    sdk.device.first_depth_sensor = lambda: SimpleNamespace(
        get_depth_scale=lambda: 0.0001, supports=lambda key: False)
    # D405 has no separate RGB sensor; opening without color options must work.
    del sdk.device.first_color_sensor
    streams = []
    sdk.config = lambda: SimpleNamespace(enable_device=lambda serial: None,
                                         enable_stream=lambda *args: streams.append(args))
    transport = _open_with(monkeypatch, sdk, expected_product="D405")
    transport._pipeline.get_active_profile = lambda: SimpleNamespace(get_device=lambda: sdk.device)
    try:
        frame = asyncio.run(transport.capture_rgbd())
        assert frame.depth_mm(0, 0) == pytest.approx(50.0)
        assert frame.metadata["depth_scale_m"] == 0.0001
        assert [args[1] for args in streams if args[0] == "infrared"] == [1, 2]
        outcome = asyncio.run(transport.write("camera.emitter_enabled", 1))
        assert not outcome.ok
        assert outcome.failure_code == "option_not_supported"
    finally:
        asyncio.run(transport.close())


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


async def _wait_started(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, 2.0), "fake worker did not start"


def test_cancelled_open_drains_startup_and_releases_pipeline(monkeypatch) -> None:
    sdk = _FakeSdk()
    monkeypatch.setattr(realsense_module, "_load_sdk", lambda: sdk)
    transport = RealSenseTransport({"serial_number": "123456789"})
    started, release = threading.Event(), threading.Event()
    original = transport._start_pipeline

    def slow_start() -> None:
        started.set()
        assert release.wait(2.0)
        original()

    monkeypatch.setattr(transport, "_start_pipeline", slow_start)

    async def scenario() -> None:
        task = asyncio.create_task(transport.open(_context()))
        await _wait_started(started)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()  # Repeated cancellation cannot release ownership early.
        await asyncio.sleep(0)
        assert not task.done()
        assert transport._capture_lock.locked()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2.0)
        assert sdk.stopped
        assert transport._pipeline is None
        assert not (await transport.probe()).connected
        assert not transport._capture_lock.locked()
        monkeypatch.setattr(transport, "_start_pipeline", original)
        assert (await transport.open(_context())).connected
        await transport.close()

    asyncio.run(scenario())


def test_cancelled_capture_keeps_close_waiting_until_worker_finishes(monkeypatch) -> None:
    transport = _open_with(monkeypatch, _FakeSdk())
    started, release = threading.Event(), threading.Event()
    original = transport._capture_rgbd

    def slow_capture(*args: object) -> RgbdFrame:
        started.set()
        assert release.wait(2.0)
        return original(*args)

    monkeypatch.setattr(transport, "_capture_rgbd", slow_capture)

    async def scenario() -> None:
        capture = asyncio.create_task(transport.capture_rgbd())
        await _wait_started(started)
        capture.cancel()
        close = asyncio.create_task(transport.close())
        await asyncio.sleep(0.05)
        # The cancelled caller may return at once; the device may not be released
        # until the work it started on the pipeline has finished.
        assert not close.done()
        assert transport._pipeline is not None
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(capture, 2.0)
        await asyncio.wait_for(close, 2.0)
        assert transport._pipeline is None

    asyncio.run(scenario())


def test_pipeline_is_released_when_post_start_setup_fails(monkeypatch) -> None:
    sdk = _FakeSdk()
    monkeypatch.setattr(realsense_module, "_load_sdk", lambda: sdk)

    def fail_align(stream):
        raise RuntimeError("fake alignment setup failure")

    monkeypatch.setattr(sdk, "align", fail_align)
    transport = RealSenseTransport({"serial_number": "123456789"})
    with pytest.raises(TransportError) as error:
        asyncio.run(transport.open(_context()))
    assert error.value.failure_code == "realsense_open_failed"
    assert sdk.stopped and transport._pipeline is None


def test_measurement_read_is_typed_unavailable_without_processor(monkeypatch) -> None:
    transport = _open_with(monkeypatch, _FakeSdk())
    for channel in ("camera.rgbd.quality", "camera.rgbd.geometry"):
        with pytest.raises(TransportError) as error:
            asyncio.run(transport.read(channel))
        assert error.value.failure_code == "depth_processor_unavailable"
    asyncio.run(transport.close())


def test_depth_measurement_read_without_arm_transport(monkeypatch) -> None:
    """Camera-only quality read uses the injected processor; no arm transport is built."""
    from leapflow.hardware.rgbd_quality import DepthLimits, RegionDepthProcessor
    from leapflow.hardware.context import Quality
    from dataclasses import asdict

    # Build a processor with synthetic configuration.
    # The fake SDK produces a 2x1 pixel frame, so use disjoint single-pixel ROIs.
    limits = DepthLimits(3, .5, 10, 10, 1, 100, 2, 2, 0)
    config = {
        "limits": asdict(limits),
        "roi": [0, 0, 1, 1],
        "background_roi": [1, 0, 2, 1],
        "calibration_digest": "test-cal",
        "mount_digest": "test-mount",
        "uncertainty_mm": .5,
    }
    processor = RegionDepthProcessor(config)

    sdk = _FakeSdk()
    monkeypatch.setattr(realsense_module, "_load_sdk", lambda: sdk)
    transport = RealSenseTransport({"serial_number": "123456789"}, processor=processor)
    asyncio.run(transport.open(_context()))
    # Transport was built without any arm-related configuration.
    reading = asyncio.run(transport.read("camera.rgbd.quality"))
    assert reading.channel_id == "camera.rgbd.quality"
    assert reading.quantity == "depth_quality"
    assert reading.quality in (Quality.OK.value, Quality.STALE.value)
    evidence = reading.value
    assert isinstance(evidence, dict)
    assert "qualified" in evidence and "distinct_frames" in evidence
    asyncio.run(transport.close())


# ── Shared framesets (FrameHub) ──────────────────────────────────────────────


class _Colorized(_RawFrame):
    """What the SDK colouriser returns: an RGB8 frame that says so in its profile."""

    def get_profile(self) -> SimpleNamespace:
        return SimpleNamespace(format=lambda: "format.rgb8")


class _Colorizer:
    def __init__(self) -> None:
        self.calls = 0

    def colorize(self, depth: object) -> _Colorized:
        self.calls += 1
        return _Colorized(2, 1, bytes((255, 0, 0, 255, 0, 0)))  # pure red in RGB order


def _hub_sdk() -> _FakeSdk:
    sdk = _FakeSdk()
    colorizer = _Colorizer()
    sdk.colorizer = lambda: colorizer  # type: ignore[method-assign]
    sdk.colorizer_instance = colorizer  # type: ignore[attr-defined]
    return sdk


def test_concurrent_previews_share_one_frameset_and_one_alignment(monkeypatch) -> None:
    sdk = _hub_sdk()
    sdk.frame_interval_s = 0.1  # all six readers must wake before the next frameset
    transport = _open_with(monkeypatch, sdk)

    async def scenario() -> list[object]:
        first = await transport.read_frame("camera.color.preview")
        target = first.sequence + 1
        reads = [
            transport.read_frame(channel, min_sequence=target)
            for channel in ("camera.color.preview", "camera.depth.preview") * 3
        ]
        return list(await asyncio.gather(*reads))

    readings = asyncio.run(scenario())

    assert len({reading.sequence for reading in readings}) == 1
    # Six readers of one frameset: alignment and colourising ran once, not per reader.
    assert sdk.aligned == 1
    assert sdk.colorizer_instance.calls == 1
    assert {r.metadata["aligned_to_color"] for r in readings if "depth" in r.channel_id} == {True}


def test_colour_and_infrared_previews_never_align(monkeypatch) -> None:
    sdk = _hub_sdk()
    transport = _open_with(monkeypatch, sdk)

    for _ in range(3):
        asyncio.run(transport.read_frame("camera.color.preview"))

    assert sdk.aligned == 0


def test_colourised_depth_is_decoded_in_the_format_the_sdk_reports(monkeypatch) -> None:
    from io import BytesIO

    from PIL import Image

    transport = _open_with(monkeypatch, _hub_sdk())

    frame = asyncio.run(transport.read_frame("camera.depth.preview", quality=100))

    red, green, blue = Image.open(BytesIO(frame.data)).convert("RGB").getpixel((0, 0))
    assert red > 200 and blue < 60, "RGB8 colourised depth must not be read as BGR"


def test_rgbd_capture_waits_for_a_frameset_that_arrived_after_the_call(monkeypatch) -> None:
    transport = _open_with(monkeypatch, _FakeSdk())
    preview = asyncio.run(transport.read_frame("camera.color.preview"))
    called_at = time.monotonic()

    frame = asyncio.run(transport.capture_rgbd())

    assert frame.monotonic_at > called_at
    assert frame.sequence > preview.sequence


def test_sequence_keeps_increasing_across_a_reopen(monkeypatch) -> None:
    transport = _open_with(monkeypatch, _FakeSdk())
    before = asyncio.run(transport.read_frame("camera.color.preview")).sequence
    asyncio.run(transport.close())
    asyncio.run(transport.open(_context()))

    after = asyncio.run(transport.read_frame("camera.color.preview")).sequence

    assert after > before


def test_close_stops_the_frame_pump_and_reports_its_health(monkeypatch) -> None:
    sdk = _FakeSdk()
    transport = _open_with(monkeypatch, sdk)
    asyncio.run(transport.read_frame("camera.color.preview"))
    health = asyncio.run(transport.read("camera.metadata")).value
    assert health["frame_hub"]["running"] is True
    assert health["frame_capabilities"]["alignment"] == "software"

    asyncio.run(transport.close())
    pulled = sdk.framesets
    time.sleep(0.05)

    assert sdk.framesets == pulled, "no frameset may be pulled after close"
    assert sdk.stopped


@pytest.mark.parametrize("encoder_kind", ["pillow", "opencv"])
def test_native_depth_preview_skips_sdk_blocks_but_task_capture_stays_aligned(
    monkeypatch, encoder_kind
) -> None:
    if encoder_kind == "opencv":
        pytest.importorskip("cv2")
        pytest.importorskip("numpy")
    sdk = _hub_sdk()
    transport = _open_with(monkeypatch, sdk, preview_depth="native", preview_encoder=encoder_kind)
    preview = asyncio.run(transport.read_frame("camera.depth.preview"))
    assert preview.data.startswith(b"\xff\xd8")
    assert preview.metadata["aligned_to_color"] is False
    assert preview.metadata["preview_depth"] == "native"
    assert preview.metadata["preview_encoder"] == encoder_kind
    assert preview.metadata["encoding"] == "16UC1"
    assert sdk.aligned == sdk.colorizer_instance.calls == 0
    raw = asyncio.run(transport.capture_rgbd())
    assert sdk.aligned == 1
    assert raw.depth_mm(0, 0) == pytest.approx(500)
    assert raw.color == bytes((0, 0, 255, 0, 255, 0))
    assert raw.intrinsics.fx == 600
    assert sdk.colorizer_instance.calls == 0


def test_default_preview_mode_remains_aligned_pillow(monkeypatch) -> None:
    sdk = _hub_sdk()
    transport = _open_with(monkeypatch, sdk)
    preview = asyncio.run(transport.read_frame("camera.depth.preview"))
    assert preview.metadata["aligned_to_color"] is True
    assert preview.metadata["preview_depth"] == "aligned"
    assert preview.metadata["preview_encoder"] == "pillow"
    assert sdk.aligned == sdk.colorizer_instance.calls == 1


def test_injected_encoder_shares_views_and_receives_native_depth(monkeypatch) -> None:
    from leapflow.hardware.ros2_image import encode_preview

    class Encoder:
        def __init__(self):
            self.planes = []

        def encode(self, plane, *, max_width, quality):
            self.planes.append(plane)
            return encode_preview(plane, max_width=max_width, quality=quality)

    encoder = Encoder()
    sdk = _hub_sdk()
    sdk.frame_interval_s = 0.1
    monkeypatch.setattr(realsense_module, "_load_sdk", lambda: sdk)
    transport = RealSenseTransport(
        {"serial_number": "123456789", "preview_depth": "native"}, preview_encoder=encoder
    )
    _OPENED.append(transport)
    asyncio.run(transport.open(_context()))

    async def reads():
        return await asyncio.gather(*(transport.read_frame("camera.depth.preview") for _ in range(6)))

    results = asyncio.run(reads())
    assert len({r.sequence for r in results}) == 1
    assert len(encoder.planes) == 1
    assert encoder.planes[0].encoding == "16UC1"
    assert all(r.metadata["preview_encoder"] == "injected" for r in results)
    assert sdk.aligned == sdk.colorizer_instance.calls == 0


def test_invalid_preview_configuration_fails_before_open() -> None:
    with pytest.raises(TransportError) as error:
        RealSenseTransport({"preview_depth": "unknown"})
    assert error.value.failure_code == "preview_config_invalid"
    with pytest.raises(TransportError) as error:
        RealSenseTransport({}, preview_encoder=object())
    assert error.value.failure_code == "preview_encoder_invalid"


def test_unavailable_encoder_fails_before_camera_pipeline_start(monkeypatch) -> None:
    sdk = _hub_sdk()
    monkeypatch.setattr(realsense_module, "_load_sdk", lambda: sdk)
    def unavailable(kind):
        raise TransportError("encoder unavailable", failure_code="preview_encoder_unavailable")
    monkeypatch.setattr(realsense_module, "build_encoder", unavailable)
    transport = RealSenseTransport({"serial_number": "123456789", "preview_encoder": "custom"})
    with pytest.raises(TransportError) as error:
        asyncio.run(transport.open(_context()))
    assert error.value.failure_code == "preview_encoder_unavailable"
    assert not asyncio.run(transport.probe()).connected
    assert sdk.framesets == 0
