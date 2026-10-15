# Copyright (c) Alibaba, Inc. and its affiliates.
"""Intel RealSense transport for node-local LeapFlow hardware profiles.

The SDK is imported only when a declared RealSense device is opened.  This
keeps a host-only LeapFlow installation importable and lets the Jetson own the
camera, while LHP forwards bounded preview frames to the host.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import time
from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image

from leapflow.hardware.context import HardwareContext, Quality
from leapflow.hardware.transport import (
    SIDE_EFFECT_NONE,
    FrameReading,
    Reading,
    TransportError,
    TransportStatus,
    WriteOutcome,
)

_COLOR_CHANNEL = "camera.color.preview"
_DEPTH_CHANNEL = "camera.depth.preview"
_IR_LEFT_CHANNEL = "camera.infrared_left.preview"
_IR_RIGHT_CHANNEL = "camera.infrared_right.preview"
_METADATA_CHANNEL = "camera.metadata"
_FRAME_CHANNELS = {_COLOR_CHANNEL, _DEPTH_CHANNEL, _IR_LEFT_CHANNEL, _IR_RIGHT_CHANNEL}

# Optional writable channels — sensor options exposed through the SDK.
_EMITTER_CHANNEL = "camera.emitter_enabled"
_LASER_POWER_CHANNEL = "camera.laser_power"
_WRITABLE_OPTIONS: dict[str, str] = {
    _EMITTER_CHANNEL: "emitter_enabled",
    _LASER_POWER_CHANNEL: "laser_power",
}


@dataclass(frozen=True)
class _StreamProfile:
    width: int
    height: int
    fps: int


class RealSenseTransport:
    """Capture one serial-addressed D435-family camera through ``pyrealsense2``.

    The transport emits only bounded RGB, colorized-depth, and IR preview
    images.  Raw Z16 remains inside the Jetson process; its calibration and
    depth-scale metadata are exposed through ``camera.metadata`` instead.
    """

    kind = "realsense"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = dict(config or {})
        self._serial_number = str(config.get("serial_number") or "").strip()
        self._expected_product = str(config.get("expected_product") or "").strip()
        self._minimum_usb = str(config.get("minimum_usb") or "3.0").strip()
        self._align_depth_to_color = bool(config.get("align_depth_to_color", True))
        self._color = _profile(config.get("color"), default=(1280, 720, 30))
        self._depth = _profile(config.get("depth"), default=(848, 480, 30))
        self._infrared = _profile(config.get("infrared"), default=(848, 480, 30))
        self._pipeline: Any = None
        self._align: Any = None
        self._colorizer: Any = None
        self._context: HardwareContext | None = None
        self._sdk: Any = None
        self._connected = False
        self._sequence = 0
        self._depth_scale = 0.0
        self._active_profile: dict[str, Any] = {}
        self._last_frame_metadata: dict[str, Any] = {}
        self._capture_lock = asyncio.Lock()

    async def open(self, context: HardwareContext) -> TransportStatus:
        """Validate the declared serial and start its SDK pipeline once."""
        if not self._serial_number:
            return TransportStatus(
                connected=False,
                halt_supported=False,
                detail="realsense transport requires transport.config.serial_number",
            )
        if self._connected:
            return await self.probe()
        self._context = context
        try:
            self._sdk = _load_sdk()
            await asyncio.to_thread(self._start_pipeline)
        except TransportError:
            self._connected = False
            raise
        except Exception as exc:
            self._connected = False
            raise TransportError(
                f"failed to open RealSense {self._serial_number!r}: {exc}",
                failure_code="realsense_open_failed",
            ) from exc
        self._connected = True
        return await self.probe()

    async def close(self) -> TransportStatus:
        """Stop the node-local pipeline and release the camera claim."""
        async with self._capture_lock:
            pipeline, self._pipeline = self._pipeline, None
            self._align = None
            self._colorizer = None
            self._connected = False
            if pipeline is not None:
                try:
                    await asyncio.to_thread(pipeline.stop)
                except Exception:
                    # Closing must be idempotent even when a camera was unplugged.
                    pass
        return TransportStatus(connected=False, halt_supported=False, detail="RealSense pipeline closed")

    async def probe(self) -> TransportStatus:
        """Report the selected physical device and active capture profile."""
        metadata = {
            "serial_number": self._serial_number,
            "expected_product": self._expected_product,
            "minimum_usb": self._minimum_usb,
            "active_profile": dict(self._active_profile),
            "depth_scale_m": self._depth_scale,
            "capturing": self._pipeline is not None,
        }
        return TransportStatus(
            connected=self._connected,
            halt_supported=False,
            detail=("RealSense pipeline active" if self._connected else "RealSense pipeline is closed"),
            metadata=metadata,
        )

    async def halt(self) -> TransportStatus:
        """Stop observation; a passive camera has no physical emergency stop."""
        await self.close()
        return TransportStatus(
            connected=False,
            halt_supported=False,
            detail="RealSense capture released; camera has no emergency stop",
        )

    async def read(self, channel_id: str) -> Reading:
        """Return bounded camera health/calibration metadata, never frame bytes."""
        self._require_open(channel_id)
        if channel_id not in {_METADATA_CHANNEL, *_FRAME_CHANNELS}:
            raise TransportError(f"unknown channel {channel_id!r}", failure_code="unknown_channel")
        if channel_id != _METADATA_CHANNEL:
            value: Any = {
                "preview_channel": channel_id,
                "available": True,
                "last_frame": dict(self._last_frame_metadata),
            }
        else:
            value = self._metadata()
        return Reading(
            device_id=self._device_id,
            channel_id=channel_id,
            value=value,
            quantity="camera_metadata",
            sequence=self._sequence,
            quality=Quality.OK.value,
        )

    async def read_frame(
        self, channel_id: str, *, max_width: int = 0, quality: int = 0, fps: float = 0.0
    ) -> FrameReading:
        """Return one encoded preview frame for a declared camera channel."""
        self._require_open(channel_id)
        if channel_id not in _FRAME_CHANNELS:
            raise TransportError(f"channel {channel_id!r} is not a preview channel", failure_code="unknown_channel")
        async with self._capture_lock:
            try:
                return await asyncio.to_thread(
                    self._capture_frame, channel_id, max_width=max_width, quality=quality, fps=fps
                )
            except TransportError:
                raise
            except Exception as exc:
                raise TransportError(
                    f"RealSense frame capture failed: {exc}", failure_code="realsense_capture_failed"
                ) from exc

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        """Write optional sensor options (emitter, laser power); refuse others."""
        if channel_id not in _WRITABLE_OPTIONS:
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error=f"RealSense channel {channel_id!r} is read-only",
                failure_code="not_writable",
            )
        return await asyncio.to_thread(
            self._write_option, channel_id, _WRITABLE_OPTIONS[channel_id], value
        )

    def _write_option(
        self, channel_id: str, option_name: str, value: Any
    ) -> WriteOutcome:
        """Set a sensor option on the active depth sensor (blocking).

        ``emitter_enabled`` expects 0 or 1; ``laser_power`` expects a float
        within the sensor's reported min/max range (typically 0–360 for D435).
        """
        from leapflow.hardware.transport import SIDE_EFFECT_COMMITTED

        pipeline = self._pipeline
        if pipeline is None:
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error="RealSense pipeline is not open",
                failure_code="transport_not_open",
            )
        sdk = self._sdk
        if sdk is None:
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error="RealSense SDK is unavailable",
                failure_code="realsense_sdk_missing",
            )
        try:
            profile = pipeline.get_active_profile()
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            option_key = getattr(sdk.option, option_name, None)
            if option_key is None:
                return WriteOutcome(
                    ok=False,
                    side_effect_state=SIDE_EFFECT_NONE,
                    error=f"SDK option {option_name!r} not available",
                    failure_code="option_unavailable",
                )
            if not depth_sensor.supports(option_key):
                return WriteOutcome(
                    ok=False,
                    side_effect_state=SIDE_EFFECT_NONE,
                    error=f"Depth sensor does not support {option_name!r}",
                    failure_code="option_not_supported",
                )
            # Validate range.
            opt_range = depth_sensor.get_option_range(option_key)
            numeric = float(value)
            if numeric < opt_range.min or numeric > opt_range.max:
                return WriteOutcome(
                    ok=False,
                    side_effect_state=SIDE_EFFECT_NONE,
                    error=(
                        f"{option_name} value {numeric} out of range "
                        f"[{opt_range.min}, {opt_range.max}]"
                    ),
                    failure_code="value_out_of_range",
                )
            depth_sensor.set_option(option_key, numeric)
            return WriteOutcome(
                ok=True,
                side_effect_state=SIDE_EFFECT_COMMITTED,
            )
        except (TypeError, ValueError) as exc:
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error=f"Invalid value for {option_name}: {exc}",
                failure_code="invalid_value",
            )
        except Exception as exc:  # noqa: BLE001
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error=f"Failed to set {option_name}: {exc}",
                failure_code="realsense_option_write_failed",
            )

    def _start_pipeline(self) -> None:
        sdk = self._sdk
        if sdk is None:
            raise TransportError("RealSense SDK is unavailable", failure_code="realsense_sdk_missing")
        device = _find_device(sdk, self._serial_number)
        name = _device_info(device, sdk.camera_info.name)
        if self._expected_product and self._expected_product.lower() not in name.lower():
            raise TransportError(
                f"RealSense serial {self._serial_number!r} reports {name!r}, "
                f"not expected {self._expected_product!r}",
                failure_code="realsense_product_mismatch",
            )
        usb_type = _device_info(device, getattr(sdk.camera_info, "usb_type_descriptor", None))
        if not _meets_usb_minimum(usb_type, self._minimum_usb):
            raise TransportError(
                f"RealSense {self._serial_number!r} USB link {usb_type or 'unknown'!r} "
                f"is below required {self._minimum_usb!r}",
                failure_code="realsense_usb_too_slow",
            )
        config = sdk.config()
        config.enable_device(self._serial_number)
        config.enable_stream(
            sdk.stream.color, self._color.width, self._color.height, sdk.format.bgr8, self._color.fps
        )
        config.enable_stream(
            sdk.stream.depth, self._depth.width, self._depth.height, sdk.format.z16, self._depth.fps
        )
        config.enable_stream(
            sdk.stream.infrared, 1, self._infrared.width, self._infrared.height, sdk.format.y8,
            self._infrared.fps,
        )
        config.enable_stream(
            sdk.stream.infrared, 2, self._infrared.width, self._infrared.height, sdk.format.y8,
            self._infrared.fps,
        )
        pipeline = sdk.pipeline()
        profile = pipeline.start(config)
        self._pipeline = pipeline
        self._align = sdk.align(sdk.stream.color) if self._align_depth_to_color else None
        self._colorizer = sdk.colorizer()
        self._depth_scale = _depth_scale(profile)
        self._active_profile = {
            "product": name,
            "usb_type": usb_type,
            "color": _profile_dict(self._color),
            "depth": _profile_dict(self._depth),
            "infrared": _profile_dict(self._infrared),
            "depth_aligned_to_color": self._align_depth_to_color,
        }

    def _capture_frame(
        self, channel_id: str, *, max_width: int, quality: int, fps: float
    ) -> FrameReading:
        pipeline = self._pipeline
        if pipeline is None:
            raise TransportError("RealSense pipeline is not open", failure_code="transport_not_open")
        frames = pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)
        if channel_id == _COLOR_CHANNEL:
            frame, mode = frames.get_color_frame(), "BGR"
        elif channel_id == _DEPTH_CHANNEL:
            depth = frames.get_depth_frame()
            frame, mode = self._colorizer.colorize(depth), "BGR"
        elif channel_id == _IR_LEFT_CHANNEL:
            frame, mode = frames.get_infrared_frame(1), "L"
        else:
            frame, mode = frames.get_infrared_frame(2), "L"
        if not frame:
            raise TransportError("requested RealSense stream has no frame", failure_code="realsense_frame_missing")
        width, height = int(frame.get_width()), int(frame.get_height())
        raw = bytes(frame.get_data())
        if mode == "BGR":
            image = Image.frombytes("RGB", (width, height), raw, "raw", "BGR")
        else:
            image = Image.frombytes("L", (width, height), raw)
        requested_width = int(max_width or 0)
        if requested_width > 0 and image.width > requested_width:
            target_height = max(1, round(image.height * requested_width / image.width))
            image = image.resize((requested_width, target_height), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        safe_quality = min(100, max(1, int(quality or 75)))
        if image.mode == "L":
            image.save(output, format="JPEG", quality=safe_quality)
        else:
            image.convert("RGB").save(output, format="JPEG", quality=safe_quality)
        self._sequence += 1
        metadata = {
            "sdk_frame_number": _safe_frame_number(frame),
            "sdk_timestamp_ms": _safe_timestamp(frame),
            "depth_scale_m": self._depth_scale,
            "aligned_to_color": bool(channel_id == _DEPTH_CHANNEL and self._align is not None),
            "active_profile": dict(self._active_profile),
        }
        self._last_frame_metadata = metadata
        return FrameReading(
            device_id=self._device_id,
            channel_id=channel_id,
            data=output.getvalue(),
            media_type="image/jpeg",
            width=image.width,
            height=image.height,
            observed_at=time.time(),
            monotonic_at=time.monotonic(),
            sequence=self._sequence,
            quality=Quality.OK.value,
            metadata=metadata,
        )

    @property
    def _device_id(self) -> str:
        return self._context.device_id if self._context is not None else ""

    def _metadata(self) -> dict[str, Any]:
        return {
            "serial_number": self._serial_number,
            "depth_scale_m": self._depth_scale,
            "active_profile": dict(self._active_profile),
            "last_frame": dict(self._last_frame_metadata),
        }

    def _require_open(self, channel_id: str) -> None:
        if not self._connected or self._pipeline is None:
            raise TransportError(
                f"RealSense transport for {channel_id!r} is not open", failure_code="transport_not_open"
            )


def _load_sdk() -> Any:
    try:
        return importlib.import_module("pyrealsense2")
    except ImportError as exc:
        raise TransportError(
            "pyrealsense2 is not installed; install the Jetson-compatible librealsense Python binding",
            failure_code="realsense_sdk_missing",
        ) from exc


def _find_device(sdk: Any, serial_number: str) -> Any:
    context = sdk.context()
    for device in context.query_devices():
        if _device_info(device, sdk.camera_info.serial_number) == serial_number:
            return device
    raise TransportError(
        f"RealSense serial {serial_number!r} is not attached", failure_code="realsense_not_found"
    )


def _device_info(device: Any, info_key: Any) -> str:
    if info_key is None:
        return ""
    try:
        if not device.supports(info_key):
            return ""
        return str(device.get_info(info_key) or "")
    except Exception:
        return ""


def _depth_scale(profile: Any) -> float:
    try:
        return float(profile.get_device().first_depth_sensor().get_depth_scale())
    except Exception:
        return 0.0


def _safe_frame_number(frame: Any) -> int:
    try:
        return int(frame.get_frame_number())
    except Exception:
        return 0


def _safe_timestamp(frame: Any) -> float:
    try:
        return float(frame.get_timestamp())
    except Exception:
        return 0.0


def _meets_usb_minimum(actual: str, minimum: str) -> bool:
    """Compare RealSense USB descriptors without assuming a vendor string format."""
    def major(value: str) -> int:
        for token in value.replace("USB", "").replace(".", " ").split():
            if token.isdigit():
                return int(token)
        return 0

    return major(actual) >= major(minimum)


def _profile(value: Any, *, default: tuple[int, int, int]) -> _StreamProfile:
    mapping = value if isinstance(value, Mapping) else {}
    return _StreamProfile(
        width=max(1, int(mapping.get("width") or default[0])),
        height=max(1, int(mapping.get("height") or default[1])),
        fps=max(1, int(mapping.get("fps") or default[2])),
    )


def _profile_dict(profile: _StreamProfile) -> dict[str, int]:
    return {"width": profile.width, "height": profile.height, "fps": profile.fps}


def build_transport(config: Mapping[str, Any] | None = None) -> RealSenseTransport:
    """Entry-point factory for the optional RealSense transport plugin."""
    return RealSenseTransport(config)


__all__ = ["RealSenseTransport", "build_transport"]
