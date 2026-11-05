# Copyright (c) Alibaba, Inc. and its affiliates.
"""Intel RealSense transport for node-local LeapFlow hardware profiles.

The SDK is imported only when a declared RealSense device is opened.  This
keeps a host-only LeapFlow installation importable and lets the Jetson own the
camera, while LHP forwards bounded preview frames to the host.

Node-local perception gets more through ``FrameSource.capture_rgbd``: the raw
colour image, depth aligned to it, and the colour stream's factory intrinsics.
Those bytes never leave the node process.

One :class:`~leapflow.hardware.vision.FrameHub` per open pipeline pulls framesets;
every preview and RGB-D capture *looks* at the newest one instead of taking its
own, so concurrent consumers share frames rather than stealing alternate ones.
Alignment and JPEG encoding run once per frameset on the hub's workers, outside
the lifecycle lock, and alignment only when a depth consumer asks for it.
"""

from __future__ import annotations

import asyncio
import importlib
import time
import hashlib
import json
import uuid
from typing import Any, Mapping

from leapflow.hardware.context import HardwareContext, Quality
from leapflow.hardware.drivers._realsense_device import (
    apply_color_options as _apply_color_options,
    color_intrinsics as _color_intrinsics,
    depth_scale as _depth_scale,
    depth_to_color_extrinsics as _depth_to_color_extrinsics,
    device_info as _device_info,
    find_device as _find_device,
    meets_usb_minimum as _meets_usb_minimum,
    profile_dict as _profile_dict,
    stop_quietly as _stop_quietly,
    stream_profile as _profile,
)
from leapflow.hardware.drivers._realsense_frames import (
    PLANE_COLOR,
    PLANE_DEPTH,
    PLANE_IR_LEFT,
    PLANE_IR_RIGHT,
    RealSenseFrames,
    safe_frame_number,
    safe_timestamp,
    source_metadata,
)
from leapflow.hardware.ros2_image import EncodedPreview, encode_preview
from leapflow.hardware.rgbd import CameraIntrinsics, RgbdFrame
from leapflow.hardware.rgbd_quality import RegionDepthProcessor, RgbdProcessor
from leapflow.hardware.rgbd_observation import ColorGeometryProcessor, GeometryProcessor
from leapflow.hardware.transport import (
    SIDE_EFFECT_NONE,
    FrameReading,
    Reading,
    TransportError,
    TransportStatus,
    WriteOutcome,
)
from leapflow.hardware.vision import FrameHub, Frameset
from leapflow.hardware.vision.encoding import PlaneEncoder, build_encoder

async def _worker(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Cancellation cannot release pipeline ownership while a worker is running."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if task.done() and not task.cancelled():
            task.exception()
        raise


_COLOR_CHANNEL = "camera.color.preview"
_DEPTH_CHANNEL = "camera.depth.preview"
_IR_LEFT_CHANNEL = "camera.infrared_left.preview"
_IR_RIGHT_CHANNEL = "camera.infrared_right.preview"
_METADATA_CHANNEL = "camera.metadata"
_PLANES = {
    _COLOR_CHANNEL: PLANE_COLOR,
    _DEPTH_CHANNEL: PLANE_DEPTH,
    _IR_LEFT_CHANNEL: PLANE_IR_LEFT,
    _IR_RIGHT_CHANNEL: PLANE_IR_RIGHT,
}
_FRAME_CHANNELS = set(_PLANES)

# Optional writable channels — sensor options exposed through the SDK.
_EMITTER_CHANNEL = "camera.emitter_enabled"
_LASER_POWER_CHANNEL = "camera.laser_power"
_WRITABLE_OPTIONS: dict[str, str] = {
    _EMITTER_CHANNEL: "emitter_enabled",
    _LASER_POWER_CHANNEL: "laser_power",
}


class RealSenseTransport:
    """Capture one serial-addressed D435-family camera through ``pyrealsense2``.

    The transport emits only bounded RGB, colorized-depth, and IR preview
    images.  Raw Z16 remains inside the Jetson process; its calibration and
    depth-scale metadata are exposed through ``camera.metadata`` instead, and
    node-local skills read raw aligned frames through ``capture_rgbd``.
    """

    kind = "realsense"
    frame_reads_concurrent = True
    """Every reader is served by the one frame hub, so reads need no device I/O lock."""

    def __init__(self, config: Mapping[str, Any] | None = None, *,
                 processor: RgbdProcessor | None = None,
                 geometry_processor: GeometryProcessor | None = None,
                 preview_encoder: PlaneEncoder | None = None) -> None:
        config = dict(config or {})
        if preview_encoder is not None and not isinstance(preview_encoder, PlaneEncoder):
            raise TransportError("preview encoder must implement PlaneEncoder",
                                 failure_code="preview_encoder_invalid")
        self._preview_encoder = preview_encoder
        self._preview_encoder_kind = str(config.get(
            "preview_encoder", "injected" if preview_encoder is not None else "pillow"
        ))
        self._preview_depth = str(config.get("preview_depth", "aligned"))
        if self._preview_depth not in {"aligned", "native"}:
            raise TransportError("preview_depth must be 'aligned' or 'native'",
                                 failure_code="preview_config_invalid")
        self._session_id = uuid.uuid4().hex
        self._frame_timeout_ms = max(1, int(config.get("frame_timeout_ms", 5000)))
        self._frame_buffer = max(1, int(config.get("frame_buffer", 2)))
        self._mount_digest = str(config.get("mount_digest", ""))
        self._calibration_digest = str(config.get("calibration_digest", ""))
        self._uncertainty_mm = config.get("measurement_uncertainty_mm")
        self._processor = processor
        if processor is None and isinstance(config.get("measurement"), Mapping):
            self._processor = RegionDepthProcessor(config["measurement"])
        self._geometry_processor = geometry_processor
        if geometry_processor is None and isinstance(config.get("geometry_measurement"), Mapping):
            self._geometry_processor = ColorGeometryProcessor(config["geometry_measurement"])
        self._measurement_lock = asyncio.Lock()
        self._serial_number = str(config.get("serial_number") or "").strip()
        self._expected_product = str(config.get("expected_product") or "").strip()
        self._minimum_usb = str(config.get("minimum_usb") or "3.0").strip()
        self._align_depth_to_color = bool(config.get("align_depth_to_color", True))
        self._color = _profile(config.get("color"), default=(1280, 720, 30))
        self._depth = _profile(config.get("depth"), default=(848, 480, 30))
        self._infrared = _profile(config.get("infrared"), default=(848, 480, 30))
        options = config.get("color_options")
        self._color_options: dict[str, Any] = (
            {str(k): v for k, v in options.items()} if isinstance(options, Mapping) else {}
        )
        self._intrinsics: CameraIntrinsics | None = None
        self._extrinsics: dict[str, Any] = {}
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
        self._hub: FrameHub | None = None
        self._frames: RealSenseFrames | None = None
        # Guards the lifecycle (open, close, hub creation, option writes). Frame
        # work never runs under it, so a slow consumer cannot delay the others.
        self._capture_lock = asyncio.Lock()

    async def open(self, context: HardwareContext) -> TransportStatus:
        async with self._measurement_lock, self._capture_lock:
            return await self._open(context)

    async def _open(self, context: HardwareContext) -> TransportStatus:
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
        self._session_id = uuid.uuid4().hex
        try:
            self._sdk = _load_sdk()
            if self._preview_encoder is None and self._preview_encoder_kind != "pillow":
                self._preview_encoder = await _worker(build_encoder, self._preview_encoder_kind)
            await _worker(self._start_pipeline)
        except asyncio.CancelledError:
            # _worker has drained startup. Release any completed pipeline while
            # still holding both lifecycle locks before propagating cancellation.
            await _worker(self._release_pipeline)
            raise
        except TransportError:
            await _worker(self._release_pipeline)
            raise
        except Exception as exc:
            await _worker(self._release_pipeline)
            raise TransportError(
                f"failed to open RealSense {self._serial_number!r}: {exc}",
                failure_code="realsense_open_failed",
            ) from exc
        self._connected = True
        return await self.probe()

    async def close(self) -> TransportStatus:
        async with self._measurement_lock:
            return await self._close()

    async def _close(self) -> TransportStatus:
        """Stop the node-local pipeline and release the camera claim."""
        async with self._capture_lock:
            await _worker(self._release_pipeline)
        return TransportStatus(
            connected=False, halt_supported=False, detail="RealSense pipeline closed"
        )

    def _release_pipeline(self) -> None:
        hub, self._hub, self._frames = self._hub, None, None
        if hub is not None:
            # Blocks until no hub thread touches the pipeline any more.
            hub.stop()
            self._sequence = max(self._sequence, int(hub.stats()["sequence"]))
        pipeline, self._pipeline = self._pipeline, None
        self._align = self._colorizer = self._intrinsics = None
        self._connected = False
        self._last_frame_metadata = {}
        if pipeline is not None:
            _stop_quietly(pipeline)

    async def probe(self) -> TransportStatus:
        """Report the selected physical device and active capture profile."""
        metadata = {
            "serial_number": self._serial_number,
            "expected_product": self._expected_product,
            "minimum_usb": self._minimum_usb,
            "active_profile": dict(self._active_profile),
            "depth_scale_m": self._depth_scale,
            "color_intrinsics": self._intrinsics.to_dict() if self._intrinsics else {},
            "capturing": self._pipeline is not None,
            "frame_hub": self._hub.stats() if self._hub is not None else {},
        }
        return TransportStatus(
            connected=self._connected,
            halt_supported=False,
            detail=(
                "RealSense pipeline active" if self._connected else "RealSense pipeline is closed"
            ),
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
        if channel_id in ("camera.rgbd.quality", "camera.rgbd.geometry"):
            return await self._read_measurement(channel_id)
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

    async def _read_measurement(self, channel_id: str) -> Reading:
        geometry = channel_id == "camera.rgbd.geometry"
        processor = self._geometry_processor if geometry else self._processor
        if processor is None:
            raise TransportError("no node-local RGB-D processor configured",
                                 failure_code="depth_processor_unavailable")
        # Serialize temporal window mutation as well as capture; concurrent reads
        # must not reorder source frames or count one bundle twice.
        async with self._measurement_lock:
            frame = await self.capture_rgbd()
            try:
                evidence = await _worker(processor.measure, frame)
            except (ValueError, KeyError) as error:
                raise TransportError(str(error), failure_code="depth_measurement_invalid") from error
            qualified = (evidence.depth.qualified or
                         (len(evidence.pad_depth) == 2 and all(p.qualified for p in evidence.pad_depth))) if geometry else evidence.qualified
            return Reading(device_id=self._device_id, channel_id=channel_id,
                           value=evidence.to_dict(), quantity="geometry_proposal" if geometry else "depth_quality",
                           observed_at=frame.observed_at, monotonic_at=frame.monotonic_at,
                           sequence=frame.sequence,
                           quality=Quality.OK.value if qualified else Quality.STALE.value)

    async def read_frame(
        self,
        channel_id: str,
        *,
        max_width: int = 0,
        quality: int = 0,
        fps: float = 0.0,
        min_sequence: int = 0,
        after_monotonic: float | None = None,
    ) -> FrameReading:
        """Encode one preview of the newest frameset satisfying the freshness bounds.

        ``min_sequence`` and ``after_monotonic`` (this process's monotonic clock)
        wait for a frameset at least that new. ``fps`` is accepted for the frame
        contract; the camera streams at its declared rate whatever a viewer asks,
        and callers pace their own reads.
        """
        self._require_open(channel_id)
        plane = _PLANES.get(channel_id)
        if plane is None:
            raise TransportError(
                f"channel {channel_id!r} is not a preview channel", failure_code="unknown_channel"
            )
        hub, frames, frameset = await self._frameset(
            channel_id, min_sequence=min_sequence, after_monotonic=after_monotonic
        )
        try:
            source = frameset.native
            if plane == PLANE_DEPTH and self._preview_depth == "aligned":
                source = await self._aligned(hub, frames, frameset)
            encoded, frame_number, timestamp_ms = await hub.view(
                frameset,
                ("jpeg", plane, int(max_width or 0), int(quality or 0)),
                lambda _: _encode(frames, source, plane, max_width, quality,
                                  encoder=self._preview_encoder,
                                  native_depth=self._preview_depth == "native"),
            )
        except TransportError:
            raise
        except Exception as exc:  # noqa: BLE001 - any SDK or codec failure is one typed capture error
            raise TransportError(
                f"RealSense frame capture failed: {exc}",
                failure_code="realsense_capture_failed",
            ) from exc
        metadata = {
            "sdk_frame_number": frame_number,
            "sdk_timestamp_ms": timestamp_ms,
            "session_id": self._session_id,
            "depth_scale_m": self._depth_scale,
            "aligned_to_color": bool(plane == PLANE_DEPTH and frames.aligns
                                     and self._preview_depth == "aligned"),
            "preview_depth": self._preview_depth,
            "preview_encoder": self._preview_encoder_kind,
            "active_profile": dict(self._active_profile),
            **dict(encoded.metadata),
        }
        self._last_frame_metadata = metadata
        return FrameReading(
            device_id=self._device_id,
            channel_id=channel_id,
            data=encoded.data,
            media_type="image/jpeg",
            width=encoded.width,
            height=encoded.height,
            observed_at=frameset.observed_at,
            monotonic_at=frameset.monotonic_at,
            sequence=frameset.sequence,
            quality=Quality.OK.value,
            metadata=metadata,
        )

    async def capture_rgbd(self, *, after_monotonic: float | None = None) -> RgbdFrame:
        """Colour with depth aligned to it, from a frameset newer than *after_monotonic*.

        Freshness is a condition on the shared stream, not a capture of its own: a
        caller asking "after my arm stopped" waits for the next frameset instead of
        taking one from a concurrent preview. The default bound is the call itself;
        a capture group passes one shared instant so every camera answers the same
        question.
        """
        self._require_open("rgbd")
        if self._align is None:
            raise TransportError(
                "RGB-D capture needs align_depth_to_color: true in the declaration",
                failure_code="realsense_not_aligned",
            )
        bound = time.monotonic() if after_monotonic is None else float(after_monotonic)
        hub, frames, frameset = await self._frameset("rgbd", after_monotonic=bound)
        try:
            aligned = await self._aligned(hub, frames, frameset)
            return await hub.view(frameset, "rgbd", lambda fs: self._capture_rgbd(fs, aligned))
        except TransportError:
            raise
        except Exception as exc:  # noqa: BLE001 - any SDK failure is one typed capture error
            raise TransportError(
                f"RealSense RGB-D capture failed: {exc}",
                failure_code="realsense_capture_failed",
            ) from exc

    async def _frameset(
        self, channel_id: str, *, min_sequence: int = 0, after_monotonic: float | None = None
    ) -> tuple[FrameHub, RealSenseFrames, Frameset]:
        async with self._capture_lock:
            self._require_open(channel_id)
            hub, frames = self._ensure_hub()
        frameset = await hub.latest(
            min_sequence=min_sequence,
            after_monotonic=after_monotonic,
            timeout_s=self._frame_timeout_ms / 1000.0,
        )
        self._sequence = max(self._sequence, frameset.sequence)
        return hub, frames, frameset

    def _ensure_hub(self) -> tuple[FrameHub, RealSenseFrames]:
        """The open pipeline's hub, started on first demand (caller holds the lifecycle lock)."""
        if self._hub is None or self._frames is None:
            frames = RealSenseFrames(self._pipeline, align=self._align, colorizer=self._colorizer)
            self._hub = FrameHub(
                frames,
                name=self._serial_number or "realsense",
                depth=self._frame_buffer,
                poll_timeout_s=min(1.0, self._frame_timeout_ms / 1000.0),
                sequence_start=self._sequence,
            ).start()
            self._frames = frames
        return self._hub, self._frames

    @staticmethod
    async def _aligned(hub: FrameHub, frames: RealSenseFrames, frameset: Frameset) -> Any:
        """Depth aligned to colour, computed at most once per frameset for all consumers."""
        return await hub.view(frameset, "aligned", lambda fs: frames.aligned(fs.native))

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        """Write optional sensor options (emitter, laser power); refuse others."""
        if channel_id not in _WRITABLE_OPTIONS:
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error=f"RealSense channel {channel_id!r} is read-only",
                failure_code="not_writable",
            )
        async with self._capture_lock:
            return await _worker(
                self._write_option, channel_id, _WRITABLE_OPTIONS[channel_id], value
            )

    def _write_option(self, channel_id: str, option_name: str, value: Any) -> WriteOutcome:
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
            raise TransportError(
                "RealSense SDK is unavailable", failure_code="realsense_sdk_missing"
            )
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
            sdk.stream.color,
            self._color.width,
            self._color.height,
            sdk.format.bgr8,
            self._color.fps,
        )
        config.enable_stream(
            sdk.stream.depth, self._depth.width, self._depth.height, sdk.format.z16, self._depth.fps
        )
        config.enable_stream(
            sdk.stream.infrared,
            1,
            self._infrared.width,
            self._infrared.height,
            sdk.format.y8,
            self._infrared.fps,
        )
        config.enable_stream(
            sdk.stream.infrared,
            2,
            self._infrared.width,
            self._infrared.height,
            sdk.format.y8,
            self._infrared.fps,
        )
        pipeline = sdk.pipeline()
        profile = pipeline.start(config)
        try:
            applied = _apply_color_options(sdk, profile.get_device(), self._color_options)
            intrinsics = _color_intrinsics(sdk, profile)
            align = sdk.align(sdk.stream.color) if self._align_depth_to_color else None
            colorizer = sdk.colorizer()
            depth_scale = _depth_scale(profile)
            extrinsics = _depth_to_color_extrinsics(sdk, profile)
        except BaseException:
            _stop_quietly(pipeline)
            raise
        self._pipeline = pipeline
        self._align = align
        self._colorizer = colorizer
        self._depth_scale = depth_scale
        self._intrinsics = intrinsics
        self._extrinsics = extrinsics
        self._active_profile = {
            "product": name,
            "usb_type": usb_type,
            "color": _profile_dict(self._color),
            "depth": _profile_dict(self._depth),
            "infrared": _profile_dict(self._infrared),
            "depth_aligned_to_color": self._align_depth_to_color,
            "color_options": applied,
        }

    def _capture_rgbd(self, frameset: Frameset, aligned: Any) -> RgbdFrame:
        intrinsics = self._intrinsics
        if intrinsics is None:
            raise TransportError(
                "RealSense pipeline is not open", failure_code="transport_not_open"
            )
        native = frameset.native
        color_source = source_metadata(native.get_color_frame())
        depth_source = source_metadata(native.get_depth_frame())
        color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
        if not color or not depth:
            raise TransportError(
                "RealSense delivered a frameset without colour and depth",
                failure_code="realsense_frame_missing",
            )
        width, height = int(color.get_width()), int(color.get_height())
        if (int(depth.get_width()), int(depth.get_height())) != (width, height):
            raise TransportError(
                "depth is not aligned to the colour image", failure_code="realsense_not_aligned"
            )
        metadata = {
            "sdk_frame_number": safe_frame_number(color),
            "sdk_timestamp_ms": safe_timestamp(color),
            "color_source": color_source,
            "depth_source": depth_source,
            "session_id": self._session_id,
            "mount_digest": self._mount_digest,
            "calibration_digest": self._calibration_digest,
            "measurement_uncertainty_mm": self._uncertainty_mm,
            "color_intrinsics": intrinsics.to_dict(),
            "profile_digest": hashlib.sha256(json.dumps({
                "serial": self._serial_number, "profile": self._active_profile,
                "intrinsics": intrinsics.to_dict(), "depth_scale_m": self._depth_scale,
            }, sort_keys=True).encode()).hexdigest(),
            "depth_scale_m": self._depth_scale,
            "aligned_to_color": True,
            "active_profile": dict(self._active_profile),
        }
        self._last_frame_metadata = metadata
        try:
            return RgbdFrame(
                device_id=self._device_id,
                width=width,
                height=height,
                color=bytes(color.get_data()),
                depth=bytes(depth.get_data()),
                depth_scale_m=self._depth_scale,
                intrinsics=intrinsics,
                observed_at=frameset.observed_at,
                monotonic_at=frameset.monotonic_at,
                sequence=frameset.sequence,
                metadata=metadata,
            )
        except ValueError as exc:
            raise TransportError(
                f"RealSense frame is inconsistent: {exc}", failure_code="realsense_frame_invalid"
            ) from exc

    @property
    def _device_id(self) -> str:
        return self._context.device_id if self._context is not None else ""

    def _metadata(self) -> dict[str, Any]:
        return {
            "serial_number": self._serial_number,
            "session_id": self._session_id,
            "depth_scale_m": self._depth_scale,
            "active_profile": dict(self._active_profile),
            "color_intrinsics": self._intrinsics.to_dict() if self._intrinsics else {},
            "depth_to_color_extrinsics": dict(self._extrinsics),
            "last_frame": dict(self._last_frame_metadata),
            "frame_capabilities": self._frames.capabilities().to_dict() if self._frames else {},
            "frame_hub": self._hub.stats() if self._hub is not None else {},
        }

    def _require_open(self, channel_id: str) -> None:
        if not self._connected or self._pipeline is None:
            raise TransportError(
                f"RealSense transport for {channel_id!r} is not open",
                failure_code="transport_not_open",
            )


def _encode(
    frames: RealSenseFrames, source: Any, plane: str, max_width: int, quality: int,
    *, encoder: PlaneEncoder | None = None, native_depth: bool = False,
) -> tuple[EncodedPreview, int, float]:
    """One plane as a bounded JPEG, with the SDK identity of the frame it came from."""
    image = frames.plane(source, plane, colorize_depth=not native_depth)
    encode = encoder.encode if encoder is not None else encode_preview
    encoded = encode(image, max_width=int(max_width or 0), quality=int(quality or 0))
    return encoded, image.frame_number, image.timestamp_ms


def _load_sdk() -> Any:
    try:
        return importlib.import_module("pyrealsense2")
    except ImportError as exc:
        raise TransportError(
            "pyrealsense2 is not installed; install the Jetson-compatible librealsense Python binding",
            failure_code="realsense_sdk_missing",
        ) from exc


def build_transport(config: Mapping[str, Any] | None = None) -> RealSenseTransport:
    """Entry-point factory for the optional RealSense transport plugin."""
    return RealSenseTransport(config)


__all__ = ["RealSenseTransport", "build_transport"]
