# Copyright (c) Alibaba, Inc. and its affiliates.
"""Device-side setup for the RealSense transport: discovery, options, calibration.

Everything here runs once per open, before a frame flows: finding the serial,
checking the USB link, applying declared colour options, and reading the factory
calibration. Frame acquisition lives in ``_realsense_frames``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from leapflow.hardware.rgbd import CameraIntrinsics
from leapflow.hardware.transport import TransportError

# A manual colour setting only sticks once its auto loop is off; perception needs
# stable exposure and white balance, or the same object segments differently
# between the frame a skill planned on and the one it verifies with.
_AUTO_OVERRIDES: dict[str, str] = {
    "exposure": "enable_auto_exposure",
    "white_balance": "enable_auto_white_balance",
}


@dataclass(frozen=True)
class StreamProfile:
    width: int
    height: int
    fps: int


def find_device(sdk: Any, serial_number: str) -> Any:
    context = sdk.context()
    for device in context.query_devices():
        if device_info(device, sdk.camera_info.serial_number) == serial_number:
            return device
    raise TransportError(
        f"RealSense serial {serial_number!r} is not attached", failure_code="realsense_not_found"
    )


def device_info(device: Any, info_key: Any) -> str:
    if info_key is None:
        return ""
    try:
        if not device.supports(info_key):
            return ""
        return str(device.get_info(info_key) or "")
    except Exception:  # noqa: BLE001 - an unreadable info field reads as absent
        return ""


def depth_scale(profile: Any) -> float:
    try:
        return float(profile.get_device().first_depth_sensor().get_depth_scale())
    except Exception:  # noqa: BLE001 - an unknown scale is reported as 0.0
        return 0.0


def apply_color_options(sdk: Any, device: Any, options: Mapping[str, Any]) -> dict[str, float]:
    """Set declared colour-sensor options, turning the matching auto loop off first.

    Fail-closed: an unknown, unsupported, non-numeric, or out-of-range option
    refuses the open, since a camera silently left on auto exposure would make
    every downstream calibration and segmentation quietly unrepeatable.
    """
    if not options:
        return {}
    sensor = device.first_color_sensor()
    steps: list[tuple[str, Any]] = [
        (auto, 0)
        for name, auto in _AUTO_OVERRIDES.items()
        if name in options and auto not in options
    ]
    steps.extend(options.items())
    applied: dict[str, float] = {}
    for name, value in steps:
        key = getattr(sdk.option, name, None)
        if key is None or not sensor.supports(key):
            raise TransportError(
                f"RealSense colour sensor does not support option {name!r}",
                failure_code="realsense_option_invalid",
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise TransportError(
                f"colour option {name!r} needs a number, got {value!r}",
                failure_code="realsense_option_invalid",
            ) from exc
        bounds = sensor.get_option_range(key)
        if not bounds.min <= numeric <= bounds.max:
            raise TransportError(
                f"colour option {name}={numeric} is outside [{bounds.min}, {bounds.max}]",
                failure_code="realsense_option_invalid",
            )
        sensor.set_option(key, numeric)
        applied[name] = numeric
    return applied


def color_intrinsics(sdk: Any, profile: Any) -> CameraIntrinsics:
    """The colour stream's factory intrinsics -- what aligned depth is expressed in."""
    raw = profile.get_stream(sdk.stream.color).as_video_stream_profile().get_intrinsics()
    try:
        return CameraIntrinsics(
            width=int(raw.width),
            height=int(raw.height),
            fx=float(raw.fx),
            fy=float(raw.fy),
            ppx=float(raw.ppx),
            ppy=float(raw.ppy),
            model=str(raw.model).rsplit(".", 1)[-1].lower(),
            coeffs=tuple(float(c) for c in raw.coeffs),
        )
    except ValueError as exc:
        raise TransportError(
            f"RealSense colour intrinsics are unusable: {exc}",
            failure_code="realsense_intrinsics_unsupported",
        ) from exc


def depth_to_color_extrinsics(sdk: Any, profile: Any) -> dict[str, Any]:
    """Depth-to-colour extrinsics for the metadata channel; informational only."""
    try:
        depth = profile.get_stream(sdk.stream.depth)
        raw = depth.get_extrinsics_to(profile.get_stream(sdk.stream.color))
        return {
            "rotation": [float(v) for v in raw.rotation],
            "translation_m": [float(v) for v in raw.translation],
        }
    except Exception:  # noqa: BLE001 - metadata must not fail an otherwise healthy open
        return {}


def stop_quietly(pipeline: Any) -> None:
    try:
        pipeline.stop()
    except Exception:  # noqa: BLE001 - cleanup after a failed open must not mask its cause
        pass


def meets_usb_minimum(actual: str, minimum: str) -> bool:
    """Compare RealSense USB descriptors without assuming a vendor string format."""

    def major(value: str) -> int:
        for token in value.replace("USB", "").replace(".", " ").split():
            if token.isdigit():
                return int(token)
        return 0

    return major(actual) >= major(minimum)


def stream_profile(value: Any, *, default: tuple[int, int, int]) -> StreamProfile:
    mapping = value if isinstance(value, Mapping) else {}
    return StreamProfile(
        width=max(1, int(mapping.get("width") or default[0])),
        height=max(1, int(mapping.get("height") or default[1])),
        fps=max(1, int(mapping.get("fps") or default[2])),
    )


def profile_dict(profile: StreamProfile) -> dict[str, int]:
    return {"width": profile.width, "height": profile.height, "fps": profile.fps}


__all__ = [
    "StreamProfile",
    "apply_color_options",
    "color_intrinsics",
    "depth_scale",
    "depth_to_color_extrinsics",
    "device_info",
    "find_device",
    "meets_usb_minimum",
    "profile_dict",
    "stop_quietly",
    "stream_profile",
]
