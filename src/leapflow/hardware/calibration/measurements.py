# Copyright (c) Alibaba, Inc. and its affiliates.
"""Pure node-side measured-data conversions; no acquisition or actuator ownership."""

from __future__ import annotations

import math
import struct
import sys
from array import array
from statistics import median
from typing import Any, Mapping, Sequence

from leapcalib.contracts.identity import ModelIdentity
from leapcalib.contracts.motion import (
    MODE_FIXED,
    MODE_WRIST,
    MotionObservation,
    PointCorrespondence,
    finite_vector,
)
from leapcalib.contracts.observations import RgbdObservation, Stamp
from leapcalib.contracts.tracking import MetricRoi, SourceStamp, TrackingConfig
from leapcalib.geometry.se3 import Transform
from leapflow.hardware.calibration.adapter import rgbd_observation, stream_profile_of
from leapflow.hardware.rgbd import CameraIntrinsics, RgbdFrame
from leapflow.hardware.skills.geometry import Transform as HardwareTransform


def si_transform(pose: HardwareTransform, *, parent: str = "", child: str = "") -> Transform:
    """Convert the existing nominal/raw transform's millimetres once; rotation is unchanged."""
    return Transform(pose.rotation, tuple(float(v) / 1000 for v in pose.translation), parent, child)


def tracked_motion(
    first: RgbdFrame,
    second: RgbdFrame,
    first_pose: HardwareTransform,
    second_pose: HardwareTransform,
    *,
    observation_id: str,
    model: ModelIdentity,
    tracks: Mapping[str, tuple[tuple[int, int], tuple[int, int]]],
    mode: str,
    scene_id: str,
    settled: bool,
    max_speed_rad_s: float,
    stamp_uncertainty_s: float,
) -> tuple[RgbdObservation, RgbdObservation, MotionObservation]:
    """Build a motion factor only from externally identified metric pixel tracks.

    Fixed mode needs tracks on one rigid downstream link during a single-joint
    motion. Wrist mode needs a static scene. This adapter never manufactures
    correspondences from equal pixel coordinates or assumes commanded arrival.
    """
    if mode not in (MODE_FIXED, MODE_WRIST):
        raise ValueError("unknown motion convention")
    if first.device_id != second.device_id or first.sequence >= second.sequence:
        raise ValueError("tracked frames must come from one increasing camera stream")
    if not tracks or len(tracks) > 2048:
        raise ValueError("tracked motion needs 1..2048 externally identified features")
    role = "fixed" if mode == MODE_FIXED else "wrist"
    profile = stream_profile_of(first, role=role)
    if profile != stream_profile_of(second, role=role):
        raise ValueError("stream profile changed between tracked frames")
    before = rgbd_observation(
        first,
        observation_id=f"{observation_id}:first",
        pixels=[p[0] for p in tracks.values()],
        profile=profile,
        stamp=Stamp(first.monotonic_at, uncertainty_s=stamp_uncertainty_s),
    )
    after = rgbd_observation(
        second,
        observation_id=f"{observation_id}:second",
        pixels=[p[1] for p in tracks.values()],
        profile=profile,
        stamp=Stamp(second.monotonic_at, uncertainty_s=stamp_uncertainty_s),
    )
    pairs = []
    for (identity, (pixel_a, pixel_b)), a, b in zip(tracks.items(), before.samples, after.samples):
        if not profile.depth_in_range(a.depth_m) or not profile.depth_in_range(b.depth_m):
            continue
        point_a = first.intrinsics.deproject(*pixel_a, a.depth_m * 1000)
        point_b = second.intrinsics.deproject(*pixel_b, b.depth_m * 1000)
        pairs.append(
            PointCorrespondence(
                identity, tuple(v / 1000 for v in point_a), tuple(v / 1000 for v in point_b)
            )
        )
    a, b = si_transform(first_pose), si_transform(second_pose)
    robot_motion = b.compose(a.inverse()) if mode == MODE_FIXED else a.inverse().compose(b)
    motion = MotionObservation(
        observation_id,
        before.observation_id,
        after.observation_id,
        profile,
        model,
        before.stamp,
        after.stamp,
        before.provenance,
        mode,
        scene_id,
        robot_motion,
        tuple(pairs),
        settled,
        max_speed_rad_s,
    )
    return before, after, motion


def frame_evidence(
    frame: RgbdFrame, *, limit: int, role: str, stamp_uncertainty_s: float
) -> dict[str, Any]:
    """A bounded, local-only metric sample record for readiness, not correspondence."""
    if type(limit) is not int or not 16 <= limit <= 2048:
        raise ValueError("metric readiness sampling needs 16..2048 points")
    stride = max(1, math.ceil(math.sqrt(frame.width * frame.height / limit)))
    pixels = [
        (u, v)
        for v in range(stride // 2, frame.height, stride)
        for u in range(stride // 2, frame.width, stride)
    ][:limit]
    profile = stream_profile_of(frame, role=role)
    observation = rgbd_observation(
        frame,
        observation_id=f"{frame.device_id}:{frame.sequence}",
        pixels=pixels,
        profile=profile,
        stamp=Stamp(frame.monotonic_at, uncertainty_s=stamp_uncertainty_s),
    )
    depths = sorted(s.depth_m for s in observation.measured_samples)
    return {
        "observation": observation.to_dict(),
        "profile_digest": profile.digest,
        "frame_sequence": frame.sequence,
        "valid_ratio": observation.valid_ratio,
        "samples": len(observation.samples),
        "measured_samples": len(depths),
        "depth_range_m": [depths[0], depths[-1]] if depths else [],
        "depth_median_m": depths[len(depths) // 2] if depths else None,
        "source_profile_digest": frame.metadata.get("profile_digest", ""),
        "source_mount_digest": frame.metadata.get("mount_digest", ""),
        "aligned_to_color": frame.metadata.get("aligned_to_color", False),
    }


def metric_roi(
    frame: RgbdFrame,
    *,
    region: Sequence[int],
    role: str,
    receipt_uncertainty_s: float,
    exposure_uncertainty_s: float,
    valid_min_m: float = 0.0,
    valid_max_m: float = 0.0,
) -> MetricRoi:
    """Copy a lossless bounded ROI off-loop, keeping exact original intrinsics.

    This never resizes a preview, changes depth scale, or invents mount/session
    provenance. Unknown source identity is an acquisition refusal.
    """
    if (
        not isinstance(frame, RgbdFrame)
        or len(region) != 4
        or any(type(v) is not int for v in region)
    ):
        raise ValueError("metric ROI needs a typed frame and integer x/y/width/height")
    x, y, width, height = region
    if (
        x < 0
        or y < 0
        or not 8 <= width <= 640
        or not 8 <= height <= 480
        or x + width > frame.width
        or y + height > frame.height
    ):
        raise ValueError("metric ROI exceeds bounded frame geometry")
    if frame.metadata.get("aligned_to_color") is not True:
        raise ValueError(
            "metric tracking requires measured depth aligned to exact color intrinsics"
        )
    metadata = frame.metadata
    mount, session, source_profile = (
        metadata.get(key) for key in ("mount_digest", "session_id", "profile_digest")
    )
    if any(not isinstance(v, str) or not v for v in (mount, session, source_profile)):
        raise ValueError("metric capture lacks exact source profile/mount/session identity")
    sources = []
    for stream in ("color", "depth"):
        source = metadata.get(f"{stream}_source")
        if not isinstance(source, Mapping):
            raise ValueError("metric capture lacks native stream clock provenance")
        sources.append(
            SourceStamp(
                stream,
                source.get("frame_number"),
                source.get("timestamp_ms"),
                source.get("timestamp_domain"),
            )
        )
    color = b"".join(
        frame.color[(row * frame.width + x) * 3 : (row * frame.width + x + width) * 3]
        for row in range(y, y + height)
    )
    units = frame.depth_units()
    depth = array(
        "f",
        (
            float(units[row * frame.width + column]) * frame.depth_scale_m
            for row in range(y, y + height)
            for column in range(x, x + width)
        ),
    )
    if depth.itemsize != 4:
        raise ValueError("platform lacks IEEE754 float32 metric ROI encoding")
    if sys.byteorder != "little":
        depth.byteswap()
    profile = stream_profile_of(frame, role=role, valid_min_m=valid_min_m, valid_max_m=valid_max_m)
    return MetricRoi(
        f"{frame.device_id}:{session}:{frame.sequence}",
        profile,
        mount,
        session,
        source_profile,
        Stamp(frame.monotonic_at, uncertainty_s=receipt_uncertainty_s),
        exposure_uncertainty_s,
        x,
        y,
        width,
        height,
        color,
        depth.tobytes(),
        "metric_rgbd",
        tuple(sources),
    )


def metric_track_point(
    roi: MetricRoi,
    pixel: Sequence[float],
    config: TrackingConfig,
) -> tuple[tuple[float, float, float], float]:
    """Independently deproject a bounded robust depth patch from immutable raw bytes.

    This owner-side check verifies numerical worker values against exact metric
    input. It neither tracks image identity nor supplies an independent witness.
    """
    if not isinstance(roi, MetricRoi) or not isinstance(config, TrackingConfig):
        raise ValueError("typed metric ROI and tracking policy required")
    x, y = finite_vector(pixel, 2)
    u, v = round(x - roi.x), round(y - roi.y)
    if not 4 <= x - roi.x < roi.width - 4 or not 4 <= y - roi.y < roi.height - 4:
        raise ValueError("metric track left its bounded depth patch")
    values = [
        struct.unpack_from("<f", roi.depth_m_f32le, 4 * (row * roi.width + column))[0]
        for row in range(v - 2, v + 3)
        for column in range(u - 2, u + 3)
    ]
    if any(not math.isfinite(z) or z < 0 for z in values):
        raise ValueError("metric track depth patch contains nonfinite/negative values")
    values = [
        z
        for z in values
        if z > max(0, roi.profile.valid_min_m)
        and (not roi.profile.valid_max_m or z <= roi.profile.valid_max_m)
    ]
    if len(values) < 13:
        raise ValueError("metric track has insufficient valid raw depth")

    # The wire depth contract is float32; preserve its rounding for even medians
    # and depth differences without importing a numerical library on the owner.
    def f32(value: float) -> float:
        return struct.unpack("<f", struct.pack("<f", value))[0]

    z = f32(median(values))
    mad = f32(median(abs(f32(value - z)) for value in values))
    if mad > config.max_depth_mad_m:
        raise ValueError("metric track raw depth dispersion exceeds the frozen policy")
    profile = roi.profile
    if any(profile.distortion_coeffs) and profile.distortion_model not in (
        "brown_conrady",
        "inverse_brown_conrady",
    ):
        raise ValueError("exact metric track distortion deprojection unavailable")
    intrinsics = CameraIntrinsics(
        profile.width,
        profile.height,
        profile.fx_px,
        profile.fy_px,
        profile.ppx_px,
        profile.ppy_px,
        profile.distortion_model,
        profile.distortion_coeffs,
    )
    try:
        point = finite_vector(
            tuple(value / 1000 for value in intrinsics.deproject(x, y, z * 1000)), 3
        )
    except (ZeroDivisionError, OverflowError) as exc:
        raise ValueError("singular metric track deprojection") from exc
    return point, max(config.depth_floor_m, 1.4826 * mad)


__all__ = ["frame_evidence", "metric_roi", "metric_track_point", "si_transform", "tracked_motion"]
