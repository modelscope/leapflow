# Copyright (c) Alibaba, Inc. and its affiliates.
"""Pure node-side measured-data conversions; no acquisition or actuator ownership."""

from __future__ import annotations

import math
from typing import Any, Mapping

from leapcalib.contracts.identity import ModelIdentity
from leapcalib.contracts.motion import (
    MODE_FIXED,
    MODE_WRIST,
    MotionObservation,
    PointCorrespondence,
)
from leapcalib.contracts.observations import RgbdObservation, Stamp
from leapcalib.geometry.se3 import Transform
from leapflow.hardware.calibration.adapter import rgbd_observation, stream_profile_of
from leapflow.hardware.rgbd import RgbdFrame
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


__all__ = ["frame_evidence", "si_transform", "tracked_motion"]
