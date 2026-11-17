# Copyright (c) Alibaba, Inc. and its affiliates.
"""Cached posture-specific reach guidance, not a new motion admission gate.

A workspace box is a safety boundary, not an IK workspace. Sample a vertical
radial slice with the calibrated TCP and margined joint bounds so callers can
choose useful hover heights. Each band belongs to the published azimuth and
yaw; neither axisymmetry nor reach at another orientation is assumed. Missing
bands are unknown, not proof of impossibility. Every actual move still passes
workspace, orientation, seeded path IK, approval and the firmware's checks.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

from leapflow.hardware.skills.adapt.spec import IkSettings
from leapflow.hardware.skills.frames import FrameGraph
from leapflow.hardware.skills.geometry import Pose, Transform, WorkspaceBox
from leapflow.hardware.skills.kinematics import DhChain
from leapflow.hardware.skills.planning import Deadline
from leapflow.hardware.skills.protocol import FAILURE_TIMEOUT, SkillError
from leapflow.hardware.skills.reach import Bounds, LineFollower, ReachSettings, joint_bounds
from leapflow.hardware.skills.workcell import WorkcellSpec, joint_channels


@dataclass(frozen=True)
class ReachBand:
    radius_mm: float
    z_min_mm: float | None
    z_max_mm: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "radius_mm": round(self.radius_mm, 1),
            "z_min_mm": None if self.z_min_mm is None else math.ceil(self.z_min_mm * 10) / 10,
            "z_max_mm": None if self.z_max_mm is None else math.floor(self.z_max_mm * 10) / 10,
        }


@dataclass(frozen=True)
class ReachEnvelope:
    """A sampled top-down slice, bounded by the cell's measured safe box."""

    bands: tuple[ReachBand, ...]
    azimuth_deg: float
    yaw_deg: float
    step_mm: float
    posture: str = "top_down"
    ik_iterations: int = 50

    def radius_range(self) -> tuple[float, float]:
        return self.bands[0].radius_mm, self.bands[-1].radius_mm

    def z_range(self, radius: float) -> tuple[float, float] | None:
        """Conservative intersection of adjacent samples; never bridge unknown bands."""
        for index, band in enumerate(self.bands):
            if math.isclose(radius, band.radius_mm, abs_tol=1e-6):
                return self._intersection((band,))
            if index and self.bands[index - 1].radius_mm < radius < band.radius_mm:
                return self._intersection((self.bands[index - 1], band))
        return None

    @staticmethod
    def _intersection(bands: Sequence[ReachBand]) -> tuple[float, float] | None:
        if any(b.z_min_mm is None or b.z_max_mm is None for b in bands):
            return None
        low = max(float(b.z_min_mm) for b in bands)  # type: ignore[arg-type]
        high = min(float(b.z_max_mm) for b in bands)  # type: ignore[arg-type]
        return (low, high) if low <= high else None

    def to_dict(self) -> dict[str, Any]:
        common = self._intersection(self.bands)
        return {
            "posture": self.posture,
            "advisory": True,
            "azimuth_deg": round(self.azimuth_deg, 1),
            "yaw_deg": round(self.yaw_deg, 1),
            "sample_step_mm": self.step_mm,
            "ik_iteration_budget": self.ik_iterations,
            "radius_range_mm": [round(v, 1) for v in self.radius_range()],
            "bands": [b.to_dict() for b in self.bands],
            "transit_z_mm": None if common is None else math.floor(common[1] * 10) / 10,
            "detail": (
                "Sampled top-down guidance at the stated azimuth/yaw, not the workspace box "
                "or a guarantee of transit clearance. Null bands are unknown. Choose heights "
                "inside adjacent bands; motion.sequence checks the actual posture and path."
            ),
        }


def envelope_view(arm: Any, spec: WorkcellSpec, frames: FrameGraph) -> dict[str, Any]:
    """The same declaration-derived guidance for the catalog and arm.state."""
    tcp, box = frames.transform("tcp"), frames.workspace
    if tcp is None or box is None or any(name in frames.stale for name in ("tcp", "workspace")):
        return {
            "available": False,
            "advisory": True,
            "detail": "TCP/workspace calibration is missing or stale",
        }
    names = joint_channels(arm)
    settings = ReachSettings.from_limits(spec.limits)
    try:
        envelope = envelope_for(
            arm,
            names,
            joint_bounds(arm, names, settings.margin_deg),
            tcp,
            box,
            spec.limits,
            spec.home,
        )
    except SkillError as error:
        if error.failure_code != FAILURE_TIMEOUT:
            raise
        return {"available": False, "advisory": True,
                "detail": "Reach guidance computation exceeded its declared time budget; reach remains unknown"}
    except ValueError as error:
        return {"available": False, "advisory": True, "detail": str(error)}
    if envelope is None:
        return {
            "available": False,
            "advisory": True,
            "detail": "No top-down slice found with the declared kinematics",
        }
    return {"available": True, "source": "declared_kinematics", **envelope.to_dict()}


def envelope_for(
    arm: Any,
    names: Sequence[str],
    bounds: Bounds,
    flange_tcp: Transform,
    box: WorkspaceBox,
    limits: Mapping[str, float],
    home: Mapping[str, float],
) -> ReachEnvelope | None:
    """Resolve a declaration-only snapshot; no device reads and no motion."""
    raw = getattr(getattr(arm, "kinematics", None), "dh", None) or {}
    if not raw:
        return None
    chain = DhChain.from_mapping(raw)
    if chain.dof != len(names):
        return None
    seed = tuple(
        min(high, max(low, float(home.get(name, (low + high) / 2))))
        for name, (lo, hi) in zip(names, bounds)
        for low, high in [(lo if lo is not None else -180.0, hi if hi is not None else 180.0)]
    )
    return build_envelope(
        chain,
        bounds,
        flange_tcp,
        box,
        ReachSettings.from_limits(limits),
        float(limits["reach_envelope_step_mm"]),
        seed,
        int(limits["reach_envelope_yaw_samples"]),
        float(limits["reach_envelope_timeout_s"]),
        int(limits["reach_envelope_ik_iterations"]),
    )


@lru_cache(maxsize=32)
def build_envelope(
    chain: DhChain,
    bounds: Bounds,
    flange_tcp: Transform,
    box: WorkspaceBox,
    settings: ReachSettings,
    step_mm: float,
    seed: tuple[float, ...],
    yaw_samples: int,
    budget_s: float = math.inf,
    ik_iterations: int = 50,
) -> ReachEnvelope | None:
    """Sweep a radial slice; only completed results are cached, including the resource basis."""
    line = LineFollower(chain, bounds, flange_tcp, settings, Deadline(time.monotonic() + budget_s),
                        ik_settings=IkSettings(max_iterations=ik_iterations))
    azimuth = math.atan2((box.y_min + box.y_max) / 2, (box.x_min + box.x_max) / 2)
    cosine, sine = math.cos(azimuth), math.sin(azimuth)
    low, high = 0.0, math.inf
    for direction, minimum, maximum in ((cosine, box.x_min, box.x_max), (sine, box.y_min, box.y_max)):
        if abs(direction) < 1e-12:
            if not minimum <= 0 <= maximum:
                return None
            continue
        entry, exit = sorted((minimum / direction, maximum / direction))
        low, high = max(low, entry), min(high, exit)
    if high <= low:
        return None
    radii = _samples(low, high, step_mm)
    center_r = math.hypot((box.x_min + box.x_max) / 2, (box.y_min + box.y_max) / 2)
    anchor = _anchor(line, center_r, cosine, sine, box, step_mm, seed, yaw_samples,
                     first_feasible=math.isfinite(budget_s))
    if anchor is None:
        return None
    yaw, q = anchor
    bands = []
    for radius in radii:
        found = _height_anchor(line, radius, cosine, sine, yaw, box, step_mm, q)
        if found is None:
            bands.append(ReachBand(radius, None, None))
            continue
        pose, q = found
        bottom = line.follow(pose, Pose(pose.x, pose.y, box.z_min, 180, 0, yaw), q, continuous=True)
        top = line.follow(pose, Pose(pose.x, pose.y, box.z_max, 180, 0, yaw), q, continuous=True)
        bands.append(
            ReachBand(
                radius,
                pose.z + (box.z_min - pose.z) * bottom.reached,
                pose.z + (box.z_max - pose.z) * top.reached,
            )
        )
    return ReachEnvelope(tuple(bands), math.degrees(azimuth), yaw, step_mm, ik_iterations=ik_iterations)


def _samples(low: float, high: float, step: float) -> tuple[float, ...]:
    count = max(1, math.ceil((high - low) / step))
    return tuple(low + (high - low) * index / count for index in range(count + 1))


def _height_anchor(
    line: LineFollower,
    radius: float,
    cosine: float,
    sine: float,
    yaw: float,
    box: WorkspaceBox,
    step: float,
    seed: tuple[float, ...],
) -> tuple[Pose, tuple[float, ...]] | None:
    heights = _samples(box.z_min, box.z_max, step)
    for z in sorted(heights, key=lambda z: abs(z - (box.z_min + box.z_max) / 2)):
        pose = Pose(radius * cosine, radius * sine, z, 180, 0, yaw)
        q = line.solve(pose, seed)
        if q is not None:
            return pose, q
    return None


def _anchor(
    line: LineFollower,
    radius: float,
    cosine: float,
    sine: float,
    box: WorkspaceBox,
    step: float,
    seed: tuple[float, ...],
    yaw_samples: int,
    *, first_feasible: bool = False,
) -> tuple[float, tuple[float, ...]] | None:
    candidates = []
    yaws = [-180 + 360 * index / yaw_samples for index in range(yaw_samples)]
    if first_feasible:
        yaws.sort(key=abs)
    for yaw in yaws:
        found = _height_anchor(line, radius, cosine, sine, yaw, box, step, seed)
        if found is not None:
            _, q = found
            if first_feasible:
                return yaw, q
            comfort = min(
                min(v - lo if lo is not None else math.inf, hi - v if hi is not None else math.inf)
                for v, (lo, hi) in zip(q, line.bounds)
            )
            candidates.append((comfort, yaw, q))
    if not candidates:
        return None
    _, yaw, q = max(candidates)
    return yaw, q


__all__ = ["ReachBand", "ReachEnvelope", "build_envelope", "envelope_for", "envelope_view"]
