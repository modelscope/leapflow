# Copyright (c) Alibaba, Inc. and its affiliates.
"""Measured motion factors, not commanded trajectories or acquisition authority.

Fixed-camera pairs map first-view rigid-link points into the second view;
wrist-camera pairs map second-view static-scene points into the first view.
In both cases the recorded robot motion A and registered camera motion B obey
A X = X B. A comes from nominal/raw feedback, so X is conditional on that model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from leapcalib.contracts.identity import ModelIdentity, StreamProfile, digest_of
from leapcalib.contracts.observations import CALIBRATION_PROVENANCE, MAX_SAMPLES, Stamp
from leapcalib.contracts.units import canonical_number
from leapcalib.geometry.se3 import Transform

MODE_FIXED = "fixed_rigid_motion"
MODE_WRIST = "wrist_static_scene"
MOTION_MODES = frozenset({MODE_FIXED, MODE_WRIST})


def finite_vector(values: Any, dimension: int) -> tuple[float, ...]:
    """Canonicalize a vector without silently admitting booleans or nonfinite data."""
    result = tuple(canonical_number(v) for v in values)
    if len(result) != dimension:
        raise ValueError(f"expected {dimension} components")
    return result


def require_measured_flags(**flags: bool) -> None:
    """A truthy string is not evidence of measurement or stillness."""
    if any(type(value) is not bool for value in flags.values()):
        raise ValueError("measurement flags must be explicit booleans")


@dataclass(frozen=True)
class PointCorrespondence:
    """One independently tracked metric feature in two views, in camera coordinates.

    Pixel-grid position alone is not feature correspondence after motion. The
    producer must establish identity (e.g. tracked native landmark/static scene).
    No nearest-neighbor ICP or symmetry resolution is implied by this contract.
    """

    feature_id: str
    first_m: tuple[float, float, float]
    second_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.feature_id, str) or not self.feature_id.strip():
            raise ValueError("a correspondence needs a feature identity")
        object.__setattr__(self, "first_m", finite_vector(self.first_m, 3))
        object.__setattr__(self, "second_m", finite_vector(self.second_m, 3))

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "first_m": list(self.first_m),
            "second_m": list(self.second_m),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PointCorrespondence:
        return cls(data["feature_id"], data["first_m"], data["second_m"])


@dataclass(frozen=True)
class MotionObservation:
    """A settled metric correspondence factor with explicit direction and source IDs."""

    observation_id: str
    first_id: str
    second_id: str
    stream: StreamProfile
    model: ModelIdentity
    first_stamp: Stamp
    stamp: Stamp
    provenance: str
    mode: str
    scene_id: str
    robot_motion: Transform
    points: tuple[PointCorrespondence, ...]
    settled: bool
    max_speed_rad_s: float

    def __post_init__(self) -> None:
        if any(
            not isinstance(v, str) or not v.strip()
            for v in (self.observation_id, self.first_id, self.second_id, self.scene_id)
        ):
            raise ValueError("motion needs observation, source-frame and scene identities")
        if self.first_id == self.second_id or self.observation_id in self.source_ids:
            raise ValueError("motion must refer to two distinct source observations")
        if not isinstance(self.provenance, str) or self.provenance not in CALIBRATION_PROVENANCE:
            raise ValueError("motion requires metric or explicitly synthetic provenance")
        if not isinstance(self.mode, str) or self.mode not in MOTION_MODES:
            raise ValueError("unknown motion convention")
        if self.stream.role != ("fixed" if self.mode == MODE_FIXED else "wrist"):
            raise ValueError("motion convention disagrees with camera role")
        require_measured_flags(settled=self.settled)
        if self.stamp.delta_s(self.first_stamp) <= 0:
            raise ValueError("motion source stamps must increase on the same clock")
        object.__setattr__(self, "max_speed_rad_s", canonical_number(self.max_speed_rad_s))
        if self.max_speed_rad_s < 0:
            raise ValueError("a measured speed bound cannot be negative")
        object.__setattr__(self, "points", tuple(self.points))
        if len(self.points) > MAX_SAMPLES:
            raise ValueError("too many metric correspondences")
        ids = [p.feature_id for p in self.points]
        if len(ids) != len(set(ids)):
            raise ValueError("correspondence feature identities must be unique")
        if self.robot_motion.parent or self.robot_motion.child:
            raise ValueError("relative robot motion uses the mode's temporal frame convention")

    @property
    def source_ids(self) -> tuple[str, str]:
        return self.first_id, self.second_id

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "motion",
            "observation_id": self.observation_id,
            "first_id": self.first_id,
            "second_id": self.second_id,
            "stream": self.stream.to_dict(),
            "model": self.model.to_dict(),
            "first_stamp": self.first_stamp.to_dict(),
            "stamp": self.stamp.to_dict(),
            "provenance": self.provenance,
            "mode": self.mode,
            "scene_id": self.scene_id,
            "robot_motion": self.robot_motion.to_dict(),
            "points": [p.to_dict() for p in self.points],
            "settled": self.settled,
            "max_speed_rad_s": self.max_speed_rad_s,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> MotionObservation:
        return cls(
            observation_id=data["observation_id"],
            first_id=data["first_id"],
            second_id=data["second_id"],
            stream=StreamProfile.from_mapping(data["stream"]),
            model=ModelIdentity(**data["model"]),
            first_stamp=Stamp.from_mapping(data["first_stamp"]),
            stamp=Stamp.from_mapping(data["stamp"]),
            provenance=data["provenance"],
            mode=data["mode"],
            scene_id=data["scene_id"],
            robot_motion=Transform.from_mapping(data["robot_motion"]),
            points=tuple(PointCorrespondence.from_mapping(p) for p in data["points"]),
            settled=data["settled"],
            max_speed_rad_s=data["max_speed_rad_s"],
        )


__all__ = [
    "MODE_FIXED",
    "MODE_WRIST",
    "MOTION_MODES",
    "MotionObservation",
    "PointCorrespondence",
    "finite_vector",
    "require_measured_flags",
]
