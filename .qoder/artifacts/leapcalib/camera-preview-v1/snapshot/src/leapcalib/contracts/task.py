# Copyright (c) Alibaba, Inc. and its affiliates.
"""Four-DoF task observations and explicitly local validity envelopes.

The four coordinates are measured xyz (metres) and yaw (radians), never commanded
arrival or pixels mislabeled as metres. A target or domain change is retained in
the evidence and rejected by the method, not silently relabeled by a producer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from leapcalib.contracts.identity import ModelIdentity, StreamProfile, digest_of
from leapcalib.contracts.motion import finite_vector, require_measured_flags
from leapcalib.contracts.observations import CALIBRATION_PROVENANCE, Stamp
from leapcalib.contracts.units import canonical_number


@dataclass(frozen=True)
class TaskState:
    """Measured endpoint and task feature, sharing a node clock and explicit identity."""

    state_id: str
    stamp: Stamp
    pose_m_rad: tuple[float, float, float, float]
    feature_m_rad: tuple[float, float, float, float]
    feature_id: str
    domain_id: str
    settled: bool
    measured: bool
    uncertainty_m_rad: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if any(
            not isinstance(v, str) or not v.strip()
            for v in (self.state_id, self.feature_id, self.domain_id)
        ):
            raise ValueError("task state needs source, feature and domain identities")
        for name in ("pose_m_rad", "feature_m_rad", "uncertainty_m_rad"):
            object.__setattr__(self, name, finite_vector(getattr(self, name), 4))
        if any(v < 0 for v in self.uncertainty_m_rad):
            raise ValueError("task uncertainty cannot be negative")
        require_measured_flags(settled=self.settled, measured=self.measured)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "stamp": self.stamp.to_dict(),
            "pose_m_rad": list(self.pose_m_rad),
            "feature_m_rad": list(self.feature_m_rad),
            "feature_id": self.feature_id,
            "domain_id": self.domain_id,
            "settled": self.settled,
            "measured": self.measured,
            "uncertainty_m_rad": list(self.uncertainty_m_rad),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TaskState:
        return cls(
            state_id=data["state_id"],
            stamp=Stamp.from_mapping(data["stamp"]),
            pose_m_rad=data["pose_m_rad"],
            feature_m_rad=data["feature_m_rad"],
            feature_id=data["feature_id"],
            domain_id=data["domain_id"],
            settled=data["settled"],
            measured=data["measured"],
            uncertainty_m_rad=data["uncertainty_m_rad"],
        )


@dataclass(frozen=True)
class TaskTransition:
    """An actual measured response sample; timestamps are not exposure claims."""

    observation_id: str
    stream: StreamProfile
    model: ModelIdentity
    first: TaskState
    second: TaskState
    provenance: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.observation_id, str)
            or not self.observation_id.strip()
            or self.first.state_id == self.second.state_id
        ):
            raise ValueError("a task transition needs distinct observation identities")
        if not isinstance(self.provenance, str) or self.provenance not in CALIBRATION_PROVENANCE:
            raise ValueError("task transitions need metric or synthetic provenance")
        if self.second.stamp.delta_s(self.first.stamp) <= 0:
            raise ValueError("task transition stamps must increase on the same clock")

    @property
    def stamp(self) -> Stamp:
        return self.second.stamp

    @property
    def source_ids(self) -> tuple[str, str]:
        return self.first.state_id, self.second.state_id

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "task_transition",
            "observation_id": self.observation_id,
            "stream": self.stream.to_dict(),
            "model": self.model.to_dict(),
            "first": self.first.to_dict(),
            "second": self.second.to_dict(),
            "provenance": self.provenance,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TaskTransition:
        return cls(
            observation_id=data["observation_id"],
            stream=StreamProfile.from_mapping(data["stream"]),
            model=ModelIdentity(**data["model"]),
            first=TaskState.from_mapping(data["first"]),
            second=TaskState.from_mapping(data["second"]),
            provenance=data["provenance"],
        )


@dataclass(frozen=True)
class TaskDomain:
    """Caller-supplied qualified local box; an optimizer cannot widen it.

    Yaw is unwrapped around the box's centre for membership checks. Clearances
    and tolerances come from task geometry; naming this domain does not qualify
    any hardware, executor, joint limits, or trajectory.
    """

    domain_id: str
    feature_id: str
    lower_m_rad: tuple[float, float, float, float]
    upper_m_rad: tuple[float, float, float, float]
    max_step_m_rad: tuple[float, float, float, float]
    tolerance_m_rad: tuple[float, float, float, float]
    clearance_m: float
    lever_arm_m: float
    speed_m_s: float
    rotation_rate_rad_s: float
    max_age_s: float

    def __post_init__(self) -> None:
        if any(not isinstance(v, str) or not v.strip() for v in (self.domain_id, self.feature_id)):
            raise ValueError("a task domain needs explicit identities")
        for name in ("lower_m_rad", "upper_m_rad", "max_step_m_rad", "tolerance_m_rad"):
            object.__setattr__(self, name, finite_vector(getattr(self, name), 4))
        if any(a >= b for a, b in zip(self.lower_m_rad, self.upper_m_rad)):
            raise ValueError("task envelope bounds must increase")
        if self.upper_m_rad[3] - self.lower_m_rad[3] > 2 * math.pi:
            raise ValueError("a local yaw envelope cannot span more than one revolution")
        if any(v <= 0 for v in self.max_step_m_rad + self.tolerance_m_rad):
            raise ValueError("task steps and tolerances must be positive")
        for name in ("clearance_m", "lever_arm_m", "speed_m_s", "rotation_rate_rad_s", "max_age_s"):
            object.__setattr__(self, name, canonical_number(getattr(self, name)))
        if (
            min(self.clearance_m, self.max_age_s) <= 0
            or min(self.lever_arm_m, self.speed_m_s, self.rotation_rate_rad_s) < 0
        ):
            raise ValueError("task clearance/age must be positive; rates/lever cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in vars(self).items()
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TaskDomain:
        return cls(**data)


__all__ = ["TaskDomain", "TaskState", "TaskTransition"]
