# Copyright (c) Alibaba, Inc. and its affiliates.
"""Measured encoder response contracts; pose-response v1 remains unchanged."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from leapcalib.contracts.identity import ModelIdentity, StreamProfile, digest_of
from leapcalib.contracts.motion import finite_vector
from leapcalib.contracts.observations import CALIBRATION_PROVENANCE, CLOCK_NODE_MONOTONIC, Stamp
from leapcalib.contracts.task import TaskDomain, TaskState

RAW_TASK_VERSION = "leapcalib.raw_task.v1"
MAX_JOINTS = 32


def identities(values: Any, *, minimum: int = 1, maximum: int = MAX_JOINTS) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not minimum <= len(values) <= maximum:
        raise ValueError("identity count exceeds contract bounds")
    result = tuple(values)
    if any(not isinstance(v, str) or not v.strip() or len(v) > 256 for v in result):
        raise ValueError("identities must be bounded nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError("identities must be distinct")
    return result


@dataclass(frozen=True)
class MetricTaskFeature:
    """Camera-frame centroid and directed planar yaw, not world or tool geometry.

    Ordered, persistent track identities freeze the centroid's support and the
    yaw axis. Changing support creates a different feature; a tracker cannot
    silently drop a point and shift the reference. Association and a stationary
    background witness require separate owner evidence.
    """

    track_ids: tuple[str, ...]
    yaw_ids: tuple[str, str]
    domain_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_ids", identities(self.track_ids, minimum=12, maximum=2048))
        object.__setattr__(self, "yaw_ids", identities(self.yaw_ids, minimum=2, maximum=2))
        identities((self.domain_id,))
        if not set(self.yaw_ids) <= set(self.track_ids):
            raise ValueError("directed yaw endpoints must belong to the frozen task feature")

    @property
    def feature_id(self) -> str:
        return digest_of(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "leapcalib.metric_task_feature.v1",
            "track_ids": list(self.track_ids),
            "yaw_ids": list(self.yaw_ids),
            "domain_id": self.domain_id,
            "coordinates": "camera_centroid_m_directed_xy_yaw_rad",
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> MetricTaskFeature:
        if (
            not isinstance(data, Mapping)
            or set(data) != {"schema", "track_ids", "yaw_ids", "domain_id", "coordinates"}
            or data["schema"] != "leapcalib.metric_task_feature.v1"
            or data["coordinates"] != "camera_centroid_m_directed_xy_yaw_rad"
        ):
            raise ValueError("unknown metric task feature schema or coordinate semantics")
        return cls(data["track_ids"], data["yaw_ids"], data["domain_id"])


@dataclass(frozen=True)
class RawTaskState:
    task: TaskState
    joint_ids: tuple[str, ...]
    joints_rad: tuple[float, ...]
    joint_uncertainty_rad: tuple[float, ...]
    frame_ids: tuple[str, ...]
    scope_digest: str
    feedback_stamp: Stamp
    witness_valid: bool
    feedback_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task, TaskState) or not isinstance(self.feedback_stamp, Stamp):
            raise ValueError("raw state requires typed measured task and feedback stamps")
        object.__setattr__(self, "feedback_ids", identities(self.feedback_ids, maximum=16))
        object.__setattr__(self, "joint_ids", identities(self.joint_ids))
        object.__setattr__(self, "frame_ids", identities(self.frame_ids, maximum=16))
        identities((self.scope_digest,))
        for field in ("joints_rad", "joint_uncertainty_rad"):
            object.__setattr__(
                self, field, finite_vector(getattr(self, field), len(self.joint_ids))
            )
        if any(v < 0 for v in self.joint_uncertainty_rad):
            raise ValueError("encoder uncertainty cannot be negative")
        if type(self.witness_valid) is not bool:
            raise ValueError("witness validity must be an explicit boolean")
        if (
            self.feedback_stamp.domain != CLOCK_NODE_MONOTONIC
            or self.task.stamp.domain != CLOCK_NODE_MONOTONIC
        ):
            raise ValueError("feedback and frame receipts need one explicit node clock")

    @property
    def source_ids(self) -> tuple[str, ...]:
        return (self.task.state_id, *self.frame_ids, *self.feedback_ids)

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "joint_ids": list(self.joint_ids),
            "joints_rad": list(self.joints_rad),
            "joint_uncertainty_rad": list(self.joint_uncertainty_rad),
            "frame_ids": list(self.frame_ids),
            "scope_digest": self.scope_digest,
            "feedback_stamp": self.feedback_stamp.to_dict(),
            "witness_valid": self.witness_valid,
            "feedback_ids": list(self.feedback_ids),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RawTaskState:
        return cls(
            **{
                **data,
                "task": TaskState.from_mapping(data["task"]),
                "feedback_stamp": Stamp.from_mapping(data["feedback_stamp"]),
            }
        )


@dataclass(frozen=True)
class RawTaskTransition:
    observation_id: str
    stream: StreamProfile
    model: ModelIdentity
    first: RawTaskState
    second: RawTaskState
    provenance: str

    def __post_init__(self) -> None:
        identities((self.observation_id,))
        if not isinstance(self.stream, StreamProfile) or not isinstance(self.model, ModelIdentity):
            raise ValueError("raw transition requires typed stream/model identities")
        if not isinstance(self.first, RawTaskState) or not isinstance(self.second, RawTaskState):
            raise ValueError("raw transition requires typed measured states")
        if len(self.first.joint_ids) != self.model.dof:
            raise ValueError("raw joint channels must match the model's declared dof")
        if self.first.joint_ids != self.second.joint_ids:
            raise ValueError("ordered encoder channels changed")
        if self.first.task.state_id == self.second.task.state_id:
            raise ValueError("transition needs distinct measured states")
        if self.second.task.stamp.delta_s(self.first.task.stamp) <= 0:
            raise ValueError("measured states must increase on the same node clock")
        if self.provenance not in CALIBRATION_PROVENANCE:
            raise ValueError("raw response needs explicit metric or synthetic provenance")

    @property
    def stamp(self) -> Stamp:
        return self.second.task.stamp

    @property
    def source_ids(self) -> tuple[str, ...]:
        return (*self.first.source_ids, *self.second.source_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "raw_task_transition",
            "schema": RAW_TASK_VERSION,
            "observation_id": self.observation_id,
            "stream": self.stream.to_dict(),
            "model": self.model.to_dict(),
            "first": self.first.to_dict(),
            "second": self.second.to_dict(),
            "provenance": self.provenance,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RawTaskTransition:
        expected = {
            "kind",
            "schema",
            "observation_id",
            "stream",
            "model",
            "first",
            "second",
            "provenance",
        }
        if (
            set(data) != expected
            or data.get("kind") != "raw_task_transition"
            or data.get("schema") != RAW_TASK_VERSION
        ):
            raise ValueError("unknown raw response schema or fields")
        return cls(
            data["observation_id"],
            StreamProfile.from_mapping(data["stream"]),
            ModelIdentity(**data["model"]),
            RawTaskState.from_mapping(data["first"]),
            RawTaskState.from_mapping(data["second"]),
            data["provenance"],
        )


@dataclass(frozen=True)
class RawTaskDomain:
    task: TaskDomain
    joint_ids: tuple[str, ...]
    reference_rad: tuple[float, ...]
    lower_rad: tuple[float, ...]
    upper_rad: tuple[float, ...]
    max_step_rad: tuple[float, ...]
    scope_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.task, TaskDomain):
            raise ValueError("raw domain requires a typed task domain")
        object.__setattr__(self, "joint_ids", identities(self.joint_ids))
        identities((self.scope_digest,))
        for name in ("reference_rad", "lower_rad", "upper_rad", "max_step_rad"):
            object.__setattr__(self, name, finite_vector(getattr(self, name), len(self.joint_ids)))
        if any(
            not lo <= q <= hi or lo >= hi
            for lo, q, hi in zip(self.lower_rad, self.reference_rad, self.upper_rad)
        ):
            raise ValueError("reference must be inside increasing raw bounds")
        if any(v <= 0 for v in self.max_step_rad):
            raise ValueError("raw joint step scales must be positive")

    def accepts(self, state: RawTaskState) -> bool:
        task = state.task
        centre = (self.task.lower_m_rad[3] + self.task.upper_m_rad[3]) / 2
        yaw = centre + (task.pose_m_rad[3] - centre + math.pi) % (2 * math.pi) - math.pi
        pose = (*task.pose_m_rad[:3], yaw)
        return (
            state.joint_ids == self.joint_ids
            and state.scope_digest == self.scope_digest
            and task.feature_id == self.task.feature_id
            and task.domain_id == self.task.domain_id
            and task.measured
            and task.settled
            and state.witness_valid
            and all(
                lo <= q <= hi for lo, q, hi in zip(self.lower_rad, state.joints_rad, self.upper_rad)
            )
            and all(
                lo <= q <= hi
                for lo, q, hi in zip(self.task.lower_m_rad, pose, self.task.upper_m_rad)
            )
        )

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "joint_ids": list(self.joint_ids),
            "reference_rad": list(self.reference_rad),
            "lower_rad": list(self.lower_rad),
            "upper_rad": list(self.upper_rad),
            "max_step_rad": list(self.max_step_rad),
            "scope_digest": self.scope_digest,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RawTaskDomain:
        return cls(**{**data, "task": TaskDomain.from_mapping(data["task"])})
