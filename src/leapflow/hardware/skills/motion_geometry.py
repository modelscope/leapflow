# Copyright (c) Alibaba, Inc. and its affiliates.
"""Detached motion geometry shared by governed commands and pure planning workers."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

from leapflow.hardware.context import Envelope
from leapflow.hardware.skills.geometry import Pose, Transform, WorkspaceBox, tool_cone_deg
from leapflow.hardware.skills.kinematics import DhChain
from leapflow.hardware.skills.planning import PlanningCheckpoint
from leapflow.hardware.skills.protocol import (
    FAILURE_CALIBRATION_MISSING, FAILURE_DEVICE_FAULT, FAILURE_INVALID_ARGS, FAILURE_SKILL_UNAVAILABLE,
    FAILURE_OUT_OF_WORKSPACE, FAILURE_UNREACHABLE, ArmState, SkillError,
)
from leapflow.hardware.skills.reach import Bounds, ReachSettings, ReachTracker

CONE_EPSILON_DEG = 1e-6


@dataclass(frozen=True)
class MotionGeometry:
    arm_id: str
    box: WorkspaceBox | None
    region_error: str
    flange_tcp: Transform
    pose_envelopes: tuple[tuple[str, Envelope], ...]
    joint_envelopes: tuple[tuple[str, Envelope], ...]
    bounds: Bounds
    chain: DhChain | None
    chain_error: str
    cone_deg: float
    fk_mm: float
    fk_deg: float
    joint_step_deg: float
    reach_settings: ReachSettings
    paths: tuple[str, ...]
    gripper_tolerance: float | None
    checkpoint: PlanningCheckpoint | None = None
    metric_bounds: Bounds | None = None

    def bind(self, checkpoint: PlanningCheckpoint) -> MotionGeometry:
        return replace(self, checkpoint=checkpoint)

    def check(self) -> None:
        if self.checkpoint is not None:
            self.checkpoint.check()

    @property
    def gripper_tolerance_mm(self) -> float:
        if self.gripper_tolerance is None:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, "this workcell binds no gripper")
        return self.gripper_tolerance

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.joint_envelopes)

    def flange_to_tcp(self, flange: Pose) -> Pose:
        return Transform.from_pose(flange).compose(self.flange_tcp).to_pose()

    def tcp_to_flange(self, tcp: Pose) -> Pose:
        return Transform.from_pose(tcp).compose(self.flange_tcp.inverse()).to_pose()

    def check_target(self, pose: Pose) -> None:
        self.check_reach(pose)
        self.check_orientation(pose)

    def check_reach(self, pose: Pose) -> None:
        self.check()
        if self.box is None:
            raise SkillError(FAILURE_CALIBRATION_MISSING, self.region_error)
        violation = self.box.violation(pose.position)
        if violation is not None:
            raise SkillError(FAILURE_OUT_OF_WORKSPACE, f"target leaves the workspace: {violation}",
                             result={"limit": self.box.exceeded(pose.position)})
        flange = self.tcp_to_flange(pose)
        for (name, envelope), value in zip(self.pose_envelopes, flange.as_tuple()):
            if not envelope.contains(value):
                raise SkillError(FAILURE_UNREACHABLE,
                    f"flange {name}={value:.1f} is outside the arm's declared envelope")

    def check_orientation(self, pose: Pose) -> None:
        self.require_cone(pose, self.cone_deg)

    def require_cone(self, pose: Pose, ceiling_deg: float) -> None:
        cone = tool_cone_deg(pose)
        if cone > ceiling_deg + CONE_EPSILON_DEG:
            raise SkillError(FAILURE_UNREACHABLE,
                f"tool would tilt {cone:.1f} deg from straight down; the workcell allows {ceiling_deg:.1f}",
                result={"limit": {"quantity": "tool_tilt", "unit": "deg",
                                  "value": round(cone, 1), "max": round(ceiling_deg, 1)}})

    def kinematics(self) -> DhChain:
        if self.chain is None:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, self.chain_error)
        return self.chain

    def check_state(self, state: ArmState) -> None:
        """Nonfinite telemetry is a fault, not an optional-kinematics degradation."""
        values = (*state.tcp.as_tuple(), *state.flange.as_tuple(), *state.joints.values())
        if not all(math.isfinite(value) for value in values):
            raise SkillError(FAILURE_DEVICE_FAULT, f"arm {self.arm_id!r} reported nonfinite pose or joints")

    def check_kinematics(self, state: ArmState) -> None:
        self.check()
        self.check_state(state)
        missing = [name for name in self.names if name not in state.joints]
        if missing:
            raise SkillError(FAILURE_DEVICE_FAULT, f"arm {self.arm_id!r} did not report {', '.join(missing)}")
        chain = self.kinematics()
        mm, deg = chain.agreement([state.joints[name] for name in self.names], state.flange)
        if mm > self.fk_mm or deg > self.fk_deg:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE,
                f"the arm's kinematics.dh puts the flange {mm:.1f} mm / {deg:.1f} deg "
                f"from where the firmware reports it (tolerance {self.fk_mm:g} mm / "
                f"{self.fk_deg:g} deg); fix the table before moving joints")

    def check_joints(self, targets: Mapping[str, float], current: Mapping[str, float] | None,
                     *, observe: Callable[[Sequence[float]], None] | None = None) -> Pose:
        self.check()
        if not targets:
            raise SkillError(FAILURE_INVALID_ARGS, "joint targets are empty")
        unknown = sorted(set(targets) - set(self.names))
        if unknown:
            raise SkillError(FAILURE_INVALID_ARGS,
                             f"{', '.join(unknown)} are not joint channels of {self.arm_id!r}")
        numeric = {name: float(value) for name, value in targets.items()}
        if not all(math.isfinite(value) for value in numeric.values()):
            raise SkillError(FAILURE_INVALID_ARGS, "joint targets must be finite")
        for name, envelope in self.joint_envelopes:
            if name in numeric and not envelope.contains(numeric[name]):
                raise SkillError(FAILURE_UNREACHABLE,
                    f"joint {name}={numeric[name]:g} is outside the arm's declared envelope")
        start = {} if current is None else dict(current)
        missing = [name for name in self.names if name not in numeric and name not in start]
        if missing:
            code = FAILURE_INVALID_ARGS if current is None else FAILURE_DEVICE_FAULT
            detail = "must be targeted" if current is None else "was not reported by the arm"
            raise SkillError(code, f"joint {', '.join(missing)} {detail}")
        end = tuple(numeric[name] if name in numeric else start[name] for name in self.names)
        chain = self.kinematics()
        begin = end if current is None else tuple(float(start[name]) for name in self.names)
        count = 1 if current is None else max(1, math.ceil(max(abs(b - a) for a, b in zip(begin, end)) / self.joint_step_deg))
        started_at = None if current is None else self.flange_to_tcp(chain.forward(begin))
        pose = self.flange_to_tcp(chain.forward(end))
        for index in range(1, count + 1):
            self.check()
            q = tuple(a + (b - a) * index / count for a, b in zip(begin, end))
            pose = self.flange_to_tcp(chain.forward(q))
            self.check_joint_target(pose, start=started_at)
            if observe is not None:
                observe(q)
        return pose

    def joint_tilt_slack_deg(self) -> float:
        return sum(float(envelope.tolerance or 0) for _, envelope in self.joint_envelopes)

    def check_joint_target(self, pose: Pose, *, start: Pose | None) -> None:
        if start is None:
            self.check_target(pose)
        else:
            self.check_carried_target(pose, start)

    def check_carried_target(self, pose: Pose, start: Pose) -> None:
        """Reach in full; attitude only as far as the posture it is carried from.

        A command that keeps its attitude does not state where the tool points, so
        the cone is a no-further-tilt ceiling here rather than an admission test.
        Otherwise a posture the cell would never choose -- one a calibration ring
        or a halt left behind -- could not be left at all.
        """
        self.check_reach(pose)
        self.require_cone(pose, max(self.cone_deg, tool_cone_deg(start) + self.joint_tilt_slack_deg()))

    def reach_tracker(self, state: ArmState) -> ReachTracker:
        self.check_state(state)
        try:
            self.check_kinematics(state)
        except SkillError as error:
            if error.failure_code != FAILURE_SKILL_UNAVAILABLE and error.failure_code != FAILURE_DEVICE_FAULT:
                raise
            return ReachTracker.unverified(state.tcp, error.message)
        return ReachTracker(tcp=state.tcp, chain=self.kinematics(), names=self.names,
            joints=state.joints, bounds=self.bounds, metric_bounds=self.metric_bounds, flange_tcp=self.flange_tcp,
            settings=self.reach_settings, checkpoint=self.checkpoint)
