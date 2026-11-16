# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bounded fixed-orientation IK paths for a qualified node-local grasp cycle."""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Sequence

from leapflow.hardware.skills.adapt.ik import solve_ik
from leapflow.hardware.skills.adapt.spec import IkSettings
from leapflow.hardware.skills.geometry import Pose, Transform
from leapflow.hardware.skills.protocol import FAILURE_UNREACHABLE, SkillError


def check_path(motion: Any, state: Any, targets: Sequence[Pose]) -> tuple[dict[str, Any], ...]:
    """Solve every waypoint and check the actual joint-space path, without writes.

    Execution consumes these approved joint targets, not a second firmware IK
    branch. Fixed orientation also bounds stationary force-characterization scope.
    Workspace checks cover TCP paths, not an unmodeled arm swept-volume certificate.
    """
    limits = motion._spec.grasp_verify
    motion.check_kinematics(state)
    chain = motion._chain()
    names = tuple(motion._joints)
    envelopes = [motion._arm.channel(name).envelope for name in names]
    if any(not e.declared or e.min_value is None or e.max_value is None for e in envelopes):
        raise SkillError(FAILURE_UNREACHABLE, "bounded joint envelopes required for grasp IK")
    bounds = [(e.min_value, e.max_value) for e in envelopes]
    seed = dict(state.joints)
    previous = state.tcp
    result = []
    settings = IkSettings(tolerance_mm=min(IkSettings().tolerance_mm, limits["max_relative_error_mm"]),
                          tolerance_deg=min(IkSettings().tolerance_deg, limits["capture_drift_deg"]))
    # Flange rotation error displaces the TCP by the tool's lever arm. Budget
    # half the TCP tolerance to translation and half to that displacement.
    length = math.sqrt(sum(v * v for v in motion._flange_tcp.translation))
    rotation_budget = (math.degrees(2 * math.asin(min(1, settings.tolerance_mm / (4 * length))))
                       if length else settings.tolerance_deg)
    solver_settings = replace(settings, tolerance_mm=settings.tolerance_mm / 2,
                              tolerance_deg=min(settings.tolerance_deg, rotation_budget))
    for index, target in enumerate(targets):
        if previous.rotation_to_deg(target) > limits["capture_drift_deg"]:
            raise SkillError(FAILURE_UNREACHABLE, "qualified cycle needs fixed-orientation waypoints")
        count = max(1, math.ceil(previous.distance_mm(target) / limits["cycle_path_step_mm"]))
        if len(result) + count > limits["cycle_max_waypoints"]:
            raise SkillError(FAILURE_UNREACHABLE, "cycle exceeds bounded IK waypoint budget")
        begin = previous
        for step in range(1, count + 1):
            fraction = step / count
            position = tuple(a + (b - a) * fraction for a, b in zip(begin.position, target.position))
            # Short wrapped deltas preserve near-fixed orientation across ±180°.
            orientation = tuple(a + ((b - a + 180) % 360 - 180) * fraction
                                for a, b in zip(begin.as_tuple()[3:], target.as_tuple()[3:]))
            pose = Pose(*position, *orientation)
            motion.check_target(pose)
            try:
                joints = solve_ik(chain, Transform.from_pose(motion.tcp_to_flange(pose)),
                                  tuple(seed[name] for name in names), bounds, solver_settings)
            except ValueError as error:
                raise SkillError(FAILURE_UNREACHABLE, f"cycle IK: {error}") from error
            solved = dict(zip(names, joints))
            predicted = motion.check_joints(solved, seed)
            if (predicted.distance_mm(pose) > settings.tolerance_mm
                    or predicted.rotation_to_deg(pose) > settings.tolerance_deg):
                raise SkillError(FAILURE_UNREACHABLE, "IK waypoint disagrees with predicted TCP")
            result.append({"target_index": index, "pose": pose.to_dict(6), "joints": solved})
            seed, previous = solved, pose
    return tuple(result)
