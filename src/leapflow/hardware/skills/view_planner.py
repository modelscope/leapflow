# Copyright (c) Alibaba, Inc. and its affiliates.
"""Pure, replaceable viewpoint proposal geometry; proposals grant no motion authority."""
from __future__ import annotations

import logging
import math
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from leapflow.hardware.rgbd import CameraIntrinsics
from leapflow.hardware.skills.geometry import Plane, Pose, Transform, Vec3, dot
from leapflow.hardware.skills.protocol import SIDE_EFFECT_NONE, LocatedObject, SkillError, SkillOutcome
from leapflow.hardware.skills.planning import Evaluation, PlanningCheckpoint

VIEW_PLANNER_ENTRY_POINT_GROUP = 'leapflow.hardware.view_planners'
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ViewGoal:
    obj: LocatedObject
    table: Plane
    camera: str
    flange_camera: Transform
    flange_tcp: Transform
    intrinsics: CameraIntrinsics
    near_mm: float
    far_mm: float
    max_candidates: int


@dataclass(frozen=True)
class ViewCandidate:
    tcp: Pose
    camera: str
    camera_to_base: Transform
    feature: str
    score: float


@runtime_checkable
class ViewPlanner(Protocol):
    @property
    def name(self) -> str: ...
    def candidates(self, goal: ViewGoal) -> tuple[ViewCandidate, ...]: ...


class RingViewPlanner:
    """A support-plane ring of optical poses, independent of tool-axis convention."""

    name = 'ring'

    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = set(config) - {'tilt_deg', 'range_fraction', 'azimuths'}
        if unknown:
            raise ValueError(f'unknown ring planner settings: {sorted(unknown)}')
        self.tilt = config.get('tilt_deg', 20.)
        self.fraction = config.get('range_fraction', .5)
        self.azimuths = config.get('azimuths', 8)
        if (type(self.tilt) not in (int, float) or type(self.fraction) not in (int, float)
                or type(self.azimuths) is not int or not math.isfinite(self.tilt)
                or not math.isfinite(self.fraction) or not 0 <= self.tilt < 90
                or not 0 < self.fraction < 1 or not 1 <= self.azimuths <= 64):
            raise ValueError('invalid ring viewpoint geometry')

    def candidates(self, goal: ViewGoal) -> tuple[ViewCandidate, ...]:
        if not 0 < goal.near_mm < goal.far_mm:
            raise ValueError('active camera needs a valid depth interval')
        normal = goal.table.normal
        reference = (1., 0., 0.) if abs(normal[0]) < .9 else (0., 1., 0.)
        u = _unit(cross(normal, reference))
        v = cross(normal, u)
        radius = goal.near_mm + (goal.far_mm-goal.near_mm)*self.fraction
        tilt = math.radians(self.tilt)
        target = (goal.obj.center[0], goal.obj.center[1], goal.obj.center[2]+goal.obj.height/2)
        result = []
        for i in range(min(self.azimuths, goal.max_candidates)):
            azimuth = 2*math.pi*i/self.azimuths
            direction = tuple(normal[j]*math.cos(tilt) + math.sin(tilt)*(u[j]*math.cos(azimuth)+v[j]*math.sin(azimuth))
                              for j in range(3))
            origin = tuple(target[j]+radius*direction[j] for j in range(3))
            z = _unit(tuple(target[j]-origin[j] for j in range(3)))
            x = _unit(cross(u if abs(dot(z, u)) < .9 else v, z))
            y = cross(z, x)
            rotation = tuple(tuple(axis[j] for axis in (x, y, z)) for j in range(3))
            camera = Transform(rotation, origin)
            tcp = camera.compose(goal.flange_camera.inverse()).compose(goal.flange_tcp).to_pose()
            result.append(ViewCandidate(tcp, goal.camera, camera, f'{goal.camera}:ring:{i}', 1.))
        return tuple(result)


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _unit(vector: Vec3) -> Vec3:
    length = math.sqrt(dot(vector, vector))
    if length <= 1e-12:
        raise ValueError('viewpoint has a degenerate axis')
    return tuple(v/length for v in vector)


@dataclass(frozen=True)
class ViewProposalJob:
    """Optional planner code is isolated in the same bounded calculation workers."""
    name: str
    config_json: str
    goal: ViewGoal

    def evaluate(self, checkpoint: PlanningCheckpoint) -> Evaluation:
        checkpoint.check()
        factory = discover_view_planners().get(self.name)
        if factory is None:
            raise SkillError('skill_unavailable', 'declared view planner is not installed')
        try:
            planner = factory(json.loads(self.config_json))
            if not isinstance(planner, ViewPlanner):
                raise ValueError('view planner does not implement the proposal contract')
            offered = planner.candidates(self.goal)
            if not isinstance(offered, tuple) or len(offered) > self.goal.max_candidates:
                raise ValueError('view planner must return a bounded tuple of candidates')
            features: set[str] = set()
            for p in offered:
                if (not isinstance(p, ViewCandidate) or not isinstance(p.tcp, Pose)
                        or not isinstance(p.camera_to_base, Transform) or p.camera != self.goal.camera
                        or len(p.camera_to_base.translation) != 3
                        or not isinstance(p.feature, str) or not 0 < len(p.feature) <= 128
                        or type(p.score) not in (int, float) or not math.isfinite(p.score)
                        or not all(math.isfinite(v) for v in p.tcp.as_tuple())
                        or not all(math.isfinite(v) for row in p.camera_to_base.rotation for v in row)
                        or not all(math.isfinite(v) for v in p.camera_to_base.translation)):
                    raise ValueError('view planner returned invalid candidate geometry')
                rows = p.camera_to_base.rotation
                if (len(rows) != 3 or any(len(row) != 3 for row in rows)
                        or any(abs(dot(rows[i], rows[j]) - float(i == j)) > 1e-6
                               for i in range(3) for j in range(3))
                        or dot(cross(rows[0], rows[1]), rows[2]) < 1-1e-6
                        or p.feature in features):
                    raise ValueError('view planner returned a nonrigid transform or duplicate feature')
                features.add(p.feature)
        except (TypeError, ValueError) as exc:
            raise SkillError('invalid_args', f'view planner refused geometry or configuration: {exc}') from exc
        checkpoint.check()
        return Evaluation(SkillOutcome.succeeded({'candidates': [
            {'tcp': p.tcp.to_dict(), 'camera': p.camera,
             'camera_to_base': p.camera_to_base.to_pose().to_dict(),
             'feature': p.feature, 'score': p.score} for p in offered]}, side_effect_state=SIDE_EFFECT_NONE))


ViewPlannerFactory = Callable[[Mapping[str, Any]], ViewPlanner]


def discover_view_planners() -> dict[str, ViewPlannerFactory]:
    from importlib.metadata import entry_points
    found: dict[str, ViewPlannerFactory] = {'ring': RingViewPlanner}
    for entry in entry_points(group=VIEW_PLANNER_ENTRY_POINT_GROUP):
        if entry.name in found:
            if entry.value != 'leapflow.hardware.skills.view_planner:RingViewPlanner':
                logger.warning('view planner %r already registered; incumbent retained', entry.name)
            continue
        try:
            factory = entry.load()
            if not callable(factory):
                raise TypeError('view planner factory is not callable')
            found[entry.name] = factory
        except Exception:  # noqa: BLE001 - optional discovery must not break unrelated skills
            logger.warning('view planner %r failed to load', entry.name, exc_info=True)
    return found
