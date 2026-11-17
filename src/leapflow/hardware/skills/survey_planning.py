# Copyright (c) Alibaba, Inc. and its affiliates.
"""Detached survey calculation and conservative declared swept-volume clearance.

The free volume is an operator declaration, not inferred from IK. Joint intervals
are inflated by a serial-chain displacement bound, covering independent joint
progress plus declared tracking error; no firmware Cartesian path is assumed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from leapflow.hardware.skills.body_visibility import capsules, segment_distance
from leapflow.hardware.skills.geometry import Plane, Transform, WorkspaceBox
from leapflow.hardware.skills.motion_geometry import MotionGeometry
from leapflow.hardware.skills.planning import Evaluation, FrozenJoints, PlanningCheckpoint
from leapflow.hardware.skills.protocol import SIDE_EFFECT_NONE, ArmState, LocatedObject, SkillError, SkillOutcome
from leapflow.hardware.skills.view_planner import ViewCandidate


@dataclass(frozen=True)
class SurveyClearance:
    free_box: WorkspaceBox
    radii: tuple[float, ...]
    tool_radius_mm: float
    tracking_deg: float
    margin_mm: float
    table: Plane
    objects: tuple[LocatedObject, ...]
    camera_mounts: tuple[Transform, ...] = ()

    def check(self, geometry: MotionGeometry, begin: Mapping[str, float], end: Mapping[str, float]) -> None:
        chain = geometry.kinematics()
        start = tuple(begin[n] for n in geometry.names)
        finish = tuple(end[n] for n in geometry.names)
        frames = chain.link_transforms(start)
        body = capsules([f.translation for f in frames], self.radii)
        lengths = [abs(k.a) + abs(k.d) for k in chain.links]
        tool_length = math.sqrt(sum(v*v for v in geometry.flange_tcp.translation))
        camera_length = max((math.sqrt(sum(v*v for v in m.translation))
                             for m in self.camera_mounts), default=0)
        # Lipschitz bound for any independently progressing revolute joint in its
        # full endpoint interval; correlated calibration error is not averaged away.
        reach = sum(lengths) + max(tool_length, camera_length) + max(self.radii)
        inflation = reach * sum(math.radians(abs(a-b) + 2*self.tracking_deg) for a, b in zip(start, finish))
        inflation += geometry.fk_mm + self.margin_mm
        for i, link in enumerate(body):
            self._segment(link.start, link.end, link.radius_mm + inflation)
            for other in body[i+2:]:
                if segment_distance(link.start, link.end, other.start, other.end) <= (
                        link.radius_mm + other.radius_mm + 2*inflation):
                    raise SkillError('observation_insufficient', 'survey cannot exclude self-collision')
        flange = frames[-1]
        tcp = flange.compose(geometry.flange_tcp).translation
        self._segment(flange.translation, tcp, self.tool_radius_mm + inflation)
        self._attachment(body[:-1], flange.translation, tcp, inflation)
        if self.table.above(tcp) <= self.tool_radius_mm + inflation:
            raise SkillError('observation_insufficient', 'survey tool swept volume intersects the table')
        for mount in self.camera_mounts:
            camera = flange.compose(mount).translation
            self._segment(flange.translation, camera, self.tool_radius_mm + inflation)
            self._attachment(body[:-1], flange.translation, camera, inflation)
            if self.table.above(camera) <= self.tool_radius_mm + inflation:
                raise SkillError('observation_insufficient', 'survey camera swept volume intersects the table')

    def _attachment(self, body: Sequence[Any], a: Sequence[float], b: Sequence[float], inflation: float) -> None:
        for link in body:
            if segment_distance(tuple(a), tuple(b), link.start, link.end) <= (
                    self.tool_radius_mm + link.radius_mm + 2*inflation):
                raise SkillError('observation_insufficient', 'survey attachment can intersect a nonattached body link')

    def _segment(self, a: Sequence[float], b: Sequence[float], radius: float) -> None:
        if not self.free_box.contains(a, margin=radius) or not self.free_box.contains(b, margin=radius):
            raise SkillError('observation_insufficient', 'survey swept volume leaves the declared obstacle-free box')
        for obj in self.objects:
            # Bounding sphere encloses the entire measured oriented box. A lower
            # bound cannot describe all obstacles: free_box remains independently required.
            extent = math.sqrt(obj.length**2 + obj.width**2 + obj.height**2)/2
            if segment_distance(tuple(a), tuple(b), obj.center, obj.center) <= radius + extent:
                raise SkillError('observation_insufficient', f'survey swept volume intersects {obj.object_id}')


@dataclass(frozen=True)
class SurveyJob:
    geometry: MotionGeometry
    state: ArmState
    destinations: tuple[ViewCandidate, ...]
    posture: FrozenJoints | None
    clearance: SurveyClearance
    max_travel_mm: float
    max_steps: int

    def evaluate(self, checkpoint: PlanningCheckpoint) -> Evaluation:
        geometry = self.geometry.bind(checkpoint)
        geometry.check_kinematics(self.state)
        current = dict(self.state.joints)
        tcp = self.state.tcp
        legs = []
        travel = 0.0
        for candidate in self.destinations:
            checkpoint.check()
            if self.posture is not None:
                endpoint = dict(self.posture)
                target = geometry.check_joints(endpoint, current)
            else:
                tracker = geometry.reach_tracker(ArmState(flange=geometry.tcp_to_flange(tcp), tcp=tcp,
                                                                        joints=FrozenJoints.capture(current)))
                tracker.move(candidate.tcp, linear=False, label=candidate.feature)
                if not tracker.verified or tracker.joints is None:
                    raise SkillError('observation_insufficient', 'survey requires verified kinematics')
                endpoint, target = dict(tracker.joints), candidate.tcp
            samples = []
            geometry.check_joints(endpoint, current, observe=lambda q: samples.append(dict(zip(geometry.names, q))))
            if len(samples) + sum(len(leg['steps']) for leg in legs) > self.max_steps:
                raise SkillError('invalid_args', 'survey joint waypoint count exceeds the workcell step budget')
            steps = []
            for q in samples:
                checkpoint.check()
                self.clearance.check(geometry, current, q)
                pose = geometry.flange_to_tcp(geometry.kinematics().forward([q[n] for n in geometry.names]))
                travel += tcp.distance_mm(pose)
                if travel > self.max_travel_mm:
                    raise SkillError('observation_insufficient', 'survey exceeds declared travel budget')
                steps.append({'joints': q, 'pose': pose.to_dict()})
                current, tcp = q, pose
            legs.append({'camera': candidate.camera, 'feature': candidate.feature,
                         'score': candidate.score, 'steps': steps, 'pose': target.to_dict()})
        return Evaluation(SkillOutcome.succeeded({'legs': legs, 'travel_mm': travel}, side_effect_state=SIDE_EFFECT_NONE),
                          {'tcp_distance_mm': travel}, verified=True)
