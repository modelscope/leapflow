# Copyright (c) Alibaba, Inc. and its affiliates.
"""Owner-side bounded survey behind ordinary skill approval, leases and motion gates."""
from __future__ import annotations

import hashlib
import math
import time
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Iterator, Mapping, TYPE_CHECKING

from leapflow.hardware.skills.geometry import Pose, WorkspaceBox
from leapflow.hardware.skills.observability import ObservationSettings
from leapflow.hardware.skills.planning import Deadline, FrozenJoints, PreparedPlanning, encoded
from leapflow.hardware.skills.protocol import (
    SIDE_EFFECT_COMMITTED, SIDE_EFFECT_NONE, SIDE_EFFECT_PARTIAL,
    LocatedObject, SkillError, SkillOutcome, SkillPlan,
)
from leapflow.hardware.skills.survey_planning import SurveyClearance, SurveyJob
from leapflow.hardware.skills.view_planner import ViewCandidate, ViewGoal, ViewProposalJob

if TYPE_CHECKING:
    from leapflow.hardware.skills.context import RunContext


class WorkcellSurvey:
    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    @property
    def settings(self) -> ObservationSettings:
        return ObservationSettings.from_mapping(self.ctx.workcell.observation)

    def scope(self) -> str:
        ctx = self.ctx
        return hashlib.sha256(encoded({'workcell': ctx.workcell.to_dict(), 'frames': dict(ctx.frames.versions),
            'geometry': repr(ctx.motion.geometry_snapshot())}).encode()).hexdigest()

    def clearance(self, camera: str | None = None) -> SurveyClearance:
        ctx, raw = self.ctx, self.ctx.workcell.observation
        declared = raw.get('clearance')
        if not declared or not raw.get('body_radii_mm'):
            raise SkillError('observation_insufficient', 'survey motion needs declared body radii and an obstacle-free volume')
        if ctx.frames.table is None:
            raise SkillError('calibration_missing', 'survey requires a calibrated table')
        mounts = tuple(ctx.frames.transform(r) for r in ctx.workcell.roles
                       if ctx.frames.parent(r) == 'flange' and ctx.frames.transform(r) is not None)
        if camera and ctx.frames.transform(camera) is None:
            raise SkillError('calibration_missing', 'survey moving camera mount is unresolved')
        objects = ctx.scene._memory.objects(ctx.scene._basis, ctx.scene._settings.memory_ttl_s)
        return SurveyClearance(WorkspaceBox.from_mapping(declared['free_box']), tuple(raw['body_radii_mm']),
            declared['tool_radius_mm'], declared['joint_tracking_deg'], self.settings.clearance_mm,
            ctx.frames.table, objects, mounts)

    def report(self, obj: LocatedObject, grasp: Any = None,
               reference: LocatedObject | None = None) -> Mapping[str, Any]:
        # A remembered grasp supplies tolerance only, not a reusable motion handle.
        # Bound width growth for any closing axis without assuming a gripper mount.
        if grasp is not None and reference is not None:
            growth = max(0., obj.length-reference.length) + max(0., obj.width-reference.width)
            rotation = math.radians((obj.yaw_deg-reference.yaw_deg+90) % 180-90)
            growth += 2*math.hypot(obj.length, obj.width)*math.sin(abs(rotation)/2)
            grasp = replace(grasp, width_mm=grasp.width_mm+growth)
        return self.ctx.scene.observation(obj, grasp)

    def adequate(self, obj: LocatedObject, grasp: Any = None,
                 reference: LocatedObject | None = None) -> bool:
        report = self.report(obj, grasp, reference)
        return bool(report.get('sufficient')) if grasp else report.get('verdict') == 'sufficient'

    def known_occlusion(self, obj: LocatedObject) -> None:
        if any(v['nearer'] > 0 and v['occluder'] == 'unknown'
               for v in self.ctx.scene.observation(obj).get('views', ())):
            raise SkillError('observation_insufficient',
                             'unknown occluder: stationary observation or explicit visual inspection required')

    async def plan(self, args: Mapping[str, Any]) -> SkillPlan:
        try:
            return await self._plan(args)
        except SkillError as exc:
            raise SkillError(exc.failure_code, exc.message, result={
                'observation_scope': self.scope(), **dict(exc.result)},
                side_effect_state=exc.side_effect_state, status=exc.status) from exc

    async def _plan(self, args: Mapping[str, Any]) -> SkillPlan:
        ctx, settings = self.ctx, self.settings
        deadline = min(ctx.planning_deadline, time.monotonic()+ctx.planning_settings.timeout_s)
        obj = ctx.scene.object(args['object'])
        if ctx.scene.held is not None:
            raise SkillError('plan_stale', 'survey cannot move with a remembered held object')
        grasp = ctx.scene.grasp(args['grasp']) if args.get('grasp') else None
        if grasp is not None and grasp.object_id != obj.object_id:
            raise SkillError('invalid_args', 'survey grasp belongs to a different object')
        basis = await ctx.planning_basis()
        scope = self.scope()
        state = await ctx.motion.state()
        state = replace(state, joints=FrozenJoints.capture(state.joints))
        legs, refused = [], []
        if not self.adequate(obj, grasp) and args['mode'] != 'passive':
            if not settings.enabled:
                raise SkillError('skill_unavailable', 'survey motion requires observation.enabled')
            proposals, posture, proposal_jobs = await self._proposals(obj, state, args, deadline)
            legs, refused = await self._routes(state, proposals, posture, deadline, proposal_jobs)
            if not legs:
                raise SkillError('observation_insufficient', 'no survey view has a verified safe route',
                                 result={'refused': refused})
        if basis != await ctx.planning_basis() or scope != self.scope():
            raise SkillError('plan_stale', 'survey owner or observation basis changed during planning')
        PreparedPlanning((), objects=(obj,), held=None, grasps=() if grasp is None else (grasp,)).check_scene(ctx)
        Deadline(deadline).check()
        planned = SkillPlan('perceive.survey', args, f"Bounded {args['mode']} survey of {obj.object_id}; {len(legs)} views",
            targets=tuple(Pose.from_mapping(leg['pose']) for leg in legs),
            limits={'speed_pct': min(settings.speed_pct, ctx.limits['speed_pct_max'])},
            basis={'object': obj.to_dict(), 'grasp': None if grasp is None else grasp.to_dict(),
                   'start': state.tcp.to_dict(), 'start_joints': dict(state.joints), 'observation_scope': scope,
                   'legs': legs, 'refused': refused})
        ctx.planning_settings.check_size(planned.to_dict())
        return planned

    async def _proposals(self, obj: LocatedObject, state: Any, args: Mapping[str, Any],
                         deadline: float) -> tuple[tuple[ViewCandidate, ...], Any, int]:
        ctx, raw = self.ctx, self.ctx.workcell.observation
        self.known_occlusion(obj)
        if args['mode'] == 'observe':
            posture = raw.get('observe')
            if not posture:
                raise SkillError('skill_unavailable', 'no observation.observe joint posture is declared')
            self.clearance()
            return (ViewCandidate(state.tcp, 'auto', PoseTransform(state.tcp), 'observe', 1.),), FrozenJoints.capture(posture), 0
        result, proposal_jobs = [], 0
        for role, values in raw.get('cameras', {}).items():
            if proposal_jobs >= ctx.planning_settings.max_jobs-1:
                break
            if ctx.frames.parent(role) != 'flange' or 'near_mm' not in values:
                continue
            mount = ctx.frames.transform(role)
            if mount is None:
                continue
            self.clearance(role)
            capture = await ctx.scene.capture(role)
            goal = ViewGoal(obj, ctx.frames.table, role, mount, ctx.motion.geometry_snapshot().flange_tcp,
                capture.frame.intrinsics, values['near_mm'], values['far_mm'], self.settings.max_candidates)
            job = ViewProposalJob(raw.get('planner', 'ring'), encoded(raw.get('planner_settings', {})), goal)
            evaluations = await ctx.calculations.evaluate([job], deadline=deadline)
            proposal_jobs += 1
            if len(evaluations) != 1 or not isinstance(evaluations[0].value, SkillOutcome):
                raise SkillError('internal_defect', 'invalid view planner worker response')
            value = evaluations[0].value
            if not value.ok:
                raise SkillError(value.failure_code, value.message, result=value.result)
            proposals = tuple(ViewCandidate(Pose.from_mapping(p['tcp']), p['camera'],
                              PoseTransform(Pose.from_mapping(p['camera_to_base'])), p['feature'], p['score'])
                              for p in value.result['candidates'])
            result.extend(p for p in proposals if _frustum(goal, p))
        if not result:
            raise SkillError('observation_insufficient', 'no declared calibrated moving camera has a usable target frustum')
        priors = args.get('priors', {}) if args.get('prior_scope') == self.scope() else {}
        result.sort(key=lambda p: (-float(priors.get(p.feature, 0)), -p.score, state.tcp.distance_mm(p.tcp), p.feature))
        return tuple(result[:self.settings.max_candidates]), None, proposal_jobs

    async def _routes(self, state: Any, proposals: tuple[ViewCandidate, ...], posture: Any,
                      deadline: float, jobs_used: int = 0) -> tuple[list[Any], list[Any]]:
        ctx, settings = self.ctx, self.settings
        remaining, legs, refused = list(proposals), [], []
        travel, steps = 0., 0
        while remaining and len(legs) < settings.max_views and jobs_used < ctx.planning_settings.max_jobs:
            offered = remaining[:ctx.planning_settings.max_jobs-jobs_used]
            jobs = [SurveyJob(ctx.motion.geometry_snapshot(), state, (p,), posture,
                    self.clearance(None if posture else p.camera), settings.max_travel_mm-travel,
                    int(ctx.limits['max_steps'])-steps) for p in offered]
            evaluations = await ctx.calculations.evaluate(jobs, deadline=deadline)
            jobs_used += len(jobs)
            if len(evaluations) != len(jobs):
                raise SkillError('internal_defect', 'survey worker response count differs')
            chosen = None
            for proposal, evaluation in zip(offered, evaluations):
                value = evaluation.value
                if not isinstance(value, SkillOutcome):
                    raise SkillError('internal_defect', 'invalid survey worker response')
                if value.failure_code in {'internal_defect', 'timeout', 'stopped'}:
                    raise SkillError(value.failure_code, value.message)
                if value.ok and chosen is None:
                    chosen = (proposal, value.result)
                elif not value.ok:
                    refused.append({'feature': proposal.feature, 'failure_code': value.failure_code, 'message': value.message})
            if chosen is None:
                break
            proposal, result = chosen
            leg = result['legs'][0]
            legs.append(leg)
            travel += result['travel_mm']
            steps += len(leg['steps'])
            endpoint = leg['steps'][-1]
            tcp = Pose.from_mapping(endpoint['pose'])
            state = replace(state, tcp=tcp, flange=ctx.motion.geometry_snapshot().tcp_to_flange(tcp),
                            joints=FrozenJoints.capture(endpoint['joints']))
            remaining.remove(proposal)
        if remaining and jobs_used >= ctx.planning_settings.max_jobs:
            refused.append({'failure_code': 'planning_budget_exhausted', 'remaining': len(remaining)})
        return legs, refused

    @contextmanager
    def trial(self, trials: list[Any], feature: str, before: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
        """Record an in-flight trial even if cancellation or timeout interrupts it."""
        started = time.monotonic()
        trial = {'feature': feature, 'before': dict(before), 'after': {},
                 'duration_s': 0., 'failure_code': 'incomplete', 'attempted_waypoints': 0}
        trials.append(trial)
        try:
            yield trial
            trial.pop('failure_code', None)
        except SkillError as exc:
            trial['failure_code'] = exc.failure_code
            raise
        finally:
            trial['duration_s'] = time.monotonic()-started

    async def execute(self, plan: SkillPlan) -> SkillOutcome:
        self.ctx.note('survey', {'scope': plan.basis['observation_scope'], 'trials': [], 'moves': 0})
        try:
            return await self._execute(plan)
        except SkillError as exc:
            partial = self.ctx.evidence.get('survey', {})
            raise SkillError(exc.failure_code, exc.message, result={
                'observation_scope': plan.basis['observation_scope'],
                'trials': partial.get('trials', []), 'moves': partial.get('moves', 0), **dict(exc.result)},
                side_effect_state=exc.side_effect_state, status=exc.status) from exc

    async def _execute(self, plan: SkillPlan) -> SkillOutcome:
        ctx, settings = self.ctx, self.settings
        if self.scope() != plan.basis['observation_scope']:
            raise SkillError('plan_stale', 'survey observation configuration or calibration changed')
        obj = ctx.scene.object(plan.args['object'])
        if obj.to_dict() != plan.basis['object'] or ctx.scene.held is not None:
            raise SkillError('plan_stale', 'survey exact object or payload changed')
        grasp = ctx.scene.grasp(plan.args['grasp']) if plan.args.get('grasp') else None
        if (None if grasp is None else grasp.to_dict()) != plan.basis['grasp']:
            raise SkillError('plan_stale', 'survey optional grasp changed since approval')
        state = await ctx.motion.state()
        if (state.tcp.distance_mm(Pose.from_mapping(plan.basis['start'])) > ctx.workcell.freshness['start_mm']
                or state.tcp.rotation_to_deg(Pose.from_mapping(plan.basis['start'])) > ctx.workcell.freshness['start_deg']
                or set(state.joints) != set(plan.basis['start_joints'])
                or any(abs(state.joints[n]-v) > ctx.workcell.freshness['start_deg'] for n, v in plan.basis['start_joints'].items())):
            raise SkillError('plan_stale', 'survey start changed since approval')
        started, trials = time.monotonic(), []
        reference = obj
        before = dict(self.report(obj, grasp, reference))
        history: list[Any] = []
        record = {'scope': plan.basis['observation_scope'], 'trials': trials, 'moves': 0,
                  'refused': plan.basis['refused']}
        ctx.note('survey', record)
        with self.trial(trials, 'passive', before) as trial:
            obj = await self._measure(obj, 'auto', history)
            trial['after'] = dict(self.report(obj, grasp, reference))
        for leg in plan.basis['legs']:
            if self.adequate(obj, grasp, reference):
                break
            with self.trial(trials, leg['feature'], self.report(obj, grasp, reference)) as trial:
                self.known_occlusion(obj)
                for waypoint in leg['steps']:
                    geometry = ctx.motion.geometry_snapshot()
                    state = await ctx.motion.state()
                    geometry.check_kinematics(state)
                    def validate(current: Any) -> None:
                        if self.scope() != plan.basis['observation_scope']:
                            raise SkillError('plan_stale', 'survey configuration changed before movement')
                        if ctx.scene.held is not None or ctx.scene.object(obj.object_id) != obj:
                            raise SkillError('plan_stale', 'survey live object or payload changed before movement')
                        geometry.check_kinematics(current)
                        clearance = self.clearance(None if leg['camera'] == 'auto' else leg['camera'])
                        clearance.check(geometry, current.joints, waypoint['joints'])
                    trial['attempted_waypoints'] += 1
                    moved = await ctx.motion.move_joints(waypoint['joints'], speed_pct=plan.limits['speed_pct'],
                                                         validate=validate)
                    record['moves'] += 1
                    if not ctx.motion.reached(Pose.from_mapping(waypoint['pose']), moved.state):
                        raise SkillError('guard_failed', 'survey approved waypoint was not reached')
                await ctx.motion.pause(settings.settle_s)
                obj = await self._measure(obj, leg['camera'], history)
                trial['after'] = dict(self.report(obj, grasp, reference))
        moves = record['moves']
        result = {'object': obj.to_dict(), 'observation': dict(self.report(obj, grasp, reference)),
                  'before': before, 'trials': trials, 'moves': moves, 'duration_s': time.monotonic()-started,
                  'observation_scope': plan.basis['observation_scope'], 'refused': plan.basis['refused'],
                  'grasp_handles_invalidated': obj.to_dict() != plan.basis['object'],
                  'grasp_is_tolerance_only': grasp is not None,
                  'next_grasp': 'grasp.propose with the updated object id'}
        effect = SIDE_EFFECT_COMMITTED if moves else SIDE_EFFECT_NONE
        if not self.adequate(obj, grasp, reference):
            result['next'] = ['perceive.snapshot with_cameras=[all] (requires node and host sharing consent)']
            raise SkillError('observation_insufficient', 'bounded survey exhausted; inspect observation and request an explicit view',
                             result=result, side_effect_state=SIDE_EFFECT_PARTIAL if moves else SIDE_EFFECT_NONE)
        return SkillOutcome.succeeded(result, side_effect_state=effect, message='Observation adequate; no images shared')

    async def _measure(self, obj: LocatedObject, role: str, history: list[Any]) -> LocatedObject:
        scene = self.ctx.scene
        captures, sync = await scene._observations.captures(scene._observations.roles() if role == 'auto' else (role,))
        if not sync.get('within_tolerance', True):
            raise SkillError('device_fault', 'survey cameras are not synchronized', result={'sync': sync})
        history.extend(captures.values())
        fresh = await scene._observations.measure(obj, history, latest=tuple(captures.values()))
        self.ctx.note('survey_capture', {'sync': sync, 'views': len(history)})
        return fresh


def PoseTransform(pose: Pose) -> Any:
    from leapflow.hardware.skills.geometry import Transform
    return Transform.from_pose(pose)


def _frustum(goal: ViewGoal, proposal: ViewCandidate) -> bool:
    from leapflow.hardware.skills.geometry import Transform
    expected = Transform.from_pose(proposal.tcp).compose(goal.flange_tcp.inverse()).compose(goal.flange_camera)
    if expected.to_pose().distance_mm(proposal.camera_to_base.to_pose()) > 1e-6 or expected.to_pose().rotation_to_deg(proposal.camera_to_base.to_pose()) > 1e-6:
        return False
    inverse = proposal.camera_to_base.inverse()
    obj, intrinsics = goal.obj, goal.intrinsics
    yaw = math.radians(obj.yaw_deg)
    for dx in (-obj.length/2, obj.length/2):
        for dy in (-obj.width/2, obj.width/2):
            for dz in (-obj.height/2, obj.height/2):
                p = inverse.apply((obj.center[0]+dx*math.cos(yaw)-dy*math.sin(yaw),
                                   obj.center[1]+dx*math.sin(yaw)+dy*math.cos(yaw), obj.center[2]+dz))
                if not goal.near_mm < p[2] < goal.far_mm:
                    return False
                u, v = intrinsics.project(p)
                if not 0 <= u < intrinsics.width or not 0 <= v < intrinsics.height:
                    return False
    return True
