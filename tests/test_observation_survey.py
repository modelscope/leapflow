# Copyright (c) Alibaba, Inc. and its affiliates.
"""Finite survey execution, fail-closed clearance, and runtime evidence ownership."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from leapflow.hardware.skills.geometry import Plane, Pose, Transform, WorkspaceBox
from leapflow.hardware.skills.kinematics import DhChain, DhLink
from leapflow.hardware.skills.observability import ViewEvidence
from leapflow.hardware.skills.planning import Evaluation, PlanningSettings
from leapflow.hardware.skills.protocol import (
    SIDE_EFFECT_COMMITTED, SIDE_EFFECT_NONE, SIDE_EFFECT_PARTIAL,
    ArmState, SkillError, SkillOutcome, SkillPlan,
)
from leapflow.hardware.skills.survey import WorkcellSurvey
from leapflow.hardware.skills.survey_planning import SurveyClearance
from leapflow.hardware.skills.view_planner import ViewCandidate
from leapflow.hardware.skills.builtin.scene import skills as scene_skills
from leapflow.hardware.skills.builtin.survey import skills as survey_skills
from tests.test_perception import CUP, _calibrated, _object, _runtime, _scene, _spec
from tests.test_observability import full_report
from tests._harness.skill_stubs import CAMERA_ID, PERCEPTION_WORKCELL_YAML, WORKCELL_ID

pytestmark = pytest.mark.usefixtures('close_skill_runtimes')


class Motion:
    def __init__(self):
        self.current = ArmState(Pose(300., 0., 250.), Pose(300., 0., 250.), {'joint.axis': 0.})
        self.writes = []
        self.failure = None

    async def state(self):
        return self.current

    def geometry_snapshot(self):
        return SimpleNamespace(check_kinematics=lambda state: None)

    async def move_joints(self, joints, *, speed_pct, validate):
        validate(self.current)
        self.writes.append((dict(joints), speed_pct))
        if self.failure:
            raise self.failure
        self.current = replace(self.current, joints=joints)
        return SimpleNamespace(state=self.current)

    def reached(self, pose, state):
        return True

    async def pause(self, duration):
        await asyncio.sleep(0)


async def rig(tmp_path, monkeypatch):
    scene = _scene(_calibrated(tmp_path, CUP), spec=_spec(observation={'enabled': True}))
    (obj,) = await scene.locate(['cup'])
    scene._observations.record(obj, ())
    evidence = {}
    motion = Motion()
    ctx = SimpleNamespace(scene=scene, workcell=scene._spec, motion=motion, evidence=evidence,
                          note=evidence.__setitem__, planning_settings=PlanningSettings(),
                          planning_deadline=float('inf'), limits={'speed_pct_max': 30., 'max_steps': 10.})
    survey = WorkcellSurvey(ctx)
    monkeypatch.setattr(survey, 'scope', lambda: 'a'*64)
    checks = []
    def check(geometry, begin, end):
        checks.append((dict(begin), dict(end)))
    monkeypatch.setattr(survey, 'clearance', lambda camera=None: SimpleNamespace(check=check))
    state = motion.current
    leg = {'camera': 'camera', 'feature': 'declared:one', 'pose': state.tcp.to_dict(),
           'steps': [{'joints': {'joint.axis': 1.}, 'pose': state.tcp.to_dict()}]}
    plan = SkillPlan('perceive.survey', {'object': obj.object_id, 'mode': 'active'}, 'one view',
        limits={'speed_pct': 7.}, basis={'object': obj.to_dict(), 'grasp': None,
        'start': state.tcp.to_dict(), 'start_joints': dict(state.joints),
        'observation_scope': 'a'*64, 'legs': [leg], 'refused': []})
    return SimpleNamespace(survey=survey, scene=scene, obj=obj, ctx=ctx, motion=motion, plan=plan, checks=checks)


async def test_fresh_passive_adequacy_executes_zero_waypoints(tmp_path, monkeypatch):
    r = await rig(tmp_path, monkeypatch)
    async def measure(obj, role, history):
        r.scene._observations.record(obj, full_report(obj).views)
        return obj
    monkeypatch.setattr(r.survey, '_measure', measure)
    result = await r.survey.execute(r.plan)
    assert result.ok and result.side_effect_state == SIDE_EFFECT_NONE
    assert result.result['moves'] == 0 and r.motion.writes == []
    assert len(result.result['trials']) == 1


async def test_movement_is_bounded_checked_and_stops_at_first_adequate_view(tmp_path, monkeypatch):
    r = await rig(tmp_path, monkeypatch)
    second = {**r.plan.basis['legs'][0], 'feature': 'declared:two'}
    plan = replace(r.plan, basis={**r.plan.basis, 'legs': [r.plan.basis['legs'][0], second]})
    async def measure(obj, role, history):
        if role != 'auto':
            r.scene._observations.record(obj, full_report(obj).views)
        return obj
    monkeypatch.setattr(r.survey, '_measure', measure)
    result = await r.survey.execute(plan)
    assert result.ok and result.result['moves'] == 1
    assert result.side_effect_state == SIDE_EFFECT_COMMITTED
    assert r.motion.writes == [({'joint.axis': 1.}, 7.)]
    assert len(r.checks) == 1 and len(result.result['trials']) == 2
    assert result.result['trials'][1]['after']['verdict'] == 'sufficient'


@pytest.mark.parametrize('kind', ['scope', 'object', 'start', 'held'])
async def test_stale_survey_refuses_before_capture_or_motion(tmp_path, monkeypatch, kind):
    r = await rig(tmp_path, monkeypatch)
    plan = r.plan
    if kind == 'scope':
        plan = replace(plan, basis={**plan.basis, 'observation_scope': 'b'*64})
    elif kind == 'object':
        r.scene._memory.update(replace(r.obj, width=r.obj.width+1), r.scene._basis)
    elif kind == 'start':
        r.motion.current = replace(r.motion.current, joints={'joint.axis': 90.})
    else:
        monkeypatch.setattr(type(r.scene), 'held', property(lambda self: object()))
    with pytest.raises(SkillError) as error:
        await r.survey.execute(plan)
    assert error.value.failure_code == 'plan_stale'
    assert r.motion.writes == []
    assert error.value.result['observation_scope'] == plan.basis['observation_scope']


async def test_new_unknown_occlusion_refuses_previously_approved_motion(tmp_path, monkeypatch):
    r = await rig(tmp_path, monkeypatch)
    async def measure(obj, role, history):
        r.scene._observations.record(obj, (ViewEvidence('camera', 'live', 'occluded', nearer=.5),))
        return obj
    monkeypatch.setattr(r.survey, '_measure', measure)
    with pytest.raises(SkillError) as error:
        await r.survey.execute(r.plan)
    assert error.value.failure_code == 'observation_insufficient' and r.motion.writes == []
    assert error.value.result['trials'][-1]['failure_code'] == 'observation_insufficient'


@pytest.mark.parametrize('failure', [SkillError('device_fault', 'lost telemetry'), asyncio.CancelledError()])
async def test_interrupted_leg_preserves_evidence_without_replay(tmp_path, monkeypatch, failure):
    r = await rig(tmp_path, monkeypatch)
    async def measure(obj, role, history):
        return obj
    monkeypatch.setattr(r.survey, '_measure', measure)
    r.motion.failure = failure
    with pytest.raises(type(failure)):
        await r.survey.execute(r.plan)
    record = r.ctx.evidence['survey']
    assert len(r.motion.writes) == 1
    assert record['trials'][-1]['attempted_waypoints'] == 1
    assert record['trials'][-1]['duration_s'] >= 0
    assert record['trials'][-1]['failure_code'] in {'incomplete', 'device_fault'}


async def test_proposal_jobs_and_routes_share_one_budget(tmp_path, monkeypatch):
    r = await rig(tmp_path, monkeypatch)
    r.ctx.planning_settings = PlanningSettings(max_jobs=2)
    seen = []
    async def evaluate(jobs, *, deadline):
        seen.extend(jobs)
        return [Evaluation(SkillOutcome.failed('unreachable', 'no safe IK')) for _ in jobs]
    r.ctx.calculations = SimpleNamespace(evaluate=evaluate)
    candidates = tuple(ViewCandidate(Pose(0., 0., 0.), 'camera', Transform(), str(i), 1.) for i in range(4))
    legs, refused = await r.survey._routes(r.motion.current, candidates, None, float('inf'), jobs_used=1)
    assert len(seen) == 1 and legs == []
    assert refused[-1]['failure_code'] == 'planning_budget_exhausted'


@pytest.mark.parametrize('problem', ['volume', 'table', 'object', 'self', 'camera'])
def test_declared_clearance_refuses_unsafe_sweeps(problem):
    chain = DhChain((DhLink(100., 0., 100.), DhLink(100., 0., 0.)), 'standard')
    geometry = SimpleNamespace(names=('a', 'b'), kinematics=lambda: chain,
        flange_tcp=Transform(translation=(20., 0., 0.)), fk_mm=.01)
    clearance = SurveyClearance(WorkspaceBox(-500., 500., -500., 500., -200., 500.),
        (1., 1.), 1., .001, .01, Plane.horizontal(-100.), ())
    if problem == 'volume':
        clearance = replace(clearance, free_box=WorkspaceBox(0., 10., 0., 10., 0., 10.))
    elif problem == 'table':
        clearance = replace(clearance, table=Plane.horizontal(101.))
    elif problem == 'object':
        clearance = replace(clearance, objects=(_object(center=(50., 0., 50.)),))
    elif problem == 'self':
        chain = DhChain((DhLink(100., 0., 100.), DhLink(-100., 0., 0.), DhLink(0., 0., -100.)), 'standard')
        geometry.names = ('a', 'b', 'c')
        clearance = replace(clearance, radii=(1., 1., 1.))
    else:
        clearance = replace(clearance, camera_mounts=(Transform(translation=(-150., 0., 0.)),))
    joints = {name: 0. for name in geometry.names}
    with pytest.raises(SkillError) as error:
        clearance.check(geometry, joints, joints)
    assert error.value.failure_code == 'observation_insufficient'


@pytest.mark.parametrize('moves', [0, 1])
async def test_exhausted_survey_uses_shared_side_effect_contract(tmp_path, monkeypatch, moves):
    r = await rig(tmp_path, monkeypatch)
    async def measure(obj, role, history):
        return obj
    monkeypatch.setattr(r.survey, '_measure', measure)
    plan = replace(r.plan, basis={**r.plan.basis, 'legs': r.plan.basis['legs'] if moves else []})
    with pytest.raises(SkillError) as error:
        await r.survey.execute(plan)
    assert error.value.failure_code == 'observation_insufficient'
    assert error.value.result['moves'] == moves
    assert error.value.side_effect_state == (SIDE_EFFECT_PARTIAL if moves else SIDE_EFFECT_NONE)


@pytest.mark.parametrize('adequate', [False, True])
async def test_runtime_zero_motion_survey_preserves_known_no_effect(tmp_path, monkeypatch, adequate):
    registry = _calibrated(tmp_path, CUP)
    runtime = _runtime(tmp_path, registry, PERCEPTION_WORKCELL_YAML, skills=(*scene_skills(), *survey_skills()))
    located = await runtime.plan(WORKCELL_ID, 'perceive.locate', {'queries': ['cup']})
    assert located.ok
    monkeypatch.setattr(WorkcellSurvey, 'adequate', lambda self, *args: adequate)
    plan = await runtime.plan(WORKCELL_ID, 'perceive.survey', {'object': 'o1', 'mode': 'passive'})
    assert isinstance(plan, SkillPlan), plan
    run = await runtime.execute(plan.digest)
    finished = await runtime.wait(run.run_id, timeout_s=5.)
    assert finished.outcome.ok is adequate
    assert finished.outcome.side_effect_state == SIDE_EFFECT_NONE
    if not adequate:
        assert finished.outcome.failure_code == 'observation_insufficient'
    assert registry.sim.batches == []


async def test_auto_camera_missing_model_refuses_without_inventing_an_auto_role(tmp_path):
    registry = _calibrated(tmp_path, CUP)
    cell = {**PERCEPTION_WORKCELL_YAML, 'models': {}}
    runtime = _runtime(tmp_path, registry, cell, skills=scene_skills())
    result = await runtime.plan(WORKCELL_ID, 'perceive.locate', {'queries': ['cup'], 'camera': 'auto'})
    assert isinstance(result, SkillOutcome) and not result.ok
    assert result.failure_code == 'skill_unavailable'
    assert "'locator' model" in result.message
    assert 'binds no auto' not in result.message
    assert 'auto is not calibrated' not in result.message


async def test_runtime_terminal_failure_keeps_survey_scope_and_capture_trial(tmp_path):
    registry = _calibrated(tmp_path, CUP)
    runtime = _runtime(tmp_path, registry, PERCEPTION_WORKCELL_YAML, skills=(*scene_skills(), *survey_skills()))
    located = await runtime.plan(WORKCELL_ID, 'perceive.locate', {'queries': ['cup']})
    assert located.ok
    plan = await runtime.plan(WORKCELL_ID, 'perceive.survey', {'object': 'o1', 'mode': 'passive'})
    assert isinstance(plan, SkillPlan), plan
    registry.devices[CAMERA_ID].fail = True
    run = await runtime.execute(plan.digest)
    finished = await runtime.wait(run.run_id, timeout_s=5.)
    assert finished.outcome.failure_code == 'device_fault'
    record = finished.outcome.evidence['survey']
    assert record['scope'] == plan.basis['observation_scope']
    assert record['trials'][0]['failure_code'] == 'device_fault'
    assert registry.sim.batches == []


@pytest.mark.parametrize('change', ['object', 'held', 'scope'])
async def test_final_write_validation_rechecks_live_scene(tmp_path, monkeypatch, change):
    r = await rig(tmp_path, monkeypatch)
    async def measure(obj, role, history):
        return obj
    monkeypatch.setattr(r.survey, '_measure', measure)
    original = r.motion.move_joints
    async def move(joints, *, speed_pct, validate):
        if change == 'object':
            r.scene._memory.update(replace(r.obj, width=r.obj.width+5), r.scene._basis)
        elif change == 'held':
            monkeypatch.setattr(type(r.scene), 'held', property(lambda self: object()))
        else:
            monkeypatch.setattr(r.survey, 'scope', lambda: 'b'*64)
        return await original(joints, speed_pct=speed_pct, validate=validate)
    r.motion.move_joints = move
    with pytest.raises(SkillError) as error:
        await r.survey.execute(r.plan)
    assert error.value.failure_code == 'plan_stale'
    assert r.motion.writes == []
