# Copyright (c) Alibaba, Inc. and its affiliates.
"""Generalized observation contracts and real synthetic depth measurements."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from leapflow.hardware.skills.body_visibility import capsules, segment_distance
from leapflow.hardware.skills.frame_status import Accuracy
from leapflow.hardware.skills.geometry import Pose, Transform
from leapflow.hardware.skills.kinematics import DhChain, DhLink
from leapflow.hardware.skills.observability import ObservationReport, ObservationSettings, ViewEvidence
from leapflow.hardware.skills.protocol import SIDE_EFFECT_NONE, ArmState, SkillError, SkillOutcome, SkillPlan
from leapflow.hardware.skills.planning import Evaluation, PlanningSettings
from leapflow.hardware.skills.view_planner import (
    RingViewPlanner, ViewGoal, ViewPlanner, ViewProposalJob,
)
from leapflow.hardware.skills.survey import WorkcellSurvey, _frustum
from tests.test_perception import CUP, _PLANNER, _HOME, _TABLE, _calibrated, _object, _scene, _spec
from tests._harness.skill_stubs import CAMERA_ID, CAMERA_POSE, IDENTITY_MATRIX, PERCEPTION_WORKCELL_YAML, SimRgbdCamera


def accurate(scene, *roles):
    for role in roles or ('camera',):
        diagnosis = scene._frames.diagnosis(role)
        scene._frames._diagnoses[role] = replace(diagnosis, achieved=Accuracy(.2, .01))
    return scene


def full_report(obj, error=.25):
    return ObservationReport(obj.object_id, (
        ViewEvidence('camera', 'c1', 'visible', support=1., error_mm=error, extent_supported=True),))


@pytest.mark.parametrize('raw', [
    {'samples': 0}, {'samples': 10.5}, {'max_views': 17}, {'min_support': 1.1},
    {'enabled': 1}, {'unexpected': 4}, {'body_radii_mm': [0]},
    {'cameras': {'camera.other': {'near_mm': 100}}}, {'observe': {'axis': float('nan')}},
    {'clearance': {'free_box': {'x': [0, 1]}}},
])
def test_policy_refuses_unbounded_or_incomplete_declarations(raw):
    with pytest.raises(ValueError):
        ObservationSettings.from_mapping(raw)


async def test_visible_extent_and_unknown_accuracy_are_separate(tmp_path):
    scene = _scene(_calibrated(tmp_path, CUP))
    (obj,) = await scene.locate(['cup'])
    capture = await scene.capture()
    unknown = scene._observations.analyze(obj, capture)
    assert unknown.support >= .85
    assert unknown.error_mm is None
    assert not unknown.extent_supported
    accurate(scene)
    evidence = scene._observations.analyze(obj, capture)
    assert evidence.extent_supported
    assert evidence.error_mm > .2
    assert ObservationReport(obj.object_id, (evidence,)).to_dict(ObservationSettings())['verdict'] == 'sufficient'


async def test_sampling_density_is_not_a_confidence_multiplier(tmp_path):
    registry = _calibrated(tmp_path, CUP)
    scene = accurate(_scene(registry))
    (obj,) = await scene.locate(['cup'])
    capture = await scene.capture()
    reports = []
    for samples in (81, 225, 400):
        scene._spec = _spec(observation={'samples': samples})
        reports.append(scene._observations.analyze(obj, capture))
    assert max(r.support for r in reports)-min(r.support for r in reports) < .1
    assert len({r.error_mm for r in reports}) == 1


async def test_missing_depth_nearer_surface_and_truncation_do_not_certify(tmp_path):
    registry = _calibrated(tmp_path, CUP)
    scene = accurate(_scene(registry))
    (obj,) = await scene.locate(['cup'])
    capture = await scene.capture()
    blind = replace(capture, frame=replace(capture.frame, depth=bytes(len(capture.frame.depth))))
    evidence = scene._observations.analyze(obj, blind)
    assert evidence.state == 'no_depth' and not evidence.extent_supported
    registry.devices[CAMERA_ID].boxes = [replace(CUP, height=100.)]
    nearer = await scene.capture()
    evidence = await scene._observations.evidence(obj, nearer)
    assert evidence.nearer > .8 and evidence.occluder == 'unknown'
    edge = replace(obj, center=(475., obj.center[1], obj.center[2]))
    evidence = scene._observations.analyze(edge, capture)
    assert evidence.truncated
    assert not ObservationReport(edge.object_id, (evidence,)).complete_views(ObservationSettings())


async def test_relocate_uses_alternative_without_grounding_or_replacing_identity(tmp_path):
    registry = _calibrated(tmp_path, CUP)
    registry.devices['second'] = SimRgbdCamera((CUP,))
    version = registry.calibrate('other.handeye', pose=CAMERA_POSE, parameters={'target_mount': IDENTITY_MATRIX})
    registry.calibrate('camera.other.verify', parameters={
        'training_disjoint': True, 'p95_mm': .5, 'depends_on': {'camera.other': version},
    })
    spec = _spec(roles={**PERCEPTION_WORKCELL_YAML['roles'], 'camera.other': 'second'},
                 frames={**PERCEPTION_WORKCELL_YAML['frames'],
                         'camera.other': {'parent': 'base', 'procedure': 'other.handeye'}})
    scene = accurate(_scene(registry, spec=spec), 'camera', 'camera.other')
    (obj,) = await scene.locate(['cup'])
    locator = registry.devices[spec.model('locator')]
    requests = len(locator.requests)
    registry.devices[CAMERA_ID].boxes = [replace(CUP, height=100.)]
    fresh = await scene.relocate(obj)
    assert fresh.object_id == obj.object_id and fresh.source == obj.source
    assert fresh.camera == 'camera.other'
    assert abs(fresh.height-obj.height) < 3
    assert len(locator.requests) == requests
    attempts = scene.observation(fresh)['attempts']
    assert attempts[0]['nearer'] > 0
    registry.devices['second'].boxes = [replace(CUP, height=100.)]
    with pytest.raises(SkillError) as error:
        await scene.relocate(fresh)
    assert error.value.failure_code == 'observation_insufficient'
    assert scene.object(obj.object_id) == fresh


@pytest.mark.parametrize('offset', [-20.5, -11.2, 20.5])
async def test_viewed_table_offset_preserves_relocate_and_survey_geometry(tmp_path, offset):
    from tests._harness.skill_stubs import TABLE_Z
    registry = _calibrated(tmp_path, CUP)
    registry.devices[CAMERA_ID].table_z = TABLE_Z + offset
    scene = accurate(_scene(registry, spec=_spec(observation={'enabled': True})))
    (obj,) = await scene.locate(['cup'])
    assert obj.table_offset_mm == pytest.approx(offset, abs=1.)
    capture = await scene.capture()
    evidence = scene._observations.analyze(obj, capture)
    assert evidence.support >= .85 and evidence.extent_supported
    confirmed = await scene.relocate(obj, commit=False)
    assert scene.object(obj.object_id) == obj
    assert confirmed.center == pytest.approx(obj.center, abs=3.)
    fitted = await scene._observations.measure(obj, (capture,))
    assert fitted.height == pytest.approx(obj.height, abs=3.)
    assert fitted.center == pytest.approx(obj.center, abs=3.)
    fresh = await scene.relocate(fitted)
    assert fresh.object_id == obj.object_id
    assert fresh.center == pytest.approx(obj.center, abs=3.)
    registry.devices[CAMERA_ID].boxes = []
    with pytest.raises(SkillError) as error:
        await scene.relocate(fresh)
    assert error.value.failure_code == 'target_not_found'


async def test_historical_cloud_cannot_certify_a_missing_live_target(tmp_path):
    registry = _calibrated(tmp_path, CUP)
    scene = accurate(_scene(registry))
    (obj,) = await scene.locate(['cup'])
    old = await scene.capture()
    registry.devices[CAMERA_ID].boxes = []
    live = await scene.capture()
    fresh = await scene._observations.measure(obj, (old, live), latest=(live,))
    assert fresh == obj
    report = scene.observation(fresh)
    assert report['verdict'] != 'sufficient'
    assert {v['capture_id'] for v in report['views']} == {live.capture_id}
    assert old.capture_id in {v['capture_id'] for v in report['attempts']}


async def test_measurement_updates_extent_and_invalidates_grasp_handles(tmp_path):
    scene = accurate(_scene(_calibrated(tmp_path, CUP)))
    (obj,) = await scene.locate(['cup'])
    (grasp, _) = scene.remember_grasps(obj, await _PLANNER.candidates(obj, table=_TABLE, current=_HOME))
    fresh = await scene._observations.measure(obj, (await scene.capture(),))
    assert fresh.object_id == obj.object_id
    with pytest.raises(SkillError):
        scene.grasp(grasp.grasp_id)
    assert scene.observation(obj)['verdict'] == 'insufficient'
    assert scene.observation(fresh)['verdict'] == 'sufficient'


async def test_grasp_tolerance_gate_is_distinct_from_reach(tmp_path):
    obj = _object()
    candidate = (await _PLANNER.candidates(obj, table=_TABLE, current=_HOME))[0]
    report = full_report(obj)
    assert report.for_grasp(obj, candidate, ObservationSettings())['sufficient']
    tight = replace(candidate, open_mm=candidate.width_mm+.5)
    assert not report.for_grasp(obj, tight, ObservationSettings())['sufficient']
    unknown = replace(report, views=(replace(report.views[0], error_mm=None),))
    assert not unknown.for_grasp(obj, candidate, ObservationSettings())['sufficient']
    scene = accurate(_scene(_calibrated(tmp_path, CUP)))
    (remembered,) = await scene.locate(['cup'])
    scene._observations.record(remembered, report.views)
    candidate = replace(candidate, object_id=remembered.object_id)
    survey = WorkcellSurvey(SimpleNamespace(scene=scene))
    larger = replace(remembered, width=remembered.width+20)
    scene._memory.update(larger, scene._basis)
    scene._observations.record(larger, report.views)
    assert not survey.adequate(larger, candidate, remembered)


@pytest.mark.parametrize('count', [1, 3, 7])
def test_capsules_and_cumulative_transforms_have_no_joint_count_assumption(count):
    chain = DhChain(tuple(DhLink(20., 0., 10.) for _ in range(count)), 'standard')
    links = chain.link_transforms([0.]*count)
    assert len(links) == count+1
    assert links[-1] == chain.transform([0.]*count)
    body = capsules([link.translation for link in links], [2.]*count)
    assert len(body) == count
    assert body[0].blocks((10., -10., 5.), (10., 10., 5.))
    assert not body[0].blocks((100., -10., 5.), (100., 10., 5.))
    with pytest.raises(ValueError):
        capsules([link.translation for link in links], [2.]*(count+1))
    assert segment_distance((0., 0., 0.), (0., 0., 0.), (1., 0., 0.), (2., 0., 0.)) == 1


@pytest.mark.parametrize('mount', [Pose(10., 20., 30.), Pose(-20., 10., 50., 90., 20., 30.)])
def test_ring_camera_geometry_composes_arbitrary_mounts(mount):
    from tests._harness.skill_stubs import SIM_INTRINSICS
    goal = ViewGoal(_object(), _TABLE, 'camera.tool', Transform.from_pose(mount),
                    Transform.from_pose(Pose(0., 0., 80.)), SIM_INTRINSICS, 150., 450., 4)
    planner = RingViewPlanner({'azimuths': 4, 'tilt_deg': 10})
    assert isinstance(planner, ViewPlanner)
    proposed = planner.candidates(goal)
    assert len(proposed) == 4
    assert all(_frustum(goal, p) for p in proposed)
    assert len({p.feature for p in proposed}) == 4
    evaluation = ViewProposalJob('ring', '{"azimuths": 4, "tilt_deg": 10}', goal).evaluate(
        SimpleNamespace(check=lambda: None))
    assert evaluation.value.ok
    assert evaluation.value.side_effect_state == SIDE_EFFECT_NONE


@pytest.mark.parametrize('bad', ['nonrigid', 'duplicate', 'camera', 'count', 'translation'])
def test_custom_planner_output_is_checked_before_owner_use(monkeypatch, bad):
    import leapflow.hardware.skills.view_planner as module
    from tests._harness.skill_stubs import SIM_INTRINSICS
    goal = ViewGoal(_object(), _TABLE, 'camera.custom', Transform(), Transform(), SIM_INTRINSICS, 100., 400., 4)
    good = RingViewPlanner({'azimuths': 1}).candidates(goal)[0]
    if bad == 'nonrigid':
        offered = (replace(good, camera_to_base=Transform(((2., 0., 0.), (0., 1., 0.), (0., 0., 1.)))),)
    elif bad == 'camera':
        offered = (replace(good, camera='camera.other'),)
    elif bad == 'count':
        offered = (good,)*5
    elif bad == 'translation':
        offered = (replace(good, camera_to_base=Transform(translation=(0., 0.))),)
    else:
        offered = (good, good)
    class Custom:
        name = 'custom'
        def candidates(self, goal):
            return offered
    monkeypatch.setattr(module, 'discover_view_planners', lambda: {'custom': lambda config: Custom()})
    with pytest.raises(SkillError) as error:
        ViewProposalJob('custom', '{}', goal).evaluate(SimpleNamespace(check=lambda: None))
    assert error.value.failure_code == 'invalid_args'


@pytest.mark.parametrize('config', [{'azimuths': True}, {'azimuths': 2.5}, {'range_fraction': True},
                                    {'tilt_deg': '20'}, {'range_fraction': float('nan')}])
def test_ring_settings_never_coerce_malformed_budgets(config):
    with pytest.raises(ValueError):
        RingViewPlanner(config)


async def test_auto_locate_fuses_unique_geometry_but_not_ambiguous_instances(tmp_path, monkeypatch):
    registry = _calibrated(tmp_path, CUP)
    registry.devices['second'] = SimRgbdCamera((CUP,))
    version = registry.calibrate('other.handeye', pose=CAMERA_POSE, parameters={'target_mount': IDENTITY_MATRIX})
    registry.calibrate('camera.other.verify', parameters={
        'training_disjoint': True, 'p95_mm': .5, 'depends_on': {'camera.other': version},
    })
    spec = _spec(roles={**PERCEPTION_WORKCELL_YAML['roles'], 'camera.other': 'second'},
                 frames={**PERCEPTION_WORKCELL_YAML['frames'],
                         'camera.other': {'parent': 'base', 'procedure': 'other.handeye'}})
    scene = accurate(_scene(registry, spec=spec), 'camera', 'camera.other')
    (obj,) = await scene.locate(['cup'], camera='auto')
    assert len(scene.observation(obj)['views']) == 2
    assert not scene.observation(obj)['conflict']
    locator = registry.devices[spec.model('locator')]
    infer = locator.infer

    async def ambiguous(request):
        result = await infer(request)
        # Two same-label model hypotheses overlap; neither is unique across views.
        return replace(result, detections=result.detections * 2)

    monkeypatch.setattr(locator, 'infer', ambiguous)
    objects = await scene.locate(['cup'], camera='auto')
    assert len(objects) == 4
    assert all(scene.observation(o)['conflict'] for o in objects)


@pytest.mark.parametrize('geometry', ['valid', 'invalid_fk', 'missing_radii'])
async def test_arm_occluder_requires_declared_geometry_and_matching_fk(tmp_path, geometry):
    raw = {'body_radii_mm': [10.]} if geometry != 'missing_radii' else {}
    registry = _calibrated(tmp_path, CUP)
    scene = accurate(_scene(registry, spec=_spec(observation=raw)))
    (obj,) = await scene.locate(['cup'])
    registry.devices[CAMERA_ID].boxes = [replace(CUP, height=100.)]
    capture = await scene.capture()
    origin = capture.placement()[0].translation
    top = (obj.center[0], obj.center[1], obj.center[2]+obj.height/2)
    middle = tuple((a+b)/2 for a, b in zip(origin, top))
    links = (Transform(translation=(middle[0]-100., middle[1], middle[2])),
             Transform(translation=(middle[0]+100., middle[1], middle[2])))

    def check(state):
        if geometry == 'invalid_fk':
            raise SkillError('guard_failed', 'FK disagrees with live telemetry')

    async def state():
        return ArmState(_HOME, _HOME, {'axis': 0.})

    snapshot = SimpleNamespace(names=('axis',), check_kinematics=check,
                               kinematics=lambda: SimpleNamespace(link_transforms=lambda joints: links))
    scene._motion = SimpleNamespace(state=state, geometry_snapshot=lambda: snapshot)
    evidence = await scene._observations.evidence(obj, capture)
    assert evidence.nearer > 0
    assert evidence.occluder == ('arm_predicted' if geometry == 'valid' else 'unknown')
    assert not evidence.extent_supported


@pytest.mark.parametrize('case', ['sufficient', 'jaw_slack', 'unknown_accuracy', 'legacy'])
async def test_production_grasp_publication_keeps_observation_and_reach_separate(tmp_path, case):
    from leapflow.hardware.skills.builtin._grasp_planning import propose

    scene = _scene(_calibrated(tmp_path, CUP), spec=_spec(observation={'enabled': case != 'legacy'}))
    (obj,) = await scene.locate(['cup'])
    report = full_report(obj)
    if case == 'unknown_accuracy':
        report = replace(report, views=(replace(report.views[0], error_mm=None),))
    scene._observations.record(obj, report.views)
    candidates = await _PLANNER.candidates(obj, table=_TABLE, current=_HOME)
    if case == 'jaw_slack':
        candidates = tuple(replace(c, open_mm=c.width_mm+.5) for c in candidates)

    async def offered(obj, **kwargs):
        return candidates

    async def evaluate(jobs, *, deadline):
        return [Evaluation(SkillOutcome.succeeded({}), verified=True) for _ in jobs]

    async def basis():
        return 'unchanged'

    async def state():
        return ArmState(_HOME, _HOME, {})

    evidence = {}
    ctx = SimpleNamespace(scene=scene, workcell=scene._spec, frames=scene._frames,
        motion=SimpleNamespace(state=state, geometry_snapshot=lambda: None),
        grasp_planner=SimpleNamespace(name='test', candidates=offered),
        calculations=SimpleNamespace(evaluate=evaluate), planning_basis=basis,
        planning_settings=PlanningSettings(), planning_deadline=float('inf'),
        evidence=evidence, note=evidence.__setitem__)
    plan = SkillPlan('grasp.propose', {'object': obj.object_id, 'top_k': 2}, 'propose')
    if case in {'jaw_slack', 'unknown_accuracy'}:
        with pytest.raises(SkillError) as error:
            await propose(ctx, plan)
        assert error.value.failure_code == 'observation_insufficient'
        assert error.value.result['reach']['verified']
        assert error.value.result['observation'][0]['reasons']
        with pytest.raises(SkillError):
            scene.grasp('g1')
    else:
        result = await propose(ctx, plan)
        assert result.ok and result.result['grasps']
        assert result.result['reach']['verified']
        assert bool(result.result['observation']) == (case != 'legacy')


def test_optional_view_planner_discovery_preserves_builtin_and_loads_plugin(monkeypatch):
    import importlib.metadata as metadata
    from leapflow.hardware.skills.view_planner import discover_view_planners
    entries = [SimpleNamespace(name='custom', value='custom:planner', load=lambda: RingViewPlanner),
               SimpleNamespace(name='ring', value='override:planner', load=lambda: None)]
    monkeypatch.setattr(metadata, 'entry_points', lambda **kw: entries)
    assert discover_view_planners() == {'ring': RingViewPlanner, 'custom': RingViewPlanner}
