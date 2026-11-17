# Copyright (c) Alibaba, Inc. and its affiliates.
"""Batch selection and owner snapshot invariants through the production evaluator."""
from __future__ import annotations

import asyncio
import pickle
import time
from dataclasses import replace
from typing import Any

import pytest

from leapflow.hardware.skills.builtin.scene import skills as scene_skills
from leapflow.hardware.skills.builtin.sequence import skills as sequence_skills
from leapflow.hardware.skills.planning import (
    Deadline, Evaluation, FrozenJoints, PlanningSettings, Selection, candidates, encoded,
)
from leapflow.hardware.skills.protocol import SkillError, SkillOutcome, SkillPlan
from tests._harness.skill_stubs import (
    ARM_ID, JOINTS, PERCEPTION_WORKCELL_YAML, SEQUENCE, WORKCELL_ID, WORKCELL_YAML, move_by,
)
from tests.test_perception import CUP, _calibrated
from tests.test_skill_runtime import _joint_registry, _registry, _runtime

pytestmark = [pytest.mark.component, pytest.mark.usefixtures("close_skill_runtimes")]


class CaptureCalculations:
    """Owner-loop spy; real spawned calculations remain the implementation under test."""
    def __init__(self, delegate: Any, after: Any = None) -> None:
        self.delegate = delegate
        self.jobs: list[Any] = []
        self.after = after

    async def evaluate(self, jobs: Any, *, deadline: float) -> Any:
        self.jobs.extend(jobs)
        values = await self.delegate.evaluate(jobs, deadline=deadline)
        if self.after is not None:
            await self.after()
        return values

    async def close(self) -> None:
        await self.delegate.close()


def offered(*args: Any) -> list[dict[str, Any]]:
    return [{"id": f"path-{i}", "args": value} for i, value in enumerate(args)]


@pytest.mark.parametrize("raw", [
    [], [{"id": "", "args": {}}], [{"id": " " * 2, "args": {}}],
    [{"id": "x" * 65, "args": {}}], [{"id": 1, "args": {}}],
    [{"id": "x", "args": []}], [{"id": "x", "args": {}, "extra": 1}],
    [{"id": "x", "args": {}}, {"id": "x", "args": {}}],
])
def test_candidate_contract_refuses_malformed_identity(raw: Any) -> None:
    with pytest.raises(SkillError) as error:
        candidates(raw, PlanningSettings())
    assert error.value.failure_code == "invalid_args"


@pytest.mark.parametrize("raw", [
    {"require_verified": 1}, {"order": []}, {"order": ["unknown"]},
    {"order": ["wait_s", "wait_s"]}, {"order": [{"metric": "wait_s", "direction": "up"}]},
    {"order": [{"metric": "wait_s", "extra": 1}]}, {"unknown": True},
])
def test_selection_contract_refuses_ambiguous_policy(raw: Any) -> None:
    with pytest.raises(SkillError):
        Selection.from_mapping(raw)


def test_lexicographic_selection_verification_missing_values_and_stable_ties() -> None:
    plan = SkillPlan(skill_id=SEQUENCE, args={}, summary="selection fixture")
    values = [Evaluation(plan, {"joint_margin": 0.9}, False),
              Evaluation(plan, {"joint_margin": None, "joint_travel": 0}, True),
              Evaluation(plan, {"joint_margin": 0.2, "joint_travel": 2}, True),
              Evaluation(plan, {"joint_margin": 0.2, "joint_travel": 1}, True),
              Evaluation(plan, {"joint_margin": 0.2, "joint_travel": 1}, True)]
    for verified in (True, False):
        selected, rationale = Selection(require_verified=verified).choose(values)
        assert selected == 3
        assert rationale["omitted_metrics"] == ["tcp_distance_mm"]
        assert rationale["omission_reason"]
    assert Selection.from_mapping({"order": [{"metric": "joint_travel", "direction": "max"}]}).choose(values)[0] == 2
    assert Selection().choose(values[:1])[0] is None
    assert Selection(require_verified=False).choose(values[:1])[0] == 0
    with pytest.raises(SkillError, match="nonfinite"):
        Selection().choose([Evaluation(plan, {"joint_margin": float("nan")}, True)])


def test_joint_snapshot_is_immutable_detached_and_pickleable() -> None:
    source = {"j1": 5.0}
    snapshot = FrozenJoints.capture(source)
    source["j1"] = 90
    assert dict(pickle.loads(pickle.dumps(snapshot))) == {"j1": 5.0}
    with pytest.raises(TypeError):
        snapshot["j1"] = 10  # type: ignore[index]


async def test_real_batch_shares_start_deduplicates_and_seals_only_selected(tmp_path: Any) -> None:
    registry = _joint_registry(tmp_path)
    runtime = _runtime(tmp_path, registry, planning=PlanningSettings(workers=2))
    capture = CaptureCalculations(runtime._calculations)
    runtime._calculations = capture
    args = offered(move_by(10), move_by(5), move_by(5), move_by(1000))
    result = await runtime.plan_batch(WORKCELL_ID, SEQUENCE, args, {"order": ["tcp_distance_mm"]})
    assert isinstance(result.selected, SkillPlan), result
    assert result.report["selected_id"] == "path-1"
    assert [r["status"] for r in result.report["candidates"]] == ["feasible", "feasible", "feasible", "refused"]
    evidence = result.report["evidence"]
    assert (evidence["candidate_count"], evidence["unique_programs"], evidence["duplicate_reuse"]) == (4, 3, 1)
    assert evidence["state_reads"] == 1 and evidence["ik_calls"] > 0 and evidence["llm_calls"] == 0
    assert len(capture.jobs) == 3
    assert all(job.state is capture.jobs[0].state for job in capture.jobs)
    assert all(job.view is capture.jobs[0].view for job in capture.jobs)
    assert isinstance(capture.jobs[0].state.joints, FrozenJoints)
    geometry = capture.jobs[0].view.motion
    assert geometry.metric_bounds != geometry.bounds
    assert all(lo <= safe_lo <= safe_hi <= hi for (lo, hi), (safe_lo, safe_hi)
               in zip(geometry.metric_bounds, geometry.bounds))
    assert len(runtime._plans) == 1
    assert result.selected.basis["batch"] == result.report
    assert result.selected.args["steps"][0]["relative"]["dz"] == 5
    assert registry.sim.batches == []
    assert len({r["evidence"].get("worker_pid") for r in result.report["candidates"]}) == 2
    # Independent mutable trackers give duplicates and separately evaluated paths stable metrics.
    assert result.report["candidates"][1]["metrics"] == result.report["candidates"][2]["metrics"]
    run = await runtime.execute(result.selected.digest)
    finished = await runtime.wait(run.run_id, timeout_s=3)
    assert finished.outcome.ok
    assert len([batch for batch in registry.sim.batches if any(n == "ee.z" for n, _ in batch)]) == 1
    assert runtime.cached_plan(result.selected.digest) is None
    assert (await runtime.execute(result.selected.digest)).failure_code == "plan_unknown"


async def test_unverified_policy_preserves_legacy_and_requires_explicit_batch_opt_out(tmp_path: Any) -> None:
    registry = _registry(tmp_path)
    runtime = _runtime(tmp_path, registry)
    assert isinstance(await runtime.plan(WORKCELL_ID, SEQUENCE, move_by(5)), SkillPlan)
    result = await runtime.plan_batch(WORKCELL_ID, SEQUENCE, offered(move_by(5)))
    assert isinstance(result.selected, SkillOutcome) and not result.selected.ok
    assert result.report["selected_id"] is None
    assert result.report["candidates"][0]["status"] == "unverified"
    result = await runtime.plan_batch(WORKCELL_ID, SEQUENCE, offered(move_by(5)), {"require_verified": False})
    assert isinstance(result.selected, SkillPlan)
    assert result.report["candidates"][0]["verified"] is False
    assert registry.sim.batches == []


@pytest.mark.parametrize("batch", [False, True])
async def test_arm_only_snapshot_plans_motion_but_refuses_gripper(tmp_path: Any, batch: bool) -> None:
    registry = _joint_registry(tmp_path)
    runtime = _runtime(tmp_path, registry, workcell={**WORKCELL_YAML, "roles": {"arm": ARM_ID}})
    capture = CaptureCalculations(runtime._calculations)
    runtime._calculations = capture
    state = await runtime.plan(WORKCELL_ID, "arm.state", {})
    assert isinstance(state, SkillOutcome) and state.ok, state
    result = (await runtime.plan_batch(WORKCELL_ID, SEQUENCE, offered(move_by(5)))).selected if batch else await runtime.plan(WORKCELL_ID, SEQUENCE, move_by(5))
    assert isinstance(result, SkillPlan), result
    geometry = capture.jobs[0].view.motion
    assert geometry.gripper_tolerance is None
    with pytest.raises(SkillError, match="no gripper"):
        _ = geometry.gripper_tolerance_mm
    runtime.discard_plan(result.digest)
    args = {"steps": [{"op": "gripper", "width_mm": 20}]}
    if batch:
        report = await runtime.plan_batch(WORKCELL_ID, SEQUENCE, offered(args))
        assert isinstance(report.selected, SkillOutcome) and not report.selected.ok
        assert report.report["selected_id"] is None
        assert report.report["candidates"][0]["problems"][0]["failure_code"] == "skill_unavailable"
    else:
        refused = await runtime.plan(WORKCELL_ID, SEQUENCE, args)
        assert isinstance(refused, SkillOutcome) and refused.failure_code == "skill_unavailable"
    assert runtime._plans == {} and registry.sim.batches == []


async def test_all_refused_and_mixed_schema_failures_report_every_id(tmp_path: Any) -> None:
    registry = _joint_registry(tmp_path)
    runtime = _runtime(tmp_path, registry)
    result = await runtime.plan_batch(WORKCELL_ID, SEQUENCE, offered(move_by(1000), {"steps": []}))
    assert isinstance(result.selected, SkillOutcome)
    assert [r["id"] for r in result.report["candidates"]] == ["path-0", "path-1"]
    assert all(r["status"] == "refused" and r["problems"] for r in result.report["candidates"])
    assert runtime._plans == {} and registry.sim.batches == []
    unsupported = await runtime.plan_batch(WORKCELL_ID, "motion.home", offered({}))
    assert unsupported.selected.failure_code == "skill_unavailable"


@pytest.mark.parametrize("changed", ["calibration", "workcell", "shutdown", "generation"])
async def test_changed_shared_basis_invalidates_entire_batch(tmp_path: Any, changed: str) -> None:
    registry = _joint_registry(tmp_path)
    runtime = _runtime(tmp_path, registry)
    original_basis = runtime._generation_basis
    generation = {"changed": False}

    async def basis(spec: Any) -> Any:
        value = await original_basis(spec)
        return {**value, "test_generation": 1} if generation["changed"] else value

    runtime._generation_basis = basis

    async def mutate() -> None:
        if changed == "calibration":
            registry.calibrate_cell()
        elif changed == "workcell":
            runtime._workcells[WORKCELL_ID] = replace(runtime._workcells[WORKCELL_ID])
        elif changed == "shutdown":
            runtime._closing = True
        else:
            generation["changed"] = True

    runtime._calculations = CaptureCalculations(runtime._calculations, mutate)
    result = await runtime.plan_batch(WORKCELL_ID, SEQUENCE, offered(move_by(5), move_by(10)))
    assert result.selected.failure_code == "plan_stale", result
    assert result.report["selected_id"] is None
    assert runtime._plans == {} and registry.sim.batches == []


@pytest.mark.parametrize("batch", [False, True])
async def test_scene_replacement_during_final_generation_read_prevents_publication(tmp_path: Any, batch: bool) -> None:
    registry = _calibrated(tmp_path, CUP)
    runtime = _runtime(tmp_path, registry, workcell=PERCEPTION_WORKCELL_YAML,
                       skills=(*scene_skills(), *sequence_skills()))
    located = await runtime.plan(WORKCELL_ID, "perceive.locate", {"queries": ["cup"]})
    obj = located.result["objects"][0]["object_id"]
    args = {"steps": [{"op": "move", "ref": f"object:{obj}.top", "offset": {"dz": 150}},
                      {"op": "move", "relative": {"dz": 5}}]}
    original = runtime._generation_basis
    calls = 0

    async def basis(spec: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            await runtime.plan(WORKCELL_ID, "perceive.locate", {"queries": ["cup"]})
        return await original(spec)

    runtime._generation_basis = basis
    result = await runtime.plan_batch(WORKCELL_ID, SEQUENCE, offered(args), {"require_verified": False}) if batch else await runtime.plan(WORKCELL_ID, SEQUENCE, args)
    outcome = result.selected if batch else result
    assert outcome.failure_code == "plan_stale", outcome
    assert runtime._plans == {} and registry.sim.batches == []


@pytest.mark.parametrize("change", ["drift", "missing", "nonfinite"])
async def test_equivalent_tcp_with_changed_start_joints_refuses_before_commands(tmp_path: Any, change: str) -> None:
    registry = _registry(tmp_path)
    runtime = _runtime(tmp_path, registry)
    result = await runtime.plan_batch(WORKCELL_ID, SEQUENCE, offered(move_by(5)), {"require_verified": False})
    assert isinstance(result.selected, SkillPlan)
    if change == "missing":
        registry.sim.joints.pop(JOINTS[0])
    else:
        registry.sim.joints[JOINTS[0]] = 10 if change == "drift" else float("nan")
    run = await runtime.execute(result.selected.digest)
    finished = await runtime.wait(run.run_id, timeout_s=3)
    assert finished.outcome.failure_code in {"plan_stale", "device_fault"}, finished.outcome
    assert registry.sim.batches == []


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("source", ["joint", "position", "rotation"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
async def test_nonfinite_start_telemetry_cannot_become_unverified_executable_plan(
    tmp_path: Any, batch: bool, source: str, value: float,
) -> None:
    registry = _registry(tmp_path)
    if source == "joint":
        registry.sim.joints[JOINTS[0]] = value
    else:
        registry.sim.pose[0 if source == "position" else 3] = value
    runtime = _runtime(tmp_path, registry)
    result = await runtime.plan_batch(WORKCELL_ID, SEQUENCE, offered(move_by(5)), {"require_verified": False}) if batch else await runtime.plan(WORKCELL_ID, SEQUENCE, move_by(5))
    outcome = result.selected if batch else result
    assert isinstance(outcome, SkillOutcome) and outcome.failure_code == "device_fault"
    assert runtime._plans == {} and registry.sim.batches == []


async def test_request_payload_and_selected_response_budgets_cache_nothing(tmp_path: Any) -> None:
    runtime = _runtime(tmp_path, _registry(tmp_path), planning=PlanningSettings(max_bytes=1024))
    huge = offered({"steps": [{"op": "wait", "seconds": 1}], "padding": "x" * 2000})
    result = await runtime.plan_batch(WORKCELL_ID, SEQUENCE, huge)
    assert result.selected.failure_code == "invalid_args" and runtime._plans == {}
    result = await runtime.plan_batch(WORKCELL_ID, SEQUENCE, offered(move_by(5)), {"require_verified": False})
    assert result.selected.failure_code == "invalid_args" and runtime._plans == {}
    assert len(encoded({"outcome": result.selected.to_dict(), "batch": result.report}).encode()) <= 1024
    assert result.report["selected_id"] is None


async def test_shutdown_cancels_owner_preparation_before_worker_admission(tmp_path: Any) -> None:
    runtime = _runtime(tmp_path, _registry(tmp_path))
    entered = asyncio.Event()
    skill = runtime._skills[SEQUENCE]
    original = skill.prepare

    async def prepare(ctx: Any, args: Any) -> Any:
        entered.set()
        await asyncio.sleep(60)
        return await original(ctx, args)

    skill.prepare = prepare
    task = asyncio.create_task(runtime.plan_batch(WORKCELL_ID, SEQUENCE, offered(move_by(5))))
    await asyncio.wait_for(entered.wait(), 2)
    await runtime.close()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime._planning_tasks == set() and runtime._plans == {}
    assert runtime._calculations._workers == {}


async def test_grasp_deadline_includes_read_plan_before_candidate_preparation(tmp_path: Any) -> None:
    registry = _calibrated(tmp_path, CUP)
    runtime = _runtime(tmp_path, registry, workcell=PERCEPTION_WORKCELL_YAML,
                       skills=scene_skills(), planning=PlanningSettings(timeout_s=0.1))
    located = await runtime.plan(WORKCELL_ID, "perceive.locate", {"queries": ["cup"]})
    obj = located.result["objects"][0]["object_id"]
    skill = runtime._skills["grasp.propose"]
    original = skill.plan

    async def delayed(ctx: Any, args: Any) -> Any:
        await asyncio.sleep(0.3)
        return await original(ctx, args)

    skill.plan = delayed
    start = time.monotonic()
    result = await runtime.plan(WORKCELL_ID, "grasp.propose", {"object": obj})
    assert result.failure_code == "timeout" and result.effect == "read"
    assert time.monotonic() - start < 0.3
    assert runtime._scenes[WORKCELL_ID]._grasps == {} and registry.sim.batches == []


async def test_single_and_batch_end_to_end_deadline_include_preparation(tmp_path: Any) -> None:
    runtime = _runtime(tmp_path, _registry(tmp_path), planning=PlanningSettings(timeout_s=0.1))
    skill = runtime._skills[SEQUENCE]
    original = skill.prepare

    async def prepare(ctx: Any, args: Any) -> Any:
        await asyncio.sleep(0.3)
        return await original(ctx, args)

    skill.prepare = prepare
    for operation in (runtime.plan(WORKCELL_ID, SEQUENCE, move_by(5)),
                      runtime.plan_batch(WORKCELL_ID, SEQUENCE, offered(move_by(5)))):
        start = time.monotonic()
        result = await operation
        outcome = result.selected if hasattr(result, "selected") else result
        assert outcome.failure_code == "timeout"
        assert time.monotonic() - start < 0.3
        assert runtime._plans == {}
    with pytest.raises(SkillError):
        Deadline(time.monotonic() - 1).check()
