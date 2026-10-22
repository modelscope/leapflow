# Copyright (c) Alibaba, Inc. and its affiliates.
"""SkillRuntime end to end against a simulated Piper workcell.

Covers the two-phase contract (plan -> digest -> execute), every refusal that
must leave the arm untouched (side effect NONE), and every way a started run
ends: arrival, stop, timeout, skill defect -- each with the right halt policy.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from leapflow.hardware.skills import builtin  # noqa: F401 - package import check
from leapflow.hardware.skills.builtin.motion import skills as motion_skills
from leapflow.hardware.skills.builtin.sequence import skills as sequence_skills
from leapflow.hardware.skills.digest import plan_digest, run_permit_digest
from leapflow.hardware.skills.kinematics import DhChain
from leapflow.hardware.skills.ports import (
    SkillContext,
)
from leapflow.hardware.skills.protocol import (
    EFFECT_READ,
    FAILURE_ARM_NOT_READY,
    FAILURE_DEVICE_BUSY,
    FAILURE_GUARD_FAILED,
    FAILURE_INTERNAL_DEFECT,
    FAILURE_INVALID_ARGS,
    FAILURE_OUT_OF_WORKSPACE,
    FAILURE_PLAN_STALE,
    FAILURE_PLAN_UNKNOWN,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_STOPPED,
    FAILURE_TIMEOUT,
    FAILURE_UNKNOWN_SKILL,
    FAILURE_UNREACHABLE,
    FAILURE_UNKNOWN_WORKCELL,
    FAILURE_WRITE_REFUSED,
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_PARTIAL,
    STATUS_FAILED,
    STATUS_HALTED,
    SkillOutcome,
    SkillPlan,
    SkillSpec,
)
from leapflow.hardware.skills.runtime import SkillRun, SkillRuntime, SkillRuntimeSettings
from tests._harness.skill_stubs import (
    JOINTS,
    SEQUENCE,
    START_FLANGE,
    TOOL_LENGTH_MM,
    WORKCELL_ID,
    WORKCELL_YAML,
    SimArmTransport,
    SkillNodeRegistry,
    grip,
    move_by,
    piper_context,
    write_workcell,
)

pytestmark = pytest.mark.asyncio

# Piper X downward-facing fixture inside the existing calibrated box/cone.
_SAFE_JOINTS = (0.0, 80.0, -55.0, 69.99, 10.0, 0.0)


def _skills() -> tuple[Any, ...]:
    return (*motion_skills(), *sequence_skills())


def _registry(tmp_path: Path, *, calibrated: bool = True, **sim: Any) -> SkillNodeRegistry:
    registry = SkillNodeRegistry(SimArmTransport(**sim), store_path=tmp_path / "instrument.duckdb")
    if calibrated:
        registry.calibrate_cell()
    return registry


def _joint_registry(tmp_path: Path, joints: tuple[float, ...] = _SAFE_JOINTS) -> SkillNodeRegistry:
    chain = DhChain.from_mapping(piper_context().kinematics.dh)
    registry = SkillNodeRegistry(
        SimArmTransport(joints=joints, forward=chain.forward),
        store_path=tmp_path / "instrument.duckdb",
    )
    registry.calibrate_cell()
    return registry


def _runtime(
    tmp_path: Path,
    registry: SkillNodeRegistry,
    *,
    workcell: Mapping[str, Any] | None = None,
    skills: Any = None,
    **settings: Any,
) -> SkillRuntime:
    directory = tmp_path / "workcells"
    write_workcell(directory, workcell)
    return SkillRuntime(
        registry,
        settings=SkillRuntimeSettings(workcells_dir=directory, **settings),
        skills=_skills() if skills is None else skills,
        poll_s=0.005,
    )


async def _run(runtime: SkillRuntime, skill: str, args: Mapping[str, Any]) -> SkillRun:
    plan = await runtime.plan(WORKCELL_ID, skill, args)
    assert isinstance(plan, SkillPlan), plan
    run = await runtime.execute(plan.digest)
    assert isinstance(run, SkillRun), run
    finished = await runtime.wait(run.run_id, timeout_s=5.0)
    assert finished is not None and not finished.running
    return finished


async def _until(predicate: Any, timeout_s: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "condition never held"
        await asyncio.sleep(0.005)


# ── Catalog ──────────────────────────────────────────────────────────────────


async def test_list_marks_cartesian_skills_unavailable_until_calibrated(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, _registry(tmp_path, calibrated=False))
    (cell,) = runtime.list()["workcells"]
    by_id = {skill["skill_id"]: skill for skill in cell["skills"]}
    assert not by_id[SEQUENCE]["available"]
    assert "run calib.tcp" in by_id[SEQUENCE]["reason"]
    assert "run calib.workspace" in by_id[SEQUENCE]["reason"]
    assert by_id["motion.home"]["available"]
    assert by_id["arm.state"]["available"]
    assert cell["calibration"] == {}


async def test_list_reports_calibration_versions_and_rejected_workcells(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    write_workcell(tmp_path / "workcells", {**WORKCELL_YAML, "workcell": "ghost", "roles": {"arm": "nope"}})
    runtime = _runtime(tmp_path, registry)
    listing = runtime.list()
    (cell,) = listing["workcells"]
    assert set(cell["calibration"]) == {"tcp", "workspace"}
    assert all(skill["available"] for skill in cell["skills"])
    (rejected,) = listing["rejected"]
    assert rejected["workcell"] == "ghost" and "not declared" in rejected["detail"]


async def test_enabled_patterns_and_workcell_skill_filter_limit_the_catalog(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    runtime = _runtime(tmp_path, registry, enabled=("motion.*",))
    assert runtime.skill_ids == ("motion.home", SEQUENCE)
    narrowed = _runtime(tmp_path, registry, workcell={**WORKCELL_YAML, "skills": ["arm.*"]})
    (cell,) = narrowed.list()["workcells"]
    assert [skill["skill_id"] for skill in cell["skills"]] == ["arm.state"]
    refused = await narrowed.plan(WORKCELL_ID, "motion.home", {})
    assert isinstance(refused, SkillOutcome) and refused.failure_code == FAILURE_UNKNOWN_SKILL


# ── Plan ─────────────────────────────────────────────────────────────────────


async def test_sequence_plan_is_sealed_digested_and_cached(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    runtime = _runtime(tmp_path, registry)
    plan = await runtime.plan(WORKCELL_ID, SEQUENCE, move_by(100))
    assert isinstance(plan, SkillPlan)
    assert (plan.workcell, plan.skill_id, plan.version) == (WORKCELL_ID, SEQUENCE, "1.0.0")
    assert plan.targets[0].z == pytest.approx(START_FLANGE[2] - TOOL_LENGTH_MM + 100)
    assert plan.limits == {
        "speed_pct": 20.0,
        "speed_pct_max": 30.0,
        "move_timeout_s": 5.0,
        "fk_tolerance_mm": 5.0,
        "fk_tolerance_deg": 2.0,
        "joint_path_step_deg": 2.0,
        "max_duration_s": 120.0,
        "max_steps": 16.0,
        "max_wait_s": 10.0,
    }
    assert set(plan.basis["calibration"]) == {"tcp", "workspace"}
    assert plan.digest == plan_digest(plan)
    assert runtime.cached_plan(plan.digest) == plan
    assert runtime.permit_digest(plan) == run_permit_digest(
        workcell=WORKCELL_ID,
        skill_id=SEQUENCE,
        version="1.0.0",
        plan_digest=plan.digest,
        limits=plan.limits,
    )
    assert registry.sim.batches == [], "planning must never write"


async def test_requested_speed_is_capped_by_workcell_and_node(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    plan = await _runtime(tmp_path, registry).plan(WORKCELL_ID, SEQUENCE, move_by(10, speed_pct=90))
    assert isinstance(plan, SkillPlan) and plan.limits["speed_pct"] == 30.0
    node_capped = _runtime(tmp_path, registry, speed_pct_max=10.0)
    plan = await node_capped.plan(WORKCELL_ID, SEQUENCE, move_by(10))
    assert isinstance(plan, SkillPlan)
    assert plan.limits["speed_pct"] == 10.0 and plan.limits["speed_pct_max"] == 10.0


def _steps(*steps: Mapping[str, Any]) -> dict[str, Any]:
    return {"steps": list(steps)}


_UP = {"op": "move", "relative": {"dz": 10}}


@pytest.mark.parametrize(
    ("workcell", "skill", "args", "code"),
    [
        ("nowhere", SEQUENCE, move_by(1), FAILURE_UNKNOWN_WORKCELL),
        (WORKCELL_ID, "motion.fly", {}, FAILURE_UNKNOWN_SKILL),
        (WORKCELL_ID, SEQUENCE, move_by(2000), FAILURE_INVALID_ARGS),
        (WORKCELL_ID, SEQUENCE, _steps({"op": "move", "relative": {"dz": True}}), FAILURE_INVALID_ARGS),
        (WORKCELL_ID, SEQUENCE, _steps({"op": "fly"}), FAILURE_INVALID_ARGS),
        (WORKCELL_ID, SEQUENCE, _steps(), FAILURE_INVALID_ARGS),
        (WORKCELL_ID, SEQUENCE, _steps({"op": "move", "pose": {"x": 250, "y": 0}}), FAILURE_INVALID_ARGS),
        (WORKCELL_ID, SEQUENCE, _steps({"op": "move"}), FAILURE_INVALID_ARGS),
        (WORKCELL_ID, SEQUENCE, _steps({**_UP, "width_mm": 3}), FAILURE_INVALID_ARGS),
        (WORKCELL_ID, SEQUENCE, _steps({**_UP, "expect": "released"}), FAILURE_INVALID_ARGS),
        (WORKCELL_ID, SEQUENCE, _steps({"op": "move", "ref": "thing:o1"}), FAILURE_INVALID_ARGS),
        (WORKCELL_ID, SEQUENCE, _steps(*[_UP] * 17), FAILURE_INVALID_ARGS),
        (WORKCELL_ID, SEQUENCE, _steps({"op": "wait", "seconds": 11}), FAILURE_INVALID_ARGS),
        (WORKCELL_ID, SEQUENCE, move_by(300), FAILURE_OUT_OF_WORKSPACE),
        (WORKCELL_ID, SEQUENCE, _steps(_UP, {"op": "move", "relative": {"dz": 300}}), FAILURE_OUT_OF_WORKSPACE),
        (
            WORKCELL_ID,
            SEQUENCE,
            _steps({"op": "gripper", "grip": "open", "width_mm": 10}),
            FAILURE_INVALID_ARGS,
        ),
        (WORKCELL_ID, SEQUENCE, _steps({"op": "gripper", "width_mm": 90}), FAILURE_INVALID_ARGS),
        (
            WORKCELL_ID,
            SEQUENCE,
            _steps({"op": "move_joints", "joints": {"elbow": 10}}),
            FAILURE_INVALID_ARGS,
        ),
    ],
)
async def test_planning_refusals_are_typed_and_touch_nothing(
    tmp_path: Path, workcell: str, skill: str, args: Mapping[str, Any], code: str
) -> None:
    registry = _registry(tmp_path)
    outcome = await _runtime(tmp_path, registry).plan(workcell, skill, args)
    assert isinstance(outcome, SkillOutcome)
    assert (outcome.failure_code, outcome.side_effect_state) == (code, SIDE_EFFECT_NONE)
    assert registry.sim.batches == []


async def test_uncalibrated_cartesian_plan_names_the_calibration_to_run(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, _registry(tmp_path, calibrated=False))
    outcome = await runtime.plan(WORKCELL_ID, SEQUENCE, move_by(10))
    assert isinstance(outcome, SkillOutcome)
    assert outcome.failure_code == FAILURE_SKILL_UNAVAILABLE
    assert "calib.tcp" in outcome.message


async def test_a_ref_to_an_unknown_handle_is_refused(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    outcome = await _runtime(tmp_path, registry).plan(
        WORKCELL_ID, SEQUENCE, _steps({"op": "move", "ref": "grasp:g9.pregrasp"})
    )
    assert isinstance(outcome, SkillOutcome) and not outcome.ok
    assert outcome.side_effect_state == SIDE_EFFECT_NONE
    assert registry.sim.batches == []


# ── Execute ──────────────────────────────────────────────────────────────────


async def test_lift_by_ten_centimetres_moves_the_flange_and_commits(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    run = await _run(_runtime(tmp_path, registry), SEQUENCE, move_by(100))
    assert run.outcome is not None and run.outcome.ok, run.outcome
    assert run.outcome.side_effect_state == SIDE_EFFECT_COMMITTED
    assert run.outcome.result["tcp"]["z"] == pytest.approx(350.0, abs=0.01)
    assert registry.sim.flange.z == pytest.approx(START_FLANGE[2] + 100, abs=0.01)
    (batch,) = registry.sim.batches
    assert batch[0] == ("arm.speed_pct", 20.0), "speed must precede the motion"
    assert batch[1] == ("arm.pose_path", "linear"), "the arm's first declared path"
    assert [cid for cid, _ in batch[2:]] == ["ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz"]
    assert registry.sim.halts == 0


async def test_tool_frame_offset_follows_the_tool_axis(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    run = await _run(_runtime(tmp_path, registry), SEQUENCE, move_by(50, frame="tool"))
    assert run.outcome is not None and run.outcome.ok, run.outcome
    # The tool points straight down, so +z of the tool is -z of the base.
    assert registry.sim.flange.z == pytest.approx(START_FLANGE[2] - 50, abs=0.01)


async def test_absolute_then_relative_steps_chain_from_the_previous_target(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    program = _steps(
        {"op": "move", "pose": {"x": 300, "y": 50, "z": 200}, "expect": "reached"},
        {"op": "move", "relative": {"dy": -50}},
    )
    run = await _run(_runtime(tmp_path, registry), SEQUENCE, program)
    assert run.outcome is not None and run.outcome.ok, run.outcome
    flange = registry.sim.flange
    assert (flange.x, flange.y, flange.z) == pytest.approx((300, 0, 200 + TOOL_LENGTH_MM), abs=0.01)
    assert [entry["step"] for entry in run.outcome.result["steps"]] == [1, 2]


async def test_home_moves_in_joint_space_after_fk_path_checks(tmp_path: Path) -> None:
    registry = _joint_registry(tmp_path)
    workcell = {**WORKCELL_YAML, "home": {"joint.j1.position": 5.0}}
    run = await _run(_runtime(tmp_path, registry, workcell=workcell), "motion.home", {})
    assert run.outcome is not None and run.outcome.ok, run.outcome
    assert registry.sim.joints["joint.j1.position"] == pytest.approx(5.0)


async def test_home_execute_rechecks_the_path_from_fresh_joints(tmp_path: Path) -> None:
    registry = _joint_registry(tmp_path)
    workcell = {**WORKCELL_YAML, "home": {"joint.j1.position": 5.0}}
    runtime = _runtime(tmp_path, registry, workcell=workcell)
    plan = await runtime.plan(WORKCELL_ID, "motion.home", {})
    assert isinstance(plan, SkillPlan), plan
    registry.sim.joints = dict(zip(JOINTS, (0.0,) * len(JOINTS)))
    registry.sim.pose = list(registry.sim._forward_pose())

    run = await runtime.execute(plan.digest)
    assert isinstance(run, SkillRun), run
    finished = await runtime.wait(run.run_id, timeout_s=5.0)
    assert finished is not None and finished.outcome is not None
    assert finished.outcome.failure_code in {FAILURE_OUT_OF_WORKSPACE, FAILURE_UNREACHABLE}
    assert registry.sim.batches == []


async def test_home_plan_refuses_a_joint_path_with_an_unsafe_middle(tmp_path: Path) -> None:
    start = (-9.6, 57.8, -22.2, 30.5, 63.1, 6.0)
    target = (-50.6, 80.4, -40.8, -4.4, 34.3, 5.7)
    registry = _joint_registry(tmp_path, start)
    home = dict(zip(JOINTS, target))
    outcome = await _runtime(tmp_path, registry, workcell={**WORKCELL_YAML, "home": home}).plan(
        WORKCELL_ID, "motion.home", {}
    )
    assert isinstance(outcome, SkillOutcome) and not outcome.ok
    assert outcome.failure_code in {FAILURE_OUT_OF_WORKSPACE, FAILURE_UNREACHABLE}
    assert registry.sim.batches == []


async def test_joint_move_predicts_the_tcp_for_a_following_relative_step(tmp_path: Path) -> None:
    registry = _joint_registry(tmp_path)
    runtime = _runtime(tmp_path, registry)
    program = _steps(
        {"op": "move_joints", "joints": {"joint.j1.position": 5.0}},
        {"op": "move", "relative": {"dz": 10.0}},
    )
    plan = await runtime.plan(WORKCELL_ID, SEQUENCE, program)
    assert isinstance(plan, SkillPlan), plan
    assert len(plan.targets) == 2
    assert plan.targets[1].z == pytest.approx(plan.targets[0].z + 10.0)
    assert registry.sim.batches == []


async def test_joint_move_without_a_dh_table_fails_closed(tmp_path: Path) -> None:
    registry = _joint_registry(tmp_path)
    assert registry._context.kinematics is not None
    registry._context = replace(  # type: ignore[attr-defined] - test owns this registry
        registry._context,
        kinematics=replace(registry._context.kinematics, dh={}),
    )
    outcome = await _runtime(tmp_path, registry).plan(WORKCELL_ID, "motion.home", {})
    assert isinstance(outcome, SkillOutcome) and not outcome.ok
    assert outcome.failure_code == FAILURE_SKILL_UNAVAILABLE
    assert "kinematics.dh" in outcome.message
    assert registry.sim.batches == []


async def test_closing_on_an_object_satisfies_expect_held(tmp_path: Path) -> None:
    registry = _registry(tmp_path, object_width_mm=30.0)
    run = await _run(_runtime(tmp_path, registry), SEQUENCE, grip("close", expect="held"))
    assert run.outcome is not None and run.outcome.ok, run.outcome
    assert run.outcome.result["gripper_mm"] == pytest.approx(30.0)
    assert run.outcome.result["held"] is None, "no object was referenced, so none is recorded"


async def test_closing_on_nothing_fails_expect_held_and_names_the_step(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    program = _steps(_UP, {"op": "gripper", "grip": "close", "expect": "held", "on_fail": "hold"})
    run = await _run(_runtime(tmp_path, registry), SEQUENCE, program)
    assert run.outcome is not None
    assert (run.outcome.failure_code, run.outcome.status, run.outcome.side_effect_state) == (
        FAILURE_GUARD_FAILED,
        STATUS_FAILED,
        SIDE_EFFECT_PARTIAL,
    )
    assert run.outcome.result["step"] == 2 and run.outcome.result["completed"] == 1
    assert run.outcome.result["guard"] == "held"
    assert registry.sim.halts == 0, "on_fail: hold leaves the arm where it is"


async def test_a_stop_guard_halts_the_arm(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    run = await _run(_runtime(tmp_path, registry), SEQUENCE, grip("close", expect="held"))
    assert run.outcome is not None
    assert (run.outcome.failure_code, run.outcome.status) == (FAILURE_GUARD_FAILED, STATUS_HALTED)
    assert registry.sim.halts == 1


async def test_opening_reaches_the_declared_width_and_is_released(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    run = await _run(_runtime(tmp_path, registry), SEQUENCE, grip("open", expect="released"))
    assert run.outcome is not None and run.outcome.ok, run.outcome
    assert run.outcome.result["gripper_mm"] == pytest.approx(70.0)


async def test_a_blocked_gripper_fails_expect_released(tmp_path: Path) -> None:
    # Something sits between the jaws at 20 mm, so moving them to 10 mm stalls.
    registry = _registry(tmp_path, gripper_mm=20.0, object_width_mm=20.0)
    program = _steps({"op": "gripper", "width_mm": 10, "expect": "released", "on_fail": "hold"})
    run = await _run(_runtime(tmp_path, registry), SEQUENCE, program)
    assert run.outcome is not None
    assert run.outcome.failure_code == FAILURE_GUARD_FAILED
    assert run.outcome.result["guard"] == "released"


async def test_wait_pauses_between_steps(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    program = _steps(_UP, {"op": "wait", "seconds": 0.05, "expect": "reached"})
    run = await _run(_runtime(tmp_path, registry), SEQUENCE, program)
    assert run.outcome is not None and run.outcome.ok, run.outcome
    assert [entry["op"] for entry in run.outcome.result["steps"]] == ["move", "wait"]


async def test_one_plan_executes_once(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, _registry(tmp_path))
    plan = await runtime.plan(WORKCELL_ID, SEQUENCE, move_by(10))
    assert isinstance(plan, SkillPlan)
    first = await runtime.execute(plan.digest)
    assert isinstance(first, SkillRun)
    again = await runtime.execute(plan.digest)
    assert isinstance(again, SkillOutcome) and again.failure_code == FAILURE_PLAN_UNKNOWN
    await runtime.wait(first.run_id, 5.0)


async def test_expired_plan_is_unknown(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, _registry(tmp_path), plan_ttl_s=0.0)
    plan = await runtime.plan(WORKCELL_ID, SEQUENCE, move_by(10))
    assert isinstance(plan, SkillPlan)
    await asyncio.sleep(0.01)
    outcome = await runtime.execute(plan.digest)
    assert isinstance(outcome, SkillOutcome) and outcome.failure_code == FAILURE_PLAN_UNKNOWN


async def test_recalibration_after_planning_makes_the_plan_stale(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    runtime = _runtime(tmp_path, registry)
    plan = await runtime.plan(WORKCELL_ID, SEQUENCE, move_by(10))
    assert isinstance(plan, SkillPlan)
    registry.calibrate("tcp.offset", pose={"x": 0, "y": 0, "z": TOOL_LENGTH_MM + 5})
    outcome = await runtime.execute(plan.digest)
    assert isinstance(outcome, SkillOutcome)
    assert (outcome.failure_code, outcome.side_effect_state) == (
        FAILURE_PLAN_STALE,
        SIDE_EFFECT_NONE,
    )
    assert registry.sim.batches == []


async def test_relative_move_refuses_a_world_that_moved_since_approval(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    runtime = _runtime(tmp_path, registry)
    plan = await runtime.plan(WORKCELL_ID, SEQUENCE, move_by(10))
    assert isinstance(plan, SkillPlan)
    registry.sim.pose[0] += 10.0  # someone nudged the arm
    run = await runtime.execute(plan.digest)
    assert isinstance(run, SkillRun)
    await runtime.wait(run.run_id, 5.0)
    assert run.outcome is not None
    assert (run.outcome.failure_code, run.outcome.side_effect_state) == (
        FAILURE_PLAN_STALE,
        SIDE_EFFECT_NONE,
    )
    assert registry.sim.batches == [] and registry.sim.halts == 0


async def test_disabled_arm_is_not_ready_and_untouched(tmp_path: Path) -> None:
    registry = _registry(tmp_path, enabled=False)
    run = await _run(_runtime(tmp_path, registry), SEQUENCE, move_by(10))
    assert run.outcome is not None
    assert (run.outcome.failure_code, run.outcome.side_effect_state) == (
        FAILURE_ARM_NOT_READY,
        SIDE_EFFECT_NONE,
    )
    assert registry.sim.batches == [] and registry.sim.halts == 0


async def test_node_safety_refusal_is_write_refused_with_no_effect(tmp_path: Path) -> None:
    registry = SkillNodeRegistry(
        store_path=tmp_path / "instrument.duckdb", safety_deny_reason="operator lockout"
    )
    registry.calibrate_cell()
    run = await _run(_runtime(tmp_path, registry), SEQUENCE, move_by(10))
    assert run.outcome is not None
    assert (run.outcome.failure_code, run.outcome.side_effect_state) == (
        FAILURE_WRITE_REFUSED,
        SIDE_EFFECT_NONE,
    )
    assert "operator lockout" in run.outcome.message
    assert registry.sim.batches == [] and registry.sim.halts == 0


# ── Stop, timeout, busy ──────────────────────────────────────────────────────


async def test_busy_devices_refuse_a_second_run_and_stop_halts_once(tmp_path: Path) -> None:
    registry = _registry(tmp_path, stall=True)
    runtime = _runtime(tmp_path, registry)
    plan = await runtime.plan(WORKCELL_ID, SEQUENCE, move_by(50))
    assert isinstance(plan, SkillPlan)
    run = await runtime.execute(plan.digest)
    assert isinstance(run, SkillRun)
    await _until(lambda: registry.sim.batches)
    assert runtime.holds_device("robot_piper_arm")

    second = await runtime.plan(WORKCELL_ID, SEQUENCE, move_by(25))
    assert isinstance(second, SkillPlan)
    busy = await runtime.execute(second.digest)
    assert isinstance(busy, SkillOutcome) and busy.failure_code == FAILURE_DEVICE_BUSY

    stopped = await runtime.stop(run.run_id, "operator pressed stop")
    assert stopped is run and run.outcome is not None
    assert (run.outcome.failure_code, run.outcome.status, run.outcome.side_effect_state) == (
        FAILURE_STOPPED,
        STATUS_HALTED,
        SIDE_EFFECT_PARTIAL,
    )
    assert run.outcome.message == "operator pressed stop"
    assert registry.sim.halts == 1, "stop halts once; the run's own end must not halt again"
    assert not runtime.holds_device("robot_piper_arm")


async def test_arrival_timeout_halts_and_reports_partial(tmp_path: Path) -> None:
    registry = _registry(tmp_path, stall=True)
    workcell = {**WORKCELL_YAML, "limits": {**WORKCELL_YAML["limits"], "move_timeout_s": 0.1}}
    run = await _run(_runtime(tmp_path, registry, workcell=workcell), SEQUENCE, move_by(10))
    assert run.outcome is not None
    assert (run.outcome.failure_code, run.outcome.status, run.outcome.side_effect_state) == (
        FAILURE_TIMEOUT,
        STATUS_HALTED,
        SIDE_EFFECT_PARTIAL,
    )
    assert registry.sim.halts == 1


async def test_a_refused_target_is_unreachable_not_a_timeout(tmp_path: Path) -> None:
    """The firmware's own refusal must not masquerade as a slow arrival.

    ``arm_status`` 0x04 is the vendor's inverse kinematics rejecting the pose
    before executing anything, so the arm is exactly where it started. Reporting
    that as a timeout would halt the run, and a halt engages the emergency stop,
    which releases the drives and drops the arm under gravity. The side effect
    stays PARTIAL because the frame did reach the transport.
    """
    registry = _registry(tmp_path, stall=True, arm_status=0x04)
    workcell = {**WORKCELL_YAML, "limits": {**WORKCELL_YAML["limits"], "move_timeout_s": 30.0}}
    run = await _run(_runtime(tmp_path, registry, workcell=workcell), SEQUENCE, move_by(10))
    assert run.outcome is not None
    assert (run.outcome.failure_code, run.outcome.status, run.outcome.side_effect_state) == (
        FAILURE_UNREACHABLE,
        STATUS_FAILED,
        SIDE_EFFECT_PARTIAL,
    )
    assert registry.sim.halts == 0, "a refused target must never engage the emergency stop"


async def test_a_stale_refusal_code_does_not_block_a_valid_move(tmp_path: Path) -> None:
    """A refusal code left over from an earlier target must not end the next move."""
    registry = _registry(tmp_path, arm_status=0x04)
    run = await _run(_runtime(tmp_path, registry), SEQUENCE, move_by(10))
    assert run.outcome is not None
    assert run.outcome.status == "succeeded", run.outcome.message


async def test_stopping_a_run_that_never_moved_does_not_halt(tmp_path: Path) -> None:
    class _Waits:
        spec = SkillSpec(
            skill_id="test.wait", version="1", card="Waits.", args_schema={"type": "object"}
        )

        async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
            return SkillPlan(skill_id="test.wait", args=args, summary="wait")

        async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
            while not ctx.cancelled:
                await asyncio.sleep(0.005)
            return SkillOutcome.succeeded()

    registry = _registry(tmp_path)
    runtime = _runtime(tmp_path, registry, skills=[_Waits()])
    plan = await runtime.plan(WORKCELL_ID, "test.wait", {})
    assert isinstance(plan, SkillPlan)
    run = await runtime.execute(plan.digest)
    assert isinstance(run, SkillRun)
    await runtime.stop(run.run_id)
    assert run.outcome is not None and run.outcome.failure_code == FAILURE_STOPPED
    assert run.outcome.side_effect_state == SIDE_EFFECT_NONE
    assert registry.sim.halts == 0, "a halt latches the e-stop; never halt a still arm"


# ── Defects and READ skills ──────────────────────────────────────────────────


async def test_skill_defects_become_internal_defect_outcomes(tmp_path: Path) -> None:
    class _Broken:
        spec = SkillSpec(
            skill_id="test.broken", version="1", card="Breaks.", args_schema={"type": "object"}
        )

        async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
            return SkillPlan(skill_id="test.broken", args=args, summary="break")

        async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
            await ctx.motion.gripper(10.0)
            raise AttributeError("no such thing")

    registry = _registry(tmp_path)
    runtime = _runtime(tmp_path, registry, skills=[_Broken()])
    run = await _run(runtime, "test.broken", {})
    assert run.outcome is not None
    assert (run.outcome.failure_code, run.outcome.side_effect_state) == (
        FAILURE_INTERNAL_DEFECT,
        SIDE_EFFECT_PARTIAL,
    )
    assert registry.sim.halts == 1


async def test_moving_during_plan_is_an_internal_defect(tmp_path: Path) -> None:
    class _Sneaky:
        spec = SkillSpec(
            skill_id="test.sneaky", version="1", card="Moves early.", args_schema={"type": "object"}
        )

        async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
            await ctx.motion.gripper(10.0)
            raise AssertionError("unreachable")

        async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
            raise AssertionError("unreachable")

    registry = _registry(tmp_path)
    outcome = await _runtime(tmp_path, registry, skills=[_Sneaky()]).plan(
        WORKCELL_ID, "test.sneaky", {}
    )
    assert isinstance(outcome, SkillOutcome)
    assert outcome.failure_code == FAILURE_INTERNAL_DEFECT
    assert registry.sim.batches == []


async def test_read_skill_answers_from_plan_without_a_cached_plan(tmp_path: Path) -> None:
    class _Where:
        spec = SkillSpec(
            skill_id="test.where",
            version="1",
            card="Reports the TCP.",
            args_schema={"type": "object"},
            effect=EFFECT_READ,
        )

        async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
            return SkillPlan(skill_id="test.where", args=args, summary="where")

        async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
            state = await ctx.motion.state()
            ctx.note("polled", True)
            return SkillOutcome.succeeded(
                {"tcp": state.tcp.to_dict()}, side_effect_state=SIDE_EFFECT_NONE
            )

    runtime = _runtime(tmp_path, _registry(tmp_path), skills=[_Where()])
    outcome = await runtime.plan(WORKCELL_ID, "test.where", {})
    assert isinstance(outcome, SkillOutcome) and outcome.ok
    assert outcome.result["tcp"]["z"] == pytest.approx(250.0)
    assert outcome.evidence == {"polled": True}
    assert runtime.status() == ()


async def test_record_calibration_is_versioned_and_visible_to_the_next_plan(
    tmp_path: Path,
) -> None:
    class _Calibrates:
        spec = SkillSpec(
            skill_id="calib.test",
            version="1",
            card="Writes the safe box.",
            args_schema={"type": "object"},
        )

        async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
            return SkillPlan(skill_id="calib.test", args=args, summary="calibrate")

        async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
            ctx.record_calibration(
                "workspace.safe_box",
                parameters={"x": [120, 420], "y": [-150, 150], "z": [90, 400]},
            )
            return SkillOutcome.succeeded(side_effect_state=SIDE_EFFECT_NONE)

    registry = _registry(tmp_path)
    runtime = _runtime(tmp_path, registry, skills=[_Calibrates(), *_skills()])
    before = (await runtime.plan(WORKCELL_ID, SEQUENCE, move_by(10))).basis["calibration"]
    run = await _run(runtime, "calib.test", {})
    assert run.outcome is not None and run.outcome.ok, run.outcome
    assert "workspace.safe_box" in run.outcome.evidence["calibrations"]
    after = (await runtime.plan(WORKCELL_ID, SEQUENCE, move_by(10))).basis["calibration"]
    assert after["workspace"] > before["workspace"] and after["tcp"] == before["tcp"]


async def test_close_stops_every_running_run(tmp_path: Path) -> None:
    registry = _registry(tmp_path, stall=True)
    runtime = _runtime(tmp_path, registry)
    plan = await runtime.plan(WORKCELL_ID, SEQUENCE, move_by(20))
    assert isinstance(plan, SkillPlan)
    run = await runtime.execute(plan.digest)
    assert isinstance(run, SkillRun)
    await _until(lambda: registry.sim.batches)
    await runtime.close()
    assert run.outcome is not None and run.outcome.failure_code == FAILURE_STOPPED
    assert registry.sim.halts == 1
