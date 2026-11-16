# Copyright (c) Alibaba, Inc. and its affiliates.
"""Calibration skills against a simulated Piper workcell.

Pins the first-run journey (fresh cell -> calib.tcp -> calib.workspace ->
Cartesian skills available), the fitting maths behind it, every out-of-tolerance
verdict (``check_failed``, NONE at planning, COMMITTED once something moved),
and staleness: a workspace measured with a superseded TCP stops resolving.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from leapflow.hardware.skill_results import next_step
from leapflow.hardware.skills.args import validate_args
from leapflow.hardware.skills.builtin.calibration import BRINGUP_PROCEDURE
from leapflow.hardware.skills.builtin.calibration import skills as calibration_skills
from leapflow.hardware.skills.builtin.motion import skills as motion_skills
from leapflow.hardware.skills.builtin.sequence import skills as sequence_skills
from leapflow.hardware.skills.geometry import (
    Pose,
    fit_plane,
    mat_vec,
    plane_height,
    rpy_to_matrix,
    solve_pivot,
)
from leapflow.hardware.skills.protocol import (
    FAILURE_CHECK_FAILED,
    FAILURE_INVALID_ARGS,
    FAILURE_OUT_OF_WORKSPACE,
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SkillError,
    SkillOutcome,
    SkillPlan,
)
from leapflow.hardware.skills.runtime import SkillRun, SkillRuntime, SkillRuntimeSettings
from tests._harness.skill_stubs import (
    ARM_ID,
    SAFE_BOX,
    SEQUENCE,
    TOOL_LENGTH_MM,
    WORKCELL_ID,
    WORKCELL_YAML,
    SimArmTransport,
    SkillNodeRegistry,
    move_by,
    write_workcell,
)

TIP = (0.0, 0.0, TOOL_LENGTH_MM)
PIVOT = (300.0, 20.0, 50.0)
PIVOT_RPY = ((180, 0, 0), (165, 15, 0), (170, -15, 45), (-170, 10, -60), (180, 25, 90))
BOOTSTRAP = WORKCELL_YAML["workspace"]["bootstrap_box"]
# z = 0.02 x + 0.01 y + 100: highest over x[160, 390] y[-150, 150] at (390, 150).
TILTED_TABLE = [
    [x, y, 0.02 * x + 0.01 * y + 100.0]
    for x, y in ((160, -150), (390, -150), (390, 150), (160, 150))
]
BOX_ARGS = {"x": [160, 390], "y": [-150, 150], "z_max": 380}


def _registry(tmp_path: Path, **sim: Any) -> SkillNodeRegistry:
    registry = SkillNodeRegistry(SimArmTransport(**sim), store_path=tmp_path / "instrument.duckdb")
    registry.last_command = lambda device_id, channel_id: None  # type: ignore[method-assign]
    return registry


def _runtime(
    tmp_path: Path,
    registry: SkillNodeRegistry,
    *,
    calibration: Mapping[str, float] | None = None,
) -> SkillRuntime:
    directory = tmp_path / "workcells"
    workcell = WORKCELL_YAML
    if calibration is not None:
        workcell = {**WORKCELL_YAML, "calibration": calibration}
    write_workcell(directory, workcell)
    return SkillRuntime(
        registry,
        settings=SkillRuntimeSettings(workcells_dir=directory),
        skills=[*motion_skills(), *sequence_skills(), *calibration_skills()],
        poll_s=0.005,
    )


async def _run(runtime: SkillRuntime, skill: str, args: Mapping[str, Any]) -> SkillRun:
    plan = await runtime.plan(WORKCELL_ID, skill, args)
    assert isinstance(plan, SkillPlan), plan
    run = await runtime.execute(plan.digest)
    assert isinstance(run, SkillRun), run
    finished = await runtime.wait(run.run_id, timeout_s=10.0)
    assert finished is not None and not finished.running
    return finished


async def _refusal(runtime: SkillRuntime, skill: str, args: Mapping[str, Any]) -> SkillOutcome:
    outcome = await runtime.plan(WORKCELL_ID, skill, args)
    assert isinstance(outcome, SkillOutcome) and not outcome.ok, outcome
    return outcome


def _skills(runtime: SkillRuntime) -> dict[str, Mapping[str, Any]]:
    (cell,) = runtime.list()["workcells"]
    return {skill["skill_id"]: skill for skill in cell["skills"]}


def _pivot_samples(tip: tuple[float, ...] = TIP) -> list[list[float]]:
    """Flange poses with the tool tip resting on PIVOT: ``p = P - R t``."""
    samples = []
    for rpy in PIVOT_RPY:
        offset = mat_vec(rpy_to_matrix(*rpy), tip)
        samples.append([PIVOT[i] - offset[i] for i in range(3)] + list(rpy))
    return samples


def _pose_batches(sim: SimArmTransport) -> list[dict[str, float]]:
    return [dict(batch) for batch in sim.batches if any(cid == "ee.x" for cid, _ in batch)]


def _gripper_targets(sim: SimArmTransport) -> list[float]:
    return [value for batch in sim.batches for cid, value in batch if cid == "gripper.position"]


# ── Geometry ─────────────────────────────────────────────────────────────────


def test_pivot_fit_recovers_tip_and_pivot() -> None:
    tip, pivot, residual = solve_pivot([Pose(*sample) for sample in _pivot_samples()])
    assert tip == pytest.approx(TIP, abs=1e-6)
    assert pivot == pytest.approx(PIVOT, abs=1e-6)
    assert residual == pytest.approx(0.0, abs=1e-6)


def test_pivot_about_one_axis_is_unobservable() -> None:
    poses = [Pose(300, 20, 170, 180, 0, rz) for rz in (0, 30, 60, 90)]
    with pytest.raises(ValueError):
        solve_pivot(poses)


def test_plane_fit_recovers_a_tilted_table() -> None:
    normal, offset, residual = fit_plane(TILTED_TABLE)
    assert normal[2] > 0 and residual == pytest.approx(0.0, abs=1e-9)
    assert plane_height(normal, offset, 390, 150) == pytest.approx(109.3)
    with pytest.raises(ValueError):
        fit_plane([(0, 0, 0), (1, 1, 1), (2, 2, 2)])


def test_nested_array_arguments_validate_each_item() -> None:
    schema = {
        "type": "object",
        "properties": {
            "points": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}, "minItems": 3},
            }
        },
    }
    assert validate_args(schema, {"points": [[1, 2, 3]]}) == {"points": [[1.0, 2.0, 3.0]]}
    with pytest.raises(SkillError) as error:
        validate_args(schema, {"points": [[1, 2, 3], [1, 2]]})
    assert error.value.failure_code == FAILURE_INVALID_ARGS
    assert "points[1]" in error.value.message


# ── First run ────────────────────────────────────────────────────────────────


async def test_a_fresh_cell_is_calibrated_in_order_and_unlocks_cartesian_skills(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    runtime = _runtime(tmp_path, registry)
    before = _skills(runtime)
    assert before["calib.bringup"]["available"] and before["calib.tcp"]["available"]
    assert not before["calib.workspace"]["available"]
    assert "tcp is not calibrated" in before["calib.workspace"]["reason"]
    assert not before[SEQUENCE]["available"]

    tcp = await _run(runtime, "calib.tcp", {"offset": [0, 0, TOOL_LENGTH_MM]})
    assert tcp.outcome is not None and tcp.outcome.ok, tcp.outcome
    assert registry.sim.batches == []  # declaring the TCP moves nothing
    assert _skills(runtime)["calib.workspace"]["available"]

    workspace = await _run(runtime, "calib.workspace", {**BOX_ARGS, "table_z": 100})
    outcome = workspace.outcome
    assert outcome is not None and outcome.ok, outcome
    assert outcome.side_effect_state == SIDE_EFFECT_COMMITTED
    assert outcome.result["workspace"] == {"x": [160, 390], "y": [-150, 150], "z": [130, 380]}
    assert outcome.result["poses_visited"] == 10
    moves = _pose_batches(registry.sim)
    assert len(moves) == 10
    for move in moves:  # every TCP target stays inside the human-declared bootstrap box
        assert BOOTSTRAP["x"][0] <= move["ee.x"] <= BOOTSTRAP["x"][1]
        assert BOOTSTRAP["y"][0] <= move["ee.y"] <= BOOTSTRAP["y"][1]
        assert BOOTSTRAP["z"][0] <= move["ee.z"] - TOOL_LENGTH_MM <= BOOTSTRAP["z"][1]
    assert moves[-1]["ee.z"] == pytest.approx(250.0 + TOOL_LENGTH_MM)  # back at the start

    (cell,) = runtime.list()["workcells"]
    assert set(cell["calibration"]) == {"tcp", "workspace"} and cell["calibration_stale"] == {}
    assert _skills(runtime)[SEQUENCE]["available"]
    lift = await _run(runtime, SEQUENCE, move_by(50))
    assert lift.outcome is not None and lift.outcome.ok, lift.outcome


# ── calib.tcp ────────────────────────────────────────────────────────────────


async def test_pivot_tcp_is_fitted_from_flange_poses(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    runtime = _runtime(tmp_path, registry)
    plan = await runtime.plan(
        WORKCELL_ID, "calib.tcp", {"method": "pivot", "samples": _pivot_samples()}
    )
    assert isinstance(plan, SkillPlan), plan
    tcp = plan.basis["tcp"]
    assert (tcp["x"], tcp["y"], tcp["z"]) == pytest.approx(TIP, abs=1e-3)
    assert plan.basis["fit"]["pivot"] == pytest.approx(PIVOT, abs=1e-3)
    assert "pivot over 5 poses" in plan.summary and "not calibrated" in plan.summary
    run = await runtime.execute(plan.digest)
    assert isinstance(run, SkillRun)
    finished = await runtime.wait(run.run_id, timeout_s=5.0)
    assert finished is not None and finished.outcome is not None and finished.outcome.ok
    record = registry.calibration_store.latest(WORKCELL_ID, "tcp.offset")
    assert record is not None and record.pose["z"] == pytest.approx(TOOL_LENGTH_MM, abs=1e-3)
    assert record.parameters["method"] == "pivot"


async def test_pivot_samples_that_disagree_fail_the_check_before_anything_runs(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path, _registry(tmp_path))
    samples = _pivot_samples()
    samples[2][0] += 10.0  # the tip slipped between two samples
    outcome = await _refusal(runtime, "calib.tcp", {"method": "pivot", "samples": samples})
    assert outcome.failure_code == FAILURE_CHECK_FAILED
    assert outcome.side_effect_state == SIDE_EFFECT_NONE
    assert outcome.result["residual_mm"] > 1.0

    tolerant = _runtime(
        tmp_path / "tolerant",
        _registry(tmp_path / "tolerant"),
        calibration={"max_pivot_residual_mm": 10.0},
    )
    accepted = await tolerant.plan(
        WORKCELL_ID, "calib.tcp", {"method": "pivot", "samples": samples}
    )
    assert isinstance(accepted, SkillPlan), accepted


async def test_pivot_needs_spread_orientations(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, _registry(tmp_path))
    samples = [[300, 20, 170, rx, ry, 0] for rx, ry in ((180, 0), (175, 5), (178, -5), (180, 8))]
    outcome = await _refusal(runtime, "calib.tcp", {"method": "pivot", "samples": samples})
    assert outcome.failure_code == FAILURE_CHECK_FAILED
    assert "orientation spread too small" in outcome.result["problems"]


async def test_declared_tcp_needs_a_plausible_offset(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, _registry(tmp_path))
    missing = await _refusal(runtime, "calib.tcp", {})
    assert missing.failure_code == FAILURE_INVALID_ARGS
    metres = await _refusal(runtime, "calib.tcp", {"offset": [0, 0, 1200]})
    assert metres.failure_code == FAILURE_INVALID_ARGS and "millimetres" in metres.message


# ── Staleness ────────────────────────────────────────────────────────────────


async def test_recalibrating_the_tcp_makes_the_dependent_workspace_stale(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.calibrate("tcp.offset", pose={"x": 0, "y": 0, "z": TOOL_LENGTH_MM}, recorded_at=1000.0)
    registry.calibrate(
        "workspace.safe_box",
        parameters={**SAFE_BOX, "depends_on": {"tcp": 1000.0}},
        recorded_at=1001.0,
    )
    runtime = _runtime(tmp_path, registry)
    assert _skills(runtime)[SEQUENCE]["available"]

    registry.calibrate("tcp.offset", pose={"x": 0, "y": 0, "z": 130}, recorded_at=2000.0)
    (cell,) = runtime.list()["workcells"]
    assert "workspace" not in cell["calibration"]
    assert "recalibrated" in cell["calibration_stale"]["workspace"]
    skills = _skills(runtime)
    assert not skills[SEQUENCE]["available"]
    assert "workspace calibration is stale" in skills[SEQUENCE]["reason"]
    assert skills["calib.workspace"]["available"]


# ── calib.workspace ──────────────────────────────────────────────────────────


async def test_a_workspace_beyond_the_bootstrap_box_cannot_be_verified(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.calibrate("tcp.offset", pose={"x": 0, "y": 0, "z": TOOL_LENGTH_MM})
    runtime = _runtime(tmp_path, registry)
    wide = {**BOX_ARGS, "x": [100, 390], "z_min": 130}
    refused = await _refusal(runtime, "calib.workspace", wide)
    assert refused.failure_code == FAILURE_OUT_OF_WORKSPACE
    assert "bootstrap_box" in refused.message and "verify=false" in refused.message

    run = await _run(runtime, "calib.workspace", {**wide, "verify": False})
    assert run.outcome is not None and run.outcome.ok, run.outcome
    assert run.outcome.result["poses_visited"] == 0 and not run.outcome.result["verified"]
    assert registry.sim.batches == []
    record = registry.calibration_store.latest(WORKCELL_ID, "workspace.safe_box")
    assert record is not None and record.parameters["x"] == [100, 390]
    assert record.parameters["depends_on"]["tcp"] == pytest.approx(
        runtime.list()["workcells"][0]["calibration"]["tcp"]
    )


async def test_workspace_verification_ignores_the_measured_box_it_is_about_to_replace(
    tmp_path: Path,
) -> None:
    """The skill that measures the workspace plans in the bootstrap box, always.

    A wide measured box would let every corner through, so a plan accepted here
    would prove the skill had trusted the geometry it is about to overwrite.
    """
    registry = _registry(tmp_path)
    registry.calibrate("tcp.offset", pose={"x": 0, "y": 0, "z": TOOL_LENGTH_MM})
    registry.calibrate("workspace.safe_box", parameters={**SAFE_BOX, "x_min": 100.0})
    runtime = _runtime(tmp_path, registry)
    wide = {**BOX_ARGS, "x": [120, 390], "z_min": 130}
    refused = await _refusal(runtime, "calib.workspace", wide)
    assert refused.failure_code == FAILURE_OUT_OF_WORKSPACE
    assert "bootstrap_box" in refused.message
    assert registry.sim.batches == []


async def test_the_floor_follows_the_highest_corner_of_a_tilted_table(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.calibrate("tcp.offset", pose={"x": 0, "y": 0, "z": TOOL_LENGTH_MM})
    runtime = _runtime(tmp_path, registry)
    plan = await runtime.plan(
        WORKCELL_ID,
        "calib.workspace",
        {**BOX_ARGS, "table_points": TILTED_TABLE, "verify": False},
    )
    assert isinstance(plan, SkillPlan), plan
    assert plan.basis["box"]["z"][0] == pytest.approx(109.3 + 30.0)
    assert "tilted" in plan.summary and plan.basis["floor"]["method"] == "table_points"

    bumpy = [list(point) for point in TILTED_TABLE]
    bumpy[0][2] += 20.0
    refused = await _refusal(runtime, "calib.workspace", {**BOX_ARGS, "table_points": bumpy})
    assert refused.failure_code == FAILURE_CHECK_FAILED
    assert refused.result["residual_mm"] > 3.0


@pytest.mark.parametrize(
    "floor", [{}, {"z_min": 130, "table_z": 100}], ids=["no-floor", "two-floors"]
)
async def test_the_workspace_takes_exactly_one_floor(
    tmp_path: Path, floor: Mapping[str, Any]
) -> None:
    registry = _registry(tmp_path)
    registry.calibrate("tcp.offset", pose={"x": 0, "y": 0, "z": TOOL_LENGTH_MM})
    runtime = _runtime(tmp_path, registry)
    refused = await _refusal(runtime, "calib.workspace", {**BOX_ARGS, **floor})
    assert refused.failure_code == FAILURE_INVALID_ARGS


# ── calib.bringup ────────────────────────────────────────────────────────────


async def test_bringup_cycles_the_gripper_and_records_against_the_arm(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    runtime = _runtime(tmp_path, registry)
    run = await _run(runtime, "calib.bringup", {"expect_zero": True})
    assert run.outcome is not None and run.outcome.ok, run.outcome
    assert _gripper_targets(registry.sim) == [70.0, 0.0, 60.0]
    assert _pose_batches(registry.sim) == []  # the arm itself never moves
    record = registry.calibration_store.latest(ARM_ID, BRINGUP_PROCEDURE)
    assert record is not None and record.parameters["zero_checked"] is True
    assert record.parameters["gripper_open_mm"] == pytest.approx(70.0)


async def test_bringup_reports_blocked_jaws_as_a_committed_check_failure(tmp_path: Path) -> None:
    registry = _registry(tmp_path, object_width_mm=30.0)
    runtime = _runtime(tmp_path, registry)
    run = await _run(runtime, "calib.bringup", {})
    outcome = run.outcome
    assert outcome is not None and outcome.failure_code == FAILURE_CHECK_FAILED
    assert outcome.side_effect_state == SIDE_EFFECT_COMMITTED
    assert "clear the jaws" in outcome.message
    assert _gripper_targets(registry.sim)[-1] == 60.0  # the width is restored
    assert registry.calibration_store.latest(ARM_ID, BRINGUP_PROCEDURE) is None


async def test_bringup_refuses_an_arm_that_does_not_read_zero(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.sim.joints["joint.j2.position"] = 3.0
    runtime = _runtime(tmp_path, registry)
    refused = await _refusal(runtime, "calib.bringup", {"expect_zero": True})
    assert refused.failure_code == FAILURE_CHECK_FAILED
    assert refused.side_effect_state == SIDE_EFFECT_NONE
    assert any("joint.j2.position" in problem for problem in refused.result["problems"])
    assert registry.sim.batches == []

    tolerant = _runtime(
        tmp_path / "tolerant",
        _registry(tmp_path / "tolerant", joints=tuple(registry.sim.joints.values())),
        calibration={"max_zero_error_deg": 4.0},
    )
    accepted = await tolerant.plan(
        WORKCELL_ID,
        "calib.bringup",
        {"expect_zero": True, "gripper_travel": False},
    )
    assert isinstance(accepted, SkillPlan), accepted
    assert accepted.basis["max_zero_error_deg"] == 4.0


# ── arm.state and the host ───────────────────────────────────────────────────


async def test_arm_state_reads_without_moving(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.calibrate_cell()
    runtime = _runtime(tmp_path, registry)
    outcome = await runtime.plan(WORKCELL_ID, "arm.state", {})
    assert isinstance(outcome, SkillOutcome) and outcome.ok, outcome
    assert outcome.side_effect_state == SIDE_EFFECT_NONE
    assert outcome.result["tcp_calibrated"] is True
    assert outcome.result["tcp"]["z"] == pytest.approx(250.0)
    assert outcome.result["workspace"]["z"] == [SAFE_BOX["z_min"], SAFE_BOX["z_max"]]
    assert registry.sim.batches == [] and runtime.status() == ()


def test_check_failed_tells_the_model_to_fix_and_rerun_calibration() -> None:
    step = next_step(FAILURE_CHECK_FAILED, SIDE_EFFECT_NONE)
    assert "calib." in step and "result.problems" in step
