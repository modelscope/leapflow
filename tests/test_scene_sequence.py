# Copyright (c) Alibaba, Inc. and its affiliates.
"""Pick and place composed by S2: perceive.locate -> grasp.propose -> motion.sequence.

The arm, camera, locator and grasp model are simulated; the runtime, gate,
scene memory, planners and skills are the production ones. The node scripts no
manipulation: it resolves the handles S2 names, refuses targets it cannot
reach before approval, refuses a stale plan before moving, and reports a
failed guard with the step it stopped at.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import pytest

from leapflow.hardware.model_contract import (
    TASK_GRASP_PROPOSAL,
    GraspProposalResult,
    ModelInfo,
    ProposedGrasp,
)
from leapflow.hardware.skill_results import next_step
from leapflow.hardware.skills.builtin.motion import skills as motion_skills
from leapflow.hardware.skills.builtin.scene import skills as scene_skills
from leapflow.hardware.skills.builtin.sequence import skills as sequence_skills
from leapflow.hardware.skills.geometry import Pose, Transform
from leapflow.hardware.skills.protocol import (
    FAILURE_GUARD_FAILED,
    FAILURE_INVALID_ARGS,
    FAILURE_OUT_OF_WORKSPACE,
    FAILURE_PLAN_STALE,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_TARGET_NOT_FOUND,
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_PARTIAL,
    STATUS_FAILED,
    STATUS_HALTED,
    STATUS_SUCCEEDED,
    SkillOutcome,
    SkillPlan,
)
from leapflow.hardware.skills.runtime import SkillRun, SkillRuntime, SkillRuntimeSettings
from tests._harness.skill_stubs import (
    CAMERA_ID,
    CAMERA_POSE,
    PERCEPTION_WORKCELL_YAML,
    SEQUENCE,
    SIM_INTRINSICS,
    TABLE_Z,
    TOOL_LENGTH_MM,
    WORKCELL_ID,
    SimArmTransport,
    SimBox,
    SimRgbdCamera,
    SkillNodeRegistry,
    perception_registry,
    write_workcell,
)

CUP = SimBox("cup", center=(300.0, 20.0), length=60.0, width=30.0, height=50.0)
CUP_WIDTH = 30.0
GRASP_ID = "grasp_server"

PICK = [
    {"op": "gripper", "grip": "open", "ref": "grasp:g1"},
    {"op": "move", "ref": "grasp:g1.pregrasp"},
    {"op": "move_linear", "ref": "grasp:g1.grasp", "expect": "reached"},
    {"op": "gripper", "grip": "close", "ref": "grasp:g1", "expect": "held"},
    {"op": "move_linear", "ref": "grasp:g1.retreat", "expect": "held"},
]
PLACE = [
    {"op": "move", "pose": {"x": 220.0, "y": -90.0, "z": 150.0}, "expect": "held"},
    {"op": "move_linear", "relative": {"dx": 0, "dy": 0, "dz": -30, "frame": "base"}},
    {"op": "gripper", "grip": "open", "expect": "released"},
    {"op": "move_linear", "relative": {"dx": 0, "dy": 0, "dz": 40, "frame": "base"}},
]


def _cell(
    tmp_path: Path,
    *boxes: SimBox,
    arm: SimArmTransport | None = None,
    workcell: Mapping[str, Any] = PERCEPTION_WORKCELL_YAML,
    devices: Mapping[str, Any] | None = None,
) -> SkillRuntime:
    registry = perception_registry(
        tmp_path / "instrument.duckdb",
        boxes,
        arm=arm or SimArmTransport(object_width_mm=CUP_WIDTH),
    )
    registry.devices.update(devices or {})
    registry.last_command = lambda device_id, channel_id: None  # type: ignore[method-assign]
    registry.calibrate_perception()
    directory = tmp_path / "workcells"
    write_workcell(directory, workcell)
    return SkillRuntime(
        registry,
        settings=SkillRuntimeSettings(workcells_dir=directory),
        skills=[*motion_skills(), *sequence_skills(), *scene_skills()],
        poll_s=0.005,
    )


def _registry(runtime: SkillRuntime) -> SkillNodeRegistry:
    return runtime._registry  # type: ignore[attr-defined] - the test owns the sim


def _sim(runtime: SkillRuntime) -> SimArmTransport:
    return _registry(runtime).sim


def _camera(runtime: SkillRuntime) -> SimRgbdCamera:
    return _registry(runtime).devices[CAMERA_ID]


async def _read(runtime: SkillRuntime, skill: str, args: Mapping[str, Any]) -> SkillOutcome:
    outcome = await runtime.plan(WORKCELL_ID, skill, args)
    assert isinstance(outcome, SkillOutcome) and outcome.ok, outcome
    assert outcome.side_effect_state == SIDE_EFFECT_NONE
    return outcome


async def _plan(runtime: SkillRuntime, steps: list[dict[str, Any]]) -> SkillPlan:
    plan = await runtime.plan(WORKCELL_ID, SEQUENCE, {"steps": steps})
    assert isinstance(plan, SkillPlan), plan
    return plan


async def _execute(runtime: SkillRuntime, plan: SkillPlan) -> SkillOutcome:
    run = await runtime.execute(plan.digest)
    assert isinstance(run, SkillRun), run
    finished = await runtime.wait(run.run_id, timeout_s=10.0)
    assert finished is not None and finished.outcome is not None
    return finished.outcome


async def _refusal(runtime: SkillRuntime, steps: list[dict[str, Any]]) -> SkillOutcome:
    outcome = await runtime.plan(WORKCELL_ID, SEQUENCE, {"steps": steps})
    assert isinstance(outcome, SkillOutcome) and not outcome.ok, outcome
    assert outcome.side_effect_state == SIDE_EFFECT_NONE
    return outcome


async def _located_and_proposed(runtime: SkillRuntime) -> SkillOutcome:
    located = await _read(runtime, "perceive.locate", {"queries": ["cup"]})
    assert [obj["object_id"] for obj in located.result["objects"]] == ["o1"]
    return await _read(runtime, "grasp.propose", {"object": "o1"})


def _gripper_targets(sim: SimArmTransport) -> list[float]:
    return [value for batch in sim.batches for cid, value in batch if cid == "gripper.position"]


def _moves(sim: SimArmTransport) -> list[dict[str, Any]]:
    return [dict(batch) for batch in sim.batches if any(cid == "ee.z" for cid, _ in batch)]


# ── Pick and place ───────────────────────────────────────────────────────────


async def test_a_composed_pick_resolves_grasp_handles_and_records_the_hold(
    tmp_path: Path,
) -> None:
    runtime = _cell(tmp_path, CUP)
    proposed = await _located_and_proposed(runtime)
    g1 = proposed.result["grasps"][0]
    assert g1["grasp_id"] == "g1" and g1["object_id"] == "o1"

    plan = await _plan(runtime, PICK)
    pregrasp, grasp, retreat = plan.targets
    assert (grasp.x, grasp.y) == pytest.approx((300.0, 20.0), abs=4.0)
    assert grasp.z == pytest.approx(TABLE_Z + 25.0, abs=3.0)
    assert pregrasp.z == retreat.z == pytest.approx(grasp.z + 80.0, abs=0.01)
    assert f"open the gripper to {g1['open_mm']:.0f} mm on g1" in plan.summary
    assert set(plan.basis["objects"]) == {"o1"} and set(plan.basis["grasps"]) == {"g1"}
    assert _sim(runtime).batches == []  # planning never moves

    outcome = await _execute(runtime, plan)
    assert outcome.status == STATUS_SUCCEEDED, outcome
    assert outcome.side_effect_state == SIDE_EFFECT_COMMITTED
    held = outcome.result["held"]
    assert held["object"]["object_id"] == "o1" and held["grasp_id"] == "g1"
    assert held["width_mm"] == pytest.approx(CUP_WIDTH, abs=1.0)
    assert held["hold_offset_mm"] == pytest.approx(25.0, abs=4.0)  # fingertips below the top
    sim = _sim(runtime)
    assert _gripper_targets(sim) == [pytest.approx(g1["open_mm"], abs=0.1), 0.0]
    moves = _moves(sim)
    assert [move["ee.z"] for move in moves] == pytest.approx(
        [target.z + TOOL_LENGTH_MM for target in plan.targets], abs=0.01
    )  # flange = TCP + tool length, pointing down
    assert [move.get("arm.pose_path") for move in moves][1:] == ["linear", "linear"]

    located = await _read(runtime, "perceive.locate", {"queries": ["cup"]})
    assert located.result["held"]["object"]["object_id"] == "o1"


async def test_a_composed_place_releases_and_forgets_the_hold(tmp_path: Path) -> None:
    runtime = _cell(tmp_path, CUP)
    await _located_and_proposed(runtime)
    await _execute(runtime, await _plan(runtime, PICK))

    plan = await _plan(runtime, PLACE)
    assert plan.basis["held"] == "o1" and plan.basis["objects"] == {}
    outcome = await _execute(runtime, plan)
    assert outcome.status == STATUS_SUCCEEDED, outcome
    assert outcome.result["held"] is None
    assert _gripper_targets(_sim(runtime))[-1] == pytest.approx(70.0)  # the full stroke
    located = await _read(runtime, "perceive.locate", {"queries": ["cup"]})
    assert located.result["held"] is None


async def test_release_guard_preserves_hold_until_width_confirms_release(tmp_path: Path) -> None:
    runtime = _cell(tmp_path, CUP)
    await _located_and_proposed(runtime)
    await _execute(runtime, await _plan(runtime, PICK))
    step = {
        "op": "gripper",
        "width_mm": CUP_WIDTH + 4.0,
        "expect": "released",
        "on_fail": "hold",
    }

    outcome = await _execute(runtime, await _plan(runtime, [step]))

    assert outcome.failure_code == FAILURE_GUARD_FAILED
    assert outcome.status == STATUS_FAILED
    assert "release requires more than 35.0" in outcome.message
    located = await _read(runtime, "perceive.locate", {"queries": ["cup"]})
    assert located.result["held"]["object"]["object_id"] == "o1"


async def test_a_move_may_name_a_path_the_arm_declares(tmp_path: Path) -> None:
    runtime = _cell(tmp_path, CUP)
    plan = await _plan(
        runtime, [{"op": "move", "pose": {"x": 220.0, "y": -90.0, "z": 150.0}, "path": "point"}]
    )
    assert "move point" in plan.summary

    outcome = await _execute(runtime, plan)

    assert outcome.status == STATUS_SUCCEEDED, outcome
    (move,) = _moves(_sim(runtime))
    assert move["arm.pose_path"] == "point", "a named path reaches the arm's path channel"


# ── Refusals before approval ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("steps", "code", "needle"),
    [
        ([{"op": "move", "pose": {"x": 300, "y": 0, "z": 500}}], FAILURE_OUT_OF_WORKSPACE, ""),
        ([
            {"op": "move", "pose": {"x": 220, "y": -90, "z": 150}, "path": "arc"},
        ], FAILURE_INVALID_ARGS, "not declared by this arm"),
        ([
            {"op": "move_linear", "pose": {"x": 220, "y": -90, "z": 150}, "path": "point"},
        ], FAILURE_INVALID_ARGS, "do(es) not apply to op move_linear"),
        ([{"op": "move", "ref": "grasp:g7.grasp"}], FAILURE_TARGET_NOT_FOUND, "g7"),
        ([{"op": "move", "ref": "object:cup"}], FAILURE_TARGET_NOT_FOUND, "cup"),
        ([{"op": "move", "ref": "grasp:g1.top"}], FAILURE_INVALID_ARGS, "must be object:"),
        ([{"op": "gripper", "grip": "open", "width_mm": 20}], FAILURE_INVALID_ARGS, "exactly one"),
        ([{"op": "gripper", "width_mm": 90}], FAILURE_INVALID_ARGS, "exceeds the gripper"),
        ([{"op": "wait", "seconds": 30}], FAILURE_INVALID_ARGS, "exceeds the workcell"),
        ([{"op": "wait", "seconds": 0}] * 17, FAILURE_INVALID_ARGS, "split the program"),
        ([{"op": "wait", "seconds": 0, "expect": "released"}], FAILURE_INVALID_ARGS, "expect"),
    ],
)
async def test_plan_refuses_what_it_cannot_resolve_or_reach(
    tmp_path: Path, steps: list[dict[str, Any]], code: str, needle: str
) -> None:
    runtime = _cell(tmp_path, CUP)
    await _located_and_proposed(runtime)
    refused = await _refusal(runtime, steps)
    assert refused.failure_code == code and needle in refused.message, refused
    assert _sim(runtime).batches == []


# ── Freshness at execute ─────────────────────────────────────────────────────


async def test_a_referenced_object_that_drifted_makes_the_plan_stale(tmp_path: Path) -> None:
    runtime = _cell(tmp_path, CUP)
    await _located_and_proposed(runtime)
    plan = await _plan(runtime, PICK)
    camera = _camera(runtime)
    camera.boxes[0] = SimBox("cup", center=(330.0, 20.0), length=60.0, width=30.0, height=50.0)
    outcome = await _execute(runtime, plan)
    assert outcome.failure_code == FAILURE_PLAN_STALE and "moved" in outcome.message
    assert outcome.side_effect_state == SIDE_EFFECT_NONE
    assert _sim(runtime).batches == []


async def test_a_tool_that_moved_since_planning_makes_the_plan_stale(tmp_path: Path) -> None:
    runtime = _cell(tmp_path, CUP)
    plan = await _plan(runtime, PLACE[1:2])
    _sim(runtime).pose[2] += 10.0
    outcome = await _execute(runtime, plan)
    assert outcome.failure_code == FAILURE_PLAN_STALE and "tool moved" in outcome.message
    assert _sim(runtime).batches == []


async def test_a_dropped_object_makes_a_plan_that_assumed_the_hold_stale(tmp_path: Path) -> None:
    runtime = _cell(tmp_path, CUP)
    await _located_and_proposed(runtime)
    await _execute(runtime, await _plan(runtime, PICK))
    plan = await _plan(runtime, PLACE)
    sim = _sim(runtime)
    sim.object_width_mm = None  # it slipped out; the jaws keep closing
    moves = len(sim.batches)
    outcome = await _execute(runtime, plan)
    assert outcome.failure_code == FAILURE_PLAN_STALE and "holds changed" in outcome.message
    assert len(sim.batches) == moves


# ── Guards ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("on_fail", "status"), [("stop", STATUS_HALTED), ("hold", STATUS_FAILED)])
async def test_closing_on_nothing_fails_the_held_guard_and_names_the_step(
    tmp_path: Path, on_fail: str, status: str
) -> None:
    runtime = _cell(tmp_path, CUP, arm=SimArmTransport())  # the cup is never between the jaws
    await _located_and_proposed(runtime)
    steps = [
        {"op": "move", "ref": "grasp:g1.pregrasp"},
        {
            "op": "gripper",
            "grip": "close",
            "ref": "object:o1",
            "expect": "held",
            "on_fail": on_fail,
        },
        {"op": "move", "ref": "grasp:g1.retreat"},
    ]
    outcome = await _execute(runtime, await _plan(runtime, steps))
    assert outcome.failure_code == FAILURE_GUARD_FAILED and outcome.status == status
    assert outcome.side_effect_state == SIDE_EFFECT_PARTIAL
    assert outcome.result["step"] == 2 and outcome.result["completed"] == 1
    assert outcome.result["guard"] == "held"
    assert "nothing between them" in outcome.message
    assert len(_moves(_sim(runtime))) == 1  # the retreat never ran
    located = await _read(runtime, "perceive.locate", {"queries": ["cup"]})
    assert located.result["held"] is None

    advice = next_step(FAILURE_GUARD_FAILED, SIDE_EFFECT_PARTIAL)
    assert "result.step" in advice and "The arm may have moved" not in advice


# ── The learned grasp planner ────────────────────────────────────────────────


class _SimGraspModel:
    """A ``grasp_proposal`` service answering around the object's camera-frame centre."""

    def __init__(self) -> None:
        self.requests: list[Mapping[str, Any]] = []

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(task=TASK_GRASP_PROPOSAL, name="sim-grasp", version="2")

    async def refresh(self) -> ModelInfo:
        return self.model_info

    async def halt(self) -> None:
        return None

    async def infer(self, request: Mapping[str, Any]) -> GraspProposalResult:
        self.requests.append(request)
        x, y, z = request["object"]["center"]

        def at(width: float, score: float, rz: float = 0.0, **kw: Any) -> ProposedGrasp:
            return ProposedGrasp((x, y, z, 0.0, 0.0, rz), width, score, **kw)

        grasps = (
            at(30.0, 0.6),
            at(80.0, 0.95),  # wider than the stroke
            at(30.0, 0.1),  # below the declared floor
            at(60.0, 0.8, rz=90.0, approach_mm=50.0),
        )
        return GraspProposalResult(grasps, 20.0, self.model_info.identity)


async def test_a_model_planner_maps_filters_and_ranks_proposed_grasps(tmp_path: Path) -> None:
    model = _SimGraspModel()
    workcell = {
        **PERCEPTION_WORKCELL_YAML,
        "models": {**PERCEPTION_WORKCELL_YAML["models"], "grasp": GRASP_ID},
        "grasp": {"planner": "model", "min_score": 0.2},
    }
    runtime = _cell(tmp_path, CUP, workcell=workcell, devices={GRASP_ID: model})
    proposed = await _located_and_proposed(runtime)
    assert proposed.result["planner"] == "model"
    first, second = proposed.result["grasps"]
    assert first["planner"] == second["planner"] == "model:sim-grasp"
    assert (first["grasp_id"], first["score"], second["score"]) == ("g1", 0.8, 0.6)
    assert first["approach_mm"] == 50.0 and second["approach_mm"] == 80.0
    assert first["open_mm"] == 70.0 and second["open_mm"] == 45.0  # capped at the stroke
    pose = second["grasp"]
    # The camera looks straight down from 600 mm: its frame maps back onto the cup.
    assert (pose["x"], pose["y"], pose["z"]) == pytest.approx((300.0, 20.0, TABLE_Z + 25), abs=4)
    assert abs(pose["rx"]) == pytest.approx(180.0, abs=1e-6)

    (request,) = model.requests
    depth = 600.0 - TABLE_Z - 25.0  # the camera-frame z of the cup's centre
    assert request["object"]["center"][2] == pytest.approx(depth, abs=4.0)
    assert request["object"]["pixel"] == pytest.approx([100.0, 75.0 - 250.0 * 20 / depth], abs=2)
    assert request["gripper"] == {"max_width_mm": 70.0} and request["top_k"] == 5
    assert isinstance(request["image"], str) and isinstance(request["depth"]["data"], str)

    plan = await _plan(runtime, PICK[:2])  # the learned grasp feeds the same program
    assert plan.targets[0].z == pytest.approx(first["grasp"]["z"] + 50.0, abs=0.01)


# ── A lifted object needs no locator ──────────────────────────────────────────

NO_LOCATOR_YAML: dict[str, Any] = {
    **{key: value for key, value in PERCEPTION_WORKCELL_YAML.items() if key != "models"},
    "share_frames": True,
}


def _top_box(box: SimBox, z: float) -> list[float]:
    """The pixel box the simulated camera sees around *box*'s top face at *z*."""
    to_camera = Transform.from_pose(Pose(**CAMERA_POSE)).inverse()
    yaw = math.radians(box.yaw_deg)
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            dx, dy = sx * box.length / 2.0, sy * box.width / 2.0
            x = box.center[0] + dx * math.cos(yaw) - dy * math.sin(yaw)
            y = box.center[1] + dx * math.sin(yaw) + dy * math.cos(yaw)
            corners.append(SIM_INTRINSICS.project(to_camera.apply((x, y, z))))
    return [
        min(u for u, _ in corners),
        min(v for _, v in corners),
        max(u for u, _ in corners),
        max(v for _, v in corners) + 1.0,
    ]


async def test_grasp_propose_plans_for_a_lifted_object_without_a_locator(
    tmp_path: Path,
) -> None:
    """S2's own grounding must reach the planner: locate needs a model, lift does not."""
    runtime = _cell(tmp_path, CUP, workcell=NO_LOCATOR_YAML)
    no_locator = await runtime.plan(WORKCELL_ID, "perceive.locate", {"queries": ["cup"]})
    assert isinstance(no_locator, SkillOutcome) and not no_locator.ok
    assert no_locator.failure_code == FAILURE_SKILL_UNAVAILABLE

    shot = await runtime.plan(WORKCELL_ID, "perceive.snapshot", {}, share_frames=True)
    assert isinstance(shot, SkillOutcome) and shot.ok, shot
    lifted = await runtime.plan(
        WORKCELL_ID,
        "perceive.lift",
        {
            "frame": shot.result["frame"],
            "targets": [{"label": "cup", "box": _top_box(CUP, TABLE_Z + CUP.height)}],
        },
        share_frames=True,
    )
    assert isinstance(lifted, SkillOutcome) and lifted.ok, lifted
    (cup,) = lifted.result["objects"]

    proposed = await _read(runtime, "grasp.propose", {"object": cup["object_id"]})
    grasp, alongside = proposed.result["grasps"]  # the narrow side first, then the long one
    assert grasp["object_id"] == cup["object_id"] and grasp["planner"] == "top_down"
    assert grasp["width_mm"] == pytest.approx(CUP_WIDTH, abs=4.0)
    assert grasp["open_mm"] == pytest.approx(min(70.0, CUP_WIDTH + 15.0), abs=4.0)
    assert grasp["score"] == 1.0 and alongside["score"] == 0.5
    assert grasp["grasp"]["z"] == pytest.approx(TABLE_Z + 25.0, abs=3.0)
