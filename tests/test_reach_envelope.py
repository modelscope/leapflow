# Copyright (c) Alibaba, Inc. and its affiliates.
"""Reach guidance and continuity use production seams without moving a device."""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.hardware.skill_results import next_step, workcell_view
from leapflow.hardware.skills.geometry import Pose, Transform, WorkspaceBox
from leapflow.hardware.skills.kinematics import DhChain
from leapflow.hardware.skills.protocol import (
    FAILURE_GRASP_INFEASIBLE,
    FAILURE_INVALID_ARGS,
    FAILURE_TIMEOUT,
    FAILURE_UNREACHABLE,
    SIDE_EFFECT_NONE,
    SkillError,
    SkillOutcome,
)
from leapflow.hardware.skills.reach import LineFollower, ReachSettings, ReachTracker, joint_bounds
from leapflow.hardware.skills.reach_envelope import (
    ReachBand,
    ReachEnvelope,
    build_envelope,
    envelope_for,
)
from leapflow.hardware.skills.tuning import DEFAULT_LIMITS, REACH_ENVELOPE_MAX_TIMEOUT_S, tuned
from tests._harness.skill_stubs import WORKCELL_ID, piper_context
from tests.test_scene_sequence import CUP, _cell, _kinematic_cell, _read, _refusal, _sim


def _probe(monkeypatch: pytest.MonkeyPatch, *, branch: bool = False) -> ReachTracker:
    def solve(self: LineFollower, pose: Pose, seed: Any) -> tuple[float, ...] | None:
        if branch:
            return (0.0 if pose.rz < 0 else 100.0,)
        return (3.0 * pose.x,)

    monkeypatch.setattr(LineFollower, "solve", solve)
    return ReachTracker(
        tcp=Pose(0, 0, 0, 180, 0, -20),
        chain=SimpleNamespace(dof=1),
        names=("j1",),
        joints={"j1": 0},
        bounds=((-180, 180),),
        settings=ReachSettings(10, 10, 1),
    )


def test_metrics_normalize_declared_limits_without_changing_solver_safety_margin() -> None:
    tracker = ReachTracker(tcp=Pose(0, 0, 0), chain=SimpleNamespace(dof=1), names=("j1",),
                           joints={"j1": 10}, bounds=((1, 99),), metric_bounds=((0, 100),),
                           settings=ReachSettings(10, 10, 1))
    tracker.observe_joints((30,))
    assert tracker.metrics() == {"joint_margin": 0.1, "joint_travel": 0.2}
    assert tracker._line.bounds == ((1, 99),)


def test_fast_continuous_joints_pass_subdivision(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = _probe(monkeypatch)
    tracker.move(Pose(10, 0, 0, 180, 0, -20), linear=True, label="fast")
    assert tracker.verified


def test_a_real_branch_switch_survives_subdivision_and_reports_degrees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _probe(monkeypatch, branch=True)
    with pytest.raises(SkillError) as refusal:
        tracker.move(Pose(0, 0, 0, 180, 0, 20), linear=True, label="rotation")
    reach = refusal.value.result["reach"]
    assert reach["cause"] == "configuration_switch"
    assert reach["requested_deg"] == 40
    assert 0 < reach["reachable_deg"] < 40
    assert "deg" in refusal.value.message and "0 of 0 mm" not in refusal.value.message


def test_unknown_radial_samples_are_not_interpolated_into_a_guarantee() -> None:
    envelope = ReachEnvelope(
        (
            ReachBand(100, 10, 150),
            ReachBand(120, 20, 140),
            ReachBand(140, None, None),
        ),
        5,
        -45,
        20,
    )
    assert envelope.z_range(110) == (20, 140)
    assert envelope.z_range(130) is None
    assert envelope.z_range(200) is None
    assert envelope.to_dict()["transit_z_mm"] is None
    assert envelope.to_dict()["advisory"] is True


def test_piper_top_down_envelope_is_cached_and_invalidates_with_geometry() -> None:
    arm = piper_context()
    chain = DhChain.from_mapping(arm.kinematics.dh)
    names = tuple(f"joint.j{i}.position" for i in range(1, 7))
    bounds = joint_bounds(arm, names, 1)
    seed = tuple((lo + hi) / 2 for lo, hi in bounds)
    box = WorkspaceBox(110, 420, -170, 220, 11.7, 330)
    tcp = Transform.from_pose(Pose(0, 0, 126.5))
    args = (chain, bounds, tcp, box, ReachSettings(10, 10, 1), 20, seed, 8)
    envelope = build_envelope(*args)
    assert envelope is not None
    low, high = envelope.z_range(350)
    assert low == pytest.approx(11.7)
    assert 150 < high < 185 < box.z_max
    assert build_envelope(*args) is envelope
    longer = build_envelope(chain, bounds, Transform.from_pose(Pose(0, 0, 146.5)), *args[3:])
    assert longer is not None and longer is not envelope
    assert longer.z_range(350)[1] < high
    assert envelope.to_dict()["transit_z_mm"] <= high


@pytest.mark.parametrize(
    "key", ["reach_continuity_rounds", "reach_refine_rounds", "reach_envelope_yaw_samples"]
)
@pytest.mark.parametrize("value", [0, 1.5, 17, math.inf, True])
def test_subdivision_and_sampling_counts_are_admission_validated(key: str, value: Any) -> None:
    with pytest.raises(ValueError):
        tuned({key: value}, "limits", DEFAULT_LIMITS)


async def test_arm_state_and_host_catalog_publish_the_same_guidance(tmp_path: Path) -> None:
    runtime = _kinematic_cell(tmp_path)
    state = await _read(runtime, "arm.state", {})
    listing = await runtime.list_async(WORKCELL_ID)
    cell = listing["workcells"][0]
    view = workcell_view("node", cell)
    assert view["reach"] == cell["reach"]
    assert state.result["reach"]["bands"] == cell["reach"]["bands"]
    assert state.result["reach"]["kinematics_verified"] is True
    assert cell["reach"]["ik_iteration_budget"] == runtime._workcells[WORKCELL_ID].limits["reach_envelope_ik_iterations"]
    azimuth = math.radians(cell["reach"]["azimuth_deg"])
    box = state.result["workspace"]
    for band in cell["reach"]["bands"]:
        radius = band["radius_mm"]
        assert box["x"][0] - 0.1 <= radius * math.cos(azimuth) <= box["x"][1] + 0.1
        assert box["y"][0] - 0.1 <= radius * math.sin(azimuth) <= box["y"][1] + 0.1
    assert state.result["workspace"]["z"][1] > max(
        b["z_max_mm"] for b in view["reach"]["bands"] if b["z_max_mm"]
    )
    assert _sim(runtime).batches == []


async def test_catalog_ik_does_not_block_the_node_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import leapflow.hardware.skills.runtime as runtime_module

    runtime = _cell(tmp_path)

    def slow_envelope(*args: Any) -> dict[str, Any]:
        time.sleep(0.1)
        return {"advisory": True}

    monkeypatch.setattr(runtime_module, "envelope_view", slow_envelope)
    listing = asyncio.create_task(runtime.list_async())
    await asyncio.sleep(0.01)
    assert not listing.done(), "the event loop must run while the envelope is computed"
    assert (await listing)["workcells"][0]["reach"]["advisory"]


@pytest.mark.parametrize("value", [0, 1.5, 51, math.inf, True])
def test_advisory_iteration_budget_is_bounded_without_relaxing_path_checks(value: Any) -> None:
    from leapflow.hardware.skills.adapt.spec import IkSettings

    with pytest.raises(ValueError):
        tuned({"reach_envelope_ik_iterations": value}, "limits", DEFAULT_LIMITS)
    assert LineFollower(SimpleNamespace(dof=1), ((-180, 180),), Transform(), ReachSettings(10, 10, 1))._ik_settings == IkSettings()


@pytest.mark.parametrize("value", [0, -1, 9.1, math.inf, True])
def test_advisory_deadline_is_positive_finite_and_fits_transport(value: Any) -> None:
    with pytest.raises(ValueError):
        tuned({"reach_envelope_timeout_s": value}, "limits", DEFAULT_LIMITS)


def test_a_slow_host_may_declare_a_longer_advisory_budget() -> None:
    # 2026-11-26: the reference cell needs 5.1 s for its declared sweep while the
    # default was 5.0, so reach answered "unknown" on about half the read-only
    # calls, and a workcell that declared 6 s was refused admission outright.
    # Patience is declarable up to the ceiling one request can carry; the slice's
    # sampling and every motion check stay unchanged.
    declared = tuned({"reach_envelope_timeout_s": REACH_ENVELOPE_MAX_TIMEOUT_S}, "limits",
                     DEFAULT_LIMITS)
    assert declared["reach_envelope_timeout_s"] == REACH_ENVELOPE_MAX_TIMEOUT_S
    assert DEFAULT_LIMITS["reach_envelope_timeout_s"] >= 5.1
    assert REACH_ENVELOPE_MAX_TIMEOUT_S < 10, "one LHP request ceiling"


def test_a_timed_out_slice_is_remembered_instead_of_recomputed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import leapflow.hardware.skills.reach_envelope as module

    arm = piper_context()
    names = tuple(f"joint.j{i}.position" for i in range(1, 7))
    bounds = joint_bounds(arm, names, 1)
    box = WorkspaceBox(110, 420, -170, 220, 11.7, 330)
    tcp = Transform.from_pose(Pose(0, 0, 126.5))
    limits = {**DEFAULT_LIMITS, "reach_envelope_timeout_s": 0.001}
    calls: list[Any] = []

    def timed_out(*args: Any, **kwargs: Any) -> None:
        calls.append(args)
        raise SkillError(FAILURE_TIMEOUT, "budget")

    module._timed_out_until.clear()
    monkeypatch.setattr(module, "build_envelope", timed_out)
    try:
        for _ in range(3):
            with pytest.raises(SkillError) as refusal:
                envelope_for(arm, names, bounds, tcp, box, limits, {})
            assert refusal.value.failure_code == FAILURE_TIMEOUT
        # A repeat read must not pay the whole budget again to learn the same thing.
        assert len(calls) == 1
    finally:
        module._timed_out_until.clear()


async def test_cold_advisory_deadline_leaves_catalog_available_and_caches_no_partial_result(tmp_path: Path) -> None:
    from dataclasses import replace

    runtime = _kinematic_cell(tmp_path)
    spec = runtime._workcells[WORKCELL_ID]
    runtime._workcells[WORKCELL_ID] = replace(spec, limits={**spec.limits, "reach_envelope_timeout_s": 0.001})
    before = build_envelope.cache_info()
    started = time.monotonic()
    listing = await runtime.list_async(WORKCELL_ID)
    reach = listing["workcells"][0]["reach"]
    assert time.monotonic() - started < 1
    assert not reach["available"] and reach["advisory"]
    assert "time budget" in reach["detail"] and "bands" not in reach
    assert listing["workcells"][0]["skills"]
    assert build_envelope.cache_info().currsize == before.currsize
    assert _sim(runtime).batches == []


async def test_sequence_collects_all_workspace_refusals_with_relative_semantics(
    tmp_path: Path,
) -> None:
    runtime = _cell(tmp_path)
    refused = await _refusal(
        runtime,
        [
            {"op": "move", "pose": {"x": 300, "y": 0, "z": 500}},
            {"op": "move_linear", "relative": {"dz": 100}},
            {"op": "move", "pose": {"x": 300, "y": 0, "z": 150}},
        ],
    )
    assert [problem["step"] for problem in refused.result["problems"]] == [1, 2]
    assert refused.result["problems"][1]["limit"]["value"] == 600
    assert refused.result["step"] == 1 and "step 2" in refused.message
    assert _sim(runtime).batches == []


async def test_sequence_collects_two_ik_limits_and_preserves_the_first_reach(
    tmp_path: Path,
) -> None:
    runtime = _kinematic_cell(tmp_path)
    refused = await _refusal(
        runtime,
        [
            {"op": "move_linear", "relative": {"dz": 200}},
            {"op": "move_linear", "relative": {"dz": 30}},
            {"op": "move", "pose": {"x": 300, "y": 0, "z": 100}},
        ],
    )
    assert refused.failure_code == FAILURE_UNREACHABLE
    assert [problem["step"] for problem in refused.result["problems"]] == [1, 2]
    assert refused.result["reach"] == refused.result["problems"][0]["reach"]
    assert refused.result["reach"]["z_max_mm"] == refused.result["reach"]["farthest"]["z"]
    assert _sim(runtime).batches == []


async def test_structural_failure_keeps_already_collected_reach_evidence(tmp_path: Path) -> None:
    runtime = _cell(tmp_path)
    refused = await _refusal(
        runtime,
        [
            {"op": "move", "pose": {"x": 300, "y": 0, "z": 500}},
            {"op": "move", "pose": {"x": 300, "y": 0, "z": 150}, "path": "arc"},
        ],
    )
    assert refused.failure_code == FAILURE_INVALID_ARGS
    assert refused.result["step"] == 2
    assert refused.result["problems"][0]["step"] == 1


async def test_grasp_propose_rejects_an_inside_box_pregrasp_that_ik_cannot_reach(
    tmp_path: Path,
) -> None:
    chain = DhChain.from_mapping(piper_context().kinematics.dh)
    from tests._harness.skill_stubs import PERCEPTION_WORKCELL_YAML, SimArmTransport
    from tests.test_scene_sequence import OVER_TABLE_JOINTS

    workcell = {**PERCEPTION_WORKCELL_YAML, "grasp": {"approach_mm": 180, "retreat_mm": 180}}
    arm = SimArmTransport(joints=OVER_TABLE_JOINTS, forward=chain.forward)
    runtime = _cell(tmp_path, CUP, arm=arm, workcell=workcell)
    await _read(runtime, "perceive.locate", {"queries": ["cup"]})
    refused = await runtime.plan(WORKCELL_ID, "grasp.propose", {"object": "o1"})
    assert isinstance(refused, SkillOutcome) and not refused.ok
    assert refused.failure_code == FAILURE_UNREACHABLE
    assert refused.result["reach"]["verified"] is True
    assert all(p["waypoint"] == "pregrasp" for p in refused.result["problems"])
    assert all("reach" in p for p in refused.result["problems"])
    assert _sim(runtime).batches == []


async def test_real_piper_yaw_motion_does_not_false_trip_a_configuration_switch(
    tmp_path: Path,
) -> None:
    from tests.test_scene_sequence import _plan

    runtime = _kinematic_cell(tmp_path)
    pose = (await _read(runtime, "arm.state", {})).result["tcp"]
    # Lift into the fixture's measured workspace before testing an in-place turn.
    target = {**pose, "z": pose["z"] + 40, "rz": pose["rz"] + 50}
    plan = await _plan(
        runtime,
        [
            {"op": "move_linear", "relative": {"dz": 40}},
            {"op": "move_linear", "pose": target},
        ],
    )
    assert plan.basis["reach"]["verified"]
    assert _sim(runtime).batches == []


async def test_joint_limit_refusals_are_collected_without_fabricating_joint_endpoints(
    tmp_path: Path,
) -> None:
    runtime = _kinematic_cell(tmp_path)
    refused = await _refusal(
        runtime,
        [
            {"op": "move_joints", "joints": {"j1": 180}},
            {"op": "move_joints", "joints": {"j2": 220}},
        ],
    )
    assert [p["step"] for p in refused.result["problems"]] == [1, 2]
    assert _sim(runtime).batches == []


def test_grasp_fit_and_arm_reach_have_distinct_recovery_guidance() -> None:
    assert "camera.wrist" in next_step(FAILURE_GRASP_INFEASIBLE, SIDE_EFFECT_NONE)
    assert "result.problems" in next_step(FAILURE_UNREACHABLE, SIDE_EFFECT_NONE)
    assert "height bands" in next_step(FAILURE_UNREACHABLE, SIDE_EFFECT_NONE)
