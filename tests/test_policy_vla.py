# Copyright (c) Alibaba, Inc. and its affiliates.
"""Declared VLA spaces end to end: what the model says it speaks becomes motion.

Two declarations shaped like the policies the node is built for drive the
simulated Piper through the production binding, adapter and shield:

- SmolVLA-like: absolute joints in radians and a normalised gripper, mean/std
  normalised, 30 Hz.
- EdgeVLA-like: end-effector deltas in metres and axis-angle in the tool frame,
  a binary gripper, 10 Hz, no state.

Plus the contract v1.1 behaviours: real-time chunking (``pending``), a model
that reports ``done``, ``progress`` echoed, IK for a pose policy bound to
joints, the declared aggregation modes, and an answer from a different model.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import pytest

from leapflow.hardware.skills.adapt.spec import stats_checksum
from leapflow.hardware.skills.protocol import (
    FAILURE_MODEL_UNAVAILABLE,
    FAILURE_SHIELD_VIOLATION,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_PARTIAL,
)
from tests._harness.policy_cell import (
    EE,
    GRIPPER,
    JOINT_START,
    commanded,
    execute_run,
    piper_chain,
    plan_run,
    policy_cell,
    refuse_run,
    sim,
)
from tests._harness.skill_stubs import (
    JOINTS,
    START_FLANGE,
    TOP_IMAGE,
    SimArmTransport,
    SimPolicy,
    action_space,
    embodiment,
)

_CAMERAS = {"top": "camera"}


def _joint_arm() -> SimArmTransport:
    return SimArmTransport(travel_reads=1, joints=JOINT_START, forward=piper_chain().forward)


def _ee_policy(step_z: float = 1.0, **kwargs: Any) -> SimPolicy:
    return SimPolicy(action_space(gripper={}), (0, 0, step_z, 0, 0, 0, 0), **kwargs)


# ── SmolVLA-like: normalised absolute joints ─────────────────────────────────


def _smolvla_space() -> dict[str, Any]:
    mean = [math.radians(v) for v in JOINT_START] + [0.5]
    std = [0.5] * 7
    space = action_space("joints", fps=30.0, unit="rad", gripper={"unit": "normalized"})
    space["normalization"] = {
        "mode": "mean_std",
        "mean": mean,
        "std": std,
        "checksum": stats_checksum("mean_std", mean, std),
    }
    return space


async def test_a_smolvla_style_policy_streams_normalised_joint_targets(tmp_path: Path) -> None:
    # 0.02 std per action is 0.01 rad (~0.57 deg) of base turn at 30 Hz.
    policy = SimPolicy(_smolvla_space(), (0.02, 0, 0, 0, 0, 0, 0))
    arm = _joint_arm()
    runtime = policy_cell(tmp_path, policy, arm=arm)
    plan = await plan_run(runtime)
    assert plan.basis["space"] == "joint" and plan.basis["channels"] == [*JOINTS, GRIPPER]
    assert "unnormalize" in plan.summary and "units" in plan.summary
    state = policy.requests[0]["state"]
    assert state[:6] == pytest.approx([0.0] * 6, abs=1e-9)  # the arm sits on the mean
    assert state[6] == pytest.approx((60.0 / 70.0 - 0.5) / 0.5, abs=1e-6)

    outcome = await execute_run(runtime, plan)
    assert outcome.ok, outcome
    base_turn = commanded(arm, JOINTS[0])
    assert base_turn == sorted(base_turn) and base_turn[-1] > 3.0
    first = policy.responses[1][0][0] * 0.5 + math.radians(JOINT_START[0])
    assert base_turn[0] == pytest.approx(math.degrees(first), abs=0.6)
    assert all(abs(v - 60.0) < 1e-3 for v in commanded(arm, GRIPPER))  # held where it was
    assert not commanded(arm, "ee.z")


# ── EdgeVLA-like: tool-frame deltas and a binary gripper ─────────────────────


def _edgevla_policy(grip: float, **kwargs: Any) -> SimPolicy:
    space = action_space(
        fps=10.0,
        unit="m",
        rotation="axis_angle",
        angle_unit="rad",
        reference="delta",
        frame="tool",
        gripper={"unit": "normalized", "binary": True},
    )
    # The flange points down, so -1 mm along tool z climbs 1 mm in base.
    rows = [[0.0, 0.0, -0.001, 0.0, 0.0, 0.0, grip]] * 5
    return SimPolicy(space, (0.0,) * 7, observation={"images": TOP_IMAGE}, rows=rows, **kwargs)


@pytest.mark.parametrize(
    ("grip", "overrides", "width_mm"),
    [
        (0.0, None, 0.0),  # close
        (1.0, None, 70.0),  # open to the workcell's declared width
        # Inside the hysteresis band: keeps the decision the open gripper implies.
        (0.55, {"action": {"gripper": {"hysteresis": 0.1}}}, 70.0),
    ],
)
async def test_an_edgevla_style_policy_streams_tool_frame_deltas(
    tmp_path: Path, grip: float, overrides: dict | None, width_mm: float
) -> None:
    policy = _edgevla_policy(grip)
    decl = embodiment(policy.space, cameras=_CAMERAS)
    if overrides is not None:
        decl["overrides"] = overrides
    runtime = policy_cell(tmp_path, policy, decl=decl)
    plan = await plan_run(runtime)
    for step in ("units", "gripper", "rotation", "reference"):
        assert step in plan.summary
    assert policy.requests[0]["state"] == []  # the model observes images only

    outcome = await execute_run(runtime, plan)
    assert outcome.ok, outcome
    arm = sim(runtime)
    heights = commanded(arm, "ee.z")
    assert heights[-1] > START_FLANGE[2] + 3.0
    assert all(b >= a - 0.5 for a, b in zip(heights, heights[1:]))
    for channel, start in zip(EE[:2], START_FLANGE[:2]):
        assert all(abs(v - start) < 1e-3 for v in commanded(arm, channel))
    assert all(abs(abs(v) - 180.0) < 1e-3 for v in commanded(arm, "ee.rx"))
    assert all(abs(v - width_mm) < 1e-6 for v in commanded(arm, GRIPPER))


# ── IK: a pose policy bound to joints ────────────────────────────────────────

_IK_DECL = {"actions": {"arm": list(JOINTS), "gripper": [GRIPPER]}, "cameras": _CAMERAS}


async def test_a_pose_policy_bound_to_joints_is_solved_by_ik(tmp_path: Path) -> None:
    policy = _ee_policy(step_z=-1.0)
    arm = _joint_arm()
    runtime = policy_cell(tmp_path, policy, arm=arm, decl=_IK_DECL)
    plan = await plan_run(runtime)
    assert plan.basis["space"] == "joint" and "ik" in plan.summary
    start = piper_chain().forward(JOINT_START)
    assert policy.requests[0]["state"][2] == pytest.approx(start.z, abs=1e-6)  # FK state

    outcome = await execute_run(runtime, plan)
    assert outcome.ok, outcome
    chain = piper_chain()
    batches = [dict(batch) for batch in arm.batches]
    poses = [chain.forward(tuple(batch[j] for j in JOINTS)) for batch in batches]
    assert poses[-1].z < start.z - 5.0
    assert all(abs(p.x - start.x) < 0.5 and abs(p.y - start.y) < 0.5 for p in poses)
    assert not commanded(arm, "ee.z")


async def test_ik_refuses_a_pose_outside_the_joint_limits(tmp_path: Path) -> None:
    # Climbing 40 mm straight up from the start needs j5 past its 70 deg limit.
    policy = _ee_policy(step_z=4.0)
    runtime = policy_cell(tmp_path, policy, arm=_joint_arm(), decl=_IK_DECL)
    outcome = await refuse_run(runtime)
    assert outcome.failure_code == FAILURE_SHIELD_VIOLATION
    assert "joint limits" in outcome.message


# ── Real-time chunking ───────────────────────────────────────────────────────


def _is_tail(rows: Sequence[Sequence[float]], of: Sequence[Sequence[float]]) -> bool:
    if not rows or len(rows) > len(of):
        return False
    tail = of[len(of) - len(rows) :]
    return all(
        all(abs(a - b) < 1e-6 for a, b in zip(row, expected)) for row, expected in zip(rows, tail)
    )


async def test_a_pending_aware_policy_is_sent_the_rows_not_yet_executed(tmp_path: Path) -> None:
    space = {**action_space(gripper={}), "pending": True}
    policy = SimPolicy(space, (0, 0, 1.0, 0, 0, 0, 0), latency_s=0.05)
    runtime = policy_cell(tmp_path, policy)
    outcome = await execute_run(runtime, await plan_run(runtime, duration_s=1.2))
    assert outcome.ok, outcome
    # Nothing is in flight for the preview, nor for the run's first chunk.
    assert all("pending" not in request for request in policy.requests[:2])
    later = policy.requests[2:]
    assert later, policy.requests
    for index, request in enumerate(later, start=2):
        assert "pending" in request, (index, len(policy.requests))
        pending = request["pending"]
        assert isinstance(pending["latency_steps"], int) and pending["latency_steps"] >= 1
        assert pending["actions"], request
        # Absolute mm/deg rows invert to themselves: the unstarted tail of the
        # chunk executing when the request went out.
        assert _is_tail(pending["actions"], policy.responses[index - 1])


# ── done, progress ───────────────────────────────────────────────────────────


async def test_a_model_reporting_done_ends_the_run_early(tmp_path: Path) -> None:
    policy = _ee_policy(done_after=3, progress=0.75)
    runtime = policy_cell(tmp_path, policy)
    plan = await plan_run(runtime, duration_s=5.0)
    assert "already reports the task done" not in plan.summary
    outcome = await execute_run(runtime, plan)
    assert outcome.ok, outcome
    report = outcome.result["policy"]
    assert report["ended"] == "done" and "ended by done" in outcome.message
    assert report["chunks"] == 2 and len(policy.requests) == 3  # no inference after done
    assert report["elapsed_s"] < 2.0 and report["progress"] == pytest.approx(0.75)


async def test_a_preview_tells_the_approver_the_model_already_reports_done(
    tmp_path: Path,
) -> None:
    policy = _ee_policy(done_after=1)
    runtime = policy_cell(tmp_path, policy)
    plan = await plan_run(runtime, duration_s=5.0)
    assert "already reports the task done" in plan.summary
    outcome = await execute_run(runtime, plan)
    assert outcome.ok and outcome.result["policy"]["ended"] == "done"
    assert outcome.result["policy"]["chunks"] == 1 and len(policy.requests) == 2


# ── Aggregation and smoothing as declared ────────────────────────────────────


@pytest.mark.parametrize(
    "extra",
    [{"aggregate": "temporal_ensemble"}, {"aggregate": "latest"}, {"smooth_hz": 5.0}],
)
async def test_declared_aggregation_streams_a_continuous_climb(
    tmp_path: Path, extra: dict[str, Any]
) -> None:
    policy = _ee_policy(step_z=1.0)
    decl = embodiment(policy.space, cameras=_CAMERAS, **extra)
    runtime = policy_cell(tmp_path, policy, decl=decl)
    outcome = await execute_run(runtime, await plan_run(runtime))
    assert outcome.ok, outcome
    heights = commanded(sim(runtime), "ee.z")
    assert heights[-1] > START_FLANGE[2] + 3.0
    assert all(b >= a - 1e-6 for a, b in zip(heights, heights[1:]))


async def test_the_adapter_digest_pins_the_declared_aggregation(tmp_path: Path) -> None:
    first = await plan_run(policy_cell(tmp_path / "a", _ee_policy()))
    decl = embodiment(action_space(gripper={}), cameras=_CAMERAS, aggregate="latest")
    second = await plan_run(policy_cell(tmp_path / "b", _ee_policy(), decl=decl))
    assert first.basis["adapter_digest"] != second.basis["adapter_digest"]


# ── Identity echo ────────────────────────────────────────────────────────────


async def test_a_first_chunk_from_a_different_model_moves_nothing(tmp_path: Path) -> None:
    policy = _ee_policy(swap_after=1)  # the preview's answer is the approved model's
    runtime = policy_cell(tmp_path, policy)
    outcome = await execute_run(runtime, await plan_run(runtime))
    assert outcome.failure_code == FAILURE_MODEL_UNAVAILABLE
    assert "different model" in outcome.message
    assert outcome.side_effect_state == SIDE_EFFECT_NONE and not sim(runtime).batches


async def test_a_model_swapped_mid_run_stops_the_stream(tmp_path: Path) -> None:
    policy = _ee_policy(swap_after=2)
    runtime = policy_cell(tmp_path, policy)
    outcome = await execute_run(runtime, await plan_run(runtime, duration_s=3.0))
    assert outcome.failure_code == FAILURE_MODEL_UNAVAILABLE
    assert "different model" in outcome.message
    assert outcome.side_effect_state == SIDE_EFFECT_PARTIAL and sim(runtime).batches
    assert len(policy.requests) == 3
