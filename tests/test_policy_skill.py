# Copyright (c) Alibaba, Inc. and its affiliates.
"""policy.run end to end: a served action-chunk policy streamed under the shield.

The arm, camera and policy server are simulated; the runtime, write gate,
binding, adapter, aggregator, shield and skill are the production ones. Pins
what an approver sees (a preview that never moves), every pre-motion refusal
(NONE), streaming in end-effector and joint space, and the run verdicts: a
a model outage stops the stream without a halt, while a shield refusal or stop
halts it. ``test_policy_vla`` covers the declared VLA spaces.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from leapflow.hardware.context import HardwareContext, Quality
from leapflow.hardware.skills.adapt.spec import stats_checksum
from leapflow.hardware.skills.policy_stream import WorkcellPolicies
from leapflow.hardware.skills.protocol import (
    FAILURE_CALIBRATION_STALE,
    FAILURE_DEVICE_FAULT,
    FAILURE_INVALID_ARGS,
    FAILURE_MODEL_UNAVAILABLE,
    FAILURE_PLAN_STALE,
    FAILURE_SHIELD_VIOLATION,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_STOPPED,
    FAILURE_TIMEOUT,
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_PARTIAL,
    STATUS_FAILED,
    STATUS_HALTED,
    SkillError,
)
from leapflow.hardware.skills.runtime import SkillRun
from leapflow.hardware.skills.shield import ContactLimit
from leapflow.hardware.transport import BatchReading, Reading
from tests._harness.policy_cell import (
    EE,
    GRIPPER,
    JOINT_START,
    TASK,
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
    PIPER_YAML,
    START_FLANGE,
    TOOL_LENGTH_MM,
    WORKCELL_ID,
    SimArmTransport,
    SimPolicy,
    action_space,
    embodiment,
)


def _ee_policy(step_z: float = 1.0, **kwargs: Any) -> SimPolicy:
    return SimPolicy(action_space(gripper={}), (0, 0, step_z, 0, 0, 0, 0), **kwargs)


def _joint_policy(**kwargs: Any) -> SimPolicy:
    return SimPolicy(action_space("joints"), (1.0, 0, 0, 0, 0, 0), **kwargs)


class _ContactRegistry:
    def __init__(self, readings: tuple[Reading, ...]) -> None:
        self._readings = readings

    async def read_batch(self, device_id: str, channel_ids: tuple[str, ...]) -> BatchReading:
        return BatchReading(device_id=device_id, readings=self._readings)


def _contact_port(readings: tuple[Reading, ...]) -> WorkcellPolicies:
    return WorkcellPolicies(
        registry=_ContactRegistry(readings),
        spec=None,
        frames=None,
        motion=None,
        settings=None,
        note=lambda _kind, _data: None,
    )


# ── Planning ─────────────────────────────────────────────────────────────────


async def test_plan_previews_the_first_chunk_without_moving(tmp_path: Path) -> None:
    policy = _ee_policy(step_z=2.0)
    runtime = policy_cell(tmp_path, policy)
    plan = await plan_run(runtime)
    assert not sim(runtime).batches and len(policy.requests) == 1
    assert (
        "'cube'" in plan.summary and "sim-policy 1" in plan.summary and "ee space" in plan.summary
    )
    assert "adapter" in plan.summary
    start_z = START_FLANGE[2] - TOOL_LENGTH_MM
    (end,) = plan.targets
    assert end.z == pytest.approx(start_z + 20.0, abs=0.5)  # ten actions of 2 mm
    assert plan.basis["model"] == {
        "name": "sim-policy",
        "version": "1",
        "checksum": "",
        "instance": "",
    }
    assert plan.basis["channels"] == [*EE, GRIPPER]
    assert str(plan.basis["adapter_digest"]).startswith("sha256:")
    request = policy.requests[0]
    assert request["task"] == TASK and set(request["images"]) == {"top"}
    assert request["state"][2] == pytest.approx(START_FLANGE[2])
    assert "pending" not in request  # the model never declared real-time chunking


async def test_plan_refuses_unbound_policies(tmp_path: Path) -> None:
    outcome = await refuse_run(policy_cell(tmp_path, _ee_policy()), policy="laundry")
    assert outcome.failure_code == FAILURE_SKILL_UNAVAILABLE


async def test_an_exact_policy_refuses_instructions_it_was_not_trained_on(tmp_path: Path) -> None:
    policy = _ee_policy(details={"instruction_policy": "exact"})
    outcome = await refuse_run(policy_cell(tmp_path, policy), task="stack the plates")
    assert outcome.failure_code == FAILURE_INVALID_ARGS
    assert "accepts only its own instructions" in outcome.message and not policy.requests


async def test_an_open_policy_takes_any_instruction(tmp_path: Path) -> None:
    policy = _ee_policy()
    await plan_run(policy_cell(tmp_path, policy), task="stack the plates")
    assert policy.requests[0]["task"] == "stack the plates"


async def test_plan_refuses_a_policy_server_that_is_down(tmp_path: Path) -> None:
    outcome = await refuse_run(policy_cell(tmp_path, _ee_policy(fail=True)))
    assert outcome.failure_code == FAILURE_MODEL_UNAVAILABLE


async def test_plan_refuses_a_duration_the_run_cap_cannot_hold(tmp_path: Path) -> None:
    policy = _ee_policy()
    outcome = await refuse_run(policy_cell(tmp_path, policy), duration_s=110.0)
    # 110 s plus the declared start-up room exceeds the runtime's 120 s cap.
    assert outcome.failure_code == FAILURE_INVALID_ARGS and "run cap of 120 s" in outcome.message
    assert "at most 105 s" in outcome.message and not policy.requests


_MIXED = {
    "fps": 20,
    "width": 12,
    "groups": [
        {"name": "arm", "kind": "ee_pose", "dims": list(range(6))},
        {"name": "wrist", "kind": "joints", "dims": list(range(6, 12))},
    ],
}
_EE_ONLY = embodiment(action_space(gripper={}), cameras={"top": "camera"})


@pytest.mark.parametrize(
    ("policy", "decl", "needle"),
    [
        (SimPolicy(_MIXED, (0.0,) * 12), None, "mix joint and end-effector"),
        (
            _ee_policy(),
            {**_EE_ONLY, "actions": {**_EE_ONLY["actions"], "arm": ["ee.x"]}},
            "must bind",
        ),
        (
            _ee_policy(),
            {**_EE_ONLY, "actions": {**_EE_ONLY["actions"], "wrist": [JOINTS[5]]}},
            "the model does not declare",
        ),
        (
            _ee_policy(),
            {**_EE_ONLY, "actions": {"arm": list(EE)}},
            "binds no channels",
        ),
        (
            _ee_policy(observation={"images": {"wrist": {"width": 16, "height": 12}}}),
            embodiment(action_space(gripper={})),
            "not a role or declared device",
        ),
        (
            _ee_policy(observation={"images": {"wrist": {"width": 16, "height": 12}}}),
            None,
            "which the model does not observe",
        ),
        (
            _ee_policy(details={"action_space": {**action_space(gripper={}), "fps": 0}}),
            None,
            "fps",
        ),
        (
            _ee_policy(
                details={
                    "action_space": {
                        **action_space(gripper={}),
                        "normalization": {
                            "mode": "min_max",
                            "min": [0.0] * 7,
                            "max": [1.0] * 7,
                            "checksum": stats_checksum("min_max", [0.0] * 7, [2.0] * 7),
                        },
                    }
                }
            ),
            None,
            "checksum",
        ),
        (_ee_policy(), {"actions": {}}, "at least one group"),
    ],
)
async def test_plan_refuses_a_binding_the_workcell_cannot_honour(
    tmp_path: Path, policy: SimPolicy, decl: dict | None, needle: str
) -> None:
    kwargs = {} if decl is None else {"decl": decl}
    runtime = policy_cell(tmp_path, policy, **kwargs)
    if WORKCELL_ID not in runtime.load_report.admitted:
        notes = " ".join(str(note.to_dict()) for note in runtime.load_report.rejected)
        assert needle in notes  # refused offline, before any model is asked
        return
    outcome = await refuse_run(runtime)
    assert outcome.failure_code == FAILURE_SKILL_UNAVAILABLE and needle in outcome.message


async def test_a_model_without_a_workcell_embodiment_is_refused(tmp_path: Path) -> None:
    outcome = await refuse_run(policy_cell(tmp_path, _ee_policy(), decl=None))
    assert outcome.failure_code == FAILURE_SKILL_UNAVAILABLE
    assert "declares no policies.cube embodiment" in outcome.message


async def test_workcell_jump_tuning_cannot_widen_the_device_limit(tmp_path: Path) -> None:
    runtime = policy_cell(tmp_path, _ee_policy(), settings={"max_jump_s": 2.0})
    outcome = await refuse_run(runtime)
    assert outcome.failure_code == FAILURE_SKILL_UNAVAILABLE
    assert "may only tighten" in outcome.message


async def test_gripper_jump_is_not_exempt_from_the_device_limit(tmp_path: Path) -> None:
    row = [*START_FLANGE, 0.0]
    arm = SimArmTransport(gripper_mm=100.0)
    outcome = await refuse_run(policy_cell(tmp_path, _ee_policy(rows=[row]), arm=arm))
    assert outcome.failure_code == FAILURE_SHIELD_VIOLATION
    assert "first 'gripper' action moves dim 0 by 100.00" in outcome.message
    assert "more than the 70" in outcome.message


async def test_plan_refuses_a_policy_trained_on_another_calibration(tmp_path: Path) -> None:
    policy = _ee_policy(details={"calibration": {"tcp": 1.0}})
    outcome = await refuse_run(policy_cell(tmp_path, policy))
    assert outcome.failure_code == FAILURE_CALIBRATION_STALE


async def test_plan_refuses_a_first_chunk_the_shield_would_stop(tmp_path: Path) -> None:
    # 6 mm per action (inside the ~6.8 mm slew limit) from z=250: thirty actions
    # cross the 80 mm floor.
    outcome = await refuse_run(policy_cell(tmp_path, _ee_policy(step_z=-6.0, horizon=30)))
    assert outcome.failure_code == FAILURE_SHIELD_VIOLATION
    assert "workspace" in outcome.message
    assert outcome.result["policy"]["violation"]


async def test_a_chunk_outrunning_the_slew_limit_trips_the_jump_check(tmp_path: Path) -> None:
    # 20 mm per action against a ~6.8 mm slew limit: the command falls ever
    # further behind the target until the gap exceeds ee.z's 150 mm jump bound.
    outcome = await refuse_run(policy_cell(tmp_path, _ee_policy(step_z=-20.0, horizon=30)))
    assert outcome.failure_code == FAILURE_SHIELD_VIOLATION
    assert "jumps" in outcome.message


async def test_plan_refuses_a_first_action_that_jumps_away_from_the_arm(tmp_path: Path) -> None:
    # ee.z may move 150 mm/s; a first action 200 mm away is a discontinuity.
    row = [*START_FLANGE[:2], START_FLANGE[2] + 200.0, *START_FLANGE[3:], 60.0]
    outcome = await refuse_run(policy_cell(tmp_path, _ee_policy(rows=[row])))
    assert outcome.failure_code == FAILURE_SHIELD_VIOLATION
    assert "adapter refused" in outcome.message and outcome.result["policy"]["violation"]


async def test_plan_refuses_a_chunk_of_the_wrong_width(tmp_path: Path) -> None:
    outcome = await refuse_run(policy_cell(tmp_path, _ee_policy(rows=[[1.0, 2.0]])))
    assert outcome.failure_code == FAILURE_MODEL_UNAVAILABLE and "unusable chunk" in outcome.message


_CONTACT = ContactLimit("joint.j1.effort", "arm", -5.0, 5.0)


@pytest.mark.parametrize(
    ("readings", "message"),
    [
        (
            (Reading("arm", _CONTACT.channel_id, 1.0, quality=Quality.STALE.value),),
            "untrustworthy",
        ),
        ((Reading("arm", _CONTACT.channel_id, math.nan),), "non-finite"),
        ((), "no reading"),
    ],
)
async def test_contact_monitoring_rejects_untrustworthy_telemetry(
    readings: tuple[Reading, ...], message: str
) -> None:
    binding = SimpleNamespace(contacts=(_CONTACT,))
    with pytest.raises(SkillError) as raised:
        await _contact_port(readings)._check_contact(binding)
    assert raised.value.failure_code == FAILURE_DEVICE_FAULT
    assert message in raised.value.message


async def test_contact_limit_violation_has_a_halted_verdict() -> None:
    reading = Reading("arm", _CONTACT.channel_id, 6.0)
    binding = SimpleNamespace(contacts=(_CONTACT,))
    with pytest.raises(SkillError) as raised:
        await _contact_port((reading,))._check_contact(binding)
    assert raised.value.failure_code == FAILURE_SHIELD_VIOLATION
    assert raised.value.status == STATUS_HALTED


# ── Streaming ────────────────────────────────────────────────────────────────


async def test_run_streams_shielded_end_effector_targets(tmp_path: Path) -> None:
    policy = _ee_policy(step_z=1.0)
    runtime = policy_cell(tmp_path, policy)
    outcome = await execute_run(runtime, await plan_run(runtime, speed_pct=10))
    assert outcome.ok, outcome
    assert outcome.side_effect_state == SIDE_EFFECT_COMMITTED
    arm = sim(runtime)
    report = outcome.result["policy"]
    assert report["chunks"] >= 2 and report["ticks"] == len(arm.batches) > 10
    assert report["violation"] == "" and report["end"] is not None
    assert report["ended"] == "duration" and "ended by duration" in outcome.message
    heights = commanded(arm, "ee.z")
    assert heights == sorted(heights) and heights[-1] > START_FLANGE[2] + 5.0
    assert commanded(arm, "arm.speed_pct") == [10.0]  # rides once, in the first batch
    assert arm.batches[0][0] == ("arm.speed_pct", 10.0)
    assert all({cid for cid, _ in batch} >= {*EE, GRIPPER} for batch in arm.batches)
    assert arm.halts == 0 and len(policy.requests) == 1 + report["chunks"]
    assert all("pending" not in request for request in policy.requests)


async def test_run_streams_joint_targets_checked_through_forward_kinematics(
    tmp_path: Path,
) -> None:
    arm = SimArmTransport(travel_reads=1, joints=JOINT_START, forward=piper_chain().forward)
    runtime = policy_cell(tmp_path, _joint_policy(), arm=arm)
    plan = await plan_run(runtime)
    assert plan.basis["space"] == "joint"
    outcome = await execute_run(runtime, plan)
    assert outcome.ok, outcome
    base_turn = commanded(arm, "joint.j1.position")
    assert base_turn == sorted(base_turn) and base_turn[-1] > 3.0
    assert not commanded(arm, "ee.z")  # never mixes motion spaces in one stream


async def test_joint_policies_refuse_a_dh_table_the_arm_disagrees_with(tmp_path: Path) -> None:
    # Joints read zero while the firmware reports START_FLANGE: the table is wrong.
    outcome = await refuse_run(policy_cell(tmp_path, _joint_policy()))
    assert outcome.failure_code == FAILURE_SKILL_UNAVAILABLE and "kinematics.dh" in outcome.message


async def test_a_shield_refusal_mid_run_halts_the_arm(tmp_path: Path) -> None:
    # TCP at z=120 descending 5 mm per action: the first chunk ends at 100,
    # the stream crosses the 80 mm floor shortly after.
    flange = (*START_FLANGE[:2], 120.0 + TOOL_LENGTH_MM, *START_FLANGE[3:])
    arm = SimArmTransport(travel_reads=1, flange=flange)
    runtime = policy_cell(tmp_path, _ee_policy(step_z=-5.0, horizon=4), arm=arm)
    outcome = await execute_run(runtime, await plan_run(runtime, duration_s=3.0))
    assert outcome.status == STATUS_HALTED, outcome
    assert outcome.failure_code == FAILURE_SHIELD_VIOLATION
    assert outcome.side_effect_state == SIDE_EFFECT_PARTIAL
    assert arm.halts == 1
    assert "workspace" in outcome.result["policy"]["violation"]
    assert min(commanded(arm, "ee.z")) >= 80.0 + TOOL_LENGTH_MM - 1e-6


async def test_a_policy_outage_before_the_first_action_moves_nothing(tmp_path: Path) -> None:
    policy = _ee_policy()
    runtime = policy_cell(tmp_path, policy)
    plan = await plan_run(runtime)
    policy.fail = True
    outcome = await execute_run(runtime, plan)
    assert outcome.failure_code == FAILURE_MODEL_UNAVAILABLE
    assert outcome.side_effect_state == SIDE_EFFECT_NONE
    assert not sim(runtime).batches and sim(runtime).halts == 0


async def test_a_stalled_policy_times_out_after_the_warmup(tmp_path: Path) -> None:
    policy = _ee_policy()
    runtime = policy_cell(tmp_path, policy, settings={"warmup_s": 0.2})
    plan = await plan_run(runtime)
    policy.latency_s = 1.0
    outcome = await execute_run(runtime, plan)
    assert outcome.status == STATUS_FAILED and outcome.failure_code == FAILURE_TIMEOUT
    assert "supplied no action" in outcome.message
    assert not sim(runtime).batches


async def test_a_changed_model_invalidates_the_approval(tmp_path: Path) -> None:
    policy = _ee_policy()
    runtime = policy_cell(tmp_path, policy)
    plan = await plan_run(runtime)
    policy.version = "2"
    outcome = await execute_run(runtime, plan)
    assert outcome.failure_code == FAILURE_PLAN_STALE
    assert not sim(runtime).batches


async def test_the_model_that_streams_is_the_one_checked_against_the_approval(
    tmp_path: Path,
) -> None:
    # The server swaps weights between two refreshes during execution: the run
    # binds once, checks that binding, and streams it -- never a later rebind.
    policy = _ee_policy(versions=("1", "2", "2"))
    runtime = policy_cell(tmp_path, policy)
    plan = await plan_run(runtime)
    assert policy.refreshes == 1
    outcome = await execute_run(runtime, plan)
    assert outcome.failure_code == FAILURE_PLAN_STALE and "approved model" in outcome.message
    assert policy.refreshes == 2 and not sim(runtime).batches


async def test_a_run_streams_the_approved_binding_only(tmp_path: Path) -> None:
    # A swap after the run's single bind cannot slip in: every response is
    # compared with the approved identity.
    policy = _ee_policy(versions=("1", "1", "2"))
    runtime = policy_cell(tmp_path, policy)
    outcome = await execute_run(runtime, await plan_run(runtime))
    assert outcome.ok, outcome
    assert policy.refreshes == 2 and policy.version == "1"


async def test_a_changed_adapter_invalidates_the_approval(tmp_path: Path) -> None:
    # Same weights, same identity -- but the declared image size moved, so the
    # adapter the approver saw is not the one that would run.
    policy = _ee_policy()
    runtime = policy_cell(tmp_path, policy)
    plan = await plan_run(runtime)
    policy.details["observation_space"] = {
        **policy.details["observation_space"],
        "images": {"top": {"width": 32, "height": 24}},
    }
    outcome = await execute_run(runtime, plan)
    assert outcome.failure_code == FAILURE_PLAN_STALE and "embodiment" in outcome.message
    assert not sim(runtime).batches


async def test_resolved_device_limits_are_pinned_by_the_adapter_digest(tmp_path: Path) -> None:
    runtime = policy_cell(tmp_path, _ee_policy())
    plan = await plan_run(runtime)
    declaration = yaml.safe_load(PIPER_YAML.read_text(encoding="utf-8"))
    ee_z = next(
        channel for channel in declaration["channels"] if channel["channel_id"] == "ee.z"
    )
    ee_z["envelope"]["tolerance"] = 1.0
    registry = runtime._registry  # type: ignore[attr-defined]
    registry._context = HardwareContext.from_mapping(declaration)

    outcome = await execute_run(runtime, plan)

    assert outcome.failure_code == FAILURE_PLAN_STALE and "embodiment" in outcome.message
    assert not sim(runtime).batches


async def test_a_stop_halts_the_stream(tmp_path: Path) -> None:
    runtime = policy_cell(tmp_path, _ee_policy(step_z=0.5))
    plan = await plan_run(runtime, duration_s=5.0)
    run = await runtime.execute(plan.digest)
    assert isinstance(run, SkillRun)
    arm = sim(runtime)
    for _ in range(200):
        if len(arm.batches) >= 3:
            break
        await asyncio.sleep(0.01)
    await runtime.stop(run.run_id)
    finished = await runtime.wait(run.run_id, timeout_s=5.0)
    assert finished is not None and finished.outcome is not None
    outcome = finished.outcome
    assert outcome.status == STATUS_HALTED and outcome.failure_code == FAILURE_STOPPED
    assert arm.halts >= 1


async def test_bad_policy_settings_refuse_the_workcell(tmp_path: Path) -> None:
    # The Piper's channels accept 100 Hz; a faster tick would outrun them.
    runtime = policy_cell(tmp_path, _ee_policy(), settings={"tick_hz": 500})
    report = runtime.load_report
    assert WORKCELL_ID not in report.admitted
    assert any("policy.tick_hz" in str(note.to_dict()) for note in report.rejected)
