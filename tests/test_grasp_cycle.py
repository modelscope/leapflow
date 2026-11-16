"""Offline cycle, cached approval and qualification contracts; no hardware claims."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import math
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from leapflow.hardware.grasp_evidence import (
    EvidenceIdentity, GraspLimits, GraspMeasurement, TemporalGraspObserver,
)
from leapflow.hardware.skills.builtin.grasp_cycle import GraspCycle, GripperQualify
from leapflow.hardware.node_gate import NodeWriteGate
from leapflow.hardware.skills.frames import FrameGraph
from leapflow.hardware.skills.geometry import Pose
from leapflow.hardware.skills.motion import GovernedMotion, _within_gripper_tolerance
from leapflow.hardware.skills.workcell import WorkcellSpec
from tests._harness.skill_stubs import ARM_ID, SimArmTransport, SkillNodeRegistry
from leapflow.hardware.skills.protocol import GraspCandidate, LocatedObject, SkillError
from leapflow.hardware.skills.tuning import DEFAULT_FRESHNESS, DEFAULT_GRASP_VERIFY
from leapflow.hardware.skills.digest import run_permit_digest, wire_plan_digest
from piper_ops import calibration, cli, grasp, health, motion, qualification
from piper_ops.config import Settings
from piper_ops.evidence import Evidence, Refusal
from tests.test_skill_tools import MOVE, WRITE_MODE_PERMIT_REQUIRED, _Tampering, rigs as skill_rigs


rigs = skill_rigs


def context(*, session_id="test"):
    obj = LocatedObject("o1", "test", 1, (0, 0, 10), 30, 20, 20, 0, time.monotonic())
    candidate = GraspCandidate(
        Pose(0, 0, 10, 180), 20, 50, 20, 1, "test", grasp_id="g1", object_id="o1"
    )
    state = SimpleNamespace(tcp=Pose(0, 0, 50, 180), joints={"j1": 0})
    paths = []

    def path(current, targets):
        paths[:] = [
            {"target_index": index, "pose": pose.to_dict(), "joints": {"j1": index}}
            for index, pose in enumerate(targets)
        ]
        return paths

    async def move(joints, **kw):
        pose = Pose.from_mapping(next(p["pose"] for p in paths if p["joints"] == joints))
        return SimpleNamespace(state=SimpleNamespace(tcp=pose))

    counter, tick, sample = 0, 0, None
    limits = GraspLimits(**{k: int(DEFAULT_GRASP_VERIFY[k]) if k == "min_frames"
                           else DEFAULT_GRASP_VERIFY[k] for k in GraspLimits.__dataclass_fields__})
    observer = TemporalGraspObserver(lambda: sample, limits=limits,
                                    current=lambda: sample.identity,
                                    clock=lambda: sample.identity.observed_mono)

    async def verify(stage, **kw):
        nonlocal counter, tick, sample
        counter += 1
        observations = []
        for _ in range(kw.get("samples", 3)):
            tick += 1
            rise = 5 if counter == 3 else 20 if counter == 4 else 0
            identity = EvidenceIdentity(session_id, "scene", "q", 100 + tick * 0.05,
                                        (f"{session_id}:{tick}:{tick}",))
            sample = GraspMeasurement(identity, (0, 0, 10 + rise), (0, 0, rise), 20, 0.1,
                                      True, True, True, counter != 1, True, True, False,
                                      counter < 3 or counter >= 5)
            evidence = observer.observe_grasp()
            observations.append({
                "stage": stage, "passed": True,
                "state": evidence.state(identity, now=identity.observed_mono, limits=limits).value,
                "qualification_digest": "q", "alignment_tool_mm": 0,
                "source_identity": copy.deepcopy(identity.__dict__),
                "evidence": evidence.to_dict(),
            })
        return {**observations[-1], "observations": observations}

    motion = SimpleNamespace(
        state=AsyncMock(return_value=state),
        check_cartesian_path=path,
        gripper=AsyncMock(return_value=SimpleNamespace(blocked=False, width_mm=50)),
        move_joints=AsyncMock(side_effect=move),
        reached=lambda pose, state: True,
    )
    scene = SimpleNamespace(
        grasp=lambda handle: candidate,
        object=lambda handle: obj,
        held=None,
        relocate=AsyncMock(side_effect=lambda obj: obj),
        release=lambda: None,
    )
    ctx = SimpleNamespace(
        motion=motion,
        scene=scene,
        workcell=SimpleNamespace(
            grasp_verify=dict(DEFAULT_GRASP_VERIFY),
            gripper_spec={"max_width_mm": 69.5},
            freshness=dict(DEFAULT_FRESHNESS),
        ),
        grasp_verification=SimpleNamespace(
            qualification=lambda: {
                "qualification_digest": "q",
                "planar_p95_mm": 1,
                "command_force_n": 1,
            },
            verify=AsyncMock(side_effect=verify),
        ),
        note=lambda *args: None,
    )
    return ctx


async def test_cycle_uses_approved_waypoints_and_retained_verifier():
    ctx = context()
    skill = GraspCycle()
    plan = await skill.plan(ctx, {"grasp_id": "g1", "place": Pose(20, 0, 10, 180).to_dict()})
    outcome = await skill.execute(ctx, plan)
    assert outcome.ok and outcome.result["cycle_passed"]
    assert ctx.motion.move_joints.await_count == 7
    assert ctx.motion.gripper.await_count == 3
    assert qualification.cycle_accepted(
        {"status": "succeeded", "side_effect_state": "committed", "result": outcome.result}, "q"
    )


@pytest.mark.parametrize("failed_call", range(1, 7))
async def test_verification_failure_stops_without_replay_or_retreat(failed_call):
    ctx = context()
    original = ctx.grasp_verification.verify.side_effect
    calls = 0

    async def verify(*args, **kw):
        nonlocal calls
        calls += 1
        if calls == failed_call:
            return {"passed": False, "state": "unknown"}
        return await original(*args, **kw)

    ctx.grasp_verification.verify.side_effect = verify
    skill = GraspCycle()
    plan = await skill.plan(ctx, {"grasp_id": "g1", "place": Pose(20, 0, 10, 180).to_dict()})
    with pytest.raises(SkillError) as error:
        await skill.execute(ctx, plan)
    assert error.value.result["consecutive_successes"] == 0
    assert ctx.motion.move_joints.await_count < 7
    assert ctx.motion.gripper.await_count <= 2


async def test_changed_joint_branch_refuses_before_any_write():
    ctx = context()
    skill = GraspCycle()
    plan = await skill.plan(ctx, {"grasp_id": "g1", "place": Pose(20, 0, 10, 180).to_dict()})
    ctx.motion.state.return_value.joints = {"j1": 10}
    with pytest.raises(SkillError, match="start changed"):
        await skill.execute(ctx, plan)
    ctx.motion.move_joints.assert_not_awaited()
    ctx.motion.gripper.assert_not_awaited()


async def test_release_requires_independent_support():
    ctx = context()
    original = ctx.grasp_verification.verify.side_effect

    async def no_support(*args, **kw):
        result = await original(*args, **kw)
        result["evidence"]["object_on_support"] = False
        return result

    ctx.grasp_verification.verify.side_effect = no_support
    skill = GraspCycle()
    plan = await skill.plan(ctx, {"grasp_id": "g1", "place": Pose(20, 0, 10, 180).to_dict()})
    with pytest.raises(SkillError, match="support required|placement lost"):
        await skill.execute(ctx, plan)
    assert ctx.motion.gripper.await_count == 2


async def test_cycle_cancellation_propagates_and_records_failure():
    ctx = context()
    notes = []
    ctx.note = lambda key, value: notes.append(value)
    ctx.motion.move_joints.side_effect = asyncio.CancelledError
    skill = GraspCycle()
    plan = await skill.plan(ctx, {"grasp_id": "g1", "place": Pose(20, 0, 10, 180).to_dict()})
    with pytest.raises(asyncio.CancelledError):
        await skill.execute(ctx, plan)
    assert notes[-1]["cycle_passed"] is False
    assert notes[-1]["consecutive_successes"] == 0


async def test_gripper_acquisition_does_not_move_arm_or_certify_encoder():
    ctx = context()
    skill = GripperQualify()
    plan = await skill.plan(ctx, {"widths": [10, 40, 68]})
    outcome = await skill.execute(ctx, plan)
    assert outcome.ok and not outcome.result["qualified"]
    ctx.motion.move_joints.assert_not_awaited()
    assert len(outcome.result["samples"]) == 7


@pytest.mark.parametrize(
    "target,width,blocked",
    [
        (69.5, 68.99999999999999, False),
        (69.5, 69.0, False),
        (69.5, 68.999999, True),
        (10.0, 10.500000000000002, False),
        (10.0, 10.500001, True),
        (69.5, 60.0, True),
    ],
)
async def test_governed_gripper_uses_same_boundary_for_arrival_and_blocked(target, width, blocked):
    clock = [0.0]

    class FixedJawFeedback(SimArmTransport):
        def _value(self, channel_id):
            return width if channel_id == "gripper.position" else super()._value(channel_id)

    registry = SkillNodeRegistry(FixedJawFeedback(gripper_mm=width))
    envelope = registry.context(ARM_ID).channel("gripper.position").envelope
    spec = WorkcellSpec.from_mapping({
        "workcell": "jaw_test", "roles": {"arm": ARM_ID, "gripper": ARM_ID},
        "gripper_spec": {"max_width_mm": 69.5},
    })
    async def pause(delay):
        clock[0] += delay

    facade = GovernedMotion(
        registry=registry, gate=NodeWriteGate(registry, clock=lambda: clock[0]),
        spec=spec, frames=FrameGraph(spec, {}, None, {}), limits=spec.limits,
        cancel=asyncio.Event(), deadline=1.0, clock=lambda: clock[0], sleep=pause, poll_s=0.1,
    )
    result = await facade.gripper(target)
    assert result.width_mm == width and result.blocked is blocked
    assert registry.sim.batches == [(("gripper.position", target),)]
    if blocked:
        assert clock[0] >= envelope.effective_settling_s
    else:
        assert clock[0] == 0.0


@pytest.mark.parametrize(
    "actual,target,tolerance",
    [(math.nan, 10.0, 0.5), (math.inf, 10.0, 0.5), (10.0, math.inf, 0.5),
     (10.0, 10.0, math.inf), (10.0, 10.0, -0.5), (10.001, 10.0, 0.0)],
)
def test_gripper_boundary_rejects_nonfinite_or_outside_exact_tolerance(actual, target, tolerance):
    assert not _within_gripper_tolerance(actual, target, tolerance)


def test_gripper_zero_tolerance_remains_exact():
    assert _within_gripper_tolerance(10.0, 10.0, 0.0)
    assert not _within_gripper_tolerance(math.nextafter(10.0, math.inf), 10.0, 0.0)


async def test_exact_saved_plan_runs_once_without_replanning(tmp_path, rigs):
    clients = []

    def wrap(client):
        proxy = _Tampering(client)
        clients.append(proxy)
        return proxy

    rig = rigs(tmp_path, write_mode=WRITE_MODE_PERMIT_REQUIRED, wrap=wrap)
    preview = await rig.service.hw_skill_run(**MOVE, dry_run=True)
    assert "basis" in preview["plan"] and "args" in preview["plan"]
    count = sum(c.calls.count("skill.plan") for c in clients)
    request = {**MOVE, "args": preview["plan"]["args"]}
    result = await rig.service.hw_skill_run(**request, approved_plan=preview["plan"])
    assert result["ok"] and result["side_effect_state"] == "committed", result
    permit_records = [r for r in rig.audit.records if r["action"] == "skill_permit"]
    assert permit_records[0]["value"]["run_id"] == result["run_id"]
    assert sum(c.calls.count("skill.plan") for c in clients) == count
    writes = len(rig.sim.batches)
    repeated = await rig.service.hw_skill_run(**request, approved_plan=preview["plan"])
    assert not repeated["ok"] and len(rig.sim.batches) == writes


@pytest.mark.parametrize("change", ["target", "args", "basis", "digest"])
async def test_saved_plan_tampering_moves_nothing(tmp_path, rigs, change):
    rig = rigs(tmp_path, write_mode=WRITE_MODE_PERMIT_REQUIRED)
    preview = await rig.service.hw_skill_run(**MOVE, dry_run=True)
    saved = copy.deepcopy(preview["plan"])
    if change == "target":
        saved["targets"][0]["z"] += 1
    elif change == "args":
        saved["args"] = {"different": True}
    elif change == "basis":
        saved["basis"]["changed"] = True
    else:
        saved["digest"] = "bad"
    result = await rig.service.hw_skill_run(
        **{**MOVE, "args": preview["plan"]["args"]}, approved_plan=saved
    )
    assert not result["ok"] and not rig.sim.batches


def test_new_cli_commands_are_preview_only_by_default():
    for command in (["gripper", "qualify"], ["grasp", "cycle"], ["qualify", "report"]):
        args = cli.parser().parse_args(command)
        assert not getattr(args, "execute", False)


async def test_grasp_preview_requires_operator_selection_and_share_consent():
    session = SimpleNamespace()
    with pytest.raises(Refusal, match="share-frames"):
        await grasp.prepare(session, "cell", {}, share_frames=False)
    with pytest.raises(Refusal, match="grasp_index"):
        await grasp.prepare(session, "cell", {"target": {"box": [0, 0, 10, 10]}}, share_frames=True)


async def test_digest_bound_operational_consent_is_single_use(tmp_path):
    evidence = Evidence(tmp_path / "consent.json")
    plan = {
        "node_id": "node",
        "workcell": "cell",
        "skill_id": "grasp.cycle",
        "plan_digest": "digest",
    }
    consent = calibration._Consent(plan, evidence)
    request = SimpleNamespace(action=SimpleNamespace(metadata=plan))
    assert (await consent.request_approval(request)).value == "allow_once"
    assert (await consent.request_approval(request)).value == "deny"


def test_offline_jaw_certificate_requires_disjoint_acquired_holdout(tmp_path):
    rows = [{"command_mm": w, "feedback_mm": w, "blocked": False} for w in (10, 40, 68)]
    data = {
        "physical": True,
        "independent_reference": "offline-test-caliper",
        "mount_digest": "test",
        "uncertainty_mm": 0.1,
        "training": [],
        "holdout": [],
    }
    for split in ("training", "holdout"):
        filename = f"{split}.json"
        acquired_artifact(tmp_path / filename, split, "gripper.qualify", {"samples": rows})
        data[split] = [
            {
                "id": f"{split}:{i}",
                "sample": i,
                "acquisition_file": filename,
                "command_mm": w,
                "feedback_mm": w,
                "caliper_mm": w,
            }
            for i, w in enumerate((10, 40, 68))
        ]
    path = tmp_path / "metrology.json"
    path.write_text(json.dumps(data))
    result = qualification.jaw_certificate(path)
    assert result["qualified"] and not result["stored"]
    data["holdout"][0]["acquisition_file"] = "training.json"
    path.write_text(json.dumps(data))
    with pytest.raises(Refusal, match="reuse"):
        qualification.jaw_certificate(path)


def acquired_artifact(path, run_id, skill, result):
    """Synthetic audit fixtures test gates, never establish hardware qualification."""
    plan = {
        "workcell": "cell",
        "skill_id": skill,
        "version": "1.0.0",
        "args": {},
        "summary": "test only",
        "targets": [],
        "limits": {},
        "basis": {"grasp_verify": dict(DEFAULT_GRASP_VERIFY)},
    }
    plan["digest"] = wire_plan_digest(plan)
    approved = {
        "skill_plan": plan,
        "node_id": "node",
        "workcell": "cell",
        "skill_id": skill,
        "declaration_digest": "test-declaration",
        "plan_digest": plan["digest"],
    }
    artifact = {
        "ok": True,
        "executed": True,
        "operation": "gripper_qualify" if skill == "gripper.qualify" else "grasp_cycle",
        "side_effect_state": "committed",
        "result": result,
        "run_id": run_id,
        "node_id": "node",
        "workcell": "node/cell",
        "skill": skill,
        "plan": plan,
        "approved_preview": approved,
        "approved_plan_hash": motion.digest_plan(approved),
        "post_exit_verified": True,
        "post_exit_snapshot": {"enabled": True},
        "events": [
            {
                "event": "connected",
                "node_id": "node",
                "write_mode": "permit_required",
                "declaration_digest": "test-declaration", "execute": True,
            },
            {"event": "calibration_consent", "approved": True,
             "binding": {k: approved[k] for k in ("node_id", "workcell", "skill_id", "plan_digest")}},
            {"event": "qualification_result", "result": {
                "run_id": run_id, "plan": plan, "result": result,
                "side_effect_state": "committed", "ok": True}},
            {"event": "after_skill_client_exit", "snapshot": {"enabled": True}},
            {"event": "connected", "node_id": "node", "write_mode": "permit_required",
             "declaration_digest": "test-declaration", "execute": False},
            {"event": "after_client_exit", "snapshot": {"enabled": True}},
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact))
    audit = [
        {"action": "skill_run", "device": "node/cell", "outcome": "started " + run_id},
        {
            "action": "skill_permit",
            "device": "node/cell",
            "value": {
                "run_id": run_id,
                "plan_digest": plan["digest"],
                "approval_id": "test-only",
                "audience": "node",
                "request_digest": run_permit_digest(
                    workcell="cell",
                    skill_id=skill,
                    version="1.0.0",
                    plan_digest=plan["digest"],
                    limits={},
                ),
            },
        },
    ]
    path.with_suffix(".audit.ndjson").write_text("\n".join(json.dumps(row) for row in audit))
    return json.loads(json.dumps(artifact))


@pytest.mark.parametrize(
    "fault",
    ["none", "can", "p95", "unavailable", "audit", "side_effect", "post_exit", "replay", "failed",
     "source_reuse", "plan_digest", "version", "malformed_runs", "malformed_run",
     "malformed_inspection", "malformed_identity", "conflicts"],
)
async def test_report_requires_all_live_and_durable_gates(fault, tmp_path, monkeypatch):
    skill = GraspCycle()
    runs = []
    previous = None
    for i in range(3):
        ctx = context(session_id=f"test-{i}")
        plan = await skill.plan(ctx, {"grasp_id": "g1", "place": Pose(20, 0, 10, 180).to_dict()})
        outcome = await skill.execute(ctx, plan)
        result = previous if fault == "source_reuse" and i == 2 else outcome.result
        previous = outcome.result
        artifact = acquired_artifact(
            tmp_path / "data/runs" / f"cycle-{i}.json", str(i), "grasp.cycle", result
        )
        runs.append(
            {
                "run_id": str(i),
                "workcell": "node/cell",
                "skill_id": "grasp.cycle",
                "plan_digest": artifact["plan"]["digest"],
                "version": artifact["plan"]["version"],
                "status": "succeeded",
                "side_effect_state": "committed",
                "result": artifact["result"],
            }
        )
    if fault == "audit":
        (tmp_path / "data/runs/cycle-2.audit.ndjson").write_text("")
    elif fault == "side_effect":
        runs[-1]["side_effect_state"] = "unknown"
    elif fault == "post_exit":
        path = tmp_path / "data/runs/cycle-2.json"
        data = json.loads(path.read_text())
        data["post_exit_verified"] = False
        path.write_text(json.dumps(data))
    elif fault == "replay":
        runs[-1] = copy.deepcopy(runs[0])
    elif fault == "failed":
        runs.append({**runs[-1], "run_id": "failure", "status": "halted"})
    elif fault == "plan_digest":
        runs[-1]["plan_digest"] = "other"
    elif fault == "version":
        runs[-1]["version"] = "other"
    elif fault == "malformed_run":
        runs.append(None)
    elif fault == "malformed_identity":
        runs[-1]["run_id"] = []
    elif fault == "conflicts":
        conflicting = acquired_artifact(tmp_path / "data/runs/conflict.json", "2", "grasp.cycle", {})
        acquired_artifact(tmp_path / "data/runs/conflict-2.json", "2", "grasp.cycle", conflicting["result"])
    inspection = {
        "ok": True,
        "result": {
            "qualification": {"qualification_digest": "q"},
            "depth_quality": {
                "qualified": True,
                "frame_ids": [str(i) for i in range(20)],
                "temporal_p95_mm": None if fault == "p95" else 0.1,
            },
        },
    }
    if fault == "malformed_inspection":
        inspection["result"] = {"qualification": None}
    service = SimpleNamespace(
        hw_skill_list=AsyncMock(return_value={"ok": True}),
        hw_skill_run=AsyncMock(return_value=inspection),
        hw_skill_status=AsyncMock(
            return_value={
                "ok": True,
                "runs": None if fault == "malformed_runs" else runs,
                "unavailable": ["node"] if fault == "unavailable" else [],
            }
        ),
    )

    @asynccontextmanager
    async def factory(*args, **kw):
        yield service

    monkeypatch.setattr(calibration, "service", factory)
    monkeypatch.setattr(
        health, "passive_can", AsyncMock(return_value={"qualified": fault != "can"})
    )
    session = SimpleNamespace(
        settings=Settings(root=tmp_path, node_id="node"),
        snapshot=AsyncMock(return_value=SimpleNamespace(to_dict=lambda: {"enabled": True})),
    )
    result = await qualification.report(session, "cell")
    assert result["qualified"] is (fault == "none")
    assert bool(result["qualification_blockers"]) is (fault != "none")


async def test_incomplete_approved_artifact_cannot_fall_back_to_replanning():
    with pytest.raises(Refusal, match="complete cached"):
        async with calibration.service(SimpleNamespace(), {"skill_id": "grasp.cycle"}):
            pytest.fail("incomplete artifact was allowed")


@pytest.mark.parametrize("field", ["grasp_verify", "freshness"])
async def test_cycle_tuning_change_refuses_before_writes(field):
    ctx = context()
    skill = GraspCycle()
    plan = await skill.plan(ctx, {"grasp_id": "g1", "place": Pose(20, 0, 10, 180).to_dict()})
    getattr(ctx.workcell, field)[next(iter(getattr(ctx.workcell, field)))] += 1
    with pytest.raises(SkillError, match="tuning changed"):
        await skill.execute(ctx, plan)
    ctx.motion.gripper.assert_not_awaited()


@pytest.mark.parametrize("fault", ["none", "errors", "frozen", "remote"])
async def test_passive_can_never_borrows_remote_host_health(fault, tmp_path, monkeypatch):
    from tests.test_piper_ops import context as arm_context, snapshot

    monkeypatch.setattr(health.sys, "platform", "linux")
    counts = {
        "rx_packets": 10,
        "tx_packets": 0,
        "rx_errors": 0,
        "tx_errors": 0,
        "rx_dropped": 0,
        "tx_dropped": 0,
    }
    later = {**counts, "rx_packets": 20, "rx_errors": int(fault == "errors")}
    mock = AsyncMock(side_effect=[{"counts": counts}, {"counts": later}])

    async def sample(*args):
        return await mock()

    monkeypatch.setattr(health.asyncio, "to_thread", sample)
    devices = tmp_path / "devices"
    devices.mkdir()
    local = arm_context()
    (devices / "piper_arm.yaml").write_text(json.dumps(local.to_dict()))
    settings = Settings(
        root=tmp_path,
        node_devices=devices,
        poll_s=0.001,
        endpoint="tcp://192.0.2.1:8767" if fault == "remote" else "tcp://127.0.0.1:8767",
    )
    session = SimpleNamespace(
        settings=settings,
        # The served listing withholds transport.config; the node profile binds it.
        context=dataclasses.replace(
            local, transport=dataclasses.replace(local.transport, config={})
        ),
        snapshot=AsyncMock(
            side_effect=[snapshot(seq=1), snapshot(seq=1 if fault == "frozen" else 2)]
        ),
    )
    result = await health.passive_can(session)
    assert result["qualified"] is (fault == "none")
    if fault == "remote":
        mock.assert_not_awaited()


@pytest.mark.parametrize("fault", ["none", "missing_readback", "extra_connection", "foreign_node",
                                   "reordered", "consent", "outcome", "snapshot"])
def test_provenance_binds_execute_then_verified_readback(fault, tmp_path):
    path = tmp_path / "run.json"
    artifact = acquired_artifact(path, "run", "gripper.qualify", {"samples": []})
    assert qualification.acquisition_provenance(path, artifact)
    events = artifact["events"]
    if fault == "missing_readback":
        del events[4]
    elif fault == "extra_connection":
        events.insert(4, copy.deepcopy(events[0]))
    elif fault == "foreign_node":
        events[4]["node_id"] = "other"
    elif fault == "reordered":
        events[3], events[4] = events[4], events[3]
    elif fault == "consent":
        events[1]["binding"]["plan_digest"] = "other"
    elif fault == "outcome":
        events[2]["result"]["run_id"] = "other"
    elif fault == "snapshot":
        artifact["post_exit_snapshot"] = {"enabled": False}
    assert qualification.acquisition_provenance(path, artifact) is (fault == "none")


@pytest.mark.parametrize("fault", ["empty", "failed", "scope", "replay", "time", "nan",
                                   "state", "summary", "order", "proof"])
async def test_report_rejects_invalid_individual_cycle_observations(fault):
    ctx = context()
    skill = GraspCycle()
    plan = await skill.plan(ctx, {"grasp_id": "g1", "place": Pose(20, 0, 10, 180).to_dict()})
    outcome = await skill.execute(ctx, plan)
    run = {"status": "succeeded", "side_effect_state": "committed", "result": outcome.result}
    assert qualification.cycle_accepted(run, "q")
    verification = run["result"]["stages"][2]["verification"]
    observations = verification["observations"]
    if fault == "empty":
        observations[0] = {}
    elif fault == "failed":
        observations[0]["passed"] = False
    elif fault == "scope":
        observations[0]["source_identity"]["scene_id"] = "other"
    elif fault == "replay":
        observations[1] = copy.deepcopy(observations[0])
    elif fault == "time":
        observations[0]["source_identity"]["observed_mono"] += 100
    elif fault == "nan":
        observations[0]["evidence"]["uncertainty_mm"] = float("nan")
    elif fault == "state":
        observations[0]["state"] = "held_verified"
    elif fault == "summary":
        verification["source_identity"] = {}
    elif fault == "order":
        run["result"]["stages"].reverse()
    else:
        run["result"]["stages"][3]["verification"]["observations"][0]["evidence"]["proof"]["identity"]["scene_id"] = "other"
    assert not qualification.cycle_accepted(run, "q")


async def test_changed_cli_jaw_widths_refuse_before_opening_session(monkeypatch, tmp_path):
    args = cli.parser().parse_args(["gripper", "qualify", "--execute", "--widths", "11,40,68"])
    plan = {"skill_id": "gripper.qualify", "workcell": "piper_cell",
            "args": {"widths": [10, 40, 68]}}
    monkeypatch.setattr(cli, "approved", lambda *a: plan)
    execution = AsyncMock()
    monkeypatch.setattr(cli, "skill_execution", execution)
    with pytest.raises(Refusal, match="widths differ"):
        await cli.hardware(args, Settings(root=tmp_path), Evidence(tmp_path / "refused.json"))
    execution.assert_not_awaited()
