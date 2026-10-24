"""Real cycle/IK/permit/lease integration over synthetic I/O; no physical claims."""

from __future__ import annotations

import asyncio
import copy
import time
import uuid
from dataclasses import replace
from types import SimpleNamespace

import pytest

from leapflow.hardware.grasp_evidence import (
    EvidenceIdentity,
    GraspLimits,
    GraspMeasurement,
    TemporalGraspObserver,
)
from leapflow.hardware.skills.builtin.grasp_cycle import GraspCycle, GripperQualify
from leapflow.hardware.skills.builtin.grasp_verify import GraspInspect
from leapflow.hardware.skills.context import RunContext
from leapflow.hardware.skills.geometry import Pose, Transform, tool_cone_deg
from leapflow.hardware.skills.protocol import GraspCandidate, LocatedObject
from leapflow.hardware.skills.tuning import DEFAULT_GRASP_VERIFY
from piper_ops import cli, grasp, qualification
from piper_ops.config import Settings
from piper_ops.evidence import Evidence, Refusal
from piper_ops.session import Session
from leapflow.hardware.lhp.permit import save_private_key
from tests.test_piper_ops import snapshot as controller_snapshot
from tests._harness.policy_cell import piper_chain
from tests._harness.skill_stubs import (
    CAMERA_ID,
    IDENTITY_MATRIX,
    JOINTS,
    WORKCELL_ID,
    WORKCELL_YAML,
    SimRgbdCamera,
)
from tests.test_skill_tools import (
    QUALIFIED,
    WRITE_MODE_PERMIT_REQUIRED,
    _Tampering,
    _until,
    rigs as skill_rigs,
)

rigs = skill_rigs
START = (4, 77, -68, 80, -15, -150)
WRIST = "wrist_camera"


class SyntheticVerifier:
    """Synthetic measured geometry following simulated FK, with a real observer."""

    def __init__(self, ctx, grasp, *, fail=False):
        self.ctx, self.grasp, self.fail = ctx, grasp, fail
        self.tick = 0
        self.session_id = "synthetic-" + uuid.uuid4().hex
        self.sample = None
        self.limits = GraspLimits(
            **{
                k: int(DEFAULT_GRASP_VERIFY[k]) if k == "min_frames" else DEFAULT_GRASP_VERIFY[k]
                for k in GraspLimits.__dataclass_fields__
            }
        )
        self.observer = TemporalGraspObserver(
            lambda: self.sample,
            limits=self.limits,
            current=lambda: self.sample.identity,
            clock=time.monotonic,
        )

    def qualification(self):
        return {"qualification_digest": "synthetic-only", "planar_p95_mm": 1, "command_force_n": 1}

    async def inspect(self):
        return {
            "qualification": self.qualification(),
            "depth_quality": {
                "qualified": True,
                "temporal_p95_mm": 0.1,
                "frame_ids": [f"{self.session_id}:{i}:{i}" for i in range(1, 21)],
            },
        }

    async def verify(self, stage, *, samples, **kwargs):
        state = await self.ctx.motion.state()
        supported = abs(state.tcp.z - self.grasp.grasp.z) < 0.5
        observations = []
        for _ in range(samples):
            self.tick += 1
            identity = EvidenceIdentity(
                self.session_id,
                "scene",
                "synthetic-only",
                time.monotonic(),
                (f"{self.session_id}:{self.tick}:{self.tick}",),
            )
            position = state.tcp.position
            self.sample = GraspMeasurement(
                identity,
                (position[0], position[1], position[2] + 10),
                position,
                20,
                0.05,
                True,
                True,
                True,
                stage != "pre_grasp",
                True,
                True,
                False,
                supported or stage == "pre_grasp",
            )
            evidence = self.observer.observe_grasp()
            observations.append(
                {
                    "stage": stage,
                    "passed": not self.fail,
                    "state": evidence.state(
                        identity, now=time.monotonic(), limits=self.limits
                    ).value,
                    "qualification_digest": "synthetic-only",
                    "alignment_tool_mm": 0,
                    "source_identity": copy.deepcopy(identity.__dict__),
                    "evidence": evidence.to_dict(),
                }
            )
            await asyncio.sleep(0.001)
        return {**observations[-1], "observations": observations}


@pytest.fixture
async def cycle_rig(tmp_path, rigs, monkeypatch):
    chain = piper_chain()
    tcp = (
        Transform.from_pose(chain.forward(START))
        .compose(Transform.from_pose(Pose(0, 0, 120)))
        .to_pose()
    )
    assert tool_cone_deg(tcp) < WORKCELL_YAML["workspace"]["tool_cone_deg"]
    candidate = GraspCandidate(
        tcp.translated(dz=-10),
        20,
        50,
        10,
        1,
        "synthetic",
        retreat_mm=9,
        grasp_id="g1",
        object_id="o1",
    )
    obj = LocatedObject(
        "o1", "synthetic", 1, candidate.grasp.position, 30, 20, 20, 0, time.monotonic()
    )
    cell = copy.deepcopy(WORKCELL_YAML)
    cell["roles"].update({"camera": CAMERA_ID, "camera.wrist": WRIST})
    cell["frames"].update(
        {
            "camera": {"parent": "base", "procedure": "camera.handeye"},
            "camera.wrist": {"parent": "flange", "procedure": "wrist.handeye"},
        }
    )
    proxies = []
    contexts = []
    failing = [False]
    original = RunContext.__init__

    def construct(ctx, **kwargs):
        original(ctx, **kwargs)
        # Only sensor/scene boundaries are synthetic; motion, FK, gate and runtime remain real.
        ctx._scene = SimpleNamespace(
            grasp=lambda handle: candidate,
            object=lambda handle: obj,
            held=None,
            relocate=lambda seen: asyncio.sleep(0, result=seen),
            release=lambda: None,
        )
        ctx._grasp_verification = SyntheticVerifier(ctx, candidate, fail=failing[0])
        contexts.append(ctx)

    def wrap(client):
        proxy = _Tampering(client)
        proxies.append(proxy)
        return proxy

    monkeypatch.setattr(RunContext, "__init__", construct)
    rig = rigs(
        tmp_path,
        write_mode=WRITE_MODE_PERMIT_REQUIRED,
        cell=cell,
        node_skills=(GraspCycle(), GripperQualify(), GraspInspect()),
        devices={CAMERA_ID: SimRgbdCamera(), WRIST: SimRgbdCamera()},
        joints=START,
        forward=chain.forward,
        travel_reads=1,
        object_width_mm=20,
        lease_s=0.2,
        wrap=wrap,
    )
    registry = rig.nodes[0].registry
    registry.calibrate(
        "camera.handeye", pose=Pose(0, 0, 0).to_dict(),
        parameters={"target_mount": IDENTITY_MATRIX},
    )
    registry.calibrate(
        "wrist.handeye", pose=Pose(0, 0, 0).to_dict(),
        parameters={"target_mount": IDENTITY_MATRIX},
    )
    request = {
        "workcell": QUALIFIED,
        "skill": "grasp.cycle",
        "args": {
            "grasp_id": "g1",
            "place": candidate.grasp.translated(dx=10).to_dict(),
            "proof_lift_mm": 7,
        },
    }
    return SimpleNamespace(
        rig=rig, request=request, contexts=contexts, proxies=proxies, failing=failing
    )


async def approved_cycle(setup):
    preview = await setup.rig.service.hw_skill_run(**setup.request, dry_run=True)
    assert preview["ok"], preview
    assert not setup.rig.sim.batches
    request = {**setup.request, "args": preview["plan"]["args"]}
    return preview, request


async def test_actual_cycle_uses_one_cached_permit_lease_and_observer(cycle_rig):
    setup = cycle_rig
    preview, request = await approved_cycle(setup)
    planned = sum(p.calls.count("skill.plan") for p in setup.proxies)
    result = await setup.rig.service.hw_skill_run(**request, approved_plan=preview["plan"])
    assert result["ok"] and result["side_effect_state"] == "committed", result
    assert qualification.cycle_accepted(result, "synthetic-only"), result
    assert sum(p.calls.count("skill.plan") for p in setup.proxies) == planned
    assert any("control.renew" in p.calls for p in setup.proxies)
    assert len(setup.rig.gate.seen) == 1
    assert len([r for r in setup.rig.audit.records if r["action"] == "skill_permit"]) == 1
    assert len(setup.contexts) == 2  # read-only plan and one continuous executing context
    assert setup.contexts[-1].grasp_verification.tick > 18
    approved_joints = [w["joints"] for w in preview["plan"]["basis"]["paths"]]
    dispatched = [dict((c, v) for c, v in b if c in JOINTS) for b in setup.rig.sim.batches]
    assert [j for j in dispatched if j] == approved_joints
    assert not setup.rig.runtime.holds_device(WRIST)
    writes = len(setup.rig.sim.batches)
    replay = await setup.rig.service.hw_skill_run(**request, approved_plan=preview["plan"])
    assert not replay["ok"] and len(setup.rig.sim.batches) == writes
    assert setup.rig.sim.halts == 0
    status = await setup.rig.service.hw_skill_status()
    (run,) = status["runs"]
    assert run["plan_digest"] == preview["plan"]["digest"]
    assert run["version"] == preview["plan"]["version"]
    assert run["started_at"] <= run["finished_at"]


async def test_actual_cycle_verifier_refusal_halts_without_retreat(cycle_rig):
    setup = cycle_rig
    preview, request = await approved_cycle(setup)
    setup.failing[0] = True
    result = await setup.rig.service.hw_skill_run(**request, approved_plan=preview["plan"])
    assert not result["ok"] and result["failure_code"] == "guard_failed", result
    assert result["side_effect_state"] == "partial"
    assert result["result"]["consecutive_successes"] == 0
    assert setup.rig.sim.halts == 1
    assert len([b for b in setup.rig.sim.batches if any(c in JOINTS for c, v in b)]) == 1


@pytest.mark.parametrize("stop", ["explicit", "lease", "cancel"])
async def test_actual_cycle_stop_settles_without_replay(cycle_rig, stop):
    setup = cycle_rig
    preview, request = await approved_cycle(setup)
    setup.rig.sim.stall = True
    if stop == "cancel":
        task = asyncio.create_task(
            setup.rig.service.hw_skill_run(**request, approved_plan=preview["plan"])
        )
        await _until(lambda: setup.rig.sim.moving)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        started = await setup.rig.service.hw_skill_run(
            **request, approved_plan=preview["plan"], wait=False
        )
        assert started["status"] == "running", started
        await _until(lambda: setup.rig.sim.moving)
        assert setup.rig.runtime.holds_device(WRIST)
        if stop == "lease":
            await setup.rig.service.close()
        else:
            await setup.rig.service.hw_skill_stop(run_id=started["run_id"])
    await _until(lambda: all(not r.running for r in setup.rig.runtime.status()))
    (run,) = setup.rig.runtime.status()
    assert run.outcome.failure_code == ("lease_lost" if stop == "lease" else "stopped")
    assert run.outcome.side_effect_state != "none"
    assert run.outcome.evidence["grasp_cycle"]["consecutive_successes"] == 0
    assert setup.rig.sim.halts == 1
    assert not setup.rig.runtime.holds_device(WRIST)
    writes = len(setup.rig.sim.batches)
    await asyncio.sleep(0.05)
    assert len(setup.rig.sim.batches) == writes


@pytest.mark.parametrize("fault", ["wrap", "orientation", "budget", "workspace"])
async def test_real_ik_path_checks_wrapping_and_refuses_unapproved_paths(cycle_rig, fault):
    from leapflow.hardware.skills.protocol import SkillError

    setup = cycle_rig
    await approved_cycle(setup)
    ctx = setup.contexts[-1]
    state = await ctx.motion.state()
    target = state.tcp.translated(dz=2)
    if fault == "wrap":
        target = replace(target, rx=target.rx + 360)
        path = ctx.motion.check_cartesian_path(state, (target,))
        assert len(path) == 1
        predicted = ctx.motion.check_joints(path[0]["joints"], state.joints)
        assert predicted.distance_mm(target) <= 0.1
        assert predicted.rotation_to_deg(target) <= 0.1
    else:
        if fault == "orientation":
            target = replace(target, rx=target.rx + 5)
        elif fault == "budget":
            ctx.workcell.grasp_verify["cycle_max_waypoints"] = 1
            target = target.translated(dz=10)
        else:
            target = target.translated(dx=1000)
        with pytest.raises(SkillError):
            ctx.motion.check_cartesian_path(state, (target,))
    assert not setup.rig.sim.batches


@pytest.fixture
async def operations(cycle_rig, tmp_path, monkeypatch):
    """Real TCP Session/SkillService clients; only device I/O is simulated."""
    setup = cycle_rig
    registry = setup.rig.nodes[0].registry
    registry._context = replace(
        registry._context,
        provenance=replace(
            registry._context.provenance, verified_by="offline-test", verified_at=time.time()
        ),
    )
    original_value = setup.rig.sim._value
    receipt = 0

    def value(channel):
        nonlocal receipt
        if channel == "arm.telemetry":
            receipt += 1
            return controller_snapshot(seq=receipt).telemetry
        if channel == "arm.status":
            return {"ctrl_mode": 1, "arm_status": 0, "motion_status": int(setup.rig.sim.moving)}
        return original_value(channel)

    monkeypatch.setattr(setup.rig.sim, "_value", value)
    host, port = await setup.rig.nodes[0].server.start(host="127.0.0.1", port=0)
    token = tmp_path / "synthetic-token"
    token.write_text("offline-local-only")
    key = tmp_path / "synthetic-issuer.key"
    save_private_key(setup.rig.issuer._private, key)
    settings = replace(
        Settings(),
        root=tmp_path,
        node_id=setup.rig.nodes[0].fleet_node.node_id,
        endpoint=f"tcp://{host}:{port}",
        token_file=token,
        issuer_key=key,
    )
    return SimpleNamespace(setup=setup, settings=settings)


async def acquire_cycle(operations, path):
    settings = operations.settings
    setup = operations.setup
    planning = Evidence(path.with_name("preview-" + path.name))
    async with Session(settings, planning) as session:
        preview = await grasp.run(
            session,
            workcell=WORKCELL_ID,
            skill="grasp.cycle",
            arguments=setup.request["args"],
            operation="grasp_cycle",
        )
    assert preview["ok"], preview
    planning.finish(preview)
    approved = preview["preview"]
    args = SimpleNamespace(
        approved_digest=approved["declaration_digest"],
        approval_note="offline synthetic contract only",
        workcell=WORKCELL_ID,
    )
    evidence = Evidence(path)
    result = await cli.skill_execution(args, settings, evidence, approved, "grasp_cycle")
    return evidence.finish(result)


async def test_actual_operations_execute_readback_and_three_cycle_report(operations, monkeypatch):
    paths = [operations.settings.root / "data/runs" / f"cycle-{i}.json" for i in range(3)]
    artifacts = [await acquire_cycle(operations, path) for path in paths]
    for path, artifact in zip(paths, artifacts):
        assert artifact["post_exit_verified"] and qualification.acquisition_provenance(
            path, artifact
        )
    evidence = Evidence(operations.settings.root / "report.json")
    # This replaces kernel I/O, not the report, live run status, governance, or evidence admission.
    monkeypatch.setattr(
        qualification.health,
        "passive_can",
        lambda session: asyncio.sleep(0, result={"qualified": True, "synthetic": True}),
    )
    async with Session(operations.settings, evidence) as session:
        report = await qualification.report(session, WORKCELL_ID)
    assert report["qualified"] and report["consecutive_successes"] == 3, report
    assert [r["run_id"] for r in report["cycles"]] == [r["run_id"] for r in artifacts]
    assert len({r["run_id"] for r in artifacts}) == 3
    assert operations.setup.rig.sim.halts == 0
    # A live failed cycle must reset the streak even without a durable success artifact.
    operations.setup.failing[0] = True
    failed = await operations.setup.rig.service.hw_skill_run(**operations.setup.request)
    assert not failed["ok"], failed
    async with Session(operations.settings, evidence) as session:
        report = await qualification.report(session, WORKCELL_ID)
    assert not report["qualified"] and report["consecutive_successes"] == 0
    # Every result here uses synthetic sensors; none is exported as physical qualification.


async def test_actual_operations_post_exit_drift_halts_once_without_replay(operations, monkeypatch):
    original = cli.verify_after_exit

    async def drift(settings, evidence, plan, final):
        operations.setup.rig.sim.gripper_mm += 3
        operations.setup.rig.sim.gripper_target = operations.setup.rig.sim.gripper_mm
        return await original(settings, evidence, plan, final)

    monkeypatch.setattr(cli, "verify_after_exit", drift)
    path = operations.settings.root / "data/runs/drift.json"
    with pytest.raises(Refusal):
        await acquire_cycle(operations, path)
    assert operations.setup.rig.sim.halts == 1
    writes = len(operations.setup.rig.sim.batches)
    await asyncio.sleep(0.05)
    assert len(operations.setup.rig.sim.batches) == writes
    assert not qualification.acquisition_provenance(path, qualification.load(path))
