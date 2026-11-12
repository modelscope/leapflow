# Copyright (c) Alibaba, Inc. and its affiliates.
"""Real owner/LHP/permit/cache/compute bridge lane with simulated sensor I/O only.

These tests neither access hardware nor establish physical calibration accuracy.
"""
from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import replace
from pathlib import Path

import pytest

from leapcalib.contracts.identity import digest_of
from leapflow.cache.manager import CacheManager
from leapflow.hardware.calibration.acquisition import CollectionProfile
from leapflow.hardware.calibration.bridge_policy import BridgePolicy
from leapflow.hardware.calibration.bridge_assessment import shadow_problem
from leapflow.hardware.calibration.worker import CalibrationWorker
from leapflow.hardware.calibration_store import CalibrationRecord
from leapflow.hardware.context import Channel, ContextProvenance, HardwareContext, TransportRef
from leapflow.hardware.lhp.permit import PermitIssuer, PermitVerifier, generate_keypair
from leapflow.hardware.lhp.server import LhpNodeServer
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.skills.builtin.bridge_calibration import skills
from leapflow.hardware.skills.frames import FrameGraph
from leapflow.hardware.skills.runtime import SkillRuntime, SkillRuntimeSettings
from leapflow.hardware.skills.vision import CharucoTarget
from leapflow.hardware.rgbd import CameraIntrinsics
from leapflow.hardware.skills.geometry import Pose, Transform
from leapflow.layout import CacheLayout
from tests._harness.skill_stubs import ARM_ID, JOINTS, WORKCELL_ID, WORKCELL_YAML, IDENTITY_MATRIX, piper_context, write_workcell
from tests.test_leapcalib_acquisition import Provider, frame, profile
from tests.test_leapcalib_bridge import budget, detection
from tests.test_leapcalib_collection import Sensor, request, permit, finished

pytestmark = pytest.mark.component


RENDER_INTRINSICS = CameraIntrinsics(640, 480, 900., 900., 320., 240.)


class Camera(Sensor):
    def __init__(self, device_id, *, rendered=False):
        super().__init__()
        self.device_id, self.rendered = device_id, rendered
        self.late_defect = ""
        if rendered:
            from tests.test_camera_calibration import _render
            self.image, self.depth = _render(Transform.from_pose(Pose(-50, -40, 500, 15, 10, 0)), RENDER_INTRINSICS, True)

    async def capture_rgbd(self, *, after_monotonic=None):
        if self.late_defect and self.calls >= 7:
            self.defect = self.late_defect
        value = replace(await super().capture_rgbd(after_monotonic=after_monotonic), device_id=self.device_id)
        if self.rendered:
            intrinsics = replace(RENDER_INTRINSICS, fx=901.) if self.defect == "intrinsics" else RENDER_INTRINSICS
            value = replace(value, width=640, height=480, color=self.image, depth=self.depth,
                            depth_scale_m=.0001, intrinsics=intrinsics)
        return value


@pytest.fixture
async def bridge_owner(tmp_path, monkeypatch, request):
    rendered = getattr(request, "param", False)
    if rendered:
        cv2 = pytest.importorskip("cv2")
        if not hasattr(getattr(cv2, "aruco", None), "CharucoDetector"):
            pytest.skip("this OpenCV build has no ChArUco detector")
    values = {**dict.fromkeys(JOINTS, 0.0), **dict(zip(("ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz"), (250, 0, 370, 180, 0, 0))),
              "arm.enabled": True, "arm.motion": True, "arm.ctrl_mode": "can", "arm.status": {"arm_status": 0, "joint_communication_errors": []}}
    arm = replace(piper_context(), transport=TransportRef("mock", {"values": values}), provenance=ContextProvenance(verified_by="software-test"))
    contexts = [arm, *[HardwareContext(device_id=name, channels=(Channel("frame", direction="read"),), transport=TransportRef("mock", {}),
                                      provenance=ContextProvenance(verified_by="software-test")) for name in ("fixed_camera", "wrist_camera")]]
    registry = HardwareRegistry(HardwareSettings(instrument_db_path=str(tmp_path / "instrument.duckdb")), providers=[Provider(c) for c in contexts])
    loaded = registry.load()
    assert not loaded.rejected, loaded
    registry.bind_persistence(cache_manager=CacheManager(CacheLayout(tmp_path / "cache"), profile_id="bridge-test"))
    worker = CalibrationWorker(tmp_path.parent / ("br-" + tmp_path.name[-8:]) / "s", enabled=True)
    registry.bind_calibration_worker(worker)
    sensors, writes = {}, []
    for name in ("fixed_camera", "wrist_camera"):
        sensor = Camera(name, rendered=rendered)
        sensors[name] = sensor
        monkeypatch.setattr(await registry.transport(name), "capture_rgbd", sensor.capture_rgbd, raising=False)
    arm_transport = await registry.transport(ARM_ID)
    async def forbidden_write(*args, **kwargs):
        writes.append((args, kwargs))
        raise AssertionError("stationary bridge must never dispatch arm writes")
    monkeypatch.setattr(arm_transport, "write", forbidden_write)
    monkeypatch.setattr(arm_transport, "halt", forbidden_write)
    write_workcell(tmp_path / "workcells", {**WORKCELL_YAML,
        "roles": {"arm": ARM_ID, "camera": "fixed_camera", "camera.wrist": "wrist_camera"},
        "frames": {**WORKCELL_YAML["frames"], "camera": {"parent": "base", "procedure": "camera.handeye"},
                   "camera.wrist": {"parent": "flange", "procedure": "camera.wrist.handeye"}}})
    runtime = SkillRuntime(registry, settings=SkillRuntimeSettings(workcells_dir=tmp_path / "workcells"), skills=skills())
    wrist = CalibrationRecord(WORKCELL_ID, "camera.wrist.handeye", time.time() - 20,
        parameters={"mounting": "eye_in_hand", "camera_role": "camera.wrist", "camera_device": "wrist_camera",
                    "units": "mm", "orientation_convention": "Rz@Ry@Rx;degrees", "target_mount": IDENTITY_MATRIX,
                    "intrinsics": (RENDER_INTRINSICS if rendered else frame(1).intrinsics).to_dict(),
                    "fit": {"translation_rms_mm": .1, "rotation_rms_deg": .01}}, matrix=IDENTITY_MATRIX)
    verification = CalibrationRecord(WORKCELL_ID, "camera.wrist.verify", time.time() - 10,
        parameters={"camera_role": "camera.wrist", "depends_on": {"camera.wrist": wrist.recorded_at}, "training_disjoint": True,
                    "points": 5, "p95_mm": .1, "rms_mm": .1, "max_p95_mm": 5., "max_rms_mm": 5.})
    assert registry.calibration_store.record(wrist) and registry.calibration_store.record(verification)
    policy = BridgePolicy("camera", "camera.wrist", "sheet", "stationary-epoch", digest_of(vars(wrist)), digest_of(vars(verification)),
        {"node_session_id": runtime._node_session_id, "mount_digest": frame(1).metadata["mount_digest"],
         "source_profile_digest": frame(1).metadata["profile_digest"], "confirmation_id": "software-fixture-not-human"},
        budget(terms={key: (.1, .2) for key in budget().terms}), .5, 50., 2, 3, 3)
    declaration = {**WORKCELL_YAML, "roles": {"arm": ARM_ID, "camera": "fixed_camera", "camera.wrist": "wrist_camera"},
        "frames": {**WORKCELL_YAML["frames"], "camera": {"parent": "base", "procedure": "camera.handeye"},
                   "camera.wrist": {"parent": "flange", "procedure": "camera.wrist.handeye"}},
        "calibration_target": (CharucoTarget(5, 4, 20, 15) if rendered else CharucoTarget(5, 4, 10, 7)).to_dict(),
        "calibration_bridge": policy.to_dict(),
        "calibration_acquisition": CollectionProfile("camera", "fixed", ("can",), profile(timeout_s=2), 60, 64 * 1024 * 1024, 6).to_dict()}
    write_workcell(tmp_path / "workcells", declaration)
    assert not runtime.load_report.rejected
    private, public = generate_keypair()
    issuer = PermitIssuer(private)
    server = LhpNodeServer(registry, node_id="node", skill_runtime=runtime, permit_verifier=PermitVerifier({issuer.key_id: public}, node_id="node"))
    from tests.test_workcell_reload import approved_reload
    await approved_reload(server, issuer)
    if not rendered:
        monkeypatch.setattr("leapflow.hardware.calibration.paired.detect_board", lambda f, t: detection())
    try:
        yield registry, runtime, server, issuer, sensors, writes, declaration, tmp_path
    finally:
        await runtime.close()
        await server.sessions.shutdown()
        await worker.close()
        await registry.close_all()


async def bridge_plan(server, *, rendered=False):
    region = [200, 140, 260, 240] if rendered else [2, 1, 8, 8]
    return await request(server, "skill.plan", workcell=WORKCELL_ID, skill="calib.bridge",
                         args={"fixed_region": region, "wrist_region": region})


@pytest.mark.parametrize("bridge_owner", [False, True], indirect=True)
async def test_real_bridge_shadow_indexed_replay_and_no_motion(bridge_owner):
    registry, runtime, server, issuer, sensors, writes, _, _ = bridge_owner
    planned = await bridge_plan(server, rendered=sensors["wrist_camera"].rendered)
    assert "plan" in planned, planned
    assert planned["consent"] == "each_run" and not planned["plan"]["targets"]
    denied = await server.handle_request({"id": "denied", "method": "skill.execute", "params": {"plan_digest": planned["plan"]["digest"], "lease_s": 15}})
    assert "error" in denied or denied["result"].get("outcome", {}).get("ok") is not True
    started = await request(server, "skill.execute", plan_digest=planned["plan"]["digest"], permit=permit(issuer, planned), lease_s=15)
    run = await finished(runtime, started)
    assert run.outcome.ok, run.outcome.to_dict()
    report = run.outcome.result
    assert report["verdict"] == "rejected" and report["candidate"] is not None
    assert report["campaign_acceptance"] == "insufficient_evidence" and report["acceptance_blockers"]
    assert not writes and report["activation"] == "none" and len(report["captures"]) == 6
    shadow = await asyncio.to_thread(registry.calibration_generations.candidate, report["shadow_record_digest"], workcell_id=WORKCELL_ID)
    assert not shadow_problem(shadow)
    verified = await registry.calibration_evidence.verify_bridge_result(report["captures"], shadow["evaluation"], workcell_id=WORKCELL_ID,
                                                                      plan=planned["plan"], run_id=run.run_id)
    assert verified == shadow
    if sensors["wrist_camera"].rendered:
        raw = await registry.calibration_evidence.read_capture(report["captures"][0]["evidence_digest"], workcell_id=WORKCELL_ID)
        for role in ("fixed", "wrist"):
            detected = raw["observation"][f"{role}_detection"]
            assert len(detected["corner_ids"]) == 12 and len(detected["marker_ids"]) == 10
            assert detected["pose_gap_px"] > .5
            assert len(detected["image_points"]) == len(detected["board_points_mm"]) == 12
    for defect in ("verdict", "quality_margins_m_rad", "candidate_digest", "scope"):
        bad = copy.deepcopy(shadow["evaluation"])
        bad[defect] = "forged"
        with pytest.raises(ValueError, match="differs from indexed"):
            await registry.calibration_evidence.verify_bridge_result(report["captures"], bad, workcell_id=WORKCELL_ID, plan=planned["plan"], run_id=run.run_id)
    for field, value in (("physical_activation", "active"), ("absolute_base_accuracy_established", True),
                         ("independent_reference_accuracy", {"rms_m": 0}), ("joint_wrist_fit", True),
                         ("schema", "forged"), ("verdict", "qualified"), ("reachability", {})):
        bad = copy.deepcopy(shadow)
        bad["evaluation"][field] = value
        assert shadow_problem(bad)
    bad = copy.deepcopy(shadow)
    bad["evaluation"]["stationary_board"]["physical_stationarity_certified"] = True
    assert shadow_problem(bad)
    # A target_mount field cannot launder this stationary-board result into hand-eye.
    imported = CalibrationRecord(WORKCELL_ID, "camera.handeye", time.time(), parameters={"method": report["method_id"], "bridge_evidence": shadow, "target_mount": IDENTITY_MATRIX}, matrix=IDENTITY_MATRIX)
    assert registry.calibration_store.record(imported)
    spec = runtime._workcells[WORKCELL_ID]
    graph = FrameGraph.snapshot(spec, registry.calibration_store)
    assert graph.transform("camera") is None and "shadow" in graph.stale["camera"]


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("defect", ["mount_digest", "session_id", "profile_digest", "intrinsics", "clock", "repeated"])
async def test_bridge_identity_failure_retains_no_candidate_and_never_writes(bridge_owner, defect, late):
    registry, runtime, server, issuer, sensors, writes, _, _ = bridge_owner
    planned = await bridge_plan(server)
    assert "plan" in planned, planned
    if late:
        sensors["wrist_camera"].late_defect = defect
    else:
        sensors["wrist_camera"].defect = defect
    run = await finished(runtime, await request(server, "skill.execute", plan_digest=planned["plan"]["digest"], permit=permit(issuer, planned), lease_s=15))
    assert not run.outcome.ok and not writes
    assert registry.calibration_store.latest(WORKCELL_ID, "camera.handeye") is None
    assert "bridge_shadow_digest" not in run.outcome.evidence
    if late:
        assert len(run.outcome.evidence["acquisitions"]) == 6
        assert run.outcome.failure_code == "plan_stale"


@pytest.mark.parametrize("retention", [{"quota_bytes": 4 * 1024 * 1024}, {"ttl_s": 1}])
async def test_bridge_retention_preflight_refuses_before_acquisition(bridge_owner, retention):
    registry, runtime, server, issuer, sensors, writes, declaration, path = bridge_owner
    changed = copy.deepcopy(declaration)
    changed["calibration_acquisition"].update(retention)
    write_workcell(path / "workcells", changed)
    from tests.test_workcell_reload import approved_reload
    await approved_reload(server, issuer)
    before = [sensor.calls for sensor in sensors.values()]
    refused = await bridge_plan(server)
    assert "plan" not in refused and "TTL/quota" in refused["outcome"]["message"]
    assert [sensor.calls for sensor in sensors.values()] == before
    assert not registry.calibration_evidence.summaries(workcell_id=WORKCELL_ID) and not writes


async def test_bridge_exclusions_remain_indexed_without_holdout_reallocation(bridge_owner, monkeypatch):
    registry, runtime, server, issuer, sensors, writes, _, _ = bridge_owner
    planned = await bridge_plan(server)
    monkeypatch.setattr("leapflow.hardware.calibration.paired.detect_board", lambda f, t: None)
    run = await finished(runtime, await request(server, "skill.execute", plan_digest=planned["plan"]["digest"],
                                               permit=permit(issuer, planned), lease_s=15))
    assert run.outcome.ok, run.outcome.to_dict()
    result = run.outcome.result
    assert result["verdict"] == "insufficient_evidence" and result["candidate"] is None
    assert len(result["captures"]) == len(result["excluded_observations"]) == 6
    assert [c["allocation"] for c in result["captures"]] == ["training"] * 3 + ["holdout"] * 3
    bad = copy.deepcopy(result["captures"])
    bad[1]["evidence_digest"] = bad[0]["evidence_digest"]
    shadow = await asyncio.to_thread(registry.calibration_generations.candidate, result["shadow_record_digest"], workcell_id=WORKCELL_ID)
    with pytest.raises(ValueError, match="overlap"):
        await registry.calibration_evidence.verify_bridge_result(bad, shadow["evaluation"], workcell_id=WORKCELL_ID,
                                                                  plan=planned["plan"], run_id=run.run_id)
    with pytest.raises(ValueError, match="shortened"):
        await registry.calibration_evidence.verify_bridge_result(result["captures"][:-1], shadow["evaluation"], workcell_id=WORKCELL_ID,
                                                                  plan=planned["plan"], run_id=run.run_id)
    assert not writes


@pytest.mark.parametrize("defect", ["units", "fit", "verification_cap", "points", "source_overlap", "source_string", "source_element", "verification_role"])
async def test_wrist_validation_cannot_be_replaced_by_a_large_bridge_budget(bridge_owner, defect):
    registry, runtime, server, issuer, sensors, writes, declaration, path = bridge_owner
    wrist = registry.calibration_store.latest(WORKCELL_ID, "camera.wrist.handeye")
    verification = registry.calibration_store.latest(WORKCELL_ID, "camera.wrist.verify")
    values, checked = copy.deepcopy(wrist.parameters), copy.deepcopy(verification.parameters)
    if defect == "units":
        values["units"] = "m"
    elif defect == "fit":
        values["fit"]["translation_rms_mm"] = 20
    elif defect == "verification_cap":
        checked["max_p95_mm"] = 100
    elif defect == "points":
        checked["points"] = True
    elif defect == "source_overlap":
        values["source_ids"] = checked["source_ids"] = ["same-refinement-frame"]
    elif defect == "source_string":
        values["source_ids"] = "untyped-frame-lineage"
    elif defect == "source_element":
        values["source_ids"] = [{"untyped": "frame"}]
    else:
        checked["camera_role"] = "camera"
    wrist = replace(wrist, recorded_at=wrist.recorded_at + 1, parameters=values)
    checked["depends_on"] = {"camera.wrist": wrist.recorded_at}
    verification = replace(verification, recorded_at=verification.recorded_at + 1, parameters=checked)
    assert registry.calibration_store.record(wrist) and registry.calibration_store.record(verification)
    changed = copy.deepcopy(declaration)
    changed["calibration_bridge"].update(wrist_record_digest=digest_of(vars(wrist)), wrist_verification_digest=digest_of(vars(verification)))
    write_workcell(path / "workcells", changed)
    from tests.test_workcell_reload import approved_reload
    await approved_reload(server, issuer)
    refused = await bridge_plan(server)
    assert "plan" not in refused and "wrist dependency unresolved" in refused["outcome"]["message"]
    assert not writes and not registry.calibration_evidence.summaries(workcell_id=WORKCELL_ID)


async def test_bridge_discovery_from_declared_entrypoint_reload_and_dispose(bridge_owner, monkeypatch):
    import importlib.metadata
    import tomllib
    from leapflow.hardware.skills.runtime import discover_skills
    registry, runtime, server, issuer, _, writes, declaration, path = bridge_owner
    manifest = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    entries = manifest["project"]["entry-points"]["leapflow.hardware.skills"]
    entry = importlib.metadata.EntryPoint(name="builtin_bridge_calibration", value=entries["builtin_bridge_calibration"], group="leapflow.hardware.skills")
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **kwargs: (entry,))
    assert any(s.spec.skill_id == "calib.bridge" for s in discover_skills())
    planned = await bridge_plan(server)
    altered = copy.deepcopy(declaration)
    altered["calibration_bridge"]["stationary_epoch"] = "operator-moved-sheet"
    write_workcell(path / "workcells", altered)
    from tests.test_workcell_reload import approved_reload
    applied = await approved_reload(server, issuer)
    assert applied["invalidated_plans"] == 1 and runtime.cached_plan(planned["plan"]["digest"]) is None
    assert not runtime.load_report.rejected
    assert "plan" in await bridge_plan(server)
    await runtime.close()
    assert not runtime.holds_device(ARM_ID) and not writes
