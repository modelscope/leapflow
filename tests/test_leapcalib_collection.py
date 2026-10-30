# Copyright (c) Alibaba, Inc. and its affiliates.
"""Actual owner/skill/LHP/private-store construction; only sensor and disk I/O is simulated.

These stationary software fixtures are not physical qualification or hardware trials.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from leapcalib.contracts.tracking import TrackingConfig
from leapflow.cache.manager import CacheManager
from leapflow.hardware.calibration.acquisition import CollectionProfile
from leapflow.hardware.calibration.evidence import CATEGORY, AcquisitionBinding
from leapflow.hardware.calibration.reference import (
    ReferenceProfile,
    assess_reference,
    validate_tracking_result,
)
from leapflow.hardware.calibration.worker import CalibrationWorker
from leapflow.hardware.context import Channel, ContextProvenance, HardwareContext, TransportRef
from leapflow.hardware.lhp.permit import PermitIssuer, PermitVerifier, generate_keypair
from leapflow.hardware.lhp.server import LhpNodeServer
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.rgbd import CameraIntrinsics
from leapflow.hardware.skills.builtin.governed_calibration import skills
from leapflow.hardware.skills.builtin.motion import skills as motion_skills
from leapflow.hardware.skills.runtime import SkillRuntime, SkillRuntimeSettings
from leapflow.hardware.transport import TransportError
from leapflow.layout import CacheLayout
from tests._harness.skill_stubs import (
    ARM_ID,
    JOINTS,
    WORKCELL_ID,
    WORKCELL_YAML,
    piper_context,
    write_workcell,
)
from tests.test_leapcalib_acquisition import Provider, frame, profile
from tests.test_leapcalib_tracking import pair

pytestmark = pytest.mark.component


class Sensor:
    """Boundary-only metric stream with explicit increasing native provenance."""

    def __init__(self) -> None:
        self.calls = 0
        self.defect = ""
        self.textured = False

    async def capture_rgbd(self, *, after_monotonic: float | None = None):
        self.calls += 1
        value = frame(time.monotonic())
        if self.textured:
            import numpy as np

            first, second = pair()
            color = first.color_bgr8
            if self.calls >= 3 and self.defect == "occluded":
                color = bytes(len(color))
            if self.calls >= 3 and self.defect == "scene-motion":
                color = np.roll(
                    np.frombuffer(second.color_bgr8, dtype=np.uint8).reshape(
                        second.height, second.width, 3
                    ),
                    8,
                    axis=1,
                ).tobytes()
            value = replace(
                value,
                width=first.width,
                height=first.height,
                color=color,
                depth=np.full((first.height, first.width), 1000, dtype=np.uint16).tobytes(),
                intrinsics=CameraIntrinsics(
                    first.width, first.height, 500, 500, first.width / 2, first.height / 2
                ),
            )
        number = 1 if self.defect == "repeated" else self.calls
        metadata = {
            **value.metadata,
            "color_source": {
                "frame_number": number,
                "timestamp_ms": number * 33.0,
                "timestamp_domain": "hardware_clock",
            },
            "depth_source": {
                "frame_number": number,
                "timestamp_ms": number * 33.0 + 0.1,
                "timestamp_domain": "hardware_clock",
            },
        }
        if self.defect in ("mount_digest", "session_id", "profile_digest"):
            metadata[self.defect] = "changed"
        if self.defect == "clock":
            metadata["depth_source"]["timestamp_domain"] = "changed-clock"
        intrinsics = (
            replace(value.intrinsics, fx=value.intrinsics.fx + 1)
            if self.defect == "intrinsics"
            else value.intrinsics
        )
        return replace(value, sequence=number, metadata=metadata, intrinsics=intrinsics)


@pytest.fixture
async def owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    values = {
        **dict.fromkeys(JOINTS, 0.0),
        **dict(zip(("ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz"), (250, 0, 370, 180, 0, 0))),
        "arm.enabled": True,
        "arm.motion": True,
        "arm.ctrl_mode": "can",
        "arm.status": {"arm_status": 0, "joint_communication_errors": []},
    }
    arm = replace(
        piper_context(),
        transport=TransportRef("mock", {"values": values}),
        provenance=ContextProvenance(verified_by="isolated-software-fixture"),
    )
    camera = HardwareContext(
        device_id="camera",
        channels=(Channel("frame", direction="read"),),
        transport=TransportRef("mock", {}),
        provenance=ContextProvenance(verified_by="software-test"),
    )
    registry = HardwareRegistry(
        HardwareSettings(instrument_db_path=str(tmp_path / "instrument.duckdb")),
        providers=[Provider(arm), Provider(camera)],
    )
    registry.load()
    cache = CacheManager(CacheLayout(tmp_path / "cache"), profile_id="isolated-software")
    registry.bind_persistence(cache_manager=cache)
    worker = CalibrationWorker(tmp_path.parent / ("lc-" + tmp_path.name[-8:]) / "s", enabled=False)
    registry.bind_calibration_worker(worker)
    sensor = Sensor()
    camera_transport = await registry.transport("camera")
    monkeypatch.setattr(camera_transport, "capture_rgbd", sensor.capture_rgbd, raising=False)
    acquisition = CollectionProfile(
        "camera", "fixed", ("can",), profile(timeout_s=2.0), 60, 4 * 1024 * 1024, 4
    )
    declaration = {
        **WORKCELL_YAML,
        "roles": {"arm": ARM_ID, "camera": "camera"},
        "frames": {
            **WORKCELL_YAML["frames"],
            "camera": {"parent": "base", "procedure": "camera.handeye"},
        },
        "calibration_acquisition": acquisition.to_dict(),
        "calibration_reference": ReferenceProfile(
            TrackingConfig(), (0.003, 0.003, 0.003, math.radians(2)), "isolated-stationary", 4, 0.3
        ).to_dict(),
    }
    write_workcell(tmp_path / "workcells", declaration)
    runtime = SkillRuntime(
        registry,
        settings=SkillRuntimeSettings(workcells_dir=tmp_path / "workcells", stop_grace_s=0.01),
        skills=(*skills(), *motion_skills()),
    )
    assert not runtime.load_report.rejected
    private, public = generate_keypair()
    issuer = PermitIssuer(private)
    verifier = PermitVerifier({issuer.key_id: public}, node_id="node")
    server = LhpNodeServer(
        registry, node_id="node", skill_runtime=runtime, permit_verifier=verifier
    )
    try:
        yield registry, cache, sensor, runtime, server, issuer
    finally:
        await runtime.close()
        await server.sessions.shutdown()
        await worker.close()
        await registry.close_all()


async def request(server: LhpNodeServer, method: str, **params: Any) -> dict[str, Any]:
    reply = await server.handle_request(
        {"id": "software-request", "method": method, "params": params}
    )
    assert "error" not in reply, reply
    return reply["result"]


async def planned(server: LhpNodeServer, frames: int = 1) -> dict[str, Any]:
    result = await request(
        server,
        "skill.plan",
        workcell=WORKCELL_ID,
        skill="calib.collect",
        args={"region": [2, 1, 8, 8], "frames": frames},
    )
    assert "plan" in result, result
    return result


def permit(issuer: PermitIssuer, result: dict[str, Any]) -> dict[str, Any]:
    return issuer.issue(
        audience="node",
        session_id="software-session",
        workspace="software-workspace",
        approval_id="boundary-human-fixture",
        digest=result["permit_digest"],
    ).to_dict()


async def finished(runtime: SkillRuntime, result: dict[str, Any]):
    assert "run" in result, result
    run = await runtime.wait(result["run"]["run_id"], timeout_s=15)
    assert run is not None and not run.running
    assert not runtime.holds_device(ARM_ID)
    return run


async def test_stationary_production_path_persists_exact_run_lineage_and_redacts_inspection(
    owner,
) -> None:
    registry, cache, sensor, runtime, server, issuer = owner
    plan = await planned(server, frames=2)
    assert plan["consent"] == "each_run"
    assert cache.list_entries(category=CATEGORY) == []
    assert registry.calibration_evidence.summaries(workcell_id=WORKCELL_ID) == ()
    proof = permit(issuer, plan)
    run = await finished(
        runtime,
        await request(
            server, "skill.execute", plan_digest=plan["plan"]["digest"], permit=proof, lease_s=10
        ),
    )
    assert run.outcome.ok and run.outcome.side_effect_state == "committed"
    captures = run.outcome.result["captures"]
    assert len(captures) == 2 and sensor.calls == 3
    for capture in captures:
        content = await asyncio.to_thread(
            registry.calibration_evidence.read, capture["evidence_digest"], workcell_id=WORKCELL_ID
        )
        binding = content["binding"]
        assert binding["plan"] == plan["plan"] and binding["run_id"] == run.run_id
        assert binding["admission"]["permit"] == proof
        assert binding["node_session_id"] == plan["plan"]["basis"]["collection"]["node_session_id"]
        assert not capture["hardware_accuracy_certified"]
    inspected = await request(
        server,
        "skill.plan",
        workcell=WORKCELL_ID,
        skill="calib.inspect",
        args={"evidence_digest": captures[0]["evidence_digest"]},
    )
    assert (
        inspected["outcome"]["status"] == "succeeded"
        and inspected["outcome"]["side_effect_state"] == "none"
    )
    assert (
        sensor.calls == 3
        and '"permit":' not in json.dumps(inspected)
        and "color_bgr8" not in json.dumps(inspected)
    )
    transport = await registry.transport(ARM_ID)
    assert transport.write_log == () and transport.halt_calls == 0
    state = await request(server, "skill.plan", workcell=WORKCELL_ID, skill="arm.state", args={})
    assert state["outcome"]["status"] == "succeeded"


async def test_inspection_of_absent_evidence_is_read_only(owner) -> None:
    registry, _, sensor, _, server, _ = owner
    holder = registry.calibration_evidence._holder
    before = holder.connection.execute(
        "SELECT table_name FROM information_schema.tables ORDER BY table_name"
    ).fetchall()
    for args in ({}, {"evidence_digest": "absent"}):
        result = await request(
            server, "skill.plan", workcell=WORKCELL_ID, skill="calib.inspect", args=args
        )
        assert result["outcome"]["side_effect_state"] == "none"
    assert (
        holder.connection.execute(
            "SELECT table_name FROM information_schema.tables ORDER BY table_name"
        ).fetchall()
        == before
    )
    assert sensor.calls == 0
    with pytest.raises(TransportError) as caught:
        registry.get_open_transport(ARM_ID)
    assert caught.value.failure_code == "transport_not_open"


@pytest.mark.parametrize(
    "defect", ["intrinsics", "repeated", "clock", "mount_digest", "session_id", "profile_digest"]
)
async def test_changed_or_replayed_metric_input_refuses_before_persistence(
    owner, defect: str
) -> None:
    registry, cache, sensor, runtime, server, issuer = owner
    plan = await planned(server)
    sensor.defect = defect
    run = await finished(
        runtime,
        await request(
            server,
            "skill.execute",
            plan_digest=plan["plan"]["digest"],
            permit=permit(issuer, plan),
            lease_s=10,
        ),
    )
    assert not run.outcome.ok and run.outcome.failure_code == "check_failed", run.outcome
    assert run.outcome.side_effect_state == "none"
    assert not cache.list_entries(category=CATEGORY)
    assert registry.calibration_evidence.summaries(workcell_id=WORKCELL_ID) == ()
    transport = await registry.transport(ARM_ID)
    assert transport.write_log == () and transport.halt_calls == 0


async def test_busy_before_permit_consumption_then_same_exact_permit_succeeds(owner) -> None:
    registry, _, _, runtime, server, issuer = owner
    plan = await planned(server)
    proof = permit(issuer, plan)
    token = server.sessions.reserve((ARM_ID,), owner_id="isolated-owner-test")
    try:
        refused = await request(
            server, "skill.execute", plan_digest=plan["plan"]["digest"], permit=proof, lease_s=10
        )
        assert refused["outcome"]["failure_code"] == "device_busy"
    finally:
        server.sessions.release_reservation(token)
    run = await finished(
        runtime,
        await request(
            server, "skill.execute", plan_digest=plan["plan"]["digest"], permit=proof, lease_s=10
        ),
    )
    assert run.outcome.ok
    assert (await registry.transport(ARM_ID)).write_log == ()


async def test_direct_runtime_without_owner_admission_cannot_persist(owner) -> None:
    registry, cache, _, runtime, server, _ = owner
    plan = await planned(server)
    run = await finished(
        runtime, {"run": (await runtime.execute(plan["plan"]["digest"])).to_dict()}
    )
    assert run.outcome.failure_code == "write_refused" and run.outcome.side_effect_state == "none"
    assert cache.list_entries(category=CATEGORY) == []
    assert registry.calibration_evidence.summaries(workcell_id=WORKCELL_ID) == ()


async def test_repeated_cancel_shutdown_and_run_holds_wait_for_real_file_settlement(
    owner, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, _, _, runtime, server, issuer = owner
    plan = await planned(server)
    entered, release = threading.Event(), threading.Event()
    fsync = os.fsync

    def delayed(fd: int) -> None:
        entered.set()
        assert release.wait(5)
        fsync(fd)

    monkeypatch.setattr("leapflow.hardware.calibration.evidence.os.fsync", delayed)
    started = await request(
        server,
        "skill.execute",
        plan_digest=plan["plan"]["digest"],
        permit=permit(issuer, plan),
        lease_s=10,
    )
    run = runtime.status(started["run"]["run_id"])[0]
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        await runtime.stop(run.run_id, "software cancellation")
        run.task.cancel()
        await asyncio.sleep(0)
        assert run.running and runtime.holds_device(ARM_ID)
        assert server.sessions.busy_device((ARM_ID,)) == ARM_ID
        closing = asyncio.create_task(runtime.close())
        await asyncio.sleep(0.01)
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        assert not closing.done() and run.running
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert not run.running and not runtime.holds_device(ARM_ID)
        assert run.outcome.failure_code == "stopped" and run.outcome.side_effect_state == "partial"
        assert len(run.outcome.evidence["acquisitions"]) == 1
        assert len(registry.calibration_evidence.summaries(workcell_id=WORKCELL_ID)) == 1
        assert (await registry.transport(ARM_ID)).halt_calls == 0
    finally:
        release.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)


async def reference_plan(owner):
    registry, _, sensor, _, server, _ = owner
    registry.calibration_worker.enabled = True  # Explicit isolated-software worker opt-in.
    sensor.textured = True
    result = await request(
        server,
        "skill.plan",
        workcell=WORKCELL_ID,
        skill="calib.align_reference",
        args={"region": [0, 0, 160, 120], "frames": 3},
    )
    assert "plan" in result, result
    return result


async def test_reference_through_actual_worker_and_exact_owner_lineage(owner) -> None:
    registry, _, sensor, runtime, server, issuer = owner
    plan = await reference_plan(owner)
    assert plan["consent"] == "each_run" and plan["plan"]["targets"] == []
    assert registry.calibration_worker._process is None
    proof = permit(issuer, plan)
    run = await finished(
        runtime,
        await request(
            server, "skill.execute", plan_digest=plan["plan"]["digest"], permit=proof, lease_s=30
        ),
    )
    assert run.outcome.ok, run.outcome
    summary = run.outcome.result
    assert summary["state"] == "REGISTERED" and summary["frames"] == 3 and summary["span_s"] >= 0.3
    assert not summary["independent_witness"] and not summary["clearance_verified"]
    assert (
        summary["physical_activation"] == "unqualified"
        and not summary["hardware_accuracy_certified"]
    )
    content = await asyncio.to_thread(
        registry.calibration_evidence.read_reference,
        summary["reference_digest"],
        workcell_id=WORKCELL_ID,
    )
    assert (
        content["binding"]["admission"]["permit"] == proof
        and content["binding"]["run_id"] == run.run_id
    )
    assert content["binding"]["plan"] == plan["plan"]
    assert len(content["assessment"]["persistent_tracks"]) >= 12 and len(content["results"]) == 2
    assert sensor.calls == 4 and len(run.outcome.evidence["acquisitions"]) == 3
    inspected = await request(
        server, "skill.plan", workcell=WORKCELL_ID, skill="calib.inspect", args={}
    )
    assert tuple(inspected["outcome"]["result"]["references"]) == (summary,)
    encoded = json.dumps(inspected)
    assert all(
        key not in encoded for key in ('"permit":', '"pixel":', "color_bgr8", "raw_joints_rad")
    )
    assert inspected["outcome"]["side_effect_state"] == "none" and sensor.calls == 4
    assert (
        registry.calibration_worker.last_metrics["peak_rss_bytes"]
        <= registry.calibration_worker.budgets.rss_bytes
    )
    transport = await registry.transport(ARM_ID)
    assert not transport.write_log and not transport.halt_calls
    with pytest.raises(ValueError, match="this workcell"):
        registry.calibration_evidence.read_reference(
            summary["reference_digest"], workcell_id="other"
        )


async def test_disabled_reference_worker_refuses_before_capture(owner) -> None:
    registry, _, sensor, _, server, _ = owner
    result = await request(
        server,
        "skill.plan",
        workcell=WORKCELL_ID,
        skill="calib.align_reference",
        args={"region": [0, 0, 160, 120]},
    )
    assert result["outcome"]["failure_code"] == "skill_unavailable"
    assert not sensor.calls and registry.calibration_worker._process is None


@pytest.mark.parametrize("defect", ["occluded", "scene-motion"])
async def test_reference_tracker_loss_or_scene_motion_never_registers_or_retries(
    owner, defect: str
) -> None:
    registry, _, sensor, runtime, server, issuer = owner
    plan = await reference_plan(owner)
    sensor.defect = defect
    run = await finished(
        runtime,
        await request(
            server,
            "skill.execute",
            plan_digest=plan["plan"]["digest"],
            permit=permit(issuer, plan),
            lease_s=30,
        ),
    )
    assert not run.outcome.ok and run.outcome.failure_code == "check_failed", run.outcome
    assert run.outcome.side_effect_state == "partial" and sensor.calls == 3
    assert not registry.calibration_evidence.reference_summaries(workcell_id=WORKCELL_ID)
    assert len(registry.calibration_evidence.summaries(workcell_id=WORKCELL_ID)) == 2
    assert not (await registry.transport(ARM_ID)).write_log


async def test_reference_expiry_preserves_summary_but_blocks_consumption(
    owner, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, _, _, runtime, server, issuer = owner
    plan = await reference_plan(owner)
    run = await finished(
        runtime,
        await request(
            server,
            "skill.execute",
            plan_digest=plan["plan"]["digest"],
            permit=permit(issuer, plan),
            lease_s=30,
        ),
    )
    assert run.outcome.ok, run.outcome
    summary = run.outcome.result
    future = time.time() + 3600
    monkeypatch.setattr("leapflow.hardware.calibration.evidence.time.time", lambda: future)
    assert registry.calibration_evidence.reference_summaries(workcell_id=WORKCELL_ID) == (summary,)
    with pytest.raises(ValueError, match="expired"):
        registry.calibration_evidence.read_reference(
            summary["reference_digest"], workcell_id=WORKCELL_ID
        )


async def test_reference_owner_rechecks_worker_values_whole_window_and_run_binding(owner) -> None:
    from leapflow.hardware.calibration.acquisition import SettledObservation, SettledProfile
    from leapflow.hardware.skills.admission import RunAdmission

    registry, _, _, runtime, server, issuer = owner
    plan = await reference_plan(owner)
    run = await finished(
        runtime,
        await request(
            server,
            "skill.execute",
            plan_digest=plan["plan"]["digest"],
            permit=permit(issuer, plan),
            lease_s=30,
        ),
    )
    assert run.outcome.ok, run.outcome
    summary, store = run.outcome.result, registry.calibration_evidence
    content = await asyncio.to_thread(
        store.read_reference, summary["reference_digest"], workcell_id=WORKCELL_ID
    )
    captures = [
        await asyncio.to_thread(store.read, identity, workcell_id=WORKCELL_ID)
        for identity in content["acquisition_digests"]
    ]
    observations = [SettledObservation.from_mapping(c["observation"]) for c in captures]
    reference = ReferenceProfile.from_mapping(content["assessment"]["profile"])
    settled = SettledProfile(**captures[0]["profile"])
    tracking = content["results"][0]
    mutations = [
        {"first_digest": "wrong"},
        {"witness_valid": True},
        {"executed": True},
        {"extra": "unexpected"},
        {"uncertainty_m_rad": [0, 0, 0, 0]},
        {"uncertainty_m_rad": [float("nan"), 1, 1, 1]},
        {"registration": {}},
        {"delta_m_rad": [0.001, 0, 0, 0]},
        {"persistent_tracks": []},
        {"timing_uncertainty_s": 0},
        {"rejected_features": [tracking["tracks"][0]["feature_id"]]},
        {"rejected_features": [None]},
        {"uncertainty_m_rad": [0.0005, 0.0005, 0.0005, 1e-12]},
    ]
    for changes in mutations:
        with pytest.raises((TypeError, ValueError)):
            validate_tracking_result(
                {**tracking, **changes}, observations[0], observations[1], reference, None
            )
    malformed = json.loads(json.dumps(tracking))
    malformed["registration"]["motion"]["rotation"][0][0] = True
    with pytest.raises(ValueError):
        validate_tracking_result(malformed, observations[0], observations[1], reference, None)
    invented = json.loads(json.dumps(tracking))
    for row in invented["tracks"]:
        row["first_m"][0] += 0.001
        row["second_m"][0] += 0.001
    with pytest.raises(ValueError, match="indexed raw depth"):
        validate_tracking_result(invented, observations[0], observations[1], reference, None)
    second_result = json.loads(json.dumps(content["results"][1]))
    ids = {row["feature_id"] for row in tracking["tracks"]}
    for index, source in ((0, 1), (1, 0)):
        for key in ("first_pixel", "second_pixel", "first_m", "second_m"):
            second_result["tracks"][index][key] = content["results"][1]["tracks"][source][key]
        second_result["persistent_tracks"][index]["pixel"] = second_result["tracks"][index][
            "second_pixel"
        ]
    with pytest.raises(ValueError, match="switched"):
        validate_tracking_result(
            second_result,
            observations[1],
            observations[2],
            reference,
            ids,
            previous_tracks=tracking["persistent_tracks"],
        )
    drifted = replace(
        observations[-1],
        feedback=tuple(
            replace(sample, joints_rad=tuple(v + 0.0009 for v in sample.joints_rad))
            for sample in observations[-1].feedback
        ),
    )
    with pytest.raises(ValueError, match="starting raw posture"):
        assess_reference(
            [*observations[:-1], drifted], content["results"], profile=reference, settled=settled
        )
    receipt = content["binding"]["admission"]
    binding = AcquisitionBinding(
        WORKCELL_ID,
        run.run_id,
        content["binding"]["node_session_id"],
        json.dumps(plan["plan"]),
        RunAdmission(receipt["mode"], receipt["action_digest"], json.dumps(receipt["permit"])),
    )
    repeated = await asyncio.to_thread(
        store.record_reference,
        content["acquisition_digests"],
        content["results"],
        profile=reference,
        binding=binding,
    )
    assert repeated == summary and store.reference_summaries(workcell_id=WORKCELL_ID) == (summary,)
    with pytest.raises(ValueError, match="different admitted run"):
        await asyncio.to_thread(
            store.record_reference,
            content["acquisition_digests"],
            content["results"],
            profile=reference,
            binding=replace(binding, run_id="another-run"),
        )


async def test_reference_worker_budget_change_stales_plan_before_capture(owner) -> None:
    registry, _, sensor, runtime, server, issuer = owner
    plan = await reference_plan(owner)
    worker = registry.calibration_worker
    worker.budgets = replace(worker.budgets, update_s=1.5)
    run = await finished(
        runtime,
        await request(
            server,
            "skill.execute",
            plan_digest=plan["plan"]["digest"],
            permit=permit(issuer, plan),
            lease_s=30,
        ),
    )
    assert run.outcome.failure_code == "plan_stale" and run.outcome.side_effect_state == "none"
    assert sensor.calls == 1 and worker._process is None


@pytest.mark.parametrize("uncertain_commit", [False, True])
async def test_reference_cancel_retains_holds_until_real_db_dispatch_settles(
    owner, monkeypatch: pytest.MonkeyPatch, uncertain_commit: bool
) -> None:
    import duckdb

    registry, _, _, runtime, server, issuer = owner
    plan = await reference_plan(owner)
    entered, release = threading.Event(), threading.Event()
    execute = duckdb.DuckDBPyConnection.execute

    def delayed(connection, query, *args, **kwargs):
        if query.startswith("INSERT INTO calib_references "):
            entered.set()
            assert release.wait(10)
            result = execute(connection, query, *args, **kwargs)
            if uncertain_commit:
                raise OSError("isolated synthetic loss of DB commit acknowledgement")
            return result
        return execute(connection, query, *args, **kwargs)

    # Only the actual DuckDB I/O boundary is delayed; owner/runtime/store remain real.
    monkeypatch.setattr(duckdb.DuckDBPyConnection, "execute", delayed)
    started = await request(
        server,
        "skill.execute",
        plan_digest=plan["plan"]["digest"],
        permit=permit(issuer, plan),
        lease_s=30,
    )
    run = runtime.status(started["run"]["run_id"])[0]
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        await runtime.stop(run.run_id, "isolated reference cancellation")
        run.task.cancel()
        await asyncio.sleep(0.01)
        run.task.cancel()
        await asyncio.sleep(0.01)
        assert run.running and runtime.holds_device(ARM_ID)
        assert server.sessions.busy_device((ARM_ID,)) == ARM_ID
        release.set()
        await run.task
        assert not runtime.holds_device(ARM_ID)
        if uncertain_commit:
            assert (
                run.outcome.side_effect_state == "unknown"
                and run.outcome.failure_code == "device_fault"
            )
            assert not run.outcome.evidence.get("references")
        else:
            assert (
                run.outcome.failure_code == "stopped" and run.outcome.side_effect_state == "partial"
            )
            assert len(run.outcome.evidence["references"]) == 1
        # Durable decision survives a lost acknowledgement; inspection never retries it.
        assert len(registry.calibration_evidence.reference_summaries(workcell_id=WORKCELL_ID)) == 1
        transport = await registry.transport(ARM_ID)
        assert not transport.write_log and not transport.halt_calls
    finally:
        release.set()
