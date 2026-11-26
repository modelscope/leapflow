"""Real qualification/store/runtime wiring over isolated synthetic sensor I/O.

Certificates here are test inputs only; no physical qualification is established.
"""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from leapflow.hardware.calibration_qualification import (
    CalibrationQualification,
    QualificationLimits,
)
from leapflow.hardware.context import HardwareContext
from leapflow.hardware.grasp_measurement import digest
from leapflow.hardware.lhp.server import WRITE_MODE_PERMIT_REQUIRED
from leapflow.hardware.skills.builtin.grasp_verify import (
    GraspEvidence,
    GraspInspect,
    GraspVerify,
    InstallGraspEvidence,
)
from leapflow.hardware.skills.frames import FrameGraph
from leapflow.hardware.skills.geometry import Pose
from leapflow.hardware.skills.grasp_verification import PROCEDURE, WorkcellGraspVerification
from leapflow.hardware.skills.protocol import SkillError
from leapflow.hardware.transport import BatchReading, Reading
from leapflow.hardware.attestation import declaration_digest
from tests._harness.skill_stubs import (
    ARM_ID,
    CAMERA_ID,
    IDENTITY_MATRIX,
    SAFE_BOX,
    WORKCELL_ID,
    WORKCELL_YAML,
    SimRgbdCamera,
)
from tests.test_grasp_measurement import geometry, qualification_data
from tests.test_skill_tools import QUALIFIED, rigs as skill_rigs

rigs = skill_rigs
WRIST = "sensor_realsense_d405"


class MeasuredCamera:
    """Native-bundle-shaped camera boundary; the production verifier is untouched."""

    def __init__(self) -> None:
        self.qualification: Any = None
        self.frame = 0
        self.fault = ""

    async def read_batch(self, channels: tuple[str, ...]) -> BatchReading:
        self.frame += 1
        frame = self.frame if self.fault != "frozen" else 1
        q = self.qualification
        now = time.monotonic()
        data = geometry(q, frame=frame)
        data["configuration_digest"] = q.data["geometry_configuration_digest"]
        data["observed_mono"] = now
        data["depth"]["observation_times"] = [now]
        for depth in data["pad_depth"]:
            depth["observation_times"] = [now]
        if self.fault == "session":
            data["session_id"] = "new-session"
        elif self.fault == "domain":
            data["source"]["depth_source"]["timestamp_domain"] = "other"
        elif self.fault == "scope":
            data["configuration_digest"] = "foreign"
        value = data
        if channels == ("camera.rgbd.quality",):
            value = {**data["depth"], "configuration_digest": q.data["depth_configuration_digest"]}
        return BatchReading(
            WRIST, tuple(Reading(WRIST, channel, value, monotonic_at=now) for channel in channels)
        )


@pytest.fixture
async def admission_rig(tmp_path: Path, rigs: Any, monkeypatch: Any) -> Any:
    cell = copy.deepcopy(WORKCELL_YAML)
    cell["roles"].update({"camera": CAMERA_ID, "camera.wrist": WRIST})
    cell["frames"].update(
        {
            "camera": {"parent": "base", "procedure": "camera.handeye"},
            "camera.wrist": {"parent": "flange", "procedure": "wrist.handeye"},
        }
    )
    camera = MeasuredCamera()
    rig = rigs(
        tmp_path,
        write_mode=WRITE_MODE_PERMIT_REQUIRED,
        cell=cell,
        node_skills=(InstallGraspEvidence(), GraspVerify(), GraspInspect(), GraspEvidence()),
        devices={CAMERA_ID: SimRgbdCamera(), WRIST: camera},
        gripper_mm=20,
    )
    registry = rig.nodes[0].registry
    declaration = yaml.safe_load(
        (
            Path(__file__).parents[1] / "src/leapflow/hardware/examples/realsense_d405.yaml"
        ).read_text()
    )
    declaration["transport"]["config"].update(
        {
            "mount_digest": "test-mount",
            "geometry_measurement": {"scene_id": "scene", "fixture": "synthetic-test"},
            "measurement": {"roi": [1, 1, 10, 10], "fixture": "synthetic-test"},
        }
    )
    overview_declaration = copy.deepcopy(declaration)
    overview_declaration["device_id"] = CAMERA_ID
    declarations = {
        WRIST: HardwareContext.from_mapping(declaration),
        CAMERA_ID: HardwareContext.from_mapping(overview_declaration),
    }
    original_context = registry.context
    monkeypatch.setattr(
        registry, "context", lambda device: declarations.get(device) or original_context(device)
    )
    registry.calibrate(
        "camera.handeye",
        pose=Pose(0, 0, 0).to_dict(),
        parameters={"target_mount": IDENTITY_MATRIX},
    )
    registry.calibrate(
        "wrist.handeye",
        pose=Pose(0, 0, 120).to_dict(),
        parameters={"target_mount": IDENTITY_MATRIX},
    )
    registry.calibrate(
        "workspace.safe_box",
        parameters={**SAFE_BOX, "floor": {"method": "table_z", "table_z": 140}},
    )
    spec = rig.runtime._workcells[WORKCELL_ID]
    frames = FrameGraph.snapshot(spec, registry.calibration_store)
    data = qualification_data()
    data.update(
        {
            "camera_device": WRIST,
            "arm_declaration_digest": declaration_digest(registry.context(ARM_ID)),
            "geometry_configuration_digest": digest(
                declarations[WRIST].transport.config["geometry_measurement"]
            ),
            "depth_configuration_digest": digest(
                declarations[WRIST].transport.config["measurement"]
            ),
            "depends_on": dict(frames.versions),
        }
    )
    for row in data["calibration"]["components"]:
        component = row["qualification"]["component"]
        frame = {
            "camera_to_base": "camera.wrist",
            "flange_tcp": "tcp",
            "table_plane": "workspace",
        }.get(component)
        if frame:
            row["geometry"] = dict(frames.parameters(frame))
            transform = frames.transform(frame)
            if transform is not None:
                row["matrix"] = transform.to_matrix()
        fields = dict(row["qualification"])
        fields["limits"] = QualificationLimits(**fields["limits"])
        row["qualification"] = asdict(
            CalibrationQualification(**fields).bind_geometry(row["geometry"], row.get("matrix"))
        )
    for row in (*data["effort"]["training"], *data["effort"]["holdout"]):
        row["flange"] = rig.sim.flange.to_dict()
    original_read = rig.sim.read_batch
    faults = {"feedback": ""}

    async def read(channels: tuple[str, ...]) -> BatchReading:
        batch = await original_read(channels)
        rows = []
        for row in batch.readings:
            if row.channel_id == "gripper.position":
                row = replace(row, unit="mm")
            elif row.channel_id == "gripper.effort":
                row = replace(row, value=1, unit="N")
            if faults["feedback"] == "unit" and row.channel_id == "gripper.effort":
                row = replace(row, unit="A")
            elif faults["feedback"] == "stale" and row.channel_id == "gripper.effort":
                row = replace(row, monotonic_at=1)
            rows.append(row)
        return replace(batch, readings=tuple(rows))

    monkeypatch.setattr(rig.sim, "read_batch", read)
    return SimpleNamespace(
        rig=rig,
        registry=registry,
        spec=spec,
        frames=frames,
        data=data,
        camera=camera,
        declarations=declarations,
        faults=faults,
    )


def real_context(setup: Any) -> Any:
    runtime = setup.rig.runtime
    return runtime._context(
        setup.spec,
        setup.frames,
        runtime._effective_limits(setup.spec, {}),
        asyncio.Event(),
        "grasp.verify",
        writes=False,
    )


async def install(setup: Any) -> dict[str, Any]:
    request = {
        "workcell": QUALIFIED,
        "skill": "calib.grasp_evidence",
        "args": {"qualification": setup.data},
    }
    preview = await setup.rig.service.hw_skill_run(**request, dry_run=True)
    assert preview["ok"], str(preview)
    assert setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE) is None
    result = await setup.rig.service.hw_skill_run(**request, approved_plan=preview["plan"])
    assert result["ok"] and result["side_effect_state"] == "committed", result
    setup.camera.qualification = real_context(setup).grasp_verification._qualification()
    return result


async def test_real_install_and_readonly_runtime_verification(admission_rig: Any) -> None:
    setup = admission_rig
    result = await install(setup)
    stored = setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE)
    assert digest(
        {k: v for k, v in stored.parameters.items() if k != "installation_scope"}
    ) == digest(setup.data)
    assert stored.parameters["installation_scope"]["workcell"] == WORKCELL_ID
    assert result["result"]["qualification_digest"] == digest(stored.parameters)
    assert len(setup.rig.gate.seen) == 1
    assert len([r for r in setup.rig.audit.records if r["action"] == "skill_permit"]) == 1
    verified = await setup.rig.service.hw_skill_run(
        workcell=QUALIFIED, skill="grasp.verify", args={"stage": "hold", "samples": 3}
    )
    assert verified["ok"] and verified["side_effect_state"] == "none", verified
    assert verified["result"]["state"] == "contact_candidate"
    assert len(verified["result"]["observations"]) == 3
    feedback = verified["operation_verification"]
    assert feedback["source"] == "node_grasp_verifier"
    (record,) = feedback["records"]
    assert record["state"] == "contact_candidate" and record["passed"] is True
    assert record["qualification_digest"] == digest(stored.parameters)
    assert (
        record["source_identity"]["session_id"]
        == verified["result"]["source_identity"]["session_id"]
    )
    assert record["observation_count"] == 3 and len(record["evidence_digest"]) == 64
    assert feedback["measured_task_outcome"] == "unverified"
    assert len(setup.rig.gate.seen) == 1
    assert not setup.rig.sim.batches and setup.rig.sim.halts == 0
    assert not setup.rig.runtime.holds_device(WRIST)


@pytest.mark.parametrize("change", ["frame", "qualification", "session"])
async def test_real_lazy_backend_retains_proof_and_rechecks_live_store(
    admission_rig: Any, change: str
) -> None:
    setup = admission_rig
    await install(setup)
    ctx = real_context(setup)
    backend = ctx.grasp_verification
    assert isinstance(backend, WorkcellGraspVerification) and ctx.grasp_verification is backend
    baseline = await backend.verify("hold", samples=3)
    assert baseline["passed"] and baseline["state"] == "contact_candidate"
    observer = backend._observer
    setup.rig.sim.pose[2] += 5  # External synthetic I/O; never dispatch a real command.
    proof = await backend.verify("hold", samples=3)
    assert proof["passed"] and proof["state"] == "held_verified"
    assert backend._observer is observer
    if change == "frame":
        setup.registry.calibrate("tcp.offset", pose=Pose(0, 0, 120).to_dict())
    elif change == "qualification":
        setup.registry.calibrate(PROCEDURE, parameters=setup.data)
    else:
        setup.camera.fault = "session"
    with pytest.raises(SkillError):
        await backend.verify("hold", samples=3)
    assert backend._observer is None
    assert not setup.rig.sim.batches


@pytest.mark.parametrize(
    "fault",
    [
        "camera",
        "arm",
        "geometry_config",
        "depth_config",
        "version",
        "geometry",
        "support",
        "physical",
    ],
)
async def test_real_admission_refuses_before_approval_or_store(
    admission_rig: Any, fault: str
) -> None:
    setup = admission_rig
    if fault == "camera":
        setup.data["camera_device"] = CAMERA_ID
    elif fault == "arm":
        setup.data["arm_declaration_digest"] = "foreign"
    elif fault in ("geometry_config", "depth_config"):
        key = (
            "geometry_configuration_digest"
            if fault == "geometry_config"
            else "depth_configuration_digest"
        )
        setup.data[key] = "foreign"
    elif fault == "version":
        setup.data["depends_on"]["tcp"] += 1
    elif fault == "geometry":
        setup.data["calibration"]["components"][0]["matrix"][0][3] += 1
    elif fault == "support":
        setup.registry.calibrate("workspace.safe_box", parameters=SAFE_BOX)
    else:
        setup.data["object_reference"]["physical"] = False
    result = await setup.rig.service.hw_skill_run(
        workcell=QUALIFIED, skill="calib.grasp_evidence", args={"qualification": setup.data}
    )
    assert not result["ok"] and result["side_effect_state"] == "none", result
    assert result["failure_code"] != "internal_defect", result
    assert not setup.rig.gate.seen
    assert setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE) is None
    assert not setup.rig.sim.batches and setup.rig.sim.halts == 0


@pytest.mark.parametrize("fault", ["unit", "stale", "frozen", "domain", "scope"])
async def test_real_sensor_admission_fails_closed(admission_rig: Any, fault: str) -> None:
    setup = admission_rig
    await install(setup)
    if fault in ("unit", "stale"):
        setup.faults["feedback"] = fault
    else:
        setup.camera.fault = fault
    result = await setup.rig.service.hw_skill_run(
        workcell=QUALIFIED, skill="grasp.verify", args={"stage": "hold", "samples": 3}
    )
    assert not result["ok"] and result["side_effect_state"] == "none", result
    assert result["failure_code"] != "internal_defect", result
    assert not setup.rig.sim.batches and setup.rig.sim.halts == 0


async def test_stored_evidence_inspection_is_readonly_while_blocked(admission_rig: Any) -> None:
    setup = admission_rig
    before = setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE)
    missing = await setup.rig.service.hw_skill_run(
        workcell=QUALIFIED, skill="grasp.evidence", args={}
    )
    assert missing["ok"] and missing["side_effect_state"] == "none", str(missing)
    assert missing["result"]["installed_version"] is None
    assert missing["result"]["stored_versions"] == []
    assert not missing["result"]["history_truncated"]
    assert not missing["result"]["execution_ready"]
    assert all(row["status"] == "missing" for row in missing["result"]["components"].values())
    assert setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE) is before
    await install(setup)
    stored = setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE)
    setup.registry.calibrate("wrist.handeye", pose=Pose(0, 0, 120).to_dict(), parameters={})
    inspected = await setup.rig.service.hw_skill_run(
        workcell=QUALIFIED, skill="grasp.evidence", args={}
    )
    assert inspected["ok"] and inspected["side_effect_state"] == "none", str(inspected)
    report = inspected["result"]
    assert (
        report["metadata_only"]
        and not report["execution_ready"]
        and not report["package_admissible"]
    )
    assert report["dependency_status"]["camera.wrist"]["status"] == "mismatched"
    assert report["force"]["status"] == "qualified_stationary_reference"
    assert report["components"]["camera_to_base"]["status"] == "invalid"
    assert "package" not in report and setup.camera.frame == 0
    assert (
        setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE).recorded_at
        == stored.recorded_at
    )
    assert len(setup.rig.gate.seen) == 1 and not setup.rig.sim.batches


async def test_history_discovers_older_reuse_identity_without_export_or_motion(
    admission_rig: Any,
) -> None:
    setup = admission_rig
    await install(setup)
    store = setup.registry.calibration_store
    valid = store.latest(WORKCELL_ID, PROCEDURE)
    setup.registry.calibrate(PROCEDURE, parameters={"broken": float("nan")})
    malformed = store.latest(WORKCELL_ID, PROCEDURE)
    inspected = await setup.rig.service.hw_skill_run(
        workcell=QUALIFIED, skill="grasp.evidence", args={"history_limit": 1}
    )
    assert inspected["ok"] and inspected["side_effect_state"] == "none", inspected
    report = inspected["result"]
    assert report["history_truncated"] and len(report["stored_versions"]) == 1
    assert report["stored_versions"][0] == {
        "recorded_at": malformed.recorded_at,
        "admissibility": "malformed",
    }
    all_versions = await real_context(setup).grasp_verification.evidence(history_limit=2)
    assert not all_versions["history_truncated"]
    assert all_versions["installed_version"] == malformed.recorded_at
    assert all_versions["stored_versions"][1] == {
        "recorded_at": valid.recorded_at,
        "qualification_digest": digest(valid.parameters),
        "admissibility": "not_assessed",
    }
    assert not all_versions["execution_ready"] and not all_versions["package_admissible"]
    assert "package" not in all_versions
    assert all(
        set(row) <= {"recorded_at", "qualification_digest", "admissibility"}
        for row in all_versions["stored_versions"]
    )
    request = {
        "workcell": QUALIFIED,
        "skill": "calib.grasp_evidence",
        "args": {"reuse_from": valid.recorded_at},
    }
    preview = await setup.rig.service.hw_skill_run(**request, dry_run=True)
    assert preview["ok"], preview
    assert store.latest(WORKCELL_ID, PROCEDURE).recorded_at == malformed.recorded_at
    assert setup.camera.frame == 0 and not setup.rig.sim.batches


@pytest.mark.parametrize("limit", [0, 101, True, 1.5, "20"])
async def test_history_rejects_unbounded_or_coerced_limit(admission_rig: Any, limit: Any) -> None:
    with pytest.raises(ValueError, match="bounded evidence"):
        await real_context(admission_rig).grasp_verification.evidence(history_limit=limit)
    assert not admission_rig.rig.sim.batches and admission_rig.camera.frame == 0


@pytest.mark.parametrize("reuse", [False, True])
async def test_identical_valid_installation_is_a_governed_noop(
    admission_rig: Any, reuse: bool
) -> None:
    setup = admission_rig
    first = await install(setup)
    stored = setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE)
    args = {"reuse_from": stored.recorded_at} if reuse else {"qualification": setup.data}
    request = {"workcell": QUALIFIED, "skill": "calib.grasp_evidence", "args": args}
    preview = await setup.rig.service.hw_skill_run(**request, dry_run=True)
    assert preview["ok"], str(preview)
    assert not InstallGraspEvidence.spec.uses_motion and InstallGraspEvidence.spec.is_actuating
    result = await setup.rig.service.hw_skill_run(**request, approved_plan=preview["plan"])
    assert result["ok"] and result["side_effect_state"] == "none", str(result)
    assert result["result"]["no_op"] and result["result"]["recorded_at"] == stored.recorded_at
    assert result["result"]["qualification_digest"] == first["result"]["qualification_digest"]
    assert len(setup.rig.gate.seen) == 2
    assert len([r for r in setup.rig.audit.records if r["action"] == "skill_permit"]) == 2
    assert (
        setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE).recorded_at
        == stored.recorded_at
    )
    assert not setup.rig.sim.batches and setup.camera.frame == 0


async def test_noop_requires_exact_package_content_not_numeric_coercion(admission_rig: Any) -> None:
    setup = admission_rig
    await install(setup)
    source = setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE)
    data = copy.deepcopy(setup.data)
    data["object_reference"]["dimensions_mm"] = [20.0, 30.0, 20.0]
    proposed = await real_context(setup).grasp_verification.installation({"qualification": data})
    assert proposed["no_op"] is None
    assert proposed["qualification_digest"] != digest(source.parameters)
    assert setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE) == source


async def test_invalid_prior_package_does_not_block_valid_governed_installation(
    admission_rig: Any,
) -> None:
    setup = admission_rig
    setup.registry.calibrate(PROCEDURE, parameters={"broken": float("nan")})
    prior = setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE)
    request = {
        "workcell": QUALIFIED,
        "skill": "calib.grasp_evidence",
        "args": {"qualification": setup.data},
    }
    preview = await setup.rig.service.hw_skill_run(**request, dry_run=True)
    assert preview["ok"] and preview["plan"]["basis"]["no_op"] is None, str(preview)
    result = await setup.rig.service.hw_skill_run(**request, approved_plan=preview["plan"])
    assert result["ok"] and result["side_effect_state"] == "committed", str(result)
    assert (
        setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE).recorded_at
        != prior.recorded_at
    )
    assert (
        setup.registry.calibration_store.get_version(WORKCELL_ID, PROCEDURE, prior.recorded_at)
        is not None
    )
    assert len(setup.rig.gate.seen) == 1 and setup.camera.frame == 0 and not setup.rig.sim.batches


def refresh_frames(setup: Any) -> None:
    setup.spec = setup.rig.runtime._workcells[WORKCELL_ID]
    setup.frames = FrameGraph.snapshot(setup.spec, setup.registry.calibration_store)


def repeat_frame(setup: Any, frame: str, *, change: str = "") -> None:
    procedure = setup.frames.procedure(frame)
    record = setup.registry.calibration_store.latest(WORKCELL_ID, procedure)
    parameters, pose, matrix = copy.deepcopy(record.parameters), record.pose, record.matrix
    if change == "provenance":
        parameters["source_evidence"] = "different-independent-measurement"
    elif change == "geometry":
        pose = {**pose, "z": pose["z"] + 1}
    setup.registry.calibrate(procedure, parameters=parameters, pose=pose, matrix=matrix)
    refresh_frames(setup)


async def test_equivalent_stored_evidence_rebinds_only_installation_metadata(
    admission_rig: Any,
) -> None:
    setup = admission_rig
    await install(setup)
    source = setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE)
    repeat_frame(setup, "tcp")
    request = {
        "workcell": QUALIFIED,
        "skill": "calib.grasp_evidence",
        "args": {"reuse_from": source.recorded_at},
    }
    preview = await setup.rig.service.hw_skill_run(**request, dry_run=True)
    assert preview["ok"], str(preview)
    result = await setup.rig.service.hw_skill_run(**request, approved_plan=preview["plan"])
    assert result["ok"] and result["side_effect_state"] == "committed", str(result)
    rebound = setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE)
    assert rebound.recorded_at != source.recorded_at
    assert rebound.parameters["depends_on"]["tcp"] == setup.frames.versions["tcp"]
    assert rebound.parameters["reused_from"] == {
        "recorded_at": source.recorded_at,
        "digest": digest(source.parameters),
    }
    for key in ("calibration", "effort", "object_reference", "scene_id", "profile_digest"):
        assert rebound.parameters[key] == source.parameters[key]
    assert (
        setup.registry.calibration_store.get_version(WORKCELL_ID, PROCEDURE, source.recorded_at)
        == source
    )
    assert real_context(setup).grasp_verification._observer is None
    assert setup.camera.frame == 0 and not setup.rig.sim.batches


@pytest.mark.parametrize(
    "fault",
    [
        "geometry",
        "provenance",
        "history",
        "legacy",
        "mount",
        "profile",
        "configuration",
        "arm",
        "device",
        "parent",
        "model",
        "drift",
    ],
)
async def test_stored_reuse_refuses_unestablished_equivalence(
    admission_rig: Any, monkeypatch: Any, fault: str
) -> None:
    setup = admission_rig
    await install(setup)
    source = setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE)
    if fault in ("geometry", "provenance", "history", "legacy"):
        if fault == "legacy":
            source = replace(
                source,
                recorded_at=source.recorded_at + 0.001,
                parameters={
                    k: v for k, v in source.parameters.items() if k != "installation_scope"
                },
            )
            assert setup.registry.calibration_store.record(source)
        repeat_frame(setup, "tcp", change=fault)
        if fault == "history":
            original = setup.registry.calibration_store.get_version
            monkeypatch.setattr(
                setup.registry.calibration_store,
                "get_version",
                lambda cell, procedure, version: (
                    None
                    if procedure == "tcp.offset"
                    and version == source.parameters["depends_on"]["tcp"]
                    else original(cell, procedure, version)
                ),
            )
    elif fault in ("mount", "profile", "configuration"):
        declaration = setup.declarations[WRIST].to_dict()
        config = declaration["transport"]["config"]
        if fault == "mount":
            config["mount_digest"] = "changed-mount"
        elif fault == "profile":
            config["depth"]["width"] = 999
        else:
            config["measurement"]["roi"] = [2, 2, 5, 5]
        setup.declarations[WRIST] = HardwareContext.from_mapping(declaration)
    elif fault == "arm":
        setup.registry._context = replace(setup.registry._context, display_name="changed-arm")
    elif fault in ("device", "parent"):
        spec = setup.spec
        if fault == "device":
            spec = replace(spec, roles={**spec.roles, "camera.wrist": CAMERA_ID})
        else:
            spec = replace(
                spec,
                frames={
                    **spec.frames,
                    "camera.wrist": replace(spec.frames["camera.wrist"], parent="base"),
                },
            )
        setup.rig.runtime._workcells[WORKCELL_ID] = spec
        refresh_frames(setup)
    elif fault == "model":
        from leapflow.hardware.skills.admission import generation_basis

        model = await generation_basis(setup.registry, setup.spec)
        monkeypatch.setattr(
            "leapflow.hardware.skills.admission.generation_basis",
            lambda *args: asyncio.sleep(0, result={**model, "generation": 123}),
        )
    else:
        setup.registry.calibrate(
            "tcp.attestation",
            parameters={"passed": False, "depends_on": {"tcp": setup.frames.versions["tcp"]}},
        )
        refresh_frames(setup)
    result = await setup.rig.service.hw_skill_run(
        workcell=QUALIFIED,
        skill="calib.grasp_evidence",
        args={"reuse_from": source.recorded_at},
        dry_run=True,
    )
    assert not result["ok"] and result["side_effect_state"] == "none", str(result)
    assert result["failure_code"] != "internal_defect", str(result)
    assert (
        setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE).recorded_at
        == source.recorded_at
    )
    assert len(setup.rig.gate.seen) == 1 and not setup.rig.sim.batches


@pytest.mark.parametrize("declared_overview", [False, True])
async def test_new_packages_enforce_minimum_and_all_declared_dependencies(
    admission_rig: Any, declared_overview: bool
) -> None:
    setup = admission_rig
    if not declared_overview:
        setup.data["depends_on"].pop("camera")
    await install(setup)
    repeat_frame(setup, "camera")
    backend = real_context(setup).grasp_verification
    if declared_overview:
        with pytest.raises(ValueError, match="dependency camera"):
            backend._qualification()
    else:
        assert backend._qualification().data["depends_on"] == setup.data["depends_on"]
    assert setup.camera.frame == 0 and not setup.rig.sim.batches


@pytest.mark.parametrize("change", ["source", "source_content", "latest", "frame"])
async def test_reuse_approval_never_survives_a_changed_basis(
    admission_rig: Any, change: str, monkeypatch: Any
) -> None:
    setup = admission_rig
    await install(setup)
    source = setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE)
    request = {
        "workcell": QUALIFIED,
        "skill": "calib.grasp_evidence",
        "args": {"reuse_from": source.recorded_at},
    }
    preview = await setup.rig.service.hw_skill_run(**request, dry_run=True)
    assert preview["ok"], str(preview)
    if change == "source":
        original = setup.registry.calibration_store.get_version
        monkeypatch.setattr(
            setup.registry.calibration_store,
            "get_version",
            lambda cell, procedure, version: (
                None if procedure == PROCEDURE else original(cell, procedure, version)
            ),
        )
    elif change == "source_content":
        changed = copy.deepcopy(source.parameters)
        changed["effort"]["independent_reference"] = "different-test-force-reference"
        assert setup.registry.calibration_store.record(replace(source, parameters=changed))
    elif change == "latest":
        setup.registry.calibrate(PROCEDURE, parameters=source.parameters)
    else:
        repeat_frame(setup, "tcp")
    before = setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE).recorded_at
    result = await setup.rig.service.hw_skill_run(**request, approved_plan=preview["plan"])
    assert not result["ok"] and result["side_effect_state"] == "none", str(result)
    assert result["failure_code"] == "plan_stale", str(result)
    assert setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE).recorded_at == before
    assert not setup.rig.sim.batches


@pytest.mark.parametrize(
    "fault",
    [
        "effort",
        "object",
        "component",
        "geometry",
        "missing_scope",
        "force_row",
        "flange",
        "calibration_row",
        "dependencies",
    ],
)
async def test_malformed_stored_packages_remain_inspectable_and_cannot_install(
    admission_rig: Any, fault: str
) -> None:
    setup = admission_rig
    data = copy.deepcopy(setup.data)
    if fault == "effort":
        data["effort"] = []
    elif fault == "object":
        data["object_reference"] = None
    elif fault == "component":
        data["calibration"]["components"][0]["qualification"]["component"] = []
    elif fault == "geometry":
        data["calibration"]["components"][0]["geometry"] = None
    elif fault == "force_row":
        data["effort"]["training"][0] = None
    elif fault == "flange":
        data["effort"]["training"][0]["flange"] = []
    elif fault == "calibration_row":
        data["calibration"]["components"][0] = None
    elif fault == "dependencies":
        data["depends_on"] = None
    else:
        data.pop("scene_id")
    setup.registry.calibrate(PROCEDURE, parameters=data)
    inspected = await setup.rig.service.hw_skill_run(
        workcell=QUALIFIED, skill="grasp.evidence", args={}
    )
    assert inspected["ok"] and inspected["side_effect_state"] == "none", str(inspected)
    assert (
        not inspected["result"]["package_admissible"] and not inspected["result"]["execution_ready"]
    )
    assert inspected["result"]["problems"]
    refused = await setup.rig.service.hw_skill_run(
        workcell=QUALIFIED, skill="calib.grasp_evidence", args={"qualification": data}
    )
    assert not refused["ok"] and refused["failure_code"] != "internal_defect", str(refused)
    assert not setup.rig.gate.seen and not setup.rig.sim.batches and setup.camera.frame == 0


@pytest.mark.parametrize("frame", ["camera.wrist", "tcp", "workspace"])
async def test_installation_requires_each_geometry_dependency(
    admission_rig: Any, frame: str
) -> None:
    setup = admission_rig
    setup.data["depends_on"].pop(frame)
    result = await setup.rig.service.hw_skill_run(
        workcell=QUALIFIED,
        skill="calib.grasp_evidence",
        args={"qualification": setup.data},
        dry_run=True,
    )
    assert not result["ok"] and result["side_effect_state"] == "none", str(result)
    assert "bind wrist camera, TCP and support-plane" in result["error"]
    assert "grasp.evidence" in result["next_step"] and not setup.rig.gate.seen


async def test_legacy_exact_binding_can_noop_without_upgrading_metadata(admission_rig: Any) -> None:
    setup = admission_rig
    setup.registry.calibrate(PROCEDURE, parameters=setup.data)
    legacy = setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE)
    request = {
        "workcell": QUALIFIED,
        "skill": "calib.grasp_evidence",
        "args": {"reuse_from": legacy.recorded_at},
    }
    preview = await setup.rig.service.hw_skill_run(**request, dry_run=True)
    assert preview["ok"], str(preview)
    result = await setup.rig.service.hw_skill_run(**request, approved_plan=preview["plan"])
    assert result["ok"] and result["result"]["no_op"] and result["side_effect_state"] == "none", (
        str(result)
    )
    current = setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE)
    assert current == legacy and "installation_scope" not in current.parameters
    assert not setup.rig.sim.batches


async def test_unknown_mount_cannot_rebind_stored_evidence(admission_rig: Any) -> None:
    setup = admission_rig
    declaration = setup.declarations[WRIST].to_dict()
    declaration["transport"]["config"].pop("mount_digest")
    setup.declarations[WRIST] = HardwareContext.from_mapping(declaration)
    await install(setup)
    source = setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE)
    assert not source.parameters["installation_scope"]["cameras"]["camera.wrist"]["mount_digest"]
    repeat_frame(setup, "tcp")
    refused = await setup.rig.service.hw_skill_run(
        workcell=QUALIFIED,
        skill="calib.grasp_evidence",
        args={"reuse_from": source.recorded_at},
        dry_run=True,
    )
    assert not refused["ok"] and "declared mounting identity" in refused["error"], str(refused)
    assert not setup.rig.sim.batches


@pytest.mark.parametrize(
    "args", [{}, {"reuse_from": 0}, {"reuse_from": 123}, {"reuse_from": 123, "qualification": {}}]
)
async def test_stored_reuse_requires_one_exact_available_source(
    admission_rig: Any, args: dict[str, Any]
) -> None:
    setup = admission_rig
    result = await setup.rig.service.hw_skill_run(
        workcell=QUALIFIED, skill="calib.grasp_evidence", args=args, dry_run=True
    )
    assert not result["ok"] and result["side_effect_state"] == "none", str(result)
    assert result["failure_code"] != "internal_defect" and not setup.rig.gate.seen
    assert setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE) is None


async def test_evidence_disclosure_is_explicit_redacted_and_bounded(admission_rig: Any) -> None:
    setup = admission_rig
    data = copy.deepcopy(setup.data)
    data["private_frames"] = ["private-camera-pixels"]
    data["effort"]["credentials"] = {"password": "not-public"}
    data["effort"]["training"][0]["raw"] = "private-force-pixels"
    setup.registry.calibrate(PROCEDURE, parameters=data)
    result = await setup.rig.service.hw_skill_run(
        workcell=QUALIFIED, skill="grasp.evidence", args={"include_package": True}
    )
    assert result["ok"], str(result)
    report = result["result"]
    assert report["package_redacted"] and "private_frames" not in report["package"]
    assert "credentials" not in report["package"]["effort"]
    assert "raw" not in report["package"]["effort"]["training"][0]
    bounded = await setup.rig.service.hw_skill_run(
        workcell=QUALIFIED, skill="grasp.evidence", args={"include_package": True, "max_bytes": 1}
    )
    assert (
        bounded["ok"]
        and "package" not in bounded["result"]
        and bounded["result"]["package_omitted"]
    )
    assert setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE).parameters[
        "private_frames"
    ] == ["private-camera-pixels"]
    assert not setup.rig.gate.seen and setup.camera.frame == 0


async def test_mount_mismatch_is_visible_and_refused_before_approval(admission_rig: Any) -> None:
    setup = admission_rig
    declaration = setup.declarations[WRIST].to_dict()
    declaration["transport"]["config"]["mount_digest"] = "different-test-mount"
    setup.declarations[WRIST] = HardwareContext.from_mapping(declaration)
    refused = await setup.rig.service.hw_skill_run(
        workcell=QUALIFIED,
        skill="calib.grasp_evidence",
        args={"qualification": setup.data},
        dry_run=True,
    )
    assert not refused["ok"] and "mounting identity" in refused["error"], str(refused)
    setup.registry.calibrate(PROCEDURE, parameters=setup.data)
    inspected = await setup.rig.service.hw_skill_run(
        workcell=QUALIFIED, skill="grasp.evidence", args={}
    )
    assert inspected["ok"] and not inspected["result"]["package_admissible"], str(inspected)
    assert any("mounting identity" in problem for problem in inspected["result"]["problems"])
    assert not setup.rig.gate.seen and not setup.rig.sim.batches and setup.camera.frame == 0


async def test_unrelated_operational_settings_do_not_prevent_persistent_reuse(
    admission_rig: Any,
) -> None:
    setup = admission_rig
    await install(setup)
    source = setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE)
    setup.rig.runtime._workcells[WORKCELL_ID] = replace(
        setup.spec,
        limits={**setup.spec.limits, "default_speed_pct": 1},
    )
    refresh_frames(setup)
    repeat_frame(setup, "tcp")
    proposed = await real_context(setup).grasp_verification.installation(
        {"reuse_from": source.recorded_at}
    )
    assert proposed["parameters"]["installation_scope"] == source.parameters["installation_scope"]
    assert proposed["parameters"]["depends_on"]["tcp"] == setup.frames.versions["tcp"]
    assert proposed["no_op"] is None and not setup.rig.sim.batches and setup.camera.frame == 0


async def test_inspection_detects_changed_physical_model_without_capture(
    admission_rig: Any, monkeypatch: Any
) -> None:
    from leapflow.hardware.skills.admission import generation_basis

    setup = admission_rig
    await install(setup)
    model = await generation_basis(setup.registry, setup.spec)
    monkeypatch.setattr(
        "leapflow.hardware.skills.admission.generation_basis",
        lambda *args: asyncio.sleep(0, result={**model, "generation": 123}),
    )
    inspected = await setup.rig.service.hw_skill_run(
        workcell=QUALIFIED, skill="grasp.evidence", args={}
    )
    assert inspected["ok"] and not inspected["result"]["package_admissible"], str(inspected)
    assert "stored physical-model installation scope changed" in inspected["result"]["problems"]
    assert not inspected["result"]["execution_ready"] and setup.camera.frame == 0


async def test_real_inspection_requires_current_overview_holdout(admission_rig: Any) -> None:
    setup = admission_rig
    await install(setup)
    request = {"workcell": QUALIFIED, "skill": "grasp.inspect", "args": {}}
    missing = await setup.rig.service.hw_skill_run(**request)
    assert not missing["ok"] and missing["side_effect_state"] == "none"
    assert missing["failure_code"] == "skill_unavailable", missing
    assert "camera.verify" in missing["error"], missing
    setup.registry.calibrate(
        "camera.verify",
        parameters={
            "camera_role": "camera",
            "training_disjoint": True,
            "depends_on": {"camera": setup.frames.versions["camera"]},
            "errors_mm": [0.1] * int(setup.spec.calibration["min_verify_points"]),
        },
    )
    inspected = await setup.rig.service.hw_skill_run(**request)
    assert inspected["ok"] and inspected["side_effect_state"] == "none", inspected
    stored = setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE)
    assert inspected["result"]["qualification"]["qualification_digest"] == digest(stored.parameters)
    assert inspected["result"]["measured_geometry"]["independent_jaws"] is True
    setup.registry.calibrate("camera.handeye", pose=Pose(0, 0, 0).to_dict())
    stale = await setup.rig.service.hw_skill_run(**request)
    assert not stale["ok"] and stale["side_effect_state"] == "none", stale
    assert stale["failure_code"] != "internal_defect", stale
    assert not setup.rig.sim.batches and setup.rig.sim.halts == 0


async def test_the_listing_reports_skills_that_need_an_uninstalled_record(
    admission_rig: Any,
) -> None:
    """Verification skills are unavailable, with the reason, until their records exist.

    The listing used to call them available, so a planner picked grasp.inspect
    and every run refused it as skill_unavailable.
    """
    setup = admission_rig

    async def availability() -> dict[str, Any]:
        listed = await setup.rig.service.hw_skill_list()
        (cell,) = [view for view in listed["workcells"] if view.get("skills")]
        return {s["skill_id"]: (s["available"], s.get("reason", "")) for s in cell["skills"]}

    before = await availability()
    assert before["calib.grasp_evidence"][0] is True
    for skill in ("grasp.verify", "grasp.inspect"):
        available, reason = before[skill]
        assert available is False and "grasp.qualification record" in reason, before[skill]

    await install(setup)
    after = await availability()
    assert after["grasp.verify"] == (True, "")
    assert after["grasp.inspect"][0] is False and "camera.verify" in after["grasp.inspect"][1]


def test_record_tokens_name_the_procedures_the_verifier_reads() -> None:
    """Skill modules spell the token literally; this keeps it on the real procedure."""
    token = f"record:{PROCEDURE}"
    assert token in GraspVerify.spec.requires and token in GraspInspect.spec.requires
    assert token not in InstallGraspEvidence.spec.requires, "installing must not need itself"


@pytest.mark.parametrize("fault", ["config", "arm", "frame"])
async def test_cached_install_refuses_changed_live_binding(admission_rig: Any, fault: str) -> None:
    setup = admission_rig
    request = {
        "workcell": QUALIFIED,
        "skill": "calib.grasp_evidence",
        "args": {"qualification": setup.data},
    }
    preview = await setup.rig.service.hw_skill_run(**request, dry_run=True)
    assert preview["ok"], str(preview)
    if fault == "config":
        declaration = setup.declarations[WRIST].to_dict()
        declaration["transport"]["config"]["measurement"]["roi"] = [2, 2, 10, 10]
        setup.declarations[WRIST] = HardwareContext.from_mapping(declaration)
    elif fault == "arm":
        setup.registry._context = replace(setup.registry._context, display_name="changed-arm")
    else:
        setup.registry.calibrate("tcp.offset", pose=Pose(0, 0, 121).to_dict())
    result = await setup.rig.service.hw_skill_run(**request, approved_plan=preview["plan"])
    assert not result["ok"] and result["side_effect_state"] == "none", result
    assert result["failure_code"] != "internal_defect", result
    assert setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE) is None
    assert not setup.rig.sim.batches and setup.rig.sim.halts == 0
