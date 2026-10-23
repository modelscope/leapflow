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
from leapflow.hardware.providers.yaml_provider import declaration_digest
from leapflow.hardware.skills.builtin.grasp_verify import (
    GraspInspect,
    GraspVerify,
    InstallGraspEvidence,
)
from leapflow.hardware.skills.frames import FrameGraph
from leapflow.hardware.skills.geometry import Pose
from leapflow.hardware.skills.grasp_verification import PROCEDURE, WorkcellGraspVerification
from leapflow.hardware.skills.protocol import SkillError
from leapflow.hardware.transport import BatchReading, Reading
from tests._harness.skill_stubs import (
    ARM_ID,
    CAMERA_ID,
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
        node_skills=(InstallGraspEvidence(), GraspVerify(), GraspInspect()),
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
            "geometry_measurement": {"scene_id": "scene", "fixture": "synthetic-test"},
            "measurement": {"roi": [1, 1, 10, 10], "fixture": "synthetic-test"},
        }
    )
    declarations = {WRIST: HardwareContext.from_mapping(declaration)}
    original_context = registry.context
    monkeypatch.setattr(
        registry, "context", lambda device: declarations.get(device) or original_context(device)
    )
    registry.calibrate("camera.handeye", pose=Pose(0, 0, 0).to_dict())
    registry.calibrate("wrist.handeye", pose=Pose(0, 0, 120).to_dict())
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
    assert preview["ok"], preview
    assert setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE) is None
    result = await setup.rig.service.hw_skill_run(**request, approved_plan=preview["plan"])
    assert result["ok"] and result["side_effect_state"] == "committed", result
    setup.camera.qualification = real_context(setup).grasp_verification._qualification()
    return result


async def test_real_install_and_readonly_runtime_verification(admission_rig: Any) -> None:
    setup = admission_rig
    result = await install(setup)
    stored = setup.registry.calibration_store.latest(WORKCELL_ID, PROCEDURE)
    assert digest(stored.parameters) == digest(setup.data)
    assert result["result"]["qualification_digest"] == digest(setup.data)
    assert len(setup.rig.gate.seen) == 1
    assert len([r for r in setup.rig.audit.records if r["action"] == "skill_permit"]) == 1
    verified = await setup.rig.service.hw_skill_run(
        workcell=QUALIFIED, skill="grasp.verify", args={"stage": "hold", "samples": 3}
    )
    assert verified["ok"] and verified["side_effect_state"] == "none", verified
    assert verified["result"]["state"] == "contact_candidate"
    assert len(verified["result"]["observations"]) == 3
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


async def test_real_inspection_requires_current_overview_holdout(admission_rig: Any) -> None:
    setup = admission_rig
    await install(setup)
    request = {"workcell": QUALIFIED, "skill": "grasp.inspect", "args": {}}
    missing = await setup.rig.service.hw_skill_run(**request)
    assert not missing["ok"] and missing["side_effect_state"] == "none"
    setup.registry.calibrate("camera.verify", parameters={
        "camera_role": "camera", "training_disjoint": True,
        "depends_on": {"camera": setup.frames.versions["camera"]},
        "errors_mm": [0.1] * int(setup.spec.calibration["min_verify_points"]),
    })
    inspected = await setup.rig.service.hw_skill_run(**request)
    assert inspected["ok"] and inspected["side_effect_state"] == "none", inspected
    assert inspected["result"]["qualification"]["qualification_digest"] == digest(setup.data)
    assert inspected["result"]["measured_geometry"]["independent_jaws"] is True
    setup.registry.calibrate("camera.handeye", pose=Pose(0, 0, 0).to_dict())
    stale = await setup.rig.service.hw_skill_run(**request)
    assert not stale["ok"] and stale["side_effect_state"] == "none", stale
    assert stale["failure_code"] != "internal_defect", stale
    assert not setup.rig.sim.batches and setup.rig.sim.halts == 0


@pytest.mark.parametrize("fault", ["config", "arm", "frame"])
async def test_cached_install_refuses_changed_live_binding(admission_rig: Any, fault: str) -> None:
    setup = admission_rig
    request = {
        "workcell": QUALIFIED,
        "skill": "calib.grasp_evidence",
        "args": {"qualification": setup.data},
    }
    preview = await setup.rig.service.hw_skill_run(**request, dry_run=True)
    assert preview["ok"], preview
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
