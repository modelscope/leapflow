"""Offline synthetic fixtures exercise physical admission; they certify no hardware."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from leapflow.hardware.calibration_qualification import (
    COMPONENTS,
    CalibrationQualification,
    QualificationLimits,
)
from leapflow.hardware.grasp_evidence import (
    EvidenceIdentity,
    GraspLimits,
    GraspMeasurement,
    GraspState,
    TemporalGraspObserver,
)
from leapflow.hardware.grasp_measurement import EffortReference, GraspQualification
from leapflow.hardware.skills.builtin.grasp_verify import GraspVerify, InstallGraspEvidence
from leapflow.hardware.skills.geometry import Plane, Pose, Transform
from leapflow.hardware.skills.grasp_verification import WorkcellGraspVerification
from leapflow.hardware.skills.protocol import SkillError
from leapflow.hardware.skills.tuning import DEFAULT_GRASP_VERIFY
from leapflow.hardware.transport import BatchReading, Reading

LIMITS = dict(DEFAULT_GRASP_VERIFY)


def effort_data():
    def rows(split):
        return [
            {
                "id": f"{split}:{label}:{i}",
                "label": label,
                "stationary": True,
                "width_mm": width,
                "flange": Pose(0, 0, 0).to_dict(),
                "effort_n": force,
                "reference_force_n": force,
            }
            for label, width, forces in (("empty", 0.1, [0, 0, 0]), ("contact", 20, [0.9, 1, 1.1]))
            for i, force in enumerate(forces)
        ]

    return {
        "units": "N",
        "physical": True,
        "condition": "stationary",
        "independent_reference": "synthetic-test-force-reference",
        "uncertainty_n": 0.05,
        "command_force_n": 1,
        "training": rows("train"),
        "holdout": rows("heldout"),
    }


def qualification_data():
    geometries = {
        "jaw_stroke": {"scale": 1, "offset_mm": 0, "min_width_mm": 0, "max_width_mm": 69.5},
        "inner_pad_midpoint": {"tool_mm": [0, 0, 100]},
        "fingertip_plane": {"tool_z_mm": 110},
    }
    components = []
    for component in sorted(COMPONENTS):
        geometry = geometries.get(component, {"offset_mm": [0, 0, 0]})
        certificate = CalibrationQualification(
            component,
            "test-v1",
            "test-mount",
            "mm",
            "Rz@Ry@Rx;degrees",
            ("train:1", "train:2"),
            (),
            ("heldout:1", "heldout:2"),
            (0.05, 0.05),
            0.05,
            "synthetic-test-independent-reference",
            True,
            QualificationLimits(2, 2, 1, 1, 1),
        ).bind_geometry(geometry)
        components.append({"qualification": asdict(certificate), "geometry": geometry})
    return {
        "calibration": {"components": components},
        "effort": effort_data(),
        "camera_device": "camera_wrist",
        "arm_declaration_digest": "arm-digest",
        "profile_digest": "profile",
        "scene_id": "scene",
        "geometry_configuration_digest": "test-config",
        "depth_configuration_digest": "test-depth-config",
        "object_reference": {
            "physical": True,
            "independent_reference": "synthetic-test-caliper",
            "measurement_id": "test-only",
            "dimensions_mm": [20, 30, 20],
            "uncertainty_mm": 0.05,
        },
    }


def qualified():
    return GraspQualification.admit(
        qualification_data(), device_id="cell", recorded_at=1, limits=LIMITS
    )


def geometry(q, *, frame=1, width=20, visible=True):
    depth = {
        "qualified": True,
        "frame_ids": [f"session:{frame}:{frame}"],
        "calibration_digest": q.package.digest,
        "mount_digest": "test-mount",
        "profile_digest": "profile",
        "uncertainty_mm": 0.05,
    }
    return {
        "session_id": "session",
        "scene_id": "scene",
        "profile_digest": "profile",
        "configuration_digest": "test-config",
        "observed_mono": time.monotonic(),
        "source": {
            k: {
                "timestamp_ms": frame * 33,
                "frame_number": frame,
                "timestamp_domain": "hardware_clock",
            }
            for k in ("color_source", "depth_source")
        },
        "depth": depth,
        "pad_depth": [dict(depth), dict(depth)],
        "independent_jaws": True,
        "jaw_points_camera_mm": [[-width / 2, 0, 100], [width / 2, 0, 100]],
        "object_detected": visible,
        "dimensions_mm": {"box": [20, 30, 20]},
        "proposals": [
            {
                "hypothesis_id": "box",
                "center_camera_mm": [0, 0, 100],
                "rpy_deg": [0, 0, 0],
                "observable": True,
                "depth_qualified": True,
                "surface_rms_mm": 0.05,
                "contour_rms_px": 0.1,
            }
        ]
        if visible
        else [],
    }


def measure(q, data, *, width=20, effort=1, rise=0):
    return q.measure(
        data,
        width_mm=width,
        effort_n=effort,
        flange=Pose(0, 0, rise),
        tcp=Pose(0, 0, rise),
        mount=Transform(),
        table=Plane.horizontal(90),
        synchronized=True,
        limits=LIMITS,
    )


def test_measured_contact_and_distinct_proof_lift_use_one_observer():
    q = qualified()
    sample, alignment = measure(q, geometry(q))
    assert sample.pad_engagement and sample.width_consistent and sample.motion_conditioned_effort
    assert sample.object_on_support and alignment == pytest.approx(0)
    assert sample.uncertainty_mm == pytest.approx(0.15)
    observer = TemporalGraspObserver(
        lambda: sample,
        limits=GraspLimits(1, 3, 5, 10, 2, 1, 1),
        current=lambda: sample.identity,
        clock=time.monotonic,
    )
    assert (
        observer.observe_grasp().state(
            sample.identity, now=time.monotonic(), limits=observer.limits
        )
        == GraspState.CONTACT_CANDIDATE
    )
    for frame in (2, 3, 4):
        sample, _ = measure(q, geometry(q, frame=frame), rise=5)
        evidence = observer.observe_grasp()
    assert (
        evidence.state(sample.identity, now=time.monotonic(), limits=observer.limits)
        == GraspState.HELD_VERIFIED
    )
    assert sample.jaws_mm[2] == pytest.approx(105)


@pytest.mark.parametrize("fault", ["none", "support", "gap", "slip", "lost", "relative"])
def test_retained_proof_during_descent_still_requires_continuous_measured_contact(fault):
    sample = None
    now = 10
    observer = TemporalGraspObserver(
        lambda: sample, limits=GraspLimits(1, 3, 5, 10, 2, 1, 1),
        current=lambda: sample.identity, clock=lambda: now,
    )
    for frame, rise in enumerate((0, 7, 7, 7, 4.5, -5), 1):
        now += 0.05
        if frame == 6 and fault == "gap":
            now += 2
        sample = GraspMeasurement(
            EvidenceIdentity("s", "scene", "q", now, (f"s:{frame}",)),
            (0, 0, 10 + rise), (0, 0, rise), 20, 0.1,
            True, True, True, True, True, True, False, frame == 1,
        )
        if frame == 6:
            if fault == "support":
                sample = replace(sample, object_on_support=True)
            elif fault == "slip":
                sample = replace(sample, width_mm=23)
            elif fault == "lost":
                sample = replace(sample, visible=False)
            elif fault == "relative":
                sample = replace(sample, object_mm=(4, 0, 10 + rise))
        evidence = observer.observe_grasp()
        if frame in (4, 5):
            assert evidence.state(sample.identity, now=now, limits=observer.limits) == GraspState.HELD_VERIFIED
    state = evidence.state(sample.identity, now=now, limits=observer.limits)
    assert (state == GraspState.HELD_VERIFIED) is (fault == "none")
    if fault == "support":
        assert state == GraspState.CONTACT_CANDIDATE and evidence.proof is None


@pytest.mark.parametrize(
    "fault",
    [
        "units",
        "physical",
        "reference",
        "duplicate",
        "holdout",
        "nan",
        "moving",
        "uncertainty",
        "force_disagreement",
    ],
)
def test_effort_admission_fails_closed(fault):
    data = effort_data()
    if fault == "units":
        data["units"] = "current"
    elif fault == "physical":
        data["physical"] = False
    elif fault == "reference":
        data["independent_reference"] = ""
    elif fault == "duplicate":
        data["holdout"][0]["id"] = data["training"][0]["id"]
    elif fault == "holdout":
        data["holdout"][-1]["effort_n"] = 0.1
    elif fault == "nan":
        data["training"][0]["effort_n"] = float("nan")
    elif fault == "moving":
        data["training"][0]["stationary"] = False
    elif fault == "uncertainty":
        data["uncertainty_n"] = 2
    elif fault == "force_disagreement":
        data["training"][-1]["reference_force_n"] = 10
    with pytest.raises(ValueError):
        EffortReference.admit(data, limits=LIMITS)


def test_positive_effort_and_uncharacterized_pose_never_establish_contact():
    reference = EffortReference.admit(effort_data(), limits=LIMITS)
    assert reference.classify(0.3, 20, Pose(0, 0, 0), limits=LIMITS) == "unknown"
    assert reference.classify(100, 20, Pose(0, 0, 0), limits=LIMITS) == "unknown"
    assert reference.classify(1, 20, Pose(0, 0, 100), limits=LIMITS) == "unknown"
    assert reference.classify(1, 30, Pose(0, 0, 0), limits=LIMITS) == "unknown"


@pytest.mark.parametrize(
    "fault",
    [
        "scope",
        "pads",
        "ambiguous",
        "dimensions",
        "unobservable",
        "axis",
        "residual",
        "configuration",
        "tip",
        "source_window",
    ],
)
def test_geometry_proposals_cannot_fabricate_qualified_contact(fault):
    q = qualified()
    data = geometry(q)
    if fault == "scope":
        data["pad_depth"][0]["mount_digest"] = "other"
    elif fault == "pads":
        data["pad_depth"][1]["qualified"] = False
    elif fault == "ambiguous":
        data["proposals"].append(copy.deepcopy(data["proposals"][0]))
    elif fault == "dimensions":
        data["dimensions_mm"]["box"][0] = 25
    elif fault == "unobservable":
        data["proposals"][0]["observable"] = False
    elif fault == "axis":
        data["proposals"][0]["rpy_deg"][2] = 30
    elif fault == "residual":
        data["proposals"][0]["surface_rms_mm"] = 5
    elif fault == "configuration":
        data["configuration_digest"] = "changed-selector"
    elif fault == "tip":
        data["jaw_points_camera_mm"][0][2] = 120
    elif fault == "source_window":
        data["depth"]["frame_ids"] = ["old-frame"]
    with pytest.raises(ValueError):
        measure(q, data)


def test_missing_object_is_not_empty_without_measured_closed_pads_and_unloaded_force():
    q = qualified()
    sample, _ = measure(q, geometry(q, visible=False))
    assert not sample.empty_closure
    sample, _ = measure(q, geometry(q, width=0.1, visible=False), width=0.1, effort=0)
    assert sample.empty_closure
    sample, _ = measure(q, geometry(q, width=0.1, visible=False), width=0.1, effort=1)
    assert not sample.empty_closure


@pytest.mark.parametrize("fault", ["stale", "duplicate", "sync", "jaws", "uncertainty"])
def test_empty_closure_cannot_bypass_source_validation(fault):
    sample = GraspMeasurement(
        EvidenceIdentity("s", "scene", "cal", 10, ("frame",)),
        (0, 0, 0),
        (0, 0, 0),
        0,
        0.1,
        False,
        True,
        True,
        False,
        True,
        False,
        True,
        False,
    )
    observer = TemporalGraspObserver(
        lambda: sample,
        limits=GraspLimits(1, 3, 5, 10, 2, 1, 1),
        current=lambda: sample.identity,
        clock=lambda: 10,
    )
    if fault == "duplicate":
        observer.observe_grasp()
    elif fault == "stale":
        sample = replace(sample, identity=replace(sample.identity, observed_mono=1))
    elif fault == "sync":
        sample = replace(sample, synchronized=False)
    elif fault == "jaws":
        sample = replace(sample, independent_jaws=False)
    elif fault == "uncertainty":
        sample = replace(sample, uncertainty_mm=2)
    assert (
        observer.observe_grasp().state(sample.identity, now=10, limits=observer.limits)
        == GraspState.UNKNOWN
    )


async def test_standalone_verify_remains_read_only_and_returns_failure_state():
    ctx = SimpleNamespace(
        grasp_verification=SimpleNamespace(
            verify=AsyncMock(return_value={"passed": False, "state": "unknown"})
        ),
        motion=SimpleNamespace(),
    )
    result = await GraspVerify().plan(ctx, {"stage": "hold", "samples": 3})
    assert not result.ok and result.side_effect_state == "none"
    assert GraspVerify.spec.effect == "read"


async def test_installation_requires_matching_read_only_admission_before_recording():
    ctx = SimpleNamespace(
        grasp_verification=SimpleNamespace(
            admit=lambda data: {"qualification_digest": "digest", "parameters": dict(data)}
        ),
        record_calibration=lambda *args, **kw: 123,
    )
    skill = InstallGraspEvidence()
    plan = await skill.plan(ctx, {"qualification": {"physical": True}})
    result = await skill.execute(ctx, plan)
    assert result.ok and result.result["recorded_at"] == 123
    ctx.grasp_verification.admit = lambda data: {
        "qualification_digest": "changed",
        "parameters": dict(data),
    }
    with pytest.raises(SkillError):
        await skill.execute(ctx, plan)


async def test_backend_preserves_quality_identity_and_source_time():
    now = time.monotonic()
    row = Reading("wrist", "camera.rgbd.geometry", {}, monotonic_at=now, quality="stale")
    registry = SimpleNamespace(read_batch=AsyncMock(return_value=BatchReading("wrist", (row,))))
    ctx = SimpleNamespace(workcell=SimpleNamespace(role=lambda role: "wrist", grasp_verify=LIMITS))
    backend = WorkcellGraspVerification(ctx, registry)
    with pytest.raises(ValueError, match="stale"):
        await backend._read("camera.wrist", ("camera.rgbd.geometry",))
    registry.read_batch.return_value = BatchReading(
        "wrist", (replace(row, quality="ok", device_id="other"),)
    )
    with pytest.raises(ValueError, match="misbound"):
        await backend._read("camera.wrist", ("camera.rgbd.geometry",))


async def test_backend_reuses_one_observer_and_serializes_finite_evidence(monkeypatch):
    q = qualified()
    ctx = SimpleNamespace(
        workcell=SimpleNamespace(grasp_verify=LIMITS),
        motion=SimpleNamespace(pause=AsyncMock()),
        note=lambda *args: None,
    )
    backend = WorkcellGraspVerification(ctx, SimpleNamespace())
    backend._qualification = lambda: q
    counter = 0
    rise = 0

    async def observe(_):
        nonlocal counter
        counter += 1
        return measure(q, geometry(q, frame=counter), rise=rise)

    backend._observe = observe
    contact = await backend.verify("hold", samples=3)
    assert contact["passed"] and contact["state"] == "contact_candidate"
    observer = backend._observer
    rise = 5
    proof = await backend.verify("hold", samples=3)
    assert proof["passed"] and proof["state"] == "held_verified"
    assert backend._observer is observer
    json.dumps(proof, allow_nan=False)


async def test_camera_warmup_discards_old_window_without_restamping(monkeypatch):
    monkeypatch.setattr("leapflow.hardware.skills.grasp_verification.time.monotonic", lambda: 100)
    old = {"frame_ids": ["s:1:1", "s:2:2"], "observation_times": [99.4, 99.8]}
    mature = {"frame_ids": ["s:3:3", "s:4:4"], "observation_times": [99.8, 99.9]}
    rows = [
        Reading("wrist", "camera.rgbd.quality", value, monotonic_at=100) for value in (old, mature)
    ]
    registry = SimpleNamespace(
        read_batch=AsyncMock(side_effect=[BatchReading("wrist", (r,)) for r in rows])
    )
    ctx = SimpleNamespace(
        workcell=SimpleNamespace(role=lambda role: "wrist", grasp_verify=LIMITS),
        motion=SimpleNamespace(pause=AsyncMock()),
    )
    backend = WorkcellGraspVerification(ctx, registry)
    backend._stage_at = 99.5
    result = await backend._capture("camera.rgbd.quality")
    assert result is rows[1] and result.monotonic_at == 100
    ctx.motion.pause.assert_awaited_once()
    assert backend._observer is None


@pytest.mark.parametrize(
    "fault", ["unordered", "future", "stale", "nan", "count", "duplicate", "misbound"]
)
async def test_camera_capture_window_fails_closed(fault, monkeypatch):
    monkeypatch.setattr("leapflow.hardware.skills.grasp_verification.time.monotonic", lambda: 100)
    data = {"frame_ids": ["s:1:1", "s:2:2"], "observation_times": [99.8, 99.9]}
    if fault == "unordered":
        data["observation_times"] = [99.9, 99.8]
    elif fault == "future":
        data["observation_times"] = [99.9, 101]
    elif fault == "stale":
        data["observation_times"] = [1, 2]
    elif fault == "nan":
        data["observation_times"][0] = float("nan")
    elif fault == "count":
        data["frame_ids"] = ["s:1:1"]
    elif fault == "duplicate":
        data["frame_ids"] = ["s:1:1", "s:1:1"]
    row = Reading(
        "other" if fault == "misbound" else "wrist", "camera.rgbd.quality", data, monotonic_at=100
    )
    registry = SimpleNamespace(read_batch=AsyncMock(return_value=BatchReading("wrist", (row,))))
    ctx = SimpleNamespace(
        workcell=SimpleNamespace(role=lambda role: "wrist", grasp_verify=LIMITS),
        motion=SimpleNamespace(pause=AsyncMock()),
    )
    backend = WorkcellGraspVerification(ctx, registry)
    backend._stage_at = 99.5
    with pytest.raises(ValueError):
        await backend._capture("camera.rgbd.quality")
    ctx.motion.pause.assert_not_awaited()


@pytest.mark.parametrize(
    "field",
    ["configuration_digest", "calibration_digest", "mount_digest", "profile_digest", "frame_ids"],
)
async def test_inspection_rejects_foreign_depth_scope(field):
    q = qualified()
    sample, alignment = measure(q, geometry(q))
    depth = {
        "qualified": True,
        "frame_ids": ["session:1:1"],
        "calibration_digest": q.package.digest,
        "mount_digest": "test-mount",
        "profile_digest": "profile",
        "configuration_digest": "test-depth-config",
    }
    depth[field] = ["foreign:1:1"] if field == "frame_ids" else "foreign"
    ctx = SimpleNamespace(
        workcell=SimpleNamespace(grasp_verify=LIMITS),
        frames=SimpleNamespace(versions={}),
        motion=SimpleNamespace(state=AsyncMock(return_value=SimpleNamespace(flange=Pose(0, 0, 0)))),
    )
    backend = WorkcellGraspVerification(ctx, SimpleNamespace())
    backend.qualification = lambda: {"qualification_digest": q.digest}
    backend._qualification = lambda: q
    backend._capture = AsyncMock(return_value=SimpleNamespace(value=depth))
    backend._observe = AsyncMock(return_value=(sample, alignment))
    with pytest.raises(SkillError, match="depth scope"):
        await backend.inspect()
