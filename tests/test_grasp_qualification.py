"""Independent hold/calibration evidence, not width or image-IoU certificates."""
from dataclasses import asdict, replace
import json

import pytest

from leapflow.hardware.grasp_evidence import (
    EvidenceIdentity, GraspEvidence, GraspLimits, GraspMeasurement, GraspState, TemporalGraspObserver,
)
from leapflow.hardware.calibration_qualification import (
    COMPONENTS, CalibrationQualification, QualificationLimits, admit_package, package_digest, tool_offset_observable,
)


IDENTITY = EvidenceIdentity("arm-session", "scene", "cal", 10.0, ("rgbd:1", "rgbd:2", "rgbd:3"))
LIMITS = GraspLimits(1, 3, 5, 10, 2, 1, 1)


def held():
    return GraspEvidence(IDENTITY, True, True, True, True, True, True, False, False,
                         5.0, 5.0, .2, .1, .5, (9.8, 9.9, 10.0))


@pytest.mark.parametrize("change,state", [
    ({}, GraspState.HELD_VERIFIED), ({"visible": False}, GraspState.UNKNOWN),
    ({"independent_jaws": False}, GraspState.UNKNOWN), ({"synchronized": False}, GraspState.UNKNOWN),
    ({"object_on_support": True}, GraspState.CONTACT_CANDIDATE),
    ({"object_rise_mm": 0}, GraspState.CONTACT_CANDIDATE),
    ({"tool_rise_mm": 20}, GraspState.CONTACT_CANDIDATE),
    ({"width_consistent": False}, GraspState.UNKNOWN),
    ({"width_change_mm": 3}, GraspState.SLIPPED), ({"empty_closure": True}, GraspState.EMPTY),
    ({"relative_error_mm": float("nan")}, GraspState.UNKNOWN),
    ({"uncertainty_mm": 3}, GraspState.UNKNOWN),
    ({"identity": replace(IDENTITY, frame_ids=("rgbd:1",))}, GraspState.CONTACT_CANDIDATE),
])
def test_verified_hold_requires_joint_visual_and_motion_evidence(change, state):
    assert replace(held(), **change).state(IDENTITY, now=10.1, limits=LIMITS) == state


@pytest.mark.parametrize("identity", [
    replace(IDENTITY, session_id="reboot"), replace(IDENTITY, scene_id="moved"),
    replace(IDENTITY, calibration_digest="remounted"), replace(IDENTITY, observed_mono=1),
    replace(IDENTITY, frame_ids=("same", "same")),
])
def test_stale_stage_identity_is_unknown(identity):
    assert replace(held(), identity=identity).state(IDENTITY, now=10.1, limits=LIMITS) == GraspState.UNKNOWN


def certificate(component="camera_to_base"):
    return CalibrationQualification(component, "v1", "mount", "mm", "Rz@Ry@Rx;degrees",
                                    ("train:1", "train:2"), ("rejected:outlier",),
                                    ("heldout:1", "heldout:2"), (.5, .8), .2,
                                    "independent-fixture", True, QualificationLimits(2, 2, 2, 3, 1))


def test_geometry_is_part_of_certificate_and_package_identity():
    cert = certificate()
    assert not cert.qualified
    cert = cert.bind_geometry({"offset_mm": [0, 0, 150]})
    assert cert.qualified
    record = cert.record(device_id="arm", recorded_at=1, parameters={"offset_mm": [0, 0, 150]})
    assert record.parameters["qualified"]
    with pytest.raises(ValueError, match="geometry differs"):
        cert.record(device_id="arm", recorded_at=1, parameters={"offset_mm": [0, 0, 128.42]})
    certs = [certificate(c).bind_geometry({"value_mm": 1}) for c in COMPONENTS]
    first = package_digest(certs)
    certs[0] = certs[0].bind_geometry({"value_mm": 2})
    assert package_digest(certs) != first
    with pytest.raises(ValueError, match="six"):
        package_digest(certs[:-1])
    with pytest.raises(ValueError, match="mixed-mount"):
        package_digest([replace(certs[0], mount_digest="other"), *certs[1:]])


def test_outlier_and_holdout_ids_are_disjoint():
    with pytest.raises(ValueError, match="disjoint"):
        replace(certificate(), holdout_ids=("train:1", "heldout:2"))
    assert not replace(certificate(), observable=False).bind_geometry({"value_mm": 1}).qualified
    assert not replace(certificate(), holdout_errors_mm=(4, 5)).bind_geometry({"value_mm": 1}).qualified


def test_fixed_orientation_and_single_axis_cannot_solve_unknown_tcp():
    kwargs = dict(independent_offset=False, min_orientation_span_deg=5)
    assert not tool_offset_observable([(0, 0, 0)] * 12, **kwargs)
    assert not tool_offset_observable([(0, 0, 0), (0, 0, 30), (0, 0, 60)], **kwargs)
    assert tool_offset_observable([(0, 0, 0), (30, 0, 0), (0, 30, 0), (0, 0, 30)], **kwargs)
    assert tool_offset_observable([(0, 0, 0)] * 12, independent_offset=True, min_orientation_span_deg=5)


def measurement(index, at, *, rise=0, support=True, **changes):
    identity = EvidenceIdentity("arm-session", "scene", "cal", at,
                                (f"rgb:{index}", f"depth:{index}"))
    sample = GraspMeasurement(identity, (0, 0, 100 + rise), (0, 0, 110 + rise),
                              56.9, .1, True, True, True, True, True, True, False, support)
    return replace(sample, **changes)


class ObserverHarness:
    def __init__(self):
        self.sample = measurement(1, 10)
        self.now = 10
        self.current = IDENTITY
        self.observer = TemporalGraspObserver(lambda: self.sample, limits=LIMITS,
                                              current=lambda: self.current, clock=lambda: self.now)

    def read(self, sample, *, now=None):
        self.sample, self.now = sample, sample.identity.observed_mono if now is None else now
        evidence = self.observer.observe_grasp()
        return evidence.state(self.current, now=self.now, limits=LIMITS)

    def proof(self):
        assert self.read(measurement(1, 10)) == GraspState.CONTACT_CANDIDATE
        assert self.read(measurement(2, 10.1, rise=5, support=False)) == GraspState.CONTACT_CANDIDATE
        assert self.read(measurement(3, 10.2, rise=5, support=False)) == GraspState.CONTACT_CANDIDATE
        assert self.read(measurement(4, 10.3, rise=5, support=False)) == GraspState.HELD_VERIFIED


def test_bundle_members_and_support_frames_are_not_independent_lift_observations():
    harness = ObserverHarness()
    for index in range(1, 5):
        assert harness.read(measurement(index, 10 + index / 10)) == GraspState.CONTACT_CANDIDATE
    assert harness.read(measurement(5, 10.5, rise=5, support=False)) == GraspState.CONTACT_CANDIDATE
    assert replace(held(), observation_times=()).state(IDENTITY, now=10.1, limits=LIMITS) == GraspState.CONTACT_CANDIDATE


def test_fresh_continuous_carry_retains_a_verified_bounded_proof():
    harness = ObserverHarness()
    harness.proof()
    assert harness.read(measurement(5, 10.4, rise=30, support=False)) == GraspState.HELD_VERIFIED
    assert harness.read(measurement(6, 10.5, rise=40, support=False)) == GraspState.HELD_VERIFIED
    assert harness.observer._proof.proof is None


@pytest.mark.parametrize("fault", ["duplicate", "retired", "stale", "gap", "engagement", "occluded", "reboot", "slip"])
def test_uncertain_carry_invalidates_proof_without_automatic_requalification(fault):
    harness = ObserverHarness()
    harness.proof()
    sample = measurement(5, 10.4, rise=30, support=False)
    now = None
    if fault in ("duplicate", "retired"):
        sample = measurement(4 if fault == "duplicate" else 1, 10.4, rise=30, support=False)
    elif fault == "stale":
        now = 12
    elif fault == "gap":
        sample = measurement(5, 12, rise=30, support=False)
    elif fault == "engagement":
        sample = replace(sample, pad_engagement=False)
    elif fault == "occluded":
        sample = replace(sample, visible=False)
    elif fault == "reboot":
        harness.current = replace(IDENTITY, session_id="reboot")
    elif fault == "slip":
        sample = replace(sample, width_mm=60)
    state = harness.read(sample, now=now)
    assert state in (GraspState.UNKNOWN, GraspState.SLIPPED)
    assert harness.observer._proof is None
    assert harness.read(measurement(6, harness.now + .1, rise=30, support=False)) != GraspState.HELD_VERIFIED


def test_initial_lift_above_approved_budget_never_certifies_a_hold():
    harness = ObserverHarness()
    harness.read(measurement(1, 10))
    assert harness.read(measurement(2, 10.1, rise=30, support=False)) == GraspState.UNKNOWN


def test_old_window_cannot_be_restamped_by_the_latest_frame():
    assert replace(held(), observation_times=(1, 9.9, 10)).state(IDENTITY, now=10.1, limits=LIMITS) == GraspState.UNKNOWN


def package_data():
    geometry = {"offset_mm": [0, 0, 150]}
    matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    return {"components": [
        {"qualification": asdict(certificate(c).bind_geometry(geometry, matrix)),
         "geometry": dict(geometry), "matrix": matrix}
        for c in sorted(COMPONENTS)
    ]}


def test_calibration_admission_binds_active_geometry_and_uses_versioned_store(tmp_path):
    from leapflow.hardware.calibration_store import CalibrationStore
    data = package_data()
    package = admit_package(data, device_id="camera", recorded_at=10)
    package.require_geometry("camera_to_base", data["components"][0]["geometry"],
                             data["components"][0]["matrix"])
    with pytest.raises(ValueError, match="active"):
        package.require_geometry("camera_to_base", {"offset_mm": [0, 0, 128.42]})
    store = CalibrationStore(db_path=tmp_path / "instrument.duckdb")
    try:
        assert all(store.record(row) for row in package.records)
        record = store.latest("camera", "camera_to_base")
        assert record.parameters["qualified"]
        assert record.parameters["digest"] == package.certificates[0].digest
        data["components"][0]["geometry"]["offset_mm"] = [0, 0, 128.42]
        assert package.records[0].parameters["geometry"]["offset_mm"] == [0, 0, 150]
    finally:
        store.close()


@pytest.mark.parametrize("fault", ["missing", "duplicate", "geometry", "matrix", "outliers",
                                   "observable", "truthy", "units", "nan", "budget"])
def test_malformed_or_unqualified_calibration_package_is_refused(fault):
    data = json.loads(json.dumps(package_data()))
    row = data["components"][0]
    kwargs = {}
    if fault == "missing":
        data["components"].pop()
    elif fault == "duplicate":
        data["components"][-1] = row
    elif fault == "geometry":
        row["geometry"]["offset_mm"][2] = 128.42
    elif fault == "matrix":
        row["matrix"][0][3] = 5
    elif fault == "outliers":
        row["qualification"]["holdout_errors_mm"] = [5, 5]
    elif fault in ("observable", "truthy"):
        row["qualification"]["observable"] = False if fault == "observable" else "true"
    elif fault == "units":
        row["qualification"]["units"] = "m"
    elif fault == "nan":
        row["qualification"]["uncertainty_mm"] = float("nan")
    elif fault == "budget":
        kwargs["max_bytes"] = 16
    with pytest.raises(ValueError):
        admit_package(data, device_id="camera", recorded_at=10, **kwargs)


def test_source_failure_invalidates_hold_proof_and_preserves_exception():
    harness = ObserverHarness()
    harness.proof()
    def fail():
        raise RuntimeError("observation unavailable")
    harness.observer.source = fail
    with pytest.raises(RuntimeError, match="observation unavailable"):
        harness.observer.observe_grasp()
    assert harness.observer._proof is None
    harness.observer.source = lambda: harness.sample
    assert harness.read(measurement(5, 10.4, rise=30, support=False)) == GraspState.UNKNOWN


def test_new_support_observation_refreshes_baseline_not_old_table_position():
    harness = ObserverHarness()
    harness.read(measurement(1, 10))
    harness.read(measurement(2, 10.1, rise=20))
    for index in (3, 4, 5):
        state = harness.read(measurement(index, 10.1 + index / 10, rise=25, support=False))
    assert state == GraspState.HELD_VERIFIED


@pytest.mark.parametrize("flag", ["visible", "synchronized", "independent_jaws", "pad_engagement",
                                  "width_consistent", "motion_conditioned_effort", "empty_closure"])
def test_truthy_untyped_flags_never_certify_grasp(flag):
    assert replace(held(), **{flag: "true"}).state(IDENTITY, now=10, limits=LIMITS) == GraspState.UNKNOWN
