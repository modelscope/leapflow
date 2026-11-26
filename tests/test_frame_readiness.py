# Copyright (c) Alibaba, Inc. and its affiliates.
"""Frame diagnosis, provenance, readiness, and promote/attest without re-measuring.

The user story: a cell was calibrated, nothing moved, yet a grasp is refused
because one camera's record predates the evidence a record now carries. The
fix must not be "calibrate again": the readiness solver should name the
cheapest safe step (vouch for the record against an already verified camera),
and that step must leave a record the frame graph then trusts.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from leapflow.hardware.calibration.frame_lifecycle import WorkcellFrameLifecycle
from leapflow.hardware.calibration_store import CalibrationRecord, CalibrationStore
from leapflow.hardware.skill_results import frame_status_view, workcell_view
from leapflow.hardware.skills.builtin.camera_calibration import skills as camera_skills
from leapflow.hardware.skills.builtin.frame_lifecycle import skills as frame_lifecycle_skills
from leapflow.hardware.skills.builtin.grasp_verify import InstallGraspEvidence
from leapflow.hardware.skills.frame_status import Accuracy, FrameCandidate, FrameStatus
from leapflow.hardware.skills.frames import FrameGraph, unmet_requirements
from leapflow.hardware.skills.geometry import Pose, Transform
from leapflow.hardware.skills.protocol import (
    EFFECT_READ,
    METHOD_MARKERLESS_SCREW,
    SkillError,
    SkillPlan,
    SkillSpec,
)
from leapflow.hardware.skills.provenance import (
    METHOD_CANDIDATE_PROMOTION,
    judge,
)
from leapflow.hardware.skills.readiness import WorkcellReadiness
from leapflow.hardware.skills.workcell import WorkcellError, WorkcellSpec

pytestmark = pytest.mark.component

CELL_ID = "bench_cell"
FIXED, WRIST = "camera", "camera.wrist"
FIXED_PROC, WRIST_PROC = "handeye.fixed", "handeye.wrist"
FLANGE = Pose(250.0, 0.0, 370.0, 180.0, 0.0, 0.0)
FIXED_MOUNT = Transform.from_pose(Pose(600.0, 50.0, 700.0, 180.0, 0.0, 90.0))
WRIST_MOUNT = Transform.from_pose(Pose(0.0, 60.0, 40.0, 0.0, 0.0, 0.0))
BOARD = Transform.from_pose(Pose(300.0, -40.0, 0.0, 0.0, 0.0, 15.0))
IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _cell(**overrides: Any) -> WorkcellSpec:
    """A two-camera cell that is deliberately not a Piper: generic ids, own procedures."""
    data: dict[str, Any] = {
        "workcell": CELL_ID,
        "roles": {"arm": "arm0", FIXED: "cam_fixed", WRIST: "cam_wrist"},
        "frames": {
            FIXED: {"parent": "base", "procedure": FIXED_PROC, "accuracy": {"translation_mm": 5.0}},
            WRIST: {"parent": "flange", "procedure": WRIST_PROC},
        },
        "calibration": {"max_verify_p95_mm": 8.0},
        "calibration_target": {"squares_x": 5, "squares_y": 7, "square_mm": 30.0, "marker_mm": 22.0},
    }
    data.update(overrides)
    return WorkcellSpec.from_mapping(data)


class _Store:
    """A real calibration store with a terse writer."""

    def __init__(self, path: Path) -> None:
        self.store = CalibrationStore(db_path=path)
        self._clock = 1_000.0

    def put(
        self,
        procedure: str,
        parameters: Mapping[str, Any] | None = None,
        matrix: Any = None,
        at: float | None = None,
    ) -> float:
        self._clock = at if at is not None else self._clock + 10.0
        assert self.store.record(
            CalibrationRecord(
                device_id=CELL_ID,
                procedure_id=procedure,
                recorded_at=self._clock,
                parameters=dict(parameters or {}),
                matrix=None if matrix is None else [list(row) for row in matrix],
            )
        )
        return self._clock


@pytest.fixture
def store(tmp_path: Path) -> _Store:
    return _Store(tmp_path / "instrument.duckdb")


def _legacy_fixed(store: _Store, mount: Transform = FIXED_MOUNT) -> float:
    """What a board-free fit wrote before it kept its screw evidence."""
    return store.put(FIXED_PROC, {"method": METHOD_MARKERLESS_SCREW}, mount.to_matrix())


def _verified_wrist(store: _Store, achieved_mm: float = 2.5) -> float:
    version = store.put(
        WRIST_PROC,
        {"target_mount": IDENTITY, "achieved_accuracy": {"translation_mm": achieved_mm}},
        WRIST_MOUNT.to_matrix(),
    )
    store.put(f"{WRIST}.verify", {"training_disjoint": True, "p95_mm": achieved_mm,
                                 "depends_on": {WRIST: version}})
    return version


def _candidate(dependency: Mapping[str, float], achieved_mm: float, digest: str = "a" * 64) -> FrameCandidate:
    return FrameCandidate(
        digest=digest,
        frame=FIXED,
        method="stationary_board_bridge.v1",
        matrix=tuple(tuple(row) for row in FIXED_MOUNT.to_matrix()),
        achieved=Accuracy(achieved_mm, 0.5),
        created_at=900.0,
        dependency=dict(dependency),
    )


def _requester(*requires: str) -> SkillSpec:
    return SkillSpec(
        skill_id="task.pick",
        version="1.0.0",
        card="A task that needs the fixed camera.",
        args_schema={"type": "object", "properties": {}},
        effect=EFFECT_READ,
        requires=requires,
    )


# ── Provenance ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("parameters", "mounted", "status"),
    [
        ({"method": METHOD_MARKERLESS_SCREW}, True, FrameStatus.LEGACY_EVIDENCE),
        ({}, True, FrameStatus.UNPROVEN),
        ({"method": "vendor_magic"}, True, FrameStatus.UNPROVEN),
        ({"target_mount": IDENTITY}, True, FrameStatus.VALID),
        ({"target_mount": "nope"}, True, FrameStatus.UNPROVEN),
        ({}, False, FrameStatus.VALID),  # a declared offset is the operator's statement
    ],
)
def test_a_record_is_judged_by_the_evidence_it_carries(parameters, mounted, status) -> None:
    record = SimpleNamespace(parameters=parameters, matrix=IDENTITY)
    assert judge(record, mounted=mounted).status is status


def test_a_promotion_must_install_exactly_the_candidate_it_names() -> None:
    promotion = {"candidate_digest": "b" * 64, "matrix": IDENTITY}
    record = SimpleNamespace(
        parameters={"method": METHOD_CANDIDATE_PROMOTION, "promotion": promotion,
                    "achieved_accuracy": {"translation_mm": 3.0}},
        matrix=IDENTITY,
    )
    verdict = judge(record, mounted=True)
    assert verdict.status is FrameStatus.VALID and verdict.achieved == Accuracy(3.0)
    tampered = SimpleNamespace(parameters=record.parameters, matrix=FIXED_MOUNT.to_matrix())
    assert judge(tampered, mounted=True).status is FrameStatus.UNPROVEN


def test_a_broken_validator_refuses_rather_than_trusts() -> None:
    class Broken:
        name = "broken"

        def applies(self, parameters: Mapping[str, Any]) -> bool:
            raise RuntimeError("boom")

        def check(self, record: Any) -> Any:  # pragma: no cover - never reached
            raise AssertionError

    record = SimpleNamespace(parameters={"target_mount": IDENTITY}, matrix=IDENTITY)
    assert judge(record, mounted=True, validators=[Broken()]).status is FrameStatus.UNPROVEN


# ── Diagnosis ────────────────────────────────────────────────────────────────


def test_a_legacy_record_is_diagnosed_not_called_stale(store: _Store) -> None:
    spec = _cell()
    _legacy_fixed(store)
    _verified_wrist(store)
    frames = FrameGraph.snapshot(spec, store.store)
    fixed = frames.diagnosis(FIXED)
    assert fixed.status is FrameStatus.LEGACY_EVIDENCE and frames.transform(FIXED) is None
    assert fixed.required == Accuracy(5.0)  # the frame's own declaration wins
    assert frames.diagnosis(WRIST).ok and frames.required_accuracy(WRIST) == Accuracy(8.0)
    assert frame_status_view({k: v.to_dict() for k, v in frames.diagnoses.items()}) == {
        FIXED: {"status": "legacy_evidence", "reason": fixed.reason, "required": {"translation_mm": 5.0}},
        WRIST: {"status": "valid"},
    }


def test_an_independent_check_bound_to_the_version_vouches_for_it(store: _Store) -> None:
    spec = _cell()
    version = _legacy_fixed(store)
    store.put(f"{FIXED}.verify", {"passed": True, "training_disjoint": True, "p95_mm": 3.2,
                                  "depends_on": {FIXED: version}})
    diagnosis = FrameGraph.snapshot(spec, store.store).diagnosis(FIXED)
    assert diagnosis.ok and diagnosis.verified_by == f"{FIXED}.verify"
    assert diagnosis.achieved == Accuracy(3.2)


def test_a_check_of_another_version_or_not_held_out_vouches_for_nothing(store: _Store) -> None:
    spec = _cell()
    version = _legacy_fixed(store)
    store.put(f"{FIXED}.verify", {"passed": True, "training_disjoint": True, "p95_mm": 1.0,
                                  "depends_on": {FIXED: version - 5.0}})
    assert FrameGraph.snapshot(spec, store.store).diagnosis(FIXED).status is FrameStatus.LEGACY_EVIDENCE
    store.put(f"{FIXED}.verify", {"passed": True, "p95_mm": 1.0, "depends_on": {FIXED: version}})
    assert FrameGraph.snapshot(spec, store.store).diagnosis(FIXED).status is FrameStatus.LEGACY_EVIDENCE


def test_a_newer_failed_attestation_marks_the_frame_drifted(store: _Store) -> None:
    spec = _cell()
    version = _legacy_fixed(store)
    store.put(f"{FIXED}.verify", {"passed": True, "training_disjoint": True, "p95_mm": 2.0,
                                  "depends_on": {FIXED: version}})
    store.put(f"{FIXED}.attestation", {"passed": False, "reason": "moved 30 mm",
                                       "depends_on": {FIXED: version}})
    frames = FrameGraph.snapshot(spec, store.store)
    assert frames.diagnosis(FIXED).status is FrameStatus.DRIFT_DETECTED
    assert frames.diagnosis(FIXED).reason == "moved 30 mm" and frames.transform(FIXED) is None


def test_a_check_coarser_than_required_leaves_the_frame_unusable(store: _Store) -> None:
    spec = _cell()
    version = _legacy_fixed(store)
    store.put(f"{FIXED}.verify", {"passed": True, "training_disjoint": True, "p95_mm": 7.0,
                                  "depends_on": {FIXED: version}})
    diagnosis = FrameGraph.snapshot(spec, store.store).diagnosis(FIXED)
    assert diagnosis.status is FrameStatus.INSUFFICIENT_ACCURACY
    assert "required 5.0 mm" in diagnosis.reason


def test_candidates_solved_through_a_replaced_frame_are_dropped(store: _Store) -> None:
    spec = _cell()
    _legacy_fixed(store)
    wrist = _verified_wrist(store)
    live, orphan = _candidate({WRIST: wrist}, 3.0), _candidate({WRIST: wrist - 1.0}, 3.0, "c" * 64)
    frames = FrameGraph.snapshot(spec, store.store, candidates={FIXED: (live, orphan)})
    assert frames.candidates(FIXED) == (live,)


def test_an_installed_candidate_is_no_longer_on_offer(store: _Store) -> None:
    spec = _cell()
    wrist = _verified_wrist(store)
    candidate = _candidate({WRIST: wrist}, 3.0)
    store.put(FIXED_PROC, {"method": METHOD_CANDIDATE_PROMOTION, "depends_on": {WRIST: wrist},
                           "promotion": {"candidate_digest": candidate.digest,
                                         "matrix": [list(r) for r in candidate.matrix]}},
              candidate.matrix)
    frames = FrameGraph.snapshot(spec, store.store, candidates={FIXED: (candidate,)})
    assert frames.diagnosis(FIXED).ok and frames.candidates(FIXED) == ()


@pytest.mark.parametrize(
    "accuracy",
    [{"translation_mm": 0}, {"translation_mm": -1}, {"rotation_deg": 1.0}, {"translation_mm": 2, "x": 1},
     {"translation_mm": True}, "5mm"],
)
def test_a_malformed_frame_accuracy_is_refused_at_admission(accuracy: Any) -> None:
    frames = {FIXED: {"parent": "base", "procedure": FIXED_PROC, "accuracy": accuracy}}
    with pytest.raises(WorkcellError, match="accuracy"):
        _cell(frames=frames)


def test_an_undeclared_accuracy_keeps_the_declaration_digest_stable() -> None:
    frames = {FIXED: {"parent": "base", "procedure": FIXED_PROC}}
    assert "accuracy" not in _cell(frames=frames).to_dict()["frames"][FIXED]


# ── Readiness ────────────────────────────────────────────────────────────────


def _catalog() -> list[Any]:
    return [*camera_skills(), *frame_lifecycle_skills()]


def test_readiness_offers_vouching_before_recalibration(store: _Store) -> None:
    spec = _cell()
    _legacy_fixed(store)
    _verified_wrist(store)
    frames = FrameGraph.snapshot(spec, store.store)
    readiness = WorkcellReadiness(_catalog(), spec, frames)
    explained = readiness.explain(_requester(f"calib:{FIXED}", f"record:{FIXED}.verify"))
    assert not explained.ready
    assert [b.status for b in explained.blockers] == ["legacy_evidence", ""]
    first = explained.next_steps[0]
    assert (first.skill_id, dict(first.args), first.motion) == (
        "calib.attest", {"frame": FIXED, "reference": WRIST}, False
    )
    # One attestation meets both blockers: the combined plan does not repeat it.
    assert explained.compact() == [f"run calib.attest frame={FIXED} reference={WRIST}"]
    options = explained.blockers[0].options
    assert options[0].motion is False and any(option.motion for option in options[1:])
    assert readiness.hint(f"calib:{FIXED}") == explained.compact()[0]


def test_a_coarse_candidate_is_promoted_then_attested(store: _Store) -> None:
    spec = _cell()
    wrist = _verified_wrist(store)  # the fixed camera has no record at all
    frames = FrameGraph.snapshot(spec, store.store, candidates={FIXED: (_candidate({WRIST: wrist}, 22.6),)})
    explained = WorkcellReadiness(_catalog(), spec, frames).explain(_requester(f"calib:{FIXED}"))
    assert explained.blockers[0].status == "missing"
    best = explained.blockers[0].options[0]
    assert [step.skill_id for step in best.steps] == ["calib.promote", "calib.attest"]
    assert best.steps[0].to_dict()["expected_status"] == "insufficient_accuracy"
    assert best.motion is False


def test_attestation_is_not_offered_for_a_frame_that_drifted(store: _Store) -> None:
    spec = _cell()
    version = _legacy_fixed(store)
    _verified_wrist(store)
    store.put(f"{FIXED}.attestation", {"passed": False, "reason": "moved", "depends_on": {FIXED: version}})
    frames = FrameGraph.snapshot(spec, store.store)
    explained = WorkcellReadiness(_catalog(), spec, frames).explain(_requester(f"calib:{FIXED}"))
    assert explained.blockers[0].status == "drift_detected"
    assert all(step.skill_id != "calib.attest" for o in explained.blockers[0].options for step in o.steps)
    assert explained.next_steps and explained.next_steps[0].motion  # only a real re-measurement fixes it


def test_without_a_reference_the_refusal_names_the_measurement(store: _Store) -> None:
    spec = _cell()
    _legacy_fixed(store)  # no verified wrist camera to compare with
    frames = FrameGraph.snapshot(spec, store.store)
    readiness = WorkcellReadiness(frame_lifecycle_skills(), spec, frames)
    assert readiness.explain(_requester(f"calib:{FIXED}")).next_steps == ()
    assert unmet_requirements((f"calib:{FIXED}",), spec, frames, readiness.hint)[0].endswith(
        f"record {FIXED_PROC}"
    )


def test_the_catalog_card_carries_the_next_step_and_status() -> None:
    cell = {
        "workcell": CELL_ID,
        "skills": [{"skill_id": "task.pick", "available": False, "reason": "camera is legacy",
                    "next": ["run calib.attest frame=camera reference=camera.wrist"],
                    "readiness": {"skill_id": "task.pick", "ready": False}}],
        "calibration_status": {FIXED: {"status": "legacy_evidence", "reason": "old"}, WRIST: {"status": "valid"}},
    }
    listed = workcell_view("node", cell)
    assert listed["skills"][0]["next"] == ["run calib.attest frame=camera reference=camera.wrist"]
    assert "readiness" not in listed["skills"][0]
    assert listed["calibration_status"][WRIST] == {"status": "valid"}
    assert workcell_view("node", cell, skill="task.pick")["skills"][0]["readiness"]["ready"] is False


# ── Promote / attest ─────────────────────────────────────────────────────────


class _View:
    """A camera that sees the board where the true mount puts it."""

    def __init__(self, to_base: Transform) -> None:
        self._seen = to_base.inverse().compose(BOARD)

    async def observe_target(self) -> Any:
        return SimpleNamespace(transform=self._seen)


class _Ctx:
    """The run-context surface the frame lifecycle uses, over a real store."""

    def __init__(self, spec: WorkcellSpec, store: _Store, *, candidates: Mapping[str, Any] | None = None,
                 fixed_truth: Transform = FIXED_MOUNT) -> None:
        self.workcell, self._store, self._candidates = spec, store, dict(candidates or {})
        flange = Transform.from_pose(FLANGE)
        self._views = {FIXED: _View(fixed_truth), WRIST: _View(flange.compose(WRIST_MOUNT))}
        self.motion = SimpleNamespace(state=self._state)
        self.refresh()

    def refresh(self) -> None:
        self.frames = FrameGraph.snapshot(self.workcell, self._store.store, candidates=self._candidates)

    async def _state(self) -> Any:
        return SimpleNamespace(flange=FLANGE)

    def camera_vision(self, role: str) -> _View:
        return self._views[role]

    def record_calibration(self, procedure_id: str, *, role: str = "", parameters: Any = None,
                           matrix: Any = None, pose: Any = None, notes: str = "") -> float:
        assert not role, "frame records are the workcell's, never a device's"
        return self._store.put(procedure_id, parameters, matrix)


def _registry(store: _Store) -> Any:
    return SimpleNamespace(calibration_store=store.store)


def _attest(ctx: _Ctx, store: _Store) -> Mapping[str, Any]:
    port = WorkcellFrameLifecycle(ctx, _registry(store))
    args = {"frame": FIXED, "reference": WRIST, "samples": 2}
    basis = asyncio.run(port.attestation_basis(args))
    return asyncio.run(port.attest(SkillPlan("calib.attest", args, "", basis={"attestation": basis})))


def test_attesting_an_unmoved_camera_restores_it_without_motion(store: _Store) -> None:
    spec = _cell()
    _legacy_fixed(store)
    _verified_wrist(store, achieved_mm=2.5)
    ctx = _Ctx(spec, store)
    result = _attest(ctx, store)
    assert result["p95_mm"] < 1e-3 and result["achieved"] == {"translation_mm": 2.5}
    ctx.refresh()
    diagnosis = ctx.frames.diagnosis(FIXED)
    assert diagnosis.ok and diagnosis.verified_by == f"{FIXED}.verify"
    assert ctx.frames.transform(FIXED) is not None
    listed = FrameGraph.snapshot(spec, store.store, (f"{FIXED}.verify",))
    assert unmet_requirements((f"calib:{FIXED}", f"record:{FIXED}.verify"), spec, listed) == []


def test_attesting_a_moved_camera_marks_it_drifted(store: _Store) -> None:
    spec = _cell()
    _legacy_fixed(store)
    _verified_wrist(store)
    moved = Transform.from_pose(Pose(630.0, 50.0, 700.0, 180.0, 0.0, 90.0))
    ctx = _Ctx(spec, store, fixed_truth=moved)
    with pytest.raises(SkillError, match="disagree"):
        _attest(ctx, store)
    ctx.refresh()
    assert ctx.frames.diagnosis(FIXED).status is FrameStatus.DRIFT_DETECTED


def test_a_reference_too_coarse_to_vouch_writes_nothing(store: _Store) -> None:
    spec = _cell()
    _legacy_fixed(store)
    _verified_wrist(store, achieved_mm=6.0)  # within the cell's 8 mm, above the fixed camera's 5 mm
    ctx = _Ctx(spec, store)
    with pytest.raises(SkillError, match="tighten"):
        _attest(ctx, store)
    ctx.refresh()
    assert ctx.frames.diagnosis(FIXED).status is FrameStatus.LEGACY_EVIDENCE


def test_attestation_refuses_a_plan_made_against_another_version(store: _Store) -> None:
    spec = _cell()
    _legacy_fixed(store)
    _verified_wrist(store)
    ctx = _Ctx(spec, store)
    port = WorkcellFrameLifecycle(ctx, _registry(store))
    args = {"frame": FIXED, "reference": WRIST}
    basis = asyncio.run(port.attestation_basis(args))
    _legacy_fixed(store)  # recalibrated meanwhile
    ctx.refresh()
    with pytest.raises(SkillError, match="plan again"):
        asyncio.run(port.attest(SkillPlan("calib.attest", args, "", basis={"attestation": basis})))


def test_promoting_a_candidate_installs_it_with_its_accuracy(store: _Store) -> None:
    spec = _cell()
    wrist = _verified_wrist(store)
    candidate = _candidate({WRIST: wrist}, 3.0)
    ctx = _Ctx(spec, store, candidates={FIXED: (candidate,)})
    port = WorkcellFrameLifecycle(ctx, _registry(store))
    basis = asyncio.run(port.promotion_basis({"frame": FIXED}))
    assert basis["predicted_status"] == "valid"
    result = asyncio.run(port.promote(SkillPlan("calib.promote", {"frame": FIXED}, "", basis={"promotion": basis})))
    assert result["candidate_digest"] == candidate.digest
    ctx.refresh()
    diagnosis = ctx.frames.diagnosis(FIXED)
    assert diagnosis.ok and diagnosis.method == METHOD_CANDIDATE_PROMOTION
    assert diagnosis.achieved == Accuracy(3.0, 0.5) and ctx.frames.candidates(FIXED) == ()


def test_promoting_a_coarse_candidate_predicts_it_still_needs_a_check(store: _Store) -> None:
    spec = _cell()
    wrist = _verified_wrist(store)
    ctx = _Ctx(spec, store, candidates={FIXED: (_candidate({WRIST: wrist}, 22.6),)})
    port = WorkcellFrameLifecycle(ctx, _registry(store))
    basis = asyncio.run(port.promotion_basis({"frame": FIXED}))
    assert basis["predicted_status"] == "insufficient_accuracy"
    asyncio.run(port.promote(SkillPlan("calib.promote", {"frame": FIXED}, "", basis={"promotion": basis})))
    ctx.refresh()
    assert ctx.frames.diagnosis(FIXED).status is FrameStatus.INSUFFICIENT_ACCURACY


def test_promotion_with_no_admissible_candidate_is_refused(store: _Store) -> None:
    spec = _cell()
    _verified_wrist(store)
    port = WorkcellFrameLifecycle(_Ctx(spec, store), _registry(store))
    with pytest.raises(SkillError, match="no admissible camera candidate"):
        asyncio.run(port.promotion_basis({"frame": FIXED}))


@pytest.mark.parametrize("change", ["version", "drift", "invalid"])
@pytest.mark.parametrize("own_evidence", [False, True])
def test_cross_view_check_loses_changed_reference_but_preserves_own_geometry(
    store: _Store, change: str, own_evidence: bool,
) -> None:
    subject = (_verified_wrist(store) if own_evidence else
               store.put(WRIST_PROC, {"method": METHOD_MARKERLESS_SCREW}, WRIST_MOUNT.to_matrix()))
    anchor = store.put(FIXED_PROC, {"target_mount": IDENTITY,
                                   "achieved_accuracy": {"translation_mm": 2}}, FIXED_MOUNT.to_matrix())
    store.put(f"{FIXED}.verify", {"training_disjoint": True, "p95_mm": 2,
                                "depends_on": {FIXED: anchor}})
    store.put(f"{WRIST}.verify", {"training_disjoint": True, "p95_mm": 3,
                                "depends_on": {WRIST: subject, FIXED: anchor}})
    assert FrameGraph.snapshot(_cell(), store.store).diagnosis(WRIST).verified_by
    if change == "version":
        store.put(FIXED_PROC, {"target_mount": IDENTITY}, FIXED_MOUNT.to_matrix())
    elif change == "invalid":
        store.put(FIXED_PROC, {}, FIXED_MOUNT.to_matrix())
    else:
        store.put(f"{FIXED}.attestation", {"passed": False, "depends_on": {FIXED: anchor}})
    diagnosis = FrameGraph.snapshot(_cell(), store.store).diagnosis(WRIST)
    assert not diagnosis.verified_by
    assert diagnosis.ok is own_evidence


def test_cross_view_proof_is_order_independent_and_cannot_bootstrap_a_cycle(store: _Store) -> None:
    a = _legacy_fixed(store)
    b = store.put(WRIST_PROC, {"method": METHOD_MARKERLESS_SCREW}, WRIST_MOUNT.to_matrix())
    for subject, reference, version, ref_version in [(FIXED, WRIST, a, b), (WRIST, FIXED, b, a)]:
        store.put(f"{subject}.verify", {"training_disjoint": True, "p95_mm": 2,
                                     "depends_on": {subject: version, reference: ref_version}})
    spec = _cell()
    reverse = _cell(frames=dict(reversed(list(spec.to_dict()["frames"].items()))))
    first, second = FrameGraph.snapshot(spec, store.store), FrameGraph.snapshot(reverse, store.store)
    assert first.diagnoses == second.diagnoses
    assert not first.versions
    store.put(f"{WRIST}.verify", {"training_disjoint": True, "p95_mm": 2, "depends_on": {WRIST: b}})
    assert all(d.ok for d in FrameGraph.snapshot(spec, store.store).diagnoses.values())


def test_attestation_unknown_reference_accuracy_refuses_before_capture(store: _Store) -> None:
    _legacy_fixed(store)
    store.put(WRIST_PROC, {"target_mount": IDENTITY}, WRIST_MOUNT.to_matrix())
    ctx = _Ctx(_cell(), store)
    ctx._views.clear()  # capture would fail if the reference gate did not run first
    with pytest.raises(SkillError, match="known positive accuracy"):
        _attest(ctx, store)
    assert store.store.latest(CELL_ID, f"{FIXED}.verify") is None


def test_unknown_rotation_does_not_satisfy_a_rotational_bound() -> None:
    assert not Accuracy(1).within(Accuracy(2, 1))
    assert Accuracy(1, .2).plus(Accuracy(1)).rotation_deg is None
    assert not Accuracy(float("nan")).within(Accuracy(2))


def test_lost_reference_lineage_invalidates_record_tokens_even_with_valid_geometry(store: _Store) -> None:
    subject = _verified_wrist(store)
    anchor = store.put(FIXED_PROC, {"target_mount": IDENTITY}, FIXED_MOUNT.to_matrix())
    store.put(f"{FIXED}.verify", {"training_disjoint": True, "p95_mm": 2, "depends_on": {FIXED: anchor}})
    store.put(f"{WRIST}.verify", {"training_disjoint": True, "p95_mm": 3,
                                 "depends_on": {WRIST: subject, FIXED: anchor}})
    token = f"record:{WRIST}.verify"
    assert not unmet_requirements((token,), _cell(), FrameGraph.snapshot(_cell(), store.store, (f"{WRIST}.verify",)))
    store.put(f"{FIXED}.verify", {"training_disjoint": False, "depends_on": {FIXED: anchor}})
    frames = FrameGraph.snapshot(_cell(), store.store, (f"{WRIST}.verify",))
    assert frames.diagnosis(WRIST).ok and not frames.diagnosis(WRIST).verified_by
    assert "independent dependency lineage" in unmet_requirements((token,), _cell(), frames)[0]


def test_unknown_reference_rotation_cannot_support_a_rotational_cross_view_check(store: _Store) -> None:
    wrist = _verified_wrist(store)
    fixed = _legacy_fixed(store)
    store.put(f"{FIXED}.verify", {"training_disjoint": True,
                                "achieved_accuracy": {"translation_mm": 3, "rotation_deg": .5},
                                "depends_on": {FIXED: fixed, WRIST: wrist}})
    spec = _cell()
    frames = spec.to_dict()["frames"]
    frames[FIXED]["accuracy"]["rotation_deg"] = 1
    spec = _cell(frames=frames)
    assert not FrameGraph.snapshot(spec, store.store).diagnosis(FIXED).ok
    ctx = _Ctx(spec, store)
    ctx._views.clear()
    with pytest.raises(SkillError, match="rotational accuracy"):
        _attest(ctx, store)


def test_readiness_marks_missing_evidence_and_ranks_complete_no_motion_options_first(store: _Store) -> None:
    install = replace(InstallGraspEvidence.spec, requires=())
    complete = SkillSpec(skill_id="evidence.restore", version="1", card="Restore a known artifact",
                         effect=EFFECT_READ, provides=install.provides, args_schema={"type": "object"})
    catalog = [SimpleNamespace(spec=install), SimpleNamespace(spec=complete)]
    readiness = WorkcellReadiness(catalog, _cell(), FrameGraph.snapshot(_cell(), store.store))
    options = readiness.options("record:grasp.qualification")
    assert options[0].steps[0].skill_id == complete.skill_id
    step = options[1].steps[0]
    assert step.skill_id == install.skill_id and step.needs_args == ("qualification",)
    assert not step.actionable and not step.motion and step.consent
    assert step.to_dict()["actionable"] is False
    assert step.describe().startswith("supply evidence/arguments")
