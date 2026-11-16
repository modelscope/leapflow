# Copyright (c) Alibaba, Inc. and its affiliates.
"""Synthetic bounded owner alignment; these tests do not qualify physical motion."""
from __future__ import annotations

import asyncio
import math
import threading
from dataclasses import replace

import numpy as np
import pytest

from leapcalib.contracts.failures import CalibrationFailure
from leapcalib.contracts.identity import digest_of
from leapcalib.contracts.observations import Stamp
from leapcalib.estimation.raw_response import estimate_raw_response, propose_raw
from leapcalib.geometry.se3 import Transform
from leapflow.hardware.calibration.alignment import (
    ALIGNED, FAILED, LOST, STOPPED, AlignmentGuard, AlignmentProfile, CorrectionBinding, ResponseSnapshot,
    check_alignment_feedback, run_alignment,
)
from leapflow.hardware.calibration.physical_model import PhysicalModel
from leapflow.hardware.skills.kinematics import DhChain, DhLink
from leapflow.hardware.skills.protocol import ArmState, SkillError
from test_leapcalib_physical_model import geometry
from test_leapcalib_raw_response import raw_study


def fixture(**profile_changes):
    bundle, domain, _ = raw_study()
    fit = estimate_raw_response(bundle, domain)
    response = ResponseSnapshot.from_fit(fit)
    source = next(o for o in bundle.observations if o.observation_id == "r40").first
    reference = replace(source, joints_rad=(0.0,) * 6)
    chain = DhChain(tuple(DhLink(1, 0, 0) for _ in range(6)), "standard")
    nominal = PhysicalModel(chain, (0.0,) * 6, Transform(), digest_of(chain.to_dict()), "synthetic-deps", "synthetic-cert")
    corrected = replace(nominal, tool_transform=Transform(translation_m=(0.001, 0, 0)))
    guard = AlignmentGuard(reference, response, nominal=nominal, corrected=corrected,
                           geometry=geometry(link_radii_mm=(1,) * 7), profile=AlignmentProfile(**profile_changes))
    return guard, fit, reference


def fresh(state, seconds, *, joints=None, feature=None):
    identity = str(seconds)
    stamp = Stamp(seconds, uncertainty_s=0.001)
    return replace(state, task=replace(state.task, stamp=stamp, state_id="state:" + identity,
                   feature_m_rad=state.task.feature_m_rad if feature is None else feature),
                   joints_rad=state.joints_rad if joints is None else joints,
                   feedback_stamp=stamp, frame_ids=("frame:" + identity,), feedback_ids=("feedback:" + identity,))


class HealthyMotion:
    """Coherent synthetic feedback; no transport or physical qualification."""

    def __init__(self, guard, current):
        self.guard = guard
        self.current = current
        self.moves = []
        self.reads = 0
        self.feedback_changes = {}
        self.after_read = lambda: None

    async def state(self):
        self.reads += 1
        pose = self.guard.nominal.forward(tuple(map(math.degrees, self.current.joints_rad)))
        state = ArmState(pose, pose, joints=dict(zip(self.current.joint_ids, map(math.degrees, self.current.joints_rad))),
                         enabled=True, arrived=True, ctrl_mode="can", arm_status=0)
        self.after_read()
        return replace(state, **self.feedback_changes)

    def check_kinematics(self, state):
        assert isinstance(state, ArmState)

    async def move_joints(self, targets, *, speed_pct):
        self.moves.append((targets, speed_pct))
        self.current = replace(self.current, joints_rad=tuple(map(math.radians, targets.values())))


def proposal(guard, fit, current):
    return propose_raw(fit, current, (0.0001, -0.0001, 0.3001, 0.001),
                       Stamp(current.task.stamp.seconds + 0.01), run_id="run", model_digest=guard.response.model_digest)


def test_proposal_is_rechecked_and_single_use_without_authority():
    guard, fit, reference = fixture()
    current = fresh(reference, 200)
    guard.start("run", 199.9)
    proposed = proposal(guard, fit, current)
    targets = guard.accept_proposal(proposed, current, now_s=200.02)
    assert tuple(targets) == reference.joint_ids
    assert tuple(targets.values()) == pytest.approx(tuple(map(math.degrees, proposed["raw_targets_rad"])))
    assert guard.summary()["pending_action"] and guard.corrections == 1
    assert guard.travel_m > 0 and guard.joint_l1_rad > 0
    with pytest.raises(SkillError, match="pending"):
        guard.accept_proposal(proposed, current, now_s=200.03)
    arrival = fresh(current, 200.1, joints=tuple(proposed["raw_targets_rad"]))
    guard.arrived(arrival, now_s=200.11)
    with pytest.raises(SkillError, match="already dispatched"):
        guard.accept_proposal(proposed, current, now_s=200.12)
    with pytest.raises(ValueError, match="reset"):
        guard.start("new-run", 201)


@pytest.mark.parametrize("field,value", [
    ("run_id", "other-client"), ("current_state_digest", "foreign"),
    ("model_digest", "other-model"), ("source_frame_digest", "reused"),
    ("candidate_digest", "forged"), ("supported_subspace_digest", "changed"),
    ("executed", True), ("accepted", 1), ("deadline_monotonic_s", 900),
    ("uncertainty_m_rad", [0, 0, 0, 0]), ("prediction_m_rad", [0.01, 0, 0, 0]),
])
def test_foreign_or_forged_worker_values_never_reserve_dispatch(field, value):
    guard, fit, reference = fixture()
    current = fresh(reference, 200)
    guard.start("run", 199.9)
    proposed = {**proposal(guard, fit, current), field: value}
    with pytest.raises(SkillError):
        guard.accept_proposal(proposed, current, now_s=200.02)
    assert guard.corrections == 0 and not guard.summary()["pending_action"]


def test_nullspace_injection_is_not_exempted_by_small_endpoint_motion():
    guard, fit, reference = fixture()
    current = fresh(reference, 200)
    guard.start("run", 199.9)
    proposed = proposal(guard, fit, current)
    basis = np.asarray(guard.response.basis)
    unsupported = (np.eye(6) - basis.T @ basis) @ np.array([1, 0, 0, 0, 0, 0])
    proposed["raw_targets_rad"] = (np.asarray(proposed["raw_targets_rad"]) + unsupported * 0.0001).tolist()
    with pytest.raises(SkillError, match="nullspace"):
        guard.accept_proposal(proposed, current, now_s=200.02)
    assert guard.corrections == 0


def test_sustained_alignment_counts_time_at_high_frame_rate_and_uncertainty():
    guard, _, reference = fixture()
    guard.start("run", 199.9)
    for i in range(4):
        stamp = 200 + i * 0.1
        assert not guard.evaluate(fresh(reference, stamp), now_s=stamp + 0.01)
    assert guard.evaluate(fresh(reference, 200.4), now_s=200.41)
    assert guard.state == ALIGNED
    assert not guard.summary()["grasp_ready"]
    assert not guard.summary()["physical_activation_qualified"]
    with pytest.raises(SkillError, match="ended"):
        guard.evaluate(fresh(reference, 200.5), now_s=200.51)


def test_reused_frame_or_lost_witness_never_becomes_aligned():
    guard, _, reference = fixture()
    guard.start("run", 199.9)
    current = fresh(reference, 200)
    assert not guard.evaluate(current, now_s=200.01)
    reused = replace(fresh(reference, 200.2), frame_ids=current.frame_ids)
    with pytest.raises(SkillError, match="fresh frames"):
        guard.evaluate(reused, now_s=200.21)
    with pytest.raises(SkillError, match="witness"):
        guard.evaluate(replace(fresh(reference, 200.4), witness_valid=False), now_s=200.41)
    with pytest.raises(SkillError, match="stale"):
        guard.evaluate(fresh(reference, 200.4), now_s=201)


def test_original_campaign_reference_cannot_be_reset_by_small_segments():
    guard, _, reference = fixture()
    guard.start("run", 199.9)
    outside = fresh(reference, 200, joints=(math.radians(5.01), 0, 0, 0, 0, 0))
    with pytest.raises(SkillError, match="original campaign"):
        guard.evaluate(outside, now_s=200.01)
    with pytest.raises(ValueError, match="clearance"):
        AlignmentGuard(reference, guard.response, nominal=guard.nominal, corrected=guard.corrected,
                       geometry=replace(guard.geometry, minimum_clearance_mm=29))


@pytest.mark.parametrize("changes", [
    {"speed_pct": 11}, {"correction_m": 0.006}, {"campaign_m": 0.011},
    {"correction_joint_rad": math.radians(1.01)}, {"campaign_yaw_rad": math.radians(3.01)},
    {"max_corrections": 9}, {"max_travel_m": 0.041}, {"max_joint_l1_rad": math.radians(12.01)},
    {"max_duration_s": 121}, {"aligned_frames": 2}, {"aligned_span_s": 0.29},
    {"speed_pct": True}, {"correction_m": float("nan")},
])
def test_authorized_envelope_can_only_be_tightened(changes):
    with pytest.raises(ValueError):
        AlignmentProfile(**changes)


async def test_runner_sustains_no_motion_and_reports_noncontact_only():
    guard, fit, reference = fixture()
    stamps = iter([200, 200.1, 200.2, 200.3, 200.4])
    clock_value = [199.9]
    class Observations:
        async def capture(self):
            stamp = next(stamps)
            clock_value[0] = stamp + 0.01
            return fresh(reference, stamp)
    motion = HealthyMotion(guard, reference)
    result = await run_alignment(guard, Observations(), worker=None, motion=motion,
        workcell_id="synthetic-cell", run_id="run", fit=fit, clock=lambda: clock_value[0], cancelled=lambda: False)
    assert result["state"] == ALIGNED and result["corrections"] == 0
    assert motion.reads == 5 and motion.moves == []


async def test_runner_worker_failure_and_cancellation_never_retry_or_recover_motion():
    guard, fit, reference = fixture()
    calls = []
    class Observations:
        async def capture(self):
            return fresh(reference, 200, feature=(0.01, 0, 0.3, 0))
    class Worker:
        async def submit(self, *args, **kwargs):
            calls.append(args[0])
            raise SkillError("check_failed", "synthetic worker failure")
    with pytest.raises(SkillError, match="worker failure"):
        await run_alignment(guard, Observations(), worker=Worker(), motion=HealthyMotion(guard, reference),
            workcell_id="synthetic-cell", run_id="run", fit=fit, clock=lambda: 200.01, cancelled=lambda: False)
    assert calls == ["propose_raw"] and guard.state == FAILED
    guard, fit, _ = fixture()
    with pytest.raises(SkillError, match="cancelled"):
        await run_alignment(guard, Observations(), worker=Worker(), motion=HealthyMotion(guard, reference),
            workcell_id="synthetic-cell", run_id="run", fit=fit, clock=lambda: 200.01, cancelled=lambda: True)
    assert guard.state == STOPPED and calls == ["propose_raw"]


@pytest.mark.parametrize("changes", [
    {"enabled": False}, {"arrived": False}, {"ctrl_mode": "teach"}, {"arm_status": False},
    {"arm_status": None}, {"joint_communication_errors": (1,)}, {"joints": {}},
    *({"joints": {f"joint_{i + 1}": value for i in range(6)}}
      for value in (float("nan"), float("inf"), True, "0")),
])
async def test_stationary_unhealthy_feedback_never_reports_aligned(changes):
    guard, fit, reference = fixture()
    motion = HealthyMotion(guard, reference)
    motion.feedback_changes = changes
    class Observations:
        async def capture(self):
            return fresh(reference, 200)
    with pytest.raises(SkillError) as caught:
        await run_alignment(guard, Observations(), worker=None, motion=motion,
            workcell_id="synthetic-cell", run_id="run", fit=fit, clock=lambda: 200.01, cancelled=lambda: False)
    assert caught.value.failure_code == "check_failed"
    assert caught.value.result["alignment"]["state"] == FAILED
    assert motion.moves == [] and guard.corrections == 0


async def test_feedback_rejects_untyped_observation_and_raw_state():
    guard, _, reference = fixture()
    live = await HealthyMotion(guard, reference).state()
    for current, state in ((None, live), (reference, None)):
        with pytest.raises(SkillError, match="typed"):
            check_alignment_feedback(guard, current, state)
    with pytest.raises(SkillError, match="changed"):
        check_alignment_feedback(guard, reference, replace(live, joints={reference.joint_ids[0]: 10}))


async def test_runner_witness_loss_is_actionable_lost_without_dispatch():
    guard, fit, reference = fixture()
    motion = HealthyMotion(guard, reference)
    class Observations:
        async def capture(self):
            return replace(fresh(reference, 200), witness_valid=False)
    with pytest.raises(SkillError) as caught:
        await run_alignment(guard, Observations(), worker=None, motion=motion,
            workcell_id="synthetic-cell", run_id="run", fit=fit, clock=lambda: 200.01, cancelled=lambda: False)
    assert caught.value.failure_code == "target_not_found"
    assert caught.value.result["alignment"]["state"] == LOST
    assert motion.moves == []


@pytest.mark.parametrize("code,expected", [("deadline", "timeout"), ("worker_unavailable", "skill_unavailable"), ("schema", "check_failed")])
async def test_runner_maps_worker_refusal_without_retry(code, expected):
    guard, fit, reference = fixture()
    motion = HealthyMotion(guard, reference)
    calls = []
    class Observations:
        async def capture(self):
            return fresh(reference, 200, feature=(0.01, 0, 0.3, 0))
    class Worker:
        async def submit(self, operation, *args, **kwargs):
            calls.append(operation)
            raise CalibrationFailure(code, "synthetic compute refusal")
    with pytest.raises(SkillError, match="no replay") as caught:
        await run_alignment(guard, Observations(), worker=Worker(), motion=motion,
            workcell_id="synthetic-cell", run_id="run", fit=fit, clock=lambda: 200.01, cancelled=lambda: False)
    assert caught.value.failure_code == expected and guard.state == FAILED
    assert calls == ["propose_raw"] and motion.moves == []


@pytest.mark.parametrize("checkpoint", ["capture", "feedback", "compute", "final_feedback"])
async def test_runner_cancellation_at_each_pre_dispatch_checkpoint(checkpoint):
    guard, fit, reference = fixture()
    current = fresh(reference, 200, feature=(0.01, 0, 0.3, 0))
    motion = HealthyMotion(guard, current)
    stopped = [False]
    calls = []
    class Observations:
        async def capture(self):
            stopped[0] = checkpoint == "capture"
            return current
    class Worker:
        async def submit(self, operation, *args, **kwargs):
            calls.append(operation)
            stopped[0] = checkpoint == "compute"
            return {"proposal": {}}
    def after_read():
        if checkpoint == "feedback" or checkpoint == "final_feedback" and motion.reads == 2:
            stopped[0] = True
    motion.after_read = after_read
    with pytest.raises(SkillError, match="cancelled") as caught:
        await run_alignment(guard, Observations(), worker=Worker(), motion=motion,
            workcell_id="synthetic-cell", run_id="run", fit=fit, clock=lambda: 200.01, cancelled=lambda: stopped[0])
    assert guard.state == STOPPED and caught.value.result["alignment"]["state"] == STOPPED
    assert motion.moves == [] and guard.corrections == 0
    assert len(calls) == (0 if checkpoint in ("capture", "feedback") else 1)


async def test_runner_task_cancellation_preserves_stopped_state():
    guard, fit, reference = fixture()
    entered = asyncio.Event()
    class Observations:
        async def capture(self):
            entered.set()
            await asyncio.Event().wait()
    motion = HealthyMotion(guard, reference)
    task = asyncio.create_task(run_alignment(guard, Observations(), worker=None, motion=motion,
        workcell_id="synthetic-cell", run_id="run", fit=fit, clock=lambda: 200.01, cancelled=lambda: False))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert guard.state == STOPPED and motion.moves == []


class ComputeWorker:
    """Real compute implementation, with optional boundary-only result corruption."""
    def __init__(self, defect=None, stopped=None):
        self.calls = []
        self.payloads = []
        self.results = []
        self.defect, self.stopped = defect, stopped

    async def submit(self, operation, payload, *, workcell_id, run_id, model_digest):
        from leapcalib.runtime.protocol import WorkerBudgets, WorkerRequest
        from leapcalib.runtime.server import compute
        self.calls.append(operation)
        self.payloads.append(payload)
        result = compute(WorkerRequest("r", "j", workcell_id, run_id, model_digest, operation,
            999, payload, digest_of(payload)), WorkerBudgets())
        if operation == "update_raw":
            if self.defect == "floor":
                result["estimate"]["metadata"]["residual_floor_m_rad"] = [0] * 4
            elif self.defect == "holdout":
                result["estimate"]["holdout_ids"] = []
            elif self.defect == "basis":
                result["estimate"]["metadata"]["supported_basis_scaled"][0][0] += 0.001
            elif self.defect == "identity":
                result["estimate"]["metadata"]["model_digest"] = "foreign"
            elif self.defect == "cancel":
                self.stopped[0] = True
            elif self.defect == "worker_failure":
                raise CalibrationFailure("worker_unavailable", "synthetic post-motion failure")
        self.results.append(result)
        return result


def correction_case(*, enabled=True, defect=None):
    """Measured synthetic endpoints differ from targets; no physical authority."""
    guard, fit, reference = fixture(response_update_enabled=enabled, max_update_norm=0.01)
    current = fresh(reference, 200, feature=(0.004, 0, 0.3, 0))
    clock = [199.9]
    stopped = [False]
    binding = CorrectionBinding("run", fit.topology.streams[0], fit.topology.model,
        guard.response.domain.scope_digest, "synthetic")
    class MeasuredMotion(HealthyMotion):
        async def move_joints(self, targets, *, speed_pct):
            first = np.asarray(self.current.joints_rad)
            await super().move_joints(targets, speed_pct=speed_pct)
            # Model measured encoder arrival, not a perfect command echo.
            actual = first + 0.999 * (np.asarray(self.current.joints_rad) - first)
            self.current = replace(self.current, joints_rad=tuple(actual))
    motion = MeasuredMotion(guard, current)
    class Observations:
        def __init__(self):
            self.captures = []
        async def capture(self):
            clock[0] += 0.1
            state = motion.current
            feature = np.asarray((0.004, 0, 0.3, 0)) + np.asarray(fit.metadata["response_scaled"]) @ (
                np.asarray(state.joints_rad) / guard.response.domain.max_step_rad
            ) * guard.response.domain.task.tolerance_m_rad
            state = fresh(state, clock[0], feature=tuple(feature))
            if len(self.captures) == 1:
                if defect == "innovation":
                    state = replace(state, task=replace(state.task, feature_m_rad=(0.02, 0, 0.3, 0)))
                elif defect == "shared_source":
                    state = replace(state, frame_ids=self.captures[0].frame_ids)
                elif defect == "excluded_source":
                    state = replace(state, frame_ids=(fit.metadata["holdout_source_ids"][0],))
            motion.current = state
            self.captures.append(state)
            return state
    return guard, fit, binding, motion, Observations(), clock, stopped


@pytest.mark.parametrize("changes", [{"response_update_enabled": 1}, {"max_update_norm": 0},
    {"max_update_norm": 0.051}, {"max_update_norm": True}, {"max_update_norm": float("nan")}])
def test_response_adaptation_is_explicit_and_can_only_tighten(changes):
    with pytest.raises(ValueError):
        AlignmentProfile(**changes)


async def test_correction_updates_use_measured_endpoints_and_preserve_holdouts_and_models():
    guard, fit, binding, motion, observations, clock, stopped = correction_case()
    worker = ComputeWorker()
    models = guard.nominal.digest, guard.corrected.digest
    original = fit.to_dict()
    original_digest = fit.digest
    result = await run_alignment(guard, observations, worker=worker, motion=motion,
        workcell_id="synthetic-cell", run_id="run", fit=fit, clock=lambda: clock[0] + 0.01,
        cancelled=lambda: stopped[0], correction_binding=binding)
    assert result["state"] == ALIGNED and len(motion.moves) == 1
    assert worker.calls == ["propose_raw", "update_raw"]
    sample = worker.payloads[1]["sample"]
    assert sample["first"]["joints_rad"] == list(observations.captures[0].joints_rad)
    assert sample["second"]["joints_rad"] == list(observations.captures[1].joints_rad)
    assert sample["second"]["joints_rad"] != list(map(math.radians, motion.moves[0][0].values()))
    assert len(result["response_updates"]) == 1
    assert result["response_updates"][0]["parent_digest"] == fit.digest
    assert guard.response.candidate_digest != fit.digest
    assert (guard.nominal.digest, guard.corrected.digest) == models
    assert fit.metadata["update_ids"] == []  # original immutable version remains unchanged
    updated = worker.results[1]["estimate"]
    assert fit.digest == original_digest and fit.to_dict() == original
    assert updated["holdout_ids"] == original["holdout_ids"]
    assert updated["accepted_ids"] == original["accepted_ids"]
    assert updated["rejected_ids"] == original["rejected_ids"]
    for key in ("holdout_source_ids", "holdout_predictions", "holdout_rejections",
                "supported_subspace_digest", "supported_basis_scaled", "domain", "config"):
        assert updated["metadata"][key] == original["metadata"][key]
    assert set(updated["metadata"]["holdout_source_ids"]).isdisjoint(updated["metadata"]["training_source_ids"])
    assert not result["physical_activation_qualified"] and not result["grasp_ready"]


async def test_legacy_no_adaptation_correction_has_no_hidden_update():
    guard, fit, binding, motion, observations, clock, stopped = correction_case(enabled=False)
    worker = ComputeWorker()
    result = await run_alignment(guard, observations, worker=worker, motion=motion,
        workcell_id="synthetic-cell", run_id="run", fit=fit, clock=lambda: clock[0] + 0.01,
        cancelled=lambda: stopped[0])
    assert result["state"] == ALIGNED and worker.calls == ["propose_raw"]
    assert result["response_updates"] == [] and guard.response.candidate_digest == fit.digest


@pytest.mark.parametrize("defect", ["floor", "holdout", "basis", "identity", "cancel", "worker_failure",
    "innovation", "shared_source", "excluded_source"])
async def test_post_correction_update_refusal_never_replays_or_promotes(defect):
    guard, fit, binding, motion, observations, clock, stopped = correction_case(defect=defect)
    worker = ComputeWorker(defect, stopped)
    with pytest.raises(SkillError) as caught:
        await run_alignment(guard, observations, worker=worker, motion=motion,
            workcell_id="synthetic-cell", run_id="run", fit=fit, clock=lambda: clock[0] + 0.01,
            cancelled=lambda: stopped[0], correction_binding=binding)
    assert len(motion.moves) == 1
    assert guard.corrections == 1
    assert guard.summary()["pending_action"] is (defect == "shared_source")
    assert guard.summary()["response_updates"] == [] and guard.response.candidate_digest == fit.digest
    assert guard.state == (STOPPED if defect == "cancel" else FAILED)
    assert caught.value.result["alignment"]["state"] == guard.state
    assert not caught.value.result["alignment"]["physical_activation_qualified"]


@pytest.mark.parametrize("defect", ["missing", "run", "scope", "provenance"])
async def test_enabled_updates_require_exact_sealed_acquisition_binding_before_motion(defect):
    guard, fit, binding, motion, observations, clock, stopped = correction_case()
    if defect == "missing":
        binding = None
    else:
        binding = replace(binding, **{"run": {"run_id": "other-client"}, "scope": {"scope_digest": "other"},
            "provenance": {"provenance": "metric_rgbd"}}[defect])
    worker = ComputeWorker()
    with pytest.raises(SkillError):
        await run_alignment(guard, observations, worker=worker, motion=motion,
            workcell_id="synthetic-cell", run_id="run", fit=fit, clock=lambda: clock[0] + 0.01,
            cancelled=lambda: stopped[0], correction_binding=binding)
    assert motion.moves == [] and worker.calls == [] and guard.state == STOPPED


@pytest.mark.parametrize("changed", ["frame", "feedback", "clock"])
def test_actual_arrival_cannot_reuse_pre_dispatch_sources_or_time(changed):
    guard, fit, reference = fixture()
    first = fresh(reference, 200)
    guard.start("run", 199.9)
    proposed = proposal(guard, fit, first)
    guard.accept_proposal(proposed, first, now_s=200.02)
    arrival = fresh(first, 200.1, joints=tuple(proposed["raw_targets_rad"]))
    arrival = replace(arrival, **{
        "frame": {"frame_ids": first.frame_ids},
        "feedback": {"feedback_ids": first.feedback_ids},
        "clock": {"feedback_stamp": first.feedback_stamp},
    }[changed])
    with pytest.raises(SkillError, match="fresh separated"):
        guard.arrived(arrival, now_s=200.11)
    assert guard.summary()["pending_action"] and guard.corrections == 1


def test_update_is_bound_to_exact_settled_endpoints_and_is_single_use():
    from leapcalib.estimation.raw_response import update_raw_response
    guard, fit, binding, _, _, _, _ = correction_case()
    first = fresh(guard.reference, 200, feature=(0.004, 0, 0.3, 0))
    guard.start("run", 199.9)
    proposed = proposal(guard, fit, first)
    guard.accept_proposal(proposed, first, now_s=200.02)
    arrival = fresh(first, 200.1, joints=tuple(proposed["raw_targets_rad"]),
        feature=tuple(a + b for a, b in zip(first.task.feature_m_rad, proposed["prediction_m_rad"])))
    guard.arrived(arrival, now_s=200.11)
    sample = binding.transition(first, arrival)
    response = ResponseSnapshot.from_fit(update_raw_response(fit, sample, max_update_norm=0.01))
    foreign = binding.transition(first, fresh(arrival, 200.2))
    with pytest.raises(SkillError, match="lacks executed"):
        guard.publish_response_update(response, foreign)
    assert guard.response.candidate_digest == fit.digest
    guard.publish_response_update(response, sample)
    with pytest.raises(SkillError, match="lacks executed"):
        guard.publish_response_update(response, sample)
    assert len(guard.summary()["response_updates"]) == 1


@pytest.mark.parametrize("failed", [False, True])
async def test_cancelled_owner_validation_retains_run_until_thread_settlement(monkeypatch, failed):
    from leapflow.hardware.calibration import alignment
    guard, fit, binding, motion, observations, clock, stopped = correction_case()
    entered, release = threading.Event(), threading.Event()
    original = alignment._validate_response_update
    def delayed(*args):
        entered.set()
        assert release.wait(3)
        if failed:
            raise ValueError("synthetic numerical refusal during cancellation")
        return original(*args)
    monkeypatch.setattr(alignment, "_validate_response_update", delayed)
    task = asyncio.create_task(run_alignment(guard, observations, worker=ComputeWorker(), motion=motion,
        workcell_id="synthetic-cell", run_id="run", fit=fit, clock=lambda: clock[0] + 0.01,
        cancelled=lambda: stopped[0], correction_binding=binding))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and len(motion.moves) == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert guard.state == STOPPED and guard.summary()["response_updates"] == []
        assert guard.response.candidate_digest == fit.digest and len(motion.moves) == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_real_subprocess_correction_update_loop_reaps_without_physical_qualification(tmp_path):
    from test_leapcalib_worker import worker_at
    guard, fit, binding, motion, observations, clock, stopped = correction_case()
    worker = worker_at(tmp_path)
    try:
        result = await run_alignment(guard, observations, worker=worker, motion=motion,
            workcell_id="synthetic-cell", run_id="run", fit=fit, clock=lambda: clock[0] + 0.01,
            cancelled=lambda: stopped[0], correction_binding=binding)
        assert result["state"] == ALIGNED and len(result["response_updates"]) == 1
        assert not result["physical_activation_qualified"]
        process = worker._process
    finally:
        await worker.close()
    assert process.returncode is not None and not worker.socket_path.exists()
