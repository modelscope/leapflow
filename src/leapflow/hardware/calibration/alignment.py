# Copyright (c) Alibaba, Inc. and its affiliates.
"""Owner-side bounded non-contact alignment; numerical output grants no authority.

The campaign reference is immutable across runs. This module consumes admitted
observations and frozen models, not camera pipelines, permits or vendor SDKs.
A caller must obtain independent witness/geometry provenance through acquisition;
a stationary registration alone does not supply that evidence.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from leapcalib.contracts.estimates import EstimateBundle
from leapcalib.contracts.failures import CalibrationFailure, FAILURE_DEADLINE, FAILURE_WORKER_UNAVAILABLE
from leapcalib.contracts.identity import ModelIdentity, StreamProfile, digest_of
from leapcalib.contracts.motion import finite_vector
from leapcalib.contracts.observations import CALIBRATION_PROVENANCE, Stamp
from leapcalib.contracts.raw_task import RawTaskDomain, RawTaskState, RawTaskTransition, identities
from leapflow.hardware.calibration.physical_model import PhysicalModel, SafetyGeometry, check_dual_sweep
from leapflow.hardware.calibration.worker import ComputeWorker as AlignmentWorker
from leapflow.hardware.skills.protocol import (
    FAILURE_CHECK_FAILED, FAILURE_PLAN_STALE, FAILURE_SKILL_UNAVAILABLE, FAILURE_STOPPED,
    FAILURE_TARGET_NOT_FOUND, FAILURE_TIMEOUT, ArmState, SkillError,
)

ALIGNING = "ALIGNING"
ALIGNED = "ALIGNED"
LOST = "LOST"
STOPPED = "STOPPED"
FAILED = "FAILED"


def _norm(values: Any) -> float:
    return math.sqrt(sum(v * v for v in values))


def _yaw_delta(first: float, second: float) -> float:
    return (first - second + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class AlignmentProfile:
    speed_pct: float = 10.0
    correction_m: float = 0.005
    correction_yaw_rad: float = math.radians(1)
    correction_joint_rad: float = math.radians(1)
    campaign_m: float = 0.010
    campaign_yaw_rad: float = math.radians(3)
    campaign_joint_rad: float = math.radians(5)
    max_corrections: int = 8
    max_travel_m: float = 0.040
    max_joint_l1_rad: float = math.radians(12)
    max_duration_s: float = 120.0
    tolerance_m_rad: tuple[float, ...] = (0.003, 0.003, 0.003, math.radians(2))
    aligned_frames: int = 3
    aligned_span_s: float = 0.3
    allowed_modes: tuple[str, ...] = ("can",)
    response_update_enabled: bool = False
    max_update_norm: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_modes, (list, tuple)) or not 1 <= len(self.allowed_modes) <= 16 or any(not isinstance(mode, str) or not mode or len(mode) > 64 for mode in self.allowed_modes):
            raise ValueError("alignment requires bounded declared healthy raw feedback modes")
        object.__setattr__(self, "allowed_modes", tuple(self.allowed_modes))
        if type(self.response_update_enabled) is not bool:
            raise ValueError("response adaptation must be explicitly enabled in the frozen policy")
        ceilings = {"max_update_norm": 0.05, "speed_pct": 10.0, "correction_m": 0.005,
                    "correction_yaw_rad": math.radians(1), "correction_joint_rad": math.radians(1),
                    "campaign_m": 0.010, "campaign_yaw_rad": math.radians(3),
                    "campaign_joint_rad": math.radians(5), "max_travel_m": 0.040,
                    "max_joint_l1_rad": math.radians(12), "max_duration_s": 120.0}
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= ceiling:
                raise ValueError(f"{name} exceeds authorized alignment envelope")
        if type(self.max_corrections) is not int or not 1 <= self.max_corrections <= 8:
            raise ValueError("correction count exceeds authorized envelope")
        object.__setattr__(self, "tolerance_m_rad", finite_vector(self.tolerance_m_rad, 4))
        if any(not 0 < a <= b for a, b in zip(self.tolerance_m_rad, (0.003, 0.003, 0.003, math.radians(2)))):
            raise ValueError("alignment tolerance exceeds declared study criterion")
        if type(self.aligned_frames) is not int or not 3 <= self.aligned_frames <= 128:
            raise ValueError("alignment needs at least three bounded fresh frames")
        if isinstance(self.aligned_span_s, bool) or not isinstance(self.aligned_span_s, (int, float)) or not math.isfinite(self.aligned_span_s) or not 0.3 <= self.aligned_span_s <= self.max_duration_s:
            raise ValueError("alignment needs a sustained fresh observation interval")

    def to_dict(self) -> dict[str, Any]:
        return {**vars(self), "tolerance_m_rad": list(self.tolerance_m_rad), "allowed_modes": list(self.allowed_modes)}


@dataclass(frozen=True)
class ResponseSnapshot:
    """Finite immutable worker values; NOT a physical qualification certificate."""
    candidate_digest: str
    model_digest: str
    domain: RawTaskDomain
    basis: tuple[tuple[float, ...], ...]
    response: tuple[tuple[float, ...], ...]
    residual_floor: tuple[float, ...]
    confidence_multiplier: float
    lower_rad: tuple[float, ...]
    upper_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.domain, RawTaskDomain) or not self.candidate_digest or not self.model_digest:
            raise ValueError("response needs frozen candidate/model/domain identity")
        n = len(self.domain.joint_ids)
        object.__setattr__(self, "basis", tuple(finite_vector(row, n) for row in self.basis))
        object.__setattr__(self, "response", tuple(finite_vector(row, n) for row in self.response))
        if not 4 <= len(self.basis) <= n or len(self.response) != 4:
            raise ValueError("response must cover four outputs in a measured input subspace")
        for i, first in enumerate(self.basis):
            for j, second in enumerate(self.basis):
                if abs(sum(a * b for a, b in zip(first, second)) - (1 if i == j else 0)) > 1e-8:
                    raise ValueError("response supported basis must be orthonormal")
        for name, count in (("residual_floor", 4), ("lower_rad", n), ("upper_rad", n)):
            object.__setattr__(self, name, finite_vector(getattr(self, name), count))
        if any(v <= 0 for v in self.residual_floor) or any(lo > hi for lo, hi in zip(self.lower_rad, self.upper_rad)):
            raise ValueError("response floors/bounds must remain conservative")
        c = self.confidence_multiplier
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or c < 1:
            raise ValueError("response confidence multiplier must be finite and conservative")

    @property
    def subspace_digest(self) -> str:
        return digest_of([list(row) for row in self.basis])

    @classmethod
    def from_fit(cls, fit: EstimateBundle) -> ResponseSnapshot:
        # Numerical validation is a cold-path operation. Live command rechecks
        # below are dependency-free dot products over the frozen bounded arrays.
        from leapcalib.estimation.raw_response import validate_raw_fit
        domain, config = validate_raw_fit(fit)
        return cls(fit.digest, fit.topology.model.digest, domain,
                   tuple(tuple(row) for row in fit.metadata["supported_basis_scaled"]),
                   tuple(tuple(row) for row in fit.metadata["response_scaled"]),
                   tuple(fit.metadata["residual_floor_m_rad"]), config.confidence_multiplier,
                   tuple(fit.metadata["support_lower_rad"]), tuple(fit.metadata["support_upper_rad"]))


@dataclass(frozen=True)
class CorrectionBinding:
    """Owner-declared acquisition identity, not a new admission or witness certificate.

    The caller seals this binding and the adaptation policy into its approved plan
    and supplies only admitted captures through AlignmentObservations. Synthetic
    bindings remain synthetic; numerical updates never promote their provenance.
    """
    run_id: str
    stream: StreamProfile
    model: ModelIdentity
    scope_digest: str
    provenance: str

    def __post_init__(self) -> None:
        identities((self.run_id,))
        identities((self.scope_digest,))
        if not isinstance(self.stream, StreamProfile) or not isinstance(self.model, ModelIdentity):
            raise ValueError("typed correction acquisition stream/model required")
        if self.provenance not in CALIBRATION_PROVENANCE:
            raise ValueError("explicit metric/replay/synthetic acquisition provenance required")

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "stream": self.stream.to_dict(),
                "model": self.model.to_dict(), "scope_digest": self.scope_digest,
                "provenance": self.provenance}

    def validate(self, fit: EstimateBundle, run_id: str, scope_digest: str) -> None:
        if (
            run_id != self.run_id or scope_digest != self.scope_digest
            or self.stream.digest != fit.metadata["stream_digest"]
            or self.model != fit.topology.model
            or fit.metadata["provenances"] != [self.provenance]
        ):
            raise ValueError("correction acquisition identity/provenance differs from frozen response")

    def transition(self, first: RawTaskState, second: RawTaskState) -> RawTaskTransition:
        if first.scope_digest != self.scope_digest or second.scope_digest != self.scope_digest:
            raise ValueError("correction capture scope changed")
        if set(first.source_ids).intersection(second.source_ids):
            raise ValueError("correction arrival reuses measured source identities")
        identity = "correction:" + digest_of({"binding": self.to_dict(), "first": first.digest, "second": second.digest})
        return RawTaskTransition(identity, self.stream, self.model, first, second, self.provenance)


class AlignmentGuard:
    """Single-use proposals and non-resettable campaign/run budgets.

    The owner still checks live feedback, generation, authority and dispatch at
    its ordinary shared gate. Passing this guard never authorizes a command.
    """
    def __init__(self, reference: RawTaskState, response: ResponseSnapshot, *,
                 nominal: PhysicalModel, corrected: PhysicalModel, geometry: SafetyGeometry,
                 profile: AlignmentProfile = AlignmentProfile()) -> None:
        if not response.domain.accepts(reference):
            raise ValueError("campaign reference needs settled admitted identity and independent witness")
        if nominal.nominal_digest != corrected.nominal_digest or nominal.dof != len(reference.joint_ids):
            raise ValueError("campaign safety/model/raw-channel identities differ")
        if any(nominal.zero_offsets_rad):
            raise ValueError("nominal safety model cannot contain physical zero corrections")
        if geometry.minimum_clearance_mm < 30 or geometry.boundary_margin_mm < 20:
            raise ValueError("campaign geometry cannot relax residual clearance or boundary margin")
        self.reference, self.response = reference, response
        self.nominal, self.corrected, self.geometry, self.profile = nominal, corrected, geometry, profile
        self._run_id = ""
        self._started_s = 0.0
        self._pending: tuple[float, ...] | None = None
        self._dispatched_state: RawTaskState | None = None
        self._settled_correction: tuple[str, str] | None = None
        self._used_sources: set[str] = set()
        self._aligned_stamps: list[tuple[float, float]] = []
        self._evaluated_frames: set[str] = set()
        self._last_stamp: float | None = None
        self._response_updates: list[dict[str, str]] = []
        self.corrections = 0
        self.travel_m = 0.0
        self.joint_l1_rad = 0.0
        self.state = STOPPED
        self._check_sweep(reference.joints_rad, reference.joints_rad)

    def start(self, run_id: str, now_s: float) -> None:
        if self._run_id or not isinstance(run_id, str) or not run_id:
            raise ValueError("an alignment run cannot be replayed or silently reset")
        if isinstance(now_s, bool) or not isinstance(now_s, (int, float)) or not math.isfinite(now_s) or now_s < 0:
            raise ValueError("finite node monotonic run start required")
        self._run_id, self._started_s, self.state = run_id, now_s, ALIGNING

    def _live(self, current: RawTaskState, now_s: float) -> None:
        if self.state != ALIGNING or not self._run_id:
            raise SkillError(FAILURE_STOPPED, "alignment run has ended; no replay or automatic recovery")
        if not math.isfinite(now_s) or now_s < self._started_s or now_s - self._started_s > self.profile.max_duration_s:
            raise SkillError(FAILURE_TIMEOUT, "alignment run deadline expired")
        if not isinstance(current, RawTaskState):
            raise SkillError(FAILURE_CHECK_FAILED, "typed measured raw alignment observation required")
        if current.witness_valid is not True:
            raise SkillError(FAILURE_TARGET_NOT_FOUND, "alignment independent witness lost; stop and inspect the scene")
        if not self.response.domain.accepts(current):
            raise SkillError(FAILURE_CHECK_FAILED, "alignment identity, stillness or independent witness lost")
        domain = self.response.domain.task
        age = now_s - current.task.stamp.seconds
        feedback_age = now_s - current.feedback_stamp.seconds
        if age < 0 or feedback_age < 0 or max(age + current.task.stamp.uncertainty_s, feedback_age + current.feedback_stamp.uncertainty_s) > domain.max_age_s:
            raise SkillError(FAILURE_CHECK_FAILED, "alignment source is stale or from another clock")
        if current.feedback_stamp.spread_s(current.task.stamp) > domain.max_age_s:
            raise SkillError(FAILURE_CHECK_FAILED, "alignment frame/raw feedback timing is incoherent")
        uncertainty = self.uncertainty(current)
        if any(u > limit for u, limit in zip(uncertainty, self.profile.tolerance_m_rad)):
            raise SkillError(FAILURE_CHECK_FAILED, "alignment uncertainty exceeds frozen criterion")
        self._check_sweep(current.joints_rad, current.joints_rad)

    def uncertainty(self, current: RawTaskState) -> tuple[float, ...]:
        return tuple(max(floor, value) * self.response.confidence_multiplier
                     for floor, value in zip(self.response.residual_floor, current.task.uncertainty_m_rad))

    def _check_sweep(self, first_rad: tuple[float, ...], second_rad: tuple[float, ...]) -> float:
        p = self.profile
        if any(abs(q - start) > p.campaign_joint_rad + 1e-12 for q, start in zip(second_rad, self.reference.joints_rad)):
            raise SkillError(FAILURE_CHECK_FAILED, "raw target leaves the original campaign envelope")
        first, second = tuple(map(math.degrees, first_rad)), tuple(map(math.degrees, second_rad))
        try:
            result = check_dual_sweep(self.nominal, self.corrected, first, second, self.geometry)
        except ValueError as exc:
            raise SkillError(FAILURE_CHECK_FAILED, f"alignment dual-model clearance refused: {exc}") from exc
        travel = 0.0
        for model in (self.nominal, self.corrected):
            reference = model.forward(tuple(map(math.degrees, self.reference.joints_rad)))
            previous = model.forward(first)
            length = 0.0
            steps = result["samples_per_model"] - 1
            for i in range(steps + 1):
                q = tuple(a + (b - a) * i / steps for a, b in zip(first, second))
                pose = model.forward(q)
                if pose.distance_mm(reference) > p.campaign_m * 1000 + 1e-9 or abs(_yaw_delta(math.radians(pose.rz), math.radians(reference.rz))) > p.campaign_yaw_rad + 1e-12:
                    raise SkillError(FAILURE_CHECK_FAILED, "intermediate TCP leaves original campaign envelope")
                length += pose.distance_mm(previous) / 1000
                previous = pose
            reach_m = sum(abs(link.a) + abs(link.d) for link in model.nominal.links) / 1000 + _norm(model.tool_transform.translation_m)
            angle_l1 = sum(abs(b - a) for a, b in zip(first_rad, second_rad))
            # Bound the chord-vs-arc length deficit using a whole-chain second
            # derivative bound; sampled chords alone undercount nonlinear travel.
            length += reach_m * angle_l1 * angle_l1 / (2 * steps)
            travel = max(travel, length)
        return travel

    def accept_proposal(self, proposal: Mapping[str, Any], current: RawTaskState, *, now_s: float) -> Mapping[str, float]:
        self._live(current, now_s)
        if self._pending is not None or current.digest in self._used_sources:
            raise SkillError(FAILURE_STOPPED, "alignment has pending or already dispatched source; no replay")
        expected = {"schema", "executed", "run_id", "raw_targets_rad", "joint_ids", "model_digest", "candidate_digest",
                    "domain_digest", "current_state_digest", "source_frame_digest", "supported_subspace_digest",
                    "prediction_m_rad", "uncertainty_m_rad", "deadline_monotonic_s", "trust_region_rad", "accepted"}
        bindings = {"schema": "leapcalib.raw_proposal.v1", "run_id": self._run_id,
                    "model_digest": self.response.model_digest, "candidate_digest": self.response.candidate_digest,
                    "domain_digest": self.response.domain.digest, "current_state_digest": current.digest,
                    "source_frame_digest": digest_of(current.frame_ids), "supported_subspace_digest": self.response.subspace_digest}
        if not isinstance(proposal, Mapping) or set(proposal) != expected or any(proposal[k] != v for k, v in bindings.items()) or proposal["executed"] is not False or proposal["accepted"] is not True or proposal["joint_ids"] != list(current.joint_ids):
            raise SkillError(FAILURE_PLAN_STALE, "worker proposal differs from frozen run/model/source policy")
        try:
            target = finite_vector(proposal["raw_targets_rad"], len(current.joint_ids))
            prediction = finite_vector(proposal["prediction_m_rad"], 4)
            uncertainty = finite_vector(proposal["uncertainty_m_rad"], 4)
            trust = finite_vector(proposal["trust_region_rad"], len(current.joint_ids))
            deadline = finite_vector((proposal["deadline_monotonic_s"],), 1)[0]
        except (TypeError, ValueError) as exc:
            raise SkillError(FAILURE_CHECK_FAILED, f"malformed finite worker proposal: {exc}") from exc
        domain, profile = self.response.domain, self.profile
        maximum_deadline = min(current.task.stamp.seconds - current.task.stamp.uncertainty_s,
                               current.feedback_stamp.seconds - current.feedback_stamp.uncertainty_s) + domain.task.max_age_s
        if not now_s < deadline <= maximum_deadline + 1e-12 or trust != domain.max_step_rad:
            raise SkillError(FAILURE_CHECK_FAILED, "worker deadline or trust region exceeds frozen policy")
        delta = tuple(b - a for a, b in zip(current.joints_rad, target))
        if any(abs(d) > min(step, profile.correction_joint_rad) + 1e-12 for d, step in zip(delta, domain.max_step_rad)):
            raise SkillError(FAILURE_CHECK_FAILED, "raw correction exceeds authorized single-step bound")
        if any(not max(lo, supported_lo) <= q <= min(hi, supported_hi) for q, lo, hi, supported_lo, supported_hi in zip(target, domain.lower_rad, domain.upper_rad, self.response.lower_rad, self.response.upper_rad)):
            raise SkillError(FAILURE_CHECK_FAILED, "raw correction extrapolates beyond measured support")
        scaled = tuple(d / step for d, step in zip(delta, domain.max_step_rad))
        coordinates = tuple(sum(a * b for a, b in zip(row, scaled)) for row in self.response.basis)
        projected = tuple(sum(row[i] * z for row, z in zip(self.response.basis, coordinates)) for i in range(len(scaled)))
        if _norm(a - b for a, b in zip(scaled, projected)) > 1e-8:
            raise SkillError(FAILURE_CHECK_FAILED, "raw correction contains unsupported nullspace motion")
        predicted = tuple(sum(a * b for a, b in zip(row, scaled)) * tolerance for row, tolerance in zip(self.response.response, domain.task.tolerance_m_rad))
        if any(abs(a - b) > 1e-9 for a, b in zip(prediction, predicted)):
            raise SkillError(FAILURE_CHECK_FAILED, "worker prediction differs from measured response")
        if _norm(prediction[:3]) > profile.correction_m + 1e-12 or abs(prediction[3]) > profile.correction_yaw_rad + 1e-12:
            raise SkillError(FAILURE_CHECK_FAILED, "task correction exceeds authorized single-step bound")
        if any(not minimum <= value <= tolerance for value, minimum, tolerance in zip(uncertainty, self.uncertainty(current), profile.tolerance_m_rad)):
            raise SkillError(FAILURE_CHECK_FAILED, "worker understates uncertainty or exceeds criterion")
        for model in (self.nominal, self.corrected):
            first = model.forward(tuple(map(math.degrees, current.joints_rad)))
            second = model.forward(tuple(map(math.degrees, target)))
            if first.distance_mm(second) > profile.correction_m * 1000 + 1e-9 or abs(_yaw_delta(math.radians(second.rz), math.radians(first.rz))) > profile.correction_yaw_rad + 1e-12:
                raise SkillError(FAILURE_CHECK_FAILED, "physical-model correction exceeds single-step bound")
        travel = self._check_sweep(current.joints_rad, target)
        joint_travel = sum(abs(d) for d in delta)
        if self.corrections >= profile.max_corrections or self.travel_m + travel > profile.max_travel_m + 1e-12 or self.joint_l1_rad + joint_travel > profile.max_joint_l1_rad + 1e-12:
            raise SkillError(FAILURE_CHECK_FAILED, "alignment run count/travel budget exhausted")
        # Reserve costs BEFORE dispatch. An uncertain action never refunds them.
        self.corrections += 1
        self.travel_m += travel
        self.joint_l1_rad += joint_travel
        self._pending = target
        self._dispatched_state = current
        self._settled_correction = None
        self._used_sources.add(current.digest)
        self._aligned_stamps.clear()
        return dict(zip(current.joint_ids, map(math.degrees, target)))

    def arrived(self, current: RawTaskState, *, now_s: float) -> None:
        self._live(current, now_s)
        first = self._dispatched_state
        if self._pending is None or first is None:
            raise SkillError(FAILURE_CHECK_FAILED, "alignment arrival has no matching dispatch")
        if (
            current.task.stamp.seconds - current.task.stamp.uncertainty_s
            <= first.task.stamp.seconds + first.task.stamp.uncertainty_s
            or current.feedback_stamp.seconds - current.feedback_stamp.uncertainty_s
            <= first.feedback_stamp.seconds + first.feedback_stamp.uncertainty_s
            or set(first.source_ids).intersection(current.source_ids)
        ):
            raise SkillError(FAILURE_CHECK_FAILED, "arrival requires fresh separated frame and feedback sources after dispatch")
        if any(abs(a - b) > max(u, 1e-6) for a, b, u in zip(current.joints_rad, self._pending, current.joint_uncertainty_rad)):
            raise SkillError(FAILURE_CHECK_FAILED, "actual raw arrival differs from bounded target")
        self._settled_correction = (first.digest, current.digest)
        self._dispatched_state = None
        self._pending = None

    def publish_response_update(self, response: ResponseSnapshot, sample: RawTaskTransition) -> None:
        """Change only the local response after settled motion and owner validation."""
        old = self.response
        if not self.profile.response_update_enabled or self.state != ALIGNING or self._pending is not None:
            raise SkillError(FAILURE_STOPPED, "response update requires enabled adaptation and settled correction")
        if (
            response.model_digest != old.model_digest or response.domain != old.domain
            or response.basis != old.basis or response.lower_rad != old.lower_rad
            or response.upper_rad != old.upper_rad or response.confidence_multiplier != old.confidence_multiplier
            or any(a < b for a, b in zip(response.residual_floor, old.residual_floor))
            or self._settled_correction != (sample.first.digest, sample.second.digest)
            or len(self._response_updates) >= self.corrections
        ):
            raise SkillError(FAILURE_PLAN_STALE, "response update changes frozen support/identity or lacks executed correction")
        self._response_updates.append({"observation_id": sample.observation_id,
            "parent_digest": old.candidate_digest, "candidate_digest": response.candidate_digest})
        self.response = response
        self._settled_correction = None

    def evaluate(self, current: RawTaskState, *, now_s: float) -> bool:
        self._live(current, now_s)
        stamp = current.task.stamp.seconds
        if self._pending is not None or self._last_stamp is not None and stamp <= self._last_stamp or self._evaluated_frames.intersection(current.frame_ids):
            raise SkillError(FAILURE_CHECK_FAILED, "alignment outcome needs increasing fresh frames and settled action")
        if len(self._evaluated_frames) + len(current.frame_ids) > 2048:
            raise SkillError(FAILURE_CHECK_FAILED, "alignment fresh-frame accounting budget exhausted")
        self._last_stamp = stamp
        self._evaluated_frames.update(current.frame_ids)
        target = self.reference.task.feature_m_rad
        actual = current.task.feature_m_rad
        error = tuple(a - b for a, b in zip(actual[:3], target[:3])) + (_yaw_delta(actual[3], target[3]),)
        if all(abs(e) + u <= tolerance for e, u, tolerance in zip(error, self.uncertainty(current), self.profile.tolerance_m_rad)):
            self._aligned_stamps.append((stamp - current.task.stamp.uncertainty_s, stamp + current.task.stamp.uncertainty_s))
            # Preserve the first sustained frame at high camera rates; a rolling
            # last-three window can otherwise never span the required 0.3 s.
            if len(self._aligned_stamps) > self.profile.aligned_frames:
                self._aligned_stamps = [self._aligned_stamps[0], *self._aligned_stamps[-(self.profile.aligned_frames - 1):]]
        else:
            self._aligned_stamps.clear()
        if len(self._aligned_stamps) >= self.profile.aligned_frames and self._aligned_stamps[-1][0] - self._aligned_stamps[0][1] >= self.profile.aligned_span_s:
            self.state = ALIGNED
            return True
        return False

    def end(self, state: str) -> None:
        if state not in (LOST, STOPPED, FAILED):
            raise ValueError("failure termination must use alignment state vocabulary")
        self.state = state

    def summary(self) -> dict[str, Any]:
        return {"schema": "leapflow.noncontact_alignment.v1", "state": self.state,
                "run_id": self._run_id, "reference_digest": self.reference.digest,
                "candidate_digest": self.response.candidate_digest,
                "nominal_model_digest": self.nominal.digest, "corrected_model_digest": self.corrected.digest,
                "corrections": self.corrections, "travel_m": self.travel_m,
                "joint_l1_rad": self.joint_l1_rad, "pending_action": self._pending is not None,
                "response_updates": [dict(update) for update in self._response_updates],
                "physical_activation_qualified": False, "grasp_ready": False}


@runtime_checkable
class AlignmentObservations(Protocol):
    """Admitted settled captures with independently validated witness and health."""
    async def capture(self) -> RawTaskState: ...


def check_alignment_feedback(
    guard: AlignmentGuard, current: RawTaskState, live: ArmState,
) -> None:
    """Check health even during the stationary sustained-outcome branch."""
    if not isinstance(current, RawTaskState) or not isinstance(live, ArmState):
        raise SkillError(FAILURE_CHECK_FAILED, "typed measured observation and coherent arm feedback required")
    if (
        live.enabled is not True or live.arrived is not True
        or not isinstance(live.ctrl_mode, str) or live.ctrl_mode not in guard.profile.allowed_modes
        or type(live.arm_status) is not int or live.arm_status != 0
        or not isinstance(live.joint_communication_errors, tuple) or live.joint_communication_errors
        or not isinstance(live.joints, Mapping)
    ):
        raise SkillError(FAILURE_CHECK_FAILED, "raw arm health/mode changed during alignment")
    for name, expected, uncertainty in zip(current.joint_ids, current.joints_rad, current.joint_uncertainty_rad):
        value = live.joints.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SkillError(FAILURE_CHECK_FAILED, "finite complete raw joint feedback required for alignment")
        if abs(math.radians(value) - expected) > max(uncertainty, 1e-6):
            raise SkillError(FAILURE_PLAN_STALE, "raw arm state changed since the measured alignment frame")


async def _compute(
    worker: AlignmentWorker, operation: str, payload: Mapping[str, Any], *,
    workcell_id: str, run_id: str, model_digest: str,
) -> Mapping[str, Any]:
    try:
        return await worker.submit(
            operation, payload, workcell_id=workcell_id, run_id=run_id, model_digest=model_digest,
        )
    except CalibrationFailure as error:
        code = FAILURE_TIMEOUT if error.code == FAILURE_DEADLINE else (
            FAILURE_SKILL_UNAVAILABLE if error.code == FAILURE_WORKER_UNAVAILABLE else FAILURE_CHECK_FAILED
        )
        raise SkillError(code, f"alignment compute refused; no replay: {error}; inspect worker and evidence") from error


def _validate_response_update(
    fit: EstimateBundle, sample: RawTaskTransition, result: Mapping[str, Any], maximum: float,
) -> tuple[EstimateBundle, ResponseSnapshot]:
    """Small cold-path owner recomputation; NOT structural or physical admission.

    Recompute from the actual measured endpoints rather than trusting a worker's
    changed metadata, floors, holdouts, covariance or rank claims. This bounded
    4-by-N check runs off the hardware loop and returns the owner's immutable fit.
    """
    from leapcalib.estimation.raw_response import update_raw_response
    from leapcalib.runtime.protocol import freeze_json

    if not isinstance(result, Mapping) or set(result) != {"estimate"}:
        raise ValueError("typed bounded recursive response result required")
    wire = freeze_json({"estimate": result["estimate"]}, 2 * 1024 * 1024)
    reported = EstimateBundle.from_mapping(wire["estimate"])
    expected = update_raw_response(fit, sample, max_update_norm=maximum)
    if reported.digest != expected.digest:
        raise ValueError("worker update differs from owner measured correction recomputation")
    return expected, ResponseSnapshot.from_fit(expected)


async def _update_response(
    guard: AlignmentGuard, fit: EstimateBundle, binding: CorrectionBinding,
    first: RawTaskState, second: RawTaskState, *, worker: AlignmentWorker,
    workcell_id: str, run_id: str, cancelled: Callable[[], bool],
) -> EstimateBundle:
    try:
        sample = binding.transition(first, second)
        computed = await _compute(worker, "update_raw",
            {"fit": fit.to_dict(), "sample": sample.to_dict(), "max_update_norm": guard.profile.max_update_norm},
            workcell_id=workcell_id, run_id=run_id, model_digest=guard.response.model_digest)
        if cancelled():
            raise SkillError(FAILURE_STOPPED, "alignment cancelled during response update; no motion replay")
        updated, response = await _settle_validation(
            _validate_response_update, fit, sample, computed, guard.profile.max_update_norm,
        )
        if cancelled():
            raise SkillError(FAILURE_STOPPED, "alignment cancelled during owner update validation; no motion replay")
        guard.publish_response_update(response, sample)
        return updated
    except (CalibrationFailure, ValueError, TypeError, KeyError) as exc:
        raise SkillError(FAILURE_CHECK_FAILED, f"measured response update refused; inspect evidence before another run: {exc}") from exc


async def _settle_validation(function: Callable[..., Any], *args: Any) -> Any:
    """Retain run ownership until bounded pure validation settles after cancellation."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break  # Inspect the settled result once; never repeat the computation.
    if cancelled:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()


async def run_alignment(guard: AlignmentGuard, observations: AlignmentObservations, *,
                        worker: AlignmentWorker | None, motion: Any, workcell_id: str, run_id: str,
                        fit: EstimateBundle, clock: Callable[[], float], cancelled: Callable[[], bool],
                        correction_binding: CorrectionBinding | None = None) -> Mapping[str, Any]:
    """One bounded loop over the existing governed motion facade, never a retry path."""
    if not isinstance(observations, AlignmentObservations) or fit.digest != guard.response.candidate_digest:
        raise SkillError(FAILURE_PLAN_STALE, "alignment needs admitted observations and frozen response")
    # Detach caller-owned metadata before the first suspension. The run owns its
    # response versions; a mutable fit container cannot change a frozen policy.
    from leapcalib.runtime.protocol import freeze_json
    fit = EstimateBundle.from_mapping(freeze_json(fit.to_dict(), 2 * 1024 * 1024))
    if guard.profile.response_update_enabled:
        if not isinstance(correction_binding, CorrectionBinding) or not isinstance(worker, AlignmentWorker):
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, "enabled adaptation requires the sealed acquisition binding and owner worker")
        try:
            correction_binding.validate(fit, run_id, guard.response.domain.scope_digest)
        except (ValueError, KeyError) as exc:
            raise SkillError(FAILURE_PLAN_STALE, str(exc)) from exc
    guard.start(run_id, clock())
    try:
        while True:
            if cancelled():
                raise SkillError(FAILURE_STOPPED, "non-contact alignment cancelled")
            current = await observations.capture()
            if cancelled():
                raise SkillError(FAILURE_STOPPED, "non-contact alignment cancelled after capture")
            live = await motion.state()
            check_alignment_feedback(guard, current, live)
            motion.check_kinematics(live)
            if cancelled():
                raise SkillError(FAILURE_STOPPED, "non-contact alignment cancelled during observation feedback check")
            if guard.evaluate(current, now_s=clock()):
                return guard.summary()
            error = tuple(abs(a - b) for a, b in zip(current.task.feature_m_rad[:3], guard.reference.task.feature_m_rad[:3])) + (abs(_yaw_delta(current.task.feature_m_rad[3], guard.reference.task.feature_m_rad[3])),)
            if all(e + u <= t for e, u, t in zip(error, guard.uncertainty(current), guard.profile.tolerance_m_rad)):
                continue  # Sustain stationary observations; never move inside tolerance.
            if not isinstance(worker, AlignmentWorker):
                raise SkillError(FAILURE_SKILL_UNAVAILABLE, "alignment requires the enabled owner compute worker")
            computed = await _compute(worker, "propose_raw", {"fit": fit.to_dict(), "current": current.to_dict(),
                "target": list(guard.reference.task.feature_m_rad),
                "now": Stamp(clock()).to_dict()},
                workcell_id=workcell_id, run_id=run_id, model_digest=guard.response.model_digest)
            if cancelled():
                raise SkillError(FAILURE_STOPPED, "non-contact alignment cancelled before dispatch")
            if not isinstance(computed, Mapping) or set(computed) != {"proposal"}:
                raise SkillError(FAILURE_CHECK_FAILED, "typed bounded raw proposal required")
            live = await motion.state()
            check_alignment_feedback(guard, current, live)
            motion.check_kinematics(live)
            if cancelled():
                raise SkillError(FAILURE_STOPPED, "non-contact alignment cancelled during final feedback check")
            targets = guard.accept_proposal(computed["proposal"], current, now_s=clock())
            await motion.move_joints(targets, speed_pct=guard.profile.speed_pct)
            if cancelled():
                raise SkillError(FAILURE_STOPPED, "non-contact alignment cancelled after dispatch; verify settled state")
            arrival = await observations.capture()
            if cancelled():
                raise SkillError(FAILURE_STOPPED, "non-contact alignment cancelled during arrival capture")
            live = await motion.state()
            check_alignment_feedback(guard, arrival, live)
            motion.check_kinematics(live)
            if cancelled():
                raise SkillError(FAILURE_STOPPED, "non-contact alignment cancelled during arrival feedback check")
            guard.arrived(arrival, now_s=clock())
            if guard.profile.response_update_enabled:
                fit = await _update_response(guard, fit, correction_binding, current, arrival,
                    worker=worker, workcell_id=workcell_id, run_id=run_id, cancelled=cancelled)
    except asyncio.CancelledError:
        guard.end(STOPPED)
        raise
    except SkillError as error:
        guard.end(STOPPED if cancelled() or error.failure_code == FAILURE_STOPPED else LOST if error.failure_code == FAILURE_TARGET_NOT_FOUND else FAILED)
        error.result = {**error.result, "alignment": guard.summary()}
        raise
    except Exception:
        guard.end(STOPPED if cancelled() else FAILED)
        # The runtime owns halt/hold and side-effect merging. Never send a recovery
        # motion here; preserve a pending action for its uncertain-dispatch handling.
        raise
