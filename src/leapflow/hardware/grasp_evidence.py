# Copyright (c) Alibaba, Inc. and its affiliates.
"""Measured grasp states and stage identity, independent of a vendor or mission."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Protocol, runtime_checkable


class GraspState(str, Enum):
    UNKNOWN = "unknown"
    EMPTY = "empty"
    CONTACT_CANDIDATE = "contact_candidate"
    HELD_VERIFIED = "held_verified"
    SLIPPED = "slipped"


@dataclass(frozen=True)
class EvidenceIdentity:
    session_id: str
    scene_id: str
    calibration_digest: str
    observed_mono: float
    frame_ids: tuple[str, ...]

    def current(self, other: EvidenceIdentity, *, now: float, max_age_s: float) -> bool:
        return (
            bool(self.session_id and self.scene_id and self.calibration_digest and self.frame_ids)
            and (self.session_id, self.scene_id, self.calibration_digest)
            == (other.session_id, other.scene_id, other.calibration_digest)
            and math.isfinite(now) and math.isfinite(self.observed_mono)
            and 0 <= now - self.observed_mono <= max_age_s
            and isinstance(self.frame_ids, (list, tuple)) and len(self.frame_ids) <= 256
            and all(isinstance(i, str) and i.strip() and len(i) <= 256 for i in self.frame_ids)
            and len(set(self.frame_ids)) == len(self.frame_ids)
        )


@dataclass(frozen=True)
class GraspLimits:
    max_age_s: float
    min_frames: int
    min_proof_lift_mm: float
    max_proof_lift_mm: float
    max_relative_error_mm: float
    max_slip_mm: float
    max_uncertainty_mm: float

    def __post_init__(self) -> None:
        values = (self.max_age_s, self.min_proof_lift_mm, self.max_proof_lift_mm,
                  self.max_relative_error_mm, self.max_slip_mm, self.max_uncertainty_mm)
        if (any(not math.isfinite(v) or v <= 0 for v in values) or isinstance(self.min_frames, bool) or not isinstance(self.min_frames, int)
                or not 2 <= self.min_frames <= 50
                or self.min_proof_lift_mm > self.max_proof_lift_mm):
            raise ValueError("invalid grasp evidence limits")


@dataclass(frozen=True)
class GraspEvidence:
    identity: EvidenceIdentity
    visible: bool
    synchronized: bool
    independent_jaws: bool
    pad_engagement: bool
    width_consistent: bool
    motion_conditioned_effort: bool
    empty_closure: bool
    object_on_support: bool
    object_rise_mm: float
    tool_rise_mm: float
    relative_error_mm: float
    width_change_mm: float
    uncertainty_mm: float
    observation_times: tuple[float, ...] = ()
    proof: GraspEvidence | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def state(self, current: EvidenceIdentity, *, now: float, limits: GraspLimits) -> GraspState:
        flags = (self.visible, self.synchronized, self.independent_jaws, self.pad_engagement,
                 self.width_consistent, self.motion_conditioned_effort, self.empty_closure,
                 self.object_on_support)
        if (any(type(flag) is not bool for flag in flags)
                or not self.identity.current(current, now=now, max_age_s=limits.max_age_s)):
            return GraspState.UNKNOWN
        values = (self.object_rise_mm, self.tool_rise_mm, self.relative_error_mm,
                  self.width_change_mm, self.uncertainty_mm)
        if any(not math.isfinite(v) for v in values) or self.relative_error_mm < 0:
            return GraspState.UNKNOWN
        if self.empty_closure:
            return GraspState.EMPTY
        if (not self.visible or not self.synchronized or not self.independent_jaws
                or not 0 <= self.uncertainty_mm <= limits.max_uncertainty_mm):
            return GraspState.UNKNOWN
        if (self.observation_times and (
                any(not math.isfinite(t) or not 0 <= now - t <= limits.max_age_s
                    for t in self.observation_times)
                or any(a >= b for a, b in zip(self.observation_times, self.observation_times[1:])))):
            return GraspState.UNKNOWN
        if abs(self.width_change_mm) > limits.max_slip_mm:
            return GraspState.SLIPPED
        contact = self.pad_engagement and self.width_consistent and self.motion_conditioned_effort
        if not contact:
            return GraspState.UNKNOWN
        if self.object_on_support:
            return GraspState.CONTACT_CANDIDATE
        count = len(self.observation_times)
        # Frame IDs within a synchronized bundle are not independent observations.
        proof_valid = (self.proof is not None and self.proof.proof is None
                       and self.proof.state(current, now=self.proof.identity.observed_mono,
                                            limits=limits) == GraspState.HELD_VERIFIED)
        lift_valid = limits.min_proof_lift_mm <= self.tool_rise_mm <= limits.max_proof_lift_mm
        if (count >= limits.min_frames and len(self.identity.frame_ids) >= count
                and self.observation_times[-1] == self.identity.observed_mono
                and (lift_valid or (proof_valid and self.tool_rise_mm >= limits.min_proof_lift_mm))
                and self.object_rise_mm - self.uncertainty_mm > 0
                and abs(self.object_rise_mm - self.tool_rise_mm) + self.uncertainty_mm
                <= limits.max_relative_error_mm
                and 0 <= self.relative_error_mm + self.uncertainty_mm <= limits.max_relative_error_mm):
            return GraspState.HELD_VERIFIED
        return GraspState.CONTACT_CANDIDATE


@runtime_checkable
class GraspObserver(Protocol):
    """Independent RGB-D/jaw observation; model geometry alone cannot satisfy it."""

    def observe_grasp(self) -> GraspEvidence: ...


@dataclass(frozen=True)
class GraspMeasurement:
    """A bounded, independently measured object/tool observation in the base frame.

    Values enter through an admitted reading source. Absence of an object is not
    a measured hold. Current/width consistency flags must come from qualified
    unloaded/stroke characterization, never a positive magnitude comparison.
    """
    identity: EvidenceIdentity
    object_mm: tuple[float, float, float]
    jaws_mm: tuple[float, float, float]
    width_mm: float
    uncertainty_mm: float
    visible: bool
    synchronized: bool
    independent_jaws: bool
    pad_engagement: bool
    width_consistent: bool
    motion_conditioned_effort: bool
    empty_closure: bool
    object_on_support: bool


class TemporalGraspObserver:
    """Derive proof-lift and relative stability from distinct measured frames."""

    def __init__(self, source: Callable[[], GraspMeasurement], *, limits: GraspLimits,
                 current: Callable[[], EvidenceIdentity], clock: Callable[[], float],
                 max_frame_ids: int = 4096):
        if isinstance(max_frame_ids, bool) or not isinstance(max_frame_ids, int) or max_frame_ids < 2 * limits.min_frames:
            raise ValueError("insufficient distinct-frame budget")
        self.source, self.limits, self.current, self.clock = source, limits, current, clock
        self.max_frame_ids = max_frame_ids
        self._contact: GraspMeasurement | None = None
        self._history: list[GraspMeasurement] = []
        self._proof: GraspEvidence | None = None
        self._seen: set[str] = set()
        self._scope: tuple[str, str, str] | None = None
        self._last_at = float("-inf")

    def _invalidate(self) -> None:
        self._contact, self._history, self._proof = None, [], None

    def observe_grasp(self) -> GraspEvidence:
        try:
            sample, current, now = self.source(), self.current(), self.clock()
        except Exception:
            # Lost observation continuity cannot retain a physical proof.
            self._invalidate()
            raise
        scope = (current.session_id, current.scene_id, current.calibration_digest)
        if scope != self._scope:
            self._invalidate()
            self._scope, self._seen, self._last_at = scope, set(), float("-inf")
        fresh = sample.identity.current(current, now=now, max_age_s=self.limits.max_age_s)
        finite = (len(sample.object_mm) == len(sample.jaws_mm) == 3
                  and all(math.isfinite(v) for v in (*sample.object_mm, *sample.jaws_mm,
                                                     sample.width_mm, sample.uncertainty_mm))
                  and sample.width_mm >= 0 and 0 <= sample.uncertainty_mm <= self.limits.max_uncertainty_mm)
        distinct = (not self._seen.intersection(sample.identity.frame_ids)
                    and sample.identity.observed_mono > self._last_at
                    and len(self._seen) + len(sample.identity.frame_ids) <= self.max_frame_ids)
        continuous = sample.identity.observed_mono - self._last_at <= self.limits.max_age_s
        flags = (sample.visible, sample.synchronized, sample.independent_jaws,
                 sample.pad_engagement, sample.width_consistent, sample.motion_conditioned_effort,
                 sample.empty_closure, sample.object_on_support)
        valid = (fresh and finite and distinct and all(type(flag) is bool for flag in flags)
                 and sample.visible and sample.synchronized and sample.independent_jaws)
        contact = (valid and sample.pad_engagement and sample.width_consistent
                   and sample.motion_conditioned_effort and not sample.empty_closure)
        if not contact or not continuous:
            self._invalidate()
        if fresh and distinct:
            self._seen.update(sample.identity.frame_ids)
            self._last_at = sample.identity.observed_mono
        if contact and sample.object_on_support:
            # A new support observation starts a new baseline, not stale geometry
            # from an earlier table placement or an old closure.
            self._contact = sample
        baseline = self._contact
        if sample.object_on_support:
            # A support observation cannot contribute to a post-lift stable window.
            self._history, self._proof = [], None
        elif baseline and contact:
            rise = sample.jaws_mm[2] - baseline.jaws_mm[2]
            object_rise = sample.object_mm[2] - baseline.object_mm[2]
            uncertainty = sample.uncertainty_mm + baseline.uncertainty_mm
            if rise < self.limits.min_proof_lift_mm or object_rise <= uncertainty:
                self._history, self._proof = [], None
            if self._proof is None and rise > self.limits.max_proof_lift_mm:
                valid = False
                self._invalidate()
                baseline = None
            self._history = [r for r in self._history
                             if 0 <= now - r.identity.observed_mono <= self.limits.max_age_s]
            if baseline and rise >= self.limits.min_proof_lift_mm and object_rise > uncertainty:
                self._history.append(sample)
            self._history = self._history[-self.limits.min_frames:]
        if not sample.object_on_support and baseline is None:
            valid = False
        evidence = self._evidence(sample, baseline, valid)
        state = evidence.state(current, now=now, limits=self.limits)
        if state == GraspState.HELD_VERIFIED and self._proof is None:
            self._proof = evidence
        elif state in (GraspState.UNKNOWN, GraspState.EMPTY, GraspState.SLIPPED):
            self._invalidate()
        return evidence

    def _evidence(self, sample: GraspMeasurement, baseline: GraspMeasurement | None,
                  valid: bool) -> GraspEvidence:
        object_rise = sample.object_mm[2] - baseline.object_mm[2] if baseline else 0.0
        tool_rise = sample.jaws_mm[2] - baseline.jaws_mm[2] if baseline else 0.0
        relative = 0.0
        if baseline and self._history:
            expected = tuple(a - b for a, b in zip(baseline.object_mm, baseline.jaws_mm))
            relative = max(math.dist(tuple(a - b for a, b in zip(r.object_mm, r.jaws_mm)), expected)
                           for r in self._history)
        frames = tuple(i for r in self._history for i in r.identity.frame_ids) or sample.identity.frame_ids
        identity = EvidenceIdentity(sample.identity.session_id, sample.identity.scene_id,
                                    sample.identity.calibration_digest, sample.identity.observed_mono, frames)
        return GraspEvidence(identity, valid, sample.synchronized, sample.independent_jaws,
                             sample.pad_engagement, sample.width_consistent, sample.motion_conditioned_effort,
                             sample.empty_closure, sample.object_on_support, object_rise, tool_rise,
                             relative, sample.width_mm - baseline.width_mm if baseline else 0.0,
                             sample.uncertainty_mm + baseline.uncertainty_mm if baseline else sample.uncertainty_mm,
                             tuple(r.identity.observed_mono for r in self._history), self._proof)
