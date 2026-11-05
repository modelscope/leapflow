# Copyright (c) Alibaba, Inc. and its affiliates.
"""Paired withheld-unit evaluation, separate from acquisition trust and activation.

Numerical support and even a passing evaluation cannot establish independent
metrology. An owner must validate indexed acquisition, witnesses, geometry and
consumer scope before physical admission. Conditional AX=XB and local controller
responses never establish individual joint zeros or globally anchored geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from leapcalib.contracts.estimates import EstimateBundle
from leapcalib.contracts.identity import digest_of
from leapcalib.contracts.motion import MotionObservation, finite_vector
from leapcalib.contracts.observations import CLOCK_NODE_MONOTONIC, Stamp
from leapcalib.contracts.raw_task import RawTaskTransition, identities
from leapcalib.contracts.units import canonical_number
from leapcalib.validation.bundle import EvidenceBundle

SCHEMA = "leapcalib.withheld_evaluation.v1"
Factor = MotionObservation | RawTaskTransition


@dataclass(frozen=True)
class EvaluationConfig:
    min_units: int = 20
    separation_s: float = 0.3
    max_error_m_rad: tuple[float, ...] = (0.003, 0.003, 0.003, math.radians(2))
    confidence_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if type(self.min_units) is not int or not 20 <= self.min_units <= 1024:
            raise ValueError("evaluation requires at least twenty bounded withheld units")
        for name in ("separation_s", "confidence_multiplier"):
            object.__setattr__(self, name, canonical_number(getattr(self, name)))
        if not 0.3 <= self.separation_s <= 120 or not 1 <= self.confidence_multiplier <= 10:
            raise ValueError("invalid separation or conservative confidence policy")
        object.__setattr__(self, "max_error_m_rad", finite_vector(self.max_error_m_rad, 4))
        if any(
            not 0 < value <= cap
            for value, cap in zip(self.max_error_m_rad, (0.003, 0.003, 0.003, math.radians(2)))
        ):
            raise ValueError("evaluation cannot widen study error-plus-uncertainty limits")

    def to_dict(self) -> dict[str, Any]:
        return {**vars(self), "max_error_m_rad": list(self.max_error_m_rad)}


@dataclass(frozen=True)
class EvaluationUnit:
    observation_id: str
    candidate_error_m_rad: tuple[float, ...]
    baseline_error_m_rad: tuple[float, ...]
    uncertainty_m_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        identities((self.observation_id,))
        for name in ("candidate_error_m_rad", "baseline_error_m_rad", "uncertainty_m_rad"):
            values = finite_vector(getattr(self, name), 4)
            if any(v < 0 for v in values):
                raise ValueError("evaluation errors and uncertainty must be nonnegative")
            object.__setattr__(self, name, values)
        if any(v <= 0 for v in self.uncertainty_m_rad):
            raise ValueError("withheld evaluation must retain a nonzero measurement floor")

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            **{
                name: list(getattr(self, name))
                for name in ("candidate_error_m_rad", "baseline_error_m_rad", "uncertainty_m_rad")
            },
        }


@dataclass(frozen=True)
class CandidateEvaluation:
    """Immutable recomputation output, not an externally admissible certificate."""

    candidate_digest: str
    evidence_digest: str
    baseline_digest: str
    baseline_evidence_digest: str
    nuisance_digest: str
    method_id: str
    scope: str
    config: EvaluationConfig
    units: tuple[EvaluationUnit, ...]
    exclusions: tuple[tuple[str, str], ...]
    provenances: tuple[str, ...]

    def __post_init__(self) -> None:
        for binding in (
            self.candidate_digest,
            self.evidence_digest,
            self.baseline_digest,
            self.baseline_evidence_digest,
            self.nuisance_digest,
        ):
            identities((binding,))
        if self.method_id not in ("metric_motion.v1", "raw_joint_response.v1"):
            raise ValueError("unknown method-specific evaluation")
        expected = (
            "conditional_camera_motion"
            if self.method_id == "metric_motion.v1"
            else "local_raw_joint_response"
        )
        if self.scope != expected or not isinstance(self.config, EvaluationConfig):
            raise ValueError("evaluation scope cannot be promoted into structural geometry")
        if (
            not isinstance(self.units, (list, tuple))
            or len(self.units) > 1024
            or any(not isinstance(u, EvaluationUnit) for u in self.units)
        ):
            raise ValueError("bounded typed paired evaluation units required")
        object.__setattr__(self, "units", tuple(self.units))
        if not isinstance(self.exclusions, (list, tuple)) or len(self.exclusions) > 1024:
            raise ValueError("bounded explicit withheld exclusions required")
        exclusions = []
        for row in self.exclusions:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise ValueError("typed withheld exclusion required")
            identity, reason = row
            identities((identity,))
            if not isinstance(reason, str) or not reason or len(reason) > 4096:
                raise ValueError("exclusion requires a bounded reason")
            exclusions.append((identity, reason))
        object.__setattr__(self, "exclusions", tuple(exclusions))
        identities(
            tuple(u.observation_id for u in self.units) + tuple(i for i, _ in exclusions),
            minimum=0,
            maximum=1024,
        )
        object.__setattr__(self, "provenances", identities(self.provenances, minimum=0, maximum=3))

    @property
    def verdict(self) -> str:
        if len(self.units) < self.config.min_units:
            return "insufficient_evidence"
        # Invalid withheld branches cannot disappear behind a passing subset.
        if self.exclusions:
            return "rejected"
        limits = self.config.max_error_m_rad
        if any(
            any(
                e + u * self.config.confidence_multiplier > cap
                for e, u, cap in zip(unit.candidate_error_m_rad, unit.uncertainty_m_rad, limits)
            )
            for unit in self.units
        ):
            return "rejected"
        improvements = []
        noise = []
        for unit in self.units:
            before = max(v / limit for v, limit in zip(unit.baseline_error_m_rad, limits))
            after = max(v / limit for v, limit in zip(unit.candidate_error_m_rad, limits))
            improvements.append(before - after)
            noise.append(max(v / limit for v, limit in zip(unit.uncertainty_m_rad, limits)))
            # A required output cannot regress outside its measurement envelope.
            if any(
                c > b + self.config.confidence_multiplier * u
                for c, b, u in zip(
                    unit.candidate_error_m_rad, unit.baseline_error_m_rad, unit.uncertainty_m_rad
                )
            ):
                return "rejected"
        mean = sum(improvements) / len(improvements)
        variance = sum((v - mean) ** 2 for v in improvements) / (len(improvements) - 1)
        lower = mean - self.config.confidence_multiplier * math.sqrt(variance / len(improvements))
        if lower <= self.config.confidence_multiplier * max(noise):
            return "no_op"
        return "numerical_pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "candidate_digest": self.candidate_digest,
            "evidence_digest": self.evidence_digest,
            "baseline_digest": self.baseline_digest,
            "baseline_evidence_digest": self.baseline_evidence_digest,
            "nuisance_digest": self.nuisance_digest,
            "method_id": self.method_id,
            "scope": self.scope,
            "config": self.config.to_dict(),
            "units": [u.to_dict() for u in self.units],
            "exclusions": [{"observation_id": i, "reason": r} for i, r in self.exclusions],
            "provenances": list(self.provenances),
            "verdict": self.verdict,
            "physical_admission": "owner_indexed_acquisition_and_scope_required",
            "global_geometry": "not_established",
        }

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())


@dataclass(frozen=True)
class MotionMeasurementBounds:
    """Explicit nuisance bounds; their physical provenance remains owner-validated.

    Rate bounds describe the measured observation interval, not firmware speed
    percentages. Point uncertainty includes depth, intrinsics and association.
    No zero registration residual can erase these declared measurement floors.
    """

    observation_digest: str
    point_uncertainty_m: float
    robot_translation_uncertainty_m: float
    robot_rotation_uncertainty_rad: float
    translation_rate_m_s: float
    rotation_rate_rad_s: float

    def __post_init__(self) -> None:
        identities((self.observation_digest,))
        for name in (
            "point_uncertainty_m",
            "robot_translation_uncertainty_m",
            "robot_rotation_uncertainty_rad",
            "translation_rate_m_s",
            "rotation_rate_rad_s",
        ):
            value = canonical_number(getattr(self, name))
            if value < 0 or ("uncertainty" in name and value == 0):
                raise ValueError(
                    "measurement floors must be positive and measured rates nonnegative"
                )
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


def _window(sample: Factor) -> tuple[float, float]:
    stamps = (
        (sample.first_stamp, sample.stamp)
        if isinstance(sample, MotionObservation)
        else (
            sample.first.task.stamp,
            sample.first.feedback_stamp,
            sample.second.task.stamp,
            sample.second.feedback_stamp,
        )
    )
    if any(not isinstance(s, Stamp) or s.domain != CLOCK_NODE_MONOTONIC for s in stamps):
        raise ValueError("independent evaluation needs one explicit node-session clock")
    return min(s.seconds - s.uncertainty_s for s in stamps), max(
        s.seconds + s.uncertainty_s for s in stamps
    )


def independent_holdouts(
    training: Sequence[Factor],
    holdouts: Sequence[Factor],
    *,
    separation_s: float,
) -> tuple[tuple[Factor, ...], tuple[tuple[str, str], ...]]:
    """Count nonoverlapping physical windows, not merely different source strings.

    All training windows, including rejected fit rows, stay excluded. Every
    holdout exclusion is retained; input order cannot select a favorable subset.
    Separation is a necessary correlation bound, not proof of spatial excitation.
    """
    separation_s = canonical_number(separation_s)
    if (
        not 0.3 <= separation_s <= 120
        or not isinstance(training, (list, tuple))
        or not isinstance(holdouts, (list, tuple))
        or len(training) + len(holdouts) > 3072
    ):
        raise ValueError("bounded independently separated evaluation factors required")
    if any(
        not isinstance(s, (MotionObservation, RawTaskTransition)) for s in (*training, *holdouts)
    ):
        raise ValueError("typed method-specific evaluation factors required")
    merged: dict[str, Factor] = {}
    for sample in training:
        previous = merged.get(sample.observation_id)
        if previous is not None and digest_of(previous.to_dict()) != digest_of(sample.to_dict()):
            raise ValueError("conflicting training factor identity")
        merged[sample.observation_id] = sample
    identities(tuple(merged), minimum=0, maximum=2048)
    identities(tuple(s.observation_id for s in holdouts), minimum=0, maximum=1024)
    blocked = [
        (s, _window(s))
        for s in sorted(merged.values(), key=lambda s: (*_window(s), s.observation_id))
    ]
    selected, exclusions = [], []
    for sample in sorted(holdouts, key=lambda s: (*_window(s), s.observation_id)):
        first, last = _window(sample)
        reason = ""
        for other, (start, end) in blocked:
            if sample.observation_id == other.observation_id or set(sample.source_ids).intersection(
                other.source_ids
            ):
                reason = "shared frame/state/raw-feedback sources"
                break
            if not (first - end >= separation_s or start - last >= separation_s):
                reason = "overlapping or correlated adjacent physical windows"
                break
        if reason:
            exclusions.append((sample.observation_id, reason))
        else:
            selected.append(sample)
        # Even an excluded window cannot provide a fresh independent neighbor.
        blocked.append((sample, (first, last)))
    return tuple(selected), tuple(exclusions)


def _factors(
    bundle: EvidenceBundle,
    cls: type,
    config: EvaluationConfig,
    baseline_bundle: EvidenceBundle | None = None,
) -> tuple[tuple[Factor, ...], tuple[tuple[str, str], ...]]:
    session = bundle.metadata.get("node_session_id")
    if not isinstance(session, str) or not session.strip() or len(session) > 256:
        raise ValueError("withheld evaluation requires explicit node-session clock identity")
    if baseline_bundle is not None and baseline_bundle.metadata.get("node_session_id") != session:
        raise ValueError("baseline and candidate node-session clocks are not comparable")
    by_id = {o.observation_id: o for o in bundle.observations}
    if any(
        not isinstance(by_id[i], cls)
        for i in (*bundle.accepted_ids, *bundle.rejected_ids, *bundle.holdout_ids)
    ):
        raise ValueError("evaluation splits must contain only method-specific factors")
    training = [by_id[i] for i in (*bundle.accepted_ids, *bundle.rejected_ids)]
    if baseline_bundle is not None:
        before = {o.observation_id: o for o in baseline_bundle.observations}
        if any(
            digest_of(before[i].to_dict()) != digest_of(by_id[i].to_dict())
            for i in set(before) & set(by_id)
        ):
            raise ValueError("conflicting source identity between baseline and candidate evidence")
        if any(
            not isinstance(before[i], cls)
            for i in (*baseline_bundle.accepted_ids, *baseline_bundle.rejected_ids)
        ):
            raise ValueError("baseline training must contain method-specific factors")
        training.extend(
            before[i] for i in (*baseline_bundle.accepted_ids, *baseline_bundle.rejected_ids)
        )
    return independent_holdouts(
        tuple(training),
        tuple(by_id[i] for i in bundle.holdout_ids),
        separation_s=config.separation_s,
    )


def evaluate_motion_candidate(
    bundle: EvidenceBundle,
    fit: EstimateBundle,
    *,
    measurement_bounds: Mapping[str, MotionMeasurementBounds],
    config: EvaluationConfig = EvaluationConfig(),
) -> CandidateEvaluation:
    """Re-solve training and re-register withheld metric points against frozen prior."""
    from leapcalib.estimation.handeye import METHOD_ID, _prepare, estimate_motion
    from leapcalib.estimation.numeric import SolveConfig
    from leapcalib.geometry.se3 import Transform

    if fit.method_id != METHOD_ID or fit.topology != bundle.topology or len(fit.estimates) != 1:
        raise ValueError("motion evaluation needs the exact method and evidence topology")
    solve = SolveConfig(**fit.metadata["config"])
    prior = Transform.from_mapping(fit.metadata["prior_transform"])
    expected = estimate_motion(
        bundle, prior, prior_uncertainty=fit.estimates[0].prior_uncertainty, config=solve
    )
    if fit.digest != expected.digest:
        raise ValueError("candidate differs from independent evidence recomputation")
    if not isinstance(measurement_bounds, Mapping) or set(measurement_bounds) != set(
        bundle.holdout_ids
    ):
        raise ValueError("each withheld motion needs exactly one explicit measurement bound")
    by_id = {o.observation_id: o for o in bundle.observations}
    for identity, bound in measurement_bounds.items():
        if (
            not isinstance(bound, MotionMeasurementBounds)
            or bound.observation_digest != by_id[identity].digest
        ):
            raise ValueError("motion measurement bounds differ from exact source factor")
    selected, exclusions = _factors(bundle, MotionObservation, config)
    if (
        not fit.supported_blocks
        or not fit.metadata["consensus_stable"]
        or fit.metadata["prior_rows_in_rank"] != 0
    ):
        excluded = dict(exclusions)
        excluded.update(
            (o.observation_id, "no stable fully data-supported camera block to evaluate")
            for o in selected
        )
        return CandidateEvaluation(
            fit.digest,
            bundle.digest,
            digest_of(prior.to_dict()),
            bundle.digest,
            digest_of({i: b.to_dict() for i, b in measurement_bounds.items()}),
            METHOD_ID,
            "conditional_camera_motion",
            config,
            (),
            tuple(sorted(excluded.items())),
            tuple(sorted({s.provenance for s in selected})),
        )
    rejected: dict[str, str] = {}
    factors = _prepare(selected, solve, rejected)
    candidate = Transform.from_mapping(expected.metadata["candidate_transform"])
    floor_m = max(solve.translation_sigma_m, *fit.estimates[0].uncertainty[:3])
    floor_rad = max(solve.rotation_sigma_rad, *fit.estimates[0].uncertainty[3:])
    units = []
    for observation, measured in factors:
        after = candidate.inverse().compose(observation.robot_motion).compose(candidate)
        before = prior.inverse().compose(observation.robot_motion).compose(prior)
        # Norm errors applied per coordinate are conservative without an anchored
        # global basis; rotations are full SO(3), not a favorable Euler component.
        uncertainty = _motion_floor(
            observation, solve, measurement_bounds[observation.observation_id], floor_m, floor_rad
        )
        units.append(
            EvaluationUnit(
                observation.observation_id,
                (after.distance_m(measured),) * 3 + (after.angle_to_rad(measured),),
                (before.distance_m(measured),) * 3 + (before.angle_to_rad(measured),),
                uncertainty,
            )
        )
    return CandidateEvaluation(
        fit.digest,
        bundle.digest,
        digest_of(prior.to_dict()),
        bundle.digest,
        digest_of({i: b.to_dict() for i, b in measurement_bounds.items()}),
        METHOD_ID,
        "conditional_camera_motion",
        config,
        tuple(units),
        (*exclusions, *sorted(rejected.items())),
        tuple(sorted({s.provenance for s in selected})),
    )


def _motion_floor(
    observation: MotionObservation,
    solve: Any,
    bound: MotionMeasurementBounds,
    model_floor_m: float,
    model_floor_rad: float,
) -> tuple[float, ...]:
    from leapcalib.estimation.numeric import numpy
    from leapcalib.estimation.registration import register_points
    from leapcalib.contracts.motion import MODE_WRIST

    np = numpy()
    registered = register_points(
        observation.points, reverse=observation.mode == MODE_WRIST, config=solve
    )
    accepted = set(registered.accepted_features)
    spreads = []
    for endpoint in ("first_m", "second_m"):
        points = np.asarray(
            [getattr(p, endpoint) for p in observation.points if p.feature_id in accepted]
        )
        singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        spreads.append(float(singular[1]) / math.sqrt(len(points)))
    point_floor = 2 * bound.point_uncertainty_m + registered.max_error_m
    timing = observation.first_stamp.uncertainty_s + observation.stamp.uncertainty_s
    # The smaller in-plane RMS extent bounds rotation support. Vanishing support
    # yields a pi uncertainty, not a precision claim from a zero fit residual.
    angular = min(math.pi, 2 * math.asin(min(1.0, point_floor / max(min(spreads), 1e-12))))
    distance = (
        model_floor_m
        + point_floor
        + bound.robot_translation_uncertainty_m
        + bound.translation_rate_m_s * timing
    )
    angle = (
        model_floor_rad
        + angular
        + bound.robot_rotation_uncertainty_rad
        + bound.rotation_rate_rad_s * timing
    )
    return (distance,) * 3 + (angle,)


def evaluate_raw_candidate(
    bundle: EvidenceBundle,
    fit: EstimateBundle,
    baseline: EstimateBundle,
    *,
    baseline_bundle: EvidenceBundle,
    config: EvaluationConfig = EvaluationConfig(),
) -> CandidateEvaluation:
    """Pair the same withheld transitions with a frozen baseline, never training loss."""
    from leapcalib.estimation.raw_response import (
        METHOD_ID,
        estimate_raw_response,
        validate_raw_fit,
    )

    from leapcalib.contracts.raw_task import RawTaskDomain
    from leapcalib.estimation.response import ResponseConfig

    if any(
        not isinstance(value, EstimateBundle) or value.method_id != METHOD_ID
        for value in (fit, baseline)
    ) or not isinstance(config, EvaluationConfig):
        raise ValueError("typed raw-response candidates and evaluation policy required")
    domain = RawTaskDomain.from_mapping(fit.metadata["domain"])
    solve = ResponseConfig(**fit.metadata["config"])
    before_domain = RawTaskDomain.from_mapping(baseline.metadata["domain"])
    before_config = ResponseConfig(**baseline.metadata["config"])
    if (
        domain != before_domain
        or fit.topology != bundle.topology
        or baseline.topology != bundle.topology
    ):
        raise ValueError("paired baseline and candidate need the same frozen identity/domain")
    if baseline_bundle.topology != bundle.topology:
        raise ValueError("baseline evidence topology differs")
    expected = estimate_raw_response(bundle, domain, config=solve)
    expected_baseline = estimate_raw_response(baseline_bundle, before_domain, config=before_config)
    if fit.digest != expected.digest or baseline.digest != expected_baseline.digest:
        raise ValueError("candidate or baseline differs from independent evidence recomputation")
    selected, exclusions = _factors(bundle, RawTaskTransition, config, baseline_bundle)
    for value in (expected, expected_baseline):
        if value.supported_blocks:
            validate_raw_fit(value)
    if not expected.supported_blocks or not expected_baseline.supported_blocks:
        from leapcalib.estimation.raw_response import _invalid

        units = []
        rejected = [
            (
                sample.observation_id,
                _invalid(sample, domain, solve)
                or _invalid(sample, domain, before_config)
                or "candidate or baseline lacks a fully data-supported raw response",
            )
            for sample in selected
        ]
    else:
        units, rejected = _raw_units(
            selected, expected, expected_baseline, domain, solve, before_config
        )
    return CandidateEvaluation(
        fit.digest,
        bundle.digest,
        baseline.digest,
        baseline_bundle.digest,
        digest_of(
            {
                "domain": domain.to_dict(),
                "candidate_config": solve.to_dict(),
                "baseline_config": before_config.to_dict(),
            }
        ),
        METHOD_ID,
        "local_raw_joint_response",
        config,
        tuple(units),
        (*exclusions, *rejected),
        tuple(sorted({s.provenance for s in selected})),
    )


def _raw_units(
    samples: Sequence[RawTaskTransition],
    fit: EstimateBundle,
    baseline: EstimateBundle,
    domain: Any,
    solve: Any,
    before_config: Any,
) -> tuple[list[EvaluationUnit], list[tuple[str, str]]]:
    from leapcalib.estimation.raw_response import _data, _floor, _invalid
    from leapcalib.estimation.numeric import numpy

    np = numpy()
    after = np.asarray(fit.metadata["response_scaled"])
    before = np.asarray(baseline.metadata["response_scaled"])
    basis = np.asarray(fit.metadata["supported_basis_scaled"])
    baseline_basis = np.asarray(baseline.metadata["supported_basis_scaled"])
    baseline_training = set(baseline.metadata["training_source_ids"])
    units, rejected = [], []
    for sample in samples:
        reason = _invalid(sample, domain, solve) or _invalid(sample, domain, before_config)
        x, y = _data(sample, domain)
        if not reason and baseline_training.intersection(sample.source_ids):
            reason = "withheld sources trained the frozen baseline"
        if not reason and any(
            float(np.linalg.norm(x - b.T @ b @ x)) > 1e-6 for b in (basis, baseline_basis)
        ):
            reason = "withheld displacement outside common measured input support"
        lower = np.maximum(
            fit.metadata["support_lower_rad"], baseline.metadata["support_lower_rad"]
        )
        upper = np.minimum(
            fit.metadata["support_upper_rad"], baseline.metadata["support_upper_rad"]
        )
        if not reason and any(
            np.any(np.asarray(s.joints_rad) < lower) or np.any(np.asarray(s.joints_rad) > upper)
            for s in (sample.first, sample.second)
        ):
            reason = "withheld raw endpoints outside common measured region"
        if reason:
            rejected.append((sample.observation_id, reason))
            continue
        floor = np.maximum(
            _floor([sample], after, domain, solve), _floor([sample], before, domain, before_config)
        )
        parameter = (
            np.maximum(
                np.asarray(fit.estimates[0].uncertainty).reshape(4, len(basis)) @ np.abs(basis @ x),
                np.asarray(baseline.estimates[0].uncertainty).reshape(4, len(baseline_basis))
                @ np.abs(baseline_basis @ x),
            )
            * domain.task.tolerance_m_rad
        )
        floor = (
            np.maximum(
                floor,
                np.maximum(
                    fit.metadata["residual_floor_m_rad"], baseline.metadata["residual_floor_m_rad"]
                ),
            )
            + parameter
        )
        tolerance = np.asarray(domain.task.tolerance_m_rad)
        units.append(
            EvaluationUnit(
                sample.observation_id,
                tuple(np.abs(after @ x - y) * tolerance),
                tuple(np.abs(before @ x - y) * tolerance),
                tuple(floor),
            )
        )
    return units, rejected
