# Copyright (c) Alibaba, Inc. and its affiliates.
"""Measured four-DoF response adaptation and data-only bounded shadow proposals.

A local response is not global geometry and is never sent to an actuator here.
Inputs are actual endpoint displacement, not commanded sub-arrival corrections.
Normalized xyz/yaw coordinates make the response coefficients dimensionless;
normalization and the original SI envelopes are part of the candidate digest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from leapcalib.contracts.estimates import EstimateBundle, ParameterEstimate, retained_estimate
from leapcalib.contracts.failures import CalibrationFailure, FAILURE_INSUFFICIENT_EVIDENCE
from leapcalib.contracts.identity import digest_of
from leapcalib.contracts.motion import finite_vector
from leapcalib.contracts.observations import Stamp
from leapcalib.contracts.parameters import (
    COMPOSE_ADDITIVE,
    KIND_TASK_RESPONSE,
    SUPPORT_PARTIAL,
    SUPPORT_SUPPORTED,
    ParameterBlock,
)
from leapcalib.contracts.task import TaskDomain, TaskState, TaskTransition
from leapcalib.contracts.units import canonical_number
from leapcalib.estimation.numeric import numpy, rank_report, wrap_rad
from leapcalib.validation.bundle import EvidenceBundle

METHOD_ID = "task_response.v1"


@dataclass(frozen=True)
class ResponseConfig:
    max_samples: int = 256
    trials: int = 48
    seed: int = 0
    min_samples: int = 8
    rank_rcond: float = 1e-4
    max_condition: float = 1e4
    residual_gate: float = 0.5
    min_inlier_ratio: float = 0.65
    min_motion_ratio: float = 0.05
    max_transition_s: float = 30.0
    damping: float = 0.05
    confidence_multiplier: float = 3.0
    uncertainty_floor_m_rad: tuple[float, float, float, float] = (0.0005, 0.0005, 0.0005, 0.005)

    def __post_init__(self) -> None:
        for name in ("max_samples", "trials", "min_samples"):
            value = getattr(self, name)
            if type(value) is not int or value < 1 or value > 1024:
                raise ValueError("response work budgets must be positive bounded integers")
        if (
            self.min_samples < 4
            or self.min_samples > self.max_samples
            or type(self.seed) is not int
            or self.seed < 0
        ):
            raise ValueError("invalid response sample/seed budget")
        for name in (
            "rank_rcond",
            "max_condition",
            "residual_gate",
            "min_inlier_ratio",
            "min_motion_ratio",
            "max_transition_s",
            "damping",
            "confidence_multiplier",
        ):
            value = canonical_number(getattr(self, name))
            object.__setattr__(self, name, value)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_inlier_ratio > 1 or self.rank_rcond >= 1 or self.max_condition <= 1:
            raise ValueError("invalid response rank/consensus bounds")
        object.__setattr__(
            self, "uncertainty_floor_m_rad", finite_vector(self.uncertainty_floor_m_rad, 4)
        )
        if any(v <= 0 for v in self.uncertainty_floor_m_rad):
            raise ValueError("response uncertainty floors must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in vars(self).items()
        }


def difference(second: Sequence[float], first: Sequence[float]) -> Any:
    np = numpy()
    result = np.asarray(second) - np.asarray(first)
    result[3] = wrap_rad(result[3])
    return result


def domain_pose(pose: Sequence[float], domain: TaskDomain) -> Any:
    np = numpy()
    result = np.asarray(pose).copy()
    centre = (domain.lower_m_rad[3] + domain.upper_m_rad[3]) / 2
    result[3] = centre + wrap_rad(result[3] - centre)
    return result


def in_domain(state: TaskState, domain: TaskDomain) -> bool:
    np = numpy()
    pose = domain_pose(state.pose_m_rad, domain)
    return bool(np.all(pose >= domain.lower_m_rad) and np.all(pose <= domain.upper_m_rad))


def _invalid(sample: TaskTransition, domain: TaskDomain, config: ResponseConfig) -> str:
    np = numpy()
    for state in (sample.first, sample.second):
        if not state.measured or not state.settled:
            return "endpoint/feature motion was not actually measured at settled states"
        if state.feature_id != domain.feature_id or state.domain_id != domain.domain_id:
            return "feature or local validity domain changed"
        if not in_domain(state, domain):
            return "measured endpoint is outside the local envelope"
        if np.any(
            np.asarray(state.uncertainty_m_rad) * config.confidence_multiplier
            > np.asarray(domain.tolerance_m_rad)
        ):
            return "measurement uncertainty exceeds the task budget"
        if state.stamp.uncertainty_s > domain.max_age_s:
            return "timestamp uncertainty exceeds the local task age budget"
    if sample.stamp.delta_s(sample.first.stamp) > config.max_transition_s:
        return "transition spans an unqualified time interval"
    delta = difference(sample.second.pose_m_rad, sample.first.pose_m_rad)
    noise = np.asarray(sample.first.uncertainty_m_rad) + sample.second.uncertainty_m_rad
    if not np.any(
        np.abs(delta)
        > np.maximum(
            np.asarray(domain.max_step_m_rad) * config.min_motion_ratio,
            noise * config.confidence_multiplier,
        )
    ):
        return "no informative actual displacement (a command is not measured motion)"
    return ""


def estimate_response(
    bundle: EvidenceBundle, domain: TaskDomain, *, config: ResponseConfig = ResponseConfig()
) -> EstimateBundle:
    by_id = {o.observation_id: o for o in bundle.observations}
    training = [by_id[i] for i in bundle.accepted_ids if isinstance(by_id[i], TaskTransition)]
    holdouts = [by_id[i] for i in bundle.holdout_ids if isinstance(by_id[i], TaskTransition)]
    if len(training) + len(holdouts) > config.max_samples:
        raise CalibrationFailure(FAILURE_INSUFFICIENT_EVIDENCE, "task sample budget exceeded")
    if len({s.stream.digest for s in training + holdouts}) > 1:
        raise ValueError("local response cannot mix stream profiles")
    reasons = {
        s.observation_id: reason for s in training if (reason := _invalid(s, domain, config))
    }
    valid = [s for s in training if s.observation_id not in reasons]
    return _fit_response(bundle, domain, config, valid, holdouts, reasons)


def _fit_response(
    bundle: EvidenceBundle,
    domain: TaskDomain,
    config: ResponseConfig,
    valid: Sequence[TaskTransition],
    holdouts: Sequence[TaskTransition],
    reasons: dict[str, str],
) -> EstimateBundle:
    np = numpy()
    steps, tolerances = np.asarray(domain.max_step_m_rad), np.asarray(domain.tolerance_m_rad)
    x = np.asarray(
        [difference(s.second.pose_m_rad, s.first.pose_m_rad) / steps for s in valid]
    ).reshape(-1, 4)
    y = np.asarray(
        [difference(s.second.feature_m_rad, s.first.feature_m_rad) / tolerances for s in valid]
    ).reshape(-1, 4)
    rank, condition, singular, null = rank_report(x, config.rank_rcond)
    inputs = [i for i in range(4) if not len(null) or float(np.linalg.norm(null[:, i])) < 1e-5]
    mask = np.zeros(len(valid), dtype=bool)
    coefficients = np.zeros((4, 4))
    if (
        len(valid) >= config.min_samples
        and inputs
        and condition is not None
        and condition <= config.max_condition
    ):
        mask, coefficients = _robust_response(x, y, inputs, config)
        rank, condition, singular, null = rank_report(x[mask], config.rank_rcond)
        final_inputs = [
            i for i in range(4) if not len(null) or float(np.linalg.norm(null[:, i])) < 1e-5
        ]
        if final_inputs != inputs or condition is None or condition > config.max_condition:
            mask[:] = False
            coefficients[:] = 0
    accepted = [s for s, ok in zip(valid, mask) if ok]
    for sample, ok in zip(valid, mask):
        if not ok:
            reasons[sample.observation_id] = "insufficient excitation or response consensus"
    block = ParameterBlock(
        "response",
        KIND_TASK_RESPONSE,
        16,
        ("",) * 16,
        composition=COMPOSE_ADDITIVE,
        notes="row-major dimensionless J: feature/tolerance = J @ pose/max_step",
    )
    limitations = (
        "shadow only; no motion or global structural calibration authority",
        "local xyz/yaw response is valid only for the recorded feature/domain/topology",
        "shared source states are correlated; uncertainty is conditional, not precision certification",
    )
    uncertainty = np.ones(16)
    residual_floor = np.asarray(config.uncertainty_floor_m_rad)
    support = tuple(row * 4 + column for row in range(4) for column in inputs) if accepted else ()
    if support:
        residual = y[mask] - x[mask] @ coefficients.T
        rms = np.sqrt(np.mean(residual**2, axis=0)) * tolerances
        # Exact residuals cannot erase stated endpoint/feature measurement uncertainty.
        measured_floors = []
        for sample in accepted:
            noise = np.asarray(sample.first.uncertainty_m_rad) + sample.second.uncertainty_m_rad
            timing = sample.first.stamp.uncertainty_s + sample.second.stamp.uncertainty_s
            noise[:3] += (
                domain.speed_m_s + domain.lever_arm_m * domain.rotation_rate_rad_s
            ) * timing
            noise[3] += domain.rotation_rate_rad_s * timing
            measured_floors.append(noise + np.abs(coefficients) @ (noise / steps) * tolerances)
        residual_floor = np.maximum.reduce([residual_floor, rms, np.max(measured_floors, axis=0)])
        covariance_x = np.linalg.pinv(
            x[mask][:, inputs].T @ x[mask][:, inputs], rcond=config.rank_rcond
        )
        for row in range(4):
            for j, column in enumerate(inputs):
                uncertainty[row * 4 + column] = max(
                    1e-6,
                    residual_floor[row] / tolerances[row] * math.sqrt(max(0, covariance_x[j, j])),
                )
        estimate = ParameterEstimate(
            block,
            tuple(coefficients.ravel()),
            (0.0,) * 16,
            SUPPORT_SUPPORTED if len(support) == 16 else SUPPORT_PARTIAL,
            tuple(uncertainty),
            (1.0,) * 16,
            support,
            method_id=METHOD_ID,
            accepted_count=len(accepted),
            limitations=limitations,
        )
    else:
        inputs = []
        estimate = retained_estimate(
            block,
            (0.0,) * 16,
            (1.0,) * 16,
            reason="actual measured excitation does not support the response",
        )
    predictions, heldout_rejections = [], {}
    for sample in holdouts:
        if reason := _invalid(sample, domain, config):
            heldout_rejections[sample.observation_id] = reason
            continue
        predicted = coefficients @ (
            difference(sample.second.pose_m_rad, sample.first.pose_m_rad) / steps
        )
        error = predicted * tolerances - difference(
            sample.second.feature_m_rad, sample.first.feature_m_rad
        )
        predictions.append({"observation_id": sample.observation_id, "error_m_rad": error.tolist()})
    states = [
        domain_pose(s.pose_m_rad, domain)
        for sample in accepted
        for s in (sample.first, sample.second)
    ]
    support_lower = np.min(states, axis=0).tolist() if states else list(domain.lower_m_rad)
    support_upper = np.max(states, axis=0).tolist() if states else list(domain.upper_m_rad)
    metadata = {
        "evidence_digest": bundle.digest,
        "domain": domain.to_dict(),
        "config": config.to_dict(),
        "data_rank": rank,
        "condition": condition,
        "nullspace_scaled": null.tolist(),
        "singular_values": singular.tolist(),
        "supported_inputs": inputs,
        "response_rank": int(np.linalg.matrix_rank(coefficients, tol=config.rank_rcond)),
        "prior_rows_in_rank": 0,
        "residual_floor_m_rad": residual_floor.tolist(),
        "support_lower_m_rad": support_lower,
        "support_upper_m_rad": support_upper,
        "rejections": reasons,
        "holdout_rejections": heldout_rejections,
        "holdout_predictions": predictions,
        "provenances": sorted({s.provenance for s in valid}),
    }
    identity = digest_of(
        {
            "evidence": bundle.digest,
            "method": METHOD_ID,
            "domain": domain.to_dict(),
            "config": config.to_dict(),
        }
    )
    return EstimateBundle(
        identity,
        bundle.topology,
        (estimate,),
        tuple(s.observation_id for s in accepted),
        tuple(sorted(set(bundle.rejected_ids) | set(reasons))),
        bundle.holdout_ids,
        METHOD_ID,
        limitations=limitations,
        metadata=metadata,
    )


def _robust_response(
    x: Any, y: Any, inputs: Sequence[int], config: ResponseConfig
) -> tuple[Any, Any]:
    np = numpy()
    rng = np.random.default_rng(config.seed)
    indices = list(inputs)
    trials = [np.arange(len(x))]
    trials.extend(
        rng.choice(len(x), max(4, len(inputs)), replace=False) for _ in range(config.trials)
    )
    best = np.zeros(len(x), dtype=bool)
    best_score = (-1, float("-inf"))
    for sample in trials:
        subset = x[sample][:, indices]
        if np.linalg.matrix_rank(subset, tol=config.rank_rcond) < len(inputs):
            continue
        fit = np.linalg.lstsq(subset, y[sample], rcond=config.rank_rcond)[0]
        errors = np.linalg.norm(x[:, indices] @ fit - y, axis=1)
        mask = errors <= config.residual_gate
        score = (int(mask.sum()), -float(np.median(errors)))
        if score > best_score:
            best, best_score = mask, score
    minimum = max(config.min_samples, math.ceil(len(x) * config.min_inlier_ratio))
    coefficients = np.zeros((4, 4))
    for _ in range(8):
        if int(best.sum()) < minimum:
            return np.zeros(len(x), dtype=bool), coefficients
        fit = np.linalg.lstsq(x[best][:, indices], y[best], rcond=config.rank_rcond)[0]
        mask = np.linalg.norm(x[:, indices] @ fit - y, axis=1) <= config.residual_gate
        if np.array_equal(mask, best):
            coefficients[:, indices] = fit.T
            return best, coefficients
        best = mask
    return np.zeros(len(x), dtype=bool), coefficients


def propose_shadow(
    fit: EstimateBundle,
    current: TaskState,
    target_m_rad: Sequence[float],
    domain: TaskDomain,
    now: Stamp,
    *,
    topology_digest: str,
) -> dict[str, Any]:
    """Return a bounded data-only proposal or explicit no-op/refusal, never execute it."""
    np = numpy()
    target = np.asarray(finite_vector(target_m_rad, 4))
    if fit.method_id != METHOD_ID or fit.topology.digest != topology_digest:
        raise ValueError("shadow request differs from candidate method/topology")
    if fit.metadata["domain"] != domain.to_dict():
        raise ValueError("shadow request cannot alter the fitted validity envelope")
    config = ResponseConfig(**fit.metadata["config"])
    age = now.delta_s(current.stamp) + now.uncertainty_s + current.stamp.uncertainty_s
    reason = ""
    if age < 0 or age > domain.max_age_s or now.seconds < current.stamp.seconds:
        reason = "observation is stale or from the future"
    elif (
        not current.measured
        or not current.settled
        or current.feature_id != domain.feature_id
        or current.domain_id != domain.domain_id
    ):
        reason = "target lost/changed or current state is not measured and settled"
    elif not in_domain(current, domain):
        reason = "current endpoint is outside the local envelope"
    current_uncertainty = np.asarray(current.uncertainty_m_rad) * config.confidence_multiplier
    delay_m = (domain.speed_m_s + domain.lever_arm_m * domain.rotation_rate_rad_s) * max(0, age)
    current_uncertainty[:3] += delay_m
    current_uncertainty[3] += domain.rotation_rate_rad_s * max(0, age)
    if not reason and np.any(current_uncertainty > domain.tolerance_m_rad):
        reason = "current measurement uncertainty/latency exceeds the task tolerance"
    error = difference(target, current.feature_m_rad)
    if reason:
        return {
            "shadow_only": True,
            "accepted": False,
            "reason": reason,
            "age_s": age,
            "candidate_digest": fit.digest,
            "executed": False,
        }
    if np.all(np.abs(error) + current_uncertainty <= domain.tolerance_m_rad):
        return {
            "shadow_only": True,
            "accepted": True,
            "no_op": True,
            "delta_m_rad": [0.0] * 4,
            "reason": "already within measured task tolerance",
            "candidate_digest": fit.digest,
            "uncertainty_m_rad": current_uncertainty.tolist(),
            "executed": False,
        }
    inputs = fit.metadata["supported_inputs"]
    if not inputs:
        return {
            "shadow_only": True,
            "accepted": False,
            "reason": "response remains unobserved",
            "candidate_digest": fit.digest,
            "executed": False,
        }
    return _bounded_proposal(fit, current, error, domain, config, inputs, age)


def _bounded_proposal(
    fit: EstimateBundle,
    current: TaskState,
    error: Any,
    domain: TaskDomain,
    config: ResponseConfig,
    inputs: Sequence[int],
    age: float,
) -> dict[str, Any]:
    np = numpy()
    steps, tolerances = np.asarray(domain.max_step_m_rad), np.asarray(domain.tolerance_m_rad)
    response = np.asarray(fit.estimates[0].values).reshape(4, 4)
    active = response[:, inputs]
    # SVD projection detects task error directions no measured input can change.
    projected = active @ np.linalg.lstsq(active, error / tolerances, rcond=config.rank_rcond)[0]
    if np.any(np.abs(projected - error / tolerances) > 1):
        return {
            "shadow_only": True,
            "accepted": False,
            "reason": "task error includes unsupported dimensions",
            "candidate_digest": fit.digest,
            "executed": False,
        }
    augmented = np.vstack([active, np.eye(len(inputs)) * config.damping])
    command = np.linalg.lstsq(
        augmented, np.r_[error / tolerances, np.zeros(len(inputs))], rcond=config.rank_rcond
    )[0]
    normalized = np.zeros(4)
    normalized[list(inputs)] = command
    normalized /= max(1, float(np.max(np.abs(normalized))))
    delta = normalized * steps
    pose = domain_pose(current.pose_m_rad, domain)
    uncertainty = np.asarray(current.uncertainty_m_rad) + np.asarray(
        fit.metadata["residual_floor_m_rad"]
    )
    coefficient_uncertainty = np.asarray(fit.estimates[0].uncertainty).reshape(4, 4)
    uncertainty += np.abs(normalized) @ coefficient_uncertainty.T * tolerances
    uncertainty *= config.confidence_multiplier
    delay_m = (domain.speed_m_s + domain.lever_arm_m * domain.rotation_rate_rad_s) * age
    uncertainty[:3] += delay_m
    uncertainty[3] += domain.rotation_rate_rad_s * age
    clearance_error = float(np.linalg.norm(uncertainty[:3]) + domain.lever_arm_m * uncertainty[3])
    lower = np.maximum(domain.lower_m_rad, fit.metadata["support_lower_m_rad"])
    upper = np.minimum(domain.upper_m_rad, fit.metadata["support_upper_m_rad"])
    proposed = pose + delta
    reason = ""
    if np.any(uncertainty > tolerances) or clearance_error >= domain.clearance_m:
        reason = "uncertainty/latency consumes the task tolerance or clearance"
    elif np.any(proposed[:3] - clearance_error < lower[:3]) or np.any(
        proposed[:3] + clearance_error > upper[:3]
    ):
        reason = "proposal leaves the uncertainty-inflated measured local envelope"
    elif proposed[3] - uncertainty[3] < lower[3] or proposed[3] + uncertainty[3] > upper[3]:
        reason = "proposal yaw leaves the measured local envelope"
    predicted = error - response @ normalized * tolerances
    if (
        not reason
        and np.linalg.norm(predicted / tolerances) >= np.linalg.norm(error / tolerances) - 1e-6
    ):
        reason = "proposal predicts no informative improvement"
    return {
        "shadow_only": True,
        "accepted": not reason,
        "reason": reason or "bounded shadow proposal",
        "delta_m_rad": delta.tolist(),
        "predicted_error_m_rad": predicted.tolist(),
        "uncertainty_m_rad": uncertainty.tolist(),
        "age_s": age,
        "clearance_error_m": clearance_error,
        "candidate_digest": fit.digest,
        "domain_digest": digest_of(domain.to_dict()),
        "executed": False,
    }


class ResponseMethod:
    method_id = METHOD_ID

    def solve(self, evidence: EvidenceBundle, options: Mapping[str, Any]) -> EstimateBundle:
        return estimate_response(
            evidence,
            TaskDomain.from_mapping(options["domain"]),
            config=ResponseConfig(**options.get("config", {})),
        )


__all__ = ["METHOD_ID", "ResponseConfig", "ResponseMethod", "estimate_response", "propose_shadow"]
