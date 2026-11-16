# Copyright (c) Alibaba, Inc. and its affiliates.
"""Data-only measured raw-encoder response; numerical proposals never grant authority."""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping

from leapcalib.contracts.estimates import EstimateBundle, ParameterEstimate, retained_estimate
from leapcalib.contracts.failures import CalibrationFailure, FAILURE_INSUFFICIENT_EVIDENCE
from leapcalib.contracts.identity import digest_of
from leapcalib.contracts.motion import finite_vector
from leapcalib.contracts.observations import Stamp
from leapcalib.contracts.parameters import COMPOSE_ADDITIVE, KIND_TASK_RESPONSE, SUPPORT_SUPPORTED, ParameterBlock
from leapcalib.contracts.raw_task import RawTaskDomain, RawTaskState, RawTaskTransition
from leapcalib.estimation.numeric import numpy, rank_report
from leapcalib.estimation.response import ResponseConfig, difference
from leapcalib.validation.bundle import EvidenceBundle

METHOD_ID = "raw_joint_response.v1"


def _invalid(sample: RawTaskTransition, domain: RawTaskDomain, config: ResponseConfig) -> str:
    np = numpy()
    for state in (sample.first, sample.second):
        if not domain.accepts(state):
            return "changed identity/domain, unsettled state, or invalid independent witness"
        uncertainty = np.asarray(state.task.uncertainty_m_rad)
        if np.any(uncertainty * config.confidence_multiplier > domain.task.tolerance_m_rad):
            return "feature uncertainty exceeds task budget"
        if state.feedback_stamp.spread_s(state.task.stamp) > domain.task.max_age_s:
            return "feedback/frame timing uncertainty exceeds budget"
    if sample.stamp.delta_s(sample.first.task.stamp) > config.max_transition_s:
        return "transition exceeds qualified timing interval"
    delta = np.asarray(sample.second.joints_rad) - sample.first.joints_rad
    noise = np.asarray(sample.first.joint_uncertainty_rad) + sample.second.joint_uncertainty_rad
    if not np.any(np.abs(delta) > np.maximum(np.asarray(domain.max_step_rad) * config.min_motion_ratio, noise * config.confidence_multiplier)):
        return "below-resolution measured encoder displacement"
    return ""


def _data(sample: RawTaskTransition, domain: RawTaskDomain) -> tuple[Any, Any]:
    np = numpy()
    return ((np.asarray(sample.second.joints_rad) - sample.first.joints_rad) / domain.max_step_rad,
            difference(sample.second.task.feature_m_rad, sample.first.task.feature_m_rad) / domain.task.tolerance_m_rad)


def _basis(x: Any, config: ResponseConfig) -> tuple[Any, int, float | None]:
    np = numpy()
    rank, condition, _, _ = rank_report(x, config.rank_rcond)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return vt[:rank], rank, condition


def _floor(samples: list[RawTaskTransition], j: Any, domain: RawTaskDomain, config: ResponseConfig) -> Any:
    np = numpy()
    floor = np.asarray(config.uncertainty_floor_m_rad)
    for sample in samples:
        feature = np.asarray(sample.first.task.uncertainty_m_rad) + sample.second.task.uncertainty_m_rad
        raw = np.asarray(sample.first.joint_uncertainty_rad) + sample.second.joint_uncertainty_rad
        timing = sum(s.task.stamp.uncertainty_s + s.feedback_stamp.uncertainty_s for s in (sample.first, sample.second))
        speed = domain.task.speed_m_s + domain.task.lever_arm_m * domain.task.rotation_rate_rad_s
        feature[:3] += speed * timing
        feature[3] += domain.task.rotation_rate_rad_s * timing
        floor = np.maximum(floor, feature + np.abs(j) @ (raw / domain.max_step_rad) * domain.task.tolerance_m_rad)
    return floor


def estimate_raw_response(bundle: EvidenceBundle, domain: RawTaskDomain, *, config: ResponseConfig = ResponseConfig()) -> EstimateBundle:
    np = numpy()
    by_id = {s.observation_id: s for s in bundle.observations}
    training = [by_id[i] for i in bundle.accepted_ids if isinstance(by_id[i], RawTaskTransition)]
    holdout = [by_id[i] for i in bundle.holdout_ids if isinstance(by_id[i], RawTaskTransition)]
    if len(training) + len(holdout) > config.max_samples:
        raise CalibrationFailure(FAILURE_INSUFFICIENT_EVIDENCE, "raw response sample budget exceeded")
    if len({s.stream.digest for s in training + holdout}) > 1:
        raise ValueError("raw response cannot mix stream identities")
    reasons = {s.observation_id: reason for s in training if (reason := _invalid(s, domain, config))}
    valid = [s for s in training if s.observation_id not in reasons]
    n = len(domain.joint_ids)
    x = np.asarray([_data(s, domain)[0] for s in valid]).reshape(-1, n)
    y = np.asarray([_data(s, domain)[1] for s in valid]).reshape(-1, 4)
    basis, rank, condition = _basis(x, config)
    mask = np.zeros(len(valid), dtype=bool)
    j = np.zeros((4, n))
    if len(valid) >= config.min_samples and rank >= 4 and condition is not None and condition <= config.max_condition:
        z = x @ basis.T
        rng = np.random.default_rng(config.seed)
        best_score = (-1, float("-inf"))
        for trial in range(config.trials + 1):
            indices = np.arange(len(valid)) if trial == 0 else rng.choice(len(valid), rank, replace=False)
            if rank_report(z[indices], config.rank_rcond)[0] != rank:
                continue
            c = np.linalg.lstsq(z[indices], y[indices], rcond=config.rank_rcond)[0]
            errors = np.linalg.norm(y - z @ c, axis=1)
            selected = errors <= config.residual_gate
            score = (int(selected.sum()), -float(np.sum(np.minimum(errors, config.residual_gate))))
            if score > best_score:
                best_score, mask = score, selected
        if mask.sum() < max(config.min_samples, len(valid) * config.min_inlier_ratio):
            mask[:] = False
        else:
            basis, rank, condition = _basis(x[mask], config)
            if rank < 4 or condition is None or condition > config.max_condition:
                mask[:] = False
            else:
                z = x[mask] @ basis.T
                c = np.linalg.lstsq(z, y[mask], rcond=config.rank_rcond)[0]
                j = c.T @ basis
                if rank_report(c.T, config.rank_rcond)[0] < 4:
                    mask[:] = False
                    j[:] = 0
    accepted = [s for s, ok in zip(valid, mask) if ok]
    for sample, ok in zip(valid, mask):
        if not ok:
            reasons[sample.observation_id] = "insufficient supported excitation, output rank, or consensus"
    floor = _floor(accepted, j, domain, config)
    dimension = 4 * rank if accepted else 4 * n
    block = ParameterBlock("raw_response_subspace", KIND_TASK_RESPONSE, dimension, ("",) * dimension,
                           composition=COMPOSE_ADDITIVE,
                           notes="row-major feature/tolerance response on measured orthonormal raw/max_step basis; not individual joint columns")
    limitations = ("local controller response only; no structural activation authority",
                   "synthetic/replayed data cannot certify physical qualification",
                   "independence requires separate physical windows, not transition count")
    covariance = np.eye(rank) if rank else np.empty((0, 0))
    if accepted:
        z = x[mask] @ basis.T
        c = np.linalg.lstsq(z, y[mask], rcond=config.rank_rcond)[0].T
        floor = np.maximum(floor, np.sqrt(np.mean((y[mask] - z @ c.T)**2, axis=0)) * domain.task.tolerance_m_rad)
        covariance = np.linalg.pinv(z.T @ z, rcond=config.rank_rcond)
        uncertainty = (floor / domain.task.tolerance_m_rad)[:, None] * np.sqrt(np.maximum(np.diag(covariance), 0))[None, :]
        estimate = ParameterEstimate(block, tuple(c.ravel()), (0.0,) * dimension, SUPPORT_SUPPORTED,
                                     tuple(np.maximum(uncertainty, 1e-6).ravel()), (1.0,) * dimension,
                                     tuple(range(dimension)), method_id=METHOD_ID, accepted_count=len(accepted), limitations=limitations)
    else:
        basis = np.empty((0, n))
        estimate = retained_estimate(block, (0.0,) * dimension, (1.0,) * dimension, reason="measured raw response does not control all four task coordinates")
    predictions, heldout_rejections = [], {}
    projector = basis.T @ basis
    for sample in holdout:
        reason = _invalid(sample, domain, config)
        sx, sy = _data(sample, domain)
        if not reason and float(np.linalg.norm(sx - projector @ sx)) > 1e-6:
            reason = "heldout displacement outside measured input subspace"
        if reason:
            heldout_rejections[sample.observation_id] = reason
        else:
            predictions.append({"observation_id": sample.observation_id, "error_m_rad": ((j @ sx - sy) * domain.task.tolerance_m_rad).tolist()})
    sources: set[str] = set()
    independent = 0
    for sample in accepted:
        if not sources.intersection(sample.source_ids):
            independent += 1
        sources.update(sample.source_ids)
    metadata = {
        "domain": domain.to_dict(), "config": config.to_dict(), "evidence_digest": bundle.digest,
        "joint_ids": list(domain.joint_ids), "response_scaled": j.tolist(),
        "supported_basis_scaled": basis.tolist(), "supported_subspace_digest": digest_of(basis.tolist()),
        "data_rank": rank, "condition": condition, "response_rank": rank_report(j, config.rank_rcond)[0],
        "prior_rows_in_rank": 0, "residual_floor_m_rad": floor.tolist(),
        "covariance_subspace": covariance.tolist(), "training_source_ids": sorted(sources),
        "holdout_source_ids": sorted({i for s in holdout for i in s.source_ids}),
        "source_disjoint_units": independent, "structurally_qualified": False,
        "support_lower_rad": np.min([s.joints_rad for t in accepted for s in (t.first, t.second)], axis=0).tolist() if accepted else list(domain.reference_rad),
        "support_upper_rad": np.max([s.joints_rad for t in accepted for s in (t.first, t.second)], axis=0).tolist() if accepted else list(domain.reference_rad),
        "rejections": reasons, "holdout_rejections": heldout_rejections, "holdout_predictions": predictions,
        "stream_digest": training[0].stream.digest if training else "", "model_digest": bundle.topology.model.digest,
        "provenances": sorted({s.provenance for s in accepted}), "update_ids": [],
    }
    return EstimateBundle(digest_of({"evidence": bundle.digest, "domain": domain.digest, "config": config.to_dict(), "method": METHOD_ID}),
                          bundle.topology, (estimate,), tuple(s.observation_id for s in accepted),
                          tuple(sorted(set(bundle.rejected_ids) | set(reasons))), bundle.holdout_ids,
                          method_id=METHOD_ID, limitations=limitations, metadata=metadata)


def validate_raw_fit(fit: EstimateBundle) -> tuple[RawTaskDomain, ResponseConfig]:
    """Check internal numerical consistency, not acquisition trust or qualification."""
    np = numpy()
    if fit.method_id != METHOD_ID or len(fit.estimates) != 1:
        raise ValueError("proposal needs separately versioned raw-joint response")
    domain = RawTaskDomain.from_mapping(fit.metadata["domain"])
    config = ResponseConfig(**fit.metadata["config"])
    n = len(domain.joint_ids)
    basis = np.asarray(fit.metadata["supported_basis_scaled"], dtype=float)
    j = np.asarray(fit.metadata["response_scaled"], dtype=float)
    if basis.size == 0:
        raise CalibrationFailure(FAILURE_INSUFFICIENT_EVIDENCE, "all four outputs are not data-controllable")
    if basis.ndim != 2 or basis.shape[1] != n or not 4 <= len(basis) <= n or j.shape != (4, n):
        raise ValueError("raw response subspace dimensions differ")
    if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(j)):
        raise ValueError("raw response matrices must be finite")
    if not np.allclose(basis @ basis.T, np.eye(len(basis)), atol=1e-8, rtol=0):
        raise ValueError("raw response basis is not orthonormal")
    if digest_of(basis.tolist()) != fit.metadata["supported_subspace_digest"]:
        raise ValueError("raw response supported-subspace digest mismatch")
    c = np.asarray(fit.estimates[0].values).reshape(4, len(basis))
    if fit.estimates[0].support != SUPPORT_SUPPORTED or not np.allclose(c @ basis, j, atol=1e-8, rtol=0):
        raise ValueError("raw response estimate and matrix differ")
    rank, condition, _, _ = rank_report(c, config.rank_rcond)
    if rank != 4 or condition is None or condition > config.max_condition:
        raise CalibrationFailure(FAILURE_INSUFFICIENT_EVIDENCE, "raw response output rank/conditioning refused")
    p = np.asarray(fit.metadata["covariance_subspace"], dtype=float)
    if p.shape != (len(basis), len(basis)) or not np.all(np.isfinite(p)) or not np.allclose(p, p.T, atol=1e-8, rtol=0):
        raise ValueError("raw response covariance must be finite and symmetric")
    if np.min(np.linalg.eigvalsh(p)) < -1e-10:
        raise ValueError("raw response covariance must be positive semidefinite")
    floor = finite_vector(fit.metadata["residual_floor_m_rad"], 4)
    if any(v < minimum for v, minimum in zip(floor, config.uncertainty_floor_m_rad)):
        raise ValueError("raw response cannot erase measurement floors")
    lo = finite_vector(fit.metadata["support_lower_rad"], n)
    hi = finite_vector(fit.metadata["support_upper_rad"], n)
    if any(a > b for a, b in zip(lo, hi)):
        raise ValueError("raw response supported bounds are reversed")
    if fit.metadata["model_digest"] != fit.topology.model.digest or fit.metadata["joint_ids"] != list(domain.joint_ids):
        raise ValueError("raw response model/channel identity differs")
    return domain, config


def propose_raw(fit: EstimateBundle, current: RawTaskState, target_m_rad: Any, now: Stamp, *, run_id: str, model_digest: str) -> dict[str, Any]:
    np = numpy()
    domain, config = validate_raw_fit(fit)
    if not run_id or model_digest != fit.metadata["model_digest"] or not domain.accepts(current):
        raise CalibrationFailure(FAILURE_INSUFFICIENT_EVIDENCE, "proposal identity/domain/witness refused")
    age = now.delta_s(current.task.stamp)
    if age < 0 or age + current.task.stamp.uncertainty_s + now.uncertainty_s > domain.task.max_age_s:
        raise CalibrationFailure(FAILURE_INSUFFICIENT_EVIDENCE, "stale proposal source")
    if current.feedback_stamp.spread_s(current.task.stamp) > domain.task.max_age_s:
        raise CalibrationFailure(FAILURE_INSUFFICIENT_EVIDENCE, "feedback/frame timing refused")
    j = np.asarray(fit.metadata["response_scaled"])
    basis = np.asarray(fit.metadata["supported_basis_scaled"])
    if basis.size == 0 or rank_report(j, config.rank_rcond)[0] != 4:
        raise CalibrationFailure(FAILURE_INSUFFICIENT_EVIDENCE, "all four outputs are not data-controllable")
    target = finite_vector(target_m_rad, 4)
    error = difference(target, current.task.feature_m_rad) / domain.task.tolerance_m_rad
    reduced = j @ basis.T
    dq = basis.T @ reduced.T @ np.linalg.solve(reduced @ reduced.T + config.damping**2 * np.eye(4), error)
    scale = max(1.0, float(np.max(np.abs(dq))))
    raw_delta = dq / scale * domain.max_step_rad
    prediction = j @ (raw_delta / domain.max_step_rad) * domain.task.tolerance_m_rad
    # Uniform scaling preserves the measured subspace; clipping individual joints does not.
    scale = max(1.0, float(np.linalg.norm(prediction[:3]) / min(domain.task.max_step_m_rad[:3])), abs(float(prediction[3])) / domain.task.max_step_m_rad[3])
    raw_delta /= scale
    prediction /= scale
    raw_targets = np.asarray(current.joints_rad) + raw_delta
    lo = np.maximum(domain.lower_rad, fit.metadata["support_lower_rad"])
    hi = np.minimum(domain.upper_rad, fit.metadata["support_upper_rad"])
    uncertainty = np.maximum(fit.metadata["residual_floor_m_rad"], current.task.uncertainty_m_rad) * config.confidence_multiplier
    if np.any(raw_targets < lo) or np.any(raw_targets > hi) or np.any(uncertainty > domain.task.tolerance_m_rad):
        raise CalibrationFailure(FAILURE_INSUFFICIENT_EVIDENCE, "unsupported raw extrapolation or uncertainty")
    return {"schema": "leapcalib.raw_proposal.v1", "executed": False, "run_id": run_id,
            "raw_targets_rad": raw_targets.tolist(), "joint_ids": list(domain.joint_ids),
            "model_digest": model_digest, "candidate_digest": fit.digest, "domain_digest": domain.digest,
            "current_state_digest": current.digest, "source_frame_digest": digest_of(current.frame_ids),
            "supported_subspace_digest": fit.metadata["supported_subspace_digest"],
            "prediction_m_rad": prediction.tolist(), "uncertainty_m_rad": uncertainty.tolist(),
            "deadline_monotonic_s": min(current.task.stamp.seconds - current.task.stamp.uncertainty_s,
                                                    current.feedback_stamp.seconds - current.feedback_stamp.uncertainty_s) + domain.task.max_age_s,
            "trust_region_rad": list(domain.max_step_rad), "accepted": True}


def update_raw_response(fit: EstimateBundle, sample: RawTaskTransition, *, max_update_norm: float = 0.05) -> EstimateBundle:
    """Bounded RLS from actual corrections, with immutable excluded evaluation sources."""
    np = numpy()
    if isinstance(max_update_norm, bool) or not isinstance(max_update_norm, (int, float)) or not math.isfinite(max_update_norm) or not 0 < max_update_norm <= 0.05:
        raise ValueError("recursive update bound must be positive and at most 0.05")
    domain, config = validate_raw_fit(fit)
    if fit.method_id != METHOD_ID or sample.stream.digest != fit.metadata["stream_digest"] or sample.model.digest != fit.metadata["model_digest"]:
        raise ValueError("recursive update identity changed")
    if sample.observation_id in (*fit.holdout_ids, *fit.accepted_ids, *fit.rejected_ids, *fit.metadata["update_ids"]):
        raise ValueError("evaluation/training transition cannot be replayed as an update")
    if set(sample.source_ids).intersection((*fit.metadata["holdout_source_ids"], *fit.metadata["training_source_ids"])):
        raise ValueError("correlated or heldout sources cannot update the response")
    if reason := _invalid(sample, domain, config):
        raise CalibrationFailure(FAILURE_INSUFFICIENT_EVIDENCE, reason)
    basis = np.asarray(fit.metadata["supported_basis_scaled"])
    x, y = _data(sample, domain)
    if basis.size == 0 or np.linalg.norm(x - basis.T @ basis @ x) > 1e-6:
        raise ValueError("recursive update cannot expand measured input support")
    z = basis @ x
    c = np.asarray(fit.estimates[0].values).reshape(4, len(basis))
    innovation = y - c @ z
    if np.linalg.norm(innovation) > config.residual_gate:
        raise CalibrationFailure(FAILURE_INSUFFICIENT_EVIDENCE, "unexpected target movement or normalized innovation")
    p = np.asarray(fit.metadata["covariance_subspace"])
    gain = p @ z / (1 + z @ p @ z)
    delta = np.outer(innovation, gain)
    delta *= min(1.0, max_update_norm / max(1e-12, float(np.linalg.norm(delta))))
    c += delta
    rank, condition, _, _ = rank_report(c, config.rank_rcond)
    if rank != 4 or condition is None or condition > config.max_condition:
        raise ValueError("recursive update loses controllable output rank/conditioning")
    j = c @ basis
    updated_covariance = p - np.outer(gain, z @ p)
    updated_covariance = (updated_covariance + updated_covariance.T) / 2
    metadata = {**fit.metadata, "response_scaled": j.tolist(),
                "covariance_subspace": updated_covariance.tolist(),
                "residual_floor_m_rad": np.maximum(fit.metadata["residual_floor_m_rad"], _floor([sample], j, domain, config)).tolist(),
                "update_ids": [*fit.metadata["update_ids"], sample.observation_id],
                "training_source_ids": sorted(set(fit.metadata["training_source_ids"]) | set(sample.source_ids))}
    estimate = replace(fit.estimates[0], values=tuple(c.ravel()))
    return replace(fit, bundle_id=digest_of({"parent": fit.digest, "update": sample.to_dict()}), estimates=(estimate,), metadata=metadata)


class RawResponseMethod:
    @property
    def method_id(self) -> str:
        return METHOD_ID

    def solve(self, evidence: EvidenceBundle, options: Mapping[str, Any]) -> EstimateBundle:
        if set(options) - {"domain", "config"}:
            raise ValueError("unknown raw-response options")
        return estimate_raw_response(evidence, RawTaskDomain.from_mapping(options["domain"]), config=ResponseConfig(**options.get("config", {})))
