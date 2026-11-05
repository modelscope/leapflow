# Copyright (c) Alibaba, Inc. and its affiliates.
"""Target-free motion-conjugacy candidates, conditional on frozen nominal geometry.

AX=XB is shared by fixed-camera rigid-link/screw motion and wrist static-scene
hand-eye. Source association, scene rigidity, and nominal robot motion are
explicit nuisance assumptions, not inferred accuracy or activation authority.
Unsupported Cartesian components are retained; coupled observable combinations
are reported through the data-only nullspace instead of fabricated axis precision.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Sequence

from leapcalib.contracts.estimates import EstimateBundle, ParameterEstimate, retained_estimate
from leapcalib.contracts.failures import CalibrationFailure, FAILURE_INSUFFICIENT_EVIDENCE
from leapcalib.contracts.identity import digest_of
from leapcalib.contracts.motion import MODE_WRIST, MotionObservation, finite_vector
from leapcalib.contracts.parameters import (
    SUPPORT_PARTIAL,
    SUPPORT_SUPPORTED,
    camera_extrinsic_block,
)
from leapcalib.geometry.se3 import Transform
from leapcalib.estimation.numeric import (
    SolveConfig,
    array_transform,
    correction_values,
    jacobian,
    numpy,
    perturbation,
    rank_report,
    rotation_vector,
    transform,
)
from leapcalib.estimation.registration import register_points
from leapcalib.validation.bundle import EvidenceBundle

METHOD_ID = "metric_motion.v1"


def _residual(
    candidate: Transform,
    factors: Sequence[tuple[MotionObservation, Transform]],
    config: SolveConfig,
) -> Any:
    np = numpy()
    rx, tx = array_transform(candidate)
    rows = []
    for observation, camera in factors:
        ra, ta = array_transform(observation.robot_motion)
        rb, tb = array_transform(camera)
        rows.extend((ra @ tx + ta - rx @ tb - tx) / config.translation_sigma_m)
        rows.extend(rotation_vector((ra @ rx).T @ (rx @ rb)) / config.rotation_sigma_rad)
    return np.asarray(rows)


def _closed_fit(
    factors: Sequence[tuple[MotionObservation, Transform]], prior: Transform, config: SolveConfig
) -> Transform:
    np = numpy()
    aa = np.asarray([rotation_vector(array_transform(o.robot_motion)[0]) for o, _ in factors])
    bb = np.asarray([rotation_vector(array_transform(b)[0]) for _, b in factors])
    u, singular, vh = np.linalg.svd(bb.T @ aa)
    if singular[1] <= max(float(singular[0]) * config.rank_rcond, 1e-10):
        rotation = np.asarray(prior.rotation)
    else:
        fix = np.eye(3)
        fix[2, 2] = np.linalg.det(vh.T @ u.T)
        rotation = vh.T @ fix @ u.T
    rows, targets = [], []
    for observation, camera in factors:
        ra, ta = array_transform(observation.robot_motion)
        _, tb = array_transform(camera)
        rows.append(ra - np.eye(3))
        targets.append(rotation @ tb - ta)
    rows, targets = np.vstack(rows), np.concatenate(targets)
    # Preserve the prior's null component rather than selecting the minimum-norm origin.
    prior_t = np.asarray(prior.translation_m)
    translation = (
        prior_t + np.linalg.lstsq(rows, targets - rows @ prior_t, rcond=config.rank_rcond)[0]
    )
    return transform(rotation, translation)


def _refine(
    factors: Sequence[tuple[MotionObservation, Transform]],
    prior: Transform,
    values: Any,
    components: Sequence[int],
    config: SolveConfig,
) -> Any:
    np = numpy()
    values = values.copy()
    unsupported = list(set(range(6)) - set(components))
    values[unsupported] = 0
    if not components:
        return values
    indices = list(components)

    def function(v: Any) -> Any:
        return _residual(prior.compose(perturbation(v)), factors, config)

    for _ in range(config.refinement_steps):
        residual = function(values)
        matrix = jacobian(function, values)[:, indices]
        step = np.linalg.lstsq(matrix, -residual, rcond=config.rank_rcond)[0]
        improved = False
        for multiplier in (1.0, 0.5, 0.25, 0.125):
            trial = values.copy()
            trial[indices] += step * multiplier
            if float(np.linalg.norm(function(trial))) < float(np.linalg.norm(residual)):
                values, improved = trial, True
                break
        if not improved or float(np.linalg.norm(step)) < 1e-9:
            break
    return values


def _consensus(
    factors: Sequence[tuple[MotionObservation, Transform]], prior: Transform, config: SolveConfig
) -> tuple[Any, Any]:
    np = numpy()
    rng = np.random.default_rng(config.seed)
    samples = [np.arange(len(factors))]
    if len(factors) >= 3:
        samples.extend(rng.choice(len(factors), 3, replace=False) for _ in range(config.trials))
    best, best_score = np.zeros(6), (-1, float("-inf"))
    for indices in samples:
        candidate = _closed_fit([factors[int(i)] for i in indices], prior, config)
        values = correction_values(prior, candidate)
        residual = _residual(candidate, factors, config).reshape(-1, 6)
        errors = np.linalg.norm(residual, axis=1)
        score = (int(np.sum(errors <= config.residual_gate)), -float(np.median(errors)))
        if score > best_score:
            best, best_score = values, score
    errors = np.linalg.norm(
        _residual(prior.compose(perturbation(best)), factors, config).reshape(-1, 6), axis=1
    )
    return best, errors <= config.residual_gate


def _prepare(
    observations: Sequence[MotionObservation], config: SolveConfig, rejections: dict[str, str]
) -> list[tuple[MotionObservation, Transform]]:
    result = []
    for observation in observations:
        reason = ""
        if not observation.settled or observation.max_speed_rad_s > config.max_settled_speed_rad_s:
            reason = "robot stillness was not measured within the configured bound"
        elif (
            max(observation.first_stamp.uncertainty_s, observation.stamp.uncertainty_s)
            > config.max_stamp_uncertainty_s
        ):
            reason = "receipt/exposure uncertainty exceeds the timing budget"
        angle = observation.robot_motion.angle_to_rad(Transform())
        if angle < config.min_rotation_rad or angle >= math.pi - 0.05:
            reason = "rotation is noninformative or near the ambiguous pi branch"
        if reason:
            rejections[observation.observation_id] = reason
            continue
        try:
            registered = register_points(
                observation.points, reverse=observation.mode == MODE_WRIST, config=config
            )
        except CalibrationFailure as failure:
            rejections[observation.observation_id] = failure.message
            continue
        result.append((observation, registered.motion))
    return result


def estimate_motion(
    bundle: EvidenceBundle,
    prior: Transform,
    *,
    prior_uncertainty: Sequence[float] = (0.05, 0.05, 0.05, 0.2, 0.2, 0.2),
    config: SolveConfig = SolveConfig(),
) -> EstimateBundle:
    """Produce a digest-linked conditional mount correction, never an activation.

    The prior is T_base_camera for fixed motion, T_flange_camera for wrist motion.
    Reported block values are the right-composed translation/rotation-vector
    correction, not Euler angles or an unlabeled absolute transform.
    """
    prior_uncertainty = finite_vector(prior_uncertainty, 6)
    if any(v < 0 for v in prior_uncertainty) or not prior.parent or not prior.child:
        raise ValueError("a mount prior needs named frames and nonnegative uncertainties")
    observations = {o.observation_id: o for o in bundle.observations}
    training = [
        observations[i]
        for i in bundle.accepted_ids
        if isinstance(observations[i], MotionObservation)
    ]
    holdouts = [
        observations[i]
        for i in bundle.holdout_ids
        if isinstance(observations[i], MotionObservation)
    ]
    if len(training) + len(holdouts) > config.max_pairs:
        raise CalibrationFailure(FAILURE_INSUFFICIENT_EVIDENCE, "motion pair budget exceeded")
    identities = {(o.mode, o.stream.digest) for o in training + holdouts}
    if len(identities) > 1:
        raise ValueError("one candidate cannot combine camera roles or stream profiles")
    if any(o.model != bundle.topology.model for o in training + holdouts):
        raise ValueError("nominal model identity differs from motion evidence")
    if bundle.topology.mounts and prior.parent not in {
        m.parent_frame for m in bundle.topology.mounts
    }:
        raise ValueError("mount prior parent differs from topology")
    return _solve(bundle, prior, prior_uncertainty, training, holdouts, config)


def _solve(
    bundle: EvidenceBundle,
    prior: Transform,
    prior_uncertainty: tuple[float, ...],
    training: Sequence[MotionObservation],
    holdouts: Sequence[MotionObservation],
    config: SolveConfig,
) -> EstimateBundle:
    np = numpy()
    reasons: dict[str, str] = {}
    factors = _prepare(training, config, reasons)
    block = replace(
        camera_extrinsic_block("mount", parent_frame=prior.parent, child_frame=prior.child),
        notes="right correction: dx,dy,dz metres and SO(3) rotation vector radians",
    )
    values = np.zeros(6)
    rank, condition, singular, null = 0, None, np.zeros(6), np.eye(6)
    supported: list[int] = []
    accepted: list[tuple[MotionObservation, Transform]] = []
    stable = False
    if len(factors) >= config.min_pairs:
        seed, mask = _consensus(factors, prior, config)
        for _ in range(config.refinement_steps):
            accepted = [f for f, ok in zip(factors, mask) if ok]
            if len(accepted) < max(
                config.min_pairs, math.ceil(len(factors) * config.min_inlier_ratio)
            ):
                break

            def function(v: Any) -> Any:
                return _residual(prior.compose(perturbation(v)), accepted, config)

            scales = np.array([config.translation_sigma_m] * 3 + [config.rotation_sigma_rad] * 3)
            matrix = jacobian(function, np.zeros(6)) * scales
            rank, condition, singular, null = rank_report(matrix, config.rank_rcond)
            supported = [
                i for i in range(6) if not len(null) or float(np.linalg.norm(null[:, i])) < 1e-5
            ]
            values = _refine(accepted, prior, seed, supported, config)
            errors = np.linalg.norm(
                _residual(prior.compose(perturbation(values)), factors, config).reshape(-1, 6),
                axis=1,
            )
            updated = errors <= config.residual_gate
            if np.array_equal(mask, updated):
                stable = True
                break
            seed, mask = values, updated
        if not stable or condition is None or condition > config.max_condition:
            supported = []
            values[:] = 0
    candidate = replace(prior.compose(perturbation(values)), parent=prior.parent, child=prior.child)
    accepted_ids = tuple(o.observation_id for o, _ in accepted) if stable else ()
    for observation, _ in factors:
        if observation.observation_id not in accepted_ids:
            reasons[observation.observation_id] = "no stable supported motion consensus"
    limitations = (
        "candidate only; no hardware accuracy qualification or activation",
        "nominal/raw robot geometry, factory intrinsics and depth scale are frozen priors",
        "correspondences require independently tracked rigid-link/static-scene identity",
        "shared source frames are correlated; conditional uncertainty is not independent metrology",
        "joint-zero, time-offset, native-mesh ICP and depth-scale estimation were not performed",
    )
    if supported and accepted_ids:

        def function(v: Any) -> Any:
            return _residual(prior.compose(perturbation(v)), accepted, config)

        matrix = jacobian(function, values)[:, supported]
        residual = function(values)
        covariance = np.linalg.pinv(matrix.T @ matrix, rcond=config.rank_rcond) * max(
            1.0, float(residual @ residual) / max(1, len(residual) - len(supported))
        )
        uncertainty = list(prior_uncertainty)
        floors = [config.translation_sigma_m] * 3 + [config.rotation_sigma_rad] * 3
        for j, i in enumerate(supported):
            uncertainty[i] = max(floors[i], float(np.sqrt(max(0, covariance[j, j]))))
        estimate = ParameterEstimate(
            block,
            tuple(values),
            (0.0,) * 6,
            SUPPORT_SUPPORTED if len(supported) == 6 else SUPPORT_PARTIAL,
            tuple(uncertainty),
            prior_uncertainty,
            tuple(supported),
            ("frozen_nominal_raw_kinematics",),
            METHOD_ID,
            len(accepted_ids),
            limitations,
        )
    else:
        estimate = retained_estimate(
            block,
            (0.0,) * 6,
            prior_uncertainty,
            reason="insufficient excitation, coupled gauge or unstable consensus",
        )
    heldout_rejections: dict[str, str] = {}
    heldout = _prepare(holdouts, config, heldout_rejections)
    predictions = []
    for observation, camera in heldout:
        predicted = candidate.inverse().compose(observation.robot_motion).compose(candidate)
        predictions.append(
            {
                "observation_id": observation.observation_id,
                "translation_error_m": predicted.distance_m(camera),
                "rotation_error_rad": predicted.angle_to_rad(camera),
            }
        )
    metadata = {
        "evidence_digest": bundle.digest,
        "config": config.to_dict(),
        "prior_transform": prior.to_dict(),
        "candidate_transform": candidate.to_dict(),
        "data_rank": rank,
        "condition": condition,
        "singular_values": singular.tolist(),
        "nullspace_scaled": null.tolist(),
        "prior_rows_in_rank": 0,
        "rejections": reasons,
        "holdout_rejections": heldout_rejections,
        "holdout_predictions": predictions,
        "consensus_stable": stable,
        "provenances": sorted({o.provenance for o in training + holdouts}),
    }
    identity = digest_of(
        {
            "evidence": bundle.digest,
            "method": METHOD_ID,
            "prior": prior.to_dict(),
            "config": config.to_dict(),
            "prior_uncertainty": list(prior_uncertainty),
        }
    )
    return EstimateBundle(
        identity,
        bundle.topology,
        (estimate,),
        accepted_ids,
        tuple(sorted(set(bundle.rejected_ids) | set(reasons))),
        bundle.holdout_ids,
        METHOD_ID,
        limitations=limitations,
        metadata=metadata,
    )


class MotionMethod:
    method_id = METHOD_ID

    def solve(self, evidence: EvidenceBundle, options: Mapping[str, Any]) -> EstimateBundle:
        return estimate_motion(
            evidence,
            Transform.from_mapping(options["prior"]),
            prior_uncertainty=options.get("prior_uncertainty", (0.05,) * 3 + (0.2,) * 3),
            config=SolveConfig(**options.get("config", {})),
        )


__all__ = ["METHOD_ID", "MotionMethod", "estimate_motion"]
