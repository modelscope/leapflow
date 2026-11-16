# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bounded robust rigid registration of identified metric correspondences.

This is not ICP: source/target identity must be supplied by a measured tracker.
Planar *identified* landmarks can constrain a rigid transform; an untextured
plane with nearest-neighbor association cannot. The latter is not accepted as
identified evidence by this API. Collinear/coincident landmarks are refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from leapcalib.contracts.failures import CalibrationFailure, FAILURE_INSUFFICIENT_EVIDENCE
from leapcalib.contracts.motion import PointCorrespondence
from leapcalib.geometry.se3 import Transform
from leapcalib.estimation.numeric import SolveConfig, numpy, transform


@dataclass(frozen=True)
class Registration:
    motion: Transform
    accepted_features: tuple[str, ...]
    rejected_features: tuple[str, ...]
    rms_m: float
    max_error_m: float
    spatial_rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "motion": self.motion.to_dict(),
            "accepted_features": list(self.accepted_features),
            "rejected_features": list(self.rejected_features),
            "rms_m": self.rms_m,
            "max_error_m": self.max_error_m,
            "spatial_rank": self.spatial_rank,
            "association": "externally_identified_metric_correspondences",
        }


def _kabsch(source: Any, target: Any, rcond: float) -> tuple[Any, Any, int]:
    np = numpy()
    centred = source - source.mean(axis=0)
    spread = np.linalg.svd(centred, compute_uv=False)
    target_spread = np.linalg.svd(target - target.mean(axis=0), compute_uv=False)
    rank = min(
        int(np.sum(spread > max(float(spread[0]) * rcond, 1e-8))),
        int(np.sum(target_spread > max(float(target_spread[0]) * rcond, 1e-8))),
    )
    if rank < 2:
        raise CalibrationFailure(
            FAILURE_INSUFFICIENT_EVIDENCE,
            "coincident/collinear features cannot establish rigid motion",
        )
    u, _, vh = np.linalg.svd(centred.T @ (target - target.mean(axis=0)))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vh.T @ u.T)
    rotation = vh.T @ correction @ u.T
    translation = target.mean(axis=0) - rotation @ source.mean(axis=0)
    return rotation, translation, rank


def register_points(
    points: Sequence[PointCorrespondence],
    *,
    reverse: bool = False,
    config: SolveConfig = SolveConfig(),
) -> Registration:
    """Map first points into second points (reverse=True maps second into first)."""
    np = numpy()
    if type(reverse) is not bool:
        raise ValueError("registration direction must be an explicit boolean")
    if not config.min_points <= len(points) <= config.max_points:
        raise CalibrationFailure(
            FAILURE_INSUFFICIENT_EVIDENCE, "correspondences are outside the configured point budget"
        )
    ids = [p.feature_id for p in points]
    if len(ids) != len(set(ids)):
        raise ValueError("feature identities must be unique")
    source = np.asarray([p.second_m if reverse else p.first_m for p in points])
    target = np.asarray([p.first_m if reverse else p.second_m for p in points])
    rng = np.random.default_rng(config.seed)
    trials = [np.arange(len(points))]
    trials.extend(rng.choice(len(points), 3, replace=False) for _ in range(config.trials))
    best = None
    best_score = (-1, float("-inf"))
    for indices in trials:
        try:
            rotation, translation, _ = _kabsch(source[indices], target[indices], config.rank_rcond)
        except CalibrationFailure:
            continue
        errors = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        inliers = errors <= config.point_gate_m
        score = (int(inliers.sum()), -float(np.median(errors)))
        if score > best_score:
            best, best_score = inliers, score
    minimum = max(config.min_points, int(np.ceil(len(points) * config.min_inlier_ratio)))
    if best is None or int(best.sum()) < minimum:
        raise CalibrationFailure(
            FAILURE_INSUFFICIENT_EVIDENCE,
            "metric correspondences have no supported rigid consensus",
        )
    for _ in range(config.refinement_steps):
        rotation, translation, spatial_rank = _kabsch(source[best], target[best], config.rank_rcond)
        errors = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        updated = errors <= config.point_gate_m
        if np.array_equal(updated, best):
            break
        best = updated
        if int(best.sum()) < minimum:
            raise CalibrationFailure(
                FAILURE_INSUFFICIENT_EVIDENCE, "registration consensus collapsed"
            )
    # Refit the final set so bounded-loop exit cannot report a stale model.
    rotation, translation, spatial_rank = _kabsch(source[best], target[best], config.rank_rcond)
    errors = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    if float(errors[best].max()) > config.point_gate_m:
        raise CalibrationFailure(FAILURE_INSUFFICIENT_EVIDENCE, "registration did not stabilize")
    return Registration(
        transform(rotation, translation),
        tuple(p.feature_id for p, ok in zip(points, best) if ok),
        tuple(p.feature_id for p, ok in zip(points, best) if not ok),
        float(np.sqrt(np.mean(errors[best] ** 2))),
        float(errors[best].max()),
        spatial_rank,
    )


__all__ = ["Registration", "register_points"]
