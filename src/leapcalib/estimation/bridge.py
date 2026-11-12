# Copyright (c) Alibaba, Inc. and its affiliates.
"""Fixed-only stationary-board bridge in SI, with separately scoped evaluation.

An exact wrist dependency is a conditional prior, not a jointly identified mount.
This method never establishes independent robot/base accuracy or activation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from leapcalib.contracts.identity import digest_of
from leapcalib.contracts.motion import finite_vector
from leapcalib.geometry.se3 import Transform

METHOD = "stationary_board_bridge.v1"
SCOPE = "conditional_fixed_camera_cross_view"
BUDGET_TERMS = frozenset({"robot_fk", "wrist_mount", "board_scale_flatness", "intrinsics", "pnp_depth", "timing", "repeatability"})


@dataclass(frozen=True)
class BridgeBudget:
    """Worst-case correlated component bounds; systematic terms never average away.

    Each term is (translation metres, rotation radians). The translation terms
    include propagated rotational lever-arm effects, checked by the owner adapter.
    Sum rather than RSS is deliberate: no independence witness is assumed.
    """
    terms: Mapping[str, Sequence[float]]
    max_error_m_rad: tuple[float, float]
    confidence_multiplier: float
    min_training: int
    min_holdout: int
    separation_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.terms, Mapping) or set(self.terms) != BUDGET_TERMS:
            raise ValueError("complete explicit bridge uncertainty budget required")
        normalized = {}
        for key, value in self.terms.items():
            values = finite_vector(value, 2)
            if any(v <= 0 for v in values):
                raise ValueError("all systematic/correlated floors must be positive")
            normalized[key] = values
        from types import MappingProxyType
        object.__setattr__(self, "terms", MappingProxyType(normalized))
        object.__setattr__(self, "max_error_m_rad", finite_vector(self.max_error_m_rad, 2))
        if not all(0 < v <= cap for v, cap in zip(self.max_error_m_rad, (0.003, math.radians(2)))):
            raise ValueError("bridge policy cannot silently widen the existing study ceilings")
        if type(self.confidence_multiplier) not in (int, float) or not math.isfinite(self.confidence_multiplier) or not 1 <= self.confidence_multiplier <= 10:
            raise ValueError("conservative confidence multiplier required")
        if any(type(v) is not int or not 3 <= v <= 64 for v in (self.min_training, self.min_holdout)):
            raise ValueError("at least three bounded training and withheld stationary pairs required")
        if type(self.separation_s) not in (int, float) or not math.isfinite(self.separation_s) or not 0.3 <= self.separation_s <= 120:
            raise ValueError("disjoint stationary window separation required")

    @property
    def uncertainty(self) -> tuple[float, float]:
        return tuple(sum(v[i] for v in self.terms.values()) for i in range(2))

    def to_dict(self) -> dict[str, Any]:
        return {**vars(self), "terms": {k: list(v) for k, v in self.terms.items()},
                "max_error_m_rad": list(self.max_error_m_rad)}

    def reachability(self) -> dict[str, Any]:
        floors = [self.confidence_multiplier * u for u in self.uncertainty]
        margins = [cap - floor for cap, floor in zip(self.max_error_m_rad, floors)]
        return {"zero_residual_floor_m_rad": floors, "available_residual_m_rad": margins,
                "reachable": all(v >= 0 for v in margins), "terms": self.to_dict()["terms"],
                "correlations": "worst_case_sum; no systematic floor reduced by sample count"}


@dataclass(frozen=True)
class BridgePair:
    evidence_digest: str
    source_ids: tuple[str, ...]
    interval_s: tuple[float, float]
    base_T_flange: Transform
    wrist_T_board: Transform
    fixed_T_board: Transform

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_digest, str) or len(self.evidence_digest) != 64 or any(c not in "0123456789abcdef" for c in self.evidence_digest):
            raise ValueError("indexed evidence digest required")
        if not isinstance(self.source_ids, (list, tuple)) or not 6 <= len(self.source_ids) <= 64 or any(not isinstance(v, str) or not 0 < len(v) <= 256 for v in self.source_ids) or len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("bounded distinct raw frame/feedback source identities required")
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        object.__setattr__(self, "interval_s", finite_vector(self.interval_s, 2))
        if not 0 <= self.interval_s[0] < self.interval_s[1]:
            raise ValueError("nonempty stationary interval required")
        for name, parent, child in (("base_T_flange", "base", "flange"), ("wrist_T_board", "wrist", "board"), ("fixed_T_board", "fixed", "board")):
            transform = getattr(self, name)
            if not isinstance(transform, Transform) or (transform.parent, transform.child) != (parent, child):
                raise ValueError("explicit parent_T_child SI transform required")

    def to_dict(self) -> dict[str, Any]:
        return {**vars(self), **{k: getattr(self, k).to_dict() for k in ("base_T_flange", "wrist_T_board", "fixed_T_board")},
                "source_ids": list(self.source_ids), "interval_s": list(self.interval_s)}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> BridgePair:
        if not isinstance(data, Mapping) or set(data) != set(cls.__dataclass_fields__):
            raise ValueError("exact bridge pair fields required")
        return cls(**{**data, **{k: Transform.from_mapping(data[k]) for k in ("base_T_flange", "wrist_T_board", "fixed_T_board")}})


def bridge_transform(pair: BridgePair, flange_T_wrist: Transform) -> Transform:
    if (flange_T_wrist.parent, flange_T_wrist.child) != ("flange", "wrist"):
        raise ValueError("fresh flange_T_wrist dependency required; joint fitting is not this mode")
    return pair.base_T_flange.compose(flange_T_wrist).compose(pair.wrist_T_board).compose(pair.fixed_T_board.inverse())


def _log(transform: Transform) -> Any:
    import numpy as np

    rotation = np.asarray(transform.rotation)
    theta = math.acos(max(-1.0, min(1.0, (float(np.trace(rotation)) - 1) / 2)))
    if theta >= math.pi - 1e-6:
        raise ValueError("bridge rotation branch is ambiguous; cannot average antipodal poses")
    omega = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]) * (0.5 if theta < 1e-7 else theta / (2 * math.sin(theta)))
    cross = _skew(omega)
    inverse_j = np.eye(3) - cross / 2 + (1 / 12 if theta < 1e-5 else (1 - theta / (2 * math.tan(theta / 2))) / theta ** 2) * cross @ cross
    return np.concatenate((inverse_j @ np.asarray(transform.translation_m), omega))


def _skew(v: Any) -> Any:
    import numpy as np
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def _exp(v: Any) -> Transform:
    import numpy as np
    theta, cross = float(np.linalg.norm(v[3:])), _skew(v[3:])
    a = 1 - theta ** 2 / 6 if theta < 1e-5 else math.sin(theta) / theta
    b = 0.5 - theta ** 2 / 24 if theta < 1e-5 else (1 - math.cos(theta)) / theta ** 2
    c = 1 / 6 - theta ** 2 / 120 if theta < 1e-5 else (theta - math.sin(theta)) / theta ** 3
    rotation = np.eye(3) + a * cross + b * cross @ cross
    translation = (np.eye(3) + b * cross + c * cross @ cross) @ v[:3]
    return Transform(tuple(tuple(float(x) for x in row) for row in rotation), tuple(float(x) for x in translation))


def fit_se3(transforms: Sequence[Transform]) -> Transform:
    """Right-invariant SE(3) mean; no Euler arithmetic, bounded convergence."""
    import numpy as np
    if not transforms or len(transforms) > 128:
        raise ValueError("bounded nonempty SE(3) support required")
    current = Transform(transforms[0].rotation, transforms[0].translation_m)
    unlabelled = [Transform(t.rotation, t.translation_m) for t in transforms]
    for _ in range(64):
        step = np.mean([_log(current.inverse().compose(t)) for t in unlabelled], axis=0)
        if float(np.linalg.norm(step)) < 1e-10:
            return Transform(current.rotation, current.translation_m, "base", "fixed")
        current = current.compose(_exp(step))
    raise ValueError("bridge SE(3) fit did not converge")


def stationary_board_check(pairs: Sequence[BridgePair], budget: BridgeBudget) -> dict[str, Any]:
    """Check the fixed view's scene-stationary assumption without refitting holdout.

    Residuals within declared view-noise bounds are conditional consistency, not
    a physical target witness. A detected change requires a new stationary epoch;
    it cannot be repaired by letting both views cancel a moving board in the chain.
    """
    anchor = pairs[0].fixed_T_board
    residuals = [[anchor.distance_m(p.fixed_T_board), anchor.angle_to_rad(p.fixed_T_board)] for p in pairs]
    # Common systematic board-scale/mount/FK terms cannot excuse movement of the
    # fixed view. Allow only the separately declared view noise/repeatability.
    limits = [budget.max_error_m_rad[i] + 2 * budget.confidence_multiplier *
              sum(budget.terms[key][i] for key in ("intrinsics", "pnp_depth", "repeatability")) for i in range(2)]
    if any(value > cap for row in residuals for value, cap in zip(row, limits)):
        raise ValueError("stationary board/fixed view changed; inspect and declare a new target epoch")
    return {"anchor_evidence_digest": pairs[0].evidence_digest, "fixed_view_residual_m_rad": residuals,
            "limits_m_rad": limits, "observed_consistency": True, "physical_stationarity_certified": False}


def solve_bridge(training: Sequence[BridgePair], holdout: Sequence[BridgePair], *,
                 flange_T_wrist: Transform, dependency_digest: str, dependency_sources: Sequence[str],
                 budget: BridgeBudget) -> dict[str, Any]:
    if not isinstance(budget, BridgeBudget) or not isinstance(dependency_digest, str) or len(dependency_digest) != 64 or any(c not in "0123456789abcdef" for c in dependency_digest):
        raise ValueError("versioned wrist dependency and frozen budget required")
    if len(training) + len(holdout) > 128 or any(not isinstance(p, BridgePair) for p in (*training, *holdout)):
        raise ValueError("bounded typed bridge pairs required")
    # No physical interval or native source may be recycled, even with new IDs.
    ordered = sorted((*training, *holdout), key=lambda p: p.interval_s)
    if not isinstance(dependency_sources, (list, tuple)) or len(dependency_sources) > 8192 or any(not isinstance(v, str) or not 0 < len(v) <= 256 for v in dependency_sources):
        raise ValueError("bounded wrist refinement source identities required")
    seen, evidence_ids = set(dependency_sources), set()
    for i, pair in enumerate(ordered):
        if pair.evidence_digest in evidence_ids:
            raise ValueError("bridge indexed evidence overlap")
        evidence_ids.add(pair.evidence_digest)
        if seen.intersection(pair.source_ids):
            raise ValueError("bridge source overlap with fitting, evaluation or wrist refinement")
        seen.update(pair.source_ids)
        if i and pair.interval_s[0] - ordered[i - 1].interval_s[1] < budget.separation_s:
            raise ValueError("bridge stationary physical windows overlap or lack frozen separation")
    result = {"schema": "leapcalib.bridge_result.v1", "method_id": METHOD, "scope": SCOPE,
              "dependency_digest": dependency_digest, "budget": budget.to_dict(),
              "reachability": budget.reachability(), "training": [p.evidence_digest for p in training],
              "holdout": [p.evidence_digest for p in holdout], "physical_activation": "shadow_only",
              "independent_reference_accuracy": None, "absolute_base_accuracy_established": False,
              "joint_wrist_fit": False, "exclusions": []}
    if len(training) < budget.min_training:
        return {**result, "verdict": "insufficient_evidence", "candidate": None}
    stationarity = stationary_board_check([*training, *holdout], budget)
    candidate = fit_se3([bridge_transform(p, flange_T_wrist) for p in training])
    def errors(pairs: Sequence[BridgePair]) -> list[list[float]]:
        return [[candidate.distance_m(t), candidate.angle_to_rad(t)] for t in (bridge_transform(p, flange_T_wrist) for p in pairs)]
    train_errors, withheld = errors(training), errors(holdout)
    # Estimate repeatability only on training. A systematic floor never shrinks
    # with support; an observed spread can only increase the frozen prior.
    terms = {key: list(value) for key, value in budget.terms.items()}
    terms["repeatability"] = [max(terms["repeatability"][i], max(row[i] for row in train_errors)) for i in range(2)]
    uncertainty = tuple(sum(value[i] for value in terms.values()) for i in range(2))
    margins = [[cap - e - budget.confidence_multiplier * u for cap, e, u in zip(budget.max_error_m_rad, row, uncertainty)] for row in withheld]
    verdict = "insufficient_evidence" if len(holdout) < budget.min_holdout else "numerical_pass" if all(min(row) >= 0 for row in margins) else "rejected"
    return {**result, "stationary_board": stationarity,
            "observability": {"solve_mode": "fixed_only_with_validated_wrist", "conditional_pose_dof": 6,
                              "joint_wrist_rank_established": False, "absolute_base_rank_established": False},
            "candidate": candidate.to_dict(), "candidate_digest": digest_of({"transform": candidate.to_dict(), "dependency": dependency_digest, "training": result["training"], "budget": budget.to_dict()}),
            "training_residual_m_rad": train_errors, "withheld_cross_view_m_rad": withheld,
            "uncertainty_m_rad": list(uncertainty), "effective_uncertainty_terms": terms,
            "uncertainty_basis": "declared_correlated_bounds_plus_training_repeatability; conditional_not_metrology",
            "quality_margins_m_rad": margins,
            "verdict": verdict, "advisories": [] if verdict == "numerical_pass" else ["candidate retained; scoped acceptance not established"]}
