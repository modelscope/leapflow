# Copyright (c) Alibaba, Inc. and its affiliates.
"""Indexed paired raw-response assessment, separate from witnesses and activation.

This cold-path module keeps paired candidate/baseline orchestration out of the
factor reconstruction boundary. Unverified indexed states retain false witness
flags; neither recomputation nor a manually constructed assessment grants motion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from leapcalib.contracts.estimates import EstimateBundle
from leapcalib.contracts.identity import digest_of
from leapcalib.contracts.raw_task import (
    MetricTaskFeature,
    RawTaskDomain,
    RawTaskTransition,
    identities,
)
from leapcalib.contracts.tracking import TrackingConfig
from leapcalib.validation.bundle import EvidenceBundle
from leapcalib.validation.evaluation import (
    CandidateEvaluation,
    EvaluationConfig,
    evaluate_raw_candidate,
)
from leapflow.hardware.calibration.factors import (
    IndexedRawTaskFactors,
    _CaptureSnapshot,
    _indexed_evaluation,
    indexed_raw_task_factors,
)
from leapflow.hardware.calibration.generations import _json
from leapflow.hardware.calibration.provenance import (
    IndexedCaptureReader,
    independent_indexed_holdouts,
)


@dataclass(frozen=True)
class IndexedRawAssessment:
    """Immutable checked paired result; not an externally admissible certificate."""

    candidate_json: str
    baseline_json: str
    factors: IndexedRawTaskFactors
    baseline_factors: IndexedRawTaskFactors
    evaluation: CandidateEvaluation
    indexed_holdout_ids: tuple[str, ...]
    indexed_exclusions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        fits = []
        for payload in (self.candidate_json, self.baseline_json):
            if not isinstance(payload, str) or len(payload.encode()) > 2 * 1024 * 1024:
                raise ValueError("bounded immutable indexed raw candidates required")
            fits.append(EstimateBundle.from_mapping(json.loads(payload)))
        if any(
            not isinstance(factors, IndexedRawTaskFactors)
            for factors in (self.factors, self.baseline_factors)
        ) or not isinstance(self.evaluation, CandidateEvaluation):
            raise ValueError("typed indexed raw factors and paired evaluation required")
        evaluation = self.evaluation
        for fit, factors, candidate_digest, evidence_digest in zip(
            fits,
            (self.factors, self.baseline_factors),
            (evaluation.candidate_digest, evaluation.baseline_digest),
            (evaluation.evidence_digest, evaluation.baseline_evidence_digest),
        ):
            if (
                fit.method_id != "raw_joint_response.v1"
                or fit.digest != candidate_digest
                or fit.metadata.get("evidence_digest") != evidence_digest
                or factors.manifest.evidence_digest != evidence_digest
                or factors.manifest.topology_digest != fit.topology.digest
                or not set((*fit.accepted_ids, *fit.rejected_ids, *fit.holdout_ids))
                <= {c.observation_id for c in factors.checks}
            ):
                raise ValueError("indexed raw candidate/evidence/split binding mismatch")
        if (
            evaluation.method_id != "raw_joint_response.v1"
            or evaluation.scope != "local_raw_joint_response"
            or fits[0].topology != fits[1].topology
            or fits[0].metadata["domain"] != fits[1].metadata["domain"]
            or self.factors.feature != self.baseline_factors.feature
            or self.factors.tracking != self.baseline_factors.tracking
            or self.factors.manifest.node_session_id
            != self.baseline_factors.manifest.node_session_id
        ):
            raise ValueError("paired indexed raw identity/domain/policies differ")
        nuisance = digest_of(
            {
                "domain": fits[0].metadata["domain"],
                "candidate_config": fits[0].metadata["config"],
                "baseline_config": fits[1].metadata["config"],
            }
        )
        if evaluation.nuisance_digest != nuisance:
            raise ValueError("indexed raw evaluation nuisance binding differs")
        object.__setattr__(
            self,
            "indexed_holdout_ids",
            identities(self.indexed_holdout_ids, minimum=0, maximum=1024),
        )
        exclusions = replace(evaluation, units=(), exclusions=self.indexed_exclusions).exclusions
        object.__setattr__(self, "indexed_exclusions", exclusions)
        allowed, excluded = set(self.indexed_holdout_ids), {i for i, _ in exclusions}
        units = {u.observation_id for u in evaluation.units}
        numerical_exclusions = {i for i, _ in evaluation.exclusions}
        if (
            allowed & excluded
            or allowed | excluded != set(fits[0].holdout_ids)
            or not units <= allowed
            or not excluded <= numerical_exclusions
            or units | numerical_exclusions != set(fits[0].holdout_ids)
        ):
            raise ValueError("indexed raw assessment must account for every withheld branch")

    def to_dict(self) -> dict[str, Any]:
        fit = EstimateBundle.from_mapping(json.loads(self.candidate_json))
        baseline = EstimateBundle.from_mapping(json.loads(self.baseline_json))
        return {
            "schema": "leapflow.indexed_raw_assessment.v1",
            "method_id": "raw_joint_response.v1",
            "scope": "local_raw_joint_response",
            "candidate_digest": fit.digest,
            "baseline_digest": baseline.digest,
            "evidence_digest": self.factors.manifest.evidence_digest,
            "baseline_evidence_digest": self.baseline_factors.manifest.evidence_digest,
            "topology_digest": fit.topology.digest,
            "nominal_model_digest": fit.topology.model.digest,
            "feature": self.factors.feature.to_dict(),
            "domain_digest": digest_of(fit.metadata["domain"]),
            "supported_blocks": list(fit.supported_blocks),
            "retained_blocks": list(fit.retained_blocks),
            "conditioning": {
                k: fit.metadata[k]
                for k in ("data_rank", "condition", "response_rank", "prior_rows_in_rank")
            },
            "baseline_conditioning": {
                k: baseline.metadata[k]
                for k in ("data_rank", "condition", "response_rank", "prior_rows_in_rank")
            },
            "supported_subspace_digest": fit.metadata["supported_subspace_digest"],
            "measurement_floor_m_rad": list(fit.metadata["residual_floor_m_rad"]),
            "factor_digest": self.factors.digest,
            "baseline_factor_digest": self.baseline_factors.digest,
            "source_manifest": self.factors.manifest.to_dict(),
            "baseline_source_manifest": self.baseline_factors.manifest.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "indexed_holdout_ids": list(self.indexed_holdout_ids),
            "indexed_exclusions": [
                {"observation_id": i, "reason": r} for i, r in self.indexed_exclusions
            ],
            "qualified_consumers": [],
            "physical_admission": "not_established",
            "global_geometry": "not_established",
            "unmet_physical_gates": [
                "independent_feature_association",
                "independent_scene_or_carried_link_witness",
                "measured_interval_peak_rates",
                "required_consumer_compatibility",
                "uncertainty_inflated_clearance",
            ],
        }

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())


def _work_budget(fit: EstimateBundle, bundle: EvidenceBundle, feature: MetricTaskFeature) -> None:
    from leapcalib.estimation.response import ResponseConfig

    if not isinstance(fit, EstimateBundle) or fit.method_id != "raw_joint_response.v1":
        raise ValueError("separately versioned raw-response candidate required")
    solve = ResponseConfig(**fit.metadata["config"])
    domain = RawTaskDomain.from_mapping(fit.metadata["domain"])
    if (
        solve.max_samples > 256
        or solve.trials > 48
        or len(domain.joint_ids) > 32
        or sum(isinstance(o, RawTaskTransition) for o in bundle.observations) > 256
    ):
        raise ValueError("owner raw assessment exceeds integrated numerical work budgets")
    if domain.task.feature_id != feature.feature_id or domain.task.domain_id != feature.domain_id:
        raise ValueError("raw response domain differs from frozen indexed task feature")


def assess_indexed_raw(
    store: IndexedCaptureReader,
    bundle: EvidenceBundle,
    fit: EstimateBundle,
    acquisition_ids: Mapping[str, str],
    tracking_results: Mapping[str, Mapping[str, Any]],
    *,
    baseline_bundle: EvidenceBundle,
    baseline: EstimateBundle,
    baseline_acquisition_ids: Mapping[str, str],
    baseline_tracking_results: Mapping[str, Mapping[str, Any]],
    feature: MetricTaskFeature,
    tracking: TrackingConfig,
    config: EvaluationConfig,
    trusted_keys: Mapping[str, Ed25519PublicKey],
    node_id: str,
) -> IndexedRawAssessment:
    """Reopen both evidence streams and recompute fits/paired heldouts without authority.

    One shared snapshot bounds the total keyframes across both streams. Rejected
    and unused raw factors are validated; complete baseline acquisition windows
    also exclude candidate holdouts. Missing witness evidence stays insufficient,
    rather than being replaced with caller flags to enable a numerical fit.
    """
    if (
        not isinstance(store, IndexedCaptureReader)
        or any(not isinstance(b, EvidenceBundle) for b in (bundle, baseline_bundle))
        or not isinstance(feature, MetricTaskFeature)
        or not isinstance(tracking, TrackingConfig)
        or not isinstance(config, EvaluationConfig)
        or any(
            not isinstance(m, Mapping)
            for m in (
                acquisition_ids,
                tracking_results,
                baseline_acquisition_ids,
                baseline_tracking_results,
                trusted_keys,
            )
        )
    ):
        raise ValueError("typed indexed raw assessment inputs required")
    _work_budget(fit, bundle, feature)
    _work_budget(baseline, baseline_bundle, feature)
    bundle = EvidenceBundle.from_mapping(json.loads(_json(bundle.to_dict())))
    baseline_bundle = EvidenceBundle.from_mapping(json.loads(_json(baseline_bundle.to_dict())))
    candidate_json, baseline_json = _json(fit.to_dict()), _json(baseline.to_dict())
    fit = EstimateBundle.from_mapping(json.loads(candidate_json))
    baseline = EstimateBundle.from_mapping(json.loads(baseline_json))
    snapshot = _CaptureSnapshot(store)
    keys = dict(trusted_keys)
    checked = indexed_raw_task_factors(
        snapshot,
        bundle,
        dict(acquisition_ids),
        json.loads(_json(tracking_results)),
        feature=feature,
        config=tracking,
        trusted_keys=keys,
        node_id=node_id,
    )
    before = indexed_raw_task_factors(
        snapshot,
        baseline_bundle,
        dict(baseline_acquisition_ids),
        json.loads(_json(baseline_tracking_results)),
        feature=feature,
        config=tracking,
        trusted_keys=keys,
        node_id=node_id,
    )
    selected, exclusions = independent_indexed_holdouts(
        bundle,
        checked.manifest,
        baseline_bundle=baseline_bundle,
        baseline_manifest=before.manifest,
        separation_s=config.separation_s,
    )
    evaluation = evaluate_raw_candidate(
        bundle, fit, baseline, baseline_bundle=baseline_bundle, config=config
    )
    return IndexedRawAssessment(
        candidate_json,
        baseline_json,
        checked,
        before,
        _indexed_evaluation(evaluation, selected, exclusions),
        selected,
        exclusions,
    )
