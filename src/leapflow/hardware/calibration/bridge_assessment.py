# Copyright (c) Alibaba, Inc. and its affiliates.
"""Owner-indexed stationary bridge verification and immutable shadow provenance.

No caller-supplied pose, verdict, or report grants frame authority. Reopening the
private index recomputes source-disjoint withheld evaluation from the sealed run.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from leapcalib.contracts.identity import digest_of
from leapcalib.estimation.bridge import BridgeBudget, METHOD, SCOPE, solve_bridge
from leapcalib.geometry.se3 import Transform
from leapflow.hardware.calibration.bridge_policy import BridgePolicy
from leapflow.hardware.calibration.paired import PairedObservation

SCHEMA = "leapflow.bridge_shadow.v1"


def _same(actual: Any, expected: Any) -> bool:
    """Strict shape/identity plus numerical roundoff only, never verdict tolerance."""
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and set(actual) == set(expected) and all(_same(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(_same(a, b) for a, b in zip(actual, expected))
    if type(expected) is float:
        return type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10)
    return type(actual) is type(expected) and actual == expected


def verify_indexed_result(store: Any, captures: Sequence[Mapping[str, str]], result: Mapping[str, Any], *,
                          workcell_id: str, plan: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    """Synchronous off-loop reconstruction; called through tracked owner storage."""
    from leapflow.hardware.calibration.bridge_port import indexed_pair

    basis = plan["basis"]["bridge"]
    policy = BridgePolicy.from_mapping(basis["policy"])
    if len(captures) != len(basis["allocation"]):
        raise ValueError("bridge allocation was shortened or reallocated")
    training, holdout, excluded, bindings = [], [], [], []
    evidence_ids, sources, intervals = set(), set(basis["dependency"]["source_ids"]), []
    for capture, slot in zip(captures, basis["allocation"]):
        if set(capture) != {"evidence_digest", "allocation"} or capture["allocation"] != slot:
            raise ValueError("bridge immutable capture allocation differs from preview")
        if capture["evidence_digest"] in evidence_ids:
            raise ValueError("bridge indexed evidence overlap, including excluded windows")
        evidence_ids.add(capture["evidence_digest"])
        content = store.read(capture["evidence_digest"], workcell_id=workcell_id)
        binding = content["binding"]
        if binding["plan"] != plan or binding["run_id"] != run_id or binding["node_session_id"] != basis["node_session_id"]:
            raise ValueError("bridge paired evidence belongs to another admitted run")
        bindings.append(digest_of(binding))
        observation = PairedObservation.from_mapping(content["observation"])
        if sources.intersection(observation.source_ids):
            raise ValueError("bridge source overlap, including excluded windows and wrist refinement")
        sources.update(observation.source_ids)
        intervals.append(observation.interval_s)
        if observation.exclusions:
            excluded.append({**capture, "reasons": list(observation.exclusions)})
        else:
            pair = indexed_pair(observation, capture["evidence_digest"], basis, policy)
            (training if slot == "training" else holdout).append(pair)
    if len(set(bindings)) != 1:
        raise ValueError("bridge acquisition authority differs across pairs")
    ordered = sorted(intervals)
    if any(b[0] - a[1] < policy.budget.separation_s for a, b in zip(ordered, ordered[1:])):
        raise ValueError("bridge physical windows overlap, including excluded observations")
    dependency = basis["dependency"]
    expected = solve_bridge(training, holdout, flange_T_wrist=Transform.from_mapping(dependency["transform"]),
                            dependency_digest=digest_of(dependency), dependency_sources=dependency["source_ids"], budget=policy.budget)
    if not _same(result, expected):
        raise ValueError("bridge worker result differs from indexed method-specific evaluation")
    # Never repeat the untrusted result: emit the freshly recomputed result.
    return {"schema": SCHEMA, "method_id": METHOD, "scope": SCOPE, "workcell_id": workcell_id,
            "run_id": run_id, "plan_digest": plan["digest"], "binding_digest": bindings[0],
            "evidence_digest": digest_of({"captures": list(captures), "binding_digest": bindings[0]}),
            "captures": list(captures), "excluded_observations": excluded, "dependency": dependency,
            "evaluation": expected, "shadow_only": True, "activation": "none", "commands": [],
            "raw_exported": False, "hardware_accuracy_certified": False,
            "wrist_refinement_disjointness": dependency["source_disjointness_basis"]}


def shadow_problem(record: Mapping[str, Any]) -> str:
    """Method-specific derived provenance checks; indexed replay remains mandatory."""
    try:
        required = {"schema", "method_id", "scope", "workcell_id", "run_id", "plan_digest", "binding_digest",
                    "evidence_digest", "captures", "excluded_observations", "dependency", "evaluation",
                    "shadow_only", "activation", "commands", "raw_exported", "hardware_accuracy_certified",
                    "wrist_refinement_disjointness"}
        if not isinstance(record, Mapping) or set(record) != required or record["schema"] != SCHEMA or record["method_id"] != METHOD or record["scope"] != SCOPE:
            return "unknown stationary bridge provenance schema/method/scope"
        if record["shadow_only"] is not True or record["activation"] != "none" or record["commands"] != [] or record["hardware_accuracy_certified"] is not False or record["raw_exported"] is not False:
            return "stationary bridge cannot claim activation or physical qualification"
        if record["evidence_digest"] != digest_of({"captures": record["captures"], "binding_digest": record["binding_digest"]}):
            return "stationary bridge evidence allocation digest mismatch"
        result = record["evaluation"]
        if result["method_id"] != METHOD or result["scope"] != SCOPE or result["dependency_digest"] != digest_of(record["dependency"]):
            return "stationary bridge wrist dependency/evaluation mismatch"
        base_fields = {"schema", "method_id", "scope", "dependency_digest", "budget", "reachability", "training", "holdout",
                       "physical_activation", "independent_reference_accuracy", "absolute_base_accuracy_established",
                       "joint_wrist_fit", "exclusions", "verdict", "candidate", "acceptance_basis"}
        fit_fields = {"stationary_board", "observability", "candidate_digest", "training_residual_m_rad",
                      "withheld_cross_view_m_rad", "uncertainty_m_rad", "effective_uncertainty_terms", "uncertainty_basis",
                      "quality_margins_m_rad", "advisories"} if result["candidate"] is not None else set()
        if set(result) != base_fields | fit_fields or result["schema"] != "leapcalib.bridge_result.v1":
            return "unknown stationary bridge evaluation schema/fields"
        if (result["physical_activation"] != "shadow_only" or result["independent_reference_accuracy"] is not None
                or result["absolute_base_accuracy_established"] is not False or result["joint_wrist_fit"] is not False
                or result["exclusions"] != [] or record["dependency"]["independent_absolute_accuracy"] is not False
                or record["wrist_refinement_disjointness"] != record["dependency"]["source_disjointness_basis"]):
            return "stationary bridge evaluation cannot claim activation or independent accuracy"
        budget = BridgeBudget(**result["budget"])
        if not _same(result["reachability"], budget.reachability()):
            return "stationary bridge frozen budget/reachability mismatch"
        if result["acceptance_basis"] != budget.acceptance_basis:
            return "stationary bridge acceptance basis disagrees with its frozen budget"
        if result["verdict"] not in ("insufficient_evidence", "rejected", "numerical_pass"):
            return "unknown stationary bridge verdict"
        if result["candidate"] is None and result["verdict"] != "insufficient_evidence":
            return "stationary bridge verdict requires a fitted candidate"
        if result["candidate"] is not None:
            if (result["stationary_board"]["physical_stationarity_certified"] is not False
                    or result["observability"] != {"solve_mode": "fixed_only_with_validated_wrist", "conditional_pose_dof": 6,
                                                   "joint_wrist_rank_established": False, "absolute_base_rank_established": False}):
                return "stationary bridge cannot claim joint rank or physical stationarity"
            transform = Transform.from_mapping(result["candidate"])
            if (transform.parent, transform.child) != ("base", "fixed"):
                return "stationary bridge transform direction mismatch"
            identity = digest_of({"transform": result["candidate"], "dependency": result["dependency_digest"],
                                  "training": result["training"], "budget": result["budget"]})
            if identity != result["candidate_digest"]:
                return "stationary bridge candidate digest mismatch"
        return ""
    except (KeyError, TypeError, ValueError):
        return "malformed stationary bridge provenance"
