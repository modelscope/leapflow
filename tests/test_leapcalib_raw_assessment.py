# Copyright (c) Alibaba, Inc. and its affiliates.
"""Paired owner reads over real private storage, with simulated/unwitnessed captures."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from leapcalib.contracts.raw_task import RawTaskDomain, RawTaskTransition
from leapcalib.contracts.task import TaskDomain
from leapcalib.contracts.tracking import TrackingConfig
from leapcalib.estimation.raw_response import estimate_raw_response
from leapcalib.estimation.tracking import track_metric
from leapcalib.validation.bundle import RAW_SCHEMA_VERSION
from leapcalib.validation.evaluation import EvaluationConfig
from leapflow.hardware.calibration.acquisition import SettledObservation
from leapflow.hardware.calibration.raw_assessment import IndexedRawAssessment
from leapflow.hardware.skills.protocol import SkillError
from tests.test_leapcalib_assessment import append_pair
from tests.test_leapcalib_factors import indexed_pair
from tests.test_leapcalib_raw_factors import definition, raw_factor

pytestmark = pytest.mark.component


def raw_domain(state):
    """Declared synthetic study box, not measured clearance or physical admission."""
    task = TaskDomain(
        state.task.domain_id,
        state.task.feature_id,
        (-1, -1, -1, -3),
        (1, 1, 1, 3),
        (0.005, 0.005, 0.005, 0.017),
        (0.003, 0.003, 0.003, 0.034),
        0.03,
        0.1,
        0.01,
        0.02,
        0.5,
    )
    return RawTaskDomain(
        task,
        state.joint_ids,
        state.joints_rad,
        (-0.1,) * len(state.joint_ids),
        (0.1,) * len(state.joint_ids),
        (0.01,) * len(state.joint_ids),
        state.scope_digest,
    )


async def inputs(tmp_path: Path, *, shift: float = 2.0):
    registry, store, captures, original, sources, results, verifier, keys = await indexed_pair(
        tmp_path
    )
    try:
        original, sources, results = await append_pair(
            store, captures, original, sources, results, shift=shift
        )
        feature = definition(results["motion"])
        training = raw_factor(captures, original.topology, results["motion"], feature, "raw")
        held_captures = []
        for capture in captures:
            content = await asyncio.to_thread(
                store.read, sources["held:" + capture.roi.frame_id], workcell_id="cell"
            )
            held_captures.append(SettledObservation.from_mapping(content["observation"]))
        # Repeated synthetic images have known support. Fresh source IDs are real
        # persisted fixture IDs; this input does not establish physical association.
        tracked = track_metric(
            *[c.roi for c in held_captures],
            tracks=[
                {
                    "feature_id": r["feature_id"],
                    "frame_id": held_captures[0].roi.frame_id,
                    "pixel": r["first_pixel"],
                }
                for r in results["motion"]["tracks"]
            ],
        )
        heldout = raw_factor(held_captures, original.topology, tracked, feature, "withheld")
        primitive = tuple(
            o for o in original.observations if o.observation_id not in ("motion", "heldout")
        )
        bundle = replace(
            original,
            observations=(*primitive, training, heldout),
            accepted_ids=("raw",),
            holdout_ids=("withheld",),
            schema_version=RAW_SCHEMA_VERSION,
        )
        baseline_sources = {
            *training.first.frame_ids,
            *training.first.feedback_ids,
            *training.second.frame_ids,
            *training.second.feedback_ids,
        }
        baseline_bundle = replace(
            bundle,
            bundle_id="baseline-indexed-raw",
            holdout_ids=(),
            observations=(
                *[o for o in primitive if o.observation_id in baseline_sources],
                training,
            ),
        )
        domain = raw_domain(training.first)
        fit = estimate_raw_response(bundle, domain)
        baseline = estimate_raw_response(baseline_bundle, domain)
        options = {
            "baseline_bundle": baseline_bundle,
            "baseline": baseline,
            "baseline_acquisition_ids": {i: sources[i] for i in baseline_sources},
            "baseline_tracking_results": {"raw": results["motion"]},
            "feature": feature,
            "tracking": TrackingConfig(),
            "config": EvaluationConfig(),
            "trusted_keys": keys,
            "node_id": "node",
        }
        return (
            registry,
            store,
            bundle,
            fit,
            sources,
            {"raw": results["motion"], "withheld": tracked},
            verifier,
            options,
        )
    except Exception:
        await registry.close_all()
        raise


@pytest.mark.parametrize("shift", [0.2, 2.0])
async def test_indexed_paired_raw_assessment_preserves_missing_witness_and_full_windows(
    tmp_path: Path, monkeypatch, shift: float
) -> None:
    registry, store, bundle, fit, sources, results, verifier, options = await inputs(
        tmp_path, shift=shift
    )
    reads = []
    original_read = store.read

    def counted(identity, *, workcell_id):
        reads.append(identity)
        return original_read(identity, workcell_id=workcell_id)

    monkeypatch.setattr(store, "read", counted)
    try:
        nonces = dict(verifier._seen_nonces)
        assessment = await store.assess_raw_candidate(bundle, fit, sources, results, **options)
        assert isinstance(assessment, IndexedRawAssessment)
        assert assessment.evaluation.verdict == "insufficient_evidence"
        assert not assessment.evaluation.units and not fit.supported_blocks
        assert len(reads) == len(set(sources.values())) == len(set(reads))
        if shift == 0.2:
            assert assessment.indexed_exclusions == (
                ("withheld", "shared or correlated indexed acquisition window"),
            )
            assert not assessment.indexed_holdout_ids
        else:
            assert (
                assessment.indexed_holdout_ids == ("withheld",)
                and not assessment.indexed_exclusions
            )
            assert "witness" in assessment.evaluation.exclusions[0][1]
        data = assessment.to_dict()
        assert data["candidate_digest"] == fit.digest
        assert data["baseline_digest"] == options["baseline"].digest
        assert data["baseline_evidence_digest"] == options["baseline_bundle"].digest
        assert data["physical_admission"] == data["global_geometry"] == "not_established"
        assert data["qualified_consumers"] == [] and data["supported_blocks"] == []
        assert data["conditioning"]["prior_rows_in_rank"] == 0
        assert "independent_scene_or_carried_link_witness" in data["unmet_physical_gates"]
        assert not any(
            k in json.dumps(data) for k in ("signature", "color_bgr8", "depth_m_f32le", '"permit":')
        )
        identity = assessment.digest
        data["evaluation"]["verdict"] = "qualified"
        data["feature"]["yaw_ids"].reverse()
        assert assessment.digest == identity
        repeated = await store.assess_raw_candidate(bundle, fit, sources, results, **options)
        assert repeated.digest == identity and verifier._seen_nonces == nonces
        assert registry.calibration_generations.snapshot("cell") is None
        assert not store._pending
        with pytest.raises(ValueError, match="every withheld"):
            replace(assessment, indexed_holdout_ids=(), indexed_exclusions=())
        with pytest.raises(ValueError, match="binding mismatch"):
            replace(assessment, baseline_json=assessment.candidate_json)
    finally:
        await registry.close_all()


@pytest.mark.parametrize(
    "defect",
    [
        "fit",
        "baseline",
        "baseline-result",
        "baseline-source",
        "baseline-evidence",
        "witness",
        "feature",
        "budgets",
        "baseline-budgets",
        "expired",
        "unsigned",
    ],
)
async def test_raw_assessment_cannot_substitute_either_indexed_evidence_stream(
    tmp_path: Path, monkeypatch, defect: str
) -> None:
    registry, store, bundle, fit, sources, results, _, options = await inputs(tmp_path)
    try:
        if defect == "fit":
            fit = replace(fit, metadata={**fit.metadata, "data_rank": 4})
        elif defect == "baseline":
            before = options["baseline"]
            options["baseline"] = replace(
                before, metadata={**before.metadata, "residual_floor_m_rad": [1e-8] * 4}
            )
        elif defect == "baseline-result":
            options["baseline_tracking_results"] = {
                "raw": {**results["raw"], "witness_valid": True}
            }
        elif defect == "baseline-source":
            options["baseline_acquisition_ids"] = {}
        elif defect == "baseline-evidence":
            before = options["baseline_bundle"]
            options["baseline_bundle"] = replace(before, bundle_id="substituted-baseline")
        elif defect == "witness":
            bundle = replace(
                bundle,
                observations=tuple(
                    replace(o, first=replace(o.first, witness_valid=True))
                    if isinstance(o, RawTaskTransition)
                    else o
                    for o in bundle.observations
                ),
            )
        elif defect == "feature":
            options["feature"] = replace(options["feature"], domain_id="different-task")
        elif defect in ("budgets", "baseline-budgets"):
            value = fit if defect == "budgets" else options["baseline"]
            value = replace(
                value,
                metadata={**value.metadata, "config": {**value.metadata["config"], "trials": 49}},
            )
            if defect == "budgets":
                fit = value
            else:
                options["baseline"] = value
        elif defect == "expired":
            monkeypatch.setattr(
                "leapflow.hardware.calibration.evidence.time.time", lambda: 9999999999
            )
        else:
            options["trusted_keys"].clear()
        with pytest.raises(ValueError):
            await store.assess_raw_candidate(bundle, fit, sources, results, **options)
        assert not store._pending
        assert registry.calibration_generations.snapshot("cell") is None
    finally:
        await registry.close_all()


@pytest.mark.parametrize("fail", [False, True])
async def test_paired_raw_assessment_freezes_both_streams_and_drains_repeated_cancellation(
    tmp_path: Path, monkeypatch, fail: bool
) -> None:
    registry, store, bundle, fit, sources, results, _, options = await inputs(tmp_path)
    entered, release = threading.Event(), threading.Event()
    read = store.read
    baseline_only_read = next(iter(options["baseline_acquisition_ids"].values()))
    reads = []

    def blocked(identity, *, workcell_id):
        reads.append(identity)
        if identity == baseline_only_read:
            entered.set()
            assert release.wait(5)
            if fail:
                raise ValueError("injected paired read failure")
        return read(identity, workcell_id=workcell_id)

    monkeypatch.setattr(store, "read", blocked)
    task = asyncio.create_task(store.assess_raw_candidate(bundle, fit, sources, results, **options))
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        fit.metadata["data_rank"] = 4
        options["baseline"].metadata["data_rank"] = 4
        bundle.metadata["node_session_id"] = "mutated"
        options["baseline_bundle"].metadata["node_session_id"] = "mutated"
        for mapping in (
            sources,
            results,
            options["baseline_acquisition_ids"],
            options["baseline_tracking_results"],
            options["trusted_keys"],
        ):
            mapping.clear()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        closing = asyncio.create_task(registry.close_all())
        await asyncio.sleep(0.01)
        assert not task.done() and not closing.done()
        release.set()
        with pytest.raises(ValueError if fail else asyncio.CancelledError):
            await task
        await closing
        assert not store._pending and len(reads) == len(set(reads))
        with pytest.raises(SkillError, match="shutting down"):
            await store.assess_raw_candidate(bundle, fit, sources, results, **options)
    finally:
        release.set()
        await asyncio.gather(task, *([closing] if closing else []), return_exceptions=True)
        await registry.close_all()
