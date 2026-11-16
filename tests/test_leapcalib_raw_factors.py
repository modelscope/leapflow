# Copyright (c) Alibaba, Inc. and its affiliates.
"""Owner-indexed encoder/features with simulated captures, never physical admission."""

from __future__ import annotations

import asyncio
import json
import math
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from leapcalib.contracts.identity import digest_of
from leapcalib.contracts.raw_task import MetricTaskFeature, RawTaskTransition
from leapcalib.contracts.tracking import TrackingConfig
from leapcalib.estimation.tracking import track_metric
from leapcalib.validation.bundle import RAW_SCHEMA_VERSION
from leapflow.hardware.calibration.factors import IndexedRawTaskFactors, metric_raw_states
from leapflow.hardware.calibration.provenance import independent_indexed_holdouts
from leapflow.hardware.skills.protocol import SkillError
from tests.test_leapcalib_assessment import append_pair
from tests.test_leapcalib_factors import indexed_pair

pytestmark = pytest.mark.component


def definition(result):
    tracks = result["tracks"]
    left = min(tracks, key=lambda t: t["first_m"][0])["feature_id"]
    right = max(tracks, key=lambda t: t["first_m"][0])["feature_id"]
    return MetricTaskFeature(tuple(t["feature_id"] for t in tracks), (left, right), "camera-task")


def raw_factor(captures, topology, result, feature, identity):
    states = metric_raw_states(*captures, result, feature, topology, config=TrackingConfig())
    return RawTaskTransition(
        identity, captures[0].roi.profile, topology.model, *states, "metric_rgbd"
    )


async def raw_inputs(tmp_path):
    registry, store, captures, original, sources, results, verifier, keys = await indexed_pair(
        tmp_path
    )
    feature = definition(results["motion"])
    factor = raw_factor(captures, original.topology, results["motion"], feature, "raw")
    bundle = replace(
        original,
        observations=(*original.observations[:-1], factor),
        accepted_ids=("raw",),
        schema_version=RAW_SCHEMA_VERSION,
    )
    return (
        registry,
        store,
        captures,
        bundle,
        sources,
        {"raw": results["motion"]},
        feature,
        verifier,
        keys,
    )


async def test_raw_task_values_are_reconstructed_from_private_indexed_sources(
    tmp_path: Path,
) -> None:
    registry, store, captures, bundle, sources, results, feature, verifier, keys = await raw_inputs(
        tmp_path
    )
    try:
        checked = await store.raw_task_factors(
            bundle,
            sources,
            results,
            feature=feature,
            config=TrackingConfig(),
            trusted_keys=keys,
            node_id="node",
        )
        assert isinstance(checked, IndexedRawTaskFactors)
        assert checked.manifest.evidence_digest == bundle.digest
        factor = bundle.observations[-1]
        assert checked.checks[0].observation_digest == digest_of(factor.to_dict())
        for state, capture, side in zip(
            (factor.first, factor.second), captures, ("first", "second")
        ):
            points = {r["feature_id"]: r[side + "_m"] for r in results["raw"]["tracks"]}
            centroid = tuple(
                math.fsum(p[k] for p in points.values()) / len(points) for k in range(3)
            )
            a, b = (points[i] for i in feature.yaw_ids)
            yaw = math.atan2(b[1] - a[1], b[0] - a[0])
            assert state.task.feature_m_rad == pytest.approx((*centroid, yaw), abs=1e-10)
            assert state.task.pose_m_rad[:3] == tuple(
                v / 1000 for v in capture.feedback[-2].tcp.position
            )
            assert state.joints_rad == capture.feedback[-2].joints_rad
            assert state.feedback_ids == tuple(f.source_id for f in capture.feedback)
            assert min(state.joint_uncertainty_rad) > 0
            assert min(state.task.uncertainty_m_rad) >= TrackingConfig().depth_floor_m
            assert state.witness_valid is False
        data = checked.to_dict()
        for field in (
            "independent_witness",
            "feature_association",
            "interval_peak_rates",
            "physical_admission",
            "global_geometry",
        ):
            assert data[field] == "not_established"
        assert data["pose_semantics"] == "nominal_firmware_flange"
        assert "signature" not in json.dumps(data) and "color_bgr8" not in json.dumps(data)
        identity = checked.digest
        data["factors"][0]["state_digests"][0] = "tampered"
        assert checked.digest == identity
        assert registry.calibration_generations.snapshot("cell") is None
        nonces = dict(verifier._seen_nonces)
        repeated = await store.raw_task_factors(
            bundle,
            sources,
            results,
            feature=feature,
            config=TrackingConfig(),
            trusted_keys=keys,
            node_id="node",
        )
        assert repeated.digest == identity and verifier._seen_nonces == nonces
    finally:
        await registry.close_all()


@pytest.mark.parametrize(
    "defect",
    [
        "raw",
        "encoder-floor",
        "feature",
        "pose",
        "yaw",
        "feature-floor",
        "state-alias",
        "feature-alias",
        "scope",
        "cohort",
        "witness",
        "missing-result",
        "unsigned",
        "expired",
    ],
)
async def test_raw_lineage_cannot_launder_task_values_or_witness_flags(
    tmp_path: Path, monkeypatch, defect: str
) -> None:
    registry, store, _, bundle, sources, results, feature, _, keys = await raw_inputs(tmp_path)
    factor = bundle.observations[-1]
    state = factor.first
    try:
        with pytest.raises(ValueError):
            if defect == "raw":
                state = replace(state, joints_rad=(state.joints_rad[0] + 0.001,))
            elif defect == "encoder-floor":
                state = replace(state, joint_uncertainty_rad=(0,))
            elif defect == "feature":
                state = replace(state, task=replace(state.task, feature_m_rad=(0, 0, 0.5, 0)))
            elif defect == "pose":
                state = replace(state, task=replace(state.task, pose_m_rad=(0, 0, 0, 0)))
            elif defect == "yaw":
                f = state.task.feature_m_rad
                state = replace(
                    state, task=replace(state.task, feature_m_rad=(*f[:3], f[3] + math.pi))
                )
            elif defect == "feature-floor":
                state = replace(state, task=replace(state.task, uncertainty_m_rad=(0,) * 4))
            elif defect == "state-alias":
                state = replace(state, task=replace(state.task, state_id="caller-state"))
            elif defect == "feature-alias":
                state = replace(state, task=replace(state.task, feature_id="different-feature"))
            elif defect == "scope":
                state = replace(state, scope_digest="different-model-tool-profile")
            elif defect == "cohort":
                state = replace(state, feedback_ids=(state.feedback_ids[-2],))
            elif defect == "witness":
                state = replace(state, witness_valid=True)
            elif defect == "missing-result":
                results.clear()
            elif defect == "unsigned":
                keys.clear()
            elif defect == "expired":
                monkeypatch.setattr(
                    "leapflow.hardware.calibration.evidence.time.time", lambda: 9999999999
                )
            factor = replace(factor, first=state)
            bundle = replace(bundle, observations=(*bundle.observations[:-1], factor))
            await store.raw_task_factors(
                bundle,
                sources,
                results,
                feature=feature,
                config=TrackingConfig(),
                trusted_keys=keys,
                node_id="node",
            )
        assert registry.calibration_generations.snapshot("cell") is None
    finally:
        await registry.close_all()


async def test_raw_factors_check_unused_rejected_and_complete_holdout_windows(
    tmp_path: Path,
) -> None:
    registry, store, captures, original, sources, results, _, keys = await indexed_pair(tmp_path)
    try:
        original, sources, results = await append_pair(
            store, captures, original, sources, results, shift=0.2
        )
        feature = definition(results["motion"])
        first = raw_factor(captures, original.topology, results["motion"], feature, "raw")
        # Reopen the actual new captures rather than inventing aliases for old ones.
        from leapflow.hardware.calibration.acquisition import SettledObservation

        held_captures = []
        for identity in ("held:" + c.roi.frame_id for c in captures):
            content = await asyncio.to_thread(store.read, sources[identity], workcell_id="cell")
            held_captures.append(SettledObservation.from_mapping(content["observation"]))
        # The repeated synthetic images have known feature correspondence; rerun
        # tracking with those IDs instead of letting fresh seeding change support.
        results["heldout"] = track_metric(
            held_captures[0].roi,
            held_captures[1].roi,
            tracks=[
                {
                    "feature_id": r["feature_id"],
                    "frame_id": held_captures[0].roi.frame_id,
                    "pixel": r["first_pixel"],
                }
                for r in results["motion"]["tracks"]
            ],
        )
        held = raw_factor(held_captures, original.topology, results["heldout"], feature, "withheld")
        primitives = tuple(
            o for o in original.observations if o.observation_id not in ("motion", "heldout")
        )
        bundle = replace(
            original,
            observations=(*primitives, first, held),
            accepted_ids=("raw",),
            holdout_ids=("withheld",),
            schema_version=RAW_SCHEMA_VERSION,
        )
        checked = await store.raw_task_factors(
            bundle,
            sources,
            {"raw": results["motion"], "withheld": results["heldout"]},
            feature=feature,
            config=TrackingConfig(),
            trusted_keys=keys,
            node_id="node",
        )
        selected, excluded = independent_indexed_holdouts(bundle, checked.manifest)
        assert not selected and excluded == (
            ("withheld", "shared or correlated indexed acquisition window"),
        )
        extra = replace(first, observation_id="unused")
        for rejected in ((), ("unused",)):
            changed = replace(
                bundle, observations=(*bundle.observations, extra), rejected_ids=rejected
            )
            with pytest.raises(ValueError, match="witness"):
                await store.raw_task_factors(
                    changed,
                    sources,
                    {
                        "raw": results["motion"],
                        "withheld": results["heldout"],
                        "unused": {**results["motion"], "witness_valid": True},
                    },
                    feature=feature,
                    config=TrackingConfig(),
                    trusted_keys=keys,
                    node_id="node",
                )
    finally:
        await registry.close_all()


@pytest.mark.parametrize("fail", [False, True])
async def test_raw_task_reads_freeze_inputs_and_settle_on_cancel_shutdown(
    tmp_path: Path, monkeypatch, fail: bool
) -> None:
    registry, store, _, bundle, sources, results, feature, _, keys = await raw_inputs(tmp_path)
    entered, release = threading.Event(), threading.Event()
    read = store.read

    def blocked(identity, *, workcell_id):
        entered.set()
        assert release.wait(5)
        if fail:
            raise ValueError("injected raw read failure")
        return read(identity, workcell_id=workcell_id)

    monkeypatch.setattr(store, "read", blocked)
    task = asyncio.create_task(
        store.raw_task_factors(
            bundle,
            sources,
            results,
            feature=feature,
            config=TrackingConfig(),
            trusted_keys=keys,
            node_id="node",
        )
    )
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        bundle.metadata["node_session_id"] = "mutated"
        sources.clear()
        results.clear()
        keys.clear()
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
        assert not store._pending
        with pytest.raises(SkillError, match="shutting down"):
            await store.raw_task_factors(
                bundle,
                sources,
                results,
                feature=feature,
                config=TrackingConfig(),
                trusted_keys=keys,
                node_id="node",
            )
    finally:
        release.set()
        await asyncio.gather(task, *([closing] if closing else []), return_exceptions=True)
        await registry.close_all()


@pytest.mark.parametrize("defect", ["unresolved-yaw", "lost-support", "wrong-topology"])
async def test_metric_task_geometry_fails_closed(tmp_path: Path, defect: str) -> None:
    registry, _, captures, bundle, _, results, feature, _, _ = await raw_inputs(tmp_path)
    try:
        result, topology = results["raw"], bundle.topology
        if defect == "unresolved-yaw":
            result = {**result, "uncertainty_m_rad": (0.3, 0.3, 0.3, 1)}
        elif defect == "lost-support":
            feature = replace(feature, track_ids=(*feature.track_ids, "unobserved-track"))
        else:
            topology = replace(
                topology, model=replace(topology.model, chain_digest="different-chain")
            )
        with pytest.raises(ValueError, match="unresolved|support|topology"):
            metric_raw_states(*captures, result, feature, topology, config=TrackingConfig())
    finally:
        await registry.close_all()


def test_metric_task_feature_identity_and_strict_schema() -> None:
    feature = MetricTaskFeature(
        tuple("track:" + str(i) for i in range(12)), ("track:0", "track:11"), "task"
    )
    assert MetricTaskFeature.from_mapping(feature.to_dict()) == feature
    assert (
        replace(feature, yaw_ids=tuple(reversed(feature.yaw_ids))).feature_id != feature.feature_id
    )
    with pytest.raises(ValueError, match="endpoints"):
        replace(feature, yaw_ids=("unknown", "track:11"))
    with pytest.raises(ValueError, match="distinct"):
        replace(feature, yaw_ids=("track:0", "track:0"))
    with pytest.raises(ValueError, match="count"):
        replace(feature, track_ids=feature.track_ids[:11])
    with pytest.raises(ValueError, match="semantics"):
        MetricTaskFeature.from_mapping({**feature.to_dict(), "coordinates": "world_tool_xyz_yaw"})
