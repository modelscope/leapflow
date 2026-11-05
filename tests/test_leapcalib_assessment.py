# Copyright (c) Alibaba, Inc. and its affiliates.
"""Actual owner/store assessment paths with simulated captures, never physical admission."""

from __future__ import annotations

import asyncio
import json
import math
import struct
import threading
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from leapcalib.contracts.motion import PointCorrespondence
from leapcalib.contracts.observations import JointObservation, PixelDepthSample, RgbdObservation
from leapcalib.contracts.tracking import TrackingConfig
from leapcalib.contracts.raw_task import MetricTaskFeature, RawTaskTransition
from leapcalib.estimation.tracking import track_metric
from leapflow.hardware.calibration.evidence import AcquisitionBinding
from leapflow.hardware.skills.admission import RunAdmission
from leapflow.hardware.skills.protocol import SkillError
from tests.test_leapcalib_acquisition import profile
from leapcalib.estimation.handeye import estimate_motion
from leapcalib.estimation.numeric import SolveConfig
from leapcalib.geometry.se3 import Transform, rpy_rad_from_rotation
from leapcalib.validation.evaluation import EvaluationConfig, MotionMeasurementBounds
from leapflow.hardware.calibration.factors import (
    IndexedMotionAssessment,
    _indexed_evaluation,
    metric_raw_states,
)
from leapflow.hardware.calibration.worker import CalibrationWorker
from leapflow.hardware.skills.geometry import Pose
from tests.test_leapcalib_evaluation import bounds, motion_study, paired_result
from tests.test_leapcalib_factors import indexed_pair

pytestmark = pytest.mark.component


async def inputs(tmp_path: Path):
    registry, store, captures, bundle, sources, results, verifier, keys = await indexed_pair(
        tmp_path
    )
    try:
        bundle, sources, results = await append_pair(
            store, captures, bundle, sources, results, shift=0.2
        )
    except Exception:
        await registry.close_all()
        raise
    heldout = bundle.observations[-1]
    prior = Transform(parent="flange", child="camera")
    fit = estimate_motion(bundle, prior)
    options = {
        "tracking": TrackingConfig(),
        "config": EvaluationConfig(),
        "trusted_keys": keys,
        "node_id": "node",
        "measurement_bounds": {
            "heldout": MotionMeasurementBounds(heldout.digest, 0.002, 0.001, 0.001, 0.01, 0.02)
        },
    }
    return registry, store, bundle, fit, sources, results, verifier, options


async def append_pair(store, captures, bundle, sources, results, *, shift: float):
    """Distinct stored frame/state identities may still occupy correlated windows."""
    content = await asyncio.to_thread(
        store.read, sources[captures[0].roi.frame_id], workcell_id="cell"
    )
    data = content["binding"]
    proof = data["admission"]
    binding = AcquisitionBinding(
        data["workcell_id"],
        data["run_id"],
        data["node_session_id"],
        json.dumps(data["plan"]),
        RunAdmission(proof["mode"], proof["action_digest"], json.dumps(proof["permit"])),
    )
    rows, fresh = [], []
    for capture in captures:
        roi = replace(
            capture.roi,
            frame_id="held:" + capture.roi.frame_id,
            stamp=replace(capture.roi.stamp, seconds=capture.roi.stamp.seconds + shift),
            source_stamps=tuple(
                replace(
                    s, frame_number=s.frame_number + 10, timestamp_ms=s.timestamp_ms + shift * 1000
                )
                for s in capture.roi.source_stamps
            ),
        )
        feedback = tuple(
            replace(
                f,
                source_id="held:" + f.source_id,
                stamp=replace(f.stamp, seconds=f.stamp.seconds + shift),
                first_receipt_s=f.first_receipt_s + shift,
                last_receipt_s=f.last_receipt_s + shift,
            )
            for f in capture.feedback
        )
        value = replace(capture, roi=roi, feedback=feedback)
        fresh.append(value)
        summary = await asyncio.to_thread(
            store.record,
            value,
            profile=profile(),
            binding=binding,
            ttl_s=60,
            quota_bytes=4 * 1024 * 1024,
        )
        original = next(o for o in bundle.observations if o.observation_id == capture.roi.frame_id)
        frame = replace(
            original,
            observation_id=roi.frame_id,
            stamp=replace(
                roi.stamp, uncertainty_s=roi.stamp.uncertainty_s + roi.exposure_uncertainty_s
            ),
        )
        joints = tuple(
            JointObservation(f.source_id, bundle.topology.model, f.stamp, f.joints_rad, still=True)
            for f in feedback
        )
        rows.extend((frame, *joints))
        sources.update(
            dict.fromkeys(
                (frame.observation_id, *(j.observation_id for j in joints)),
                summary["evidence_digest"],
            )
        )
    tracked = track_metric(fresh[0].roi, fresh[1].roi)
    factor = replace(
        bundle.observations[-1],
        observation_id="heldout",
        first_id=fresh[0].roi.frame_id,
        second_id=fresh[1].roi.frame_id,
        first_stamp=rows[0].stamp,
        stamp=rows[5].stamp,
        points=tuple(
            PointCorrespondence(r["feature_id"], r["first_m"], r["second_m"])
            for r in tracked["tracks"]
        ),
    )
    bundle = replace(
        bundle, observations=(*bundle.observations, *rows, factor), holdout_ids=("heldout",)
    )
    return bundle, sources, {**results, "heldout": tracked}


async def test_owner_assessment_combines_raw_derived_fit_and_complete_window_checks(
    tmp_path: Path,
) -> None:
    registry, store, bundle, fit, sources, results, verifier, options = await inputs(tmp_path)
    try:
        nonces = dict(verifier._seen_nonces)
        assessment = await store.assess_motion_candidate(bundle, fit, sources, results, **options)
        assert isinstance(assessment, IndexedMotionAssessment)
        assert assessment.evaluation.verdict == "insufficient_evidence"
        assert assessment.indexed_holdout_ids == ()
        assert assessment.indexed_exclusions == (
            ("heldout", "shared or correlated indexed acquisition window"),
        )
        assert len(assessment.evaluation.exclusions) == 1
        reason = assessment.evaluation.exclusions[0][1]
        assert "correlated" in reason and "indexed acquisition" in reason
        assert assessment.evaluation.units == ()
        data = assessment.to_dict()
        assert data["retained_blocks"] == list(fit.retained_blocks)
        assert data["supported_blocks"] == [] and data["qualified_consumers"] == []
        assert data["physical_admission"] == "not_established"
        assert "independently_anchored_geometry" in data["unmet_physical_gates"]
        assert data["conditioning"]["prior_rows_in_rank"] == 0
        assert data["candidate_digest"] == fit.digest and data["evidence_digest"] == bundle.digest
        assert data["source_manifest"]["scope"] == "indexed_raw_values"
        assert data["factor_digest"] == assessment.factors.digest
        assert verifier._seen_nonces == nonces
        assert registry.calibration_generations.snapshot("cell") is None
        encoded = json.dumps(data)
        assert all(
            k not in encoded for k in ("signature", "color_bgr8", "depth_m_f32le", '"permit":')
        )
        identity = assessment.digest
        data["evaluation"]["verdict"] = "qualified"
        data["estimates"][0]["values"][0] = 2
        assert assessment.digest == identity
        repeated = await store.assess_motion_candidate(bundle, fit, sources, results, **options)
        assert repeated.digest == identity
        assert not store._pending
    finally:
        await registry.close_all()


@pytest.mark.parametrize(
    "defect", ["fit", "raw", "derived", "floor", "receipt", "expired", "bounds", "budgets"]
)
async def test_owner_assessment_cannot_bypass_a_failed_indexed_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    registry, store, bundle, fit, sources, results, _, options = await inputs(tmp_path)
    try:
        if defect == "fit":
            fit = replace(fit, metadata={**fit.metadata, "data_rank": 6})
        elif defect == "raw":
            row = bundle.observations[0]
            row = replace(row, samples=(replace(row.samples[0], depth_m=0.501),))
            bundle = replace(bundle, observations=(row, *bundle.observations[1:]))
            fit = estimate_motion(bundle, Transform(parent="flange", child="camera"))
        elif defect == "derived":
            factor = replace(bundle.observations[-1], robot_motion=Transform())
            bundle = replace(bundle, observations=(*bundle.observations[:-1], factor))
            fit = estimate_motion(bundle, Transform(parent="flange", child="camera"))
            options["measurement_bounds"]["heldout"] = replace(
                options["measurement_bounds"]["heldout"], observation_digest=factor.digest
            )
        elif defect == "floor":
            bound = options["measurement_bounds"]["heldout"]
            options["measurement_bounds"]["heldout"] = replace(bound, point_uncertainty_m=1e-8)
        elif defect == "receipt":
            options["trusted_keys"] = {}
        elif defect == "expired":
            monkeypatch.setattr(
                "leapflow.hardware.calibration.evidence.time.time", lambda: 9999999999
            )
        elif defect == "bounds":
            options["measurement_bounds"] = {}
        else:
            fit = replace(
                fit,
                metadata={**fit.metadata, "config": {**fit.metadata["config"], "max_points": 4096}},
            )
        with pytest.raises(ValueError):
            await store.assess_motion_candidate(bundle, fit, sources, results, **options)
        assert registry.calibration_generations.snapshot("cell") is None
        assert not store._pending
    finally:
        await registry.close_all()


async def test_owner_assessment_freezes_inputs_before_suspending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, store, bundle, fit, sources, results, _, options = await inputs(tmp_path)
    entered, release = threading.Event(), threading.Event()
    read = store.read

    def blocked(identity, *, workcell_id):
        entered.set()
        assert release.wait(5)
        return read(identity, workcell_id=workcell_id)

    monkeypatch.setattr(store, "read", blocked)
    expected = fit.digest
    task = asyncio.create_task(
        store.assess_motion_candidate(bundle, fit, sources, results, **options)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        fit.metadata["data_rank"] = 6
        results["heldout"]["executed"] = True
        sources.clear()
        options["measurement_bounds"].clear()
        options["trusted_keys"].clear()
        release.set()
        assessment = await task
        assert assessment.evaluation.candidate_digest == expected
        assert assessment.evaluation.verdict == "insufficient_evidence"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await registry.close_all()


@pytest.mark.parametrize("fail", [False, True])
async def test_assessment_cancel_shutdown_wait_for_indexed_reads_and_numerical_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
) -> None:
    registry, store, bundle, fit, sources, results, _, options = await inputs(tmp_path)
    entered, release = threading.Event(), threading.Event()
    read = store.read

    def blocked(identity, *, workcell_id):
        entered.set()
        assert release.wait(5)
        if fail:
            raise ValueError("injected assessment read failure")
        return read(identity, workcell_id=workcell_id)

    monkeypatch.setattr(store, "read", blocked)
    task = asyncio.create_task(
        store.assess_motion_candidate(bundle, fit, sources, results, **options)
    )
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
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
            await store.assess_motion_candidate(bundle, fit, sources, results, **options)
    finally:
        release.set()
        await asyncio.gather(task, *([closing] if closing else []), return_exceptions=True)
        await registry.close_all()


def test_indexed_filter_cannot_keep_correlated_units_or_erase_numerical_rejections() -> None:
    numerical = paired_result()
    assert numerical.verdict == "numerical_pass"
    selected = tuple(u.observation_id for u in numerical.units[1:])
    result = _indexed_evaluation(numerical, selected, (("0", "indexed correlation"),))
    assert len(result.units) == 19 and result.verdict == "insufficient_evidence"
    assert result.exclusions == (("0", "indexed correlation"),)
    numerical = replace(
        numerical, units=numerical.units[1:], exclusions=(("0", "ambiguous branch"),)
    )
    result = _indexed_evaluation(numerical, selected, (("0", "indexed correlation"),))
    assert result.exclusions == (("0", "ambiguous branch; indexed correlation"),)


async def test_indexed_assessment_contract_cannot_promote_scope_or_drop_branches(
    tmp_path: Path,
) -> None:
    registry, store, bundle, fit, sources, results, _, options = await inputs(tmp_path)
    try:
        assessment = await store.assess_motion_candidate(bundle, fit, sources, results, **options)
        with pytest.raises(ValueError, match="binding"):
            replace(assessment, evaluation=replace(assessment.evaluation, candidate_digest="other"))
        with pytest.raises(ValueError, match="every split"):
            replace(assessment, indexed_exclusions=())
        with pytest.raises(ValueError, match="every split"):
            replace(assessment, evaluation=replace(assessment.evaluation, exclusions=()))
        with pytest.raises(ValueError, match="scope"):
            replace(assessment, evaluation=replace(assessment.evaluation, scope="global_geometry"))
        with pytest.raises(ValueError, match="immutable"):
            replace(assessment, candidate_json={})
    finally:
        await registry.close_all()


def test_retained_motion_candidate_has_explicit_withheld_insufficiency_and_unchanged_priors() -> (
    None
):
    from leapcalib.validation.evaluation import evaluate_motion_candidate

    bundle, prior, _ = motion_study()
    bundle = replace(bundle, accepted_ids=bundle.accepted_ids[:1], rejected_ids=())
    fit = estimate_motion(bundle, prior, config=SolveConfig())
    assert fit.retained_blocks and not fit.supported_blocks
    result = evaluate_motion_candidate(bundle, fit, measurement_bounds=bounds(bundle))
    assert result.verdict == "insufficient_evidence" and not result.units
    assert {i for i, _ in result.exclusions} == set(bundle.holdout_ids)
    assert all("data-supported" in reason for _, reason in result.exclusions)
    assert all(e.values == e.prior for e in fit.estimates)
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_motion_candidate(bundle, fit, measurement_bounds={})


def projected_scene(roi, motion: Transform):
    """Render a textured plane under an exact synthetic 3-D correspondence motion."""
    import cv2
    import numpy as np

    p = roi.profile
    intrinsics = np.asarray(((p.fx_px, 0, p.ppx_px), (0, p.fy_px, p.ppy_px), (0, 0, 1)))
    rotation, translation = np.asarray(motion.rotation), np.asarray(motion.translation_m)
    homography = (
        intrinsics @ (rotation + np.outer(translation, (0, 0, 1)) / 0.5) @ np.linalg.inv(intrinsics)
    )
    color = np.frombuffer(roi.color_bgr8, dtype=np.uint8).reshape(roi.height, roi.width, 3)
    shifted = cv2.warpPerspective(color, homography, (roi.width, roi.height))
    v, u = np.mgrid[: roi.height, : roi.width]
    rays = np.stack(((u - p.ppx_px) / p.fx_px, (v - p.ppy_px) / p.fy_px, np.ones_like(u)), axis=-1)
    normal = rotation[:, 2]
    depth = ((0.5 + normal @ translation) / (rays @ normal)).astype("<f4")
    return replace(roi, color_bgr8=shifted.tobytes(), depth_m_f32le=depth.tobytes())


async def record_simulated_capture(store, capture, binding, model):
    summary = await asyncio.to_thread(
        store.record,
        capture,
        profile=profile(),
        binding=binding,
        ttl_s=120,
        quota_bytes=32 * 1024 * 1024,
    )
    roi = capture.roi
    frame = RgbdObservation(
        roi.frame_id,
        roi.profile,
        replace(roi.stamp, uncertainty_s=roi.stamp.uncertainty_s + roi.exposure_uncertainty_s),
        "metric_rgbd",
        (
            PixelDepthSample(
                5, 5, struct.unpack_from("<f", roi.depth_m_f32le, 4 * (5 * roi.width + 5))[0]
            ),
        ),
    )
    joints = tuple(
        JointObservation(f.source_id, model, f.stamp, f.joints_rad, still=True)
        for f in capture.feedback
    )
    rows = (frame, *joints)
    return rows, dict.fromkeys((r.observation_id for r in rows), summary["evidence_digest"])


async def test_full_supported_indexed_assessment_through_real_worker_and_shared_holder(
    tmp_path: Path,
) -> None:
    """Simulated flange motions/image renders test wiring, not unbiased robot geometry."""
    import numpy as np

    registry, store, templates, original, sources, _, verifier, keys = await indexed_pair(tmp_path)
    worker = CalibrationWorker(
        tmp_path.resolve().parent / ("lc-" + uuid.uuid4().hex[:8]) / "s", enabled=True
    )
    registry.bind_calibration_worker(worker)
    try:
        content = await asyncio.to_thread(
            store.read, sources[templates[0].roi.frame_id], workcell_id="cell"
        )
        data, proof = content["binding"], content["binding"]["admission"]
        binding = AcquisitionBinding(
            "cell",
            data["run_id"],
            data["node_session_id"],
            json.dumps(data["plan"]),
            RunAdmission(proof["mode"], proof["action_digest"], json.dumps(proof["permit"])),
        )
        tracking = TrackingConfig(max_tracks=32)
        primitives, factors, results, sources = [], [], {}, {}
        before = Transform(templates[0].feedback[-2].tcp.rotation, (0.1, 0, 0))
        for i in range(26):
            angles = [0.0, 0.0, 0.0]
            angles[i % 3] = 0.016 + i * 0.0001
            rotated = Transform.from_rpy_rad(*angles)
            center = np.asarray((0, 0, 0.5))
            translation = (
                center - np.asarray(rotated.rotation) @ center + np.asarray((0.0004, -0.0002, 0))
            )
            camera_motion = replace(rotated, translation_m=tuple(translation))
            after = before.compose(camera_motion.inverse())
            captures, frames = [], []
            for j, template in enumerate(templates):
                roi = (
                    templates[0].roi if j == 0 else projected_scene(templates[0].roi, camera_motion)
                )
                stamp = 1 + 3 * i + j
                roi = replace(
                    roi,
                    frame_id=f"render:{i}:{j}",
                    stamp=replace(roi.stamp, seconds=stamp),
                    source_stamps=tuple(
                        replace(s, frame_number=100 + 3 * i + j, timestamp_ms=1000 * stamp)
                        for s in roi.source_stamps
                    ),
                )
                transform = before if j == 0 else after
                pose = Pose(
                    *[v * 1000 for v in transform.translation_m],
                    *map(math.degrees, rpy_rad_from_rotation(transform.rotation)),
                )
                shift = stamp - template.roi.stamp.seconds
                feedback = tuple(
                    replace(
                        f,
                        source_id=f"render:{i}:{j}:{k}",
                        tcp=pose,
                        stamp=replace(f.stamp, seconds=f.stamp.seconds + shift),
                        first_receipt_s=f.first_receipt_s + shift,
                        last_receipt_s=f.last_receipt_s + shift,
                    )
                    for k, f in enumerate(template.feedback)
                )
                capture = replace(template, roi=roi, feedback=feedback)
                captures.append(capture)
                rows, indexed = await record_simulated_capture(
                    store, capture, binding, original.topology.model
                )
                primitives.extend(rows)
                frames.append(rows[0])
                sources.update(indexed)
            computed = await worker.submit(
                "track_metric",
                {
                    "first": captures[0].roi.to_dict(),
                    "second": captures[1].roi.to_dict(),
                    "tracks": [],
                    "config": vars(tracking),
                },
                workcell_id="cell",
                run_id="simulated-assessment",
                model_digest=original.topology.model.digest,
            )
            tracked = computed["tracking"]
            factor = replace(
                original.observations[-1],
                observation_id=f"render-motion:{i}",
                first_id=frames[0].observation_id,
                second_id=frames[1].observation_id,
                first_stamp=frames[0].stamp,
                stamp=frames[1].stamp,
                robot_motion=camera_motion.inverse(),
                max_speed_rad_s=0,
                points=tuple(
                    PointCorrespondence(r["feature_id"], r["first_m"], r["second_m"])
                    for r in tracked["tracks"]
                ),
            )
            factors.append(factor)
            results[factor.observation_id] = tracked
        bundle = replace(
            original,
            observations=(*primitives, *factors),
            accepted_ids=tuple(f.observation_id for f in factors[:6]),
            holdout_ids=tuple(f.observation_id for f in factors[6:]),
        )
        prior = Transform(translation_m=(0.02, 0, 0), parent="flange", child="camera")
        fit = estimate_motion(
            bundle,
            prior,
            config=SolveConfig(trials=4, translation_sigma_m=0.0005, rotation_sigma_rad=0.002),
        )
        assert fit.supported_blocks and fit.metadata["consensus_stable"], fit.metadata
        measurement = {
            f.observation_id: MotionMeasurementBounds(f.digest, 0.002, 0.001, 0.001, 0.01, 0.02)
            for f in factors[6:]
        }
        nonces = dict(verifier._seen_nonces)
        assessed = await store.assess_motion_candidate(
            bundle,
            fit,
            sources,
            results,
            tracking=tracking,
            config=EvaluationConfig(),
            measurement_bounds=measurement,
            trusted_keys=keys,
            node_id="node",
        )
        assert len(assessed.evaluation.units) == 20 and not assessed.evaluation.exclusions
        assert len(assessed.indexed_holdout_ids) == 20 and not assessed.indexed_exclusions
        assert len(assessed.factors.checks) == 26
        assert assessed.evaluation.verdict != "insufficient_evidence"
        assert assessed.to_dict()["physical_admission"] == "not_established"
        assert assessed.to_dict()["qualified_consumers"] == []
        assert assessed.to_dict()["conditioning"]["data_rank"] == 6
        assert verifier._seen_nonces == nonces
        assert registry.calibration_generations.snapshot("cell") is None
        assert worker.last_metrics["peak_rss_bytes"] <= worker.budgets.rss_bytes
        # The same real worker output also feeds the raw-encoder/local-feature
        # boundary. Simulated correspondence never supplies a physical witness.
        rows = tracked["tracks"]
        feature = MetricTaskFeature(
            tuple(r["feature_id"] for r in rows),
            (
                min(rows, key=lambda r: r["first_m"][0])["feature_id"],
                max(rows, key=lambda r: r["first_m"][0])["feature_id"],
            ),
            "rendered-camera-task",
        )
        states = metric_raw_states(*captures, tracked, feature, bundle.topology, config=tracking)
        raw = RawTaskTransition(
            "raw-rendered", captures[0].roi.profile, bundle.topology.model, *states, "metric_rgbd"
        )
        from leapcalib.validation.bundle import RAW_SCHEMA_VERSION

        raw_bundle = replace(
            bundle,
            observations=(*primitives[-10:], raw),
            accepted_ids=("raw-rendered",),
            rejected_ids=(),
            holdout_ids=(),
            schema_version=RAW_SCHEMA_VERSION,
        )
        raw_sources = {o.observation_id: sources[o.observation_id] for o in primitives[-10:]}
        raw_checked = await store.raw_task_factors(
            raw_bundle,
            raw_sources,
            {"raw-rendered": tracked},
            feature=feature,
            config=tracking,
            trusted_keys=keys,
            node_id="node",
        )
        assert len(raw_checked.checks) == 1 and all(not s.witness_valid for s in states)
        assert raw_checked.to_dict()["physical_admission"] == "not_established"
        assert registry.calibration_generations.snapshot("cell") is None
        assert verifier._seen_nonces == nonces
    finally:
        await worker.close()
        await registry.close_all()
