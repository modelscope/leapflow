# Copyright (c) Alibaba, Inc. and its affiliates.
"""Real indexed owner wiring with synthetic imagery, never physical witness proof."""

from __future__ import annotations

import asyncio
import copy
import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from leapcalib.contracts.failures import CalibrationFailure
from leapcalib.contracts.identity import digest_of
from leapcalib.contracts.tracking import SceneConfig
from leapcalib.estimation.tracking import compare_scene_supports, track_scene
from leapflow.hardware.calibration.scene import IndexedSceneCheck, IndexedSceneReplay
from leapflow.hardware.skills.protocol import SkillError
from tests.test_leapcalib_factors import indexed_pair
from tests.test_leapcalib_scene import numerical_policy, scene_pair
from tests.test_leapcalib_worker import worker_at

pytestmark = pytest.mark.component


async def inputs(tmp_path: Path):
    first, second, task, background = scene_pair(common_shift=2)
    registry, store, captures, bundle, sources, _, verifier, keys = await indexed_pair(
        tmp_path, rois=(first, second)
    )
    bundle = replace(
        bundle,
        observations=tuple(o for o in bundle.observations if o.observation_id != "motion"),
        accepted_ids=(),
    )
    frames = [c.roi.frame_id for c in captures]
    config = numerical_policy()
    scene = track_scene(
        *[c.roi for c in captures], task_tracks=task, background_tracks=background, config=config
    )
    options = {
        "task_tracks": task,
        "background_tracks": background,
        "tracking": config,
        "scene_config": SceneConfig(),
        "trusted_keys": keys,
        "node_id": "node",
    }
    return registry, store, bundle, sources, frames, scene, options, verifier


async def test_indexed_scene_checks_both_raw_windows_without_authority(tmp_path: Path, monkeypatch):
    registry, store, bundle, sources, frames, scene, options, verifier = await inputs(tmp_path)
    reads = []
    read = store.read

    def counted(identity, *, workcell_id):
        reads.append(identity)
        return read(identity, workcell_id=workcell_id)

    monkeypatch.setattr(store, "read", counted)
    try:
        nonces = dict(verifier._seen_nonces)
        checked = await store.check_scene(bundle, sources, frames, scene, **options)
        assert isinstance(checked, IndexedSceneCheck)
        assert len(reads) == len(set(reads)) == 2
        assert checked.to_dict()["comparison"] == scene["comparison"]
        assert checked.to_dict()["comparison"]["verdict"] == "consistent"
        assert checked.to_dict()["physical_admission"] == "not_established"
        assert checked.to_dict()["feature_association"] == "not_established"
        assert checked.to_dict()["witness_valid"] is False
        assert checked.to_dict()["qualified_consumers"] == []
        assert not any(
            k in json.dumps(checked.to_dict())
            for k in ("signature", "color_bgr8", "depth_m_f32le", '"permit":')
        )
        identity = checked.digest
        mutated = checked.to_dict()
        mutated["comparison"]["witness_valid"] = True
        assert checked.digest == identity
        assert (
            await store.check_scene(bundle, sources, frames, scene, **options)
        ).digest == identity
        assert verifier._seen_nonces == nonces
        assert registry.calibration_generations.snapshot("cell") is None
        assert not store._pending
    finally:
        await registry.close_all()


@pytest.mark.parametrize(
    "defect",
    [
        "verdict",
        "uncertainty",
        "policy",
        "digest",
        "authority",
        "task-depth",
        "background-depth",
        "source",
        "switched",
        "alias",
        "missing-feedback",
        "extra-capture",
        "expired",
        "key",
        "budget",
        "schema",
        "tracking-authority",
        "feedback-value",
        "malformed-support",
        "malformed-acquisition",
        "frames",
    ],
)
async def test_scene_owner_rejects_substitution_and_unverified_promotion(tmp_path: Path, defect):
    registry, store, bundle, sources, frames, scene, options, _ = await inputs(tmp_path)
    try:
        if defect == "verdict":
            scene["comparison"]["verdict"] = "relative_motion"
        elif defect == "uncertainty":
            scene["comparison"]["uncertainty_m_rad"] = [0, 0]
        elif defect == "policy":
            options["scene_config"] = SceneConfig(max_translation_m=0.002)
        elif defect == "digest":
            scene["comparison"]["background_tracking_digest"] = "substituted"
        elif defect == "authority":
            scene["comparison"]["witness_valid"] = True
        elif defect.endswith("-depth"):
            group = "task_tracking" if defect == "task-depth" else "background_tracking"
            row = scene[group]["tracks"][0]
            row["first_m"] = [row["first_m"][0] + 0.001, *row["first_m"][1:]]
        elif defect == "source":
            scene["background_tracking"]["first_digest"] = "foreign"
        elif defect == "switched":
            options["background_tracks"][0]["pixel"][0] += 1
        elif defect == "alias":
            options["background_tracks"][0]["feature_id"] = options["task_tracks"][0]["feature_id"]
        elif defect == "missing-feedback":
            row = next(o for o in bundle.observations if o.observation_id not in frames)
            bundle = replace(bundle, observations=tuple(o for o in bundle.observations if o != row))
            sources.pop(row.observation_id)
        elif defect == "extra-capture":
            sources["foreign"] = "third"
        elif defect == "expired":
            store._holder.connection.execute("UPDATE calib_acquisition SET expires_at=0")
        elif defect == "key":
            options["trusted_keys"] = {}
        elif defect == "budget":
            options["background_tracks"] *= 100
        elif defect == "schema":
            scene["unexpected"] = True
        elif defect == "tracking-authority":
            scene["background_tracking"]["witness_valid"] = True
        elif defect == "malformed-support":
            options["background_tracks"][0] = {}
        elif defect == "malformed-acquisition":
            sources[frames[0]] = []
        elif defect == "feedback-value":
            row = next(o for o in bundle.observations if o.observation_id not in frames)
            bundle = replace(
                bundle,
                observations=tuple(
                    replace(o, q_raw_rad=(0.01,)) if o == row else o for o in bundle.observations
                ),
            )
        else:
            frames.reverse()
        with pytest.raises(ValueError):
            await store.check_scene(bundle, sources, frames, scene, **options)
    finally:
        await registry.close_all()


async def test_scene_reads_freeze_inputs_and_settle_after_repeated_cancel(
    tmp_path: Path, monkeypatch
):
    registry, store, bundle, sources, frames, scene, options, _ = await inputs(tmp_path)
    entered, release = threading.Event(), threading.Event()
    read = store.read

    def blocked(identity, *, workcell_id):
        entered.set()
        assert release.wait(5)
        return read(identity, workcell_id=workcell_id)

    monkeypatch.setattr(store, "read", blocked)
    task = asyncio.create_task(store.check_scene(bundle, sources, frames, scene, **options))
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        sources.clear()
        frames.reverse()
        scene["comparison"]["witness_valid"] = True
        options["task_tracks"].clear()
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        closing = asyncio.create_task(registry.close_all())
        await asyncio.sleep(0.01)
        assert not task.done() and not closing.done() and store._pending
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await closing
        assert not store._pending
        with pytest.raises(SkillError, match="shutting down"):
            await store.check_scene(bundle, sources, frames, scene, **options)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if closing is not None:
            await closing
        await registry.close_all()


async def test_scene_cancelled_read_error_is_not_replaced_by_success(tmp_path: Path, monkeypatch):
    registry, store, bundle, sources, frames, scene, options, _ = await inputs(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def failed(identity, *, workcell_id):
        entered.set()
        assert release.wait(5)
        raise ValueError("injected read failure")

    monkeypatch.setattr(store, "read", failed)
    task = asyncio.create_task(
        store.check_scene(bundle, sources, frames, copy.deepcopy(scene), **options)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        release.set()
        with pytest.raises(ValueError, match="injected"):
            await task
        assert not store._pending
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await registry.close_all()


async def test_indexed_optical_replay_uses_private_images_and_redacts_raw_data(
    tmp_path, monkeypatch
):
    registry, store, bundle, sources, frames, scene, options, verifier = await inputs(tmp_path)
    worker = worker_at(tmp_path)
    reads, requests = [], []
    read, submit = store.read, worker.submit

    def counted(identity, *, workcell_id):
        reads.append(identity)
        return read(identity, workcell_id=workcell_id)

    async def submitted(operation, payload, **identity):
        requests.append((operation, digest_of(payload), identity))
        return await submit(operation, payload, **identity)

    monkeypatch.setattr(store, "read", counted)
    monkeypatch.setattr(worker, "submit", submitted)
    try:
        nonces = dict(verifier._seen_nonces)
        replay = await store.replay_scene(
            bundle, sources, frames, worker=worker, run_id="optical-run", **options
        )
        assert isinstance(replay, IndexedSceneReplay)
        assert len(reads) == len(set(reads)) == 2
        assert requests == [
            (
                "track_scene",
                replay.input_digest,
                {
                    "workcell_id": "cell",
                    "run_id": "optical-run",
                    "model_digest": bundle.topology.model.digest,
                },
            )
        ]
        assert digest_of(replay.tracking_result()) == digest_of(scene)
        result = replay.tracking_result()
        result["comparison"]["witness_valid"] = True
        assert replay.tracking_result()["comparison"]["witness_valid"] is False
        report = replay.to_dict()
        assert report["feature_association"] == "owner_worker_optical_flow"
        assert report["physical_surface_identity"] == "not_established"
        assert report["physical_admission"] == "not_established"
        assert report["witness_valid"] is False and report["qualified_consumers"] == []
        assert not any(
            key in json.dumps(report)
            for key in (
                "color_bgr8",
                "depth_m_f32le",
                "signature",
                '"permit":',
                "first_pixel",
            )
        )
        assert verifier._seen_nonces == nonces
        assert registry.calibration_generations.snapshot("cell") is None
        assert not store._pending and not store._computations
        assert worker._process is not None
    finally:
        await worker.close()
        await registry.close_all()


@pytest.mark.parametrize(
    "defect", ["expired", "key", "missing-feedback", "budget", "alias", "frames", "reversed"]
)
async def test_optical_replay_rejects_invalid_sources_before_worker_launch(tmp_path, defect):
    registry, store, bundle, sources, frames, _, options, _ = await inputs(tmp_path)
    worker = worker_at(tmp_path)
    try:
        if defect == "expired":
            store._holder.connection.execute("UPDATE calib_acquisition SET expires_at=0")
        elif defect == "key":
            options["trusted_keys"] = {}
        elif defect == "missing-feedback":
            sources.pop(next(identity for identity in sources if identity not in frames))
        elif defect == "budget":
            options["background_tracks"] *= 100
        elif defect == "alias":
            options["background_tracks"][0]["feature_id"] = options["task_tracks"][0]["feature_id"]
        elif defect == "reversed":
            frames.reverse()
        else:
            frames[:] = ["foreign", frames[0]]
        with pytest.raises(ValueError):
            await store.replay_scene(
                bundle, sources, frames, worker=worker, run_id="replay", **options
            )
        assert worker._process is None
        assert not store._pending and not store._computations
    finally:
        await worker.close()
        await registry.close_all()


async def test_depth_consistent_forgery_does_not_survive_actual_image_replay(tmp_path):
    # Destroy only private image texture. Fabricated correspondences can still
    # satisfy raw-depth geometry, but cannot establish optical-flow association.
    first, second, task, background = scene_pair(common_shift=2)
    erased = replace(second, color_bgr8=bytes(len(second.color_bgr8)))
    registry, store, captures, bundle, sources, _, _, keys = await indexed_pair(
        tmp_path, rois=(first, erased), include_motion=False
    )
    bundle = replace(
        bundle, observations=tuple(o for o in bundle.observations if o.observation_id != "motion")
    )
    frames = [c.roi.frame_id for c in captures]
    config = numerical_policy()
    forged = track_scene(
        captures[0].roi,
        replace(captures[1].roi, color_bgr8=second.color_bgr8),
        task_tracks=task,
        background_tracks=background,
        config=config,
    )
    for group in ("task_tracking", "background_tracking"):
        forged[group]["second_digest"] = captures[1].roi.digest
    forged["comparison"] = compare_scene_supports(
        captures[0].roi,
        forged["task_tracking"],
        forged["background_tracking"],
        task_tracks=task,
        background_tracks=background,
        config=config,
        scene_config=SceneConfig(),
    )
    options = {
        "task_tracks": task,
        "background_tracks": background,
        "tracking": config,
        "scene_config": SceneConfig(),
        "trusted_keys": keys,
        "node_id": "node",
    }
    worker = worker_at(tmp_path)
    try:
        geometry = await store.check_scene(bundle, sources, frames, forged, **options)
        assert geometry.to_dict()["feature_association"] == "not_established"
        with pytest.raises(CalibrationFailure) as error:
            await store.replay_scene(
                bundle, sources, frames, worker=worker, run_id="replay", **options
            )
        assert error.value.code == "insufficient_evidence"
        assert not store._pending and not store._computations
        assert registry.calibration_generations.snapshot("cell") is None
    finally:
        await worker.close()
        await registry.close_all()


@pytest.mark.parametrize("defect", ["envelope", "geometry", "authority"])
async def test_optical_replay_checks_computed_output_independently(tmp_path, monkeypatch, defect):
    registry, store, bundle, sources, frames, _, options, _ = await inputs(tmp_path)
    worker = worker_at(tmp_path)
    submit = worker.submit

    async def changed(operation, payload, **identity):
        result = await submit(operation, payload, **identity)
        if defect == "envelope":
            result["extra"] = True
        elif defect == "geometry":
            row = result["scene"]["background_tracking"]["tracks"][0]
            row["first_m"] = [row["first_m"][0] + 0.001, *row["first_m"][1:]]
        else:
            result["scene"]["comparison"]["witness_valid"] = True
        return result

    monkeypatch.setattr(worker, "submit", changed)
    try:
        with pytest.raises(ValueError):
            await store.replay_scene(
                bundle, sources, frames, worker=worker, run_id="replay", **options
            )
        assert not store._pending and not store._computations
    finally:
        await worker.close()
        await registry.close_all()


async def test_replay_cancellation_settles_reads_without_starting_compute(tmp_path, monkeypatch):
    registry, store, bundle, sources, frames, _, options, _ = await inputs(tmp_path)
    worker = worker_at(tmp_path)
    entered, release = threading.Event(), threading.Event()
    read = store.read

    def blocked(identity, *, workcell_id):
        entered.set()
        assert release.wait(5)
        return read(identity, workcell_id=workcell_id)

    monkeypatch.setattr(store, "read", blocked)
    task = asyncio.create_task(
        store.replay_scene(bundle, sources, frames, worker=worker, run_id="replay", **options)
    )
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        sources.clear()
        frames.reverse()
        options["background_tracks"].clear()
        options["trusted_keys"].clear()
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        closing = asyncio.create_task(registry.close_all())
        await asyncio.sleep(0.01)
        assert not task.done() and not closing.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await closing
        assert worker._process is None and not store._pending and not store._computations
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if closing is not None:
            await closing
        await worker.close()
        await registry.close_all()


@pytest.mark.parametrize("shutdown", [False, True])
async def test_replay_cancel_or_shutdown_reaps_actual_compute(tmp_path, monkeypatch, shutdown):
    registry, store, bundle, sources, frames, _, options, _ = await inputs(tmp_path)
    worker = worker_at(tmp_path)
    entered = asyncio.Event()

    async def withheld(*args):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(worker, "_response", withheld)
    task = asyncio.create_task(
        store.replay_scene(bundle, sources, frames, worker=worker, run_id="replay", **options)
    )
    try:
        await asyncio.wait_for(entered.wait(), 10)
        process = worker._process
        assert process is not None and process.returncode is None
        if shutdown:
            await registry.close_all()
        else:
            task.cancel()
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.returncode is not None and worker._process is None
        assert worker._pending == 0 and not store._pending and not store._computations
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await worker.close()
        await registry.close_all()


async def test_replay_freezes_nested_sources_supports_and_keys_before_suspension(
    tmp_path, monkeypatch
):
    registry, store, bundle, sources, frames, scene, options, _ = await inputs(tmp_path)
    worker = worker_at(tmp_path)
    entered, release = threading.Event(), threading.Event()
    read = store.read

    def blocked(identity, *, workcell_id):
        entered.set()
        assert release.wait(5)
        return read(identity, workcell_id=workcell_id)

    monkeypatch.setattr(store, "read", blocked)
    task = asyncio.create_task(
        store.replay_scene(bundle, sources, frames, worker=worker, run_id="frozen", **options)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        sources.clear()
        frames.reverse()
        options["background_tracks"][0]["pixel"][0] += 20
        options["task_tracks"].clear()
        options["trusted_keys"].clear()
        release.set()
        replay = await task
        assert digest_of(replay.tracking_result()) == digest_of(scene)
        assert not store._pending and not store._computations
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await worker.close()
        await registry.close_all()


async def test_replay_discards_late_result_even_if_compute_adapter_swallows_cancel(
    tmp_path, monkeypatch
):
    registry, store, bundle, sources, frames, _, options, _ = await inputs(tmp_path)
    worker = worker_at(tmp_path)
    completed = asyncio.Event()
    submit = worker.submit

    async def late(operation, payload, **identity):
        result = await submit(operation, payload, **identity)
        completed.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return result

    monkeypatch.setattr(worker, "submit", late)
    task = asyncio.create_task(
        store.replay_scene(bundle, sources, frames, worker=worker, run_id="late", **options)
    )
    try:
        await asyncio.wait_for(completed.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert registry.calibration_generations.snapshot("cell") is None
        assert not store._pending and not store._computations
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await worker.close()
        await registry.close_all()
