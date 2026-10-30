# Copyright (c) Alibaba, Inc. and its affiliates.
"""Camera shadow pipeline wiring with private indexed captures and real isolated compute.

Rendered scenes and nominal flange feedback are simulated, never physical qualification.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from leapcalib.geometry.se3 import Transform, rpy_rad_from_rotation
from leapcalib.runtime.protocol import WorkerBudgets
from leapflow.hardware.calibration.acquisition import CollectionProfile
from leapflow.hardware.calibration.evidence import AcquisitionBinding
from leapflow.hardware.calibration.trial import TrialPolicy, run_motion_trial
from leapflow.hardware.calibration.worker import CalibrationWorker
from leapflow.hardware.skills.admission import RunAdmission
from leapflow.hardware.skills.builtin.governed_calibration import AssessMotion, CollectPoses
from leapflow.hardware.skills.geometry import Pose
from tests.test_leapcalib_acquisition import profile
from tests.test_leapcalib_assessment import projected_scene, record_simulated_capture
from tests.test_leapcalib_factors import indexed_pair
from tests import test_leapcalib_collection as collection_tests
from tests.test_leapcalib_collection import (
    planned,
    request,
    permit,
    finished,
)  # noqa: F401
from tests._harness.skill_stubs import ARM_ID, WORKCELL_ID

pytestmark = pytest.mark.component
stationary_owner = collection_tests.owner


def collection_profile() -> CollectionProfile:
    return CollectionProfile(
        "camera.wrist", "wrist", ("can",), profile(), 120, 32 * 1024 * 1024, 128
    )


async def rendered_campaign(tmp_path: Path, count: int = 26):
    registry, store, templates, original, sources, _, verifier, keys = await indexed_pair(tmp_path)
    content = await store.read_capture(sources[templates[0].roi.frame_id], workcell_id="cell")
    data, proof = content["binding"], content["binding"]["admission"]
    binding = AcquisitionBinding(
        "cell",
        data["run_id"],
        data["node_session_id"],
        json.dumps(data["plan"]),
        RunAdmission(proof["mode"], proof["action_digest"], json.dumps(proof["permit"])),
    )
    pairs = []
    before = Transform(templates[0].feedback[-2].tcp.rotation, (0.1, 0, 0))
    for i in range(count):
        angles = [0.0, 0.0, 0.0]
        angles[i % 3] = 0.016 + i * 0.0001
        rotation = Transform.from_rpy_rad(*angles)
        center = np.asarray((0, 0, 0.5))
        translation = (
            center - np.asarray(rotation.rotation) @ center + np.asarray((0.0004, -0.0002, 0))
        )
        motion = replace(rotation, translation_m=tuple(translation))
        after = before.compose(motion.inverse())
        pair = []
        for j, template in enumerate(templates):
            roi = templates[0].roi if j == 0 else projected_scene(templates[0].roi, motion)
            stamp = 1 + 3 * i + j
            roi = replace(
                roi,
                frame_id=f"trial-render:{i}:{j}",
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
                    source_id=f"trial-render:{i}:{j}:{k}",
                    tcp=pose,
                    stamp=replace(f.stamp, seconds=f.stamp.seconds + shift),
                    first_receipt_s=f.first_receipt_s + shift,
                    last_receipt_s=f.last_receipt_s + shift,
                )
                for k, f in enumerate(template.feedback)
            )
            capture = replace(template, roi=roi, feedback=feedback)
            rows, indexed = await record_simulated_capture(
                store, capture, binding, original.topology.model
            )
            pair.append(indexed[rows[0].observation_id])
        pairs.append(pair)
    args = {
        "pairs": pairs,
        "training": min(6, count - 1),
        "mount": original.topology.mounts[0].to_dict(),
        "prior": Transform(translation_m=(0.02, 0, 0), parent="flange", child="camera").to_dict(),
        "tracking": {"max_tracks": 32},
        "solve": {"trials": 4, "translation_sigma_m": 0.0005, "rotation_sigma_rad": 0.002},
    }
    return registry, store, original.topology.model, verifier, keys, args


async def test_trial_real_worker_disjoint_evaluation_no_activation(tmp_path: Path) -> None:
    registry, store, model, verifier, keys, args = await rendered_campaign(tmp_path)
    worker = CalibrationWorker(
        tmp_path.resolve().parent / ("lc-" + uuid.uuid4().hex[:8]) / "s", enabled=True
    )
    registry.bind_calibration_worker(worker)
    registry.bind_approval_keys(keys, node_id="node")
    nonces = dict(verifier._seen_nonces)
    operations = []
    original = worker.submit

    async def submit(operation, payload, **kwargs):
        operations.append(operation)
        if operation == "solve_motion":
            content = payload["evidence"]["content"]
            assert len(content["observations"]) == 26 * 5
            assert len(content["accepted_ids"]) == 6
            assert len(content["holdout_ids"]) == 20
        return await original(operation, payload, **kwargs)

    worker.submit = submit
    try:
        report = await run_motion_trial(
            store,
            worker,
            policy=TrialPolicy.from_mapping(args, worker.budgets),
            profile=collection_profile(),
            model=model,
            workcell_id="cell",
            run_id="trial",
            node_session_id="node-boot",
            approval_keys=registry.approval_keys,
            blocked=lambda: False,
        )
        assert operations == ["track_metric"] * 26 + ["solve_motion"]
        assert report["evaluation"]["verdict"] == "rejected", report["evaluation"]
        assert report[
            "supported_blocks"
        ]  # Conservative declared rates can still reject a fitted candidate.
        assert len(report["indexed_holdout_ids"]) == 20
        assert report["activation"] == "none" and report["qualified_consumers"] == []
        assert report["physical_admission"] == "not_established" and not report["raw_exported"]
        assert report["unmet_physical_gates"] and not report["tracking_refusals"]
        assert verifier._seen_nonces == nonces
        assert "color_bgr8" not in json.dumps(report) and "depth_m_f32le" not in json.dumps(report)
    finally:
        await registry.close_all()


async def test_tracking_exclusion_never_promotes_withheld_to_training(tmp_path: Path) -> None:
    registry, store, model, _, keys, args = await rendered_campaign(tmp_path, count=3)
    worker = CalibrationWorker(
        tmp_path.resolve().parent / ("lc-" + uuid.uuid4().hex[:8]) / "s", enabled=True
    )
    registry.bind_calibration_worker(worker)
    args["training"] = 1
    original = worker.submit

    async def submit(operation, payload, **kwargs):
        if operation == "track_metric" and payload["first"]["frame_id"] == "trial-render:0:0":
            return {"tracking": {}}
        return await original(operation, payload, **kwargs)

    worker.submit = submit
    try:
        report = await run_motion_trial(
            store,
            worker,
            policy=TrialPolicy.from_mapping(args, worker.budgets),
            profile=collection_profile(),
            model=model,
            workcell_id="cell",
            run_id="trial",
            node_session_id="node-boot",
            approval_keys=(keys, "node"),
            blocked=lambda: False,
        )
        assert report["split"]["accepted_ids"] == []
        assert report["split"]["holdout_ids"] == ["motion:001", "motion:002"]
        assert report["tracking_refusals"][0]["split"] == "training"
        assert report["evaluation"]["verdict"] == "insufficient_evidence"
    finally:
        await registry.close_all()


async def test_trial_leased_read_through_real_owner_lhp_port(
    stationary_owner, monkeypatch: pytest.MonkeyPatch
) -> None:  # noqa: F811
    from leapcalib.contracts.identity import MountIdentity

    registry, _, sensor, runtime, server, issuer = stationary_owner
    mount = MountIdentity("fixed", "base")
    original_capture = sensor.capture_rgbd

    async def capture(**kwargs):
        frame = await original_capture(**kwargs)
        return replace(frame, metadata={**frame.metadata, "mount_digest": mount.digest})

    monkeypatch.setattr(await registry.transport("camera"), "capture_rgbd", capture)
    worker = registry.calibration_worker
    worker.enabled = True
    registry.bind_approval_keys({issuer.key_id: issuer._private.public_key()}, node_id="node")
    pairs = []
    for _ in range(2):
        collected = await planned(server, frames=2)
        run = await finished(
            runtime,
            await request(
                server,
                "skill.execute",
                plan_digest=collected["plan"]["digest"],
                permit=permit(issuer, collected),
                lease_s=10,
            ),
        )
        assert run.outcome.ok
        pairs.append([c["evidence_digest"] for c in run.outcome.result["captures"]])
    args = {
        "pairs": pairs,
        "training": 1,
        "mount": mount.to_dict(),
        "prior": Transform(parent="base", child="camera").to_dict(),
    }
    trial = await request(
        server, "skill.plan", workcell=WORKCELL_ID, skill="calib.assess_motion", args=args
    )
    assert "plan" in trial, trial
    assert trial["consent"] == "each_run"
    transport = await registry.transport(ARM_ID)
    before = list(transport._write_log)
    run = await finished(
        runtime,
        await request(
            server,
            "skill.execute",
            plan_digest=trial["plan"]["digest"],
            permit=permit(issuer, trial),
            lease_s=10,
        ),
    )
    assert run.outcome.ok, run.outcome
    assert run.outcome.side_effect_state == "none" and run.actuators == ()
    assert run.outcome.result["evaluation"]["verdict"] == "insufficient_evidence"
    assert transport._write_log == before and not runtime.holds_device(ARM_ID)


async def test_public_key_exposure_is_read_only_and_cannot_rebind_owner(tmp_path: Path) -> None:
    registry, _, _, _, _, _, verifier, keys = await indexed_pair(tmp_path)
    try:
        assert verifier.node_id == "node"
        public = verifier.trusted_public_keys
        with pytest.raises(TypeError):
            public["other"] = next(iter(keys.values()))
        registry.bind_approval_keys(public, node_id=verifier.node_id)
        registry.bind_approval_keys(keys, node_id="node")
        with pytest.raises(TypeError):
            registry.approval_keys[0]["other"] = next(iter(keys.values()))
        with pytest.raises(ValueError, match="competing"):
            registry.bind_approval_keys(keys, node_id="other-node")
        assert dict(verifier.trusted_public_keys) == keys
    finally:
        await registry.close_all()


def test_skill_authority_and_trial_reused_source_refusal() -> None:
    assert AssessMotion.spec.runs_leased and not AssessMotion.spec.uses_motion
    assert AssessMotion.spec.effect == "read" and AssessMotion.spec.consent == "each_run"
    assert CollectPoses.spec.uses_motion and CollectPoses.spec.consent == "each_run"
    with pytest.raises(ValueError, match="source reuse"):
        TrialPolicy.from_mapping(
            {"pairs": [["a" * 64, "b" * 64], ["b" * 64, "c" * 64]], "training": 1}, WorkerBudgets()
        )
