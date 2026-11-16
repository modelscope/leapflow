# Copyright (c) Alibaba, Inc. and its affiliates.
"""Indexed derived-value reconstruction with simulated measurements, not metrology."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import struct
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from leapcalib.contracts.identity import DeviceIdentity, MountIdentity, TopologyIdentity, digest_of
from leapcalib.contracts.motion import (
    MODE_FIXED,
    MODE_WRIST,
    MotionObservation,
    PointCorrespondence,
)
from leapcalib.contracts.observations import (
    JointObservation,
    PixelDepthSample,
    RgbdObservation,
    Stamp,
)
from leapcalib.contracts.tracking import SourceStamp, TrackingConfig
from leapcalib.estimation.tracking import track_metric
from leapcalib.geometry.se3 import Transform
from leapcalib.validation.bundle import EvidenceBundle, Retention
from leapcalib.validation.evaluation import MotionMeasurementBounds
from leapflow.hardware.calibration.adapter import model_identity
from leapflow.hardware.calibration.evidence import AcquisitionBinding
from leapflow.hardware.calibration.factors import IndexedMotionFactors, MotionFactorCheck
from leapflow.hardware.skills.geometry import Pose
from leapflow.hardware.skills.kinematics import DhChain, DhLink
from tests.test_leapcalib_acquisition import profile
from tests.test_leapcalib_evidence import registry_and_cache, sealed_plan
from tests.test_leapcalib_provenance import admitted
from tests.test_leapcalib_tracking import pair, settled_fixture

pytestmark = pytest.mark.component


async def indexed_pair(tmp_path: Path, *, mode: str = MODE_WRIST, rois=None, include_motion=True):
    """Actual private store/holder and gate; camera/feedback values are synthetic."""
    registry, _, store = registry_and_cache(tmp_path)
    model = model_identity(DhChain((DhLink(100, 0, 0, 0),)), model_id="arm")
    role = "fixed" if mode == MODE_FIXED else "wrist"
    mount = MountIdentity(role, "base" if role == "fixed" else "flange")
    first, second = pair() if rois is None else rois
    captures = []
    for i, roi in enumerate((first, second)):
        roi = replace(
            roi,
            profile=replace(roi.profile, role=role),
            mount_digest=mount.digest,
            stamp=Stamp(1 + i, uncertainty_s=0.001),
            provenance="metric_rgbd",
            source_stamps=(
                SourceStamp("color", 20 + i, 1000 + 1000 * i, "hardware"),
                SourceStamp("depth", 21 + i, 1000.1 + 1000 * i, "hardware"),
            ),
        )
        value = settled_fixture(roi, i * 0.01)
        captures.append(
            replace(
                value,
                feedback=tuple(
                    replace(
                        f,
                        device_id="arm",
                        model_digest=model.digest,
                        tcp=Pose(100 + 2 * i, i, 0, 0, 0, 10 + i),
                    )
                    for f in value.feedback
                ),
                profile_digest=digest_of(vars(profile())),
            )
        )
    first, second = captures
    anchor, roi = first.feedback[0], first.roi
    basis = {
        "collection": {
            "model_digest": model.digest,
            "declaration_digest": anchor.declaration_digest,
            "joint_ids": list(anchor.joint_ids),
            "node_session_id": anchor.node_session_id,
            "camera_device_id": roi.profile.device_id,
            "mount_digest": mount.digest,
            "camera_session_id": roi.session_id,
            "source_profile_digest": roi.source_profile_digest,
            "metric_profile_digest": roi.profile.digest,
            "region": [roi.x, roi.y, roi.width, roi.height],
            "profile": {"settled": vars(profile())},
        }
    }
    plan = sealed_plan(basis=basis)
    receipt, verifier, keys = admitted(plan)
    binding = AcquisitionBinding("cell", "run", "node-boot", json.dumps(plan.to_dict()), receipt)
    sources: dict[str, str] = {}
    primitives = []
    for capture in captures:
        summary = await asyncio.to_thread(
            store.record,
            capture,
            profile=profile(),
            binding=binding,
            ttl_s=60,
            quota_bytes=4 * 1024 * 1024,
        )
        roi = capture.roi
        stamp = replace(
            roi.stamp, uncertainty_s=roi.stamp.uncertainty_s + roi.exposure_uncertainty_s
        )
        row = RgbdObservation(
            roi.frame_id,
            roi.profile,
            stamp,
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
        primitives.extend((row, *joints))
        sources.update(
            dict.fromkeys(
                (row.observation_id, *(j.observation_id for j in joints)),
                summary["evidence_digest"],
            )
        )
    if not include_motion:
        topology = TopologyIdentity(
            "cell", DeviceIdentity("arm"), model, (first.roi.profile,), (mount,)
        )
        bundle = EvidenceBundle(
            "indexed-primitive-software",
            topology,
            tuple(primitives),
            1700000000,
            Retention(60),
            metadata={"node_session_id": "node-boot"},
        )
        return registry, store, captures, bundle, sources, {}, verifier, keys
    tracking = track_metric(first.roi, second.roi)
    # Independent fixture formulation uses the measured flange poses, not commands.
    a = Transform(first.feedback[-2].tcp.rotation, (0.1, 0, 0))
    b = Transform(second.feedback[-2].tcp.rotation, (0.102, 0.001, 0))
    relative = b.compose(a.inverse()) if mode == MODE_FIXED else a.inverse().compose(b)
    factor = MotionObservation(
        "motion",
        first.roi.frame_id,
        second.roi.frame_id,
        first.roi.profile,
        model,
        primitives[0].stamp,
        primitives[5].stamp,
        "metric_rgbd",
        mode,
        "unwitnessed-scene",
        relative,
        tuple(
            PointCorrespondence(row["feature_id"], row["first_m"], row["second_m"])
            for row in tracking["tracks"]
        ),
        True,
        0.1,
    )
    topology = TopologyIdentity(
        "cell", DeviceIdentity("arm"), model, (first.roi.profile,), (mount,)
    )
    bundle = EvidenceBundle(
        "indexed-motion-software",
        topology,
        (*primitives, factor),
        1700000000,
        Retention(60),
        accepted_ids=(factor.observation_id,),
        metadata={"node_session_id": "node-boot"},
    )
    return registry, store, captures, bundle, sources, {"motion": tracking}, verifier, keys


@pytest.mark.parametrize("mode", [MODE_FIXED, MODE_WRIST])
async def test_indexed_motion_reconstructs_metric_points_and_nominal_direction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    registry, store, captures, bundle, sources, results, verifier, keys = await indexed_pair(
        tmp_path, mode=mode
    )
    calls = []
    read = store.read

    def counted(identity, *, workcell_id):
        calls.append(identity)
        return read(identity, workcell_id=workcell_id)

    monkeypatch.setattr(store, "read", counted)
    try:
        nonces = dict(verifier._seen_nonces)
        checked = await store.motion_factors(
            bundle, sources, results, config=TrackingConfig(), trusted_keys=keys, node_id="node"
        )
        assert sorted(calls) == sorted(set(sources.values())) and len(calls) == 2
        assert checked.manifest.evidence_digest == bundle.digest
        assert checked.checks[0].observation_digest == bundle.observations[-1].digest
        assert checked.checks[0].pose_source_ids == tuple(
            c.feedback[-2].source_id for c in captures
        )
        assert checked.checks[0].robot_translation_uncertainty_m >= 0.00002
        assert checked.checks[0].robot_rotation_uncertainty_rad == pytest.approx(math.radians(0.01))
        data = checked.to_dict()
        assert (
            data["scope"] == "conditional_camera_motion"
            and data["robot_pose_semantics"] == "nominal_firmware_flange"
        )
        for field in (
            "feature_association",
            "independent_witness",
            "rigid_link_identity",
            "interval_peak_rates",
            "physical_admission",
            "global_geometry",
        ):
            assert data[field] == "not_established"
        assert "signature" not in json.dumps(data) and "color_bgr8" not in json.dumps(data)
        assert dict(verifier._seen_nonces) == nonces
        data["factors"][0]["observation_id"] = "changed"
        assert checked.to_dict()["factors"][0]["observation_id"] == "motion"
        repeated = await store.motion_factors(
            EvidenceBundle.from_mapping(bundle.to_dict()),
            sources,
            results,
            config=TrackingConfig(),
            trusted_keys=keys,
            node_id="node",
        )
        assert checked.digest == repeated.digest
    finally:
        await registry.close_all()


@pytest.mark.parametrize(
    "defect",
    [
        "commanded-pose",
        "inverse-direction",
        "points",
        "feature-alias",
        "settled",
        "stamp",
        "missing-result",
        "extra-result",
        "witness",
        "worker-point",
        "model-source",
    ],
)
async def test_matching_raw_lineage_cannot_launder_derived_motion(
    tmp_path: Path, defect: str
) -> None:
    registry, store, _, bundle, sources, results, _, keys = await indexed_pair(tmp_path)
    factor = bundle.observations[-1]
    try:
        if defect == "commanded-pose":
            factor = replace(factor, robot_motion=Transform())
        elif defect == "inverse-direction":
            factor = replace(factor, robot_motion=factor.robot_motion.inverse())
        elif defect == "points":
            point = factor.points[0]
            factor = replace(
                factor,
                points=(
                    replace(point, first_m=(point.first_m[0] + 0.0001, *point.first_m[1:])),
                    *factor.points[1:],
                ),
            )
        elif defect == "feature-alias":
            factor = replace(
                factor, points=(replace(factor.points[0], feature_id="alias"), *factor.points[1:])
            )
        elif defect == "settled":
            factor = replace(factor, settled=False)
        elif defect == "stamp":
            factor = replace(factor, stamp=replace(factor.stamp, uncertainty_s=0.001))
        elif defect == "missing-result":
            results = {}
        elif defect == "extra-result":
            results["extra"] = results["motion"]
        elif defect == "witness":
            results["motion"]["witness_valid"] = True
        elif defect == "worker-point":
            results = copy.deepcopy(results)
            results["motion"]["tracks"][0]["first_m"] = (0, 0, 1)
        else:
            joints = bundle.observations[1]
            bundle = replace(
                bundle,
                observations=(
                    bundle.observations[0],
                    replace(joints, q_raw_rad=(0.001,)),
                    *bundle.observations[2:],
                ),
            )
        with pytest.raises(ValueError):
            bundle = replace(bundle, observations=(*bundle.observations[:-1], factor))
            await store.motion_factors(
                bundle, sources, results, config=TrackingConfig(), trusted_keys=keys, node_id="node"
            )
    finally:
        await registry.close_all()


async def test_rejected_and_unused_motion_rows_are_recomputed(tmp_path: Path) -> None:
    registry, store, _, bundle, sources, results, _, keys = await indexed_pair(tmp_path)
    try:
        extra = replace(bundle.observations[-1], observation_id="unused", robot_motion=Transform())
        bundle = replace(bundle, observations=(*bundle.observations, extra))
        results["unused"] = results["motion"]
        with pytest.raises(ValueError, match="nominal flange feedback"):
            await store.motion_factors(
                bundle, sources, results, config=TrackingConfig(), trusted_keys=keys, node_id="node"
            )
        bundle = replace(bundle, rejected_ids=("unused",))
        with pytest.raises(ValueError, match="nominal flange feedback"):
            await store.motion_factors(
                bundle, sources, results, config=TrackingConfig(), trusted_keys=keys, node_id="node"
            )
    finally:
        await registry.close_all()


async def test_indexed_factor_floors_cannot_be_understated(tmp_path: Path) -> None:
    registry, store, _, bundle, sources, results, _, keys = await indexed_pair(tmp_path)
    try:
        checked = await store.motion_factors(
            bundle, sources, results, config=TrackingConfig(), trusted_keys=keys, node_id="node"
        )
        factor = checked.checks[0]
        bound = MotionMeasurementBounds(
            factor.observation_digest,
            factor.point_uncertainty_m,
            factor.robot_translation_uncertainty_m,
            factor.robot_rotation_uncertainty_rad,
            0,
            0,
        )
        checked.validate_measurement_bounds({"motion": bound})
        # Even valid floors do not establish the unobserved peak rates or witnesses.
        assert checked.to_dict()["interval_peak_rates"] == "not_established"
        for field in (
            "point_uncertainty_m",
            "robot_translation_uncertainty_m",
            "robot_rotation_uncertainty_rad",
        ):
            with pytest.raises(ValueError, match="understate"):
                checked.validate_measurement_bounds(
                    {"motion": replace(bound, **{field: getattr(bound, field) / 2})}
                )
        with pytest.raises(ValueError, match="exact checked"):
            checked.validate_measurement_bounds(
                {"motion": replace(bound, observation_digest="different")}
            )
        with pytest.raises(ValueError, match="unchecked"):
            checked.validate_measurement_bounds({"other": bound})
    finally:
        await registry.close_all()


@pytest.mark.parametrize("fail", [False, True])
async def test_factor_read_repeated_cancel_and_shutdown_settle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    registry, store, _, bundle, sources, results, _, keys = await indexed_pair(tmp_path)
    entered, release = threading.Event(), threading.Event()
    read = store.read

    def blocked(identity, *, workcell_id):
        entered.set()
        assert release.wait(5)
        if fail:
            raise ValueError("injected indexed read failure")
        return read(identity, workcell_id=workcell_id)

    monkeypatch.setattr(store, "read", blocked)
    task = asyncio.create_task(
        store.motion_factors(
            bundle, sources, results, config=TrackingConfig(), trusted_keys=keys, node_id="node"
        )
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
        if fail:
            with pytest.raises(ValueError, match="indexed read failure"):
                await task
        else:
            with pytest.raises(asyncio.CancelledError):
                await task
        await closing
        assert not store._pending
    finally:
        release.set()
        await asyncio.gather(task, *([closing] if closing else []), return_exceptions=True)
        await registry.close_all()


def test_factor_check_contract_floors_and_typed_sources() -> None:
    with pytest.raises(ValueError, match="zero"):
        MotionFactorCheck("factor", "digest", "tracking", ("a", "b"), 0, 0.001, 0.001)
    with pytest.raises(ValueError, match="typed indexed"):
        IndexedMotionFactors(None, TrackingConfig(), ())
