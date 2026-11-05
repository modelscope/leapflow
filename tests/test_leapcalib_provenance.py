# Copyright (c) Alibaba, Inc. and its affiliates.
"""Indexed raw lineage and historical signatures; all acquisitions are synthetic fixtures."""

from __future__ import annotations

import asyncio
import json
import struct
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from leapcalib.contracts.identity import DeviceIdentity, MountIdentity, TopologyIdentity
from leapcalib.contracts.motion import MotionObservation
from leapcalib.contracts.observations import JointObservation, PixelDepthSample, RgbdObservation
from leapcalib.validation.bundle import EvidenceBundle, Retention
from leapflow.hardware.calibration.adapter import model_identity
from leapflow.hardware.calibration.evidence import AcquisitionBinding
from leapflow.hardware.calibration.provenance import (
    IndexedSourceManifest,
    approval_provenance,
    independent_indexed_holdouts,
)
from leapflow.hardware.lhp.authority import WriteAuthority
from leapflow.hardware.lhp.permit import (
    ExecutionPermit,
    PermitIssuer,
    PermitVerifier,
    generate_keypair,
)
from leapflow.hardware.skills.digest import run_permit_digest
from leapflow.hardware.skills.kinematics import DhChain, DhLink
from tests.test_leapcalib_acquisition import profile
from tests.test_leapcalib_evidence import binding, captured, registry_and_cache, sealed_plan
from tests.test_leapcalib_evaluation import motion_study

pytestmark = pytest.mark.component


def admitted(plan):
    private, public = generate_keypair()
    issuer = PermitIssuer(private)
    verifier = PermitVerifier({issuer.key_id: public}, node_id="node")
    digest = run_permit_digest(
        workcell=plan.workcell,
        skill_id=plan.skill_id,
        version=plan.version,
        plan_digest=plan.digest,
        limits=plan.limits,
    )
    permit = issuer.issue(
        audience="node",
        session_id="client",
        workspace="workspace",
        approval_id="software-human-fixture",
        digest=digest,
    )
    refusal, receipt = WriteAuthority("permit_required", verifier).admit_run(
        {"permit": permit.to_dict()},
        expected_digest=digest,
    )
    assert refusal is None
    return receipt, verifier, {issuer.key_id: public}


async def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    value, _ = await captured(monkeypatch)
    mount = MountIdentity("wrist", "flange")
    value = replace(value, roi=replace(value.roi, mount_digest=mount.digest))
    first, roi = value.feedback[0], value.roi
    basis = {
        "collection": {
            "model_digest": first.model_digest,
            "declaration_digest": first.declaration_digest,
            "joint_ids": list(first.joint_ids),
            "node_session_id": first.node_session_id,
            "camera_device_id": roi.profile.device_id,
            "mount_digest": roi.mount_digest,
            "camera_session_id": roi.session_id,
            "source_profile_digest": roi.source_profile_digest,
            "metric_profile_digest": roi.profile.digest,
            "region": [roi.x, roi.y, roi.width, roi.height],
            "profile": {"settled": vars(profile())},
        }
    }
    plan = sealed_plan(basis=basis)
    receipt, verifier, keys = admitted(plan)
    lineage = AcquisitionBinding("cell", "run", "node-boot", json.dumps(plan.to_dict()), receipt)
    registry, cache, store = registry_and_cache(tmp_path)
    summary = await asyncio.to_thread(
        store.record,
        value,
        profile=profile(),
        binding=lineage,
        ttl_s=60,
        quota_bytes=4 * 1024 * 1024,
    )
    model = model_identity(DhChain((DhLink(100, 0, 0, 0),)), model_id="arm")
    topology = TopologyIdentity("cell", DeviceIdentity("arm"), model, (roi.profile,), (mount,))
    stamp = replace(roi.stamp, uncertainty_s=roi.stamp.uncertainty_s + roi.exposure_uncertainty_s)
    depth = struct.unpack_from("<f", roi.depth_m_f32le, 0)[0]
    frame = RgbdObservation(
        roi.frame_id,
        roi.profile,
        stamp,
        "metric_rgbd",
        (PixelDepthSample(float(roi.x), float(roi.y), depth),),
    )
    joints = tuple(
        JointObservation(f.source_id, model, f.stamp, f.joints_rad, still=True)
        for f in value.feedback
    )
    bundle = EvidenceBundle(
        "indexed-software",
        topology,
        (frame, *joints),
        1700000000,
        Retention(60),
        metadata={"node_session_id": "node-boot"},
    )
    sources = dict.fromkeys(
        (o.observation_id for o in bundle.observations), summary["evidence_digest"]
    )
    return registry, cache, store, value, bundle, sources, lineage, verifier, keys


async def test_exact_source_values_approval_and_acquisition_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, _, store, value, bundle, sources, _, verifier, keys = await fixture(
        tmp_path, monkeypatch
    )
    try:
        nonces = dict(verifier._seen_nonces)
        result = await store.source_manifest(bundle, sources, trusted_keys=keys, node_id="node")
        data = result.to_dict()
        assert result.evidence_digest == bundle.digest
        assert result.topology_digest == bundle.topology.digest
        assert len(data["sources"]) == len(bundle.observations)
        assert data["physical_admission"] == data["feature_association"] == "not_established"
        assert all(
            row["acquisition_window_s"]
            == [
                value.feedback[0].first_receipt_s - value.feedback[0].stamp.uncertainty_s,
                value.feedback[-1].last_receipt_s + value.feedback[-1].stamp.uncertainty_s,
            ]
            for row in data["sources"]
        )
        assert "signature" not in json.dumps(data) and "software-human-fixture" not in json.dumps(
            data
        )
        assert dict(verifier._seen_nonces) == nonces
        data["sources"][0]["source_id"] = "changed"
        assert result.to_dict()["sources"][0]["source_id"] != "changed"
        repeated = await store.source_manifest(
            bundle, dict(reversed(tuple(sources.items()))), trusted_keys=keys, node_id="node"
        )
        assert repeated == result and repeated.digest == result.digest
    finally:
        await registry.close_all()


@pytest.mark.parametrize(
    "defect",
    [
        "depth",
        "pixel",
        "frame_alias",
        "stamp",
        "profile",
        "provenance",
        "joints",
        "feedback_alias",
        "joint_stamp",
        "moving",
        "session",
        "mount",
        "device",
        "missing",
        "extra",
    ],
)
async def test_source_and_topology_laundering_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    registry, _, store, _, bundle, sources, _, _, keys = await fixture(tmp_path, monkeypatch)
    try:
        frame, *joints = bundle.observations
        old_frame, old_joint = frame.observation_id, joints[0].observation_id
        topology, metadata = bundle.topology, dict(bundle.metadata)
        if defect == "depth":
            frame = replace(frame, samples=(replace(frame.samples[0], depth_m=0.5),))
        elif defect == "pixel":
            frame = replace(frame, samples=(replace(frame.samples[0], u_px=2.5),))
        elif defect == "frame_alias":
            frame = replace(frame, observation_id="alias")
        elif defect == "stamp":
            frame = replace(frame, stamp=replace(frame.stamp, uncertainty_s=0.001))
        elif defect == "profile":
            frame = replace(frame, stream=replace(frame.stream, fx_px=300))
            topology = replace(topology, streams=(frame.stream,))
        elif defect == "provenance":
            frame = replace(frame, provenance="synthetic")
        elif defect == "joints":
            joints[0] = replace(joints[0], q_raw_rad=(0.001,))
        elif defect == "feedback_alias":
            joints[0] = replace(joints[0], observation_id="alias")
        elif defect == "joint_stamp":
            joints[0] = replace(joints[0], stamp=replace(joints[0].stamp, uncertainty_s=0.0001))
        elif defect == "moving":
            joints[0] = replace(joints[0], still=False)
        elif defect == "session":
            metadata["node_session_id"] = "different-node"
        elif defect == "mount":
            topology = replace(topology, mounts=(MountIdentity("other", "flange"),))
        elif defect == "device":
            topology = replace(topology, device=DeviceIdentity("other"))
        elif defect == "missing":
            sources.pop(old_joint)
        else:
            sources["extra"] = next(iter(sources.values()))
        if old_frame != frame.observation_id:
            sources[frame.observation_id] = sources.pop(old_frame)
        if old_joint != joints[0].observation_id:
            sources[joints[0].observation_id] = sources.pop(old_joint)
        altered = replace(
            bundle, topology=topology, observations=(frame, *joints), metadata=metadata
        )
        with pytest.raises(ValueError):
            await store.source_manifest(altered, sources, trusted_keys=keys, node_id="node")
    finally:
        await registry.close_all()


async def test_expired_raw_and_unsigned_legacy_never_qualify_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, cache, store, value, bundle, sources, _, _, keys = await fixture(
        tmp_path, monkeypatch
    )
    try:
        entry = cache.list_entries(category="calibration_acquisition")[0]
        monkeypatch.setattr(
            "leapflow.hardware.calibration.evidence.time.time", lambda: entry.expires_at + 1
        )
        with pytest.raises(ValueError, match="expired"):
            await store.source_manifest(bundle, sources, trusted_keys=keys, node_id="node")
        assert store.summary(next(iter(sources.values())), workcell_id="cell")
        with pytest.raises(ValueError, match="legacy"):
            approval_provenance(binding().admission.to_dict(), trusted_keys=keys, node_id="node")
        assert value.digest
    finally:
        await registry.close_all()


async def test_historical_signature_remains_evidence_after_expiry_without_nonce_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, _, _, _, _, _, lineage, verifier, keys = await fixture(tmp_path, monkeypatch)
    try:
        receipt = lineage.admission.to_dict()
        monkeypatch.setattr(
            "leapflow.hardware.lhp.permit.time.time", lambda: receipt["permit"]["expires_at"] + 100
        )
        nonces = dict(verifier._seen_nonces)
        proof = approval_provenance(receipt, trusted_keys=keys, node_id="node")
        assert proof["action_digest"] == receipt["action_digest"]
        assert dict(verifier._seen_nonces) == nonces
        assert (
            verifier.verify(
                ExecutionPermit(**receipt["permit"]), expected_digest=receipt["action_digest"]
            ).code
            == "permit_expired"
        )
    finally:
        await registry.close_all()


@pytest.mark.parametrize(
    "field,bad",
    [
        ("signature", "00" * 64),
        ("audience", "other"),
        ("action_digest", "other"),
        ("session_id", "other"),
        ("workspace", "other"),
        ("v", True),
        ("max_uses", 1.5),
        ("issued_at", "1700000000"),
        ("expires_at", float("nan")),
        ("key_id", "untrusted"),
        ("approval_id", False),
    ],
)
async def test_tampered_or_lossy_historical_receipts_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, bad: Any
) -> None:
    registry, _, _, _, _, _, lineage, _, keys = await fixture(tmp_path, monkeypatch)
    try:
        receipt = lineage.admission.to_dict()
        receipt["permit"][field] = bad
        with pytest.raises(ValueError):
            approval_provenance(receipt, trusted_keys=keys, node_id="node")
    finally:
        await registry.close_all()


@pytest.mark.parametrize("failure", [False, True])
async def test_cancel_and_shutdown_wait_for_indexed_read_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    registry, _, store, _, bundle, sources, _, _, keys = await fixture(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = store.read

    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        if failure:
            raise ValueError("isolated invalid raw evidence")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "read", slow)
    task = asyncio.create_task(
        store.source_manifest(bundle, sources, trusted_keys=keys, node_id="node")
    )
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        closing = asyncio.create_task(registry.close_all())
        await asyncio.sleep(0.01)
        assert not task.done() and not closing.done()
        release.set()
        with pytest.raises(ValueError if failure else asyncio.CancelledError):
            await task
        await closing
        assert not store._pending
    finally:
        release.set()
        await asyncio.gather(task, *([closing] if closing else []), return_exceptions=True)
        await registry.close_all()


def software_windows(**windows):
    """Construct numerical window fixtures, not an owner-recomputed manifest."""
    bundle, _, _ = motion_study()
    factors = {o.observation_id: o for o in bundle.observations if isinstance(o, MotionObservation)}
    chosen = ("m1", "m24", "m25")
    source_ids = {identity for key in chosen for identity in factors[key].source_ids}
    observations = tuple(
        o for o in bundle.observations if o.observation_id in source_ids | set(chosen)
    )
    bundle = replace(
        bundle,
        observations=observations,
        accepted_ids=("m1",),
        rejected_ids=(),
        holdout_ids=("m24", "m25"),
    )
    by_id = {o.observation_id: o for o in observations}
    rows = []
    for key in chosen:
        factor = factors[key]
        window = windows.get(key, (factor.first_stamp.seconds - 0.1, factor.stamp.seconds + 0.1))
        for identity in factor.source_ids:
            rows.append(
                {
                    "source_id": identity,
                    "source_digest": by_id[identity].digest,
                    "acquisition_digest": f"acquisition:{identity}",
                    "observation_digest": f"capture:{identity}",
                    "plan_digest": "software-plan",
                    "run_id": "software-run",
                    "approval": {
                        key: f"software:{key}"
                        for key in (
                            "receipt_digest",
                            "action_digest",
                            "approval_digest",
                            "client_digest",
                            "key_id",
                            "node_id",
                        )
                    },
                    "acquisition_window_s": window,
                }
            )
    manifest = IndexedSourceManifest(
        bundle.digest, bundle.topology.digest, bundle.metadata["node_session_id"], json.dumps(rows)
    )
    return bundle, manifest


def test_indexed_windows_are_stricter_than_endpoint_separation() -> None:
    bundle, manifest = software_windows()
    assert independent_indexed_holdouts(bundle, manifest) == (("m24", "m25"), ())
    bundle, manifest = software_windows(m24=(2.0, 49.1))
    accepted, excluded = independent_indexed_holdouts(bundle, manifest)
    assert accepted == ("m25",) and excluded[0][0] == "m24"
    # A rejected fit row remains a training exclusion.
    rejected = replace(bundle, accepted_ids=(), rejected_ids=("m1",))
    rebound = replace(manifest, evidence_digest=rejected.digest)
    assert independent_indexed_holdouts(rejected, rebound) == (accepted, excluded)
    bundle, manifest = software_windows(m25=(49.2, 51.1))
    accepted, excluded = independent_indexed_holdouts(bundle, manifest)
    assert accepted == ("m24",) and excluded[0][0] == "m25"
    reordered = replace(bundle, holdout_ids=tuple(reversed(bundle.holdout_ids)))
    assert independent_indexed_holdouts(
        reordered, replace(manifest, evidence_digest=reordered.digest)
    ) == (accepted, excluded)


def test_indexed_baseline_training_and_lineage_conflicts_are_not_hidden() -> None:
    bundle, manifest = software_windows()
    baseline = replace(bundle, accepted_ids=("m24",), holdout_ids=())
    baseline_manifest = replace(manifest, evidence_digest=baseline.digest)
    accepted, excluded = independent_indexed_holdouts(
        bundle, manifest, baseline_bundle=baseline, baseline_manifest=baseline_manifest
    )
    assert accepted == ("m25",) and excluded[0][0] == "m24"
    rows = json.loads(baseline_manifest.rows_json)
    rows[0]["acquisition_digest"] = "different-acquisition"
    with pytest.raises(ValueError, match="lineage conflicts"):
        independent_indexed_holdouts(
            bundle,
            manifest,
            baseline_bundle=baseline,
            baseline_manifest=replace(baseline_manifest, rows_json=json.dumps(rows)),
        )
    with pytest.raises(ValueError, match="paired"):
        independent_indexed_holdouts(bundle, manifest, baseline_bundle=baseline)
    with pytest.raises(ValueError, match="session"):
        independent_indexed_holdouts(bundle, replace(manifest, node_session_id="other"))


@pytest.mark.parametrize("bad", [False, "0.3", 0.2, float("inf"), 121])
def test_indexed_separation_policy_cannot_be_coerced_or_weakened(bad: Any) -> None:
    bundle, manifest = software_windows()
    with pytest.raises(ValueError):
        independent_indexed_holdouts(bundle, manifest, separation_s=bad)


@pytest.mark.parametrize("defect", ["extra", "duplicate", "window", "approval", "source_digest"])
def test_manifest_structure_and_source_binding_refuse(defect: str) -> None:
    bundle, manifest = software_windows()
    rows = json.loads(manifest.rows_json)
    if defect == "extra":
        rows[0]["qualified"] = True
    elif defect == "duplicate":
        rows.append(rows[0])
    elif defect == "window":
        rows[0]["acquisition_window_s"] = [False, 3]
    elif defect == "approval":
        rows[0]["approval"]["signature"] = "not-a-proof"
    else:
        rows[0]["source_digest"] = "changed"
    with pytest.raises(ValueError):
        altered = replace(manifest, rows_json=json.dumps(rows))
        independent_indexed_holdouts(bundle, altered)
