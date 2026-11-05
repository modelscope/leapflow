# Copyright (c) Alibaba, Inc. and its affiliates.
"""Private acquisition lineage and settlement; all captures here are software fixtures."""
from __future__ import annotations

import asyncio
import json
import os
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from leapflow.cache.manager import CacheManager
from leapflow.hardware.calibration.acquisition import SettledObservation
from leapflow.hardware.calibration.evidence import (
    CATEGORY, MAX_RAW_BYTES, AcquisitionBinding, CalibrationEvidenceStore, RecordedObservationSource,
)
from leapflow.hardware.lhp.authority import WriteAuthority
from leapflow.hardware.lhp.permit import PermitIssuer, PermitVerifier, generate_keypair
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.skills.admission import RunAdmission
from leapflow.hardware.skills.digest import plan_digest, run_permit_digest
from leapflow.hardware.skills.geometry import Pose
from leapflow.hardware.skills.protocol import SkillError, SkillPlan
from leapflow.layout import CacheLayout
from tests.test_leapcalib_acquisition import Camera, observation, profile, sources

pytestmark = pytest.mark.component


def sealed_plan(**changes: Any) -> SkillPlan:
    plan = SkillPlan(skill_id="calib.collect", workcell="cell", version="1.0.0",
                     args={"synthetic_fixture": True}, summary="Stationary software capture",
                     targets=(Pose(0.0, 0.0, 0.0),), limits={"max_duration_s": 2.0}, created_at=1700000000.0, **changes)
    return replace(plan, digest=plan_digest(plan))


def binding(plan: SkillPlan | None = None) -> AcquisitionBinding:
    plan = plan or sealed_plan()
    expected = run_permit_digest(workcell=plan.workcell, skill_id=plan.skill_id,
                                 version=plan.version, plan_digest=plan.digest, limits=plan.limits)
    return AcquisitionBinding("cell", "run", "node-boot", json.dumps(plan.to_dict()),
                              RunAdmission("trusted_v1", expected))


def registry_and_cache(tmp_path: Path):
    registry = HardwareRegistry(HardwareSettings(instrument_db_path=str(tmp_path / "instrument.duckdb")), providers=[])
    cache = CacheManager(CacheLayout(tmp_path / "cache"), profile_id="test")
    store = registry.calibration_evidence
    registry.bind_persistence(cache_manager=cache)
    return registry, cache, store


async def captured(monkeypatch: pytest.MonkeyPatch):
    registry, _, feedback, clock = await sources(monkeypatch)
    result = await observation(feedback, Camera(clock), clock).capture()
    await registry.close_all()
    return result, clock


def record(store, value, **changes: Any):
    return store.record(value, **{"profile": profile(), "binding": binding(), "ttl_s": 60,
                                "quota_bytes": MAX_RAW_BYTES, **changes})


async def test_roundtrip_binds_values_and_refuses_malformed_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    value, _ = await captured(monkeypatch)
    assert SettledObservation.from_mapping(json.loads(json.dumps(value.to_dict()))) == value
    altered = replace(value, feedback=(replace(value.feedback[0], joints_rad=(0.0001,)), *value.feedback[1:]))
    assert altered.feedback[0].source_id == value.feedback[0].source_id
    assert altered.digest != value.digest
    for key, bad in (("tcp", None), ("stamp", []), ("joints_rad", [0] * 17),
                     ("joint_ids", []), ("communication_errors", [True])):
        data = value.to_dict()
        data["feedback"][0][key] = bad
        with pytest.raises(ValueError):
            SettledObservation.from_mapping(data)
    with pytest.raises(ValueError):
        SettledObservation.from_mapping(None)


async def test_private_shared_store_roundtrip_restart_and_idempotency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value, _ = await captured(monkeypatch)
    registry, cache, store = registry_and_cache(tmp_path)
    try:
        assert store._holder is registry.calibration_generations._holder
        summary = await asyncio.to_thread(record, store, value)
        identity = summary["evidence_digest"]
        (entry,) = cache.list_entries(category=CATEGORY)
        assert entry.sensitive and not entry.syncable
        assert entry.path.stat().st_mode & 0o777 == 0o600
        assert entry.path.parent.stat().st_mode & 0o777 == 0o700
        assert not summary["hardware_accuracy_certified"] and not summary["raw_exported"]
        raw = await asyncio.to_thread(store.read, identity, workcell_id="cell")
        assert SettledObservation.from_mapping(raw["observation"]) == value
        assert await asyncio.to_thread(record, store, value) == summary
        assert cache.list_entries(category=CATEGORY)[0].expires_at == entry.expires_at
        with pytest.raises(ValueError, match="different run"):
            await asyncio.to_thread(record, store, value, binding=replace(binding(), run_id="other-run"))
        restarted = CalibrationEvidenceStore(store._holder, cache)
        assert await asyncio.to_thread(restarted.read, identity, workcell_id="cell") == raw
        with pytest.raises(ValueError, match="workcell"):
            await asyncio.to_thread(store.read, identity, workcell_id="other")
    finally:
        await registry.close_all()


@pytest.mark.parametrize("unavailable", ["expired", "evicted", "hash", "permissions", "missing"])
async def test_raw_unavailability_preserves_summary_but_never_authorizes_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unavailable: str) -> None:
    value, _ = await captured(monkeypatch)
    registry, cache, store = registry_and_cache(tmp_path)
    try:
        summary = await asyncio.to_thread(record, store, value)
        identity = summary["evidence_digest"]
        (entry,) = cache.list_entries(category=CATEGORY)
        if unavailable == "expired":
            monkeypatch.setattr("leapflow.hardware.calibration.evidence.time.time", lambda: entry.expires_at + 1)
        elif unavailable == "evicted":
            cache.cleanup_quota(scope="profile", category=CATEGORY, max_bytes=0)
        elif unavailable == "hash":
            entry.path.write_bytes(b"{}")
        elif unavailable == "permissions":
            os.chmod(entry.path, 0o644)
        else:
            entry.path.rename(entry.path.with_suffix(".unavailable"))
        with pytest.raises(ValueError):
            await asyncio.to_thread(store.read, identity, workcell_id="cell")
        with pytest.raises(ValueError):
            await asyncio.to_thread(record, store, value)
        assert await asyncio.to_thread(store.summary, identity, workcell_id="cell") == summary
    finally:
        await registry.close_all()


async def test_owner_rechecks_acquisition_identity_and_uncertainty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value, _ = await captured(monkeypatch)
    registry, _, store = registry_and_cache(tmp_path)
    try:
        for sample in (replace(value.feedback[0], node_session_id="other"),
                       replace(value.feedback[0], joint_uncertainty_rad=0.00001),
                       replace(value.feedback[0], arrived=False)):
            with pytest.raises(ValueError):
                await asyncio.to_thread(record, store, replace(value, feedback=(sample, *value.feedback[1:])))
        with pytest.raises(ValueError, match="digest mismatch"):
            binding(replace(sealed_plan(), summary="Changed after approval"))
        with pytest.raises(ValueError, match="different plan"):
            replace(binding(), admission=RunAdmission("trusted_v1", "different"))
    finally:
        await registry.close_all()


@pytest.mark.parametrize("failure", [False, True])
async def test_repeated_cancel_and_registry_shutdown_wait_for_private_settlement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: bool) -> None:
    value, clock = await captured(monkeypatch)
    registry, _, store = registry_and_cache(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = store.record
    commits = []

    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        if failure:
            raise OSError("isolated persistence failure")
        return original(*args, **kwargs)

    class Source:
        async def capture(self):
            return value

    monkeypatch.setattr(store, "record", slow)
    wrapper = RecordedObservationSource(Source(), store, profile=profile(), binding=binding(),
                                        ttl_s=60, quota_bytes=MAX_RAW_BYTES, on_commit=commits.append, clock=clock)
    task = asyncio.create_task(wrapper.capture())
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
        if failure:
            with pytest.raises(SkillError) as caught:
                await task
            assert caught.value.side_effect_state == "unknown"
            assert not commits
        else:
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(commits) == 1
        await closing
        assert not store._pending
    finally:
        release.set()
        await asyncio.gather(task, *([closing] if closing else []), return_exceptions=True)
        await registry.close_all()


@pytest.mark.parametrize("component", ["../escape", "/absolute", ".", "..", ""])
def test_private_cache_refuses_path_components(tmp_path: Path, component: str) -> None:
    cache = CacheManager(CacheLayout(tmp_path / "cache"), profile_id="test")
    with pytest.raises(ValueError):
        cache.private_path(category=component, source="settled")
    with pytest.raises(ValueError):
        cache.private_path(category="acquisition", source=component)


def test_private_cache_refuses_symlink_before_creation(tmp_path: Path) -> None:
    cache = CacheManager(CacheLayout(tmp_path / "cache"), profile_id="test")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (tmp_path / "cache" / "profile" / "acquisition").symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        cache.private_path(category="acquisition", source="settled")
    assert not (foreign / "settled").exists()


def test_exact_existing_authority_receipt_consumes_nonce_once() -> None:
    private, public = generate_keypair()
    issuer = PermitIssuer(private)
    verifier = PermitVerifier({issuer.key_id: public}, node_id="node")
    authority = WriteAuthority("permit_required", verifier)
    digest = binding().admission.action_digest
    permit = issuer.issue(audience="node", session_id="session", workspace="workspace", approval_id="software-human-fixture", digest=digest)
    refusal, receipt = authority.admit_run({"permit": permit.to_dict()}, expected_digest=digest)
    assert refusal is None and receipt.to_dict()["permit"] == permit.to_dict()
    refused, absent = authority.admit_run({"permit": permit.to_dict()}, expected_digest=digest)
    assert not refused.ok and absent is None
    assert refused.failure_code == "permit_replayed"
    assert WriteAuthority("observe_only").admit_run({}, expected_digest=digest)[1] is None
    assert WriteAuthority("trusted_v1").admit_run({}, expected_digest=digest)[1].permit_json == ""
