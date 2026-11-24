# Copyright (c) Alibaba, Inc. and its affiliates.
"""Governed passive evidence: real admission, cache, approval and node boundaries."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from PIL import Image

from leapflow.cache.manager import CacheManager
from leapflow.config import load_config
from leapflow.config_service import ConfigService
from leapflow.domain.events import RecordScope
from leapflow.hardware.snapshot_evidence import (
    EvidenceError, SnapshotEvidenceService, SnapshotEvidenceSettings, _decode, export_snapshot_result,
)
from leapflow.hardware.stream import EventKind, HardwareEvent
from leapflow.layout import workspace_id_for_path
from leapflow.memory.providers.episodic import EpisodicMemoryProvider
from leapflow.memory.providers.working import WorkingMemoryProvider
from leapflow.monitor.signal_noise import SignalNoiseGate
from leapflow.platform.event_bus import EventBus
from leapflow.privacy.policy import PrivacyManager
from leapflow.security.orchestrator import ApprovalOrchestrator
from leapflow.tools.execution_context import ToolExecutionContext, reset_tool_context, set_tool_context
from test_model_tools import Human, compose
from test_skill_tools import WORKCELL_ID, _until, rigs  # noqa: F401 - shared real-node fixture


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LEAPFLOW_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("LEAPFLOW_HARDWARE_SHARE_FRAMES", "true")
    import leapflow.config as config
    monkeypatch.setattr(config, "_settings_instance", None)
    return load_config()


def bus():
    immediate, working = EpisodicMemoryProvider(), WorkingMemoryProvider()
    return EventBus(immediate, working, privacy_filter=PrivacyManager())


def snapshot(format="PNG"):
    output = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(output, format=format)
    return {"frame": "f1", "capture_id": "capture-one", "camera": "camera.wrist",
            "observed_at": time.time(), "valid_for_s": 60.0,
            "validity_timebase": "node_remaining_duration",
            "image": {"mime": {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[format],
                      "data": base64.b64encode(output.getvalue()).decode()}}


def context(settings, gate, session="own", workspace=None):
    return set_tool_context(ToolExecutionContext.from_strings(
        workspace_root=str(workspace or settings.workspace_root), session_id=session,
        task_id=f"turn-{session}", orchestrator=gate))


def cache(settings):
    return CacheManager(settings.profile_layout.cache, profile_id=settings.profile)


def persist(manager, content=b"content", *, session="own", workspace="workspace", ttl=60, quota=100):
    return manager.persist_session_content(content, workspace_id=workspace, session_id=session,
        category="snapshot_evidence", source="hardware_snapshot", suffix=".png",
        expires_at=time.time() + ttl, monotonic_deadline=time.monotonic() + ttl,
        max_session_bytes=quota, metadata={"mime": "image/png"})


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP"])
def test_bounded_media(format):
    shot = snapshot(format)
    content, mime, suffix = _decode(shot["image"], SnapshotEvidenceSettings())
    assert content and mime == shot["image"]["mime"] and suffix in (".png", ".jpg", ".webp")
    with pytest.raises(EvidenceError):
        _decode(shot["image"], SnapshotEvidenceSettings(max_pixels=1))
    with pytest.raises(EvidenceError):
        _decode({**shot["image"], "mime": "application/octet-stream"}, SnapshotEvidenceSettings())
    with pytest.raises(EvidenceError):
        _decode({"mime": mime, "data": "not base64"}, SnapshotEvidenceSettings())


@pytest.mark.parametrize("values", [[], {"unknown": 1}, {"ttl_s": 301}, {"ttl_s": True},
    {"ttl_s": float("nan")}, {"quota_bytes": 67108865}, {"max_frame_bytes": 0},
    {"max_pixels": 1.2}, {"cleanup_interval_s": 0}])
def test_invalid_settings_never_written(settings, values):
    service = ConfigService(settings)
    path = settings.profile_layout.runtime_config_path
    before = path.read_bytes() if path.exists() else None
    with pytest.raises(ValueError):
        service.set("hardware.snapshot_evidence", json.dumps(values))
    assert (path.read_bytes() if path.exists() else None) == before


async def test_private_record_has_no_global_delivery_and_dedup_is_scoped():
    events = bus()
    delivered = []
    events.subscribe(delivered.append)
    payload = HardwareEvent(EventKind.SNAPSHOT_EVIDENCE, "cell", "camera", "snapshot", "reference").to_payload()
    own = RecordScope("profile", "workspace", "own")
    async def admit(scope):
        return await events.admit_record("hw.snapshot_evidence", payload, scope=scope,
                                         dedup_key="capture", noise_gate=SignalNoiseGate())
    assert (await admit(own)).admitted
    assert (await admit(own)).reason == "duplicate"
    assert (await admit(replace(own, session_id="other"))).admitted
    assert (await admit(replace(own, workspace_id="other"))).admitted
    assert (await admit(replace(own, profile_id="other"))).admitted
    assert not delivered and not events._event_buffer and events._immediate.active_count == 0
    assert not events._working._items
    invalid = await events.admit_record("hw.snapshot_evidence", {**payload, "value": b"bytes"}, scope=own,
                                       dedup_key="new", noise_gate=SignalNoiseGate())
    assert invalid.reason == "invalid_metadata"


async def test_retention_then_specialist_uses_opaque_id_and_independent_consent(settings):
    events = bus()
    sink = SnapshotEvidenceService(lambda: settings, events)
    human = Human()
    gate = ApprovalOrchestrator(human)
    token = context(settings, gate)
    owner, registry, _, _, requests = compose({"selected_id": "a", "rationale": "eligible"}, processing_location="remote")
    registry.bind_runtime(settings=settings, model_approval_gate=gate)
    try:
        result = await export_snapshot_result(sink, {"ok": True, "result": snapshot()},
                                               workcell="node/cell", started_at=time.monotonic())
        assert result["ok"], result
        assert "image" not in result["result"] and "_attachments" not in result
        reference = result["evidence"]
        assert reference["sensitive"] and not reference["syncable"]
        entry = cache(settings).resolve_session_entry(reference["artifact_id"],
            workspace_id=workspace_id_for_path(settings.workspace_root), session_id="own", max_bytes=1048576)
        assert entry.metadata["record"]["event_class"] == "record"
        assert entry.metadata["record"]["value"]["observation_id"] == reference["observation_id"]
        assert os.stat(entry.path).st_mode & 0o777 == 0o600
        assert os.stat(entry.path.parent).st_mode & 0o777 == 0o700
        assert 0 < reference["expires_at"] - time.time() <= 60
        answer = await registry.tool_handlers["model_query"]("specialist", {
            "goal": "choose", "candidates": [{"id": "a", "description": "eligible"}]},
            artifacts={"image": reference["artifact_id"]})
        assert answer["ok"], answer
        assert len(requests) == 1 and not answer["provenance"]["physical_authority"]
        kinds = [request.action.kind for request in human.requests]
        assert "file.write" in kinds and "file.read" in kinds and "network.fetch" in kinds
        assert all(request.action.session_id == "own" for request in human.requests)
        assert not events._event_buffer and events._immediate.active_count == 0
    finally:
        reset_tool_context(token)
        await owner.aclose()
        await sink.aclose()
    assert not cache(settings).list_entries(category="snapshot_evidence")


@pytest.mark.parametrize("fault", ["denied", "missing", "broken", "sharing", "stale", "image"])
async def test_retention_refusals_do_not_retain_or_attach(settings, fault):
    class Broken:
        async def evaluate(self, action):
            raise RuntimeError("test-only broken route")
    human = Human(allow=fault != "denied")
    gate = None if fault == "missing" else Broken() if fault == "broken" else ApprovalOrchestrator(human)
    runtime = replace(settings, hardware_share_frames=False) if fault == "sharing" else settings
    sink = SnapshotEvidenceService(lambda: runtime, bus())
    shot = snapshot()
    if fault == "stale":
        shot["valid_for_s"] = 0
    if fault == "image":
        shot.pop("image")
    token = context(settings, gate)
    try:
        result = await export_snapshot_result(sink, {"ok": True, "result": shot, "_attachments": ["forbidden"]},
                                               workcell="node/cell", started_at=time.monotonic())
        assert not result["ok"] and "image" not in result["result"] and "_attachments" not in result
        assert not cache(settings).list_entries(category="snapshot_evidence")
        if fault in ("denied", "missing", "broken"):
            assert "Do not retry" in result["next_step"]
        else:
            assert not human.requests
    finally:
        reset_tool_context(token)
        await sink.aclose()


@pytest.mark.parametrize("fault", ["privacy", "redaction", "noise"])
async def test_admission_refuses_before_retention_consent(settings, monkeypatch, fault):
    class Privacy:
        def should_ingest(self, *args):
            return fault != "privacy"
        def redact_payload(self, event_type, payload):
            return {**payload, "value": "[REDACTED]"} if fault == "redaction" else payload
    events = bus()
    events.set_privacy_filter(Privacy())
    if fault == "noise":
        monkeypatch.setattr(SignalNoiseGate, "should_pass", lambda self, event: False)
    sink = SnapshotEvidenceService(lambda: settings, events)
    human = Human()
    token = context(settings, ApprovalOrchestrator(human))
    try:
        with pytest.raises(EvidenceError, match="admission|provenance"):
            await sink.retain_snapshot(snapshot(), workcell="node/cell", started_at=time.monotonic())
        assert not human.requests and not cache(settings).list_entries(category="snapshot_evidence")
    finally:
        reset_tool_context(token)
        await sink.aclose()


async def test_concurrent_sessions_retain_independent_authority(settings, tmp_path):
    sink = SnapshotEvidenceService(lambda: settings, bus())
    gate = ApprovalOrchestrator(Human())
    shot = snapshot()
    async def retain(session):
        token = context(settings, gate, session=session, workspace=tmp_path / session)
        try:
            return await sink.retain_snapshot(shot, workcell="node/cell", started_at=time.monotonic())
        finally:
            reset_tool_context(token)
    try:
        left, right = await asyncio.gather(retain("left"), retain("right"))
        assert left.artifact_id != right.artifact_id and left.content_hash == right.content_hash
        manager = cache(settings)
        with pytest.raises(ValueError, match="unavailable"):
            manager.resolve_session_entry(left.artifact_id, workspace_id=workspace_id_for_path(tmp_path / "right"),
                                          session_id="right", max_bytes=1048576)
    finally:
        await sink.aclose()


async def test_cancel_during_real_approval_retains_nothing(settings):
    entered = asyncio.Event()
    class WaitingHuman(Human):
        async def request_approval(self, request):
            entered.set()
            await asyncio.Event().wait()
    sink = SnapshotEvidenceService(lambda: settings, bus())
    token = context(settings, ApprovalOrchestrator(WaitingHuman()))
    try:
        call = asyncio.create_task(sink.retain_snapshot(snapshot(), workcell="node/cell", started_at=time.monotonic()))
        await asyncio.wait_for(entered.wait(), 3)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(call, 1)
        assert not cache(settings).list_entries(category="snapshot_evidence")
    finally:
        reset_tool_context(token)
        await sink.aclose()


async def test_cancel_during_disk_write_discards_late_result(settings, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = CacheManager.persist_session_content
    def delayed(manager, *args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(manager, *args, **kwargs)
    monkeypatch.setattr(CacheManager, "persist_session_content", delayed)
    sink = SnapshotEvidenceService(lambda: settings, bus())
    token = context(settings, ApprovalOrchestrator(Human()))
    try:
        call = asyncio.create_task(sink.retain_snapshot(snapshot(), workcell="node/cell", started_at=time.monotonic()))
        await _until(entered.is_set)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(call, 1)
        release.set()
        await _until(lambda: not sink._cleanup_tasks)
        assert not cache(settings).list_entries(category="snapshot_evidence")
    finally:
        release.set()
        reset_tool_context(token)
        await sink.aclose()


def test_atomic_quota_and_monotonic_expiry(settings, monkeypatch):
    manager = cache(settings)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(persist, manager, quota=7) for _ in range(2)]
        results = []
        for job in jobs:
            try:
                results.append(job.result())
            except ValueError as exc:
                assert "quota" in str(exc)
    assert len(results) == 1
    entry = results[0]
    monkeypatch.setattr("leapflow.cache.manager.time.monotonic", lambda: entry.metadata["_monotonic_deadline"] + 1)
    with pytest.raises(ValueError, match="expired"):
        manager.resolve_session_entry(entry.entry_id, workspace_id="workspace", session_id="own", max_bytes=100)
    assert manager.cleanup_expired(now=entry.created_at - 100) == 1
    assert not entry.path.exists()


def test_secure_storage_and_reads_reject_path_replacement(settings, tmp_path):
    manager = cache(settings)
    entry = persist(manager)
    content = manager.read_session_entry(entry, max_bytes=100)
    assert content == b"content"
    entry.path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        manager.read_session_entry(entry, max_bytes=100)
    moved = entry.path.with_suffix(".original")
    entry.path.rename(moved)
    entry.path.symlink_to(moved)
    with pytest.raises(ValueError, match="escaped"):
        manager.read_session_entry(entry, max_bytes=100)
    # Descriptor-relative cleanup removes the final link, not its target.
    manager.discard_session_content(entry)
    assert moved.read_bytes() == b"changed" and not entry.path.is_symlink()
    parent = moved.parent
    relocated = parent.with_name("relocated")
    parent.rename(relocated)
    parent.symlink_to(relocated, target_is_directory=True)
    with pytest.raises(OSError):
        persist(manager)
    assert list(relocated.iterdir()) == [relocated / moved.name]


def test_fifo_is_refused_without_blocking(settings):
    manager = cache(settings)
    entry = persist(manager)
    entry.path.rename(entry.path.with_suffix(".saved"))
    os.mkfifo(entry.path)
    with pytest.raises(ValueError, match="regular file"):
        manager.read_session_entry(entry, max_bytes=100)
    manager.discard_session_content(entry)


def test_transient_reference_invalidated_by_runtime_restart(settings, monkeypatch):
    manager = cache(settings)
    entry = persist(manager)
    monkeypatch.setattr("leapflow.cache.manager._VALIDITY_PROCESS", "another-runtime")
    with pytest.raises(ValueError, match="restarted"):
        manager.resolve_session_entry(entry.entry_id, workspace_id="workspace", session_id="own", max_bytes=100)


async def test_expiry_during_human_consent_is_not_extended(settings):
    class SlowHuman(Human):
        async def request_approval(self, request):
            await asyncio.sleep(.3)
            return await super().request_approval(request)
    runtime = replace(settings, hardware_snapshot_evidence='{"ttl_s":0.2}')
    sink = SnapshotEvidenceService(lambda: runtime, bus())
    human = SlowHuman()
    token = context(settings, ApprovalOrchestrator(human))
    try:
        with pytest.raises(EvidenceError, match="expired"):
            await sink.retain_snapshot(snapshot(), workcell="node/cell", started_at=time.monotonic())
        assert not cache(settings).list_entries(category="snapshot_evidence")
    finally:
        reset_tool_context(token)
        await sink.aclose()


async def test_failed_export_consumes_images_without_calling_sink():
    class RefuseUse:
        async def retain_snapshot(self, *args, **kwargs):
            pytest.fail("failed snapshot must not reach retention consent")
        async def aclose(self):
            pass
    result = await export_snapshot_result(RefuseUse(), {"ok": False, "result": snapshot(), "_attachments": ["image"]},
                                         workcell="node/cell", started_at=time.monotonic())
    assert result["evidence"]["status"] == "missing"
    assert "image" not in result["result"] and "_attachments" not in result


@pytest.mark.parametrize("node_share", [True, False])
async def test_real_passive_node_snapshot_exports_without_motion(settings, tmp_path, rigs, node_share):  # noqa: F811 - imported fixture
    from leapflow.hardware.skills.builtin.scene import skills as scene_skills
    from tests._harness.skill_stubs import PERCEPTION_WORKCELL_YAML
    from tests.test_perception import CUP, _calibrated
    devices = _calibrated(tmp_path / "devices", CUP).devices
    rig = rigs(tmp_path / "node", cell={**PERCEPTION_WORKCELL_YAML, "share_frames": node_share},
               node_skills=scene_skills(), devices=devices)
    rig.nodes[0].registry.calibrate_perception()
    rig.service._share_frames = True
    sink = SnapshotEvidenceService(lambda: settings, bus())
    rig.service.set_evidence_sink(sink)
    token = context(settings, ApprovalOrchestrator(Human()))
    try:
        result = await rig.service.hw_skill_run(workcell=WORKCELL_ID, skill="perceive.snapshot",
                                                args={"camera": "camera"}, export_evidence=True)
        assert result["ok"] is node_share, json.dumps(result)
        assert result["evidence"]["status"] == ("retained" if node_share else "missing")
        assert "image" not in result["result"] and "_attachments" not in result
        assert not rig.sim.batches and not rig.gate.seen
        if node_share:
            ordinary = await rig.service.hw_skill_run(workcell=WORKCELL_ID, skill="perceive.snapshot", args={})
            assert ordinary["ok"] and ordinary.get("_attachments")
    finally:
        reset_tool_context(token)
        await sink.aclose()


@pytest.mark.parametrize("stage", ["cache", "admission"])
async def test_preconsent_stages_are_bounded_by_native_validity(settings, monkeypatch, stage):
    human = Human()
    events = bus()
    sink = SnapshotEvidenceService(lambda: settings, events)
    runtime = replace(settings, hardware_snapshot_evidence='{"ttl_s":0.05}')
    sink._settings = lambda: runtime
    async def stalled(*args, **kwargs):
        await asyncio.Event().wait()
    if stage == "cache":
        monkeypatch.setattr(sink, "_preflight", stalled)
    else:
        monkeypatch.setattr(events, "admit_record", stalled)
    token = context(settings, ApprovalOrchestrator(human))
    try:
        with pytest.raises(EvidenceError) as error:
            await asyncio.wait_for(sink.retain_snapshot(snapshot(), workcell="node/cell", started_at=time.monotonic()), 1)
        assert error.value.failure.code == "evidence_stale"
        assert not human.requests and not cache(settings).list_entries(category="snapshot_evidence")
    finally:
        reset_tool_context(token)
        await sink.aclose()


async def test_sweeper_survives_invalid_hot_reload(settings):
    runtime = replace(settings, hardware_snapshot_evidence='{"ttl_s":0.08,"cleanup_interval_s":0.02}')
    sink = SnapshotEvidenceService(lambda: runtime, bus())
    token = context(settings, ApprovalOrchestrator(Human()))
    try:
        reference = await sink.retain_snapshot(snapshot(), workcell="node/cell", started_at=time.monotonic())
        runtime = replace(settings, hardware_snapshot_evidence='{"ttl_s":999}')
        await _until(lambda: reference.artifact_id not in sink._retained)
        assert not sink._sweeper.done()
        assert not cache(settings).list_entries(category="snapshot_evidence")
    finally:
        reset_tool_context(token)
        await sink.aclose()


@pytest.mark.parametrize("daemon_mode", [False, True])
async def test_real_runtime_modes_publish_reload_and_dispose_evidence(settings, monkeypatch, daemon_mode):
    from leapflow.cli.context import Context
    import leapflow.plugins as plugins
    from leapflow.hardware.snapshot_evidence import SessionEvidenceSink
    from conftest import make_settings

    monkeypatch.setattr(plugins, "_registry", None)
    monkeypatch.setattr(plugins, "_scoped_registry", None)
    runtime = replace(make_settings(str(settings.layout.root)), hardware_enabled=True,
                      hardware_providers="yaml", hardware_share_frames=True,
                      hardware_stream_enabled=False, hardware_fleet_nodes="[]")
    ctx = Context(runtime, mock_host=True)
    try:
        await ctx.initialize_critical(daemon_mode=daemon_mode)
        sink = ctx._hardware_evidence_sink
        registry = plugins.get_registry()
        plugin = registry.get_plugin("hardware_context")
        assert isinstance(sink, SessionEvidenceSink) and plugin._evidence_sink is sink
        assert sink._sweeper is None and ctx._daemon_mode is daemon_mode
        snapshot_handlers = dict(registry.tool_handlers)
        version = registry.version
        plugins.reload_plugin("hardware_context")
        fresh = registry.get_plugin("hardware_context")
        assert fresh is not plugin and fresh._evidence_sink is sink
        assert not sink._closed and registry.version > version
        assert "hw_read" in snapshot_handlers and "hw_read" in registry.tool_handlers
        assert snapshot_handlers["hw_read"] is not registry.tool_handlers["hw_read"]
        await plugins.get_scoped_registry().get_fiber("hardware_context").scope.async_dispose()
        assert registry.get_plugin("hardware_context") is None
        # The runtime owns the shared sink, not the retiring plugin fiber.
        assert not sink._closed
    finally:
        await ctx.cleanup()
    assert sink._closed and sink._scope.is_disposed
