# Copyright (c) Alibaba, Inc. and its affiliates.
"""Real registry, provider and approval composition for read-only model tools."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time

import httpx
import pytest

from leapflow.cache.manager import CacheManager
from leapflow.config import load_config
from leapflow.layout import workspace_id_for_path
from leapflow.llm.provider_registry import LLMProviderRegistry
from leapflow.llm.services.config import parse_services
from leapflow.llm.services.http_client import BoundedTransport
from leapflow.llm.services.runtime import ModelServiceRuntime
from leapflow.plugins.registry import ToolPluginRegistry
from leapflow.plugins.scoped_registry import ScopedToolRegistry
from leapflow.plugins.tool_plugins.model_services import ModelServicesPlugin
from leapflow.security.approval import ApprovalDecision, SessionAwareGate
from leapflow.security.orchestrator import ApprovalOrchestrator
from leapflow.tools.execution_context import ToolExecutionContext, reset_tool_context, set_tool_context
from leapflow.tools.model_fixture import check_fixture
from leapflow.cli.commands.model_service_config import execute_fixture, parse_service_arguments
from leapflow.world_model.specialist_contract import validate_result
from test_model_services import declaration, recorded_response


class Human:
    def __init__(self, allow=True):
        self.allow = allow
        self.requests = []

    async def request_approval(self, request):
        self.requests.append(request)
        return ApprovalDecision.ALLOW_ONCE if self.allow else ApprovalDecision.DENY


def compose(payload, *, task="decision", **changes):
    providers = LLMProviderRegistry()
    providers.discover_builtin()
    requests = []

    async def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=recorded_response(payload))

    owner = ModelServiceRuntime(providers, parse_services({"specialist": declaration(task=task, **changes)}),
        client_factory=lambda spec: httpx.AsyncClient(transport=BoundedTransport(spec, httpx.MockTransport(respond))))
    registry = ToolPluginRegistry()
    plugin = ModelServicesPlugin()
    registry.register(plugin)
    registry.assemble()
    assert not registry.tool_handlers
    scoped = ScopedToolRegistry(registry)
    scoped.adopt_existing_plugins()
    registry.bind_runtime(model_services=owner)
    registry.publish_plugin_tools_atomic(plugin)
    return owner, registry, scoped, plugin, requests


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["query", "fixture"])
async def test_disabled_tools_keep_catalog_but_never_read_approve_or_send(tmp_path, monkeypatch, mode):
    from leapflow.tools import model_fixture, model_tools

    owner, registry, _, plugin, requests = compose({"selected_id": "a", "rationale": "eligible"}, enabled=False)
    human = Human()
    accessed = []
    monkeypatch.setattr(model_fixture, "_read_fixture", lambda *args: accessed.append("fixture"))
    monkeypatch.setattr(model_tools, "classify_url", lambda *args: accessed.append("dns"))
    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(tmp_path), session_id="own",
        orchestrator=ApprovalOrchestrator(human)))
    try:
        assert {"model_query", "model_describe"} <= {tool.name for tool in plugin.tools}
        status = await registry.tool_handlers["model_describe"]("specialist")
        assert status["services"][0]["unavailable_reason"] == "disabled"
        if mode == "query":
            result = await registry.tool_handlers["model_query"]("specialist", {
                "goal": "choose", "candidates": [{"id": "a", "description": "eligible"}]})
        else:
            result = await check_fixture(owner, "specialist", tmp_path / "unread.json")
        assert result["failure_code"] == "model_service_disabled" and not result["retryable"]
        assert not human.requests and not accessed and not requests and not owner._handles
    finally:
        reset_tool_context(token)
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["query", "fixture"])
async def test_disable_during_owned_cleanup_discards_success_even_after_reenable(tmp_path, monkeypatch, mode):
    owner, registry, _, _, requests = compose({"selected_id": "a", "rationale": "eligible"})
    original = owner._release

    async def revoke_after_release(handle):
        await original(handle)
        assert not owner._generations
        owner.publish(parse_services({"specialist": declaration(enabled=False)}))
        owner.publish(parse_services({"specialist": declaration()}))

    monkeypatch.setattr(owner, "_release", revoke_after_release)
    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(tmp_path), session_id="own",
        orchestrator=ApprovalOrchestrator(Human())))
    try:
        if mode == "query":
            result = await registry.tool_handlers["model_query"]("specialist", {
                "goal": "choose", "candidates": [{"id": "a", "description": "eligible"}]})
        else:
            result = await check_fixture(owner, "specialist", fixture(tmp_path / "case.json"))
        assert result["failure_code"] == "model_service_disabled" and "result" not in result
        assert len(requests) == 1 and not owner._handles
    finally:
        reset_tool_context(token)
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["network.fetch", "file.read"])
async def test_disable_during_artifact_consent_never_reads_or_sends(tmp_path, monkeypatch, boundary):
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path / "data"))
    settings = load_config()
    cache = CacheManager(settings.profile_layout.cache, profile_id=settings.profile)
    entry = session_artifact(cache, tmp_path, "own", "image.jpg", b"\xff\xd8\xfffixture")
    owner, registry, _, _, requests = compose({"selected_id": "a", "rationale": "eligible"}, processing_location="local")
    accessed = []
    original = CacheManager.read_session_entry

    def record_read(*args, **kwargs):
        accessed.append("image")
        return original(*args, **kwargs)

    monkeypatch.setattr(CacheManager, "read_session_entry", record_read)

    class RevokingHuman(Human):
        async def request_approval(self, request):
            if request.action.kind == boundary:
                owner.publish(parse_services({"specialist": declaration(enabled=False, processing_location="local")}))
                owner.publish(parse_services({"specialist": declaration(processing_location="local")}))
            return await super().request_approval(request)

    human = RevokingHuman()
    registry.bind_runtime(settings=settings)
    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(tmp_path), session_id="own",
        orchestrator=ApprovalOrchestrator(human)))
    try:
        result = await registry.tool_handlers["model_query"]("specialist", {
            "goal": "choose", "candidates": [{"id": "a", "description": "eligible"}]}, {"image": entry.entry_id})
        assert result["failure_code"] == "model_service_disabled", result
        assert any(r.action.kind == boundary for r in human.requests)
        assert not requests and not accessed and not owner._handles
    finally:
        reset_tool_context(token)
        await owner.aclose()


@pytest.mark.asyncio
async def test_fixture_disable_during_file_approval_never_reads(tmp_path, monkeypatch):
    from leapflow.tools import model_fixture

    owner, _, _, _, requests = compose({"selected_id": "a", "rationale": "eligible"})
    accessed = []
    monkeypatch.setattr(model_fixture, "_read_fixture", lambda *args: accessed.append("read"))

    class RevokingGate(ApprovalOrchestrator):
        async def evaluate(self, action):
            decision = await super().evaluate(action)
            if action.kind == "file.read":
                owner.publish(parse_services({"specialist": declaration(enabled=False)}))
            return decision

    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(tmp_path), session_id="own",
        orchestrator=RevokingGate(Human())))
    try:
        result = await check_fixture(owner, "specialist", fixture(tmp_path / "case.json"))
        assert result["failure_code"] == "model_service_disabled"
        assert not accessed and not requests and not owner._handles
    finally:
        reset_tool_context(token)
        await owner.aclose()


@pytest.mark.asyncio
async def test_query_real_approval_and_registry(tmp_path):
    owner, registry, scoped, plugin, requests = compose({"selected_id": "a", "rationale": "best supplied choice"})
    human = Human()
    gate = ApprovalOrchestrator(human)
    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(tmp_path), session_id="own", orchestrator=gate))
    try:
        result = await registry.tool_handlers["model_query"]("specialist", {
            "goal": "choose", "candidates": [{"id": "a", "description": "eligible"}]})
        assert result["ok"] and result["provenance"]["physical_authority"] is False
        assert result["provenance"]["scores_calibrated"] is False
        assert not requests[0].get("stream", False)
        envelope = json.loads(requests[0]["messages"][-1]["content"][0]["text"])
        assert envelope["context"]["session_id"] == "own"
        assert gate.audit.entries
        owner.publish(())
        registry.unregister_tools(registry.tool_owners)
        assert plugin.tools == []
    finally:
        reset_tool_context(token)
        await scoped.get_fiber(plugin.plugin_id).scope.async_dispose()
        await owner.aclose()


@pytest.mark.asyncio
async def test_denial_and_unknown_id_never_execute(tmp_path):
    owner, registry, _, _, requests = compose({"selected_id": "unknown", "rationale": "invalid"})
    gate = ApprovalOrchestrator(Human(allow=False))
    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(tmp_path), session_id="s", orchestrator=gate))
    try:
        result = await registry.tool_handlers["model_query"]("specialist", {
            "goal": "choose", "candidates": [{"id": "a", "description": "eligible"}]})
        assert result["failure_code"] == "permission_denied" and "User denied" in result["error"]
        assert not requests
        gate = ApprovalOrchestrator(Human())
        reset_tool_context(token)
        token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(tmp_path), session_id="s", orchestrator=gate))
        result = await registry.tool_handlers["model_query"]("specialist", {
            "goal": "choose", "candidates": [{"id": "a", "description": "eligible"}]})
        assert result["failure_code"] == "model_invalid_output"
        assert len(requests) == 1
        assert not owner._handles
    finally:
        reset_tool_context(token)
        await owner.aclose()


def fixture(path, *, data=None, artifacts=None, scope="synthetic", expected=None):
    path.write_text(json.dumps({"schema_version": 1, "artifact_scope": scope,
        "input": data or {"goal": "choose", "candidates": [{"id": "a", "description": "eligible"}]},
        "artifacts": artifacts or {}, "expected": expected if expected is not None else {
            "type": "object", "properties": {"selected_id": {"enum": ["a"]}},
            "required": ["selected_id"]}}))
    return path


@pytest.mark.asyncio
async def test_fixture_uses_real_provider_and_is_not_qualification(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path / "profile-data"))
    monkeypatch.setenv("LEAPFLOW_WORKSPACE_ROOT", str(tmp_path))
    settings = load_config()
    owner, _, _, _, requests = compose({"selected_id": "a", "rationale": "eligible"})
    gate = ApprovalOrchestrator(Human())
    fixture(tmp_path / "case.json")
    try:
        result = await execute_fixture(settings, parse_service_arguments([
            "check", "specialist", "--case", "case.json"]), client=owner, gate=gate, session_id="s")
        assert result["ok"], result
        assert result["stages"]["fixture_assertions"] == "pass"
        assert result["stages"]["identity_provenance"] == "not_qualified"
        assert result["stages"]["execution_consent"] == "not_requested"
        assert not result["physical_authority"]
        assert len(requests) == 1 and not owner._handles and not owner._closed
        assert gate.audit.entries
        result = await execute_fixture(settings, parse_service_arguments([
            "check", "specialist", "--case", "case.json"]), client=owner)
        assert result["failure_code"] == "permission_denied" and len(requests) == 1
    finally:
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("expected", [
    {"properties": []}, {"required": "selected_id"}, {"required": [[]]},
    {"items": []}, {"minItems": True}, {"minimum": "bad"},
    {"enum": []}, {"enum": "a"}, {"additionalProperties": {}},
    {"minLength": 5, "maxLength": 1}, {"$ref": "file:///unapproved"},
    {"type": []}, {"type": {}}, {"type": ""},
])
async def test_fixture_invalid_assertions_never_reach_endpoint(tmp_path, expected):
    owner, _, _, _, requests = compose({"selected_id": "a", "rationale": "eligible"})
    gate = ApprovalOrchestrator(Human())
    token = set_tool_context(ToolExecutionContext.from_strings(
        workspace_root=str(tmp_path), session_id="s", orchestrator=gate))
    try:
        result = await check_fixture(owner, "specialist", fixture(tmp_path / "case.json", expected=expected))
        assert result["failure_code"] == "model_invalid_fixture", result
        assert result["failed_stage"] == "fixture_input" and not requests and not owner._handles
    finally:
        reset_tool_context(token)
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope,allow,code", [
    ("recorded", True, "model_privacy_unknown"), ("synthetic", False, "permission_denied"),
])
async def test_fixture_privacy_and_denial_are_terminal(tmp_path, scope, allow, code):
    owner, _, _, _, requests = compose({"selected_id": "a", "rationale": "eligible"})
    gate = ApprovalOrchestrator(Human(allow=allow))
    token = set_tool_context(ToolExecutionContext.from_strings(
        workspace_root=str(tmp_path), session_id="s", orchestrator=gate))
    try:
        result = await check_fixture(owner, "specialist", fixture(tmp_path / "case.json", scope=scope))
        assert result["failure_code"] == code, result
        assert not requests and not owner._handles
        if not allow:
            assert "User denied" in result["error"]
    finally:
        reset_tool_context(token)
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("valid_depth", [True, False])
async def test_grasp_fixture_has_explicit_bounded_depth(tmp_path, valid_depth):
    owner, _, _, _, requests = compose({"grasps": []}, task="grasp_proposal", processing_location="local")
    gate = ApprovalOrchestrator(Human())
    token = set_tool_context(ToolExecutionContext.from_strings(
        workspace_root=str(tmp_path), session_id="s", orchestrator=gate))
    data = {"depth": {"scale_m": .001, "data": base64.b64encode(b"\0\0" if valid_depth else b"\0").decode()},
            "intrinsics": {"width": 1, "height": 1, "fx": 1, "fy": 1, "ppx": 0, "ppy": 0},
            "object": {"center": [0, 0, 100]}, "gripper": {"max_width_mm": 80}, "top_k": 1}
    image = {"media_type": "image/jpeg", "data": base64.b64encode(b"\xff\xd8\xfffixture").decode()}
    try:
        result = await check_fixture(owner, "specialist", fixture(tmp_path / "case.json", data=data,
                                    artifacts={"image": image}, expected={"properties": {"grasps": {"maxItems": 0}}}))
        assert result["ok"] is valid_depth, result
        assert len(requests) == int(valid_depth) and not owner._handles
    finally:
        reset_tool_context(token)
        await owner.aclose()


@pytest.mark.asyncio
async def test_fixture_approval_wait_does_not_expire_inference_budget(tmp_path):
    class DeliberatingHuman(Human):
        async def request_approval(self, request):
            await asyncio.sleep(.08)
            return await super().request_approval(request)

    owner, _, _, _, requests = compose({"selected_id": "a", "rationale": "eligible"}, limits={"deadline_s": .06})
    human = DeliberatingHuman()
    gate = ApprovalOrchestrator(human)
    token = set_tool_context(ToolExecutionContext.from_strings(
        workspace_root=str(tmp_path), session_id="s", orchestrator=gate))
    try:
        result = await check_fixture(owner, "specialist", fixture(tmp_path / "case.json"))
        assert result["ok"], result
        assert human.requests and len(requests) == 1
    finally:
        reset_tool_context(token)
        await owner.aclose()


@pytest.mark.parametrize("task,data,result", [
    ("decision", {"candidates": [{"id": "a"}]}, {"selected_id": "a", "ranking": ["a", "a"]}),
    ("prediction", {"candidates": [{"id": "a"}], "horizon_s": 1}, {"predictions": []}),
    ("prediction", {"candidates": [{"id": "a"}], "horizon_s": 1},
     {"predictions": [{"candidate_id": "a", "horizon_s": 2}]}),
    ("evaluation", {"evidence": [{"id": "a"}]}, {"evidence_ids": ["other"]}),
])
def test_relational_validation(task, data, result):
    with pytest.raises(ValueError):
        validate_result(task, data, result)


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", [ApprovalDecision.ALLOW_SESSION, ApprovalDecision.ALLOW_ALL_SESSION])
async def test_shared_gate_keeps_specialist_consent_in_caller_session(tmp_path, choice):
    class ScopedHuman(Human):
        async def request_approval(self, request):
            self.requests.append(request)
            return choice if request.action.session_id == "a" else ApprovalDecision.DENY

    owner, registry, _, _, requests = compose({"selected_id": "a", "rationale": "eligible"})
    human = ScopedHuman()
    wrapper = SessionAwareGate(human)
    gate = ApprovalOrchestrator(wrapper)
    try:
        for session, approved in (("a", True), ("b", False), ("a", True)):
            token = set_tool_context(ToolExecutionContext.from_strings(
                workspace_root=str(tmp_path / session), session_id=session, orchestrator=gate))
            try:
                result = await registry.tool_handlers["model_query"]("specialist", {
                    "goal": "choose", "candidates": [{"id": "a", "description": "eligible"}]})
                assert result["ok"] is approved, result
            finally:
                reset_tool_context(token)
        assert len(requests) == 2
        assert [request.action.session_id for request in human.requests] == ["a", "b"]
        wrapper.reset()
        assert not wrapper._session_bypasses and not wrapper.approved_categories
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_fixture_cancellation_during_consent_releases_only_its_handle(tmp_path):
    waiting = asyncio.Event()

    class WaitingHuman(Human):
        async def request_approval(self, request):
            waiting.set()
            await asyncio.Event().wait()

    owner, _, _, _, requests = compose({"selected_id": "a", "rationale": "eligible"})
    path = fixture(tmp_path / "case.json")
    other = await owner.acquire("specialist")
    token = set_tool_context(ToolExecutionContext.from_strings(
        workspace_root=str(tmp_path), session_id="a", orchestrator=ApprovalOrchestrator(WaitingHuman())))
    try:
        pending = asyncio.create_task(check_fixture(owner, "specialist", path))
        await waiting.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert owner._handles == {other} and not requests
        assert not other._closed
    finally:
        reset_tool_context(token)
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload,code,stage", [
    ({"selected_id": "other", "rationale": "invalid"}, "model_invalid_output", "task_result_schema"),
    ({"selected_id": "a", "rationale": "eligible"}, "model_fixture_mismatch", "fixture_assertions"),
])
async def test_fixture_failures_identify_validation_stage(tmp_path, payload, code, stage):
    owner, _, _, _, requests = compose(payload)
    gate = ApprovalOrchestrator(Human())
    token = set_tool_context(ToolExecutionContext.from_strings(
        workspace_root=str(tmp_path), session_id="a", orchestrator=gate))
    try:
        path = fixture(tmp_path / "case.json", expected={"properties": {"selected_id": {"enum": ["b"]}}})
        result = await check_fixture(owner, "specialist", path)
        assert result["failure_code"] == code and result["failed_stage"] == stage, result
        assert len(requests) == 1 and not owner._handles
    finally:
        reset_tool_context(token)
        await owner.aclose()


def test_artifact_session_ttl_checksum_and_quota(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path / "data"))
    settings = load_config()
    cache = CacheManager(settings.profile_layout.cache, profile_id=settings.profile)
    workspace = workspace_id_for_path(tmp_path)
    path = cache.path(scope="session", category="model", workspace_id=workspace, session_id="s") / "fixture.json"
    content = b'{"episode":"synthetic"}'
    path.write_bytes(content)
    entry = cache.register(path=path, scope="session", category="model", source="fixture",
                           workspace_id=workspace, session_id="s", content_hash=hashlib.sha256(content).hexdigest(),
                           expires_at=time.time() + 60, sensitive=True, syncable=False)
    assert cache.read_session_entry(entry, max_bytes=100) == content
    for session, budget in [("other", 100), ("s", 1)]:
        with pytest.raises(ValueError):
            cache.resolve_session_entry(entry.entry_id, workspace_id=workspace, session_id=session, max_bytes=budget)
    path.write_bytes(b'x' * len(content))
    with pytest.raises(ValueError, match="checksum"):
        cache.read_session_entry(entry, max_bytes=100)
    cache.register(path=path, scope="session", category="model", source="fixture", workspace_id=workspace,
                   session_id="s", content_hash=entry.content_hash, expires_at=time.time() - 1, sensitive=True, syncable=False)
    with pytest.raises(ValueError, match="expired"):
        cache.resolve_session_entry(entry.entry_id, workspace_id=workspace, session_id="s", max_bytes=100)


def session_artifact(cache, workspace, session, name, content, *, expires_at=None):
    workspace_id = workspace_id_for_path(workspace)
    path = cache.path(scope="session", category="model", workspace_id=workspace_id, session_id=session) / name
    path.write_bytes(content)
    return cache.register(path=path, scope="session", category="model", source="fixture",
        workspace_id=workspace_id, session_id=session, content_hash=hashlib.sha256(content).hexdigest(),
        expires_at=expires_at if expires_at is not None else time.time() + 120, sensitive=True, syncable=False)


@pytest.mark.asyncio
async def test_concurrent_model_queries_keep_artifacts_and_results_in_own_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path / "data"))
    settings = load_config()
    cache = CacheManager(settings.profile_layout.cache, profile_id=settings.profile)
    entries = {session: session_artifact(cache, tmp_path / session, session, "observation.json",
                                         json.dumps({"owner": session}).encode()) for session in ("a", "b")}
    owner, registry, _, _, requests = compose({"predictions": [{"candidate_id": "a",
        "outcome_summary": "hypothetical", "horizon_s": 1}]}, task="prediction",
        processing_location="local", limits={"max_inflight": 2})
    human = Human()
    gate = ApprovalOrchestrator(SessionAwareGate(human))
    registry.bind_runtime(settings=settings, model_approval_gate=gate)

    async def query(session, reference):
        token = set_tool_context(ToolExecutionContext.from_strings(
            workspace_root=str(tmp_path / session), session_id=session, orchestrator=gate))
        try:
            return await registry.tool_handlers["model_query"]("specialist", {
                "observation_summary": "recorded evidence", "horizon_s": 1,
                "candidates": [{"id": "a", "description": "eligible"}],
                "observation_refs": [reference]})
        finally:
            reset_tool_context(token)

    try:
        results = await asyncio.gather(*(query(s, e.entry_id) for s, e in entries.items()))
        assert all(r["ok"] and not r["provenance"]["physical_authority"] for r in results), results
        for request in requests:
            data = json.loads(request["messages"][-1]["content"][0]["text"])
            session = data["context"]["session_id"]
            assert data["artifact_evidence"] == {entries[session].entry_id: {"owner": session}}
            assert str(tmp_path) not in json.dumps(data)
        refused = await query("b", entries["a"].entry_id)
        assert refused["failure_code"] == "model_invalid_request"
        assert len(requests) == 2 and not owner._handles
        assert {r.action.session_id for r in human.requests} == {"a", "b"}
    finally:
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["expiry", "checksum", "catalog"])
async def test_artifact_change_during_consent_refuses_without_provider_call(tmp_path, monkeypatch, change):
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path / "data"))
    settings = load_config()
    cache = CacheManager(settings.profile_layout.cache, profile_id=settings.profile)
    content = b'{"owner":"a"}'
    entry = session_artifact(cache, tmp_path, "a", "observation.json", content)
    owner, registry, _, _, requests = compose({"predictions": [{"candidate_id": "a",
        "outcome_summary": "hypothetical", "horizon_s": 1}]}, task="prediction", processing_location="local")

    class ChangingHuman(Human):
        async def request_approval(self, request):
            if request.action.kind == "file.read":
                if change == "checksum":
                    entry.path.write_bytes(b'x' * len(content))
                else:
                    session_artifact(cache, tmp_path, "a", "observation.json", content,
                                     expires_at=0 if change == "expiry" else time.time() + 200)
            return await super().request_approval(request)

    human = ChangingHuman()
    gate = ApprovalOrchestrator(human)
    registry.bind_runtime(settings=settings, model_approval_gate=gate)
    token = set_tool_context(ToolExecutionContext.from_strings(
        workspace_root=str(tmp_path), session_id="a", orchestrator=gate))
    try:
        result = await registry.tool_handlers["model_query"]("specialist", {
            "observation_summary": "recorded evidence", "horizon_s": 1,
            "candidates": [{"id": "a", "description": "eligible"}],
            "observation_refs": [entry.entry_id]})
        assert result["failure_code"] == "model_invalid_request", result
        assert any(r.action.kind == "file.read" for r in human.requests)
        assert not requests and not owner._handles
    finally:
        reset_tool_context(token)
        await owner.aclose()


def test_session_artifact_requires_checksum_and_rejects_parent_symlinks(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path / "data"))
    settings = load_config()
    cache = CacheManager(settings.profile_layout.cache, profile_id=settings.profile)
    entry = session_artifact(cache, tmp_path, "a", "observation.json", b'{}')
    link = tmp_path / "linked-directory"
    link.symlink_to(entry.path.parent, target_is_directory=True)
    with pytest.raises(OSError):
        cache._open_session_file(link / entry.path.name)
    unhashed = cache.register(path=entry.path, scope="session", category="model", source="fixture",
                   workspace_id=entry.workspace_id, session_id="a", expires_at=time.time() + 60,
                   sensitive=True, syncable=False)
    with pytest.raises(ValueError, match="checksum-bound"):
        cache.resolve_session_entry(unhashed.entry_id, workspace_id=entry.workspace_id, session_id="a", max_bytes=100)


def test_identical_artifacts_keep_independent_session_and_path_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path / "data"))
    settings = load_config()
    cache = CacheManager(settings.profile_layout.cache, profile_id=settings.profile)
    entries = [session_artifact(cache, tmp_path / workspace, session, name, b'{}')
               for workspace, session, name in (("one", "a", "one.json"), ("one", "b", "one.json"),
                                                 ("two", "a", "one.json"), ("one", "a", "two.json"))]
    assert len({e.entry_id for e in entries}) == len(entries)
    reopened = CacheManager(settings.profile_layout.cache, profile_id=settings.profile)
    for entry in entries:
        admitted = reopened.resolve_session_entry(entry.entry_id, workspace_id=entry.workspace_id,
                                                  session_id=entry.session_id, max_bytes=100)
        assert admitted == entry and reopened.read_session_entry(admitted, max_bytes=100) == b'{}'
    first = entries[0]
    refreshed = session_artifact(cache, tmp_path / "one", "a", "one.json", b'{}', expires_at=0)
    assert refreshed.entry_id == first.entry_id
    with pytest.raises(ValueError, match="expired"):
        reopened.resolve_session_entry(first.entry_id, workspace_id=first.workspace_id, session_id="a", max_bytes=100)
    for entry in entries[1:]:
        assert reopened.read_session_entry(entry, max_bytes=100) == b'{}'


@pytest.mark.asyncio
async def test_action_query_and_upload_bypass_refuse(tmp_path):
    owner, registry, _, _, requests = compose({}, task="action_chunk")
    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(tmp_path), session_id="s"))
    try:
        result = await registry.tool_handlers["model_query"]("specialist", {})
        assert not result["ok"] and "policy.run" in result["error"]
        assert not requests
        owner.publish(parse_services({"specialist": declaration()}))
        result = await registry.tool_handlers["model_query"]("specialist", {
            "goal": "choose", "candidates": [{"id": "a", "description": "a"}], "_images": ["unapproved"]})
        assert not result["ok"] and not requests
    finally:
        reset_tool_context(token)
        await owner.aclose()
