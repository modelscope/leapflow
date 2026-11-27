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


def compose(payload, *, task="decision", transport_error=None, **changes):
    providers = LLMProviderRegistry()
    providers.discover_builtin()
    requests = []

    async def respond(request):
        requests.append(json.loads(request.content))
        if transport_error is not None:
            raise transport_error
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
async def test_disable_during_artifact_consent_never_reads_or_sends(tmp_path, monkeypatch):
    boundary = "network.fetch"
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
        from leapflow.llm.services.protocol import canonical
        assert result["consultation"]["result_digest"] == hashlib.sha256(canonical(result["result"]).encode()).hexdigest()
        from leapflow.engine.context.context_control import ToolEvidenceBuilder
        from leapflow.engine.context.result_budget import fit_result_to_budget
        fitted = json.loads(fit_result_to_budget(ToolEvidenceBuilder().build("model_query", {}, result), 819))
        assert fitted["result"] == result["result"]
        assert fitted["provenance"]["scores_calibrated"] is False
        assert fitted["provenance"]["physical_authority"] is False
        assert fitted["consultation"]["call_id"] == result["consultation"]["call_id"]
        assert "attempts" in result["consultation"], "budget fitting cannot mutate audit evidence"
        assert not requests[0].get("stream", False)
        envelope = json.loads(requests[0]["messages"][-1]["content"][0]["text"])
        assert envelope["context"]["session_id"] == "own"
        assert gate.audit.entries
        timing = result["consultation"]["timing"]
        assert {"acquisition_ms", "approval_wait_ms", "encoding_ms", "network_inference_ms",
                "validation_ms", "cleanup_ms"} <= timing.keys()
        assert all(value >= 0 for value in timing.values())
        assert timing["network_inference_ms"] == sum(
            item["elapsed_ms"] for item in result["consultation"]["attempts"])
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
            if request.action.kind == "network.fetch":
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
        assert len(human.requests) == 1 and human.requests[0].action.kind == "network.fetch"
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


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["allow", "deny", "invalid", "timeout", "cancel"])
async def test_query_phase_timing_retains_failed_and_cancelled_consultations(tmp_path, mode):
    from leapflow.domain.events import RecordScope
    from leapflow.llm.services.telemetry import capture_inference

    waiting = asyncio.Event()

    class DeliberatingHuman(Human):
        async def request_approval(self, request):
            waiting.set()
            if mode == "cancel":
                await asyncio.Event().wait()
            await asyncio.sleep(.02)
            return await super().request_approval(request)

    owner, registry, _, _, requests = compose(
        {"selected_id": "foreign" if mode == "invalid" else "a", "rationale": "advisory"},
        transport_error=httpx.ReadTimeout("offline timeout") if mode == "timeout" else None)
    scope = RecordScope("", workspace_id_for_path(tmp_path), "own")
    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(tmp_path),
        session_id="own", orchestrator=ApprovalOrchestrator(DeliberatingHuman(allow=mode != "deny"))))
    try:
        with capture_inference(scope=scope) as capture:
            pending = asyncio.create_task(registry.tool_handlers["model_query"]("specialist", {
                "goal": "choose", "candidates": [{"id": "a", "description": "eligible"}]}))
            await asyncio.wait_for(waiting.wait(), timeout=2)
            if mode == "cancel":
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                payload = await pending
                assert payload["ok"] is (mode == "allow"), payload
        (consultation,) = capture.consultations
        timing = consultation.to_dict()["timing"]
        assert {"acquisition_ms", "approval_wait_ms", "cleanup_ms"} <= timing.keys()
        assert all(value >= 0 for value in timing.values())
        assert not owner._handles
        if mode in {"deny", "cancel"}:
            assert not requests and "network_inference_ms" not in timing
            assert "encoding_ms" not in timing and "validation_ms" not in timing
        else:
            assert len(requests) == 1
            assert timing["network_inference_ms"] == sum(item.elapsed_ms for item in capture.attempts)
            assert ("validation_ms" in timing) is (mode != "timeout")
        if mode != "cancel":
            assert timing["approval_wait_ms"] >= 10
        assert consultation.status == ("cancelled" if mode == "cancel" else
                                       "eligible" if mode == "allow" else "refused")
    finally:
        reset_tool_context(token)
        await owner.aclose()


class NovelTaskContract:
    task = "test.label_choice.v1"
    revision = "test.v1"
    classification = "read_only"
    description = "Select exactly one caller-declared label; advisory only."

    def schemas(self):
        return ({"type": "object", "properties": {"labels": {"type": "array", "minItems": 1,
                    "items": {"type": "string"}}}, "required": ["labels"], "additionalProperties": False},
                {"type": "object", "properties": {"selected_id": {"type": "string"}},
                 "required": ["selected_id"], "additionalProperties": False})

    def validate_input(self, request):
        if len(request["labels"]) != len(set(request["labels"])):
            raise ValueError("Labels must be unique")

    def validate_result(self, request, result):
        if result["selected_id"] not in request["labels"]:
            raise ValueError("Selected label must be caller-eligible")

    def evidence_references(self, request):
        return {}


class NovelContractPlugin:
    plugin_id = "test_contract_plugin"
    category = "model"
    dependencies = ["model_services", "effect_scope"]
    tools = []

    def __init__(self):
        self.scope = None
        self.owner = None
        self.registered = False

    def bind_runtime(self, **deps):
        self.scope = deps.get("effect_scope", self.scope)
        self.owner = deps.get("model_services", self.owner)
        if self.scope is not None and self.owner is not None and not self.registered:
            from leapflow.llm.services.codec import JsonCodec
            self.scope.effect(self.owner.register_contract(NovelTaskContract()))
            self.scope.effect(self.owner.register_codec("test.json.v1", JsonCodec()))
            self.registered = True


@pytest.mark.asyncio
async def test_new_contract_and_codec_plugin_share_query_fixture_and_cli_admission(tmp_path, monkeypatch):
    from leapflow.config_service import ConfigService
    from leapflow.tools.model_fixture import check_inline_fixture
    from leapflow.cli.commands.model_service_config import execute_service_config

    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path / "home"))
    settings = load_config()
    declared = declaration(task="test.label_choice.v1", codec="test.json.v1")
    owner, registry, scoped, tools_plugin, requests = compose({"selected_id": "a"},
        task="test.label_choice.v1", codec="test.json.v1")
    contract_plugin = NovelContractPlugin()
    registry.register(contract_plugin)
    scoped.adopt_existing_plugins()
    registry.bind_runtime(model_services=owner)
    registry.publish_plugin_tools_atomic(tools_plugin)
    service = ConfigService(settings, model_services=owner)
    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(tmp_path), session_id="s",
                            orchestrator=ApprovalOrchestrator(Human())))
    try:
        assert service.set_model_service("specialist", declared).ok
        checked = execute_service_config(service, parse_service_arguments(["check", "specialist"]))
        assert checked["services"][0]["usable"] and not checked["services"][0]["endpoint_probed"]
        described = await registry.tool_handlers["model_describe"]("specialist")
        assert "labels" in described["input_schema"]["properties"]
        result = await registry.tool_handlers["model_query"]("specialist", {"labels": ["a", "b"]})
        assert result["ok"] and result["provenance"]["contract_revision"] == "test.v1"
        case = {"schema_version": 1, "artifact_scope": "synthetic", "input": {"labels": ["a", "b"]},
                "expected": {"properties": {"selected_id": {"enum": ["a"]}}}}
        assert (await check_inline_fixture(owner, "specialist", case))["ok"]
        invalid = await registry.tool_handlers["model_query"]("specialist", {"labels": ["a", "a"]})
        assert not invalid["ok"] and len(requests) == 2
        scoped.dispose_plugin(contract_plugin.plugin_id)
        assert not service.check_model_services("specialist")["services"][0]["usable"]
        refused = await registry.tool_handlers["model_query"]("specialist", {"labels": ["a"]})
        assert not refused["ok"] and len(requests) == 2 and not owner._handles
        with pytest.raises(ValueError, match="admission"):
            service.set_model_service("unavailable", declared)
    finally:
        reset_tool_context(token)
        await owner.aclose()


class UnifiedConsultationScript:
    """Script only the primary I/O boundary; dispatch, governance and admission are real."""

    def __init__(self, behavior, *, artifact="", consultation_input=None):
        self.behavior = behavior
        self.artifact = artifact
        self.consultation_input = consultation_input
        self.calls = []
        self.scopes = []
        self.catalogs = []

    async def achat(self, messages, **kwargs):
        from leapflow.llm.base import LLMChatResponse, ToolCallInfo
        from leapflow.llm.services.telemetry import current_capture

        self.calls.append(json.loads(json.dumps(messages)))
        self.scopes.append(current_capture().scope)
        self.catalogs.append(kwargs.get("tools", []))
        step = len(self.calls)
        if step == 1 or step == 3:
            call = ToolCallInfo(id=f"describe-{step}", name="model_describe", arguments={"service_id": "specialist"})
        elif step == 2 and self.behavior != "decline":
            args = {"service_id": "specialist", "input": {"goal": "Choose a safe next observation",
                "candidates": [{"id": "a", "description": "Inspect again"},
                               {"id": "b", "description": "Stop and request clarification"}]}}
            if self.consultation_input is not None:
                args["input"] = json.loads(json.dumps(self.consultation_input))
            if self.artifact:
                args["artifacts"] = {"evidence": self.artifact}
            call = ToolCallInfo(id="consult-2", name="model_query", arguments=args)
        else:
            return LLMChatResponse(content="I choose b independently." if self.behavior == "override"
                else "Read-only analysis completed without physical action.",
                usage={"prompt_tokens": 20, "completion_tokens": 5})
        return LLMChatResponse(content="", tool_calls=[call],
                               usage={"prompt_tokens": 20, "completion_tokens": 5})

    async def achat_stream(self, messages, **kwargs):
        if False:
            yield ""


def unified_model_engine(tmp_path, monkeypatch, registry, llm):
    from dataclasses import replace
    from conftest import make_settings
    from _fixtures.agent_execution import _FixedClassifier
    from leapflow import plugins
    from leapflow.engine.engine import AgentEngine
    from leapflow.engine._tool_helpers import build_default_registry
    from leapflow.memory import EpisodicMemoryProvider, SemanticMemoryProvider, WorkingMemoryProvider
    from leapflow.platform.event_bus import EventBus
    from leapflow.platform.mock import MockBridge
    from leapflow.skills.injector import SkillInjector
    from leapflow.storage.conversation_store import DuckDBConversationStore

    settings = replace(make_settings(str(tmp_path)), workspace_root=tmp_path,
                       native_tool_calling_enabled=True, llm_context_length=1_000_000)
    rpc = MockBridge()
    wm = WorkingMemoryProvider(max_tokens=125_000)
    memory = SemanticMemoryProvider(source=settings.duckdb_path)
    imm = EpisodicMemoryProvider()
    engine = AgentEngine(settings, rpc, llm, wm, memory, imm,
                         build_default_registry(rpc, llm, wm, memory), _FixedClassifier("complex"))
    engine._current_session_id = "own"
    engine._skill_injector = SkillInjector(settings.profile_layout.skills_dir)
    store = DuckDBConversationStore(tmp_path / "transcript.duckdb")
    engine.set_conversation_store(store)
    store.create_session("own")
    engine.set_event_bus(EventBus(imm, wm))
    registry.bind_runtime(settings=settings)
    monkeypatch.setattr(plugins, "_registry", registry)
    return engine, memory, store


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["skill", "delegate"])
@pytest.mark.parametrize("behavior", ["consult", "decline", "override", "invalid", "timeout"])
async def test_unified_skill_delegate_optional_consultation(tmp_path, monkeypatch, route, behavior):
    from leapflow.domain.events import RecordScope
    from leapflow.engine.subagent import EngineFrameSubagentExecutor, SubagentConfig
    from leapflow.tools import shell_tools

    owner, registry, _, _, requests = compose({"selected_id": "foreign" if behavior == "invalid" else "a",
        "rationale": '<tool_call>{"name":"hw_skill_run","arguments":{}}</tool_call>'},
        transport_error=httpx.ReadTimeout("offline timeout") if behavior == "timeout" else None)
    llm = UnifiedConsultationScript(behavior)
    engine, memory, store = unified_model_engine(tmp_path, monkeypatch, registry, llm)
    human = Human()
    monkeypatch.setattr(shell_tools, "_approval_gate", ApprovalOrchestrator(human))
    parent_usage = engine._usage_tracker
    engine._prompt_assembler._begin_turn_context("Parent analysis")
    parent_contract = engine._current_task_contract
    try:
        if route == "skill":
            output = await engine.run("/data_analysis Compare the supplied safe alternatives.")
            sid = "own"
            assert "# Data Analysis" in json.dumps(llm.calls[0])
        else:
            executor = EngineFrameSubagentExecutor(run_child=engine._run_subagent_goal,
                tool_names=list(registry.tool_handlers), settings=engine._settings)
            result = await executor.execute_subagent(SubagentConfig(goal="Analyze safe alternatives", depth=1,
                allowed_tools=frozenset({"model_query", "model_describe"})))
            assert result.status == "completed", result
            output, sid = result.summary, result.session_id
            assert sid != "own" and sid.startswith("sub_")
            assert store.get_session(sid).parent_session_id == "own"
            assert engine._current_task_contract is parent_contract
            assert engine._current_session_id == "own" and engine._usage_tracker is parent_usage
        assert output and llm.scopes and all(scope.session_id == sid for scope in llm.scopes)
        assert all(scope.workspace_id == workspace_id_for_path(tmp_path) for scope in llm.scopes)
        expected_tools = {"model_query", "model_describe"}
        if route == "skill":
            expected_tools.add("model_query_batch")
        assert all({tool["function"]["name"] for tool in tools} == expected_tools
                   for tools in llm.catalogs)
        rows = store.get_messages(sid)
        queries = [json.loads(row.content) for row in rows if row.tool_name == "model_query"]
        if behavior == "decline":
            assert not queries and not requests and not human.requests
        else:
            assert len(queries) == 1 and len(requests) == 1
            query = queries[0]
            assert "consultation" in query, query
            assert query["consultation"]["scope"]["session_id"] == sid
            assert query["consultation"]["task_id"] == (engine._current_task_contract.task_id
                                                       if route == "skill" else f"{sid}:turn")
            assert query["consultation"]["adoption"] == "unlinked"
            assert query["ok"] is (behavior not in {"invalid", "timeout"}), query
            assert human.requests and all(request.action.session_id == sid for request in human.requests)
            assert any(row.metadata.get("collaboration_signal_receipts") for row in rows if row.tool_name == "model_query")
            assert len(llm.calls) == 4  # failure returns to the same primary loop
        usage = parent_usage.combined_usage()
        assert usage["primary"]["calls"] == len(llm.calls)
        assert usage["specialist"]["calls"] == len(requests)
        assert len(parent_usage.inference_records()) == len(llm.calls) + len(requests)
        assert not any(row.tool_name == "hw_skill_run" for row in rows)
        if route == "delegate":
            scope = RecordScope(engine._settings.profile, workspace_id_for_path(tmp_path), "own")
            assert parent_usage.collaboration_feedback(scope)["records"] == []
            # Retention is idempotent and includes cancelled/failed boundaries too.
            before = parent_usage.combined_usage()
            for attempt in parent_usage.inference_records():
                parent_usage.record_inference(attempt)
            assert parent_usage.combined_usage() == before
    finally:
        store.close()
        memory.close()
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["skill", "delegate"])
@pytest.mark.parametrize("perturbation", ["expired", "contradictory", "unreachable", "order"])
async def test_replay_complex_observation_perturbations_keep_advice_non_authoritative(
    tmp_path, monkeypatch, route, perturbation,
):
    """Replay boundary tests, not semantic qualification or a physical task script."""
    from leapflow.engine.subagent import EngineFrameSubagentExecutor, SubagentConfig
    from leapflow.llm.services.telemetry import current_capture
    from leapflow.tools import shell_tools

    observations = {
        "expired": ["Replay: initial placement evidence has expired; current pose is unknown."],
        "contradictory": ["Replay view 1: the object is supported.", "Replay view 2: support is unverified."],
        "unreachable": ["Replay: the proposed outward corridor failed the workspace reachability gate."],
        "order": ["Replay: candidate ordering is presentation, not identity or execution priority."],
    }[perturbation]
    candidates = [{"id": "a", "description": "Inspect current geometry without motion"},
                  {"id": "b", "description": "Ask for independent evidence and refuse unsupported motion"}]
    if perturbation == "order":
        candidates.reverse()
    request = {"goal": "Reassess an uncertain pickup, exhibition, placement, and return goal; advisory only",
               "observations": observations, "candidates": candidates}
    owner, registry, _, _, requests = compose({"selected_id": "a", "rationale": "Advisory hypothesis only"},
        processing_location="local")

    class PerturbedPrimary(UnifiedConsultationScript):
        async def achat(self, messages, **options):
            if perturbation == "expired" and len(self.calls) == 1:
                scope = current_capture().scope
                cache = CacheManager(engine._settings.profile_layout.cache, profile_id=scope.profile_id)
                entry = session_artifact(cache, tmp_path, scope.session_id, "expired.json", b'{}', expires_at=0)
                self.artifact = entry.entry_id
            return await super().achat(messages, **options)

    llm = PerturbedPrimary("override", consultation_input=request)
    engine, memory, store = unified_model_engine(tmp_path, monkeypatch, registry, llm)
    human = Human()
    monkeypatch.setattr(shell_tools, "_approval_gate", ApprovalOrchestrator(human))
    try:
        if route == "skill":
            output = await engine.run("/data_analysis Assess uncertainty and safe alternatives; do not actuate.")
            sid = "own"
        else:
            result = await EngineFrameSubagentExecutor(run_child=engine._run_subagent_goal,
                tool_names=list(registry.tool_handlers), settings=engine._settings).execute_subagent(
                    SubagentConfig(goal="Assess uncertainty without actuation", depth=1,
                        allowed_tools=frozenset({"model_describe", "model_query"})))
            assert result.status == "completed"
            output, sid = result.summary, result.session_id
        assert "I choose b independently." in output
        rows = store.get_messages(sid)
        (query,) = [json.loads(row.content) for row in rows if row.tool_name == "model_query"]
        assert not any(row.tool_name == "hw_skill_run" for row in rows)
        assert len(llm.calls) == 4
        if perturbation == "expired":
            assert not query["ok"] and not requests and not human.requests
            assert query["failure_code"] == "model_invalid_request"
            assert "expired" in query["error"]
        else:
            assert query["ok"] and query["result"]["selected_id"] == "a"
            assert query["consultation"]["adoption"] == "unlinked"
            assert len(requests) == 1
            envelope = json.loads(requests[0]["messages"][-1]["content"][0]["text"])
            assert {key: envelope[key] for key in request} == request
            assert envelope["context"]["session_id"] == sid
            assert envelope["artifact_order"] == []
            assert query["provenance"]["physical_authority"] is False
        usage = engine._usage_tracker.combined_usage()
        assert usage["primary"]["calls"] == 4 and usage["specialist"]["calls"] == len(requests)
    finally:
        store.close()
        memory.close()
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("persist", [False, True])
async def test_cancelled_delegate_retains_cost_without_parent_authority(tmp_path, monkeypatch, persist):
    import asyncio
    from leapflow.domain.events import RecordScope
    from leapflow.engine.subagent import EngineFrameSubagentExecutor, SubagentConfig
    from leapflow.tools import shell_tools

    class PausedScript(UnifiedConsultationScript):
        def __init__(self):
            super().__init__("consult")
            self.paused = asyncio.Event()

        async def achat(self, messages, **kwargs):
            response = await super().achat(messages, **kwargs)
            if len(self.calls) == 3:
                self.paused.set()
                await asyncio.Event().wait()
            return response

    owner, registry, _, _, requests = compose({"selected_id": "a", "rationale": "inspect"})
    llm = PausedScript()
    engine, memory, store = unified_model_engine(tmp_path, monkeypatch, registry, llm)
    if not persist:
        engine.set_conversation_store(None)
    monkeypatch.setattr(shell_tools, "_approval_gate", ApprovalOrchestrator(Human()))
    engine._prompt_assembler._begin_turn_context("Parent analysis")
    parent_contract, parent_usage = engine._current_task_contract, engine._usage_tracker
    executor = EngineFrameSubagentExecutor(run_child=engine._run_subagent_goal,
        tool_names=list(registry.tool_handlers), settings=engine._settings)
    task = asyncio.create_task(executor.execute_subagent(SubagentConfig(goal="Read-only analysis", depth=1,
        allowed_tools=frozenset({"model_query", "model_describe"}))))
    try:
        await asyncio.wait_for(llm.paused.wait(), timeout=5)
        child_scope = llm.scopes[0]
        assert child_scope.session_id != "own"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert engine._current_session_id == "own" and engine._current_task_contract is parent_contract
        assert engine._usage_tracker is parent_usage and len(requests) == 1
        totals = parent_usage.combined_usage()
        assert totals["primary"]["calls"] == 3 and totals["specialist"]["calls"] == 1
        assert totals["primary"]["failed_attempts"] == 1
        assert all(attempt.scope == child_scope for attempt in parent_usage.inference_records())
        parent_scope = RecordScope(engine._settings.profile, workspace_id_for_path(tmp_path), "own")
        assert parent_usage.collaboration_feedback(parent_scope)["records"] == []
        assert parent_usage.consultation_reference(child_scope, f"{child_scope.session_id}:turn",
            next(attempt.call_id for attempt in parent_usage.inference_records() if attempt.source == "specialist")) is None
        if persist:
            assert store.get_session(child_scope.session_id).parent_session_id == "own"
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        store.close()
        memory.close()
        await owner.aclose()


@pytest.mark.asyncio
async def test_unified_delegate_cannot_borrow_parent_artifact(tmp_path, monkeypatch):
    from leapflow.engine.subagent import EngineFrameSubagentExecutor, SubagentConfig
    from leapflow.tools import shell_tools

    owner, registry, _, _, requests = compose({"selected_id": "a", "rationale": "inspect"}, processing_location="local")
    llm = UnifiedConsultationScript("consult")
    engine, memory, store = unified_model_engine(tmp_path, monkeypatch, registry, llm)
    cache = CacheManager(engine._settings.profile_layout.cache, profile_id=engine._settings.profile)
    entry = session_artifact(cache, tmp_path, "own", "observation.json", b'{"owner":"parent"}')
    llm.artifact = entry.entry_id
    human = Human()
    monkeypatch.setattr(shell_tools, "_approval_gate", ApprovalOrchestrator(human))
    try:
        executor = EngineFrameSubagentExecutor(run_child=engine._run_subagent_goal,
            tool_names=list(registry.tool_handlers), settings=engine._settings)
        result = await executor.execute_subagent(SubagentConfig(goal="Inspect evidence", depth=1,
            allowed_tools=frozenset({"model_query", "model_describe"})))
        query = next(json.loads(row.content) for row in store.get_messages(result.session_id)
                     if row.tool_name == "model_query")
        assert not query["ok"] and query["failure_code"] == "model_invalid_request"
        assert not requests and not human.requests  # feasibility before consent/read/HTTP
        assert result.status == "completed" and len(llm.calls) == 4
        assert all(scope.session_id != "own" for scope in llm.scopes)
    finally:
        store.close()
        memory.close()
        await owner.aclose()


@pytest.mark.asyncio
async def test_concurrent_unified_delegates_keep_parent_prompt_and_memory_private(tmp_path, monkeypatch):
    import asyncio
    from leapflow.engine.subagent import EngineFrameSubagentExecutor, SubagentConfig
    from leapflow.llm.base import LLMChatResponse
    from leapflow.llm.services.telemetry import current_capture

    class ConcurrentScript:
        model = "offline-primary"

        def __init__(self):
            self.calls = []
            self.both_started = asyncio.Event()

        async def achat(self, messages, **options):
            self.calls.append((json.dumps(messages), current_capture().scope))
            if len(self.calls) == 2:
                self.both_started.set()
            await asyncio.wait_for(self.both_started.wait(), timeout=3)
            return LLMChatResponse(content="Independent analysis complete.", usage={})

    owner, registry, _, _, requests = compose({"selected_id": "a", "rationale": "inspect"})
    llm = ConcurrentScript()
    engine, memory, store = unified_model_engine(tmp_path, monkeypatch, registry, llm)
    engine._prompt_assembler._begin_turn_context("Parent-only work")
    engine._wm.remember_chat({"role": "user", "content": "PARENT_PRIVATE_HISTORY"})
    engine._memory_context_snapshot = "PARENT_PRIVATE_RETRIEVAL"
    engine._frozen_system_prompt = "PARENT_PRIVATE_FROZEN_PREFIX"
    engine._frozen_tool_schema = json.dumps([{"type": "function", "function": {"name": "hw_skill_run"}}])
    parent_contract, parent_usage = engine._current_task_contract, engine._usage_tracker
    parent_messages = engine._wm.as_chat_messages()
    executor = EngineFrameSubagentExecutor(run_child=engine._run_subagent_goal,
        tool_names=list(registry.tool_handlers), settings=engine._settings)
    try:
        results = await asyncio.gather(*(executor.execute_subagent(SubagentConfig(goal=goal, depth=1,
            allowed_tools=frozenset({"model_describe"}))) for goal in ("Child alpha", "Child beta")))
        assert all(result.status == "completed" for result in results), results
        assert len({result.session_id for result in results}) == 2
        assert {scope.session_id for _, scope in llm.calls} == {result.session_id for result in results}
        assert len(llm.calls) == 2 and not requests
        assert all("PARENT_PRIVATE" not in prompt for prompt, _ in llm.calls)
        for result in results:
            assert store.get_session(result.session_id).parent_session_id == "own"
            assert store.get_messages(result.session_id)
        assert engine._current_session_id == "own" and engine._current_task_contract is parent_contract
        assert engine._usage_tracker is parent_usage
        assert parent_usage.combined_usage()["primary"]["calls"] == 2
        assert engine._wm.as_chat_messages() == parent_messages
        assert engine._memory_context_snapshot == "PARENT_PRIVATE_RETRIEVAL"
        assert engine._frozen_system_prompt == "PARENT_PRIVATE_FROZEN_PREFIX"
    finally:
        store.close()
        memory.close()
        await owner.aclose()


@pytest.mark.asyncio
async def test_failed_unified_delegate_reports_actual_persisted_child_identity(tmp_path, monkeypatch):
    from leapflow.engine.subagent import EngineFrameSubagentExecutor, SubagentConfig
    from leapflow.llm.base import LLMChatResponse
    from leapflow.llm.services.telemetry import current_capture

    scopes = []

    class BrokenAdapter:
        model = "offline-primary"

        async def achat(self, messages, **options):
            scopes.append(current_capture().scope)
            return LLMChatResponse(content=42, usage={})  # Adapter violates the text contract.

    owner, registry, _, _, _ = compose({"selected_id": "a", "rationale": "inspect"})
    engine, memory, store = unified_model_engine(tmp_path, monkeypatch, registry, BrokenAdapter())
    executor = EngineFrameSubagentExecutor(run_child=engine._run_subagent_goal,
        tool_names=list(registry.tool_handlers), settings=engine._settings)
    try:
        result = await executor.execute_subagent(SubagentConfig(goal="Analyze", depth=1,
            allowed_tools=frozenset({"model_describe"})))
        assert result.status == "failed", result
        assert result.session_id == scopes[0].session_id
        assert store.get_session(result.session_id).parent_session_id == "own"
        assert engine._current_session_id == "own"
        assert engine._usage_tracker.combined_usage()["primary"]["calls"] == 1
        assert result.metadata["failure"]["category"] == "internal_defect"
        assert result.metadata["failure"]["side_effect_state"] == "unknown"
        wire = json.loads(json.dumps(result.metadata))
        assert wire["failure"]["source"] == "system"
        assert wire["failure"]["recoverability"] == "non_recoverable"
    finally:
        store.close()
        memory.close()
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_first", [False, True])
async def test_shared_delegate_plugin_uses_owning_session_runner(tmp_path, monkeypatch, cancel_first):
    import asyncio
    from leapflow.engine.context.context_disclosure import DisclosureLevel, PromptAssemblyPlan
    from leapflow.engine.session.session_factory import build_session_engine
    from leapflow.engine.subagent import EngineFrameSubagentExecutor, SubagentManager
    from leapflow.llm.base import LLMChatResponse
    from leapflow.llm.services.telemetry import current_capture
    from leapflow.memory import WorkingMemoryProvider
    from leapflow.plugins.tool_plugins.orchestration import OrchestrationPlugin

    class PausedPrimary:
        model = "offline-primary"

        def __init__(self):
            self.calls = []
            self.both_started = asyncio.Event()
            self.release = asyncio.Event()

        async def achat(self, messages, **options):
            self.calls.append((json.dumps(messages), current_capture().scope))
            if len(self.calls) == 2:
                self.both_started.set()
            await self.release.wait()
            return LLMChatResponse(content="Independent analysis complete.", usage={})

    owner, registry, _, _, requests = compose({"selected_id": "a", "rationale": "inspect"})
    llm = PausedPrimary()
    base, memory, store = unified_model_engine(tmp_path, monkeypatch, registry, llm)
    manager = SubagentManager(executor=EngineFrameSubagentExecutor(
        run_child=base._run_subagent_goal, tool_names=list(registry.tool_handlers), settings=base._settings))
    plugin = OrchestrationPlugin()
    registry.register(plugin)
    registry.bind_runtime(subagent_manager=manager)
    registry.publish_plugin_tools_atomic(plugin)
    registry.set_subagent_manager(manager)
    engines = [build_session_engine(base, session_id=sid,
        working_memory=WorkingMemoryProvider(max_tokens=512), workspace_root=tmp_path / sid)
        for sid in ("parent-a", "parent-b")]
    for engine in engines:
        store.create_session(engine._current_session_id)
        engine._prompt_assembler._begin_turn_context("Parent-private task")
    tasks = [asyncio.create_task(engine._tool_dispatch._execute_tool_scoped(
        {"name": "delegate_task", "arguments": {"goal": f"Analyze {engine._current_session_id}"}},
        registry.tool_handlers, tool_call_id=f"delegate-{index}"))
        for index, engine in enumerate(engines)]
    try:
        await asyncio.wait_for(llm.both_started.wait(), timeout=5)
        active = manager.get_active_state()["active"]
        assert {item["parent_session_id"] for item in active} == {"parent-a", "parent-b"}
        plan = PromptAssemblyPlan(level=DisclosureLevel.EXPANDED)
        for engine in engines:
            section = engine._prompt_assembler._active_subagent_status_section(plan)
            for item in active:
                assert (item["subagent_id"] in section) is (item["parent_session_id"] == engine._current_session_id)
        if cancel_first:
            engines[0].cancel()
            first = await asyncio.wait_for(tasks[0], timeout=5)
            assert first["status"] == "cancelled" and not first["ok"]
            assert not tasks[1].done()  # Cancellation never crosses to the other workspace.
        llm.release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
        assert results[1]["status"] == "completed"
        assert results[0]["status"] == ("cancelled" if cancel_first else "completed")
        assert len({result["session_id"] for result in results}) == 2
        for engine, result in zip(engines, results):
            sid = result["session_id"]
            assert store.get_session(sid).parent_session_id == engine._current_session_id
            assert engine._usage_tracker.combined_usage()["primary"]["calls"] == 1
            attempts = engine._usage_tracker.inference_records()
            assert len(attempts) == 1 and attempts[0].scope.session_id == sid
            assert attempts[0].scope.workspace_id == workspace_id_for_path(engine._settings.workspace_root)
        assert base._current_session_id == "own"
        assert not base._usage_tracker.inference_records() and not base._child_tasks
        assert not requests and not manager.has_active()
    finally:
        llm.release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        store.close()
        memory.close()
        await owner.aclose()
