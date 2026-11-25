# Copyright (c) Alibaba, Inc. and its affiliates.
"""Service-bound consent, persistent revocation, and owned-evidence boundaries."""
from __future__ import annotations

import asyncio
import json
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
from prompt_toolkit.document import Document

from leapflow.cache.manager import CacheManager
from leapflow.cli.commands.config import _cmd_llm
from leapflow.cli.commands.model_service_config import execute_service_config, parse_service_arguments
from leapflow.cli.commands.slash_handlers import build_config_payload, build_config_payload_async
from leapflow.config import load_config
from leapflow.config_service import ConfigService
from leapflow.llm.services.config import parse_services
from leapflow.security.actions import ActionDescriptor
from leapflow.security.approval import ApprovalDecision, ApprovalRequest, SessionAwareGate
from leapflow.security.grants import ApprovalAuditLog, ApprovalGrant, JsonApprovalGrantStore, grant_key
from leapflow.security.network import NetworkTarget
from leapflow.security.orchestrator import ApprovalOrchestrator
from leapflow.security.risk import DefaultRiskClassifier, RiskLevel
from leapflow.tools.execution_context import ToolExecutionContext, reset_tool_context, set_tool_context
from leapflow.tools.model_tools import authorize_endpoint
from test_model_services import declaration
from test_model_tools import Human, compose, session_artifact

pytestmark = pytest.mark.component

INPUT = {"goal": "choose", "candidates": [{"id": "a", "description": "eligible"}]}
RESULT = {"selected_id": "a", "rationale": "eligible"}
IMAGE = b"\xff\xd8\xfffixture"


class ChoosingHuman(Human):
    def __init__(self, choice=ApprovalDecision.ALLOW_ALWAYS):
        super().__init__()
        self.choice = choice

    async def request_approval(self, request):
        self.requests.append(request)
        return self.choice


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LEAPFLOW_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    return load_config()


async def query(registry, gate, workspace, session, artifacts=None):
    token = set_tool_context(ToolExecutionContext.from_strings(
        workspace_root=str(workspace), session_id=session, task_id="turn", orchestrator=gate))
    try:
        return await registry.tool_handlers["model_query"]("specialist", INPUT, artifacts or {})
    finally:
        reset_tool_context(token)


def persistent_gate(settings, human):
    return ApprovalOrchestrator(SessionAwareGate(human),
        grants=JsonApprovalGrantStore(settings.profile_layout.approval.grants_path),
        audit=ApprovalAuditLog(settings.profile_layout.approval.audit_path))


@pytest.mark.asyncio
async def test_local_one_consent_covers_text_multiple_new_images_and_fresh_sessions(settings, tmp_path):
    owner, registry, _, _, sent = compose(RESULT, processing_location="local")
    registry.bind_runtime(settings=settings)
    cache = CacheManager(settings.profile_layout.cache, profile_id=settings.profile)
    human = ChoosingHuman()
    gate = persistent_gate(settings, human)
    workspace = tmp_path / "one"
    try:
        assert (await query(registry, gate, workspace, "a"))["ok"]
        for count in (2, 3):
            artifacts = {str(i): session_artifact(cache, workspace, "a", f"{count}-{i}.jpg", IMAGE).entry_id
                         for i in range(count)}
            assert (await query(registry, gate, workspace, "a", artifacts))["ok"]
        fresh_human = ChoosingHuman(ApprovalDecision.DENY)
        fresh = persistent_gate(settings, fresh_human)
        image = session_artifact(cache, tmp_path / "two", "b", "fresh.jpg", IMAGE)
        assert (await query(registry, fresh, tmp_path / "two", "b", {"image": image.entry_id}))["ok"]
        assert len(human.requests) == 1 and not fresh_human.requests and len(sent) == 4
        request = human.requests[0]
        assert request.risk.level == RiskLevel.MEDIUM and "allow_always" in request.choices
        assert request.action.metadata["approval_binding"]["evidence_scope"] == "text_and_evidence"
        assert "Loopback may be forwarded" in request.risk.explanation
        assert not gate._gate.approved_categories
        durable = json.dumps(gate.grants.list()[0].to_dict())
        audit = json.dumps(gate.audit.entries)
        for private in ("eligible", "choose", str(workspace), IMAGE.hex()):
            assert private not in durable and private not in audit
        assert gate.audit.entries[0]["actor"] == "user" and gate.audit.entries[-1]["actor"] == "grant"
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_remote_loopback_text_does_not_authorize_evidence_and_both_can_be_remembered(settings, tmp_path):
    owner, registry, _, _, sent = compose(RESULT, processing_location="remote")
    registry.bind_runtime(settings=settings)
    cache = CacheManager(settings.profile_layout.cache, profile_id=settings.profile)
    human = ChoosingHuman()
    gate = persistent_gate(settings, human)
    image = session_artifact(cache, tmp_path, "a", "image.jpg", IMAGE)
    try:
        assert (await query(registry, gate, tmp_path, "a"))["ok"]
        assert (await query(registry, gate, tmp_path, "a", {"image": image.entry_id}))["ok"]
        assert (await query(registry, gate, tmp_path, "a", {"image": image.entry_id}))["ok"]
        assert len(human.requests) == 2 and len(sent) == 3
        assert all(r.risk.level == RiskLevel.HIGH and "allow_always" in r.choices for r in human.requests)
        assert [r.action.metadata["approval_binding"]["evidence_scope"] for r in human.requests] == ["text", "evidence"]
        assert all("remote" in r.display["summary"] for r in human.requests)
        fresh = persistent_gate(settings, ChoosingHuman(ApprovalDecision.DENY))
        assert (await query(registry, fresh, tmp_path, "new"))["ok"]
        owned = session_artifact(cache, tmp_path, "new", "new.jpg", IMAGE)
        assert (await query(registry, fresh, tmp_path, "new", {"new": owned.entry_id}))["ok"]
        assert not fresh._gate._delegate.requests
    finally:
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("choice,expected", [(ApprovalDecision.ALLOW_ONCE, 3), (ApprovalDecision.ALLOW_SESSION, 2)])
async def test_once_and_session_keep_narrow_meanings(settings, tmp_path, choice, expected):
    owner, registry, _, _, _ = compose(RESULT, processing_location="local")
    human = ChoosingHuman(choice)
    gate = persistent_gate(settings, human)
    try:
        for session in ("a", "a", "b"):
            assert (await query(registry, gate, tmp_path, session))["ok"]
        assert len(human.requests) == expected
        assert all(g.scope != "profile" for g in gate.grants.list())
    finally:
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [
    {"model": "changed"}, {"base_url": "http://localhost:8081/v1"},
    {"processing_location": "remote"}, {"allow_remote": True},
    {"auth": {"mode": "secret", "ref": "secret://profile/models/new"}},
    {"request_options": {"temperature": 0.1}}, {"limits": {"max_inflight": 2}},
    {"prompt_template": "private-template-content"},
    {"identity": {"mode": "server_reported"},
     "model_info": {"name": "new", "version": "v1", "checksum": "digest", "task": "decision"},
     "deployment": {"engine_version": "v1", "adapter_digest": "adapter", "qualification_ref": "fixture"}},
])
async def test_all_declaration_changes_get_new_consent(settings, changes):
    human = ChoosingHuman()
    gate = persistent_gate(settings, human)
    original = declaration(processing_location="local")
    first = parse_services({"specialist": original})[0]
    second = parse_services({"specialist": {**original, **changes}})[0]
    await authorize_endpoint(first, gate, sensitive=False)
    await authorize_endpoint(second, gate, sensitive=False)
    assert len(human.requests) == 2
    assert human.requests[0].action.signature() != human.requests[1].action.signature()
    binding = json.dumps(gate.grants.list()[1].binding)
    assert "secret://" not in binding and "private-template-content" not in binding
    assert "private-template-content" not in json.dumps(gate.audit.entries)


@pytest.mark.asyncio
async def test_dns_destination_change_same_category_reprompts_and_wait_change_refuses(settings, monkeypatch):
    from leapflow.tools import model_tools

    address = "127.0.0.1"

    async def resolve(url):
        return NetworkTarget(url, "http", "localhost", 8080, "http://localhost:8080", "loopback", (address,))

    monkeypatch.setattr(model_tools, "classify_url", resolve)
    human = ChoosingHuman()
    gate = persistent_gate(settings, human)
    spec = parse_services({"specialist": declaration(processing_location="local")})[0]
    await authorize_endpoint(spec, gate, sensitive=False)
    address = "127.0.0.2"
    await authorize_endpoint(spec, gate, sensitive=False)
    assert len(human.requests) == 2

    class RebindingHuman(ChoosingHuman):
        async def request_approval(self, request):
            nonlocal address
            address = "127.0.0.4"
            return await super().request_approval(request)

    address = "127.0.0.3"
    from leapflow.llm.services.protocol import ServiceError
    with pytest.raises(ServiceError, match="destination changed"):
        await authorize_endpoint(spec, persistent_gate(settings, RebindingHuman()), sensitive=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["metadata", "link_local", "unspecified", "reserved", "unknown"])
async def test_restricted_targets_never_offer_or_accept_permanent_consent(settings, monkeypatch, category):
    from leapflow.tools import model_tools

    async def resolve(url):
        return NetworkTarget(url, "http", "localhost", 8080, "http://localhost:8080", category, ("127.0.0.1",))

    monkeypatch.setattr(model_tools, "classify_url", resolve)
    human = ChoosingHuman()
    gate = persistent_gate(settings, human)
    spec = parse_services({"specialist": declaration(processing_location="local")})[0]
    from leapflow.llm.services.protocol import ServiceError
    with pytest.raises(ServiceError):
        await authorize_endpoint(spec, gate, sensitive=False)
    assert "allow_always" not in human.requests[0].choices and not gate.grants.list()


@pytest.mark.asyncio
async def test_public_known_text_still_has_first_use_consent_and_credentials_have_no_exemption(settings, monkeypatch):
    from leapflow.tools import model_tools

    async def resolve(url):
        return NetworkTarget(url, "https", "example.test", 443, "https://example.test", "public", ("8.8.8.8",))

    monkeypatch.setattr(model_tools, "classify_url", resolve)
    human = ChoosingHuman()
    gate = persistent_gate(settings, human)
    spec = parse_services({"specialist": declaration(base_url="https://example.test/v1", allow_remote=True,
                                                    processing_location="remote")})[0]
    await authorize_endpoint(spec, gate, sensitive=False)
    assert len(human.requests) == 1 and "allow_always" in human.requests[0].choices
    action = human.requests[0].action
    risk = DefaultRiskClassifier().assess(replace(action, metadata={**action.metadata, "has_credentials": True}))
    assert risk.level == RiskLevel.HIGH and not risk.allow_permanent


@pytest.mark.asyncio
async def test_revoke_via_cli_and_slash_refreshes_running_gate_and_preserves_other_grants(settings, tmp_path, capsys):
    owner, registry, _, _, sent = compose(RESULT, processing_location="local")
    human = ChoosingHuman()
    gate = persistent_gate(settings, human)
    service = ConfigService(settings)
    service.set_model_service("specialist", declaration(processing_location="local"))
    unrelated = ApprovalGrant("other", "profile", "allow", "shell.command", "execute", resource="private-other")
    other_model = ApprovalGrant("other-model", "profile", "allow", "network.fetch", "read",
        binding={"platform": "model_services", "service_id": "other-service", "evidence_scope": "text"})
    gate.grants.put(unrelated)
    gate.grants.put(other_model)
    ctx = SimpleNamespace(settings=settings)
    try:
        assert (await query(registry, gate, tmp_path, "a"))["ok"]
        before = service.model_services()
        args = parse_service_arguments(["permissions", "specialist"])
        args.llm_action = "services"
        assert _cmd_llm(service, args, settings=settings) == 0
        visible = json.loads(capsys.readouterr().out)
        assert len(visible["permissions"]) == 1 and "private-other" not in json.dumps(visible)
        assert build_config_payload(ctx, "llm services permissions absent")["result"]["permissions"] == []
        revoked = await build_config_payload_async(ctx, "llm services revoke specialist")
        assert revoked["ok"] and revoked["result"]["revoked"] == 1
        assert service.model_services() == before
        human.choice = ApprovalDecision.DENY
        assert (await query(registry, gate, tmp_path, "a"))["failure_code"] == "permission_denied"
        assert len(sent) == 1 and len(human.requests) == 2
        assert {g.key for g in gate.grants.list()} == {"other", "other-model"}
        noop = execute_service_config(service, parse_service_arguments(["revoke", "specialist"]))
        assert noop["ok"] and noop["revoked"] == 0
        audit = [json.loads(line) for line in settings.profile_layout.approval.audit_path.read_text().splitlines()]
        assert any(e["decision"] == "revoke" and e["actor"] == "user" for e in audit)
        assert "other-service" in json.dumps(build_config_payload(ctx, "llm services permissions")["result"])
        args = parse_service_arguments(["revoke", "other-service"])
        args.llm_action = "services"
        assert _cmd_llm(service, args, settings=settings) == 0
        assert gate.grants.get("other-model") is None and gate.grants.get("other") == unrelated
    finally:
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("location,choice", [
    ("unknown", ApprovalDecision.ALLOW_ALWAYS), ("local", ApprovalDecision.DENY),
    ("local", ApprovalDecision.CANCEL_WORKFLOW),
])
async def test_unknown_or_denied_owned_evidence_never_reads_or_sends(settings, tmp_path, monkeypatch, location, choice):
    owner, registry, _, _, sent = compose(RESULT, processing_location=location)
    registry.bind_runtime(settings=settings)
    cache = CacheManager(settings.profile_layout.cache, profile_id=settings.profile)
    image = session_artifact(cache, tmp_path, "a", "image.jpg", IMAGE)
    read = []
    monkeypatch.setattr(CacheManager, "read_session_entry", lambda *args, **kwargs: read.append(True))
    human = ChoosingHuman(choice)
    try:
        result = await query(registry, persistent_gate(settings, human), tmp_path, "a", {"image": image.entry_id})
        assert not result["ok"] and not read and not sent
        if location == "unknown":
            assert result["failure_code"] == "model_privacy_unknown" and not human.requests
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_cancel_wait_before_evidence_read(settings, tmp_path, monkeypatch):
    entered = asyncio.Event()

    class WaitingHuman(Human):
        async def request_approval(self, request):
            entered.set()
            await asyncio.Event().wait()

    owner, registry, _, _, sent = compose(RESULT, processing_location="local")
    registry.bind_runtime(settings=settings)
    cache = CacheManager(settings.profile_layout.cache, profile_id=settings.profile)
    image = session_artifact(cache, tmp_path, "a", "image.jpg", IMAGE)
    read = []
    monkeypatch.setattr(CacheManager, "read_session_entry", lambda *args, **kwargs: read.append(True))
    try:
        task = asyncio.create_task(query(registry, persistent_gate(settings, WaitingHuman()), tmp_path, "a", {"i": image.entry_id}))
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not read and not sent and not owner._handles
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_remembered_permission_never_shares_artifacts_across_workspaces_or_sessions(settings, tmp_path, monkeypatch):
    owner, registry, _, _, sent = compose(RESULT, processing_location="local", limits={"max_inflight": 2})
    registry.bind_runtime(settings=settings)
    cache = CacheManager(settings.profile_layout.cache, profile_id=settings.profile)
    entries = {s: session_artifact(cache, tmp_path / s, s, "image.jpg", IMAGE + s.encode()) for s in ("a", "b")}
    human = ChoosingHuman()
    gate = persistent_gate(settings, human)
    try:
        assert (await query(registry, gate, tmp_path / "a", "a"))["ok"]
        fresh = persistent_gate(settings, ChoosingHuman(ApprovalDecision.DENY))
        results = await asyncio.gather(*(query(registry, fresh, tmp_path / s, s, {"i": e.entry_id}) for s, e in entries.items()))
        assert all(r["ok"] for r in results)
        assert len(sent) == 3 and not fresh._gate._delegate.requests
        for request in sent[1:]:
            contents = request["messages"][-1]["content"]
            envelope = json.loads(contents[0]["text"])
            session = envelope["context"]["session_id"]
            import base64
            image = base64.b64decode(contents[1]["image_url"]["url"].split(",", 1)[1])
            assert image == IMAGE + session.encode()
        reads = []
        monkeypatch.setattr(CacheManager, "read_session_entry", lambda *args, **kwargs: reads.append(True))
        for workspace, session in ((tmp_path / "b", "b"), (tmp_path / "b", "a")):
            result = await query(registry, fresh, workspace, session, {"i": entries["a"].entry_id})
            assert result["failure_code"] == "model_invalid_request"
        assert not reads and len(sent) == 3
    finally:
        await owner.aclose()


def _grant_writer(path, barrier, prefix):
    store = JsonApprovalGrantStore(path)
    barrier.wait(timeout=20)
    for index in range(8):
        store.put(ApprovalGrant(f"{prefix}-{index}", "profile", "allow", "shell.command", "execute"))


def test_store_process_and_thread_writers_refresh_remove_and_fail_closed(tmp_path):
    path = tmp_path / "grants.json"
    running = JsonApprovalGrantStore(path)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    processes = [context.Process(target=_grant_writer, args=(path, barrier, str(i))) for i in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert not process.is_alive() and process.exitcode == 0
    assert len(running.list()) == 16
    stores = [JsonApprovalGrantStore(path) for _ in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda pair: pair[1].put(ApprovalGrant(f"thread-{pair[0]}", "profile", "allow",
                      "shell.command", "execute")), enumerate(stores)))
    assert len(running.list()) == 24
    removed = stores[0].remove_where(lambda g: g.key.startswith("0-"))
    assert len(removed) == 8 and len(running.list()) == 16
    assert len(json.loads(path.read_text())) == 16
    # A previously opened writer must not resurrect revoked grants.
    stores[1].put(ApprovalGrant("later", "profile", "allow", "shell.command", "execute"))
    assert len(running.list()) == 17 and not any(g.key.startswith("0-") for g in running.list())
    path.write_text("invalid-json")
    assert not running.list() and running.get("thread-0") is None
    with pytest.raises(ValueError, match="refusing to overwrite"):
        running.put(ApprovalGrant("new", "profile", "allow", "shell.command", "execute"))
    assert path.read_text() == "invalid-json"


@pytest.mark.asyncio
async def test_legacy_gate_cache_is_preserved_but_orchestrator_ignores_it(settings):
    human = ChoosingHuman(ApprovalDecision.ALLOW_SESSION)
    wrapper = SessionAwareGate(human)
    action = replace(ActionDescriptor.shell("ssh example.test"), session_id="a")
    legacy = ApprovalRequest(category=action.kind, detail=action.detail, action=action)
    assert await wrapper.request_approval(legacy) == ApprovalDecision.ALLOW_SESSION
    assert await wrapper.request_approval(legacy) == ApprovalDecision.ALLOW
    assert len(human.requests) == 1 and wrapper.approved_categories
    gate = ApprovalOrchestrator(wrapper)
    assert (await gate.evaluate(action)).approved
    assert len(human.requests) == 2
    gate.grants.remove_where(lambda grant: True)
    assert (await gate.evaluate(action)).approved and len(human.requests) == 3


@pytest.mark.asyncio
async def test_grant_lookup_put_and_audit_run_off_event_loop(settings):
    loop_thread = threading.get_ident()
    observed = []

    class ObservedStore(JsonApprovalGrantStore):
        def get(self, key):
            observed.append(threading.get_ident())
            return super().get(key)

        def put(self, grant):
            observed.append(threading.get_ident())
            return super().put(grant)

        def put_if_current(self, grant, revision):
            observed.append(threading.get_ident())
            return super().put_if_current(grant, revision)

    class ObservedAudit(ApprovalAuditLog):
        def record(self, **kwargs):
            observed.append(threading.get_ident())
            return super().record(**kwargs)

    gate = ApprovalOrchestrator(SessionAwareGate(ChoosingHuman()),
        grants=ObservedStore(settings.profile_layout.approval.grants_path), audit=ObservedAudit())
    spec = parse_services({"specialist": declaration(processing_location="local")})[0]
    await authorize_endpoint(spec, gate, sensitive=False)
    await authorize_endpoint(spec, gate, sensitive=False)
    assert observed and all(thread != loop_thread for thread in observed)


@pytest.mark.asyncio
async def test_cli_parser_completion_and_prompt_guidance():
    from leapflow.cli.tui_app.input import SlashCommandCompleter
    from leapflow.plugins.tool_plugins.model_services import ModelServicesPlugin

    with pytest.raises(ValueError):
        parse_service_arguments(["revoke"])
    completions = {item.text for item in SlashCommandCompleter((("config", "Configuration"),)).get_completions(
        Document("/config llm services "), None)}
    assert {"permissions", "revoke"} <= completions
    owner, _, _, plugin, _ = compose(RESULT)
    description = next(t.description for t in plugin.tools if t.name == "model_query")
    assert "Call directly" in description and "duplicate consent" in description
    assert isinstance(plugin, ModelServicesPlugin)
    await owner.aclose()


@pytest.mark.asyncio
async def test_unknown_text_keeps_session_only_internal_safeguards(settings, tmp_path):
    owner, registry, _, _, sent = compose(RESULT)
    human = ChoosingHuman(ApprovalDecision.ALLOW_SESSION)
    gate = persistent_gate(settings, human)
    try:
        for session in ("a", "a", "b"):
            assert (await query(registry, gate, tmp_path, session))["ok"]
        assert len(sent) == 3 and len(human.requests) == 2
        assert all("allow_always" not in r.choices and r.risk.level == RiskLevel.HIGH for r in human.requests)
    finally:
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("addresses,permanent,scope", [
    (("127.0.0.1", "169.254.1.1"), False, "text"),
    (("127.0.0.1", "8.8.8.8"), True, "text"),
    (("127.0.0.1", "::1"), True, "text_and_evidence"),
])
async def test_mixed_dns_records_never_claim_loopback_only_local_processing(settings, monkeypatch, addresses, permanent, scope):
    from leapflow.tools import model_tools

    async def resolve(url):
        return NetworkTarget(url, "http", "localhost", 8080, "http://localhost:8080", "loopback", addresses)

    monkeypatch.setattr(model_tools, "classify_url", resolve)
    human = ChoosingHuman(ApprovalDecision.ALLOW_ONCE)
    gate = persistent_gate(settings, human)
    spec = parse_services({"specialist": declaration(processing_location="local")})[0]
    await authorize_endpoint(spec, gate, sensitive=False)
    request = human.requests[0]
    assert ("allow_always" in request.choices) is permanent
    assert request.action.metadata["approval_binding"]["evidence_scope"] == scope
    assert request.risk.level == (RiskLevel.MEDIUM if scope == "text_and_evidence" else RiskLevel.HIGH)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["expired", "replacement"])
async def test_metadata_change_during_consent_is_refused_before_any_read(settings, tmp_path, monkeypatch, change):
    owner, registry, _, _, sent = compose(RESULT, processing_location="local")
    registry.bind_runtime(settings=settings)
    cache = CacheManager(settings.profile_layout.cache, profile_id=settings.profile)
    images = {str(i): session_artifact(cache, tmp_path, "a", f"{i}.jpg", IMAGE) for i in range(2)}
    reads = []
    monkeypatch.setattr(CacheManager, "read_session_entry", lambda *args, **kwargs: reads.append(True))

    class ChangingHuman(ChoosingHuman):
        async def request_approval(self, request):
            session_artifact(cache, tmp_path, "a", "1.jpg", IMAGE,
                             expires_at=0 if change == "expired" else 9999999999)
            return await super().request_approval(request)

    try:
        result = await query(registry, persistent_gate(settings, ChangingHuman()), tmp_path, "a",
                             {label: entry.entry_id for label, entry in images.items()})
        assert result["failure_code"] == "model_invalid_request" and not sent and not reads
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_post_assembly_freshness_still_refuses_expired_earlier_image(settings, tmp_path, monkeypatch):
    owner, registry, _, _, sent = compose(RESULT, processing_location="local")
    registry.bind_runtime(settings=settings)
    cache = CacheManager(settings.profile_layout.cache, profile_id=settings.profile)
    images = {str(i): session_artifact(cache, tmp_path, "a", f"{i}.jpg", IMAGE) for i in range(2)}
    original = CacheManager.read_session_entry
    reads = []

    def read_and_expire(self, entry, **kwargs):
        content = original(self, entry, **kwargs)
        reads.append(entry.entry_id)
        if entry.entry_id == images["1"].entry_id:
            session_artifact(cache, tmp_path, "a", "0.jpg", IMAGE, expires_at=0)
        return content

    monkeypatch.setattr(CacheManager, "read_session_entry", read_and_expire)
    try:
        result = await query(registry, persistent_gate(settings, ChoosingHuman()), tmp_path, "a",
                             {label: entry.entry_id for label, entry in images.items()})
        assert result["failure_code"] == "model_invalid_request" and len(reads) == 2 and not sent
    finally:
        await owner.aclose()


def test_failed_atomic_save_does_not_leave_an_unsaved_cache_grant(settings, monkeypatch):
    store = JsonApprovalGrantStore(settings.profile_layout.approval.grants_path)
    old = ApprovalGrant("old", "profile", "allow", "shell.command", "execute")
    store.put(old)
    import leapflow.security.grants as grants

    def fail_replace(*args):
        raise OSError("atomic replace unavailable")

    monkeypatch.setattr(grants.os, "replace", fail_replace)
    with pytest.raises(OSError):
        store.put(ApprovalGrant("new", "profile", "allow", "shell.command", "execute"))
    assert store.get("new") is None and store.get("old") == old


@pytest.mark.asyncio
async def test_model_grant_never_authorizes_generic_web_or_sensitive_file_access(settings):
    human = ChoosingHuman()
    gate = persistent_gate(settings, human)
    spec = parse_services({"specialist": declaration(processing_location="local")})[0]
    await authorize_endpoint(spec, gate, sensitive=False)
    human.choice = ApprovalDecision.DENY
    web = ActionDescriptor.network_fetch(spec.base_url, origin="http://localhost:8080",
                                         metadata={"target_category": "loopback"})
    sensitive = ActionDescriptor.file_read("/owned/image.jpg", metadata={
        "sensitive": True, "sensitivity_category": "cache_sensitive"})
    for action in (web, sensitive):
        result = await gate.evaluate(action)
        assert not result.approved and not result.risk.allow_permanent
    assert len(human.requests) == 3


@pytest.mark.asyncio
async def test_legacy_unbound_grants_cannot_silently_authorize_new_model_identity(settings):
    human = ChoosingHuman()
    gate = persistent_gate(settings, human)
    old = ActionDescriptor.network_fetch("http://localhost:8080/v1", origin="http://localhost:8080",
        method="POST", metadata={"platform": "model_services", "processing_location": "local"})
    gate.grants.put(ApprovalGrant(grant_key(old, "profile"), "profile", "allow", old.kind, old.effect))
    spec = parse_services({"specialist": declaration(processing_location="local")})[0]
    await authorize_endpoint(spec, gate, sensitive=False)
    assert len(human.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("binding", [None, {}, {"platform": "model_services", "service_id": "other"}])
async def test_missing_or_changed_persisted_binding_cannot_authorize(settings, binding):
    human = ChoosingHuman()
    gate = persistent_gate(settings, human)
    spec = parse_services({"specialist": declaration(processing_location="local")})[0]
    await authorize_endpoint(spec, gate, sensitive=False)
    path = settings.profile_layout.approval.grants_path
    records = json.loads(path.read_text())
    if binding is None:
        records[0].pop("binding")
    else:
        records[0]["binding"] = binding
    path.write_text(json.dumps(records))
    denying = ChoosingHuman(ApprovalDecision.DENY)
    from leapflow.llm.services.protocol import ServiceError
    with pytest.raises(ServiceError):
        await authorize_endpoint(spec, persistent_gate(settings, denying), sensitive=False)
    assert len(denying.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", [ApprovalDecision.ALLOW_ALWAYS, ApprovalDecision.ALLOW_SESSION,
                                   ApprovalDecision.DENY_ALWAYS])
async def test_cancelled_persistence_cannot_resurrect_revoked_consent(settings, choice):
    entered, release = threading.Event(), threading.Event()

    class DelayedStore(JsonApprovalGrantStore):
        def put_if_current(self, grant, revision):
            entered.set()
            if not release.wait(10):
                raise TimeoutError("Test did not release consent worker")
            return super().put_if_current(grant, revision)

    human = ChoosingHuman(choice)
    gate = ApprovalOrchestrator(SessionAwareGate(human),
        grants=DelayedStore(settings.profile_layout.approval.grants_path), audit=ApprovalAuditLog())
    spec = parse_services({"specialist": declaration(processing_location="local")})[0]
    task = asyncio.create_task(authorize_endpoint(spec, gate, sensitive=False))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert gate._persistence_tasks
        revoked = await asyncio.to_thread(ConfigService(settings).revoke_model_service_permissions, "specialist")
        assert revoked["revoked"] == 0
        release.set()
        await asyncio.wait_for(asyncio.gather(*gate._persistence_tasks), 5)
        assert not gate.grants.list()
        assert gate.audit.entries[-1]["decision"] == "deny"
        assert gate.audit.entries[-1]["reason"] == "consent_revoked_during_approval"
        denying = ChoosingHuman(ApprovalDecision.DENY)
        from leapflow.llm.services.protocol import ServiceError
        with pytest.raises(ServiceError):
            await authorize_endpoint(spec, persistent_gate(settings, denying), sensitive=False)
        assert len(denying.requests) == 1
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, *gate._persistence_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_revocation_during_prompt_requires_a_new_human_choice(settings):
    entered, release = asyncio.Event(), asyncio.Event()

    class WaitingHuman(ChoosingHuman):
        async def request_approval(self, request):
            entered.set()
            await release.wait()
            return await super().request_approval(request)

    gate = persistent_gate(settings, WaitingHuman())
    spec = parse_services({"specialist": declaration(processing_location="local")})[0]
    task = asyncio.create_task(authorize_endpoint(spec, gate, sensitive=False))
    from leapflow.llm.services.protocol import ServiceError
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.to_thread(ConfigService(settings).revoke_model_service_permissions, "specialist")
        release.set()
        with pytest.raises(ServiceError):
            await task
        assert not gate.grants.list()
        fresh = persistent_gate(settings, ChoosingHuman())
        await authorize_endpoint(spec, fresh, sensitive=False)
        assert len(fresh.grants.list()) == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


def test_durable_revisions_are_scoped_and_cannot_be_reused_after_repeated_revoke(tmp_path):
    path = tmp_path / "grants.json"
    stale = JsonApprovalGrantStore(path)
    selector = {"platform": "model_services", "service_id": "specialist"}
    binding = {**selector, "evidence_scope": "text"}
    grant = ApprovalGrant("model", "profile", "allow", "network.fetch", "read", binding=binding)
    revision = stale.revision(binding)
    fresh = JsonApprovalGrantStore(path)
    fresh.revoke_where(lambda g: g.key == "model", binding=selector)
    assert not stale.put_if_current(grant, revision)
    after_revoke = stale.revision(binding)
    assert stale.put_if_current(grant, after_revoke)
    fresh.revoke_where(lambda g: g.key == "model", binding=selector)
    assert not stale.put_if_current(grant, after_revoke) and not stale.list()
    other = replace(grant, key="other", binding={**binding, "service_id": "other"})
    assert stale.put_if_current(other, revision)
    fresh.put(ApprovalGrant("legacy", "profile", "allow", "shell.command", "execute"))
    reopened = JsonApprovalGrantStore(path)
    assert reopened.revision(binding) == stale.revision(binding)
    assert {g.key for g in reopened.list()} == {"other", "legacy"}
