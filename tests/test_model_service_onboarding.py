# Copyright (c) Alibaba, Inc. and its affiliates.
"""Governed empty-catalog onboarding and durable concurrent config mutations."""
from __future__ import annotations

import asyncio
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import pytest

from leapflow.config import load_config
from leapflow.config_service import ConfigService
from leapflow.plugins.registry import ToolPluginRegistry
from leapflow.plugins.tool_plugins.config_tools import ConfigToolsPlugin
from leapflow.security.orchestrator import ApprovalOrchestrator
from leapflow.tools import config_tools
from leapflow.tools.config_model_service import config_model_service_handler
from leapflow.tools.execution_context import ToolExecutionContext, reset_tool_context, set_tool_context
from leapflow.tools.model_fixture import check_inline_fixture
from test_model_services import declaration
from test_model_tools import Human, compose, fixture


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path / "home"))
    monkeypatch.setenv("LEAPFLOW_WORKSPACE_ROOT", str(tmp_path))
    config_tools.set_config_context(None)
    config_tools.set_config_approval_gate(None)
    import leapflow.config as config

    monkeypatch.setattr(config, "_settings_instance", None)
    value = load_config()
    yield value
    config_tools.set_config_context(None)
    config_tools.set_config_approval_gate(None)


def _toggle_writer(barrier, service_id):
    service = ConfigService(load_config())
    barrier.wait(timeout=20)
    for _ in range(8):
        service.set_model_service_enabled(service_id, True)
        service.set_model_service_enabled(service_id, False)
        service.set("llm.context_length", 54321)


def test_cross_process_toggle_same_and_distinct_ids_preserve_declarations(settings):
    service = ConfigService(settings)
    declared = declaration(base_url="http://localhost:8099/v1", model="preserved",
        limits={"max_inflight": 2}, auth={"mode": "secret", "ref": "secret://profile/models/a"})
    for name in ("a", "b"):
        service.set_model_service(name, declared)
    service.set("llm.model", "primary")
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(3)
    processes = [ctx.Process(target=_toggle_writer, args=(barrier, name)) for name in ("a", "a", "b")]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert not process.is_alive() and process.exitcode == 0
    assert service.model_services() == {name: {**declared, "enabled": False,
        "media": {"max_images": 16, "max_videos": 0}} for name in ("a", "b")}
    assert load_config().llm_model == "primary" and load_config().llm_context_length == 54321


def test_toggle_cli_and_slash_roundtrip_noop_unknown_and_invalid(settings):
    from types import SimpleNamespace
    from leapflow.cli.commands.model_service_config import execute_service_config, parse_service_arguments
    from leapflow.cli.commands.slash_handlers import build_config_payload

    service = ConfigService(settings)
    service.set_model_service("d3", declaration(enabled=False))
    path = settings.profile_layout.llm_config_path
    before = path.read_bytes(), path.stat().st_mtime_ns
    noop = execute_service_config(service, parse_service_arguments(["disable", "d3"]))
    assert noop.ok and not noop.changed_keys
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    with pytest.raises(ValueError, match="Unknown"):
        service.set_model_service_enabled("unknown", True)
    for invalid in (None, "false", 1, 0):
        with pytest.raises(ValueError, match="boolean"):
            service.set_model_service_enabled("d3", invalid)
    ctx = SimpleNamespace(settings=settings, reload_runtime_config_if_changed=lambda **kwargs: False)
    enabled = build_config_payload(ctx, "llm services enable d3")
    assert enabled["ok"] and service.model_services()["d3"]["enabled"]
    disabled = execute_service_config(service, parse_service_arguments(["disable", "d3"]))
    assert disabled.ok and not service.model_services()["d3"]["enabled"]
    checked = service.check_model_services("d3")["services"][0]
    assert checked["configured"] and not checked["usable"] and checked["unavailable_reason"] == "disabled"
    assert not checked["endpoint_probed"]


def _catalog_writer(barrier, prefix):
    service = ConfigService(load_config())
    barrier.wait(timeout=20)
    for index in range(8):
        service.set_model_service(f"{prefix}_{index}", declaration(task="decision"))
        service.set("llm.model" if prefix == "left" else "llm.base_url", "left-model" if prefix == "left" else "http://localhost:1234/v1")


def test_cross_process_distinct_ids_and_adjacent_writers(settings):
    service = ConfigService(settings)
    service.set("llm.context_length", 54321)
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    processes = [ctx.Process(target=_catalog_writer, args=(barrier, prefix)) for prefix in ("left", "right")]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert not process.is_alive()
        assert process.exitcode == 0
    assert len(service.model_services()) == 16
    updated = load_config()
    assert updated.llm_model == "left-model"
    assert updated.llm_base_url == "http://localhost:1234/v1"
    assert updated.llm_context_length == 54321


def test_unset_and_secret_writer_preserve_unrelated_catalog(settings):
    service = ConfigService(settings)
    service.set_model_service("keep", declaration())
    service.set_model_service("remove", declaration())
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(service.unset_model_service, "remove"),
                pool.submit(service.set, "llm.api_key", "test-only-credential"),
                pool.submit(service.set_model_service, "add", declaration())]
        for job in jobs:
            assert job.result().ok
    assert set(service.model_services()) == {"keep", "add"}
    assert "test-only-credential" not in settings.profile_layout.llm_config_path.read_text()


def test_same_id_last_serialized_writer_and_corrupt_yaml(settings):
    service = ConfigService(settings)
    first = declaration(model="first")
    second = declaration(model="second")
    service.set_model_service("shared", first)
    service.set_model_service("shared", second)
    assert service.model_services()["shared"]["model"] == "second"
    path = settings.profile_layout.llm_config_path
    path.write_text("llm: [invalid\n")
    with pytest.raises(ValueError, match="refusing"):
        service.set_model_service("new", first)
    assert path.read_text() == "llm: [invalid\n"


@pytest.mark.asyncio
async def test_empty_catalog_registry_onboarding_and_routed_denial(settings):
    registry = ToolPluginRegistry()
    registry.register(ConfigToolsPlugin())
    registry.assemble()
    handler = registry.tool_handlers["config_model_service"]
    assert (await handler({"operation": "show"}))["services"] == {}
    human = Human()
    gate = ApprovalOrchestrator(human)
    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(settings.workspace_root),
        session_id="own", task_id="turn-1", orchestrator=gate))
    try:
        initial_primary = settings.llm_model
        result = await handler({"operation": "set", "service_id": "adviser", "declaration": declaration(task="decision")})
        assert result["ok"], result
        assert result["declaration"]["task"] == "decision"
        assert load_config().llm_model == initial_primary
        assert human.requests
        action = gate.audit.entries[-1]
        assert "own" in str(action) and "turn-1" in str(action)
        human.allow = False
        result = await handler({"operation": "unset", "service_id": "adviser"})
        assert result["failure_code"] == "permission_denied" and "STOP" in result["llm_instruction"]
        assert "adviser" in ConfigService(settings).model_services()
    finally:
        reset_tool_context(token)


@pytest.mark.asyncio
async def test_absent_route_and_invalid_provider_never_save_or_prompt(settings):
    config_tools.set_config_approval_gate(ApprovalOrchestrator(Human()))
    args = {"operation": "set", "service_id": "x", "declaration": declaration(task="decision")}
    result = await config_model_service_handler(args)
    assert result["failure_code"] == "permission_denied"
    human = Human()
    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(settings.workspace_root),
        session_id="s", orchestrator=ApprovalOrchestrator(human)))
    try:
        args["declaration"]["provider"] = "not-installed"
        result = await config_model_service_handler(args)
        assert result["failure_code"] == "model_provider_unavailable"
        assert not human.requests and not ConfigService(settings).model_services()
    finally:
        reset_tool_context(token)


@pytest.mark.asyncio
async def test_inline_fixture_shares_file_validation_and_refuses_images(settings, tmp_path):
    owner, _, _, _, requests = compose({"selected_id": "a", "rationale": "eligible"})
    gate = ApprovalOrchestrator(Human())
    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(tmp_path), session_id="s", orchestrator=gate))
    try:
        case = json.loads(fixture(tmp_path / "case.json").read_text())
        result = await check_inline_fixture(owner, "specialist", case)
        assert result["ok"] and result["stages"]["fixture_assertions"] == "pass", result
        assert not result["physical_authority"] and len(requests) == 1
        case["artifacts"] = {"image": {"media_type": "image/jpeg", "data": "unapproved"}}
        result = await check_inline_fixture(owner, "specialist", case)
        assert result["failure_code"] == "model_invalid_fixture" and len(requests) == 1
        case["artifacts"] = {}
        case["expected"] = {"$ref": "https://invalid.example"}
        result = await check_inline_fixture(owner, "specialist", case)
        assert result["failed_stage"] == "fixture_input" and len(requests) == 1
        case["expected"] = {"properties": {"selected_id": {"enum": ["wrong"]}}}
        result = await check_inline_fixture(owner, "specialist", case)
        assert result["failure_code"] == "model_fixture_mismatch"
        assert not owner._handles
    finally:
        reset_tool_context(token)
        await owner.aclose()


@pytest.mark.asyncio
async def test_committed_write_is_not_reported_unsaved_after_readback_failure(settings, monkeypatch):
    original = ConfigService.model_services
    calls = 0

    def failing_readback(self):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("readback unavailable")
        return original(self)

    monkeypatch.setattr(ConfigService, "model_services", failing_readback)
    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(settings.workspace_root),
        session_id="own", task_id="save", orchestrator=ApprovalOrchestrator(Human())))
    try:
        result = await config_model_service_handler({"operation": "set", "service_id": "saved",
            "declaration": declaration(task="decision")})
        assert result["ok"] and result["saved"] and result["side_effect_state"] == "committed"
        assert result["read_back_status"] == "unavailable" and not result["retryable"]
        assert "saved" in original(ConfigService(settings))
    finally:
        reset_tool_context(token)


def test_overlapping_same_id_writes_serialize_after_atomic_replace(settings, monkeypatch):
    import threading
    import leapflow.config_service as module

    service = ConfigService(settings)
    entered, release, attempted = threading.Event(), threading.Event(), threading.Event()
    original = module._write_yaml_atomic

    def paused_first_write(path, data):
        if data["llm"]["services"]["shared"]["model"] == "first":
            entered.set()
            assert release.wait(5)
        original(path, data)

    def second_write():
        attempted.set()
        return service.set_model_service("shared", declaration(model="second"))

    monkeypatch.setattr(module, "_write_yaml_atomic", paused_first_write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service.set_model_service, "shared", declaration(model="first"))
        assert entered.wait(5)
        second = pool.submit(second_write)
        assert attempted.wait(5) and not second.done()
        release.set()
        assert first.result(timeout=5).ok and second.result(timeout=5).ok
    assert service.model_services()["shared"]["model"] == "second"


@pytest.mark.asyncio
async def test_real_context_publication_preserves_two_workspace_session_snapshots(settings, monkeypatch, tmp_path):
    from leapflow.cli.context import Context
    from leapflow.daemon.service import RuntimeLeapService
    from leapflow.plugins import get_registry
    from tests.test_session_factory import _build_base_engine, _EchoLLM

    ctx = Context(settings, mock_host=True)
    base, memory = _build_base_engine(str(tmp_path / "engine"), _EchoLLM())
    ctx.engine = base
    daemon = RuntimeLeapService(settings)
    daemon._ctx = ctx
    sessions = daemon._ensure_session_registry(base)
    a = await sessions.acquire("a", workspace_root=tmp_path / "workspace-a")
    b = await sessions.acquire("b", workspace_root=tmp_path / "workspace-b")
    ctx._runtime_config_reload_callback = daemon._on_runtime_config_reloaded
    monkeypatch.setattr(ctx, "_configure_mcp_manager", lambda _: None)
    config_tools.set_config_context(ctx)
    registry = get_registry()
    ctx._bind_model_services()
    snapshot = dict(registry.tool_handlers)
    version = registry.version
    primary = (ctx.llm, ctx.settings.llm_model, ctx.settings.llm_base_url)
    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(tmp_path / "workspace-a"),
        session_id="a", task_id="onboard", orchestrator=ApprovalOrchestrator(Human())))
    try:
        a.engine._wm.remember_chat({"role": "user", "content": "only-in-a"})
        result = await registry.tool_handlers["config_model_service"]({"operation": "set", "service_id": "adviser",
            "declaration": declaration(task="decision")})
        assert result["ok"] and result["session_reloaded"], result
        assert "model_query" in registry.tool_handlers and "model_query" not in snapshot
        assert registry.version > version
        assert ctx.model_services.describe("adviser")["services"][0]["available"]
        assert a.engine._settings.llm_services == b.engine._settings.llm_services == ctx.settings.llm_services
        assert a.engine._settings.workspace_root != b.engine._settings.workspace_root
        assert list(b.engine._wm.as_chat_messages()) == []
        assert primary == (ctx.llm, ctx.settings.llm_model, ctx.settings.llm_base_url)
        admitted = await ctx.model_services.acquire("adviser")
        stable_tools = set(registry.tool_handlers)
        result = await registry.tool_handlers["config_model_service"]({"operation": "disable", "service_id": "adviser"})
        assert result["ok"] and result["runtime_publication_confirmed"] and result["effective_enabled"] is False
        assert not ctx.model_services.describe("adviser")["services"][0]["usable"]
        assert a.engine._settings.llm_services == b.engine._settings.llm_services == ctx.settings.llm_services
        assert not a.engine._settings.llm_services[0].enabled
        assert stable_tools == set(registry.tool_handlers)
        from leapflow.llm.services.protocol import ServiceError
        with pytest.raises(ServiceError) as revoked:
            admitted.check_admission()
        assert revoked.value.failure.code == "model_service_disabled"
        result = await registry.tool_handlers["config_model_service"]({"operation": "enable", "service_id": "adviser"})
        assert result["runtime_publication_confirmed"] and result["effective_enabled"] is True
        with pytest.raises(ServiceError):
            admitted.check_admission()
        # External writes become visible at the existing reload entry point.
        ConfigService(settings).set_model_service_enabled("adviser", False)
        assert ctx.model_services.describe("adviser")["services"][0]["enabled"]
        ctx.reload_runtime_config_if_changed(force=True)
        assert not ctx.model_services.describe("adviser")["services"][0]["enabled"]
        assert not b.engine._settings.llm_services[0].enabled
        assert primary == (ctx.llm, ctx.settings.llm_model, ctx.settings.llm_base_url)
        await admitted.aclose()
        result = await registry.tool_handlers["config_model_service"]({"operation": "unset", "service_id": "adviser"})
        assert result["ok"] and "model_query" not in registry.tool_handlers
        assert ctx.model_services.describe()["services"] == []
        assert a.engine._settings.llm_services == b.engine._settings.llm_services == ()
    finally:
        reset_tool_context(token)
        config_tools.set_config_context(None)
        await ctx.cleanup()
        memory.close()


@pytest.mark.asyncio
async def test_toggle_reload_failure_reports_saved_but_not_effectively_disabled(settings):
    from types import SimpleNamespace

    service = ConfigService(settings)
    service.set_model_service("specialist", declaration())
    owner, _, _, _, _ = compose({"selected_id": "a", "rationale": "eligible"})

    def fail_reload(**kwargs):
        raise RuntimeError("reload unavailable")

    config_tools.set_config_context(SimpleNamespace(settings=settings, model_services=owner,
        reload_runtime_config_if_changed=fail_reload))
    human = Human()
    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(settings.workspace_root), session_id="own",
        orchestrator=ApprovalOrchestrator(human)))
    try:
        result = await config_model_service_handler({"operation": "disable", "service_id": "specialist"})
        assert result["ok"] and result["saved"] and result["side_effect_state"] == "committed"
        assert not result["runtime_publication_confirmed"] and result["effective_enabled"] is None
        assert not result["session_reloaded"] and not service.model_services()["specialist"]["enabled"]
        assert owner.describe("specialist")["services"][0]["enabled"]
        noop = await config_model_service_handler({"operation": "disable", "service_id": "specialist"})
        assert noop["saved"] and noop["side_effect_state"] == "none"
        human.allow = False
        denied = await config_model_service_handler({"operation": "enable", "service_id": "specialist"})
        assert denied["failure_code"] == "permission_denied" and "STOP" in denied["llm_instruction"]
        assert not service.model_services()["specialist"]["enabled"]
    finally:
        reset_tool_context(token)
        config_tools.set_config_context(None)
        await owner.aclose()


@pytest.mark.asyncio
async def test_cancel_configuration_approval_does_not_write(settings):
    entered = asyncio.Event()

    class WaitingHuman:
        async def request_approval(self, request):
            entered.set()
            await asyncio.Event().wait()

    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(settings.workspace_root), session_id="s",
        orchestrator=ApprovalOrchestrator(WaitingHuman())))
    try:
        task = asyncio.create_task(config_model_service_handler({"operation": "set", "service_id": "cancelled",
            "declaration": declaration(task="decision")}))
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not ConfigService(settings).model_services()
    finally:
        reset_tool_context(token)


@pytest.mark.asyncio
async def test_chat_onboards_registered_contract_codec_and_second_endpoint_without_core_edit(settings):
    from types import SimpleNamespace
    from leapflow.llm.services.config import parse_services
    from test_model_tools import NovelContractPlugin

    owner, registry, scoped, _, requests = compose({"selected_id": "a"})
    plugin = NovelContractPlugin()
    registry.register(plugin)
    scoped.adopt_existing_plugins()
    registry.bind_runtime(model_services=owner)
    service = ConfigService(settings, model_services=owner)

    def reload(**kwargs):
        owner.publish(parse_services(service.model_services()))
        return True

    config_tools.set_config_context(SimpleNamespace(settings=settings, model_services=owner,
        reload_runtime_config_if_changed=reload))
    human = Human()
    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(settings.workspace_root), session_id="own",
                            orchestrator=ApprovalOrchestrator(human)))
    try:
        for name, declared in (
            ("first", declaration()),
            ("second", declaration(model="compatible-checkpoint", base_url="http://localhost:8099/v1")),
            ("novel", declaration(task="test.label_choice.v1", codec="test.json.v1")),
        ):
            result = await config_model_service_handler({"operation": "set", "service_id": name, "declaration": declared})
            assert result["ok"] and result["runtime_publication_confirmed"], result
            checked = await config_model_service_handler({"operation": "check", "service_id": name})
            assert checked["services"][0]["usable"] and not checked["services"][0]["endpoint_probed"]
        assert not requests and len(service.model_services()) == 3
        scoped.dispose_plugin(plugin.plugin_id)
        checked = await config_model_service_handler({"operation": "check", "service_id": "novel"})
        assert not checked["services"][0]["usable"]
        prompts = len(human.requests)
        rejected = await config_model_service_handler({"operation": "set", "service_id": "another",
            "declaration": declaration(task="test.label_choice.v1", codec="test.json.v1")})
        assert not rejected["ok"] and len(human.requests) == prompts
        assert "another" not in service.model_services()
    finally:
        reset_tool_context(token)
        config_tools.set_config_context(None)
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [
    {"task": "prediction", "output": "json_schema"},
    {"task": "unknown.task.v1"}, {"codec": "unknown.codec.v1"},
    {"provider": "anthropic", "request_options": {"seed": 3}},
])
async def test_incompatible_declarations_refused_before_cli_write_or_chat_consent(settings, changes):
    service = ConfigService(settings)
    with pytest.raises(ValueError):
        service.set_model_service("bad", declaration(**changes))
    human = Human()
    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=str(settings.workspace_root), session_id="s",
                            orchestrator=ApprovalOrchestrator(human)))
    try:
        result = await config_model_service_handler({"operation": "set", "service_id": "bad",
                                                    "declaration": declaration(**changes)})
        assert not result["ok"] and not human.requests and not service.model_services()
    finally:
        reset_tool_context(token)
