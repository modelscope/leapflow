"""Real-artifact tests for the restricted DSH/Cordis compatibility path."""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from leapflow.learning.compatibility import (
    ComponentStatus,
    PluginSourceKind,
    Verdict,
    assess_plugin,
    inspect_plugin_source,
)
from leapflow.learning.plugin_generator import PluginValidator
from leapflow.plugins.dsh.bundle import promote_staging_bundle, stage_runtime_bundle
from leapflow.plugins.dsh.capabilities import (
    DshCapabilityBroker,
    DshCapabilityError,
    parse_curl_get,
)
from leapflow.plugins.dsh.installer import DshInstallError, prepare_dsh_installation
from leapflow.plugins.dsh.node_host import DshNodeHost

_FIXTURES = Path(__file__).parent / "_fixtures"
_DYNAMIC = _FIXTURES / "dsh_exports" / "stock-panel"
_PACKAGE = _FIXTURES / "dsh_packages" / "echo"


def _write_package(root: Path, body: str, *, name: str = "fixture-package") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "package.json").write_text(
        json.dumps(
            {
                "name": name,
                "version": "1.0.0",
                "main": "index.cjs",
                "keywords": ["tools"],
            }
        ),
        encoding="utf-8",
    )
    (root / "index.cjs").write_text(body, encoding="utf-8")
    return root


def test_dynamic_export_static_assessment_is_honestly_partial() -> None:
    inspection = inspect_plugin_source(_DYNAMIC)
    report = assess_plugin(_DYNAMIC)

    assert inspection.execution_plan.source_kind == PluginSourceKind.CORDIS_DYNAMIC_EXPORT
    assert inspection.manifest.version == "0.0.0+export"
    assert inspection.execution_plan.bundle_sha256
    assert report.final_verdict == Verdict.PARTIAL
    assert report.is_installable() is False  # discovery has not run
    assert report.execution_plan is not None
    assert report.execution_plan.installable_candidate is True
    statuses = {item.kind.value: item.status for item in report.execution_plan.components}
    assert statuses == {
        "host": ComponentStatus.CANDIDATE,
        "client": ComponentStatus.UNSUPPORTED,
    }
    assert "client.js UI" in report.execution_plan.limitations[0]


def test_prebuilt_package_static_assessment_requires_runtime_discovery() -> None:
    report = assess_plugin(_PACKAGE)

    assert report.final_verdict == Verdict.ADAPTABLE
    assert report.execution_plan is not None
    assert report.execution_plan.source_kind == PluginSourceKind.DSH_PACKAGE
    assert report.execution_plan.entry_point == "index.cjs"
    assert report.execution_plan.requires_discovery is True
    assert report.is_installable() is False


def test_package_with_dependency_is_not_installable_in_p0(tmp_path: Path) -> None:
    source = tmp_path / "package"
    source.mkdir()
    (source / "index.cjs").write_text("module.exports={apply(){}}", encoding="utf-8")
    (source / "package.json").write_text(
        json.dumps(
            {
                "name": "dependency-plugin",
                "version": "1.0.0",
                "main": "index.cjs",
                "keywords": ["tools"],
                "dependencies": {"left-pad": "1.3.0"},
            }
        ),
        encoding="utf-8",
    )

    report = assess_plugin(source)

    assert report.final_verdict == Verdict.INCOMPATIBLE
    assert report.is_installable() is False
    assert "npm install/build" in (report.rejection_reason or "")


def test_package_peer_dependencies_and_architecture_category_are_preserved(
    tmp_path: Path,
) -> None:
    source = tmp_path / "agent-loop"
    source.mkdir()
    (source / "index.js").write_text("export function apply() {}", encoding="utf-8")
    (source / "package.json").write_text(
        json.dumps(
            {
                "name": "@deepseek-ai/dsh-agent-loop",
                "version": "0.1.0-rc.7",
                "type": "module",
                "main": "index.js",
                "peerDependencies": {"@deepseek-ai/dsh-session": "workspace:^"},
            }
        ),
        encoding="utf-8",
    )

    report = assess_plugin(source)

    assert report.final_verdict == Verdict.INCOMPATIBLE
    assert report.manifest.category == "agent-loop"
    assert "single hardened OODA execution loop" in (report.rejection_reason or "")
    assert report.execution_plan is not None
    assert report.execution_plan.dependencies == ("@deepseek-ai/dsh-session",)
    assert report.execution_plan.blockers


def test_dynamic_declared_inject_service_is_rejected_before_discovery(
    tmp_path: Path,
) -> None:
    source = tmp_path / "service-consumer"
    source.mkdir()
    (source / "meta.json").write_text(
        '{"name":"service-consumer","category":"tools"}', encoding="utf-8"
    )
    (source / "host.js").write_text(
        "return {inject:['greeter','tools'],apply(ctx){harness.registerTool(ctx,"
        "harness.defineTool({name:'greet',description:'greet',parameters:{},"
        "execute:async()=>ctx.greeter.greet('x')}))}}",
        encoding="utf-8",
    )

    report = assess_plugin(source)

    assert report.final_verdict == Verdict.INCOMPATIBLE
    assert report.execution_plan is not None
    assert "required DSH host service: greeter" in (report.rejection_reason or "")
    assert "required DSH host service: tools" not in (report.rejection_reason or "")


def test_source_inspector_rejects_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "plugin"
    source.mkdir()
    (source / "meta.json").write_text('{"name":"bad"}', encoding="utf-8")
    (source / "host.js").write_text("return {apply(){}}", encoding="utf-8")
    (source / "escape").symlink_to(tmp_path / "outside")

    with pytest.raises(ValueError, match="symlink"):
        inspect_plugin_source(source)


@pytest.mark.asyncio
async def test_prebuilt_package_discover_and_invoke() -> None:
    host = DshNodeHost(
        _PACKAGE,
        source_kind=PluginSourceKind.DSH_PACKAGE.value,
        entry_point="index.cjs",
    )
    try:
        discovered = await host.discover()
        assert discovered.ok is True, host.stderr_tail
        assert [tool["name"] for tool in discovered.result["tools"]] == ["fixture_echo"]

        invoked = await host.invoke("fixture_echo", {"text": "hello"})
        assert invoked.ok is True
        assert invoked.result == {"ok": True, "echo": "hello"}
    finally:
        await host.stop()


@pytest.mark.asyncio
async def test_dynamic_export_discovery_and_typed_http_invocation(tmp_path: Path) -> None:
    inspection = inspect_plugin_source(_DYNAMIC)
    staging, entry = stage_runtime_bundle(inspection, tmp_path, "stock")
    fetches: list[dict] = []

    async def fake_fetch(params: dict) -> dict:
        fetches.append(dict(params))
        return {"ok": True, "text": "fixture quote"}

    host = DshNodeHost(
        staging,
        source_kind=inspection.execution_plan.source_kind.value,
        entry_point=entry,
        broker=DshCapabilityBroker(web_fetch=fake_fetch),
    )
    try:
        discovered = await host.discover()
        assert discovered.ok is True, host.stderr_tail
        assert [tool["name"] for tool in discovered.result["tools"]] == [
            "fixture_stock_quote"
        ]
        assert discovered.result["handler_channels"] == ["fetch-quote"]

        invoked = await host.invoke("fixture_stock_quote", {"symbol": "AAPL"})
        assert invoked.ok is True
        assert invoked.result["raw"] == "fixture quote"
        assert fetches[0]["url"] == "https://example.test/quote?q=AAPL"
    finally:
        await host.stop()


@pytest.mark.asyncio
async def test_shell_injection_is_denied_before_any_fetch(tmp_path: Path) -> None:
    inspection = inspect_plugin_source(_DYNAMIC)
    staging, entry = stage_runtime_bundle(inspection, tmp_path, "stock")
    fetches: list[dict] = []

    async def fake_fetch(params: dict) -> dict:
        fetches.append(dict(params))
        return {"ok": True, "text": "must not run"}

    host = DshNodeHost(
        staging,
        source_kind=inspection.execution_plan.source_kind.value,
        entry_point=entry,
        broker=DshCapabilityBroker(web_fetch=fake_fetch),
    )
    try:
        invoked = await host.invoke(
            "fixture_stock_quote",
            {"symbol": "AAPL'; echo LEAPFLOW_DSH_INJECTION; #"},
        )
        assert invoked.ok is False
        assert "only permits" in invoked.error
        assert fetches == []
    finally:
        await host.stop()


@pytest.mark.parametrize(
    "command",
    [
        "curl -sS https://example.test",
        "curl -sS -m 5 'https://example.test'; id",
        "curl -sS -m 5 'https://example.test' > /tmp/out",
        "curl -sS -m 5 'https://example.test' | sh",
        "wget 'https://example.test'",
        "curl -sS -m 5 'http://$(whoami)'",
    ],
)
def test_curl_shim_rejects_every_non_contract_shape(command: str) -> None:
    with pytest.raises(DshCapabilityError):
        parse_curl_get(command)


def test_curl_shim_accepts_the_exact_legacy_shapes() -> None:
    plain = parse_curl_get("curl -sS -m 15 'https://example.test/a?q=1'")
    assert plain.url == "https://example.test/a?q=1"
    assert plain.decode_gb18030 is False

    gb = parse_curl_get(
        "curl -sS -m 20 'https://example.test/a' | iconv -f GB18030 -t UTF-8"
    )
    assert gb.decode_gb18030 is True


@pytest.mark.asyncio
async def test_worker_timeout_terminates_the_process(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        '{"name":"hang","version":"1.0.0","main":"index.cjs","keywords":["tools"]}',
        encoding="utf-8",
    )
    (tmp_path / "index.cjs").write_text(
        "module.exports={apply(ctx){harness.registerTool(ctx,harness.defineTool({"
        "name:'hang_forever',description:'hang',parameters:{},"
        "execute:async()=>new Promise(()=>{})}))}}",
        encoding="utf-8",
    )
    host = DshNodeHost(
        tmp_path,
        source_kind=PluginSourceKind.DSH_PACKAGE.value,
        entry_point="index.cjs",
        invoke_timeout_s=0.05,
    )
    try:
        assert (await host.discover()).ok is True
        response = await host.invoke("hang_forever", {})
        assert response.ok is False
        assert response.error_type == "timeout"
    finally:
        await host.stop()


@pytest.mark.asyncio
async def test_console_output_does_not_pollute_protocol(tmp_path: Path) -> None:
    (tmp_path / "index.cjs").write_text(
        "console.log('plugin boot noise');module.exports={apply(ctx){"
        "harness.registerTool(ctx,harness.defineTool({name:'noisy_tool',"
        "description:'noisy',parameters:{},execute:async()=>{console.log('tool noise');"
        "return {ok:true}}}))}}",
        encoding="utf-8",
    )
    host = DshNodeHost(
        tmp_path,
        source_kind=PluginSourceKind.DSH_PACKAGE.value,
        entry_point="index.cjs",
    )
    try:
        assert (await host.discover()).ok is True
        assert (await host.invoke("noisy_tool", {})).ok is True
        await asyncio.sleep(0)
        assert "plugin boot noise" in host.stderr_tail
        assert "tool noise" in host.stderr_tail
    finally:
        await host.stop()


@pytest.mark.asyncio
async def test_prepared_installation_generates_valid_native_wrapper(tmp_path: Path) -> None:
    prepared = await prepare_dsh_installation(
        _PACKAGE,
        plugin_id="fixture-echo",
        plugins_dir=tmp_path / "plugins",
        dsh_plugins_dir=tmp_path / "plugins" / "dsh",
    )
    try:
        promote_staging_bundle(prepared.staging_root, prepared.final_root)
        result = await PluginValidator().validate(
            prepared.plugin_id, prepared.wrapper_source
        )
        assert result.ok is True, result.error
        assert prepared.compatibility.is_installable() is True
        assert prepared.plugin_id == "fixture_echo"
        assert [tool.name for tool in prepared.descriptor.tools] == ["fixture_echo"]
        assert "plugin = DshBridgePlugin" in prepared.wrapper_source
    finally:
        shutil.rmtree(tmp_path / "plugins", ignore_errors=True)


@pytest.mark.asyncio
async def test_installed_descriptor_rejects_source_tampering(tmp_path: Path) -> None:
    prepared = await prepare_dsh_installation(
        _PACKAGE,
        plugin_id="fixture-echo",
        plugins_dir=tmp_path / "plugins",
        dsh_plugins_dir=tmp_path / "plugins" / "dsh",
    )
    promote_staging_bundle(prepared.staging_root, prepared.final_root)
    try:
        (prepared.final_root / "index.cjs").write_text(
            "module.exports={apply(){/* tampered */}}", encoding="utf-8"
        )
        namespace: dict = {}
        with pytest.raises(ValueError, match="hash does not match"):
            exec(compile(prepared.wrapper_source, "<tampered-wrapper>", "exec"), namespace)
    finally:
        shutil.rmtree(tmp_path / "plugins", ignore_errors=True)


@pytest.mark.asyncio
async def test_self_management_dsh_install_reload_status_and_remove(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import leapflow.plugins as plugins
    from leapflow.plugins.registry import ToolPluginRegistry
    from leapflow.plugins.scoped_registry import ScopedToolRegistry
    from leapflow.plugins.tool_plugins import _load_plugin_from_file
    from leapflow.plugins.tool_plugins.self_management import SelfManagementPlugin

    registry = ToolPluginRegistry()
    scoped = ScopedToolRegistry(registry)
    monkeypatch.setattr(plugins, "_registry", registry)
    monkeypatch.setattr(plugins, "_scoped_registry", scoped)

    class Approval:
        actions: list = []

        async def evaluate(self, action):
            self.actions.append(action)
            return type("Result", (), {"approved": True, "denial_message": ""})()

    class VersionStore:
        def record_source(self, *args, **kwargs):
            return {"version": "fixture-v1"}

        def active(self, plugin_id):
            return None

        def versions(self, plugin_id):
            return []

    manager = SelfManagementPlugin()
    manager._plugin_install_dir = str(tmp_path / "plugins")
    manager._plugin_version_store = VersionStore()
    manager._plugin_approval_gate = Approval()

    result = await manager._plugin_install_handler(
        plugin_id="fixture-stock",
        source_path=str(_DYNAMIC),
    )
    assert result["ok"] is True, result
    assert result["plugin_id"] == "fixture_stock"
    assert result["verdict"] == "partial"
    assert result["limitations"]
    assert result["installed_tools"] == ["fixture_stock_quote"]
    assert len(manager._plugin_approval_gate.actions) == 1
    approval_metadata = manager._plugin_approval_gate.actions[0].metadata
    assert approval_metadata["bundle_sha256"] == result["bundle_sha256"]
    assert approval_metadata["verdict"] == "partial"
    assert "compat.shell.curl_get" in approval_metadata["permissions"]
    assert {item["kind"]: item["status"] for item in approval_metadata["components"]} == {
        "host": "candidate",
        "client": "unsupported",
    }

    plugin = registry.get_plugin("fixture_stock")
    assert plugin is not None

    async def fake_fetch(params: dict) -> dict:
        return {"ok": True, "text": f"quote:{params['url']}"}

    plugin.bind_runtime(web_fetch=fake_fetch)
    invoked = await plugin.tools[0].handler(symbol="AAPL")
    assert invoked["ok"] is True
    assert invoked["raw"].startswith("quote:https://example.test/quote?q=AAPL")

    status = await manager._plugin_status_handler("fixture_stock")
    assert status["ok"] is True
    assert status["dsh"]["runtime"] == "node"
    assert status["dsh"]["verdict"] == "partial"
    assert status["dsh"]["limitations"]
    assert status["dsh"]["client_components"][0]["status"] == "unsupported"

    wrapper = tmp_path / "plugins" / "fixture_stock.py"
    restarted = _load_plugin_from_file(wrapper)
    assert restarted is not None
    assert [tool.name for tool in restarted.tools] == ["fixture_stock_quote"]

    old_generation = scoped.get_fiber("fixture_stock").generation
    reloaded = await manager._plugin_reload_handler("fixture_stock")
    assert reloaded["ok"] is True
    assert reloaded["new_generation"] > old_generation

    disabled = await manager._plugin_disable_handler("fixture_stock")
    assert disabled["ok"] is True
    assert registry.get_plugin("fixture_stock") is None
    enabled = await manager._plugin_enable_handler("fixture_stock")
    assert enabled["ok"] is True
    assert registry.get_plugin("fixture_stock") is not None

    rollback = await manager._plugin_rollback_handler("fixture_stock", "fixture-v1")
    assert rollback["ok"] is False
    assert rollback["failure_code"] == "dsh_rollback_unsupported"

    removed = await manager._plugin_remove_handler("fixture_stock")
    assert removed["ok"] is True
    assert wrapper.exists() is False
    assert (tmp_path / "plugins" / "dsh" / "fixture_stock").exists() is False
    assert registry.get_plugin("fixture_stock") is None


def test_ui_only_dynamic_export_is_incompatible(tmp_path: Path) -> None:
    source = tmp_path / "ui-only"
    source.mkdir()
    (source / "meta.json").write_text('{"name":"ui-only"}', encoding="utf-8")
    (source / "host.js").write_text(
        "return {apply(){harness.handle('private-only', async()=>({ok:true}))}}",
        encoding="utf-8",
    )
    (source / "client.js").write_text(
        "return {apply(ctx){ctx.slots.inject('main', {})}}", encoding="utf-8"
    )

    report = assess_plugin(source)

    assert report.final_verdict == Verdict.INCOMPATIBLE
    assert report.is_installable() is False
    assert "no statically visible registerTool" in (report.rejection_reason or "")


def test_unsupported_host_service_is_a_static_blocker(tmp_path: Path) -> None:
    source = tmp_path / "unsupported-service"
    source.mkdir()
    (source / "meta.json").write_text('{"name":"unsupported-service"}', encoding="utf-8")
    (source / "host.js").write_text(
        "return {apply(ctx){ctx.get('storage');harness.registerTool(ctx,"
        "harness.defineTool({name:'bad_service',description:'bad',parameters:{},"
        "execute:async()=>({ok:true})}))}}",
        encoding="utf-8",
    )

    report = assess_plugin(source)

    assert report.final_verdict == Verdict.INCOMPATIBLE
    assert "required DSH host service: storage" in (report.rejection_reason or "")


@pytest.mark.asyncio
async def test_discovery_rejects_invalid_tool_schema_and_cleans_staging(
    tmp_path: Path,
) -> None:
    source = _write_package(
        tmp_path / "source",
        "module.exports={apply(ctx){harness.registerTool(ctx,harness.defineTool({"
        "name:'bad_schema',description:'bad schema',parameters:{type:'array'},"
        "execute:async()=>[]}))}}",
        name="bad-schema",
    )
    dsh_root = tmp_path / "plugins" / "dsh"

    with pytest.raises(DshInstallError, match="discovery failed"):
        await prepare_dsh_installation(
            source,
            plugin_id="bad-schema",
            plugins_dir=tmp_path / "plugins",
            dsh_plugins_dir=dsh_root,
        )

    assert list(dsh_root.glob(".bad_schema.staging-*")) == []


@pytest.mark.asyncio
async def test_worker_bounds_stderr_tail_and_oversized_result(tmp_path: Path) -> None:
    source = _write_package(
        tmp_path / "bounded",
        "console.error('noise-'+'x'.repeat(10000)+'-tail-marker');"
        "module.exports={apply(ctx){harness.registerTool(ctx,harness.defineTool({"
        "name:'large_result',description:'large',parameters:{},"
        "execute:async()=>({value:'y'.repeat(5000)})}))}}",
        name="bounded-worker",
    )
    host = DshNodeHost(
        source,
        source_kind=PluginSourceKind.DSH_PACKAGE.value,
        entry_point="index.cjs",
        max_line_bytes=1024,
        max_stderr_bytes=1024,
    )
    try:
        assert (await host.discover()).ok is True
        await asyncio.sleep(0.02)
        assert len(host.stderr_tail.encode("utf-8")) <= 1024
        assert "tail-marker" in host.stderr_tail

        response = await host.invoke("large_result", {})
        assert response.ok is False
        assert response.error_type == "response_too_large"

        oversized_request = await host.invoke("large_result", {"value": "z" * 5000})
        assert oversized_request.ok is False
        assert oversized_request.error_type == "request_too_large"
    finally:
        await host.stop()


@pytest.mark.asyncio
async def test_host_rejects_a_mismatched_response_request_id(tmp_path: Path) -> None:
    class FakeStdin:
        def write(self, value: bytes) -> None:
            return None

        async def drain(self) -> None:
            return None

    class FakeStdout:
        async def readline(self) -> bytes:
            return b'{"version":1,"type":"response","request_id":"wrong","ok":true}\n'

    class FakeProcess:
        stdin = FakeStdin()
        stdout = FakeStdout()
        stderr = None
        returncode = 0

    host = DshNodeHost(
        tmp_path,
        source_kind=PluginSourceKind.DSH_PACKAGE.value,
        entry_point="index.cjs",
    )
    host._proc = FakeProcess()  # type: ignore[assignment]

    response = await host.discover()

    assert response.ok is False
    assert response.error_type == "protocol_error"
    assert "response id mismatch" in response.error
    await host.stop()


@pytest.mark.asyncio
async def test_bridge_invocations_do_not_share_node_module_state(tmp_path: Path) -> None:
    source = _write_package(
        tmp_path / "stateful",
        "let counter=0;module.exports={apply(ctx){harness.registerTool(ctx,"
        "harness.defineTool({name:'isolated_counter',description:'counter',parameters:{},"
        "execute:async()=>{counter+=1;await new Promise(r=>setTimeout(r,20));"
        "return {counter}}}))}}",
        name="stateful",
    )
    prepared = await prepare_dsh_installation(
        source,
        plugin_id="stateful",
        plugins_dir=tmp_path / "plugins",
        dsh_plugins_dir=tmp_path / "plugins" / "dsh",
    )
    promote_staging_bundle(prepared.staging_root, prepared.final_root)
    try:
        namespace: dict = {}
        exec(compile(prepared.wrapper_source, "<stateful-wrapper>", "exec"), namespace)
        plugin = namespace["plugin"]

        first, second = await asyncio.gather(
            plugin.tools[0].handler(), plugin.tools[0].handler()
        )

        assert first["counter"] == 1
        assert second["counter"] == 1
    finally:
        shutil.rmtree(tmp_path / "plugins", ignore_errors=True)


@pytest.mark.asyncio
async def test_capability_returns_raw_json_and_enforces_utf8_byte_limit() -> None:
    calls: list[dict] = []

    async def fake_fetch(params: dict) -> dict:
        calls.append(dict(params))
        return {"ok": True, "data": {"message": "中文内容"}}

    broker = DshCapabilityBroker(web_fetch=fake_fetch)
    result = await broker.dispatch(
        "compat.shell.run",
        {
            "command": (
                "curl -sS -m 5 'https://example.test/data' "
                "| iconv -f GB18030 -t UTF-8"
            ),
            "stdoutMaxBytes": 12,
        },
    )

    assert calls[0]["extract"] == "raw_text"
    assert calls[0]["encoding"] == "gb18030"
    assert len(result["stdout"]["text"].encode("utf-8")) <= 12
    assert result["stdout"]["text"].startswith('{"message"')


@pytest.mark.asyncio
async def test_source_path_install_uses_workspace_approval_before_plugin_approval(
    tmp_path: Path,
) -> None:
    from leapflow.plugins.tool_plugins.self_management import SelfManagementPlugin
    from leapflow.security.actions import ActionKind
    from leapflow.tools.execution_context import (
        ToolExecutionContext,
        reset_tool_context,
        set_tool_context,
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class WorkspaceGate:
        actions: list = []

        async def evaluate(self, action):
            self.actions.append(action)
            return type(
                "Result", (), {"approved": False, "denial_message": "outside denied"}
            )()

    class PluginGate:
        async def evaluate(self, action):
            raise AssertionError("plugin approval must not run after workspace denial")

    gate = WorkspaceGate()
    manager = SelfManagementPlugin()
    manager._plugin_approval_gate = PluginGate()
    token = set_tool_context(
        ToolExecutionContext.from_strings(
            workspace_root=str(workspace),
            session_id="dsh-workspace-gate",
            orchestrator=gate,
        )
    )
    try:
        result = await manager._plugin_install_handler(source_path=str(_DYNAMIC))
    finally:
        reset_tool_context(token)

    assert result["ok"] is False
    assert result["error"] == "outside denied"
    assert len(gate.actions) == 1
    assert gate.actions[0].kind == ActionKind.WORKSPACE_ESCAPE.value


@pytest.mark.asyncio
async def test_failed_registration_rolls_back_wrapper_and_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import leapflow.plugins as plugins
    from leapflow.plugins.registry import ToolPluginRegistry
    from leapflow.plugins.scoped_registry import ScopedToolRegistry
    from leapflow.plugins.tool_plugins.self_management import SelfManagementPlugin

    registry = ToolPluginRegistry()
    scoped = ScopedToolRegistry(registry)
    monkeypatch.setattr(plugins, "_registry", registry)
    monkeypatch.setattr(plugins, "_scoped_registry", scoped)

    class Approval:
        async def evaluate(self, action):
            return type("Result", (), {"approved": True, "denial_message": ""})()

    manager = SelfManagementPlugin()
    manager._plugin_install_dir = str(tmp_path / "plugins")
    manager._plugin_approval_gate = Approval()
    monkeypatch.setattr(
        manager,
        "_register_inprocess",
        lambda plugin_id, module_name, target: {
            "ok": False,
            "error": "forced registration failure",
        },
    )

    result = await manager._plugin_install_handler(
        plugin_id="rollback-fixture",
        source_path=str(_PACKAGE),
    )

    assert result == {"ok": False, "error": "forced registration failure"}
    assert not (tmp_path / "plugins" / "rollback_fixture.py").exists()
    assert not (tmp_path / "plugins" / "dsh" / "rollback_fixture").exists()
