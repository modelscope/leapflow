"""Scenario-based tests for safety: confirmation levels, action policy, sandbox."""

from __future__ import annotations

from typing import Optional

import pytest

from conftest import make_skill
from leapflow.engine.confirmation import ConfirmationHandler, ConfirmLevel
from leapflow.skills.action_policy import (
    PolicyContext,
    PolicyDecision,
    PolicyEngine,
    Verdict,
    default_rules,
)
from leapflow.skills.sandbox import SandboxedNamespace
from leapflow.skills.tool_executor import ToolCall


# ═══════════════════════════════════════════════════════════════════
# Confirmation levels
# ═══════════════════════════════════════════════════════════════════


def test_confirm_level_v1_requires_step():
    handler = ConfirmationHandler()
    skill = make_skill(version=1, confidence=0.9)
    assert handler.determine_level(skill) == ConfirmLevel.STEP


def test_confirm_level_low_confidence_requires_step():
    handler = ConfirmationHandler()
    skill = make_skill(version=3, confidence=0.5)
    assert handler.determine_level(skill) == ConfirmLevel.STEP


def test_confirm_level_destructive_requires_confirm():
    handler = ConfirmationHandler()
    skill = make_skill(
        version=2,
        confidence=0.7,
        description="delete files from disk",
    )
    assert handler.determine_level(skill) == ConfirmLevel.CONFIRM


def test_confirm_level_mature_skill_auto():
    handler = ConfirmationHandler()
    skill = make_skill(version=3, confidence=0.9)
    assert handler.determine_level(skill) == ConfirmLevel.AUTO


def test_confirm_level_override_wins():
    handler = ConfirmationHandler()
    skill = make_skill(version=3, confidence=0.9)
    assert (
        handler.determine_level(skill, override=ConfirmLevel.STEP)
        == ConfirmLevel.STEP
    )


# ═══════════════════════════════════════════════════════════════════
# Action policy
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_policy_safe_action_allowed():
    engine = PolicyEngine(default_rules())
    ctx = PolicyContext(skill_name="test", iteration=0)
    call = ToolCall(name="observe_ui", params={})
    decision = await engine.evaluate(call, ctx)
    assert decision.verdict == Verdict.ALLOW


@pytest.mark.asyncio
async def test_policy_send_action_requires_ask():
    engine = PolicyEngine(default_rules())
    ctx = PolicyContext(skill_name="test", iteration=0)
    call = ToolCall(name="click", params={"selector": "AXButton[label=Send]"})
    decision = await engine.evaluate(call, ctx)
    assert decision.verdict == Verdict.ASK
    assert decision.reason == "send_action"


class _DenyAllRule:
    def check(self, call: ToolCall, context: PolicyContext) -> Optional[PolicyDecision]:
        return PolicyDecision(verdict=Verdict.DENY, reason="blocked")


@pytest.mark.asyncio
async def test_policy_custom_rule_injection():
    engine = PolicyEngine(default_rules())
    engine.add_rule(_DenyAllRule())
    ctx = PolicyContext(skill_name="test", iteration=0)
    call = ToolCall(name="observe_ui", params={})
    decision = await engine.evaluate(call, ctx)
    assert decision.verdict == Verdict.DENY
    assert decision.reason == "blocked"


# ═══════════════════════════════════════════════════════════════════
# Sandboxed skill execution
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_sandbox_safe_execution():
    code = """
async def greet(execution, perception, **params):
    name = params.get("name", "world")
    return {"ok": True, "result": f"Hello, {name}!"}
"""
    fn = SandboxedNamespace.compile_skill(code, "greet")
    result = await fn(None, None, name="Alice")
    assert result == {"ok": True, "result": "Hello, Alice!"}


def test_sandbox_blocks_dangerous_builtins():
    ns = SandboxedNamespace.create()
    builtins = ns["__builtins__"]
    for name in ("open", "eval", "exec", "compile", "getattr", "setattr", "delattr"):
        assert name not in builtins


@pytest.mark.asyncio
async def test_sandbox_complex_skill_with_json():
    code = """
async def summarize(execution, perception, **params):
    items = params.get("items", [])
    return {"ok": True, "payload": json.dumps({"count": len(items), "items": items})}
"""
    fn = SandboxedNamespace.compile_skill(code, "summarize")
    result = await fn(None, None, items=["a", "b", "c"])
    assert result["ok"] is True
    assert '"count": 3' in result["payload"]
    assert '"items": ["a", "b", "c"]' in result["payload"]


# ═══════════════════════════════════════════════════════════════════
# Gateway safety boundaries
# ═══════════════════════════════════════════════════════════════════


def test_session_key_normalizes_non_string_fields() -> None:
    from leapflow.gateway.session_router import SessionKey

    key = SessionKey(
        profile="default",
        platform="telegram",
        chat_type="group",
        chat_id=123456,
        thread_id=789,
        user_id=42,
    )

    assert key.chat_id == "123456"
    assert key.thread_id == "789"
    assert key.user_id == "42"
    assert str(key) == "default:telegram:group:123456:789:42"


def test_session_key_rejects_unsafe_serialized_fields() -> None:
    from leapflow.gateway.session_router import SessionKey

    with pytest.raises(ValueError, match="Unsafe"):
        SessionKey(
            profile="default",
            platform="telegram",
            chat_type="group",
            chat_id="safe",
            thread_id="../escape",
        )


def test_gateway_adapter_import_error_preserves_internal_imports(tmp_path, monkeypatch) -> None:
    import sys

    from leapflow.gateway.manifest import AdapterSpec, PlatformManifest
    from leapflow.gateway.server import GatewayServer

    adapter_module = tmp_path / "adapter_mod.py"
    adapter_module.write_text("import missing_subdependency_for_test\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("adapter_mod", None)

    manifest = PlatformManifest(
        platform_id="demo",
        display_name="Demo",
        adapter=AdapterSpec(
            module="adapter_mod",
            class_name="Adapter",
            dependencies=("demo-sdk",),
        ),
    )

    with pytest.raises(ModuleNotFoundError, match="missing_subdependency_for_test"):
        GatewayServer._instantiate_adapter(manifest, {}, {})


def test_gateway_adapter_missing_module_has_install_hint() -> None:
    from leapflow.gateway.manifest import AdapterSpec, PlatformManifest
    from leapflow.gateway.server import GatewayServer

    manifest = PlatformManifest(
        platform_id="demo",
        display_name="Demo",
        adapter=AdapterSpec(
            module="missing_adapter_module_for_test",
            class_name="Adapter",
            dependencies=("demo-sdk",),
        ),
    )

    with pytest.raises(ImportError, match="Install dependencies: pip install demo-sdk"):
        GatewayServer._instantiate_adapter(manifest, {}, {})


@pytest.mark.asyncio
async def test_gateway_start_skips_failed_auto_connect_platform(tmp_path, monkeypatch) -> None:
    from leapflow.gateway.config_store import GatewayConfig, PlatformConfig
    from leapflow.gateway.manifest import PlatformManifest
    from leapflow.gateway.server import GatewayServer

    class FakeConfigStore:
        def load(self) -> GatewayConfig:
            return GatewayConfig(
                platforms={
                    "bad": PlatformConfig(enabled=True),
                    "good": PlatformConfig(enabled=True),
                },
                auto_connect=["bad", "good"],
            )

        def load_platform_credentials(self, platform_id, manifest):
            if platform_id == "bad":
                raise RuntimeError("corrupt credentials")
            return {"token": "ok"}

    server = GatewayServer(tmp_path)
    server._manifests = {
        "bad": PlatformManifest(platform_id="bad", display_name="Bad"),
        "good": PlatformManifest(platform_id="good", display_name="Good"),
    }
    server._config_store = FakeConfigStore()
    monkeypatch.setattr(server, "discover_manifests", lambda: server._manifests)

    async def connect_platform(platform_id, credentials, options=None, *, is_reconnect=False):
        return {"ok": platform_id == "good"}

    monkeypatch.setattr(server, "connect_platform", connect_platform)

    assert await server.start() == 1


@pytest.mark.asyncio
async def test_file_read_gate_supports_legacy_two_argument_check(tmp_path) -> None:
    from leapflow.tools import registry_bootstrap
    from leapflow.tools.file_operations import file_read

    class LegacyReadGate:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def check(self, path: str, mode: str) -> bool:
            self.calls.append((path, mode))
            return True

    target = tmp_path / ".env"
    target.write_text("SECRET=value", encoding="utf-8")
    gate = LegacyReadGate()
    registry_bootstrap.set_file_read_gate(gate)
    try:
        result = await file_read({"path": str(target), "mode": "raw"})
    finally:
        registry_bootstrap.set_file_read_gate(None)

    assert result["ok"] is True
    assert gate.calls == [(str(target.resolve()), "raw")]


@pytest.mark.asyncio
async def test_file_write_gate_supports_legacy_three_argument_check(tmp_path) -> None:
    from leapflow.tools import registry_bootstrap
    from leapflow.tools.file_operations import file_write

    class LegacyWriteGate:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        async def check(self, path: str, content: str, mode: str) -> bool:
            self.calls.append((path, content, mode))
            return True

    target = tmp_path / ".env"
    gate = LegacyWriteGate()
    registry_bootstrap.set_file_write_gate(gate)
    try:
        result = await file_write({"path": str(target), "content": "SECRET=value", "mode": "overwrite"})
    finally:
        registry_bootstrap.set_file_write_gate(None)

    assert result["ok"] is True
    assert target.read_text(encoding="utf-8") == "SECRET=value"
    assert gate.calls == [(str(target.resolve()), "SECRET=value", "overwrite")]
