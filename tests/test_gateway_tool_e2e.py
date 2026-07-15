from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from leapflow.gateway.connectors.protocol import ActionFailure, ActionPreview, ActionResult, ActionSpec, BackendKind
from leapflow.gateway.protocol import OutboundContent, SendResult, SendTarget
from leapflow.gateway.server import GatewayServer
from leapflow.tools.gateway_tool import (
    build_app_connector_prompt_section,
    gateway_connect_handler,
    gateway_send_handler,
    platform_action_handler,
    platform_connect_handler,
    set_gateway_approval_gate,
    set_gateway_server,
)


@pytest.mark.asyncio
async def test_app_slash_payloads_reuse_platform_connect_and_manifests(tmp_path) -> None:
    from leapflow.cli.commands.slash_handlers import build_app_payload

    server = GatewayServer(tmp_path)
    server.discover_manifests()
    ctx = SimpleNamespace(gateway_server=server)
    set_gateway_server(server)
    set_gateway_approval_gate(None)

    try:
        listed = await build_app_payload(ctx, "")
        guide = await build_app_payload(ctx, "FeiShu")
        status = await build_app_payload(ctx, "status feishu")
        actions = await build_app_payload(ctx, "actions feishu")
        telegram_connect = await build_app_payload(ctx, "connect telegram")
        invalid_option = await build_app_payload(ctx, "connect feishu --unknown value")
    finally:
        await server.stop()
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert listed["ok"] is True
    assert listed["view"] == "list"
    assert {entry["id"] for entry in listed["result"]["platforms"]} >= {"feishu", "telegram", "dingtalk"}
    assert guide["ok"] is True
    assert guide["view"] == "guide"
    assert guide["result"]["platform"] == "飞书 (Feishu/Lark)"
    assert guide["result"]["setup_form"]["backend"]["kind"] == "cli"
    assert status["ok"] is True
    assert status["result"]["platform"] == "feishu"
    assert actions["ok"] is True
    assert set(actions["actions"]) == {
        "im.send_message", "im.reply_message", "im.update_message",
        "im.add_reaction", "im.remove_reaction", "im.update_card",
        "im.download_resource",
        "im.list_chats", "im.search_chats",
        "im.list_messages", "im.get_messages", "im.search_messages",
        "im.list_thread_messages",
        "docs.create_markdown", "calendar.create_event",
        "drive.search", "sheets.append_row", "mail.search_unread", "task.create",
    }
    assert telegram_connect["ok"] is True
    assert telegram_connect["view"] == "guide"
    assert telegram_connect["result"]["required_fields"][0]["key"] == "bot_token"
    assert invalid_option["ok"] is False
    assert "Unknown option" in invalid_option["error"]
    assert "feishu" in invalid_option["available"]


def test_gateway_config_store_persists_secret_refs_and_env_override(monkeypatch, tmp_path) -> None:
    from leapflow.gateway.config_store import GatewayConfigStore
    from leapflow.gateway.credential_vault import CredentialVault
    from leapflow.gateway.manifest import CredentialField, PlatformManifest
    from leapflow.security.secrets import FernetSecretVault, secret_ref

    manifest = PlatformManifest(
        platform_id="fake",
        display_name="Fake",
        credentials=(
            CredentialField(key="api_key", label="API Key", secret=True),
            CredentialField(key="base_url", label="Base URL", secret=False),
        ),
    )
    vault = CredentialVault(tmp_path / "secrets")
    store = GatewayConfigStore(tmp_path / "gateway.yaml", vault)

    store.save_platform(
        "fake",
        {"api_key": "sk-gateway", "base_url": "https://gateway.example.invalid"},
        {},
        manifest,
    )

    saved = store.load().platforms["fake"].credentials
    api_key_ref = secret_ref("profile", "gateway", "fake", "api_key")
    assert saved == {"api_key_ref": api_key_ref, "base_url": "https://gateway.example.invalid"}
    assert "sk-gateway" not in (tmp_path / "gateway.yaml").read_text(encoding="utf-8")
    assert FernetSecretVault(tmp_path / "secrets" / "vault.json", tmp_path / "secrets" / "vault.key").get(api_key_ref) == "sk-gateway"

    assert store.load_platform_credentials("fake", manifest) == {
        "api_key": "sk-gateway",
        "base_url": "https://gateway.example.invalid",
    }

    monkeypatch.setenv("LEAPFLOW_FAKE_API_KEY", "sk-env")
    assert store.load_platform_credentials("fake", manifest)["api_key"] == "sk-env"


def test_gateway_config_store_warns_for_missing_secret_ref(caplog, tmp_path) -> None:
    from leapflow.gateway.config_store import GatewayConfig, GatewayConfigStore, PlatformConfig
    from leapflow.gateway.credential_vault import CredentialVault
    from leapflow.gateway.manifest import CredentialField, PlatformManifest

    manifest = PlatformManifest(
        platform_id="fake",
        display_name="Fake",
        credentials=(CredentialField(key="api_key", label="API Key", secret=True),),
    )
    store = GatewayConfigStore(tmp_path / "gateway.yaml", CredentialVault(tmp_path / "secrets"))
    store.save(GatewayConfig(platforms={
        "fake": PlatformConfig(credentials={"api_key_ref": "secret://profile/gateway/fake/api_key"}),
    }))

    with caplog.at_level("WARNING"):
        assert store.load_platform_credentials("fake", manifest) == {}

    assert "Missing gateway credential ref" in caplog.text


@pytest.mark.asyncio
async def test_gateway_connect_tool_can_connect_builtin_webhook(tmp_path) -> None:
    server = GatewayServer(tmp_path)
    server.discover_manifests()
    set_gateway_server(server)

    try:
        listed = await gateway_connect_handler({"action": "list"})
        assert listed["ok"] is True
        assert {entry["id"] for entry in listed["platforms"]} >= {"webhook", "api_server"}

        guide = await gateway_connect_handler({"action": "guide", "platform": "webhook"})
        assert guide["ok"] is True
        assert guide["platform"] == "Webhook (Generic)"
        assert "setup_form" in guide

        connected = await gateway_connect_handler({
            "action": "connect",
            "platform": "webhook",
            "credentials": {"webhook_secret": ""},
            "options": {"host": "127.0.0.1", "port": 0, "path": "/webhook"},
        })
        assert connected["ok"] is True
        assert connected["status"] == "connected"

        status = await gateway_connect_handler({"action": "status", "platform": "webhook"})
        assert status["ok"] is True
        assert status["connected"] is True
    finally:
        await server.stop()
        set_gateway_approval_gate(None)
        set_gateway_server(None)


class FakeSendAdapter:
    platform_id = "fake"
    supports_async_delivery = True
    splits_long_messages = False
    max_message_length = 0

    def __init__(self) -> None:
        self.sent: list[tuple[SendTarget, OutboundContent]] = []
        self.spec = ActionSpec(
            name="im.send_message",
            backend_kind=BackendKind.CLI.value,
            effect="send",
            schema={
                "type": "object",
                "required": ["chat_id", "text"],
                "properties": {"chat_id": {"type": "string"}, "text": {"type": "string"}},
            },
            risk_level="high",
        )

    async def connect(self, *, is_reconnect: bool = False) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def send(self, target: SendTarget, content: OutboundContent) -> SendResult:
        self.sent.append((target, content))
        return SendResult(ok=True, message_id="fake-1")

    def action_spec(self, action: str) -> ActionSpec | None:
        return self.spec if action == self.spec.name else None

    def action_specs(self) -> dict[str, ActionSpec]:
        return {self.spec.name: self.spec}

    async def preview_action(self, action: str, payload: dict) -> ActionPreview:
        if action != self.spec.name:
            return ActionPreview(ok=False, error="unknown action")
        return ActionPreview(ok=True, summary=f"preview {action}")

    async def execute_action(self, action: str, payload: dict) -> ActionResult:
        self.sent.append((
            SendTarget(
                platform="fake",
                chat_id=str(payload.get("chat_id", "")),
                thread_id=str(payload.get("thread_id", "")),
            ),
            OutboundContent(text=str(payload.get("text", ""))),
        ))
        return ActionResult(ok=True, resource_id="fake-action-1")


class DenyGate:
    async def evaluate(self, action):
        class Result:
            approved = False
            denial_message = "denied for test"

        return Result()


@pytest.mark.asyncio
async def test_gateway_send_tool_dispatches_to_connected_adapter(tmp_path) -> None:
    server = GatewayServer(tmp_path)
    adapter = FakeSendAdapter()
    server._adapters["fake"] = adapter
    set_gateway_server(server)
    set_gateway_approval_gate(None)

    try:
        result = await gateway_send_handler({
            "platform": "fake",
            "chat_id": "chat-1",
            "text": "hello outbound",
            "thread_id": "thread-1",
        })
    finally:
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert result["ok"] is True
    assert result["resource_id"] == "fake-action-1"
    assert result["message_id"] == "fake-action-1"
    assert result["source_tool"] == "gateway_send"
    target, content = adapter.sent[0]
    assert target.chat_id == "chat-1"
    assert target.thread_id == "thread-1"
    assert content.text == "hello outbound"


@pytest.mark.asyncio
async def test_gateway_send_tool_honors_approval_denial(tmp_path) -> None:
    server = GatewayServer(tmp_path)
    adapter = FakeSendAdapter()
    server._adapters["fake"] = adapter
    set_gateway_server(server)
    set_gateway_approval_gate(DenyGate())

    try:
        result = await gateway_send_handler({
            "platform": "fake",
            "chat_id": "chat-1",
            "text": "blocked outbound",
        })
    finally:
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert result == {"ok": False, "error": "denied for test"}
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_gateway_send_returns_onboarding_hint_when_registered_action_is_not_connected(tmp_path) -> None:
    server = GatewayServer(tmp_path)
    server.discover_manifests()
    set_gateway_server(server)
    set_gateway_approval_gate(None)

    try:
        result = await gateway_send_handler({
            "platform": "feishu",
            "chat_id": "oc_1",
            "text": "hello",
        })
    finally:
        await server.stop()
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert result["ok"] is False
    assert result["failure_code"] == "platform_not_connected"
    assert "registered action 'im.send_message'" in result["error"]
    assert "platform_connect action=preflight platform=feishu" in result["next_steps"]


@pytest.mark.asyncio
async def test_platform_action_distinguishes_registered_but_not_connected_action(tmp_path) -> None:
    server = GatewayServer(tmp_path)
    server.discover_manifests()
    set_gateway_server(server)
    set_gateway_approval_gate(None)

    try:
        result = await platform_action_handler({
            "platform": "feishu",
            "action": "im.send_message",
            "payload": {"chat_id": "oc_1", "text": "hello"},
        })
    finally:
        await server.stop()
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert result["ok"] is False
    assert result["failure_code"] == "platform_not_connected"
    assert result["recovery_hint"]
    assert "platform_connect action=connect platform=feishu" in result["next_steps"]


@pytest.mark.asyncio
async def test_platform_action_reports_unknown_action_with_expanded_registry(tmp_path) -> None:
    server = GatewayServer(tmp_path)
    server.discover_manifests()
    set_gateway_server(server)
    set_gateway_approval_gate(None)

    try:
        # im.fake_invented is NOT in the YAML registry
        result = await platform_action_handler({
            "platform": "feishu",
            "action": "im.fake_invented",
            "payload": {},
        })
    finally:
        await server.stop()
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert result["ok"] is False
    assert result["failure_code"] == "unknown_platform_action"
    assert result["retryable"] is True
    # All 9 registered actions should be in available_action_names
    assert "im.send_message" in result["available_action_names"]
    assert "im.list_chats" in result["available_action_names"]
    assert "im.search_chats" in result["available_action_names"]
    assert any(item["name"] == "im.send_message" for item in result["available_actions"])


@pytest.mark.asyncio
async def test_platform_action_list_chats_and_search_chats_are_registered(tmp_path) -> None:
    """im.list_chats and im.search_chats are now registered discovery actions."""
    from leapflow.gateway.action_packs.feishu import ACTION_SPECS

    assert "im.list_chats" in ACTION_SPECS
    assert "im.search_chats" in ACTION_SPECS
    assert ACTION_SPECS["im.list_chats"].effect == "read"
    assert ACTION_SPECS["im.search_chats"].effect == "read"
    assert ACTION_SPECS["im.search_chats"].schema.get("required") == ["query"]

    server = GatewayServer(tmp_path)
    server.discover_manifests()
    set_gateway_server(server)
    set_gateway_approval_gate(None)

    try:
        # These should be platform_not_connected (registered but not connected)
        # rather than unknown_platform_action
        result_list = await platform_action_handler({
            "platform": "feishu",
            "action": "im.list_chats",
            "payload": {},
        })
        result_search = await platform_action_handler({
            "platform": "feishu",
            "action": "im.search_chats",
            "payload": {"query": "LeapFlow"},
        })
    finally:
        await server.stop()
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert result_list["failure_code"] == "platform_not_connected"
    assert result_search["failure_code"] == "platform_not_connected"


@pytest.mark.asyncio
async def test_platform_action_reports_wrong_namespace_for_management_action(tmp_path) -> None:
    server = GatewayServer(tmp_path)
    server.discover_manifests()
    set_gateway_server(server)
    set_gateway_approval_gate(None)

    try:
        result = await platform_action_handler({
            "platform": "feishu",
            "action": "list",
            "payload": {},
        })
    finally:
        await server.stop()
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert result["ok"] is False
    assert result["failure_code"] == "wrong_action_namespace"
    assert result["correct_tool"] == "platform_connect"
    assert result["retryable"] is True
    assert "list" in result["available_management_actions"]


@pytest.mark.asyncio
async def test_platform_action_reports_unknown_platform_with_available_list(tmp_path) -> None:
    server = GatewayServer(tmp_path)
    server.discover_manifests()
    set_gateway_server(server)
    set_gateway_approval_gate(None)

    try:
        result = await platform_action_handler({
            "platform": "gateway",
            "action": "list_actions",
            "payload": {"platform": "feishu"},
        })
    finally:
        await server.stop()
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert result["ok"] is False
    assert result["failure_code"] == "unknown_platform"
    assert result["retryable"] is True
    assert "feishu" in result["available_platforms"]


class CaptureGate:
    def __init__(self) -> None:
        self.actions = []

    async def evaluate(self, action):
        self.actions.append(action)

        class Result:
            approved = True
            denial_message = ""

        return Result()


class PermissionFailingAdapter(FakeSendAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.read_spec = ActionSpec(
            name="im.list_messages",
            backend_kind=BackendKind.CLI.value,
            effect="read",
            capability="im.message.read",
            schema={
                "type": "object",
                "required": ["chat_id"],
                "properties": {"chat_id": {"type": "string"}},
            },
            risk_level="low",
            output_policy="raw",
        )
        self.execute_calls: list[str] = []

    def action_spec(self, action: str) -> ActionSpec | None:
        if action == self.spec.name:
            return self.spec
        if action == self.read_spec.name:
            return self.read_spec
        return None

    def action_specs(self) -> dict[str, ActionSpec]:
        return {self.spec.name: self.spec, self.read_spec.name: self.read_spec}

    async def preview_action(self, action: str, payload: dict) -> ActionPreview:
        spec = self.action_spec(action)
        if spec is None:
            return ActionPreview(ok=False, error="unknown action")
        return ActionPreview(ok=True, summary=f"preview {action}")

    async def execute_action(self, action: str, payload: dict) -> ActionResult:
        self.execute_calls.append(action)
        if action == "im.list_messages":
            failure = ActionFailure(
                failure_class="authorization",
                failure_code="missing_scope",
                message="access denied for this operation",
                recoverability="admin_required",
                retryable=False,
                missing_scopes=("im:message.group_msg",),
                capability="im.message.read",
                blocks_approval=True,
            )
            return ActionResult(
                ok=False,
                error="access denied for this operation",
                failure=failure,
            )
        return await super().execute_action(action, payload)


@pytest.mark.asyncio
async def test_platform_action_blocks_known_permission_failure_before_approval(tmp_path) -> None:
    """Auth failure degrades side-effects while read actions can revalidate."""
    server = GatewayServer(tmp_path)
    adapter = PermissionFailingAdapter()
    gate = CaptureGate()
    server._adapters["fake"] = adapter
    set_gateway_server(server)
    set_gateway_approval_gate(gate)

    try:
        first = await platform_action_handler({
            "platform": "fake",
            "action": "im.list_messages",
            "payload": {"chat_id": "chat-1"},
        })
        second = await platform_action_handler({
            "platform": "fake",
            "action": "im.list_messages",
            "payload": {"chat_id": "chat-1"},
        })
        send = await platform_action_handler({
            "platform": "fake",
            "action": "im.send_message",
            "payload": {"chat_id": "chat-1", "text": "should be blocked"},
        })
    finally:
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    # First call executes and fails with auth error
    assert first["ok"] is False
    assert first["failure_class"] == "authorization"
    assert first["blocks_approval"] is True
    # Read calls are allowed to revalidate and therefore both reach execution.
    assert len(gate.actions) == 2
    assert adapter.execute_calls == ["im.list_messages", "im.list_messages"]

    # Second call still reports the fresh backend failure.
    assert second["ok"] is False
    assert second["failure_class"] == "authorization"
    assert second["failure_code"] == "missing_scope"
    assert second["capability"] == "im.message.read"

    # Send blocked by platform degradation (no execution, no approval)
    assert send["ok"] is False
    assert send["failure_code"] == "platform_degraded"
    assert "llm_instruction" in send


class RecoveringPermissionAdapter(PermissionFailingAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_read = True

    async def execute_action(self, action: str, payload: dict) -> ActionResult:
        if action == "im.list_messages" and self.fail_next_read:
            self.fail_next_read = False
            return await super().execute_action(action, payload)
        self.execute_calls.append(action)
        return ActionResult(ok=True, resource_id=f"ok_{len(self.execute_calls)}", data=dict(payload))


@pytest.mark.asyncio
async def test_platform_action_success_revalidates_and_clears_permission_failure(tmp_path) -> None:
    server = GatewayServer(tmp_path)
    adapter = RecoveringPermissionAdapter()
    gate = CaptureGate()
    server._adapters["fake"] = adapter
    set_gateway_server(server)
    set_gateway_approval_gate(gate)

    try:
        first = await platform_action_handler({
            "platform": "fake",
            "action": "im.list_messages",
            "payload": {"chat_id": "chat-1"},
        })
        second = await platform_action_handler({
            "platform": "fake",
            "action": "im.list_messages",
            "payload": {"chat_id": "chat-1"},
        })
        send = await platform_action_handler({
            "platform": "fake",
            "action": "im.send_message",
            "payload": {"chat_id": "chat-1", "text": "allowed after revalidation"},
        })
    finally:
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert first["ok"] is False
    assert first["failure_class"] == "authorization"
    assert second["ok"] is True
    assert send["ok"] is True
    assert send.get("failure_code") != "platform_degraded"
    assert adapter.execute_calls == ["im.list_messages", "im.list_messages", "im.send_message"]


@pytest.mark.asyncio
async def test_platform_action_uses_registered_spec_for_approval_and_summary(tmp_path) -> None:
    server = GatewayServer(tmp_path)
    adapter = FakeSendAdapter()
    gate = CaptureGate()
    server._adapters["fake"] = adapter
    set_gateway_server(server)
    set_gateway_approval_gate(gate)

    try:
        result = await platform_action_handler({
            "platform": "fake",
            "action": "im.send_message",
            "payload": {"chat_id": "chat-1", "text": "hello"},
        })
    finally:
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert result["ok"] is True
    assert result["platform"] == "fake"
    assert result["action"] == "im.send_message"
    assert result["output_policy"] == "summary"
    assert gate.actions[0].effect == "send"
    assert gate.actions[0].metadata["backend_kind"] == "cli"
    assert gate.actions[0].metadata["risk_level"] == "high"


@pytest.mark.asyncio
async def test_platform_events_report_unavailable_for_feishu(tmp_path) -> None:
    from leapflow.gateway.adapters.feishu import FeishuAdapter
    from leapflow.gateway.connectors.protocol import BackendStatus

    class ReadyBackend:
        kind = "cli"

        async def status(self) -> BackendStatus:
            return BackendStatus(ok=True, backend_kind=self.kind)

        async def authenticate(self, payload):
            return BackendStatus(ok=True, backend_kind=self.kind)

        async def execute(self, spec, payload):
            return ActionResult(ok=True)

    server = GatewayServer(tmp_path)
    adapter = FeishuAdapter(backend=ReadyBackend())
    server._adapters["feishu"] = adapter
    set_gateway_server(server)
    set_gateway_approval_gate(None)

    try:
        status = await platform_connect_handler({"action": "events_status", "platform": "feishu"})
        started = await platform_connect_handler({"action": "events_start", "platform": "feishu"})
    finally:
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert status["ok"] is False
    assert started["ok"] is False
    assert status["metadata"]["available"] is False
    assert "not enabled" in status["detail"].lower()


@pytest.mark.asyncio
async def test_platform_connect_can_configure_credentialless_platform(tmp_path) -> None:
    server = GatewayServer(tmp_path)
    server.discover_manifests()
    set_gateway_server(server)
    set_gateway_approval_gate(None)

    try:
        guide = await platform_connect_handler({"action": "guide", "platform": "feishu"})
    finally:
        await server.stop()
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert guide["ok"] is True
    assert guide["required_fields"] == []
    assert guide["setup_form"]["backend"]["kind"] == "cli"
    assert guide["setup_form"]["actions"]["pack"] == "leapflow.gateway.action_packs.feishu"
    assert guide["preflight_checks"][0]["command"] == "lark-cli --version"
    assert guide["preflight_checks"][0]["kind"] == "check"
    assert guide["preflight_checks"][1]["command"] == "lark-cli auth login --json"
    assert guide["preflight_checks"][1]["kind"] == "interactive_auth"
    assert guide["preflight_checks"][2]["command"] == "lark-cli auth status --json"
    assert guide["preflight_result"]["backend_kind"] == "cli"
    assert guide["onboarding_state"]["platform_id"] == "feishu"
    assert guide["recovery_hint"]
    assert "lark-cli" in guide["setup_guide"]


@pytest.mark.asyncio
async def test_platform_connect_records_pending_onboarding_state_for_cli_preflight(tmp_path) -> None:
    server = GatewayServer(tmp_path)
    server.discover_manifests()
    set_gateway_server(server)
    set_gateway_approval_gate(None)

    try:
        guide = await platform_connect_handler({
            "action": "guide",
            "platform": "feishu",
            "options": {"binary": "definitely-missing-cli-for-onboarding-test"},
        })
        section = build_app_connector_prompt_section()
        preflight = await platform_connect_handler({
            "action": "preflight",
            "platform": "feishu",
            "options": {"binary": "definitely-missing-cli-for-onboarding-test"},
        })
    finally:
        await server.stop()
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert guide["ok"] is True
    assert guide["preflight_result"]["ready"] is False
    assert guide["onboarding_state"]["stage"] == "cli_missing"
    assert guide["preflight_result"]["checks"][0]["status"] == "failed"
    assert guide["preflight_result"]["checks"][1]["status"] == "blocked"
    assert "Pending App Onboarding State" in section
    assert "platform=`feishu`" in section
    assert preflight["ok"] is False
    assert preflight["onboarding_state"]["stage"] == "cli_missing"


@pytest.mark.asyncio
async def test_app_connector_prompt_section_exposes_supported_apps_without_classifying_text(tmp_path) -> None:
    server = GatewayServer(tmp_path)
    server.discover_manifests()
    set_gateway_server(server)
    set_gateway_approval_gate(None)

    try:
        section = build_app_connector_prompt_section()
    finally:
        await server.stop()
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert "App Connector Capability Index" in section
    assert "`feishu`" in section
    assert "`telegram`" in section
    assert "`dingtalk`" in section
    assert "platform_connect" in section
    assert "SDK/Webhook sample code" in section
    assert "im.send_message" in section
    assert "management namespace" in section


@pytest.mark.asyncio
async def test_platform_connect_prompt_contract_still_uses_gateway_handler(tmp_path) -> None:
    server = GatewayServer(tmp_path)
    server.discover_manifests()
    set_gateway_server(server)
    set_gateway_approval_gate(None)

    try:
        guide = await platform_connect_handler({"action": "guide", "platform": "telegram"})
        connect_without_secret = await platform_connect_handler({"action": "connect", "platform": "telegram"})
    finally:
        await server.stop()
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert guide["ok"] is True
    assert guide["platform"] == "Telegram"
    assert guide["required_fields"][0]["key"] == "bot_token"
    assert connect_without_secret["ok"] is True
    assert connect_without_secret["setup_form"]["fields"][0]["type"] == "password"


@pytest.mark.asyncio
async def test_platform_connect_failure_returns_feishu_recovery_hint(tmp_path) -> None:
    server = GatewayServer(tmp_path)
    server.discover_manifests()
    set_gateway_server(server)
    set_gateway_approval_gate(None)

    try:
        result = await platform_connect_handler({
            "action": "connect",
            "platform": "feishu",
            "options": {"binary": "definitely-missing-lark-cli-for-test"},
        })
    finally:
        await server.stop()
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert result["ok"] is False
    assert "Connection failed" in result["error"]
    assert "Install 'definitely-missing-lark-cli-for-test'" in result["recovery_hint"]
    assert result["next_steps"]
    assert result["diagnostics"]["profile"] == "default"


@pytest.mark.asyncio
async def test_platform_status_exposes_feishu_connector_diagnostics(tmp_path) -> None:
    from leapflow.gateway.adapters.feishu import FeishuAdapter
    from leapflow.gateway.connectors.protocol import BackendStatus

    class ReadyBackend:
        kind = "cli"
        binary = "lark-cli"
        profile = "work"
        identity = "bot"

        async def status(self) -> BackendStatus:
            return BackendStatus(ok=True, backend_kind=self.kind)

        async def authenticate(self, payload):
            return BackendStatus(ok=True, backend_kind=self.kind)

        async def execute(self, spec, payload):
            return ActionResult(ok=True)

    server = GatewayServer(tmp_path)
    server.discover_manifests()
    server._adapters["feishu"] = FeishuAdapter(backend=ReadyBackend())
    server._connected_since["feishu"] = 1.0
    set_gateway_server(server)
    set_gateway_approval_gate(None)

    try:
        status = await platform_connect_handler({"action": "status", "platform": "feishu"})
        listed = await platform_connect_handler({"action": "list"})
    finally:
        await server.stop()
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert status["ok"] is True
    assert status["diagnostics"]["backend_kind"] == "cli"
    assert status["diagnostics"]["profile"] == "work"
    assert status["diagnostics"]["event_source"]["mode"] == "outbound_actions_only"
    feishu_entry = next(entry for entry in listed["platforms"] if entry["id"] == "feishu")
    assert feishu_entry["diagnostics"]["identity"] == "bot"


@pytest.mark.asyncio
async def test_platform_action_honors_approval_denial(tmp_path) -> None:
    server = GatewayServer(tmp_path)
    adapter = FakeSendAdapter()
    server._adapters["fake"] = adapter
    set_gateway_server(server)
    set_gateway_approval_gate(DenyGate())

    try:
        result = await platform_action_handler({
            "platform": "fake",
            "action": "im.send_message",
            "payload": {"chat_id": "chat-1", "text": "blocked outbound"},
        })
    finally:
        set_gateway_approval_gate(None)
        set_gateway_server(None)

    assert result == {"ok": False, "error": "denied for test"}
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_gateway_send_reply_returns_success_for_empty_text(tmp_path) -> None:
    from leapflow.gateway.protocol import MessageSource

    server = GatewayServer(tmp_path)
    adapter = FakeSendAdapter()
    server._adapters["fake"] = adapter

    result = await server.send_reply(MessageSource(platform="fake", chat_id="chat-1"), "")

    assert result is not None
    assert result.ok is True
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_gateway_consumer_saves_checkpoint_and_dedup_on_source_crash(tmp_path) -> None:
    from leapflow.gateway.connectors.protocol import EventSourceStatus

    class CrashingSource:
        platform_id = "fake"
        backend_kind = "test"

        async def start(self, *, checkpoint: str = "") -> EventSourceStatus:
            return EventSourceStatus(ok=True, backend_kind=self.backend_kind)

        async def stop(self) -> EventSourceStatus:
            return EventSourceStatus(ok=True, backend_kind=self.backend_kind)

        async def status(self) -> EventSourceStatus:
            return EventSourceStatus(ok=True, backend_kind=self.backend_kind)

        async def events(self):
            raise RuntimeError("source crashed")
            if False:
                yield None

    server = GatewayServer(tmp_path)
    saved: list[str] = []
    server._save_checkpoint = lambda platform_id: saved.append(f"checkpoint:{platform_id}")  # type: ignore[method-assign]
    server._save_dedup_state = lambda platform_id: saved.append(f"dedup:{platform_id}")  # type: ignore[method-assign]

    await server._consume_platform_events("fake", CrashingSource())  # type: ignore[arg-type]

    assert saved == ["checkpoint:fake", "dedup:fake"]


@pytest.mark.asyncio
async def test_composite_event_source_cleans_child_tasks_on_cancel() -> None:
    from leapflow.gateway.connectors.composite_event_source import CompositeEventSource
    from leapflow.gateway.connectors.protocol import EventSourceStatus

    class HangingSource:
        platform_id = "fake"
        backend_kind = "test"

        def __init__(self) -> None:
            self.cancelled = False

        async def start(self, *, checkpoint: str = "") -> EventSourceStatus:
            return EventSourceStatus(ok=True, backend_kind=self.backend_kind)

        async def stop(self) -> EventSourceStatus:
            return EventSourceStatus(ok=True, backend_kind=self.backend_kind)

        async def status(self) -> EventSourceStatus:
            return EventSourceStatus(ok=True, backend_kind=self.backend_kind)

        async def events(self):
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True
            if False:
                yield None

    source = HangingSource()
    composite = CompositeEventSource([source], platform_id="fake")
    await composite.start()
    iterator = composite.events()
    task = asyncio.create_task(iterator.__anext__())
    await asyncio.sleep(0)

    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, StopAsyncIteration):
        pass

    assert source.cancelled is True
    assert composite._tasks == []


@pytest.mark.asyncio
async def test_lark_event_source_kills_identity_subprocess_on_timeout(monkeypatch) -> None:
    import leapflow.gateway.connectors.lark_event_source as lark_event_source

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode = None
            self.killed = False
            self.waited = False

        async def communicate(self):
            return b"{}", b""

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> None:
            self.waited = True

    proc = FakeProcess()

    async def fake_create_subprocess_exec(*args, **kwargs):
        return proc

    async def fake_wait_for(awaitable, timeout):
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(lark_event_source.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(lark_event_source.asyncio, "wait_for", fake_wait_for)

    source = lark_event_source.LarkCliEventSource(binary="lark-cli", profile="default")
    identity = await source.fetch_bot_identity()

    assert identity.open_id == ""
    assert proc.killed is True
    assert proc.waited is True


def test_duckdb_deduplication_store_uses_batch_insert() -> None:
    from leapflow.gateway.checkpoint_store import DuckDBDeduplicationStore

    class FakeConnection:
        def __init__(self) -> None:
            self.executed: list[tuple[str, object]] = []
            self.executemany_calls: list[tuple[str, list[tuple[str, str, float]]]] = []

        def execute(self, sql, params=None):
            self.executed.append((str(sql), params))
            return self

        def executemany(self, sql, rows):
            self.executemany_calls.append((str(sql), list(rows)))
            return self

    class Holder:
        def __init__(self) -> None:
            self.connection = FakeConnection()

    holder = Holder()
    store = DuckDBDeduplicationStore(holder)
    store.save_batch("feishu", [f"event-{i}" for i in range(1005)])

    assert len(holder.connection.executemany_calls) == 1
    _, rows = holder.connection.executemany_calls[0]
    assert len(rows) == 1000
    assert rows[0][1] == "event-5"
    assert rows[-1][1] == "event-1004"
