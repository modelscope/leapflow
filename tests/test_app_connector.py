from __future__ import annotations

import pytest

from leapflow.gateway.adapters.feishu import FeishuAdapter
from leapflow.gateway.capability_health import CapabilityHealthLedger
from leapflow.gateway.backends.lark_cli_errors import classify_lark_cli_failure
from leapflow.gateway.connectors.action_registry import (
    ActionRegistry,
    ValidationResult,
    normalize_payload,
    summarize_action_result,
    validate_payload,
)
from leapflow.gateway.connectors.protocol import (
    ActionAuthSpec,
    ActionFailure,
    ActionPreview,
    ActionResult,
    ActionSpec,
    BackendStatus,
)


class FakeBackend:
    kind = "cli"

    async def status(self) -> BackendStatus:
        return BackendStatus(ok=True, backend_kind=self.kind)

    async def authenticate(self, payload) -> BackendStatus:
        return BackendStatus(ok=True, backend_kind=self.kind, metadata=dict(payload))

    async def execute(self, spec, payload) -> ActionResult:
        return ActionResult(ok=True, resource_id=spec.name, data=dict(payload))

    async def preview(self, spec, payload) -> ActionPreview:
        return ActionPreview(ok=True, summary=f"preview {spec.name}")


@pytest.mark.asyncio
async def test_feishu_adapter_exposes_app_connector_actions() -> None:
    adapter = FeishuAdapter(backend=FakeBackend())

    specs = adapter.action_specs()
    result = await adapter.execute_action(
        "im.send_message",
        {"chat_id": "oc_1", "text": "hello"},
    )

    assert "im.send_message" in specs
    assert "docs.create_markdown" in specs
    assert "calendar.create_event" in specs
    assert "mail.search_unread" in specs
    assert result.ok is True
    assert result.resource_id == "im.send_message"


@pytest.mark.asyncio
async def test_feishu_adapter_rejects_unknown_action() -> None:
    adapter = FeishuAdapter(backend=FakeBackend())

    result = await adapter.execute_action("docs.unknown", {})

    assert result.ok is False
    assert "Unknown Feishu action" in result.error


@pytest.mark.asyncio
async def test_feishu_adapter_validates_registered_action_schema() -> None:
    adapter = FeishuAdapter(backend=FakeBackend())

    missing = await adapter.execute_action("docs.create_markdown", {"title": "Doc"})
    wrong_type = await adapter.execute_action(
        "sheets.append_row",
        {"spreadsheet_token": "s", "sheet_id": "sid", "values": "not-array"},
    )

    assert missing.ok is False
    assert "Missing required fields: markdown" in missing.error
    assert wrong_type.ok is False
    assert "values" in wrong_type.error


@pytest.mark.asyncio
async def test_feishu_adapter_returns_cli_preview() -> None:
    adapter = FeishuAdapter(profile="bot-reader", backend=FakeBackend())

    preview = await adapter.preview_action(
        "im.send_message",
        {"chat_id": "oc_1", "text": "hello"},
    )

    assert preview.ok is True
    assert preview.summary == "preview im.send_message"


def test_action_registry_loads_standard_action_pack_export() -> None:
    registry = ActionRegistry.from_module("leapflow.gateway.action_packs.feishu")

    assert registry.get("im.send_message") is not None
    assert registry.get("task.create") is not None


def test_yaml_action_pack_loads_auth_contract() -> None:
    registry = ActionRegistry.from_module("leapflow.gateway.action_packs.feishu")

    list_messages = registry.get("im.list_messages")
    send_message = registry.get("im.send_message")
    search_messages = registry.get("im.search_messages")

    assert list_messages is not None
    assert list_messages.capability == "im.message.read"
    assert list_messages.auth.identities == ("user", "bot")
    assert "im:message:readonly" in list_messages.auth.scopes["common"]
    assert "im:message.group_msg:get_as_user" in list_messages.auth.scopes["user"]
    assert "im:message.group_msg" in list_messages.auth.scopes["bot"]
    assert list_messages.auth.resource_fields == ("chat_id",)

    assert send_message is not None
    assert send_message.capability == "im.message.send"
    assert "im:message:send_as_bot" in send_message.auth.scopes["bot"]

    assert search_messages is not None
    assert search_messages.auth.identities == ("user",)
    assert "search:message" in search_messages.auth.scopes["user"]


def test_summarize_action_result_preserves_structured_failure() -> None:
    spec = ActionSpec(
        name="im.list_messages",
        backend_kind="cli",
        capability="im.message.read",
        output_policy="raw",
    )
    failure = ActionFailure(
        failure_class="authorization",
        failure_code="missing_scope",
        message="access denied",
        recoverability="admin_required",
        retryable=False,
        missing_scopes=("im:message.group_msg",),
        capability="im.message.read",
        blocks_approval=True,
    )

    summary = summarize_action_result(
        spec,
        ActionResult(ok=False, error="access denied", failure=failure),
    )

    assert summary["ok"] is False
    assert summary["failure_class"] == "authorization"
    assert summary["failure_code"] == "missing_scope"
    assert summary["recoverability"] == "admin_required"
    assert summary["blocks_approval"] is True
    assert summary["missing_scopes"] == ["im:message.group_msg"]
    assert summary["capability"] == "im.message.read"


def test_capability_health_ledger_blocks_only_matching_capability() -> None:
    ledger = CapabilityHealthLedger()
    read_spec = ActionSpec(
        name="im.list_messages",
        backend_kind="cli",
        capability="im.message.read",
        auth=ActionAuthSpec(scopes={"common": ("im:message:readonly",)}),
    )
    send_spec = ActionSpec(
        name="im.send_message",
        backend_kind="cli",
        capability="im.message.send",
        auth=ActionAuthSpec(scopes={"common": ("im:message:send_as_bot",)}),
    )
    failure = ActionFailure(
        failure_class="authorization",
        failure_code="missing_scope",
        message="access denied",
        recoverability="admin_required",
        retryable=False,
        missing_scopes=("im:message.group_msg",),
        capability="im.message.read",
        blocks_approval=True,
    )

    ledger.record_failure("feishu", read_spec.capability, failure)

    blocked = ledger.check_feasibility("feishu", read_spec)
    allowed = ledger.check_feasibility("feishu", send_spec)

    assert blocked["ok"] is False
    assert blocked["skip_approval"] is True
    assert blocked["failure_code"] == "missing_scope"
    assert blocked["capability"] == "im.message.read"
    assert allowed == {"ok": True}


def test_lark_cli_error_classifier_from_plain_access_denied() -> None:
    spec = ActionSpec(name="docs.create_markdown", backend_kind="cli", capability="docs.create")

    failure = classify_lark_cli_failure(
        spec,
        "access denied for this operation; possible causes: missing scope",
        {},
        binary="lark-cli",
        profile="work",
        identity="bot",
    )

    assert failure.failure_class == "authorization"
    assert failure.failure_code == "access_denied"
    assert failure.recoverability == "admin_required"
    assert failure.retryable is False
    assert failure.blocks_approval is True
    assert failure.capability == "docs.create"


# ═══════════════════════════════════════════════════════════════
# Payload normalization tests
# ═══════════════════════════════════════════════════════════════

_SEND_SPEC = ActionSpec(
    name="im.send_message",
    backend_kind="cli",
    schema={
        "required": ["chat_id", "text"],
        "properties": {
            "chat_id": {"type": "string", "description": "Target chat ID."},
            "text": {"type": "string", "description": "Message text."},
            "thread_id": {"type": "string", "description": "Optional thread."},
        },
    },
)

_LIST_SPEC = ActionSpec(
    name="im.list_messages",
    backend_kind="cli",
    schema={
        "required": ["chat_id"],
        "properties": {
            "chat_id": {"type": "string", "description": "Chat ID."},
        },
    },
)


class TestNormalizePayload:
    """Tests for normalize_payload — generous input acceptance."""

    def test_correct_payload_passes_through(self) -> None:
        params = {
            "platform": "feishu",
            "action": "im.send_message",
            "payload": {"chat_id": "oc_1", "text": "hello"},
        }
        result = normalize_payload(_SEND_SPEC, params)
        assert result == {"chat_id": "oc_1", "text": "hello"}

    def test_top_level_fields_lifted_into_payload(self) -> None:
        params = {
            "platform": "feishu",
            "action": "im.list_messages",
            "chat_id": "oc_1",
        }
        result = normalize_payload(_LIST_SPEC, params)
        assert result == {"chat_id": "oc_1"}

    def test_top_level_fields_lifted_when_payload_empty(self) -> None:
        params = {
            "platform": "feishu",
            "action": "im.send_message",
            "payload": {},
            "chat_id": "oc_1",
            "text": "hello",
        }
        result = normalize_payload(_SEND_SPEC, params)
        assert result == {"chat_id": "oc_1", "text": "hello"}

    def test_explicit_payload_takes_precedence(self) -> None:
        params = {
            "platform": "feishu",
            "action": "im.send_message",
            "payload": {"chat_id": "oc_correct", "text": "right"},
            "chat_id": "oc_wrong",
            "text": "wrong",
        }
        result = normalize_payload(_SEND_SPEC, params)
        assert result["chat_id"] == "oc_correct"
        assert result["text"] == "right"

    def test_meta_keys_never_lifted(self) -> None:
        params = {
            "platform": "feishu",
            "action": "im.send_message",
            "payload": {"chat_id": "oc_1", "text": "hi"},
        }
        result = normalize_payload(_SEND_SPEC, params)
        assert "platform" not in result
        assert "action" not in result

    def test_missing_payload_key_creates_from_top_level(self) -> None:
        params = {
            "platform": "feishu",
            "action": "im.list_messages",
            "chat_id": "oc_1",
        }
        result = normalize_payload(_LIST_SPEC, params)
        assert result == {"chat_id": "oc_1"}

    def test_partial_payload_supplemented_from_top(self) -> None:
        params = {
            "platform": "feishu",
            "action": "im.send_message",
            "payload": {"chat_id": "oc_1"},
            "text": "supplemented",
        }
        result = normalize_payload(_SEND_SPEC, params)
        assert result == {"chat_id": "oc_1", "text": "supplemented"}


class TestValidatePayloadStructured:
    """Tests for enhanced structured ValidationResult."""

    def test_valid_payload_returns_ok(self) -> None:
        result = validate_payload(_SEND_SPEC, {"chat_id": "oc_1", "text": "hi"})
        assert result.ok is True
        assert result.failure_code == ""

    def test_missing_fields_returns_structured_error(self) -> None:
        result = validate_payload(_SEND_SPEC, {"chat_id": "oc_1"})
        assert result.ok is False
        assert result.failure_code == "missing_required_fields"
        assert "text" in result.missing_fields
        assert "payload.text" in result.recovery_hint

    def test_type_mismatch_returns_structured_error(self) -> None:
        spec = ActionSpec(
            name="test.action",
            backend_kind="cli",
            schema={
                "required": ["values"],
                "properties": {"values": {"type": "array"}},
            },
        )
        result = validate_payload(spec, {"values": "not-array"})
        assert result.ok is False
        assert result.failure_code == "type_mismatch"
        assert len(result.type_errors) == 1

    def test_empty_required_field_counts_as_missing(self) -> None:
        result = validate_payload(_SEND_SPEC, {"chat_id": "oc_1", "text": ""})
        assert result.ok is False
        assert "text" in result.missing_fields


# ═══════════════════════════════════════════════════════════════
# Task-scoped side-effect dedup tests
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_side_effect_dedup_blocks_duplicate_send() -> None:
    """Second identical send call returns already_executed without execution."""
    from leapflow.gateway.connectors.protocol import ActionPreview, ActionResult, BackendStatus
    from leapflow.gateway.adapters.feishu import FeishuAdapter
    from leapflow.gateway.server import GatewayServer
    from leapflow.tools.gateway_tool import (
        platform_action_handler,
        reset_platform_action_scope,
        set_gateway_approval_gate,
        set_gateway_server,
    )
    import tempfile
    from pathlib import Path

    class CountingBackend:
        kind = "cli"
        call_count = 0

        async def status(self):
            return BackendStatus(ok=True, backend_kind=self.kind)

        async def authenticate(self, payload):
            return BackendStatus(ok=True, backend_kind=self.kind)

        async def execute(self, spec, payload):
            self.call_count += 1
            return ActionResult(ok=True, resource_id=f"msg_{self.call_count}", data=dict(payload))

        async def preview(self, spec, payload):
            return ActionPreview(ok=True, summary=f"preview {spec.name}")

    backend = CountingBackend()
    server = GatewayServer(Path(tempfile.mkdtemp()))
    server.discover_manifests()
    adapter = FeishuAdapter(backend=backend)
    server._adapters["feishu"] = adapter
    set_gateway_server(server)
    set_gateway_approval_gate(None)
    reset_platform_action_scope()

    try:
        params = {
            "platform": "feishu",
            "action": "im.send_message",
            "payload": {"chat_id": "oc_1", "text": "hello"},
        }

        result1 = await platform_action_handler(params)
        result2 = await platform_action_handler(params)

        assert result1.get("ok") is True
        assert result2.get("ok") is True
        assert result2.get("already_executed") is True
        assert "Do not re-invoke" in result2.get("execution_note", "")
        assert backend.call_count == 1
    finally:
        await server.stop()
        set_gateway_approval_gate(None)
        set_gateway_server(None)
        reset_platform_action_scope()


@pytest.mark.asyncio
async def test_side_effect_dedup_allows_read_actions_to_repeat() -> None:
    """Read actions (effect=read) are not subject to dedup."""
    from leapflow.gateway.connectors.protocol import ActionPreview, ActionResult, BackendStatus
    from leapflow.gateway.adapters.feishu import FeishuAdapter
    from leapflow.gateway.server import GatewayServer
    from leapflow.tools.gateway_tool import (
        platform_action_handler,
        reset_platform_action_scope,
        set_gateway_approval_gate,
        set_gateway_server,
    )
    import tempfile
    from pathlib import Path

    class CountingBackend:
        kind = "cli"
        call_count = 0

        async def status(self):
            return BackendStatus(ok=True, backend_kind=self.kind)

        async def authenticate(self, payload):
            return BackendStatus(ok=True, backend_kind=self.kind)

        async def execute(self, spec, payload):
            self.call_count += 1
            return ActionResult(ok=True, resource_id=f"r_{self.call_count}", data=dict(payload))

        async def preview(self, spec, payload):
            return ActionPreview(ok=True, summary=f"preview {spec.name}")

    backend = CountingBackend()
    server = GatewayServer(Path(tempfile.mkdtemp()))
    server.discover_manifests()
    adapter = FeishuAdapter(backend=backend)
    server._adapters["feishu"] = adapter
    set_gateway_server(server)
    set_gateway_approval_gate(None)
    reset_platform_action_scope()

    try:
        params = {
            "platform": "feishu",
            "action": "im.list_messages",
            "payload": {"chat_id": "oc_1"},
        }

        result1 = await platform_action_handler(params)
        result2 = await platform_action_handler(params)

        assert result1.get("ok") is True
        assert result2.get("ok") is True
        assert result2.get("already_executed") is None
        assert backend.call_count == 2
    finally:
        await server.stop()
        set_gateway_approval_gate(None)
        set_gateway_server(None)
        reset_platform_action_scope()


@pytest.mark.asyncio
async def test_side_effect_dedup_resets_across_turns() -> None:
    """After reset_platform_action_scope, same action can execute again."""
    from leapflow.gateway.connectors.protocol import ActionPreview, ActionResult, BackendStatus
    from leapflow.gateway.adapters.feishu import FeishuAdapter
    from leapflow.gateway.server import GatewayServer
    from leapflow.tools.gateway_tool import (
        platform_action_handler,
        reset_platform_action_scope,
        set_gateway_approval_gate,
        set_gateway_server,
    )
    import tempfile
    from pathlib import Path

    class CountingBackend:
        kind = "cli"
        call_count = 0

        async def status(self):
            return BackendStatus(ok=True, backend_kind=self.kind)

        async def authenticate(self, payload):
            return BackendStatus(ok=True, backend_kind=self.kind)

        async def execute(self, spec, payload):
            self.call_count += 1
            return ActionResult(ok=True, resource_id=f"msg_{self.call_count}", data=dict(payload))

        async def preview(self, spec, payload):
            return ActionPreview(ok=True, summary=f"preview {spec.name}")

    backend = CountingBackend()
    server = GatewayServer(Path(tempfile.mkdtemp()))
    server.discover_manifests()
    adapter = FeishuAdapter(backend=backend)
    server._adapters["feishu"] = adapter
    set_gateway_server(server)
    set_gateway_approval_gate(None)
    reset_platform_action_scope()

    try:
        params = {
            "platform": "feishu",
            "action": "im.send_message",
            "payload": {"chat_id": "oc_1", "text": "hello"},
        }

        await platform_action_handler(params)
        reset_platform_action_scope()
        result2 = await platform_action_handler(params)

        assert result2.get("ok") is True
        assert result2.get("already_executed") is None
        assert backend.call_count == 2
    finally:
        await server.stop()
        set_gateway_approval_gate(None)
        set_gateway_server(None)
        reset_platform_action_scope()
