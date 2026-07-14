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


def test_capability_health_ledger_degrades_platform_for_side_effects() -> None:
    """Hard auth failure on one capability blocks side-effect actions on the whole platform."""
    ledger = CapabilityHealthLedger()
    read_spec = ActionSpec(
        name="im.list_messages",
        backend_kind="cli",
        effect="read",
        capability="im.message.read",
        auth=ActionAuthSpec(scopes={"common": ("im:message:readonly",)}),
    )
    send_spec = ActionSpec(
        name="im.send_message",
        backend_kind="cli",
        effect="send",
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

    blocked_read = ledger.check_feasibility("feishu", read_spec)
    blocked_send = ledger.check_feasibility("feishu", send_spec)

    # Read actions pass through so they can revalidate external permission changes.
    assert blocked_read["ok"] is True
    assert blocked_read["permission_revalidation"] is True
    assert blocked_read["previous_failure_code"] == "missing_scope"
    assert blocked_read["capability"] == "im.message.read"

    # Platform degradation blocks the side-effect action
    assert blocked_send["ok"] is False
    assert blocked_send["failure_code"] == "platform_degraded"
    assert blocked_send["platform_degraded"] is True
    assert "llm_instruction" in blocked_send

    # Unrelated platform is not affected
    other_spec = ActionSpec(name="send", backend_kind="cli", effect="send", capability="im.send")
    assert ledger.check_feasibility("dingtalk", other_spec) == {"ok": True}


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


def test_lark_cli_error_classifier_declares_scopes_from_action_spec() -> None:
    """When no authoritative missing_scopes are available, the classifier must
    fall back to the action's OWN declared auth.scopes contract — never a
    global CLI scope registry or a guessed scope name."""
    spec = ActionSpec(
        name="im.list_chats",
        backend_kind="cli",
        capability="im.chat.read",
        auth=ActionAuthSpec(scopes={"common": ("im:chat:read",)}),
    )

    failure = classify_lark_cli_failure(
        spec,
        "missing scope for this operation",
        {},
        binary="lark-cli",
        profile="work",
        identity="bot",
    )

    assert failure.required_scopes == ("im:chat:read",)
    assert failure.scope_source == "declared"
    assert failure.scope_relation == "all_required"
    assert "im:chat:read" in failure.recovery_hint


def test_lark_cli_error_classifier_typed_error_marks_authoritative_scopes() -> None:
    """missing_scopes from the upstream typed error is authoritative and must
    take priority over the action's declared contract."""
    spec = ActionSpec(
        name="im.list_chats",
        backend_kind="cli",
        capability="im.chat.read",
        auth=ActionAuthSpec(scopes={"common": ("im:chat:read",)}),
    )

    failure = classify_lark_cli_failure(
        spec,
        "",
        {
            "error": {
                "type": "authorization",
                "subtype": "missing_scope",
                "message": "access denied",
                "missing_scopes": ["im:chat:read"],
                "console_url": "https://open.feishu.cn/app/cli_xxx/auth",
            }
        },
        binary="lark-cli",
        profile="work",
        identity="bot",
    )

    assert failure.missing_scopes == ("im:chat:read",)
    assert failure.scope_source == "authoritative"
    assert failure.console_url == "https://open.feishu.cn/app/cli_xxx/auth"


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


# ═══════════════════════════════════════════════════════════════
# Platform degradation tests
# ═══════════════════════════════════════════════════════════════


def test_platform_degradation_blocks_side_effects() -> None:
    """Authorization failure on a read capability degrades the platform for all side-effects."""
    from leapflow.gateway.capability_health import CapabilityHealthLedger

    ledger = CapabilityHealthLedger()
    failure = ActionFailure(
        failure_class="authorization",
        failure_code="access_denied",
        message="Missing scope im:chat:read",
        recoverability="admin_required",
        retryable=False,
        blocks_approval=True,
    )
    ledger.record_failure("feishu", "im.chat.read", failure)

    assert ledger.is_platform_degraded("feishu")
    assert not ledger.is_platform_degraded("dingtalk")

    send_spec = ActionSpec(
        name="im.send_message",
        backend_kind="cli",
        effect="send",
        capability="im.message.send",
        auth=ActionAuthSpec(resource_fields=("chat_id",)),
    )
    result = ledger.check_feasibility("feishu", send_spec)
    assert result["ok"] is False
    assert result["failure_code"] == "platform_degraded"
    assert result["skip_approval"] is True
    assert "llm_instruction" in result


def test_platform_degradation_allows_read_retry() -> None:
    """A read action with a previous auth failure can revalidate live permissions."""
    from leapflow.gateway.capability_health import CapabilityHealthLedger

    ledger = CapabilityHealthLedger()
    failure = ActionFailure(
        failure_class="authorization",
        failure_code="missing_scope",
        message="Scope im:chat:read required",
        recoverability="admin_required",
        retryable=False,
        blocks_approval=True,
    )
    ledger.record_failure("feishu", "im.chat.read", failure)

    read_spec = ActionSpec(
        name="im.list_chats",
        backend_kind="cli",
        effect="read",
        capability="im.chat.read",
    )
    result = ledger.check_feasibility("feishu", read_spec)
    assert result["ok"] is True
    assert result["permission_revalidation"] is True
    assert result["previous_failure_code"] == "missing_scope"


def test_platform_degradation_record_success_restores_matching_capability() -> None:
    """Successful revalidation removes the matching capability failure."""
    from leapflow.gateway.capability_health import CapabilityHealthLedger

    ledger = CapabilityHealthLedger()
    failure = ActionFailure(
        failure_class="authorization",
        failure_code="missing_scope",
        message="Missing scope",
        recoverability="admin_required",
        retryable=False,
        blocks_approval=True,
    )
    ledger.record_failure("feishu", "im.chat.read", failure)
    assert ledger.is_platform_degraded("feishu")

    assert ledger.record_success("feishu", "im.chat.read") is True
    assert not ledger.is_platform_degraded("feishu")
    assert ledger.summary() == []


def test_platform_degradation_clear_restores_access() -> None:
    """Clearing a platform removes degradation status."""
    from leapflow.gateway.capability_health import CapabilityHealthLedger

    ledger = CapabilityHealthLedger()
    failure = ActionFailure(
        failure_class="authorization",
        failure_code="access_denied",
        message="Missing scope",
        recoverability="admin_required",
        retryable=False,
        blocks_approval=True,
    )
    ledger.record_failure("feishu", "im.chat.read", failure)
    assert ledger.is_platform_degraded("feishu")

    ledger.clear("feishu")
    assert not ledger.is_platform_degraded("feishu")

    send_spec = ActionSpec(
        name="im.send_message",
        backend_kind="cli",
        effect="send",
        capability="im.message.send",
    )
    result = ledger.check_feasibility("feishu", send_spec)
    assert result["ok"] is True


def test_transient_failure_does_not_degrade_platform() -> None:
    """Timeout/rate-limit failures do not trigger platform degradation."""
    from leapflow.gateway.capability_health import CapabilityHealthLedger

    ledger = CapabilityHealthLedger()
    failure = ActionFailure(
        failure_class="timeout",
        failure_code="request_timeout",
        message="Timed out",
        recoverability="retryable",
        retryable=True,
        blocks_approval=False,
    )
    ledger.record_failure("feishu", "im.chat.read", failure)
    assert not ledger.is_platform_degraded("feishu")


# ═══════════════════════════════════════════════════════════════
# Resource provenance tests
# ═══════════════════════════════════════════════════════════════


def test_resource_provenance_verified_after_registration() -> None:
    """Registered resource IDs are verified in subsequent checks."""
    from leapflow.gateway.resource_provenance import ProvenanceStatus, ResourceProvenancePool

    pool = ResourceProvenancePool()
    pool.register("feishu", "chat_id", "oc_real_123")
    pool.register("feishu", "chat_id", "oc_real_456")

    assert pool.check("feishu", "chat_id", "oc_real_123").status == ProvenanceStatus.VERIFIED
    assert pool.check("feishu", "chat_id", "oc_real_456").status == ProvenanceStatus.VERIFIED
    assert pool.check("feishu", "chat_id", "oc_fake").status == ProvenanceStatus.UNVERIFIED
    assert pool.check("feishu", "message_id", "msg_1").status == ProvenanceStatus.UNKNOWN


def test_resource_provenance_extract_from_nested_result() -> None:
    """register_from_result extracts resource IDs from nested API results."""
    from leapflow.gateway.resource_provenance import ProvenanceStatus, ResourceProvenancePool

    pool = ResourceProvenancePool()
    result_data = {
        "items": [
            {"chat_id": "oc_a", "name": "Group A"},
            {"chat_id": "oc_b", "name": "Group B"},
        ],
        "page_token": "",
    }
    count = pool.register_from_result("feishu", ["chat_id"], result_data)
    assert count == 2
    assert pool.check("feishu", "chat_id", "oc_a").status == ProvenanceStatus.VERIFIED
    assert pool.check("feishu", "chat_id", "oc_b").status == ProvenanceStatus.VERIFIED


def test_resource_provenance_check_payload() -> None:
    """check_payload returns results for all resource fields in one call."""
    from leapflow.gateway.resource_provenance import ProvenanceStatus, ResourceProvenancePool

    pool = ResourceProvenancePool()
    pool.register("feishu", "chat_id", "oc_valid")

    results = pool.check_payload(
        "feishu", ["chat_id", "message_id"], {"chat_id": "oc_valid", "message_id": "msg_1"}
    )
    assert len(results) == 2
    assert results[0].status == ProvenanceStatus.VERIFIED
    assert results[1].status == ProvenanceStatus.UNKNOWN


@pytest.mark.asyncio
async def test_platform_degradation_blocks_send_after_list_fails() -> None:
    """End-to-end: list_chats auth failure degrades platform, blocking send_message."""
    from leapflow.gateway.adapters.feishu import FeishuAdapter
    from leapflow.gateway.connectors.protocol import ActionPreview, ActionResult, BackendStatus
    from leapflow.gateway.server import GatewayServer
    from leapflow.tools.gateway_tool import (
        platform_action_handler,
        reset_platform_action_scope,
        set_gateway_approval_gate,
        set_gateway_server,
    )
    import tempfile
    from pathlib import Path

    class AuthFailBackend:
        kind = "cli"

        async def status(self):
            return BackendStatus(ok=True, backend_kind=self.kind)

        async def authenticate(self, payload):
            return BackendStatus(ok=True, backend_kind=self.kind)

        async def execute(self, spec, payload):
            return ActionResult(
                ok=False,
                error="access denied",
                failure=ActionFailure(
                    failure_class="authorization",
                    failure_code="access_denied",
                    message="Missing scope im:chat:read",
                    recoverability="admin_required",
                    retryable=False,
                    blocks_approval=True,
                    capability="im.chat.read",
                ),
            )

        async def preview(self, spec, payload):
            return ActionPreview(ok=True, summary=f"preview {spec.name}")

    backend = AuthFailBackend()
    server = GatewayServer(Path(tempfile.mkdtemp()))
    server.discover_manifests()
    adapter = FeishuAdapter(backend=backend)
    server._adapters["feishu"] = adapter
    set_gateway_server(server)
    set_gateway_approval_gate(None)
    reset_platform_action_scope()

    try:
        # First: list_chats fails with authorization error
        list_result = await platform_action_handler({
            "platform": "feishu",
            "action": "im.list_chats",
            "payload": {"page_size": 20},
        })
        assert list_result["ok"] is False

        # Now: send_message should be blocked by platform degradation
        send_result = await platform_action_handler({
            "platform": "feishu",
            "action": "im.send_message",
            "payload": {"chat_id": "oc_hallucinated", "text": "test"},
        })
        assert send_result["ok"] is False
        assert send_result.get("failure_code") == "platform_degraded"
        assert "llm_instruction" in send_result
    finally:
        await server.stop()
        set_gateway_approval_gate(None)
        set_gateway_server(None)
        reset_platform_action_scope()
