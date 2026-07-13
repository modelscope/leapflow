from __future__ import annotations

import pytest

from leapflow.gateway.adapters.feishu import FeishuAdapter
from leapflow.gateway.connectors.action_registry import ActionRegistry
from leapflow.gateway.connectors.protocol import ActionPreview, ActionResult, BackendStatus


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
