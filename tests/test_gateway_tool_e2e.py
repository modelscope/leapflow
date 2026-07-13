from __future__ import annotations

import pytest

from leapflow.gateway.protocol import OutboundContent, SendResult, SendTarget
from leapflow.gateway.server import GatewayServer
from leapflow.tools.gateway_tool import (
    gateway_connect_handler,
    gateway_send_handler,
    set_gateway_approval_gate,
    set_gateway_server,
)


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

    async def connect(self, *, is_reconnect: bool = False) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def send(self, target: SendTarget, content: OutboundContent) -> SendResult:
        self.sent.append((target, content))
        return SendResult(ok=True, message_id="fake-1")


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

    assert result == {"ok": True, "message_id": "fake-1"}
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
