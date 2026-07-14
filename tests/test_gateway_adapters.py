from __future__ import annotations

from typing import Any, Mapping

import pytest

from leapflow.gateway.adapters.api_server import APIServerAdapter
from leapflow.gateway.adapters.common import JsonBody, UrlLibJsonHttpClient, post_json_for_test
from leapflow.gateway.adapters.dingtalk import DingTalkAdapter
from leapflow.gateway.adapters.feishu import FeishuAdapter
from leapflow.gateway.adapters.telegram import TelegramAdapter
from leapflow.gateway.adapters.webhook import WebhookAdapter
from leapflow.gateway.connectors.protocol import ActionResult, BackendStatus
from leapflow.gateway.manifest import ManifestLoader
from leapflow.gateway.protocol import InboundMessage, OutboundContent, SendTarget
from leapflow.gateway.server import GatewayServer


class FakeJsonHttpClient:
    def __init__(self, responses: Mapping[str, tuple[int, JsonBody]]) -> None:
        self.responses = dict(responses)
        self.requests: list[dict[str, Any]] = []

    async def request_json(
        self,
        method: str,
        url: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_s: float = 10.0,
    ) -> tuple[int, JsonBody]:
        self.requests.append({
            "method": method,
            "url": url,
            "json_body": dict(json_body or {}),
            "headers": dict(headers or {}),
            "timeout_s": timeout_s,
        })
        for marker, response in self.responses.items():
            if marker in url:
                return response
        return 200, {"ok": True}


def test_builtin_manifest_adapter_modules_are_importable() -> None:
    manifests = ManifestLoader().discover()
    credentials = {
        "api_server": {"api_key": "0123456789abcdef"},
        "webhook": {"webhook_secret": ""},
        "telegram": {"bot_token": "token"},
        "feishu": {},
        "dingtalk": {"app_key": "key", "app_secret": "secret"},
    }

    for platform_id in {"api_server", "webhook", "telegram", "feishu", "dingtalk"}:
        manifest = manifests[platform_id]
        adapter = GatewayServer._instantiate_adapter(
            manifest,
            credentials[platform_id],
            {"auto_poll": False, "port": 0},
        )
        assert adapter.platform_id == platform_id


@pytest.mark.asyncio
async def test_webhook_adapter_http_post_emits_message() -> None:
    received: list[InboundMessage] = []
    adapter = WebhookAdapter(port=0)
    adapter.on_message = received.append

    await adapter.connect()
    try:
        status, data = await post_json_for_test(adapter.local_url, {
            "text": "hello webhook",
            "chat_id": "chat-1",
            "user_id": "user-1",
            "chat_type": "group",
        })
    finally:
        await adapter.disconnect()

    assert status == 202
    assert data["ok"] is True
    assert received[0].text == "hello webhook"
    assert received[0].source.platform == "webhook"
    assert received[0].source.chat_id == "chat-1"
    assert received[0].source.user_id == "user-1"


@pytest.mark.asyncio
async def test_api_server_adapter_accepts_openai_chat_completion() -> None:
    received: list[InboundMessage] = []
    adapter = APIServerAdapter(api_key="0123456789abcdef", port=0)
    adapter.on_message = received.append
    client = UrlLibJsonHttpClient()

    await adapter.connect()
    try:
        status, data = await client.request_json(
            "POST",
            f"{adapter.local_url}/v1/chat/completions",
            json_body={
                "model": "leapflow-test",
                "user": "api-user",
                "messages": [
                    {"role": "system", "content": "ignore"},
                    {"role": "user", "content": "hello api"},
                ],
            },
            headers={"Authorization": "Bearer 0123456789abcdef"},
        )
    finally:
        await adapter.disconnect()

    assert status == 200
    assert data["object"] == "chat.completion"
    assert received[0].text == "hello api"
    assert received[0].source.platform == "api_server"
    assert received[0].source.chat_type == "api"


@pytest.mark.asyncio
async def test_telegram_adapter_send_and_update_normalization() -> None:
    fake_http = FakeJsonHttpClient({"sendMessage": (200, {"ok": True, "result": {"message_id": 42}})})
    received: list[InboundMessage] = []
    adapter = TelegramAdapter(bot_token="token", auto_poll=False, http_client=fake_http)
    adapter.on_message = received.append

    await adapter.connect()
    result = await adapter.send(
        SendTarget(platform="telegram", chat_id="100", reply_to_id="7"),
        OutboundContent(text="x" * 5000),
    )
    await adapter.handle_update({
        "update_id": 7,
        "message": {
            "message_id": 8,
            "text": "incoming",
            "chat": {"id": 100, "type": "private"},
            "from": {"id": 200, "username": "alice"},
        },
    })
    await adapter.disconnect()

    assert result.ok is True
    assert result.message_id == "42,42"
    assert len(fake_http.requests) == 2
    assert len(fake_http.requests[0]["json_body"]["text"]) == adapter.max_message_length
    assert len(fake_http.requests[1]["json_body"]["text"]) == 5000 - adapter.max_message_length
    assert fake_http.requests[0]["json_body"]["reply_to_message_id"] == "7"
    assert "reply_to_message_id" not in fake_http.requests[1]["json_body"]
    assert received[0].source.chat_type == "dm"
    assert received[0].text == "incoming"


class FakeExecutionBackend:
    kind = "cli"

    def __init__(self) -> None:
        self.executed: list[tuple[str, Mapping[str, Any]]] = []

    async def status(self) -> BackendStatus:
        return BackendStatus(ok=True, backend_kind=self.kind)

    async def authenticate(self, payload: Mapping[str, Any]) -> BackendStatus:
        return BackendStatus(ok=True, backend_kind=self.kind, metadata=dict(payload))

    async def execute(self, spec, payload: Mapping[str, Any]) -> ActionResult:
        self.executed.append((spec.name, payload))
        return ActionResult(ok=True, resource_id="om_1", data={"message_id": "om_1"})


@pytest.mark.asyncio
async def test_feishu_adapter_uses_cli_backend_for_send() -> None:
    backend = FakeExecutionBackend()
    adapter = FeishuAdapter(profile="bot-reader", backend=backend)

    await adapter.connect()
    result = await adapter.send(
        SendTarget(platform="feishu", chat_id="oc_1"),
        OutboundContent(text="hello feishu"),
    )
    await adapter.disconnect()

    assert result.ok is True
    assert result.message_id == "om_1"
    assert backend.executed == [
        ("im.send_message", {"chat_id": "oc_1", "thread_id": "", "text": "hello feishu"}),
    ]


@pytest.mark.asyncio
async def test_dingtalk_adapter_connect_send_and_event_normalization() -> None:
    fake_http = FakeJsonHttpClient({
        "/v1.0/oauth2/accessToken": (200, {"errcode": 0, "accessToken": "access-token"}),
        "/topapi/robot/send": (200, {"errcode": 0, "task_id": "task-1"}),
    })
    adapter = DingTalkAdapter(
        app_key="key",
        app_secret="secret",
        robot_code="robot",
        port=0,
        http_client=fake_http,
    )

    await adapter.connect()
    source = adapter.event_source()
    assert source is not None

    try:
        await source.start()

        result = await adapter.send(
            SendTarget(platform="dingtalk", chat_id="cid-1"),
            OutboundContent(text="hello dingtalk"),
        )
        status, data = await post_json_for_test(source._server.url_base + "/dingtalk/events", {
            "conversationId": "cid-1",
            "conversationType": "2",
            "senderStaffId": "staff-1",
            "senderNick": "Bob",
            "msgId": "msg-1",
            "text": {"content": "incoming dingtalk"},
        })
    finally:
        await source.stop()
        await adapter.disconnect()

    assert result.ok is True
    assert status == 202
    assert data["ok"] is True
    assert result.message_id == "task-1"
    assert fake_http.requests[1]["json_body"]["robotCode"] == "robot"
