"""Feishu/Lark gateway adapter."""
from __future__ import annotations

import json
from typing import Any, Mapping

from leapflow.gateway.adapters.common import (
    AdapterLifecycle,
    HttpRequest,
    HttpResponse,
    JsonHttpClient,
    TinyJsonHttpServer,
    UrlLibJsonHttpClient,
    parse_bind_port,
    parse_json_object,
    stable_message_id,
)
from leapflow.gateway.protocol import InboundMessage, OutboundContent, SendResult, SendTarget


class FeishuAdapter(AdapterLifecycle):
    """Feishu adapter with token refresh, outbound text, and event normalisation."""

    platform_id = "feishu"
    supports_async_delivery = True
    max_message_length = 8000

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        connection_mode: str = "webhook",
        profile: str = "default",
        api_base: str = "https://open.feishu.cn",
        host: str = "127.0.0.1",
        port: str | int = "9091",
        path: str = "/feishu/events",
        http_client: JsonHttpClient | None = None,
        **_: Any,
    ) -> None:
        super().__init__(profile=profile)
        self._app_id = app_id
        self._app_secret = app_secret
        self._connection_mode = connection_mode or "webhook"
        self._api_base = api_base.rstrip("/")
        self._host = host or "127.0.0.1"
        self._port = parse_bind_port(port, 9091)
        self._path = path if str(path).startswith("/") else f"/{path}"
        self._http = http_client or UrlLibJsonHttpClient()
        self._tenant_access_token = ""
        self._server: TinyJsonHttpServer | None = None

    @property
    def local_url(self) -> str:
        if self._server is None:
            return f"http://{self._host}:{self._port}{self._path}"
        return f"{self._server.url_base}{self._path}"

    async def connect(self, *, is_reconnect: bool = False) -> None:
        self._tenant_access_token = await self._fetch_tenant_access_token()
        if self._connection_mode != "webhook":
            raise NotImplementedError(
                "Feishu built-in adapter currently supports connection_mode='webhook'. "
                "WebSocket SDK mode is planned as a later adapter enhancement.",
            )
        if self._server is None:
            self._server = TinyJsonHttpServer(self._host, self._port, self._handle_request)
            await self._server.start()
        await super().connect(is_reconnect=is_reconnect)

    async def disconnect(self) -> None:
        if self._server is not None:
            await self._server.stop()
            self._server = None
        await super().disconnect()

    async def send(self, target: SendTarget, content: OutboundContent) -> SendResult:
        token = self._tenant_access_token or await self._fetch_tenant_access_token()
        url = f"{self._api_base}/open-apis/im/v1/messages?receive_id_type=chat_id"
        payload = {
            "receive_id": target.chat_id,
            "msg_type": "text",
            "content": json.dumps({"text": content.text[:self.max_message_length]}),
        }
        status, data = await self._http.request_json(
            "POST",
            url,
            json_body=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout_s=10,
        )
        if status >= 400 or data.get("code", 0) != 0:
            return SendResult(ok=False, error=str(data.get("msg") or data))
        message_id = data.get("data", {}).get("message_id", "")
        return SendResult(ok=True, message_id=str(message_id))

    async def handle_event(self, payload: Mapping[str, Any]) -> InboundMessage | None:
        """Normalise a Feishu event callback payload and emit it."""
        message = self.message_from_event(payload)
        if message is None:
            return None
        await self._emit(message)
        return message

    async def _handle_request(self, request: HttpRequest) -> HttpResponse:
        if request.path.split("?", 1)[0] != self._path:
            return HttpResponse(404, {"ok": False, "error": "not found"})
        if request.method != "POST":
            return HttpResponse(405, {"ok": False, "error": "method not allowed"})
        payload = parse_json_object(request.body)
        if payload.get("type") == "url_verification" and payload.get("challenge"):
            return HttpResponse(200, {"challenge": payload["challenge"]})
        message = await self.handle_event(payload)
        if message is None:
            return HttpResponse(202, {"ok": True, "ignored": True})
        return HttpResponse(202, {"ok": True, "message_id": message.message_id})

    def message_from_event(self, payload: Mapping[str, Any]) -> InboundMessage | None:
        event = payload.get("event") if isinstance(payload.get("event"), dict) else payload
        if not isinstance(event, Mapping):
            return None
        raw_message = event.get("message") if isinstance(event.get("message"), dict) else {}
        raw_sender = event.get("sender") if isinstance(event.get("sender"), dict) else {}
        content = raw_message.get("content", "")
        text = self._parse_text_content(content)
        if not text:
            return None
        chat_id = str(raw_message.get("chat_id") or event.get("chat_id") or "feishu")
        chat_type = str(raw_message.get("chat_type") or "group")
        user_id = self._sender_id(raw_sender)
        user_name = str(raw_sender.get("sender_type") or "")
        message_id = str(raw_message.get("message_id") or stable_message_id("feishu"))
        return InboundMessage(
            source=self._source(
                chat_id=chat_id,
                chat_type=chat_type,
                user_id=user_id,
                user_name=user_name,
            ),
            text=text,
            message_id=message_id,
            metadata={"connection_mode": self._connection_mode},
        )

    async def _fetch_tenant_access_token(self) -> str:
        status, data = await self._http.request_json(
            "POST",
            f"{self._api_base}/open-apis/auth/v3/tenant_access_token/internal",
            json_body={"app_id": self._app_id, "app_secret": self._app_secret},
            timeout_s=10,
        )
        token = data.get("tenant_access_token", "")
        if status >= 400 or data.get("code", 0) != 0 or not token:
            raise RuntimeError(str(data.get("msg") or "Feishu token request failed"))
        return str(token)

    @staticmethod
    def _sender_id(sender: Mapping[str, Any]) -> str:
        raw_id = sender.get("sender_id") if isinstance(sender.get("sender_id"), dict) else {}
        if isinstance(raw_id, dict):
            return str(raw_id.get("open_id") or raw_id.get("union_id") or raw_id.get("user_id") or "")
        return ""

    @staticmethod
    def _parse_text_content(content: Any) -> str:
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                return content
            if isinstance(parsed, dict):
                return str(parsed.get("text") or parsed.get("content") or "")
            return str(parsed)
        if isinstance(content, dict):
            return str(content.get("text") or content.get("content") or "")
        return str(content or "")
