"""OpenAI-compatible API server gateway adapter."""
from __future__ import annotations

from typing import Any, Mapping

from leapflow.gateway.adapters.common import (
    AdapterLifecycle,
    HttpRequest,
    HttpResponse,
    TinyJsonHttpServer,
    parse_bind_port,
    parse_json_object,
    stable_message_id,
)
from leapflow.gateway.protocol import InboundMessage, OutboundContent, SendResult, SendTarget


class APIServerAdapter(AdapterLifecycle):
    """Expose a local OpenAI-compatible chat completions ingress endpoint."""

    platform_id = "api_server"
    supports_async_delivery = False
    max_message_length = 0

    def __init__(
        self,
        api_key: str,
        host: str = "127.0.0.1",
        port: str | int = "8080",
        profile: str = "default",
        **_: Any,
    ) -> None:
        super().__init__(profile=profile)
        self._api_key = api_key
        self._host = host or "127.0.0.1"
        self._port = parse_bind_port(port, 8080)
        self._server: TinyJsonHttpServer | None = None

    @property
    def local_url(self) -> str:
        if self._server is None:
            return f"http://{self._host}:{self._port}"
        return self._server.url_base

    async def connect(self, *, is_reconnect: bool = False) -> None:
        if len(self._api_key) < 16:
            raise ValueError("API key must be at least 16 characters")
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
        return SendResult(ok=False, error="api_server ingress does not support outbound delivery")

    async def _handle_request(self, request: HttpRequest) -> HttpResponse:
        path = request.path.split("?", 1)[0]
        if path == "/health":
            return HttpResponse(200, {"ok": True, "platform": self.platform_id})
        if path != "/v1/chat/completions":
            return HttpResponse(404, {"ok": False, "error": "not found"})
        if request.method != "POST":
            return HttpResponse(405, {"ok": False, "error": "method not allowed"})
        if not self._authorized(request.headers):
            return HttpResponse(401, {"error": {"message": "unauthorized", "type": "auth_error"}})
        payload = parse_json_object(request.body)
        message = self.message_from_payload(payload)
        await self._emit(message)
        return HttpResponse(200, self._accepted_response(payload, message.message_id))

    def _authorized(self, headers: Mapping[str, str]) -> bool:
        auth = headers.get("authorization", "")
        if auth == f"Bearer {self._api_key}":
            return True
        return headers.get("x-api-key", "") == self._api_key

    def message_from_payload(self, payload: Mapping[str, Any]) -> InboundMessage:
        messages = payload.get("messages")
        text = ""
        if isinstance(messages, list):
            for item in reversed(messages):
                if not isinstance(item, dict):
                    continue
                if item.get("role") == "user":
                    content = item.get("content", "")
                    text = self._content_to_text(content)
                    break
        if not text:
            text = str(payload.get("prompt") or payload.get("input") or "")
        chat_id = str(payload.get("user") or payload.get("session_id") or "api")
        message_id = stable_message_id("api")
        return InboundMessage(
            source=self._source(chat_id=chat_id, chat_type="api", user_id=chat_id),
            text=text,
            message_id=message_id,
            metadata={"model": str(payload.get("model", ""))},
        )

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    value = block.get("text") or block.get("content")
                    if value:
                        parts.append(str(value))
            return "\n".join(parts)
        return str(content or "")

    @staticmethod
    def _accepted_response(payload: Mapping[str, Any], message_id: str) -> dict[str, Any]:
        return {
            "id": message_id,
            "object": "chat.completion",
            "created": 0,
            "model": str(payload.get("model") or "leapflow-gateway"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Request accepted by LeapFlow gateway.",
                    },
                    "finish_reason": "stop",
                },
            ],
        }
