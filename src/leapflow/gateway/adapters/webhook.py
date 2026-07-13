"""Generic HTTP webhook gateway adapter."""
from __future__ import annotations

import hashlib
import hmac
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


class WebhookAdapter(AdapterLifecycle):
    """Receive generic JSON webhook events and normalise them to gateway messages."""

    platform_id = "webhook"
    supports_async_delivery = False
    max_message_length = 0

    def __init__(
        self,
        webhook_secret: str = "",
        host: str = "127.0.0.1",
        port: str | int = "9090",
        path: str = "/webhook",
        profile: str = "default",
        **_: Any,
    ) -> None:
        super().__init__(profile=profile)
        self._secret = webhook_secret or ""
        self._host = host or "127.0.0.1"
        self._port = parse_bind_port(port, 9090)
        self._path = path if str(path).startswith("/") else f"/{path}"
        self._server: TinyJsonHttpServer | None = None

    @property
    def local_url(self) -> str:
        if self._server is None:
            return f"http://{self._host}:{self._port}{self._path}"
        return f"{self._server.url_base}{self._path}"

    async def connect(self, *, is_reconnect: bool = False) -> None:
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
        return SendResult(ok=False, error="webhook receiver does not support outbound delivery")

    async def _handle_request(self, request: HttpRequest) -> HttpResponse:
        if request.path.split("?", 1)[0] != self._path:
            return HttpResponse(404, {"ok": False, "error": "not found"})
        if request.method != "POST":
            return HttpResponse(405, {"ok": False, "error": "method not allowed"})
        if self._secret and not self._valid_signature(request.headers, request.body):
            return HttpResponse(401, {"ok": False, "error": "invalid signature"})
        payload = parse_json_object(request.body)
        message = self.message_from_payload(payload)
        await self._emit(message)
        return HttpResponse(202, {"ok": True, "message_id": message.message_id})

    def _valid_signature(self, headers: Mapping[str, str], body: bytes) -> bool:
        expected = hmac.new(self._secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        candidates = [
            headers.get("x-leapflow-signature", ""),
            headers.get("x-hub-signature-256", ""),
            headers.get("x-signature", ""),
        ]
        for candidate in candidates:
            if candidate.startswith("sha256="):
                candidate = candidate[len("sha256="):]
            if candidate and hmac.compare_digest(candidate, expected):
                return True
        return False

    def message_from_payload(self, payload: Mapping[str, Any]) -> InboundMessage:
        source_raw = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        source = source_raw if isinstance(source_raw, dict) else {}
        text = str(
            payload.get("text")
            or payload.get("message")
            or payload.get("content")
            or payload.get("prompt")
            or ""
        )
        chat_id = str(payload.get("chat_id") or source.get("chat_id") or "webhook")
        user_id = str(payload.get("user_id") or source.get("user_id") or "")
        user_name = str(payload.get("user_name") or source.get("user_name") or "")
        chat_type = str(payload.get("chat_type") or source.get("chat_type") or "dm")
        thread_id = str(payload.get("thread_id") or source.get("thread_id") or "")
        message_id = str(payload.get("message_id") or stable_message_id("webhook"))
        return InboundMessage(
            source=self._source(
                chat_id=chat_id,
                chat_type=chat_type,
                user_id=user_id,
                user_name=user_name,
                thread_id=thread_id,
            ),
            text=text,
            message_id=message_id,
            metadata={"payload_keys": tuple(sorted(str(key) for key in payload.keys()))},
        )
