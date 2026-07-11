"""Common helpers for built-in gateway adapters.

The helpers here intentionally stay small: they provide shared lifecycle,
JSON-over-HTTP, and tiny local HTTP server primitives without becoming a
Hermes-style base class that owns platform behaviour.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Protocol
from urllib import error, request

from leapflow.gateway.mixin import PlatformAdapterMixin
from leapflow.gateway.protocol import InboundMessage, MessageHandler, MessageSource

logger = logging.getLogger(__name__)

JsonBody = Dict[str, Any]
HttpHeaders = Dict[str, str]


class JsonHttpClient(Protocol):
    """Minimal async JSON HTTP client contract used by adapters."""

    async def request_json(
        self,
        method: str,
        url: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_s: float = 10.0,
    ) -> tuple[int, JsonBody]:
        """Send a JSON request and return ``(status_code, parsed_json)``."""
        ...


class UrlLibJsonHttpClient:
    """Small stdlib-backed JSON HTTP client with timeout support."""

    async def request_json(
        self,
        method: str,
        url: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_s: float = 10.0,
    ) -> tuple[int, JsonBody]:
        return await asyncio.to_thread(
            self._request_json_sync,
            method,
            url,
            dict(json_body or {}),
            dict(headers or {}),
            timeout_s,
        )

    @staticmethod
    def _request_json_sync(
        method: str,
        url: str,
        json_body: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_s: float,
    ) -> tuple[int, JsonBody]:
        payload = json.dumps(json_body).encode("utf-8") if json_body else None
        request_headers = {"Accept": "application/json", **headers}
        if payload is not None:
            request_headers.setdefault("Content-Type", "application/json")
        req = request.Request(
            url,
            data=payload,
            headers=request_headers,
            method=method.upper(),
        )
        try:
            with request.urlopen(req, timeout=timeout_s) as resp:
                status = int(getattr(resp, "status", 200))
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            status = int(exc.code)
            raw = exc.read().decode("utf-8", errors="replace")
        data = parse_json_object(raw)
        return status, data


def parse_json_object(raw: str | bytes) -> JsonBody:
    """Parse a JSON object; non-object payloads are wrapped as text."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"text": raw}
    return value if isinstance(value, dict) else {"value": value}


def parse_bind_port(value: str | int | None, default: int) -> int:
    """Parse a bind port while preserving explicit ``0`` for ephemeral ports."""
    if value is None or value == "":
        return default
    return int(value)


def stable_message_id(prefix: str) -> str:
    """Return a compact unique message ID for synthetic gateway events."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def bool_option(value: Any, *, default: bool = True) -> bool:
    """Parse common boolean option representations."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def chunk_text(text: str, limit: int) -> list[str]:
    """Split text into non-empty chunks without truncating content."""
    if limit <= 0 or len(text) <= limit:
        return [text]
    return [text[index:index + limit] for index in range(0, len(text), limit)]


class AdapterLifecycle(PlatformAdapterMixin):
    """Tiny lifecycle helper shared by built-in adapters."""

    platform_id = ""
    supports_async_delivery = True
    splits_long_messages = False
    max_message_length = 4000

    def __init__(self, *, profile: str = "default") -> None:
        self._profile = profile or "default"
        self._on_message: Optional[MessageHandler] = None
        self._connected = False

    @property
    def on_message(self) -> Optional[MessageHandler]:
        return self._on_message

    @on_message.setter
    def on_message(self, handler: MessageHandler) -> None:
        self._on_message = handler

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self, *, is_reconnect: bool = False) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    def _source(
        self,
        *,
        chat_id: str,
        chat_type: str = "dm",
        user_id: str = "",
        user_name: str = "",
        thread_id: str = "",
        scope_id: str = "",
    ) -> MessageSource:
        return MessageSource(
            platform=self.platform_id,
            chat_id=str(chat_id or self.platform_id),
            chat_type=chat_type or "dm",
            user_id=str(user_id or ""),
            user_name=str(user_name or ""),
            thread_id=str(thread_id or ""),
            scope_id=str(scope_id or ""),
            profile=self._profile,
        )

    async def _emit(self, message: InboundMessage) -> None:
        if self._on_message is None:
            logger.warning("No gateway message handler set for %s", self.platform_id)
            return
        result = self._on_message(message)
        if asyncio.iscoroutine(result):
            await result


@dataclass(frozen=True)
class HttpRequest:
    """Parsed HTTP request passed to local gateway endpoints."""

    method: str
    path: str
    headers: Mapping[str, str]
    body: bytes = b""


@dataclass(frozen=True)
class HttpResponse:
    """Minimal HTTP response returned by local gateway endpoints."""

    status: int
    body: Mapping[str, Any] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)


HttpHandler = Callable[[HttpRequest], Awaitable[HttpResponse]]


class TinyJsonHttpServer:
    """Small asyncio JSON HTTP server for local gateway adapters."""

    def __init__(self, host: str, port: int, handler: HttpHandler) -> None:
        self._host = host
        self._port = port
        self._handler = handler
        self._server: asyncio.AbstractServer | None = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        if self._server is None:
            return self._port
        sockets = self._server.sockets or []
        if not sockets:
            return self._port
        return int(sockets[0].getsockname()[1])

    @property
    def url_base(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_client,
            host=self._host,
            port=self._port,
        )

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            request_obj = await self._read_request(reader)
            response = await self._handler(request_obj)
        except Exception as exc:
            logger.debug("gateway.http.error", exc_info=True)
            response = HttpResponse(500, {"ok": False, "error": type(exc).__name__})
        self._write_response(writer, response)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    @staticmethod
    async def _read_request(reader: asyncio.StreamReader) -> HttpRequest:
        header_bytes = await reader.readuntil(b"\r\n\r\n")
        header_text = header_bytes.decode("iso-8859-1")
        lines = header_text.split("\r\n")
        method, path, _version = lines[0].split(" ", 2)
        headers: Dict[str, str] = {}
        for line in lines[1:]:
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
        content_length = int(headers.get("content-length", "0") or "0")
        body = await reader.readexactly(content_length) if content_length else b""
        return HttpRequest(method=method.upper(), path=path, headers=headers, body=body)

    @staticmethod
    def _write_response(writer: asyncio.StreamWriter, response: HttpResponse) -> None:
        reason = {
            200: "OK",
            202: "Accepted",
            400: "Bad Request",
            401: "Unauthorized",
            404: "Not Found",
            405: "Method Not Allowed",
            500: "Internal Server Error",
        }.get(response.status, "OK")
        body = json.dumps(dict(response.body), ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(body)),
            "Connection": "close",
            **dict(response.headers),
        }
        header_lines = [f"HTTP/1.1 {response.status} {reason}"]
        header_lines.extend(f"{key}: {value}" for key, value in headers.items())
        writer.write("\r\n".join(header_lines).encode("ascii") + b"\r\n\r\n" + body)


async def post_json_for_test(url: str, payload: Mapping[str, Any]) -> tuple[int, JsonBody]:
    """Helper used by tests and smoke checks to POST JSON to local adapters."""
    client = UrlLibJsonHttpClient()
    return await client.request_json("POST", url, json_body=payload, timeout_s=5)
