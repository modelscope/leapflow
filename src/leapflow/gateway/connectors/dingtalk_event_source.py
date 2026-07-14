"""DingTalk webhook event source.

Implements ``BackendEventSource`` by running a lightweight HTTP server
that receives DingTalk callback payloads and yields them as
``BackendEvent`` objects.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Mapping

from leapflow.gateway.adapters.common import (
    HttpRequest,
    HttpResponse,
    TinyJsonHttpServer,
    parse_bind_port,
    parse_json_object,
)
from leapflow.gateway.connectors.protocol import BackendEvent, EventSourceStatus

logger = logging.getLogger(__name__)


class DingTalkWebhookEventSource:
    """BackendEventSource wrapping a local HTTP webhook receiver for DingTalk.

    Yields ``BackendEvent`` objects for each incoming DingTalk callback.
    """

    platform_id = "dingtalk"
    backend_kind = "webhook"

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int | str = 9092,
        path: str = "/dingtalk/events",
        robot_code: str = "",
    ) -> None:
        self._host = host or "127.0.0.1"
        self._port = parse_bind_port(port, 9092)
        self._path = path if str(path).startswith("/") else f"/{path}"
        self._robot_code = robot_code
        self._server: TinyJsonHttpServer | None = None
        self._queue: asyncio.Queue[BackendEvent] = asyncio.Queue()
        self._running = False

    async def start(self, *, checkpoint: str = "") -> EventSourceStatus:
        if self._running:
            return await self.status()
        self._server = TinyJsonHttpServer(self._host, self._port, self._handle_request)
        await self._server.start()
        self._running = True
        url = f"{self._server.url_base}{self._path}"
        logger.info("DingTalk webhook event source listening at %s", url)
        return EventSourceStatus(
            ok=True,
            backend_kind=self.backend_kind,
            detail=f"Webhook listening at {url}",
            checkpoint=checkpoint,
            metadata={"url": url},
        )

    async def stop(self) -> EventSourceStatus:
        self._running = False
        if self._server is not None:
            await self._server.stop()
            self._server = None
        return EventSourceStatus(
            ok=True,
            backend_kind=self.backend_kind,
            detail="Webhook stopped",
        )

    async def events(self) -> AsyncIterator[BackendEvent]:
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=5.0)
                yield event
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return

    async def status(self) -> EventSourceStatus:
        return EventSourceStatus(
            ok=self._running and self._server is not None,
            backend_kind=self.backend_kind,
            detail="running" if self._running else "stopped",
            metadata={"queue_size": self._queue.qsize()},
        )

    async def _handle_request(self, request: HttpRequest) -> HttpResponse:
        if request.path.split("?", 1)[0] != self._path:
            return HttpResponse(404, {"ok": False, "error": "not found"})
        if request.method != "POST":
            return HttpResponse(405, {"ok": False, "error": "method not allowed"})

        payload = parse_json_object(request.body)
        event = self._to_backend_event(payload)
        if event is not None:
            await self._queue.put(event)
            return HttpResponse(202, {"ok": True, "event_id": event.event_id})
        return HttpResponse(202, {"ok": True, "ignored": True})

    def _to_backend_event(self, payload: Mapping[str, Any]) -> BackendEvent | None:
        msg_id = str(payload.get("msgId") or payload.get("message_id") or "")
        event_type_raw = str(
            payload.get("msgtype") or payload.get("chatbotCorpId", "message") or "message"
        )
        return BackendEvent(
            event_id=msg_id,
            event_type=event_type_raw,
            platform_id=self.platform_id,
            payload=dict(payload),
        )
