"""Telegram polling event source.

Implements ``BackendEventSource`` by long-polling the Telegram Bot API
for updates and yielding them as ``BackendEvent`` objects.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Mapping

from leapflow.gateway.connectors.protocol import BackendEvent, EventSourceStatus

logger = logging.getLogger(__name__)


class TelegramPollingEventSource:
    """BackendEventSource wrapping Telegram getUpdates long-polling.

    Constructed with the bot token and API base, runs its own poll loop
    and yields BackendEvent per update.
    """

    platform_id = "telegram"
    backend_kind = "http_poll"

    def __init__(
        self,
        *,
        bot_token: str,
        api_base: str = "https://api.telegram.org",
        poll_interval_s: float = 1.0,
        http_client: Any = None,
    ) -> None:
        self._bot_token = bot_token
        self._api_base = api_base.rstrip("/")
        self._poll_interval_s = poll_interval_s
        from leapflow.gateway.adapters.common import UrlLibJsonHttpClient
        self._http = http_client or UrlLibJsonHttpClient()
        self._offset = 0
        self._running = False
        self._bot_id = ""

    async def start(self, *, checkpoint: str = "") -> EventSourceStatus:
        if self._running:
            return await self.status()
        if checkpoint:
            try:
                self._offset = int(checkpoint)
            except ValueError:
                pass
        self._running = True
        await self._fetch_bot_id()
        return EventSourceStatus(
            ok=True,
            backend_kind=self.backend_kind,
            detail="Telegram polling started",
            checkpoint=str(self._offset),
            metadata={"bot_id": self._bot_id},
        )

    async def stop(self) -> EventSourceStatus:
        self._running = False
        return EventSourceStatus(
            ok=True,
            backend_kind=self.backend_kind,
            detail="Telegram polling stopped",
            checkpoint=str(self._offset),
        )

    async def events(self) -> AsyncIterator[BackendEvent]:
        while self._running:
            try:
                status_code, data = await self._api(
                    "getUpdates",
                    {"offset": self._offset, "timeout": 25},
                    timeout_s=30,
                )
                if status_code < 400 and data.get("ok"):
                    for update in data.get("result", []) or []:
                        if not isinstance(update, dict):
                            continue
                        update_id = int(update.get("update_id", self._offset))
                        self._offset = max(self._offset, update_id + 1)
                        event = self._to_backend_event(update)
                        if event is not None:
                            yield event
            except asyncio.CancelledError:
                return
            except Exception:
                logger.debug("Telegram poll error", exc_info=True)
                await asyncio.sleep(self._poll_interval_s)
            await asyncio.sleep(self._poll_interval_s)

    async def status(self) -> EventSourceStatus:
        return EventSourceStatus(
            ok=self._running,
            backend_kind=self.backend_kind,
            detail="running" if self._running else "stopped",
            checkpoint=str(self._offset),
            metadata={"bot_id": self._bot_id},
        )

    @property
    def bot_id(self) -> str:
        return self._bot_id

    def _to_backend_event(self, update: Mapping[str, Any]) -> BackendEvent | None:
        update_id = str(update.get("update_id", ""))
        event_type = "message"
        if "callback_query" in update:
            event_type = "callback_query"
        elif "edited_message" in update:
            event_type = "edited_message"
        elif "my_chat_member" in update or "chat_member" in update:
            event_type = "chat_member_update"
        elif "message_reaction" in update:
            event_type = "message_reaction"
        return BackendEvent(
            event_id=update_id,
            event_type=event_type,
            platform_id=self.platform_id,
            payload=dict(update),
        )

    async def _fetch_bot_id(self) -> None:
        """Resolve bot user_id via getMe for self-message filtering."""
        try:
            status_code, data = await self._api("getMe", {}, timeout_s=10)
            if status_code < 400 and data.get("ok"):
                result = data.get("result", {})
                self._bot_id = str(result.get("id", ""))
                bot_name = str(result.get("username", ""))
                if self._bot_id:
                    logger.info(
                        "Telegram bot identity: id=%s username=%s",
                        self._bot_id, bot_name,
                    )
        except Exception:
            logger.debug("Failed to fetch Telegram bot identity", exc_info=True)

    async def _api(
        self,
        method: str,
        payload: Mapping[str, Any],
        *,
        timeout_s: float = 10.0,
    ) -> tuple[int, dict[str, Any]]:
        url = f"{self._api_base}/bot{self._bot_token}/{method}"
        return await self._http.request_json(
            "POST", url, json_body=payload, timeout_s=timeout_s,
        )
