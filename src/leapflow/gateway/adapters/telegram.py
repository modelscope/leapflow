"""Telegram Bot gateway adapter."""
from __future__ import annotations

import asyncio
from typing import Any, Mapping

from leapflow.gateway.adapters.common import (
    AdapterLifecycle,
    JsonHttpClient,
    UrlLibJsonHttpClient,
    bool_option,
    chunk_text,
    stable_message_id,
)
from leapflow.gateway.connectors.protocol import BackendEventSource
from leapflow.gateway.connectors.telegram_event_source import TelegramPollingEventSource
from leapflow.gateway.protocol import InboundMessage, OutboundContent, SendResult, SendTarget


class TelegramAdapter(AdapterLifecycle):
    """Telegram bot adapter supporting polling and outbound text messages."""

    platform_id = "telegram"
    supports_async_delivery = True
    max_message_length = 4096

    def __init__(
        self,
        bot_token: str,
        transport: str = "polling",
        webhook_url: str = "",
        profile: str = "default",
        api_base: str = "https://api.telegram.org",
        poll_interval_s: float = 1.0,
        auto_poll: bool | str = True,
        http_client: JsonHttpClient | None = None,
        **_: Any,
    ) -> None:
        super().__init__(profile=profile)
        self._bot_token = bot_token
        self._transport = transport or "polling"
        self._webhook_url = webhook_url
        self._api_base = api_base.rstrip("/")
        self._poll_interval_s = float(poll_interval_s)
        self._auto_poll = bool_option(auto_poll, default=True)
        self._http = http_client or UrlLibJsonHttpClient()
        self._poll_task: asyncio.Task[None] | None = None
        self._offset = 0
        self._event_source_instance: TelegramPollingEventSource | None = None
        if self._transport == "polling":
            self._event_source_instance = TelegramPollingEventSource(
                bot_token=self._bot_token,
                api_base=self._api_base,
                poll_interval_s=self._poll_interval_s,
                http_client=self._http,
            )

    def event_source(self) -> BackendEventSource | None:
        """Return the polling-based BackendEventSource for the unified pipeline."""
        return self._event_source_instance  # type: ignore[return-value]

    @property
    def bot_identity(self) -> Any:
        """Return bot identity for normalizer injection."""
        if self._event_source_instance:
            from leapflow.gateway.connectors.lark_event_source import BotIdentity
            return BotIdentity(
                open_id=self._event_source_instance.bot_id,
                app_name="",
            )
        return None

    async def connect(self, *, is_reconnect: bool = False) -> None:
        await super().connect(is_reconnect=is_reconnect)
        if self._transport == "webhook" and self._webhook_url:
            await self._api("setWebhook", {"url": self._webhook_url})

    async def disconnect(self) -> None:
        task = self._poll_task
        self._poll_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await super().disconnect()

    async def send(self, target: SendTarget, content: OutboundContent) -> SendResult:
        message_ids: list[str] = []
        for chunk in chunk_text(content.text, self.max_message_length):
            payload = {
                "chat_id": target.chat_id,
                "text": chunk,
            }
            if target.reply_to_id and not message_ids:
                payload["reply_to_message_id"] = target.reply_to_id
            status, data = await self._api("sendMessage", payload)
            if status >= 400 or not data.get("ok", False):
                return SendResult(ok=False, error=str(data.get("description") or data))
            message_id = data.get("result", {}).get("message_id", "")
            if message_id:
                message_ids.append(str(message_id))
        return SendResult(ok=True, message_id=",".join(message_ids))

    async def handle_update(self, update: Mapping[str, Any]) -> InboundMessage | None:
        """Normalise a Telegram update and emit it when it contains text."""
        message = self.message_from_update(update)
        if message is None:
            return None
        await self._emit(message)
        return message

    def message_from_update(self, update: Mapping[str, Any]) -> InboundMessage | None:
        raw_message = update.get("message") or update.get("edited_message")
        if not isinstance(raw_message, dict):
            return None
        text = raw_message.get("text") or raw_message.get("caption") or ""
        if not text:
            return None
        chat = raw_message.get("chat") if isinstance(raw_message.get("chat"), dict) else {}
        user = raw_message.get("from") if isinstance(raw_message.get("from"), dict) else {}
        chat_id = str(chat.get("id") or raw_message.get("chat_id") or "telegram")
        chat_type = str(chat.get("type") or "dm")
        if chat_type == "private":
            chat_type = "dm"
        user_id = str(user.get("id") or "")
        user_name = str(user.get("username") or user.get("first_name") or "")
        message_id = str(raw_message.get("message_id") or stable_message_id("telegram"))
        return InboundMessage(
            source=self._source(
                chat_id=chat_id,
                chat_type=chat_type,
                user_id=user_id,
                user_name=user_name,
            ),
            text=str(text),
            message_id=message_id,
            metadata={"update_id": str(update.get("update_id", ""))},
        )

    async def _poll_loop(self) -> None:
        while self.connected:
            try:
                status, data = await self._api(
                    "getUpdates",
                    {"offset": self._offset, "timeout": 25},
                    timeout_s=30,
                )
                if status < 400 and data.get("ok"):
                    for update in data.get("result", []) or []:
                        if not isinstance(update, dict):
                            continue
                        update_id = int(update.get("update_id", self._offset))
                        self._offset = max(self._offset, update_id + 1)
                        await self.handle_update(update)
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(self._poll_interval_s)
            await asyncio.sleep(self._poll_interval_s)

    async def _api(
        self,
        method: str,
        payload: Mapping[str, Any],
        *,
        timeout_s: float = 10.0,
    ) -> tuple[int, dict[str, Any]]:
        url = f"{self._api_base}/bot{self._bot_token}/{method}"
        return await self._http.request_json(
            "POST",
            url,
            json_body=payload,
            timeout_s=timeout_s,
        )
