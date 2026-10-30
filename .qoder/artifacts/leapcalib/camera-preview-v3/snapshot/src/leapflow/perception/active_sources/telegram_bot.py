# Copyright (c) Alibaba, Inc. and its affiliates.
"""Telegram Bot ActiveSignalSource.

Subscribes to Telegram Bot messages via long polling and emits
InteractionSignal for each incoming message. Demonstrates the ActiveSignalSource
pattern for pull-based (as opposed to Feishu's webhook push-based) IM protocols.

Requires a Telegram Bot token (from @BotFather). No third-party SDK — uses
stdlib urllib for the HTTP calls.

Architecture:
    Telegram getUpdates long poll → TelegramBotSignalSource → emit(signal)
    → SignalBuffer + CausalPipeline

Signal format:
    signal_type = "im_message"
    detail = JSON string with: sender, chat_id, chat_type, text_preview, platform
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from leapflow.perception.active_signal_source import EmitCallback
from leapflow.perception.types import InteractionSignal

logger = logging.getLogger(__name__)


class TelegramBotSignalSource:
    """ActiveSignalSource for Telegram Bot messages via long polling.

    Uses Telegram's `getUpdates` endpoint with long-polling timeout to
    receive new messages without a webhook. The bot token is required.

    Thread safety:
        The polling loop runs in an asyncio task on the event loop. The
        `emit` callback is safe to call from asyncio context (queue.put_nowait
        via call_soon_threadsafe from the manager).
    """

    _API_BASE = "https://api.telegram.org/bot"

    def __init__(
        self,
        *,
        bot_token: str,
        source_id: str = "telegram_bot",
        poll_timeout_s: int = 30,
        request_timeout_s: float = 35.0,
    ) -> None:
        if not bot_token:
            raise ValueError("bot_token is required")
        self._bot_token = bot_token
        self._source_id = source_id
        self._poll_timeout_s = poll_timeout_s
        self._request_timeout_s = request_timeout_s
        self._emit: Optional[EmitCallback] = None
        self._poll_task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._last_update_id: int = 0

    @property
    def source_id(self) -> str:
        """Unique identifier for this source instance."""
        return self._source_id

    @property
    def channel_id(self) -> str:
        """Channel identifier for gating by config.signal_channels."""
        return "im_message"

    async def start(self, emit: EmitCallback) -> None:
        """Start the long-polling loop as a background task."""
        self._emit = emit
        self._running = True
        self._poll_task = asyncio.create_task(
            self._poll_loop(), name=f"telegram-poll:{self._source_id}"
        )
        logger.info(
            "TelegramBotSignalSource '%s' started (long polling)", self._source_id
        )

    async def _poll_loop(self) -> None:
        """Continuously poll getUpdates until stopped."""
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                updates = await loop.run_in_executor(None, self._fetch_updates)
                for update in updates:
                    self._process_update(update)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - polling loop must not die from transient errors
                logger.warning(
                    "TelegramBotSignalSource poll error (retrying): %s", exc
                )
                await asyncio.sleep(2.0)  # backoff on error

    def _fetch_updates(self) -> list[dict[str, Any]]:
        """Blocking HTTP call to Telegram's getUpdates. Runs in executor."""
        url = f"{self._API_BASE}{self._bot_token}/getUpdates"
        params: dict[str, int] = {
            "timeout": self._poll_timeout_s,
        }
        if self._last_update_id:
            params["offset"] = self._last_update_id + 1
        query = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{url}?{query}"
        try:
            with urlopen(full_url, timeout=self._request_timeout_s) as resp:  # noqa: S310
                data = json.loads(resp.read())
        except (URLError, HTTPError, OSError, json.JSONDecodeError) as exc:
            logger.debug("Telegram getUpdates failed: %s", exc)
            return []

        if not data.get("ok"):
            logger.warning(
                "Telegram API returned not-ok: %s",
                data.get("description", "unknown"),
            )
            return []

        return data.get("result", [])

    def _process_update(self, update: dict[str, Any]) -> None:
        """Parse a Telegram Update and emit as InteractionSignal."""
        if not self._emit or not self._running:
            return

        update_id = update.get("update_id", 0)
        if update_id > self._last_update_id:
            self._last_update_id = update_id

        message = update.get("message")
        if not message:
            return  # skip non-message updates (edited_message, callback_query, etc.)

        from_user = message.get("from", {})
        chat = message.get("chat", {})
        text = message.get("text", "")

        detail = json.dumps(
            {
                "sender": from_user.get("username") or str(from_user.get("id", "unknown")),
                "chat_id": chat.get("id"),
                "chat_type": chat.get("type", "private"),
                "text_preview": text[:100],
                "platform": "telegram",
            },
            ensure_ascii=False,
        )

        signal = InteractionSignal(
            timestamp=time.time(),
            signal_type="im_message",
            app="telegram",
            detail=detail[:500],
        )
        self._emit(signal)

    async def stop(self) -> None:
        """Stop the polling loop."""
        self._running = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await asyncio.wait_for(self._poll_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._poll_task = None
        self._emit = None
        logger.info("TelegramBotSignalSource '%s' stopped", self._source_id)
