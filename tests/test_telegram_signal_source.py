# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for TelegramBotSignalSource.

Verifies ActiveSignalSource protocol conformance, fail-fast construction,
signal emission from parsed Telegram updates, update_id tracking, and
start/stop lifecycle — all without real network calls.
"""

from __future__ import annotations

import asyncio
import json
from typing import List
from unittest.mock import patch

import pytest

from leapflow.perception.active_signal_source import ActiveSignalSource, EmitCallback
from leapflow.perception.active_sources.telegram_bot import TelegramBotSignalSource
from leapflow.perception.types import InteractionSignal


# ═══════════════════════════════════════════════════════════════════
# Protocol Conformance
# ═══════════════════════════════════════════════════════════════════


class TestTelegramProtocolConformance:
    """TelegramBotSignalSource satisfies the ActiveSignalSource protocol."""

    def test_telegram_source_protocol_conformance(self) -> None:
        """isinstance(src, ActiveSignalSource) is True."""
        src = TelegramBotSignalSource(bot_token="fake:token")
        assert isinstance(src, ActiveSignalSource)

    def test_telegram_requires_bot_token(self) -> None:
        """Empty bot_token raises ValueError at construction."""
        with pytest.raises(ValueError, match="bot_token is required"):
            TelegramBotSignalSource(bot_token="")

    def test_telegram_source_id_and_channel_id(self) -> None:
        """Default source_id and channel_id match spec."""
        src = TelegramBotSignalSource(bot_token="123:ABC")
        assert src.source_id == "telegram_bot"
        assert src.channel_id == "im_message"

    def test_telegram_custom_source_id(self) -> None:
        """Custom source_id is respected."""
        src = TelegramBotSignalSource(bot_token="123:ABC", source_id="tg_custom")
        assert src.source_id == "tg_custom"


# ═══════════════════════════════════════════════════════════════════
# Signal Emission
# ═══════════════════════════════════════════════════════════════════


class TestTelegramSignalEmission:
    """_process_update emits correct InteractionSignal."""

    def test_telegram_process_update_emits_signal(self) -> None:
        """Feed a fake update dict, verify emit called with correct signal."""
        src = TelegramBotSignalSource(bot_token="fake:token")
        emitted: List[InteractionSignal] = []
        src._emit = emitted.append  # type: ignore[assignment]
        src._running = True

        update = {
            "update_id": 100,
            "message": {
                "from": {"id": 42, "username": "testuser"},
                "chat": {"id": -1001, "type": "group"},
                "text": "Hello from Telegram!",
            },
        }
        src._process_update(update)

        assert len(emitted) == 1
        signal = emitted[0]
        assert signal.signal_type == "im_message"
        assert signal.app == "telegram"

        detail = json.loads(signal.detail)
        assert detail["sender"] == "testuser"
        assert detail["chat_id"] == -1001
        assert detail["chat_type"] == "group"
        assert detail["text_preview"] == "Hello from Telegram!"
        assert detail["platform"] == "telegram"

    def test_telegram_process_update_sender_fallback_to_id(self) -> None:
        """When username is absent, sender falls back to str(id)."""
        src = TelegramBotSignalSource(bot_token="fake:token")
        emitted: List[InteractionSignal] = []
        src._emit = emitted.append  # type: ignore[assignment]
        src._running = True

        update = {
            "update_id": 101,
            "message": {
                "from": {"id": 999},
                "chat": {"id": 123, "type": "private"},
                "text": "no username",
            },
        }
        src._process_update(update)

        assert len(emitted) == 1
        detail = json.loads(emitted[0].detail)
        assert detail["sender"] == "999"

    def test_telegram_process_update_ignores_non_message(self) -> None:
        """Updates without a 'message' key are skipped."""
        src = TelegramBotSignalSource(bot_token="fake:token")
        emitted: List[InteractionSignal] = []
        src._emit = emitted.append  # type: ignore[assignment]
        src._running = True

        # edited_message, callback_query, etc. — no "message" key
        update_edited = {"update_id": 200, "edited_message": {"text": "edited"}}
        update_callback = {"update_id": 201, "callback_query": {"data": "click"}}

        src._process_update(update_edited)
        src._process_update(update_callback)

        assert len(emitted) == 0

    def test_telegram_process_update_truncates_text_preview(self) -> None:
        """Text preview is truncated to 100 chars."""
        src = TelegramBotSignalSource(bot_token="fake:token")
        emitted: List[InteractionSignal] = []
        src._emit = emitted.append  # type: ignore[assignment]
        src._running = True

        long_text = "x" * 200
        update = {
            "update_id": 300,
            "message": {
                "from": {"id": 1, "username": "u"},
                "chat": {"id": 1, "type": "private"},
                "text": long_text,
            },
        }
        src._process_update(update)

        detail = json.loads(emitted[0].detail)
        assert len(detail["text_preview"]) == 100


# ═══════════════════════════════════════════════════════════════════
# Update ID Tracking
# ═══════════════════════════════════════════════════════════════════


class TestTelegramUpdateIdTracking:
    """_last_update_id advances correctly after processing updates."""

    def test_telegram_update_id_tracking(self) -> None:
        """After processing update with id=5, _last_update_id becomes 5."""
        src = TelegramBotSignalSource(bot_token="fake:token")
        src._emit = lambda s: None  # type: ignore[assignment]
        src._running = True

        src._process_update({
            "update_id": 5,
            "message": {
                "from": {"id": 1},
                "chat": {"id": 1, "type": "private"},
                "text": "hi",
            },
        })
        assert src._last_update_id == 5

    def test_telegram_update_id_monotonic(self) -> None:
        """update_id never decreases — out-of-order updates don't regress."""
        src = TelegramBotSignalSource(bot_token="fake:token")
        src._emit = lambda s: None  # type: ignore[assignment]
        src._running = True

        src._process_update({
            "update_id": 10,
            "message": {"from": {"id": 1}, "chat": {"id": 1, "type": "private"}, "text": "a"},
        })
        src._process_update({
            "update_id": 8,
            "message": {"from": {"id": 1}, "chat": {"id": 1, "type": "private"}, "text": "b"},
        })
        assert src._last_update_id == 10

    def test_telegram_fetch_updates_uses_offset(self) -> None:
        """After update_id=5 processed, next _fetch_updates builds offset=6."""
        src = TelegramBotSignalSource(bot_token="fake:token", poll_timeout_s=10)
        src._last_update_id = 5

        # Intercept urlopen to verify the URL
        called_urls: List[str] = []

        def fake_urlopen(url, *, timeout=None):
            called_urls.append(url)
            raise OSError("mocked")

        with patch(
            "leapflow.perception.active_sources.telegram_bot.urlopen",
            side_effect=fake_urlopen,
        ):
            result = src._fetch_updates()

        assert result == []
        assert len(called_urls) == 1
        assert "offset=6" in called_urls[0]
        assert "timeout=10" in called_urls[0]


# ═══════════════════════════════════════════════════════════════════
# Lifecycle
# ═══════════════════════════════════════════════════════════════════


class TestTelegramLifecycle:
    """start/stop lifecycle without real network calls."""

    async def test_telegram_start_stop_lifecycle(self) -> None:
        """start creates poll task; stop cancels and cleans up."""
        src = TelegramBotSignalSource(bot_token="fake:token")
        emitted: List[InteractionSignal] = []

        # Monkey-patch _fetch_updates to avoid real network
        src._fetch_updates = lambda: []  # type: ignore[assignment]

        await src.start(emitted.append)

        # Task should be running
        assert src._poll_task is not None
        assert not src._poll_task.done()
        assert src._running is True

        # Give the loop a tick
        await asyncio.sleep(0.05)

        await src.stop()

        # After stop, task is cleaned up
        assert src._poll_task is None
        assert src._running is False
        assert src._emit is None

    async def test_telegram_stop_idempotent(self) -> None:
        """Calling stop() twice does not raise."""
        src = TelegramBotSignalSource(bot_token="fake:token")
        src._fetch_updates = lambda: []  # type: ignore[assignment]

        await src.start(lambda s: None)
        await asyncio.sleep(0.02)
        await src.stop()
        # Second stop is safe
        await src.stop()

    async def test_telegram_no_emit_after_stop(self) -> None:
        """After stop, _process_update does not emit."""
        src = TelegramBotSignalSource(bot_token="fake:token")
        emitted: List[InteractionSignal] = []
        src._emit = emitted.append  # type: ignore[assignment]
        src._running = False  # simulate stopped state

        src._process_update({
            "update_id": 1,
            "message": {"from": {"id": 1}, "chat": {"id": 1, "type": "private"}, "text": "x"},
        })
        assert len(emitted) == 0
