# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for SlackBotSignalSource and DiscordBotSignalSource.

Verifies ActiveSignalSource protocol conformance, URL verification handling,
signal emission from parsed events, and start/stop lifecycle — all without
real network calls (mock server binds to ephemeral loopback ports).
"""

from __future__ import annotations

import asyncio
import json
from typing import List

import pytest

from leapflow.perception.active_signal_source import ActiveSignalSource
from leapflow.perception.active_sources.discord_bot import DiscordBotSignalSource
from leapflow.perception.active_sources.slack_bot import SlackBotSignalSource
from leapflow.perception.types import InteractionSignal


# ═══════════════════════════════════════════════════════════════════
# Slack — Protocol Conformance
# ═══════════════════════════════════════════════════════════════════


class TestSlackProtocolConformance:
    """SlackBotSignalSource satisfies the ActiveSignalSource protocol."""

    def test_slack_source_protocol_conformance(self) -> None:
        """isinstance(src, ActiveSignalSource) is True."""
        src = SlackBotSignalSource()
        assert isinstance(src, ActiveSignalSource)

    def test_slack_source_id_and_channel_id_defaults(self) -> None:
        """Default source_id and channel_id match spec."""
        src = SlackBotSignalSource()
        assert src.source_id == "slack_bot"
        assert src.channel_id == "im_message"

    def test_slack_custom_source_id(self) -> None:
        """Custom source_id is respected."""
        src = SlackBotSignalSource(source_id="slack_custom")
        assert src.source_id == "slack_custom"


# ═══════════════════════════════════════════════════════════════════
# Slack — Event Processing
# ═══════════════════════════════════════════════════════════════════


class TestSlackEventProcessing:
    """_process_event handles verification challenge and message events."""

    def test_slack_url_verification_challenge(self) -> None:
        """url_verification payload returns the challenge token."""
        src = SlackBotSignalSource()
        body = json.dumps(
            {"type": "url_verification", "challenge": "abc123"}
        ).encode("utf-8")

        response = src._process_event(body)

        payload = json.loads(response)
        assert payload == {"challenge": "abc123"}

    def test_slack_message_event_emits_signal(self) -> None:
        """event_callback with a message event emits an InteractionSignal."""
        src = SlackBotSignalSource()
        emitted: List[InteractionSignal] = []
        src._emit = emitted.append  # type: ignore[assignment]
        src._running = True

        body = json.dumps(
            {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "user": "U0001",
                    "channel": "C0002",
                    "text": "Hello from Slack!",
                },
            }
        ).encode("utf-8")

        response = src._process_event(body)

        assert response == b""
        assert len(emitted) == 1
        sig = emitted[0]
        assert sig.signal_type == "im_message"
        assert sig.app == "slack"

        detail = json.loads(sig.detail)
        assert detail["sender"] == "U0001"
        assert detail["channel_id"] == "C0002"
        assert detail["text_preview"] == "Hello from Slack!"
        assert detail["platform"] == "slack"

    def test_slack_message_subtype_is_skipped(self) -> None:
        """Messages with a subtype (bot_message, message_changed) are ignored."""
        src = SlackBotSignalSource()
        emitted: List[InteractionSignal] = []
        src._emit = emitted.append  # type: ignore[assignment]
        src._running = True

        body = json.dumps(
            {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "subtype": "bot_message",
                    "user": "U0001",
                    "channel": "C0002",
                    "text": "bot said this",
                },
            }
        ).encode("utf-8")

        src._process_event(body)
        assert len(emitted) == 0

    def test_slack_non_message_event_ignored(self) -> None:
        """Non-message event types are silently ignored."""
        src = SlackBotSignalSource()
        emitted: List[InteractionSignal] = []
        src._emit = emitted.append  # type: ignore[assignment]
        src._running = True

        body = json.dumps(
            {
                "type": "event_callback",
                "event": {"type": "reaction_added", "user": "U1"},
            }
        ).encode("utf-8")

        src._process_event(body)
        assert len(emitted) == 0

    def test_slack_text_preview_truncated(self) -> None:
        """Long text is truncated to 100 chars in the detail preview."""
        src = SlackBotSignalSource()
        emitted: List[InteractionSignal] = []
        src._emit = emitted.append  # type: ignore[assignment]
        src._running = True

        body = json.dumps(
            {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "user": "U",
                    "channel": "C",
                    "text": "x" * 500,
                },
            }
        ).encode("utf-8")

        src._process_event(body)

        detail = json.loads(emitted[0].detail)
        assert len(detail["text_preview"]) == 100

    def test_slack_no_emit_before_start(self) -> None:
        """Without _emit set (not started), no signal is produced."""
        src = SlackBotSignalSource()
        # _emit is None, _running is False
        body = json.dumps(
            {
                "type": "event_callback",
                "event": {"type": "message", "user": "u", "channel": "c", "text": "x"},
            }
        ).encode("utf-8")
        response = src._process_event(body)
        assert response == b""

    def test_slack_empty_body_returns_empty(self) -> None:
        """Empty body returns empty response, does not raise."""
        src = SlackBotSignalSource()
        assert src._process_event(b"") == b""

    def test_slack_invalid_json_returns_empty(self) -> None:
        """Malformed JSON returns empty response, does not raise."""
        src = SlackBotSignalSource()
        assert src._process_event(b"not-json{") == b""


# ═══════════════════════════════════════════════════════════════════
# Slack — Lifecycle
# ═══════════════════════════════════════════════════════════════════


class TestSlackLifecycle:
    """start/stop lifecycle without real network traffic (loopback bind only)."""

    async def test_slack_start_stop_lifecycle(self, unused_tcp_port: int) -> None:
        """start binds server, stop closes it. No external network."""
        src = SlackBotSignalSource(listen_port=unused_tcp_port)
        emitted: List[InteractionSignal] = []
        await src.start(emitted.append)

        assert src._server is not None
        assert src._running is True

        await src.stop()

        assert src._server is None
        assert src._running is False
        assert src._emit is None

    async def test_slack_stop_idempotent(self, unused_tcp_port: int) -> None:
        """Calling stop twice does not raise."""
        src = SlackBotSignalSource(listen_port=unused_tcp_port)
        await src.start(lambda s: None)
        await src.stop()
        await src.stop()

    async def test_slack_bind_failure_is_logged_not_raised(
        self, unused_tcp_port: int
    ) -> None:
        """When the port is already in use, start does not raise."""
        # Bind first instance
        src1 = SlackBotSignalSource(listen_port=unused_tcp_port)
        await src1.start(lambda s: None)
        try:
            # Second instance on the same port should fail bind but not raise
            src2 = SlackBotSignalSource(listen_port=unused_tcp_port)
            await src2.start(lambda s: None)
            # server attribute should remain None on bind failure
            assert src2._server is None
            await src2.stop()
        finally:
            await src1.stop()


# ═══════════════════════════════════════════════════════════════════
# Discord — Protocol Conformance
# ═══════════════════════════════════════════════════════════════════


class TestDiscordProtocolConformance:
    """DiscordBotSignalSource satisfies the ActiveSignalSource protocol."""

    def test_discord_source_protocol_conformance(self) -> None:
        """isinstance(src, ActiveSignalSource) is True."""
        src = DiscordBotSignalSource()
        assert isinstance(src, ActiveSignalSource)

    def test_discord_source_id_and_channel_id_defaults(self) -> None:
        """Default source_id and channel_id match spec."""
        src = DiscordBotSignalSource()
        assert src.source_id == "discord_bot"
        assert src.channel_id == "im_message"

    def test_discord_custom_source_id(self) -> None:
        """Custom source_id is respected."""
        src = DiscordBotSignalSource(source_id="discord_custom")
        assert src.source_id == "discord_custom"


# ═══════════════════════════════════════════════════════════════════
# Discord — Event Processing
# ═══════════════════════════════════════════════════════════════════


class TestDiscordEventProcessing:
    """_process_event handles PING verification and message events."""

    def test_discord_ping_returns_pong(self) -> None:
        """Interaction type=1 (PING) returns type=1 (PONG) JSON."""
        src = DiscordBotSignalSource()
        body = json.dumps({"type": 1}).encode("utf-8")

        response = src._process_event(body)
        payload = json.loads(response)
        assert payload == {"type": 1}

    def test_discord_gateway_message_create_emits_signal(self) -> None:
        """Gateway-style MESSAGE_CREATE event emits an InteractionSignal."""
        src = DiscordBotSignalSource()
        emitted: List[InteractionSignal] = []
        src._emit = emitted.append  # type: ignore[assignment]
        src._running = True

        body = json.dumps(
            {
                "t": "MESSAGE_CREATE",
                "d": {
                    "author": {"username": "alice", "bot": False},
                    "channel_id": "chan_123",
                    "guild_id": "guild_456",
                    "content": "Hello from Discord!",
                },
            }
        ).encode("utf-8")

        response = src._process_event(body)
        assert response == b""
        assert len(emitted) == 1

        sig = emitted[0]
        assert sig.signal_type == "im_message"
        assert sig.app == "discord"

        detail = json.loads(sig.detail)
        assert detail["sender"] == "alice"
        assert detail["channel_id"] == "chan_123"
        assert detail["guild_id"] == "guild_456"
        assert detail["text_preview"] == "Hello from Discord!"
        assert detail["platform"] == "discord"

    def test_discord_direct_webhook_message_emits_signal(self) -> None:
        """Simplified webhook payload (author+content at top level) emits signal."""
        src = DiscordBotSignalSource()
        emitted: List[InteractionSignal] = []
        src._emit = emitted.append  # type: ignore[assignment]
        src._running = True

        body = json.dumps(
            {
                "author": {"username": "bob"},
                "channel_id": "cx",
                "content": "hi there",
            }
        ).encode("utf-8")

        src._process_event(body)
        assert len(emitted) == 1
        detail = json.loads(emitted[0].detail)
        assert detail["sender"] == "bob"

    def test_discord_bot_author_is_skipped(self) -> None:
        """Messages authored by bots (author.bot=True) are ignored."""
        src = DiscordBotSignalSource()
        emitted: List[InteractionSignal] = []
        src._emit = emitted.append  # type: ignore[assignment]
        src._running = True

        body = json.dumps(
            {
                "t": "MESSAGE_CREATE",
                "d": {
                    "author": {"username": "botty", "bot": True},
                    "channel_id": "c",
                    "content": "bot self message",
                },
            }
        ).encode("utf-8")

        src._process_event(body)
        assert len(emitted) == 0

    def test_discord_text_preview_truncated(self) -> None:
        """Long content is truncated to 100 chars."""
        src = DiscordBotSignalSource()
        emitted: List[InteractionSignal] = []
        src._emit = emitted.append  # type: ignore[assignment]
        src._running = True

        body = json.dumps(
            {
                "t": "MESSAGE_CREATE",
                "d": {
                    "author": {"username": "u"},
                    "channel_id": "c",
                    "content": "y" * 500,
                },
            }
        ).encode("utf-8")

        src._process_event(body)
        detail = json.loads(emitted[0].detail)
        assert len(detail["text_preview"]) == 100

    def test_discord_unknown_event_ignored(self) -> None:
        """Unknown gateway event types produce no signal."""
        src = DiscordBotSignalSource()
        emitted: List[InteractionSignal] = []
        src._emit = emitted.append  # type: ignore[assignment]
        src._running = True

        body = json.dumps({"t": "TYPING_START", "d": {}}).encode("utf-8")
        src._process_event(body)
        assert len(emitted) == 0

    def test_discord_no_emit_before_start(self) -> None:
        """Without _emit set (not started), no signal is produced."""
        src = DiscordBotSignalSource()
        body = json.dumps(
            {
                "t": "MESSAGE_CREATE",
                "d": {"author": {"username": "u"}, "channel_id": "c", "content": "x"},
            }
        ).encode("utf-8")
        response = src._process_event(body)
        assert response == b""

    def test_discord_empty_body_returns_empty(self) -> None:
        """Empty body returns empty response, does not raise."""
        src = DiscordBotSignalSource()
        assert src._process_event(b"") == b""

    def test_discord_invalid_json_returns_empty(self) -> None:
        """Malformed JSON returns empty response, does not raise."""
        src = DiscordBotSignalSource()
        assert src._process_event(b"{{{not-json") == b""


# ═══════════════════════════════════════════════════════════════════
# Discord — Lifecycle
# ═══════════════════════════════════════════════════════════════════


class TestDiscordLifecycle:
    """start/stop lifecycle without real network traffic."""

    async def test_discord_start_stop_lifecycle(self, unused_tcp_port: int) -> None:
        """start binds server, stop closes it. No external network."""
        src = DiscordBotSignalSource(listen_port=unused_tcp_port)
        emitted: List[InteractionSignal] = []
        await src.start(emitted.append)

        assert src._server is not None
        assert src._running is True

        await src.stop()

        assert src._server is None
        assert src._running is False
        assert src._emit is None

    async def test_discord_stop_idempotent(self, unused_tcp_port: int) -> None:
        """Calling stop twice does not raise."""
        src = DiscordBotSignalSource(listen_port=unused_tcp_port)
        await src.start(lambda s: None)
        await src.stop()
        await src.stop()


# ═══════════════════════════════════════════════════════════════════
# End-to-end HTTP loopback smoke — real asyncio server, no third-party HTTP client
# ═══════════════════════════════════════════════════════════════════


async def _post_json(host: str, port: int, path: str, payload: dict) -> bytes:
    """Minimal HTTP client using asyncio streams. Returns response body."""
    body = json.dumps(payload).encode("utf-8")
    request = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode("utf-8") + body

    reader, writer = await asyncio.open_connection(host, port)
    try:
        writer.write(request)
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(-1), timeout=2.0)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (OSError, RuntimeError):
            pass

    # Split HTTP response headers from body
    marker = raw.find(b"\r\n\r\n")
    return raw[marker + 4 :] if marker >= 0 else b""


class TestLoopbackSmokeE2E:
    """Drive the real asyncio server end-to-end over loopback."""

    async def test_slack_loopback_url_verification(
        self, unused_tcp_port: int
    ) -> None:
        """Full HTTP POST → server → challenge response over loopback."""
        src = SlackBotSignalSource(listen_port=unused_tcp_port)
        await src.start(lambda s: None)
        try:
            body = await _post_json(
                "127.0.0.1",
                unused_tcp_port,
                "/",
                {"type": "url_verification", "challenge": "zzz"},
            )
            assert json.loads(body) == {"challenge": "zzz"}
        finally:
            await src.stop()

    async def test_slack_loopback_message_emits_signal(
        self, unused_tcp_port: int
    ) -> None:
        """POST a message event; ensure a signal reaches the emit callback."""
        src = SlackBotSignalSource(listen_port=unused_tcp_port)
        emitted: List[InteractionSignal] = []
        await src.start(emitted.append)
        try:
            await _post_json(
                "127.0.0.1",
                unused_tcp_port,
                "/",
                {
                    "type": "event_callback",
                    "event": {
                        "type": "message",
                        "user": "U9",
                        "channel": "C9",
                        "text": "loopback",
                    },
                },
            )
            # Give the event loop a tick to process
            for _ in range(20):
                if emitted:
                    break
                await asyncio.sleep(0.02)
            assert len(emitted) == 1
            assert emitted[0].app == "slack"
        finally:
            await src.stop()

    async def test_discord_loopback_ping_pong(self, unused_tcp_port: int) -> None:
        """Discord PING → PONG response over loopback."""
        src = DiscordBotSignalSource(listen_port=unused_tcp_port)
        await src.start(lambda s: None)
        try:
            body = await _post_json(
                "127.0.0.1", unused_tcp_port, "/", {"type": 1}
            )
            assert json.loads(body) == {"type": 1}
        finally:
            await src.stop()

    async def test_discord_loopback_message_emits_signal(
        self, unused_tcp_port: int
    ) -> None:
        """POST a MESSAGE_CREATE; ensure a signal reaches emit."""
        src = DiscordBotSignalSource(listen_port=unused_tcp_port)
        emitted: List[InteractionSignal] = []
        await src.start(emitted.append)
        try:
            await _post_json(
                "127.0.0.1",
                unused_tcp_port,
                "/",
                {
                    "t": "MESSAGE_CREATE",
                    "d": {
                        "author": {"username": "eve"},
                        "channel_id": "c1",
                        "content": "hi",
                    },
                },
            )
            for _ in range(20):
                if emitted:
                    break
                await asyncio.sleep(0.02)
            assert len(emitted) == 1
            assert emitted[0].app == "discord"
        finally:
            await src.stop()


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def unused_tcp_port() -> int:
    """Pick an unused TCP port on 127.0.0.1 (pytest-asyncio provides this, but
    we define a fallback so the test file works without that plugin flag)."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
