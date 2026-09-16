# Copyright (c) Alibaba, Inc. and its affiliates.
"""Discord Bot ActiveSignalSource.

Receives Discord interaction events via HTTP webhook (Interactions Endpoint)
and converts them into InteractionSignals, enabling the agent to observe and
respond to Discord messages in real-time.

Discord supports an Interactions Endpoint URL that receives POST requests for
slash commands and message components. For general message observation, this
source listens for forwarded message events from a Discord bot's event webhook.

Architecture:
    Discord webhook POST -> DiscordBotSignalSource.start(emit) -> emit(signal)
    -> AsyncQueue -> consumer -> SignalBuffer + CausalPipeline

Signal format:
    signal_type = "im_message"
    detail = JSON string with: sender, channel_id, guild_id, text_preview, platform

Usage:
    source = DiscordBotSignalSource(
        public_key="your_discord_application_public_key",
        listen_port=9879,
    )
    manager.register(source)
    await manager.start_all()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

from leapflow.perception.active_signal_source import EmitCallback
from leapflow.perception.types import InteractionSignal

logger = logging.getLogger(__name__)


class DiscordBotSignalSource:
    """ActiveSignalSource that receives Discord webhook events.

    Runs a minimal HTTP server and processes incoming event payloads from
    Discord's Interactions Endpoint or a message-forwarding webhook.

    Handles Discord's PING verification (type=1) and MESSAGE_CREATE events.

    Signal format:
        signal_type = "im_message"
        detail = JSON string with: sender, channel_id, guild_id, text_preview, platform

    Thread safety:
        Event reception happens on asyncio server callbacks; emission via the
        manager's emit callback is thread-safe (call_soon_threadsafe).
    """

    # Discord Interaction types
    _INTERACTION_PING = 1
    _INTERACTION_APPLICATION_COMMAND = 2
    _INTERACTION_MESSAGE_COMPONENT = 3

    def __init__(
        self,
        *,
        source_id: str = "discord_bot",
        listen_port: int = 9879,
        public_key: str = "",
    ) -> None:
        self._source_id = source_id
        self._listen_port = listen_port
        self._public_key = public_key
        self._emit: Optional[EmitCallback] = None
        self._server: Optional[asyncio.Server] = None
        self._running = False

    @property
    def source_id(self) -> str:
        """Unique identifier for this source instance."""
        return self._source_id

    @property
    def channel_id(self) -> str:
        """Channel identifier for gating by config.signal_channels."""
        return "im_message"

    async def start(self, emit: EmitCallback) -> None:
        """Start a minimal HTTP callback server for Discord Interactions Endpoint.

        Discord sends POST requests to this endpoint for interactions and
        message events. The server handles PING verification and message events.
        """
        self._emit = emit
        self._running = True

        async def _handle_connection(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            """Handle incoming HTTP POST from Discord."""
            try:
                # Read HTTP request line (minimal parsing)
                request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if not request_line:
                    writer.close()
                    return

                # Read headers to find Content-Length
                content_length = 0
                while True:
                    header = await asyncio.wait_for(reader.readline(), timeout=5.0)
                    if header in (b"\r\n", b"\n", b""):
                        break
                    if header.lower().startswith(b"content-length:"):
                        content_length = int(header.split(b":")[1].strip())

                # Read body
                body = b""
                if content_length > 0:
                    body = await asyncio.wait_for(
                        reader.readexactly(content_length), timeout=5.0
                    )

                # Process the event
                response_body = self._process_event(body)

                # Send HTTP response
                if response_body:
                    resp = (
                        b"HTTP/1.1 200 OK\r\n"
                        b"Content-Type: application/json\r\n"
                        b"Content-Length: " + str(len(response_body)).encode() + b"\r\n"
                        b"\r\n" + response_body
                    )
                else:
                    resp = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
                writer.write(resp)
                await writer.drain()
            except (asyncio.TimeoutError, ConnectionResetError, OSError):
                pass
            finally:
                try:
                    writer.close()
                except (OSError, RuntimeError):
                    pass

        try:
            self._server = await asyncio.start_server(
                _handle_connection,
                host="127.0.0.1",
                port=self._listen_port,
            )
            logger.info(
                "DiscordBotSignalSource '%s' listening on 127.0.0.1:%d",
                self._source_id,
                self._listen_port,
            )
        except OSError as exc:
            logger.error(
                "DiscordBotSignalSource failed to bind port %d: %s",
                self._listen_port,
                exc,
            )

    def _process_event(self, body: bytes) -> bytes:
        """Parse Discord event JSON and emit as InteractionSignal.

        Returns optional response body (for PING verification).
        Handles two event formats:
        - Discord Interaction (type=1 PING): returns type=1 PONG
        - Message event (type="MESSAGE_CREATE" or embedded in interaction data):
          processes and emits signal
        """
        if not body:
            return b""

        try:
            data: dict[str, Any] = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return b""

        # Handle Discord's PING verification (Interaction type 1)
        interaction_type = data.get("type")
        if interaction_type == self._INTERACTION_PING:
            pong_resp = json.dumps({"type": 1})
            return pong_resp.encode("utf-8")

        if not self._emit or not self._running:
            return b""

        # Handle MESSAGE_CREATE events (forwarded by bot gateway or webhook)
        event_type = data.get("t", "")
        if event_type == "MESSAGE_CREATE":
            event_data = data.get("d", {})
            self._emit_message_signal(event_data)
            return b""

        # Handle direct message payload (simplified webhook format)
        if "content" in data and "author" in data:
            self._emit_message_signal(data)
            return b""

        return b""

    def _emit_message_signal(self, message: dict[str, Any]) -> None:
        """Extract message fields and emit as InteractionSignal."""
        if not self._emit or not self._running:
            return

        author = message.get("author", {})
        sender = author.get("username", "unknown")
        channel = message.get("channel_id", "")
        guild_id = message.get("guild_id", "")
        content = message.get("content", "")

        # Skip bot messages
        if author.get("bot", False):
            return

        # Build signal detail as bounded JSON
        detail = json.dumps(
            {
                "sender": sender,
                "channel_id": channel,
                "guild_id": guild_id,
                "text_preview": content[:100],
                "platform": "discord",
            },
            ensure_ascii=False,
        )

        signal = InteractionSignal(
            timestamp=time.time(),
            signal_type="im_message",
            app="discord",
            detail=detail[:500],  # bounded for safety
        )
        self._emit(signal)

    async def stop(self) -> None:
        """Stop the callback server."""
        self._running = False
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            self._server = None
        self._emit = None
        logger.info("DiscordBotSignalSource '%s' stopped", self._source_id)
