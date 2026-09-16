# Copyright (c) Alibaba, Inc. and its affiliates.
"""Slack Bot ActiveSignalSource.

Receives Slack events via HTTP webhook (Events API) and converts them into
InteractionSignals, enabling the agent to observe and respond to Slack
workspace messages in real-time.

Slack's Events API sends POST requests to a configured URL when subscribed
events occur. This source runs a minimal asyncio HTTP server to receive those
events — same pattern as FeishuIMSignalSource.

Architecture:
    Slack Events API POST -> SlackBotSignalSource.start(emit) -> emit(signal)
    -> AsyncQueue -> consumer -> SignalBuffer + CausalPipeline

Signal format:
    signal_type = "im_message"
    detail = JSON string with: sender, channel_id, text_preview, platform

Usage:
    source = SlackBotSignalSource(
        signing_secret="your_slack_signing_secret",
        listen_port=9878,
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


class SlackBotSignalSource:
    """ActiveSignalSource that receives Slack Events API webhooks.

    Runs a minimal HTTP server and processes incoming event payloads from
    Slack. Handles the URL verification challenge and message events.

    Signal format:
        signal_type = "im_message"
        detail = JSON string with: sender, channel_id, text_preview, platform

    Thread safety:
        Event reception happens on asyncio server callbacks; emission via the
        manager's emit callback is thread-safe (call_soon_threadsafe).
    """

    def __init__(
        self,
        *,
        source_id: str = "slack_bot",
        listen_port: int = 9878,
        signing_secret: str = "",
    ) -> None:
        self._source_id = source_id
        self._listen_port = listen_port
        self._signing_secret = signing_secret
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
        """Start a minimal HTTP callback server for Slack Events API.

        Slack sends POST requests to this endpoint when events occur.
        The server handles URL verification challenges and event callbacks.
        """
        self._emit = emit
        self._running = True

        async def _handle_connection(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            """Handle incoming HTTP POST from Slack Events API."""
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
                "SlackBotSignalSource '%s' listening on 127.0.0.1:%d",
                self._source_id,
                self._listen_port,
            )
        except OSError as exc:
            logger.error(
                "SlackBotSignalSource failed to bind port %d: %s",
                self._listen_port,
                exc,
            )

    def _process_event(self, body: bytes) -> bytes:
        """Parse Slack Events API JSON and emit as InteractionSignal.

        Returns optional response body (for URL verification challenge).
        Handles two payload types:
        - url_verification: returns the challenge token
        - event_callback: processes the event and emits a signal
        """
        if not body:
            return b""

        try:
            data: dict[str, Any] = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return b""

        # Handle Slack's URL verification challenge
        payload_type = data.get("type", "")
        if payload_type == "url_verification":
            challenge = data.get("challenge", "")
            challenge_resp = json.dumps({"challenge": challenge})
            return challenge_resp.encode("utf-8")

        if not self._emit or not self._running:
            return b""

        # Process event_callback payloads
        if payload_type != "event_callback":
            return b""

        event = data.get("event", {})
        event_type = event.get("type", "")

        # Only process message events (ignore subtypes like bot_message)
        if event_type != "message":
            return b""

        # Skip bot messages and message_changed subtypes
        if event.get("subtype"):
            return b""

        sender = event.get("user", "unknown")
        channel = event.get("channel", "")
        text = event.get("text", "")

        # Build signal detail as bounded JSON
        detail = json.dumps(
            {
                "sender": sender,
                "channel_id": channel,
                "text_preview": text[:100],
                "platform": "slack",
            },
            ensure_ascii=False,
        )

        signal = InteractionSignal(
            timestamp=time.time(),
            signal_type="im_message",
            app="slack",
            detail=detail[:500],  # bounded for safety
        )
        self._emit(signal)
        return b""

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
        logger.info("SlackBotSignalSource '%s' stopped", self._source_id)
