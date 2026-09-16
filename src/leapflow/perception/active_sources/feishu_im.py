# Copyright (c) Alibaba, Inc. and its affiliates.
"""Feishu IM Bot ActiveSignalSource.

Receives Feishu instant messages and converts them into InteractionSignals,
enabling the agent to observe and respond to collaboration signals from
the IM environment in real-time.

This is the primary demonstration of LeapFlow's "observe real-world signals"
capability: external IM events flow through the same signal pipeline
(SignalBuffer -> CausalFusionPipeline) as UI events and file changes.

Architecture:
    Feishu webhook/event -> FeishuIMSignalSource.start(emit) -> emit(signal)
    -> AsyncQueue -> consumer -> SignalBuffer + CausalPipeline

Usage:
    source = FeishuIMSignalSource(
        app_id="cli_...",
        event_types=["im.message.receive_v1"],
    )
    manager.register(source)
    await manager.start_all()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional, Sequence

from leapflow.perception.active_signal_source import EmitCallback
from leapflow.perception.types import InteractionSignal

logger = logging.getLogger(__name__)


class FeishuIMSignalSource:
    """ActiveSignalSource that subscribes to Feishu IM events.

    Connects to Feishu's event subscription (via webhook callback server or
    long-poll mechanism) and emits InteractionSignal for each received message.

    Signal format:
        signal_type = "im_message"
        detail = JSON string with: sender, chat_id, message_type, text_preview

    Thread safety:
        Event reception may happen on a server callback thread; emission
        via the manager's emit callback is thread-safe (call_soon_threadsafe).
    """

    def __init__(
        self,
        *,
        source_id: str = "feishu_im",
        listen_port: int = 9876,
        event_types: Optional[Sequence[str]] = None,
    ) -> None:
        self._source_id = source_id
        self._listen_port = listen_port
        self._event_types = set(event_types or ["im.message.receive_v1"])
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
        """Start a minimal HTTP callback server for Feishu event subscription.

        In production, Feishu sends POST requests to this endpoint when
        configured as a webhook URL in the Feishu bot settings.
        """
        self._emit = emit
        self._running = True

        async def _handle_connection(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            """Handle incoming HTTP POST from Feishu webhook."""
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

                # Send 200 OK response (Feishu requires acknowledgment)
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
                "FeishuIMSignalSource '%s' listening on 127.0.0.1:%d",
                self._source_id,
                self._listen_port,
            )
        except OSError as exc:
            logger.error(
                "FeishuIMSignalSource failed to bind port %d: %s",
                self._listen_port,
                exc,
            )

    def _process_event(self, body: bytes) -> bytes:
        """Parse Feishu event JSON and emit as InteractionSignal.

        Returns optional response body (for URL verification challenge).
        """
        if not body:
            return b""

        try:
            data: dict[str, Any] = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return b""

        # Handle Feishu's URL verification challenge
        if "challenge" in data:
            challenge_resp = json.dumps({"challenge": data["challenge"]})
            return challenge_resp.encode("utf-8")

        if not self._emit or not self._running:
            return b""

        # Extract event header
        header = data.get("header", {})
        event_type = header.get("event_type", "")

        if event_type not in self._event_types:
            return b""

        # Extract message details from the event payload
        event = data.get("event", {})
        message = event.get("message", {})
        sender = (
            event.get("sender", {}).get("sender_id", {}).get("open_id", "unknown")
        )
        chat_id = message.get("chat_id", "")
        msg_type = message.get("message_type", "text")

        # Extract text preview (for text messages)
        text_preview = self._extract_text_preview(message, msg_type)

        # Build signal detail as bounded JSON
        detail = json.dumps(
            {
                "sender": sender,
                "chat_id": chat_id,
                "message_type": msg_type,
                "text_preview": text_preview,
                "event_type": event_type,
            },
            ensure_ascii=False,
        )

        signal = InteractionSignal(
            timestamp=time.time(),
            signal_type="im_message",
            app="feishu",
            detail=detail[:500],  # bounded for safety
        )
        self._emit(signal)
        return b""

    @staticmethod
    def _extract_text_preview(message: dict[str, Any], msg_type: str) -> str:
        """Extract a bounded text preview from the message content."""
        content_str = message.get("content", "")
        if not content_str:
            return f"[{msg_type}]"
        try:
            content = json.loads(content_str)
            text = content.get("text", "")
            return text[:100] if text else f"[{msg_type}]"
        except (json.JSONDecodeError, ValueError):
            return f"[{msg_type}]"

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
        logger.info("FeishuIMSignalSource '%s' stopped", self._source_id)
