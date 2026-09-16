# Copyright (c) Alibaba, Inc. and its affiliates.
"""Built-in ActiveSignalSource implementations.

The FileWatchSignalSource is the community-extension exemplar: it demonstrates
the correct pattern for external event streams (filesystem in this case),
including thread-safe emission and graceful shutdown.

Also provides:
- WebhookSignalSource: receives signals via HTTP webhook (stdlib asyncio only)
- CronSignalSource: emits periodic timer signals at configurable intervals
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from leapflow.perception.active_signal_source import EmitCallback
from leapflow.perception.types import InteractionSignal

logger = logging.getLogger(__name__)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    _WATCHDOG_AVAILABLE = True
except ImportError:
    _WATCHDOG_AVAILABLE = False
    FileSystemEventHandler = object  # type: ignore[assignment,misc]


class _WatchdogHandler(FileSystemEventHandler):  # type: ignore[misc]
    """Adapts watchdog FileSystemEventHandler to our EmitCallback."""

    def __init__(self, emit: EmitCallback, source_id: str) -> None:
        super().__init__()
        self._emit = emit
        self._source_id = source_id

    def on_any_event(self, event: Any) -> None:
        """Called by watchdog for every filesystem event. Emits an InteractionSignal."""
        try:
            event_type = getattr(event, "event_type", "unknown")
            src_path = getattr(event, "src_path", "")
            signal = InteractionSignal(
                timestamp=time.time(),
                signal_type="file_change",
                detail=f"{event_type}:{src_path}",
            )
            self._emit(signal)
        except (AttributeError, TypeError):
            # Swallow to protect the observer thread
            pass


class FileWatchSignalSource:
    """Watches configured filesystem paths and emits InteractionSignal on changes.

    Uses watchdog (already a project dependency) with event-driven monitoring.
    Emits signals with signal_type="file_change" and detail="{event_type}:{path}".

    Thread safety:
        watchdog's Observer runs in a background thread; the emit callback is
        called from that thread. ActiveSourceManager's emit uses
        asyncio.Queue.put_nowait() which is thread-safe.
    """

    def __init__(
        self,
        watch_paths: Sequence[str | Path],
        *,
        source_id: str = "file_watch",
        recursive: bool = True,
    ) -> None:
        self._watch_paths = [Path(p) for p in watch_paths]
        self._source_id = source_id
        self._recursive = recursive
        self._observer: Optional[Any] = None
        self._emit: Optional[EmitCallback] = None

    @property
    def source_id(self) -> str:
        """Unique identifier for this source instance."""
        return self._source_id

    @property
    def channel_id(self) -> str:
        """Channel identifier for gating by config.signal_channels."""
        return "file_watch"

    async def start(self, emit: EmitCallback) -> None:
        """Start the watchdog Observer on all configured paths."""
        if not _WATCHDOG_AVAILABLE:
            logger.error(
                "watchdog not available; FileWatchSignalSource %r cannot start",
                self._source_id,
            )
            return

        self._emit = emit
        handler = _WatchdogHandler(emit, self._source_id)
        self._observer = Observer()

        for path in self._watch_paths:
            if not path.exists():
                logger.warning(
                    "FileWatchSignalSource: path does not exist: %s", path
                )
                continue
            self._observer.schedule(handler, str(path), recursive=self._recursive)

        self._observer.start()
        logger.info(
            "FileWatchSignalSource %r watching %d paths",
            self._source_id, len(self._watch_paths),
        )

    async def stop(self) -> None:
        """Stop the observer and wait for its thread to exit."""
        if self._observer is None:
            return
        try:
            self._observer.stop()
            # observer.join is blocking -- wrap in executor to be async-friendly
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, lambda: self._observer.join(timeout=2.0)  # type: ignore[union-attr]
            )
        except (RuntimeError, AttributeError) as exc:
            logger.debug(
                "FileWatchSignalSource stop error: %s", exc, exc_info=True
            )
        finally:
            self._observer = None
            self._emit = None


class WebhookSignalSource:
    """Receives signals via HTTP webhook endpoint.

    Starts a minimal asyncio TCP server on a configured port. External
    services POST JSON payloads to ``/signal`` which are transformed into
    InteractionSignals. Uses only stdlib (asyncio.start_server), no aiohttp.
    """

    def __init__(self, port: int = 8765, host: str = "127.0.0.1") -> None:
        self._port = port
        self._host = host
        self._server: Optional[asyncio.AbstractServer] = None
        self._emit: Optional[EmitCallback] = None

    @property
    def source_id(self) -> str:
        """Unique identifier for this source instance."""
        return "webhook"

    @property
    def channel_id(self) -> str:
        """Channel identifier for gating by config.signal_channels."""
        return "webhook"

    async def start(self, emit: EmitCallback) -> None:
        """Start the HTTP server and begin accepting webhook POSTs."""
        self._emit = emit
        self._server = await asyncio.start_server(
            self._handle_connection, self._host, self._port
        )
        logger.info(
            "WebhookSignalSource listening on %s:%d", self._host, self._port
        )

    async def stop(self) -> None:
        """Close the server and release resources."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._emit = None

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a single HTTP connection (minimal HTTP parser)."""
        try:
            # Read request line and headers
            request_line = await reader.readline()
            if not request_line:
                writer.close()
                return

            headers: dict[str, str] = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                decoded = line.decode("utf-8", errors="replace").strip()
                if ":" in decoded:
                    key, val = decoded.split(":", 1)
                    headers[key.strip().lower()] = val.strip()

            # Read body if Content-Length present
            content_length = int(headers.get("content-length", "0"))
            body = b""
            if content_length > 0:
                body = await reader.readexactly(content_length)

            # Parse method and path
            parts = request_line.decode("utf-8", errors="replace").split()
            method = parts[0] if parts else ""
            path = parts[1] if len(parts) > 1 else "/"

            if method == "POST" and path == "/signal":
                self._emit_signal(body)
                response = (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: 15\r\n"
                    b"\r\n"
                    b'{"status":"ok"}'
                )
            else:
                response = (
                    b"HTTP/1.1 404 Not Found\r\n"
                    b"Content-Length: 0\r\n"
                    b"\r\n"
                )

            writer.write(response)
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError, OSError):
            pass
        finally:
            writer.close()

    def _emit_signal(self, body: bytes) -> None:
        """Parse JSON body and emit as InteractionSignal."""
        if self._emit is None:
            return
        try:
            payload = _json.loads(body) if body else {}
        except (ValueError, _json.JSONDecodeError):
            payload = {"raw": body.decode("utf-8", errors="replace")}

        detail = _json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else str(payload)
        signal = InteractionSignal(
            timestamp=time.time(),
            signal_type="webhook",
            detail=detail,
        )
        self._emit(signal)


class CronSignalSource:
    """Emits periodic timer signals at configurable intervals.

    Useful for scheduled checks, heartbeats, or periodic automation triggers.
    Emits signals with signal_type="cron" and detail=label.
    """

    def __init__(self, interval_s: float = 60.0, label: str = "tick") -> None:
        self._interval_s = max(0.1, interval_s)
        self._label = label
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None

    @property
    def source_id(self) -> str:
        """Unique identifier for this source instance."""
        return "cron"

    @property
    def channel_id(self) -> str:
        """Channel identifier for gating by config.signal_channels."""
        return "cron"

    async def start(self, emit: EmitCallback) -> None:
        """Start the periodic timer loop."""
        self._running = True
        self._task = asyncio.create_task(self._tick_loop(emit))
        logger.info(
            "CronSignalSource started: interval=%.1fs, label=%r",
            self._interval_s, self._label,
        )

    async def stop(self) -> None:
        """Stop the timer loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _tick_loop(self, emit: EmitCallback) -> None:
        """Internal loop emitting signals at the configured interval."""
        while self._running:
            signal = InteractionSignal(
                timestamp=time.time(),
                signal_type="cron",
                detail=self._label,
            )
            emit(signal)
            await asyncio.sleep(self._interval_s)
