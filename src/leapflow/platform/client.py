"""Async Unix Domain Socket RPC client with bidirectional event support."""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional

from leapflow.platform.protocol import (
    EventHandler,
    HostRpc,
    RpcError,
    decode_packet,
    encode_packet,
    make_request,
)

logger = logging.getLogger(__name__)

_RECONNECT_DELAY_S = 1.5
_MAX_RECONNECT_ATTEMPTS = 5
_CALL_TIMEOUT_S = 30.0  # Default fallback timeout when no method-specific entry matches.
_KEEPALIVE_INTERVAL_S = 15.0

# ── Per-method RPC timeout map ──
# Keys are matched against the method *prefix* (segment before first '.').
# This keeps fast operations (ping, ui_action) snappy while allowing
# legitimately slow operations (file IO) more headroom. Override per-instance
# via ``BridgeClient(timeout_overrides=...)`` or via Settings.
_RPC_TIMEOUT_MAP: Dict[str, float] = {
    "ping": 2.0,
    "screenshot": 5.0,
    "ax": 5.0,
    "ui_action": 3.0,
    "input": 3.0,
    "file": 15.0,
}


class BridgeClient(HostRpc):
    """Bidirectional Unix Domain Socket client.

    Supports:
    - Request/response RPC calls (Python → Swift)
    - Server-pushed events (Swift → Python) via registered handlers
    - Automatic reconnection with exponential backoff
    """

    def __init__(
        self,
        socket_path: Path,
        *,
        timeout_overrides: Optional[Dict[str, float]] = None,
        default_timeout: float = _CALL_TIMEOUT_S,
    ) -> None:
        self._path = socket_path
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._rx_buf = bytearray()
        self._call_lock = asyncio.Lock()
        self._pending: Dict[str, asyncio.Future[Any]] = {}
        self._event_handlers: List[EventHandler] = []
        self._listen_task: Optional[asyncio.Task[None]] = None
        self._reconnect_callbacks: List[Callable[[], Awaitable[None]]] = []
        self._keepalive_task: Optional[asyncio.Task[None]] = None
        self._closed = False
        self._disconnected = False
        # Merge defaults with optional caller overrides; caller wins on conflict.
        self._timeout_map: Dict[str, float] = dict(_RPC_TIMEOUT_MAP)
        if timeout_overrides:
            self._timeout_map.update(timeout_overrides)
        self._default_timeout: float = default_timeout

    def _resolve_timeout(self, method: str) -> float:
        """Resolve the call timeout for ``method`` via prefix matching.

        The prefix is the segment before the first ``.`` (e.g. ``ax.tree`` →
        ``ax``). Falls back to the configured default when no entry matches.
        """
        prefix = method.split(".", 1)[0] if method else ""
        return self._timeout_map.get(prefix, self._default_timeout)

    @property
    def connected(self) -> bool:
        return (
            self._writer is not None
            and not self._writer.is_closing()
            and not self._disconnected
        )

    def on_event(self, handler: EventHandler) -> None:
        """Register a handler invoked for every server-pushed event."""
        self._event_handlers.append(handler)

    def on_reconnect(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register a callback invoked after successful reconnect."""
        self._reconnect_callbacks.append(callback)

    async def fire_reconnect_callbacks(self) -> None:
        """Fire all reconnect callbacks (used after background connect)."""
        for cb in self._reconnect_callbacks:
            try:
                await cb()
            except Exception:
                logger.debug("Reconnect callback error", exc_info=True)

    async def try_connect(self) -> bool:
        """Single non-blocking connection attempt. Returns True on success."""
        if self.connected:
            return True
        self._disconnected = False
        try:
            reader, writer = await asyncio.open_unix_connection(path=str(self._path))
            self._reader = reader
            self._writer = writer
            self._rx_buf.clear()
            self._pending.clear()
            if self._listen_task is None or self._listen_task.done():
                self._listen_task = asyncio.create_task(self._receive_loop())
            if self._keepalive_task is None or self._keepalive_task.done():
                self._keepalive_task = asyncio.create_task(self._keepalive_loop())
            logger.info("Bridge connected to %s", self._path)
            return True
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return False

    async def connect(self) -> None:
        if self.connected:
            return
        self._disconnected = False
        for attempt in range(_MAX_RECONNECT_ATTEMPTS):
            try:
                reader, writer = await asyncio.open_unix_connection(path=str(self._path))
                self._reader = reader
                self._writer = writer
                self._rx_buf.clear()
                self._pending.clear()
                if self._listen_task is None or self._listen_task.done():
                    self._listen_task = asyncio.create_task(self._receive_loop())
                if self._keepalive_task is None or self._keepalive_task.done():
                    self._keepalive_task = asyncio.create_task(self._keepalive_loop())
                logger.info("Bridge connected to %s", self._path)
                return
            except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
                if attempt >= _MAX_RECONNECT_ATTEMPTS - 1:
                    raise ConnectionError(f"Cannot connect to OSHost: {exc}") from exc
                delay = _RECONNECT_DELAY_S * (2**attempt)
                logger.warning("Bridge connect attempt %d failed, retrying in %.1fs", attempt + 1, delay)
                await asyncio.sleep(delay)

    async def _reconnect(self) -> None:
        """Attempt reconnection after a detected disconnect."""
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._writer = None
        self._reader = None
        self._rx_buf.clear()
        self._disconnected = False
        await self.connect()

    async def close(self) -> None:
        self._closed = True
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        self._writer = None
        self._reader = None
        self._rx_buf.clear()
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("bridge closed"))
        self._pending.clear()

    async def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Invoke an RPC method and await the result.

        Retries once with a single reconnection attempt on failure.
        """
        for attempt in range(2):
            try:
                return await self._do_call(method, params)
            except ConnectionError as e:
                if attempt == 0 and not self._closed:
                    ok = await self.try_connect()
                    if not ok:
                        raise
                else:
                    raise

    async def _do_call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Single-attempt RPC call."""
        async with self._call_lock:
            if not self.connected:
                ok = await self.try_connect()
                if not ok:
                    raise ConnectionError("bridge not connected")
            rid = str(uuid.uuid4())
            fut: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
            self._pending[rid] = fut
            frame = encode_packet(make_request(rid, method, params))
            try:
                self._writer.write(frame)
                await self._writer.drain()
            except (ConnectionError, OSError) as exc:
                self._pending.pop(rid, None)
                self._disconnected = True
                raise ConnectionError(f"bridge write failed: {exc}") from exc

        timeout_s = self._resolve_timeout(method)
        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            logger.warning("RPC %s timed out after %.2fs", method, timeout_s)
            raise RpcError("timeout", f"RPC {method} timed out after {timeout_s}s", {})

    async def _receive_loop(self) -> None:
        """Background task: read frames and dispatch responses/events."""
        try:
            while not self._closed and self._reader:
                chunk = await self._reader.read(65536)
                if not chunk:
                    logger.warning("Bridge connection EOF")
                    break
                self._rx_buf.extend(chunk)
                self._drain_frames()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("Bridge receive loop error: %s", exc, exc_info=True)
        finally:
            self._handle_disconnect()
            if not self._closed:
                asyncio.ensure_future(self._auto_reconnect())

    async def _auto_reconnect(self) -> None:
        """Attempt reconnection with exponential backoff after unexpected disconnect."""
        for attempt in range(_MAX_RECONNECT_ATTEMPTS):
            delay = _RECONNECT_DELAY_S * (2 ** attempt)
            await asyncio.sleep(delay)
            if self._closed:
                return
            ok = await self.try_connect()
            if ok:
                logger.info("Bridge auto-reconnected (attempt %d)", attempt + 1)
                await self.fire_reconnect_callbacks()
                return
        logger.warning(
            "Bridge auto-reconnect failed after %d attempts",
            _MAX_RECONNECT_ATTEMPTS,
        )

    async def _keepalive_loop(self) -> None:
        """Send periodic pings to prevent idle disconnects."""
        try:
            while not self._closed and not self._disconnected:
                await asyncio.sleep(_KEEPALIVE_INTERVAL_S)
                if self._closed or self._disconnected:
                    break
                try:
                    await self._do_call("ping", {})
                except Exception:
                    break
        except asyncio.CancelledError:
            pass

    def _drain_frames(self) -> None:
        """Parse all complete frames from the buffer."""
        while True:
            try:
                obj, rest = decode_packet(bytes(self._rx_buf))
                self._rx_buf = bytearray(rest)
                self._dispatch_frame(obj)
            except ValueError:
                break

    def _dispatch_frame(self, obj: Dict[str, Any]) -> None:
        frame_type = obj.get("type", "")
        if frame_type == "response":
            rid = obj.get("id", "")
            fut = self._pending.pop(rid, None)
            if fut is None or fut.done():
                return
            if obj.get("ok") is True:
                fut.set_result(obj.get("result"))
            else:
                err = obj.get("error") or {}
                fut.set_exception(
                    RpcError(
                        str(err.get("code", "rpc_error")),
                        str(err.get("message", "RPC failed")),
                        dict(err.get("details") or {}),
                    )
                )
        elif frame_type == "event":
            event_type = str(obj.get("event", ""))
            payload = dict(obj.get("payload") or {})
            frame_ts = obj.get("ts")
            if frame_ts is not None:
                payload["_mono_ts"] = float(frame_ts)
            for handler in self._event_handlers:
                asyncio.create_task(self._safe_handle_event(handler, event_type, payload))

    async def _safe_handle_event(self, handler: EventHandler, event_type: str, payload: Dict[str, Any]) -> None:
        try:
            await handler(event_type, payload)
        except Exception:
            logger.error("Event handler error for %s", event_type, exc_info=True)

    def _handle_disconnect(self) -> None:
        self._disconnected = True
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
        self._writer = None
        self._reader = None
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("bridge disconnected"))
        self._pending.clear()
        self._rx_buf.clear()


def fire_and_forget(coro: Coroutine) -> Optional[asyncio.Task]:
    """Schedule a coroutine as a fire-and-forget task with error suppression.

    Prevents "Task exception was never retrieved" warnings by catching and
    logging exceptions from unawaited tasks (e.g. best-effort RPC calls).
    """

    async def _wrapper() -> None:
        try:
            await coro
        except Exception as e:
            logger.debug("Fire-and-forget failed: %s", e)

    try:
        return asyncio.get_running_loop().create_task(_wrapper())
    except RuntimeError:
        return None
