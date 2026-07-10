"""Thin client for connecting LeapFlow CLI processes to leapd."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from leapflow.daemon.lifecycle import (
    DaemonInfo,
    DaemonLock,
    cleanup_stale,
    spawn_daemon,
    wait_ready,
)
from leapflow.daemon.protocol import RpcRequest
from leapflow.engine import StreamEvent

logger = logging.getLogger(__name__)

StatusCallback = Callable[[str], None]


class DaemonUnavailableError(RuntimeError):
    """Raised when a usable leapd daemon cannot be reached."""


class DaemonClient:
    """Small JSON-RPC client that opens one Unix socket per request."""

    def __init__(self, sock_path: Path, *, timeout_s: float = 30.0) -> None:
        self._sock_path = sock_path
        self._timeout_s = timeout_s

    @property
    def sock_path(self) -> Path:
        """Return the Unix socket path used by this client."""
        return self._sock_path

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send one non-streaming JSON-RPC request and return its result."""
        request = RpcRequest(method=method, params=params or {})
        reader, writer = await self._open()
        try:
            await _send(writer, request.to_json())
            while True:
                payload = await self._read_payload(reader)
                if payload.get("id") != request.id:
                    continue
                if "error" in payload:
                    raise DaemonUnavailableError(_format_rpc_error(payload["error"]))
                return payload.get("result")
        finally:
            await _close_writer(writer)

    async def engine_chat(
        self,
        message: str,
        *,
        enable_thinking: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        """Stream chat events from the daemon-owned AgentEngine."""
        request = RpcRequest(
            method="engine.chat",
            params={"message": message, "enable_thinking": enable_thinking},
        )
        reader, writer = await self._open()
        try:
            await _send(writer, request.to_json())
            while True:
                payload = await self._read_payload(reader)
                method = payload.get("method")
                params = dict(payload.get("params") or {})
                if method == "stream.chunk" and params.get("id") == request.id:
                    if params.get("done"):
                        continue
                    yield _event_from_params(params)
                    continue
                if payload.get("id") == request.id:
                    if "error" in payload:
                        raise DaemonUnavailableError(_format_rpc_error(payload["error"]))
                    break
        finally:
            await _close_writer(writer)

    async def session_resume(self, session_id: str) -> dict[str, Any]:
        """Ask the daemon to load an existing conversation session."""
        result = await self.request("session.resume", {"session_id": session_id})
        return dict(result or {})

    async def status(self) -> dict[str, Any]:
        """Return daemon status."""
        result = await self.request("daemon.status")
        return dict(result or {})

    async def approval_status(self) -> dict[str, Any]:
        """Return pending daemon approval requests."""
        result = await self.request("approval.status")
        return dict(result or {})

    async def approval_resolve(
        self,
        pending_id: str,
        decision: str,
        *,
        reason: str = "",
    ) -> dict[str, Any]:
        """Resolve a pending daemon approval request."""
        result = await self.request(
            "approval.resolve",
            {"pending_id": pending_id, "decision": decision, "reason": reason},
        )
        return dict(result or {})

    async def approval_cancel(self, pending_id: str, *, reason: str = "cancelled") -> dict[str, Any]:
        """Cancel a pending daemon approval request."""
        result = await self.request(
            "approval.cancel",
            {"pending_id": pending_id, "reason": reason},
        )
        return dict(result or {})

    async def shutdown(self) -> None:
        """Request graceful daemon shutdown."""
        await self.request("daemon.shutdown")

    async def _open(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        try:
            return await asyncio.wait_for(
                asyncio.open_unix_connection(str(self._sock_path)),
                timeout=self._timeout_s,
            )
        except (TimeoutError, OSError) as exc:
            raise DaemonUnavailableError(
                f"Cannot connect to leapd at {self._sock_path}: {exc}"
            ) from exc

    async def _read_payload(self, reader: asyncio.StreamReader) -> dict[str, Any]:
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=self._timeout_s)
        except TimeoutError as exc:
            raise DaemonUnavailableError("Timed out waiting for leapd response") from exc
        if not raw:
            raise DaemonUnavailableError("leapd closed the connection unexpectedly")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DaemonUnavailableError("Received invalid JSON from leapd") from exc
        if not isinstance(payload, dict):
            raise DaemonUnavailableError("Received invalid JSON-RPC payload from leapd")
        return payload


async def ensure_daemon_client(
    settings: Any,
    *,
    mock_host: bool = False,
    status_callback: StatusCallback | None = None,
) -> DaemonClient:
    """Return a client connected to a healthy daemon, starting one if needed."""
    run_dir = settings.profile_dir / "run"
    sock_path = run_dir / "leapd.sock"
    info = DaemonInfo.discover(run_dir)
    if info.is_healthy:
        _emit(status_callback, f"Connected to leapd (pid={info.pid}).")
        return DaemonClient(sock_path)

    if info.pid is not None and not info.is_running:
        cleanup_stale(run_dir)
    elif info.is_running and not info.is_healthy:
        raise DaemonUnavailableError(
            f"leapd is running but unhealthy (pid={info.pid}). "
            "Run 'leap daemon stop' and retry."
        )

    lock = DaemonLock(run_dir / "leapd.lock")
    if lock.acquire():
        try:
            _emit(status_callback, "Starting leapd daemon...")
            spawn_daemon(settings, mock_host=mock_host)
        finally:
            lock.release()
    else:
        _emit(status_callback, "Waiting for leapd daemon...")

    ready = wait_ready(run_dir, timeout_s=_daemon_start_timeout())
    if not ready.is_healthy:
        raise DaemonUnavailableError(
            "leapd did not become ready. Run 'leap daemon status' for details."
        )
    _emit(status_callback, f"Connected to leapd (pid={ready.pid}).")
    return DaemonClient(sock_path)


def _event_from_params(params: dict[str, Any]) -> StreamEvent:
    event_type = str(params.get("event_type") or "chunk")
    if event_type not in {
        "chunk",
        "final",
        "tool_start",
        "tool_complete",
        "thinking",
        "status",
        "error",
        "approval_request",
        "approval_response",
    }:
        event_type = "chunk"
    metadata = params.get("metadata")
    return StreamEvent(
        type=event_type,  # type: ignore[arg-type]
        content=str(params.get("content") or ""),
        metadata=dict(metadata) if isinstance(metadata, dict) else None,
    )


def _format_rpc_error(error: object) -> str:
    if isinstance(error, dict):
        message = str(error.get("message") or "Daemon request failed")
        data = error.get("data")
        return f"{message}: {data}" if data else message
    return str(error)


def _emit(callback: StatusCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _daemon_start_timeout() -> float:
    raw = os.getenv("LEAPFLOW_DAEMON_START_TIMEOUT", "30").strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 30.0


async def _send(writer: asyncio.StreamWriter, text: str) -> None:
    writer.write(text.encode("utf-8") + b"\n")
    await writer.drain()


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except (BrokenPipeError, ConnectionError, OSError):
        logger.debug("daemon client: socket closed with transport error", exc_info=True)
