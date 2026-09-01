"""Thin client for connecting LeapFlow CLI processes to leapd."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from leapflow.daemon._transport import get_transport
from leapflow.daemon.lifecycle import (
    DaemonInfo,
    DaemonLock,
    cleanup_stale,
    spawn_daemon,
    stop_daemon,
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

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        on_stream_event: Callable[[StreamEvent], Any] | None = None,
    ) -> Any:
        """Send one non-streaming JSON-RPC request and return its result.

        Handles server-sent heartbeat notifications transparently: each
        heartbeat resets the read timeout, keeping the connection alive for
        long-running handlers without raising ``DaemonUnavailableError``.
        """
        request = RpcRequest(method=method, params=params or {})
        reader, writer = await self._open()
        try:
            await _send(writer, request.to_json())
            while True:
                payload = await self._read_payload(reader)
                # Skip heartbeat notifications sent by the server for
                # long-running handlers — they carry no result.
                if payload.get("method") == "stream.chunk":
                    p = payload.get("params") or {}
                    if isinstance(p.get("metadata"), dict) and p["metadata"].get("heartbeat"):
                        continue
                    if p.get("id") == request.id and on_stream_event is not None:
                        event = _event_from_params(dict(p))
                        result = on_stream_event(event)
                        if hasattr(result, "__await__"):
                            await result
                        continue
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
        session_id: str = "",
        workspace_root: str = "",
    ) -> AsyncIterator[StreamEvent]:
        """Stream chat events from the daemon-owned AgentEngine.

        ``session_id`` routes the turn to that session's engine, so distinct
        sessions (e.g. two TUI clients) run concurrently and isolated when the
        daemon admits concurrency (daemon.max_concurrent_turns > 1). Empty means
        the daemon's current session (single-session behavior unchanged).
        ``workspace_root`` is the client process' active workspace/cwd. It is
        routed to the daemon so each TUI session gets its own project context
        instead of inheriting the shared daemon process cwd.
        """
        params: dict[str, Any] = {"message": message, "enable_thinking": enable_thinking}
        if session_id:
            params["session_id"] = session_id
        if workspace_root:
            params["workspace_root"] = workspace_root
        request = RpcRequest(
            method="engine.chat",
            params=params,
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

    async def subscribe_notifications(self) -> AsyncIterator[dict[str, Any]]:
        """Long-lived subscription stream for daemon push notifications.

        Yields notification dicts with keys: event_type, payload, timestamp.
        Runs until the server closes the stream or connection is lost.
        """
        request = RpcRequest(method="events.subscribe", params={})
        reader, writer = await self._open()
        try:
            await _send(writer, request.to_json())
            while True:
                payload = await self._read_payload(reader)
                method = payload.get("method")
                params = dict(payload.get("params") or {})
                if method == "stream.chunk" and params.get("id") == request.id:
                    if params.get("done"):
                        break
                    metadata = params.get("metadata") or {}
                    if metadata.get("event_type"):
                        yield metadata
                    continue
                if payload.get("id") == request.id:
                    break
        finally:
            await _close_writer(writer)

    async def engine_cancel(self, request_id: str = "") -> bool:
        """Request cancellation of the daemon-owned active engine turn.

        With ``request_id`` the daemon targets that specific turn; without one it
        cancels the active turn(s) (at N=1 the single running turn).
        """
        result = await self.request("engine.cancel", {"request_id": request_id} if request_id else None)
        return bool(result)

    async def session_resume(self, session_id: str) -> dict[str, Any]:
        """Ask the daemon to load an existing conversation session."""
        result = await self.request("session.resume", {"session_id": session_id})
        return dict(result or {})

    async def status(self, session_id: str = "") -> dict[str, Any]:
        """Return daemon status, scoped to ``session_id`` when the caller has one.

        Passing the caller's own session is what makes the reply describe *this*
        client: without it the daemon has no way to know which of several live
        sessions to report, and any session identity it returned would belong to
        somebody else.

        ``daemon.status`` is read-only and idempotent, so it tolerates the short
        startup/restart window where a socket is not yet accepting, or accepts
        and closes before the control-plane request is handled.
        """
        params = {"session_id": session_id} if session_id else {}
        last_error: DaemonUnavailableError | None = None
        retry_budget_s = 30.0 if self._timeout_s > 60.0 else 5.0
        deadline = asyncio.get_running_loop().time() + min(max(self._timeout_s, 1.0), retry_budget_s)
        attempt = 0
        while True:
            try:
                result = await self.request("daemon.status", params)
                return dict(result or {})
            except DaemonUnavailableError as exc:
                last_error = exc
                message = str(exc)
                transient_startup = (
                    "closed the connection unexpectedly" in message
                    or "Cannot connect to leapd" in message
                )
                if not transient_startup or asyncio.get_running_loop().time() >= deadline:
                    raise
                attempt += 1
                await asyncio.sleep(min(0.5, 0.05 * attempt))
        assert last_error is not None
        raise last_error

    async def host_status(self) -> dict[str, Any]:
        """Return daemon-owned host backend status."""
        result = await self.request("host.status")
        return dict(result or {})

    async def host_start(self) -> dict[str, Any]:
        """Start the daemon-owned host backend."""
        result = await self.request("host.start")
        return dict(result or {})

    async def host_stop(self) -> dict[str, Any]:
        """Stop the daemon-owned host backend."""
        result = await self.request("host.stop")
        return dict(result or {})

    async def host_restart(self) -> dict[str, Any]:
        """Restart the daemon-owned host backend."""
        result = await self.request("host.restart")
        return dict(result or {})

    async def hardware_pause(self, device: str) -> dict[str, Any]:
        """Pause the daemon-owned hardware sampling loop for one device."""
        result = await self.request("hardware.pause", {"device": device})
        return dict(result or {})

    async def hardware_resume(self, device: str) -> dict[str, Any]:
        """Resume the daemon-owned hardware sampling loop for one device."""
        result = await self.request("hardware.resume", {"device": device})
        return dict(result or {})

    async def tools_list(self) -> dict[str, Any]:
        """Return daemon-owned tool summary for slash-command rendering."""
        result = await self.request("tools.list")
        return dict(result or {})

    async def usage_summary(self) -> dict[str, Any]:
        """Return token usage for the daemon-owned session."""
        result = await self.request("usage.summary")
        return dict(result or {})

    async def app_command(self, args: str = "") -> dict[str, Any]:
        """Return daemon-owned App Connector command payload."""
        result = await self.request("app.command", {"args": args})
        return dict(result or {})

    async def command_execute(
        self,
        name: str,
        args: str = "",
        session_id: str = "",
        *,
        on_stream_event: Callable[[StreamEvent], Any] | None = None,
    ) -> dict[str, Any]:
        """Execute any engine-routed slash command via daemon.

        ``session_id`` tells the daemon which client session the command belongs
        to, so session-scoped commands (e.g. ``/board``) observe the caller's
        conversation instead of whichever session was last active.
        """
        result = await self.request(
            "command.execute",
            {"name": name, "args": args, "session_id": session_id},
            on_stream_event=on_stream_event,
        )
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

    # ── Watch runtime (monitor subsystem) ──

    async def watch_arm(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Arm a monitor watch on the daemon. Returns its runtime view."""
        return dict(await self.request("watch.arm", {"spec": spec}) or {})

    async def watch_list(self) -> list[dict[str, Any]]:
        """List daemon-hosted watches."""
        return list(await self.request("watch.list") or [])

    async def watch_get(self, watch_id: str) -> dict[str, Any]:
        """Return a single watch view."""
        return dict(await self.request("watch.get", {"watch_id": watch_id}) or {})

    async def watch_pause(self, watch_id: str) -> dict[str, Any]:
        """Suspend a watch until resumed."""
        return dict(await self.request("watch.pause", {"watch_id": watch_id}) or {})

    async def watch_resume(self, watch_id: str) -> dict[str, Any]:
        """Re-arm a suspended watch."""
        return dict(await self.request("watch.resume", {"watch_id": watch_id}) or {})

    async def watch_stop(self, watch_id: str) -> dict[str, Any]:
        """Terminally stop a watch."""
        return dict(await self.request("watch.stop", {"watch_id": watch_id}) or {})

    async def watch_mute(self, watch_id: str, *, muted: bool = True) -> dict[str, Any]:
        """Toggle whether a watch's findings are pushed."""
        return dict(await self.request("watch.mute", {"watch_id": watch_id, "muted": muted}) or {})

    async def watch_refresh(self, watch_id: str) -> dict[str, Any]:
        """Run one observation cycle immediately."""
        return dict(await self.request("watch.refresh", {"watch_id": watch_id}) or {})

    async def watch_findings(
        self, *, watch_id: str = "", limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Return persisted findings, newest first."""
        return list(await self.request(
            "watch.findings", {"watch_id": watch_id, "limit": limit, "offset": offset}
        ) or [])

    async def session_history(self, *, limit: int = 200, session_id: str = "") -> dict[str, Any]:
        """Return a session's transcript and counts (empty id = current session)."""
        return dict(await self.request(
            "session.history", {"limit": limit, "session_id": session_id},
        ) or {})

    async def session_detail(
        self,
        session_id: str,
        *,
        limit: int = 200,
        offset: int = 0,
        include_inactive: bool = True,
        workspace_root: str = "",
    ) -> dict[str, Any]:
        """Return a persisted historical session transcript by id."""
        return dict(await self.request(
            "session.detail",
            {
                "session_id": session_id,
                "limit": limit,
                "offset": offset,
                "include_inactive": include_inactive,
                "workspace_root": workspace_root,
            },
        ) or {})

    async def session_analyze(self) -> dict[str, Any]:
        """Ensure a session-analysis watch and run one analysis cycle now."""
        return dict(await self.request("session.analyze") or {})

    async def signal_record(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Inject a signal event into the daemon's EventBus."""
        return dict(await self.request(
            "signal.record", {"signal_data": {"type": event_type, "payload": payload}},
        ) or {})

    async def monitor_signal_metrics(self) -> dict[str, Any]:
        """Fetch signal flow metrics from daemon."""
        return dict(await self.request("monitor.signal_metrics") or {})

    async def shutdown(self) -> None:
        """Request graceful daemon shutdown."""
        await self.request("daemon.shutdown")

    async def _open(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        try:
            # _timeout_s is the per-request read budget (heartbeat tests set it
            # to 0.1s); connection establishment needs its own, larger floor or
            # every RPC flakes under load when the handshake exceeds it.
            return await asyncio.wait_for(
                get_transport().connect(self._sock_path.parent),
                timeout=max(self._timeout_s, 5.0),
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
    runtime_dir = settings.runtime_dir
    sock_path = get_transport().readiness_path(runtime_dir)
    info = DaemonInfo.discover(runtime_dir)
    if info.is_healthy:
        _emit(status_callback, f"Connected to leapd (pid={info.pid}).")
        return DaemonClient(sock_path)

    if info.pid is not None and not info.is_running:
        cleanup_stale(runtime_dir)
    elif info.is_running and not info.is_healthy:
        raise DaemonUnavailableError(
            f"leapd is running but unhealthy (pid={info.pid}). "
            "Run 'leap daemon stop' and retry."
        )

    lock = DaemonLock(runtime_dir / "leapd.lock")
    if lock.acquire():
        try:
            _emit(status_callback, "Starting leapd daemon...")
            spawn_daemon(settings, mock_host=mock_host)
        finally:
            lock.release()
    else:
        _emit(status_callback, "Waiting for leapd daemon...")

    ready = wait_ready(runtime_dir, timeout_s=_daemon_start_timeout())
    if not ready.is_healthy:
        raise DaemonUnavailableError(
            "leapd did not become ready. Run 'leap daemon status' for details."
        )
    _emit(status_callback, f"Connected to leapd (pid={ready.pid}).")
    return DaemonClient(sock_path)


async def recover_daemon_client(
    settings: Any,
    *,
    mock_host: bool = False,
    status_callback: StatusCallback | None = None,
) -> DaemonClient:
    """Return a usable daemon client, restarting an unresponsive daemon once."""
    runtime_dir = settings.runtime_dir
    try:
        client = await ensure_daemon_client(
            settings,
            mock_host=mock_host,
            status_callback=status_callback,
        )
        await _probe_daemon_status(client, timeout_s=_daemon_recovery_probe_timeout())
        return client
    except DaemonUnavailableError as exc:
        info = DaemonInfo.discover(runtime_dir)
        if not info.is_running:
            raise
        _emit(status_callback, f"Restarting unresponsive leapd (pid={info.pid})...")
        result = await asyncio.to_thread(
            stop_daemon,
            runtime_dir,
            timeout_s=10.0,
            force=True,
            force_timeout_s=5.0,
            on_progress=status_callback,
        )
        if not result.stopped:
            raise exc
        client = await ensure_daemon_client(
            settings,
            mock_host=mock_host,
            status_callback=status_callback,
        )
        await _probe_daemon_status(client, timeout_s=_daemon_recovery_probe_timeout())
        return client


async def _probe_daemon_status(client: DaemonClient, *, timeout_s: float) -> None:
    """Verify that the daemon RPC loop responds, not just that its socket accepts."""
    await DaemonClient(client.sock_path, timeout_s=timeout_s).status()


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


def _daemon_recovery_probe_timeout() -> float:
    raw = os.getenv("LEAPFLOW_DAEMON_RECOVERY_PROBE_TIMEOUT", "3").strip()
    try:
        return max(0.5, float(raw))
    except ValueError:
        return 3.0


async def _send(writer: asyncio.StreamWriter, text: str) -> None:
    writer.write(text.encode("utf-8") + b"\n")
    await writer.drain()


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except (BrokenPipeError, ConnectionError, OSError):
        logger.debug("daemon client: socket closed with transport error", exc_info=True)
