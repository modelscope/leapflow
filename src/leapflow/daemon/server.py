"""Unix socket JSON-RPC server for leapd."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Any

from leapflow.daemon.lease import default_lease_ttl_s, read_active_client_leases
from leapflow.daemon.lifecycle import cleanup_run_dir, write_pid_file
from leapflow.daemon.protocol import ErrorCode, METHOD_REGISTRY, RpcRequest, RpcResponse, StreamChunk

logger = logging.getLogger(__name__)

_DEFAULT_STREAM_HEARTBEAT_S = 10.0
_DEFAULT_IDLE_TIMEOUT_S = 600.0


def _stream_heartbeat_interval() -> float:
    raw = os.getenv("LEAPFLOW_DAEMON_STREAM_HEARTBEAT", str(_DEFAULT_STREAM_HEARTBEAT_S)).strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return _DEFAULT_STREAM_HEARTBEAT_S


def _daemon_idle_timeout() -> float:
    raw = os.getenv("LEAPFLOW_DAEMON_IDLE_TIMEOUT_S", str(_DEFAULT_IDLE_TIMEOUT_S)).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_IDLE_TIMEOUT_S


class UnixRpcServer:
    """Newline-delimited JSON-RPC server bound to one Unix socket."""

    def __init__(
        self,
        service: Any,
        *,
        sock_path: Path,
        run_dir: Path,
        stream_heartbeat_s: float | None = None,
        on_shutdown: Callable[[], None] | None = None,
    ) -> None:
        self._service = service
        self._sock_path = sock_path
        self._run_dir = run_dir
        self._stream_heartbeat_s = stream_heartbeat_s or _stream_heartbeat_interval()
        self._on_shutdown = on_shutdown
        self._server: asyncio.AbstractServer | None = None
        self._active_connections = 0
        if hasattr(service, "set_client_count_provider"):
            service.set_client_count_provider(lambda: self._active_connections)
        if hasattr(service, "set_client_lease_provider"):
            service.set_client_lease_provider(lambda: read_active_client_leases(self._run_dir))

    @property
    def run_dir(self) -> Path:
        """Return the daemon runtime directory."""
        return self._run_dir

    @property
    def active_connections(self) -> int:
        """Return the current number of connected clients."""
        return self._active_connections

    async def serve_forever(self) -> None:
        """Start listening and serve until cancelled."""
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._sock_path.unlink(missing_ok=True)
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self._sock_path),
        )
        write_pid_file(self._run_dir)
        try:
            async with self._server:
                await self._server.serve_forever()
        except asyncio.CancelledError:
            raise
        finally:
            self._sock_path.unlink(missing_ok=True)

    async def stop(self) -> None:
        """Stop accepting clients and close the listening socket."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        self._sock_path.unlink(missing_ok=True)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._active_connections += 1
        try:
            while not reader.at_eof():
                raw = await reader.readline()
                if not raw:
                    break
                await self._handle_line(raw, writer)
        finally:
            self._active_connections -= 1
            await _close_writer(writer)

    async def _handle_line(self, raw: bytes, writer: asyncio.StreamWriter) -> None:
        try:
            payload = json.loads(raw.decode("utf-8"))
            request = RpcRequest.from_dict(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            response = RpcResponse.fail(
                "",
                ErrorCode.PARSE_ERROR,
                "Invalid JSON-RPC request",
                data=str(exc),
            )
            await _write_json(writer, response.to_json())
            return

        try:
            await self._dispatch(request, writer)
        except NotImplementedError as exc:
            response = RpcResponse.fail(
                request.id,
                ErrorCode.METHOD_NOT_FOUND,
                str(exc),
            )
            await _write_json(writer, response.to_json())
        except Exception as exc:
            logger.exception("daemon: request failed method=%s", request.method)
            response = RpcResponse.fail(
                request.id,
                ErrorCode.INTERNAL_ERROR,
                "Daemon request failed",
                data=str(exc),
            )
            await _write_json(writer, response.to_json())

    async def _dispatch(self, request: RpcRequest, writer: asyncio.StreamWriter) -> None:
        attr = METHOD_REGISTRY.get(request.method)
        if attr is None:
            response = RpcResponse.fail(
                request.id,
                ErrorCode.METHOD_NOT_FOUND,
                f"Unknown method: {request.method}",
            )
            await _write_json(writer, response.to_json())
            return

        method = getattr(self._service, attr)
        params = dict(request.params or {})
        if request.method == "engine.chat":
            await self._dispatch_stream(request, method, params, writer)
            return

        result = method(**params)
        if hasattr(result, "__await__"):
            result = await result
        response = RpcResponse.success(request.id, result)
        await _write_json(writer, response.to_json())
        if request.method == "daemon.shutdown" and self._on_shutdown is not None:
            self._on_shutdown()

    async def _dispatch_stream(
        self,
        request: RpcRequest,
        method: Callable[..., Any],
        params: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        stream = None
        pending: asyncio.Task | None = None
        try:
            stream = method(**params)
            pending = asyncio.create_task(anext(stream))
            while True:
                done, _ = await asyncio.wait({pending}, timeout=self._stream_heartbeat_s)
                if not done:
                    await self._write_stream_heartbeat(request.id, writer)
                    continue
                try:
                    chunk = pending.result()
                except StopAsyncIteration:
                    pending = None
                    break
                notification = StreamChunk(
                    request_id=request.id,
                    content=chunk.content,
                    done=chunk.done,
                    event_type=chunk.event_type,
                    metadata=chunk.metadata,
                ).to_notification()
                await _write_json(writer, notification.to_json())
                pending = asyncio.create_task(anext(stream))
        except Exception as exc:
            if pending is not None and not pending.done():
                pending.cancel()
            if stream is not None and hasattr(stream, "aclose"):
                try:
                    await stream.aclose()
                except Exception:
                    logger.debug("daemon: failed to close stream after error", exc_info=True)
            logger.exception("daemon: stream failed method=%s", request.method)
            response = RpcResponse.fail(
                request.id,
                ErrorCode.INTERNAL_ERROR,
                "Daemon stream failed",
                data=str(exc),
            )
            await _write_json(writer, response.to_json())
            return

        done = StreamChunk(request_id=request.id, content="", done=True).to_notification()
        await _write_json(writer, done.to_json())
        response = RpcResponse.success(request.id, {"ok": True})
        await _write_json(writer, response.to_json())

    async def _write_stream_heartbeat(
        self,
        request_id: str,
        writer: asyncio.StreamWriter,
    ) -> None:
        notification = StreamChunk(
            request_id=request_id,
            content="Still working...",
            event_type="status",
            metadata={"heartbeat": True},
        ).to_notification()
        await _write_json(writer, notification.to_json())


async def serve_daemon(settings: Any, *, mock_host: bool = False) -> int:
    """Run a daemon server for the provided settings until signalled."""
    from leapflow.daemon.service import RuntimeLeapService

    run_dir = settings.profile_dir / "run"
    sock_path = run_dir / "leapd.sock"
    service = RuntimeLeapService(settings, mock_host=mock_host)
    await service.start()
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_stop() -> None:
        stop_event.set()

    server = UnixRpcServer(
        service,
        sock_path=sock_path,
        run_dir=run_dir,
        on_shutdown=_request_stop,
    )

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            logger.debug("daemon: signal handlers unsupported on this event loop")

    task = asyncio.create_task(server.serve_forever())
    idle_task = asyncio.create_task(
        _watch_idle_shutdown(
            server,
            stop_event,
            idle_timeout_s=_daemon_idle_timeout(),
        )
    )
    try:
        await stop_event.wait()
    finally:
        idle_task.cancel()
        try:
            await idle_task
        except asyncio.CancelledError:
            pass
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await server.stop()
        await service.shutdown()
        cleanup_run_dir(run_dir)
    return 0


async def _watch_idle_shutdown(
    server: UnixRpcServer,
    stop_event: asyncio.Event,
    *,
    idle_timeout_s: float,
    lease_ttl_s: float | None = None,
    poll_interval_s: float | None = None,
) -> None:
    if idle_timeout_s <= 0:
        return
    last_active = asyncio.get_running_loop().time()
    interval = poll_interval_s or min(30.0, max(1.0, idle_timeout_s / 10.0))
    max_lease_age = default_lease_ttl_s() if lease_ttl_s is None else lease_ttl_s
    while not stop_event.is_set():
        has_lease = await asyncio.to_thread(
            read_active_client_leases,
            server.run_dir,
            ttl_s=max_lease_age,
        )
        if server.active_connections > 0 or has_lease:
            last_active = asyncio.get_running_loop().time()
        elif asyncio.get_running_loop().time() - last_active >= idle_timeout_s:
            logger.info("daemon: idle timeout reached; shutting down")
            stop_event.set()
            return
        await asyncio.sleep(interval)


async def _write_json(writer: asyncio.StreamWriter, text: str) -> None:
    writer.write(text.encode("utf-8") + b"\n")
    await writer.drain()


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except (BrokenPipeError, ConnectionError, OSError):
        logger.debug("daemon: client connection closed with transport error", exc_info=True)
