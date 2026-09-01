"""Unix socket JSON-RPC server for leapd."""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Any

from leapflow.daemon._transport import get_transport
from leapflow.daemon.lease import default_lease_ttl_s, read_active_client_leases
from leapflow.daemon.lifecycle import cleanup_runtime_dir, write_pid_file
from leapflow.daemon.protocol import ErrorCode, METHOD_REGISTRY, RpcRequest, RpcResponse, StreamChunk

logger = logging.getLogger(__name__)

_DEFAULT_STREAM_HEARTBEAT_S = 10.0
_DEFAULT_IDLE_TIMEOUT_S = 600.0

_APPROVAL_ROUTED_METHODS = frozenset({
    "command.execute",
    # A device observation that discloses the surroundings needs consent, and consent
    # needs somewhere to ask. Without a route the coordinator denies immediately
    # (``request_approval`` returns "deny" when ``route is None``), which is fail-closed
    # but leaves a caller that *can* present a prompt -- the board -- unable to obtain
    # one. The prompt travels as an interleaved ``stream.chunk`` notification on this
    # request's own socket, which ``DaemonClient.request(on_stream_event=...)`` already
    # forwards to whoever asked.
    "hardware.frame",
    "hardware.read",
})
"""RPCs that may raise an approval prompt, and therefore get a route installed.

Deliberately a small allow-list rather than "every awaitable method". A route makes the
handler wait for a human, so a method that acquires one must be a call somebody is
watching. The lifecycle -- register, deny-on-exit, unregister -- is identical for all of
them, so a routed request cannot leak a pending approval when its caller disconnects.
"""


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
        runtime_dir: Path,
        stream_heartbeat_s: float | None = None,
        on_shutdown: Callable[[], None] | None = None,
    ) -> None:
        self._service = service
        self._sock_path = sock_path
        self._runtime_dir = runtime_dir
        self._stream_heartbeat_s = stream_heartbeat_s or _stream_heartbeat_interval()
        self._on_shutdown = on_shutdown
        self._server: asyncio.AbstractServer | None = None
        self._active_connections = 0
        if hasattr(service, "set_client_count_provider"):
            service.set_client_count_provider(lambda: self._active_connections)
        if hasattr(service, "set_client_lease_provider"):
            service.set_client_lease_provider(lambda: read_active_client_leases(self._runtime_dir))

    @property
    def runtime_dir(self) -> Path:
        """Return the daemon runtime directory."""
        return self._runtime_dir

    @property
    def active_connections(self) -> int:
        """Return the current number of connected clients."""
        return self._active_connections

    def has_keepalive_work(self) -> bool:
        """Return True when the service has work that must keep the daemon alive.

        Persistent monitor watches must survive client disconnects, so an armed
        watch counts as activity for the idle-shutdown watchdog.
        """
        checker = getattr(self._service, "has_active_watches", None)
        if not callable(checker):
            return False
        try:
            return bool(checker())
        except Exception:
            return False

    async def serve_forever(self) -> None:
        """Start listening and serve until cancelled."""
        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        transport = get_transport()
        transport.cleanup(self._runtime_dir)
        self._server = await transport.start_server(
            self._handle_client,
            self._runtime_dir,
        )
        write_pid_file(self._runtime_dir)
        try:
            async with self._server:
                await self._server.serve_forever()
        except asyncio.CancelledError:
            raise
        finally:
            transport.cleanup(self._runtime_dir)

    async def stop(self) -> None:
        """Stop accepting clients and close the listening socket."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        get_transport().cleanup(self._runtime_dir)

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
            params.setdefault("request_id", request.id)
        if request.method in ("engine.chat", "events.subscribe"):
            await self._dispatch_stream(request, method, params, writer)
            return

        result = method(**params)
        if hasattr(result, "__await__"):
            approval_queue: asyncio.Queue[StreamChunk] | None = None
            route_token: contextvars.Token[Any] | None = None
            if request.method in _APPROVAL_ROUTED_METHODS:
                from leapflow.daemon.approval_route import approval_route as _approval_route

                approval_queue = asyncio.Queue()
                route_token = _approval_route.set((approval_queue, request.id))
                try:
                    self._service._approval_coordinator.register_route(request.id)
                except AttributeError:
                    pass
            try:
                result = await self._await_with_heartbeat(
                    request.id,
                    result,
                    writer,
                    approval_queue=approval_queue,
                )
            finally:
                if route_token is not None:
                    _approval_route.reset(route_token)
                    try:
                        self._service._approval_coordinator.unregister_route(request.id)
                        self._service._approval_coordinator.deny_for_request(request.id, reason="command_ended")
                    except AttributeError:
                        pass
        response = RpcResponse.success(request.id, result)
        await _write_json(writer, response.to_json())
        if request.method == "daemon.shutdown" and self._on_shutdown is not None:
            self._on_shutdown()

    async def _await_with_heartbeat(
        self,
        request_id: str,
        coro: Any,
        writer: asyncio.StreamWriter,
        *,
        approval_queue: "asyncio.Queue[StreamChunk] | None" = None,
    ) -> Any:
        """Await a coroutine while sending periodic heartbeats to keep the client alive.

        Long-running RPC handlers (e.g. /plugin generate with LLM calls) can
        exceed the client read timeout. Sending heartbeat notifications at a
        regular interval resets the client's per-read deadline, preventing
        spurious timeout disconnections.
        """
        task = asyncio.ensure_future(coro)
        approval_get: asyncio.Task[Any] | None = None
        while True:
            wait_set: set[asyncio.Task[Any]] = {task}
            if approval_queue is not None:
                if approval_get is None or approval_get.done():
                    approval_get = asyncio.create_task(approval_queue.get())
                wait_set.add(approval_get)
            done, _ = await asyncio.wait(
                wait_set,
                timeout=self._stream_heartbeat_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if approval_get is not None and approval_get in done:
                chunk = approval_get.result()
                approval_get = None
                try:
                    await _write_json(writer, chunk.to_notification().to_json())
                except (ConnectionResetError, BrokenPipeError):
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                    raise
                continue
            if task in done:
                if approval_get is not None and not approval_get.done():
                    approval_get.cancel()
                return task.result()
            # Send heartbeat to keep client connection alive
            heartbeat = StreamChunk(
                request_id=request_id,
                content="",
                event_type="status",
                metadata={"heartbeat": True},
            ).to_notification()
            try:
                await _write_json(writer, heartbeat.to_json())
            except (ConnectionResetError, BrokenPipeError):
                # Client disconnected; cancel the handler and re-raise
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                raise

    async def _dispatch_stream(
        self,
        request: RpcRequest,
        method: Callable[..., Any],
        params: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        stream = None
        pending: asyncio.Task | None = None
        # Pin every per-chunk task driving this one stream to a single, shared
        # Context. asyncio.create_task() defaults to copying the *current*
        # context on each call, so re-creating ``pending`` per chunk (needed
        # below to interleave heartbeats via asyncio.wait(..., timeout=...))
        # would otherwise hand the streamed generator a fresh Context every
        # time it resumes. A ContextVar.set()/reset() pair inside that
        # generator (e.g. the daemon's per-turn approval routing) binds its
        # token to the Context active at set()-time; resetting it from a
        # different Context object raises "was created in a different
        # Context". Sharing one Context across all chunks keeps such
        # set()/reset() pairs valid for the whole life of the stream.
        ctx = contextvars.copy_context()
        try:
            stream = method(**params)
            pending = asyncio.create_task(anext(stream), context=ctx)
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
                pending = asyncio.create_task(anext(stream), context=ctx)
        except (ConnectionResetError, BrokenPipeError) as exc:
            # Client vanished mid-stream (e.g. TUI closed while a heartbeat
            # or chunk write was in flight). This is routine churn — log a
            # single debug line, no traceback, and skip the error response
            # since the pipe is already gone.
            if pending is not None and not pending.done():
                pending.cancel()
            if stream is not None and hasattr(stream, "aclose"):
                try:
                    await stream.aclose()
                except Exception:
                    logger.debug("daemon: failed to close stream after disconnect", exc_info=True)
            logger.debug(
                "daemon: client disconnected during stream method=%s (%s)",
                request.method, type(exc).__name__,
            )
            return
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

    runtime_dir = settings.runtime_dir
    sock_path = get_transport().readiness_path(runtime_dir)
    service = RuntimeLeapService(settings, mock_host=mock_host, auto_start_deferred=False)
    await service.start()
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_stop() -> None:
        stop_event.set()

    server = UnixRpcServer(
        service,
        sock_path=sock_path,
        runtime_dir=runtime_dir,
        on_shutdown=_request_stop,
    )

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            logger.debug("daemon: signal handlers unsupported on this event loop")

    task = asyncio.create_task(server.serve_forever())
    await _wait_server_listening(server, task)
    service.start_deferred_init()
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
        cleanup_runtime_dir(runtime_dir)
    return 0


async def _wait_server_listening(server: UnixRpcServer, task: asyncio.Task[None]) -> None:
    """Wait until the RPC server has bound its transport before background init."""
    while getattr(server, "_server", None) is None:
        if task.done():
            await task
        await asyncio.sleep(0.01)


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
            server.runtime_dir,
            ttl_s=max_lease_age,
        )
        keepalive = getattr(server, "has_keepalive_work", None)
        if server.active_connections > 0 or has_lease or (callable(keepalive) and keepalive()):
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
