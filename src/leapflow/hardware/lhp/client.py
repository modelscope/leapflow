# Copyright (c) Alibaba, Inc. and its affiliates.
"""LHP client bindings: the host side of the protocol.

Two bindings, one interface (``connect`` / ``request`` / ``close``):

- ``DirectLhpClient`` -- in-process binding to an ``LhpNodeServer`` object.
  Same process means same trust domain, so the token gate is skipped (the
  server still version-checks ``hello``). This is what the transport
  conformance suite and embedded placements use; it exercises the *full*
  codec and dispatch path without a socket.

- ``SocketLhpClient`` -- TCP or Unix-domain binding. The first frame on every
  connection must be ``hello`` carrying the shared token; the server answers
  ``unauthorized`` and closes otherwise. Requests are multiplexed by id with a
  single reader task, so a stalled read cannot queue behind itself -- which is
  what keeps ``probe`` and ``halt`` able to answer while another call hangs.

Neither binding retries a request. A retry after ``timeout`` or
``connection_closed`` could re-send a write whose effect is unknown, so
reconnection is deliberately deferred to the *next* call: a failed exchange
poisons nothing, and a later request reconnects lazily. Only the caller -- who
knows whether the operation was idempotent -- may decide to repeat it.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from contextlib import suppress
from typing import Any, Mapping
from urllib.parse import urlparse

from leapflow.hardware.lhp.protocol import (
    ERR_CONNECTION_CLOSED,
    ERR_PROTOCOL_MISMATCH,
    ERR_TIMEOUT,
    LHP_PROTOCOL_VERSION,
    MAX_FRAME_BYTES,
    LhpRemoteError,
    decode_frame,
    encode_frame,
    make_request,
    response_error,
)

logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT_S = 10.0
"""Per-request ceiling. A hung node must surface as a failed call, not a hung turn."""

DEFAULT_CONNECT_TIMEOUT_S = 5.0


class DirectLhpClient:
    """In-process binding: calls ``LhpNodeServer.handle_request`` directly.

    No sockets, no timeouts (an in-process await either completes or the task
    scheduling it is cancelled), and no token gate -- the same-process trust
    domain argument from the protocol spec. The hello exchange still runs, so
    version negotiation and the response shape are identical to the socket path.
    """

    kind = "direct"

    def __init__(self, server: Any) -> None:
        self._server = server
        self._ids = itertools.count(1)
        self._node_id = ""
        self._closed = False

    async def connect(self) -> Mapping[str, Any]:
        """Perform the hello exchange. Returns the node's hello payload."""
        hello = await self._serve(
            make_request(
                next(self._ids),
                "hello",
                {"protocol": LHP_PROTOCOL_VERSION, "token": "", "client": "direct"},
            )
        )
        if not isinstance(hello, Mapping):
            raise LhpRemoteError(ERR_PROTOCOL_MISMATCH, "malformed hello payload")
        remote_protocol = hello.get("protocol")
        if remote_protocol != LHP_PROTOCOL_VERSION:
            raise LhpRemoteError(
                ERR_PROTOCOL_MISMATCH,
                f"node speaks protocol {remote_protocol!r}, expected {LHP_PROTOCOL_VERSION}",
            )
        self._node_id = str(hello.get("node_id") or "")
        return hello

    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        """Send one request; raise ``LhpRemoteError`` when the node answers an error."""
        if self._closed:
            raise LhpRemoteError(ERR_CONNECTION_CLOSED, "client closed")
        return await self._serve(make_request(next(self._ids), method, params))

    async def close(self) -> None:
        """No-op: the server belongs to the process, not to this client."""
        self._closed = True

    async def _serve(self, request: Mapping[str, Any]) -> Any:
        response = await self._server.handle_request(dict(request), via="direct")
        error = response_error(response)
        if error is not None:
            raise LhpRemoteError(error[0], error[1])
        return response.get("result")


class SocketLhpClient:
    """TCP/Unix binding with hello-first authentication and id multiplexing."""

    kind = "socket"

    def __init__(
        self,
        endpoint: str,
        *,
        token: str = "",
        client_name: str = "leapflow",
        timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
    ) -> None:
        self._endpoint = str(endpoint)
        self._token = str(token or "")
        self._client_name = str(client_name or "leapflow")
        self._timeout_s = max(0.1, float(timeout_s or DEFAULT_REQUEST_TIMEOUT_S))
        self._connect_timeout_s = max(0.1, float(connect_timeout_s or DEFAULT_CONNECT_TIMEOUT_S))
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._ids = itertools.count(1)
        self._write_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._closed = False
        self._node_id = ""
        self._hello: Mapping[str, Any] = {}

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def connect(self) -> Mapping[str, Any]:
        """Ensure a live connection; return the node's hello payload.

        Same contract as ``DirectLhpClient.connect``: the answer of the hello
        exchange that established the current connection (a copy, so callers
        cannot mutate client state).
        """
        await self._ensure_connected()
        return dict(self._hello)

    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        """Send one request over the (lazily established) connection."""
        await self._ensure_connected()
        return await self._exchange(method, params)

    async def close(self) -> None:
        """Drop the connection and fail every in-flight request. Never raises.

        Takes ``_connect_lock`` so close cannot interleave with a connect in
        progress: closing while ``_open_transport`` is still awaiting would
        tear down nothing (``_writer`` is not assigned yet), the handshake
        would then complete, and the client would be left holding a live
        socket that every later call refuses to use -- a leak nothing could
        ever reach.
        """
        self._closed = True
        async with self._connect_lock:
            await self._teardown_locked(ERR_CONNECTION_CLOSED, "client closed")

    # ── Connection lifecycle ──

    async def _ensure_connected(self) -> None:
        if self._closed:
            raise LhpRemoteError(ERR_CONNECTION_CLOSED, "client closed")
        if self.connected:
            return
        async with self._connect_lock:
            # Re-check under the lock: a concurrent first call may have connected.
            if self.connected or self._closed:
                if self._closed:
                    raise LhpRemoteError(ERR_CONNECTION_CLOSED, "client closed")
                return
            await self._open_transport()
            if self._closed:
                # ``close()`` ran while the transport was being opened; the
                # check at method entry predates the flag. Without this the
                # handshake below would complete on a closed client and leave
                # the fresh socket live.
                await self._teardown_locked(ERR_CONNECTION_CLOSED, "client closed")
                raise LhpRemoteError(ERR_CONNECTION_CLOSED, "client closed")
            try:
                hello = await self._exchange(
                    "hello",
                    {
                        "protocol": LHP_PROTOCOL_VERSION,
                        "token": self._token,
                        "client": self._client_name,
                    },
                )
            except BaseException:
                # A failed handshake must not leave a half-open socket behind.
                await self._teardown_locked(ERR_CONNECTION_CLOSED, "handshake failed")
                raise
            self._node_id = str((hello or {}).get("node_id") or "")
            remote_protocol = (hello or {}).get("protocol")
            if remote_protocol != LHP_PROTOCOL_VERSION:
                await self._teardown_locked(
                    ERR_PROTOCOL_MISMATCH,
                    f"node speaks protocol {remote_protocol!r}",
                )
                raise LhpRemoteError(
                    ERR_PROTOCOL_MISMATCH,
                    f"node speaks protocol {remote_protocol!r}, expected {LHP_PROTOCOL_VERSION}",
                )
            self._hello = dict(hello) if isinstance(hello, Mapping) else {}

    async def _open_transport(self) -> None:
        parsed = urlparse(self._endpoint)
        try:
            if parsed.scheme == "tcp":
                if not parsed.hostname or not parsed.port:
                    raise ValueError(f"tcp endpoint needs host and port: {self._endpoint!r}")
                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_connection(
                        parsed.hostname, parsed.port, limit=MAX_FRAME_BYTES + 4096
                    ),
                    timeout=self._connect_timeout_s,
                )
            elif parsed.scheme == "unix":
                path = parsed.path or ""
                if not path:
                    raise ValueError(f"unix endpoint needs a path: {self._endpoint!r}")
                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_unix_connection(path, limit=MAX_FRAME_BYTES + 4096),
                    timeout=self._connect_timeout_s,
                )
            else:
                raise ValueError(
                    f"unsupported LHP endpoint {self._endpoint!r}; use tcp://host:port or unix:///path"
                )
        except asyncio.TimeoutError as exc:
            raise LhpRemoteError(
                ERR_CONNECTION_CLOSED, f"connect to {self._endpoint!r} timed out"
            ) from exc
        except OSError as exc:
            raise LhpRemoteError(
                ERR_CONNECTION_CLOSED, f"cannot connect to {self._endpoint!r}: {exc}"
            ) from exc
        self._reader_task = asyncio.get_running_loop().create_task(self._read_loop())

    async def _exchange(self, method: str, params: Mapping[str, Any] | None) -> Any:
        reader = self._reader
        writer = self._writer
        if reader is None or writer is None:
            raise LhpRemoteError(ERR_CONNECTION_CLOSED, "not connected")
        request_id = next(self._ids)
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            async with self._write_lock:
                writer.write(encode_frame(make_request(request_id, method, params)))
                await writer.drain()
        except (ConnectionError, OSError, RuntimeError) as exc:
            self._pending.pop(request_id, None)
            future.cancel()
            await self._teardown_locked(ERR_CONNECTION_CLOSED, f"send failed: {exc}")
            raise LhpRemoteError(ERR_CONNECTION_CLOSED, f"send failed: {exc}") from exc
        try:
            return await asyncio.wait_for(future, timeout=self._timeout_s)
        except asyncio.TimeoutError as exc:
            # The late response (if it ever arrives) carries this id, finds no
            # pending entry, and is dropped -- ids are never reused.
            self._pending.pop(request_id, None)
            raise LhpRemoteError(
                ERR_TIMEOUT, f"{method!r} timed out after {self._timeout_s:g}s"
            ) from exc
        finally:
            self._pending.pop(request_id, None)

    async def _read_loop(self) -> None:
        reader = self._reader
        assert reader is not None
        failure: tuple[str, str] = (ERR_CONNECTION_CLOSED, "connection closed")
        try:
            while True:
                try:
                    line = await reader.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    failure = (ERR_CONNECTION_CLOSED, "frame exceeded the protocol line limit")
                    break
                if not line:
                    break
                try:
                    message = decode_frame(line)
                except Exception as exc:  # noqa: BLE001 - a bad frame is dropped, not fatal for others
                    logger.warning("LHP client dropped an undecodable frame: %s", exc)
                    continue
                request_id = message.get("id")
                if not isinstance(request_id, int):
                    continue
                future = self._pending.pop(request_id, None)
                if future is None or future.done():
                    continue
                error = response_error(message)
                if error is not None:
                    future.set_exception(LhpRemoteError(error[0], error[1]))
                else:
                    future.set_result(message.get("result"))
        except asyncio.CancelledError:
            failure = (ERR_CONNECTION_CLOSED, "client closed")
            raise
        except (ConnectionError, OSError) as exc:
            failure = (ERR_CONNECTION_CLOSED, f"connection lost: {exc}")
        finally:
            self._fail_pending(*failure)
            if self._reader is reader:
                writer, self._writer = self._writer, None
                self._reader = None
                self._reader_task = None
                # This exit was the loop's own (EOF, oversized frame, lost
                # connection), not a teardown -- teardown nulls the reader
                # and closes the socket itself. An unclosed writer here stays
                # registered with the event loop and leaves the peer waiting
                # forever for a next frame. An oversized frame only exceeded
                # the *line* limit; the connection was still healthy and must
                # be closed here.
                if writer is not None:
                    with suppress(Exception):
                        writer.close()
                        await writer.wait_closed()

    def _fail_pending(self, code: str, message: str) -> None:
        pending, self._pending = self._pending, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(LhpRemoteError(code, message))

    async def _teardown_locked(self, code: str, message: str) -> None:
        """Close the connection and fail in-flight requests. Never raises."""
        writer, self._writer = self._writer, None
        self._reader = None
        task, self._reader_task = self._reader_task, None
        self._fail_pending(code, message)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if writer is not None:
            with suppress(Exception):
                writer.close()
                await writer.wait_closed()


__all__ = [
    "DEFAULT_CONNECT_TIMEOUT_S",
    "DEFAULT_REQUEST_TIMEOUT_S",
    "DirectLhpClient",
    "SocketLhpClient",
]
