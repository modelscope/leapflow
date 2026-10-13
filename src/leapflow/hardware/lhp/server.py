# Copyright (c) Alibaba, Inc. and its affiliates.
"""LHP node server: the node side of the fabric.

Wraps one ``HardwareRegistry`` and answers LHP requests against it, exactly the
way the in-process tool layer would: channel validation, safety policy,
per-device I/O serialisation, then the transport. The registry contract is
duck-typed (``context`` / ``contexts`` / ``device_io`` / ``transport`` /
``read_batch`` / ``check_safety_policy``), so the production registry and a
test stub are equally admissible -- the MCP transport has the same property on
the client side.

Two entry points share one dispatch path:

- ``handle_request`` -- the pure dispatcher. ``DirectLhpClient`` calls it
  in-process; the socket connection handler calls it with ``via="socket"``,
  which additionally enforces the shared token during ``hello``.
- ``start`` / ``start_unix`` -- the socket listener. The first frame on every
  connection must be ``hello`` carrying a valid token; anything else is
  answered ``unauthorized`` and the connection closes. After that, each
  request runs in its own task, so ``probe`` and ``halt`` answer while another
  call is stuck -- the same deliberateness as the transport-level lock-freedom
  of those two methods, one layer up.

Write semantics follow the frozen protocol spec: a refusal the node can prove
(wrong direction, safety policy, batch precheck) is a ``WriteOutcome`` with
``side_effect_state="none"``; a failure that might have reached the device
maps to an ``internal_error`` error frame, which the host classifies as
``unknown``. A ``TransportError`` -- by contract "could not attempt" -- maps to
an error frame carrying its ``failure_code`` so the host can raise the same
typed error it would have seen locally.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import inspect
import json
import logging
import math
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from leapflow.hardware.context import HardwareContext
from leapflow.hardware.lhp.protocol import (
    ERR_BAD_REQUEST,
    ERR_FRAME_TOO_LARGE,
    ERR_INTERNAL_ERROR,
    ERR_PROTOCOL_MISMATCH,
    ERR_TRANSPORT_ERROR,
    ERR_UNAUTHORIZED,
    ERR_UNKNOWN_CHANNEL,
    ERR_UNKNOWN_DEVICE,
    ERR_UNSUPPORTED_CAPABILITY,
    FAILURE_BATCH_ABORTED,
    FAILURE_NOT_WRITABLE,
    FAILURE_SAFETY_DENIED,
    LHP_DEFAULT_PORT,
    LHP_SUPPORTED_VERSIONS,
    MAX_FRAME_BYTES,
    MAX_FRAME_DATA_BYTES,
    MAX_FRAME_RATE_HZ,
    MAX_FRAME_WIDTH,
    SAFE_FRAME_MEDIA_TYPES,
    LhpProtocolError,
    capabilities_for_version,
    decode_frame,
    encode_frame,
    error_response,
    is_valid_device_id,
    ok_response,
)
from leapflow.hardware.transport import (
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_PARTIAL,
    BatchTransport,
    FrameTransport,
    TransportError,
    TransportStatus,
    WriteOutcome,
    worst_side_effect_state,
)

logger = logging.getLogger(__name__)


@dataclass
class _ConnectionState:
    """Per-connection gate: nothing but ``hello`` may precede authentication."""

    authenticated: bool = False


class LhpNodeServer:
    """Export one registry's devices over the newest mutually supported LHP version."""

    def __init__(
        self,
        registry: Any,
        *,
        node_id: str,
        token: str = "",
        device_ids: Sequence[str] = (),
        max_concurrent_requests: int = 32,
    ) -> None:
        text = str(node_id or "").strip()
        if not is_valid_device_id(text):
            raise ValueError(
                f"node_id {node_id!r} must be non-empty and match [a-z0-9_]+ "
                "(it becomes the host-side device namespace prefix)"
            )
        self._registry = registry
        self._node_id = text
        self._token = str(token or "")
        self._device_ids = tuple(str(item) for item in device_ids)
        self._max_concurrent_requests = max(1, int(max_concurrent_requests))
        self._server: asyncio.AbstractServer | None = None
        self._unix_path = ""
        self._writers: set[asyncio.StreamWriter] = set()
        self._handlers: dict[str, Any] = {
            "hello": self._handle_hello,
            "list": self._handle_list,
            "read": self._handle_read,
            "read_batch": self._handle_read_batch,
            "read_frame": self._handle_read_frame,
            "release_frame": self._handle_release_frame,
            "write": self._handle_write,
            "write_batch": self._handle_write_batch,
            "probe": self._handle_probe,
            "halt": self._handle_halt,
        }

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def listening(self) -> bool:
        return self._server is not None

    # ── Dispatch (shared by direct and socket bindings) ──

    async def handle_request(self, request: Mapping[str, Any], *, via: str = "direct") -> dict[str, Any]:
        """Validate, dispatch, and answer one request. Never raises.

        ``via`` distinguishes the trust domains: ``"direct"`` is the same
        process and skips the token gate; ``"socket"`` enforces it during
        ``hello``. The version check runs in both.
        """
        if not isinstance(request, Mapping):
            return error_response(None, ERR_BAD_REQUEST, "request must be a JSON object")
        request_id = request.get("id")
        method = request.get("method")
        if not isinstance(method, str) or not method:
            return error_response(request_id, ERR_BAD_REQUEST, "request needs a 'method'")
        handler = self._handlers.get(method)
        if handler is None:
            return error_response(request_id, ERR_BAD_REQUEST, f"unknown method {method!r}")
        params: Any = request.get("params")
        if params is None:
            params = {}
        if not isinstance(params, Mapping):
            return error_response(request_id, ERR_BAD_REQUEST, "'params' must be an object")
        try:
            if method == "hello":
                result = await handler(params, via=via)
            else:
                result = await handler(params)
        except LhpProtocolError as exc:
            return error_response(request_id, exc.code, str(exc))
        except TransportError as exc:
            return error_response(request_id, exc.failure_code or ERR_TRANSPORT_ERROR, str(exc))
        except Exception as exc:  # noqa: BLE001 - any handler fault is an internal_error frame
            logger.error("LHP %s handler failed: %s", method, exc, exc_info=True)
            return error_response(request_id, ERR_INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
        return ok_response(request_id, result)

    # ── Handlers ──

    async def _handle_hello(self, params: Mapping[str, Any], *, via: str) -> dict[str, Any]:
        protocol = params.get("protocol")
        if protocol not in LHP_SUPPORTED_VERSIONS:
            raise LhpProtocolError(
                f"protocol mismatch: client speaks {protocol!r}, node supports "
                f"{LHP_SUPPORTED_VERSIONS!r}",
                code=ERR_PROTOCOL_MISMATCH,
            )
        if via != "direct" and self._token:
            presented = str(params.get("token") or "")
            if not hmac.compare_digest(presented, self._token):
                raise LhpProtocolError("invalid or missing token", code=ERR_UNAUTHORIZED)
        contexts = self._exported_contexts()
        return {
            "protocol": protocol,
            "node_id": self._node_id,
            "device_count": len(contexts),
            "declarations": _declarations_digest(contexts),
            "capabilities": list(capabilities_for_version(protocol)),
        }

    async def _handle_list(self, params: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "node_id": self._node_id,
            "devices": [_wire_context(context) for context in self._exported_contexts()],
        }

    async def _handle_read(self, params: Mapping[str, Any]) -> dict[str, Any]:
        device_id = _require_str(params, "device_id")
        channel_id = _require_str(params, "channel_id")
        _, channel = self._resolve_channel(device_id, channel_id)
        if not channel.is_readable:
            raise LhpProtocolError(
                f"channel {channel_id!r} on {device_id!r} is not readable",
                code=ERR_UNKNOWN_CHANNEL,
            )
        async with self._registry.device_io(device_id):
            transport = await self._registry.transport(device_id)
            reading = await transport.read(channel_id)
        _verify_reading_identity(reading, device_id, channel_id)
        return reading.to_dict()

    async def _handle_read_batch(self, params: Mapping[str, Any]) -> dict[str, Any]:
        device_id = _require_str(params, "device_id")
        channel_ids = _require_channel_ids(params)
        for channel_id in channel_ids:
            _, channel = self._resolve_channel(device_id, channel_id)
            if not channel.is_readable:
                raise LhpProtocolError(
                    f"channel {channel_id!r} on {device_id!r} is not readable",
                    code=ERR_UNKNOWN_CHANNEL,
                )
        batch = await self._registry.read_batch(device_id, tuple(channel_ids))
        readings = tuple(getattr(batch, "readings", ()) or ())
        requested = set(channel_ids)
        for reading in readings:
            actual_channel = str(getattr(reading, "channel_id", ""))
            if actual_channel not in requested:
                raise LhpProtocolError(
                    f"transport returned a reading for channel {actual_channel!r}, "
                    "which was not requested",
                    code=ERR_INTERNAL_ERROR,
                )
            actual_device = str(getattr(reading, "device_id", ""))
            if actual_device and actual_device != device_id:
                raise LhpProtocolError(
                    f"transport returned a reading for device {actual_device!r}, not {device_id!r}",
                    code=ERR_INTERNAL_ERROR,
                )
        return {
            "device_id": str(getattr(batch, "device_id", "") or device_id),
            "readings": [reading.to_dict() for reading in readings],
            "observed_at": float(getattr(batch, "observed_at", 0.0) or 0.0),
        }

    async def _handle_read_frame(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Return one bounded encoded frame over the negotiated v2 control plane."""
        device_id = _require_str(params, "device_id")
        channel_id = _require_str(params, "channel_id")
        _, channel = self._resolve_channel(device_id, channel_id)
        if not channel.is_readable or not bool(getattr(channel, "is_media", False)):
            raise LhpProtocolError(
                f"channel {channel_id!r} on {device_id!r} is not a readable frame channel",
                code=ERR_UNKNOWN_CHANNEL,
            )
        max_width = _optional_int(params, "max_width", upper=MAX_FRAME_WIDTH)
        quality = _optional_int(params, "quality", upper=100)
        fps = _optional_float(params, "fps", upper=MAX_FRAME_RATE_HZ)
        async with self._registry.device_io(device_id):
            transport = await self._registry.transport(device_id)
            if not isinstance(transport, FrameTransport):
                raise LhpProtocolError(
                    f"transport for {device_id!r} cannot produce frames",
                    code=ERR_UNSUPPORTED_CAPABILITY,
                )
            frame_kwargs: dict[str, Any] = {
                "max_width": max_width,
                "quality": quality,
            }
            # ``fps`` postdates the original FrameTransport side protocol. Preserve
            # structural compatibility with third-party transports while keeping a
            # TypeError raised inside a driver visible as a real internal defect.
            if _accepts_keyword(transport.read_frame, "fps"):
                frame_kwargs["fps"] = fps
            frame = await transport.read_frame(channel_id, **frame_kwargs)
        _verify_reading_identity(frame, device_id, channel_id)
        data = getattr(frame, "data", None)
        if not isinstance(data, (bytes, bytearray)) or not data:
            raise LhpProtocolError("frame transport returned no encoded bytes", code=ERR_INTERNAL_ERROR)
        if len(data) > MAX_FRAME_DATA_BYTES:
            raise LhpProtocolError(
                f"encoded frame is {len(data)} bytes; limit is {MAX_FRAME_DATA_BYTES}",
                code=ERR_FRAME_TOO_LARGE,
            )
        media_type = str(getattr(frame, "media_type", "") or "")
        if media_type not in SAFE_FRAME_MEDIA_TYPES:
            raise LhpProtocolError(
                f"frame media type {media_type!r} is not supported",
                code=ERR_UNSUPPORTED_CAPABILITY,
            )
        width = int(getattr(frame, "width", 0) or 0)
        height = int(getattr(frame, "height", 0) or 0)
        if width < 0 or height < 0 or width > MAX_FRAME_WIDTH or height > MAX_FRAME_WIDTH:
            raise LhpProtocolError("frame dimensions are outside protocol bounds", code=ERR_INTERNAL_ERROR)
        payload = frame.to_metadata()
        payload["data_b64"] = base64.b64encode(bytes(data)).decode("ascii")
        return payload

    async def _handle_release_frame(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Release node-local capture when the host's last preview lease closes."""
        device_id = _require_str(params, "device_id")
        channel_id = _require_str(params, "channel_id")
        _, channel = self._resolve_channel(device_id, channel_id)
        if not bool(getattr(channel, "is_media", False)):
            raise LhpProtocolError(
                f"channel {channel_id!r} on {device_id!r} is not a frame channel",
                code=ERR_UNKNOWN_CHANNEL,
            )
        drop_transport = getattr(self._registry, "drop_transport", None)
        if not callable(drop_transport):
            raise LhpProtocolError(
                "node registry cannot release frame transports",
                code=ERR_UNSUPPORTED_CAPABILITY,
            )
        async with self._registry.device_io(device_id):
            await drop_transport(device_id)
        return {"device_id": device_id, "channel_id": channel_id, "released": True}

    async def _handle_write(self, params: Mapping[str, Any]) -> dict[str, Any]:
        device_id = _require_str(params, "device_id")
        channel_id = _require_str(params, "channel_id")
        if "value" not in params:
            raise LhpProtocolError("write needs a 'value' (null is a value; omission is not)")
        value = params.get("value")
        context, channel = self._resolve_channel(device_id, channel_id)
        refusal = self._write_precheck(context, channel, value)
        if refusal is not None:
            return refusal.to_dict()
        async with self._registry.device_io(device_id):
            transport = await self._registry.transport(device_id)
            outcome = await transport.write(channel_id, value)
        if outcome.readback is not None:
            # Evidence for this write must be about this channel: presenting a
            # readback for some other channel as this one's would be worse than
            # presenting none, so the driver contract is enforced here too.
            _verify_reading_identity(outcome.readback, device_id, channel_id)
        return outcome.to_dict()

    async def _handle_write_batch(self, params: Mapping[str, Any]) -> dict[str, Any]:
        device_id = _require_str(params, "device_id")
        commands = _require_commands(params)
        context = self._registry.context(device_id)
        if context is None:
            raise LhpProtocolError(f"unknown device {device_id!r}", code=ERR_UNKNOWN_DEVICE)
        for channel_id, _ in commands:
            if context.channel(channel_id) is None:
                raise LhpProtocolError(
                    f"unknown channel {channel_id!r} on device {device_id!r}",
                    code=ERR_UNKNOWN_CHANNEL,
                )
        # A batch is atomic in intent: one refused member refuses the whole
        # batch *before anything is attempted*, so every outcome is provably
        # NONE and nothing needs verification before a retry.
        verdicts = [
            self._write_precheck(context, context.channel(channel_id), value)
            for channel_id, value in commands
        ]
        if any(verdict is not None for verdict in verdicts):
            aborted = WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error="batch refused: another command was rejected before anything was attempted",
                failure_code=FAILURE_BATCH_ABORTED,
            )
            outcomes = tuple(verdict if verdict is not None else aborted for verdict in verdicts)
            return self._batch_payload(ok=False, outcomes=outcomes)
        async with self._registry.device_io(device_id):
            transport = await self._registry.transport(device_id)
            if isinstance(transport, BatchTransport):
                batch = await transport.write_batch(tuple(commands))
                outcomes = tuple(batch.outcomes)
                ok = bool(batch.ok)
            else:
                outcomes_list: list[WriteOutcome] = []
                for channel_id, value in commands:
                    try:
                        outcomes_list.append(await transport.write(channel_id, value))
                    except TransportError as exc:
                        outcomes_list.append(
                            WriteOutcome(
                                ok=False,
                                side_effect_state=SIDE_EFFECT_NONE,
                                error=str(exc),
                                failure_code=exc.failure_code,
                            )
                        )
                outcomes = tuple(outcomes_list)
                ok = all(o.ok for o in outcomes)
        for index, outcome in enumerate(outcomes):
            # Batch outcomes are per-command evidence, in command order. The
            # single-write path refuses a readback for another channel and the
            # batch path must not be a quieter way in: presenting a readback
            # for one channel as another's is worse than presenting none.
            if outcome.readback is not None and index < len(commands):
                _verify_reading_identity(outcome.readback, device_id, commands[index][0])
        return self._batch_payload(ok=ok, outcomes=outcomes)

    async def _handle_probe(self, params: Mapping[str, Any]) -> dict[str, Any]:
        device_id = _require_str(params, "device_id")
        if self._registry.context(device_id) is None:
            raise LhpProtocolError(f"unknown device {device_id!r}", code=ERR_UNKNOWN_DEVICE)
        try:
            transport = await self._registry.transport(device_id)
            status = await transport.probe()
        except TransportError as exc:
            return TransportStatus(connected=False, halt_supported=False, detail=str(exc)).to_dict()
        return status.to_dict()

    async def _handle_halt(self, params: Mapping[str, Any]) -> dict[str, Any]:
        device_id = _require_str(params, "device_id")
        if self._registry.context(device_id) is None:
            raise LhpProtocolError(f"unknown device {device_id!r}", code=ERR_UNKNOWN_DEVICE)
        try:
            transport = await self._registry.transport(device_id)
            status = await transport.halt()
        except TransportError as exc:
            logger.error("LHP halt for %r failed: %s", device_id, exc, exc_info=True)
            return TransportStatus(connected=False, halt_supported=False, detail=str(exc)).to_dict()
        return status.to_dict()

    # ── Internals ──

    def _exported_contexts(self) -> tuple[HardwareContext, ...]:
        contexts = self._registry.contexts()
        if not self._device_ids:
            return tuple(contexts)
        allowed = set(self._device_ids)
        return tuple(context for context in contexts if context.device_id in allowed)

    def _resolve_channel(self, device_id: str, channel_id: str) -> tuple[HardwareContext, Any]:
        context = self._registry.context(device_id)
        if context is None:
            raise LhpProtocolError(f"unknown device {device_id!r}", code=ERR_UNKNOWN_DEVICE)
        channel = context.channel(channel_id)
        if channel is None:
            raise LhpProtocolError(
                f"unknown channel {channel_id!r} on device {device_id!r}",
                code=ERR_UNKNOWN_CHANNEL,
            )
        return context, channel

    def _write_precheck(self, context: HardwareContext, channel: Any, value: Any) -> WriteOutcome | None:
        """Return a provable refusal for *value*, or None when the write may proceed.

        Defense in depth: the host already runs its own feasibility checks, but a
        node must never rely on a remote caller having done so -- the same reason
        ``check_safety_policy`` exists below the tool layer.
        """
        if not channel.is_writable:
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error=f"channel {channel.channel_id!r} is not writable",
                failure_code=FAILURE_NOT_WRITABLE,
            )
        allowed, reason = self._registry.check_safety_policy(context, channel, value)
        if not allowed:
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error=reason,
                failure_code=FAILURE_SAFETY_DENIED,
            )
        return None

    @staticmethod
    def _batch_payload(*, ok: bool, outcomes: tuple[WriteOutcome, ...]) -> dict[str, Any]:
        merged = worst_side_effect_state(outcome.side_effect_state for outcome in outcomes)
        # A vector command that left the device mixed (some channels landed, one
        # provably did not) is PARTIAL, not COMMITTED -- "committed" would claim
        # the batch as a whole took effect.
        if not ok and merged == SIDE_EFFECT_COMMITTED:
            merged = SIDE_EFFECT_PARTIAL
        return {
            "ok": bool(ok),
            "outcomes": [outcome.to_dict() for outcome in outcomes],
            "side_effect_state": merged,
        }

    # ── Socket listener ──

    async def start(self, *, host: str = "127.0.0.1", port: int = LHP_DEFAULT_PORT) -> tuple[str, int]:
        """Bind a TCP listener. Returns the actual ``(host, port)`` bound."""
        if self._server is not None:
            raise RuntimeError("LHP node server is already started")
        self._server = await asyncio.start_server(
            self._serve_connection, host=host, port=port, limit=MAX_FRAME_BYTES + 4096
        )
        sockets = self._server.sockets or ()
        if sockets:
            sockname = sockets[0].getsockname()
            return str(sockname[0]), int(sockname[1])
        return host, port

    async def start_unix(self, path: str | Path) -> str:
        """Bind a Unix-domain socket. Returns the socket path."""
        if self._server is not None:
            raise RuntimeError("LHP node server is already started")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        self._server = await asyncio.start_unix_server(
            self._serve_connection, path=str(target), limit=MAX_FRAME_BYTES + 4096
        )
        self._unix_path = str(target)
        return self._unix_path

    async def stop(self) -> None:
        """Stop the listener and drop live connections. Never raises.

        The writers are closed *before* ``wait_closed()``: since Python 3.12
        that call waits for every active connection to finish, so waiting
        first would deadlock on the very connections the close is meant to
        drop.
        """
        server, self._server = self._server, None
        if server is not None:
            server.close()
        for writer in list(self._writers):
            with suppress(Exception):
                writer.close()
        self._writers.clear()
        if server is not None:
            with suppress(Exception):
                await server.wait_closed()
        if self._unix_path:
            with suppress(OSError):
                Path(self._unix_path).unlink(missing_ok=True)
            self._unix_path = ""

    async def _serve_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        state = _ConnectionState()
        semaphore = asyncio.Semaphore(self._max_concurrent_requests)
        write_lock = asyncio.Lock()
        handlers: set[asyncio.Task[None]] = set()
        self._writers.add(writer)
        try:
            while True:
                try:
                    line = await reader.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    await _send_frame(
                        writer,
                        write_lock,
                        error_response(None, ERR_BAD_REQUEST, "frame exceeded the line limit"),
                    )
                    break
                if not line:
                    break
                try:
                    request = decode_frame(line)
                except LhpProtocolError as exc:
                    await _send_frame(writer, write_lock, error_response(None, exc.code, str(exc)))
                    if not state.authenticated:
                        break
                    continue
                if not state.authenticated:
                    # Synchronous gate: the hello exchange completes (or fails)
                    # before another frame is even read, so no authenticated work
                    # can race ahead of it.
                    if request.get("method") != "hello":
                        await _send_frame(
                            writer,
                            write_lock,
                            error_response(
                                request.get("id"), ERR_UNAUTHORIZED, "first frame must be hello"
                            ),
                        )
                        break
                    response = await self.handle_request(request, via="socket")
                    await _send_frame(writer, write_lock, response)
                    if not bool(response.get("ok")):
                        break
                    state.authenticated = True
                    continue
                task = asyncio.get_running_loop().create_task(
                    self._dispatch_and_reply(request, writer, write_lock, semaphore)
                )
                handlers.add(task)
                task.add_done_callback(handlers.discard)
        except (ConnectionError, OSError):
            pass
        finally:
            self._writers.discard(writer)
            with suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def _dispatch_and_reply(
        self,
        request: Mapping[str, Any],
        writer: asyncio.StreamWriter,
        write_lock: asyncio.Lock,
        semaphore: asyncio.Semaphore,
    ) -> None:
        async with semaphore:
            response = await self.handle_request(request, via="socket")
            await _send_frame(writer, write_lock, response)


def _declarations_digest(contexts: Sequence[HardwareContext]) -> str:
    """Fingerprint the exported declaration set, so a host can see it change.

    Carried in ``hello`` because a heartbeat is the only conversation a host keeps
    with an idle node. A node restarted inside one heartbeat interval never looks
    offline, so liveness alone cannot reveal that it now declares a corrected
    envelope or a camera plugged in since. Hashing the wire form, not the full
    context, keeps node-private bindings out of the fingerprint too.
    """
    wire = [_wire_context(context) for context in contexts]
    encoded = json.dumps(wire, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _wire_context(context: HardwareContext) -> dict[str, Any]:
    """Serialize a declaration for the wire without its node-private binding.

    ``transport.config`` is how *this node* reaches the device -- serial paths,
    driver credentials, an MCP token, even the shared secret of a chained
    ``lhp`` hop. The host replaces the binding with its own ``lhp`` reference
    anyway, so exporting it could only leak node secrets. The kind stays as a
    diagnostic of the southbound transport (``ros2``, ``host``, ...).

    ``control_bindings`` is the *unresolved* form of the same binding (MCP
    endpoints, ROS2 graph names, bus addresses) and is withheld for the same
    reason: the host can only reach the device through this node.
    """
    payload = context.to_dict()
    payload["transport"] = {"kind": context.transport.kind, "config": {}}
    payload.pop("control_bindings", None)
    return payload


def _accepts_keyword(callable_: Any, name: str) -> bool:
    """Return whether a frame callable accepts an optional keyword."""
    try:
        parameters = inspect.signature(callable_).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _require_str(params: Mapping[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value:
        raise LhpProtocolError(f"{key!r} must be a non-empty string")
    return value


def _optional_int(params: Mapping[str, Any], key: str, *, upper: int) -> int:
    value = params.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > upper:
        raise LhpProtocolError(f"{key!r} must be an integer from 0 through {upper}")
    return value


def _optional_float(params: Mapping[str, Any], key: str, *, upper: float) -> float:
    value = params.get(key, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LhpProtocolError(f"{key!r} must be a number from 0 through {upper:g}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or numeric > upper:
        raise LhpProtocolError(f"{key!r} must be a number from 0 through {upper:g}")
    return numeric


def _require_channel_ids(params: Mapping[str, Any]) -> list[str]:
    raw = params.get("channel_ids")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise LhpProtocolError("'channel_ids' must be a non-empty list of channel ids")
    channel_ids: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            raise LhpProtocolError("every entry of 'channel_ids' must be a non-empty string")
        channel_ids.append(item)
    return channel_ids


def _require_commands(params: Mapping[str, Any]) -> list[tuple[str, Any]]:
    raw = params.get("commands")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise LhpProtocolError("'commands' must be a non-empty list of [channel, value] pairs")
    commands: list[tuple[str, Any]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise LhpProtocolError("every command must be a [channel, value] pair")
        channel_id = item[0]
        if not isinstance(channel_id, str) or not channel_id:
            raise LhpProtocolError("a command channel must be a non-empty string")
        commands.append((channel_id, item[1]))
    return commands


def _verify_reading_identity(reading: Any, device_id: str, channel_id: str) -> None:
    """Refuse a reading whose identity contradicts the request.

    A transport that returns another channel's value produces a result that is
    indistinguishable from success, so this is checked here -- where the request
    identity is still known -- rather than trusted downstream.
    """
    actual_channel = str(getattr(reading, "channel_id", ""))
    if actual_channel != channel_id:
        raise LhpProtocolError(
            f"transport returned a reading for channel {actual_channel!r}, not {channel_id!r}",
            code=ERR_INTERNAL_ERROR,
        )
    actual_device = str(getattr(reading, "device_id", ""))
    if actual_device and actual_device != device_id:
        raise LhpProtocolError(
            f"transport returned a reading for device {actual_device!r}, not {device_id!r}",
            code=ERR_INTERNAL_ERROR,
        )


async def _send_frame(
    writer: asyncio.StreamWriter, write_lock: asyncio.Lock, payload: Mapping[str, Any]
) -> None:
    """Write one response frame, serialised per connection. Never raises."""
    try:
        async with write_lock:
            writer.write(encode_frame(payload))
            await writer.drain()
    except (ConnectionError, OSError, RuntimeError) as exc:
        logger.debug("LHP reply could not be delivered: %s", exc)


__all__ = ["LhpNodeServer"]
