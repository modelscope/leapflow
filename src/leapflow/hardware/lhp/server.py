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

Every actuating method (``write``, ``write_batch``, ``control.open``,
``control.update``, ``skill.execute``) passes the one ``WriteAuthority``; every
leased activity (control loops, skill runs) lives in the one ``NodeSessions``,
which is also the device mutex wire writes respect.

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
import json
import logging
import uuid
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from leapflow.hardware.context import HardwareContext
from leapflow.hardware.lhp.authority import (
    WRITE_MODE_OBSERVE_ONLY,
    WRITE_MODE_PERMIT_REQUIRED,
    WRITE_MODE_TRUSTED_V1,
    WriteAuthority,
)
from leapflow.hardware.lhp.control_sessions import NodeSessions
from leapflow.hardware.lhp.params import (
    freshness_bounds,
    optional_float,
    optional_int,
    require_channel_ids,
    require_commands,
    require_str,
)
from leapflow.hardware.lhp.protocol import (
    CAP_CONTROL_V1,
    CAP_FRAME_FRESHNESS_V1,
    CAP_PERMIT_V1,
    CAP_SKILL_V1,
    ERR_BAD_REQUEST,
    ERR_FRAME_TOO_LARGE,
    ERR_INTERNAL_ERROR,
    ERR_PROTOCOL_MISMATCH,
    ERR_TRANSPORT_ERROR,
    ERR_UNAUTHORIZED,
    ERR_UNKNOWN_CHANNEL,
    ERR_UNKNOWN_DEVICE,
    ERR_UNSUPPORTED_CAPABILITY,
    FAILURE_DEVICE_BUSY,
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
from leapflow.hardware.node_gate import (
    NodeWriteGate,
    ReadingIdentityError,
    merged_batch_state,
    refusal,
    verify_reading_identity,
)
from leapflow.hardware.transport import (
    FrameTransport,
    TransportError,
    TransportStatus,
    WriteOutcome,
)
from leapflow.hardware.attestation import declaration_payload
from leapflow.hardware.vision.frames import accepts_keyword as _accepts_keyword

logger = logging.getLogger(__name__)

_CONNECTION_ID: ContextVar[str] = ContextVar("lhp_connection_id", default="")
"""The socket connection a request arrived on; each request task sees its own."""


@dataclass
class _ConnectionState:
    """Per-connection gate: nothing but ``hello`` may precede authentication."""

    authenticated: bool = False
    connection_id: str = ""


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
        write_mode: str = WRITE_MODE_PERMIT_REQUIRED,
        permit_verifier: Any | None = None,
        skill_runtime: Any | None = None,
    ) -> None:
        text = str(node_id or "").strip()
        if not is_valid_device_id(text):
            raise ValueError(
                f"node_id {node_id!r} must be non-empty and match [a-z0-9_]+ "
                "(it becomes the host-side device namespace prefix)"
            )
        self._authority = WriteAuthority(
            str(write_mode or WRITE_MODE_PERMIT_REQUIRED), permit_verifier
        )
        self._registry = registry
        self._node_id = text
        self._token = str(token or "")
        self._device_ids = tuple(str(item) for item in device_ids)
        self._max_concurrent_requests = max(1, int(max_concurrent_requests))
        self._gate = NodeWriteGate(registry)
        self._skill_runtime = skill_runtime
        self._sessions = NodeSessions(
            registry,
            device_busy=skill_runtime.holds_device if skill_runtime is not None else None,
        )
        self._server: asyncio.AbstractServer | None = None
        self._unix_path = ""
        self._writers: set[asyncio.StreamWriter] = set()
        self._connection_states: dict[str, _ConnectionState] = {}
        # Frame channels each connection is consuming, as (device_id, holder) pairs. The
        # registry closes a node camera only when its last holder leaves, and a dropped
        # connection releases everything it held -- a host that vanished cannot keep a
        # camera on.
        self._frame_holds: dict[str, set[tuple[str, str]]] = {}
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
            "control.open": self._handle_control_open,
            "control.renew": self._handle_control_renew,
            "control.update": self._handle_control_update,
            "control.close": self._handle_control_close,
            "control.status": self._handle_control_status,
        }
        if skill_runtime is not None:
            from leapflow.hardware.lhp.skill_methods import SkillMethods

            self._handlers.update(
                SkillMethods(
                    skill_runtime,
                    authority=self._authority,
                    sessions=self._sessions,
                    connection_id=_CONNECTION_ID.get,
                ).handlers()
            )

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def listening(self) -> bool:
        return self._server is not None

    @property
    def write_mode(self) -> str:
        return self._authority.mode

    @property
    def sessions(self) -> NodeSessions:
        return self._sessions

    # ── Dispatch (shared by direct and socket bindings) ──

    async def handle_request(
        self, request: Mapping[str, Any], *, via: str = "direct"
    ) -> dict[str, Any]:
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
        except ReadingIdentityError as exc:
            return error_response(request_id, ERR_INTERNAL_ERROR, str(exc))
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
        if self._authority.mode == WRITE_MODE_TRUSTED_V1 and via != "direct":
            logger.warning(
                "LHP node %r operating in trusted_v1 mode — writes are accepted "
                "without a permit (token-only authentication). Consider upgrading "
                "to permit_required.",
                self._node_id,
            )
        contexts = self._exported_contexts()
        verifier = self._authority.verifier
        caps = list(capabilities_for_version(protocol))
        if verifier is not None:
            caps.append(CAP_PERMIT_V1)
        caps.append(CAP_CONTROL_V1)
        if protocol >= 2:
            caps.append(CAP_FRAME_FRESHNESS_V1)
        if self._skill_runtime is not None:
            caps.append(CAP_SKILL_V1)
        result: dict[str, Any] = {
            "protocol": protocol,
            "node_id": self._node_id,
            "device_count": len(contexts),
            "declarations": _declarations_digest(contexts),
            "capabilities": caps,
            "write_mode": self._authority.mode,
        }
        if verifier is not None:
            result["permit_audience"] = self._node_id
            result["trusted_issuers"] = list(getattr(verifier, "trusted_key_ids", ()) or ())
        return result

    async def _handle_list(self, params: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "node_id": self._node_id,
            "devices": [_wire_context(context) for context in self._exported_contexts()],
        }

    async def _handle_read(self, params: Mapping[str, Any]) -> dict[str, Any]:
        device_id = require_str(params, "device_id")
        channel_id = require_str(params, "channel_id")
        _, channel = self._resolve_channel(device_id, channel_id)
        if not channel.is_readable:
            raise LhpProtocolError(
                f"channel {channel_id!r} on {device_id!r} is not readable",
                code=ERR_UNKNOWN_CHANNEL,
            )
        async with self._registry.device_io(device_id):
            transport = await self._registry.transport(device_id)
            reading = await transport.read(channel_id)
        verify_reading_identity(reading, device_id, channel_id)
        return reading.to_dict()

    async def _handle_read_batch(self, params: Mapping[str, Any]) -> dict[str, Any]:
        device_id = require_str(params, "device_id")
        channel_ids = require_channel_ids(params)
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
        device_id = require_str(params, "device_id")
        channel_id = require_str(params, "channel_id")
        _, channel = self._resolve_channel(device_id, channel_id)
        if not channel.is_readable or not bool(getattr(channel, "is_media", False)):
            raise LhpProtocolError(
                f"channel {channel_id!r} on {device_id!r} is not a readable frame channel",
                code=ERR_UNKNOWN_CHANNEL,
            )
        max_width = optional_int(params, "max_width", upper=MAX_FRAME_WIDTH)
        quality = optional_int(params, "quality", upper=100)
        fps = optional_float(params, "fps", upper=MAX_FRAME_RATE_HZ)
        freshness = freshness_bounds(params)
        frame = await self._read_held_frame(
            device_id,
            channel_id,
            {"max_width": max_width, "quality": quality, "fps": fps},
            freshness,
        )
        verify_reading_identity(frame, device_id, channel_id)
        data = getattr(frame, "data", None)
        if not isinstance(data, (bytes, bytearray)) or not data:
            raise LhpProtocolError(
                "frame transport returned no encoded bytes", code=ERR_INTERNAL_ERROR
            )
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
            raise LhpProtocolError(
                "frame dimensions are outside protocol bounds", code=ERR_INTERNAL_ERROR
            )
        payload = frame.to_metadata()
        payload["data_b64"] = base64.b64encode(bytes(data)).decode("ascii")
        return payload

    async def _read_held_frame(
        self,
        device_id: str,
        channel_id: str,
        request: Mapping[str, Any],
        freshness: Mapping[str, Any],
    ) -> Any:
        """Read one frame for this connection, holding the device for it.

        A transport declaring ``frame_reads_concurrent`` serves many readers from one
        producer (a frame hub, a latest-frame pump), so only resolving the transport
        and recording the hold need the device's I/O lock: the read itself runs
        outside it. Otherwise a preview poll waiting on a fresh frame would stall a
        node-local skill's grasp capture on the same camera for a whole frame period,
        which is exactly the serialisation the hub exists to remove. The hold comes
        first so the transport cannot be released between lock and read; it stays
        even if the read fails, because the connection still consumes the channel
        and ``release_frame`` or disconnect ends it.
        """
        async with self._registry.device_io(device_id):
            transport = await self._registry.transport(device_id)
            if not isinstance(transport, FrameTransport):
                raise LhpProtocolError(
                    f"transport for {device_id!r} cannot produce frames",
                    code=ERR_UNSUPPORTED_CAPABILITY,
                )
            kwargs = _frame_kwargs(transport, request, freshness)
            concurrent = bool(getattr(transport, "frame_reads_concurrent", False)) and callable(
                getattr(self._registry, "hold", None)
            )
            if not concurrent:
                frame = await transport.read_frame(channel_id, **kwargs)
                self._hold_frame(device_id, channel_id)
                return frame
            self._hold_frame(device_id, channel_id)
        return await transport.read_frame(channel_id, **kwargs)

    async def _handle_release_frame(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Release this connection's claim on a node-local frame channel.

        The camera itself closes only when nobody else holds it: another connection's
        preview, or a node-local consumer, must not lose its device because one host
        closed a panel.
        """
        device_id = require_str(params, "device_id")
        channel_id = require_str(params, "channel_id")
        _, channel = self._resolve_channel(device_id, channel_id)
        if not bool(getattr(channel, "is_media", False)):
            raise LhpProtocolError(
                f"channel {channel_id!r} on {device_id!r} is not a frame channel",
                code=ERR_UNKNOWN_CHANNEL,
            )
        if callable(getattr(self._registry, "release_hold", None)):
            holder = _frame_holder(_CONNECTION_ID.get(), channel_id)
            self._frame_holds.get(_CONNECTION_ID.get(), set()).discard((device_id, holder))
            closed = await self._registry.release_hold(device_id, holder)
            return {
                "device_id": device_id,
                "channel_id": channel_id,
                "released": True,
                "device_closed": bool(closed),
            }
        # A registry without a holder ledger: the pre-ledger behaviour, last caller wins.
        drop_transport = getattr(self._registry, "drop_transport", None)
        if not callable(drop_transport):
            raise LhpProtocolError(
                "node registry cannot release frame transports",
                code=ERR_UNSUPPORTED_CAPABILITY,
            )
        async with self._registry.device_io(device_id):
            await drop_transport(device_id)
        return {"device_id": device_id, "channel_id": channel_id, "released": True}

    def _hold_frame(self, device_id: str, channel_id: str) -> None:
        """Record that the requesting connection now consumes *channel_id*."""
        hold = getattr(self._registry, "hold", None)
        if not callable(hold):
            return
        connection_id = _CONNECTION_ID.get()
        if connection_id and connection_id not in self._connection_states:
            # The request outlived its connection, whose holds are already released;
            # recording one now would keep the camera on with nobody left to drop it.
            return
        holder = _frame_holder(connection_id, channel_id)
        hold(device_id, holder)
        self._frame_holds.setdefault(connection_id, set()).add((device_id, holder))

    async def _release_connection_frames(self, connection_id: str) -> None:
        """Release every frame channel a closed connection still held. Never raises."""
        for device_id, holder in sorted(self._frame_holds.pop(connection_id, set())):
            try:
                closed = await self._registry.release_hold(device_id, holder)
            except Exception as exc:  # noqa: BLE001 - one device must not strand the rest
                logger.error(
                    "LHP frame release for %r after disconnect failed: %s",
                    device_id,
                    exc,
                    exc_info=True,
                )
                continue
            logger.info(
                "LHP connection closed; released %s on %s (device closed: %s)",
                holder,
                device_id,
                closed,
            )

    async def _handle_write(self, params: Mapping[str, Any]) -> dict[str, Any]:
        device_id = require_str(params, "device_id")
        channel_id = require_str(params, "channel_id")
        if "value" not in params:
            raise LhpProtocolError("write needs a 'value' (null is a value; omission is not)")
        value = params.get("value")
        self._resolve_channel(device_id, channel_id)
        gate_refusal = self._busy_refusal(device_id) or self._authority.check("write", params)
        if gate_refusal is not None:
            return gate_refusal.to_dict()
        outcome = await self._gate.write(device_id, channel_id, value)
        return outcome.to_dict()

    async def _handle_write_batch(self, params: Mapping[str, Any]) -> dict[str, Any]:
        device_id = require_str(params, "device_id")
        commands = require_commands(params)
        context = self._registry.context(device_id)
        if context is None:
            raise LhpProtocolError(f"unknown device {device_id!r}", code=ERR_UNKNOWN_DEVICE)
        for channel_id, _ in commands:
            if context.channel(channel_id) is None:
                raise LhpProtocolError(
                    f"unknown channel {channel_id!r} on device {device_id!r}",
                    code=ERR_UNKNOWN_CHANNEL,
                )
        gate_refusal = self._busy_refusal(device_id) or self._authority.check("write_batch", params)
        if gate_refusal is not None:
            outcomes = tuple(gate_refusal for _ in commands)
            return self._batch_payload(ok=False, outcomes=outcomes)
        ok, outcomes = await self._gate.write_batch(device_id, commands)
        return self._batch_payload(ok=ok, outcomes=outcomes)

    async def _handle_probe(self, params: Mapping[str, Any]) -> dict[str, Any]:
        device_id = require_str(params, "device_id")
        if self._registry.context(device_id) is None:
            raise LhpProtocolError(f"unknown device {device_id!r}", code=ERR_UNKNOWN_DEVICE)
        try:
            transport = await self._registry.transport(device_id)
            status = await transport.probe()
        except TransportError as exc:
            return TransportStatus(connected=False, halt_supported=False, detail=str(exc)).to_dict()
        payload = status.to_dict()
        hub = (status.metadata or {}).get("frame_hub")
        if isinstance(hub, Mapping) and hub:
            payload["frame_hub"] = dict(hub)  # pump health for the host's monitor; JSON-safe
        return payload

    async def _handle_halt(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Stop a device now, over the link that is already open.

        A halt never brings a device up. Opening one runs the whole bring-up
        (warm-up, firmware-profile verification), so a profile the arm could not
        report would refuse the stop itself -- a routine shutdown sweep logged
        ``cannot verify the arm firmware profile`` at ERROR for an arm this node
        had never opened, and a successful open would have woken a device purely
        to put it down. A device with no open link here is not being driven by
        this process and holds no motion this call could stop.

        ``halt_supported`` stays as the device declared it: the device can be
        stopped and this node simply has nothing to send the stop over, which is
        not the same as a device declaring that it cannot halt.

        The registry owns the link cache, so the stop goes through it: a stop that
        ends the link (a camera releases its device) must take the instance out of
        the cache, or every later frame read on this node fails with "is not open".
        """
        device_id = require_str(params, "device_id")
        context = self._registry.context(device_id)
        if context is None:
            raise LhpProtocolError(f"unknown device {device_id!r}", code=ERR_UNKNOWN_DEVICE)
        try:
            status = await self._registry.halt(device_id)
        except TransportError as exc:
            logger.error("LHP halt for %r failed: %s", device_id, exc, exc_info=True)
            return TransportStatus(
                connected=False,
                halt_supported=getattr(context, "halt_supported", None) is not False,
                halt_delivered=False,
                detail=str(exc),
            ).to_dict()
        if status is None:
            logger.info("LHP halt for %r: no open link on this node, nothing to stop", device_id)
            return TransportStatus(
                connected=False,
                halt_supported=getattr(context, "halt_supported", None) is not False,
                detail="no link is open to this device on the node, so nothing is being driven "
                       "from here",
            ).to_dict()
        return status.to_dict()

    # ── Control session handlers ──

    async def _handle_control_open(self, params: Mapping[str, Any]) -> dict[str, Any]:
        gate_refusal = self._authority.check("control.open", params)
        if gate_refusal is not None:
            return gate_refusal.to_dict()
        return await self._sessions.open(params, connection_id=_CONNECTION_ID.get())

    async def _handle_control_renew(self, params: Mapping[str, Any]) -> dict[str, Any]:
        return await self._sessions.renew(params, connection_id=_CONNECTION_ID.get())

    async def _handle_control_update(self, params: Mapping[str, Any]) -> dict[str, Any]:
        gate_refusal = self._authority.check("control.update", params)
        if gate_refusal is not None:
            return gate_refusal.to_dict()
        return await self._sessions.update(params, connection_id=_CONNECTION_ID.get())

    async def _handle_control_close(self, params: Mapping[str, Any]) -> dict[str, Any]:
        return await self._sessions.close(params, connection_id=_CONNECTION_ID.get())

    async def _handle_control_status(self, params: Mapping[str, Any]) -> dict[str, Any]:
        return await self._sessions.status(params)

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

    def _busy_refusal(self, device_id: str) -> WriteOutcome | None:
        """Refuse a wire write while a session or a skill run holds *device_id*.

        Checked before the permit so a refusal for a held device does not burn it.
        """
        if not self._sessions.busy_device((device_id,)):
            return None
        return refusal(
            f"device {device_id!r} is held by a running session; stop it or wait for it to end",
            FAILURE_DEVICE_BUSY,
        )

    @staticmethod
    def _batch_payload(*, ok: bool, outcomes: tuple[WriteOutcome, ...]) -> dict[str, Any]:
        return {
            "ok": bool(ok),
            "outcomes": [outcome.to_dict() for outcome in outcomes],
            "side_effect_state": merged_batch_state(ok, outcomes),
        }

    # ── Socket listener ──

    async def start(
        self, *, host: str = "127.0.0.1", port: int = LHP_DEFAULT_PORT
    ) -> tuple[str, int]:
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
        try:
            await self._sessions.shutdown()
        except Exception as exc:  # noqa: BLE001 - shutdown must still drop the connections
            logger.error("LHP session shutdown failed: %s", exc, exc_info=True)
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
        connection_id = str(uuid.uuid4())
        state.connection_id = connection_id
        self._connection_states[connection_id] = state
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
                    self._dispatch_and_reply(
                        request,
                        writer,
                        write_lock,
                        semaphore,
                        connection_id=connection_id,
                    )
                )
                handlers.add(task)
                task.add_done_callback(handlers.discard)
        except (ConnectionError, OSError):
            pass
        finally:
            # Sessions owned by a dropped connection fail safe.
            try:
                await self._sessions.on_disconnect(state.connection_id)
            except Exception as exc:  # noqa: BLE001 - a fail-safe fault must not leak the writer
                logger.error("LHP disconnect fail-safe failed: %s", exc, exc_info=True)
            self._connection_states.pop(connection_id, None)
            await self._release_connection_frames(state.connection_id)
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
        connection_id: str = "",
    ) -> None:
        async with semaphore:
            _CONNECTION_ID.set(connection_id)  # task-local: concurrent requests cannot mix
            response = await self.handle_request(request, via="socket")
            await _send_frame(writer, write_lock, response)


def _frame_holder(connection_id: str, channel_id: str) -> str:
    """Name one connection's claim on one frame channel in the registry's holder ledger."""
    return f"lhp:{connection_id or 'direct'}:{channel_id}"


def _declarations_digest(contexts: Sequence[HardwareContext]) -> str:
    """Fingerprint the exported declaration set, so a host can see it change.

    Carried in ``hello`` because a heartbeat is the only conversation a host keeps
    with an idle node. A node restarted inside one heartbeat interval never looks
    offline, so liveness alone cannot reveal that it now declares a corrected
    envelope or a camera plugged in since. Hashing the same canonical form the
    host uses to bind a review (``verification.declaration_payload``) keeps
    node-private bindings out of the fingerprint and keeps the two sides'
    digests comparable.
    """
    payloads = [declaration_payload(context) for context in contexts]
    encoded = json.dumps(payloads, sort_keys=True, separators=(",", ":"), default=str)
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


def _frame_kwargs(
    transport: Any, request: Mapping[str, Any], freshness: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the keywords one transport's ``read_frame`` can honour.

    ``fps`` postdates the original ``FrameTransport`` side protocol and is dropped
    for a transport that cannot take it, preserving structural compatibility while
    a ``TypeError`` raised inside a driver stays visible as a real defect. A
    freshness bound is different: silently dropping it would return a stale frame
    to a caller that asked for a fresh one, so it is refused instead.
    """
    kwargs: dict[str, Any] = {"max_width": request["max_width"], "quality": request["quality"]}
    if _accepts_keyword(transport.read_frame, "fps"):
        kwargs["fps"] = request["fps"]
    for name, value in freshness.items():
        if not _accepts_keyword(transport.read_frame, name):
            raise LhpProtocolError(
                f"frame transport cannot honour {name!r}", code=ERR_UNSUPPORTED_CAPABILITY
            )
        kwargs[name] = value
    return kwargs


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


__all__ = [
    "WRITE_MODE_OBSERVE_ONLY",
    "WRITE_MODE_PERMIT_REQUIRED",
    "WRITE_MODE_TRUSTED_V1",
    "LhpNodeServer",
]
