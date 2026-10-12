# Copyright (c) Alibaba, Inc. and its affiliates.
"""LHP v1 wire protocol: constants, frame codec, and the error taxonomy.

The frozen specification lives in ``temp/devices/plan/02-lhp-wire-protocol-v1.md``;
this module is its executable half. LHP is a *hardware-domain* protocol, not a
general-purpose RPC: it carries exactly the six-method device semantics
(``open``/``close`` are connection lifecycle, not wire methods) plus batch
variants. The control plane is JSON Lines; the streaming plane (subscriptions,
binary frames) is reserved for a later version and must not claim fields here.

Error semantics are the heart of the protocol. An error response means
"could not attempt" -- the command provably never reached a device. Anything
that may have reached the device travels as a ``WriteOutcome`` inside a
successful ``result``, because only that shape can express "it might have
landed". The three client-generated codes below are the exception that keeps
the rule safe across a broken connection: a write that timed out or lost its
connection has an *unknown* effect, never a negative one.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

LHP_PROTOCOL_VERSION = 1
"""Bump on any wire-incompatible change; both sides carry it in ``hello``."""

LHP_DEFAULT_PORT = 8767
"""Default TCP port for a node listener (8765 is reserved by Foxglove on the bench)."""

MAX_FRAME_BYTES = 1024 * 1024
"""One JSON line may not exceed 1 MiB; an over-long frame closes the connection."""

_DEVICE_ID_CHARSET = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")


def is_valid_device_id(value: str) -> bool:
    """Return whether *value* matches the ``[a-z0-9_]+`` identity charset.

    The same rule the registry admits device ids under, shared here so a node id
    (which becomes a namespace prefix, ``<node>__<device>``) cannot introduce a
    character that would later be rejected at admission -- a failure that would
    surface far from its cause.
    """
    return bool(value) and all(ch in _DEVICE_ID_CHARSET for ch in value)


# ── Error codes ──

ERR_BAD_REQUEST = "bad_request"
ERR_UNAUTHORIZED = "unauthorized"
ERR_PROTOCOL_MISMATCH = "protocol_mismatch"
ERR_UNKNOWN_DEVICE = "unknown_device"
ERR_UNKNOWN_CHANNEL = "unknown_channel"
ERR_TRANSPORT_ERROR = "transport_error"
ERR_INTERNAL_ERROR = "internal_error"
ERR_TIMEOUT = "timeout"
ERR_CONNECTION_CLOSED = "connection_closed"

WRITE_EFFECT_UNKNOWN_CODES = frozenset({ERR_INTERNAL_ERROR, ERR_TIMEOUT, ERR_CONNECTION_CLOSED})
"""Error codes after which a *write* may have reached the device.

The host-side write classifier consults exactly this set: any other error code
is a "could not attempt" and may raise a ``TransportError``. These three must
instead degrade to ``WriteOutcome(side_effect_state=unknown)`` so the recovery
layer never replays a physical command whose fate is unproven.
"""

# Node-reported failure codes carried inside a WriteOutcome (never as error frames,
# because a refusal is a definitive verdict about a command that never landed).
FAILURE_SAFETY_DENIED = "safety_policy_denied"
FAILURE_NOT_WRITABLE = "not_writable"
FAILURE_BATCH_ABORTED = "batch_aborted"

LHP_METHODS: tuple[str, ...] = (
    "hello",
    "list",
    "read",
    "read_batch",
    "write",
    "write_batch",
    "probe",
    "halt",
)

LHP_CAPABILITIES: tuple[str, ...] = tuple(m for m in LHP_METHODS if m != "hello")
"""Capability set advertised in ``hello``; adding a method updates this tuple."""


class LhpProtocolError(Exception):
    """A frame or request violated the protocol envelope.

    Raised internally by the server handlers and translated to an error response;
    the client side never raises this (it raises ``LhpRemoteError`` for what the
    peer answered and ``ValueError`` for a local misuse).
    """

    def __init__(self, message: str, *, code: str = ERR_BAD_REQUEST) -> None:
        super().__init__(message)
        self.code = code


class LhpRemoteError(RuntimeError):
    """The exchange with the node failed.

    Covers both an error frame from the peer and locally generated transport
    failures of the exchange itself (``timeout``, ``connection_closed``). The
    ``code`` always comes from the taxonomy above, so callers classify on it
    rather than on message text.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message


def encode_frame(payload: Mapping[str, Any]) -> bytes:
    """Serialize one JSON-lines frame. ``ensure_ascii=False``, compact separators."""
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def decode_frame(line: bytes | str) -> dict[str, Any]:
    """Parse one JSON-lines frame, or raise ``LhpProtocolError``.

    Strict on shape: a frame that is not a JSON object is a protocol error, not
    an empty request -- silently coercing it would make a broken encoder look
    like a well-behaved peer.
    """
    try:
        text = line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else str(line)
    except UnicodeDecodeError as exc:
        raise LhpProtocolError("frame is not valid UTF-8") from exc
    text = text.strip()
    if not text:
        raise LhpProtocolError("empty frame")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LhpProtocolError(f"invalid JSON frame: {exc}") from exc
    if not isinstance(payload, dict):
        raise LhpProtocolError("frame must be a JSON object")
    return payload


def make_request(request_id: Any, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a request frame."""
    return {"id": request_id, "method": str(method), "params": dict(params or {})}


def ok_response(request_id: Any, result: Any) -> dict[str, Any]:
    """Build a success response frame."""
    return {"id": request_id, "ok": True, "result": result}


def error_response(request_id: Any, code: str, message: str) -> dict[str, Any]:
    """Build an error response frame ("could not attempt")."""
    return {"id": request_id, "ok": False, "error": {"code": str(code), "message": str(message)}}


def response_error(response: Mapping[str, Any]) -> tuple[str, str] | None:
    """Return ``(code, message)`` when *response* is an error frame, else None."""
    if bool(response.get("ok")):
        return None
    error = response.get("error")
    if not isinstance(error, Mapping):
        return ERR_INTERNAL_ERROR, "malformed error frame"
    return (
        str(error.get("code") or ERR_INTERNAL_ERROR),
        str(error.get("message") or ""),
    )


__all__ = [
    "ERR_BAD_REQUEST",
    "ERR_CONNECTION_CLOSED",
    "ERR_INTERNAL_ERROR",
    "ERR_PROTOCOL_MISMATCH",
    "ERR_TIMEOUT",
    "ERR_TRANSPORT_ERROR",
    "ERR_UNAUTHORIZED",
    "ERR_UNKNOWN_CHANNEL",
    "ERR_UNKNOWN_DEVICE",
    "FAILURE_BATCH_ABORTED",
    "FAILURE_NOT_WRITABLE",
    "FAILURE_SAFETY_DENIED",
    "LHP_CAPABILITIES",
    "LHP_DEFAULT_PORT",
    "LHP_METHODS",
    "LHP_PROTOCOL_VERSION",
    "MAX_FRAME_BYTES",
    "WRITE_EFFECT_UNKNOWN_CODES",
    "LhpProtocolError",
    "LhpRemoteError",
    "decode_frame",
    "encode_frame",
    "error_response",
    "is_valid_device_id",
    "make_request",
    "ok_response",
    "response_error",
]
