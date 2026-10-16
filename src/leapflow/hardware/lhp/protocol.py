# Copyright (c) Alibaba, Inc. and its affiliates.
"""LHP wire protocol: versions, frame codec, and the error taxonomy.

This module is the frozen, executable specification of the wire contract.
Version 2 is an additive, capability-negotiated extension for bounded media frame
requests and explicit remote capture release. LHP remains a *hardware-domain*
protocol rather than a general-purpose RPC: scalar and write semantics are unchanged,
and frames use request/response JSON Lines instead of introducing an unbounded stream.

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

LHP_PROTOCOL_VERSION = 2
"""Newest protocol version offered by clients and nodes."""

LHP_SUPPORTED_VERSIONS: tuple[int, ...] = (2, 1)
"""Supported versions in preference order; clients fall back only on mismatch."""

LHP_DEFAULT_PORT = 8767
"""Default TCP port for a node listener (8765 is reserved by Foxglove on the bench)."""

MAX_FRAME_BYTES = 1024 * 1024
"""One JSON line may not exceed 1 MiB; an over-long frame closes the connection."""

MAX_FRAME_DATA_BYTES = 700 * 1024
"""Maximum encoded image bytes before base64, leaving safe JSON-envelope headroom."""

MAX_FRAME_WIDTH = 4096
MAX_FRAME_RATE_HZ = 60.0
SAFE_FRAME_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})

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
ERR_UNSUPPORTED_CAPABILITY = "unsupported_capability"
ERR_FRAME_TOO_LARGE = "frame_too_large"

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
FAILURE_INTERLOCK_UNSATISFIED = "interlock_unsatisfied"
FAILURE_DEVICE_DEGRADED = "device_degraded"

# Permit-related failure codes (side_effect=NONE; command never attempted).
FAILURE_PERMIT_REQUIRED = "permit_required"
FAILURE_PERMIT_INVALID = "permit_invalid"
FAILURE_PERMIT_EXPIRED = "permit_expired"
FAILURE_PERMIT_REPLAYED = "permit_replayed"
FAILURE_PERMIT_AUDIENCE_MISMATCH = "permit_audience_mismatch"
FAILURE_PERMIT_DIGEST_MISMATCH = "permit_digest_mismatch"
FAILURE_WRITES_DISABLED = "writes_disabled"
FAILURE_REMOTE_CONTROL_UNSUPPORTED = "remote_control_unsupported"

LHP_V1_CAPABILITIES: tuple[str, ...] = (
    "list",
    "read",
    "read_batch",
    "write",
    "write_batch",
    "probe",
    "halt",
)

LHP_V2_CAPABILITIES: tuple[str, ...] = (*LHP_V1_CAPABILITIES, "read_frame", "release_frame")

# v2 optional capability tokens negotiated at hello.
CAP_PERMIT_V1 = "permit.v1"
CAP_CONTROL_V1 = "control.v1"

LHP_METHODS: tuple[str, ...] = (
    "hello",
    *LHP_V2_CAPABILITIES,
    # control.v1 methods are only accepted when the capability is advertised.
    "control.open",
    "control.renew",
    "control.update",
    "control.close",
    "control.status",
)
LHP_CAPABILITIES: tuple[str, ...] = LHP_V2_CAPABILITIES
"""Capabilities of the newest protocol; hello advertises the negotiated version's set."""


def capabilities_for_version(version: int) -> tuple[str, ...]:
    """Return the method capabilities available in a negotiated protocol version."""
    if version == 1:
        return LHP_V1_CAPABILITIES
    if version == 2:
        return LHP_V2_CAPABILITIES
    return ()


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
    "CAP_CONTROL_V1",
    "CAP_PERMIT_V1",
    "ERR_BAD_REQUEST",
    "ERR_CONNECTION_CLOSED",
    "ERR_FRAME_TOO_LARGE",
    "ERR_INTERNAL_ERROR",
    "ERR_PROTOCOL_MISMATCH",
    "ERR_TIMEOUT",
    "ERR_TRANSPORT_ERROR",
    "ERR_UNAUTHORIZED",
    "ERR_UNKNOWN_CHANNEL",
    "ERR_UNKNOWN_DEVICE",
    "ERR_UNSUPPORTED_CAPABILITY",
    "FAILURE_BATCH_ABORTED",
    "FAILURE_DEVICE_DEGRADED",
    "FAILURE_INTERLOCK_UNSATISFIED",
    "FAILURE_NOT_WRITABLE",
    "FAILURE_PERMIT_AUDIENCE_MISMATCH",
    "FAILURE_PERMIT_DIGEST_MISMATCH",
    "FAILURE_PERMIT_EXPIRED",
    "FAILURE_PERMIT_INVALID",
    "FAILURE_PERMIT_REPLAYED",
    "FAILURE_PERMIT_REQUIRED",
    "FAILURE_REMOTE_CONTROL_UNSUPPORTED",
    "FAILURE_SAFETY_DENIED",
    "FAILURE_WRITES_DISABLED",
    "LHP_CAPABILITIES",
    "LHP_DEFAULT_PORT",
    "LHP_METHODS",
    "LHP_PROTOCOL_VERSION",
    "LHP_SUPPORTED_VERSIONS",
    "LHP_V1_CAPABILITIES",
    "LHP_V2_CAPABILITIES",
    "MAX_FRAME_BYTES",
    "MAX_FRAME_DATA_BYTES",
    "MAX_FRAME_RATE_HZ",
    "MAX_FRAME_WIDTH",
    "SAFE_FRAME_MEDIA_TYPES",
    "WRITE_EFFECT_UNKNOWN_CODES",
    "LhpProtocolError",
    "LhpRemoteError",
    "capabilities_for_version",
    "decode_frame",
    "encode_frame",
    "error_response",
    "is_valid_device_id",
    "make_request",
    "ok_response",
    "response_error",
]
