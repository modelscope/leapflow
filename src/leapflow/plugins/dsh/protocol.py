"""Versioned NDJSON protocol for the restricted DSH Node worker."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 1_000_000


class DshProtocolError(ValueError):
    """A worker message violated the bridge protocol."""


@dataclass(frozen=True)
class DshRequest:
    request_id: str
    method: str
    payload: dict[str, Any] = field(default_factory=dict)
    version: int = PROTOCOL_VERSION
    type: str = "request"

    def to_json(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class DshResponse:
    request_id: str
    ok: bool
    result: Any = None
    error: str = ""
    error_type: str = ""
    version: int = PROTOCOL_VERSION
    type: str = "response"

    @classmethod
    def from_message(cls, message: dict[str, Any]) -> "DshResponse":
        _validate_common(message, expected_type="response")
        if not isinstance(message.get("ok"), bool):
            raise DshProtocolError("response.ok must be a boolean")
        return cls(
            request_id=str(message["request_id"]),
            ok=bool(message["ok"]),
            result=message.get("result"),
            error=str(message.get("error") or ""),
            error_type=str(message.get("error_type") or ""),
        )


@dataclass(frozen=True)
class CapabilityRequest:
    request_id: str
    parent_request_id: str
    capability: str
    arguments: dict[str, Any]

    @classmethod
    def from_message(cls, message: dict[str, Any]) -> "CapabilityRequest":
        _validate_common(message, expected_type="capability_request")
        parent = message.get("parent_request_id")
        capability = message.get("capability")
        arguments = message.get("arguments")
        if not isinstance(parent, str) or not parent:
            raise DshProtocolError("capability_request.parent_request_id is required")
        if not isinstance(capability, str) or not capability:
            raise DshProtocolError("capability_request.capability is required")
        if not isinstance(arguments, dict):
            raise DshProtocolError("capability_request.arguments must be an object")
        return cls(
            request_id=str(message["request_id"]),
            parent_request_id=parent,
            capability=capability,
            arguments=arguments,
        )


def capability_response(
    request_id: str,
    *,
    ok: bool,
    result: Any = None,
    error: str = "",
    error_type: str = "",
) -> str:
    return json.dumps(
        {
            "version": PROTOCOL_VERSION,
            "type": "capability_response",
            "request_id": request_id,
            "ok": ok,
            "result": result,
            "error": error,
            "error_type": error_type,
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )


def parse_message(raw: bytes, *, max_bytes: int = MAX_LINE_BYTES) -> dict[str, Any]:
    if len(raw) > max_bytes:
        raise DshProtocolError(f"worker message exceeds {max_bytes} bytes")
    try:
        decoded = raw.decode("utf-8")
        message = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DshProtocolError(f"invalid worker JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise DshProtocolError("worker message must be a JSON object")
    _validate_common(message)
    return message


def _validate_common(
    message: dict[str, Any], *, expected_type: str | None = None
) -> None:
    if message.get("version") != PROTOCOL_VERSION:
        raise DshProtocolError(
            f"unsupported DSH bridge protocol version: {message.get('version')!r}"
        )
    message_type = message.get("type")
    if not isinstance(message_type, str) or not message_type:
        raise DshProtocolError("worker message.type is required")
    if expected_type is not None and message_type != expected_type:
        raise DshProtocolError(
            f"expected worker message type {expected_type!r}, got {message_type!r}"
        )
    request_id = message.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise DshProtocolError("worker message.request_id is required")
