"""MsgPack-RPC framing, method constants, and event bus protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol, Tuple, runtime_checkable

import msgpack

PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class RpcError(Exception):
    """Structured RPC error."""

    code: str
    message: str
    details: Dict[str, Any]

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.code}: {self.message}"


def encode_packet(payload: Dict[str, Any]) -> bytes:
    """Encode a dict as length-prefixed MsgPack."""
    body = msgpack.packb(payload, use_bin_type=True)
    return len(body).to_bytes(4, "big", signed=False) + body


def decode_packet(data: bytes) -> Tuple[Dict[str, Any], bytes]:
    """Decode one length-prefixed MsgPack object from a buffer.

    Returns:
        (obj, remaining_bytes)
    """
    if len(data) < 4:
        raise ValueError("buffer too small for frame header")
    (n,) = (int.from_bytes(data[0:4], "big", signed=False),)
    if len(data) < 4 + n:
        raise ValueError("buffer too small for frame body")
    body = data[4 : 4 + n]
    obj = msgpack.unpackb(body, raw=False, strict_map_key=False)
    if not isinstance(obj, dict):
        raise ValueError("RPC frame must unpack to a dict")
    rest = data[4 + n :]
    return obj, rest


class Methods:
    """RPC method names."""

    PING = "ping"

    FILE_LIST = "file.list"
    FILE_MOVE = "file.move"
    FILE_COPY = "file.copy"
    FILE_DELETE = "file.delete"

    FS_SUBSCRIBE = "fs.subscribe"

    AX_TREE = "ax.tree"
    AX_PERFORM = "ax.perform"
    AX_SCROLL = "ax.scroll"

    APP_LAUNCH = "app.launch"
    APP_ACTIVATE = "app.activate"
    APP_LIST = "app.list"

    CLIPBOARD_GET = "clipboard.get"
    CLIPBOARD_SET = "clipboard.set"
    CLIPBOARD_LAST_CHANGE = "clipboard.last_change"

    INPUT_TYPE_TEXT = "input.type_text"
    INPUT_SHORTCUT = "input.shortcut"
    INPUT_SELECT_TEXT = "input.select_text"

    SYSTEM_INFO = "system.info"
    SYSTEM_MANIFEST = "system.manifest"

    INTENT_DISCOVER = "intent.discover"
    INTENT_PERFORM = "intent.perform"

    SCREEN_START_CAPTURE = "screen.start_capture"
    SCREEN_STOP_CAPTURE = "screen.stop_capture"
    SCREEN_CAPTURE_FRAME = "screen.capture_frame"
    SCREEN_PERMISSION_STATUS = "screen.permission_status"

    RECORDING_START = "recording.start"
    RECORDING_STOP = "recording.stop"


class EventTypes:
    """Server-pushed event type constants."""

    FS_CHANGE = "event.fs_change"
    CLIPBOARD_CHANGE = "event.clipboard_change"
    APP_FOCUS_CHANGE = "event.app_focus_change"
    UI_ACTION = "event.ui_action"
    CONTEXT_CHANGE = "event.context_change"
    SCREEN_FRAME_CAPTURED = "event.screen_frame_captured"


@runtime_checkable
class HostRpc(Protocol):
    """Shared interface for all host RPC callers (DIP: skills depend on this)."""

    async def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any: ...


EventHandler = Callable[[str, Dict[str, Any]], Awaitable[None]]


def make_request(request_id: str, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "v": PROTOCOL_VERSION,
        "type": "request",
        "id": request_id,
        "method": method,
        "params": params or {},
    }


def make_response_ok(request_id: str, result: Any) -> Dict[str, Any]:
    return {"v": PROTOCOL_VERSION, "type": "response", "id": request_id, "ok": True, "result": result}


def make_response_err(request_id: str, code: str, message: str, details: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "v": PROTOCOL_VERSION,
        "type": "response",
        "id": request_id,
        "ok": False,
        "error": {"code": code, "message": message, "details": details or {}},
    }


def make_event(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Construct a server-pushed event frame (no request ID, no response expected)."""
    return {
        "v": PROTOCOL_VERSION,
        "type": "event",
        "event": event_type,
        "payload": payload,
    }
