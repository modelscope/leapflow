# Copyright (c) Alibaba, Inc. and its affiliates.
"""Request-parameter validation shared by the LHP node method handlers.

Every helper raises ``LhpProtocolError`` (``bad_request``) on a malformed value,
which the server's dispatcher turns into an error frame: a request the node
cannot parse never reaches a device.
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

from leapflow.hardware.lhp.protocol import MAX_FRAME_AGE_MS, MAX_FRAME_SEQUENCE, LhpProtocolError


def require_str(params: Mapping[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value:
        raise LhpProtocolError(f"{key!r} must be a non-empty string")
    return value


def optional_str(params: Mapping[str, Any], key: str, *, max_len: int = 0) -> str:
    value = params.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise LhpProtocolError(f"{key!r} must be a string")
    if max_len and len(value) > max_len:
        raise LhpProtocolError(f"{key!r} must be at most {max_len} characters")
    return value


def optional_mapping(params: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = params.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise LhpProtocolError(f"{key!r} must be an object")
    return dict(value)


def require_bool(params: Mapping[str, Any], key: str) -> bool:
    value = params.get(key)
    if not isinstance(value, bool):
        raise LhpProtocolError(f"{key!r} must be a boolean")
    return value


def optional_bool(params: Mapping[str, Any], key: str) -> bool:
    value = params.get(key, False)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise LhpProtocolError(f"{key!r} must be a boolean")
    return value


def optional_int(params: Mapping[str, Any], key: str, *, upper: int) -> int:
    value = params.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > upper:
        raise LhpProtocolError(f"{key!r} must be an integer from 0 through {upper}")
    return value


def optional_float(params: Mapping[str, Any], key: str, *, upper: float) -> float:
    value = params.get(key, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LhpProtocolError(f"{key!r} must be a number from 0 through {upper:g}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or numeric > upper:
        raise LhpProtocolError(f"{key!r} must be a number from 0 through {upper:g}")
    return numeric


def require_channel_ids(params: Mapping[str, Any]) -> list[str]:
    raw = params.get("channel_ids")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise LhpProtocolError("'channel_ids' must be a non-empty list of channel ids")
    channel_ids: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            raise LhpProtocolError("every entry of 'channel_ids' must be a non-empty string")
        channel_ids.append(item)
    return channel_ids


def require_commands(params: Mapping[str, Any]) -> list[tuple[str, Any]]:
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


def freshness_bounds(params: Mapping[str, Any]) -> dict[str, Any]:
    """Parse ``min_sequence`` / ``max_age_ms`` into node-local ``read_frame`` keywords.

    ``max_age_ms`` becomes an ``after_monotonic`` on *this* node's clock. Absence
    and ``0`` differ: an explicit ``0`` asks for a capture started after the request
    arrived, which is how a host says "after the motion I just commanded".
    """
    bounds: dict[str, Any] = {}
    min_sequence = optional_int(params, "min_sequence", upper=MAX_FRAME_SEQUENCE)
    if min_sequence:
        bounds["min_sequence"] = min_sequence
    if params.get("max_age_ms") is not None:
        max_age_ms = optional_float(params, "max_age_ms", upper=MAX_FRAME_AGE_MS)
        bounds["after_monotonic"] = time.monotonic() - max_age_ms / 1000.0
    return bounds


__all__ = [
    "freshness_bounds",
    "optional_float",
    "optional_int",
    "optional_mapping",
    "optional_str",
    "require_bool",
    "require_channel_ids",
    "require_commands",
    "require_str",
]
