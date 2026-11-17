# Copyright (c) Alibaba, Inc. and its affiliates.
"""Declared-envelope checks for Piper commands: refuse a target, clamp a hold.

The vendor clamps every commanded joint into its own firmware limit map
without saying so, so a value outside the declared envelope either executes
something other than what was planned or, for a *held* axis, moves an axis
nobody named. These checks decide that before any frame is sent.
"""
from __future__ import annotations

from typing import Any

from leapflow.hardware.drivers._piper_motion import HoldRefused, numeric


def target_refusal(channel: Any, channel_id: str, value: Any) -> tuple[str, str] | None:
    """``(error, failure_code)`` refusing a commanded value, or None when it is allowed."""
    number = numeric(value)
    if number is None:
        return f"invalid value for {channel_id!r}: {value!r}", "invalid_value"
    envelope = channel.envelope if channel is not None else None
    if envelope is not None and envelope.declared and not envelope.contains(number):
        return (f"{channel_id}={number} is outside the declared envelope "
                f"[{envelope.min_value}, {envelope.max_value}]", "value_out_of_range")
    return None


def held_value(channel: Any, channel_id: str, value: float) -> float:
    """Keep a held axis inside its declared envelope, or refuse to send it.

    Holding an axis means commanding it where it already is. The vendor clamps
    whatever it is given into its own firmware limit map, so a hold outside the
    envelope does not hold at all: the arm travels to the limit while the write
    reports success, and the caller never named that axis.

    Inside the channel's own declared tolerance the clamp is applied here, so
    what the driver sends is what the arm executes. Beyond it ``HoldRefused`` is
    raised before any frame goes out, naming the axis, its reading, and its
    limit -- the arm has to be brought inside the envelope by a command that
    names it.
    """
    envelope = channel.envelope if channel is not None else None
    if envelope is None or not envelope.declared or envelope.contains(value):
        return value
    clamped = value
    if envelope.min_value is not None:
        clamped = max(clamped, envelope.min_value)
    if envelope.max_value is not None:
        clamped = min(clamped, envelope.max_value)
    drift = abs(clamped - value)
    if drift > float(envelope.tolerance or 0.0):
        unit = channel.unit if channel is not None else ""
        raise HoldRefused(
            f"{channel_id} stands at {value:.3f}{' ' + unit if unit else ''}, "
            f"{drift:.3f} outside its declared envelope "
            f"[{envelope.min_value}, {envelope.max_value}]; holding it would move an "
            "axis this command does not name, so name it and bring it inside first",
            failure_code="piper_hold_outside_envelope",
        )
    return clamped


__all__ = ["held_value", "target_refusal"]
