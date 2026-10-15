# Copyright (c) Alibaba, Inc. and its affiliates.
"""Interlock evaluation: fail-closed checks for physical write preconditions.

Extracted from ``HardwareTools._failed_interlocks`` so that both the host-side
tools layer and the LHP node server share one evaluation path. The semantics
are identical: a missing interlock, an unreadable source, or an exception all
count as unsatisfied. "Cannot check" and "not satisfied" have the same
consequence.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def failed_interlocks(
    registry: Any,
    context: Any,
    channel: Any,
) -> tuple[str, ...]:
    """Return the interlocks that do not currently hold.

    Fails closed in every uncertain case: a missing interlock, an unreadable
    source channel, or a read that raises all count as unsatisfied. "Cannot
    check" and "not satisfied" must have the same consequence.

    When ``SafetyPolicy.require_safety_interlock`` is set on the context,
    every interlock declared on the device is checked, not just those on the
    channel's envelope.
    """
    # Per-channel interlocks
    required = list(channel.envelope.requires_interlocks or ())

    # Device-level blanket requirement
    safety = getattr(context, "safety", None)
    if safety is not None and getattr(safety, "require_safety_interlock", False):
        # Add all device-declared interlocks not already in the channel set
        all_interlocks = getattr(context, "interlocks", None)
        if callable(all_interlocks):
            # context.interlocks is a property returning the dict/mapping
            pass  # interlocks are accessed via context.interlock(name)
        # Walk the context's declared channels to find all interlock names
        for ch in getattr(context, "channels", ()):
            ch_obj = ch if not isinstance(ch, str) else context.channel(ch)
            if ch_obj is not None:
                for name in getattr(getattr(ch_obj, "envelope", None), "requires_interlocks", ()) or ():
                    if name not in required:
                        required.append(name)

    if not required:
        return ()

    failed: list[str] = []
    for name in required:
        lock = context.interlock(name)
        if lock is None:
            failed.append(name)
            continue
        try:
            transport = await registry.transport(context.device_id)
            reading = await transport.read(lock.channel_id)
        except Exception as exc:  # noqa: BLE001 - see below
            # Deliberately wider than TransportError. A driver is *supposed* to
            # raise only that, but this is a safety precondition: if a
            # third-party driver raises anything else, the correct reading is
            # "the interlock could not be checked", not "the turn crashes".
            logger.warning(
                "Interlock %r could not be evaluated on %s.%s (%s); treating as unsatisfied",
                name,
                context.device_id,
                lock.channel_id,
                exc,
                exc_info=True,
            )
            failed.append(name)
            continue
        if not lock.evaluate(reading.value):
            failed.append(name)
    return tuple(failed)


__all__ = ["failed_interlocks"]
