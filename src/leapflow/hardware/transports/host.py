"""Host transport: reads this machine's resource channels in-process.

The counterpart to ``providers/host_provider.py``, and like it a thin shell over
``host_metrics.HostMetrics`` -- the probe table is the single source of truth for
what exists and how to read it, so this file contains no knowledge of what a CPU or
a mount point is.

Every write path refuses. That is not a limitation to be lifted later: a discovered
declaration carries no envelope a person is accountable for, and commanding a host
resource through an unverified declaration is exactly what the admission rules
exist to prevent. A host control surface would arrive as its own declared device
with its own transport, not by loosening this one.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from leapflow.hardware.context import HardwareContext, Quality
from leapflow.hardware.host_metrics import HostMetrics
from leapflow.hardware.transport import (
    SIDE_EFFECT_NONE,
    Reading,
    TransportError,
    TransportStatus,
    WriteOutcome,
)


class HostTransport:
    """Serves host resource readings from the shared probe table."""

    kind = "host"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._metrics = HostMetrics(config)
        self._context: HardwareContext | None = None
        self._connected = False
        self._sequence: dict[str, int] = {}
        self._opened_at = 0.0

    # ── Lifecycle ──

    async def open(self, context: HardwareContext) -> TransportStatus:
        """Bind the context. Idempotent, and touches no hardware.

        There is nothing to connect to -- the probes are in-process reads -- so
        "open" means the probe table has been enumerated and the channel ids the
        context declares can be resolved.
        """
        self._context = context
        self._connected = True
        self._opened_at = time.monotonic()
        self._metrics.probes()
        return await self.probe()

    async def close(self) -> TransportStatus:
        self._connected = False
        return TransportStatus(connected=False, halt_supported=False, detail="closed")

    async def probe(self) -> TransportStatus:
        known = self._metrics.known_channels()
        declared = {c.channel_id for c in (self._context.channels if self._context else ())}
        # Reported rather than logged: a probe that vanished after discovery (an
        # unmounted volume, a sensor that went away) is the difference between an
        # empty trace and a broken one, and only this comparison can tell them apart.
        missing = sorted(declared - known)
        return TransportStatus(
            connected=self._connected,
            halt_supported=False,
            detail=f"host probes: {len(known)} available",
            metadata={
                "backends": list(self._metrics.backends),
                "available_channels": len(known),
                "unavailable_channels": missing,
                "uptime_s": round(time.monotonic() - self._opened_at, 1) if self._connected else 0.0,
            },
        )

    async def halt(self) -> TransportStatus:
        """Report that a host has no emergency stop.

        Not an error: ``halt_supported=False`` is the declared capability that makes
        admission refuse writable channels, so answering honestly here is what keeps
        the rest of the system consistent.
        """
        return TransportStatus(
            connected=self._connected,
            halt_supported=False,
            detail="a host resource has no emergency stop",
        )

    # ── Data plane ──

    async def read(self, channel_id: str) -> Reading:
        if not self._connected:
            raise TransportError(
                f"host transport is not open (channel {channel_id!r})",
                failure_code="transport_not_open",
            )
        channel = self._context.channel(channel_id) if self._context is not None else None
        if channel is None and channel_id not in self._metrics.known_channels():
            raise TransportError(f"unknown channel {channel_id!r}", failure_code="unknown_channel")

        sequence = self._sequence.get(channel_id, 0) + 1
        self._sequence[channel_id] = sequence
        value = self._metrics.read(channel_id)
        return Reading(
            device_id=self._context.device_id if self._context is not None else "",
            channel_id=channel_id,
            value=value,
            quantity=channel.quantity if channel is not None else "",
            unit=channel.unit if channel is not None else "",
            sequence=sequence,
            # A probe that cannot answer right now is suspect, not zero. Reporting a
            # value would put a fabricated number into the downsample window, where
            # nothing downstream could ever tell it apart from a measurement.
            quality=Quality.OK.value if value is not None else Quality.SUSPECT.value,
        )

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        """Refuse every write, proving no effect landed.

        ``SIDE_EFFECT_NONE`` is provable here, unlike in most transports: the call
        never reaches anything, so recovery is free to treat it as a clean refusal
        rather than an uncertain one.
        """
        return WriteOutcome(
            ok=False,
            side_effect_state=SIDE_EFFECT_NONE,
            error=(
                f"host channel {channel_id!r} is read-only: discovered declarations carry no "
                "operating envelope a person is accountable for, so they cannot authorize a write"
            ),
            failure_code="host_channel_read_only",
        )


def build_transport(config: Mapping[str, Any] | None = None) -> HostTransport:
    """Factory registered in the transport table."""
    return HostTransport(config)


__all__ = ["HostTransport", "build_transport"]
