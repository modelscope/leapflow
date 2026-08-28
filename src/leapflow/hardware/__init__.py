"""Hardware Context Protocol -- safe agent operation of physical devices.

The protocol splits device integration along the axis of what can be known:

``context`` (stable)
    What an agent must know to operate a device safely -- quantities, units,
    operating envelopes, interlocks, settling time, reversibility. Determined by
    physics and by governance requirements, so it is defined here and frozen.

``transport`` (volatile)
    How a command reaches the device. Determined by whichever southbound standard
    or vendor SDK is in play, so it lives behind a six-method Protocol and can be
    swapped without touching anything else.

Two seams follow from that split. ``providers/`` answers where device knowledge
comes from; ``transports/`` answers how an operation executes. Both are
``kind -> factory`` lookup tables, so supporting a new standard is a module plus a
row -- and no upstream concept is permitted to leak into ``context.py``.

Nothing here is enabled by default: without a bound registry the plugin exposes no
tools, and the rest of the system behaves exactly as it did before.
"""

from __future__ import annotations

from leapflow.hardware.context import (
    HC_VERSION,
    SUPPORTED_HC_VERSIONS,
    Channel,
    ContextProvenance,
    ContextSource,
    Direction,
    Envelope,
    HardwareContext,
    HardwareEffect,
    Interlock,
    Quality,
    TransportRef,
)
from leapflow.hardware.registry import (
    AdmissionNote,
    HardwareRegistry,
    HardwareSettings,
    LoadReport,
    UnverifiedContextPolicy,
)
from leapflow.hardware.transport import (
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_PARTIAL,
    SIDE_EFFECT_UNKNOWN,
    HardwareTransport,
    Reading,
    TransportError,
    TransportStatus,
    WriteOutcome,
)

__all__ = [
    "HC_VERSION",
    "SIDE_EFFECT_COMMITTED",
    "SIDE_EFFECT_NONE",
    "SIDE_EFFECT_PARTIAL",
    "SIDE_EFFECT_UNKNOWN",
    "SUPPORTED_HC_VERSIONS",
    "AdmissionNote",
    "Channel",
    "ContextProvenance",
    "ContextSource",
    "Direction",
    "Envelope",
    "HardwareContext",
    "HardwareEffect",
    "HardwareRegistry",
    "HardwareSettings",
    "HardwareTransport",
    "Interlock",
    "LoadReport",
    "Quality",
    "Reading",
    "TransportError",
    "TransportRef",
    "TransportStatus",
    "UnverifiedContextPolicy",
    "WriteOutcome",
]
