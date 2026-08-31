"""Hardware context: the declarative half of the Hardware Context Protocol.

This module is deliberately free of any transport, vendor, or upstream-standard
concept. It describes what an agent must know to operate a device safely --
facts determined by physics and by LeapFlow's governance requirements, not by
whatever wire protocol eventually carries the command.

Keeping it that way is the single architectural red line of the protocol: an
upstream hardware standard changes ``providers/`` and ``transports/`` only, never
this module. A test in ``tests/test_architecture_contracts.py`` enforces it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

HC_VERSION = "hc.v0"
"""Protocol version of a device declaration.

``hc.v0`` is a pre-standard draft and makes no backward-compatibility promise.
The registry refuses declarations carrying an unknown version rather than
guessing: declarations are durable user assets, and silently rewriting one is
worse than rejecting it with a reason.
"""

SUPPORTED_HC_VERSIONS: frozenset[str] = frozenset({HC_VERSION})

_DEFAULT_HYSTERESIS_SPAN_FRACTION = 0.01
"""Settle band used when a channel declares no quantization, as a share of span."""

_MAX_HYSTERESIS_SPAN_FRACTION = 0.25
"""Upper bound on the settle band, so a coarse quantization cannot close it.

Without the cap, a channel whose quantization approaches its own span would
produce a settle band wider than the range itself: the value could never clear
it, and a breach would be reported as permanent.
"""


class ContextSource(str, Enum):
    """Where a hardware context came from -- drives how much it is trusted."""

    DECLARED = "declared"
    """Hand-written declaration file."""

    INTERVIEW = "interview"
    """Captured by asking the operator; awaiting human confirmation."""

    DISCOVERED = "discovered"
    """Introspected from a transport (e.g. a CLI's own help output)."""

    IMPORTED = "imported"
    """Mapped from an upstream hardware standard's device descriptor."""


class Direction(str, Enum):
    """Whether a channel can be read, written, or both."""

    READ = "read"
    WRITE = "write"
    READWRITE = "readwrite"


class HardwareEffect(str, Enum):
    """Physical effect class of a channel operation.

    Separate from ``ActionEffect`` because these risk profiles have no software
    analogue: dispensing consumes an irreversible resource, actuating carries
    kinetic energy, emitting radiates. Each maps to its own ``ActionKind`` so the
    risk classifier dispatches on a decision rather than on a fallback value.
    """

    READ = "read"
    CONFIGURE = "configure"
    ACTUATE = "actuate"
    DISPENSE = "dispense"
    EMIT = "emit"

    @classmethod
    def writable(cls) -> frozenset[str]:
        """Return the effect classes that change the physical world."""
        return frozenset({cls.CONFIGURE.value, cls.ACTUATE.value, cls.DISPENSE.value, cls.EMIT.value})


class Quality(str, Enum):
    """Verdict on whether a sample can be trusted."""

    OK = "ok"
    SUSPECT = "suspect"
    STALE = "stale"
    SATURATED = "saturated"


@dataclass(frozen=True)
class ContextProvenance:
    """Provenance and verification state of one hardware context.

    A pseudo-implementation of an unpublished standard must be honest about
    being a guess, and this type is that honesty. An unverified context cannot
    authorize a write under the default policy, and a lossy import records
    exactly which upstream fields could not be mapped -- so a downgrade in
    fidelity is visible in ``hw_describe`` instead of being silently absorbed.
    """

    source: str = ContextSource.DECLARED.value
    verified_by: str = ""
    verified_at: float = 0.0
    upstream_version: str = ""
    lossy_fields: tuple[str, ...] = ()
    notes: str = ""

    @property
    def is_verified(self) -> bool:
        """Return whether a human has taken responsibility for this context."""
        return bool(self.verified_by.strip())

    @property
    def is_lossy(self) -> bool:
        return bool(self.lossy_fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at,
            "upstream_version": self.upstream_version,
            "lossy_fields": list(self.lossy_fields),
            "notes": self.notes,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ContextProvenance":
        data = data or {}
        return cls(
            source=str(data.get("source") or ContextSource.DECLARED.value),
            verified_by=str(data.get("verified_by") or ""),
            verified_at=_as_float(data.get("verified_at"), default=0.0) or 0.0,
            upstream_version=str(data.get("upstream_version") or ""),
            lossy_fields=tuple(str(item) for item in data.get("lossy_fields") or ()),
            notes=str(data.get("notes") or ""),
        )


@dataclass(frozen=True)
class Envelope:
    """Declared physical operating limits for one channel.

    This is the machine-readable form of knowledge that previously lived in
    paper manuals and tacit expertise. It is consumed by the risk classifier
    *before* approval and never enforced inside a transport: a transport that
    policed itself could not be audited, and would be bypassed by every other
    control plane that reaches the same device.

    ``declared`` is explicit rather than inferred from "all fields are None",
    because an undeclared envelope and an intentionally unbounded one must not
    look alike. An undeclared envelope on a writable channel is a hardline deny.
    """

    declared: bool = False
    min_value: float | None = None
    max_value: float | None = None
    max_rate: float | None = None
    quantization: float | None = None
    settling_time_s: float = 0.0
    reversible: bool = False
    requires_interlocks: tuple[str, ...] = ()
    notes: str = ""

    @property
    def is_numeric(self) -> bool:
        """Return whether this envelope constrains a numeric quantity.

        Derived from the presence of numeric bounds rather than declared
        separately, so a state channel (a boolean, a mode) simply omits them
        instead of having to say it is not numeric.
        """
        return any(
            bound is not None
            for bound in (self.min_value, self.max_value, self.max_rate, self.quantization)
        )

    def contains(self, value: Any, *, margin: float = 0.0) -> bool:
        """Return True when *value* lies inside the declared bounds.

        Three cases, and the middle one is the one that matters. An undeclared
        envelope admits nothing. A *numeric* envelope handed a non-numeric value
        (a string, a boolean, NaN, infinity) admits nothing either: the bounds
        cannot be evaluated, and "cannot evaluate" must carry the same weight as
        "out of range" or an unparseable command would slip past the one check
        standing between it and the device. Only an envelope with no numeric
        bounds -- a state channel -- admits an arbitrary value.

        ``margin`` narrows the band inward. It exists so a breach can end on a
        stricter test than it began on (see ``settle_margin``); every safety
        caller leaves it at zero, because a hardline must be evaluated against
        the limit a human actually declared.
        """
        if not self.declared:
            return False
        numeric = as_numeric(value)
        if numeric is None:
            return not self.is_numeric
        inward = max(0.0, margin)
        if self.min_value is not None and numeric < self.min_value + inward:
            return False
        if self.max_value is not None and numeric > self.max_value - inward:
            return False
        return True

    @property
    def settle_margin(self) -> float:
        """Return the inward margin a value must clear before a breach is over.

        Derived, never declared. A boundary crossing and a boundary *hover* are
        different observations, but a plain in/out test cannot tell them apart:
        a sensor resting on its limit alternates threshold_exceeded and settled
        at the sampling rate, which is the same flood the event layer exists to
        prevent -- and it buries the one crossing that mattered.

        ``quantization`` is the natural width when declared: a change smaller
        than the device's own resolution is not a change. Absent it, a small
        fraction of the declared span is used. The result is capped so the
        settle band can never collapse to nothing on a narrow envelope, which
        would replace flapping with a breach that never clears.
        """
        span = _declared_span(self)
        cap = span * _MAX_HYSTERESIS_SPAN_FRACTION
        if self.quantization is not None and self.quantization > 0:
            return min(self.quantization, cap) if cap > 0 else self.quantization
        return span * _DEFAULT_HYSTERESIS_SPAN_FRACTION

    def rate_wait_s(self, *, delta: float, elapsed_s: float) -> float:
        """Return how long to wait before a change of *delta* respects ``max_rate``.

        Zero means it may proceed now. This is the single implementation of the slew
        constraint, kept on the envelope because it is a property of the declaration
        rather than of whoever happens to be enforcing it.

        A zero or negative interval cannot be measured, so it is treated as no time
        having passed rather than as time enough: an unmeasurable rate must not pass
        as a safe one.
        """
        if self.max_rate is None or self.max_rate <= 0.0:
            return 0.0
        magnitude = abs(delta)
        if magnitude == 0.0:
            return 0.0
        required = magnitude / self.max_rate
        return max(0.0, required - max(0.0, elapsed_s))

    def rate_exceeded(self, *, delta: float, elapsed_s: float) -> bool:
        """Return True when a change of *delta* over *elapsed_s* is too fast."""
        return self.rate_wait_s(delta=delta, elapsed_s=elapsed_s) > 0.0

    def band_key(self) -> str:
        """Return a stable identifier for this envelope band.

        Participates in the approval grant identity, so widening a declared
        envelope invalidates the narrower grant it was given under instead of
        silently inheriting it.
        """
        if not self.declared:
            return "undeclared"
        parts = (
            _format_bound(self.min_value),
            _format_bound(self.max_value),
            _format_bound(self.max_rate),
            "rev" if self.reversible else "irrev",
        )
        return ":".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared": self.declared,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "max_rate": self.max_rate,
            "quantization": self.quantization,
            "settling_time_s": self.settling_time_s,
            "reversible": self.reversible,
            "requires_interlocks": list(self.requires_interlocks),
            "notes": self.notes,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "Envelope":
        data = data or {}
        return cls(
            declared=bool(data.get("declared", False)),
            min_value=_as_float(data.get("min_value")),
            max_value=_as_float(data.get("max_value")),
            max_rate=_as_float(data.get("max_rate")),
            quantization=_as_float(data.get("quantization")),
            settling_time_s=_as_float(data.get("settling_time_s"), default=0.0) or 0.0,
            reversible=bool(data.get("reversible", False)),
            requires_interlocks=tuple(str(item) for item in data.get("requires_interlocks") or ()),
            notes=str(data.get("notes") or ""),
        )


@dataclass(frozen=True)
class Channel:
    """One readable or writable endpoint on a device.

    ``sample_rate_hz > 0`` is the only switch that matters downstream: it decides
    whether this channel becomes a streaming signal source or stays a
    request/response tool call. No device-type enumeration is involved anywhere.
    """

    channel_id: str
    direction: str = Direction.READ.value
    quantity: str = ""
    unit: str = ""
    effect: str = HardwareEffect.READ.value
    envelope: Envelope = field(default_factory=Envelope)
    sample_rate_hz: float = 0.0
    verify_after_write: bool = False
    description: str = ""

    @property
    def is_writable(self) -> bool:
        return self.direction in {Direction.WRITE.value, Direction.READWRITE.value}

    @property
    def is_readable(self) -> bool:
        return self.direction in {Direction.READ.value, Direction.READWRITE.value}

    @property
    def is_streaming(self) -> bool:
        return self.sample_rate_hz > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "direction": self.direction,
            "quantity": self.quantity,
            "unit": self.unit,
            "effect": self.effect,
            "envelope": self.envelope.to_dict(),
            "sample_rate_hz": self.sample_rate_hz,
            "verify_after_write": self.verify_after_write,
            "description": self.description,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Channel":
        return cls(
            channel_id=str(data.get("channel_id") or ""),
            direction=str(data.get("direction") or Direction.READ.value),
            quantity=str(data.get("quantity") or ""),
            unit=str(data.get("unit") or ""),
            effect=str(data.get("effect") or HardwareEffect.READ.value),
            envelope=Envelope.from_mapping(data.get("envelope")),
            sample_rate_hz=_as_float(data.get("sample_rate_hz"), default=0.0) or 0.0,
            verify_after_write=bool(data.get("verify_after_write", False)),
            description=str(data.get("description") or ""),
        )

    def without_write(self) -> "Channel":
        """Return this channel demoted to read-only.

        Used by admission checks that must revoke write capability while keeping the
        channel readable, because reads remain valuable for diagnosis exactly when a
        device is not trusted to be commanded.

        ``effect`` is preserved rather than reset to ``read``. The declared effect is
        what the channel is *for*, and erasing it would make a demoted channel report
        an effect-class mismatch instead of the demotion that actually blocked it --
        sending whoever reads the error to the wrong place.
        """
        if not self.is_writable:
            return self
        return Channel(
            channel_id=self.channel_id,
            direction=Direction.READ.value,
            quantity=self.quantity,
            unit=self.unit,
            effect=self.effect,
            envelope=self.envelope,
            sample_rate_hz=self.sample_rate_hz,
            verify_after_write=False,
            description=self.description,
        )


@dataclass(frozen=True)
class Interlock:
    """A precondition that must hold before a guarded write is permitted.

    Expressed as a channel comparison rather than free text so it can be
    evaluated deterministically. Natural-language conditions are not accepted:
    a safety precondition that needs interpretation is not a precondition.
    """

    interlock_id: str
    channel_id: str
    operator: str = "eq"
    value: Any = True
    description: str = ""

    def evaluate(self, reading: Any) -> bool:
        """Return True when *reading* satisfies this interlock.

        An unknown operator or an incomparable pair returns False. Interlocks
        fail closed: "cannot tell" and "not satisfied" must have the same
        consequence, or an unevaluable interlock would become an open door.
        """
        comparator = _OPERATORS.get(self.operator)
        if comparator is None:
            return False
        try:
            return bool(comparator(reading, self.value))
        except TypeError:
            return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "interlock_id": self.interlock_id,
            "channel_id": self.channel_id,
            "operator": self.operator,
            "value": self.value,
            "description": self.description,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Interlock":
        return cls(
            interlock_id=str(data.get("interlock_id") or ""),
            channel_id=str(data.get("channel_id") or ""),
            operator=str(data.get("operator") or "eq"),
            value=data.get("value", True),
            description=str(data.get("description") or ""),
        )


@dataclass(frozen=True)
class TransportRef:
    """Reference to the transport that executes operations for a device."""

    kind: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "config": dict(self.config)}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "TransportRef":
        data = data or {}
        config = data.get("config")
        return cls(
            kind=str(data.get("kind") or ""),
            config=dict(config) if isinstance(config, Mapping) else {},
        )


@dataclass(frozen=True)
class HardwareContext:
    """Everything an agent must know about one device. The SSOT of the protocol.

    Tools, gate rules, stream sources, and the reference document are all
    derived from this object deterministically -- no model call, no heuristics --
    which is what makes each of them testable in isolation.
    """

    device_id: str
    hc_version: str = HC_VERSION
    display_name: str = ""
    transport: TransportRef = field(default_factory=TransportRef)
    channels: tuple[Channel, ...] = ()
    interlocks: tuple[Interlock, ...] = ()
    vendor: str = ""
    model: str = ""
    location: str = ""
    halt_supported: bool = False
    notes: str = ""
    provenance: ContextProvenance = field(default_factory=ContextProvenance)

    def channel(self, channel_id: str) -> Channel | None:
        return next((c for c in self.channels if c.channel_id == channel_id), None)

    def interlock(self, interlock_id: str) -> Interlock | None:
        return next((i for i in self.interlocks if i.interlock_id == interlock_id), None)

    @property
    def writable_channels(self) -> tuple[Channel, ...]:
        return tuple(c for c in self.channels if c.is_writable)

    @property
    def streaming_channels(self) -> tuple[Channel, ...]:
        return tuple(c for c in self.channels if c.is_streaming)

    @property
    def label(self) -> str:
        """Return a human-facing name, falling back to the id."""
        return self.display_name or self.device_id

    def with_channels(self, channels: tuple[Channel, ...]) -> "HardwareContext":
        """Return a copy carrying *channels*, used by admission demotions."""
        return HardwareContext(
            device_id=self.device_id,
            hc_version=self.hc_version,
            display_name=self.display_name,
            transport=self.transport,
            channels=channels,
            interlocks=self.interlocks,
            vendor=self.vendor,
            model=self.model,
            location=self.location,
            halt_supported=self.halt_supported,
            notes=self.notes,
            provenance=self.provenance,
        )

    def read_only(self) -> "HardwareContext":
        """Return a copy with every channel demoted to read-only."""
        return self.with_channels(tuple(c.without_write() for c in self.channels))

    def to_dict(self) -> dict[str, Any]:
        return {
            "hc_version": self.hc_version,
            "device_id": self.device_id,
            "display_name": self.display_name,
            "vendor": self.vendor,
            "model": self.model,
            "location": self.location,
            "halt_supported": self.halt_supported,
            "notes": self.notes,
            "transport": self.transport.to_dict(),
            "provenance": self.provenance.to_dict(),
            "channels": [c.to_dict() for c in self.channels],
            "interlocks": [i.to_dict() for i in self.interlocks],
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "HardwareContext":
        """Build a context from a parsed declaration.

        Structural validation lives in the registry, not here: this constructor
        is a faithful reader so that a malformed declaration can be reported
        with a reason instead of raising during parse.
        """
        channels = tuple(
            Channel.from_mapping(item)
            for item in data.get("channels") or ()
            if isinstance(item, Mapping)
        )
        interlocks = tuple(
            Interlock.from_mapping(item)
            for item in data.get("interlocks") or ()
            if isinstance(item, Mapping)
        )
        return cls(
            device_id=str(data.get("device_id") or ""),
            hc_version=str(data.get("hc_version") or ""),
            display_name=str(data.get("display_name") or ""),
            transport=TransportRef.from_mapping(data.get("transport")),
            channels=channels,
            interlocks=interlocks,
            vendor=str(data.get("vendor") or ""),
            model=str(data.get("model") or ""),
            location=str(data.get("location") or ""),
            halt_supported=bool(data.get("halt_supported", False)),
            notes=str(data.get("notes") or ""),
            provenance=ContextProvenance.from_mapping(data.get("provenance")),
        )


_OPERATORS: dict[str, Any] = {
    "eq": lambda left, right: left == right,
    "ne": lambda left, right: left != right,
    "lt": lambda left, right: left < right,
    "le": lambda left, right: left <= right,
    "gt": lambda left, right: left > right,
    "ge": lambda left, right: left >= right,
}


def as_numeric(value: Any, *, default: float | None = None) -> float | None:
    """Coerce *value* to a finite float, returning *default* when it is not one.

    Booleans are rejected on purpose: ``True`` is a state, not the number 1, and
    silently treating it as one would let a boolean slip through a range check on
    a numeric channel. NaN and infinity are rejected for the same reason -- they
    compare in ways that make every bound look satisfied.
    """
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        numeric = float(value)
        return default if math.isnan(numeric) or math.isinf(numeric) else numeric
    if isinstance(value, str):
        try:
            numeric = float(value.strip())
        except ValueError:
            return default
        return default if math.isnan(numeric) or math.isinf(numeric) else numeric
    return default


_as_float = as_numeric
"""Internal alias used by the declaration readers above."""


def _format_bound(value: float | None) -> str:
    return "-" if value is None else f"{value:g}"


def _declared_span(envelope: "Envelope") -> float:
    """Return the width of a two-sided declared range, or 0.0 when there isn't one.

    A one-sided or unbounded envelope has no span, and every caller must treat
    that as "no scale available" rather than as zero width.
    """
    if envelope.min_value is None or envelope.max_value is None:
        return 0.0
    span = envelope.max_value - envelope.min_value
    return span if span > 0 else 0.0


__all__ = [
    "HC_VERSION",
    "SUPPORTED_HC_VERSIONS",
    "Channel",
    "ContextProvenance",
    "ContextSource",
    "Direction",
    "Envelope",
    "HardwareContext",
    "HardwareEffect",
    "Interlock",
    "Quality",
    "TransportRef",
    "as_numeric",
]
