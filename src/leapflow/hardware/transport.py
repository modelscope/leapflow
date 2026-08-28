"""Hardware transport: the executable half of the Hardware Context Protocol.

Six methods, nothing more. Deliberately narrower than ``ExecutionBackend``: a
device has channels rather than named actions, a physical open/close lifecycle,
and a halt path that must exist independently of any command queue.

This is the volatile half of the protocol. An in-process driver, a vendor CLI,
an MCP server, and a future standardized driver are all implementations of these
six methods, and swapping one for another must not reach any other module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from leapflow.hardware.context import HardwareContext, Quality


SIDE_EFFECT_NONE = "none"
SIDE_EFFECT_COMMITTED = "committed"
SIDE_EFFECT_PARTIAL = "partial"
SIDE_EFFECT_UNKNOWN = "unknown"
"""Side-effect verdicts, mirroring ``engine.failure_envelope.SideEffectState``.

Expressed as strings rather than imported, because ``leapflow.hardware`` must not
depend on ``leapflow.engine``: the dependency runs the other way, and a cycle
here would make the domain model unimportable standalone. The values are
converted at the boundary where a tool result is built.
"""


@dataclass(frozen=True)
class Reading:
    """One sampled value from a channel.

    Kept inside ``leapflow.hardware`` on purpose. Raw readings must not enter
    ``SignalBuffer`` or the causal graph: a 10 Hz channel would flush a 50-slot
    buffer in five seconds and drive causal fusion at sampling rate. Only derived
    events cross that boundary.

    ``sequence`` exists so that dropped samples become detectable rather than
    silent -- a gap in the sequence is the only evidence that a bounded queue
    discarded something.
    """

    device_id: str
    channel_id: str
    value: Any
    quantity: str = ""
    unit: str = ""
    timestamp: float = 0.0
    sequence: int = 0
    quality: str = Quality.OK.value

    @property
    def is_trustworthy(self) -> bool:
        return self.quality == Quality.OK.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "channel_id": self.channel_id,
            "value": self.value,
            "quantity": self.quantity,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "quality": self.quality,
        }


@dataclass(frozen=True)
class WriteOutcome:
    """Result of a channel write, carrying an explicit side-effect verdict.

    The transport is the only component close enough to the device to know
    whether a failed write landed, so it reports that verdict here rather than
    raising a bare exception. An error is not proof that nothing happened, and a
    caller that assumes otherwise is the mechanism by which a failed physical
    operation gets blindly repeated.

    A failing outcome must never report ``SIDE_EFFECT_NONE`` unless the transport
    can genuinely prove the command never reached the device; ``UNKNOWN`` is the
    correct answer when it cannot, and it blocks replay exactly like
    ``COMMITTED`` does.
    """

    ok: bool
    side_effect_state: str = SIDE_EFFECT_UNKNOWN
    readback: Reading | None = None
    settled: bool = False
    error: str = ""
    failure_code: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def effect_may_have_landed(self) -> bool:
        """Return True when the physical effect is not known to be absent."""
        return self.side_effect_state != SIDE_EFFECT_NONE

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "side_effect_state": self.side_effect_state,
            "settled": self.settled,
        }
        if self.readback is not None:
            payload["readback"] = self.readback.to_dict()
        if self.error:
            payload["error"] = self.error
        if self.failure_code:
            payload["failure_code"] = self.failure_code
        return payload


@dataclass(frozen=True)
class TransportStatus:
    """Transport health and declared capabilities."""

    connected: bool = False
    halt_supported: bool = False
    detail: str = ""
    latency_ms: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "halt_supported": self.halt_supported,
            "detail": self.detail,
            "latency_ms": self.latency_ms,
        }


class TransportError(RuntimeError):
    """Raised by a transport when an operation cannot be attempted at all.

    Intentionally a plain exception subclass and never frozen: CPython assigns
    ``__traceback__`` on every re-raise, and a frozen exception type would
    replace the real failure with a ``FrozenInstanceError`` about that
    assignment.

    A transport raises this only for "could not attempt" (channel unknown, not
    open). Anything that may have reached the device must come back as a
    ``WriteOutcome`` carrying a side-effect verdict instead, because an exception
    cannot express "it might have landed".
    """

    def __init__(self, message: str, *, failure_code: str = "transport_error") -> None:
        super().__init__(message)
        self.failure_code = failure_code


@runtime_checkable
class HardwareTransport(Protocol):
    """Southbound contract for one device."""

    kind: str

    async def open(self, context: HardwareContext) -> TransportStatus:
        """Establish the connection. Must be idempotent."""
        ...

    async def close(self) -> TransportStatus:
        """Release the connection. Must be idempotent and must never raise."""
        ...

    async def read(self, channel_id: str) -> Reading:
        """Read one channel. Side-effect free."""
        ...

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        """Write one channel, reporting whether the effect may have landed."""
        ...

    async def probe(self) -> TransportStatus:
        """Liveness and health check. Side-effect free."""
        ...

    async def halt(self) -> TransportStatus:
        """Stop all motion and output as fast as the device allows.

        A transport that cannot halt returns ``halt_supported=False`` rather than
        raising, and the registry then refuses to expose any writable channel for
        that device while keeping its readable channels available. "Cannot stop"
        becomes a discoverable capability degradation instead of a silent
        assumption.
        """
        ...


__all__ = [
    "SIDE_EFFECT_COMMITTED",
    "SIDE_EFFECT_NONE",
    "SIDE_EFFECT_PARTIAL",
    "SIDE_EFFECT_UNKNOWN",
    "HardwareTransport",
    "Reading",
    "TransportError",
    "TransportStatus",
    "WriteOutcome",
]
