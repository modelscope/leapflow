"""Hardware transport: the executable half of the Hardware Context Protocol.

Six methods, nothing more. Deliberately narrower than ``ExecutionBackend``: a
device has channels rather than named actions, a physical open/close lifecycle,
and a halt path that must exist independently of any command queue.

This is the volatile half of the protocol. An in-process driver, a vendor CLI,
an MCP server, and a future standardized driver are all implementations of these
six methods, and swapping one for another must not reach any other module.
"""

from __future__ import annotations

import time
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

    Timebase convention, identical to ``domain.events.SystemEvent``:

    - ``observed_at``: wall-clock (``time.time()`` epoch seconds). The only clock
      that may be persisted, rendered, or correlated with anything outside this
      process -- audit entries, approval records, findings and session events are
      all wall-clock.
    - ``monotonic_at``: ``time.monotonic()``. The only clock that may be used for
      intervals, slew rates and staleness, because wall-clock jumps (NTP, suspend,
      manual adjustment) would fabricate rates that no device produced.

    Both are populated by default so an out-of-tree driver that omits them still
    gets a usable pair rather than epoch zero. Carrying one field for both roles
    is what made downsampled history unorderable across a restart: a monotonic
    origin resets on reboot, so ``ORDER BY ended_at DESC`` silently returned the
    oldest rows as the newest.
    """

    device_id: str
    channel_id: str
    value: Any
    quantity: str = ""
    unit: str = ""
    observed_at: float = field(default_factory=time.time)
    monotonic_at: float = field(default_factory=time.monotonic)
    sequence: int = 0
    quality: str = Quality.OK.value

    @property
    def is_trustworthy(self) -> bool:
        return self.quality == Quality.OK.value

    def to_dict(self) -> dict[str, Any]:
        """Return the raw-evidence form. Wall-clock only.

        ``monotonic_at`` is deliberately absent: these records are read by humans
        and by later analysis, for whom a per-boot counter is noise that invites
        exactly the confusion this pair exists to prevent.
        """
        return {
            "device_id": self.device_id,
            "channel_id": self.channel_id,
            "value": self.value,
            "quantity": self.quantity,
            "unit": self.unit,
            "observed_at": self.observed_at,
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

    ``preview`` is the dry-run contract flag. ``preview=True`` means the call was
    a dry run (the tool was invoked with ``parameters.dry_run=true``): the write
    path ran every feasibility check and built the approval descriptor, but never
    reached ``transport.write``, so it carries no physical effect and always
    pairs with ``SIDE_EFFECT_NONE``. The accompanying ``plan`` describes the
    command that *would* have been issued had this not been a dry run, so intent
    can be confirmed before committing an irreversible effect. It is a pure
    additive field defaulting to ``False``: when unset the outcome carries
    ordinary write semantics -- a committed (or attempted) physical command --
    so every existing construction site (real transports, the mock, and any
    out-of-tree driver) keeps its current meaning untouched.
    """

    ok: bool
    side_effect_state: str = SIDE_EFFECT_UNKNOWN
    readback: Reading | None = None
    settled: bool = False
    error: str = ""
    failure_code: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)
    preview: bool = False

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
        # Emitted only when set, so an ordinary write's result shape is unchanged.
        if self.preview:
            payload["preview"] = True
        return payload


@dataclass(frozen=True)
class FrameReading:
    """One frame sampled from a ``representation=frame`` channel.

    Deliberately *not* a ``Reading``. Readings are appended to NDJSON segments and
    downsampled into DuckDB windows, and a frame has neither a mean nor a bound; a
    few hundred kilobytes per sample would also turn the raw segment writer into a
    disk filler with a schedule. So frames travel on their own type, are never
    persisted by the sampling loop, and never enter a finding payload.

    ``data`` is already encoded in ``media_type`` -- encoding happens inside the
    transport, because only the driver knows the native pixel format and doing it
    later would mean carrying raw buffers across a process boundary.

    Timebase follows ``Reading``: ``observed_at`` is wall-clock and the only clock
    that may be rendered or correlated; ``monotonic_at`` is for intervals only.
    """

    device_id: str
    channel_id: str
    data: bytes
    media_type: str = "image/jpeg"
    width: int = 0
    height: int = 0
    observed_at: float = field(default_factory=time.time)
    monotonic_at: float = field(default_factory=time.monotonic)
    sequence: int = 0
    quality: str = Quality.OK.value

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    def to_metadata(self) -> dict[str, Any]:
        """Return the frame's description *without* its bytes.

        The only form that may reach a tool result, a log line, or a JSON-RPC
        reply. Inlining the bytes -- even base64 -- is what makes one camera read
        cost more context than an entire conversation.
        """
        return {
            "device_id": self.device_id,
            "channel_id": self.channel_id,
            "media_type": self.media_type,
            "width": self.width,
            "height": self.height,
            "observed_at": self.observed_at,
            "sequence": self.sequence,
            "quality": self.quality,
            "size_bytes": self.size_bytes,
        }


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


@runtime_checkable
class FrameTransport(Protocol):
    """Optional capability: this transport can produce frames from a channel.

    A *side* protocol, not a seventh core method. ``HardwareTransport`` is six
    methods on purpose, and most devices will never have an image; widening the
    core contract would force every existing driver -- in-tree and out -- to grow
    a method it cannot implement.

    Capability is therefore discovered with ``isinstance(transport,
    FrameTransport)``, exactly as ``halt_supported=False`` makes "cannot stop" a
    discoverable degradation rather than a silent assumption: a device declaring a
    frame channel whose transport does not satisfy this Protocol has that channel
    refused at admission, with a reason, instead of failing on first preview.
    """

    async def read_frame(
        self, channel_id: str, *, max_width: int = 0, quality: int = 0, fps: float = 0.0
    ) -> FrameReading:
        """Capture one frame. Side-effect free.

        ``max_width``, ``quality`` and ``fps`` are requests, not guarantees: a driver that
        cannot scale, re-encode or choose a cadence returns its native frame, and the
        caller reads actual dimensions from the result. The broker clamps every request
        against the declaration and runtime ceilings before it reaches this method, so a
        page's economy/balanced/detail setting changes real encoder work without allowing
        it to demand an arbitrary capture rate.
        """
        ...


__all__ = [
    "SIDE_EFFECT_COMMITTED",
    "SIDE_EFFECT_NONE",
    "SIDE_EFFECT_PARTIAL",
    "SIDE_EFFECT_UNKNOWN",
    "FrameReading",
    "FrameTransport",
    "HardwareTransport",
    "Reading",
    "TransportError",
    "TransportStatus",
    "WriteOutcome",
]
