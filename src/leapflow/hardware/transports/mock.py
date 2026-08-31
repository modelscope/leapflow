"""Programmable in-memory transport for tests and dry runs.

Deliberately device-agnostic: it holds channel values, applies writes, and
injects failures entirely from its declaration config. There is no notion of a
temperature, an arm, or any other specific instrument anywhere in this file --
binding the mock to one device class would make it useless for the next one and
would smuggle device semantics into the protocol.

Behaviour is configured through ``TransportRef.config``:

    kind: mock
    config:
      values: {channel_id: initial_value}
      halt_supported: true
      latency_ms: 0.0
      failures:
        - channel_id: aspirate
          on_call: 1               # 1-based write attempt to fail
          side_effect_state: partial
          error: "liquid detection error"
          failure_code: fluid_detection
          repeat: false            # true = fail every attempt from on_call onward
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from leapflow.hardware.context import HardwareContext, Quality
from leapflow.hardware.transport import (
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_UNKNOWN,
    Reading,
    TransportError,
    TransportStatus,
    WriteOutcome,
)


@dataclass(frozen=True)
class _FailureRule:
    """One declarative write-failure injection."""

    channel_id: str
    on_call: int = 1
    side_effect_state: str = SIDE_EFFECT_UNKNOWN
    error: str = "injected transport failure"
    failure_code: str = "injected_failure"
    repeat: bool = False

    def matches(self, channel_id: str, call_index: int) -> bool:
        if self.channel_id not in {channel_id, "*"}:
            return False
        return call_index >= self.on_call if self.repeat else call_index == self.on_call


class MockTransport:
    """In-memory transport whose entire behaviour comes from configuration."""

    kind = "mock"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        raw_values = config.get("values")
        self._values: dict[str, Any] = dict(raw_values) if isinstance(raw_values, Mapping) else {}
        self._halt_supported = bool(config.get("halt_supported", True))
        self._latency_ms = float(config.get("latency_ms", 0.0) or 0.0)
        self._failures = tuple(_parse_failures(config.get("failures")))
        self._connected = False
        self._context: HardwareContext | None = None
        self._sequence: dict[str, int] = {}
        self._write_calls: dict[str, int] = {}
        self._halt_calls = 0
        self._write_log: list[tuple[str, Any]] = []

    # ── Lifecycle ──

    async def open(self, context: HardwareContext) -> TransportStatus:
        self._context = context
        self._connected = True
        for channel in context.channels:
            self._values.setdefault(channel.channel_id, None)
        return await self.probe()

    async def close(self) -> TransportStatus:
        # Must never raise: teardown runs during interpreter shutdown paths where
        # an exception would mask the original failure.
        self._connected = False
        return TransportStatus(connected=False, halt_supported=self._halt_supported, detail="closed")

    async def probe(self) -> TransportStatus:
        return TransportStatus(
            connected=self._connected,
            halt_supported=self._halt_supported,
            detail="mock transport",
            latency_ms=self._latency_ms,
            metadata={"channels": len(self._values)},
        )

    async def halt(self) -> TransportStatus:
        self._halt_calls += 1
        if not self._halt_supported:
            return TransportStatus(
                connected=self._connected,
                halt_supported=False,
                detail="halt not supported by this device",
            )
        return TransportStatus(connected=self._connected, halt_supported=True, detail="halted")

    # ── Data plane ──

    async def read(self, channel_id: str) -> Reading:
        self._require_known_channel(channel_id)
        sequence = self._sequence.get(channel_id, 0) + 1
        self._sequence[channel_id] = sequence
        channel = self._context.channel(channel_id) if self._context is not None else None
        return Reading(
            device_id=self._context.device_id if self._context is not None else "",
            channel_id=channel_id,
            value=self._values.get(channel_id),
            quantity=channel.quantity if channel is not None else "",
            unit=channel.unit if channel is not None else "",
            sequence=sequence,
            quality=Quality.OK.value,
        )

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        self._require_known_channel(channel_id)
        call_index = self._write_calls.get(channel_id, 0) + 1
        self._write_calls[channel_id] = call_index

        rule = next((r for r in self._failures if r.matches(channel_id, call_index)), None)
        if rule is not None:
            # A partial or unknown verdict means the commanded effect may already
            # have reached the device, so the stored value is left untouched
            # rather than rolled back -- the mock must not pretend to know more
            # than a real transport could.
            return WriteOutcome(
                ok=False,
                side_effect_state=rule.side_effect_state,
                error=rule.error,
                failure_code=rule.failure_code,
                raw={"call_index": call_index},
            )

        self._values[channel_id] = value
        self._write_log.append((channel_id, value))
        readback = await self.read(channel_id) if self._needs_readback(channel_id) else None
        return WriteOutcome(
            ok=True,
            side_effect_state=SIDE_EFFECT_COMMITTED,
            readback=readback,
            settled=self._settling_time(channel_id) <= 0.0,
            raw={"call_index": call_index},
        )

    # ── Test introspection ──

    @property
    def write_log(self) -> tuple[tuple[str, Any], ...]:
        """Every accepted write, in order. Lets tests assert on retry behaviour."""
        return tuple(self._write_log)

    def write_attempts(self, channel_id: str) -> int:
        """Attempts made against *channel_id*, including rejected ones."""
        return self._write_calls.get(channel_id, 0)

    @property
    def halt_calls(self) -> int:
        return self._halt_calls

    def set_value(self, channel_id: str, value: Any) -> None:
        """Set a channel value directly, simulating a change in the world."""
        self._values[channel_id] = value

    # ── Internals ──

    def _require_known_channel(self, channel_id: str) -> None:
        if not self._connected:
            raise TransportError(
                f"transport for {channel_id!r} is not open", failure_code="transport_not_open"
            )
        known = self._context.channel(channel_id) if self._context is not None else None
        if known is None and channel_id not in self._values:
            raise TransportError(f"unknown channel {channel_id!r}", failure_code="unknown_channel")

    def _needs_readback(self, channel_id: str) -> bool:
        channel = self._context.channel(channel_id) if self._context is not None else None
        return bool(channel is not None and channel.verify_after_write)

    def _settling_time(self, channel_id: str) -> float:
        channel = self._context.channel(channel_id) if self._context is not None else None
        return channel.envelope.settling_time_s if channel is not None else 0.0


def _parse_failures(raw: Any) -> list[_FailureRule]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    rules: list[_FailureRule] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        rules.append(
            _FailureRule(
                channel_id=str(item.get("channel_id") or "*"),
                on_call=int(item.get("on_call") or 1),
                side_effect_state=str(item.get("side_effect_state") or SIDE_EFFECT_UNKNOWN),
                error=str(item.get("error") or "injected transport failure"),
                failure_code=str(item.get("failure_code") or "injected_failure"),
                repeat=bool(item.get("repeat", False)),
            )
        )
    return rules


def build_transport(config: Mapping[str, Any] | None = None) -> MockTransport:
    """Factory registered in the transport table."""
    return MockTransport(config)


__all__ = ["MockTransport", "SIDE_EFFECT_NONE", "build_transport"]
