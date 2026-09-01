"""Parameterised simulation transport for end-to-end and long-running tests.

Like :mod:`leapflow.hardware.transports.mock`, this transport is entirely
device-agnostic: it holds no notion of a temperature, an arm, or any specific
instrument. Everything it does -- the shape of the values it reports, the way a
channel degrades, when the link drops -- is read from its declaration config,
never from code.

Where the mock is a programmable *store* (you set values and it hands them
back), the simulated transport is a programmable *signal source*: it generates
readings from a waveform against a logical clock, and models the ways a real
link misbehaves -- latency, dropped samples, reordering, quality degradation,
and disconnect/reconnect sequences. That makes it the substrate for L3
simulation journeys and for long-run tests, which drive time with
:meth:`advance_clock` instead of sleeping.

The logical clock is the reason a "7-day" test finishes in milliseconds. Both
clocks a :class:`~leapflow.hardware.transport.Reading` carries advance together
by the same logical amount, so wall-clock ordering and monotonic intervals stay
consistent no matter how far the clock is fast-forwarded.

Behaviour is configured through ``TransportRef.config``::

    kind: simulated
    config:
      values: {setpoint: 50.0}       # static values for non-waveform channels
      halt_supported: true
      init_required: false            # gate read/write until an init write lands
      init_channel: __init__          # write here once to initialise (see write())
      latency_ms: 0.0                 # per-operation link latency
      drop_probability: 0.0           # chance a sample is dropped (a seq gap)
      reorder: false                  # deliver adjacent samples swapped
      quality_degradation: 0.0        # chance a reading is marked degraded
      degraded_quality: suspect
      seed: 1337
      waveforms:
        sensor: {kind: sine, offset: 21.0, amplitude: 5.0, period_s: 60.0}
      failures:                       # write-side injection (see MockTransport)
        - {channel_id: setpoint, on_call: 1, side_effect_state: partial}
      disconnects:
        - {on_read: 5, reconnect_after: 2}
"""

from __future__ import annotations

import math
import random
import time
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
class _Waveform:
    """A declarative value generator sampled against the logical clock."""

    kind: str = "constant"
    value: float = 0.0
    amplitude: float = 1.0
    offset: float = 0.0
    period_s: float = 1.0
    phase: float = 0.0
    levels: tuple[float, ...] = ()
    step_interval_s: float = 1.0
    mean: float = 0.0
    stddev: float = 1.0

    def sample(self, elapsed_s: float, rng: random.Random) -> float:
        """Return the value at logical time *elapsed_s*.

        ``noise`` draws from *rng* so a fixed seed makes a run reproducible; the
        other shapes are pure functions of time and need no randomness.
        """
        if self.kind == "sine":
            if self.period_s <= 0.0:
                return self.offset
            return self.offset + self.amplitude * math.sin(
                2.0 * math.pi * (elapsed_s / self.period_s) + self.phase
            )
        if self.kind == "step":
            if not self.levels:
                return self.offset
            if self.step_interval_s <= 0.0:
                return self.levels[0]
            index = int(elapsed_s // self.step_interval_s) % len(self.levels)
            return self.levels[index]
        if self.kind == "noise":
            return self.mean if self.stddev <= 0.0 else rng.gauss(self.mean, self.stddev)
        return self.value


@dataclass(frozen=True)
class _FailureRule:
    """One declarative write-failure injection, mirroring ``MockTransport``."""

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


@dataclass(frozen=True)
class _DisconnectRule:
    """A scheduled link drop, keyed by the read attempt at which it fires.

    ``reconnect_after`` is the total number of read attempts that observe the
    link down, counted from ``on_read`` inclusive; the link recovers on its own
    at attempt ``on_read + reconnect_after``. For example ``on_read=2,
    reconnect_after=2`` fails attempts 2 and 3 and recovers on attempt 4. ``0``
    means it stays down until an explicit :meth:`SimulatedTransport.open` or an
    injected reconnect.
    """

    on_read: int
    reconnect_after: int = 0


class SimulatedTransport:
    """Transport that synthesises readings and models link misbehaviour.

    Also implements :class:`leapflow.hardware.testing.SignalInjector`, so a test
    can steer it deterministically -- forcing a specific value, punching a gap in
    the sequence, dropping the link, or advancing the logical clock -- without
    reaching into private state.
    """

    kind = "simulated"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        raw_values = config.get("values")
        self._values: dict[str, Any] = dict(raw_values) if isinstance(raw_values, Mapping) else {}
        self._halt_supported = bool(config.get("halt_supported", True))
        self._init_required = bool(config.get("init_required", False))
        self._init_channel = str(config.get("init_channel") or "__init__")
        self._latency_ms = float(config.get("latency_ms", 0.0) or 0.0)
        self._drop_probability = float(config.get("drop_probability", 0.0) or 0.0)
        self._reorder = bool(config.get("reorder", False))
        self._quality_degradation = float(config.get("quality_degradation", 0.0) or 0.0)
        self._degraded_quality = str(config.get("degraded_quality") or Quality.SUSPECT.value)
        self._waveforms = _parse_waveforms(config.get("waveforms"))
        self._failures = tuple(_parse_failures(config.get("failures")))
        self._disconnects = tuple(_parse_disconnects(config.get("disconnects")))
        self._rng = random.Random(int(config.get("seed", 1337)))

        self._connected = False
        # ``init_required`` gates the data plane until an init write lands. Off by
        # default, so every existing declaration is ready the moment it opens and
        # keeps its current behaviour untouched (backward-compatible default).
        # Turning it on is a **one-way contract**: an existing profile that has
        # never declared ``init_required: true`` continues to work without change;
        # once a profile opts in, the transport refuses reads/writes until the init
        # handshake completes.  When on, the init channel is registered as a known
        # channel here so the single init handshake write passes channel validation
        # like any other write.
        self._initialized = not self._init_required
        if self._init_required:
            self._values.setdefault(self._init_channel, None)
        self._context: HardwareContext | None = None
        self._sequence: dict[str, int] = {}
        self._injected: dict[str, list[tuple[Any, str | None]]] = {}
        self._reorder_held: dict[str, Reading] = {}
        self._write_calls: dict[str, int] = {}
        self._write_log: list[tuple[str, Any]] = []
        self._read_attempts = 0
        self._halt_calls = 0
        self._reconnect_at: int | None = None

        # Both clocks share one logical origin so they advance together. The wall
        # base is a real epoch instant (so ``observed_at`` is orderable across a
        # restart), the monotonic base is a per-boot counter, and every reading is
        # stamped ``base + elapsed``.
        self._wall_base = time.time()
        self._mono_base = time.monotonic()
        self._elapsed = 0.0

    # ── Lifecycle ──

    async def open(self, context: HardwareContext) -> TransportStatus:
        self._context = context
        self._connected = True
        self._reconnect_at = None
        # A physical re-open returns the device to its power-on state, so a
        # transport that requires initialisation is uninitialised again until the
        # next init write. With ``init_required`` off this stays ready.
        self._initialized = not self._init_required
        for channel in context.channels:
            self._values.setdefault(channel.channel_id, None)
        return await self.probe()

    async def close(self) -> TransportStatus:
        # Must never raise: teardown may run during interpreter shutdown, where an
        # exception would mask the original failure.
        self._connected = False
        return TransportStatus(
            connected=False, halt_supported=self._halt_supported, detail="closed"
        )

    async def probe(self) -> TransportStatus:
        return TransportStatus(
            connected=self._connected,
            halt_supported=self._halt_supported,
            detail="simulated transport",
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
        attempt = self._read_attempts + 1
        self._read_attempts = attempt
        self._apply_connectivity(attempt)
        self._require_open(channel_id)
        self._require_initialized(channel_id)
        self._advance_latency()
        if self._reorder:
            held = self._reorder_held.get(channel_id)
            if held is None:
                # Hold this sample and deliver the following one, so two adjacent
                # samples arrive swapped -- the sequence numbers prove the reorder.
                self._reorder_held[channel_id] = self._generate_reading(channel_id)
                return self._generate_reading(channel_id)
            del self._reorder_held[channel_id]
            return held
        return self._generate_reading(channel_id)

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        self._require_open(channel_id)
        if not self._initialized:
            return self._handle_uninitialized_write(channel_id, value)
        call_index = self._write_calls.get(channel_id, 0) + 1
        self._write_calls[channel_id] = call_index
        self._advance_latency()

        rule = next((r for r in self._failures if r.matches(channel_id, call_index)), None)
        if rule is not None:
            # A partial or unknown verdict means the commanded effect may already
            # have reached the device, so the stored value is left untouched rather
            # than rolled back: the transport must not pretend to know more than a
            # real one could.
            return WriteOutcome(
                ok=False,
                side_effect_state=rule.side_effect_state,
                error=rule.error,
                failure_code=rule.failure_code,
                raw={"call_index": call_index},
            )

        self._values[channel_id] = value
        self._write_log.append((channel_id, value))
        readback = self._readback(channel_id) if self._needs_readback(channel_id) else None
        return WriteOutcome(
            ok=True,
            side_effect_state=SIDE_EFFECT_COMMITTED,
            readback=readback,
            settled=self._settling_time(channel_id) <= 0.0,
            raw={"call_index": call_index},
        )

    # ── SignalInjector: deterministic test control ──

    def inject_reading(self, channel_id: str, value: Any, *, quality: str | None = None) -> None:
        """Queue *value* to be returned by the next read of *channel_id*.

        Injected readings take priority over the waveform or the stored value and
        are consumed in order, one per read, then behaviour reverts to normal.
        """
        self._injected.setdefault(channel_id, []).append((value, quality))

    def inject_gap(self, channel_id: str, *, dropped: int = 1) -> None:
        """Advance *channel_id*'s sequence by *dropped*, punching a visible gap.

        This is how a long-run test asserts that a bounded queue discarded
        samples: the next delivered reading skips *dropped* sequence numbers.
        """
        current = self._sequence.get(channel_id, 0)
        self._sequence[channel_id] = current + max(0, int(dropped))

    def inject_disconnect(self, *, reconnect_after: int | None = None) -> None:
        """Drop the link now. Every read fails until it recovers.

        ``reconnect_after`` schedules automatic recovery after that many read
        attempts; ``None`` leaves the link down until :meth:`open` is called
        again.
        """
        self._connected = False
        if reconnect_after is not None and reconnect_after > 0:
            self._reconnect_at = self._read_attempts + int(reconnect_after)
        else:
            self._reconnect_at = None

    def advance_clock(self, seconds: float) -> None:
        """Advance the logical clock by *seconds* without sleeping.

        Both clocks a reading carries move forward by this amount, so a test can
        simulate hours or days of elapsed time in-process while keeping wall-clock
        ordering and monotonic intervals consistent.
        """
        self._elapsed += max(0.0, float(seconds))

    # ── Test introspection ──

    @property
    def write_log(self) -> tuple[tuple[str, Any], ...]:
        """Every accepted write, in order."""
        return tuple(self._write_log)

    @property
    def read_attempts(self) -> int:
        """Reads attempted, including those refused by a dropped link."""
        return self._read_attempts

    @property
    def halt_calls(self) -> int:
        return self._halt_calls

    @property
    def initialized(self) -> bool:
        """Whether the data plane is open for reads and writes.

        Always ``True`` unless ``init_required`` is set, in which case it is
        ``False`` from ``open`` until the declared init channel is written.
        """
        return self._initialized

    # ── Internals ──

    def _apply_connectivity(self, attempt: int) -> None:
        """Recover a scheduled reconnect, then fire any drop due at *attempt*."""
        if (
            not self._connected
            and self._reconnect_at is not None
            and attempt >= self._reconnect_at
        ):
            self._connected = True
            self._reconnect_at = None
        for rule in self._disconnects:
            if rule.on_read == attempt and self._connected:
                self._connected = False
                self._reconnect_at = (
                    attempt + rule.reconnect_after if rule.reconnect_after > 0 else None
                )
                break

    def _require_initialized(self, channel_id: str) -> None:
        """Refuse a read until the device has been initialised.

        A read against an uninitialised device is a "could not attempt", so it
        surfaces as :class:`TransportError` with ``failure_code="not_initialized"``
        -- never as a fabricated reading. With ``init_required`` off the device is
        always initialised, so this is a no-op on the default path.
        """
        if not self._initialized:
            raise TransportError(
                f"transport for {channel_id!r} is not initialized",
                failure_code="not_initialized",
            )

    def _handle_uninitialized_write(self, channel_id: str, value: Any) -> WriteOutcome:
        """Perform the init handshake, or refuse a write that is not one.

        A write to the declared ``init_channel`` runs the one-shot init handshake
        and returns a committed outcome; ordinary reads and writes are open from
        then on. Any other write is refused with ``failure_code="not_initialized"``
        and ``SIDE_EFFECT_NONE`` -- the command was rejected before it could reach
        the device, so recovery is free to replay it once the device is
        initialised. This is the trigger an L3 init/calibration stage drives.
        """
        if channel_id == self._init_channel:
            self._initialized = True
            self._advance_latency()
            self._write_log.append((channel_id, value))
            return WriteOutcome(
                ok=True,
                side_effect_state=SIDE_EFFECT_COMMITTED,
                settled=True,
                raw={"init": True},
            )
        return WriteOutcome(
            ok=False,
            side_effect_state=SIDE_EFFECT_NONE,
            error="transport is not initialized",
            failure_code="not_initialized",
        )

    def _require_open(self, channel_id: str) -> None:
        if not self._connected:
            raise TransportError(
                f"transport for {channel_id!r} is not open", failure_code="transport_not_open"
            )
        known = self._context.channel(channel_id) if self._context is not None else None
        if known is None and channel_id not in self._values:
            raise TransportError(f"unknown channel {channel_id!r}", failure_code="unknown_channel")

    def _generate_reading(self, channel_id: str) -> Reading:
        sequence = self._sequence.get(channel_id, 0) + 1
        # A dropped sample leaves no reading, only a hole in the numbering: bump the
        # counter an extra step so the next delivered reading shows the gap.
        if self._drop_probability > 0.0 and self._rng.random() < self._drop_probability:
            sequence += 1
        self._sequence[channel_id] = sequence

        value, forced_quality = self._resolve_value(channel_id)
        quality = forced_quality if forced_quality is not None else self._resolve_quality()
        wall, monotonic = self._now()
        channel = self._context.channel(channel_id) if self._context is not None else None
        return Reading(
            device_id=self._context.device_id if self._context is not None else "",
            channel_id=channel_id,
            value=value,
            quantity=channel.quantity if channel is not None else "",
            unit=channel.unit if channel is not None else "",
            observed_at=wall,
            monotonic_at=monotonic,
            sequence=sequence,
            quality=quality,
        )

    def _readback(self, channel_id: str) -> Reading:
        """Produce a verification reading without reorder or latency side effects."""
        return self._generate_reading(channel_id)

    def _resolve_value(self, channel_id: str) -> tuple[Any, str | None]:
        queue = self._injected.get(channel_id)
        if queue:
            return queue.pop(0)
        waveform = self._waveforms.get(channel_id)
        if waveform is not None:
            return waveform.sample(self._elapsed, self._rng), None
        return self._values.get(channel_id), None

    def _resolve_quality(self) -> str:
        if self._quality_degradation > 0.0 and self._rng.random() < self._quality_degradation:
            return self._degraded_quality
        return Quality.OK.value

    def _now(self) -> tuple[float, float]:
        return self._wall_base + self._elapsed, self._mono_base + self._elapsed

    def _advance_latency(self) -> None:
        if self._latency_ms > 0.0:
            self._elapsed += self._latency_ms / 1000.0

    def _needs_readback(self, channel_id: str) -> bool:
        channel = self._context.channel(channel_id) if self._context is not None else None
        return bool(channel is not None and channel.verify_after_write)

    def _settling_time(self, channel_id: str) -> float:
        channel = self._context.channel(channel_id) if self._context is not None else None
        return channel.envelope.settling_time_s if channel is not None else 0.0


def _parse_waveforms(raw: Any) -> dict[str, _Waveform]:
    if not isinstance(raw, Mapping):
        return {}
    waveforms: dict[str, _Waveform] = {}
    for channel_id, spec in raw.items():
        if not isinstance(spec, Mapping):
            continue
        waveforms[str(channel_id)] = _Waveform(
            kind=str(spec.get("kind") or "constant"),
            value=_as_float(spec.get("value"), 0.0),
            amplitude=_as_float(spec.get("amplitude"), 1.0),
            offset=_as_float(spec.get("offset"), 0.0),
            period_s=_as_float(spec.get("period_s"), 1.0),
            phase=_as_float(spec.get("phase"), 0.0),
            levels=tuple(_as_float(level, 0.0) for level in spec.get("levels") or ()),
            step_interval_s=_as_float(spec.get("step_interval_s"), 1.0),
            mean=_as_float(spec.get("mean"), 0.0),
            stddev=_as_float(spec.get("stddev"), 1.0),
        )
    return waveforms


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


def _parse_disconnects(raw: Any) -> list[_DisconnectRule]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    rules: list[_DisconnectRule] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        rules.append(
            _DisconnectRule(
                on_read=int(item.get("on_read") or 0),
                reconnect_after=int(item.get("reconnect_after") or 0),
            )
        )
    return rules


def _as_float(value: Any, default: float) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_transport(config: Mapping[str, Any] | None = None) -> SimulatedTransport:
    """Factory registered in the transport table."""
    return SimulatedTransport(config)


__all__ = ["SimulatedTransport", "build_transport"]
