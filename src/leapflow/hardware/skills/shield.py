# Copyright (c) Alibaba, Inc. and its affiliates.
"""Per-tick safety filter between a streaming policy and the actuators.

A learned policy is a proposal source, not an authority. Every action it
streams passes ``SafetyShield.filter`` before it becomes a command:

1. Cyclic angle channels are wrapped into their declared range.
2. A value outside the channel envelope by more than the channel tolerance is a
   *violation*; a value within tolerance of a bound is clamped onto it.
3. A step from the last command larger than ``max_jump`` is a violation: a
   discontinuity in the stream (a chunk answering another scene, a wrongly
   declared space) must stop the run, not be dragged through by the slew limit.
4. The rate of change is limited -- first its change by ``max_accel * dt``,
   then its magnitude by ``max_rate * rate_margin`` -- so the node gate's slew
   check never has to stall the stream. A limited step never overshoots the
   target. A command that ends further than the channel tolerance from what the
   policy asked counts as an intervention.
5. An optional pose check judges where the filtered command puts the tool
   (workspace box, tool cone, reach) and turns a bad answer into a violation.

Every limit comes from the embodiment's declarations; nothing a model returns
can widen one. ``ContactLimit`` adds the read side: a declared force/torque
channel that leaves its envelope stops the stream (checked per observation).

A violation never yields a command: the caller stops streaming and reports
why. Pure apart from the injected pose check, so it is tested without a device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from leapflow.hardware.node_gate import is_cyclic
from leapflow.hardware.skills.adapt.aggregate import Action
from leapflow.hardware.skills.geometry import wrap_deg

PoseCheck = Callable[[Action], str]
"""Judge a filtered action; return a reason to refuse it, or ``""`` when it is fine."""

CONTACT_QUANTITIES = frozenset({"force", "torque", "effort"})
_EXACT = 1e-9


@dataclass(frozen=True)
class ShieldChannel:
    """The declared limits of one streamed channel."""

    channel_id: str
    device_id: str
    low: float | None = None
    high: float | None = None
    max_rate: float | None = None
    tolerance: float = 0.0
    cyclic: bool = False
    max_accel: float | None = None
    max_jump: float | None = None

    @property
    def label(self) -> str:
        return f"{self.device_id}.{self.channel_id}"

    @classmethod
    def from_channel(
        cls, device_id: str, channel: Any, *, jump_s: float | None = None
    ) -> "ShieldChannel":
        """The channel's envelope; *jump_s* sets ``max_jump`` to that many seconds at full rate."""
        envelope = channel.envelope
        rate = envelope.max_rate
        jump = rate * jump_s if jump_s and rate and rate > 0 else None
        return cls(
            channel_id=channel.channel_id,
            device_id=device_id,
            low=envelope.min_value,
            high=envelope.max_value,
            max_rate=rate,
            tolerance=float(envelope.tolerance or 0.0),
            cyclic=is_cyclic(channel),
            max_accel=getattr(envelope, "max_accel", None),
            max_jump=jump,
        )

    def wrap(self, value: float) -> float:
        low = -180.0 if self.low is None else self.low
        return low + (value - low) % 360.0

    def delta(self, value: float, last: float) -> float:
        return wrap_deg(value - last) if self.cyclic else value - last


@dataclass(frozen=True)
class ShieldVerdict:
    """The command to send (when ``ok``), how often it was clamped, or why it was refused.

    ``rates`` is the commanded rate per channel (units/s), what the next tick's
    acceleration limit starts from.
    """

    values: Action
    clamped: int = 0
    violation: str = ""
    rates: Action = ()

    @property
    def ok(self) -> bool:
        return not self.violation


class SafetyShield:
    """Envelope, jump, acceleration, slew and pose filter for one action vector layout."""

    def __init__(
        self,
        channels: Sequence[ShieldChannel],
        *,
        rate_margin: float = 0.9,
        pose_check: PoseCheck | None = None,
    ) -> None:
        if not channels:
            raise ValueError("a shield needs at least one channel")
        if not 0.0 < rate_margin <= 1.0:
            raise ValueError("rate_margin must be in (0, 1]")
        self._channels = tuple(channels)
        self._margin = float(rate_margin)
        self._pose_check = pose_check

    @property
    def channels(self) -> tuple[ShieldChannel, ...]:
        return self._channels

    def filter(
        self,
        target: Sequence[float],
        previous: Sequence[float | None],
        dt: float,
        rates: Sequence[float] | None = None,
    ) -> ShieldVerdict:
        """Filter *target* given the last command, *dt* seconds since it, and its *rates*."""
        width = len(self._channels)
        if len(target) != width or len(previous) != width:
            raise ValueError("target and previous must match the shield's channels")
        if rates is not None and len(rates) != width:
            raise ValueError("rates must match the shield's channels")
        values: list[float] = []
        new_rates: list[float] = []
        clamped = 0
        for index, (channel, wanted, last) in enumerate(zip(self._channels, target, previous)):
            value, problem = self._bound(channel, float(wanted))
            if not problem and last is not None:
                problem = self._jump(channel, value, float(last))
            if problem:
                return ShieldVerdict(tuple(values), clamped, problem, tuple(new_rates))
            rate = None if rates is None else float(rates[index])
            limited, commanded_rate = self._limit(channel, value, last, dt, rate)
            if abs(channel.delta(limited, float(wanted))) > max(channel.tolerance, _EXACT):
                clamped += 1
            values.append(limited)
            new_rates.append(commanded_rate)
        action = tuple(values)
        problem = self._pose_check(action) if self._pose_check is not None else ""
        return ShieldVerdict(action, clamped, problem, tuple(new_rates))

    @staticmethod
    def _bound(channel: ShieldChannel, value: float) -> tuple[float, str]:
        if not math.isfinite(value):
            return value, f"{channel.label} target is not a finite number"
        if channel.cyclic:
            return channel.wrap(value), ""
        low, high, slack = channel.low, channel.high, channel.tolerance
        if low is not None and value < low:
            if value < low - slack:
                return value, f"{channel.label} target {value:.2f} is below its limit {low:g}"
            return low, ""
        if high is not None and value > high:
            if value > high + slack:
                return value, f"{channel.label} target {value:.2f} is above its limit {high:g}"
            return high, ""
        return value, ""

    @staticmethod
    def _jump(channel: ShieldChannel, value: float, last: float) -> str:
        if channel.max_jump is None:
            return ""
        step = abs(channel.delta(value, last))
        if step <= channel.max_jump:
            return ""
        return (
            f"{channel.label} target jumps {step:.2f} from the last command, more than the "
            f"{channel.max_jump:g} a stream may step at once"
        )

    def _limit(
        self,
        channel: ShieldChannel,
        value: float,
        last: float | None,
        dt: float,
        rate: float | None,
    ) -> tuple[float, float]:
        """The command after the acceleration and rate limits, and its rate."""
        if last is None or dt <= 0.0:
            return value, 0.0
        delta = channel.delta(value, float(last))
        wanted_rate = delta / dt
        commanded = wanted_rate
        if channel.max_accel is not None and channel.max_accel > 0.0 and rate is not None:
            reach = channel.max_accel * dt
            commanded = min(max(commanded, rate - reach), rate + reach)
        if channel.max_rate is not None and channel.max_rate > 0.0:
            cap = channel.max_rate * self._margin
            commanded = min(max(commanded, -cap), cap)
        step = commanded * dt
        if delta * step <= 0.0 and delta != 0.0 and commanded != wanted_rate:
            step = 0.0  # still braking the other way: hold rather than reverse past it
        if abs(step) > abs(delta):
            step = delta  # never overshoot the target
        limited = float(last) + step
        if channel.cyclic:
            limited = channel.wrap(limited)
        else:
            if channel.low is not None:
                limited = max(limited, channel.low)
            if channel.high is not None:
                limited = min(limited, channel.high)
        return limited, step / dt


@dataclass(frozen=True)
class ContactLimit:
    """A read-only force/torque channel whose declared envelope bounds contact."""

    channel_id: str
    device_id: str
    low: float | None = None
    high: float | None = None

    @property
    def label(self) -> str:
        return f"{self.device_id}.{self.channel_id}"

    def check(self, value: Any) -> str:
        """Why *value* means unsafe contact, or ``""``."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{self.label} reported no usable reading ({value!r})"
        reading = float(value)
        if not math.isfinite(reading):
            return f"{self.label} reported a non-finite reading"
        if self.low is not None and reading < self.low:
            return f"{self.label} reads {reading:.2f}, below its contact limit {self.low:g}"
        if self.high is not None and reading > self.high:
            return f"{self.label} reads {reading:.2f}, above its contact limit {self.high:g}"
        return ""


def contact_limits(devices: Iterable[tuple[str, Any]]) -> tuple[ContactLimit, ...]:
    """Every readable, not writable force/torque channel with declared bounds on *devices*.

    A writable effort channel is a command (a gripper's grip force), not a
    measurement, so it never arms a contact limit.
    """
    found: dict[str, ContactLimit] = {}
    for device_id, context in devices:
        for channel in getattr(context, "channels", ()) if context is not None else ():
            envelope = channel.envelope
            if (
                channel.is_readable
                and not channel.is_writable
                and str(getattr(channel, "quantity", "")).lower() in CONTACT_QUANTITIES
                and envelope.declared
                and (envelope.min_value is not None or envelope.max_value is not None)
            ):
                limit = ContactLimit(
                    channel.channel_id, device_id, envelope.min_value, envelope.max_value
                )
                found.setdefault(limit.label, limit)
    return tuple(found.values())


def contact_violation(
    limits: Sequence[ContactLimit], readings: Mapping[str, Mapping[str, Any]]
) -> str:
    """The first contact limit *readings* (device id -> channel id -> value) break, or ``""``."""
    for limit in limits:
        problem = limit.check(readings.get(limit.device_id, {}).get(limit.channel_id))
        if problem:
            return problem
    return ""


__all__ = [
    "CONTACT_QUANTITIES",
    "ContactLimit",
    "PoseCheck",
    "SafetyShield",
    "ShieldChannel",
    "ShieldVerdict",
    "contact_limits",
    "contact_violation",
]
