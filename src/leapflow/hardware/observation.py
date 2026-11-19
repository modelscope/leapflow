# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bounded operational observation, independent of command and task authority.

One helper for policy validation, confirmation and bounded dispersion. Raw evidence
is never rewritten. Episode is also used by advisory reporting, so notification
backoff has one implementation and does not imply firmware protection.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass, field, replace
from statistics import median
from typing import Any, Mapping
from uuid import uuid4

from leapflow.hardware.context import Channel, as_numeric
from leapflow.hardware.transport import Reading

DEFAULT_REPORT_AFTER_S = 60.0
DEFAULT_REPORT_MAX_S = 900.0
MAX_PENDING_TRANSITIONS = 32


def calibration_identity(value: Mapping[str, Any]) -> str:
    """Stable calibration scope from existing records or lifecycle telemetry."""
    if value.get("kind") == "invalidated" and value.get("event_id"):
        return str(value["event_id"])
    if value.get("workcell_id") and value.get("generation") is not None:
        # Generations are immutable within a workcell. Rollback telemetry carries
        # the generation identity but need not repeat its model/dependency digests.
        return json.dumps(
            {key: value[key] for key in ("workcell_id", "generation")}, sort_keys=True,
        )
    if value.get("procedure_id") and value.get("recorded_at") is not None:
        return repr((value["procedure_id"], float(value["recorded_at"])))
    return str(value.get("event_id") or repr(dict(value)))


@dataclass(frozen=True)
class ObservationPolicy:
    soft_min: float | None = None
    soft_max: float | None = None
    margin_floor: float | None = None
    margin_max: float | None = None
    confirm_s: float = 0.2
    stale_confirm_s: float = 0.5
    rate_confirm_s: float = 0.5
    rate_recover_s: float = 1.0
    recover_s: float = 0.5
    adaptive: bool = False
    noise_window_s: float = 5.0
    min_fresh_samples: int = 32
    noise_multiplier: float = 3.0
    report_after_s: float = DEFAULT_REPORT_AFTER_S
    report_max_s: float = DEFAULT_REPORT_MAX_S

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "adaptive":
                if not isinstance(value, bool):
                    raise ValueError("observation.adaptive must be boolean")
                continue
            if value is None and name in {"soft_min", "soft_max", "margin_floor", "margin_max"}:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"observation.{name} must be finite numeric data")
            if name not in {"soft_min", "soft_max"} and value < 0:
                raise ValueError(f"observation.{name} must not be negative")
        if (
            self.soft_min is not None
            and self.soft_max is not None
            and self.soft_min >= self.soft_max
        ):
            raise ValueError("observation soft boundaries must be ordered")
        if self.margin_max is not None and self.margin_max < (self.margin_floor or 0.0):
            raise ValueError("observation.margin_max must not be below the floor")
        if self.adaptive and self.margin_max is None:
            raise ValueError("adaptive observation requires an authorized margin_max")
        if not isinstance(self.min_fresh_samples, int) or not 4 <= self.min_fresh_samples <= 4096:
            raise ValueError("observation.min_fresh_samples must be an integer in 4..4096")
        if self.noise_window_s <= 0 or self.noise_multiplier <= 0 or self.report_after_s <= 0:
            raise ValueError(
                "observation window, multiplier and reminder interval must be positive"
            )
        if self.rate_confirm_s <= 0 or self.rate_recover_s <= 0:
            raise ValueError("observation rate confirmation and recovery must be positive")
        if self.report_max_s < self.report_after_s:
            raise ValueError("observation.report_max_s must not be below report_after_s")

    def for_channel(self, channel: Channel) -> ObservationPolicy:
        envelope = channel.envelope
        floor = self.margin_floor
        if floor is None:
            floor = max(envelope.tolerance or 0.0, envelope.quantization or 0.0)
        stale_window = max(0.5, self.stale_confirm_s, 2.0 / channel.sample_rate_hz) if channel.sample_rate_hz > 0 else max(0.5, self.stale_confirm_s)
        policy = replace(self, margin_floor=floor, stale_confirm_s=stale_window)
        ceiling = policy.margin_max if policy.margin_max is not None else floor
        if ceiling < floor:
            raise ValueError(
                f"observation ceiling below declared measurement floor on {channel.channel_id}"
            )
        if policy.soft_min is None and policy.soft_max is None:
            return policy
        if (
            not channel.is_readable
            or channel.is_media
            or not envelope.declared
            or envelope.allowed_values
        ):
            raise ValueError(
                f"observation soft boundaries require a declared numeric reading: {channel.channel_id}"
            )
        for soft, direction in ((policy.soft_min, -1), (policy.soft_max, 1)):
            if soft is None:
                continue
            low, high = envelope.min_value, envelope.max_value
            if low is not None and soft - ceiling < low:
                raise ValueError(
                    f"observation threshold outside hard lower boundary: {channel.channel_id}"
                )
            if high is not None and soft + ceiling > high:
                raise ValueError(
                    f"observation threshold outside hard upper boundary: {channel.channel_id}"
                )
            if direction == -1 and high is not None and soft + ceiling >= high:
                raise ValueError("observation lower recovery band is collapsed")
            if direction == 1 and low is not None and soft - ceiling <= low:
                raise ValueError("observation upper recovery band is collapsed")
        if policy.soft_min is not None and policy.soft_max is not None:
            if policy.soft_min + ceiling >= policy.soft_max - ceiling:
                raise ValueError("observation recovery bands must not collapse")
        return policy


@dataclass(frozen=True)
class ObservationSettings:
    defaults: ObservationPolicy = field(default_factory=ObservationPolicy)
    overrides: tuple[tuple[str, str, ObservationPolicy], ...] = ()

    @classmethod
    def parse(cls, raw: Any) -> ObservationSettings:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError("hardware.observation must be valid JSON") from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping) or set(raw) - {"defaults", "overrides"}:
            raise ValueError("hardware.observation accepts only defaults and exact overrides")
        defaults = raw.get("defaults", {})
        if (
            not isinstance(defaults, Mapping)
            or set(defaults) - ObservationPolicy.__dataclass_fields__.keys()
        ):
            raise ValueError("unknown observation default keys")
        policy = ObservationPolicy(**defaults)
        entries = raw.get("overrides", [])
        if not isinstance(entries, (list, tuple)):
            raise ValueError("observation.overrides must be an array")
        overrides = []
        identities = set()
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("observation override must be a mapping")
            data = dict(entry)
            device, channel = data.pop("device_id", None), data.pop("channel_id", None)
            if not all(
                isinstance(v, str) and v and not any(c in v for c in "*?[]")
                for v in (device, channel)
            ):
                raise ValueError("observation override requires exact device_id and channel_id")
            if (device, channel) in identities or set(
                data
            ) - ObservationPolicy.__dataclass_fields__.keys():
                raise ValueError("duplicate observation override or unknown keys")
            identities.add((device, channel))
            overrides.append((device, channel, replace(policy, **data)))
        return cls(policy, tuple(overrides))

    def resolve(self, device_id: str, channel: Channel) -> ObservationPolicy:
        policy = next(
            (p for d, c, p in self.overrides if (d, c) == (device_id, channel.channel_id)),
            self.defaults,
        )
        return policy.for_channel(channel)

    def validate_contexts(self, contexts: Any) -> None:
        admitted = {(ctx.device_id, ch.channel_id): ch for ctx in contexts for ch in ctx.channels}
        for device, channel, _ in self.overrides:
            if (device, channel) not in admitted:
                raise ValueError(
                    f"observation override names an unadmitted channel: {device}.{channel}"
                )
        for (device, _), channel in admitted.items():
            self.resolve(device, channel)


@dataclass
class Episode:
    """Shared entry/recovery and exponential reminder lifecycle, no polling loop."""

    active: bool = False
    known: bool = False
    since: float = 0.0
    next_report: float = 0.0
    interval: float = 0.0
    folded: int = 0
    episode: str = ""
    sequence: int = 0
    candidate_at: float | None = None
    recovering_at: float | None = None

    def unknown(self) -> None:
        self.known = False
        self.candidate_at = self.recovering_at = None

    def _revalidate(self, breached: bool, clear: bool, now: float, confirm: float,
                    recover: float, report_after: float) -> str:
        """An unknown active episode needs new continuous evidence, not a new UUID."""
        if breached:
            self.recovering_at = None
            if self.candidate_at is None:
                self.candidate_at = now
            if now - self.candidate_at + 1e-9 < confirm:
                return ""
            self.since, self.candidate_at, self.known = self.candidate_at, None, True
            self.next_report = now + report_after
            self.sequence += 1
            return "entry"
        self.candidate_at = None
        if clear:
            if self.recovering_at is None:
                self.recovering_at = now
            if now - self.recovering_at + 1e-9 >= recover:
                self.active, self.known, self.recovering_at = False, True, None
                self.sequence += 1
                return "recovery"
        else:
            self.recovering_at = None
        return ""

    def observe(
        self,
        *,
        breached: bool,
        clear: bool,
        now: float,
        confirm_s: float = 0.0,
        recover_s: float = 0.0,
        revalidate_s: float = 0.0,
        report_after_s: float = DEFAULT_REPORT_AFTER_S,
        report_max_s: float = DEFAULT_REPORT_MAX_S,
    ) -> str:
        if self.active and not self.known and revalidate_s > 0:
            return self._revalidate(breached, clear, now, revalidate_s, recover_s, report_after_s)
        self.known = True
        if not self.active:
            self.recovering_at = None
            if not breached:
                self.candidate_at = None
                return ""
            if self.candidate_at is None:
                self.candidate_at = now
            if now - self.candidate_at + 1e-9 < confirm_s:
                return ""
            self.active, self.since = True, self.candidate_at
            self.episode, self.sequence = uuid4().hex, 0
            self.interval, self.next_report = report_after_s, now + report_after_s
            self.folded = 0
            return "entry"
        self.candidate_at = None
        if clear:
            if self.recovering_at is None:
                self.recovering_at = now
            if now - self.recovering_at + 1e-9 >= recover_s:
                self.active, self.recovering_at = False, None
                self.sequence += 1
                return "recovery"
        else:
            self.recovering_at = None
        if now >= self.next_report:
            self.sequence += 1
            self.interval = min(self.interval * 2.0, report_max_s)
            self.next_report, self.folded = now + self.interval, 0
            return "reminder"
        self.folded += 1
        return ""


class ObservationJudgment:
    """Per-channel state and notices. Rings supply bounded raw noise evidence."""

    def __init__(self, device_id: str, channel: Channel, policy: ObservationPolicy) -> None:
        self.device_id, self.channel, self.policy = device_id, channel, policy.for_channel(channel)
        self.identity = hashlib.sha256(repr(channel.to_dict()).encode()).hexdigest()[:20]
        self._declaration_identity = self.identity
        self.runtime_id = uuid4().hex
        self.states: dict[str, Episode] = {}
        self.pending: dict[str, deque[dict[str, Any]]] = {}
        self.latest: dict[str, dict[str, Any]] = {}
        self.margin = float(self.policy.margin_floor or 0.0)
        self._training_since: float | None = None
        self._estimated_at = -math.inf
        self._last_at: float | None = None
        self._epoch = ""
        self._noise_excess = False
        self._noise_valid = False
        self._revisions: dict[str, int] = {}
        self._pending_omitted: dict[str, int] = {}

    def prepare_epoch(self, reading: Reading) -> None:
        if reading.acquisition_epoch != self._epoch:
            self.reconfigure(self.policy)
            self._epoch = reading.acquisition_epoch

    def reconfigure(self, policy: ObservationPolicy) -> None:
        self.policy = policy.for_channel(self.channel)
        # Already confirmed evidence survives reload until the producer checkpoints it.
        for name in ("soft_lower", "soft_upper", "observation_quality"):
            if name in self.states:
                self.states[name] = Episode()
        self.invalidate()

    def reset_identity(self, identity: str) -> None:
        changed = hashlib.sha256(f"{self._declaration_identity}/{identity}".encode()).hexdigest()[
            :20
        ]
        if changed != self.identity:
            self.identity = changed
            self.reconfigure(self.policy)

    def invalidate(self) -> None:
        for name, state in self.states.items():
            # Numeric/soft evidence can be invalid while explicit freshness still
            # has a continuous candidate. Resetting it each stale sample defeats debounce.
            if name not in {"stale", "quality_degraded"}:
                state.unknown()
        self._training_since = None
        self._estimated_at = -math.inf
        self._last_at = None
        if not self._margin_frozen:
            self.margin = float(self.policy.margin_floor or 0.0)
        self._noise_excess = False
        self._noise_valid = False

    @property
    def _margin_frozen(self) -> bool:
        return any(
            state.active
            for name, state in self.states.items()
            if name.startswith("soft_") or name == "observation_quality"
        )

    def condition(
        self,
        name: str,
        *,
        breached: bool,
        clear: bool,
        reading: Reading,
        now: float,
        valid: bool = True,
        soft: bool = False,
        reason: str = "",
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        state = self.states.setdefault(name, Episode())
        policy = self.policy
        phase = ""
        confirm = (policy.stale_confirm_s if name == "stale" else
                   policy.rate_confirm_s if name == "rate_exceeded" else
                   policy.confirm_s if soft else 0.0)
        recovery = policy.rate_recover_s if name == "rate_exceeded" else policy.recover_s if soft else 0.0
        revalidating = name == "rate_exceeded" and state.active and not state.known
        if not valid:
            state.unknown()
        else:
            phase = state.observe(
                breached=breached,
                clear=clear,
                now=now,
                confirm_s=confirm,
                recover_s=recovery,
                revalidate_s=confirm if name == "rate_exceeded" else 0.0,
                report_after_s=policy.report_after_s,
                report_max_s=policy.report_max_s,
            )
        data = {
            "key": f"{self.device_id}/{self.channel.channel_id}/{name}",
            "condition": name,
            "device_id": self.device_id,
            "channel_id": self.channel.channel_id,
            "identity": self.identity,
            "runtime_id": self.runtime_id,
            "episode": state.episode,
            "sequence": state.sequence,
            "active": state.active,
            "known": state.known,
            "phase": phase,
            "value": as_numeric(reading.value),
            "unit": self.channel.unit,
            "reason": reason,
            "duration_s": max(0.0, now - state.since) if state.episode else 0.0,
            "observed_at": reading.observed_at,
            "freshness_s": 2.0 / self.channel.sample_rate_hz if self.channel.sample_rate_hz > 0 else 0.0,
            "margin": self.margin,
            "soft_min": policy.soft_min,
            "soft_max": policy.soft_max,
            "hard_min": self.channel.envelope.min_value,
            "hard_max": self.channel.envelope.max_value,
            "report_after_s": policy.report_after_s,
            "report_max_s": policy.report_max_s,
            "confirm_s": confirm,
            "recover_s": recovery,
            "clear": clear and valid,
            "reconfirmed": revalidating and phase == "entry",
        }
        data.update(extra or {})
        self.latest[name] = data
        if not phase:
            return None
        notices = self.pending.setdefault(name, deque(maxlen=MAX_PENDING_TRANSITIONS))
        # Repeated reminders can be folded, but entry and recovery are distinct evidence.
        if phase == "reminder" and notices and notices[-1]["phase"] == "reminder":
            notices.pop()
        revision = self._revisions.get(name, 0) + 1
        self._revisions[name] = revision
        data["revision"] = revision
        if len(notices) == MAX_PENDING_TRANSITIONS:
            self._pending_omitted[name] = self._pending_omitted.get(name, 0) + 1
        notices.append(dict(data))
        return data

    def observe_soft(
        self,
        reading: Reading,
        *,
        ring: Any = None,
        pending_motion: bool = False,
        fault_free: bool = False,
    ) -> list[dict[str, Any]]:
        policy = self.policy
        if policy.soft_min is None and policy.soft_max is None:
            return []
        numeric = as_numeric(reading.value)
        at = reading.acquisition_at if reading.acquisition_at is not None else reading.monotonic_at
        valid = numeric is not None and reading.is_trustworthy and math.isfinite(at)
        self.prepare_epoch(reading)
        if self._last_at is not None and at <= self._last_at:
            valid = False
        if self._last_at is not None and self.channel.sample_rate_hz > 0:
            if at - self._last_at > 2.0 / self.channel.sample_rate_hz:
                self.invalidate()
        if not valid:
            self.invalidate()
            return []
        self._last_at = at
        notices = []
        for name, bound, direction in (
            ("soft_lower", policy.soft_min, -1),
            ("soft_upper", policy.soft_max, 1),
        ):
            if bound is None:
                continue
            delta = direction * (numeric - bound)
            notice = self.condition(
                name,
                breached=delta > self.margin,
                clear=delta < -self.margin,
                reading=reading,
                now=at,
                soft=True,
                reason="continuous raw observation outside operational band",
            )
            if notice:
                notices.append(notice)
        healthy = (
            not pending_motion
            and fault_free
            and self.channel.envelope.contains(numeric)
            and all(
                not s.active and s.candidate_at is None
                for name, s in self.states.items()
                if name != "observation_quality"
            )
            and (policy.soft_min is None or numeric > policy.soft_min + self.margin)
            and (policy.soft_max is None or numeric < policy.soft_max - self.margin)
        )
        if not healthy:
            self._training_since = None
            self._noise_valid = False
        elif policy.adaptive and ring is not None:
            self._estimate(ring, at)
        if policy.adaptive:
            notice = self.condition(
                "observation_quality",
                breached=self._noise_excess,
                clear=not self._noise_excess,
                reading=reading,
                now=at,
                soft=True,
                valid=self._noise_valid,
                reason="healthy dispersion exceeds authorized margin ceiling",
            )
            if notice:
                notices.append(notice)
        return notices

    def _estimate(self, ring: Any, at: float) -> None:
        policy = self.policy
        if self._training_since is None:
            self._training_since = at
        if at - self._training_since < self.channel.envelope.effective_settling_s:
            return
        if at - self._estimated_at < policy.noise_window_s / 4.0:
            return
        floor, ceiling = float(policy.margin_floor or 0.0), float(policy.margin_max or 0.0)
        cutoff = max(
            self._training_since + self.channel.envelope.effective_settling_s,
            at - policy.noise_window_s,
        )
        self._noise_valid = False
        numbers: list[float] = []
        previous: float | None = None
        for reading in ring:
            stamp = (
                reading.acquisition_at
                if reading.acquisition_at is not None
                else reading.monotonic_at
            )
            if stamp < cutoff:
                continue
            value = as_numeric(reading.value)
            if (
                value is None
                or not reading.is_trustworthy
                or reading.acquisition_epoch != self._epoch
                or not math.isfinite(stamp)
                or (
                    previous is not None
                    and (
                        stamp <= previous
                        or (
                            self.channel.sample_rate_hz > 0
                            and stamp - previous > 2.0 / self.channel.sample_rate_hz
                        )
                    )
                )
            ):
                return
            previous = stamp
            if not self.channel.envelope.contains(value, margin=ceiling + floor):
                return
            numbers.append(value)
        if len(numbers) < policy.min_fresh_samples:
            if not self._margin_frozen:
                self.margin = floor
            self._noise_valid = False
            return
        center = median(numbers)
        sigma = 1.4826 * median(abs(v - center) for v in numbers)
        # A moving baseline is not random noise. No bias is inferred or compensated.
        half = len(numbers) // 2
        trend = abs(median(numbers[:half]) - median(numbers[half:]))
        if trend > max(floor, sigma):
            self._training_since = at
            if not self._margin_frozen:
                self.margin = floor
            return
        self._estimated_at = at
        self._noise_valid = True
        requested = max(floor, policy.noise_multiplier * sigma)
        self._noise_excess = requested > ceiling
        # Dispersion can qualify quality recovery without retraining the margin
        # during a warning episode.
        if not self._margin_frozen:
            self.margin = min(ceiling, requested)

    def acknowledge(self, checkpoints: Mapping[str, Mapping[str, Any]]) -> None:
        """Discard only notices confirmed durable in the existing finding store."""
        for name, notices in self.pending.items():
            checkpoint = checkpoints.get(f"{self.device_id}/{self.channel.channel_id}/{name}", {})
            if checkpoint.get("runtime_id") != self.runtime_id:
                continue
            revision = checkpoint.get("revision", 0)
            while notices and notices[0].get("revision", 0) <= revision:
                notices.popleft()

    def snapshots(self) -> tuple[dict[str, Any], ...]:
        rows = []
        for name, state in self.states.items():
            row = dict(self.latest.get(name, {}))
            if not row:
                continue
            row.update(
                active=state.active,
                known=state.known,
                identity=self.identity,
                margin=self.margin,
                pending_omitted=self._pending_omitted.get(name, 0),
                pending=tuple(dict(n) for n in self.pending.get(name, ())),
            )
            rows.append(row)
        return tuple(rows)
