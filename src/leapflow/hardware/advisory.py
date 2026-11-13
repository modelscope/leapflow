# Copyright (c) Alibaba, Inc. and its affiliates.
"""Advisory limits: monitor-only conditions a device already protects itself against.

Some limits -- a drive's temperature, its supply voltage, a firmware over-heat flag --
are enforced by the device's own firmware. Refusing commands on them a second time
in LeapFlow only stops work the hardware would have allowed, so they are *reported*
instead: a breach becomes an ``advisory_exceeded`` signal on the board, never a gate.

Limits come from host configuration (``hardware.advisory_limits``), not from device
declarations, so tuning what is watched never changes a verified declaration digest.

Reporting follows exponential backoff. Entering a breach is reported at once; a
breach that persists is re-reported after ``report_after_s``, then at doubling
intervals up to ``report_max_s``, each repeat naming how long it has lasted and how
many polls were folded into it. A recovery past the hysteresis band is reported once
as ``advisory_cleared`` and resets the backoff, so the next breach is immediate again.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from leapflow.hardware.stream import EventKind, HardwareEvent

logger = logging.getLogger(__name__)

ADVISORY_EXCEEDED = EventKind.ADVISORY_EXCEEDED
ADVISORY_CLEARED = EventKind.ADVISORY_CLEARED
ADVISORY_KINDS = frozenset({ADVISORY_EXCEEDED, ADVISORY_CLEARED})

DEFAULT_REPORT_AFTER_S = 60.0
DEFAULT_REPORT_MAX_S = 900.0
DEFAULT_READ_TIMEOUT_S = 2.0
_WILDCARD = "*"


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _truthy(item: Any) -> bool:
    if isinstance(item, bool):
        return item
    number = _finite(item)
    return number is not None and number != 0.0


@dataclass(frozen=True)
class AdvisoryLimit:
    """One monitored quantity on one channel.

    ``field`` is a dotted path into a structured reading; ``*`` matches every list
    item or mapping key at that level (``motors.*.driver_temperature_c``). An empty
    path watches the reading value itself. A numeric limit breaches outside
    ``min_value``..``max_value``; a ``flag`` limit breaches on any truthy match.
    """

    device_id: str
    channel_id: str
    field: str = ""
    label: str = ""
    min_value: float | None = None
    max_value: float | None = None
    flag: bool = False
    unit: str = ""
    hysteresis: float = 0.0
    report_after_s: float = DEFAULT_REPORT_AFTER_S
    report_max_s: float = DEFAULT_REPORT_MAX_S

    @property
    def name(self) -> str:
        return self.label or self.field or self.channel_id

    @property
    def is_valid(self) -> bool:
        return bool(self.device_id and self.channel_id) and (
            self.flag or self.min_value is not None or self.max_value is not None
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AdvisoryLimit":
        after = _finite(data.get("report_after_s"))
        ceiling = _finite(data.get("report_max_s"))
        after = after if after is not None and after > 0 else DEFAULT_REPORT_AFTER_S
        return cls(
            device_id=str(data.get("device_id") or ""),
            channel_id=str(data.get("channel_id") or ""),
            field=str(data.get("field") or ""),
            label=str(data.get("label") or ""),
            min_value=_finite(data.get("min")),
            max_value=_finite(data.get("max")),
            flag=bool(data.get("flag", False)),
            unit=str(data.get("unit") or ""),
            hysteresis=max(0.0, _finite(data.get("hysteresis")) or 0.0),
            report_after_s=after,
            report_max_s=max(after, ceiling if ceiling is not None else DEFAULT_REPORT_MAX_S),
        )

    def judge(self, value: Any) -> "_Judgement":
        """Return what *value* says about this limit (``unknown`` when nothing matched)."""
        matches = resolve_field(value, self.field)
        if self.flag:
            breached = [path for path, item in matches if _truthy(item)]
            known = any(isinstance(item, bool) or _finite(item) is not None for _, item in matches)
            return _Judgement(known, bool(breached), not breached, breached, None)
        numbers = [(path, n) for path, item in matches if (n := _finite(item)) is not None]
        if not numbers:
            return _Judgement(False, False, False, [], None)
        low, high = self.min_value, self.max_value
        breached = [p for p, n in numbers if (low is not None and n < low) or (high is not None and n > high)]
        margin = self.hysteresis
        clear = all((low is None or n >= low + margin) and (high is None or n <= high - margin)
                    for _, n in numbers)
        worst = max(numbers, key=lambda row: self._excess(row[1]))[1]
        return _Judgement(True, bool(breached), clear, breached, worst)

    def _excess(self, number: float) -> float:
        excess = -math.inf
        if self.max_value is not None:
            excess = max(excess, number - self.max_value)
        if self.min_value is not None:
            excess = max(excess, self.min_value - number)
        return excess

    def bounds(self) -> str:
        if self.flag:
            return "flag set"
        low = "-inf" if self.min_value is None else f"{self.min_value:g}"
        high = "+inf" if self.max_value is None else f"{self.max_value:g}"
        return f"outside [{low}, {high}]{f' {self.unit}' if self.unit else ''}"


@dataclass(frozen=True)
class _Judgement:
    known: bool
    breached: bool
    clear: bool
    paths: list[str]
    worst: float | None


def resolve_field(value: Any, path: str) -> list[tuple[str, Any]]:
    """Return ``(concrete_path, item)`` for every match of a dotted *path* in *value*."""
    rows: list[tuple[str, Any]] = [("", value)]
    for part in [p for p in path.split(".") if p] if path else []:
        nxt: list[tuple[str, Any]] = []
        for where, item in rows:
            for key, child in _children(item, part):
                nxt.append((f"{where}.{key}" if where else str(key), child))
        rows = nxt
    return rows


def _children(item: Any, part: str) -> Iterable[tuple[Any, Any]]:
    if isinstance(item, Mapping):
        if part == _WILDCARD:
            return list(item.items())
        return [(part, item[part])] if part in item else []
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
        if part == _WILDCARD:
            return list(enumerate(item))
        if part.isdigit() and int(part) < len(item):
            return [(int(part), item[int(part)])]
    return []


@dataclass
class _State:
    active: bool = False
    since: float = 0.0
    next_report: float = 0.0
    interval: float = 0.0
    folded: int = 0


@dataclass
class AdvisoryTracker:
    """Turn successive judgements into backed-off breach and recovery events."""

    clock: Callable[[], float] = time.time
    _states: dict[tuple[str, str, str], _State] = field(default_factory=dict)

    def observe(self, limit: AdvisoryLimit, value: Any) -> list[HardwareEvent]:
        judgement = limit.judge(value)
        if not judgement.known:
            return []  # nothing matched: not evidence either way
        key = (limit.device_id, limit.channel_id, limit.name)
        state = self._states.setdefault(key, _State())
        now = float(self.clock())
        if judgement.breached:
            if not state.active:
                state.active, state.since, state.folded = True, now, 0
                state.interval = limit.report_after_s
                state.next_report = now + state.interval
                return [self._event(ADVISORY_EXCEEDED, limit, judgement, now, "")]
            if now < state.next_report:
                state.folded += 1
                return []
            note = f"; still breached after {now - state.since:.0f}s, {state.folded} poll(s) folded"
            state.interval = min(state.interval * 2.0, limit.report_max_s)
            state.next_report, state.folded = now + state.interval, 0
            return [self._event(ADVISORY_EXCEEDED, limit, judgement, now, note)]
        if state.active and judgement.clear:
            lasted = now - state.since
            self._states.pop(key, None)
            return [self._event(ADVISORY_CLEARED, limit, judgement, now, f" after {lasted:.0f}s")]
        return []

    @staticmethod
    def _event(kind: str, limit: AdvisoryLimit, judgement: _Judgement, now: float,
               note: str) -> HardwareEvent:
        if kind == ADVISORY_CLEARED:
            detail = f"{limit.name} back within limits{note}"
        else:
            where = ", ".join(judgement.paths[:6]) + ("…" if len(judgement.paths) > 6 else "")
            detail = (f"{limit.name} {limit.bounds()} at {where}; device protection active, "
                      f"reported only{note}")
        return HardwareEvent(
            kind=kind,
            device_id=limit.device_id,
            channel_id=limit.channel_id,
            quantity=limit.name,
            detail=detail,
            value=judgement.worst,
            unit=limit.unit,
            observed_at=now,
        )


class AdvisoryWatch:
    """Read each watched channel once per poll and judge every limit on it."""

    def __init__(self, registry: Any, limits: Sequence[AdvisoryLimit], *,
                 read_timeout_s: float = DEFAULT_READ_TIMEOUT_S,
                 clock: Callable[[], float] = time.time) -> None:
        self._registry = registry
        self._limits = tuple(limits)
        self._timeout = max(0.1, float(read_timeout_s))
        self._tracker = AdvisoryTracker(clock=clock)

    @property
    def limits(self) -> tuple[AdvisoryLimit, ...]:
        return self._limits

    async def poll(self) -> list[HardwareEvent]:
        """Return this poll's advisory events. Never raises; an unread channel is skipped."""
        events: list[HardwareEvent] = []
        channels: dict[tuple[str, str], list[AdvisoryLimit]] = {}
        for limit in self._limits:
            channels.setdefault((limit.device_id, limit.channel_id), []).append(limit)
        for (device_id, channel_id), limits in channels.items():
            value = await self._read(device_id, channel_id)
            if value is None:
                continue
            for limit in limits:
                events.extend(self._tracker.observe(limit, value))
        return events

    async def _read(self, device_id: str, channel_id: str) -> Any:
        try:
            reading = await asyncio.wait_for(
                self._registry.read(device_id, channel_id), timeout=self._timeout
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - liveness is the probe's job, not this watch's
            logger.debug("advisory read failed for %s.%s", device_id, channel_id, exc_info=True)
            return None
        if str(getattr(reading, "quality", "ok")) != "ok":
            return None
        return getattr(reading, "value", reading)


def load_advisory_limits(settings: Any) -> tuple[AdvisoryLimit, ...]:
    """Load ``hardware.advisory_limits``; absent, empty or malformed yields no limits."""
    raw = getattr(settings, "hardware_advisory_limits", None)
    if not raw:
        return ()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("hardware.advisory_limits must be valid JSON; ignoring")
            return ()
    if not isinstance(raw, (list, tuple)):
        logger.warning("hardware.advisory_limits must be a list; ignoring")
        return ()
    limits: list[AdvisoryLimit] = []
    for index, entry in enumerate(raw):
        limit = AdvisoryLimit.from_mapping(entry) if isinstance(entry, Mapping) else None
        if limit is None or not limit.is_valid:
            logger.warning(
                "hardware.advisory_limits[%d] needs device_id, channel_id and min/max or flag; "
                "skipping", index,
            )
            continue
        limits.append(limit)
    return tuple(limits)


def active_advisories(events: Iterable[Mapping[str, Any]]) -> list[str]:
    """Name the advisories whose newest event is a breach (events newest first)."""
    seen: set[tuple[str, str, str]] = set()
    active: list[str] = []
    for event in events:
        kind = str(event.get("kind", ""))
        if kind not in ADVISORY_KINDS:
            continue
        key = (str(event.get("device_id", "")), str(event.get("channel_id", "")),
               str(event.get("quantity", "")))
        if key in seen:
            continue
        seen.add(key)
        if kind == ADVISORY_EXCEEDED:
            active.append(f"{key[0]}.{key[2] or key[1]}")
    return active


__all__ = [
    "ADVISORY_CLEARED",
    "ADVISORY_EXCEEDED",
    "ADVISORY_KINDS",
    "AdvisoryLimit",
    "AdvisoryTracker",
    "AdvisoryWatch",
    "active_advisories",
    "load_advisory_limits",
    "resolve_field",
]
