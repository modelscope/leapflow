"""Interval-based trigger: fires every N seconds/minutes/hours/days."""

from __future__ import annotations

import re
from typing import ClassVar, Dict


_UNIT_MAP: Dict[str, float] = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
}

_INTERVAL_RE = re.compile(
    r"(?:every\s+)?(\d+(?:\.\d+)?)\s*([a-zA-Z]+)", re.IGNORECASE
)


def _parse_interval(spec: str) -> float:
    """Parse an interval specification string into seconds.

    Supported formats:
        "30m", "2h", "1d", "every 5m", "90s", "1.5h"
    """
    match = _INTERVAL_RE.match(spec.strip())
    if not match:
        # Try pure numeric (seconds)
        try:
            return float(spec.strip())
        except ValueError:
            raise ValueError(f"Cannot parse interval: {spec!r}")

    value = float(match.group(1))
    unit = match.group(2).lower()
    multiplier = _UNIT_MAP.get(unit)
    if multiplier is None:
        raise ValueError(f"Unknown interval unit: {unit!r} in {spec!r}")
    return value * multiplier


class IntervalTrigger:
    """Fires at fixed time intervals.

    Accepts human-readable specs like "30m", "2h", "1d", "every 5m".
    """

    trigger_type: ClassVar[str] = "interval"

    def __init__(self, interval_seconds: float, *, next_due_at: float = 0.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._interval_seconds = interval_seconds
        self._next_due_at = next_due_at

    # ------------------------------------------------------------------
    # Trigger Protocol
    # ------------------------------------------------------------------

    @property
    def trigger_type(self) -> str:  # type: ignore[override]
        return "interval"

    def is_due(self, now: float) -> bool:
        """Return True if now >= next_due_at."""
        return now >= self._next_due_at

    def advance(self, now: float) -> None:
        """Set next trigger point to now + interval."""
        self._next_due_at = now + self._interval_seconds

    @property
    def next_due_at(self) -> float:
        return self._next_due_at

    def serialize(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "trigger_type": "interval",
            "interval_seconds": self._interval_seconds,
            "next_due_at": self._next_due_at,
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def deserialize(cls, data: dict) -> "IntervalTrigger":
        """Reconstruct from serialized dict."""
        return cls(
            interval_seconds=data["interval_seconds"],
            next_due_at=data.get("next_due_at", 0.0),
        )

    @classmethod
    def from_spec(cls, spec: str, *, now: float = 0.0) -> "IntervalTrigger":
        """Create from a human-readable spec string.

        Args:
            spec: e.g. "30m", "2h", "every 5m"
            now: current timestamp; first trigger at now + interval
        """
        seconds = _parse_interval(spec)
        return cls(interval_seconds=seconds, next_due_at=now + seconds)

    @property
    def interval_seconds(self) -> float:
        """The configured interval in seconds."""
        return self._interval_seconds

    def __repr__(self) -> str:
        return (
            f"IntervalTrigger(interval_seconds={self._interval_seconds}, "
            f"next_due_at={self._next_due_at})"
        )
