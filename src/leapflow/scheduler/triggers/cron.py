"""Cron-expression trigger with graceful fallback.

Uses ``croniter`` when available for full cron expression support.
Falls back to a minimal daily HH:MM parser otherwise.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import ClassVar, Optional

try:
    from croniter import croniter as _croniter  # type: ignore[import-untyped]

    _HAS_CRONITER = True
except ImportError:
    _croniter = None
    _HAS_CRONITER = False


class CronTrigger:
    """Cron-based trigger: fires according to a cron expression.

    Full cron support requires the ``croniter`` package. Without it,
    only simple "HH:MM" daily schedules are supported.
    """

    trigger_type: ClassVar[str] = "cron"

    def __init__(self, expression: str, *, next_due_at: float = 0.0) -> None:
        self._expression = expression.strip()
        self._next_due_at = next_due_at

        # Validate and compute first due time if not provided
        if self._next_due_at <= 0.0:
            self._next_due_at = self._compute_next(time.time())

    # ------------------------------------------------------------------
    # Trigger Protocol
    # ------------------------------------------------------------------

    @property
    def trigger_type(self) -> str:  # type: ignore[override]
        return "cron"

    def is_due(self, now: float) -> bool:
        """Return True if now >= next_due_at."""
        return now >= self._next_due_at

    def advance(self, now: float) -> None:
        """Compute and set the next trigger time after *now*."""
        self._next_due_at = self._compute_next(now)

    @property
    def next_due_at(self) -> float:
        return self._next_due_at

    def serialize(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "trigger_type": "cron",
            "expression": self._expression,
            "next_due_at": self._next_due_at,
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def deserialize(cls, data: dict) -> "CronTrigger":
        """Reconstruct from serialized dict."""
        return cls(
            expression=data["expression"],
            next_due_at=data.get("next_due_at", 0.0),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_next(self, after: float) -> float:
        """Compute the next trigger time after the given timestamp."""
        if _HAS_CRONITER:
            return self._compute_next_croniter(after)
        return self._compute_next_fallback(after)

    def _compute_next_croniter(self, after: float) -> float:
        """Use croniter for full cron expression support."""
        dt = datetime.fromtimestamp(after, tz=timezone.utc)
        cron = _croniter(self._expression, dt)
        next_dt = cron.get_next(datetime)
        return next_dt.timestamp()

    def _compute_next_fallback(self, after: float) -> float:
        """Fallback: support only 'HH:MM' daily schedules or '* * * * *' style.

        Parses simple HH:MM format and schedules for the next occurrence.
        For full cron expressions without croniter, raises a clear error.
        """
        parsed = self._try_parse_hhmm(self._expression)
        if parsed is None:
            raise ValueError(
                f"Cannot parse cron expression {self._expression!r} without "
                f"the 'croniter' package. Install it or use HH:MM format."
            )
        hour, minute = parsed
        dt_after = datetime.fromtimestamp(after, tz=timezone.utc)
        # Build candidate for today
        candidate = dt_after.replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        if candidate.timestamp() <= after:
            # Already passed today, schedule for tomorrow
            from datetime import timedelta

            candidate = candidate + timedelta(days=1)
        return candidate.timestamp()

    @staticmethod
    def _try_parse_hhmm(expression: str) -> Optional[tuple[int, int]]:
        """Try to parse 'HH:MM' or cron '0 9 * * *' into (hour, minute).

        Returns None if parsing fails.
        """
        # Try HH:MM format
        if ":" in expression and len(expression) <= 5:
            parts = expression.split(":")
            if len(parts) == 2:
                try:
                    h, m = int(parts[0]), int(parts[1])
                    if 0 <= h <= 23 and 0 <= m <= 59:
                        return (h, m)
                except ValueError:
                    pass

        # Try standard 5-field cron where day/month/dow are all *
        fields = expression.split()
        if len(fields) == 5 and fields[2] == "*" and fields[3] == "*" and fields[4] == "*":
            try:
                minute = int(fields[0])
                hour = int(fields[1])
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    return (hour, minute)
            except ValueError:
                pass

        return None

    @property
    def expression(self) -> str:
        """The cron expression string."""
        return self._expression

    def __repr__(self) -> str:
        return (
            f"CronTrigger(expression={self._expression!r}, "
            f"next_due_at={self._next_due_at})"
        )
