"""Event-based trigger: fires when an external event matches a pattern."""

from __future__ import annotations

import fnmatch
import time
from typing import ClassVar


class EventTrigger:
    """Fires when an external event matching the configured pattern occurs.

    The trigger is activated by calling :meth:`set_triggered` from an
    EventBus subscriber or similar mechanism. Pattern matching uses
    fnmatch-style glob patterns (e.g. "ci.passed", "fs.change:*.pdf").
    """

    trigger_type: ClassVar[str] = "event"

    def __init__(
        self,
        event_pattern: str,
        *,
        next_due_at: float = 0.0,
    ) -> None:
        if not event_pattern:
            raise ValueError("event_pattern must not be empty")
        self._event_pattern = event_pattern
        self._next_due_at = next_due_at
        self._triggered = False
        self._last_event: str = ""

    # ------------------------------------------------------------------
    # Trigger Protocol
    # ------------------------------------------------------------------

    @property
    def trigger_type(self) -> str:  # type: ignore[override]
        return "event"

    def is_due(self, now: float) -> bool:
        """Return True if the trigger has been activated by a matching event."""
        return self._triggered

    def advance(self, now: float) -> None:
        """Reset the triggered flag after the task has been executed."""
        self._triggered = False
        self._next_due_at = 0.0  # No predictable next time for events

    @property
    def next_due_at(self) -> float:
        return self._next_due_at

    def serialize(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "trigger_type": "event",
            "event_pattern": self._event_pattern,
            "next_due_at": self._next_due_at,
        }

    # ------------------------------------------------------------------
    # Event interface
    # ------------------------------------------------------------------

    def set_triggered(self, event_name: str = "") -> None:
        """Mark this trigger as fired.

        Called by external EventBus subscribers when a matching event arrives.
        Optionally records the event name for diagnostics.

        Args:
            event_name: The specific event that triggered this (for logging).
        """
        self._triggered = True
        self._last_event = event_name
        self._next_due_at = time.time()

    def matches(self, event_name: str) -> bool:
        """Check if an event name matches this trigger's pattern.

        Uses fnmatch-style glob matching. The pattern may contain ':' as
        a namespace separator (e.g. "fs.change:*.pdf" matches
        "fs.change:report.pdf").

        Args:
            event_name: The event to check against the pattern.
        """
        return fnmatch.fnmatch(event_name, self._event_pattern)

    def notify(self, event_name: str) -> bool:
        """Convenience: check match and set_triggered if matched.

        Returns True if the event matched and the trigger was activated.
        """
        if self.matches(event_name):
            self.set_triggered(event_name)
            return True
        return False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def deserialize(cls, data: dict) -> "EventTrigger":
        """Reconstruct from serialized dict."""
        return cls(
            event_pattern=data["event_pattern"],
            next_due_at=data.get("next_due_at", 0.0),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def event_pattern(self) -> str:
        """The event pattern this trigger listens for."""
        return self._event_pattern

    @property
    def is_triggered(self) -> bool:
        """Whether the trigger is currently in fired state."""
        return self._triggered

    @property
    def last_event(self) -> str:
        """The last event name that activated this trigger."""
        return self._last_event

    def __repr__(self) -> str:
        return (
            f"EventTrigger(event_pattern={self._event_pattern!r}, "
            f"triggered={self._triggered})"
        )
