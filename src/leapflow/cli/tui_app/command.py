"""Command lifecycle primitives for the interactive TUI.

The module is intentionally UI-framework agnostic: it models submitted user
commands and their lifecycle so the prompt_toolkit application, Rich console,
and future schedulers can share the same contract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from enum import Enum

_DEFAULT_SUMMARY_LIMIT = 96
_MAX_ERROR_LENGTH = 240


def _single_line(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, limit: int) -> str:
    if limit <= 1:
        return "…"
    return text if len(text) <= limit else text[: limit - 1] + "…"


class TuiCommandStatus(str, Enum):
    """Lifecycle states for a TUI-submitted command."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class TuiCommand:
    """Immutable command request tracked by the TUI scheduler."""

    id: int
    text: str
    status: TuiCommandStatus
    created_at: float
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""

    @classmethod
    def create(cls, *, command_id: int, text: str) -> "TuiCommand":
        """Create a queued command from user input."""
        return cls(
            id=command_id,
            text=text,
            status=TuiCommandStatus.QUEUED,
            created_at=time.monotonic(),
        )

    @property
    def label(self) -> str:
        """Human-readable command label."""
        return f"#{self.id}"

    @property
    def elapsed_s(self) -> float:
        """Return command runtime when available, otherwise current age."""
        if self.started_at <= 0:
            return 0.0
        end = self.finished_at if self.finished_at > 0 else time.monotonic()
        return max(0.0, end - self.started_at)

    def summary(self, *, limit: int = _DEFAULT_SUMMARY_LIMIT) -> str:
        """Return a single-line summary suitable for compact task cards."""
        return _truncate(_single_line(self.text), limit)

    def mark_running(self) -> "TuiCommand":
        """Return a copy marked as running."""
        return replace(
            self,
            status=TuiCommandStatus.RUNNING,
            started_at=time.monotonic(),
            error="",
        )

    def mark_done(self) -> "TuiCommand":
        """Return a copy marked as successfully completed."""
        return replace(
            self,
            status=TuiCommandStatus.DONE,
            finished_at=time.monotonic(),
            error="",
        )

    def mark_failed(self, error: str) -> "TuiCommand":
        """Return a copy marked as failed with a concise error message."""
        return replace(
            self,
            status=TuiCommandStatus.FAILED,
            finished_at=time.monotonic(),
            error=_truncate(_single_line(error), _MAX_ERROR_LENGTH),
        )
