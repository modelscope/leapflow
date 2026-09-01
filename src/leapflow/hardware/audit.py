"""Structured NDJSON audit log for hardware operations.

Every read, write, and emergency-stop that passes through ``HardwareTools`` is
recorded as one line of append-only NDJSON in the profile's audit directory.

The schema is deliberately flat:

    {ts, action, device, channel, value, outcome, identity}

``ts`` is wall-clock (``time.time()`` epoch seconds) — the only timebase that may
appear in audit and compliance artefacts (see ``Reading`` docstring).

The writer is synchronous, blocking, and contained: a failed append is logged and
dropped, never propagated into the tool path. Losing an audit line is bad; failing
a physical command because the audit disk filled up is worse.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuditEntry:
    """One hardware audit record."""

    ts: float
    action: str
    device: str
    channel: str
    value: Any
    outcome: str
    identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "action": self.action,
            "device": self.device,
            "channel": self.channel,
            "value": self.value,
            "outcome": self.outcome,
            "identity": self.identity,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> AuditEntry:
        return AuditEntry(
            ts=float(data.get("ts", 0.0)),
            action=str(data.get("action", "")),
            device=str(data.get("device", "")),
            channel=str(data.get("channel", "")),
            value=data.get("value"),
            outcome=str(data.get("outcome", "")),
            identity=str(data.get("identity", "")),
        )


class HardwareAuditLog:
    """Append-only NDJSON hardware audit writer.

    Constructed once per ``HardwareTools`` instance. The path is resolved from
    ``ProfileLayout`` at construction, so a missing profile gracefully degrades to
    a no-op (path is None).
    """

    def __init__(self, path: Path | None) -> None:
        self._path = path

    @property
    def path(self) -> Path | None:
        return self._path

    def record(
        self,
        *,
        action: str,
        device: str,
        channel: str = "",
        value: Any = None,
        outcome: str = "ok",
        identity: str = "",
    ) -> AuditEntry | None:
        """Append one audit line. Returns the entry on success, None on failure."""
        entry = AuditEntry(
            ts=time.time(),
            action=action,
            device=device,
            channel=channel,
            value=value,
            outcome=outcome,
            identity=identity,
        )
        if self._path is None:
            return entry

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.to_dict(), ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            logger.warning("Could not write hardware audit entry to %s: %s", self._path, exc)
            return None
        return entry

    def read_entries(self) -> list[AuditEntry]:
        """Read all entries from the log. For diagnostics and testing."""
        if self._path is None or not self._path.exists():
            return []
        entries: list[AuditEntry] = []
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(AuditEntry.from_dict(json.loads(line)))
                except (json.JSONDecodeError, ValueError):
                    continue
        except OSError:
            return []
        return entries
