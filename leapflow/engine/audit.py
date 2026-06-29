"""Structured JSONL audit logger for mode transitions, skill executions, and learning events."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, List

logger = logging.getLogger(__name__)


class AuditLogger:
    """Append-only JSONL audit trail.

    Each record: ``{"ts": float, "event": str, "session": str, ...data}``

    Thread/async-safe for single-writer usage (one ``AuditLogger`` per process).
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", encoding="utf-8")

    def log(self, event: str, *, session_id: str = "", **data: Any) -> None:
        record = {"ts": time.time(), "event": event, "session": session_id}
        record.update(data)
        try:
            self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._file.flush()
        except Exception as e:
            logger.debug("audit.write_failed event=%s error=%s", event, e)

    def query(self, *, event_prefix: str = "", limit: int = 50) -> List[dict]:
        """Read recent audit records (for debugging / inspection)."""
        records: List[dict] = []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if event_prefix and not rec.get("event", "").startswith(event_prefix):
                            continue
                        records.append(rec)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        return records[-limit:]

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


class NullAuditLogger:
    """No-op audit logger for testing and when auditing is disabled."""

    def log(self, event: str, *, session_id: str = "", **data: Any) -> None:
        pass

    def query(self, *, event_prefix: str = "", limit: int = 50) -> List[dict]:
        return []

    def close(self) -> None:
        pass
