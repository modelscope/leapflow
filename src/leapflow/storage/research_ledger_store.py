"""DuckDB-backed persistence for the per-session research ledger (S1).

Durable Orient: the structured long-task state (findings, open questions,
decisions, next step) survives process restarts and is reloaded when a session
resumes, so multi-turn deep work is not lost. Keyed by ``session_id`` and stored
in the shared ``leap.duckdb`` via a ``ConnectionHolder`` (daemon-owned), matching
the conversation/finding store conventions.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from leapflow.storage.connection import ConnectionHolder, LocalConnectionHolder

logger = logging.getLogger(__name__)

_TABLE = "research_ledger"
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    session_id TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT current_timestamp
)
"""


class ResearchLedgerStore:
    """Persist / reload the research ledger snapshot per session."""

    def __init__(self, source: Union[ConnectionHolder, Path, str]) -> None:
        if isinstance(source, (str, Path)):
            source = LocalConnectionHolder(Path(source))
        self._holder = source
        self._ensured = False

    def _ensure_schema(self) -> None:
        if not self._ensured:
            self._holder.connection.execute(_SCHEMA)
            self._ensured = True

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the persisted ledger state for a session, or None."""
        if not session_id:
            return None
        try:
            self._ensure_schema()
            row = self._holder.connection.execute(
                f"SELECT state_json FROM {_TABLE} WHERE session_id = ?",
                [session_id],
            ).fetchone()
            if not row or not row[0]:
                return None
            return json.loads(row[0])
        except Exception as exc:  # graceful: persistence must never break a turn
            logger.warning("ResearchLedgerStore.load failed: %s", exc)
            return None

    def save(self, session_id: str, state: Dict[str, Any]) -> None:
        """Upsert the ledger state for a session."""
        if not session_id:
            return
        try:
            self._ensure_schema()
            payload = json.dumps(state, ensure_ascii=False)
            self._holder.connection.execute(
                f"INSERT OR REPLACE INTO {_TABLE} (session_id, state_json, updated_at) "
                f"VALUES (?, ?, current_timestamp)",
                [session_id, payload],
            )
        except Exception as exc:  # graceful: persistence must never break a turn
            logger.warning("ResearchLedgerStore.save failed: %s", exc)

    def clear(self, session_id: str) -> None:
        """Remove a session's persisted ledger (e.g. explicit reset)."""
        if not session_id:
            return
        try:
            self._ensure_schema()
            self._holder.connection.execute(
                f"DELETE FROM {_TABLE} WHERE session_id = ?",
                [session_id],
            )
        except Exception as exc:
            logger.warning("ResearchLedgerStore.clear failed: %s", exc)

    def close(self) -> None:
        """Close the underlying connection (owned holders only)."""
        try:
            self._holder.close()
        except Exception:
            pass
