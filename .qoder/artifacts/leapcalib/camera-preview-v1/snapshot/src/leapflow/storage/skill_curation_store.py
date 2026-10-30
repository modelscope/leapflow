# Copyright (c) Alibaba, Inc. and its affiliates.
"""DuckDB-backed store for skill curation state.

Follows the project convention: stores receive a ConnectionHolder rather
than a raw path, and schema is registered via the central migration in
``storage/schema.py``.
"""

from __future__ import annotations

import logging
from typing import Optional

from leapflow.skills.curator import CurationState, SkillCurationEntry
from leapflow.storage.connection import ConnectionHolder

logger = logging.getLogger(__name__)

# Table DDL — also registered as a schema migration in schema.py.
_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS skill_curation (
    skill_name TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'active',
    pinned BOOLEAN NOT NULL DEFAULT FALSE,
    last_activity_at DOUBLE,
    created_at DOUBLE NOT NULL,
    archive_reason TEXT
)
"""


class DuckDBSkillCurationStore:
    """Persistent curation store backed by DuckDB.

    Implements the ``SkillCurationStore`` Protocol defined in
    ``leapflow.skills.curator``.
    """

    def __init__(self, holder: ConnectionHolder) -> None:
        self._holder = holder
        self._ensure_table()

    @property
    def _con(self):
        """Thread-safe connection access (never cache the result)."""
        return self._holder.connection

    def _ensure_table(self) -> None:
        """Idempotent table creation — safe to call on every instantiation."""
        try:
            self._con.execute(_TABLE_DDL)
        except Exception:
            logger.debug("skill_curation: table creation skipped", exc_info=True)

    # ── Protocol implementation ──

    def load_all(self) -> list[SkillCurationEntry]:
        rows = self._con.execute(
            "SELECT skill_name, state, pinned, last_activity_at, created_at, archive_reason "
            "FROM skill_curation ORDER BY skill_name"
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def load(self, skill_name: str) -> Optional[SkillCurationEntry]:
        rows = self._con.execute(
            "SELECT skill_name, state, pinned, last_activity_at, created_at, archive_reason "
            "FROM skill_curation WHERE skill_name = ?",
            [skill_name],
        ).fetchall()
        if not rows:
            return None
        return self._row_to_entry(rows[0])

    def save(self, entry: SkillCurationEntry) -> None:
        self._con.execute(
            """
            INSERT INTO skill_curation
                (skill_name, state, pinned, last_activity_at, created_at, archive_reason)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (skill_name) DO UPDATE SET
                state = EXCLUDED.state,
                pinned = EXCLUDED.pinned,
                last_activity_at = EXCLUDED.last_activity_at,
                archive_reason = EXCLUDED.archive_reason
            """,
            [
                entry.skill_name,
                entry.state.value,
                entry.pinned,
                entry.last_activity_at,
                entry.created_at,
                entry.archive_reason,
            ],
        )

    def delete(self, skill_name: str) -> bool:
        before = self._con.execute(
            "SELECT COUNT(*) FROM skill_curation WHERE skill_name = ?",
            [skill_name],
        ).fetchone()
        self._con.execute(
            "DELETE FROM skill_curation WHERE skill_name = ?",
            [skill_name],
        )
        return bool(before and before[0] > 0)

    # ── Helpers ──

    @staticmethod
    def _row_to_entry(row: tuple) -> SkillCurationEntry:
        skill_name, state_str, pinned, last_activity_at, created_at, archive_reason = row
        return SkillCurationEntry(
            skill_name=str(skill_name),
            state=CurationState(state_str),
            pinned=bool(pinned),
            last_activity_at=float(last_activity_at) if last_activity_at is not None else None,
            created_at=float(created_at),
            archive_reason=str(archive_reason) if archive_reason else None,
        )


__all__ = ["DuckDBSkillCurationStore"]
