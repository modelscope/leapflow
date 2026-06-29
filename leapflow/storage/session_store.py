"""Persistent store for learning session metadata.

Enables `leap learn --resume` by persisting LearningSession records across
process restarts. Uses the same DuckDB instance as TrajectoryStore.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb

logger = logging.getLogger(__name__)


class LearningSessionStore:
    """DuckDB-backed CRUD for learning session metadata."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self._path))
        self._init_schema()

    def close(self) -> None:
        self._con.close()

    def _init_schema(self) -> None:
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS leap_learning_session (
                session_id TEXT PRIMARY KEY,
                trajectory_id TEXT NOT NULL,
                goal TEXT NOT NULL DEFAULT '',
                start_time DOUBLE NOT NULL,
                end_time DOUBLE,
                status TEXT NOT NULL DEFAULT 'recording',
                annotations TEXT,
                metadata TEXT,
                created_at DOUBLE NOT NULL
            )
        """)
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_traj "
            "ON leap_learning_session(trajectory_id)"
        )
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_status "
            "ON leap_learning_session(status)"
        )

    def save(
        self,
        session_id: str,
        trajectory_id: str,
        goal: str = "",
        start_time: float = 0.0,
        *,
        end_time: Optional[float] = None,
        status: str = "recording",
        annotations: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist or update a learning session record."""
        now = time.time()
        self._con.execute(
            "INSERT OR REPLACE INTO leap_learning_session "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                session_id,
                trajectory_id,
                goal,
                start_time or now,
                end_time,
                status,
                json.dumps(annotations or [], ensure_ascii=False),
                json.dumps(metadata or {}, ensure_ascii=False),
                now,
            ],
        )

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a session by session_id."""
        rows = self._con.execute(
            "SELECT session_id, trajectory_id, goal, start_time, end_time, "
            "status, annotations, metadata "
            "FROM leap_learning_session WHERE session_id = ?",
            [session_id],
        ).fetchall()
        if not rows:
            return None
        return self._row_to_dict(rows[0])

    def find_by_trajectory(self, trajectory_id: str) -> Optional[Dict[str, Any]]:
        """Find a session by its associated trajectory_id."""
        rows = self._con.execute(
            "SELECT session_id, trajectory_id, goal, start_time, end_time, "
            "status, annotations, metadata "
            "FROM leap_learning_session WHERE trajectory_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            [trajectory_id],
        ).fetchall()
        if not rows:
            return None
        return self._row_to_dict(rows[0])

    def mark_completed(self, session_id: str, end_time: Optional[float] = None) -> None:
        """Mark a session as completed (learning triggered)."""
        self._con.execute(
            "UPDATE leap_learning_session SET status = 'completed', end_time = ? "
            "WHERE session_id = ?",
            [end_time or time.time(), session_id],
        )

    def mark_abandoned(self, session_id: str) -> None:
        """Mark a session as abandoned (quit without learning)."""
        self._con.execute(
            "UPDATE leap_learning_session SET status = 'abandoned', end_time = ? "
            "WHERE session_id = ?",
            [time.time(), session_id],
        )

    def list_resumable(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List sessions that can be resumed (status='recording', no end_time)."""
        rows = self._con.execute(
            "SELECT session_id, trajectory_id, goal, start_time, end_time, "
            "status, annotations, metadata "
            "FROM leap_learning_session "
            "WHERE status = 'recording' "
            "ORDER BY start_time DESC LIMIT ?",
            [limit],
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List all recent sessions regardless of status."""
        rows = self._con.execute(
            "SELECT session_id, trajectory_id, goal, start_time, end_time, "
            "status, annotations, metadata "
            "FROM leap_learning_session "
            "ORDER BY start_time DESC LIMIT ?",
            [limit],
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: tuple) -> Dict[str, Any]:
        annotations_raw = row[6]
        metadata_raw = row[7]
        return {
            "session_id": row[0],
            "trajectory_id": row[1],
            "goal": row[2],
            "start_time": row[3],
            "end_time": row[4],
            "status": row[5],
            "annotations": json.loads(annotations_raw) if annotations_raw else [],
            "metadata": json.loads(metadata_raw) if metadata_raw else {},
        }
