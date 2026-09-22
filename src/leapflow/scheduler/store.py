# Copyright (c) Alibaba, Inc. and its affiliates.
"""DuckDB-backed persistence for armed tasks.

Provides atomic CRUD and query operations for the scheduler.
At-most-once guarantee: advance_next_due() atomically updates
before execution to prevent double-fire on crash recovery.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Union

from leapflow.scheduler.types import ArmedTask
from leapflow.storage.connection import ConnectionHolder, LocalConnectionHolder
from leapflow.storage.write_buffer import execute_with_retry

logger = logging.getLogger(__name__)


class TaskStore:
    """DuckDB-backed persistence for armed tasks.

    Provides atomic CRUD and query operations.
    At-most-once guarantee: advance_next_due() atomically updates
    before execution to prevent double-fire on crash recovery.

    Accepts ``ConnectionHolder`` (shared) or legacy ``Path``.
    """

    def __init__(self, source: Union[ConnectionHolder, Path, str]) -> None:
        """Initialize store, creating table if not exists."""
        self._owns_holder = isinstance(source, (str, Path))
        if self._owns_holder:
            source = LocalConnectionHolder(Path(source))
        self._holder = source
        self._ensure_table()

    @property
    def _con(self) -> Any:
        """Resolve per call, so each thread gets its own cursor.

        See ``LocalConnectionHolder.connection``: caching this puts two threads on
        one connection. ``load_all`` is reached from the ``daemon.status`` RPC on the
        event loop while deferred init reads other tables in a worker, so a shared
        connection here stalls the whole control plane.
        """
        return self._holder.connection

    def close(self) -> None:
        """Close the DuckDB connection if owned by this store."""
        if self._owns_holder:
            self._holder.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        """CREATE TABLE IF NOT EXISTS armed_tasks (...)"""
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS armed_tasks (
                task_id TEXT PRIMARY KEY,
                skill_name TEXT NOT NULL,
                trigger_type TEXT NOT NULL,
                trigger_config TEXT NOT NULL,
                state TEXT DEFAULT 'armed',
                execution_tier TEXT DEFAULT 'auto',
                context_snapshot TEXT DEFAULT '{}',
                confidence DOUBLE DEFAULT 0.0,
                created_at DOUBLE NOT NULL,
                next_due_at DOUBLE DEFAULT 0.0,
                last_run_at DOUBLE DEFAULT 0.0,
                run_count INTEGER DEFAULT 0,
                max_runs INTEGER DEFAULT -1,
                grace_seconds DOUBLE DEFAULT 120.0,
                parameters TEXT DEFAULT '{}',
                cloud_worker_id TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}',
                max_retries INTEGER DEFAULT 0,
                retry_count INTEGER DEFAULT 0,
                retry_backoff_s DOUBLE DEFAULT 60.0
            )
        """)
        self._migrate_retry_columns()

    def _migrate_retry_columns(self) -> None:
        """Idempotent migration: add retry columns to pre-existing tables."""
        for col, dtype, default in (
            ("max_retries", "INTEGER", "0"),
            ("retry_count", "INTEGER", "0"),
            ("retry_backoff_s", "DOUBLE", "60.0"),
        ):
            try:
                self._con.execute(
                    f"ALTER TABLE armed_tasks ADD COLUMN {col} {dtype} DEFAULT {default}"
                )
            except Exception:  # noqa: BLE001 — column already exists
                pass

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, task: ArmedTask) -> None:
        """Insert or update a task."""
        execute_with_retry(
            self._con,
            """
            INSERT OR REPLACE INTO armed_tasks (
                task_id, skill_name, trigger_type, trigger_config,
                state, execution_tier, context_snapshot, confidence,
                created_at, next_due_at, last_run_at, run_count,
                max_runs, grace_seconds, parameters, cloud_worker_id, metadata,
                max_retries, retry_count, retry_backoff_s
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                task.task_id,
                task.skill_name,
                task.trigger_type,
                json.dumps(task.trigger_config) if isinstance(task.trigger_config, dict) else task.trigger_config,
                task.state,
                task.execution_tier,
                json.dumps(task.context_snapshot) if isinstance(task.context_snapshot, dict) else task.context_snapshot,
                task.confidence,
                task.created_at,
                task.next_due_at,
                task.last_run_at,
                task.run_count,
                task.max_runs,
                task.grace_seconds,
                json.dumps(task.parameters) if isinstance(task.parameters, dict) else task.parameters,
                task.cloud_worker_id,
                json.dumps(task.metadata) if isinstance(task.metadata, dict) else task.metadata,
                task.max_retries,
                task.retry_count,
                task.retry_backoff_s,
            ],
        )

    def load(self, task_id: str) -> Optional[ArmedTask]:
        """Load a single task by ID."""
        result = self._con.execute(
            "SELECT * FROM armed_tasks WHERE task_id = ?", [task_id]
        ).fetchone()
        if result is None:
            return None
        return self._row_to_task(result)

    def load_all(self) -> List[ArmedTask]:
        """Load all tasks."""
        rows = self._con.execute("SELECT * FROM armed_tasks").fetchall()
        return [self._row_to_task(row) for row in rows]

    def delete(self, task_id: str) -> None:
        """Remove a task."""
        execute_with_retry(
            self._con,
            "DELETE FROM armed_tasks WHERE task_id = ?", [task_id],
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_due_tasks(self, now: float) -> List[ArmedTask]:
        """Get tasks where next_due_at <= now AND state = 'armed'."""
        rows = self._con.execute(
            """
            SELECT * FROM armed_tasks
            WHERE next_due_at <= ? AND next_due_at > 0 AND state = 'armed'
            ORDER BY next_due_at ASC
            """,
            [now],
        ).fetchall()
        return [self._row_to_task(row) for row in rows]

    # ------------------------------------------------------------------
    # Atomic updates
    # ------------------------------------------------------------------

    def advance_next_due(self, task_id: str, new_due_at: float) -> None:
        """Atomically advance next_due_at (at-most-once guarantee)."""
        execute_with_retry(
            self._con,
            "UPDATE armed_tasks SET next_due_at = ? WHERE task_id = ?",
            [new_due_at, task_id],
        )

    def update_state(self, task_id: str, new_state: str) -> None:
        """Update task state."""
        execute_with_retry(
            self._con,
            "UPDATE armed_tasks SET state = ? WHERE task_id = ?",
            [new_state, task_id],
        )

    def increment_run_count(self, task_id: str) -> None:
        """Increment run_count and update last_run_at."""
        now = time.time()
        execute_with_retry(
            self._con,
            """
            UPDATE armed_tasks
            SET run_count = run_count + 1, last_run_at = ?
            WHERE task_id = ?
            """,
            [now, task_id],
        )

    # ------------------------------------------------------------------
    # CRUD extensions (Phase 1B)
    # ------------------------------------------------------------------

    _MUTABLE_COLUMNS = frozenset({
        "trigger_type", "trigger_config", "state", "next_due_at",
        "parameters", "max_runs", "grace_seconds", "metadata",
        "max_retries", "retry_count", "retry_backoff_s",
    })

    def update_task(self, task_id: str, **fields: Any) -> bool:
        """Update mutable fields on an armed task.

        Returns True if the task existed and was updated, False otherwise.
        Raises ValueError for unknown field names.
        """
        unknown = set(fields) - self._MUTABLE_COLUMNS
        if unknown:
            raise ValueError(f"Cannot update field(s): {', '.join(sorted(unknown))}")
        if not fields:
            return False
        set_clauses = []
        values: list[Any] = []
        for col, val in fields.items():
            set_clauses.append(f"{col} = ?")
            if col in ("trigger_config", "parameters", "metadata") and isinstance(val, dict):
                values.append(json.dumps(val))
            else:
                values.append(val)
        values.append(task_id)
        execute_with_retry(
            self._con,
            f"UPDATE armed_tasks SET {', '.join(set_clauses)} WHERE task_id = ?",
            values,
        )
        return self.load(task_id) is not None

    def set_state(self, task_id: str, state: str) -> bool:
        """Convenience wrapper: update only the state column.

        Returns True if the task existed.
        """
        return self.update_task(task_id, state=state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_task(self, row: tuple) -> ArmedTask:
        """Convert a DuckDB row tuple to an ArmedTask dataclass."""
        return ArmedTask(
            task_id=row[0],
            skill_name=row[1],
            trigger_type=row[2],
            trigger_config=self._safe_json_loads(row[3]),
            state=row[4],
            execution_tier=row[5],
            context_snapshot=self._safe_json_loads(row[6]),
            confidence=row[7],
            created_at=row[8],
            next_due_at=row[9],
            last_run_at=row[10],
            run_count=row[11],
            max_runs=row[12],
            grace_seconds=row[13],
            parameters=self._safe_json_loads(row[14]),
            cloud_worker_id=row[15],
            metadata=self._safe_json_loads(row[16]),
            max_retries=row[17] if len(row) > 17 else 0,
            retry_count=row[18] if len(row) > 18 else 0,
            retry_backoff_s=row[19] if len(row) > 19 else 60.0,
        )

    @staticmethod
    def _safe_json_loads(value: str | dict | None) -> dict:
        """Safely parse a JSON string, returning empty dict on failure."""
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
