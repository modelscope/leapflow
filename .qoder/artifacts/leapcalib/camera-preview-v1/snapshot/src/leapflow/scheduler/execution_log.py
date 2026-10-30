# Copyright (c) Alibaba, Inc. and its affiliates.
"""DuckDB-backed execution log for scheduled tasks.

Provides durable, queryable history of every scheduler execution —
start, finish, status, result summary, and error. Governance/logging
is cold-path only: it piggybacks on the scheduler tick and adds no
measurable per-turn cost to the hot path.

DB lives in the same DuckDB file as the TaskStore (``leap.duckdb``),
as a new ``scheduler_execution_log`` table.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Protocol, Union, runtime_checkable

from leapflow.storage.connection import ConnectionHolder, LocalConnectionHolder
from leapflow.storage.write_buffer import execute_with_retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionLogRecord:
    """Immutable record of a single scheduler execution."""

    task_id: str
    execution_id: str
    trigger_type: str
    started_at: float
    finished_at: Optional[float]
    status: str  # running | success | failed | skipped
    result_summary: str
    error: str


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ExecutionLogStore(Protocol):
    """Durable store for scheduler execution history.

    All methods are synchronous, matching :class:`TaskStore`.
    """

    def record_start(
        self,
        task_id: str,
        trigger_type: str,
    ) -> str:
        """Record that an execution has started.

        Returns:
            A unique ``execution_id`` for pairing with :meth:`record_finish`.
        """
        ...

    def record_finish(
        self,
        execution_id: str,
        status: str,
        result_summary: str = "",
        error: str = "",
    ) -> None:
        """Record the outcome of a previously started execution."""
        ...

    def get_history(
        self,
        task_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[ExecutionLogRecord]:
        """Return execution records, newest first.

        If *task_id* is ``None``, return records for all tasks.
        """
        ...

    def cleanup(self, max_age_hours: float = 168.0) -> int:
        """Delete records older than *max_age_hours*.

        Returns:
            Number of records deleted.
        """
        ...


# ---------------------------------------------------------------------------
# DuckDB implementation
# ---------------------------------------------------------------------------


class DuckDBExecutionLogStore:
    """DuckDB-backed :class:`ExecutionLogStore`.

    Shares the same DuckDB file (via ``ConnectionHolder``) as the
    ``TaskStore`` — no extra file.  Schema is created idempotently.

    Accepts ``ConnectionHolder`` (shared) or a legacy ``Path``/``str``
    for standalone usage or testing.
    """

    def __init__(self, source: Union[ConnectionHolder, Path, str]) -> None:
        self._owns_holder = isinstance(source, (str, Path))
        if self._owns_holder:
            source = LocalConnectionHolder(Path(source))
        self._holder: ConnectionHolder = source
        self._ensure_table()

    @property
    def _con(self) -> Any:
        """Resolve per call — thread-safety follows TaskStore's pattern."""
        return self._holder.connection

    def close(self) -> None:
        """Close the DuckDB connection if owned by this store."""
        if self._owns_holder:
            self._holder.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        """Idempotent table creation.

        Uses ``CREATE TABLE IF NOT EXISTS`` to avoid migration pitfalls.
        """
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS scheduler_execution_log (
                execution_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                trigger_type TEXT NOT NULL,
                started_at DOUBLE NOT NULL,
                finished_at DOUBLE,
                status TEXT NOT NULL DEFAULT 'running',
                result_summary TEXT DEFAULT '',
                error TEXT DEFAULT ''
            )
        """)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_start(
        self,
        task_id: str,
        trigger_type: str,
    ) -> str:
        execution_id = uuid.uuid4().hex
        now = time.time()
        execute_with_retry(
            self._con,
            """
            INSERT INTO scheduler_execution_log
                (execution_id, task_id, trigger_type, started_at, status)
            VALUES (?, ?, ?, ?, 'running')
            """,
            [execution_id, task_id, trigger_type, now],
        )
        return execution_id

    def record_finish(
        self,
        execution_id: str,
        status: str,
        result_summary: str = "",
        error: str = "",
    ) -> None:
        now = time.time()
        execute_with_retry(
            self._con,
            """
            UPDATE scheduler_execution_log
            SET finished_at = ?, status = ?, result_summary = ?, error = ?
            WHERE execution_id = ?
            """,
            [now, status, result_summary, error, execution_id],
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_history(
        self,
        task_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[ExecutionLogRecord]:
        if task_id is not None:
            rows = self._con.execute(
                """
                SELECT task_id, execution_id, trigger_type, started_at,
                       finished_at, status, result_summary, error
                FROM scheduler_execution_log
                WHERE task_id = ?
                ORDER BY started_at DESC
                LIMIT ?
                """,
                [task_id, limit],
            ).fetchall()
        else:
            rows = self._con.execute(
                """
                SELECT task_id, execution_id, trigger_type, started_at,
                       finished_at, status, result_summary, error
                FROM scheduler_execution_log
                ORDER BY started_at DESC
                LIMIT ?
                """,
                [limit],
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup(self, max_age_hours: float = 168.0) -> int:
        """Delete records older than *max_age_hours* (default 7 days)."""
        cutoff = time.time() - max_age_hours * 3600.0
        before = self._con.execute(
            "SELECT COUNT(*) FROM scheduler_execution_log WHERE started_at < ?",
            [cutoff],
        ).fetchone()[0]
        if before > 0:
            execute_with_retry(
                self._con,
                "DELETE FROM scheduler_execution_log WHERE started_at < ?",
                [cutoff],
            )
        return before

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: tuple) -> ExecutionLogRecord:
        return ExecutionLogRecord(
            task_id=row[0],
            execution_id=row[1],
            trigger_type=row[2],
            started_at=row[3],
            finished_at=row[4],
            status=row[5],
            result_summary=row[6] or "",
            error=row[7] or "",
        )
