# Copyright (c) Alibaba, Inc. and its affiliates.
"""DuckDB-backed file checkpoint store.

Persists file snapshots per turn so ``/checkpoint rollback`` can restore files
to their pre-mutation state. The DB is profile-scoped and separate from
``leap.duckdb`` because checkpoint blobs grow on a different curve and are
pruned on their own TTL schedule.

Design:
    - One row per snapshotted file; grouped by ``turn_id``.
    - ``inline_content`` stores small file bytes as BLOB.
    - ``temp_ref`` stores the path to a temp copy for large files.
    - Schema is created idempotently (``CREATE TABLE IF NOT EXISTS``).
    - Cross-process lock strategy matches existing stores (ConnectionHolder).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional, Union

from leapflow.engine.file_checkpoint import (
    FileSnapshot,
    RollbackResult,
    TurnCheckpoint,
    restore_from_snapshot,
)
from leapflow.storage.connection import ConnectionHolder, LocalConnectionHolder
from leapflow.storage.write_buffer import execute_with_retry

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS file_checkpoints (
    turn_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL DEFAULT '',
    seq INTEGER NOT NULL,
    file_path VARCHAR NOT NULL,
    content_hash VARCHAR NOT NULL DEFAULT '',
    existed BOOLEAN NOT NULL DEFAULT TRUE,
    inline_content BLOB,
    temp_ref VARCHAR,
    size BIGINT NOT NULL DEFAULT 0,
    created_at DOUBLE NOT NULL,
    PRIMARY KEY (turn_id, seq)
)
"""

_CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_fchk_session ON file_checkpoints(session_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_fchk_turn ON file_checkpoints(turn_id)",
)


class DuckDBFileCheckpointStore:
    """Durable file checkpoint store backed by DuckDB.

    Implements the ``FileCheckpointStore`` protocol.
    """

    def __init__(self, source: Union[ConnectionHolder, Path, str]) -> None:
        self._owns_holder = isinstance(source, (str, Path))
        if self._owns_holder:
            source = LocalConnectionHolder(Path(source))
        self._holder: ConnectionHolder = source
        self._ensure_schema()

    @property
    def _conn(self) -> Any:
        """Resolve per call for thread safety (see LocalConnectionHolder docs)."""
        return self._holder.connection

    def _execute_write(self, sql: str, params: Any = None) -> None:
        execute_with_retry(self._conn, sql, params)

    def _ensure_schema(self) -> None:
        """Create tables and indexes idempotently."""
        conn = self._conn
        conn.execute(_CREATE_TABLE_SQL)
        for idx_sql in _CREATE_INDEX_SQL:
            conn.execute(idx_sql)

    def save_turn(self, checkpoint: TurnCheckpoint) -> None:
        """Persist all snapshots for a completed turn."""
        for seq, snap in enumerate(checkpoint.snapshots):
            self._execute_write(
                """
                INSERT INTO file_checkpoints
                    (turn_id, session_id, seq, file_path, content_hash,
                     existed, inline_content, temp_ref, size, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (turn_id, seq) DO UPDATE SET
                    file_path = EXCLUDED.file_path,
                    content_hash = EXCLUDED.content_hash,
                    existed = EXCLUDED.existed,
                    inline_content = EXCLUDED.inline_content,
                    temp_ref = EXCLUDED.temp_ref,
                    size = EXCLUDED.size
                """,
                [
                    checkpoint.turn_id,
                    checkpoint.session_id,
                    seq,
                    snap.path,
                    snap.content_hash,
                    snap.existed,
                    snap.inline_content,
                    snap.temp_ref,
                    snap.size,
                    checkpoint.created_at,
                ],
            )

    def get_turn(self, turn_id: str) -> Optional[TurnCheckpoint]:
        """Retrieve the checkpoint for a given turn."""
        rows = self._conn.execute(
            """
            SELECT turn_id, session_id, seq, file_path, content_hash,
                   existed, inline_content, temp_ref, size, created_at
            FROM file_checkpoints
            WHERE turn_id = ?
            ORDER BY seq ASC
            """,
            [turn_id],
        ).fetchall()

        if not rows:
            return None

        snapshots = []
        session_id = ""
        created_at = 0.0
        for row in rows:
            session_id = row[1] or ""
            created_at = float(row[9] or 0.0)
            inline_content = row[6]
            if isinstance(inline_content, memoryview):
                inline_content = bytes(inline_content)
            snapshots.append(FileSnapshot(
                path=str(row[3]),
                content_hash=str(row[4] or ""),
                existed=bool(row[5]),
                inline_content=inline_content,
                temp_ref=str(row[7]) if row[7] else None,
                size=int(row[8] or 0),
                timestamp=float(row[9] or 0.0),
            ))

        return TurnCheckpoint(
            turn_id=turn_id,
            session_id=session_id,
            snapshots=tuple(snapshots),
            created_at=created_at,
        )

    def list_turns(self, session_id: str, *, limit: int = 20) -> list[TurnCheckpoint]:
        """List recent checkpoints for a session, newest first."""
        rows = self._conn.execute(
            """
            SELECT DISTINCT turn_id, MIN(created_at) as min_created
            FROM file_checkpoints
            WHERE session_id = ?
            GROUP BY turn_id
            ORDER BY min_created DESC
            LIMIT ?
            """,
            [session_id, limit],
        ).fetchall()

        checkpoints = []
        for row in rows:
            turn_id = str(row[0])
            cp = self.get_turn(turn_id)
            if cp is not None:
                checkpoints.append(cp)
        return checkpoints

    def rollback_turn(self, turn_id: str) -> RollbackResult:
        """Restore files from a turn's snapshots."""
        checkpoint = self.get_turn(turn_id)
        if checkpoint is None:
            return RollbackResult(
                restored=(),
                skipped=(),
                failed=(("unknown", f"No checkpoint found for turn {turn_id}"),),
            )

        restored: list[str] = []
        skipped: list[str] = []
        failed: list[tuple[str, str]] = []

        for snap in checkpoint.snapshots:
            success, reason = restore_from_snapshot(snap)
            if not success:
                failed.append((snap.path, reason))
            elif "unchanged" in reason or "already absent" in reason:
                skipped.append(snap.path)
            else:
                restored.append(snap.path)

        return RollbackResult(
            restored=tuple(restored),
            skipped=tuple(skipped),
            failed=tuple(failed),
        )

    def cleanup(self, *, max_age_hours: float = 24.0) -> int:
        """Delete checkpoints older than the cutoff. Returns count deleted."""
        cutoff = time.time() - (max_age_hours * 3600)

        # Find temp_ref paths to clean up before deleting rows
        rows = self._conn.execute(
            "SELECT temp_ref FROM file_checkpoints WHERE created_at < ? AND temp_ref IS NOT NULL",
            [cutoff],
        ).fetchall()
        for row in rows:
            temp_ref = row[0]
            if temp_ref:
                try:
                    p = Path(temp_ref)
                    if p.exists():
                        p.unlink()
                except OSError as exc:
                    logger.debug("file_checkpoint cleanup: %s: %s", temp_ref, exc)

        count_row = self._conn.execute(
            "SELECT COUNT(*) FROM file_checkpoints WHERE created_at < ?",
            [cutoff],
        ).fetchone()
        count = int(count_row[0]) if count_row else 0

        if count > 0:
            self._execute_write(
                "DELETE FROM file_checkpoints WHERE created_at < ?",
                [cutoff],
            )
            logger.info("file_checkpoint: cleaned up %d snapshot rows", count)
        return count

    def close(self) -> None:
        """Close the owned connection (if any)."""
        if self._owns_holder:
            try:
                self._holder.close()
            except Exception:
                pass


__all__ = ["DuckDBFileCheckpointStore"]
