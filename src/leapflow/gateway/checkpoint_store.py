"""Gateway event checkpoint persistence.

Stores the last-consumed event_id per platform so event sources can
resume from where they left off after a restart.  Uses the shared
DuckDB connection via ``ConnectionHolder``.
"""
from __future__ import annotations

import logging
import time
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_TABLE = "gateway_checkpoints"
_DDL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    platform_id VARCHAR PRIMARY KEY,
    checkpoint  VARCHAR NOT NULL DEFAULT '',
    updated_at  DOUBLE  NOT NULL DEFAULT 0
)
"""


@runtime_checkable
class CheckpointStore(Protocol):
    """Read/write last-consumed checkpoint per platform."""

    def save(self, platform_id: str, checkpoint: str) -> None: ...
    def load(self, platform_id: str) -> str: ...


@runtime_checkable
class DeduplicationStore(Protocol):
    """Persist seen event_ids for cross-restart dedup."""

    def save_batch(self, platform_id: str, event_ids: list[str]) -> None: ...
    def load_recent(self, platform_id: str, limit: int = 10000) -> list[str]: ...


_DEDUP_TABLE = "gateway_seen_events"
_DEDUP_DDL = f"""
CREATE TABLE IF NOT EXISTS {_DEDUP_TABLE} (
    platform_id VARCHAR NOT NULL,
    event_id    VARCHAR NOT NULL,
    seen_at     DOUBLE  NOT NULL DEFAULT 0,
    PRIMARY KEY (platform_id, event_id)
)
"""


class DuckDBDeduplicationStore:
    """DuckDB-backed dedup store for cross-restart event deduplication."""

    def __init__(self, connection_holder: object) -> None:
        self._holder = connection_holder
        self._ensure_table()

    def _ensure_table(self) -> None:
        try:
            conn = self._holder.connection  # type: ignore[union-attr]
            conn.execute(_DEDUP_DDL)
        except Exception:
            logger.warning("Failed to create dedup table", exc_info=True)

    def save_batch(self, platform_id: str, event_ids: list[str]) -> None:
        if not platform_id or not event_ids:
            return
        try:
            from leapflow.storage.write_buffer import execute_with_retry

            conn = self._holder.connection  # type: ignore[union-attr]
            now = time.time()
            for eid in event_ids[-1000:]:
                execute_with_retry(
                    conn,
                    f"""
                    INSERT OR IGNORE INTO {_DEDUP_TABLE}
                    (platform_id, event_id, seen_at) VALUES (?, ?, ?)
                    """,
                    [platform_id, eid, now],
                )
            cutoff = now - 86400 * 7
            execute_with_retry(
                conn,
                f"DELETE FROM {_DEDUP_TABLE} WHERE platform_id = ? AND seen_at < ?",
                [platform_id, cutoff],
            )
        except Exception:
            logger.debug("Failed to save dedup batch for %s", platform_id, exc_info=True)

    def load_recent(self, platform_id: str, limit: int = 10000) -> list[str]:
        if not platform_id:
            return []
        try:
            conn = self._holder.connection  # type: ignore[union-attr]
            rows = conn.execute(
                f"SELECT event_id FROM {_DEDUP_TABLE} WHERE platform_id = ? ORDER BY seen_at DESC LIMIT ?",
                [platform_id, limit],
            ).fetchall()
            return [str(r[0]) for r in rows]
        except Exception:
            logger.debug("Failed to load dedup state for %s", platform_id, exc_info=True)
            return []


class DuckDBCheckpointStore:
    """DuckDB-backed checkpoint store for gateway event sources."""

    def __init__(self, connection_holder: object) -> None:
        self._holder = connection_holder
        self._ensure_table()

    def _ensure_table(self) -> None:
        try:
            conn = self._holder.connection  # type: ignore[union-attr]
            conn.execute(_DDL)
        except Exception:
            logger.warning("Failed to create checkpoint table", exc_info=True)

    def save(self, platform_id: str, checkpoint: str) -> None:
        """Upsert the checkpoint for a platform."""
        if not platform_id or not checkpoint:
            return
        try:
            from leapflow.storage.write_buffer import execute_with_retry

            conn = self._holder.connection  # type: ignore[union-attr]
            execute_with_retry(
                conn,
                f"""
                INSERT INTO {_TABLE} (platform_id, checkpoint, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT (platform_id) DO UPDATE
                SET checkpoint = excluded.checkpoint,
                    updated_at = excluded.updated_at
                """,
                [platform_id, checkpoint, time.time()],
            )
        except Exception:
            logger.debug("Failed to save checkpoint for %s", platform_id, exc_info=True)

    def load(self, platform_id: str) -> str:
        """Load the last checkpoint for a platform, or empty string."""
        if not platform_id:
            return ""
        try:
            conn = self._holder.connection  # type: ignore[union-attr]
            result = conn.execute(
                f"SELECT checkpoint FROM {_TABLE} WHERE platform_id = ?",
                [platform_id],
            ).fetchone()
            return str(result[0]) if result else ""
        except Exception:
            logger.debug("Failed to load checkpoint for %s", platform_id, exc_info=True)
            return ""
