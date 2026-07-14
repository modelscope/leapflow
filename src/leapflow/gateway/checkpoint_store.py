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
