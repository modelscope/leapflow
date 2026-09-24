# Copyright (c) Alibaba, Inc. and its affiliates.
"""Evidence store: persists operation evidence bundles for audit and learning.

Evidence bundles capture the post-operation state of a device -- what was
commanded, what the transport reported, and what the sensors observed after
the physical system settled.  This store persists that evidence so it can
be queried for audit trails, fed into the learning pipeline, and used to
diagnose failures long after the operation occurred.

Storage layout:
- DuckDB table ``hardware_evidence`` for structured metadata (operation_id,
  device_id, channel_id, timestamp, verdict, deviation, etc.)
- File system directory for binary artifacts (camera frames) referenced
  by operation_id.
- Retention policy: configurable days, default 30, with periodic cleanup.

The store is append-only by design: evidence is immutable once recorded.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from leapflow.storage.connection import ConnectionHolder, LocalConnectionHolder

logger = logging.getLogger(__name__)

EVIDENCE_CATEGORY = "hardware_evidence"
"""Cache category for the evidence tier of ``instrument.duckdb``."""

EVIDENCE_SCHEMA_VERSION = 1
"""Row format version.  Recorded on every row and read as an exact-match
filter, so an incompatible format bump excludes rows written under any other
layout -- both older and newer -- rather than silently mixing them."""

DEFAULT_RETENTION_DAYS = 30.0
"""How long evidence rows and associated frames survive before cleanup."""

DEFAULT_MAX_FRAMES_MB = 500.0
"""Maximum total size of stored frame artifacts on disk."""

_PRUNE_INTERVAL_S = 3600.0
"""Floor on how often retention runs, matching reading_store."""


class EvidenceStore:
    """Persists EvidenceBundle and OperationVerdict for audit and learning.

    Thread-safe: all DuckDB writes go through a dedicated writer thread
    via ``asyncio.to_thread``, matching the pattern in ``reading_store.py``.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        frames_dir: str | Path | None = None,
        retention_days: float = DEFAULT_RETENTION_DAYS,
        max_frames_mb: float = DEFAULT_MAX_FRAMES_MB,
        connection_holder: ConnectionHolder | None = None,
    ) -> None:
        """Initialize the evidence store.

        Args:
            db_path: Path to the DuckDB database file.
            frames_dir: Directory for frame artifact storage.
                If None, frames are stored alongside the database.
            retention_days: How long to keep evidence (default 30 days).
            max_frames_mb: Maximum total size of stored frames.
            connection_holder: Shared DuckDB connection holder.  When
                provided the store uses it; otherwise it creates and owns
                a ``LocalConnectionHolder`` from *db_path*.
        """
        self._db_path = Path(db_path)
        self._frames_dir = Path(frames_dir) if frames_dir is not None else self._db_path.parent / "evidence_frames"
        self._retention_days = max(0.0, float(retention_days))
        self._max_frames_mb = max(0.0, float(max_frames_mb))

        # Connection management -- same pattern as CalibrationStore.
        self._holder: ConnectionHolder | None = connection_holder
        self._owns_holder = False
        if self._holder is None:
            self._holder = LocalConnectionHolder(self._db_path)
            self._owns_holder = True

        self._db_ready = False
        self._records_written = 0
        self._write_failures = 0
        self._last_prune_at = 0.0

    # ── Schema ──

    def _ensure_schema(self) -> None:
        """Create the evidence table if it doesn't exist.

        The index comes after the table, matching the calibration/reading
        store ordering: it is a maintenance aid, not a correctness
        requirement.
        """
        if self._holder is None:
            return
        connection = self._holder.connection
        connection.execute(_SCHEMA)
        try:
            connection.execute(_INDEX_DEVICE_TIME)
        except Exception:  # noqa: BLE001 - an unindexed table still answers
            logger.debug("hardware_evidence index unavailable", exc_info=True)
        try:
            connection.execute(_INDEX_VERDICT)
        except Exception:  # noqa: BLE001
            logger.debug("hardware_evidence verdict index unavailable", exc_info=True)

    def _ensure_ready(self) -> Any:
        """Ensure schema exists and return the connection.

        Returns the DuckDB connection, or raises if the holder is absent.
        """
        if self._holder is None:
            raise RuntimeError("EvidenceStore has no connection holder")
        connection = self._holder.connection
        if not self._db_ready:
            self._ensure_schema()
            self._db_ready = True
        return connection

    # ── Record ──

    async def record(
        self,
        bundle: Any,  # EvidenceBundle from verification.py
        verdict: Any | None = None,  # OperationVerdict from verification.py
    ) -> str:
        """Persist an evidence bundle and optional verdict.

        Returns the operation_id.

        1. Serialize bundle metadata to DuckDB row.
        2. If bundle has frames, write them to ``frames_dir/{operation_id}/``.
        3. If verdict is provided, include the verdict columns.
        """
        operation_id = getattr(bundle, "operation_id", "") or ""
        if not operation_id:
            raise ValueError("EvidenceBundle must have a non-empty operation_id")

        # Prepare the row values from the bundle.
        outcome = getattr(bundle, "actual_outcome", None)
        outcome_ok: bool | None = None
        outcome_side_effect: str | None = None
        if outcome is not None:
            outcome_ok = getattr(outcome, "ok", None)
            outcome_side_effect = getattr(outcome, "side_effect", None)

        # Verdict columns -- filled when provided.
        verdict_status: str | None = None
        verdict_confidence: float | None = None
        verdict_deviation: float | None = None
        verdict_detail: str | None = None
        verifier_id: str | None = None
        if verdict is not None:
            verdict_status = getattr(verdict, "status", None)
            verdict_confidence = getattr(verdict, "confidence", None)
            verdict_deviation = getattr(verdict, "deviation", None)
            verdict_detail = getattr(verdict, "detail", None) or None
            verifier_id = getattr(verdict, "verifier_id", None)

        # Serialize readings and metadata.
        readings_json = _serialize_readings(getattr(bundle, "post_settle_readings", ()))
        metadata_json = _serialize_value(dict(getattr(bundle, "metadata", {})))

        # Write frames to the file system before the DB row so the row
        # never references frames that do not exist yet.
        frame_paths = await self._write_frames(bundle)
        frame_paths_json = json.dumps(frame_paths, ensure_ascii=False) if frame_paths else None

        row = (
            operation_id,
            getattr(bundle, "device_id", ""),
            getattr(bundle, "channel_id", ""),
            float(getattr(bundle, "timestamp", 0.0)),
            _serialize_value(getattr(bundle, "intended_value", None)),
            outcome_ok,
            outcome_side_effect,
            float(getattr(bundle, "settle_delay_s", 0.0)),
            readings_json,
            verdict_status,
            verdict_confidence,
            verdict_deviation,
            verdict_detail,
            verifier_id,
            frame_paths_json,
            metadata_json,
            EVIDENCE_SCHEMA_VERSION,
        )

        await asyncio.to_thread(self._write_row, row)
        return operation_id

    def _write_row(self, row: tuple[Any, ...]) -> None:
        """Blocking DuckDB insert, safe to call from a worker thread."""
        try:
            connection = self._ensure_ready()
            connection.execute(_INSERT, row)
            self._records_written += 1
        except Exception as exc:  # noqa: BLE001 - storage must not fail the operation
            self._write_failures += 1
            logger.warning("Could not persist evidence record: %s", exc)

    # ── Verdict update ──

    async def record_verdict(
        self,
        operation_id: str,
        verdict: Any,  # OperationVerdict from verification.py
    ) -> None:
        """Update verdict for an existing evidence record.

        Used when verification runs asynchronously after the initial record.
        """
        verdict_status = getattr(verdict, "status", None)
        verdict_confidence = getattr(verdict, "confidence", None)
        verdict_deviation = getattr(verdict, "deviation", None)
        verdict_detail = getattr(verdict, "detail", None) or None
        verifier_id = getattr(verdict, "verifier_id", None)

        params = (
            verdict_status,
            verdict_confidence,
            verdict_deviation,
            verdict_detail,
            verifier_id,
            operation_id,
        )
        await asyncio.to_thread(self._update_verdict_row, params)

    def _update_verdict_row(self, params: tuple[Any, ...]) -> None:
        """Blocking DuckDB update for verdict columns."""
        try:
            connection = self._ensure_ready()
            connection.execute(_UPDATE_VERDICT, params)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not update evidence verdict: %s", exc)

    # ── Frame persistence ──

    async def _write_frames(self, bundle: Any) -> list[str]:
        """Write frame artifacts to the file system.

        Returns a list of relative paths (relative to ``frames_dir``) for
        storage in the DuckDB row.  Frame files are named
        ``{operation_id}/{sequence}.{ext}``.
        """
        frames = getattr(bundle, "frames", ())
        if not frames:
            return []
        operation_id = getattr(bundle, "operation_id", "unknown")
        op_dir = self._frames_dir / operation_id
        return await asyncio.to_thread(self._write_frames_sync, frames, op_dir, operation_id)

    @staticmethod
    def _write_frames_sync(
        frames: tuple[Any, ...],
        op_dir: Path,
        operation_id: str,
    ) -> list[str]:
        """Blocking frame write, safe for a worker thread."""
        paths: list[str] = []
        try:
            op_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create frame directory %s: %s", op_dir, exc)
            return paths

        for seq, frame in enumerate(frames):
            # FrameReading carries ``data`` (bytes) and ``format`` (str).
            data = getattr(frame, "data", None)
            fmt = getattr(frame, "format", "bin") or "bin"
            if data is None:
                continue
            filename = f"{seq:04d}.{fmt}"
            file_path = op_dir / filename
            try:
                file_path.write_bytes(data if isinstance(data, bytes) else bytes(data))
                paths.append(f"{operation_id}/{filename}")
            except OSError as exc:
                logger.warning("Could not write frame %s: %s", file_path, exc)
        return paths

    # ── Query ──

    async def query(
        self,
        *,
        device_id: str | None = None,
        channel_id: str | None = None,
        verdict_status: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query evidence records with optional filters.

        Returns dicts with all columns, suitable for the learning pipeline
        and audit display.
        """
        return await asyncio.to_thread(
            self._query_sync, device_id, channel_id, verdict_status, since, limit
        )

    def _query_sync(
        self,
        device_id: str | None,
        channel_id: str | None,
        verdict_status: str | None,
        since: float | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Blocking query implementation."""
        if self._holder is None:
            return []
        if self._db_path is not None and not self._db_path.exists():
            return []
        try:
            connection = self._ensure_ready()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not open evidence store for query: %s", exc)
            return []

        # Build dynamic WHERE clause.
        conditions: list[str] = [f"schema_version = {EVIDENCE_SCHEMA_VERSION}"]
        params: list[Any] = []
        if device_id is not None:
            conditions.append("device_id = ?")
            params.append(device_id)
        if channel_id is not None:
            conditions.append("channel_id = ?")
            params.append(channel_id)
        if verdict_status is not None:
            conditions.append("verdict_status = ?")
            params.append(verdict_status)
        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(float(since))

        where = " AND ".join(conditions)
        sql = f"SELECT {', '.join(_COLUMNS)} FROM hardware_evidence WHERE {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(int(limit))

        try:
            rows = connection.execute(sql, params).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Evidence query failed: %s", exc)
            return []
        return [dict(zip(_COLUMNS, row)) for row in rows]

    async def get_frame_paths(self, operation_id: str) -> list[Path]:
        """Return file paths for frames associated with an operation."""
        return await asyncio.to_thread(self._get_frame_paths_sync, operation_id)

    def _get_frame_paths_sync(self, operation_id: str) -> list[Path]:
        """Blocking frame path lookup."""
        if self._holder is None:
            return []
        try:
            connection = self._ensure_ready()
            rows = connection.execute(
                "SELECT frame_paths FROM hardware_evidence WHERE operation_id = ?",
                (operation_id,),
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not query frame paths: %s", exc)
            return []
        if not rows or rows[0][0] is None:
            return []
        try:
            relative_paths = json.loads(rows[0][0])
        except (TypeError, ValueError):
            return []
        return [self._frames_dir / rp for rp in relative_paths]

    # ── Analytics ──

    async def success_rate(
        self,
        device_id: str,
        *,
        channel_id: str | None = None,
        window_days: float = 7.0,
    ) -> dict[str, Any]:
        """Compute success/failure/inconclusive rate for a device or channel.

        Returns a dict with total, success, failure, inconclusive counts
        and a success_rate in [0.0, 1.0].
        """
        return await asyncio.to_thread(
            self._success_rate_sync, device_id, channel_id, window_days
        )

    def _success_rate_sync(
        self,
        device_id: str,
        channel_id: str | None,
        window_days: float,
    ) -> dict[str, Any]:
        """Blocking success rate computation."""
        result: dict[str, Any] = {
            "total": 0,
            "success": 0,
            "failure": 0,
            "inconclusive": 0,
            "success_rate": 0.0,
            "window_days": window_days,
        }
        if self._holder is None:
            return result
        if self._db_path is not None and not self._db_path.exists():
            return result
        try:
            connection = self._ensure_ready()
        except Exception:  # noqa: BLE001
            return result

        cutoff = time.time() - window_days * 86400.0
        conditions = ["device_id = ?", "timestamp >= ?", f"schema_version = {EVIDENCE_SCHEMA_VERSION}"]
        params: list[Any] = [device_id, cutoff]
        if channel_id is not None:
            conditions.append("channel_id = ?")
            params.append(channel_id)
        where = " AND ".join(conditions)
        sql = (
            f"SELECT verdict_status, COUNT(*) FROM hardware_evidence "
            f"WHERE {where} GROUP BY verdict_status"
        )
        try:
            rows = connection.execute(sql, params).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Evidence success_rate query failed: %s", exc)
            return result

        total = 0
        for status_val, count in rows:
            total += count
            if status_val == "success":
                result["success"] = count
            elif status_val == "failure":
                result["failure"] = count
            else:
                result["inconclusive"] += count
        result["total"] = total
        if total > 0:
            result["success_rate"] = result["success"] / total
        return result

    # ── Cleanup ──

    async def cleanup(self) -> dict[str, int]:
        """Remove evidence older than ``retention_days``.

        Returns ``{"rows_deleted": N, "frames_deleted": M, "bytes_freed": B}``.
        """
        return await asyncio.to_thread(self._cleanup_sync)

    def _cleanup_sync(self) -> dict[str, int]:
        """Blocking cleanup of expired evidence rows and frame artifacts."""
        stats: dict[str, int] = {"rows_deleted": 0, "frames_deleted": 0, "bytes_freed": 0}
        if self._holder is None or self._retention_days <= 0:
            return stats

        cutoff = time.time() - self._retention_days * 86400.0

        try:
            connection = self._ensure_ready()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not open evidence store for cleanup: %s", exc)
            return stats

        # First retrieve frame_paths for rows that will be deleted so we
        # can clean up the file system artifacts.
        try:
            expired_rows = connection.execute(
                "SELECT operation_id, frame_paths FROM hardware_evidence WHERE timestamp < ?",
                (cutoff,),
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not query expired evidence: %s", exc)
            return stats

        # Delete the DuckDB rows.
        try:
            deleted = connection.execute(
                "DELETE FROM hardware_evidence WHERE timestamp < ? RETURNING 1",
                (cutoff,),
            ).fetchall()
            stats["rows_deleted"] = len(deleted)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not prune expired evidence rows: %s", exc)
            return stats

        # Clean up frame directories.
        for operation_id, frame_paths_json in expired_rows:
            op_dir = self._frames_dir / str(operation_id)
            if not op_dir.exists():
                continue
            try:
                dir_size = sum(f.stat().st_size for f in op_dir.rglob("*") if f.is_file())
                file_count = sum(1 for f in op_dir.rglob("*") if f.is_file())
                shutil.rmtree(op_dir)
                stats["frames_deleted"] += file_count
                stats["bytes_freed"] += dir_size
            except OSError as exc:
                logger.warning("Could not remove frame directory %s: %s", op_dir, exc)

        self._last_prune_at = time.time()
        if stats["rows_deleted"]:
            logger.info(
                "hardware evidence: pruned %d row(s), %d frame(s), %d bytes freed",
                stats["rows_deleted"],
                stats["frames_deleted"],
                stats["bytes_freed"],
            )
        return stats

    # ── Introspection ──

    @property
    def records_written(self) -> int:
        """Total evidence records successfully persisted."""
        return self._records_written

    @property
    def write_failures(self) -> int:
        """Evidence records that could not be persisted."""
        return self._write_failures

    def close(self) -> None:
        """Close the database connection.

        Only closes the holder it created itself: when the registry injected
        the shared holder, closing it here would pull the connection out from
        under sibling stores.
        """
        if self._owns_holder and self._holder is not None:
            try:
                self._holder.close()
            except Exception:  # noqa: BLE001 - teardown must not propagate
                logger.debug("evidence holder close failed", exc_info=True)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _serialize_value(value: Any) -> str:
    """JSON-serialize a value for DuckDB storage.

    Falls back to ``str(value)`` wrapped in quotes when json.dumps cannot
    handle the type, so a row is never lost to a serialisation failure.
    """
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value))


def _serialize_readings(readings: tuple[Any, ...]) -> str:
    """Serialize a tuple of Reading objects to JSON.

    Each reading is converted via its ``to_dict()`` method when available,
    falling back to ``str()`` for opaque objects.
    """
    items: list[Any] = []
    for r in readings:
        to_dict = getattr(r, "to_dict", None)
        if callable(to_dict):
            items.append(to_dict())
        else:
            items.append(str(r))
    return json.dumps(items, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

_COLUMNS = (
    "operation_id",
    "device_id",
    "channel_id",
    "timestamp",
    "intended_value",
    "outcome_ok",
    "outcome_side_effect",
    "settle_delay_s",
    "readings_json",
    "verdict_status",
    "verdict_confidence",
    "verdict_deviation",
    "verdict_detail",
    "verifier_id",
    "frame_paths",
    "metadata",
    "created_at",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hardware_evidence (
    operation_id       VARCHAR PRIMARY KEY,
    device_id          VARCHAR NOT NULL,
    channel_id         VARCHAR NOT NULL,
    timestamp          DOUBLE NOT NULL,
    intended_value     VARCHAR,
    outcome_ok         BOOLEAN,
    outcome_side_effect VARCHAR,
    settle_delay_s     DOUBLE,
    readings_json      VARCHAR,
    verdict_status     VARCHAR,
    verdict_confidence DOUBLE,
    verdict_deviation  DOUBLE,
    verdict_detail     VARCHAR,
    verifier_id        VARCHAR,
    frame_paths        VARCHAR,
    metadata           VARCHAR,
    schema_version     INTEGER DEFAULT 1,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_INSERT = """
INSERT OR REPLACE INTO hardware_evidence (
    operation_id, device_id, channel_id, timestamp,
    intended_value, outcome_ok, outcome_side_effect, settle_delay_s,
    readings_json,
    verdict_status, verdict_confidence, verdict_deviation, verdict_detail,
    verifier_id, frame_paths, metadata, schema_version
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_UPDATE_VERDICT = """
UPDATE hardware_evidence
SET verdict_status = ?,
    verdict_confidence = ?,
    verdict_deviation = ?,
    verdict_detail = ?,
    verifier_id = ?
WHERE operation_id = ?
"""

_INDEX_DEVICE_TIME = """
CREATE INDEX IF NOT EXISTS idx_evidence_device_time
ON hardware_evidence (device_id, channel_id, timestamp)
"""
"""Serves per-device/channel queries ordered by recency."""

_INDEX_VERDICT = """
CREATE INDEX IF NOT EXISTS idx_evidence_verdict
ON hardware_evidence (verdict_status, timestamp)
"""
"""Serves verdict-filtered analytics queries."""


__all__ = [
    "DEFAULT_MAX_FRAMES_MB",
    "DEFAULT_RETENTION_DAYS",
    "EVIDENCE_CATEGORY",
    "EVIDENCE_SCHEMA_VERSION",
    "EvidenceStore",
]
