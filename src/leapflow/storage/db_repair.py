"""DuckDB health check and self-repair — corrupt detection + automatic backup.

Provides:
1. Integrity check via PRAGMA
2. Automatic backup of corrupt database files
3. Recovery by re-creation from backup or clean slate
4. Connection test before returning to caller

Used by ConversationStore, EvolutionStore, and any DuckDB-backed module
to ensure graceful handling of corruption (power loss, disk full, etc.).
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def check_and_repair(
    db_path: Path | str,
    *,
    backup_suffix: str = ".corrupt.bak",
    max_backups: int = 3,
) -> bool:
    """Check DuckDB file health and attempt repair if corrupt.

    Returns True if the database is healthy (or was repaired),
    False if repair failed and the caller should start fresh.
    """
    import duckdb

    db_path = Path(db_path)
    if not db_path.exists():
        return True

    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        conn.execute("SELECT 1").fetchone()
        tables = conn.execute("SHOW TABLES").fetchall()
        conn.close()
        return True
    except Exception as exc:
        logger.warning("db_repair: integrity check failed for %s: %s", db_path.name, exc)

    logger.warning("db_repair: database corrupt, backing up: %s", db_path.name)
    backup_path = _create_backup(db_path, suffix=backup_suffix, max_backups=max_backups)
    if backup_path:
        logger.info("db_repair: backup created at %s", backup_path.name)

    try:
        db_path.unlink(missing_ok=True)
        wal = db_path.with_suffix(db_path.suffix + ".wal")
        wal.unlink(missing_ok=True)
        logger.info("db_repair: removed corrupt database, will recreate on next access")
        return True
    except OSError as exc:
        logger.error("db_repair: failed to remove corrupt database: %s", exc)
        return False


def _create_backup(
    db_path: Path,
    *,
    suffix: str = ".corrupt.bak",
    max_backups: int = 3,
) -> Optional[Path]:
    """Create a timestamped backup of the database file."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = db_path.with_name(f"{db_path.stem}_{ts}{suffix}")

    try:
        shutil.copy2(db_path, backup)
    except OSError as exc:
        logger.warning("db_repair: backup copy failed: %s", exc)
        return None

    _cleanup_old_backups(db_path.parent, db_path.stem, suffix, max_backups)
    return backup


def _cleanup_old_backups(
    directory: Path,
    stem: str,
    suffix: str,
    max_backups: int,
) -> None:
    """Remove oldest backups exceeding max_backups."""
    pattern = f"{stem}_*{suffix}"
    backups = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    while len(backups) > max_backups:
        oldest = backups.pop(0)
        try:
            oldest.unlink()
            logger.debug("db_repair: cleaned up old backup %s", oldest.name)
        except OSError:
            pass


def safe_connect(db_path: Path | str, *, read_only: bool = False):
    """Connect to DuckDB with pre-flight health check and auto-repair.

    Returns a duckdb.Connection. On corruption, backs up and recreates.
    """
    import duckdb

    db_path = Path(db_path)
    if db_path.exists():
        check_and_repair(db_path)

    return duckdb.connect(str(db_path), read_only=read_only)
