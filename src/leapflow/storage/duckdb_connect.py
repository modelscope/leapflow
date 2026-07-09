"""Centralized DuckDB connection factory with lock detection and repair.

Replaces bare ``duckdb.connect()`` calls throughout the codebase with a
single entry point that:

1. Detects lock conflicts (another process holds exclusive lock) and raises
   a clear, actionable ``DatabaseLockedError`` instead of a cryptic traceback.
2. Performs corruption detection via read-only probe before opening read-write.
3. Auto-repairs corrupt databases (backup + recreate).

All stores must use ``connect()`` from this module rather than calling
``duckdb.connect()`` directly.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import duckdb

logger = logging.getLogger(__name__)

LOCK_KEYWORDS = frozenset({"lock", "locked", "exclusive", "cannot set lock"})

_CONNECT_RETRIES = int(os.getenv("LEAPFLOW_DB_CONNECT_RETRIES", "3"))
_CONNECT_BACKOFF_S = float(os.getenv("LEAPFLOW_DB_CONNECT_BACKOFF_S", "0.5"))


class DatabaseLockedError(RuntimeError):
    """Raised when DuckDB cannot acquire exclusive lock on the database file.

    Provides a clear, actionable error message for the user.
    """

    def __init__(self, db_path: Path, original: Exception) -> None:
        self.db_path = db_path
        self.original = original
        msg = (
            f"Database is locked: {db_path.name}\n"
            f"Another LeapFlow instance holds the database.\n"
            f"Run 'leap daemon status' to check, or close the other instance."
        )
        super().__init__(msg)


def is_lock_error(exc: Exception) -> bool:
    """Heuristic check whether exception is a DuckDB lock conflict."""
    msg = str(exc).lower()
    return any(kw in msg for kw in LOCK_KEYWORDS)


def _probe_corruption(db_path: Path) -> bool:
    """Read-only probe to detect corruption. Returns True if healthy."""
    if not db_path.exists():
        return True
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        conn.execute("SELECT 1").fetchone()
        conn.close()
        return True
    except Exception as exc:
        if is_lock_error(exc):
            return True
        logger.warning("duckdb_connect: corruption detected in %s: %s", db_path.name, exc)
        return False


def _repair(db_path: Path) -> None:
    """Backup corrupt database and delete for recreation on next connect."""
    from leapflow.storage.db_repair import check_and_repair
    check_and_repair(db_path)


def connect(
    db_path: Path | str,
    *,
    read_only: bool = False,
    retries: Optional[int] = None,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with lock detection, retry, and repair.

    Parameters
    ----------
    db_path : Path or str
        Path to the ``.duckdb`` file.
    read_only : bool
        Open in read-only mode (no lock contention).
    retries : int, optional
        Maximum retry attempts.  Defaults to env
        ``LEAPFLOW_DB_CONNECT_RETRIES`` (3).

    Returns
    -------
    duckdb.DuckDBPyConnection

    Raises
    ------
    DatabaseLockedError
        If the database is locked by another process after all retries.
    RuntimeError
        If the database cannot be opened for reasons other than lock.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    max_attempts = retries if retries is not None else _CONNECT_RETRIES

    if not read_only and not _probe_corruption(db_path):
        _repair(db_path)

    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            conn = duckdb.connect(str(db_path), read_only=read_only)
            return conn
        except Exception as exc:
            last_exc = exc
            if is_lock_error(exc):
                if attempt < max_attempts - 1:
                    backoff = _CONNECT_BACKOFF_S * (attempt + 1)
                    logger.info(
                        "duckdb_connect: %s locked, retry %d/%d in %.1fs",
                        db_path.name, attempt + 1, max_attempts, backoff,
                    )
                    time.sleep(backoff)
                    continue
                raise DatabaseLockedError(db_path, exc) from exc
            raise

    assert last_exc is not None
    raise last_exc
