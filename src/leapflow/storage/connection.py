"""ConnectionHolder protocol and implementation for shared DuckDB access.

All stores receive a ``ConnectionHolder`` instead of a raw ``db_path``.
This enables:

- **P0a-P0b**: Centralized lock-aware connection with retry
- **P1**: Single ``leap.duckdb`` shared by all stores (6→1 consolidation)
- **P4**: leapd daemon owns the connection; stores are thin wrappers

The holder creates the connection lazily and shares it. Stores MUST NOT
call ``duckdb.connect()`` themselves.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import duckdb

from leapflow.storage.duckdb_connect import connect as _lock_aware_connect

logger = logging.getLogger(__name__)


@runtime_checkable
class ConnectionHolder(Protocol):
    """Protocol for obtaining a shared DuckDB connection."""

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Return the managed DuckDB connection."""
        ...

    @property
    def db_path(self) -> Path:
        """Path to the DuckDB file."""
        ...

    def close(self) -> None:
        """Close the managed connection."""
        ...


class LocalConnectionHolder:
    """In-process holder that lazily opens a single DuckDB connection.

    Thread-safety: DuckDB's embedded connection is single-writer.
    Within one process, all stores share this holder and access is
    serialized by DuckDB's internal lock. For multi-process, use the
    leapd daemon (P4).
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = _lock_aware_connect(self._db_path)
            logger.info("duckdb: opened %s", self._db_path.name)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
                logger.info("duckdb: closed %s", self._db_path.name)
            except Exception:
                pass
            self._conn = None
