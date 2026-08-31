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
import tempfile
import threading
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import duckdb

from leapflow.storage.duckdb_connect import (
    DatabaseLockedError,
    connect as _lock_aware_connect,
)

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

    Thread-safety: ``DuckDBPyConnection`` is NOT thread-safe. The root
    connection is owned by the thread that first opens it (typically the
    event loop thread). Any other thread that asks for ``connection`` gets
    a thread-local ``cursor()`` — a full duplicate connection sharing the
    same database, which is DuckDB's documented multi-threading pattern.
    Writes across root connection and cursors are safe (DuckDB applies
    optimistic concurrency control internally).
    """

    def __init__(self, db_path: Path, *, volatile_on_lock: bool = False) -> None:
        self._db_path = db_path
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._volatile_on_lock = volatile_on_lock
        self._volatile_dir: tempfile.TemporaryDirectory[str] | None = None
        self._locked_error: DatabaseLockedError | None = None
        self._owner_thread_id: Optional[int] = None
        self._thread_local = threading.local()
        self._open_lock = threading.Lock()

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def is_volatile(self) -> bool:
        return self._volatile_dir is not None

    @property
    def locked_error(self) -> DatabaseLockedError | None:
        return self._locked_error

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Return the connection *for the calling thread*.

        **Never store the result.** The thread-affinity is the whole point: the
        value returned depends on who is asking, so a reference captured in one
        thread and used from another defeats it entirely. Resolve it per call --
        ``self._holder.connection.execute(...)``, or a ``_con`` property that
        forwards here.

        Caching it silently freezes the process. Two threads then execute on one
        ``DuckDBPyConnection``, which is not thread-safe, so the second blocks
        inside DuckDB until the first query finishes. When one of those threads is
        the event loop, every RPC stops being answered for the duration -- a
        deferred skill-library scan against a status call was enough to hang the
        daemon on roughly a third of starts, with nothing in the log after the last
        successful init line.
        """
        # Root connection is created and owned by the first thread that
        # opens it (normally the event loop thread). Other threads receive
        # a thread-local cursor, DuckDB's documented multi-threaded pattern.
        if self._conn is None:
            with self._open_lock:
                if self._conn is None:
                    self._conn = self._connect()
                    self._owner_thread_id = threading.get_ident()
                    logger.info("duckdb: opened %s", self._db_path.name)
                    return self._conn
        if threading.get_ident() == self._owner_thread_id:
            return self._conn
        cursor = getattr(self._thread_local, "cursor", None)
        if cursor is None:
            cursor = self._conn.cursor()
            self._thread_local.cursor = cursor
            logger.debug(
                "duckdb: created thread-local cursor for %s (thread=%d)",
                self._db_path.name, threading.get_ident(),
            )
        return cursor

    def _connect(self) -> duckdb.DuckDBPyConnection:
        try:
            return _lock_aware_connect(self._db_path)
        except DatabaseLockedError as exc:
            if not self._volatile_on_lock:
                raise
            self._locked_error = exc
            self._volatile_dir = tempfile.TemporaryDirectory(prefix="leapflow-volatile-")
            self._db_path = Path(self._volatile_dir.name) / "leap.duckdb"
            logger.warning(
                "duckdb: primary database locked; using volatile session database at %s",
                self._db_path,
            )
            return _lock_aware_connect(self._db_path)

    def close(self) -> None:
        # Only the root connection is closed here: thread-local cursors
        # become invalid together with it. Callers must shut down worker
        # threads (e.g. the deferred-DB executor) before invoking close().
        if self._conn is not None:
            try:
                self._conn.close()
                logger.info("duckdb: closed %s", self._db_path.name)
            except Exception:
                pass
            self._conn = None
            self._owner_thread_id = None
            self._thread_local = threading.local()
        if self._volatile_dir is not None:
            self._volatile_dir.cleanup()
            self._volatile_dir = None
