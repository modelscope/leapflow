"""Batched write buffer for DuckDB stores.

High-frequency signal writes to DuckDB benefit from batching:
- Reduces WAL pressure and MVCC overhead
- Provides resilience against transient lock errors
- Flushes on count threshold OR timer, whichever fires first

Each store that does frequent inserts should use ``WriteBuffer`` to
accumulate operations and flush them as a batch.

Additionally provides ``execute_with_retry`` for stores that need
jitter-based retry on individual write operations.
"""
from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, List, Tuple

import duckdb

from leapflow.storage.duckdb_connect import is_lock_error

logger = logging.getLogger(__name__)

_WRITE_RETRIES = int(os.getenv("LEAPFLOW_DB_WRITE_RETRIES", "10"))
_JITTER_MIN_MS = float(os.getenv("LEAPFLOW_DB_JITTER_MIN_MS", "15"))
_JITTER_MAX_MS = float(os.getenv("LEAPFLOW_DB_JITTER_MAX_MS", "120"))


def execute_with_retry(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    params: Any = None,
    *,
    retries: int = _WRITE_RETRIES,
    jitter_ms: Tuple[float, float] = (_JITTER_MIN_MS, _JITTER_MAX_MS),
) -> None:
    """Execute a write SQL with jitter retry on lock errors.

    Raises the original exception if all retries are exhausted or if the
    error is not a lock conflict.
    """
    for attempt in range(retries):
        try:
            if params:
                conn.execute(sql, params)
            else:
                conn.execute(sql)
            return
        except Exception as exc:
            if is_lock_error(exc) and attempt < retries - 1:
                jitter = random.uniform(*jitter_ms) / 1000
                time.sleep(jitter)
                continue
            raise


_Op = Tuple[str, str, Any]  # (op_tag, sql, params)


class WriteBuffer:
    """Time-and-count gated write buffer.

    Accumulates ``(tag, sql, params)`` tuples and flushes them as a
    batch when either the count threshold or the timer fires.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        The database connection to flush to.
    max_count : int
        Flush when the buffer reaches this many operations.
    max_interval_s : float
        Flush when this many seconds have elapsed since the last flush.
    max_capacity : int
        Hard cap on buffer size; oldest entries are dropped when exceeded.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        max_count: int = 100,
        max_interval_s: float = 0.5,
        max_capacity: int = 2000,
    ) -> None:
        self._conn = conn
        self._max_count = max_count
        self._max_interval_s = max_interval_s
        self._max_capacity = max_capacity
        self._buffer: List[_Op] = []
        self._last_flush: float = time.monotonic()

    @property
    def pending(self) -> int:
        return len(self._buffer)

    def append(self, tag: str, sql: str, params: Any = None) -> None:
        """Add an operation to the buffer, flushing if thresholds are met."""
        self._buffer.append((tag, sql, params))
        if len(self._buffer) > self._max_capacity:
            self._buffer.pop(0)
        if self._should_flush():
            self.flush()

    def _should_flush(self) -> bool:
        if len(self._buffer) >= self._max_count:
            return True
        if time.monotonic() - self._last_flush >= self._max_interval_s:
            return True
        return False

    def flush(self) -> int:
        """Execute all buffered operations. Returns count flushed."""
        if not self._buffer:
            return 0

        flushed = 0
        remaining: List[_Op] = []
        for tag, sql, params in self._buffer:
            try:
                execute_with_retry(self._conn, sql, params)
                flushed += 1
            except Exception as exc:
                if is_lock_error(exc):
                    logger.warning("write_buffer: transient flush failed for %s: %s", tag, exc)
                    remaining.append((tag, sql, params))
                    continue
                logger.error(
                    "write_buffer: dropping permanent failed op %s: %s",
                    tag,
                    exc,
                )

        self._buffer = remaining
        self._last_flush = time.monotonic()
        if flushed:
            logger.debug("write_buffer: flushed %d ops, %d remaining", flushed, len(remaining))
        return flushed

    def update_connection(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Update the underlying connection (used during connection sharing)."""
        self._conn = conn
