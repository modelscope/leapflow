# Copyright (c) Alibaba, Inc. and its affiliates.
"""Cache-index bookkeeping for the raw reading tier's NDJSON segments.

Split from ``ReadingStore`` because deciding *when* a segment's index entry is
refreshed is its own concern, with its own cost model: every write to the cache index
is a DuckDB open, update and checkpointing close on a file shared with other LeapFlow
processes. Re-registering every segment on every append -- one write per channel per
drain, over one a second on a multi-joint arm -- kept several cores busy, for a recorded
size and TTL that are only ever read at minute-to-day granularity.

It also owns the other end of that TTL: nothing else in the daemon expires raw segments,
and before the sweep here they accumulated for months past their seven days.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RAW_INDEX_REFRESH_S = 60.0
"""Minimum spacing between index refreshes of a segment that is still being written.

New segments, rolled segments and forced flushes are indexed immediately regardless, so
the interval only bounds how stale the size and TTL anchor of an open segment can be.
"""

DEFAULT_RAW_SWEEP_INTERVAL_S = 3600.0
"""Minimum spacing between retention sweeps, matching the history tier's prune floor.

The TTL is measured in days, so hourly precision costs nothing, and one sweep is one more
connection to the shared index.
"""


class RawSegmentIndex:
    """Track appended segments and index the due ones in a single batch.

    Keyed by path, so re-indexing refreshes the recorded size and restarts the TTL from
    the most recent sample -- which is what "keep for seven days" has to mean for a file
    that is still being written to. A segment is due when it is new, has just rolled,
    was last indexed ``refresh_s`` ago, or the flush is forced; the TTL anchor therefore
    trails the last sample by at most ``refresh_s``.

    Thread-safe: every stream source writes from its own worker thread into one store.
    The lock is held across the cache write, so two writers never register the same
    segment twice; the cache manager serializes its own writes anyway.
    """

    def __init__(
        self,
        cache_manager: Any,
        *,
        category: str,
        workspace_id: str = "",
        session_id: str = "",
        ttl_s: float,
        refresh_s: float = DEFAULT_RAW_INDEX_REFRESH_S,
        sweep_interval_s: float = DEFAULT_RAW_SWEEP_INTERVAL_S,
    ) -> None:
        self._cache = cache_manager
        self._category = category
        self._workspace_id = workspace_id
        self._session_id = session_id
        self._ttl_s = ttl_s
        self._refresh_s = max(0.0, float(refresh_s))
        self._sweep_interval_s = max(0.0, float(sweep_interval_s))
        self._last_sweep_at: float | None = None
        self._lock = threading.Lock()
        # When each open segment was last indexed (monotonic clock).
        self._indexed_at: dict[Path, float] = {}
        # Segments appended to since they were last indexed.
        self._unindexed: set[Path] = set()
        # Segments closed at their cap that still need their final size recorded.
        self._rolled: set[Path] = set()

    def mark_appended(self, path: Path) -> None:
        if self._cache is None:
            return
        with self._lock:
            self._unindexed.add(path)

    def mark_rolled(self, path: Path) -> None:
        if self._cache is None:
            return
        with self._lock:
            self._rolled.add(path)

    def index_due(self, *, force: bool = False) -> int:
        """Register every due segment through one ``register_many``; return how many.

        One call per write, so a drain covering every channel costs a single index
        connection rather than one per channel. Contained: a failure is logged and the
        segments stay marked, so the next write retries rather than losing the entry.
        """
        if self._cache is None:
            return 0
        with self._lock:
            moment = time.monotonic()
            due = [path for path in sorted(self._unindexed) if force or self._is_due(path, moment)]
            if not due:
                return 0
            # A non-positive TTL means "keep", recorded as no expiry: an expiry of now
            # would hand the segment to any other sweep of the shared index.
            expires_at = time.time() + self._ttl_s if self._ttl_s > 0 else None
            try:
                self._cache.register_many(
                    self._cache.build_entry(
                        path=path,
                        scope="session",
                        category=self._category,
                        source=str(path.name),
                        workspace_id=self._workspace_id,
                        session_id=self._session_id,
                        expires_at=expires_at,
                        sensitive=True,
                        syncable=False,
                        owner_component="hardware",
                    )
                    for path in due
                )
            except Exception as exc:  # noqa: BLE001 - indexing must not break sampling
                logger.warning("Could not index %d hardware reading file(s): %s", len(due), exc)
                return 0
            for path in due:
                self._unindexed.discard(path)
                if path in self._rolled:
                    # Finished: nothing appends to it again, so stop tracking it.
                    self._rolled.discard(path)
                    self._indexed_at.pop(path, None)
                else:
                    self._indexed_at[path] = moment
            return len(due)

    def _is_due(self, path: Path, moment: float) -> bool:
        if path in self._rolled or path not in self._indexed_at:
            return True
        return moment - self._indexed_at[path] >= self._refresh_s

    def sweep_expired(self) -> int:
        """Delete raw segments past their TTL, at most once per sweep interval.

        Runs on the write path rather than a timer, like the history prune: the process
        that grows the tier is the one that trims it, so retention cannot silently stop.
        The first write after start sweeps immediately, which is what clears a backlog
        left by an earlier process. Segments this store is still writing are spared --
        their recorded expiry can trail the file by up to the refresh interval -- and so
        is any segment *another* store sampled within the TTL, which only its file can
        tell: the index row of a segment another process is appending to may be stale.
        A non-positive TTL disables the sweep instead of expiring everything at once.
        """
        if self._cache is None or self._ttl_s <= 0:
            return 0
        with self._lock:
            moment = time.monotonic()
            if self._last_sweep_at is not None and moment - self._last_sweep_at < self._sweep_interval_s:
                return 0
            self._last_sweep_at = moment
            keep = set(self._indexed_at) | self._unindexed
        try:
            removed = int(
                self._cache.cleanup_expired(category=self._category, keep=keep, min_idle_s=self._ttl_s)
            )
        except Exception as exc:  # noqa: BLE001 - retention is maintenance, not a duty
            logger.warning("Could not expire hardware reading files: %s", exc)
            return 0
        if removed:
            logger.info("hardware readings: removed %d raw segment(s) past retention", removed)
        return removed
