"""Durable storage for sampled hardware readings.

Two tiers, because raw samples and long-term history have different lifetimes and
different sensitivity.

Raw samples land in a session-scoped cache directory as newline-delimited JSON,
registered with ``CacheManager`` as **sensitive and non-syncable**. A qPCR curve can
carry patient sample information and a production temperature trace can be a trade
secret, so raw physical data is treated like the session's visual and VLM artifacts:
TTL-bounded, quota-managed, and never synced anywhere.

Downsampled history lands in the profile's ``instrument.duckdb``. That is the tier a
later analysis reads, and it is the reason this module exists at all: before it, samples
lived only in an in-memory ring and vanished with the process, which made every form of
learning from physical experience impossible. Parameter reuse -- discovering that a
viscous sample wants 10 uL/s and *remembering* it -- cannot exist without a durable
series to derive it from.

Nothing here decides *what* is interesting; that stays with the envelope-derived event
detector. This module only persists what was observed.

Every persisted instant is ``Reading.observed_at`` -- wall-clock. Window *boundaries*
are monotonic, because deciding "has a minute elapsed" is an interval question. Mixing
the two is what made this tier unusable before: a monotonic origin resets on reboot, so
``ORDER BY ended_at DESC`` returned the oldest rows as the newest, silently.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from leapflow.hardware.context import as_numeric
from leapflow.hardware.transport import Reading

logger = logging.getLogger(__name__)

READINGS_CATEGORY = "hardware_readings"
"""Cache category for raw sample files, mirroring the visual/video artifact categories."""

DEFAULT_FLUSH_INTERVAL_S = 5.0
DEFAULT_DOWNSAMPLE_INTERVAL_S = 60.0
DEFAULT_RAW_TTL_S = 7 * 24 * 3600.0
DEFAULT_HISTORY_TTL_S = 90 * 24 * 3600.0
"""How long downsampled windows survive. Longer than the raw tier, still bounded.

The table grows at a fixed rate per streaming channel -- eight channels on the default
sixty-second interval is about 11,500 rows a day -- and before this nothing deleted
from it at all. An unbounded table is also an unbounded privacy exposure: whatever
exists is what a profile backup carries away.
"""

DEFAULT_RAW_SEGMENT_BYTES = 32 * 1024 * 1024
"""Size at which a raw file is closed and a new segment begins.

Segments exist for two reasons that a single append-only file cannot serve. A finished
segment is written once, so the size recorded in the cache index is its real size --
an endlessly appended file is indexed at whatever it weighed on first registration.
And expiry becomes possible at all: dropping last week's samples must not mean deleting
the file currently being written to.
"""

SCHEMA_VERSION = 1
"""Row format version. Read as a filter, not just recorded.

Version 0 rows are pre-dated: they carry monotonic values in ``started_at`` and
``ended_at``, which are not comparable to wall-clock ones or to each other across a
reboot. Queries exclude them rather than blending two incompatible timebases into one
series -- a chart drawn from that mixture is wrong in a way nobody can see.
"""


@dataclass(frozen=True)
class ReadingBatch:
    """One channel's buffered samples, detached from the store and ready to write.

    Exists so buffer mutation and blocking I/O can happen in different places: the
    sampling loop drains on the event loop, then hands the batch to a worker.
    """

    readings: tuple[Reading, ...]
    dropped: int


@dataclass(frozen=True)
class ReadingWindow:
    """One downsampled interval for a single channel.

    Stores the shape of the interval rather than a single average, because the reason to
    keep history is to answer "what happened", and a mean alone hides the excursion that
    made the interval worth keeping.
    """

    device_id: str
    channel_id: str
    quantity: str
    unit: str
    started_at: float
    ended_at: float
    samples: int
    dropped: int
    min_value: float | None
    max_value: float | None
    mean_value: float | None
    last_value: Any
    quality_worst: str

    def to_row(self) -> tuple[Any, ...]:
        return (
            self.device_id,
            self.channel_id,
            self.quantity,
            self.unit,
            self.started_at,
            self.ended_at,
            self.samples,
            self.dropped,
            self.min_value,
            self.max_value,
            self.mean_value,
            None if self.last_value is None else str(self.last_value),
            self.quality_worst,
            SCHEMA_VERSION,
        )


def summarize_window(readings: Sequence[Reading], *, dropped: int = 0) -> ReadingWindow | None:
    """Reduce a run of readings to one interval, or None when there is nothing to store."""
    if not readings:
        return None
    first, last = readings[0], readings[-1]
    numeric = [n for r in readings if (n := as_numeric(r.value)) is not None]
    # Worst quality wins: an interval containing one saturated sample is not an "ok"
    # interval, and collapsing to the latest quality would hide exactly the sample
    # somebody would later want to find.
    worst = _worst_quality(r.quality for r in readings)
    return ReadingWindow(
        device_id=first.device_id,
        channel_id=first.channel_id,
        quantity=first.quantity,
        unit=first.unit,
        started_at=first.observed_at,
        ended_at=last.observed_at,
        samples=len(readings),
        dropped=dropped,
        min_value=min(numeric) if numeric else None,
        max_value=max(numeric) if numeric else None,
        mean_value=(sum(numeric) / len(numeric)) if numeric else None,
        last_value=last.value,
        quality_worst=worst,
    )


class ReadingStore:
    """Persists raw samples to session cache and downsampled windows to DuckDB.

    Constructed per registry, not per channel: one append-only file and one DuckDB
    connection serve every channel, because a bench with eight channels would otherwise
    hold eight file handles and eight connections for data that is written in small
    bursts.

    Every write path is contained. Losing observability is bad; taking a sampling loop
    down because a disk filled up is worse, so a failed flush is logged and dropped
    rather than propagated into the loop that produced it.
    """

    def __init__(
        self,
        *,
        raw_dir: Path | None = None,
        db_path: Path | None = None,
        cache_manager: Any = None,
        workspace_id: str = "",
        session_id: str = "",
        raw_ttl_s: float = DEFAULT_RAW_TTL_S,
        downsample_interval_s: float = DEFAULT_DOWNSAMPLE_INTERVAL_S,
        history_ttl_s: float = DEFAULT_HISTORY_TTL_S,
        raw_segment_bytes: int = DEFAULT_RAW_SEGMENT_BYTES,
    ) -> None:
        self._raw_dir = raw_dir
        self._db_path = db_path
        self._cache = cache_manager
        self._workspace_id = workspace_id
        self._session_id = session_id
        self._raw_ttl_s = raw_ttl_s
        self._downsample_interval_s = max(1.0, downsample_interval_s)
        self._history_ttl_s = max(0.0, history_ttl_s)
        self._raw_segment_bytes = max(1, int(raw_segment_bytes))
        self._pending: dict[tuple[str, str], list[Reading]] = {}
        self._dropped: dict[tuple[str, str], int] = {}
        self._window_start: dict[tuple[str, str], float] = {}
        self._segments: dict[tuple[str, str], int] = {}
        self._db_ready = False
        self._raw_writes = 0
        self._windows_written = 0
        self._write_failures = 0
        self._rows_pruned = 0
        self._last_prune_at = 0.0

    # ── Ingest ──

    def record(self, reading: Reading, *, dropped: int = 0) -> None:
        """Buffer one sample. Cheap by design; the sampling loop calls it per reading."""
        key = (reading.device_id, reading.channel_id)
        self._pending.setdefault(key, []).append(reading)
        if dropped:
            self._dropped[key] = self._dropped.get(key, 0) + dropped
        # Monotonic, matching ``due_for_flush``. The window boundary is an elapsed-time
        # question, and wall-clock can step backwards mid-window.
        self._window_start.setdefault(key, reading.monotonic_at)

    def due_for_flush(self, *, now: float | None = None) -> bool:
        """Return whether any channel has accumulated a full downsample interval."""
        if not self._pending:
            return False
        moment = now if now is not None else time.monotonic()
        return any(
            moment - self._window_start.get(key, moment) >= self._downsample_interval_s
            for key in self._pending
        )

    def drain(self, *, force: bool = False, now: float | None = None) -> tuple[ReadingBatch, ...]:
        """Detach every closed window from the buffers, without doing any I/O.

        Split from the write so the caller can move the blocking part off the event
        loop while buffer mutation stays single-threaded. Draining and *not* writing
        loses the batch, so every caller must pass what it gets to ``write_batches``.
        """
        if not self._pending:
            return ()
        moment = now if now is not None else time.monotonic()
        batches: list[ReadingBatch] = []
        for key in list(self._pending):
            started = self._window_start.get(key, moment)
            if not force and moment - started < self._downsample_interval_s:
                continue
            readings = self._pending.pop(key, [])
            dropped = self._dropped.pop(key, 0)
            self._window_start.pop(key, None)
            if not readings:
                continue
            batches.append(ReadingBatch(readings=tuple(readings), dropped=dropped))
        return tuple(batches)

    def write_batches(self, batches: Sequence[ReadingBatch]) -> int:
        """Persist drained batches, returning how many windows were written.

        Blocking: appends files and opens DuckDB. Safe to call from a worker thread
        because it touches no buffer the sampling loop also touches.
        """
        if not batches:
            return 0
        windows: list[ReadingWindow] = []
        for batch in batches:
            self._append_raw(batch.readings)
            window = summarize_window(batch.readings, dropped=batch.dropped)
            if window is not None:
                windows.append(window)
        written = self._write_windows(windows)
        self._windows_written += written
        return written

    def flush(self, *, force: bool = False, now: float | None = None) -> int:
        """Drain and write in one call, for teardown and for callers off the hot path."""
        return self.write_batches(self.drain(force=force, now=now))

    # ── Raw tier ──

    def _append_raw(self, readings: Sequence[Reading]) -> None:
        """Append samples as NDJSON to the current segment for this channel.

        NDJSON rather than a binary format because these files are evidence: when an
        experiment goes wrong somebody needs to read them with ordinary tools, and a
        partially written line is recoverable where a truncated binary record is not.

        The segment is re-indexed after every append, and rolled once it passes its
        byte cap. Both are required by how ``CacheManager`` accounts for an artifact:
        it records the size it finds at registration, so a file registered once and
        appended to forever is counted at the few kilobytes it started as, and its TTL
        runs from the first sample rather than the last.
        """
        if self._raw_dir is None:
            return
        first = readings[0]
        key = (first.device_id, first.channel_id)
        path = self._segment_path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                for reading in readings:
                    handle.write(json.dumps(reading.to_dict(), ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Could not append hardware readings to %s: %s", path, exc)
            return
        self._raw_writes += len(readings)
        self._index_raw_file(path)
        self._roll_if_full(key, path)

    def _segment_path(self, key: tuple[str, str]) -> Path:
        device_id, channel_id = key
        index = self._segments.setdefault(key, self._resume_segment(key))
        assert self._raw_dir is not None
        return self._raw_dir / f"{device_id}.{channel_id}.{index:04d}.ndjson"

    def _resume_segment(self, key: tuple[str, str]) -> int:
        """Continue after the highest existing segment for this channel.

        A restart within one session must not reopen a segment that was already closed
        and indexed at its final size, nor overwrite one.
        """
        if self._raw_dir is None:
            return 0
        device_id, channel_id = key
        highest = -1
        try:
            for existing in self._raw_dir.glob(f"{device_id}.{channel_id}.*.ndjson"):
                parts = existing.name.split(".")
                if len(parts) >= 4 and parts[-2].isdigit():
                    highest = max(highest, int(parts[-2]))
        except OSError:
            return 0
        return highest + 1 if highest >= 0 else 0

    def _roll_if_full(self, key: tuple[str, str], path: Path) -> None:
        """Start a new segment once this one passes its cap.

        The finished segment keeps the registration it already has, which is now its
        final and correct size.
        """
        try:
            if path.stat().st_size < self._raw_segment_bytes:
                return
        except OSError:
            return
        self._segments[key] = self._segments.get(key, 0) + 1
        logger.debug("hardware readings: rolled %s at its size cap", path.name)

    def _index_raw_file(self, path: Path) -> None:
        """Index the file as sensitive, non-syncable, TTL-bounded session data.

        Re-registered on every append rather than once per file. The index is keyed by
        path, so this refreshes the recorded size and restarts the TTL from the most
        recent sample -- which is what "keep for seven days" has to mean for a file
        that is still being written to.
        """
        if self._cache is None:
            return
        try:
            self._cache.register(
                path=path,
                scope="session",
                category=READINGS_CATEGORY,
                source=str(path.name),
                workspace_id=self._workspace_id,
                session_id=self._session_id,
                expires_at=time.time() + self._raw_ttl_s,
                sensitive=True,
                syncable=False,
                owner_component="hardware",
            )
        except Exception as exc:  # noqa: BLE001 - indexing must not break sampling
            logger.warning("Could not index hardware reading file %s: %s", path, exc)

    # ── Downsampled tier ──

    def _write_windows(self, windows: Sequence[ReadingWindow]) -> int:
        """Insert every window over one connection, returning how many landed.

        One connection per drain rather than per window: a bench with eight channels
        would otherwise open and close DuckDB eight times a minute for rows that
        arrive together.

        A failure here is counted, not just logged. Losing windows to a locked
        database is the one storage fault that leaves no trace in the data itself --
        ``windows_written`` alone is a numerator with no denominator, so an outage
        looks identical to an idle bench.
        """
        if not windows or self._db_path is None:
            return 0
        try:
            import duckdb
        except ImportError:
            logger.debug("duckdb unavailable; hardware history not persisted")
            self._db_path = None
            return 0
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = duckdb.connect(str(self._db_path))
        except Exception as exc:  # noqa: BLE001 - a locked DB must not stop sampling
            self._write_failures += len(windows)
            logger.warning("Could not open %s for hardware history: %s", self._db_path, exc)
            return 0
        written = 0
        try:
            if not self._db_ready:
                self._ensure_schema(connection)
                self._db_ready = True
            for window in windows:
                connection.execute(_INSERT, window.to_row())
                written += 1
            self._prune(connection)
        except Exception as exc:  # noqa: BLE001 - as above
            self._write_failures += len(windows) - written
            logger.warning("Could not write hardware history window: %s", exc)
        finally:
            try:
                connection.close()
            except Exception:  # noqa: BLE001 - close must never raise here
                logger.debug("hardware history connection close failed", exc_info=True)
        return written

    @staticmethod
    def _ensure_schema(connection: Any) -> None:
        """Create the table, add the version column, then index it.

        The column is added with default 0, not the current version: rows already on
        disk hold monotonic instants, and defaulting them to 1 would relabel unusable
        data as current and quietly readmit it into every query.

        The index comes last, and that order is load-bearing. Indexing a column added
        by the same migration fails with a binder error on an older database, and the
        failure surfaces as a write that never lands rather than as a schema problem.
        """
        connection.execute(_SCHEMA)
        try:
            connection.execute(
                "ALTER TABLE reading_windows ADD COLUMN IF NOT EXISTS schema_version INTEGER DEFAULT 0"
            )
        except Exception:  # noqa: BLE001 - already present, or an engine without IF NOT EXISTS
            logger.debug("reading_windows schema_version column already present", exc_info=True)
        try:
            connection.execute(_INDEX)
        except Exception:  # noqa: BLE001 - an unindexed table still answers, only slower
            logger.debug("reading_windows index unavailable", exc_info=True)

    def _prune(self, connection: Any) -> None:
        """Drop windows past the retention horizon.

        Runs on the write path rather than a timer, so retention needs no scheduler and
        cannot silently stop: the only process that grows this table is the one that
        trims it. Rate-limited because a delete per flush would cost more than the
        insert it follows, and bounded data does not need minute-level precision.

        Version-0 rows are deleted by *age against the current clock*, which they will
        always fail, because their instants are monotonic and cannot be compared to a
        wall-clock cutoff at all. Excluding them from queries left them on disk forever;
        this is what finally removes them.
        """
        if self._history_ttl_s <= 0:
            return
        now = time.time()
        if now - self._last_prune_at < _PRUNE_INTERVAL_S:
            return
        self._last_prune_at = now
        cutoff = now - self._history_ttl_s
        try:
            deleted = connection.execute(_PRUNE, (cutoff, SCHEMA_VERSION)).fetchall()
        except Exception as exc:  # noqa: BLE001 - retention is maintenance, not a duty
            logger.debug("Could not prune hardware history: %s", exc)
            return
        # ``RETURNING 1`` yields one row per deleted window, so the count is the number
        # of rows -- not the value in the first one, which is always the literal 1. Read
        # that way the counter reported "1" for every prune regardless of size, and the
        # only assertion covering it was ``>= 1``, which the wrong reading satisfies.
        count = len(deleted)
        if count:
            self._rows_pruned += count
            logger.info("hardware history: pruned %d window(s) past retention", count)

    # ── Query ──

    def history(
        self, device_id: str, channel_id: str, *, limit: int = 200
    ) -> tuple[dict[str, Any], ...]:
        """Return recent downsampled windows, oldest first.

        This is what makes physical experience reusable across sessions -- the point of
        persisting at all. It returns windows, never raw samples: the raw tier is evidence
        for a human, not context for a model.
        """
        if self._db_path is None or not self._db_path.exists():
            return ()
        try:
            import duckdb

            connection = duckdb.connect(str(self._db_path), read_only=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not read hardware history: %s", exc)
            return ()
        try:
            rows = connection.execute(_SELECT, (device_id, channel_id, int(limit))).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Hardware history query failed: %s", exc)
            return ()
        finally:
            try:
                connection.close()
            except Exception:  # noqa: BLE001
                logger.debug("hardware history connection close failed", exc_info=True)
        return tuple(dict(zip(_COLUMNS, row)) for row in reversed(rows))

    # ── Introspection ──

    @property
    def raw_writes(self) -> int:
        return self._raw_writes

    @property
    def windows_written(self) -> int:
        return self._windows_written

    @property
    def write_failures(self) -> int:
        """Windows that were drained but could not be persisted."""
        return self._write_failures

    @property
    def rows_pruned(self) -> int:
        """History windows removed by retention over this store's lifetime."""
        return self._rows_pruned

    @property
    def pending_channels(self) -> int:
        return len(self._pending)

    def close(self) -> None:
        """Flush whatever is buffered. Must never raise.

        Called during teardown, where an exception would mask the failure that caused the
        shutdown -- and where losing the last interval of a long run is exactly the data
        somebody will want.
        """
        try:
            self.flush(force=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hardware reading flush failed during close: %s", exc, exc_info=True)


_QUALITY_ORDER = ("ok", "suspect", "stale", "saturated")


def _worst_quality(values: Iterable[str]) -> str:
    worst = "ok"
    worst_rank = 0
    for value in values:
        rank = _QUALITY_ORDER.index(value) if value in _QUALITY_ORDER else len(_QUALITY_ORDER)
        if rank > worst_rank:
            worst, worst_rank = value, rank
    return worst


_COLUMNS = (
    "device_id",
    "channel_id",
    "quantity",
    "unit",
    "started_at",
    "ended_at",
    "samples",
    "dropped",
    "min_value",
    "max_value",
    "mean_value",
    "last_value",
    "quality_worst",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reading_windows (
    device_id     VARCHAR NOT NULL,
    channel_id    VARCHAR NOT NULL,
    quantity      VARCHAR,
    unit          VARCHAR,
    started_at    DOUBLE NOT NULL,
    ended_at      DOUBLE NOT NULL,
    samples       BIGINT NOT NULL,
    dropped       BIGINT NOT NULL,
    min_value     DOUBLE,
    max_value     DOUBLE,
    mean_value    DOUBLE,
    last_value    VARCHAR,
    quality_worst VARCHAR,
    schema_version INTEGER DEFAULT 0
)
"""

_INSERT = """
INSERT INTO reading_windows (
    device_id, channel_id, quantity, unit, started_at, ended_at,
    samples, dropped, min_value, max_value, mean_value, last_value, quality_worst,
    schema_version
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INDEX = """
CREATE INDEX IF NOT EXISTS idx_reading_windows_channel
ON reading_windows (device_id, channel_id, ended_at)
"""
"""Matches the only query shape: filter by channel, order by recency.

Without it every ``history()`` call scans the whole table, and the table is the one
thing here that grows without bound between prunes.
"""

_PRUNE_INTERVAL_S = 3600.0
"""Floor on how often retention runs. A delete per flush would cost more than the
insert it follows, and a bounded table does not need minute-level precision."""

_PRUNE = """
DELETE FROM reading_windows
WHERE ended_at < ? OR schema_version < ?
RETURNING 1
"""
"""Age *or* an unreadable timebase. Version-0 rows carry monotonic instants, so no
wall-clock cutoff can ever match them -- excluding them from queries left them on
disk forever."""

_SELECT = f"""
SELECT {", ".join(_COLUMNS)}
FROM reading_windows
WHERE device_id = ? AND channel_id = ? AND schema_version >= {SCHEMA_VERSION}
ORDER BY ended_at DESC
LIMIT ?
"""


__all__ = [
    "DEFAULT_DOWNSAMPLE_INTERVAL_S",
    "DEFAULT_FLUSH_INTERVAL_S",
    "DEFAULT_HISTORY_TTL_S",
    "DEFAULT_RAW_SEGMENT_BYTES",
    "DEFAULT_RAW_TTL_S",
    "READINGS_CATEGORY",
    "SCHEMA_VERSION",
    "ReadingBatch",
    "ReadingStore",
    "ReadingWindow",
    "summarize_window",
]
