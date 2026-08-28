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
        started_at=first.timestamp,
        ended_at=last.timestamp,
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
    ) -> None:
        self._raw_dir = raw_dir
        self._db_path = db_path
        self._cache = cache_manager
        self._workspace_id = workspace_id
        self._session_id = session_id
        self._raw_ttl_s = raw_ttl_s
        self._downsample_interval_s = max(1.0, downsample_interval_s)
        self._pending: dict[tuple[str, str], list[Reading]] = {}
        self._dropped: dict[tuple[str, str], int] = {}
        self._window_start: dict[tuple[str, str], float] = {}
        self._registered_files: set[Path] = set()
        self._db_ready = False
        self._raw_writes = 0
        self._windows_written = 0

    # ── Ingest ──

    def record(self, reading: Reading, *, dropped: int = 0) -> None:
        """Buffer one sample. Cheap by design; the sampling loop calls it per reading."""
        key = (reading.device_id, reading.channel_id)
        self._pending.setdefault(key, []).append(reading)
        if dropped:
            self._dropped[key] = self._dropped.get(key, 0) + dropped
        self._window_start.setdefault(key, reading.timestamp or time.monotonic())

    def due_for_flush(self, *, now: float | None = None) -> bool:
        """Return whether any channel has accumulated a full downsample interval."""
        if not self._pending:
            return False
        moment = now if now is not None else time.monotonic()
        return any(
            moment - self._window_start.get(key, moment) >= self._downsample_interval_s
            for key in self._pending
        )

    def flush(self, *, force: bool = False, now: float | None = None) -> int:
        """Persist buffered samples, returning how many windows were written."""
        if not self._pending:
            return 0
        moment = now if now is not None else time.monotonic()
        written = 0
        for key in list(self._pending):
            started = self._window_start.get(key, moment)
            if not force and moment - started < self._downsample_interval_s:
                continue
            readings = self._pending.pop(key, [])
            dropped = self._dropped.pop(key, 0)
            self._window_start.pop(key, None)
            if not readings:
                continue
            self._append_raw(readings)
            window = summarize_window(readings, dropped=dropped)
            if window is not None and self._write_window(window):
                written += 1
        self._windows_written += written
        return written

    # ── Raw tier ──

    def _append_raw(self, readings: Sequence[Reading]) -> None:
        """Append samples as NDJSON to the session cache.

        NDJSON rather than a binary format because these files are evidence: when an
        experiment goes wrong somebody needs to read them with ordinary tools, and a
        partially written line is recoverable where a truncated binary record is not.
        """
        if self._raw_dir is None:
            return
        first = readings[0]
        path = self._raw_dir / f"{first.device_id}.{first.channel_id}.ndjson"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                for reading in readings:
                    handle.write(json.dumps(reading.to_dict(), ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Could not append hardware readings to %s: %s", path, exc)
            return
        self._raw_writes += len(readings)
        self._register_raw_file(path)

    def _register_raw_file(self, path: Path) -> None:
        """Index the file as sensitive, non-syncable, TTL-bounded session data.

        Registered once per file rather than per flush: the index tracks the artifact, and
        re-registering on every append would grow the index at sampling rate.
        """
        if self._cache is None or path in self._registered_files:
            return
        self._registered_files.add(path)
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
            self._registered_files.discard(path)

    # ── Downsampled tier ──

    def _write_window(self, window: ReadingWindow) -> bool:
        if self._db_path is None:
            return False
        try:
            import duckdb
        except ImportError:
            logger.debug("duckdb unavailable; hardware history not persisted")
            self._db_path = None
            return False
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = duckdb.connect(str(self._db_path))
        except Exception as exc:  # noqa: BLE001 - a locked DB must not stop sampling
            logger.warning("Could not open %s for hardware history: %s", self._db_path, exc)
            return False
        try:
            if not self._db_ready:
                connection.execute(_SCHEMA)
                self._db_ready = True
            connection.execute(_INSERT, window.to_row())
            return True
        except Exception as exc:  # noqa: BLE001 - as above
            logger.warning("Could not write hardware history window: %s", exc)
            return False
        finally:
            try:
                connection.close()
            except Exception:  # noqa: BLE001 - close must never raise here
                logger.debug("hardware history connection close failed", exc_info=True)

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
    quality_worst VARCHAR
)
"""

_INSERT = """
INSERT INTO reading_windows (
    device_id, channel_id, quantity, unit, started_at, ended_at,
    samples, dropped, min_value, max_value, mean_value, last_value, quality_worst
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT = f"""
SELECT {", ".join(_COLUMNS)}
FROM reading_windows
WHERE device_id = ? AND channel_id = ?
ORDER BY ended_at DESC
LIMIT ?
"""


__all__ = [
    "DEFAULT_DOWNSAMPLE_INTERVAL_S",
    "DEFAULT_FLUSH_INTERVAL_S",
    "DEFAULT_RAW_TTL_S",
    "READINGS_CATEGORY",
    "ReadingStore",
    "ReadingWindow",
    "summarize_window",
]
