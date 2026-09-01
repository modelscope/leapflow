"""Versioned storage for device calibration results.

A calibration is not a reading. A reading is a sample of what a channel is doing right
now; a calibration is the transform that makes those samples *mean* something -- the
matrix, the fitted parameters, the mounted pose that a later reading is interpreted
through. It changes rarely, it must survive across sessions, and every version of it
matters: comparing today's run against last week's is impossible if the calibration
that stood between the raw signal and the number silently changed in between.

So each result is kept as its own row, keyed by ``(device_id, procedure_id, recorded_at)``
and stamped with a schema version, in the profile's ``instrument.duckdb`` alongside the
downsampled history it will be read next to. Nothing here is overwritten: a re-run of the
same procedure lands a new row at a new instant, and ``latest`` is simply the most recent
one. The old rows are the audit trail of how the instrument's frame of reference moved.

Every write path is contained, for the same reason the reading store's is: losing the
ability to record a calibration is bad, but taking a calibration procedure down because
the database was momentarily locked is worse.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from leapflow.storage.connection import ConnectionHolder, LocalConnectionHolder

logger = logging.getLogger(__name__)

CALIBRATION_CATEGORY = "hardware_calibration"
"""Cache category for the calibration tier of ``instrument.duckdb``.

Shares the file, and therefore the sensitivity posture, of the downsampled history
tier: a calibration can encode the geometry of a proprietary fixture, so it inherits
the same sensitive/non-syncable default the reading store applies to the same file.
"""

CALIBRATION_SCHEMA_VERSION = 1
"""Row format version. Recorded on every row and read as a filter, so a future format
change can exclude rows written under an incompatible layout rather than blending them
into a query that assumes today's shape."""


@dataclass(frozen=True)
class CalibrationRecord:
    """One versioned calibration result for a single device and procedure.

    ``parameters`` carries the fitted scalars a procedure produced; ``matrix`` the
    transform (a list of rows) when one applies; ``pose`` the mounted position and
    orientation. All three are stored as JSON, because their shape is procedure-defined
    and pinning a column per field would make every new procedure a schema migration.
    """

    device_id: str
    procedure_id: str
    recorded_at: float
    parameters: Mapping[str, Any] = field(default_factory=dict)
    matrix: Sequence[Sequence[float]] | None = None
    pose: Mapping[str, Any] | None = None
    notes: str = ""

    def to_row(self) -> tuple[Any, ...]:
        return (
            self.device_id,
            self.procedure_id,
            float(self.recorded_at),
            json.dumps(dict(self.parameters), ensure_ascii=False),
            None if self.matrix is None else json.dumps(self.matrix),
            None if self.pose is None else json.dumps(dict(self.pose), ensure_ascii=False),
            self.notes,
            CALIBRATION_SCHEMA_VERSION,
        )


class CalibrationStore:
    """Persists versioned calibration results to the profile's ``instrument.duckdb``.

    Shares the DuckDB file -- and, when one is injected, the very connection -- with the
    reading store, because a single process must not hold two independent read-write
    connections to the same DuckDB file. The registry owns one ``ConnectionHolder`` for
    ``instrument.duckdb`` and hands it to both stores; the holder's thread-local cursor
    keeps a write off the event loop from blocking a read on it.
    """

    def __init__(
        self,
        *,
        db_path: Path | None = None,
        connection_holder: ConnectionHolder | None = None,
        cache_manager: Any = None,
        sensitive: bool = True,
    ) -> None:
        self._db_path = db_path
        # Prefer an injected holder; fall back to owning one only when handed a bare
        # path. The registry always injects, so the fallback exists for direct use
        # (tests, tools) rather than the running system.
        self._holder: ConnectionHolder | None = connection_holder
        self._owns_holder = False
        if self._holder is None and self._db_path is not None:
            self._holder = LocalConnectionHolder(self._db_path)
            self._owns_holder = True
        self._cache = cache_manager
        self._sensitive = bool(sensitive)
        self._db_registered = False
        self._db_ready = False
        self._records_written = 0
        self._write_failures = 0

    # ── Ingest ──

    def record(self, record: CalibrationRecord) -> bool:
        """Persist one calibration result, returning whether it landed.

        Contained: a locked or missing database is logged and counted, never raised into
        the calibration procedure that produced the result. Idempotent on
        ``(device_id, procedure_id, recorded_at)`` -- re-recording the same instant
        replaces the row rather than duplicating it.
        """
        if self._holder is None:
            return False
        try:
            connection = self._holder.connection
        except Exception as exc:  # noqa: BLE001 - a locked DB must not stop calibration
            self._write_failures += 1
            logger.warning("Could not open %s for calibration: %s", self._db_path, exc)
            return False
        try:
            if not self._db_ready:
                self._ensure_schema(connection)
                self._db_ready = True
                self._register_db()
            connection.execute(_INSERT, record.to_row())
        except Exception as exc:  # noqa: BLE001 - as above
            self._write_failures += 1
            logger.warning("Could not persist calibration record: %s", exc)
            return False
        self._records_written += 1
        return True

    def _register_db(self) -> None:
        """Index ``instrument.duckdb`` so a profile backup honours its sensitivity.

        Keyed by path and guarded by a flag, so it is a single idempotent call. Registers
        the same posture the reading store applies to the same file, so the calibration
        tier is governed even in a profile that persists calibrations but not readings.
        """
        if self._cache is None or self._db_path is None or self._db_registered:
            return
        try:
            self._cache.register(
                path=self._db_path,
                scope="profile",
                category=CALIBRATION_CATEGORY,
                source=str(self._db_path.name),
                sensitive=self._sensitive,
                syncable=not self._sensitive,
                owner_component="hardware",
            )
            self._db_registered = True
        except Exception as exc:  # noqa: BLE001 - indexing must not break calibration
            logger.warning(
                "Could not index calibration database %s: %s", self._db_path, exc
            )

    @staticmethod
    def _ensure_schema(connection: Any) -> None:
        """Create the calibration table and its lookup index.

        The index comes after the table, matching the reading store's ordering: it is a
        maintenance aid, not a correctness requirement, so a failure to build it leaves a
        table that still answers, only slower.
        """
        connection.execute(_SCHEMA)
        try:
            connection.execute(_INDEX)
        except Exception:  # noqa: BLE001 - an unindexed table still answers, only slower
            logger.debug("calibration_records index unavailable", exc_info=True)

    # ── Query ──

    def latest(
        self, device_id: str, procedure_id: str | None = None
    ) -> CalibrationRecord | None:
        """Return the most recent calibration for a device, or a specific procedure.

        With ``procedure_id`` omitted, returns the newest calibration of any procedure on
        the device -- the answer to "when was this device last calibrated at all".
        """
        rows = self._query(device_id, procedure_id, limit=1)
        return rows[0] if rows else None

    def latest_time(self, device_id: str, procedure_id: str | None = None) -> float | None:
        """Return the instant of the most recent calibration, or None if never calibrated.

        This is what ``hw_describe`` surfaces: a single wall-clock number a reader can
        compare against, cheaper to carry in a reference document than a whole record.
        """
        record = self.latest(device_id, procedure_id)
        return record.recorded_at if record is not None else None

    def history(
        self, device_id: str, procedure_id: str | None = None, *, limit: int = 50
    ) -> tuple[CalibrationRecord, ...]:
        """Return recent calibrations for a device, newest first.

        The versions are the point: a drift in a fitted parameter across successive runs
        is exactly what tells an operator the instrument's frame of reference is moving.
        """
        return self._query(device_id, procedure_id, limit=limit)

    def _query(
        self, device_id: str, procedure_id: str | None, *, limit: int
    ) -> tuple[CalibrationRecord, ...]:
        if self._holder is None:
            return ()
        # Opening a not-yet-created file would materialise an empty database; report no
        # history instead, matching the reading store's read guard.
        if self._db_path is not None and not self._db_path.exists():
            return ()
        try:
            connection = self._holder.connection
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not read calibration history: %s", exc)
            return ()
        if procedure_id is None:
            sql = _SELECT_BY_DEVICE
            params: tuple[Any, ...] = (device_id, CALIBRATION_SCHEMA_VERSION, int(limit))
        else:
            sql = _SELECT_BY_PROCEDURE
            params = (device_id, procedure_id, CALIBRATION_SCHEMA_VERSION, int(limit))
        try:
            rows = connection.execute(sql, params).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Calibration history query failed: %s", exc)
            return ()
        return tuple(_row_to_record(row) for row in rows)

    # ── Introspection ──

    @property
    def records_written(self) -> int:
        return self._records_written

    @property
    def write_failures(self) -> int:
        return self._write_failures

    def close(self) -> None:
        """Close an owned connection. Must never raise.

        Only closes the holder it created itself: when the registry injected the shared
        ``instrument.duckdb`` holder, closing it here would pull the connection out from
        under the reading store, so that responsibility stays with the owner.
        """
        if self._owns_holder and self._holder is not None:
            try:
                self._holder.close()
            except Exception:  # noqa: BLE001 - teardown must not propagate
                logger.debug("calibration holder close failed", exc_info=True)


def _row_to_record(row: Sequence[Any]) -> CalibrationRecord:
    device_id, procedure_id, recorded_at, parameters, matrix, pose, notes = row
    return CalibrationRecord(
        device_id=device_id,
        procedure_id=procedure_id,
        recorded_at=recorded_at,
        parameters=_loads(parameters, {}),
        matrix=_loads(matrix, None),
        pose=_loads(pose, None),
        notes=notes or "",
    )


def _loads(value: Any, default: Any) -> Any:
    """Decode a stored JSON column, falling back rather than raising on a bad value.

    A row whose JSON somehow does not parse is degraded data, not a reason to fail the
    whole query: return the default and keep the rest of the history readable.
    """
    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        logger.debug("Could not decode calibration column %r", value, exc_info=True)
        return default


_COLUMNS = (
    "device_id",
    "procedure_id",
    "recorded_at",
    "parameters",
    "matrix",
    "pose",
    "notes",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calibration_records (
    device_id      VARCHAR NOT NULL,
    procedure_id   VARCHAR NOT NULL,
    recorded_at    DOUBLE NOT NULL,
    parameters     VARCHAR,
    matrix         VARCHAR,
    pose           VARCHAR,
    notes          VARCHAR,
    schema_version INTEGER DEFAULT 1,
    PRIMARY KEY (device_id, procedure_id, recorded_at)
)
"""

_INSERT = """
INSERT OR REPLACE INTO calibration_records (
    device_id, procedure_id, recorded_at, parameters, matrix, pose, notes, schema_version
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_INDEX = """
CREATE INDEX IF NOT EXISTS idx_calibration_device_time
ON calibration_records (device_id, recorded_at)
"""
"""Serves the "latest for a device" and history queries, which order by recency within a
device across every procedure -- a shape the primary key's ``(device_id, procedure_id)``
prefix cannot answer without a scan."""

_SELECT_BY_DEVICE = f"""
SELECT {", ".join(_COLUMNS)}
FROM calibration_records
WHERE device_id = ? AND schema_version >= ?
ORDER BY recorded_at DESC
LIMIT ?
"""

_SELECT_BY_PROCEDURE = f"""
SELECT {", ".join(_COLUMNS)}
FROM calibration_records
WHERE device_id = ? AND procedure_id = ? AND schema_version >= ?
ORDER BY recorded_at DESC
LIMIT ?
"""


__all__ = [
    "CALIBRATION_CATEGORY",
    "CALIBRATION_SCHEMA_VERSION",
    "CalibrationRecord",
    "CalibrationStore",
]
