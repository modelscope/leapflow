"""Durable persistence of sampled hardware readings.

This closes the gap that made every form of learning from physical experience
impossible: before it, samples lived only in a bounded in-memory ring and vanished with
the process, so there was no series to learn *from*. Parameter reuse -- discovering that
a viscous sample wants a slow rate and remembering it next time -- starts here.

Two tiers are asserted separately because they carry different obligations. Raw samples
are session-scoped, sensitive, and non-syncable: a qPCR curve can carry patient sample
information. Downsampled windows are the long-term tier a later analysis reads.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from leapflow.cache.manager import CacheManager, CacheScope
from leapflow.hardware.context import (
    HC_VERSION,
    Channel,
    ContextProvenance,
    Direction,
    Envelope,
    HardwareContext,
    Quality,
    TransportRef,
)
from leapflow.hardware.reading_store import (
    READINGS_CATEGORY,
    ReadingStore,
    summarize_window,
)
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.transport import Reading
from leapflow.layout import build_layout


# ════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════


_WALL_EPOCH = time.time() - 3600.0
"""Wall-clock base for ``observed_at``: an hour ago.

Relative to now, not a fixed constant. A fixed literal drifted past the history
retention horizon as time passed, and retention then deleted the fixture out from
under assertions that had nothing to do with it -- a test that decays into a failure
on a calendar. Still three orders of magnitude above the small monotonic values used
for window boundaries, so persisting the wrong clock remains obvious."""


def _reading(
    value: Any,
    *,
    sequence: int = 1,
    at: float = 0.0,
    quality: str = Quality.OK.value,
    channel: str = "level",
) -> Reading:
    return Reading(
        device_id="dev",
        channel_id=channel,
        value=value,
        quantity="generic.level",
        unit="unit",
        monotonic_at=at,
        observed_at=_WALL_EPOCH + at,
        sequence=sequence,
        quality=quality,
    )


def _store(tmp_path: Path, **kwargs: Any) -> ReadingStore:
    return ReadingStore(
        raw_dir=tmp_path / "raw",
        db_path=tmp_path / "instrument.duckdb",
        downsample_interval_s=kwargs.pop("downsample_interval_s", 10.0),
        **kwargs,
    )


class _StaticProvider:
    kind = "static"

    def __init__(self, *contexts: HardwareContext) -> None:
        self._contexts = contexts

    def discover(self) -> tuple[HardwareContext, ...]:
        return self._contexts


def _context(sample_rate_hz: float = 50.0) -> HardwareContext:
    return HardwareContext(
        device_id="dev",
        hc_version=HC_VERSION,
        halt_supported=True,
        transport=TransportRef(kind="mock", config={"values": {"level": 20.0}}),
        channels=(
            Channel(
                channel_id="level",
                direction=Direction.READ.value,
                quantity="generic.level",
                unit="unit",
                sample_rate_hz=sample_rate_hz,
                envelope=Envelope(declared=True, min_value=0.0, max_value=100.0),
            ),
        ),
        provenance=ContextProvenance(verified_by="tester"),
    )


# ════════════════════════════════════════════════════════════════
# Window summarisation
# ════════════════════════════════════════════════════════════════


def test_window_keeps_the_shape_not_just_the_mean() -> None:
    """A mean alone hides the excursion that made the interval worth keeping."""
    window = summarize_window(
        [
            _reading(10.0, sequence=1, at=1.0),
            _reading(90.0, sequence=2, at=2.0),
            _reading(20.0, sequence=3, at=3.0),
        ]
    )
    assert window is not None
    assert window.min_value == 10.0
    assert window.max_value == 90.0
    assert window.mean_value == pytest.approx(40.0)
    assert window.samples == 3
    # Wall-clock, not the monotonic boundary clock. A window is read back weeks later
    # and lined up against approvals and audit entries; a per-boot counter cannot be.
    assert window.started_at == _WALL_EPOCH + 1.0
    assert window.ended_at == _WALL_EPOCH + 3.0


def test_window_reports_the_worst_quality_not_the_last() -> None:
    """One saturated sample does not make an "ok" interval.

    Collapsing to the latest quality would hide exactly the sample somebody would later
    go looking for.
    """
    window = summarize_window(
        [
            _reading(1.0, sequence=1, quality=Quality.OK.value),
            _reading(2.0, sequence=2, quality=Quality.SATURATED.value),
            _reading(3.0, sequence=3, quality=Quality.OK.value),
        ]
    )
    assert window is not None
    assert window.quality_worst == Quality.SATURATED.value


def test_window_carries_the_dropped_count() -> None:
    window = summarize_window([_reading(1.0)], dropped=7)
    assert window is not None and window.dropped == 7


def test_window_tolerates_non_numeric_values() -> None:
    """A state channel has no min/max; the window must not invent them."""
    window = summarize_window([_reading("idle"), _reading("busy")])
    assert window is not None
    assert window.min_value is None
    assert window.mean_value is None
    assert window.last_value == "busy"


def test_empty_window_is_none_not_a_zero_row() -> None:
    """Absent data is omitted rather than stored as a row of zeros."""
    assert summarize_window([]) is None


# ════════════════════════════════════════════════════════════════
# Raw tier
# ════════════════════════════════════════════════════════════════


def test_raw_samples_are_written_as_readable_ndjson(tmp_path: Path) -> None:
    """These files are evidence: somebody must be able to read them with ordinary tools."""
    store = _store(tmp_path)
    for index in range(5):
        store.record(_reading(float(index), sequence=index, at=float(index)))
    store.flush(force=True)

    files = list((tmp_path / "raw").glob("*.ndjson"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    first = json.loads(lines[0])
    assert first["channel_id"] == "level"
    assert first["value"] == 0.0
    assert first["sequence"] == 0


def test_raw_files_are_separated_per_channel(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record(_reading(1.0, channel="level"))
    store.record(_reading(2.0, channel="other"))
    store.flush(force=True)
    names = sorted(p.name for p in (tmp_path / "raw").glob("*.ndjson"))
    assert names == ["dev.level.0000.ndjson", "dev.other.0000.ndjson"]


def test_raw_writes_append_across_flushes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record(_reading(1.0, sequence=1))
    store.flush(force=True)
    store.record(_reading(2.0, sequence=2))
    store.flush(force=True)
    path = tmp_path / "raw" / "dev.level.0000.ndjson"
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_raw_samples_are_indexed_as_sensitive_and_non_syncable(tmp_path: Path) -> None:
    """The central privacy obligation: physical data must not leave the machine.

    A qPCR curve can carry patient sample information and a production temperature trace
    can be a trade secret, so raw samples are treated like session visual artifacts --
    sensitive, non-syncable, and TTL bounded.
    """
    layout = build_layout(tmp_path / "data")
    profile_layout = layout.ensure(profile_id="default")
    cache_layout = profile_layout.cache
    readings_dir = cache_layout.category_dir(
        scope=CacheScope.SESSION.value,
        category=READINGS_CATEGORY,
        workspace_id="ws",
        session_id="sess",
    )
    manager = CacheManager(cache_layout, profile_id="default")
    store = ReadingStore(
        raw_dir=readings_dir,
        db_path=tmp_path / "instrument.duckdb",
        cache_manager=manager,
        workspace_id="ws",
        session_id="sess",
        raw_ttl_s=3600.0,
    )
    store.record(_reading(1.0))
    store.flush(force=True)

    entries = [e for e in manager.list_entries() if e.category == READINGS_CATEGORY]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.sensitive is True
    assert entry.syncable is False
    assert entry.scope == CacheScope.SESSION.value
    assert entry.session_id == "sess"
    assert entry.expires_at is not None and entry.expires_at > time.time()


def test_a_segment_is_re_indexed_in_place_with_a_current_size_and_ttl(tmp_path: Path) -> None:
    """Re-registration refreshes the artifact; it must not duplicate it.

    This test previously asserted the opposite rationale -- index once, never again --
    and it kept passing after the behaviour was reversed, because the index is keyed by
    path either way. What it never checked is the thing that was actually wrong:
    ``CacheManager`` records the size it finds at registration, so a file indexed once
    and appended to for hours is accounted at the few bytes it started as, and its TTL
    counts down from the first sample rather than the last.
    """
    layout = build_layout(tmp_path / "data")
    profile_layout = layout.ensure(profile_id="default")
    manager = CacheManager(profile_layout.cache, profile_id="default")
    store = ReadingStore(
        raw_dir=tmp_path / "raw",
        db_path=tmp_path / "instrument.duckdb",
        cache_manager=manager,
        workspace_id="ws",
        session_id="sess",
        raw_ttl_s=3600.0,
    )

    def _entry() -> Any:
        rows = [e for e in manager.list_entries() if e.category == READINGS_CATEGORY]
        assert len(rows) == 1, f"one entry per segment, found {len(rows)}"
        return rows[0]

    store.record(_reading(1.0, sequence=1, at=3600.0))
    store.flush(force=True)
    first = _entry()

    for index in range(2, 40):
        store.record(_reading(float(index), sequence=index, at=3600.0))
        store.flush(force=True)
    latest = _entry()

    assert latest.size_bytes > first.size_bytes, (
        "the indexed size must follow the file; a stale size makes the cache quota "
        "under-count this artifact by orders of magnitude"
    )
    assert latest.size_bytes == (tmp_path / "raw" / "dev.level.0000.ndjson").stat().st_size
    assert latest.expires_at is not None and first.expires_at is not None
    assert latest.expires_at >= first.expires_at, (
        "the TTL must run from the most recent sample: anchored to the first, a long "
        "run's evidence expires while it is still being written"
    )


def test_a_raw_file_rolls_at_its_size_cap(tmp_path: Path) -> None:
    """Segments bound the file and make expiry possible at all.

    A single append-only file cannot be partly expired: dropping last week's samples
    would mean deleting the file currently being written to.
    """
    layout = build_layout(tmp_path / "data")
    profile_layout = layout.ensure(profile_id="default")
    manager = CacheManager(profile_layout.cache, profile_id="default")
    store = ReadingStore(
        raw_dir=tmp_path / "raw",
        db_path=tmp_path / "instrument.duckdb",
        cache_manager=manager,
        session_id="sess",
        raw_segment_bytes=200,
    )
    for index in range(1, 12):
        store.record(_reading(float(index), sequence=index, at=3600.0))
        store.flush(force=True)

    names = sorted(p.name for p in (tmp_path / "raw").glob("*.ndjson"))
    assert len(names) > 1, f"expected the segment to roll, got {names}"
    assert names[0] == "dev.level.0000.ndjson"
    indexed = {e.path.name for e in manager.list_entries() if e.category == READINGS_CATEGORY}
    assert indexed == set(names), "every segment must be indexed, including finished ones"


def test_a_new_store_continues_after_the_highest_segment(tmp_path: Path) -> None:
    """A restart must not reopen or overwrite a segment already closed at its final size."""
    first = ReadingStore(raw_dir=tmp_path / "raw", db_path=tmp_path / "db.duckdb", raw_segment_bytes=120)
    for index in range(1, 8):
        first.record(_reading(float(index), sequence=index, at=3600.0))
        first.flush(force=True)
    before = sorted(p.name for p in (tmp_path / "raw").glob("*.ndjson"))
    assert len(before) > 1

    second = ReadingStore(raw_dir=tmp_path / "raw", db_path=tmp_path / "db.duckdb")
    second.record(_reading(99.0, sequence=99, at=3600.0))
    second.flush(force=True)
    after = sorted(p.name for p in (tmp_path / "raw").glob("*.ndjson"))
    assert len(after) == len(before) + 1, f"expected a fresh segment, {before} -> {after}"


def test_persistence_without_a_raw_dir_is_silent(tmp_path: Path) -> None:
    """Missing targets degrade to no-op rather than raising into the sampling loop."""
    store = ReadingStore(raw_dir=None, db_path=None)
    store.record(_reading(1.0))
    assert store.flush(force=True) == 0
    assert store.raw_writes == 0


def test_unwritable_raw_dir_does_not_raise(tmp_path: Path) -> None:
    """A full or read-only disk must not take a sampling loop down."""
    blocker = tmp_path / "raw"
    blocker.write_text("not a directory", encoding="utf-8")
    store = ReadingStore(raw_dir=blocker, db_path=tmp_path / "instrument.duckdb")
    store.record(_reading(1.0))
    store.flush(force=True)
    assert store.raw_writes == 0


# ════════════════════════════════════════════════════════════════
# Downsampled tier
# ════════════════════════════════════════════════════════════════


def test_history_survives_the_store_instance(tmp_path: Path) -> None:
    """The whole point: physical history must outlive the process that observed it."""
    db_path = tmp_path / "instrument.duckdb"
    first = ReadingStore(raw_dir=tmp_path / "raw", db_path=db_path)
    for index in range(3):
        first.record(_reading(float(index * 10), sequence=index, at=float(index)))
    first.flush(force=True)
    first.close()

    # A brand new store, as a later process would build.
    second = ReadingStore(raw_dir=tmp_path / "raw", db_path=db_path)
    history = second.history("dev", "level")
    assert len(history) == 1
    assert history[0]["min_value"] == 0.0
    assert history[0]["max_value"] == 20.0
    assert history[0]["samples"] == 3


def test_history_is_ordered_oldest_first(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for window_index in range(3):
        store.record(
            _reading(float(window_index), sequence=window_index, at=float(window_index))
        )
        store.flush(force=True)
    history = store.history("dev", "level")
    ends = [row["ended_at"] for row in history]
    assert ends == sorted(ends)


def test_history_is_limited(tmp_path: Path) -> None:
    """Disclosure must stay bounded; a long bench would otherwise be unaffordable."""
    store = _store(tmp_path)
    for index in range(20):
        store.record(_reading(float(index), sequence=index, at=float(index)))
        store.flush(force=True)
    assert len(store.history("dev", "level", limit=5)) == 5


def test_history_is_scoped_to_the_channel(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record(_reading(1.0, channel="level"))
    store.record(_reading(2.0, channel="other"))
    store.flush(force=True)
    assert len(store.history("dev", "level")) == 1
    assert len(store.history("dev", "other")) == 1
    assert store.history("dev", "absent") == ()


def test_history_of_a_missing_database_is_empty(tmp_path: Path) -> None:
    store = ReadingStore(raw_dir=None, db_path=tmp_path / "never_written.duckdb")
    assert store.history("dev", "level") == ()


# ════════════════════════════════════════════════════════════════
# Flush scheduling
# ════════════════════════════════════════════════════════════════


def test_flush_waits_for_the_downsample_interval(tmp_path: Path) -> None:
    """Writing per sample would defeat downsampling and hammer the disk."""
    store = _store(tmp_path, downsample_interval_s=60.0)
    store.record(_reading(1.0, at=100.0))
    assert store.flush(now=100.5) == 0
    assert store.pending_channels == 1
    assert store.flush(now=200.0) == 1
    assert store.pending_channels == 0


def test_due_for_flush_reports_the_interval(tmp_path: Path) -> None:
    store = _store(tmp_path, downsample_interval_s=10.0)
    assert store.due_for_flush() is False
    store.record(_reading(1.0, at=100.0))
    assert store.due_for_flush(now=105.0) is False
    assert store.due_for_flush(now=115.0) is True


def test_close_flushes_the_final_interval(tmp_path: Path) -> None:
    """Losing the last interval of a long run loses exactly what somebody wanted."""
    store = _store(tmp_path, downsample_interval_s=3600.0)
    store.record(_reading(42.0, at=1.0))
    store.close()
    assert len(store.history("dev", "level")) == 1


def test_close_never_raises(tmp_path: Path) -> None:
    """Teardown must not mask the failure that caused the shutdown."""
    store = _store(tmp_path)
    store.record(_reading(1.0))
    store._db_path = tmp_path / "nested" / "deep" / "x.duckdb"  # noqa: SLF001
    store.close()


# ════════════════════════════════════════════════════════════════
# Registry integration
# ════════════════════════════════════════════════════════════════


def test_registry_exposes_no_store_when_persistence_is_off() -> None:
    registry = HardwareRegistry(
        HardwareSettings(enabled=True, persist_readings=False),
        providers=[_StaticProvider(_context())],
    )
    registry.load()
    assert registry.reading_store is None
    assert registry.channel_history("dev", "level") == ()


def test_registry_rebuilds_the_store_when_persistence_is_rebound(tmp_path: Path) -> None:
    """The raw directory is session scoped, and no session exists at construction."""
    registry = HardwareRegistry(
        HardwareSettings(enabled=True, instrument_db_path=str(tmp_path / "i.duckdb")),
        providers=[_StaticProvider(_context())],
    )
    registry.load()
    first = registry.reading_store
    registry.bind_persistence(readings_dir=tmp_path / "raw", session_id="sess")
    assert registry.reading_store is not first


@pytest.mark.asyncio
async def test_sampling_persists_through_the_registry(tmp_path: Path) -> None:
    """End to end: a running sampler leaves durable history behind."""
    registry = HardwareRegistry(
        HardwareSettings(
            enabled=True,
            stream_ring_capacity=64,
            instrument_db_path=str(tmp_path / "i.duckdb"),
            downsample_interval_s=0.05,
        ),
        providers=[_StaticProvider(_context(sample_rate_hz=100.0))],
    )
    registry.load()
    registry.bind_persistence(readings_dir=tmp_path / "raw", session_id="sess")

    await registry.start_streams()
    import asyncio

    await asyncio.sleep(0.2)
    await registry.close_all()

    assert registry.reading_store.raw_writes > 0
    assert (tmp_path / "raw" / "dev.level.0000.ndjson").exists()
    assert len(registry.channel_history("dev", "level")) >= 1


@pytest.mark.asyncio
async def test_read_tool_discloses_stored_windows(tmp_path: Path) -> None:
    """The only path by which an earlier run's behaviour reaches a later decision."""
    from leapflow.hardware.tools import HardwareTools

    registry = HardwareRegistry(
        HardwareSettings(
            enabled=True,
            instrument_db_path=str(tmp_path / "i.duckdb"),
            downsample_interval_s=0.05,
        ),
        providers=[_StaticProvider(_context(sample_rate_hz=100.0))],
    )
    registry.load()
    registry.bind_persistence(readings_dir=tmp_path / "raw", session_id="sess")
    await registry.start_streams()
    import asyncio

    await asyncio.sleep(0.2)
    await registry.stop_streams()
    registry.reading_store.close()

    tools = HardwareTools(registry, session_id="sess")
    result = await tools.hw_read(device_id="dev", channel_id="level")
    assert result["ok"] is True
    assert result["stored_windows"]
    # Windows, never the raw series: the raw tier is evidence for a human, not context.
    assert "readings" not in str(result["stored_windows"])
    await registry.close_all()


@pytest.mark.asyncio
async def test_a_failing_store_does_not_stop_sampling(tmp_path: Path) -> None:
    """Losing observability is bad; taking the loop down is worse."""
    registry = HardwareRegistry(
        HardwareSettings(enabled=True, stream_ring_capacity=64),
        providers=[_StaticProvider(_context(sample_rate_hz=100.0))],
    )
    registry.load()

    class _BrokenStore:
        def record(self, reading: Any, *, dropped: int = 0) -> None:
            raise RuntimeError("disk is full")

        def flush(self, **_: Any) -> int:
            raise RuntimeError("disk is full")

        def close(self) -> None:
            raise RuntimeError("disk is full")

    registry._reading_store = _BrokenStore()  # noqa: SLF001
    registry._stream_sources = None  # noqa: SLF001
    from leapflow.hardware.stream import build_stream_sources

    registry._stream_sources = build_stream_sources(  # noqa: SLF001
        registry, ring_capacity=64, reading_store=_BrokenStore()
    )
    source = registry.stream_sources()[0]
    await source.start(lambda signal: None)
    import asyncio

    await asyncio.sleep(0.1)
    await source.stop()
    # Sampling kept going even though every persistence call raised.
    assert len(source.ring) >= 2


def test_persistence_config_keys_are_discoverable() -> None:
    """A durable setting that only exists in YAML is not a supported surface."""
    from leapflow.config import get_settings
    from leapflow.config_service import ConfigService

    service = ConfigService(get_settings())
    for key in (
        "hardware.persist_readings",
        "hardware.downsample_interval_s",
        "hardware.raw_retention_days",
    ):
        view = service.describe(key)
        assert view.description
        assert view.hot_reload == "restart-required"


# ════════════════════════════════════════════════════════════════
# Clock migration, drain/write split, failure accounting
# ════════════════════════════════════════════════════════════════


def test_history_excludes_rows_written_before_the_clock_was_fixed(tmp_path: Path) -> None:
    """Version-0 rows hold monotonic instants and must not enter a series.

    They are not merely old: a per-boot counter is not comparable to a wall-clock one,
    nor to another boot's. Blending them yields a chart that is wrong in a way nobody
    can see, and an ``ORDER BY ended_at DESC`` that returns the oldest row first.
    """
    import duckdb

    store = _store(tmp_path)
    store.record(_reading(42.0, at=1.0))
    assert store.flush(force=True) == 1

    db = tmp_path / "instrument.duckdb"
    connection = duckdb.connect(str(db))
    try:
        # A row as the previous implementation would have written it: monotonic
        # instants, and no version column value.
        connection.execute(
            "INSERT INTO reading_windows (device_id, channel_id, quantity, unit, "
            "started_at, ended_at, samples, dropped, min_value, max_value, mean_value, "
            "last_value, quality_worst, schema_version) "
            "VALUES ('dev', 'level', 'generic.level', 'unit', 900000.0, 900060.0, "
            "10, 0, 1.0, 2.0, 1.5, '2.0', 'ok', 0)"
        )
    finally:
        connection.close()

    rows = store.history("dev", "level", limit=50)
    assert len(rows) == 1, "the pre-fix row must not be returned"
    assert rows[0]["started_at"] == _WALL_EPOCH + 1.0


def test_drained_batches_are_written_by_the_caller(tmp_path: Path) -> None:
    """The split is a contract: drain detaches, write persists.

    Draining without writing loses the batch, so the two must be exercised as a pair
    exactly as the sampling loop uses them.
    """
    store = _store(tmp_path)
    for index in range(3):
        store.record(_reading(float(index), sequence=index, at=float(index)))

    batches = store.drain(force=True)
    assert len(batches) == 1
    assert store.pending_channels == 0, "drain must detach, not copy"
    assert store.drain(force=True) == (), "a second drain has nothing left"

    assert store.write_batches(batches) == 1
    assert store.windows_written == 1
    assert len(store.history("dev", "level")) == 1


def test_write_batches_is_a_no_op_for_nothing(tmp_path: Path) -> None:
    """The sampling loop calls this whenever a window closes, including empty ones."""
    store = _store(tmp_path)
    assert store.write_batches(()) == 0
    assert store.write_failures == 0


def test_failed_writes_are_counted_not_only_logged(tmp_path: Path) -> None:
    """A locked database is the one storage fault that leaves no trace in the data.

    Without a count, ``windows_written`` is a numerator with no denominator and an
    outage is indistinguishable from an idle bench.
    """
    # A directory where the database file belongs: opening it cannot succeed.
    blocked = tmp_path / "db"
    blocked.mkdir(parents=True, exist_ok=True)
    (blocked / "instrument.duckdb").mkdir()
    store = ReadingStore(
        raw_dir=tmp_path / "raw",
        db_path=blocked / "instrument.duckdb",
        downsample_interval_s=1.0,
    )
    store.record(_reading(1.0, at=1.0))
    assert store.flush(force=True) == 0
    assert store.write_failures == 1
    # Raw evidence still landed: losing history must not also lose the samples.
    assert store.raw_writes == 1


def test_window_boundaries_use_the_monotonic_clock(tmp_path: Path) -> None:
    """"Has an interval elapsed" is an interval question.

    Wall-clock can step backwards mid-window (NTP, suspend), which would either close
    a window early or never close it at all.
    """
    store = _store(tmp_path, downsample_interval_s=10.0)
    store.record(_reading(1.0, at=100.0))
    assert store.due_for_flush(now=105.0) is False
    assert store.due_for_flush(now=110.0) is True


def test_a_table_written_before_versioning_is_migrated_not_broken(tmp_path: Path) -> None:
    """The real migration path: a table that genuinely lacks the version column.

    The earlier test inserts a version-0 row into a table that already has the column,
    which exercises the filter but not the ``ALTER TABLE``. If that statement were
    unsupported its exception would be swallowed and every subsequent insert would fail
    against a missing column -- windows lost, with only a counter to show it. So the
    migration is asserted end to end: legacy rows survive on disk, land at version 0,
    stay out of queries, and new rows still write.
    """
    import duckdb

    db = tmp_path / "db" / "instrument.duckdb"
    db.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(db))
    try:
        connection.execute(
            "CREATE TABLE reading_windows ("
            "device_id VARCHAR NOT NULL, channel_id VARCHAR NOT NULL, quantity VARCHAR, "
            "unit VARCHAR, started_at DOUBLE NOT NULL, ended_at DOUBLE NOT NULL, "
            "samples BIGINT NOT NULL, dropped BIGINT NOT NULL, min_value DOUBLE, "
            "max_value DOUBLE, mean_value DOUBLE, last_value VARCHAR, quality_worst VARCHAR)"
        )
        # Monotonic instants, exactly as the pre-fix implementation wrote them.
        connection.execute(
            "INSERT INTO reading_windows VALUES "
            "('dev','level','generic.level','unit',900000.0,900060.0,10,0,1.0,2.0,1.5,'2.0','ok')"
        )
    finally:
        connection.close()

    store = ReadingStore(raw_dir=tmp_path / "raw", db_path=db, downsample_interval_s=1.0)
    store.record(_reading(7.0, at=1.0))
    assert store.flush(force=True) == 1, "the new row must write against the migrated table"
    assert store.write_failures == 0, "a swallowed ALTER would surface here as a lost window"

    rows = store.history("dev", "level", limit=50)
    assert len(rows) == 1
    assert rows[0]["started_at"] == _WALL_EPOCH + 1.0

    connection = duckdb.connect(str(db), read_only=True)
    try:
        versions = connection.execute(
            "SELECT schema_version, COUNT(*) FROM reading_windows GROUP BY 1 ORDER BY 1"
        ).fetchall()
    finally:
        connection.close()
    # Retention removes the legacy row rather than relabelling it. Excluding it from
    # queries left it on disk forever, and its monotonic instants can never satisfy a
    # wall-clock cutoff, so age alone would never have collected it.
    assert versions == [(1, 1)]


# ════════════════════════════════════════════════════════════════
# History retention and the index it needs
# ════════════════════════════════════════════════════════════════


def test_history_past_the_horizon_is_pruned(tmp_path: Path) -> None:
    """Nothing else was ever going to delete from this table.

    Eight channels on the default interval is roughly 11,500 rows a day, and before
    retention existed the only bound was the disk. Unbounded history is also unbounded
    exposure: whatever exists is what a profile backup carries away.
    """
    store = _store(tmp_path, history_ttl_s=60.0)
    store.record(_reading(1.0, at=0.0))  # an hour old
    store.flush(force=True)
    assert store.history("dev", "level") == (), "a window past the horizon must not survive"
    assert store.rows_pruned == 1


def test_the_pruned_count_is_the_number_of_rows_removed(tmp_path: Path) -> None:
    """``>= 1`` was too weak, and the implementation it accepted was wrong.

    ``RETURNING 1`` yields one row per deleted window. Reading the first row's first
    column instead of the row count reported "1" for every prune regardless of how many
    windows it removed -- and a lower-bound assertion is satisfied by exactly that. The
    count feeds the board's storage panel, so an operator would have been told retention
    removed one row when it removed hundreds.
    """
    store = _store(tmp_path, history_ttl_s=0.0)  # retention off while history builds up
    for index in range(5):
        store.record(_reading(float(index), sequence=index, at=0.0))
        store.flush(force=True)
    assert len(store.history("dev", "level")) == 5

    pruning = _store(tmp_path, history_ttl_s=60.0)
    pruning.record(_reading(99.0, sequence=99, at=0.0))
    pruning.flush(force=True)
    assert pruning.history("dev", "level") == ()
    assert pruning.rows_pruned == 6, (
        "five accumulated windows plus the one just written are all past the horizon"
    )


def test_fresh_history_is_untouched_by_retention(tmp_path: Path) -> None:
    """Retention must remove only what it was asked to.

    The horizon is half an hour, not a minute: ``_WALL_EPOCH`` is fixed at import and
    the full suite runs for several minutes, so a tight horizon would let retention
    collect this fixture and fail a test that has nothing to do with age.
    """
    store = _store(tmp_path, history_ttl_s=1800.0)
    store.record(_reading(7.0, at=3600.0))  # now
    store.flush(force=True)
    assert len(store.history("dev", "level")) == 1


def test_retention_is_rate_limited_off_the_write_path(tmp_path: Path) -> None:
    """A delete per flush would cost more than the insert it follows.

    Bounded data does not need minute-level precision, so the second flush inside the
    interval leaves the row alone even though it is past the horizon.
    """
    store = _store(tmp_path, history_ttl_s=1800.0)
    store.record(_reading(1.0, at=3600.0))
    store.flush(force=True)  # first flush runs the prune
    pruned_after_first = store.rows_pruned

    store.record(_reading(2.0, at=0.0))  # an hour old, past the horizon
    store.flush(force=True)
    assert store.rows_pruned == pruned_after_first, "the prune must not run twice in one hour"
    assert len(store.history("dev", "level")) == 2


def test_zero_retention_keeps_everything(tmp_path: Path) -> None:
    """An operator who wants unbounded history must be able to say so."""
    store = _store(tmp_path, history_ttl_s=0.0)
    store.record(_reading(1.0, at=0.0))
    store.flush(force=True)
    assert len(store.history("dev", "level")) == 1
    assert store.rows_pruned == 0


def test_the_history_table_is_indexed_for_its_only_query_shape(tmp_path: Path) -> None:
    """Without an index every history call scans the table.

    Created after the version column is added, and that order matters: indexing a
    column introduced by the same migration fails with a binder error on an older
    database, and the failure shows up as a write that never lands.
    """
    import duckdb

    store = _store(tmp_path)
    store.record(_reading(1.0, at=3600.0))
    store.flush(force=True)

    connection = duckdb.connect(str(tmp_path / "instrument.duckdb"), read_only=True)
    try:
        rows = connection.execute(
            "SELECT index_name, sql FROM duckdb_indexes() WHERE table_name = 'reading_windows'"
        ).fetchall()
    finally:
        connection.close()
    names = {row[0] for row in rows}
    assert "idx_reading_windows_channel" in names, f"no channel index, found {names}"
