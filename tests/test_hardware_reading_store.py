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


def _reading(
    value: Any,
    *,
    sequence: int = 1,
    timestamp: float = 0.0,
    quality: str = Quality.OK.value,
    channel: str = "level",
) -> Reading:
    return Reading(
        device_id="dev",
        channel_id=channel,
        value=value,
        quantity="generic.level",
        unit="unit",
        timestamp=timestamp,
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
            _reading(10.0, sequence=1, timestamp=1.0),
            _reading(90.0, sequence=2, timestamp=2.0),
            _reading(20.0, sequence=3, timestamp=3.0),
        ]
    )
    assert window is not None
    assert window.min_value == 10.0
    assert window.max_value == 90.0
    assert window.mean_value == pytest.approx(40.0)
    assert window.samples == 3
    assert window.started_at == 1.0
    assert window.ended_at == 3.0


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
        store.record(_reading(float(index), sequence=index, timestamp=float(index)))
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
    assert names == ["dev.level.ndjson", "dev.other.ndjson"]


def test_raw_writes_append_across_flushes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record(_reading(1.0, sequence=1))
    store.flush(force=True)
    store.record(_reading(2.0, sequence=2))
    store.flush(force=True)
    path = tmp_path / "raw" / "dev.level.ndjson"
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


def test_raw_file_is_indexed_once_not_per_flush(tmp_path: Path) -> None:
    """Re-registering on every append would grow the index at sampling rate."""
    layout = build_layout(tmp_path / "data")
    profile_layout = layout.ensure(profile_id="default")
    manager = CacheManager(profile_layout.cache, profile_id="default")
    store = ReadingStore(
        raw_dir=tmp_path / "raw",
        db_path=tmp_path / "instrument.duckdb",
        cache_manager=manager,
        workspace_id="ws",
        session_id="sess",
    )
    for index in range(4):
        store.record(_reading(float(index), sequence=index))
        store.flush(force=True)
    entries = [e for e in manager.list_entries() if e.category == READINGS_CATEGORY]
    assert len(entries) == 1


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
        first.record(_reading(float(index * 10), sequence=index, timestamp=float(index)))
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
            _reading(float(window_index), sequence=window_index, timestamp=float(window_index))
        )
        store.flush(force=True)
    history = store.history("dev", "level")
    ends = [row["ended_at"] for row in history]
    assert ends == sorted(ends)


def test_history_is_limited(tmp_path: Path) -> None:
    """Disclosure must stay bounded; a long bench would otherwise be unaffordable."""
    store = _store(tmp_path)
    for index in range(20):
        store.record(_reading(float(index), sequence=index, timestamp=float(index)))
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
    store.record(_reading(1.0, timestamp=100.0))
    assert store.flush(now=100.5) == 0
    assert store.pending_channels == 1
    assert store.flush(now=200.0) == 1
    assert store.pending_channels == 0


def test_due_for_flush_reports_the_interval(tmp_path: Path) -> None:
    store = _store(tmp_path, downsample_interval_s=10.0)
    assert store.due_for_flush() is False
    store.record(_reading(1.0, timestamp=100.0))
    assert store.due_for_flush(now=105.0) is False
    assert store.due_for_flush(now=115.0) is True


def test_close_flushes_the_final_interval(tmp_path: Path) -> None:
    """Losing the last interval of a long run loses exactly what somebody wanted."""
    store = _store(tmp_path, downsample_interval_s=3600.0)
    store.record(_reading(42.0, timestamp=1.0))
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
    assert (tmp_path / "raw" / "dev.level.ndjson").exists()
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
