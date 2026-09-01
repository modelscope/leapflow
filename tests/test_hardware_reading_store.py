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
# Adaptive window policy
# ════════════════════════════════════════════════════════════════


def test_default_policy_holds_the_base_interval_until_an_alert() -> None:
    """With nothing happening the window is the coarse steady-state interval."""
    from leapflow.hardware.reading_store import DefaultAdaptiveWindowPolicy

    policy = DefaultAdaptiveWindowPolicy(60.0)
    assert policy.interval_s(now=0.0) == 60.0
    assert policy.interval_s(now=10_000.0) == 60.0


def test_default_policy_tightens_to_the_floor_on_an_alert() -> None:
    """An alert shrinks the window to min(10, base/4) so the excursion keeps its shape."""
    from leapflow.hardware.reading_store import DefaultAdaptiveWindowPolicy

    policy = DefaultAdaptiveWindowPolicy(60.0)
    policy.note_alert(now=100.0)
    # base/4 == 15, floored at 10.
    assert policy.interval_s(now=100.0) == 10.0


def test_default_policy_tighten_is_derived_from_base_not_current() -> None:
    """Repeated alerts must not compound the interval down toward zero."""
    from leapflow.hardware.reading_store import DefaultAdaptiveWindowPolicy

    # base/4 == 5, below the 10s floor, so the tightened value is 5 -- and stays 5
    # no matter how many alerts arrive.
    policy = DefaultAdaptiveWindowPolicy(20.0)
    policy.note_alert(now=0.0)
    assert policy.interval_s(now=0.0) == 5.0
    policy.note_alert(now=1.0)
    policy.note_alert(now=2.0)
    assert policy.interval_s(now=2.0) == 5.0


def test_default_policy_relaxes_after_five_quiet_minutes() -> None:
    """Steady state returns only once the bench has been quiet for the recovery window.

    Recovery is measured from the last alert, not the first, so a run of alerts keeps
    the fine window open rather than snapping back to coarse mid-event.
    """
    from leapflow.hardware.reading_store import DefaultAdaptiveWindowPolicy

    policy = DefaultAdaptiveWindowPolicy(60.0, recovery_s=300.0)
    policy.note_alert(now=1000.0)
    assert policy.interval_s(now=1000.0) == 10.0
    assert policy.interval_s(now=1000.0 + 299.0) == 10.0, "still tight within recovery"
    # A second alert extends the fine window from its own timestamp.
    policy.note_alert(now=1000.0 + 250.0)
    assert policy.interval_s(now=1000.0 + 250.0 + 299.0) == 10.0
    assert policy.interval_s(now=1000.0 + 250.0 + 300.0) == 60.0, "relaxed after quiet"


def test_store_window_tightens_after_an_alert(tmp_path: Path) -> None:
    """The store closes windows on the policy's interval, not a fixed constant.

    Same buffered gap, two verdicts: coarse when steady (the window is still open),
    tight after an alert (the window has closed and drains).
    """
    store = _store(tmp_path, downsample_interval_s=60.0)
    store.record(_reading(1.0, at=0.0))
    # 20s elapsed: below the 60s steady window, so nothing is due yet.
    assert store.due_for_flush(now=20.0) is False
    assert store.drain(now=20.0) == ()

    # An alert tightens the window to 10s; the same 20s gap is now overdue.
    store.note_alert(now=20.0)
    assert store.due_for_flush(now=20.0) is True
    batches = store.drain(now=20.0)
    assert len(batches) == 1


def test_store_window_relaxes_back_to_base_when_quiet(tmp_path: Path) -> None:
    """Once past the recovery horizon the coarse steady-state window returns."""
    store = _store(tmp_path, downsample_interval_s=60.0)
    store.note_alert(now=0.0)
    store.record(_reading(1.0, at=1000.0))
    # Long after recovery: a 20s gap is once again below the restored 60s window.
    assert store.due_for_flush(now=1020.0) is False


def test_alert_events_tighten_the_store_window_through_the_registry(tmp_path: Path) -> None:
    """An alert-severity event routed through record_event reaches the window policy.

    This is the wiring the sampling loop relies on: the event sink is the registry's
    record_event, and an alert kind there must tighten the store's window without the
    store ever seeing the event object.
    """
    from types import SimpleNamespace

    store = _store(tmp_path, downsample_interval_s=60.0)
    registry = HardwareRegistry.__new__(HardwareRegistry)
    registry._recent_events = []  # type: ignore[attr-defined]
    registry._reading_store = store  # type: ignore[attr-defined]

    # An informational event leaves the window coarse.
    registry.record_event(SimpleNamespace(kind="settled"))
    store.record(_reading(1.0, at=0.0))
    assert store.due_for_flush(now=20.0) is False

    # An alert-severity event tightens it.
    registry.record_event(SimpleNamespace(kind="threshold_exceeded"))
    assert store.due_for_flush(now=20.0) is True


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


def test_history_database_is_indexed_as_sensitive_and_non_syncable(tmp_path: Path) -> None:
    """The durable tier inherits the raw tier's privacy posture, not its lifetime.

    instrument.duckdb is profile-scoped and durable -- it carries no TTL -- but it still
    holds physical series that can be a trade secret or carry sample information. It is
    registered with CacheManager as sensitive and non-syncable so a profile backup
    honours the same non-syncable posture the raw tier already has.
    """
    from leapflow.hardware.reading_store import HISTORY_CATEGORY

    layout = build_layout(tmp_path / "data")
    profile_layout = layout.ensure(profile_id="default")
    manager = CacheManager(profile_layout.cache, profile_id="default")
    db_path = tmp_path / "instrument.duckdb"
    store = ReadingStore(
        raw_dir=tmp_path / "raw",
        db_path=db_path,
        cache_manager=manager,
        session_id="sess",
    )
    store.record(_reading(1.0, at=3600.0))
    store.flush(force=True)

    entries = [e for e in manager.list_entries() if e.category == HISTORY_CATEGORY]
    assert len(entries) == 1, "the history database must be indexed exactly once"
    entry = entries[0]
    assert entry.path == db_path.resolve()
    assert entry.sensitive is True
    assert entry.syncable is False
    assert entry.owner_component == "hardware"
    assert entry.scope == CacheScope.PROFILE.value
    assert entry.expires_at is None, "the durable tier is bounded by prune, not TTL"


def test_history_database_registration_is_idempotent(tmp_path: Path) -> None:
    """Keyed by path and flag-guarded: many drains, one index entry."""
    from leapflow.hardware.reading_store import HISTORY_CATEGORY

    layout = build_layout(tmp_path / "data")
    profile_layout = layout.ensure(profile_id="default")
    manager = CacheManager(profile_layout.cache, profile_id="default")
    store = ReadingStore(
        raw_dir=tmp_path / "raw",
        db_path=tmp_path / "instrument.duckdb",
        cache_manager=manager,
        session_id="sess",
    )
    for index in range(1, 8):
        store.record(_reading(float(index), sequence=index, at=3600.0))
        store.flush(force=True)

    entries = [e for e in manager.list_entries() if e.category == HISTORY_CATEGORY]
    assert len(entries) == 1


def test_history_database_sensitivity_is_configurable(tmp_path: Path) -> None:
    """Opting out lets a bench known to produce no sensitive series sync normally.

    ``hardware.reading_store_sensitive=False`` flips the registration to
    non-sensitive/syncable, so an operator who is sure the data carries nothing
    private can include it in a backup.
    """
    from leapflow.hardware.reading_store import HISTORY_CATEGORY

    layout = build_layout(tmp_path / "data")
    profile_layout = layout.ensure(profile_id="default")
    manager = CacheManager(profile_layout.cache, profile_id="default")
    store = ReadingStore(
        raw_dir=tmp_path / "raw",
        db_path=tmp_path / "instrument.duckdb",
        cache_manager=manager,
        session_id="sess",
        reading_store_sensitive=False,
    )
    store.record(_reading(1.0, at=3600.0))
    store.flush(force=True)

    entries = [e for e in manager.list_entries() if e.category == HISTORY_CATEGORY]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.sensitive is False
    assert entry.syncable is True


def test_history_database_is_not_indexed_without_a_cache_manager(tmp_path: Path) -> None:
    """No cache manager, no registration -- and no crash on the write path."""
    store = ReadingStore(
        raw_dir=tmp_path / "raw",
        db_path=tmp_path / "instrument.duckdb",
    )
    store.record(_reading(1.0, at=3600.0))
    written = store.flush(force=True)
    assert written == 1
    store.close()


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

    # Close the store so its ConnectionHolder releases the database before
    # opening a separate read-only verification connection.
    store.close()
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
    # Close the store so its ConnectionHolder releases the database before
    # opening a separate read-only verification connection.
    store.close()

    connection = duckdb.connect(str(tmp_path / "instrument.duckdb"), read_only=True)
    try:
        rows = connection.execute(
            "SELECT index_name, sql FROM duckdb_indexes() WHERE table_name = 'reading_windows'"
        ).fetchall()
    finally:
        connection.close()
    names = {row[0] for row in rows}
    assert "idx_reading_windows_channel" in names, f"no channel index, found {names}"


# ════════════════════════════════════════════════════════════════
# CBAG G16: wall-clock persistence and digest payload clock
# ════════════════════════════════════════════════════════════════


def test_persisted_window_timestamps_are_wall_clock(tmp_path: Path) -> None:
    """Every persisted ``started_at`` / ``ended_at`` must be a wall-clock epoch.

    CBAG G16: the store previously used ``Reading.monotonic_at`` for window
    boundaries, which resets on reboot. A genuine wall-clock instant built from
    ``time.time()`` is always later than yesterday; a monotonic value (uptime
    since boot) is typically far smaller and predates it.

    If this assertion fails, persisted instants are not real wall-clock epochs
    and every cross-session query is silently broken.
    """
    import duckdb

    store = _store(tmp_path)
    # Use explicit wall-clock observed_at via _reading's default (which adds
    # _WALL_EPOCH + at). _WALL_EPOCH is time.time()-3600, so all observed_at
    # values are genuine wall-clock.
    for index in range(3):
        store.record(_reading(float(index * 10), sequence=index, at=float(index)))
    store.flush(force=True)
    # Close the store so its ConnectionHolder releases the database before
    # opening a separate read-only verification connection.
    store.close()

    db = tmp_path / "instrument.duckdb"
    connection = duckdb.connect(str(db), read_only=True)
    try:
        rows = connection.execute(
            "SELECT started_at, ended_at, schema_version FROM reading_windows"
        ).fetchall()
    finally:
        connection.close()

    assert len(rows) >= 1
    yesterday = time.time() - 86400.0
    for started_at, ended_at, schema_version in rows:
        # G16: timestamps must be wall-clock (later than yesterday).
        assert started_at > yesterday, (
            f"started_at={started_at} predates yesterday, so it looks like a "
            "monotonic value rather than wall-clock; G16 regression: monotonic "
            "instants silently break cross-session ordering"
        )
        assert ended_at > yesterday, (
            f"ended_at={ended_at} predates yesterday, so it looks like a "
            "monotonic value rather than wall-clock; G16 regression"
        )
        # G16: schema version must be current so the row is queryable.
        from leapflow.hardware.reading_store import SCHEMA_VERSION
        assert schema_version == SCHEMA_VERSION, (
            f"schema_version={schema_version}, expected {SCHEMA_VERSION}; "
            "a version-0 row is excluded from queries and invisible"
        )


def test_digest_payload_declares_wall_clock(tmp_path: Path) -> None:
    """The observability payload must state ``clock=='wall'``.

    CBAG G16: a chart drawn from monotonic instants looks correct while being
    wrong by decades. The ``clock`` field lets the renderer verify it is
    displaying the right timebase.
    """
    from leapflow.hardware.observability import WALL_CLOCK, build_digest
    from leapflow.hardware.observability.series import SERIES_SCHEMA_VERSION

    # Minimal registry fake with one readable history window.
    from types import SimpleNamespace

    window = {
        "ended_at": _WALL_EPOCH + 60.0,
        "mean_value": 42.0,
        "min_value": 40.0,
        "max_value": 44.0,
        "samples": 10,
        "dropped": 0,
        "quality_worst": "ok",
    }
    channel_fake = SimpleNamespace(
        channel_id="level",
        quantity="generic.level",
        unit="C",
        sample_rate_hz=10.0,
        is_readable=True,
        is_writable=False,
        envelope=SimpleNamespace(
            declared=True, min_value=0.0, max_value=100.0, quantization=0.5, notes="",
        ),
    )
    context_fake = SimpleNamespace(
        device_id="dev",
        display_name="Dev",
        location="lab",
        halt_supported=True,
        channels=(channel_fake,),
        writable_channels=(),
        transport=SimpleNamespace(kind="mock"),
        provenance=SimpleNamespace(verified_by="tester"),
    )

    class _FakeRegistry:
        reading_store = None
        outcome_recorder = None

        def contexts(self) -> tuple:
            return (context_fake,)

        def opened_devices(self) -> tuple:
            return ()

        def channel_history(self, d: str, c: str, *, limit: int = 200) -> list:
            return [window]

        def recent_events(self, device_id: str = "", limit: int = 10) -> tuple:
            return ()

        def stream_sources(self) -> tuple:
            return ()

    payload = build_digest(_FakeRegistry()).to_payload()
    assert payload["clock"] == WALL_CLOCK, (
        f"payload clock={payload['clock']!r}, expected {WALL_CLOCK!r}; "
        "G16 regression: the chart must know it is drawing wall-clock"
    )
    assert payload["schema_version"] == SERIES_SCHEMA_VERSION
    # Every point's x must be a wall-clock epoch (later than yesterday).
    yesterday = time.time() - 86400.0
    for series in payload["series"]:
        for point in series["points"]:
            assert point["x"] > yesterday, (
                f"series point x={point['x']} predates yesterday, not wall-clock; "
                "G16 regression"
            )


# ════════════════════════════════════════════════════════════════
# Calibration store (IC-7)
# ════════════════════════════════════════════════════════════════


def _calibration_store(tmp_path: Path, **kwargs: Any):
    from leapflow.hardware.calibration_store import CalibrationStore

    return CalibrationStore(db_path=tmp_path / "instrument.duckdb", **kwargs)


def _calibration_record(
    *,
    device_id: str = "dev",
    procedure_id: str = "gain",
    recorded_at: float = 1000.0,
    parameters: dict | None = None,
    matrix: Any = None,
    pose: dict | None = None,
    notes: str = "",
):
    from leapflow.hardware.calibration_store import CalibrationRecord

    return CalibrationRecord(
        device_id=device_id,
        procedure_id=procedure_id,
        recorded_at=recorded_at,
        parameters=parameters if parameters is not None else {"slope": 1.5},
        matrix=matrix,
        pose=pose,
        notes=notes,
    )


def test_calibration_result_round_trips_through_the_store(tmp_path: Path) -> None:
    """Matrix, parameters and pose survive the JSON columns they are stored in.

    A calibration is the transform a later reading is interpreted through; storing it
    lossily would silently change what every subsequent number means.
    """
    store = _calibration_store(tmp_path)
    record = _calibration_record(
        parameters={"slope": 1.5, "offset": -0.2},
        matrix=[[1.0, 0.0], [0.0, 1.0]],
        pose={"x": 1.0, "y": 2.0, "theta": 0.5},
        notes="bench A",
    )
    assert store.record(record) is True

    latest = store.latest("dev", "gain")
    assert latest is not None
    assert latest.parameters == {"slope": 1.5, "offset": -0.2}
    assert latest.matrix == [[1.0, 0.0], [0.0, 1.0]]
    assert latest.pose == {"x": 1.0, "y": 2.0, "theta": 0.5}
    assert latest.notes == "bench A"
    store.close()


def test_calibration_latest_returns_the_most_recent_version(tmp_path: Path) -> None:
    """Nothing is overwritten: a re-run lands a new row and latest is the newest."""
    store = _calibration_store(tmp_path)
    store.record(_calibration_record(recorded_at=1000.0, parameters={"slope": 1.0}))
    store.record(_calibration_record(recorded_at=2000.0, parameters={"slope": 2.0}))

    latest = store.latest("dev", "gain")
    assert latest is not None
    assert latest.recorded_at == 2000.0
    assert latest.parameters == {"slope": 2.0}
    # Both versions remain: the old rows are the audit trail of how the frame moved.
    assert len(store.history("dev", "gain")) == 2
    store.close()


def test_calibration_latest_time_is_none_until_a_calibration_exists(tmp_path: Path) -> None:
    """hw_describe reads this: a device never calibrated reports no instant, not zero."""
    store = _calibration_store(tmp_path)
    assert store.latest_time("dev") is None
    store.record(_calibration_record(recorded_at=1234.0))
    assert store.latest_time("dev") == 1234.0
    store.close()


def test_calibration_is_keyed_and_idempotent_on_device_procedure_instant(tmp_path: Path) -> None:
    """Re-recording the same (device, procedure, ts) replaces the row, never duplicates."""
    store = _calibration_store(tmp_path)
    store.record(_calibration_record(recorded_at=1000.0, parameters={"slope": 1.0}))
    store.record(_calibration_record(recorded_at=1000.0, parameters={"slope": 9.9}))

    history = store.history("dev", "gain")
    assert len(history) == 1
    assert history[0].parameters == {"slope": 9.9}
    store.close()


def test_calibration_latest_spans_procedures_for_a_device(tmp_path: Path) -> None:
    """With no procedure named, latest answers 'when was this device last calibrated at all'."""
    store = _calibration_store(tmp_path)
    store.record(_calibration_record(procedure_id="gain", recorded_at=1000.0))
    store.record(_calibration_record(procedure_id="offset", recorded_at=3000.0))
    store.record(_calibration_record(procedure_id="gain", recorded_at=2000.0))

    latest = store.latest("dev")
    assert latest is not None
    assert latest.procedure_id == "offset"
    assert latest.recorded_at == 3000.0
    # A procedure filter still narrows to that procedure's newest.
    assert store.latest("dev", "gain").recorded_at == 2000.0
    store.close()


def test_calibration_write_is_contained_without_a_holder() -> None:
    """No database configured: record reports failure, reads are empty, nothing raises."""
    from leapflow.hardware.calibration_store import CalibrationStore

    store = CalibrationStore()
    assert store.record(_calibration_record()) is False
    assert store.latest("dev") is None
    assert store.latest_time("dev") is None
    assert store.history("dev") == ()
    store.close()


def test_calibration_reads_empty_before_the_file_exists(tmp_path: Path) -> None:
    """Reading a not-yet-written store must not materialise an empty database."""
    store = _calibration_store(tmp_path)
    assert store.latest("dev") is None
    assert store.history("dev") == ()
    assert not (tmp_path / "instrument.duckdb").exists()
    store.close()


def test_calibration_database_is_indexed_as_sensitive_and_non_syncable(tmp_path: Path) -> None:
    """The calibration tier shares the file's sensitivity: a fixture geometry can be private."""
    from leapflow.hardware.calibration_store import CALIBRATION_CATEGORY, CalibrationStore

    layout = build_layout(tmp_path / "data")
    profile_layout = layout.ensure(profile_id="default")
    manager = CacheManager(profile_layout.cache, profile_id="default")
    db_path = tmp_path / "instrument.duckdb"
    store = CalibrationStore(db_path=db_path, cache_manager=manager)
    store.record(_calibration_record())

    entries = [e for e in manager.list_entries() if e.category == CALIBRATION_CATEGORY]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.path == db_path.resolve()
    assert entry.sensitive is True
    assert entry.syncable is False
    assert entry.owner_component == "hardware"
    assert entry.scope == CacheScope.PROFILE.value
    assert entry.expires_at is None
    store.close()


def test_registry_shares_one_instrument_connection_across_both_stores(tmp_path: Path) -> None:
    """A single process must not open two read-write connections to one DuckDB file."""
    registry = HardwareRegistry(
        HardwareSettings(enabled=True, instrument_db_path=str(tmp_path / "i.duckdb")),
        providers=[_StaticProvider(_context())],
    )
    registry.load()
    reading = registry.reading_store
    calibration = registry.calibration_store
    assert reading is not None and calibration is not None
    assert reading._holder is calibration._holder  # noqa: SLF001
    assert reading._holder is registry._instrument_conn  # noqa: SLF001


def test_registry_calibration_store_is_independent_of_reading_persistence(tmp_path: Path) -> None:
    """A bench can want durable calibration history without streaming sample history."""
    registry = HardwareRegistry(
        HardwareSettings(
            enabled=True,
            persist_readings=False,
            instrument_db_path=str(tmp_path / "i.duckdb"),
        ),
        providers=[_StaticProvider(_context())],
    )
    registry.load()
    assert registry.reading_store is None
    assert registry.calibration_store is not None


def test_registry_has_no_calibration_store_without_an_instrument_database() -> None:
    registry = HardwareRegistry(
        HardwareSettings(enabled=True),
        providers=[_StaticProvider(_context())],
    )
    registry.load()
    assert registry.calibration_store is None


@pytest.mark.asyncio
async def test_hw_describe_reports_the_last_calibration_time(tmp_path: Path) -> None:
    """A reference document states a calibration age so a reader can judge its currency."""
    from leapflow.hardware.tools import HardwareTools

    registry = HardwareRegistry(
        HardwareSettings(enabled=True, instrument_db_path=str(tmp_path / "i.duckdb")),
        providers=[_StaticProvider(_context())],
    )
    registry.load()
    tools = HardwareTools(registry, session_id="sess")

    # Absent before any calibration is recorded.
    before = await tools.hw_describe(device_id="dev")
    assert before["ok"] is True
    assert "last_calibrated_at" not in before

    registry.calibration_store.record(_calibration_record(recorded_at=4242.0))
    after = await tools.hw_describe(device_id="dev")
    assert after["last_calibrated_at"] == 4242.0
    await registry.close_all()


@pytest.mark.asyncio
async def test_calibration_survives_close_all_and_reopens_for_reads(tmp_path: Path) -> None:
    """close_all closes the shared holder last; a later read reopens it lazily."""
    registry = HardwareRegistry(
        HardwareSettings(enabled=True, instrument_db_path=str(tmp_path / "i.duckdb")),
        providers=[_StaticProvider(_context())],
    )
    registry.load()
    registry.calibration_store.record(_calibration_record(recorded_at=555.0))
    await registry.close_all()

    # The holder was closed during teardown; reading reopens it rather than failing.
    assert registry.calibration_store.latest_time("dev") == 555.0

