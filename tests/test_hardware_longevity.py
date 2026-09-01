"""Long-run persistence properties of :class:`ReadingStore`, in milliseconds.

A bench that runs for a shift or a week is where storage either stays bounded or
quietly fails: raw segments must roll rather than grow into one unopenable file,
the downsampled table must not accumulate one row per sample, retention must
actually delete, and history must survive the process that observed it. None of
that is exercised by the unit suite, which writes a handful of windows and stops.

The trick that makes a "7-day" scenario finish in-process is
:meth:`SimulatedTransport.advance_clock`: it moves both clocks a
:class:`~leapflow.hardware.transport.Reading` carries forward together, so
``observed_at`` sweeps across days of wall-clock while no real time passes. Every
test here drives time that way -- there is no ``time.sleep`` anywhere, and every
wall-clock assertion is written relative to ``time.time()`` rather than a fixed
epoch, so nothing decays into a failure as the calendar moves.

Where a store branch keys off the real wall clock rather than a clock the
transport controls -- retention compares window age against ``time.time()`` -- the
test reaches that branch through the store's own seams (a past transport base, a
fresh store whose rate-limit clock starts at zero), never by editing production.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import pytest

from leapflow.hardware.context import (
    Channel,
    ContextProvenance,
    Direction,
    Envelope,
    HardwareContext,
    TransportRef,
)
from leapflow.hardware.reading_store import (
    DEFAULT_RAW_SEGMENT_BYTES,
    SCHEMA_VERSION,
    ReadingStore,
)
from leapflow.hardware.transports.simulated import SimulatedTransport

_EIGHT_HOURS_S = 8 * 3600.0
_SEVEN_DAYS_S = 7 * 24 * 3600.0
_DEVICE = "dev"
_CHANNEL = "sensor"


# ════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════


def _context() -> HardwareContext:
    """A single readable channel -- all four scenarios only need to sample."""
    return HardwareContext(
        device_id=_DEVICE,
        halt_supported=True,
        transport=TransportRef(kind="simulated", config={}),
        channels=(
            Channel(
                channel_id=_CHANNEL,
                direction=Direction.READ.value,
                quantity="generic.sensor",
                unit="unit",
                sample_rate_hz=1.0,
                envelope=Envelope(declared=True, min_value=0.0, max_value=100.0),
            ),
        ),
        provenance=ContextProvenance(verified_by="longevity"),
    )


def _sine_source(period_s: float) -> dict[str, Any]:
    """A waveform config so successive samples actually differ within a window."""
    return {
        "waveforms": {
            _CHANNEL: {"kind": "sine", "offset": 20.0, "amplitude": 5.0, "period_s": period_s}
        }
    }


async def _open(config: dict[str, Any] | None = None) -> SimulatedTransport:
    transport = SimulatedTransport(config or {})
    await transport.open(_context())
    return transport


def _count_windows(db_path: Path) -> int:
    """Every persisted window, version filter ignored -- the raw table size."""
    import duckdb

    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        return int(connection.execute("SELECT COUNT(*) FROM reading_windows").fetchone()[0])
    finally:
        connection.close()


def _current_window_rows(db_path: Path) -> list[tuple[Any, ...]]:
    """``(started_at, ended_at, samples)`` for queryable (current-version) rows."""
    import duckdb

    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        return connection.execute(
            "SELECT started_at, ended_at, samples FROM reading_windows "
            f"WHERE schema_version >= {SCHEMA_VERSION} ORDER BY ended_at"
        ).fetchall()
    finally:
        connection.close()


# ════════════════════════════════════════════════════════════════
# 1. Raw segment rotation
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_raw_segments_roll_and_stay_bounded_over_a_long_run(tmp_path: Path) -> None:
    """A shift's worth of raw samples must roll into bounded segments, not one file.

    A single append-only file cannot be partly expired -- dropping last week's
    samples would mean deleting the file currently being written to -- so the store
    closes a segment once it reaches its byte cap and opens the next. Rotation is
    append-then-check: the write that tips a segment over the cap has already landed,
    so a finished segment overshoots by at most one record and never grows without
    bound. The real 32 MB default would need millions of samples to exercise, so the
    cap is scaled down; the invariant asserted is the one the default relies on.
    """
    cap = 4096
    store = ReadingStore(raw_dir=tmp_path / "raw", db_path=None, raw_segment_bytes=cap)
    assert cap < DEFAULT_RAW_SEGMENT_BYTES  # a scaled-down stand-in for the 32 MB default

    transport = await _open(_sine_source(period_s=_EIGHT_HOURS_S))
    # One reading per flush, so each raw append is a single line: that keeps the
    # by-one-record overshoot to exactly one line and mirrors a slow sampling loop
    # stepped forward eight hours in even strides.
    samples = 240
    step_s = _EIGHT_HOURS_S / samples
    for _ in range(samples):
        store.record(await transport.read(_CHANNEL))
        store.flush(force=True)
        transport.advance_clock(step_s)

    segments = sorted(
        (tmp_path / "raw").glob(f"{_DEVICE}.{_CHANNEL}.*.ndjson"),
        key=lambda p: int(p.name.split(".")[-2]),
    )
    assert len(segments) >= 3, f"expected the segment to roll several times, got {len(segments)}"

    longest_line = max(
        len(line.encode("utf-8")) + 1  # the trailing newline the writer appends
        for segment in segments
        for line in segment.read_text(encoding="utf-8").splitlines()
    )
    for segment in segments:
        size = segment.stat().st_size
        # The core bound: no segment exceeds the rotation threshold by more than the
        # one record that tipped it over. A leak in rotation shows up here as a file
        # that ran far past the cap.
        assert size <= cap + longest_line, (
            f"{segment.name} is {size} bytes, past the {cap}-byte cap by more than one record"
        )
    # Every segment but the one still being written reached the cap before rolling.
    for finished in segments[:-1]:
        assert finished.stat().st_size >= cap, f"{finished.name} rolled early, below the cap"


# ════════════════════════════════════════════════════════════════
# 2. Downsampled table row bound
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_downsampled_rows_are_bounded_by_duration_not_sample_count(tmp_path: Path) -> None:
    """History grows with time, never with sampling rate.

    The whole point of a downsampled tier is that a fast channel does not write a
    row per sample: eight hours on a sixty-second interval is a few hundred windows
    no matter whether the channel is read once a minute or a thousand times. If the
    table instead grew per sample it would be an unbounded privacy exposure and an
    unbounded disk cost, which is the failure this bound guards.
    """
    db_path = tmp_path / "instrument.duckdb"
    interval_s = 60.0
    store = ReadingStore(
        raw_dir=tmp_path / "raw", db_path=db_path, downsample_interval_s=interval_s
    )
    transport = await _open(_sine_source(period_s=600.0))

    windows = int(_EIGHT_HOURS_S // interval_s)  # 480
    reads_per_window = 3
    for _ in range(windows):
        for _ in range(reads_per_window):
            store.record(await transport.read(_CHANNEL))
            transport.advance_clock(interval_s / reads_per_window)
        store.flush(force=True)  # close this window, exactly as a sampling loop would

    total_reads = windows * reads_per_window
    upper_bound = math.ceil(_EIGHT_HOURS_S / interval_s)  # rows can never exceed the intervals

    assert store.windows_written == windows
    # Release the read-write connection before the read-only inspection below: DuckDB
    # rejects a second connection to the same file under a different configuration.
    store.close()
    row_count = _count_windows(db_path)
    assert row_count <= upper_bound, f"{row_count} rows exceeds the {upper_bound}-interval ceiling"
    assert row_count < total_reads, (
        f"{row_count} rows from {total_reads} samples is no downsampling at all"
    )
    # The compression is real: each row aggregates a whole window, not a lone sample.
    samples_per_row = {row[2] for row in _current_window_rows(db_path)}
    assert samples_per_row == {reads_per_window}


# ════════════════════════════════════════════════════════════════
# 3. History retention
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_windows_past_the_history_ttl_are_pruned(tmp_path: Path) -> None:
    """Retention must actually delete, or the table grows until the disk stops it.

    Retention keys off the real wall clock, which ``advance_clock`` cannot move, so
    the run is staged the way a real deployment reaches this branch. The transport's
    wall base is placed seven days in the past; advancing the logical clock then
    sweeps ``observed_at`` from a week ago up toward now, building a full week of
    history while retention is switched off. A second store -- a restart -- opens the
    same table with a finite horizon, and its first flush runs the prune, because the
    rate-limit clock starts at zero. That is exactly how a restart trims a table that
    grew while retention was disabled.
    """
    db_path = tmp_path / "instrument.duckdb"
    interval_s = 3600.0
    windows = int(_SEVEN_DAYS_S // interval_s)  # 168 hourly windows
    wall_base = time.time() - _SEVEN_DAYS_S

    builder = ReadingStore(
        raw_dir=tmp_path / "raw",
        db_path=db_path,
        downsample_interval_s=interval_s,
        history_ttl_s=0.0,  # retention off while a week of history accumulates
    )
    transport = await _open(_sine_source(period_s=interval_s))
    transport._wall_base = wall_base  # noqa: SLF001 - begin the run seven days ago
    for _ in range(windows):
        builder.record(await transport.read(_CHANNEL))
        builder.flush(force=True)
        transport.advance_clock(interval_s)
    builder.close()
    assert builder.windows_written == windows
    assert builder.rows_pruned == 0, "retention was off; nothing should have been deleted yet"

    # A horizon landing mid-interval (3.5 days back), so the boundary is a clear
    # 30 minutes from any window and sub-second scheduling jitter cannot shift it.
    history_ttl_s = 84.5 * 3600.0
    keeper = ReadingStore(
        raw_dir=tmp_path / "raw",
        db_path=db_path,
        downsample_interval_s=interval_s,
        history_ttl_s=history_ttl_s,
    )
    fresh = await _open({"values": {_CHANNEL: 42.0}})
    keeper.record(await fresh.read(_CHANNEL))  # a window at ~now, safely inside the horizon
    keeper.flush(force=True)

    cutoff = time.time() - history_ttl_s
    expected_pruned = sum(1 for k in range(windows) if wall_base + k * interval_s < cutoff)
    assert expected_pruned > 0  # the staged week must actually straddle the horizon

    assert keeper.rows_pruned == expected_pruned
    # Release the writer before the read-only inspection: DuckDB refuses a second
    # connection to the same file under a different configuration.
    keeper.close()
    assert _count_windows(db_path) == windows + 1 - expected_pruned
    survivors = _current_window_rows(db_path)
    assert survivors, "the recent windows and the fresh one must remain"
    oldest_surviving = min(row[1] for row in survivors)  # ended_at
    assert oldest_surviving >= cutoff, "a window past the horizon survived the prune"


# ════════════════════════════════════════════════════════════════
# 4. Wall-clock ordering across a restart
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_history_stays_wall_ordered_across_a_restart(tmp_path: Path) -> None:
    """A reboot resets the monotonic origin; wall-clock ordering must survive it.

    Carrying one clock for both roles is what made this tier unusable before: a
    monotonic origin restarts on reboot, so a later boot's rows carry *smaller*
    boundary values than an earlier boot's, and ``ORDER BY ended_at DESC`` returned
    the oldest rows as the newest. Here two store instances write across a simulated
    restart -- the second transport's monotonic base is reset near zero and its wall
    base is more recent -- and the second run's windows must still sort *after* the
    first's, because boundaries are persisted as wall-clock ``observed_at``. A
    pre-fix row carrying monotonic instants (schema version 0) must stay out of the
    series entirely rather than blend two incompatible timebases into one chart.
    """
    import duckdb

    db_path = tmp_path / "instrument.duckdb"

    first = ReadingStore(raw_dir=tmp_path / "raw", db_path=db_path)
    boot_one = await _open({"values": {_CHANNEL: 1.0}})
    boot_one._wall_base = time.time() - 7200.0  # noqa: SLF001 - the earlier run, two hours ago
    for _ in range(3):
        first.record(await boot_one.read(_CHANNEL))
        first.flush(force=True)
        boot_one.advance_clock(60.0)
    first.close()

    # The restart: a fresh store and a fresh transport whose monotonic counter has
    # restarted near zero, while its wall clock is genuinely more recent.
    second = ReadingStore(raw_dir=tmp_path / "raw", db_path=db_path)
    boot_two = await _open({"values": {_CHANNEL: 2.0}})
    boot_two._mono_base = 1.0  # noqa: SLF001 - a per-boot counter restarted at reboot
    boot_two._wall_base = time.time() - 3600.0  # noqa: SLF001 - one hour ago, after boot one
    for _ in range(3):
        second.record(await boot_two.read(_CHANNEL))
        second.flush(force=True)
        boot_two.advance_clock(60.0)

    # A row exactly as the pre-fix implementation wrote them: monotonic instants in
    # the boundary columns, and schema version 0. Inserted after the flushes so no
    # prune runs against it -- history() must exclude it on read.
    connection = duckdb.connect(str(db_path))
    try:
        connection.execute(
            "INSERT INTO reading_windows (device_id, channel_id, quantity, unit, "
            "started_at, ended_at, samples, dropped, min_value, max_value, mean_value, "
            "last_value, quality_worst, schema_version) "
            f"VALUES ('{_DEVICE}', '{_CHANNEL}', 'generic.sensor', 'unit', 900000.0, "
            "900060.0, 10, 0, 1.0, 2.0, 1.5, '2.0', 'ok', 0)"
        )
    finally:
        connection.close()

    rows = second.history(_DEVICE, _CHANNEL, limit=50)
    ends = [row["ended_at"] for row in rows]

    assert len(rows) == 6, "both runs are present and the pre-fix monotonic row is excluded"
    assert 900060.0 not in ends, "a monotonic-boundary row must never enter the series"
    assert ends == sorted(ends), "history is ordered oldest-first by wall-clock ended_at"
    assert max(ends[:3]) < min(ends[3:]), (
        "the second run must sort after the first even though its monotonic base is smaller"
    )
    # Every boundary is a real wall-clock epoch, not a per-boot counter.
    a_day_ago = time.time() - 86400.0
    assert all(end > a_day_ago for end in ends)
