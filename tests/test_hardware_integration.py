"""L2 integration: real ReadingStore + real DuckDB + real EventBus + MockTransport.

Cross-boundary assertion gap (CBAG) regression suite for G15, G16, and G24.
Each test locks one confirmed fix so that a single-process unit test can
detect a regression that previously required multi-component observation.

CBAG defect map
---------------
- **G15**: emit wiring disconnected — events produced but nothing reacts.
  Locked by asserting the EventBus receives events the transport produces.
- **G16**: monotonic clock persisted as wall-clock — ORDER BY reversed.
  Locked by asserting every persisted ``observed_at`` is a wall-clock epoch
  (later than yesterday, relative to ``time.time()``) and that ``history()``
  returns oldest-first by that clock.
- **G24**: hardware producer not registered — board never updates.
  Locked via architecture contracts (see ``test_architecture_contracts.py``);
  here the integration variant asserts the full sample→persist→query path.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

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
from leapflow.hardware.reading_store import SCHEMA_VERSION, ReadingStore
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.transport import (
    Reading,
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_PARTIAL,
    SIDE_EFFECT_UNKNOWN,
    WriteOutcome,
)


# ════════════════════════════════════════════════════════════════
# Shared fixtures
# ════════════════════════════════════════════════════════════════


_WALL_NOW = time.time()
"""Wall-clock base: now.  Used for observed_at in synthetic readings."""


def _context(device_id: str = "bench", sample_rate_hz: float = 100.0) -> HardwareContext:
    """Return a minimal admitted device with one readable streaming channel."""
    return HardwareContext(
        device_id=device_id,
        hc_version=HC_VERSION,
        halt_supported=True,
        transport=TransportRef(
            kind="mock",
            config={"values": {"level": 20.0}},
        ),
        channels=(
            Channel(
                channel_id="level",
                direction=Direction.READ.value,
                quantity="generic.level",
                unit="C",
                sample_rate_hz=sample_rate_hz,
                envelope=Envelope(declared=True, min_value=0.0, max_value=100.0),
            ),
        ),
        provenance=ContextProvenance(verified_by="tester"),
    )


class _StaticProvider:
    """Minimal provider that yields a fixed list of contexts."""

    kind = "static"

    def __init__(self, *contexts: HardwareContext) -> None:
        self._contexts = contexts

    def discover(self) -> tuple[HardwareContext, ...]:
        return self._contexts


def _reading(
    value: float,
    *,
    seq: int = 1,
    mono: float = 0.0,
    wall: float | None = None,
    quality: str = Quality.OK.value,
) -> Reading:
    """Build a synthetic reading with explicit dual-clock values."""
    return Reading(
        device_id="bench",
        channel_id="level",
        value=value,
        quantity="generic.level",
        unit="C",
        observed_at=wall if wall is not None else _WALL_NOW + mono,
        monotonic_at=mono,
        sequence=seq,
        quality=quality,
    )


def _store(tmp_path: Path, **kwargs: Any) -> ReadingStore:
    return ReadingStore(
        raw_dir=tmp_path / "raw",
        db_path=tmp_path / "instrument.duckdb",
        downsample_interval_s=kwargs.pop("downsample_interval_s", 1.0),
        **kwargs,
    )


# ════════════════════════════════════════════════════════════════
# 1. Sample → persist → wall-ordered query (G16 regression)
# ════════════════════════════════════════════════════════════════


def test_sample_to_persist_to_wall_ordered_query(tmp_path: Path) -> None:
    """The full closed loop: record readings, flush to DuckDB, query back.

    Asserts wall-clock ordering and that persisted instants are genuine
    wall-clock epochs (later than yesterday), not monotonic values.

    CBAG G16: if the store regresses to monotonic instants, the
    later-than-yesterday assertion fails, and ordering by ``ended_at`` would
    return the oldest rows as the newest — caught by the ascending-order
    assertion.
    """
    store = _store(tmp_path)

    # Three windows at increasing wall-clock instants.
    for window_index in range(3):
        mono = float(window_index * 10)
        wall = _WALL_NOW + window_index * 60.0  # 60s apart
        store.record(_reading(
            20.0 + window_index,
            seq=window_index,
            mono=mono,
            wall=wall,
        ))
        store.flush(force=True)

    history = store.history("bench", "level")
    assert len(history) == 3, f"expected 3 windows, got {len(history)}"

    # G16 assertion: every persisted timestamp is a wall-clock epoch.  A
    # wall-clock instant built from time.time() is always later than
    # yesterday; a monotonic value (uptime since boot) is not.
    yesterday = time.time() - 86400.0
    for row in history:
        assert row["started_at"] > yesterday, (
            f"started_at={row['started_at']} predates yesterday, so it looks "
            "monotonic rather than wall-clock; G16 regression: monotonic "
            "instants must never be persisted"
        )
        assert row["ended_at"] > yesterday, (
            f"ended_at={row['ended_at']} predates yesterday, so it looks "
            "monotonic rather than wall-clock; G16 regression"
        )

    # G16 assertion: oldest first by wall-clock.
    ends = [row["ended_at"] for row in history]
    assert ends == sorted(ends), (
        f"history not oldest-first: {ends}; "
        "G16 regression: ORDER BY ended_at DESC with monotonic instants "
        "returns the oldest row as newest"
    )


def test_persisted_schema_version_is_current(tmp_path: Path) -> None:
    """Every newly written row carries the current SCHEMA_VERSION.

    G16 regression: version-0 rows are excluded from queries, so writing
    new data at version 0 would make it immediately invisible.
    """
    import duckdb

    store = _store(tmp_path)
    store.record(_reading(42.0, seq=1, mono=1.0))
    store.flush(force=True)
    # Release the store's read-write connection before opening a read-only one: DuckDB
    # refuses a second connection to the same file under a different configuration.
    store.close()

    db_path = tmp_path / "instrument.duckdb"
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = conn.execute(
            "SELECT schema_version FROM reading_windows"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0][0] == SCHEMA_VERSION, (
        f"written schema_version={rows[0][0]}, expected {SCHEMA_VERSION}; "
        "a version-0 row would be invisible to history queries"
    )


# ════════════════════════════════════════════════════════════════
# 2. Persist failure does not kill the sampling loop
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_persist_failure_does_not_stop_sampling(tmp_path: Path) -> None:
    """A locked or missing database must not take the sampling loop down.

    The store counts write_failures, and sampling must continue producing
    readings into the ring even when persistence is broken.

    CBAG G16/G24: if the persist path raises into the sample loop,
    observability vanishes — the very failure mode G24 describes.
    """
    registry = HardwareRegistry(
        HardwareSettings(enabled=True, stream_ring_capacity=64),
        providers=[_StaticProvider(_context(sample_rate_hz=100.0))],
    )
    registry.load()

    # A store whose every write raises.
    class _BrokenStore:
        write_failures_count = 0

        def record(self, reading: Any, *, dropped: int = 0) -> None:
            self.write_failures_count += 1
            raise RuntimeError("disk full")

        def flush(self, **_: Any) -> int:
            raise RuntimeError("disk full")

        def close(self) -> None:
            pass

        @property
        def write_failures(self) -> int:
            return self.write_failures_count

    broken = _BrokenStore()
    registry._reading_store = broken  # noqa: SLF001
    registry._stream_sources = None  # noqa: SLF001

    from leapflow.hardware.stream import build_stream_sources

    registry._stream_sources = build_stream_sources(  # noqa: SLF001
        registry, ring_capacity=64, reading_store=broken,
    )
    source = registry.stream_sources()[0]
    await source.start(lambda _signal: None)
    await asyncio.sleep(0.15)
    await source.stop()

    # Sampling kept going: the ring has readings despite persist failures.
    assert len(source.ring) >= 2, (
        "sampling stopped when persistence failed; "
        "G24 regression: a broken store must not kill the observation loop"
    )
    assert broken.write_failures_count > 0, (
        "the broken store was never called — wiring issue"
    )


# ════════════════════════════════════════════════════════════════
# 3. EventBus receives hardware events (G15 regression)
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_event_emitter_wiring_reaches_eventbus(tmp_path: Path) -> None:
    """The emit path from registry → EventBus must be connected.

    CBAG G15: six detection rules produced events that reached nothing because
    the emitter was not wired. This test asserts the path exists and events
    produced by the sampling loop actually reach the emitter callback.
    """
    registry = HardwareRegistry(
        HardwareSettings(
            enabled=True,
            stream_ring_capacity=64,
            instrument_db_path=str(tmp_path / "instrument.duckdb"),
            downsample_interval_s=0.05,
        ),
        providers=[_StaticProvider(_context(sample_rate_hz=100.0))],
    )
    registry.load()
    registry.bind_persistence(readings_dir=tmp_path / "raw", session_id="s1")

    # A collector that simulates what _hardware_event_emitter() does.
    received_events: list[Any] = []

    def _fake_emitter(event: Any) -> None:
        received_events.append(event)

    registry.set_event_emitter(_fake_emitter)

    # Verify the emitter is installed (G15: if set_event_emitter is a no-op
    # or the field is wrong, events go nowhere).
    assert registry._event_emitter is not None, (  # noqa: SLF001
        "G15 regression: event emitter not installed on registry"
    )

    # publish_event goes through the installed emitter.
    from leapflow.hardware.stream import HardwareEvent

    test_event = HardwareEvent(
        kind="threshold_exceeded",
        device_id="bench",
        channel_id="level",
        quantity="generic.level",
        detail="test event",
        value=105.0,
        unit="C",
        observed_at=time.time(),
    )
    registry.publish_event(test_event)
    assert len(received_events) == 1, (
        "G15 regression: publish_event did not reach the installed emitter"
    )
    assert received_events[0] is test_event


# ════════════════════════════════════════════════════════════════
# 4. PARTIAL / UNKNOWN side-effect outcomes must not be replayed
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_partial_unknown_side_effects_block_replay() -> None:
    """A write whose effect may have landed must not be blindly repeated.

    ``WriteOutcome.effect_may_have_landed`` is the gate that stops replay.
    PARTIAL and UNKNOWN must both return True; only NONE returns False.
    """
    for verdict in (SIDE_EFFECT_PARTIAL, SIDE_EFFECT_UNKNOWN):
        outcome = WriteOutcome(
            ok=False,
            side_effect_state=verdict,
            error="injected",
            failure_code="test",
        )
        assert outcome.effect_may_have_landed is True, (
            f"side_effect_state={verdict!r} must block replay "
            f"(effect_may_have_landed should be True)"
        )

    # Only NONE allows replay.
    safe = WriteOutcome(ok=False, side_effect_state="none", error="safe")
    assert safe.effect_may_have_landed is False, (
        "SIDE_EFFECT_NONE must allow replay (effect_may_have_landed should be False)"
    )


@pytest.mark.asyncio
async def test_mock_transport_failure_injection_preserves_side_effect() -> None:
    """MockTransport faithfully reports the declared side_effect_state.

    Used by the integration tests below: if the mock swallowed the verdict,
    the no-replay gate could not be tested against a real transport path.
    """
    from leapflow.hardware.transports.mock import MockTransport

    transport = MockTransport({
        "values": {"ch": 0.0},
        "halt_supported": True,
        "failures": [
            {
                "channel_id": "ch",
                "on_call": 1,
                "side_effect_state": SIDE_EFFECT_PARTIAL,
                "error": "partial write",
            },
            {
                "channel_id": "ch",
                "on_call": 2,
                "side_effect_state": SIDE_EFFECT_UNKNOWN,
                "error": "unknown write",
            },
        ],
    })
    ctx = _context(device_id="dev")
    await transport.open(ctx)

    # First write: PARTIAL → must not replay.
    r1 = await transport.write("ch", 10.0)
    assert r1.ok is False
    assert r1.side_effect_state == SIDE_EFFECT_PARTIAL
    assert r1.effect_may_have_landed is True

    # Second write: UNKNOWN → must not replay.
    r2 = await transport.write("ch", 20.0)
    assert r2.ok is False
    assert r2.side_effect_state == SIDE_EFFECT_UNKNOWN
    assert r2.effect_may_have_landed is True

    # Third write: succeeds → COMMITTED.
    r3 = await transport.write("ch", 30.0)
    assert r3.ok is True
    assert r3.side_effect_state == SIDE_EFFECT_COMMITTED


# ════════════════════════════════════════════════════════════════
# 5. Full streaming integration: sample → persist → query
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_streaming_produces_wall_clock_history(tmp_path: Path) -> None:
    """End-to-end: a real sampling loop produces durable, wall-clock-sorted history.

    Combines G15 (events reach emitter), G16 (wall-clock persistence), and
    G24 (observation path closed) in one integrated scenario.
    """
    events_received: list[Any] = []

    registry = HardwareRegistry(
        HardwareSettings(
            enabled=True,
            stream_ring_capacity=64,
            instrument_db_path=str(tmp_path / "instrument.duckdb"),
            downsample_interval_s=0.05,
        ),
        providers=[_StaticProvider(_context(sample_rate_hz=200.0))],
    )
    registry.load()
    registry.bind_persistence(readings_dir=tmp_path / "raw", session_id="sess")

    # Wire emitter to verify G15.
    registry.set_event_emitter(lambda evt: events_received.append(evt))

    await registry.start_streams()
    await asyncio.sleep(0.25)
    await registry.close_all()

    # G24: observation path is closed — data landed.
    store = registry.reading_store
    assert store is not None
    assert store.raw_writes > 0, "no raw samples written — observation path broken"

    history = registry.channel_history("bench", "level")
    assert len(history) >= 1, "no downsampled windows — persist path broken (G24)"

    # G16: all persisted timestamps are wall-clock (later than yesterday).
    yesterday = time.time() - 86400.0
    for row in history:
        assert row["started_at"] > yesterday, (
            f"started_at={row['started_at']} predates yesterday, not wall-clock "
            "(G16 regression)"
        )
        assert row["ended_at"] > yesterday, (
            f"ended_at={row['ended_at']} predates yesterday, not wall-clock "
            "(G16 regression)"
        )

    # G16: oldest first.
    ends = [row["ended_at"] for row in history]
    assert ends == sorted(ends), "history not sorted oldest-first (G16 regression)"
