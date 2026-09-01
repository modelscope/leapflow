"""Continuous sampling: the ring, the detector, and the signal source.

The layering under test is a boundary decision, not an optimisation. Raw readings must
stay inside the hardware package -- a single 10 Hz channel would flush a 50-slot
``SignalBuffer`` in five seconds and drive causal fusion at sampling rate. What crosses
into the interaction pipeline is derived events, at the rate something notable happens.

Every detection rule is asserted to come from the channel's declared ``Envelope``, so a
future change cannot quietly introduce a threshold that no human wrote down.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from leapflow.hardware.context import (
    HC_VERSION,
    Channel,
    ContextProvenance,
    Direction,
    Envelope,
    HardwareContext,
    HardwareEffect,
    Quality,
    TransportRef,
)
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.stream import (
    EventKind,
    HardwareEvent,
    HardwareEventDetector,
    HardwareStreamSource,
    ReadingRing,
    build_stream_sources,
)
from leapflow.hardware.transport import Reading


# ════════════════════════════════════════════════════════════════
# Fixtures -- a generic sampled channel, no device type anywhere
# ════════════════════════════════════════════════════════════════


def _context(*, sample_rate_hz: float = 10.0, max_rate: float | None = None) -> HardwareContext:
    return HardwareContext(
        device_id="sampled_device",
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
                envelope=Envelope(
                    declared=True, min_value=0.0, max_value=100.0, max_rate=max_rate
                ),
            ),
            Channel(
                channel_id="knob",
                direction=Direction.WRITE.value,
                quantity="generic.knob",
                effect=HardwareEffect.CONFIGURE.value,
                envelope=Envelope(declared=True, min_value=0.0, max_value=1.0, reversible=True),
            ),
        ),
        provenance=ContextProvenance(verified_by="tester"),
    )


class _StaticProvider:
    kind = "static"

    def __init__(self, *contexts: HardwareContext) -> None:
        self._contexts = contexts

    def discover(self) -> tuple[HardwareContext, ...]:
        return self._contexts


def _registry(context: HardwareContext, **overrides: Any) -> HardwareRegistry:
    registry = HardwareRegistry(
        HardwareSettings(enabled=True, **overrides), providers=[_StaticProvider(context)]
    )
    registry.load()
    return registry


_WALL_EPOCH = 1_780_000_000.0
"""An arbitrary but realistic wall-clock base, three orders of magnitude above the
monotonic values these tests use, so a clock mix-up is visible rather than plausible."""


def _reading(
    value: Any,
    *,
    sequence: int,
    at: float = 0.0,
    quality: str = Quality.OK.value,
):
    """Build a reading whose two clocks are distinguishable.

    ``observed_at`` is offset onto a plausible wall-clock epoch so any code that
    confuses the two produces an obviously wrong number instead of a subtly wrong
    one -- the failure mode this pair exists to prevent is silence.
    """
    return Reading(
        device_id="sampled_device",
        channel_id="level",
        value=value,
        quantity="generic.level",
        unit="unit",
        monotonic_at=at,
        observed_at=_WALL_EPOCH + at,
        sequence=sequence,
        quality=quality,
    )


# ════════════════════════════════════════════════════════════════
# ReadingRing
# ════════════════════════════════════════════════════════════════


def test_ring_is_bounded() -> None:
    """An unbounded history is a memory leak with a schedule."""
    ring = ReadingRing(capacity=4)
    for index in range(20):
        ring.record(_reading(float(index), sequence=index))
    assert len(ring) == 4
    assert ring.latest.value == 19.0


def test_ring_detects_a_sequence_gap() -> None:
    """Dropping samples is acceptable; dropping them silently is not.

    A break in the transport's sequence numbering is the only evidence that something
    was lost between the device and here.
    """
    ring = ReadingRing()
    assert ring.record(_reading(1.0, sequence=1)) == 0
    assert ring.record(_reading(2.0, sequence=2)) == 0
    lost = ring.record(_reading(3.0, sequence=7))
    assert lost == 4
    assert ring.dropped == 4


def test_ring_does_not_invent_a_gap_on_the_first_sample() -> None:
    ring = ReadingRing()
    assert ring.record(_reading(1.0, sequence=500)) == 0
    assert ring.dropped == 0


def test_ring_summary_is_compact_and_has_no_raw_series() -> None:
    """The summary is what reaches a model; the series must never leave the ring."""
    ring = ReadingRing()
    for index in range(50):
        ring.record(_reading(float(index), sequence=index))
    summary = ring.summary()
    assert summary["samples"] == 50
    assert summary["latest"] == 49.0
    assert summary["min"] == 0.0
    assert summary["max"] == 49.0
    assert summary["trend"] == "rising"
    # No key carries the individual readings.
    assert not any(isinstance(value, (list, tuple)) for value in summary.values())


def test_ring_summary_reports_a_falling_trend() -> None:
    ring = ReadingRing()
    for index in range(20):
        ring.record(_reading(float(100 - index), sequence=index))
    assert ring.summary()["trend"] == "falling"


def test_ring_summary_of_an_empty_ring_is_honest() -> None:
    assert ReadingRing().summary() == {"samples": 0}


def test_ring_summary_tolerates_non_numeric_values() -> None:
    """A state channel has no min/max; the summary must not invent them."""
    ring = ReadingRing()
    for index, state in enumerate(("idle", "busy", "idle")):
        ring.record(_reading(state, sequence=index))
    summary = ring.summary()
    assert summary["samples"] == 3
    assert "mean" not in summary


# ════════════════════════════════════════════════════════════════
# HardwareEventDetector -- rules derived from the envelope
# ════════════════════════════════════════════════════════════════


def _detector(**kwargs: Any) -> HardwareEventDetector:
    context = _context(**kwargs)
    return HardwareEventDetector(context, context.channel("level"))


def test_threshold_event_fires_once_per_excursion() -> None:
    """A breach is an event; staying breached is not.

    Re-reporting every sample above the limit would reproduce the sampling-rate flood
    this layer exists to prevent.
    """
    detector = _detector()
    assert detector.observe(_reading(50.0, sequence=1, at=1.0)) == ()
    first = detector.observe(_reading(150.0, sequence=2, at=2.0))
    assert [e.kind for e in first] == [EventKind.THRESHOLD_EXCEEDED]
    assert detector.observe(_reading(160.0, sequence=3, at=3.0)) == ()


def test_returning_to_range_is_reported_as_recovery() -> None:
    """Recovery must be observable, not inferred from silence."""
    detector = _detector()
    detector.observe(_reading(150.0, sequence=1, at=1.0))
    events = detector.observe(_reading(50.0, sequence=2, at=2.0))
    assert [e.kind for e in events] == [EventKind.SETTLED]


def test_rate_event_uses_the_declared_max_rate() -> None:
    detector = _detector(max_rate=5.0)
    detector.observe(_reading(10.0, sequence=1, at=1.0))
    # 40 units in one second, against a declared 5/s.
    events = detector.observe(_reading(50.0, sequence=2, at=2.0))
    assert EventKind.RATE_EXCEEDED in [e.kind for e in events]


def test_no_rate_event_without_a_declared_limit() -> None:
    """No rule may exist that a human did not write down."""
    detector = _detector(max_rate=None)
    detector.observe(_reading(10.0, sequence=1, at=1.0))
    events = detector.observe(_reading(90.0, sequence=2, at=2.0))
    assert EventKind.RATE_EXCEEDED not in [e.kind for e in events]


def test_sample_loss_is_reported() -> None:
    detector = _detector()
    events = detector.observe(_reading(20.0, sequence=5, at=1.0), lost=3)
    assert [e.kind for e in events] == [EventKind.SAMPLE_LOSS]
    assert "3 sample" in events[0].detail


def test_quality_degradation_needs_a_streak() -> None:
    """One suspect sample is noise; a run of them is a fault."""
    detector = _detector()
    kinds: list[str] = []
    for index in range(3):
        events = detector.observe(
            _reading(20.0, sequence=index, at=float(index), quality=Quality.SUSPECT.value)
        )
        kinds.extend(e.kind for e in events)
    assert kinds.count(EventKind.QUALITY_DEGRADED) == 1


def test_quality_streak_resets_on_a_good_sample() -> None:
    detector = _detector()
    detector.observe(_reading(20.0, sequence=1, at=1.0, quality=Quality.SUSPECT.value))
    detector.observe(_reading(20.0, sequence=2, at=2.0, quality=Quality.SUSPECT.value))
    detector.observe(_reading(20.0, sequence=3, at=3.0))
    events = detector.observe(
        _reading(20.0, sequence=4, at=4.0, quality=Quality.SUSPECT.value)
    )
    assert EventKind.QUALITY_DEGRADED not in [e.kind for e in events]


def test_silence_on_a_declared_rate_is_itself_an_observation() -> None:
    """A 10 Hz channel that says nothing for a second has failed."""
    detector = _detector(sample_rate_hz=10.0)
    detector.observe(_reading(20.0, sequence=1, at=100.0))
    assert detector.check_stale(now=100.05) == ()
    events = detector.check_stale(now=101.0)
    assert [e.kind for e in events] == [EventKind.STALE]
    # Reported once, not on every check.
    assert detector.check_stale(now=102.0) == ()


def test_staleness_does_not_apply_to_an_unsampled_channel() -> None:
    detector = _detector(sample_rate_hz=0.0)
    detector.observe(_reading(20.0, sequence=1, at=100.0))
    assert detector.check_stale(now=1000.0) == ()


def test_event_detail_is_a_single_readable_line() -> None:
    event = HardwareEvent(
        kind=EventKind.THRESHOLD_EXCEEDED,
        device_id="dev",
        channel_id="ch",
        quantity="generic.level",
        detail="left the declared range (0..100)",
        value=150.0,
        unit="unit",
    )
    rendered = event.to_detail()
    assert rendered.startswith("[threshold_exceeded] dev.ch")
    assert "150.0 unit" in rendered
    assert "\n" not in rendered


# ════════════════════════════════════════════════════════════════
# Source construction
# ════════════════════════════════════════════════════════════════


def test_only_streaming_channels_become_sources() -> None:
    """``sample_rate_hz > 0`` is the only switch; no device type is consulted."""
    registry = _registry(_context())
    sources = build_stream_sources(registry)
    assert [s.source_id for s in sources] == ["hw:sampled_device:level"]


def test_no_sources_when_streaming_is_disabled() -> None:
    registry = _registry(_context(), stream_enabled=False)
    assert registry.stream_sources() == ()


def test_sources_are_built_once_and_cached() -> None:
    """The manager rejects registration after start; fresh instances would be orphans."""
    registry = _registry(_context())
    assert registry.stream_sources() is registry.stream_sources()


def test_reload_discards_cached_sources() -> None:
    """A cached source would keep sampling a channel the declaration removed."""
    registry = _registry(_context())
    first = registry.stream_sources()
    registry.load()
    assert registry.stream_sources() is not first


def test_source_channel_id_gates_the_whole_device() -> None:
    """One config-level channel per device, so a whole rig can be muted at once."""
    registry = _registry(_context())
    source = registry.stream_sources()[0]
    assert source.channel_id == "hw.sampled_device"


def test_source_satisfies_the_active_signal_source_protocol() -> None:
    """Device observations ride the existing signal path, not a parallel one."""
    from leapflow.perception.active_signal_source import ActiveSignalSource

    registry = _registry(_context())
    assert isinstance(registry.stream_sources()[0], ActiveSignalSource)


# ════════════════════════════════════════════════════════════════
# Sampling lifecycle
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_start_returns_promptly_and_samples_in_the_background() -> None:
    """A source that sampled inline would stall every source started after it."""
    registry = _registry(_context(sample_rate_hz=50.0))
    source = registry.stream_sources()[0]
    emitted: list[Any] = []
    await asyncio.wait_for(source.start(emitted.append), timeout=0.5)
    await asyncio.sleep(0.1)
    await source.stop()
    assert len(source.ring) >= 2


@pytest.mark.asyncio
async def test_stop_is_idempotent() -> None:
    registry = _registry(_context(sample_rate_hz=50.0))
    source = registry.stream_sources()[0]
    await source.start(lambda signal: None)
    await source.stop()
    await source.stop()


@pytest.mark.asyncio
async def test_events_cross_the_boundary_as_hardware_events() -> None:
    """Only derived events reach the emit sink, and they keep their structure.

    The sink receives the event rather than a flattened signal because the family,
    the value and the unit are all needed downstream: an event type of
    ``hw.<kind>`` is what makes the board group these without any enumeration, and
    a detail string would force every consumer to parse it back apart.
    """
    context = _context(sample_rate_hz=50.0)
    registry = _registry(context)
    source = registry.stream_sources()[0]
    emitted: list[Any] = []
    await source.start(emitted.append)
    await asyncio.sleep(0.05)
    # Push the value out of the declared range through the mock transport.
    transport = await registry.transport("sampled_device")
    transport.set_value("level", 500.0)
    await asyncio.sleep(0.1)
    await source.stop()

    assert emitted, "a threshold excursion should have produced an event"
    event = emitted[0]
    assert isinstance(event, HardwareEvent)
    assert event.kind == EventKind.THRESHOLD_EXCEEDED
    assert event.event_type == "hw.threshold_exceeded"
    assert event.source == "sampled_device.level"
    # Wall-clock: this instant is sorted against findings and other signal families,
    # all of which are wall-clock. A monotonic value here lands decades away.
    assert event.observed_at > 1_500_000_000.0
    payload = event.to_payload()
    assert payload["ts"] == event.observed_at
    # The reorder buffer keys on this; omitting it leaves hardware events unorderable
    # against every other source.
    assert "_mono_ts" in payload


@pytest.mark.asyncio
async def test_repeated_events_of_one_kind_are_paced() -> None:
    """A level-triggered kind must not emit once per sample.

    Without this floor a slew that stays above ``max_rate`` for a whole ramp
    reproduces, on the consumer side, exactly the sampling-rate flood that keeping
    raw readings inside this module prevents on the producer side.
    """
    context = _context(sample_rate_hz=50.0)
    registry = _registry(context)
    source = registry.stream_sources()[0]
    emitted: list[Any] = []
    await source.start(emitted.append)
    transport = await registry.transport("sampled_device")
    transport.set_value("level", 500.0)
    # Long enough for many samples at 50 Hz, but shorter than the pacing floor.
    await asyncio.sleep(0.2)
    await source.stop()

    breaches = [e for e in emitted if e.kind == EventKind.THRESHOLD_EXCEEDED]
    assert len(breaches) == 1, "one excursion is one event, however many samples it spans"


def test_same_kind_different_channels_are_not_cross_throttled() -> None:
    """A threshold breach on channel A must not suppress a simultaneous breach on B.

    Before this fix the pacing key was ``event.kind`` alone, so the second
    channel's first breach was silently dropped whenever it arrived within
    ``MIN_EVENT_INTERVAL_S`` of the first channel's breach.
    """
    context = _context()
    source = HardwareStreamSource(None, context, context.channels[0])

    event_ch1 = HardwareEvent(
        kind=EventKind.THRESHOLD_EXCEEDED,
        device_id="dev",
        channel_id="ch_a",
        quantity="q",
        detail="breach on A",
        observed_at=time.time(),
    )
    event_ch2 = HardwareEvent(
        kind=EventKind.THRESHOLD_EXCEEDED,
        device_id="dev",
        channel_id="ch_b",
        quantity="q",
        detail="breach on B",
        observed_at=time.time(),
    )

    emitted: list[Any] = []
    source._dispatch([event_ch1, event_ch2], emitted.append)

    assert len(emitted) == 2, (
        "same kind on different channels must not suppress each other"
    )
    assert {e.channel_id for e in emitted} == {"ch_a", "ch_b"}


def test_same_channel_same_kind_is_still_suppressed() -> None:
    """Level-triggered events on the same channel are still paced.

    The per-channel key must not accidentally defeat the rate floor that
    prevents a flood of identical observations on the same channel.
    """
    context = _context()
    source = HardwareStreamSource(None, context, context.channels[0])

    event = HardwareEvent(
        kind=EventKind.THRESHOLD_EXCEEDED,
        device_id="sampled_device",
        channel_id="level",
        quantity="q",
        detail="breach",
        observed_at=time.time(),
    )

    emitted: list[Any] = []
    # Dispatch twice without waiting for the pacing interval to elapse.
    source._dispatch([event], emitted.append)
    source._dispatch([event], emitted.append)

    assert len(emitted) == 1, (
        "same kind on the same channel should be suppressed by the rate floor"
    )


def test_paced_out_counter_reflects_suppressed_events() -> None:
    """``_paced_out`` increments exactly once per suppressed event."""
    context = _context()
    source = HardwareStreamSource(None, context, context.channels[0])
    assert source._paced_out == 0

    event_a = HardwareEvent(
        kind=EventKind.RATE_EXCEEDED,
        device_id="sampled_device",
        channel_id="level",
        quantity="q",
        detail="fast",
        observed_at=time.time(),
    )

    source._dispatch([event_a], None)  # admitted
    assert source._paced_out == 0

    source._dispatch([event_a], None)  # suppressed
    assert source._paced_out == 1

    source._dispatch([event_a], None)  # suppressed again
    assert source._paced_out == 2


def test_different_devices_same_channel_id_are_independent() -> None:
    """Two devices with identical channel names must not suppress each other."""
    context = _context()
    source = HardwareStreamSource(None, context, context.channels[0])

    ev1 = HardwareEvent(
        kind=EventKind.STALE,
        device_id="device_alpha",
        channel_id="level",
        quantity="q",
        detail="stale alpha",
        observed_at=time.time(),
    )
    ev2 = HardwareEvent(
        kind=EventKind.STALE,
        device_id="device_beta",
        channel_id="level",
        quantity="q",
        detail="stale beta",
        observed_at=time.time(),
    )

    emitted: list[Any] = []
    source._dispatch([ev1, ev2], emitted.append)

    assert len(emitted) == 2, (
        "same channel_id on different devices must not suppress each other"
    )


def test_a_value_resting_on_the_boundary_does_not_flap() -> None:
    """Recovery must clear an inward margin, or a hovering value alternates forever.

    This is the difference between a crossing and a hover. Judged by a plain in/out
    test they are identical, and the hover buries the crossing that mattered.
    """
    channel = Channel(
        channel_id="level",
        direction=Direction.READ.value,
        quantity="generic.level",
        unit="unit",
        envelope=Envelope(declared=True, min_value=0.0, max_value=100.0),
    )
    detector = HardwareEventDetector(_context(), channel)
    detector.observe(_reading(50.0, sequence=1, at=1.0))
    # Leave the range, then sit just barely back inside it.
    breach = detector.observe(_reading(100.5, sequence=2, at=2.0))
    assert [e.kind for e in breach] == [EventKind.THRESHOLD_EXCEEDED]
    assert detector.observe(_reading(99.99, sequence=3, at=3.0)) == ()
    assert detector.observe(_reading(100.5, sequence=4, at=4.0)) == ()
    # Well inside now: recovery is reported exactly once.
    settled = detector.observe(_reading(50.0, sequence=5, at=5.0))
    assert [e.kind for e in settled] == [EventKind.SETTLED]


@pytest.mark.asyncio
async def test_raw_readings_never_reach_the_signal_pipeline() -> None:
    """The central boundary: samples stay in the ring, events cross.

    A 10 Hz channel emitting per sample would flush a 50-slot SignalBuffer in five
    seconds and run causal fusion at sampling rate.
    """
    registry = _registry(_context(sample_rate_hz=100.0))
    source = registry.stream_sources()[0]
    emitted: list[Any] = []
    await source.start(emitted.append)
    await asyncio.sleep(0.15)
    await source.stop()
    # Many samples were taken; none of them were emitted, because nothing notable
    # happened -- the value sat inside its declared range the whole time.
    assert len(source.ring) >= 5
    assert emitted == []


@pytest.mark.asyncio
async def test_a_failing_read_does_not_stop_sampling() -> None:
    """One unreachable device must not take the loop down with it."""
    registry = _registry(_context(sample_rate_hz=50.0))
    source = registry.stream_sources()[0]

    class _Broken:
        async def read(self, channel_id: str):
            raise RuntimeError("cable unplugged")

    async def _broken_transport(device_id: str):
        return _Broken()

    registry.transport = _broken_transport  # type: ignore[assignment]
    await source.start(lambda signal: None)
    await asyncio.sleep(0.1)
    await source.stop()
    assert len(source.ring) == 0


@pytest.mark.asyncio
async def test_a_raising_sink_does_not_stop_sampling() -> None:
    registry = _registry(_context(sample_rate_hz=50.0))
    source = registry.stream_sources()[0]

    def _explode(signal: Any) -> None:
        raise RuntimeError("downstream consumer is broken")

    await source.start(_explode)
    await asyncio.sleep(0.05)
    transport = await registry.transport("sampled_device")
    transport.set_value("level", 500.0)
    await asyncio.sleep(0.08)
    await source.stop()
    assert len(source.ring) >= 2


# ════════════════════════════════════════════════════════════════
# Disclosure
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_read_tool_discloses_a_summary_not_a_series() -> None:
    from leapflow.hardware.tools import HardwareTools

    registry = _registry(_context(sample_rate_hz=100.0))
    source = registry.stream_sources()[0]
    await source.start(lambda signal: None)
    await asyncio.sleep(0.1)
    await source.stop()

    tools = HardwareTools(registry, session_id="s")
    result = await tools.hw_read(device_id="sampled_device", channel_id="level")
    assert result["ok"] is True
    assert "history" in result
    assert result["history"]["samples"] >= 2
    assert "readings" not in result["history"]


@pytest.mark.asyncio
async def test_read_tool_omits_history_for_an_unsampled_channel() -> None:
    """Absent data is omitted rather than reported as empty."""
    from leapflow.hardware.tools import HardwareTools

    registry = _registry(_context(sample_rate_hz=0.0))
    tools = HardwareTools(registry, session_id="s")
    result = await tools.hw_read(device_id="sampled_device", channel_id="level")
    assert result["ok"] is True
    assert "history" not in result


# ════════════════════════════════════════════════════════════════
# Boot-order constraint
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_sources_must_be_registered_before_the_manager_starts() -> None:
    """Pins the ordering constraint that makes hardware streaming work at all.

    ``ActiveSourceManager.register`` raises once ``start_all`` has run, so hardware
    sources have to be bound before the perception session starts. Asserting it here
    means a future boot reordering fails loudly instead of silently turning device
    observation off.
    """
    from leapflow.perception.active_signal_source import ActiveSourceManager
    from leapflow.perception.signals import SignalBuffer

    class _NoopPipeline:
        def fuse(self, **kwargs: Any) -> None:
            return None

    registry = _registry(_context(sample_rate_hz=20.0))
    source = registry.stream_sources()[0]
    manager = ActiveSourceManager(SignalBuffer(), _NoopPipeline(), object())
    manager.register(source)
    await manager.start_all()
    try:
        with pytest.raises(RuntimeError):
            manager.register(
                HardwareStreamSource(
                    registry, registry.context("sampled_device"), Channel(channel_id="late")
                )
            )
    finally:
        await manager.dispose()


# ════════════════════════════════════════════════════════════════
# Device I/O serialisation and sampling health
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_concurrent_reads_on_one_device_are_serialised() -> None:
    """Two channels of one instrument are one conversation.

    A serial line, an I2C bus or a GPIB address handles one request at a time. Two
    coroutines reading concurrently interleave request and response frames, and the
    result is not an error -- it is a plausible reading carrying the wrong channel's
    value, which nothing downstream can detect. Streaming makes this the common case
    because one task per channel starts automatically.
    """
    context = _context(sample_rate_hz=10.0)
    registry = _registry(context)
    overlaps = 0
    active = 0

    async def _contend() -> None:
        nonlocal overlaps, active
        async with registry.device_io("sampled_device"):
            active += 1
            if active > 1:
                overlaps += 1
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(*(_contend() for _ in range(5)))
    assert overlaps == 0, "data-plane access to one device must not overlap"


@pytest.mark.asyncio
async def test_each_device_gets_its_own_lock() -> None:
    """Serialisation is per device; one slow instrument must not stall another."""
    context = _context()
    registry = _registry(context)
    assert registry.device_io("a") is registry.device_io("a")
    assert registry.device_io("a") is not registry.device_io("b")


@pytest.mark.asyncio
async def test_health_compares_observed_rate_against_the_declaration() -> None:
    """A cadence shortfall is invisible in the data itself.

    The stored series looks entirely normal when a channel runs at two thirds of its
    declared rate -- the window records the samples it actually got, and nothing else
    compares that against what was declared.
    """
    context = _context(sample_rate_hz=50.0)
    registry = _registry(context)
    source = registry.stream_sources()[0]
    await source.start(None)
    await asyncio.sleep(0.15)
    await source.stop()

    health = source.health
    assert health["declared_hz"] == 50.0
    assert health["samples"] > 0
    assert health["observed_hz"] > 0.0
    assert 0.0 < health["rate_ratio"] <= 1.5
    assert health["channel_id"] == "level"


# ════════════════════════════════════════════════════════════════
# G-2: First-order settling model integration
# ════════════════════════════════════════════════════════════════


def test_first_order_settling_flows_through_detector_channel() -> None:
    """Detector carries the envelope's effective settling through the channel.

    The stream layer does not enforce settling itself (that is outcome.py's
    concern), but the detector must faithfully expose the model so downstream
    consumers (alert policy, settling heuristics) can query it.
    """
    context = HardwareContext(
        device_id="heater",
        hc_version=HC_VERSION,
        halt_supported=True,
        transport=TransportRef(kind="mock", config={"values": {"temp": 25.0}}),
        channels=(
            Channel(
                channel_id="temp",
                direction=Direction.READ.value,
                quantity="temperature.heater",
                unit="degC",
                sample_rate_hz=10.0,
                envelope=Envelope(
                    declared=True,
                    min_value=0.0,
                    max_value=200.0,
                    settling_model="first_order",
                    settling_tau_s=2.0,
                ),
            ),
        ),
        provenance=ContextProvenance(verified_by="tester"),
    )
    detector = HardwareEventDetector(context, context.channel("temp"))
    # The detector's channel envelope reports 5τ = 10 s.
    assert detector._channel.envelope.effective_settling_s == pytest.approx(10.0)
    assert detector._channel.envelope.settling_model == "first_order"


def test_step_settling_default_unchanged_through_detector() -> None:
    """Default step model yields the plain settling_time_s (backward compat)."""
    context = _context()
    detector = HardwareEventDetector(context, context.channel("level"))
    assert detector._channel.envelope.settling_model == "step"
    assert detector._channel.envelope.effective_settling_s == 0.0
