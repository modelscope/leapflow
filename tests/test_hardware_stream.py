# Copyright (c) Alibaba, Inc. and its affiliates.
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
    # Continuous 10 Hz acquisition: 4 units in 0.1 seconds against 5/s.
    events = detector.observe(_reading(14.0, sequence=2, at=1.1))
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
    assert detector.check_stale(now=101.0) == ()
    assert detector.safety_evidence[EventKind.STALE] is not None
    events = detector.check_stale(now=101.5)
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


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("quality", [Quality.OK.value, Quality.STALE.value])
def test_channel_summary_resolves_existing_scalar_and_batch_rings(
    batch: bool, quality: str,
) -> None:
    from dataclasses import replace
    from leapflow.hardware.observability.inventory import build_device_view
    from leapflow.hardware.stream import BatchStreamCoordinator

    context = _context()
    if batch:
        context = replace(context, transport=TransportRef(kind="lhp"))
    registry = _registry(context, persist_readings=False)
    source = registry.stream_sources()[0]
    assert isinstance(source, BatchStreamCoordinator if batch else HardwareStreamSource)
    ring = source.ring("level") if batch else source.ring
    assert ring is not None
    assert registry.channel_summary(context.device_id, "level") is None
    ring.record(_reading(42.0, sequence=1, quality=quality))
    summary = registry.channel_summary(context.device_id, "level")
    assert summary == ring.summary()
    assert summary["latest"] == 42.0 and summary["quality"] == quality
    assert registry.latest_sample(context.device_id, "level") == ring.latest
    assert registry.channel_summary("another_device", "level") is None
    assert registry.channel_summary(context.device_id, "knob") is None
    assert registry.channel_summary(context.device_id, "unknown") is None
    view = build_device_view(registry, context.device_id)
    row = next(ch for ch in view["channels"] if ch["channel_id"] == "level")
    assert row["value"] == 42.0 and row["samples"] == 1 and row["quality"] == quality
    assert registry.opened_devices() == ()


def test_channel_summary_does_not_build_sampling_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry(_context(), persist_readings=False)

    def unexpected_build():
        raise AssertionError("An observation must not build sampling sources")

    monkeypatch.setattr(registry, "stream_sources", unexpected_build)
    assert registry.channel_summary("sampled_device", "level") is None
    assert registry.latest_sample("sampled_device", "level") is None


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


@pytest.mark.parametrize("raw", [
    {"defaults": {"adaptive": True}},
    {"defaults": {"margin_floor": -1}},
    {"defaults": {"soft_max": float("nan")}},
    {"defaults": {"confirm_s": True}},
    {"defaults": {"min_fresh_samples": 32.0}},
    {"defaults": {"unexpected": 1}},
    {"overrides": [{"device_id": "*", "channel_id": "level"}]},
])
def test_observation_rejects_invalid_policy(raw: Any) -> None:
    from leapflow.hardware.observation import ObservationSettings
    with pytest.raises(ValueError):
        ObservationSettings.parse(raw)


def test_fixed_soft_observation_confirmation_recovery_and_recurrence() -> None:
    from leapflow.hardware.observation import ObservationPolicy
    context = _context()
    detector = HardwareEventDetector(context, context.channel("level"),
                                    ObservationPolicy(soft_min=10, soft_max=80, margin_floor=2))
    def sample(value: float, index: int) -> tuple[HardwareEvent, ...]:
        return detector.observe(_reading(value, sequence=index, at=1 + index * 0.1))
    assert not sample(83, 0)
    assert not sample(83, 1)
    entry = sample(83, 2)
    assert [e.kind for e in entry] == [EventKind.OBSERVATION_WARNING]
    assert entry[0].resolved_event_class == "record"
    assert sample(79, 3) == ()  # neither entering nor clearing the band
    for index in range(4, 9):
        assert sample(77, index) == ()
    clear = sample(77, 9)
    assert [e.kind for e in clear] == [EventKind.OBSERVATION_CLEARED]
    assert not sample(83, 10)
    assert not sample(83, 11)
    recurrence = sample(83, 12)
    assert recurrence[0].observation["episode"] != entry[0].observation["episode"]
    assert context.channel("level").envelope.max_value == 100


def test_soft_confirmation_resets_on_invalid_and_missing_evidence() -> None:
    from leapflow.hardware.observation import ObservationPolicy
    context = _context()
    detector = HardwareEventDetector(context, context.channel("level"),
                                    ObservationPolicy(soft_max=80, margin_floor=2))
    detector.observe(_reading(83, sequence=0, at=1))
    detector.observe(_reading(83, sequence=1, at=1.1, quality="stale"))
    for index in range(2):
        assert EventKind.OBSERVATION_WARNING not in [e.kind for e in detector.observe(
            _reading(83, sequence=2 + index, at=1.2 + 0.1 * index))]
    assert EventKind.OBSERVATION_WARNING in [e.kind for e in detector.observe(_reading(83, sequence=4, at=1.4))]
    detector.check_stale(now=2)
    soft = next(row for row in detector.observation_snapshots if row["condition"] == "soft_upper")
    assert soft["active"] and not soft["known"]
    assert all(n["phase"] != "recovery" for n in soft["pending"])


def test_observation_channel_validation_and_atomic_reload() -> None:
    context = _context()
    registry = _registry(context)
    sources = registry.stream_sources()
    valid = '{"overrides":[{"device_id":"sampled_device","channel_id":"level","soft_max":80,"margin_floor":2}]}'
    registry.reconfigure_observation(valid)
    policy = sources[0].observation_detectors[0]._observation.policy
    assert policy.soft_max == 80
    for raw in ('{"defaults":{"soft_max":100,"margin_floor":2}}',
                '{"overrides":[{"device_id":"missing","channel_id":"level"}]}'):
        with pytest.raises(ValueError):
            registry.reconfigure_observation(raw)
        assert sources[0].observation_detectors[0]._observation.policy == policy
        assert registry.settings.hardware_observation == valid


def test_global_poll_sequences_do_not_claim_channel_loss() -> None:
    from dataclasses import replace
    ring = ReadingRing()
    assert ring.record(replace(_reading(1, sequence=1), sequence_scope="device_read")) == 0
    assert ring.record(replace(_reading(1, sequence=99), sequence_scope="device_read")) == 0
    assert ring.dropped == 0
    assert ring.record(replace(_reading(1, sequence=100), acquisition_epoch="reconnect")) == 0


def test_reordered_or_repeated_channel_sequences_do_not_invent_loss() -> None:
    ring = ReadingRing()
    for sequence in (1, 2, 1, 3, 3, 4):
        assert ring.record(_reading(50, sequence=sequence)) == 0
    assert ring.dropped == 0
    assert ring.record(_reading(50, sequence=6)) == 1


def test_circular_rates_and_discontinuous_history() -> None:
    from dataclasses import replace
    context = _context(max_rate=50)
    channel = replace(context.channel("level"), unit="deg", envelope=Envelope(
        declared=True, min_value=-180, max_value=180, max_rate=50))
    detector = HardwareEventDetector(context, channel)
    detector.observe(_reading(179, sequence=1, at=1))
    assert EventKind.RATE_EXCEEDED not in [e.kind for e in detector.observe(_reading(-179, sequence=2, at=1.1))]
    detector.observe(_reading(None, sequence=3, at=1.2))
    assert EventKind.RATE_EXCEEDED not in [e.kind for e in detector.observe(_reading(0, sequence=4, at=1.3))]
    detector.observe(_reading(10, sequence=5, at=1.3))
    assert EventKind.RATE_EXCEEDED not in [e.kind for e in detector.observe(_reading(100, sequence=6, at=1.4))]


def _adaptive_replay(values: list[float], *, adaptive: bool = True, ceiling: float = 8,
                     pending_motion: bool = False, fault_free: bool = True) -> tuple[Any, list[Any]]:
    from leapflow.hardware.observation import ObservationPolicy
    context = _context(sample_rate_hz=100)
    detector = HardwareEventDetector(context, context.channel("level"), ObservationPolicy(
        soft_min=10, soft_max=80, margin_floor=1, margin_max=ceiling, adaptive=adaptive,
        noise_window_s=1, min_fresh_samples=32))
    ring = ReadingRing(128)
    events = []
    for index, value in enumerate(values):
        reading = _reading(value, sequence=index + 1, at=1 + index * .01)
        ring.record(reading)
        events.extend(detector.observe(reading, ring=ring, pending_motion=pending_motion, fault_free=fault_free))
    return detector, events


@pytest.mark.parametrize("amplitude", [0, .1, 1])
def test_synthetic_in_tolerance_replay_has_no_false_transitions(amplitude: float) -> None:
    values = [50 + amplitude * (-1, 0, 1)[index % 3] for index in range(400)]
    for adaptive in (False, True):
        detector, events = _adaptive_replay(values, adaptive=adaptive)
        assert not events
        assert 1 <= detector._observation.margin <= 8
        if adaptive and amplitude == 1:
            assert detector._observation.margin == pytest.approx(3 * 1.4826)


def test_spikes_sustained_excursion_recovery_and_drift_on_identical_replays() -> None:
    values = [50.] * 100 + [90.] + [50.] * 10 + [83.] * 40 + [70.] * 80
    for adaptive in (False, True):
        _, events = _adaptive_replay(values, adaptive=adaptive)
        warnings = [e for e in events if e.kind == EventKind.OBSERVATION_WARNING]
        clears = [e for e in events if e.kind == EventKind.OBSERVATION_CLEARED]
        assert len(warnings) == len(clears) == 1
        assert warnings[0].observation["duration_s"] == pytest.approx(.2)
        assert warnings[0].value == 83  # baseline shift never moves the operational boundary


def test_adaptive_margin_freezes_during_warning_and_resets_on_reconnect() -> None:
    from dataclasses import replace
    detector, _ = _adaptive_replay([[49., 50., 51.][i % 3] for i in range(200)])
    frozen = detector._observation.margin
    ring = ReadingRing()
    for index in range(40):
        reading = _reading(86., sequence=201 + index, at=3 + index * .01)
        ring.record(reading)
        detector.observe(reading, ring=ring, fault_free=True)
        assert detector._observation.margin == frozen
    assert detector._observation.states["soft_upper"].active
    detector.invalidate_observation()
    assert not detector._observation.states["soft_upper"].known
    assert detector._observation.margin == frozen
    healthy, _ = _adaptive_replay([[49., 50., 51.][i % 3] for i in range(200)])
    healthy.observe(replace(_reading(50., sequence=1, at=4), acquisition_epoch="new-source"),
                    ring=ReadingRing(), fault_free=True)
    assert healthy._observation.margin == 1


@pytest.mark.parametrize("pending_motion,fault_free", [(True, True), (False, False)])
def test_unqualified_or_pending_motion_never_trains_noise(pending_motion: bool, fault_free: bool) -> None:
    detector, _ = _adaptive_replay([[49., 50., 51.][i % 3] for i in range(200)], pending_motion=pending_motion, fault_free=fault_free)
    assert detector._observation.margin == 1
    assert detector._observation._training_since is None


def test_rising_noise_quality_warning_is_bounded_and_missing_data_is_not_recovery() -> None:
    detector, events = _adaptive_replay([[46., 50., 54.][i % 3] for i in range(400)], ceiling=5)
    quality = [e for e in events if e.observation.get("condition") == "observation_quality"]
    assert len(quality) == 1 and quality[0].kind == EventKind.OBSERVATION_WARNING
    assert detector._observation.margin == 5
    detector.observe(_reading(None, sequence=401, at=5, quality="stale"))
    state = detector._observation.states["observation_quality"]
    assert state.active and not state.known
    assert not any(n["phase"] == "recovery" for n in detector._observation.pending["observation_quality"])


def test_quality_episode_freezes_margin_through_unknown_and_recovery() -> None:
    detector, _ = _adaptive_replay([[46., 50., 54.][i % 3] for i in range(400)], ceiling=5)
    judgment = detector._observation
    assert judgment.states["observation_quality"].active
    detector.invalidate_observation()
    assert judgment.margin == 5
    ring = ReadingRing(128)
    recovery = None
    for index in range(300):
        reading = _reading(50, sequence=401 + index, at=5 + index * .01)
        ring.record(reading)
        events = detector.observe(reading, ring=ring, fault_free=True)
        if recovery is None:
            assert judgment.margin == 5
        for event in events:
            if (event.kind == EventKind.OBSERVATION_CLEARED
                    and event.observation["condition"] == "observation_quality"):
                recovery = event
                assert event.observation["margin"] == 5
    assert recovery is not None
    assert not judgment.states["observation_quality"].active
    assert judgment.margin == 1  # retraining only after the episode has recovered


def test_sustained_trend_and_hard_boundary_proximity_never_relax_margin() -> None:
    trend, _ = _adaptive_replay([50 + index * .05 for index in range(400)])
    assert trend._observation.margin == 1
    boundary, _ = _adaptive_replay([1., 2.] * 100)
    assert boundary._observation.margin == 1


def test_consecutive_hard_safety_uses_samples_not_breach_entry_events() -> None:
    from leapflow.hardware.alert_policy import AlertRule, HardwareAlertPolicy
    from leapflow.hardware.stream import _observe_sample
    context = _context(max_rate=5)
    detector = HardwareEventDetector(context, context.channel("level"))
    policy = HardwareAlertPolicy([AlertRule(event_kind="threshold_exceeded", action="hw_estop", require_consecutive=3)])
    ring = ReadingRing()
    emitted = []
    for index, value in enumerate([101, 101, 101, 101, 50, 101, 101, 101]):
        reading = _reading(value, sequence=index + 1, at=1 + index * .1)
        lost = ring.record(reading)
        emitted.extend(_observe_sample(detector, reading, ring, None, policy, lost=lost))
        if index == 2:
            assert policy.fired_count == 1
    assert policy.fired_count == 2
    assert sum(e.kind == EventKind.THRESHOLD_EXCEEDED for e in emitted) == 2


def test_record_only_observation_cannot_be_mapped_to_reflex_action() -> None:
    from leapflow.hardware.alert_policy import AlertRule, HardwareAlertPolicy
    policy = HardwareAlertPolicy([AlertRule(event_kind="observation_warning", action="hw_estop")])
    policy.evaluate(HardwareEvent(kind="observation_warning", device_id="d", channel_id="c", quantity="q", detail="reported only"))
    assert policy.fired_count == 0


def test_true_hard_recurrence_survives_presentation_pacing() -> None:
    context = _context()
    detector = HardwareEventDetector(context, context.channel("level"))
    source = HardwareStreamSource(None, context, context.channel("level"))
    emitted = []
    for index, value in enumerate([101, 50, 101, 50]):
        source._dispatch(detector.observe(_reading(value, sequence=index + 1, at=1 + index * .1)), emitted.append)
    assert [e.kind for e in emitted] == ["threshold_exceeded", "settled"] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("adaptive", [False, True])
async def test_scalar_batch_sampling_policy_parity(adaptive: bool) -> None:
    from types import SimpleNamespace
    from leapflow.hardware.stream import BatchStreamCoordinator
    from leapflow.hardware.observation import ObservationSettings
    values = [49., 50., 51.] * 40 + [90.] * 10 + [50.] * 20
    context = _context(sample_rate_hz=1000)
    tuning = ObservationSettings.parse({"overrides": [{"device_id": context.device_id, "channel_id": "level",
        "soft_max": 80, "margin_floor": 1, "margin_max": 8, "adaptive": adaptive,
        "noise_window_s": .05, "confirm_s": .002, "recover_s": .005}]})
    outputs = []
    for batch_mode in (False, True):
        class ReplayIO:
            index = 0
            def device_io(self, device_id: str) -> Any:
                return asyncio.Lock()
            async def transport(self, device_id: str) -> Any:
                return self
            def observation_policy(self, device_id: str, channel: Any) -> Any:
                return tuning.resolve(device_id, channel)
            def observation_fault_free(self, device_id: str) -> bool:
                return True
            async def read(self, channel_id: str) -> Reading:
                index = self.index
                self.index += 1
                if self.index == len(values):
                    source._stopping.set()
                return _reading(values[index], sequence=index + 1, at=1 + index * .001)
            async def read_batch(self, device_id: str, channel_ids: Any) -> Any:
                return SimpleNamespace(readings=(await self.read("level"),))
        io = ReplayIO()
        source = (BatchStreamCoordinator(io, context, (context.channel("level"),)) if batch_mode
                  else HardwareStreamSource(io, context, context.channel("level")))
        events = []
        await asyncio.wait_for(source._run(events.append), timeout=2)
        outputs.append([(e.kind, e.value, e.observation.get("duration_s")) for e in events])
    assert outputs[0] == outputs[1]
    assert [row[0] for row in outputs[0]] == ["observation_warning", "observation_cleared"]


async def _sampling_cadence_probe(path: Any, *, adaptive: bool = False, factory: Any = HardwareStreamSource) -> dict[str, Any]:
    """Offline cadence probe through real acquisition, buffering, and DuckDB writes."""
    import json

    context = _context(sample_rate_hz=500)
    registry = _registry(context, instrument_db_path=str(path), downsample_interval_s=1,
        hardware_observation=json.dumps({"overrides": [{
            "device_id": context.device_id, "channel_id": "level", "soft_max": 80,
            "margin_floor": 1, "margin_max": 8, "adaptive": adaptive,
        }]}))
    store = registry.reading_store
    source = factory(registry, context, context.channel("level"), reading_store=store)
    events = []
    try:
        await source.start(events.append)
        await asyncio.sleep(2.2)
        await source.stop()
        return {**source.health, "windows": store.windows_written, "events": len(events)}
    finally:
        await source.stop()
        await registry.close_all()
        registry._instrument_conn.close()


@pytest.mark.asyncio
async def test_observation_modes_preserve_scheduled_acquisition_and_persistence(tmp_path: Any) -> None:
    for adaptive in (False, True):
        result = await _sampling_cadence_probe(tmp_path / f"cadence-{adaptive}.duckdb", adaptive=adaptive)
        assert result["windows"] > 0
        assert result["events"] == 0
        # Leave headroom for shared CI scheduling; this is a regression guard,
        # not qualification of a physical device's maximum acquisition rate.
        assert result["observed_hz"] >= result["declared_hz"] * .8


def test_stale_level_evidence_fires_before_notification_reminders(monkeypatch: Any) -> None:
    from leapflow.hardware.alert_policy import AlertRule, HardwareAlertPolicy
    from leapflow.hardware.stream import _stale_observation
    context = _context()
    detector = HardwareEventDetector(context, context.channel("level"))
    policy = HardwareAlertPolicy([AlertRule(event_kind="stale", action="hw_estop", require_consecutive=3)])
    detector.observe(_reading(50, sequence=1, at=1))
    monkeypatch.setattr("leapflow.hardware.stream.time.monotonic", lambda: 1.3)
    assert not _stale_observation(detector, policy)
    assert not _stale_observation(detector, policy)
    assert not _stale_observation(detector, policy)
    assert policy.fired_count == 1
    assert [e.observation["phase"] for e in detector.check_stale(now=1.8)] == ["entry"]
    assert [e.observation["phase"] for e in detector.check_stale(now=61.8)] == ["reminder"]
    detector.observe(_reading(50, sequence=2, at=62))
    assert not detector._observation.states["stale"].active
    assert detector._observation.pending["stale"][-1]["phase"] == "recovery"


def test_sampling_silence_breaks_other_consecutive_safety_runs(monkeypatch: Any) -> None:
    from leapflow.hardware.alert_policy import AlertRule, HardwareAlertPolicy
    from leapflow.hardware.stream import _observe_sample, _stale_observation
    context = _context()
    detector = HardwareEventDetector(context, context.channel("level"))
    policy = HardwareAlertPolicy([AlertRule(
        event_kind="threshold_exceeded", action="hw_estop", require_consecutive=3)])
    ring = ReadingRing()
    for index in range(2):
        reading = _reading(101, sequence=index + 1, at=1 + index * .1)
        _observe_sample(detector, reading, ring, None, policy, lost=ring.record(reading))
    monkeypatch.setattr("leapflow.hardware.stream.time.monotonic", lambda: 1.2)
    _stale_observation(detector, policy)  # acquisition failed, even before stale deadline
    assert policy.fired_count == 0
    for index in range(3):
        reading = _reading(101, sequence=3 + index, at=1.3 + index * .1)
        _observe_sample(detector, reading, ring, None, policy, lost=ring.record(reading))
        assert policy.fired_count == (1 if index == 2 else 0)


def test_missing_numeric_values_do_not_hide_explicit_stale_quality(monkeypatch: Any) -> None:
    context = _context()
    detector = HardwareEventDetector(context, context.channel("level"))
    monkeypatch.setattr("leapflow.hardware.stream.time.monotonic", lambda: 1.0)
    detector.observe(_reading(None, sequence=1, at=0, quality="stale"))
    state = detector._observation.states["stale"]
    assert not state.active and state.known
    assert detector.safety_evidence[EventKind.STALE] is not None
    monkeypatch.setattr("leapflow.hardware.stream.time.monotonic", lambda: 1.5)
    detector.observe(_reading(None, sequence=2, at=0, quality="stale"))
    assert state.active and state.known
    assert detector._observation.latest["stale"]["duration_s"] == pytest.approx(.5)
    assert not detector._observation.states["quality_degraded"].active
    detector.observe(_reading({"status": "fresh"}, sequence=3, at=1.6))
    assert not state.active and state.known


def test_startup_stale_clears_before_confirmation(monkeypatch: Any) -> None:
    detector = _detector()
    monkeypatch.setattr("leapflow.hardware.stream.time.monotonic", lambda: 100.0)
    for sequence in range(3):
        assert not detector.observe(_reading(None, sequence=sequence, quality="stale"))
    detector.observe(_reading(20, sequence=4, at=100.11))
    assert not detector._observation.states["stale"].active
    assert not detector._observation.pending.get("stale")
    assert not detector._observation.pending.get("quality_degraded")


@pytest.mark.parametrize("at,quality,expected", [
    (12.0, "ok", 12.0), (0.0, "ok", 100.0),
    (float("nan"), "ok", 100.0), (float("inf"), "ok", 100.0),
    (-1.0, "ok", 100.0), (12.0, "stale", 100.0),
])
def test_observation_clock_uses_only_trustworthy_positive_time(
    monkeypatch: Any, at: float, quality: str, expected: float,
) -> None:
    from leapflow.hardware.stream import _observation_now
    monkeypatch.setattr("leapflow.hardware.stream.time.monotonic", lambda: 100.0)
    assert _observation_now(_reading(None, sequence=1, at=at, quality=quality)) == expected


@pytest.mark.parametrize("rate,window", [(10, .5), (1, 2), (0, .5)])
def test_stale_confirmation_respects_declared_rate(rate: float, window: float) -> None:
    from leapflow.hardware.observation import ObservationPolicy
    assert ObservationPolicy().for_channel(_context(sample_rate_hz=rate).channel("level")).stale_confirm_s == window


def test_calibration_identity_reset_preserves_notices_and_is_idempotent() -> None:
    from leapflow.hardware.observation import ObservationPolicy
    context = _context()
    detector = HardwareEventDetector(context, context.channel("level"), ObservationPolicy(
        soft_max=80, margin_floor=1, confirm_s=0))
    detector.observe(_reading(83, sequence=1, at=1))
    before = detector.observation_snapshots
    old_identity = detector._observation.identity
    detector.invalidate_observation(identity="qualified-generation-2")
    snapshot = next(row for row in detector.observation_snapshots if row["condition"] == "soft_upper")
    assert not snapshot["known"] and not snapshot["active"]
    assert snapshot["identity"] != old_identity
    assert snapshot["pending"][0]["identity"] == old_identity
    assert snapshot["pending"][0] == next(row for row in before if row["condition"] == "soft_upper")["pending"][0]
    detector.observe(_reading(83, sequence=2, at=1.1))
    episode = detector._observation.states["soft_upper"].episode
    detector.invalidate_observation(identity="qualified-generation-2")
    assert detector._observation.states["soft_upper"].episode == episode


@pytest.mark.component
def test_persisted_calibration_identity_is_initialized_without_restoring_health(tmp_path: Any) -> None:
    from leapflow.hardware.calibration_store import CalibrationRecord
    from leapflow.hardware.observation import ObservationPolicy
    path = str(tmp_path / "instrument.duckdb")
    context = _context()
    registry = _registry(context, instrument_db_path=path)
    record = CalibrationRecord(context.device_id, "zero", 123, {"qualified": "software test"})
    try:
        assert registry.calibration_store.record(record)
        registry.record_event(HardwareEvent(
            kind="calibration_completed", device_id=context.device_id, channel_id="level",
            quantity="q", detail="software test", value={
                "procedure_id": record.procedure_id, "recorded_at": record.recorded_at}))
        detector = registry.stream_sources()[0].observation_detectors[0]
        before = detector._observation.identity
        detector.reconfigure_observation(ObservationPolicy(soft_max=80, margin_floor=1, confirm_s=0))
        detector.observe(_reading(83, sequence=1, at=1))
        assert detector._observation.states["soft_upper"].active
    finally:
        registry._instrument_conn.close()
    restarted = _registry(context, instrument_db_path=path)
    try:
        fresh = restarted.stream_sources()[0].observation_detectors[0]
        assert fresh._observation.identity == before
        assert fresh._observation.margin == 0
        assert fresh.observation_snapshots == ()
    finally:
        restarted._instrument_conn.close()


@pytest.mark.component
@pytest.mark.parametrize("kind", ["activated", "invalidated", "rolled_back"])
def test_generation_identity_matches_lifecycle_evidence_after_restart(tmp_path: Any, kind: str) -> None:
    from leapflow.hardware.calibration.lifecycle import project_decision
    context = _context()
    path = str(tmp_path / "instrument.duckdb")
    registry = _registry(context, instrument_db_path=path)
    try:
        store = registry.calibration_generations
        generation = store.commit_generation(
            workcell_id="cell", request_id="software-test", expected_generation=None,
            dependency_digest="deps", model={"device_id": context.device_id, "synthetic": True},
            accepted_blocks={"test": {"synthetic": True}}, causal={"software_test": True})
        if kind == "invalidated":
            store.invalidate(workcell_id="cell", request_id="invalidate-test",
                expected_generation=generation.generation, dependency_digest="deps",
                causal={"software_test": True})
        elif kind == "rolled_back":
            store.commit_generation(
                workcell_id="cell", request_id="second-software-test", expected_generation=1,
                dependency_digest="deps", model={"device_id": context.device_id, "synthetic": 2},
                accepted_blocks={"test": {"synthetic": True}}, causal={"software_test": True})
            store.rollback(workcell_id="cell", request_id="rollback-test", expected_generation=2,
                dependency_digest="deps", causal={"software_test": True})
        decision = next(event for event in store.pending_events() if event["kind"] == kind)
        registry.record_event(HardwareEvent(
            kind=f"calibration_{decision['kind']}", device_id=context.device_id,
            channel_id="calibration_lifecycle", quantity="q", detail="software test",
            value=project_decision(decision, context.device_id)))
        before = registry.stream_sources()[0].observation_detectors[0]._observation.identity
    finally:
        registry._instrument_conn.close()
    restarted = _registry(context, instrument_db_path=path)
    try:
        detector = restarted.stream_sources()[0].observation_detectors[0]
        assert detector._observation.identity == before
        assert detector.observation_snapshots == ()
    finally:
        restarted._instrument_conn.close()


def test_notice_overflow_is_bounded_disclosed_and_durable_notices_are_retired() -> None:
    from leapflow.hardware.observation import MAX_PENDING_TRANSITIONS
    context = _context()
    detector = HardwareEventDetector(context, context.channel("level"))
    for index in range(40):
        detector.observe(_reading(101 if index % 2 == 0 else 50, sequence=index + 1, at=1 + index * .1))
    row = next(row for row in detector.observation_snapshots if row["condition"] == "threshold_exceeded")
    assert len(row["pending"]) == MAX_PENDING_TRANSITIONS
    assert row["pending_omitted"] == 40 - MAX_PENDING_TRANSITIONS
    registry = _registry(context)
    registered = registry.stream_sources()[0].observation_detectors[0]
    registered.observe(_reading(101, sequence=1, at=1))
    notice = registered._observation.pending["threshold_exceeded"][0]
    registry.acknowledge_observation({notice["key"]: {**notice, "runtime_id": "another-runtime"}})
    assert registered._observation.pending["threshold_exceeded"]
    registry.acknowledge_observation({notice["key"]: notice})
    assert not registered._observation.pending["threshold_exceeded"]
    assert registered._observation.states["threshold_exceeded"].active


@pytest.mark.parametrize("patch,expected", [
    ({}, True), ({"settled": False}, False), ({"receive_qualified": False}, False),
    ({"fault_free": False}, False), ({"status": {"error_code": 1}}, False),
    ({"motors": [{"faults": {"overheat": True}}]}, False),
    ({"motors": [{"faults": {"overheat": None}}]}, False),
])
def test_noise_training_requires_complete_fresh_fault_and_settled_evidence(patch: Any, expected: bool,
                                                                        monkeypatch: Any) -> None:
    from dataclasses import replace
    registry = _registry(_context())
    value = {"receive_qualified": True, "settled": True, "fault_free": True, **patch}
    reading = _reading(value, sequence=1, at=1)
    monkeypatch.setattr("leapflow.hardware.registry.time.monotonic", lambda: 1.1)
    registry.note_observation(reading)
    assert registry.observation_fault_free("sampled_device") is expected
    registry.note_observation(replace(reading, monotonic_at=2))
    assert not registry.observation_fault_free("sampled_device")
    registry.note_observation(replace(reading, monotonic_at=0))
    assert not registry.observation_fault_free("sampled_device")


@pytest.mark.asyncio
async def test_batch_judges_every_hard_guard_before_persistence() -> None:
    from dataclasses import replace
    from types import SimpleNamespace
    from leapflow.hardware.stream import BatchStreamCoordinator
    from leapflow.hardware.alert_policy import AlertRule, HardwareAlertPolicy
    base = _context()
    context = replace(base, channels=(base.channel("level"), replace(base.channel("level"), channel_id="other")))
    class ReplayIO:
        async def read_batch(self, *_: Any) -> Any:
            source._stopping.set()
            return SimpleNamespace(readings=(_reading(50, sequence=1, at=1), replace(_reading(101, sequence=1, at=1), channel_id="other")))
    observed_guards = []
    class Store:
        calls = 0
        def record(self, *_: Any, **__: Any) -> None:
            self.calls += 1
            observed_guards.append(policy.fired_count)
            raise RuntimeError("persistence is unavailable")
    policy = HardwareAlertPolicy([AlertRule(event_kind="threshold_exceeded", action="hw_estop", require_consecutive=1)])
    source = BatchStreamCoordinator(ReplayIO(), context, context.channels, alert_policy=policy, reading_store=Store())
    await asyncio.wait_for(source._run(lambda _: None), timeout=2)
    assert policy.fired_count == 1
    assert source._store.calls == 2
    assert observed_guards == [1, 1]


@pytest.mark.parametrize("step", [.02, .04, .06])
def test_rate_flapping_never_confirms_but_raw_safety_stays_immediate(step: float) -> None:
    ctx = _context(sample_rate_hz=10, max_rate=1)
    detector = HardwareEventDetector(ctx, ctx.channel("level"))
    notices, raw = [], 0
    value = 20.0
    for index in range(300):
        if index % 2:
            value = 21.0 if value == 20.0 else 20.0
        events = detector.observe(_reading(value, sequence=index, at=10 + index * step))
        raw += detector.safety_evidence[EventKind.RATE_EXCEEDED] is not None
        notices.extend(e.observation for e in events if e.observation.get("condition") == "rate_exceeded")
    assert raw == 150
    assert not notices
    assert not detector._observation.states["rate_exceeded"].active


def test_rate_dwell_uses_acquisition_clock_and_delays_recovery() -> None:
    from dataclasses import replace
    ctx = _context(sample_rate_hz=50, max_rate=1)
    detector = HardwareEventDetector(ctx, ctx.channel("level"))
    notices = []
    for index in range(27):
        reading = replace(_reading(20 + index * .1, sequence=index, at=1000 + index * .0001),
                          acquisition_at=100 + index * .02, acquisition_epoch="node", rate_timebase="source")
        events = detector.observe(reading)
        notices.extend(e.observation for e in events if e.observation.get("condition") == "rate_exceeded")
    assert [n["phase"] for n in notices] == ["entry"]
    assert notices[0]["duration_s"] == pytest.approx(.5)
    assert notices[0]["measured_rate"] == pytest.approx(5)
    for index in range(27, 78):
        reading = replace(_reading(22.6, sequence=index, at=1000 + index * .0001),
                          acquisition_at=100 + index * .02, acquisition_epoch="node", rate_timebase="source")
        events = detector.observe(reading)
        notices.extend(e.observation for e in events if e.observation.get("condition") == "rate_exceeded")
        if index < 77:
            assert detector._observation.states["rate_exceeded"].active
    assert [n["phase"] for n in notices] == ["entry", "recovery"]
    assert not detector._observation.states["rate_exceeded"].active


@pytest.mark.parametrize("invalid", ["gap", "epoch", "timebase", "sequence", "quality"])
def test_rate_gap_requalifies_same_episode_without_confirming_unknown(invalid: str) -> None:
    from dataclasses import replace
    ctx = _context(sample_rate_hz=10, max_rate=1)
    detector = HardwareEventDetector(ctx, ctx.channel("level"))
    for index in range(7):
        detector.observe(_reading(20 + index, sequence=index, at=10 + index * .1))
    state = detector._observation.states["rate_exceeded"]
    assert state.active and state.known
    episode = state.episode
    reading = _reading(27, sequence=7, at=10.7 if invalid != "gap" else 11)
    patches = {"epoch": {"acquisition_epoch": "new"}, "timebase": {"rate_timebase": "unavailable"},
               "quality": {"quality": "stale"}}
    reading = replace(reading, **patches.get(invalid, {}))
    events = detector.observe(reading, lost=1 if invalid == "sequence" else 0)
    assert not state.known
    assert any(e.kind == EventKind.OBSERVATION_UNKNOWN for e in events)
    notices = []
    for index in range(8, 15):
        events = detector.observe(replace(_reading(20 + index, sequence=index,
                                                  at=reading.monotonic_at + (index - 7) * .1),
                                         acquisition_epoch=reading.acquisition_epoch))
        notices.extend(e.observation for e in events if e.observation.get("condition") == "rate_exceeded")
        if index < 13:
            assert not state.known
    assert state.known and state.active and state.episode == episode
    assert [n["phase"] for n in notices] == ["entry"]
    assert notices[0]["reconfirmed"]
