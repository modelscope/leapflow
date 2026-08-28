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


def _reading(value: Any, *, sequence: int, timestamp: float = 0.0, quality: str = Quality.OK.value):
    return Reading(
        device_id="sampled_device",
        channel_id="level",
        value=value,
        quantity="generic.level",
        unit="unit",
        timestamp=timestamp,
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
    assert detector.observe(_reading(50.0, sequence=1, timestamp=1.0)) == ()
    first = detector.observe(_reading(150.0, sequence=2, timestamp=2.0))
    assert [e.kind for e in first] == [EventKind.THRESHOLD_EXCEEDED]
    assert detector.observe(_reading(160.0, sequence=3, timestamp=3.0)) == ()


def test_returning_to_range_is_reported_as_recovery() -> None:
    """Recovery must be observable, not inferred from silence."""
    detector = _detector()
    detector.observe(_reading(150.0, sequence=1, timestamp=1.0))
    events = detector.observe(_reading(50.0, sequence=2, timestamp=2.0))
    assert [e.kind for e in events] == [EventKind.SETTLED]


def test_rate_event_uses_the_declared_max_rate() -> None:
    detector = _detector(max_rate=5.0)
    detector.observe(_reading(10.0, sequence=1, timestamp=1.0))
    # 40 units in one second, against a declared 5/s.
    events = detector.observe(_reading(50.0, sequence=2, timestamp=2.0))
    assert EventKind.RATE_EXCEEDED in [e.kind for e in events]


def test_no_rate_event_without_a_declared_limit() -> None:
    """No rule may exist that a human did not write down."""
    detector = _detector(max_rate=None)
    detector.observe(_reading(10.0, sequence=1, timestamp=1.0))
    events = detector.observe(_reading(90.0, sequence=2, timestamp=2.0))
    assert EventKind.RATE_EXCEEDED not in [e.kind for e in events]


def test_sample_loss_is_reported() -> None:
    detector = _detector()
    events = detector.observe(_reading(20.0, sequence=5, timestamp=1.0), lost=3)
    assert [e.kind for e in events] == [EventKind.SAMPLE_LOSS]
    assert "3 sample" in events[0].detail


def test_quality_degradation_needs_a_streak() -> None:
    """One suspect sample is noise; a run of them is a fault."""
    detector = _detector()
    kinds: list[str] = []
    for index in range(3):
        events = detector.observe(
            _reading(20.0, sequence=index, timestamp=float(index), quality=Quality.SUSPECT.value)
        )
        kinds.extend(e.kind for e in events)
    assert kinds.count(EventKind.QUALITY_DEGRADED) == 1


def test_quality_streak_resets_on_a_good_sample() -> None:
    detector = _detector()
    detector.observe(_reading(20.0, sequence=1, timestamp=1.0, quality=Quality.SUSPECT.value))
    detector.observe(_reading(20.0, sequence=2, timestamp=2.0, quality=Quality.SUSPECT.value))
    detector.observe(_reading(20.0, sequence=3, timestamp=3.0))
    events = detector.observe(
        _reading(20.0, sequence=4, timestamp=4.0, quality=Quality.SUSPECT.value)
    )
    assert EventKind.QUALITY_DEGRADED not in [e.kind for e in events]


def test_silence_on_a_declared_rate_is_itself_an_observation() -> None:
    """A 10 Hz channel that says nothing for a second has failed."""
    detector = _detector(sample_rate_hz=10.0)
    detector.observe(_reading(20.0, sequence=1, timestamp=100.0))
    assert detector.check_stale(now=100.05) == ()
    events = detector.check_stale(now=101.0)
    assert [e.kind for e in events] == [EventKind.STALE]
    # Reported once, not on every check.
    assert detector.check_stale(now=102.0) == ()


def test_staleness_does_not_apply_to_an_unsampled_channel() -> None:
    detector = _detector(sample_rate_hz=0.0)
    detector.observe(_reading(20.0, sequence=1, timestamp=100.0))
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
async def test_events_cross_the_boundary_as_interaction_signals() -> None:
    """Only derived events reach the signal pipeline, and in its own type."""
    from leapflow.perception.types import InteractionSignal

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

    assert emitted, "a threshold excursion should have produced a signal"
    signal = emitted[0]
    assert isinstance(signal, InteractionSignal)
    assert signal.signal_type == "hw_event"
    assert signal.app == "sampled_device"
    assert "threshold_exceeded" in signal.detail


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
