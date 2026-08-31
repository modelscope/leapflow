"""The path a derived hardware event actually travels, end to end.

Every assertion here checks a *connection*, not a capability. The defect this file
exists to prevent shipped with a green suite: the detector was correct, the event type
was correct, the conversion helper was correct -- and the sink was ``None`` at the one
call site that mattered, so nothing downstream ever saw an event. Asserting that a
converter exists proves nothing about whether anything calls it.

Three links are covered:

1. the emitter is built from the wiring layer and publishes to the bus it was given;
2. both normalizers pass ``hw.*`` through instead of collapsing it to
   ``internal.unmapped``, which would silently destroy the family;
3. the family the board groups on comes out as ``hw``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from leapflow.dashboard.service import _event_family
from leapflow.domain.events import PRE_NORMALIZED_EVENT_PREFIXES
from leapflow.hardware.stream import EventKind, HardwareEvent


def _event(kind: str = EventKind.THRESHOLD_EXCEEDED) -> HardwareEvent:
    return HardwareEvent(
        kind=kind,
        device_id="bench",
        channel_id="temperature",
        quantity="thermal.temperature",
        detail="left the declared range (0..100)",
        value=140.5,
        unit="C",
        observed_at=1_787_000_000.0,
    )


class _RecordingBus:
    """Minimal stand-in for EventBus, capturing exactly what the emitter publishes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def handle_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.calls.append((event_type, payload))


class _Wiring:
    """Borrow the production wiring methods without constructing a whole Context.

    Bound off the real class so a change to either method is exercised here rather
    than against a copy of it that can drift.
    """

    from leapflow.cli.context import Context as _Context

    _hardware_event_emitter = _Context._hardware_event_emitter
    _start_hardware_streams = _Context._start_hardware_streams

    def __init__(self, event_bus: Any, registry: Any = None) -> None:
        self.event_bus = event_bus
        self._hardware_registry = registry

    def _bind_hardware_persistence(self, registry: Any) -> None:
        """Stubbed: persistence paths are covered by the reading-store tests."""


class _RecordingRegistry:
    """Captures the sink the wiring layer installs, and whether sampling started."""

    def __init__(self) -> None:
        self.emit: Any = "<never installed>"
        self.started = False

    def set_event_emitter(self, emit: Any) -> None:
        self.emit = emit

    async def start_streams(self) -> int:
        self.started = True
        return 1


# ════════════════════════════════════════════════════════════════
# Link 1: the wiring layer supplies a real sink
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_emitter_publishes_the_event_to_the_bus() -> None:
    """The sink handed to ``start_streams`` must actually reach the bus.

    This is the assertion whose absence let the whole path stay dead.
    """
    bus = _RecordingBus()
    emit = _Wiring(bus)._hardware_event_emitter()
    assert emit is not None, "a bus is present, so a sink must be produced"

    emit(_event())
    # The handoff is a task because ingestion is async and sampling is not.
    await asyncio.sleep(0)

    assert len(bus.calls) == 1
    event_type, payload = bus.calls[0]
    assert event_type == "hw.threshold_exceeded"
    assert payload["device_id"] == "bench"
    assert payload["channel_id"] == "temperature"
    assert payload["value"] == 140.5
    assert payload["ts"] == 1_787_000_000.0
    assert "_mono_ts" in payload


@pytest.mark.asyncio
async def test_stream_startup_installs_the_emitter_on_the_registry() -> None:
    """The one assertion that fails if the sink stops being handed over.

    Every other test here proves the emitter *works*. None of them proved anybody
    *installs* it: dropping the handover left this file entirely green, which is the
    same class of blind spot -- asserting the callable rather than the call site --
    that left the whole path dead in the first place.

    Asserted on the installation rather than on an argument to ``start_streams``,
    because that is now the single place the connection is made: the command path
    reports through the same sink, and it must work even when streaming is disabled
    and no source is ever started.
    """
    registry = _RecordingRegistry()
    await _Wiring(_RecordingBus(), registry)._start_hardware_streams()

    assert registry.emit is not None and callable(registry.emit), (
        "no sink was installed; hardware events would be recorded for hw_status and "
        "reach nothing else"
    )
    assert registry.started is True, "sampling must still be started"


@pytest.mark.asyncio
async def test_stream_startup_is_a_no_op_without_a_registry() -> None:
    """Hardware is off by default; startup must not depend on it existing."""
    await _Wiring(_RecordingBus(), None)._start_hardware_streams()


@pytest.mark.asyncio
async def test_a_registry_that_cannot_sample_does_not_stop_initialization() -> None:
    """A bench that fails to start must not prevent the process coming up."""

    class _BrokenRegistry:
        async def start_streams(self, emit: Any = None) -> int:
            raise RuntimeError("bus not present")

    await _Wiring(_RecordingBus(), _BrokenRegistry())._start_hardware_streams()


def test_emitter_is_absent_rather_than_broken_without_a_bus() -> None:
    """No bus means no sink, not a sink that raises inside the sampling loop.

    Sampling must still run and still record events for ``hw_status``: losing the
    push is a degradation, losing the samples is an outage.
    """
    assert _Wiring(None)._hardware_event_emitter() is None
    assert _Wiring(object())._hardware_event_emitter() is None


@pytest.mark.asyncio
async def test_a_failing_bus_does_not_escape_into_the_sampling_loop() -> None:
    """Dispatch contains sink failures; the loop that produced the event must survive."""

    class _AngryBus:
        async def handle_event(self, event_type: str, payload: dict[str, Any]) -> None:
            raise RuntimeError("ingestion is down")

    emit = _Wiring(_AngryBus())._hardware_event_emitter()
    assert emit is not None
    emit(_event())
    # The task raises, not the caller. Draining it here keeps the failure from
    # surfacing as an unretrieved-exception warning in unrelated tests.
    await asyncio.sleep(0)
    tasks = [t for t in asyncio.all_tasks() if t.get_name().startswith("hw-event:")]
    for task in tasks:
        with pytest.raises(RuntimeError):
            await task


# ════════════════════════════════════════════════════════════════
# Link 2: normalization preserves the type
# ════════════════════════════════════════════════════════════════


def test_hardware_prefix_is_registered_as_pre_normalized() -> None:
    """The shared list is the single place this is decided.

    Two normalizers consult it. Held as one constant so adding a producer cannot
    leave a matching pair half-updated -- which is how one of them would keep
    collapsing a family the other preserved.
    """
    assert "hw." in PRE_NORMALIZED_EVENT_PREFIXES


def test_event_bus_fallback_keeps_the_hardware_event_type() -> None:
    """Without a manifest-driven normalizer, the type must still survive.

    An unlisted type becomes ``internal.unmapped``: watch triggers stop matching and
    the board loses the family, with no error anywhere.
    """
    from leapflow.platform.event_bus import EventBus

    bus = EventBus(immediate=None, working=None)  # type: ignore[arg-type]
    event = _event()
    normalized = bus._fallback_normalize(event.event_type, event.to_payload())

    assert normalized.event_type == "hw.threshold_exceeded"
    assert normalized.source == "bench.temperature"
    assert normalized.timestamp == 1_787_000_000.0


def test_manifest_normalizer_keeps_the_hardware_event_type() -> None:
    """Same guarantee on the configured path, which is what production uses."""
    from leapflow.domain.platform import PlatformManifest
    from leapflow.platform.normalizer import EventNormalizer

    normalizer = EventNormalizer(PlatformManifest.default_darwin())
    event = _event()
    normalized = normalizer.normalize(event.event_type, event.to_payload())

    assert normalized.event_type == "hw.threshold_exceeded"
    assert normalized.source == "bench.temperature"


# ════════════════════════════════════════════════════════════════
# Link 3: the board groups it correctly
# ════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "kind",
    [
        EventKind.THRESHOLD_EXCEEDED,
        EventKind.RATE_EXCEEDED,
        EventKind.STALE,
        EventKind.SAMPLE_LOSS,
        EventKind.QUALITY_DEGRADED,
        EventKind.SETTLED,
    ],
)
def test_every_kind_lands_in_the_hardware_family(kind: str) -> None:
    """One family for all kinds, derived from the type rather than enumerated.

    The board's grouping splits on the first separator, so ``hw.<kind>`` needs no
    registration anywhere -- a new kind is visible the day it is added.
    """
    assert _event_family(_event(kind).event_type) == "hw"
