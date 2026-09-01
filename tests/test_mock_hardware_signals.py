"""Contract tests for the mock ``HardwareSignalGenerator``.

The mock signal framework lives outside ``src/`` and deliberately does not import
the production hardware types, so nothing structurally forces its payloads to stay
aligned with what the real pipeline consumes. These tests are that missing force:
they pin the generator's payload keys against the field sets of the production
``Reading`` and ``HardwareEvent`` so a rename on either side breaks loudly here
rather than silently in an integration run.

The field sets are derived from the production classes at test time rather than
hard-coded, so the contract tracks the source of truth instead of a copy of it.
"""

from __future__ import annotations

from leapflow.hardware.stream import HardwareEvent
from leapflow.hardware.transport import Reading

from tests.mock_signals.generators import (
    GENERATOR_REGISTRY,
    HardwareChannelSpec,
    HardwareSignalGenerator,
    SignalConfig,
)
from tests.mock_signals.profiles import PROFILES


# ── Field sets derived from the production types (source of truth) ──


def _reading_payload_fields() -> set[str]:
    """Return ``Reading.to_dict()`` keys plus the mock-only ``monotonic_at``.

    ``monotonic_at`` is intentionally absent from ``Reading.to_dict()`` -- it is
    persistence-facing and wall-clock only -- but the mock adds it back for the
    test pipeline's monotonic ordering, so the contract expects it here.
    """
    reading = Reading(device_id="d", channel_id="c", value=1.0)
    return set(reading.to_dict().keys()) | {"monotonic_at"}


def _hw_event_payload_fields() -> set[str]:
    """Return ``HardwareEvent.to_payload()`` keys."""
    event = HardwareEvent(
        kind="threshold_exceeded",
        device_id="d",
        channel_id="c",
        quantity="temperature",
        detail="left the declared range",
    )
    return set(event.to_payload().keys())


def _collect_first(
    generator: HardwareSignalGenerator, reading_type: str, event_type: str
) -> tuple[dict[str, object], dict[str, object]]:
    """Drive ``generate()`` until one reading and one hardware event are seen.

    Returns ``(reading_payload, event_payload)``. Raises if the generator's
    bounded run ends before both families appear -- that itself is a contract
    failure worth surfacing.
    """
    reading_payload: dict[str, object] | None = None
    event_payload: dict[str, object] | None = None
    for evt_type, payload in generator.generate():
        if evt_type == "__wait__":
            continue
        if evt_type == reading_type and reading_payload is None:
            reading_payload = payload
        elif evt_type == event_type and event_payload is None:
            event_payload = payload
        if reading_payload is not None and event_payload is not None:
            break
    if reading_payload is None or event_payload is None:
        raise AssertionError(
            "generate() ended before yielding both a reading and a hardware event"
        )
    return reading_payload, event_payload


# ── Registry and profile wiring ──


def test_registry_resolves_hardware_signal_generator() -> None:
    """The registry name must resolve to the concrete generator class."""
    assert GENERATOR_REGISTRY["HardwareSignalGenerator"] is HardwareSignalGenerator


def test_hardware_profile_uses_the_hardware_generator() -> None:
    """Every generator in the ``hardware`` profile must be the hardware one.

    The profile exists to exercise the hardware path specifically, so a stray
    non-hardware generator would silently weaken the scenario.
    """
    profile = PROFILES["hardware"]
    assert profile.generators, "the hardware profile must declare at least one generator"
    names = {name for name, _ in profile.generators}
    assert names == {"HardwareSignalGenerator"}
    # Every declared name must resolve through the registry.
    for name, _ in profile.generators:
        assert name in GENERATOR_REGISTRY


# ── Payload contract: reading ──


def test_reading_payload_matches_reading_to_dict_plus_monotonic() -> None:
    """``hw.reading`` payload keys equal ``Reading.to_dict()`` plus ``monotonic_at``."""
    generator = HardwareSignalGenerator(
        SignalConfig(),
        device_id="mock_bench_0",
        channels=[HardwareChannelSpec(channel_id="ch_temp", quantity="temperature")],
    )
    payload = generator._make_reading(generator.channels[0])
    assert set(payload.keys()) == _reading_payload_fields()
    # A real reading must be reconstructable from the payload minus the mock-only field.
    reconstructable = {k: v for k, v in payload.items() if k != "monotonic_at"}
    assert set(reconstructable.keys()) == set(
        Reading(device_id="d", channel_id="c", value=1.0).to_dict().keys()
    )


# ── Payload contract: hardware event ──


def test_hw_event_payload_matches_hardware_event_to_payload() -> None:
    """``hw.<kind>`` payload keys equal ``HardwareEvent.to_payload()`` keys."""
    generator = HardwareSignalGenerator(
        SignalConfig(),
        device_id="mock_bench_0",
        channels=[HardwareChannelSpec(channel_id="ch_temp", quantity="temperature")],
    )
    payload = generator._make_hw_event(generator.channels[0], "threshold_exceeded")
    assert set(payload.keys()) == _hw_event_payload_fields()
    # The generator must fill the platform contract fields, not leave them empty.
    assert payload["kind"] == "threshold_exceeded"
    assert payload["source"] == "mock_bench_0.ch_temp"
    assert payload["ts"]  # wall-clock, non-zero
    assert payload["_mono_ts"]  # monotonic, non-zero


# ── Payload contract exercised through generate() ──


def test_generate_yields_contract_conforming_reading_and_event() -> None:
    """``generate()`` yields ``hw.reading`` and ``hw.<kind>`` tuples that conform.

    Drives the real iterator (not just the payload builders) so the event-type
    naming (``hw.reading`` / ``hw.<kind>``) is covered alongside the field sets.
    """
    generator = HardwareSignalGenerator(
        # A short run with certain event injection keeps the test fast and
        # deterministic: every reading is followed by a hardware event.
        SignalConfig(frequency_hz=100.0, duration_s=1.0),
        device_id="mock_bench_0",
        channels=[HardwareChannelSpec(channel_id="ch_temp", quantity="temperature")],
        event_kinds=["threshold_exceeded"],
        event_probability=1.0,
    )
    reading_payload, event_payload = _collect_first(
        generator, "hw.reading", "hw.threshold_exceeded"
    )
    assert set(reading_payload.keys()) == _reading_payload_fields()
    assert set(event_payload.keys()) == _hw_event_payload_fields()
    assert event_payload["kind"] == "threshold_exceeded"
