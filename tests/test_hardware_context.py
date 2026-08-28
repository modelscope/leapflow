"""Hardware context domain model and registry admission rules.

Admission is where a declaration becomes something the agent may act on, so each
rule gets a negative case. The recurring theme in the assertions below is that
uncertainty and refusal must have the same consequence: an interlock that cannot be
evaluated, an envelope that was never declared, and a device that cannot be stopped
all remove the ability to command it, rather than being treated as permissive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from leapflow.hardware.context import (
    HC_VERSION,
    Channel,
    ContextProvenance,
    ContextSource,
    Direction,
    Envelope,
    HardwareContext,
    HardwareEffect,
    Interlock,
    TransportRef,
)
from leapflow.hardware.providers import ProviderError, available_providers, build_provider
from leapflow.hardware.providers.yaml_provider import YamlContextProvider
from leapflow.hardware.reference import describe, render_reference, summarize
from leapflow.hardware.registry import (
    HardwareRegistry,
    HardwareSettings,
    UnverifiedContextPolicy,
)
from leapflow.hardware.transport import TransportError
from leapflow.hardware.transports import (
    available_transports,
    build_transport,
    register_transport,
)


# ════════════════════════════════════════════════════════════════
# Helpers -- deliberately device-agnostic
# ════════════════════════════════════════════════════════════════


class _StaticProvider:
    """Provider returning fixed contexts, standing in for any real source."""

    kind = "static"

    def __init__(self, *contexts: HardwareContext) -> None:
        self._contexts = contexts

    def discover(self) -> tuple[HardwareContext, ...]:
        return self._contexts


def _channel(
    channel_id: str = "setpoint",
    *,
    direction: str = Direction.READWRITE.value,
    effect: str = HardwareEffect.CONFIGURE.value,
    envelope: Envelope | None = None,
    **kwargs: Any,
) -> Channel:
    return Channel(
        channel_id=channel_id,
        direction=direction,
        quantity=kwargs.pop("quantity", "generic.value"),
        unit=kwargs.pop("unit", "unit"),
        effect=effect,
        envelope=envelope if envelope is not None else Envelope(declared=True, min_value=0.0, max_value=100.0),
        **kwargs,
    )


def _context(
    device_id: str = "device_a",
    *,
    channels: tuple[Channel, ...] | None = None,
    verified: bool = True,
    halt: bool = True,
    transport_kind: str = "mock",
    hc_version: str = HC_VERSION,
    interlocks: tuple[Interlock, ...] = (),
) -> HardwareContext:
    return HardwareContext(
        device_id=device_id,
        hc_version=hc_version,
        display_name=device_id,
        transport=TransportRef(kind=transport_kind, config={}),
        channels=channels if channels is not None else (_channel(),),
        interlocks=interlocks,
        halt_supported=halt,
        provenance=ContextProvenance(verified_by="tester" if verified else ""),
    )


def _load(context: HardwareContext, **setting_overrides: Any) -> HardwareRegistry:
    settings = HardwareSettings(enabled=True, **setting_overrides)
    registry = HardwareRegistry(settings, providers=[_StaticProvider(context)])
    registry.load()
    return registry


def _notes_for(registry: HardwareRegistry, rule: str) -> tuple[str, ...]:
    return tuple(note.detail for note in registry.report.notes if note.rule == rule)


# ════════════════════════════════════════════════════════════════
# Envelope semantics
# ════════════════════════════════════════════════════════════════


def test_undeclared_envelope_admits_nothing() -> None:
    """An undeclared envelope is not an unbounded one."""
    envelope = Envelope()
    assert envelope.contains(0.0) is False
    assert envelope.contains(None) is False
    assert envelope.band_key() == "undeclared"


def test_declared_envelope_bounds_numeric_values() -> None:
    envelope = Envelope(declared=True, min_value=0.0, max_value=80.0)
    assert envelope.contains(0.0) is True
    assert envelope.contains(80.0) is True
    assert envelope.contains(80.001) is False
    assert envelope.contains(-0.001) is False


def test_declared_envelope_admits_non_numeric_states() -> None:
    """A channel with no numeric bounds is a state channel; any state is in range."""
    envelope = Envelope(declared=True)
    assert envelope.contains(True) is True
    assert envelope.contains("standby") is True


@pytest.mark.parametrize("bad_value", [True, False, "fast", None, float("nan"), float("inf")])
def test_numeric_envelope_refuses_an_unevaluable_value(bad_value: Any) -> None:
    """A numeric envelope handed something it cannot compare must refuse it.

    "Cannot evaluate" has to carry the same weight as "out of range". Admitting the
    value instead would let an unparseable command slip past the one check standing
    between it and the device -- and ``True``, NaN, and infinity are exactly the
    values that compare in ways which make every bound look satisfied.
    """
    envelope = Envelope(declared=True, min_value=0.0, max_value=80.0)
    assert envelope.is_numeric is True
    assert envelope.contains(bad_value) is False


def test_envelope_is_numeric_is_derived_from_its_bounds() -> None:
    assert Envelope(declared=True).is_numeric is False
    assert Envelope(declared=True, max_rate=1.0).is_numeric is True
    assert Envelope(declared=True, quantization=0.5).is_numeric is True


def test_unmeasurable_rate_counts_as_exceeding() -> None:
    """A zero interval cannot be measured, so it must not pass as safe."""
    envelope = Envelope(declared=True, max_rate=10.0)
    assert envelope.rate_exceeded(delta=1.0, elapsed_s=0.0) is True
    assert envelope.rate_exceeded(delta=1.0, elapsed_s=1.0) is False
    assert envelope.rate_exceeded(delta=50.0, elapsed_s=1.0) is True


def test_rate_wait_is_the_shortfall_not_the_whole_interval() -> None:
    """The advised wait must be actionable: only the time still owed."""
    envelope = Envelope(declared=True, max_rate=10.0)
    # A delta of 5 at 10/s needs 0.5s; 0.2s has already elapsed.
    assert envelope.rate_wait_s(delta=5.0, elapsed_s=0.2) == pytest.approx(0.3)
    # Enough time has passed, so nothing is owed.
    assert envelope.rate_wait_s(delta=5.0, elapsed_s=0.9) == 0.0
    # Direction is irrelevant; magnitude is what costs time.
    assert envelope.rate_wait_s(delta=-5.0, elapsed_s=0.0) == pytest.approx(0.5)


def test_no_rate_limit_never_waits() -> None:
    assert Envelope(declared=True).rate_wait_s(delta=1000.0, elapsed_s=0.0) == 0.0


def test_repeating_the_same_value_costs_no_time() -> None:
    """Re-commanding the current value is not a change and cannot be too fast."""
    envelope = Envelope(declared=True, max_rate=1.0)
    assert envelope.rate_wait_s(delta=0.0, elapsed_s=0.0) == 0.0


def test_band_key_changes_when_the_envelope_widens() -> None:
    """Grant identity must not survive a change in what it was granted under."""
    narrow = Envelope(declared=True, min_value=0.0, max_value=200.0, max_rate=50.0)
    wide = Envelope(declared=True, min_value=0.0, max_value=500.0, max_rate=50.0)
    assert narrow.band_key() != wide.band_key()


def test_band_key_is_stable_for_the_same_declaration() -> None:
    first = Envelope(declared=True, min_value=0.0, max_value=200.0, reversible=False)
    second = Envelope(declared=True, min_value=0.0, max_value=200.0, reversible=False)
    assert first.band_key() == second.band_key()


def test_reversibility_participates_in_the_band() -> None:
    reversible = Envelope(declared=True, min_value=0.0, max_value=1.0, reversible=True)
    irreversible = Envelope(declared=True, min_value=0.0, max_value=1.0, reversible=False)
    assert reversible.band_key() != irreversible.band_key()


# ════════════════════════════════════════════════════════════════
# Interlock semantics
# ════════════════════════════════════════════════════════════════


def test_interlock_evaluates_declared_comparison() -> None:
    lock = Interlock(interlock_id="ready", channel_id="state", operator="eq", value=True)
    assert lock.evaluate(True) is True
    assert lock.evaluate(False) is False


def test_unknown_interlock_operator_fails_closed() -> None:
    lock = Interlock(interlock_id="ready", channel_id="state", operator="approximately", value=1)
    assert lock.evaluate(1) is False


def test_incomparable_interlock_fails_closed() -> None:
    """"Cannot tell" and "not satisfied" must have the same consequence."""
    lock = Interlock(interlock_id="ready", channel_id="state", operator="gt", value=5)
    assert lock.evaluate("not a number") is False


# ════════════════════════════════════════════════════════════════
# Admission rules
# ════════════════════════════════════════════════════════════════


def test_disabled_registry_admits_nothing() -> None:
    """With hardware off the subsystem must be inert, not merely quiet."""
    registry = HardwareRegistry(HardwareSettings(enabled=False), providers=[_StaticProvider(_context())])
    report = registry.load()
    assert report.admitted == ()
    assert registry.contexts() == ()


def test_v1_unknown_protocol_version_is_rejected_not_migrated() -> None:
    registry = _load(_context(hc_version="hc.v99"))
    assert registry.context("device_a") is None
    assert any("unsupported hc_version" in detail for detail in _notes_for(registry, "V1"))


def test_v2_rejects_unusable_device_id() -> None:
    registry = _load(_context(device_id="Device A"))
    assert registry.contexts() == ()
    assert _notes_for(registry, "V2")


def test_v2_rejects_duplicate_device_ids() -> None:
    settings = HardwareSettings(enabled=True)
    registry = HardwareRegistry(
        settings, providers=[_StaticProvider(_context("dup"), _context("dup"))]
    )
    report = registry.load()
    assert report.admitted == ("dup",)
    assert "dup" in report.rejected


def test_v3_writable_channel_without_envelope_is_demoted_not_removed() -> None:
    """The channel stays readable: reads matter most when something is wrong."""
    registry = _load(_context(channels=(_channel(envelope=Envelope(declared=False)),)))
    context = registry.context("device_a")
    assert context is not None
    assert context.channel("setpoint").is_writable is False
    assert context.channel("setpoint").is_readable is True
    assert _notes_for(registry, "V3")


def test_v4_unknown_transport_kind_is_rejected() -> None:
    registry = _load(_context(transport_kind="no_such_transport"))
    assert registry.contexts() == ()
    assert any("unknown transport kind" in detail for detail in _notes_for(registry, "V4"))


def test_v5_device_that_cannot_halt_loses_every_writable_channel() -> None:
    registry = _load(_context(halt=False))
    context = registry.context("device_a")
    assert context is not None
    assert context.writable_channels == ()
    assert _notes_for(registry, "V5")


def test_v6_interlock_on_unreadable_channel_is_reported() -> None:
    registry = _load(
        _context(
            channels=(
                _channel(
                    envelope=Envelope(
                        declared=True, min_value=0.0, max_value=1.0, requires_interlocks=("ready",)
                    )
                ),
            ),
            interlocks=(Interlock(interlock_id="ready", channel_id="absent_channel"),),
        )
    )
    notes = _notes_for(registry, "V6")
    assert notes
    assert any("hardline-den" in detail or "unsatisfied" in detail for detail in notes)


def test_v7_unverified_context_cannot_authorize_writes() -> None:
    registry = _load(_context(verified=False))
    context = registry.context("device_a")
    assert context is not None
    assert context.writable_channels == ()
    assert _notes_for(registry, "V7")


def test_v7_policy_can_be_relaxed_explicitly() -> None:
    """Relaxing the policy is possible, but it has to be said out loud."""
    registry = _load(
        _context(verified=False),
        unverified_context_policy=UnverifiedContextPolicy.ALLOW,
    )
    context = registry.context("device_a")
    assert context is not None
    assert len(context.writable_channels) == 1


def test_v8_device_count_is_capped() -> None:
    contexts = tuple(_context(f"device_{index}") for index in range(5))
    registry = HardwareRegistry(
        HardwareSettings(enabled=True, max_devices=2), providers=[_StaticProvider(*contexts)]
    )
    report = registry.load()
    assert len(report.admitted) == 2
    assert len(report.rejected) == 3
    assert _notes_for(registry, "V8")


def test_declaration_without_channels_is_rejected() -> None:
    registry = _load(_context(channels=()))
    assert registry.contexts() == ()


def test_one_bad_declaration_does_not_hide_the_others() -> None:
    """A malformed device must not make an entire bench disappear."""
    registry = HardwareRegistry(
        HardwareSettings(enabled=True),
        providers=[_StaticProvider(_context("good_one"), _context("Bad ID"))],
    )
    report = registry.load()
    assert report.admitted == ("good_one",)
    assert registry.context("good_one") is not None


def test_provider_failure_is_reported_not_fatal() -> None:
    class _Exploding:
        kind = "exploding"

        def discover(self) -> tuple[HardwareContext, ...]:
            raise ValueError("device catalogue unreachable")

    registry = HardwareRegistry(
        HardwareSettings(enabled=True), providers=[_Exploding(), _StaticProvider(_context())]
    )
    report = registry.load()
    assert report.admitted == ("device_a",)
    assert any(note.rule == "provider" for note in report.notes)


def test_load_is_idempotent_and_reflects_declaration_changes() -> None:
    registry = _load(_context())
    assert registry.report.admitted == ("device_a",)
    registry.load()
    assert registry.report.admitted == ("device_a",)


# ════════════════════════════════════════════════════════════════
# Transport resolution
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_transport_is_opened_lazily_on_first_use() -> None:
    """``load()`` must not touch hardware: discovery works with the device off."""
    registry = _load(_context())
    assert registry.opened_devices() == ()
    await registry.transport("device_a")
    assert registry.opened_devices() == ("device_a",)


@pytest.mark.asyncio
async def test_unknown_device_transport_raises_structured_error() -> None:
    registry = _load(_context())
    with pytest.raises(TransportError) as excinfo:
        await registry.transport("not_here")
    assert excinfo.value.failure_code == "unknown_device"


@pytest.mark.asyncio
async def test_close_all_isolates_failures() -> None:
    """One device refusing to close must not leave the others open."""

    class _StubbornTransport:
        kind = "stubborn"

        async def open(self, context: HardwareContext):
            from leapflow.hardware.transport import TransportStatus

            return TransportStatus(connected=True, halt_supported=True)

        async def close(self):
            raise RuntimeError("device will not release the port")

        async def read(self, channel_id: str):  # pragma: no cover - not exercised
            raise TransportError("unused")

        async def write(self, channel_id: str, value: Any):  # pragma: no cover
            raise TransportError("unused")

        async def probe(self):  # pragma: no cover
            from leapflow.hardware.transport import TransportStatus

            return TransportStatus(connected=True)

        async def halt(self):  # pragma: no cover
            from leapflow.hardware.transport import TransportStatus

            return TransportStatus(connected=True, halt_supported=True)

    globals()["_build_stubborn"] = lambda config=None: _StubbornTransport()
    # The transport table is process-global, so the registration is undone here.
    # Leaving it behind leaked into the conformance suite's coverage guard in a
    # different file, which is the same class of cross-test contamination the undo
    # callable exists to prevent in production reloads.
    undo = register_transport("stubborn_test", f"{__name__}:_build_stubborn")
    try:
        registry = _load(_context(transport_kind="stubborn_test"))
        await registry.transport("device_a")
        await registry.close_all()
        assert registry.opened_devices() == ()
    finally:
        undo()


def test_registration_undo_restores_the_table() -> None:
    """A plugin's transport must disappear with the plugin, not outlive it."""
    before = available_transports()
    undo = register_transport("temporary_kind", f"{__name__}:_build_stubborn")
    assert "temporary_kind" in available_transports()
    undo()
    assert available_transports() == before


def test_unknown_transport_kind_raises_structured_error() -> None:
    with pytest.raises(TransportError) as excinfo:
        build_transport("definitely_not_registered", {})
    assert excinfo.value.failure_code == "unknown_transport_kind"


def test_python_transport_reports_a_missing_driver_clearly() -> None:
    """A missing external driver is an unusable device, not a crash."""
    with pytest.raises(TransportError) as excinfo:
        build_transport("python", {"module": "leapflow_no_such_driver_module"})
    assert excinfo.value.failure_code == "driver_import_failed"


def test_python_transport_requires_a_module() -> None:
    with pytest.raises(TransportError) as excinfo:
        build_transport("python", {})
    assert excinfo.value.failure_code == "driver_module_missing"


def test_python_transport_rejects_a_non_conforming_driver() -> None:
    globals()["_not_a_transport"] = lambda: object()
    with pytest.raises(TransportError) as excinfo:
        build_transport("python", {"module": __name__, "factory": "_not_a_transport"})
    assert excinfo.value.failure_code == "driver_protocol_mismatch"


# ════════════════════════════════════════════════════════════════
# YAML provider
# ════════════════════════════════════════════════════════════════


def _write_declaration(directory: Path, name: str, payload: dict[str, Any]) -> Path:
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_yaml_provider_requires_a_directory() -> None:
    with pytest.raises(ProviderError):
        YamlContextProvider({})


def test_yaml_provider_reads_a_declaration(tmp_path: Path) -> None:
    _write_declaration(
        tmp_path,
        "bench_node",
        {
            "hc_version": HC_VERSION,
            "device_id": "bench_node",
            "display_name": "Bench node",
            "halt_supported": True,
            "transport": {"kind": "mock", "config": {"values": {"level": 1.0}}},
            "channels": [
                {
                    "channel_id": "level",
                    "direction": "readwrite",
                    "quantity": "generic.level",
                    "unit": "unit",
                    "effect": "configure",
                    "envelope": {"declared": True, "min_value": 0.0, "max_value": 10.0},
                }
            ],
        },
    )
    provider = YamlContextProvider({"devices_dir": tmp_path})
    contexts = provider.discover()
    assert len(contexts) == 1
    assert contexts[0].device_id == "bench_node"
    assert contexts[0].channel("level").envelope.max_value == 10.0
    assert contexts[0].provenance.source == ContextSource.DECLARED.value


def test_yaml_provider_skips_unparseable_files_without_losing_the_rest(tmp_path: Path) -> None:
    (tmp_path / "broken.yaml").write_text("this: [is: not: valid", encoding="utf-8")
    _write_declaration(
        tmp_path,
        "fine",
        {
            "hc_version": HC_VERSION,
            "device_id": "fine",
            "transport": {"kind": "mock"},
            "channels": [{"channel_id": "value", "direction": "read"}],
        },
    )
    contexts = YamlContextProvider({"devices_dir": tmp_path}).discover()
    assert [c.device_id for c in contexts] == ["fine"]


def test_yaml_provider_applies_out_of_band_verification(tmp_path: Path) -> None:
    """Confirmation lives outside the declaration so it is not self-attested."""
    devices = tmp_path / "devices"
    devices.mkdir()
    _write_declaration(
        devices,
        "node",
        {
            "hc_version": HC_VERSION,
            "device_id": "node",
            "transport": {"kind": "mock"},
            "channels": [{"channel_id": "value", "direction": "read"}],
        },
    )
    verified = tmp_path / "verified.json"
    verified.write_text(yaml.safe_dump({"node": "jason"}), encoding="utf-8")
    contexts = YamlContextProvider(
        {"devices_dir": devices, "verified_path": verified}
    ).discover()
    assert contexts[0].provenance.verified_by == "jason"
    assert contexts[0].provenance.is_verified is True


def test_yaml_provider_tolerates_a_missing_directory(tmp_path: Path) -> None:
    provider = YamlContextProvider({"devices_dir": tmp_path / "absent"})
    assert provider.discover() == ()


def test_yaml_provider_is_registered() -> None:
    assert "yaml" in available_providers()
    assert isinstance(build_provider("yaml", {"devices_dir": "."}), YamlContextProvider)


# ════════════════════════════════════════════════════════════════
# Reference rendering
# ════════════════════════════════════════════════════════════════


def test_reference_states_unverified_provenance_prominently() -> None:
    """A guess must announce itself, or it will be read as a specification."""
    text = render_reference(_context(verified=False))
    assert "UNVERIFIED" in text.splitlines()[1]


def test_reference_renders_irreversibility_from_the_declaration() -> None:
    context = _context(
        channels=(
            _channel(
                effect=HardwareEffect.DISPENSE.value,
                envelope=Envelope(declared=True, min_value=0.0, max_value=200.0, reversible=False),
            ),
        )
    )
    text = render_reference(context)
    assert "NOT reversible" in text


def test_reference_warns_when_an_envelope_is_missing() -> None:
    context = _context(channels=(_channel(envelope=Envelope(declared=False)),))
    assert "NO DECLARED ENVELOPE" in render_reference(context)


def test_reference_surfaces_channel_notes_to_the_model() -> None:
    """Operator knowledge is useless if it never reaches the reader."""
    context = _context(
        channels=(
            _channel(
                envelope=Envelope(
                    declared=True,
                    min_value=0.0,
                    max_value=10.0,
                    notes="Overshooting damages the seal.",
                )
            ),
        )
    )
    assert "Overshooting damages the seal." in render_reference(context)


def test_reference_declares_missing_halt_capability() -> None:
    assert "emergency stop: NOT SUPPORTED" in render_reference(_context(halt=False))


def test_reference_reports_lossy_imports() -> None:
    """A drop in fidelity is visible, not silently absorbed."""
    context = HardwareContext(
        device_id="imported_device",
        transport=TransportRef(kind="mock"),
        channels=(_channel(),),
        provenance=ContextProvenance(
            source=ContextSource.IMPORTED.value,
            upstream_version="upstream/0.3",
            lossy_fields=("coupled_axis_limits",),
        ),
    )
    text = render_reference(context)
    assert "FIDELITY" in text
    assert "coupled_axis_limits" in text


def test_summary_index_omits_envelopes() -> None:
    """The index is what keeps the tool surface cheap; limits are one call away."""
    payload = summarize(_context())
    assert "envelope" not in yaml.safe_dump(payload)
    assert payload["device_id"] == "device_a"
    assert payload["writable"] == 1


def test_describe_carries_both_text_and_machine_fields() -> None:
    payload = describe(_context())
    assert payload["reference"].startswith("DEVICE device_a")
    assert payload["channels"][0]["channel_id"] == "setpoint"
    assert payload["writable_channels"] == ["setpoint"]


# ════════════════════════════════════════════════════════════════
# Round-trip
# ════════════════════════════════════════════════════════════════


def test_context_round_trips_through_a_mapping() -> None:
    original = _context(
        channels=(
            _channel(
                effect=HardwareEffect.DISPENSE.value,
                sample_rate_hz=5.0,
                verify_after_write=True,
                envelope=Envelope(
                    declared=True,
                    min_value=0.0,
                    max_value=200.0,
                    max_rate=50.0,
                    quantization=0.1,
                    settling_time_s=1.5,
                    reversible=False,
                    requires_interlocks=("ready",),
                    notes="handle with care",
                ),
            ),
        ),
        interlocks=(Interlock(interlock_id="ready", channel_id="setpoint", operator="eq", value=True),),
    )
    restored = HardwareContext.from_mapping(original.to_dict())
    assert restored == original
