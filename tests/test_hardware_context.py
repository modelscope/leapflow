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


# ════════════════════════════════════════════════════════════════
# Hysteresis is derived, and never weakens the safety check
# ════════════════════════════════════════════════════════════════


def test_settle_margin_prefers_the_declared_quantization() -> None:
    """A change below the device's own resolution is not a change.

    Deriving the band from ``quantization`` keeps the declaration the single source of
    truth: nothing new has to be written down, and nothing can drift out of sync.
    """
    envelope = Envelope(declared=True, min_value=0.0, max_value=100.0, quantization=0.5)
    assert envelope.settle_margin == pytest.approx(0.5)


def test_settle_margin_falls_back_to_a_fraction_of_the_span() -> None:
    """Most channels declare no quantization but still must not flap."""
    envelope = Envelope(declared=True, min_value=0.0, max_value=200.0)
    assert envelope.settle_margin == pytest.approx(2.0)


def test_settle_margin_is_capped_so_a_breach_can_always_clear() -> None:
    """A coarse quantization must not produce a band wider than the range.

    Uncapped, this channel would demand the value land inside 0..100 by 80 on each
    side -- an empty band. The breach would then be permanent, trading a flood of
    events for a stuck one, which is worse: a flood is visible.
    """
    envelope = Envelope(declared=True, min_value=0.0, max_value=100.0, quantization=80.0)
    assert envelope.settle_margin == pytest.approx(25.0)
    assert envelope.contains(50.0, margin=envelope.settle_margin) is True


def test_settle_margin_is_zero_without_a_two_sided_range() -> None:
    """A one-sided envelope has no scale, so no fraction of it can be taken."""
    assert Envelope(declared=True, min_value=0.0).settle_margin == 0.0
    assert Envelope(declared=True).settle_margin == 0.0


def test_margin_narrows_inward_and_never_widens_the_band() -> None:
    """The margin may only make the test stricter.

    A margin that widened the range would turn a recovery aid into a hole in the one
    check standing between a command and the device.
    """
    envelope = Envelope(declared=True, min_value=0.0, max_value=100.0)
    assert envelope.contains(0.0) is True
    assert envelope.contains(0.0, margin=1.0) is False
    assert envelope.contains(100.0, margin=1.0) is False
    assert envelope.contains(50.0, margin=1.0) is True
    # A negative margin must not reopen the band.
    assert envelope.contains(101.0, margin=-5.0) is False


def test_margin_defaults_to_zero_so_the_hardline_is_unchanged() -> None:
    """Safety callers evaluate the limit a human declared, not a softened one."""
    envelope = Envelope(declared=True, min_value=0.0, max_value=100.0, quantization=10.0)
    assert envelope.contains(100.0) is True, "the declared bound is inclusive"
    assert envelope.contains(100.0, margin=envelope.settle_margin) is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "fast", None])
def test_margin_does_not_reopen_the_non_numeric_path(value: Any) -> None:
    """Still fail-closed: "cannot evaluate" carries the same weight as "out of range"."""
    envelope = Envelope(declared=True, min_value=0.0, max_value=100.0)
    assert envelope.contains(value, margin=1.0) is False


# ════════════════════════════════════════════════════════════════
# Enumerated (allowed_values) envelope
# ════════════════════════════════════════════════════════════════


def test_enum_envelope_admits_declared_values() -> None:
    """An enumerated envelope checks membership, not numeric range."""
    envelope = Envelope(declared=True, allowed_values=("ac", "battery"))
    assert envelope.contains("ac") is True
    assert envelope.contains("battery") is True
    assert envelope.contains("dc") is False
    assert envelope.contains(42) is False


def test_enum_envelope_boolean_values() -> None:
    """Boolean enum: True/False are discrete states, not numbers."""
    envelope = Envelope(declared=True, allowed_values=(True, False))
    assert envelope.contains(True) is True
    assert envelope.contains(False) is True
    assert envelope.contains("on") is False


def test_enum_envelope_undeclared_still_admits_nothing() -> None:
    """An undeclared envelope with allowed_values is still undeclared."""
    envelope = Envelope(declared=False, allowed_values=("a", "b"))
    assert envelope.contains("a") is False


def test_empty_allowed_values_preserves_numeric_behavior() -> None:
    """Default empty allowed_values is fully backward-compatible."""
    envelope = Envelope(declared=True, min_value=0.0, max_value=100.0)
    assert envelope.allowed_values == ()
    assert envelope.contains(50.0) is True
    assert envelope.contains(200.0) is False


def test_empty_allowed_values_state_channel_unchanged() -> None:
    """A state channel with no allowed_values admits any value (existing behavior)."""
    envelope = Envelope(declared=True)
    assert envelope.allowed_values == ()
    assert envelope.contains("standby") is True
    assert envelope.contains(True) is True


def test_enum_envelope_is_not_numeric() -> None:
    """An enum envelope with no numeric bounds is not numeric."""
    envelope = Envelope(declared=True, allowed_values=("ac", "battery"))
    assert envelope.is_numeric is False


def test_enum_envelope_band_key_is_stable() -> None:
    """Band key for enum envelopes must be deterministic."""
    a = Envelope(declared=True, allowed_values=("battery", "ac"))
    b = Envelope(declared=True, allowed_values=("ac", "battery"))
    assert a.band_key() == b.band_key()
    assert a.band_key().startswith("enum:")


def test_enum_envelope_band_key_changes_on_set_change() -> None:
    """Grant identity must be invalidated when allowed values change."""
    narrow = Envelope(declared=True, allowed_values=("ac", "battery"))
    wider = Envelope(declared=True, allowed_values=("ac", "battery", "usb"))
    assert narrow.band_key() != wider.band_key()


def test_enum_envelope_round_trips_through_mapping() -> None:
    """Serialization and deserialization preserve allowed_values."""
    original = Envelope(declared=True, allowed_values=("ac", "battery"))
    restored = Envelope.from_mapping(original.to_dict())
    assert restored == original
    assert restored.allowed_values == ("ac", "battery")


def test_enum_envelope_from_mapping_ignores_non_list() -> None:
    """Non-list allowed_values in a mapping is treated as absent."""
    envelope = Envelope.from_mapping({"declared": True, "allowed_values": "not_a_list"})
    assert envelope.allowed_values == ()


def test_yaml_enum_envelope(tmp_path: Path) -> None:
    """YAML declarations can specify allowed_values for enum channels."""
    _write_declaration(
        tmp_path,
        "enum_device",
        {
            "hc_version": HC_VERSION,
            "device_id": "enum_device",
            "display_name": "Enum device",
            "halt_supported": True,
            "transport": {"kind": "mock"},
            "channels": [
                {
                    "channel_id": "power_source",
                    "direction": "read",
                    "quantity": "power.source",
                    "effect": "read",
                    "envelope": {
                        "declared": True,
                        "allowed_values": ["ac", "battery"],
                    },
                }
            ],
        },
    )
    provider = YamlContextProvider({"devices_dir": tmp_path})
    contexts = provider.discover()
    assert len(contexts) == 1
    ch = contexts[0].channel("power_source")
    assert ch is not None
    assert ch.envelope.allowed_values == ("ac", "battery")
    assert ch.envelope.contains("ac") is True
    assert ch.envelope.contains("dc") is False


# ════════════════════════════════════════════════════════════════
# Tolerance (G-1) and settling model (G-2)
# ════════════════════════════════════════════════════════════════


def test_tolerance_defaults_to_zero() -> None:
    """Zero tolerance preserves existing span-based normalisation."""
    envelope = Envelope(declared=True, min_value=0.0, max_value=100.0)
    assert envelope.tolerance == 0.0


def test_tolerance_field_is_declared() -> None:
    """A positive tolerance is carried through construction."""
    envelope = Envelope(declared=True, min_value=0.0, max_value=100.0, tolerance=0.5)
    assert envelope.tolerance == 0.5


def test_settling_model_defaults_to_step() -> None:
    """Default settling model is the existing step behaviour."""
    envelope = Envelope(declared=True)
    assert envelope.settling_model == "step"
    assert envelope.settling_tau_s == 0.0


def test_effective_settling_step_uses_settling_time_s() -> None:
    """Step model delegates to the existing settling_time_s."""
    envelope = Envelope(declared=True, settling_time_s=3.0)
    assert envelope.effective_settling_s == pytest.approx(3.0)


def test_effective_settling_first_order_uses_five_tau() -> None:
    """First-order model: 5τ gives 99 % convergence."""
    envelope = Envelope(
        declared=True, settling_model="first_order", settling_tau_s=2.0
    )
    assert envelope.effective_settling_s == pytest.approx(10.0)


def test_effective_settling_both_declared_takes_max() -> None:
    """When both settling_time_s and tau are declared, the larger governs."""
    # tau*5 = 10.0 > settling_time_s = 3.0 → 10.0
    envelope = Envelope(
        declared=True,
        settling_time_s=3.0,
        settling_model="first_order",
        settling_tau_s=2.0,
    )
    assert envelope.effective_settling_s == pytest.approx(10.0)

    # settling_time_s = 20.0 > tau*5 = 10.0 → 20.0
    envelope2 = Envelope(
        declared=True,
        settling_time_s=20.0,
        settling_model="first_order",
        settling_tau_s=2.0,
    )
    assert envelope2.effective_settling_s == pytest.approx(20.0)


def test_effective_settling_zero_tau_on_first_order_falls_back_to_step() -> None:
    """A first_order declaration with tau=0 degrades to step."""
    envelope = Envelope(
        declared=True, settling_time_s=5.0,
        settling_model="first_order", settling_tau_s=0.0,
    )
    assert envelope.effective_settling_s == pytest.approx(5.0)


def test_tolerance_and_settling_round_trip_through_mapping() -> None:
    """New fields survive serialise → deserialise without loss."""
    original = Envelope(
        declared=True,
        min_value=0.0,
        max_value=100.0,
        tolerance=0.5,
        settling_model="first_order",
        settling_tau_s=3.0,
        settling_time_s=2.0,
        reversible=True,
    )
    restored = Envelope.from_mapping(original.to_dict())
    assert restored == original
    assert restored.tolerance == 0.5
    assert restored.settling_model == "first_order"
    assert restored.settling_tau_s == 3.0


def test_yaml_tolerance_and_settling_model(tmp_path: Path) -> None:
    """YAML declarations support tolerance and settling_model."""
    _write_declaration(
        tmp_path,
        "sensor",
        {
            "hc_version": HC_VERSION,
            "device_id": "sensor",
            "display_name": "Sensor",
            "halt_supported": True,
            "transport": {"kind": "mock", "config": {"values": {"temp": 25.0}}},
            "channels": [
                {
                    "channel_id": "temp",
                    "direction": "read",
                    "quantity": "temperature.ambient",
                    "unit": "degC",
                    "envelope": {
                        "declared": True,
                        "min_value": -40.0,
                        "max_value": 85.0,
                        "tolerance": 0.5,
                        "settling_model": "first_order",
                        "settling_tau_s": 2.0,
                    },
                }
            ],
        },
    )
    provider = YamlContextProvider({"devices_dir": tmp_path})
    contexts = provider.discover()
    assert len(contexts) == 1
    ch = contexts[0].channel("temp")
    assert ch is not None
    assert ch.envelope.tolerance == 0.5
    assert ch.envelope.settling_model == "first_order"
    assert ch.envelope.settling_tau_s == 2.0
    assert ch.envelope.effective_settling_s == pytest.approx(10.0)
