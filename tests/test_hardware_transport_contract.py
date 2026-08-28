"""Transport conformance suite -- the executable definition of pluggability.

Every registered transport must pass these cases. That is the point: when a driver
for an upstream hardware standard is written, "done" already has a definition, and
it was fixed before the standard existed.

New transports are added to ``_TRANSPORT_CASES``. A transport that cannot satisfy a
case must declare the shortfall (``halt_supported=False``) rather than special-case
itself out of the suite.
"""

from __future__ import annotations

from typing import Any

import pytest

from leapflow.hardware.context import (
    Channel,
    ContextProvenance,
    Direction,
    Envelope,
    HardwareContext,
    HardwareEffect,
    TransportRef,
)
from leapflow.hardware.transport import (
    SIDE_EFFECT_NONE,
    HardwareTransport,
    Reading,
    TransportError,
    TransportStatus,
    WriteOutcome,
)
from leapflow.hardware.transports import available_transports, build_transport


def _conformance_context(transport_kind: str, config: dict[str, Any]) -> HardwareContext:
    """A device declaration exercising read, write, and streaming shapes."""
    return HardwareContext(
        device_id="conformance_device",
        display_name="Conformance device",
        transport=TransportRef(kind=transport_kind, config=config),
        halt_supported=True,
        channels=(
            Channel(
                channel_id="sensor",
                direction=Direction.READ.value,
                quantity="generic.sensor",
                unit="unit",
                sample_rate_hz=1.0,
                envelope=Envelope(declared=True, min_value=0.0, max_value=100.0),
            ),
            Channel(
                channel_id="setpoint",
                direction=Direction.READWRITE.value,
                quantity="generic.setpoint",
                unit="unit",
                effect=HardwareEffect.CONFIGURE.value,
                verify_after_write=True,
                envelope=Envelope(
                    declared=True, min_value=0.0, max_value=100.0, reversible=True
                ),
            ),
        ),
        provenance=ContextProvenance(verified_by="conformance"),
    )


# Each case is (transport_kind, transport_config). A transport needing external
# resources supplies a config that keeps it self-contained, or is not listed.
_TRANSPORT_CASES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("mock", {"values": {"sensor": 21.5, "setpoint": 50.0}, "halt_supported": True}),
)

_EXTERNAL_ONLY_TRANSPORTS = frozenset(
    {
        # Needs an importable third-party driver module by definition, so it cannot
        # be exercised without one. Its failure modes are covered separately in
        # test_hardware_context.py.
        "python",
    }
)


@pytest.fixture(params=_TRANSPORT_CASES, ids=[case[0] for case in _TRANSPORT_CASES])
def transport_case(request: pytest.FixtureRequest) -> tuple[HardwareTransport, HardwareContext]:
    kind, config = request.param
    return build_transport(kind, config), _conformance_context(kind, config)


def test_every_registered_transport_is_covered_or_declared_external() -> None:
    """No transport may quietly escape the conformance suite.

    Registering a transport without covering it is how an unverified driver reaches
    a physical device, so the omission is a test failure rather than a gap someone
    notices later.
    """
    covered = {case[0] for case in _TRANSPORT_CASES} | _EXTERNAL_ONLY_TRANSPORTS
    missing = set(available_transports()) - covered
    assert not missing, (
        f"transports {sorted(missing)} are registered but not conformance-tested; "
        "add a case to _TRANSPORT_CASES or justify it in _EXTERNAL_ONLY_TRANSPORTS"
    )


@pytest.mark.asyncio
async def test_open_is_idempotent(transport_case) -> None:
    transport, context = transport_case
    first = await transport.open(context)
    second = await transport.open(context)
    assert first.connected is True
    assert second.connected is True


@pytest.mark.asyncio
async def test_close_is_idempotent_and_never_raises(transport_case) -> None:
    transport, context = transport_case
    await transport.open(context)
    assert (await transport.close()).connected is False
    # A second close during teardown must not raise: an exception here would mask
    # whatever failure caused teardown in the first place.
    assert (await transport.close()).connected is False


@pytest.mark.asyncio
async def test_probe_is_side_effect_free(transport_case) -> None:
    transport, context = transport_case
    await transport.open(context)
    before = await transport.read("sensor")
    await transport.probe()
    await transport.probe()
    after = await transport.read("sensor")
    assert before.value == after.value


@pytest.mark.asyncio
async def test_read_sequence_increases_monotonically(transport_case) -> None:
    """Sequence numbers are the only evidence that a bounded queue dropped a sample."""
    transport, context = transport_case
    await transport.open(context)
    sequences = [(await transport.read("sensor")).sequence for _ in range(4)]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)


@pytest.mark.asyncio
async def test_read_returns_a_reading_with_channel_identity(transport_case) -> None:
    transport, context = transport_case
    await transport.open(context)
    reading = await transport.read("sensor")
    assert isinstance(reading, Reading)
    assert reading.channel_id == "sensor"
    assert reading.device_id == context.device_id


@pytest.mark.asyncio
async def test_unknown_channel_raises_transport_error(transport_case) -> None:
    transport, context = transport_case
    await transport.open(context)
    with pytest.raises(TransportError):
        await transport.read("no_such_channel")


@pytest.mark.asyncio
async def test_operating_before_open_raises_rather_than_guessing(transport_case) -> None:
    transport, _ = transport_case
    with pytest.raises(TransportError):
        await transport.read("sensor")


@pytest.mark.asyncio
async def test_successful_write_reports_a_definite_side_effect(transport_case) -> None:
    transport, context = transport_case
    await transport.open(context)
    outcome = await transport.write("setpoint", 42.0)
    assert isinstance(outcome, WriteOutcome)
    assert outcome.ok is True
    # A successful physical write has definitely landed; reporting "none" would
    # tell the recovery layer it is safe to replay.
    assert outcome.side_effect_state != SIDE_EFFECT_NONE


@pytest.mark.asyncio
async def test_verify_after_write_channel_returns_a_readback(transport_case) -> None:
    transport, context = transport_case
    await transport.open(context)
    outcome = await transport.write("setpoint", 33.0)
    assert outcome.readback is not None
    assert outcome.readback.channel_id == "setpoint"


@pytest.mark.asyncio
async def test_failed_write_never_claims_no_side_effect(transport_case) -> None:
    """The central contract: an error is not proof that nothing happened.

    A transport reporting ``none`` on failure would let the recovery layer replay a
    physical command, which is precisely how a failed dispense becomes a double
    dispense.
    """
    kind, config = _TRANSPORT_CASES[0]
    failing = build_transport(
        kind,
        {
            **config,
            "failures": [{"channel_id": "setpoint", "on_call": 1, "side_effect_state": "partial"}],
        },
    )
    context = _conformance_context(kind, config)
    await failing.open(context)
    outcome = await failing.write("setpoint", 10.0)
    assert outcome.ok is False
    assert outcome.side_effect_state != SIDE_EFFECT_NONE
    assert outcome.effect_may_have_landed is True


@pytest.mark.asyncio
async def test_halt_reports_capability_instead_of_raising(transport_case) -> None:
    transport, context = transport_case
    await transport.open(context)
    status = await transport.halt()
    assert isinstance(status, TransportStatus)
    assert isinstance(status.halt_supported, bool)


@pytest.mark.asyncio
async def test_transport_without_halt_declares_it(transport_case) -> None:
    """"Cannot stop" must be discoverable, never a silent assumption."""
    kind, config = _TRANSPORT_CASES[0]
    transport = build_transport(kind, {**config, "halt_supported": False})
    await transport.open(_conformance_context(kind, config))
    status = await transport.halt()
    assert status.halt_supported is False


@pytest.mark.asyncio
async def test_satisfies_the_protocol(transport_case) -> None:
    transport, _ = transport_case
    assert isinstance(transport, HardwareTransport)
    assert isinstance(transport.kind, str) and transport.kind
