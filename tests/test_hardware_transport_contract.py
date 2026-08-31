"""Transport conformance suite -- the executable definition of pluggability.

Every registered transport must pass these cases. That is the point: when a driver
for an upstream hardware standard is written, "done" already has a definition, and
it was fixed before the standard existed.

New transports are added to ``_TRANSPORT_CASES``. A transport that cannot satisfy a
case must declare the shortfall (``halt_supported=False``) rather than special-case
itself out of the suite.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class _Case:
    """One transport under conformance, plus the two variants the suite needs.

    ``failing_write`` and ``without_halt`` exist because the two most consequential
    contracts here -- an error is not proof that nothing happened, and "cannot stop"
    must be declared -- cannot be exercised from the happy-path config. They used to
    be tested against ``_TRANSPORT_CASES[0]`` with mock-specific config keys, so every
    transport after the first was covered by fourteen cases and silently exempt from
    the two that matter most.
    """

    kind: str
    config: dict[str, Any]
    failing_write: dict[str, Any]
    without_halt: dict[str, Any]


_MOCK_CONFIG: dict[str, Any] = {
    "values": {"sensor": 21.5, "setpoint": 50.0},
    "halt_supported": True,
}


def _mcp_config(**overrides: Any) -> dict[str, Any]:
    """An MCP declaration wired to an in-process server stub.

    The stub stands in for the server, not for the transport: sequence numbering,
    idempotent lifecycle, error mapping, side-effect verdicts and both clocks are all
    the transport's own responsibilities and are what these cases exercise.
    """
    config: dict[str, Any] = {
        "server": "bench-stub",
        "read_tool": "bench_read",
        "write_tool": "bench_write",
        "probe_tool": "bench_status",
        "halt_tool": "bench_estop",
        "channel_arg": "channel",
        "value_arg": "value",
        "value_path": "value",
        "client": _StubMcpServer({"sensor": 21.5, "setpoint": 50.0}),
    }
    config.update(overrides)
    return config


class _StubMcpServer:
    """Minimal stand-in for ``McpManager``: one ``call_tool`` over held values."""

    def __init__(self, values: dict[str, Any], *, fail_writes: bool = False) -> None:
        self._values = dict(values)
        self._fail_writes = fail_writes
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((tool_name, dict(arguments)))
        channel = arguments.get("channel", "")
        if tool_name == "bench_read":
            if channel not in self._values:
                return {"ok": False, "error": f"unknown channel {channel!r}"}
            return {"ok": True, "value": self._values[channel], "quality": "ok"}
        if tool_name == "bench_write":
            if self._fail_writes:
                # No verdict declared, so the transport must assume the command may
                # already have reached the device.
                return {"ok": False, "error": "actuator refused"}
            self._values[channel] = arguments.get("value")
            return {"ok": True}
        if tool_name in ("bench_status", "bench_estop"):
            return {"ok": True}
        return {"ok": False, "error": f"unknown tool {tool_name!r}"}


_TRANSPORT_CASES: tuple[_Case, ...] = (
    _Case(
        kind="mock",
        config=_MOCK_CONFIG,
        failing_write={
            **_MOCK_CONFIG,
            "failures": [
                {"channel_id": "setpoint", "on_call": 1, "side_effect_state": "partial"}
            ],
        },
        without_halt={**_MOCK_CONFIG, "halt_supported": False},
    ),
    _Case(
        kind="mcp",
        config=_mcp_config(),
        failing_write=_mcp_config(
            client=_StubMcpServer({"sensor": 21.5, "setpoint": 50.0}, fail_writes=True)
        ),
        # No halt tool named, so the device cannot be stopped. A declaration may not
        # claim the reverse: a tool that was never named cannot be called.
        without_halt=_mcp_config(halt_tool=""),
    ),
)

_EXTERNAL_ONLY_TRANSPORTS = frozenset(
    {
        # Needs an importable third-party driver module by definition, so it cannot
        # be exercised without one. Its failure modes are covered separately in
        # test_hardware_context.py.
        "python",
    }
)


@pytest.fixture(params=_TRANSPORT_CASES, ids=[case.kind for case in _TRANSPORT_CASES])
def case(request: pytest.FixtureRequest) -> _Case:
    """The declaration under test, so a case can build its own variants."""
    return request.param


@pytest.fixture
def transport_case(case: _Case) -> tuple[HardwareTransport, HardwareContext]:
    return build_transport(case.kind, case.config), _conformance_context(case.kind, case.config)


def test_every_registered_transport_is_covered_or_declared_external() -> None:
    """No transport may quietly escape the conformance suite.

    Registering a transport without covering it is how an unverified driver reaches
    a physical device, so the omission is a test failure rather than a gap someone
    notices later.
    """
    covered = {case.kind for case in _TRANSPORT_CASES} | _EXTERNAL_ONLY_TRANSPORTS
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
async def test_failed_write_never_claims_no_side_effect(case: _Case) -> None:
    """The central contract: an error is not proof that nothing happened.

    A transport reporting ``none`` on failure would let the recovery layer replay a
    physical command, which is precisely how a failed dispense becomes a double
    dispense.
    """
    failing = build_transport(case.kind, case.failing_write)
    context = _conformance_context(case.kind, case.failing_write)
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
async def test_transport_without_halt_declares_it(case: _Case) -> None:
    """"Cannot stop" must be discoverable, never a silent assumption."""
    transport = build_transport(case.kind, case.without_halt)
    await transport.open(_conformance_context(case.kind, case.without_halt))
    status = await transport.halt()
    assert status.halt_supported is False


@pytest.mark.asyncio
async def test_satisfies_the_protocol(transport_case) -> None:
    transport, _ = transport_case
    assert isinstance(transport, HardwareTransport)
    assert isinstance(transport.kind, str) and transport.kind


# ════════════════════════════════════════════════════════════════
# Reading carries two clocks, and every transport must populate both
# ════════════════════════════════════════════════════════════════


def test_reading_populates_both_clocks_by_default() -> None:
    """A driver that names neither clock still gets a usable pair.

    Out-of-tree drivers construct ``Reading`` themselves, so the safe values have to
    be the defaults. The previous single field defaulted to 0.0, which reads as epoch
    zero on the wall and as "just booted" on the monotonic side -- both wrong, and
    neither detectable.
    """
    import time

    reading = Reading(device_id="d", channel_id="c", value=1.0)
    assert reading.observed_at > 1_500_000_000.0, "observed_at must be wall-clock"
    assert reading.monotonic_at < 1_500_000_000.0, "monotonic_at must not be wall-clock"
    assert abs(reading.observed_at - time.time()) < 5.0


def test_reading_evidence_form_carries_only_the_wall_clock() -> None:
    """Raw NDJSON is read by people and by later analysis.

    A per-boot counter in an evidence file cannot be lined up with anything, and
    including it alongside the wall-clock value invites picking the wrong one.
    """
    reading = Reading(device_id="d", channel_id="c", value=1.0, observed_at=1_787_000_000.0)
    payload = reading.to_dict()
    assert payload["observed_at"] == 1_787_000_000.0
    assert "monotonic_at" not in payload
    assert "timestamp" not in payload, "the ambiguous name must not come back"


@pytest.mark.asyncio
async def test_every_transport_stamps_both_clocks(transport_case) -> None:
    """Part of the transport contract, checked against each implementation.

    Downstream code divides by the interval between two readings and persists the
    instant of each; a transport that leaves either clock at zero breaks one of those
    without failing.
    """
    transport, context = transport_case
    channel = next(c for c in context.channels if c.is_readable)
    await transport.open(context)
    reading = await transport.read(channel.channel_id)
    assert reading.observed_at > 1_500_000_000.0
    assert reading.monotonic_at > 0.0


# ════════════════════════════════════════════════════════════════
# MCP transport: everything device-specific is declared, never inferred
# ════════════════════════════════════════════════════════════════


def _mcp_context(config: dict[str, Any]) -> HardwareContext:
    return _conformance_context("mcp", config)


@pytest.mark.asyncio
async def test_mcp_calls_only_the_tools_it_was_told_about() -> None:
    """No name matching, no verb enumeration, no "try these" chain.

    A guess that lands on the wrong tool is a physical action nobody authorised, and
    nothing downstream can tell that it happened: the reading comes back with the
    channel id the caller asked for either way.
    """
    stub = _StubMcpServer({"sensor": 21.5, "setpoint": 50.0})
    config = _mcp_config(client=stub)
    transport = build_transport("mcp", config)
    await transport.open(_mcp_context(config))
    await transport.read("sensor")
    await transport.write("setpoint", 42.0)
    await transport.halt()

    tools = [name for name, _ in stub.calls]
    assert set(tools) <= {"bench_read", "bench_write", "bench_status", "bench_estop"}
    read_args = next(args for name, args in stub.calls if name == "bench_read")
    assert read_args["channel"] == "sensor", "the declared channel_arg must carry the channel"
    write_args = next(args for name, args in stub.calls if name == "bench_write")
    assert write_args == {"channel": "setpoint", "value": 42.0}


@pytest.mark.asyncio
async def test_mcp_merges_declared_extra_arguments_into_every_call() -> None:
    """Rig or address selection belongs in the declaration, not in the transport."""
    stub = _StubMcpServer({"sensor": 1.0})
    config = _mcp_config(client=stub, extra_args={"rig": "A"})
    transport = build_transport("mcp", config)
    await transport.open(_mcp_context(config))
    await transport.read("sensor")
    assert all(args.get("rig") == "A" for _, args in stub.calls)


@pytest.mark.asyncio
async def test_mcp_refuses_to_open_when_a_needed_tool_is_undeclared() -> None:
    """A configuration fault must surface at admission, not mid-experiment.

    The conformance declaration exposes a writable channel, so a config with no write
    tool describes a device that cannot do what it claims.
    """
    config = _mcp_config(write_tool="")
    transport = build_transport("mcp", config)
    with pytest.raises(TransportError) as caught:
        await transport.open(_mcp_context(config))
    assert caught.value.failure_code == "mcp_write_tool_missing"


@pytest.mark.asyncio
async def test_mcp_without_a_client_fails_closed() -> None:
    """No client installed is a refusal, never a silent no-op."""
    config = _mcp_config()
    config.pop("client")
    transport = build_transport("mcp", config)
    with pytest.raises(TransportError) as caught:
        await transport.open(_mcp_context(config))
    assert caught.value.failure_code == "mcp_client_unavailable"


@pytest.mark.asyncio
async def test_mcp_reports_the_keys_it_got_when_the_declared_path_is_absent() -> None:
    """The alternative to a fallback chain is a diagnosable error.

    Reading whichever key happens to be present is how one channel's value is
    reported under another channel's identity.
    """

    class _WrongShape:
        async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
            return {"ok": True, "reading": 21.5, "units": "C"}

    config = _mcp_config(client=_WrongShape())
    transport = build_transport("mcp", config)
    await transport.open(_mcp_context(config))
    with pytest.raises(TransportError) as caught:
        await transport.read("sensor")
    assert caught.value.failure_code == "mcp_value_path_missing"
    assert "reading" in str(caught.value) and "units" in str(caught.value)


@pytest.mark.asyncio
async def test_mcp_honours_a_server_declared_side_effect_verdict() -> None:
    """A server that knows its command never left reports it; otherwise unknown.

    The default cannot be ``none``: the MCP client turns a timeout into an ordinary
    error reply, so a failure genuinely cannot distinguish "never sent" from "sent, no
    answer", and reporting ``none`` would let recovery replay a physical command.
    """

    class _Declaring:
        def __init__(self, state: str | None) -> None:
            self._state = state

        async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
            if tool_name != "bench_write":
                return {"ok": True, "value": 1.0}
            body: dict[str, Any] = {"ok": False, "error": "refused"}
            if self._state is not None:
                body["side_effect_state"] = self._state
            return body

    for declared, expected in (("none", "none"), ("partial", "partial"), (None, "unknown")):
        config = _mcp_config(client=_Declaring(declared))
        transport = build_transport("mcp", config)
        await transport.open(_mcp_context(config))
        outcome = await transport.write("setpoint", 1.0)
        assert outcome.ok is False
        assert outcome.side_effect_state == expected, declared


@pytest.mark.asyncio
async def test_mcp_treats_a_raising_client_as_an_unusable_device() -> None:
    """A client fault is not a device answer, so it must not become a reading."""

    class _Exploding:
        async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
            # Only the read fails, so the raise is isolated from open()'s own probe.
            if tool_name == "bench_read":
                raise RuntimeError("session closed")
            return {"ok": True}

    config = _mcp_config(client=_Exploding())
    transport = build_transport("mcp", config)
    await transport.open(_mcp_context(config))
    with pytest.raises(TransportError) as caught:
        await transport.read("sensor")
    assert caught.value.failure_code == "mcp_call_raised"


@pytest.mark.asyncio
async def test_mcp_prefers_a_server_supplied_sequence_over_its_own_counter() -> None:
    """A local counter never gaps, so it cannot show that a sample was dropped.

    Only a server that numbers its own samples can, which is why the path is declarable
    and why the fallback is documented as a real loss of information rather than an
    equivalent.
    """

    class _Numbering:
        def __init__(self) -> None:
            self._n = 40

        async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
            if tool_name != "bench_read":
                return {"ok": True}
            self._n += 2  # a gap the transport must pass through, not smooth over
            return {"ok": True, "value": 1.0, "seq": self._n}

    config = _mcp_config(client=_Numbering(), sequence_path="seq")
    transport = build_transport("mcp", config)
    await transport.open(_mcp_context(config))
    first = await transport.read("sensor")
    second = await transport.read("sensor")
    assert (first.sequence, second.sequence) == (42, 44)

    local = build_transport("mcp", _mcp_config())
    local_config = _mcp_config()
    await local.open(_mcp_context(local_config))
    a = await local.read("sensor")
    b = await local.read("sensor")
    assert (a.sequence, b.sequence) == (1, 2), "without a declared path, numbering is local"


@pytest.mark.asyncio
async def test_mcp_probe_without_a_probe_tool_reports_local_state_only() -> None:
    """An undeclared probe is not an error; it is less information."""
    config = _mcp_config(probe_tool="")
    stub = config["client"]
    transport = build_transport("mcp", config)
    status = await transport.open(_mcp_context(config))
    assert status.connected is True
    assert [name for name, _ in stub.calls] == [], "no tool may be invented for a probe"


def test_the_client_provider_is_restorable() -> None:
    """A process-global mutation with a lifetime must hand back its undo.

    MCP servers are rebuilt on every runtime config reload, so a provider that could
    not be replaced would leave a device calling a closed session.
    """
    from leapflow.hardware.transports.mcp import set_mcp_client_provider

    sentinel = _StubMcpServer({})
    undo = set_mcp_client_provider(lambda: sentinel)
    try:
        transport = build_transport("mcp", _mcp_config_without_client())
        assert transport._require_client() is sentinel
    finally:
        undo()
    after = build_transport("mcp", _mcp_config_without_client())
    with pytest.raises(TransportError):
        after._require_client()


def _mcp_config_without_client() -> dict[str, Any]:
    config = _mcp_config()
    config.pop("client")
    return config
