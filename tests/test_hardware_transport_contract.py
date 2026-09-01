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
from typing import Any, Callable, Mapping

import pytest

from leapflow.hardware.context import (
    Channel,
    ContextProvenance,
    Direction,
    Envelope,
    HardwareContext,
    HardwareEffect,
    Quality,
    TransportRef,
)
from leapflow.hardware.testing import run_transport_conformance
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
    writable: bool = True
    """Whether this transport can command anything at all.

    A declared shortfall in the same spirit as ``halt_supported=False``: a sensor-only
    bus has no write path, so the three write cases are replaced by the stricter
    contract that every write is refused *and* proves no effect landed. Not an
    exemption -- a read-only transport that accepted a write, or refused it with an
    uncertain verdict, still fails.
    """


_MOCK_CONFIG: dict[str, Any] = {
    "values": {"sensor": 21.5, "setpoint": 50.0},
    "halt_supported": True,
}

_SIMULATED_CONFIG: dict[str, Any] = {
    # Static values on both channels: the generic conformance cases assume a read
    # is repeatable, so the default declaration must not drive a waveform. Fault
    # injection is exercised by the dedicated cases lower in this file.
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


_HOST_CONFIG: dict[str, Any] = {
    # Narrowed to the two channels the standard library can always answer, so the case
    # behaves identically on every runner. The conformance context declares its own
    # ``sensor``/``setpoint`` channels; the host transport resolves those against the
    # declaration rather than the probe table, and reports SUSPECT quality for a
    # channel it cannot read -- which is exactly the contract under test.
    "include": "cpu",
}


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
    _Case(
        kind="simulated",
        config=_SIMULATED_CONFIG,
        failing_write={
            **_SIMULATED_CONFIG,
            "failures": [
                {"channel_id": "setpoint", "on_call": 1, "side_effect_state": "partial"}
            ],
        },
        without_halt={**_SIMULATED_CONFIG, "halt_supported": False},
    ),
    _Case(
        kind="host",
        config=_HOST_CONFIG,
        # Both variants are the base config: this transport has one write path and one
        # halt answer, and neither is configurable. A host has no emergency stop and
        # no commandable channel, so the shortfall is structural rather than a setting.
        failing_write=_HOST_CONFIG,
        without_halt=_HOST_CONFIG,
        writable=False,
    ),
)

_EXTERNAL_ONLY_TRANSPORTS = frozenset(
    {
        # Needs an importable third-party driver module by definition, so it cannot
        # be exercised without one. Its failure modes are covered separately in
        # test_hardware_context.py.
        "python",
        # Needs a physical camera or microphone *and* a capture backend (ffmpeg or
        # opencv-python) by definition. Running the generic cases against it would
        # open the machine's camera during `pytest` -- on macOS that raises a system
        # permission dialog mid-run, which is a worse outcome than the coverage is
        # worth. Its contracts are exercised against a fake frame source in
        # tests/test_hardware_media.py: read-only refusal with a provable NONE verdict,
        # frame metadata without bytes, and the FrameTransport capability check.
        "media",
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
async def test_successful_write_reports_a_definite_side_effect(case: _Case, transport_case) -> None:
    transport, context = transport_case
    await transport.open(context)
    outcome = await transport.write("setpoint", 42.0)
    assert isinstance(outcome, WriteOutcome)
    if not case.writable:
        # The read-only shortfall, and a stricter contract than the writable one: the
        # call never reached anything, so "no effect" is provable and must be reported
        # -- an uncertain verdict here would block replay of every later action.
        assert outcome.ok is False
        assert outcome.side_effect_state == SIDE_EFFECT_NONE
        assert outcome.error
        return
    assert outcome.ok is True
    # A successful physical write has definitely landed; reporting "none" would
    # tell the recovery layer it is safe to replay.
    assert outcome.side_effect_state != SIDE_EFFECT_NONE


@pytest.mark.asyncio
async def test_verify_after_write_channel_returns_a_readback(case: _Case, transport_case) -> None:
    if not case.writable:
        pytest.skip("a read-only transport never performs the write a readback follows")
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

    A read-only transport is the one honest exception, and it is covered by
    ``test_successful_write_reports_a_definite_side_effect`` above: there the refusal
    *must* report ``none``, because the call provably never reached a device.
    """
    if not case.writable:
        pytest.skip("covered by the read-only refusal contract, which requires 'none'")
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
# Conformance suite: reusable runner exercised as a regression guard
# ════════════════════════════════════════════════════════════════


def _factory_for_kind(kind: str) -> Callable[[Mapping[str, Any]], HardwareTransport]:
    """Return a factory that wraps ``build_transport`` for the given kind."""
    def _factory(config: Mapping[str, Any]) -> HardwareTransport:
        return build_transport(kind, config)
    return _factory


@pytest.mark.parametrize("case", _TRANSPORT_CASES, ids=[c.kind for c in _TRANSPORT_CASES])
def test_conformance_suite_passes(case: _Case) -> None:
    """The reusable conformance runner must agree with the hand-written cases.

    This is the regression guard: if the extracted suite diverges from the
    hand-written tests, this test fails and pinpoints the disagreement.
    """
    factory = _factory_for_kind(case.kind)
    report = run_transport_conformance(
        factory,
        case.config,
        failing_write_config=case.failing_write,
        no_halt_config=case.without_halt,
        writable=case.writable,
    )
    failures = [r for r in report.results if not r.passed]
    assert report.failed == 0, (
        f"{case.kind}: {report.failed} conformance checks failed:\n"
        + "\n".join(str(f) for f in failures)
    )


@pytest.mark.parametrize(
    "case",
    [c for c in _TRANSPORT_CASES if c.kind == "simulated"],
    ids=[c.kind for c in _TRANSPORT_CASES if c.kind == "simulated"],
)
def test_conformance_suite_with_init_required(case: _Case) -> None:
    """The init_required variant must also pass for transports that support it."""
    factory = _factory_for_kind(case.kind)
    report = run_transport_conformance(factory, case.config, include_init=True)
    failures = [r for r in report.results if not r.passed]
    assert report.failed == 0, (
        f"{case.kind} (init_required): {report.failed} conformance checks failed:\n"
        + "\n".join(str(f) for f in failures)
    )


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


# ═════════════════════════════════════════════════════════════════
# SimulatedTransport: parameterised fault injection and a logical clock
# ═════════════════════════════════════════════════════════════════
#
# The generic conformance cases above already prove SimulatedTransport satisfies
# the six-method contract and stamps both clocks. These cases exercise the
# behaviour that only exists to be injected, so a downstream L3 journey or a
# long-run test can assert on it.


async def _open_simulated(**config: Any):
    """Build and open a simulated transport against the conformance declaration."""
    transport = build_transport("simulated", config)
    await transport.open(_conformance_context("simulated", config))
    return transport


@pytest.mark.asyncio
async def test_simulated_waveform_is_a_pure_function_of_the_logical_clock() -> None:
    """A sine channel returns a stable value until the clock advances.

    Time is driven logically, so two reads at the same instant must agree; that
    is what lets a long-run test fast-forward without the value drifting on real
    wall-clock time between samples.
    """
    transport = await _open_simulated(
        values={"setpoint": 50.0},
        waveforms={
            "sensor": {"kind": "sine", "offset": 20.0, "amplitude": 5.0, "period_s": 60.0}
        },
    )
    first = await transport.read("sensor")
    same_instant = await transport.read("sensor")
    assert first.value == pytest.approx(20.0)
    assert same_instant.value == pytest.approx(first.value)

    transport.advance_clock(15.0)  # a quarter period -> peak of the sine
    at_peak = await transport.read("sensor")
    assert at_peak.value == pytest.approx(25.0)


@pytest.mark.asyncio
async def test_simulated_latency_advances_both_clocks_in_lockstep() -> None:
    """Latency shows up as elapsed time, and both clocks move together.

    A downstream rate calculation divides by the monotonic interval and persists
    the wall instant; if the two clocks disagreed on how much time passed, one of
    those would be wrong without failing.
    """
    transport = await _open_simulated(values={"sensor": 1.0}, latency_ms=50.0)
    first = await transport.read("sensor")
    second = await transport.read("sensor")
    wall_delta = second.observed_at - first.observed_at
    mono_delta = second.monotonic_at - first.monotonic_at
    assert wall_delta == pytest.approx(0.05)
    assert mono_delta == pytest.approx(0.05)
    assert (await transport.probe()).latency_ms == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_simulated_dropped_samples_leave_a_sequence_gap() -> None:
    """A dropped sample is invisible except as a hole in the numbering."""
    transport = await _open_simulated(values={"sensor": 1.0}, drop_probability=1.0)
    sequences = [(await transport.read("sensor")).sequence for _ in range(3)]
    gaps = [b - a for a, b in zip(sequences, sequences[1:])]
    assert all(gap > 1 for gap in gaps), (
        f"a dropped sample must widen the sequence step, got {sequences}"
    )


@pytest.mark.asyncio
async def test_simulated_reorder_delivers_adjacent_samples_swapped() -> None:
    """Reordering breaks the sorted order while keeping every sample unique."""
    transport = await _open_simulated(values={"sensor": 1.0}, reorder=True)
    sequences = [(await transport.read("sensor")).sequence for _ in range(4)]
    assert sorted(sequences) != sequences, f"reorder must not stay sorted: {sequences}"
    assert len(set(sequences)) == len(sequences), "reorder must not duplicate a sample"


@pytest.mark.asyncio
async def test_simulated_quality_degradation_marks_readings_untrustworthy() -> None:
    """A degraded channel reports a non-OK quality the pipeline can filter on."""
    transport = await _open_simulated(values={"sensor": 1.0}, quality_degradation=1.0)
    reading = await transport.read("sensor")
    assert reading.quality != Quality.OK.value
    assert reading.is_trustworthy is False


@pytest.mark.asyncio
async def test_simulated_disconnect_sequence_fails_then_recovers() -> None:
    """A scheduled drop refuses reads until the declared reconnect point.

    A refusal is a "could not attempt", so it surfaces as ``TransportError`` --
    never as a fabricated reading, which is the failure mode the sequence exists
    to catch.
    """
    transport = await _open_simulated(
        values={"sensor": 1.0},
        disconnects=[{"on_read": 2, "reconnect_after": 2}],
    )
    assert (await transport.read("sensor")).sequence == 1
    with pytest.raises(TransportError):
        await transport.read("sensor")  # attempt 2: link drops here
    with pytest.raises(TransportError):
        await transport.read("sensor")  # attempt 3: still down
    recovered = await transport.read("sensor")  # attempt 4: reconnect_at reached
    assert isinstance(recovered, Reading)


@pytest.mark.asyncio
async def test_simulated_init_required_gates_the_data_plane_until_initialised() -> None:
    """With ``init_required`` on, reads and writes are refused until an init write.

    This is the readiness contract an L3 journey asserts: a device that has opened
    the link but not run its init/calibration step must refuse to read or actuate,
    and must do so with a verdict recovery can act on -- ``not_initialized`` with
    no side effect -- rather than by fabricating a reading or a silent success.
    """
    transport = await _open_simulated(
        init_required=True,
        init_channel="__init__",
        values={"sensor": 21.5, "setpoint": 50.0},
    )

    # A read before init is a "could not attempt", so it raises.
    with pytest.raises(TransportError) as caught:
        await transport.read("sensor")
    assert caught.value.failure_code == "not_initialized"

    # An ordinary write before init is refused, and proves nothing reached the
    # device so recovery may replay it once the device is ready.
    refused = await transport.write("setpoint", 42.0)
    assert refused.ok is False
    assert refused.failure_code == "not_initialized"
    assert refused.side_effect_state == SIDE_EFFECT_NONE
    assert refused.effect_may_have_landed is False

    # The declared init channel is the one write allowed before initialisation; it
    # runs the handshake and opens the data plane.
    initialised = await transport.write("__init__", "go")
    assert initialised.ok is True
    assert initialised.side_effect_state != SIDE_EFFECT_NONE
    assert transport.initialized is True

    # From here reads and writes behave exactly as an un-gated transport would.
    reading = await transport.read("sensor")
    assert isinstance(reading, Reading)
    assert reading.value == pytest.approx(21.5)
    assert (await transport.write("setpoint", 42.0)).ok is True


@pytest.mark.asyncio
async def test_simulated_without_init_required_is_ready_the_moment_it_opens() -> None:
    """Regression guard: the default declares no init step, so nothing is gated.

    ``init_required`` defaults off, so every existing declaration reads and writes
    immediately after open with no handshake -- the behaviour the rest of the
    conformance suite already assumes.
    """
    transport = await _open_simulated(values={"sensor": 21.5, "setpoint": 50.0})
    assert transport.initialized is True
    assert isinstance(await transport.read("sensor"), Reading)
    assert (await transport.write("setpoint", 42.0)).ok is True


# ═════════════════════════════════════════════════════════════════
# SignalInjector: the deterministic control surface long-run tests drive
# ═════════════════════════════════════════════════════════════════


def test_simulated_transport_implements_the_signal_injector_protocol() -> None:
    """The injector contract is satisfied by ``isinstance``, not by subclassing."""
    from leapflow.hardware.testing import SignalInjector

    transport = build_transport("simulated", _SIMULATED_CONFIG)
    assert isinstance(transport, SignalInjector)


@pytest.mark.asyncio
async def test_injected_reading_takes_priority_then_reverts() -> None:
    """An injected value is delivered once, then normal behaviour resumes."""
    transport = await _open_simulated(values={"sensor": 21.5})
    transport.inject_reading("sensor", 99.9, quality=Quality.SUSPECT.value)
    forced = await transport.read("sensor")
    assert forced.value == pytest.approx(99.9)
    assert forced.quality == Quality.SUSPECT.value
    reverted = await transport.read("sensor")
    assert reverted.value == pytest.approx(21.5)
    assert reverted.quality == Quality.OK.value


@pytest.mark.asyncio
async def test_injected_gap_skips_the_declared_number_of_sequence_numbers() -> None:
    transport = await _open_simulated(values={"sensor": 1.0})
    first = await transport.read("sensor")
    transport.inject_gap("sensor", dropped=5)
    second = await transport.read("sensor")
    assert second.sequence - first.sequence == 6, "one live sample plus five dropped"


@pytest.mark.asyncio
async def test_injected_disconnect_refuses_reads_until_reopened() -> None:
    transport = await _open_simulated(values={"sensor": 1.0})
    assert isinstance(await transport.read("sensor"), Reading)
    transport.inject_disconnect()
    with pytest.raises(TransportError):
        await transport.read("sensor")


@pytest.mark.asyncio
async def test_advance_clock_fast_forwards_without_sleeping() -> None:
    """A day of simulated time passes in-process, on both clocks equally.

    This is the mechanism a 7-day longevity test relies on: no real sleep, and
    wall and monotonic advance by the same amount so ordering stays intact.
    """
    transport = await _open_simulated(values={"sensor": 1.0})
    before = await transport.read("sensor")
    transport.advance_clock(86_400.0)
    after = await transport.read("sensor")
    assert after.observed_at - before.observed_at == pytest.approx(86_400.0)
    assert after.monotonic_at - before.monotonic_at == pytest.approx(86_400.0)


@pytest.mark.asyncio
async def test_advance_clock_ignores_negative_values() -> None:
    """The logical clock never runs backwards, even if asked to."""
    transport = await _open_simulated(values={"sensor": 1.0})
    before = await transport.read("sensor")
    transport.advance_clock(-100.0)
    after = await transport.read("sensor")
    assert after.observed_at >= before.observed_at
