"""Test-facing protocols and reusable conformance suite for hardware transports.

This module holds no pytest dependency and nothing heavier than the standard
library plus ``leapflow.hardware.transport`` / ``leapflow.hardware.context``:
it is designed to be importable standalone so that out-of-tree driver test
scripts can ``from leapflow.hardware.testing import run_transport_conformance``
and exercise the full transport contract without pulling in pytest.

``SignalInjector`` is the deterministic control surface. Its distinguishing
method is :meth:`~SignalInjector.advance_clock`: long-run and time-acceleration
tests move the logical clock forward instead of sleeping, so a "7-day" scenario
runs in-process in milliseconds while wall-clock ordering and monotonic
intervals stay consistent.

``run_transport_conformance`` is the reusable conformance runner.  It exercises
the core six-method contract (open / close / read / write / probe / halt),
Reading dual-clock semantics, WriteOutcome side-effect verdicts, TransportError
conventions, and optionally the ``init_required`` gate.  The report it returns
is a frozen dataclass that can be inspected programmatically or printed.
"""

from __future__ import annotations

import asyncio
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

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


@runtime_checkable
class SignalInjector(Protocol):
    """Deterministic control surface over a simulated signal source.

    An implementation lets a test force a specific value, punch a gap in the
    sequence, drop and recover the link, and advance a logical clock -- all
    without reaching into private state and without any real time passing.
    """

    def inject_reading(self, channel_id: str, value: Any, *, quality: str | None = None) -> None:
        """Queue *value* as the next reading of *channel_id*.

        Injected readings take priority over any waveform or stored value and are
        consumed one per read, in the order queued. ``quality`` overrides the
        reported quality for that reading; ``None`` leaves it to the source.
        """
        ...

    def inject_gap(self, channel_id: str, *, dropped: int = 1) -> None:
        """Discard *dropped* samples from *channel_id*, leaving a sequence gap.

        The samples are never delivered; the next reading simply skips *dropped*
        sequence numbers, which is the only evidence that a bounded queue lost
        them.
        """
        ...

    def inject_disconnect(self, *, reconnect_after: int | None = None) -> None:
        """Drop the link now, so reads fail until it recovers.

        ``reconnect_after`` schedules automatic recovery after that many read
        attempts; ``None`` leaves the link down until it is explicitly reopened.
        """
        ...

    def advance_clock(self, seconds: float) -> None:
        """Advance the logical clock by *seconds* without sleeping.

        Every clock a reading carries moves forward by this amount together, so a
        test can fast-forward hours or days in-process. Negative values are
        ignored: the logical clock never runs backwards.
        """
        ...


# ════════════════════════════════════════════════════════════════
# Conformance report types
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ConformanceResult:
    """Outcome of a single conformance check."""

    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        mark = "  ok " if self.passed else " FAIL"
        suffix = f"  {self.detail}" if self.detail else ""
        return f"[{mark}] {self.name}{suffix}"


@dataclass(frozen=True)
class ConformanceReport:
    """Aggregate outcome of a transport conformance run."""

    passed: int
    failed: int
    results: tuple[ConformanceResult, ...]

    def __str__(self) -> str:
        lines = [str(r) for r in self.results]
        lines.append(f"\n{self.passed + self.failed} checks: {self.passed} ok, {self.failed} fail")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════

TransportFactory = Callable[[Mapping[str, Any]], HardwareTransport]


def _conformance_context(
    transport_kind: str, config: Mapping[str, Any],
) -> HardwareContext:
    """Build a two-channel device declaration for conformance checking."""
    return HardwareContext(
        device_id="conformance_device",
        display_name="Conformance device",
        transport=TransportRef(kind=transport_kind, config=dict(config)),
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
                    declared=True, min_value=0.0, max_value=100.0, reversible=True,
                ),
            ),
        ),
        provenance=ContextProvenance(verified_by="conformance"),
    )


def _build_failing_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Derive a config variant that injects a write failure on the first setpoint write."""
    failing = dict(config)
    failing["failures"] = [
        {"channel_id": "setpoint", "on_call": 1, "side_effect_state": "partial"}
    ]
    return failing


def _build_no_halt_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Derive a config variant with halt disabled."""
    no_halt = dict(config)
    no_halt["halt_supported"] = False
    return no_halt


class _Collector:
    """Accumulates individual check results during a conformance run."""

    def __init__(self) -> None:
        self._results: list[ConformanceResult] = []

    async def check(self, name: str, coro: Any) -> None:
        """Run one async check, capturing pass/fail without raising."""
        try:
            detail = await coro
            self._results.append(ConformanceResult(name=name, passed=True, detail=str(detail or "")))
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exception_only(type(exc), exc)
            self._results.append(
                ConformanceResult(name=name, passed=False, detail="".join(tb).strip())
            )

    def report(self) -> ConformanceReport:
        results = tuple(self._results)
        passed = sum(1 for r in results if r.passed)
        return ConformanceReport(passed=passed, failed=len(results) - passed, results=results)


def _assert(condition: bool, message: str) -> None:
    """Raise AssertionError when *condition* is False -- pytest-free assertion."""
    if not condition:
        raise AssertionError(message)


# ════════════════════════════════════════════════════════════════
# Core contract checks (async coroutine functions)
# ════════════════════════════════════════════════════════════════


async def _check_satisfies_protocol(
    factory: TransportFactory, config: Mapping[str, Any],
) -> str:
    transport = factory(config)
    _assert(isinstance(transport, HardwareTransport), "does not satisfy HardwareTransport")
    _assert(isinstance(transport.kind, str) and len(transport.kind) > 0, "kind must be a non-empty str")
    return f"kind={transport.kind}"


async def _check_open_is_idempotent(
    factory: TransportFactory, config: Mapping[str, Any],
) -> str:
    transport = factory(config)
    context = _conformance_context(transport.kind, config)
    first = await transport.open(context)
    second = await transport.open(context)
    _assert(first.connected is True, "first open did not report connected")
    _assert(second.connected is True, "second open did not report connected")
    await transport.close()
    return "idempotent open"


async def _check_close_is_idempotent(
    factory: TransportFactory, config: Mapping[str, Any],
) -> str:
    transport = factory(config)
    context = _conformance_context(transport.kind, config)
    await transport.open(context)
    first_close = await transport.close()
    _assert(first_close.connected is False, "first close still connected")
    second_close = await transport.close()
    _assert(second_close.connected is False, "second close still connected")
    return "double close safe"


async def _check_probe_is_side_effect_free(
    factory: TransportFactory, config: Mapping[str, Any],
) -> str:
    transport = factory(config)
    context = _conformance_context(transport.kind, config)
    await transport.open(context)
    before = await transport.read("sensor")
    await transport.probe()
    await transport.probe()
    after = await transport.read("sensor")
    _assert(before.value == after.value, f"probe changed the value: {before.value} -> {after.value}")
    await transport.close()
    return "probe side-effect free"


async def _check_read_returns_reading_with_identity(
    factory: TransportFactory, config: Mapping[str, Any],
) -> str:
    transport = factory(config)
    context = _conformance_context(transport.kind, config)
    await transport.open(context)
    reading = await transport.read("sensor")
    _assert(isinstance(reading, Reading), f"expected Reading, got {type(reading).__name__}")
    _assert(reading.channel_id == "sensor", f"channel_id was {reading.channel_id!r}")
    _assert(reading.device_id == context.device_id, f"device_id was {reading.device_id!r}")
    await transport.close()
    return f"value={reading.value!r}"


async def _check_read_sequence_increases_monotonically(
    factory: TransportFactory, config: Mapping[str, Any],
) -> str:
    transport = factory(config)
    context = _conformance_context(transport.kind, config)
    await transport.open(context)
    sequences = [(await transport.read("sensor")).sequence for _ in range(4)]
    _assert(sequences == sorted(sequences), f"not sorted: {sequences}")
    _assert(len(set(sequences)) == len(sequences), f"duplicates in: {sequences}")
    await transport.close()
    return f"sequence {sequences}"


async def _check_unknown_channel_raises(
    factory: TransportFactory, config: Mapping[str, Any],
) -> str:
    transport = factory(config)
    context = _conformance_context(transport.kind, config)
    await transport.open(context)
    try:
        await transport.read("no_such_channel")
        raise AssertionError("reading an unknown channel did not raise TransportError")
    except TransportError as exc:
        await transport.close()
        return f"failure_code={exc.failure_code}"


async def _check_operating_before_open_raises(
    factory: TransportFactory, config: Mapping[str, Any],
) -> str:
    transport = factory(config)
    try:
        await transport.read("sensor")
        raise AssertionError("read before open did not raise TransportError")
    except TransportError as exc:
        return f"failure_code={exc.failure_code}"


async def _check_successful_write_reports_definite_effect(
    factory: TransportFactory, config: Mapping[str, Any],
) -> str:
    transport = factory(config)
    context = _conformance_context(transport.kind, config)
    await transport.open(context)
    outcome = await transport.write("setpoint", 42.0)
    _assert(isinstance(outcome, WriteOutcome), f"expected WriteOutcome, got {type(outcome).__name__}")
    _assert(outcome.ok is True, f"write failed: {outcome.error}")
    _assert(
        outcome.side_effect_state != SIDE_EFFECT_NONE,
        "a successful write claimed no side effect",
    )
    await transport.close()
    return f"side_effect={outcome.side_effect_state}"


async def _check_verified_channel_returns_readback(
    factory: TransportFactory, config: Mapping[str, Any],
) -> str:
    transport = factory(config)
    context = _conformance_context(transport.kind, config)
    await transport.open(context)
    outcome = await transport.write("setpoint", 33.0)
    _assert(outcome.readback is not None, "verify_after_write channel gave no readback")
    _assert(
        outcome.readback.channel_id == "setpoint",
        f"readback channel was {outcome.readback.channel_id!r}",
    )
    await transport.close()
    return f"readback={outcome.readback.value!r}"


async def _check_failed_write_never_claims_no_effect(
    factory: TransportFactory,
    config: Mapping[str, Any],
    failing_write_config: Mapping[str, Any] | None = None,
) -> str:
    failing_config = dict(failing_write_config) if failing_write_config is not None else _build_failing_config(config)
    transport = factory(failing_config)
    context = _conformance_context(transport.kind, failing_config)
    await transport.open(context)
    outcome = await transport.write("setpoint", 10.0)
    _assert(outcome.ok is False, "the failure injection did not produce a failure")
    _assert(
        outcome.side_effect_state != SIDE_EFFECT_NONE,
        "a FAILED write claimed no side effect: an error is not proof that nothing happened",
    )
    _assert(
        outcome.effect_may_have_landed is True,
        "effect_may_have_landed should be True for a failed write",
    )
    await transport.close()
    return f"side_effect={outcome.side_effect_state}"


async def _check_halt_reports_capability(
    factory: TransportFactory, config: Mapping[str, Any],
) -> str:
    transport = factory(config)
    context = _conformance_context(transport.kind, config)
    await transport.open(context)
    status = await transport.halt()
    _assert(isinstance(status, TransportStatus), f"expected TransportStatus, got {type(status).__name__}")
    _assert(isinstance(status.halt_supported, bool), "halt_supported must be bool")
    await transport.close()
    return f"halt_supported={status.halt_supported}"


async def _check_transport_without_halt_declares_it(
    factory: TransportFactory,
    config: Mapping[str, Any],
    no_halt_config: Mapping[str, Any] | None = None,
) -> str:
    no_halt = dict(no_halt_config) if no_halt_config is not None else _build_no_halt_config(config)
    transport = factory(no_halt)
    context = _conformance_context(transport.kind, no_halt)
    await transport.open(context)
    status = await transport.halt()
    _assert(status.halt_supported is False, "halt was declared unsupported but reported otherwise")
    await transport.close()
    return "halt correctly declared unsupported"


async def _check_reading_stamps_both_clocks(
    factory: TransportFactory, config: Mapping[str, Any],
) -> str:
    transport = factory(config)
    context = _conformance_context(transport.kind, config)
    await transport.open(context)
    reading = await transport.read("sensor")
    _assert(
        reading.observed_at > 1_500_000_000.0,
        f"observed_at must be wall-clock, got {reading.observed_at}",
    )
    _assert(
        reading.monotonic_at > 0.0,
        f"monotonic_at must be positive, got {reading.monotonic_at}",
    )
    await transport.close()
    return f"wall={reading.observed_at:.1f} mono={reading.monotonic_at:.3f}"


async def _check_reading_evidence_form(
    factory: TransportFactory, config: Mapping[str, Any],
) -> str:
    transport = factory(config)
    context = _conformance_context(transport.kind, config)
    await transport.open(context)
    reading = await transport.read("sensor")
    payload = reading.to_dict()
    _assert("observed_at" in payload, "to_dict() missing observed_at")
    _assert("monotonic_at" not in payload, "to_dict() must not include monotonic_at")
    _assert("timestamp" not in payload, "the ambiguous name 'timestamp' must not appear")
    await transport.close()
    return "evidence form correct"


# ── init_required contract checks ──


async def _check_init_required_read_refused(
    factory: TransportFactory, config: Mapping[str, Any],
) -> str:
    """With init_required on, reads are refused before init."""
    init_config = dict(config)
    init_config["init_required"] = True
    init_config["init_channel"] = "__init__"
    transport = factory(init_config)
    context = _conformance_context(transport.kind, init_config)
    await transport.open(context)
    try:
        await transport.read("sensor")
        raise AssertionError("read before init did not raise TransportError")
    except TransportError as exc:
        _assert(
            exc.failure_code == "not_initialized",
            f"expected failure_code='not_initialized', got {exc.failure_code!r}",
        )
    await transport.close()
    return "read correctly refused before init"


async def _check_init_required_write_refused(
    factory: TransportFactory, config: Mapping[str, Any],
) -> str:
    """With init_required on, ordinary writes are refused with no side effect."""
    init_config = dict(config)
    init_config["init_required"] = True
    init_config["init_channel"] = "__init__"
    transport = factory(init_config)
    context = _conformance_context(transport.kind, init_config)
    await transport.open(context)
    refused = await transport.write("setpoint", 42.0)
    _assert(refused.ok is False, "write before init should fail")
    _assert(
        refused.failure_code == "not_initialized",
        f"expected failure_code='not_initialized', got {refused.failure_code!r}",
    )
    _assert(
        refused.side_effect_state == SIDE_EFFECT_NONE,
        f"un-init write must have no side effect, got {refused.side_effect_state!r}",
    )
    _assert(
        refused.effect_may_have_landed is False,
        "un-init write must not claim effect may have landed",
    )
    await transport.close()
    return "write correctly refused before init"


async def _check_init_required_init_then_operates(
    factory: TransportFactory, config: Mapping[str, Any],
) -> str:
    """After writing the init channel, reads and writes are unblocked."""
    init_config = dict(config)
    init_config["init_required"] = True
    init_config["init_channel"] = "__init__"
    transport = factory(init_config)
    context = _conformance_context(transport.kind, init_config)
    await transport.open(context)
    # Init write should succeed and have a real side effect.
    init_outcome = await transport.write("__init__", "go")
    _assert(init_outcome.ok is True, f"init write failed: {init_outcome.error}")
    _assert(
        init_outcome.side_effect_state != SIDE_EFFECT_NONE,
        "init write must report a side effect",
    )
    # After init, reads and writes must work normally.
    reading = await transport.read("sensor")
    _assert(isinstance(reading, Reading), "read after init did not return a Reading")
    write_outcome = await transport.write("setpoint", 42.0)
    _assert(write_outcome.ok is True, f"write after init failed: {write_outcome.error}")
    await transport.close()
    return "init -> read -> write all passed"


# ════════════════════════════════════════════════════════════════
# Public entry point
# ════════════════════════════════════════════════════════════════


def run_transport_conformance(
    transport_factory: TransportFactory,
    config: Mapping[str, Any],
    *,
    include_init: bool = False,
    failing_write_config: Mapping[str, Any] | None = None,
    no_halt_config: Mapping[str, Any] | None = None,
) -> ConformanceReport:
    """Execute the transport conformance suite and return a structured report.

    This function does **not** depend on pytest -- it is a pure function that can
    be called from any Python environment::

        from leapflow.hardware.testing import run_transport_conformance
        from leapflow.hardware.transports.mock import build_transport

        report = run_transport_conformance(build_transport, {"values": {"sensor": 21.5, "setpoint": 50.0}})
        assert report.failed == 0

    Parameters
    ----------
    transport_factory:
        A callable ``(config) -> HardwareTransport``.  Typically the ``build_transport``
        function from a transport module.
    config:
        The transport configuration dict (same shape as ``TransportRef.config``).
    include_init:
        When ``True``, also verify the ``init_required`` gate semantics.  Only
        applicable to transports that support the ``init_required`` config key
        (currently ``simulated`` and ``mock`` with matching support).
    failing_write_config:
        Optional config variant that provokes a write failure.  When ``None``,
        a generic derivation is attempted (adding a ``failures`` entry).  When
        the transport uses a different mechanism (e.g. MCP stub with
        ``fail_writes=True``), pass the pre-built config here.
    no_halt_config:
        Optional config variant with halt disabled.  When ``None``, a generic
        derivation is attempted (setting ``halt_supported=False``).  When the
        transport disables halt differently (e.g. MCP with ``halt_tool=""``),
        pass the pre-built config here.

    Returns
    -------
    ConformanceReport
        Frozen dataclass with ``passed``, ``failed`` counts and the full
        ``results`` tuple.
    """
    collector = _Collector()

    core_checks: list[tuple[str, Any]] = [
        ("satisfies HardwareTransport", _check_satisfies_protocol(transport_factory, config)),
        ("open is idempotent", _check_open_is_idempotent(transport_factory, config)),
        ("close is idempotent and never raises", _check_close_is_idempotent(transport_factory, config)),
        ("probe is side-effect free", _check_probe_is_side_effect_free(transport_factory, config)),
        ("read returns a Reading with identity", _check_read_returns_reading_with_identity(transport_factory, config)),
        ("read sequence increases monotonically", _check_read_sequence_increases_monotonically(transport_factory, config)),
        ("unknown channel raises TransportError", _check_unknown_channel_raises(transport_factory, config)),
        ("operating before open raises", _check_operating_before_open_raises(transport_factory, config)),
        ("successful write reports a definite effect", _check_successful_write_reports_definite_effect(transport_factory, config)),
        ("verified channel returns a readback", _check_verified_channel_returns_readback(transport_factory, config)),
        ("failed write never claims no effect", _check_failed_write_never_claims_no_effect(transport_factory, config, failing_write_config)),
        ("halt reports capability", _check_halt_reports_capability(transport_factory, config)),
        ("transport without halt declares it", _check_transport_without_halt_declares_it(transport_factory, config, no_halt_config)),
        ("reading stamps both clocks", _check_reading_stamps_both_clocks(transport_factory, config)),
        ("reading evidence form carries only wall clock", _check_reading_evidence_form(transport_factory, config)),
    ]

    if include_init:
        core_checks.extend([
            ("init_required: read refused before init", _check_init_required_read_refused(transport_factory, config)),
            ("init_required: write refused before init", _check_init_required_write_refused(transport_factory, config)),
            ("init_required: init then operates", _check_init_required_init_then_operates(transport_factory, config)),
        ])

    async def _run_all() -> ConformanceReport:
        for name, coro in core_checks:
            await collector.check(name, coro)
        return collector.report()

    # Support being called from both sync and async contexts.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Already inside an event loop (e.g. pytest-asyncio, Jupyter).
        # Create a new loop in a thread to avoid nested-loop issues.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _run_all())
            return future.result()
    else:
        return asyncio.run(_run_all())


__all__ = [
    "ConformanceReport",
    "ConformanceResult",
    "SignalInjector",
    "run_transport_conformance",
]
