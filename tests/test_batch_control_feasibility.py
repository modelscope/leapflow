# Copyright (c) Alibaba, Inc. and its affiliates.
"""BatchFeasibilityResult, ControlFeasibilityResult, and batch/control
three-phase pipeline unit tests.

Covers:
- BatchFeasibilityResult and ControlFeasibilityResult are frozen
- _check_batch_feasibility returns ok on valid commands
- _check_batch_feasibility returns error for: unknown device, unknown channel,
  not writable, wrong effect, out of envelope, empty commands, describe required,
  degraded device
- batch_actuate orchestrates feasibility → approval → execution
- _check_control_feasibility returns ok on valid control params
- _check_control_feasibility returns error for: unknown device, describe required,
  LHP remote, invalid frequency, degraded device
- control_start orchestrates feasibility → approval → execution
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.hardware.audit import HardwareAuditLog
from leapflow.hardware.tools import HardwareTools
from leapflow.hardware.write_governor import (
    BatchFeasibilityResult,
    ControlFeasibilityResult,
)


# ── Stubs ───────────────────────────────────────────────────────────────


class _FakeChannel:
    def __init__(
        self,
        channel_id: str = "joint0",
        is_writable: bool = True,
        is_readable: bool = True,
        effect: str = "actuate",
        quantity: str = "angle",
        unit: str = "deg",
        envelope: Any = None,
        privacy: str = "",
        is_privacy_gated: bool = False,
        representation: str = "scalar",
        verification_channel_id: str = "",
    ) -> None:
        self.channel_id = channel_id
        self.is_writable = is_writable
        self.is_readable = is_readable
        self.effect = effect
        self.quantity = quantity
        self.unit = unit
        self.privacy = privacy
        self.is_privacy_gated = is_privacy_gated
        self.representation = representation
        self.verification_channel_id = verification_channel_id
        self.envelope = envelope or _FakeEnvelope()


class _FakeEnvelope:
    def __init__(
        self,
        contains_result: bool = True,
        max_rate: float | None = None,
        reversible: bool = True,
        settling_time_s: float = 0.0,
        tolerance: float = 0.01,
        min_value: float = 0.0,
        max_value: float = 100.0,
    ) -> None:
        self._contains = contains_result
        self.max_rate = max_rate
        self.reversible = reversible
        self.settling_time_s = settling_time_s
        self.tolerance = tolerance
        self.min_value = min_value
        self.max_value = max_value
        self.requires_interlocks: tuple[str, ...] = ()

    def contains(self, value: Any) -> bool:
        return self._contains

    def rate_wait_s(self, delta: float = 0.0, elapsed_s: float = 0.0) -> float:
        return 0.0

    def band_key(self) -> str:
        return "0-100"

    @property
    def effective_settling_s(self) -> float:
        return self.settling_time_s


class _FakeContext:
    def __init__(
        self,
        device_id: str = "arm",
        channels: list[_FakeChannel] | None = None,
        location: str = "bench-1",
        transport_kind: str = "simulated",
    ) -> None:
        self.device_id = device_id
        default_channels = [
            _FakeChannel(channel_id="joint0"),
            _FakeChannel(channel_id="joint1"),
        ]
        self._channels = {c.channel_id: c for c in (channels or default_channels)}
        self.channels = list(self._channels.values())
        self.writable_channels = [c for c in self.channels if c.is_writable]
        self.location = location
        self.transport = SimpleNamespace(kind=transport_kind)
        self.provenance = SimpleNamespace(verified_by="tester")

    def channel(self, channel_id: str) -> _FakeChannel | None:
        return self._channels.get(channel_id)

    def interlock(self, name: str) -> Any:
        return None


class _FakeTransport:
    def __init__(self, ok: bool = True) -> None:
        self._ok = ok
        self.write_log: list[tuple[str, Any]] = []
        self.write_batch_log: list[Any] = []

    async def write(self, channel_id: str, value: Any) -> SimpleNamespace:
        self.write_log.append((channel_id, value))
        readback = (
            SimpleNamespace(value=value, to_dict=lambda: {"value": value}) if self._ok else None
        )
        return SimpleNamespace(
            ok=self._ok,
            to_dict=lambda: {"ok": self._ok, "value": value},
            readback=readback,
            settled=True,
            failure_code="" if self._ok else "error",
            effect_may_have_landed=False,
            side_effect_state="committed" if self._ok else "none",
            error="" if self._ok else "write failed",
        )

    async def write_batch(self, commands: tuple) -> SimpleNamespace:
        self.write_batch_log.append(commands)
        outcomes = []
        for channel_id, value in commands:
            outcome = await self.write(channel_id, value)
            outcomes.append(outcome)
        return SimpleNamespace(
            ok=all(o.ok for o in outcomes),
            outcomes=tuple(outcomes),
            side_effect_state="committed" if self._ok else "none",
        )

    async def probe(self) -> SimpleNamespace:
        return SimpleNamespace(connected=True, detail="ok")


class _FakeRegistry:
    def __init__(
        self,
        contexts: list[_FakeContext] | None = None,
        transport: _FakeTransport | None = None,
    ) -> None:
        self._contexts = {c.device_id: c for c in (contexts or [_FakeContext()])}
        self._transport = transport or _FakeTransport()
        self._described: set[tuple[str, str]] = set()
        self._degraded: set[str] = set()
        self.settings = SimpleNamespace(
            require_describe_before_write=False,
            trust_skip_enabled=False,
            envelope_grant=True,
        )
        self.outcome_recorder = None
        self._commands: dict[str, tuple[float, float]] = {}
        self._control_loops: list[Any] = []

    def contexts(self) -> list[_FakeContext]:
        return list(self._contexts.values())

    def context(self, device_id: str) -> _FakeContext | None:
        return self._contexts.get(device_id)

    async def transport(self, device_id: str) -> _FakeTransport:
        return self._transport

    async def drop_transport(self, device_id: str) -> None:
        pass

    def mark_described(self, session_id: str, device_id: str) -> None:
        self._described.add((session_id, device_id))

    def was_described(self, session_id: str, device_id: str) -> bool:
        return (session_id, device_id) in self._described

    def is_device_degraded(self, device_id: str) -> bool:
        return device_id in self._degraded

    def check_safety_policy(self, context: Any, channel: Any, value: Any) -> tuple[bool, str]:
        return True, ""

    def last_command(self, device_id: str, channel_id: str) -> tuple[float, float] | None:
        return self._commands.get(f"{device_id}.{channel_id}")

    def record_command(self, device_id: str, channel_id: str, value: float) -> None:
        import time

        self._commands[f"{device_id}.{channel_id}"] = (value, time.monotonic())

    @asynccontextmanager
    async def device_io(self, device_id: str):
        yield

    def channel_summary(self, device_id: str, channel_id: str) -> dict[str, Any]:
        return {}

    def channel_history(self, device_id: str, channel_id: str, limit: int = 12) -> list:
        return []

    def record_event(self, event: Any) -> None:
        pass

    def publish_event(self, event: Any) -> None:
        pass

    def start_control_loop(self, device_id: str, policy: Any, *, config: Any = None) -> None:
        self._control_loops.append({"device_id": device_id, "policy": policy, "config": config})


def _async_gate(approved: bool = True, denial: str = "") -> SimpleNamespace:
    async def evaluate(descriptor: Any) -> SimpleNamespace:
        return SimpleNamespace(approved=approved, denial_message=denial)

    return SimpleNamespace(evaluate=evaluate)


def _make_tools(
    registry: _FakeRegistry | None = None,
    gate: Any = None,
) -> HardwareTools:
    reg = registry or _FakeRegistry()
    return HardwareTools(
        reg,
        gate=gate or _async_gate(),
        audit_log=HardwareAuditLog(None),
    )


# ── BatchFeasibilityResult tests ──────────────────────────────────────


class TestBatchFeasibilityResult:
    def test_frozen(self) -> None:
        r = BatchFeasibilityResult(ok=True)
        with pytest.raises(AttributeError):
            r.ok = False  # type: ignore[misc]

    def test_ok_carries_context(self) -> None:
        ctx = _FakeContext()
        r = BatchFeasibilityResult(
            ok=True,
            context=ctx,
            resolved=(("ch", 1.0),),
            any_irreversible=False,
        )
        assert r.ok
        assert r.context is ctx
        assert r.error is None

    def test_error_carries_payload(self) -> None:
        r = BatchFeasibilityResult(
            ok=False,
            error={"failure_code": "empty_commands"},
        )
        assert not r.ok
        assert r.error is not None
        assert r.error["failure_code"] == "empty_commands"


# ── ControlFeasibilityResult tests ────────────────────────────────────


class TestControlFeasibilityResult:
    def test_frozen(self) -> None:
        r = ControlFeasibilityResult(ok=True)
        with pytest.raises(AttributeError):
            r.ok = False  # type: ignore[misc]

    def test_ok_carries_plan(self) -> None:
        ctx = _FakeContext()
        r = ControlFeasibilityResult(
            ok=True,
            context=ctx,
            mode="pid",
            frequency_hz=100.0,
            targets={"joint0": 45.0},
        )
        assert r.ok
        assert r.mode == "pid"
        assert r.frequency_hz == 100.0
        assert r.targets == {"joint0": 45.0}


# ── _check_batch_feasibility tests ────────────────────────────────────


class TestCheckBatchFeasibility:
    @pytest.mark.asyncio
    async def test_valid_batch(self) -> None:
        tools = _make_tools()
        result = await tools._check_batch_feasibility(
            {
                "device_id": "arm",
                "commands": [
                    {"channel_id": "joint0", "value": 45.0},
                    {"channel_id": "joint1", "value": 30.0},
                ],
            }
        )
        assert result.ok
        assert result.context is not None
        assert len(result.resolved) == 2
        assert result.descriptor is not None

    @pytest.mark.asyncio
    async def test_unknown_device(self) -> None:
        tools = _make_tools()
        result = await tools._check_batch_feasibility(
            {
                "device_id": "nope",
                "commands": [{"channel_id": "joint0", "value": 1.0}],
            }
        )
        assert not result.ok
        assert "nope" in str(result.error)

    @pytest.mark.asyncio
    async def test_empty_commands(self) -> None:
        tools = _make_tools()
        result = await tools._check_batch_feasibility(
            {
                "device_id": "arm",
                "commands": [],
            }
        )
        assert not result.ok
        assert result.error is not None
        assert result.error.get("failure_code") == "empty_commands"

    @pytest.mark.asyncio
    async def test_unknown_channel(self) -> None:
        tools = _make_tools()
        result = await tools._check_batch_feasibility(
            {
                "device_id": "arm",
                "commands": [{"channel_id": "nonexistent", "value": 1.0}],
            }
        )
        assert not result.ok
        assert "unknown_channel" in str(result.error)

    @pytest.mark.asyncio
    async def test_not_writable(self) -> None:
        ch = _FakeChannel(channel_id="joint0", is_writable=False)
        ctx = _FakeContext(channels=[ch])
        tools = _make_tools(registry=_FakeRegistry(contexts=[ctx]))
        result = await tools._check_batch_feasibility(
            {
                "device_id": "arm",
                "commands": [{"channel_id": "joint0", "value": 1.0}],
            }
        )
        assert not result.ok
        assert "channel_not_writable" in str(result.error)

    @pytest.mark.asyncio
    async def test_wrong_effect(self) -> None:
        ch = _FakeChannel(channel_id="joint0", effect="configure")
        ctx = _FakeContext(channels=[ch])
        tools = _make_tools(registry=_FakeRegistry(contexts=[ctx]))
        result = await tools._check_batch_feasibility(
            {
                "device_id": "arm",
                "commands": [{"channel_id": "joint0", "value": 1.0}],
            }
        )
        assert not result.ok
        assert "effect_class_mismatch" in str(result.error)

    @pytest.mark.asyncio
    async def test_out_of_envelope(self) -> None:
        ch = _FakeChannel(
            channel_id="joint0",
            envelope=_FakeEnvelope(contains_result=False),
        )
        ctx = _FakeContext(channels=[ch])
        tools = _make_tools(registry=_FakeRegistry(contexts=[ctx]))
        result = await tools._check_batch_feasibility(
            {
                "device_id": "arm",
                "commands": [{"channel_id": "joint0", "value": 999.0}],
            }
        )
        assert not result.ok
        assert "value_out_of_envelope" in str(result.error)

    @pytest.mark.asyncio
    async def test_degraded_device(self) -> None:
        reg = _FakeRegistry()
        reg._degraded.add("arm")
        tools = _make_tools(registry=reg)
        result = await tools._check_batch_feasibility(
            {
                "device_id": "arm",
                "commands": [{"channel_id": "joint0", "value": 1.0}],
            }
        )
        assert not result.ok
        assert "device_degraded" in str(result.error)

    @pytest.mark.asyncio
    async def test_dry_run_flag(self) -> None:
        tools = _make_tools()
        result = await tools._check_batch_feasibility(
            {
                "device_id": "arm",
                "commands": [{"channel_id": "joint0", "value": 1.0}],
                "dry_run": True,
            }
        )
        assert result.ok
        assert result.dry_run is True

    @pytest.mark.asyncio
    async def test_describe_required(self) -> None:
        reg = _FakeRegistry()
        reg.settings.require_describe_before_write = True
        tools = _make_tools(registry=reg)
        result = await tools._check_batch_feasibility(
            {
                "device_id": "arm",
                "commands": [{"channel_id": "joint0", "value": 1.0}],
            }
        )
        assert not result.ok
        assert "describe_required" in str(result.error)


# ── batch_actuate orchestration tests ─────────────────────────────────


class TestBatchActuateOrchestration:
    @pytest.mark.asyncio
    async def test_end_to_end_ok(self) -> None:
        transport = _FakeTransport(ok=True)
        reg = _FakeRegistry(transport=transport)
        tools = _make_tools(registry=reg)
        result = await tools.batch_actuate(
            {
                "device_id": "arm",
                "commands": [
                    {"channel_id": "joint0", "value": 45.0},
                    {"channel_id": "joint1", "value": 30.0},
                ],
            }
        )
        assert result["ok"] is True
        assert result["batch_size"] == 2
        # Falls back to sequential writes (FakeTransport doesn't implement BatchTransport)
        assert len(transport.write_log) == 2

    @pytest.mark.asyncio
    async def test_denied_no_transport(self) -> None:
        """Denied approval never touches the transport."""
        transport = _FakeTransport()
        reg = _FakeRegistry(transport=transport)
        tools = _make_tools(registry=reg, gate=_async_gate(approved=False, denial="no"))
        result = await tools.batch_actuate(
            {
                "device_id": "arm",
                "commands": [{"channel_id": "joint0", "value": 1.0}],
            }
        )
        assert result["ok"] is False
        assert "approval_denied" in str(result.get("failure_code", ""))
        assert len(transport.write_log) == 0
        assert len(transport.write_batch_log) == 0

    @pytest.mark.asyncio
    async def test_dry_run(self) -> None:
        transport = _FakeTransport()
        reg = _FakeRegistry(transport=transport)
        tools = _make_tools(registry=reg)
        result = await tools.batch_actuate(
            {
                "device_id": "arm",
                "commands": [{"channel_id": "joint0", "value": 1.0}],
                "dry_run": True,
            }
        )
        assert result.get("preview") is True
        assert len(transport.write_log) == 0
        assert len(transport.write_batch_log) == 0


# ── _check_control_feasibility tests ──────────────────────────────────


class TestCheckControlFeasibility:
    @pytest.mark.asyncio
    async def test_valid_pid(self) -> None:
        tools = _make_tools()
        result = await tools._check_control_feasibility(
            {
                "device_id": "arm",
                "mode": "pid",
                "frequency_hz": 100,
                "targets": {"joint0": 45.0},
            }
        )
        assert result.ok
        assert result.mode == "pid"
        assert result.frequency_hz == 100.0
        assert result.context is not None

    @pytest.mark.asyncio
    async def test_unknown_device(self) -> None:
        tools = _make_tools()
        result = await tools._check_control_feasibility(
            {
                "device_id": "nope",
                "mode": "pid",
            }
        )
        assert not result.ok
        assert "nope" in str(result.error)

    @pytest.mark.asyncio
    async def test_describe_required(self) -> None:
        reg = _FakeRegistry()
        reg.settings.require_describe_before_write = True
        tools = _make_tools(registry=reg)
        result = await tools._check_control_feasibility(
            {
                "device_id": "arm",
                "mode": "pid",
            }
        )
        assert not result.ok
        assert "describe_required" in str(result.error)

    @pytest.mark.asyncio
    async def test_lhp_remote_rejected(self) -> None:
        ctx = _FakeContext(transport_kind="lhp")
        reg = _FakeRegistry(contexts=[ctx])
        tools = _make_tools(registry=reg)
        result = await tools._check_control_feasibility(
            {
                "device_id": "arm",
                "mode": "pid",
            }
        )
        assert not result.ok
        assert "remote_control_unsupported" in str(result.error)

    @pytest.mark.asyncio
    async def test_invalid_frequency(self) -> None:
        tools = _make_tools()
        result = await tools._check_control_feasibility(
            {
                "device_id": "arm",
                "mode": "pid",
                "frequency_hz": -10,
            }
        )
        assert not result.ok
        assert "invalid_frequency" in str(result.error)

    @pytest.mark.asyncio
    async def test_nan_frequency(self) -> None:
        tools = _make_tools()
        result = await tools._check_control_feasibility(
            {
                "device_id": "arm",
                "mode": "pid",
                "frequency_hz": float("nan"),
            }
        )
        assert not result.ok
        assert "invalid_frequency" in str(result.error)

    @pytest.mark.asyncio
    async def test_degraded_device(self) -> None:
        reg = _FakeRegistry()
        reg._degraded.add("arm")
        tools = _make_tools(registry=reg)
        result = await tools._check_control_feasibility(
            {
                "device_id": "arm",
                "mode": "pid",
                "targets": {"joint0": 45.0},
            }
        )
        assert not result.ok
        assert "device_degraded" in str(result.error)

    @pytest.mark.asyncio
    async def test_dry_run_flag(self) -> None:
        tools = _make_tools()
        result = await tools._check_control_feasibility(
            {
                "device_id": "arm",
                "mode": "pid",
                "targets": {"joint0": 45.0},
                "dry_run": True,
            }
        )
        assert result.ok
        assert result.dry_run is True


# ── control_start orchestration tests ─────────────────────────────────


class TestControlStartOrchestration:
    @pytest.mark.asyncio
    async def test_end_to_end_ok(self) -> None:
        reg = _FakeRegistry()
        tools = _make_tools(registry=reg)
        result = await tools.control_start(
            device_id="arm",
            mode="pid",
            frequency_hz=100,
            targets={"joint0": 45.0},
        )
        assert result["ok"] is True
        assert result["mode"] == "pid"
        assert result["frequency_hz"] == 100
        assert len(reg._control_loops) == 1

    @pytest.mark.asyncio
    async def test_denied_no_loop(self) -> None:
        reg = _FakeRegistry()
        tools = _make_tools(registry=reg, gate=_async_gate(approved=False, denial="no"))
        result = await tools.control_start(
            device_id="arm",
            mode="pid",
            targets={"joint0": 45.0},
        )
        assert result["ok"] is False
        assert "approval_denied" in str(result.get("failure_code", ""))
        assert len(reg._control_loops) == 0

    @pytest.mark.asyncio
    async def test_dry_run(self) -> None:
        reg = _FakeRegistry()
        tools = _make_tools(registry=reg)
        result = await tools.control_start(
            device_id="arm",
            mode="pid",
            targets={"joint0": 45.0},
            dry_run=True,
        )
        assert result.get("preview") is True
        assert len(reg._control_loops) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["batch", "control"])
async def test_tools_require_secondary_channel_trust(shape: str) -> None:
    reg = _FakeRegistry()
    reg.settings.trust_skip_enabled = True
    trust = SimpleNamespace(may_skip_approval=lambda device, channel, **kw: channel == "joint0")
    tools = HardwareTools(reg, gate=_async_gate(False), hardware_trust_gate=trust)
    if shape == "batch":
        result = await tools.batch_actuate(
            {
                "device_id": "arm",
                "commands": [
                    {"channel_id": "joint0", "value": 1},
                    {"channel_id": "joint1", "value": 1},
                ],
            }
        )
    else:
        result = await tools.control_start(
            device_id="arm",
            mode="pid",
            targets={"joint0": 1, "joint1": 1},
        )
    assert result["failure_code"] == "approval_denied"
    assert result["side_effect_state"] == "none"
    assert not reg._transport.write_log
    assert not reg._control_loops


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["none", "partial", "unknown", "transport", "defect"])
@pytest.mark.parametrize("first_succeeds", [False, True])
async def test_failed_batch_skips_remaining_and_preserves_their_trust_and_pending(
    failure: str,
    first_succeeds: bool,
) -> None:
    from leapflow.hardware.transport import TransportError, WriteOutcome

    transport = _FakeTransport()
    failed_channel = "joint1" if first_succeeds else "joint0"

    async def write(channel_id: str, value: Any) -> WriteOutcome:
        transport.write_log.append((channel_id, value))
        if first_succeeds and channel_id == "joint0":
            return WriteOutcome(ok=True, side_effect_state="committed")
        if failure == "transport":
            raise TransportError("refused")
        if failure == "defect":
            raise TypeError("driver defect")
        return WriteOutcome(ok=False, side_effect_state=failure)

    transport.write = write
    reg = _FakeRegistry(
        contexts=[_FakeContext(channels=[_FakeChannel(channel_id=f"joint{i}") for i in range(3)])],
        transport=transport,
    )
    dropped: list[str] = []
    reg.outcome_recorder = SimpleNamespace(
        record_command=lambda **kw: None,
        drop_pending=lambda device, channel: dropped.append(channel),
    )
    failures: list[tuple[str, bool]] = []
    trust = SimpleNamespace(
        record_failure=lambda device, channel, **kw: failures.append(
            (channel, kw.get("hard", False))
        )
    )
    tools = HardwareTools(reg, gate=_async_gate(True), hardware_trust_gate=trust)
    result = await tools.batch_actuate(
        {
            "device_id": "arm",
            "commands": [{"channel_id": f"joint{i}", "value": 1} for i in range(3)],
        }
    )
    expected = (
        "unknown"
        if failure in ("unknown", "defect")
        else ("partial" if first_succeeds or failure == "partial" else "none")
    )
    assert result["side_effect_state"] == expected
    assert bool(result.get("effect_uncertain")) == (expected != "none")
    assert transport.write_log == (
        [("joint0", 1), ("joint1", 1)] if first_succeeds else [("joint0", 1)]
    )
    assert failures == [(failed_channel, failure == "defect")]
    assert dropped == [failed_channel]
    assert result["channels"][-1]["attempted"] is False
    assert result["channels"][-1]["side_effect_state"] == "none"
    assert result["channels"][-1]["failure_code"] == "batch_not_attempted"
    assert reg.last_command("arm", "joint2") is None
    assert bool(reg.last_command("arm", "joint0")) == first_succeeds


@pytest.mark.asyncio
async def test_explicit_zero_control_frequency_is_refused() -> None:
    result = await _make_tools().control_start(
        device_id="arm",
        mode="pid",
        targets={"joint0": 1},
        frequency_hz=0,
    )
    assert result["failure_code"] == "invalid_frequency"
    assert result["side_effect_state"] == "none"
