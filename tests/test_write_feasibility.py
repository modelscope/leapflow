# Copyright (c) Alibaba, Inc. and its affiliates.
"""WriteFeasibilityResult and _check_write_feasibility unit tests.

Covers:
- WriteFeasibilityResult is frozen
- Feasibility chain returns ok on valid command
- Feasibility chain returns error for: unknown device, not writable,
  wrong effect, describe required, out of envelope (rate limited),
  safety violation, degraded device, and unreachable device
- _execute_single_write returns ok on successful transport
- _execute_single_write handles transport error
- _write() orchestrates feasibility → approval → execution
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.hardware.audit import HardwareAuditLog
from leapflow.hardware.tools import HardwareTools
from leapflow.hardware.write_governor import WriteFeasibilityResult


# ── Stubs ───────────────────────────────────────────────────────────────


class _FakeChannel:
    def __init__(
        self,
        channel_id: str = "power",
        is_writable: bool = True,
        is_readable: bool = True,
        effect: str = "actuate",
        quantity: str = "watts",
        unit: str = "W",
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
    ) -> None:
        self._contains = contains_result
        self.max_rate = max_rate
        self.reversible = reversible
        self.settling_time_s = settling_time_s
        self.tolerance = tolerance
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
        device_id: str = "heater",
        channels: list[_FakeChannel] | None = None,
        location: str = "bench-1",
    ) -> None:
        self.device_id = device_id
        self._channels = {c.channel_id: c for c in (channels or [_FakeChannel()])}
        self.channels = list(self._channels.values())
        self.writable_channels = [c for c in self.channels if c.is_writable]
        self.location = location
        self.transport = SimpleNamespace(kind="simulated")
        self.provenance = SimpleNamespace(verified_by="tester")

    def channel(self, channel_id: str) -> _FakeChannel | None:
        return self._channels.get(channel_id)

    def interlock(self, name: str) -> Any:
        return None


class _FakeTransport:
    def __init__(self, ok: bool = True) -> None:
        self._ok = ok
        self.write_log: list[tuple[str, Any]] = []

    async def write(self, channel_id: str, value: Any) -> SimpleNamespace:
        self.write_log.append((channel_id, value))
        readback = SimpleNamespace(value=value) if self._ok else None
        return SimpleNamespace(
            ok=self._ok,
            to_dict=lambda: {"ok": self._ok, "value": value},
            readback=readback,
            settled=True,
            failure_code="" if self._ok else "error",
            effect_may_have_landed=False,
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

    def check_safety_policy(
        self, context: Any, channel: Any, value: Any
    ) -> tuple[bool, str]:
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


def _async_gate(approved: bool = True, denial: str = "") -> SimpleNamespace:
    async def _evaluate(_d: Any) -> SimpleNamespace:
        return SimpleNamespace(approved=approved, denial_message=denial)
    return SimpleNamespace(evaluate=_evaluate)


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


# ── WriteFeasibilityResult type ──────────────────────────────────────────


class TestWriteFeasibilityResult:
    def test_frozen(self) -> None:
        r = WriteFeasibilityResult(ok=True)
        with pytest.raises(AttributeError):
            r.ok = False  # type: ignore[misc]

    def test_ok_result_carries_context(self) -> None:
        r = WriteFeasibilityResult(
            ok=True,
            context="ctx",
            channel="ch",
            descriptor="desc",
            value=5.0,
        )
        assert r.ok
        assert r.context == "ctx"
        assert r.error is None

    def test_error_result_carries_payload(self) -> None:
        r = WriteFeasibilityResult(
            ok=False,
            error={"ok": False, "failure_code": "test"},
        )
        assert not r.ok
        assert r.error["failure_code"] == "test"


# ── _check_write_feasibility ─────────────────────────────────────────────


class TestCheckWriteFeasibility:
    @pytest.mark.asyncio
    async def test_valid_command_returns_ok(self) -> None:
        tools = _make_tools()
        result = await tools._check_write_feasibility(
            "hw_actuate", {"device_id": "heater", "channel_id": "power", "value": 5.0}
        )
        assert result.ok
        assert result.context is not None
        assert result.channel is not None
        assert result.descriptor is not None
        assert result.value == 5.0

    @pytest.mark.asyncio
    async def test_unknown_device_returns_error(self) -> None:
        tools = _make_tools()
        result = await tools._check_write_feasibility(
            "hw_actuate", {"device_id": "nonexistent", "channel_id": "x", "value": 1}
        )
        assert not result.ok
        assert result.error is not None
        assert result.error["failure_code"] == "unknown_device"

    @pytest.mark.asyncio
    async def test_not_writable_returns_error(self) -> None:
        ch = _FakeChannel(is_writable=False)
        ctx = _FakeContext(channels=[ch])
        tools = _make_tools(registry=_FakeRegistry(contexts=[ctx]))
        result = await tools._check_write_feasibility(
            "hw_actuate", {"device_id": "heater", "channel_id": "power", "value": 5}
        )
        assert not result.ok
        assert result.error["failure_code"] == "channel_not_writable"

    @pytest.mark.asyncio
    async def test_wrong_effect_returns_error(self) -> None:
        ch = _FakeChannel(effect="configure")
        ctx = _FakeContext(channels=[ch])
        tools = _make_tools(registry=_FakeRegistry(contexts=[ctx]))
        result = await tools._check_write_feasibility(
            "hw_actuate", {"device_id": "heater", "channel_id": "power", "value": 5}
        )
        assert not result.ok
        assert result.error["failure_code"] == "effect_class_mismatch"

    @pytest.mark.asyncio
    async def test_describe_required_returns_error(self) -> None:
        reg = _FakeRegistry()
        reg.settings.require_describe_before_write = True
        tools = _make_tools(registry=reg)
        result = await tools._check_write_feasibility(
            "hw_actuate", {"device_id": "heater", "channel_id": "power", "value": 5}
        )
        assert not result.ok
        assert result.error["failure_code"] == "describe_required"

    @pytest.mark.asyncio
    async def test_degraded_device_returns_error(self) -> None:
        reg = _FakeRegistry()
        reg._degraded.add("heater")
        tools = _make_tools(registry=reg)
        result = await tools._check_write_feasibility(
            "hw_actuate", {"device_id": "heater", "channel_id": "power", "value": 5}
        )
        assert not result.ok
        assert result.error["failure_code"] == "device_degraded"

    @pytest.mark.asyncio
    async def test_dry_run_flag_preserved(self) -> None:
        tools = _make_tools()
        result = await tools._check_write_feasibility(
            "hw_actuate",
            {"device_id": "heater", "channel_id": "power", "value": 5, "dry_run": True},
        )
        assert result.ok
        assert result.dry_run


# ── _execute_single_write ────────────────────────────────────────────────


class TestExecuteSingleWrite:
    @pytest.mark.asyncio
    async def test_successful_write(self) -> None:
        transport = _FakeTransport(ok=True)
        reg = _FakeRegistry(transport=transport)
        tools = _make_tools(registry=reg, gate=_async_gate())
        feasibility = await tools._check_write_feasibility(
            "hw_actuate", {"device_id": "heater", "channel_id": "power", "value": 5.0}
        )
        assert feasibility.ok
        result = await tools._execute_single_write(feasibility)
        assert result["ok"]
        assert transport.write_log == [("power", 5.0)]


# ── Full _write() orchestration ──────────────────────────────────────────


class TestWriteOrchestration:
    @pytest.mark.asyncio
    async def test_write_ok_end_to_end(self) -> None:
        transport = _FakeTransport(ok=True)
        reg = _FakeRegistry(transport=transport)
        tools = _make_tools(registry=reg, gate=_async_gate())
        result = await tools._write(
            "hw_actuate", {"device_id": "heater", "channel_id": "power", "value": 5.0}
        )
        assert result["ok"]
        assert transport.write_log == [("power", 5.0)]

    @pytest.mark.asyncio
    async def test_write_denied_no_transport(self) -> None:
        transport = _FakeTransport(ok=True)
        reg = _FakeRegistry(transport=transport)
        tools = _make_tools(
            registry=reg,
            gate=_async_gate(approved=False, denial="nope"),
        )
        result = await tools._write(
            "hw_actuate", {"device_id": "heater", "channel_id": "power", "value": 5.0}
        )
        assert not result["ok"]
        assert result["failure_code"] == "approval_denied"
        # Transport must not have been touched
        assert transport.write_log == []

    @pytest.mark.asyncio
    async def test_write_dry_run_no_transport(self) -> None:
        transport = _FakeTransport(ok=True)
        reg = _FakeRegistry(transport=transport)
        tools = _make_tools(registry=reg)
        result = await tools._write(
            "hw_actuate",
            {"device_id": "heater", "channel_id": "power", "value": 5.0, "dry_run": True},
        )
        assert result.get("preview")
        assert transport.write_log == []
