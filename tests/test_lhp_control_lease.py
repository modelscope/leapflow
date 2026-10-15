# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for control.v1 lease/deadman and session management (P0-D / plan §Test).

Covers: open/renew/update/close; lease expiry triggering halt; TCP disconnect
triggering comm_loss_policy; hold-not-halt; cross-connection rejection; registry
guard for LHP transports.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.hardware.lhp.control_sessions import (
    DEFAULT_LEASE_S,
    MAX_LEASE_S,
    MIN_LEASE_S,
    NodeControlSessions,
)


# ── Stubs ──────────────────────────────────────────────────────────────────


class _StubTransport:
    """Minimal transport that tracks halt calls."""

    halt_count: int = 0

    async def halt(self) -> None:
        self.halt_count += 1


class _StubChannel:
    channel_id = "ch"
    is_writable = True
    is_readable = True
    direction = "readwrite"
    quantity = "test"
    unit = "unit"
    effect = "configure"
    envelope = SimpleNamespace(
        declared=False, min_value=0.0, max_value=100.0,
        requires_interlocks=(), reversible=True,
    )

    @property
    def interlock_channels(self) -> tuple:
        return ()


@dataclass
class _StubContext:
    device_id: str = "arm"
    channels: tuple = field(default_factory=lambda: (_StubChannel(),))
    transport: Any = field(default_factory=lambda: SimpleNamespace(kind="mock", config={}))


class _StubRegistry:
    """Minimal registry for control session tests."""

    def __init__(self, *, degraded: bool = False):
        self._contexts = {"arm": _StubContext()}
        self._transport = _StubTransport()
        self._degraded = degraded
        self._loop_running = False

    def context(self, device_id: str):
        return self._contexts.get(device_id)

    async def transport(self, device_id: str):
        return self._transport

    def is_device_degraded(self, device_id: str) -> bool:
        return self._degraded

    def start_control_loop(self, device_id, policy, *, config=None):
        self._loop_running = True

    def stop_control_loop(self, device_id) -> dict:
        self._loop_running = False
        return {"stats": {"cycle_count": 42}}


def _make_sessions(*, degraded: bool = False) -> tuple[NodeControlSessions, _StubRegistry]:
    reg = _StubRegistry(degraded=degraded)
    sessions = NodeControlSessions(reg)
    return sessions, reg


CONN_A = "conn-a-0001"
CONN_B = "conn-b-0002"


# ── Open / close lifecycle ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_open_and_close_lifecycle():
    sessions, reg = _make_sessions()
    result = await sessions.open(
        {"device_id": "arm", "mode": "pid", "frequency_hz": 100.0,
         "channels": ["ch"], "lease_s": 2.0, "comm_loss_policy": "halt"},
        connection_id=CONN_A,
    )
    assert result["ok"] is True
    assert "control_id" in result
    assert result["lease_s"] == 2.0
    cid = result["control_id"]

    # Close it
    close = await sessions.close({"control_id": cid}, connection_id=CONN_A)
    assert close["ok"] is True
    assert close["status"] == "stopped"
    assert close["reason"] == "close"
    assert "uptime_s" in close


@pytest.mark.asyncio
async def test_renew_extends_lease():
    sessions, _ = _make_sessions()
    result = await sessions.open(
        {"device_id": "arm", "mode": "pid", "lease_s": 1.0, "comm_loss_policy": "halt"},
        connection_id=CONN_A,
    )
    cid = result["control_id"]

    renew = await sessions.renew({"control_id": cid}, connection_id=CONN_A)
    assert renew["ok"] is True
    assert renew["expires_in_s"] > 0.5  # Just renewed, should have > 0.5s


@pytest.mark.asyncio
async def test_update_renews_lease():
    sessions, _ = _make_sessions()
    result = await sessions.open(
        {"device_id": "arm", "mode": "pid", "lease_s": 2.0, "comm_loss_policy": "halt"},
        connection_id=CONN_A,
    )
    cid = result["control_id"]

    update = await sessions.update(
        {"control_id": cid, "targets": {"level": 50.0}},
        connection_id=CONN_A,
    )
    assert update["ok"] is True
    assert update["expires_in_s"] > 1.5


@pytest.mark.asyncio
async def test_status_reports_active_session():
    sessions, _ = _make_sessions()
    result = await sessions.open(
        {"device_id": "arm", "mode": "pid", "comm_loss_policy": "halt"},
        connection_id=CONN_A,
    )
    status = await sessions.status({"device_id": "arm"})
    assert status["ok"] is True
    assert len(status["sessions"]) == 1
    assert status["sessions"][0]["mode"] == "pid"


# ── Cross-connection rejection ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cross_connection_renew_rejected():
    sessions, _ = _make_sessions()
    result = await sessions.open(
        {"device_id": "arm", "mode": "pid", "comm_loss_policy": "halt"},
        connection_id=CONN_A,
    )
    cid = result["control_id"]

    renew = await sessions.renew({"control_id": cid}, connection_id=CONN_B)
    assert renew["ok"] is False
    assert renew["failure_code"] == "not_session_owner"


@pytest.mark.asyncio
async def test_cross_connection_update_rejected():
    sessions, _ = _make_sessions()
    result = await sessions.open(
        {"device_id": "arm", "mode": "pid", "comm_loss_policy": "halt"},
        connection_id=CONN_A,
    )
    cid = result["control_id"]

    update = await sessions.update({"control_id": cid}, connection_id=CONN_B)
    assert update["ok"] is False
    assert update["failure_code"] == "not_session_owner"


@pytest.mark.asyncio
async def test_cross_connection_close_rejected():
    sessions, _ = _make_sessions()
    result = await sessions.open(
        {"device_id": "arm", "mode": "pid", "comm_loss_policy": "halt"},
        connection_id=CONN_A,
    )
    cid = result["control_id"]

    close = await sessions.close({"control_id": cid}, connection_id=CONN_B)
    assert close["ok"] is False
    assert close["failure_code"] == "not_session_owner"


# ── Lease expiry / comm_loss_policy ───────────────────────────────────────


@pytest.mark.asyncio
async def test_lease_expiry_triggers_halt():
    sessions, reg = _make_sessions()
    result = await sessions.open(
        {"device_id": "arm", "mode": "pid", "lease_s": 0.2,
         "comm_loss_policy": "halt"},
        connection_id=CONN_A,
    )
    cid = result["control_id"]

    # Wait for the lease to expire + watchdog to fire
    await asyncio.sleep(0.5)

    # Device should have been halted exactly once
    assert reg._transport.halt_count == 1
    # Session should be halted
    session = sessions._sessions[cid]
    assert session.halted is True


@pytest.mark.asyncio
async def test_hold_policy_does_not_halt():
    sessions, reg = _make_sessions()
    result = await sessions.open(
        {"device_id": "arm", "mode": "pid", "lease_s": 0.2,
         "comm_loss_policy": "hold"},
        connection_id=CONN_A,
    )
    cid = result["control_id"]

    await asyncio.sleep(0.5)

    # Hold policy: loop stopped, but device NOT halted
    assert reg._transport.halt_count == 0
    session = sessions._sessions[cid]
    assert session.halted is True


@pytest.mark.asyncio
async def test_disconnect_triggers_comm_loss():
    sessions, reg = _make_sessions()
    result = await sessions.open(
        {"device_id": "arm", "mode": "pid", "lease_s": 5.0,
         "comm_loss_policy": "halt"},
        connection_id=CONN_A,
    )

    # Simulate connection drop
    await sessions.on_disconnect(CONN_A)

    assert reg._transport.halt_count == 1


@pytest.mark.asyncio
async def test_disconnect_other_connection_does_not_affect_session():
    sessions, reg = _make_sessions()
    await sessions.open(
        {"device_id": "arm", "mode": "pid", "lease_s": 5.0,
         "comm_loss_policy": "halt"},
        connection_id=CONN_A,
    )

    # Different connection disconnects
    await sessions.on_disconnect(CONN_B)

    assert reg._transport.halt_count == 0


# ── Precheck: degradation ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_degraded_device_rejected():
    sessions, _ = _make_sessions(degraded=True)
    result = await sessions.open(
        {"device_id": "arm", "mode": "pid", "comm_loss_policy": "halt"},
        connection_id=CONN_A,
    )
    assert result["ok"] is False
    assert result["failure_code"] == "device_degraded"


# ── Duplicate session rejection ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_duplicate_session_rejected():
    sessions, _ = _make_sessions()
    await sessions.open(
        {"device_id": "arm", "mode": "pid", "lease_s": 5.0,
         "comm_loss_policy": "halt"},
        connection_id=CONN_A,
    )

    result = await sessions.open(
        {"device_id": "arm", "mode": "pid", "lease_s": 5.0,
         "comm_loss_policy": "halt"},
        connection_id=CONN_B,
    )
    assert result["ok"] is False
    assert result["failure_code"] == "control_session_exists"


# ── Lease clamping ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lease_clamped_to_bounds():
    sessions, _ = _make_sessions()

    # Too small → clamp to MIN
    r1 = await sessions.open(
        {"device_id": "arm", "mode": "pid", "lease_s": 0.01,
         "comm_loss_policy": "halt"},
        connection_id=CONN_A,
    )
    assert r1["ok"] is True
    assert r1["lease_s"] == MIN_LEASE_S

    # Clean up for next test
    await sessions.close({"control_id": r1["control_id"]}, connection_id=CONN_A)

    # Too large → clamp to MAX
    r2 = await sessions.open(
        {"device_id": "arm", "mode": "pid", "lease_s": 999.0,
         "comm_loss_policy": "halt"},
        connection_id=CONN_A,
    )
    assert r2["ok"] is True
    assert r2["lease_s"] == MAX_LEASE_S


# ── Unknown device / session ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_device_rejected():
    sessions, _ = _make_sessions()
    result = await sessions.open(
        {"device_id": "nonexistent", "mode": "pid", "comm_loss_policy": "halt"},
        connection_id=CONN_A,
    )
    assert result["ok"] is False
    assert result["failure_code"] == "unknown_device"


@pytest.mark.asyncio
async def test_renew_unknown_session_rejected():
    sessions, _ = _make_sessions()
    result = await sessions.renew(
        {"control_id": "not-real"},
        connection_id=CONN_A,
    )
    assert result["ok"] is False
    assert result["failure_code"] == "unknown_control_session"


# ── Registry guard: LHP transports must not run ControlBus locally ─────


def test_registry_refuses_lhp_control_bus():
    """Host registry raises RuntimeError if start_control_loop is called for LHP transport."""
    from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
    from leapflow.hardware.context import (
        HC_VERSION,
        Channel,
        Direction,
        HardwareContext,
        TransportRef,
    )

    ctx = HardwareContext(
        device_id="remote",
        hc_version=HC_VERSION,
        display_name="Remote",
        transport=TransportRef(kind="lhp", config={"endpoint": "tcp://node:8767"}),
        channels=(
            Channel(
                channel_id="ch",
                direction=Direction.READWRITE.value,
                quantity="test",
                unit="unit",
            ),
        ),
    )

    class _StaticProvider:
        kind = "static"

        def discover(self):
            return (ctx,)

    registry = HardwareRegistry(
        HardwareSettings(enabled=True),
        providers=[_StaticProvider()],
    )
    registry.load()

    with pytest.raises(RuntimeError, match="LHP"):
        registry.start_control_loop("remote", object())


# ── Shutdown ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_shutdown_stops_all_sessions():
    sessions, reg = _make_sessions()
    await sessions.open(
        {"device_id": "arm", "mode": "pid", "lease_s": 10.0,
         "comm_loss_policy": "halt"},
        connection_id=CONN_A,
    )
    assert len(sessions._sessions) == 1

    await sessions.shutdown()
    assert len(sessions._sessions) == 0
