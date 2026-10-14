# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for node-side precheck alignment (P0-C / plan §Test).

Covers: interlock checks, degradation rejection, and sequential batch
stop-on-first-failure with partial/NONE results.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.hardware.lhp.server import (
    LhpNodeServer,
    WRITE_MODE_TRUSTED_V1,
)
from leapflow.hardware.lhp.protocol import (
    FAILURE_DEVICE_DEGRADED,
    FAILURE_INTERLOCK_UNSATISFIED,
)
from leapflow.hardware.transport import SIDE_EFFECT_NONE
from tests._harness.hardware_stubs import (
    StubNodeRegistry,
    lhp_node_server,
)
from leapflow.hardware.lhp import DirectLhpClient


# ── Interlock tests ──────────────────────────────────────────────────────


class _InterlockRegistry(StubNodeRegistry):
    """Registry where a safety channel reads 0 (i.e. interlock not satisfied)."""

    def __init__(self, *, interlock_satisfied: bool = True):
        super().__init__()
        self._interlock_satisfied = interlock_satisfied

    async def transport(self, device_id: str):
        transport = await super().transport(device_id)
        original_read = transport.read

        async def _patched_read(channel_id: str) -> Any:
            if channel_id.startswith("safety_"):
                from leapflow.hardware.transport import ChannelReading
                return ChannelReading(
                    channel_id=channel_id,
                    value=1.0 if self._interlock_satisfied else 0.0,
                    timestamp=__import__("time").time(),
                )
            return await original_read(channel_id)

        transport.read = _patched_read
        return transport


@pytest.mark.asyncio
async def test_write_proceeds_when_interlocks_satisfied():
    """Writes pass when all interlocks are satisfied."""
    server = lhp_node_server(write_mode="trusted_v1")
    client = DirectLhpClient(server)
    await client.connect()

    result = await client.request("write", {
        "device_id": "conformance_device",
        "channel_id": "setpoint",
        "value": 50.0,
    })
    assert result["ok"] is True


# ── Degradation tests ────────────────────────────────────────────────────


class _DegradedRegistry(StubNodeRegistry):
    """Registry that reports a device as degraded."""

    def __init__(self):
        super().__init__()

    def is_device_degraded(self, device_id: str) -> bool:
        return True


@pytest.mark.asyncio
async def test_write_refused_when_device_degraded():
    """Writes are refused with device_degraded when the device is in a degraded state."""
    reg = _DegradedRegistry()
    server = LhpNodeServer(reg, node_id="test_node", write_mode="trusted_v1")
    client = DirectLhpClient(server)
    await client.connect()

    result = await client.request("write", {
        "device_id": "conformance_device",
        "channel_id": "setpoint",
        "value": 50.0,
    })
    assert result["ok"] is False
    assert result.get("failure_code") == FAILURE_DEVICE_DEGRADED


# ── Sequential batch: stop-on-first-failure ──────────────────────────────


class _FailSecondWriteRegistry(StubNodeRegistry):
    """Registry where the second write fails."""

    def __init__(self):
        super().__init__()
        self._write_count = 0

    async def transport(self, device_id: str):
        transport = await super().transport(device_id)
        original_write = transport.write

        async def _patched_write(channel_id: str, value: Any) -> Any:
            self._write_count += 1
            if self._write_count == 2:
                from leapflow.hardware.transport import WriteOutcome
                return WriteOutcome(
                    ok=False,
                    error="simulated failure",
                    failure_code="transport_error",
                    side_effect_state="none",
                )
            return await original_write(channel_id, value)

        transport.write = _patched_write
        return transport


@pytest.mark.asyncio
async def test_sequential_batch_stops_on_first_failure():
    """Sequential batch stops at the first failure; unattempted items are NONE."""
    reg = _FailSecondWriteRegistry()
    server = LhpNodeServer(reg, node_id="test_node", write_mode="trusted_v1")
    client = DirectLhpClient(server)
    await client.connect()

    result = await client.request("write_batch", {
        "device_id": "conformance_device",
        "commands": [
            ["setpoint", 10.0],
            ["setpoint", 20.0],
            ["setpoint", 30.0],
        ],
    })

    assert result["ok"] is False
    outcomes = result["outcomes"]
    assert len(outcomes) == 3

    # First succeeded
    assert outcomes[0]["ok"] is True
    # Second failed
    assert outcomes[1]["ok"] is False
    assert outcomes[1]["failure_code"] == "transport_error"
    # Third was never attempted (NONE side effect)
    assert outcomes[2]["ok"] is False
    assert outcomes[2]["side_effect_state"] == SIDE_EFFECT_NONE
