# Copyright (c) Alibaba, Inc. and its affiliates.
"""SafetyPolicy runtime enforcement on write paths.

Verifies that the device-level SafetyPolicy declared in hc.v1 contexts is
checked by registry.check_safety_policy(), the _write() path in HardwareTools,
and the batch_actuate() path.  Also verifies that hc.v0 devices (safety=None)
pass all checks unchanged.
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
    Interlock,
    SafetyPolicy,
    TransportRef,
)
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.tools import HardwareTools
from leapflow.hardware.transport import SIDE_EFFECT_NONE
from leapflow.security.approval import ApprovalDecision
from tests._harness.hardware_stubs import ScriptedHuman

SESSION = "safety-test"

# ════════════════════════════════════════════════════════════════
# Declarations
# ════════════════════════════════════════════════════════════════

_SAFETY = SafetyPolicy(
    max_velocity_rad_s=3.14,
    max_force_n=10.0,
    collision_zones=(),
    emergency_decel_s=0.5,
    require_safety_interlock=False,
)

_SAFETY_WITH_INTERLOCK = SafetyPolicy(
    max_velocity_rad_s=3.14,
    max_force_n=10.0,
    collision_zones=(),
    emergency_decel_s=0.5,
    require_safety_interlock=True,
)


def _robot_context(
    *,
    safety: SafetyPolicy | None = _SAFETY,
    verified: bool = True,
    interlocks: tuple[Interlock, ...] = (),
) -> HardwareContext:
    """A simulated robot arm with velocity and force channels."""
    return HardwareContext(
        device_id="robot_arm_1",
        hc_version="hc.v1",
        display_name="Test Robot",
        vendor="test",
        model="sim",
        location="lab-1",
        halt_supported=True,
        transport=TransportRef(
            kind="mock",
            config={
                "values": {
                    "joint_velocity": 0.0,
                    "joint_force": 0.0,
                    "joint_position": 0.0,
                    "gripper": 0.0,
                    "homed": True,
                },
                "halt_supported": True,
            },
        ),
        provenance=ContextProvenance(
            source="declared",
            verified_by="test" if verified else "",
        ),
        safety=safety,
        interlocks=interlocks,
        channels=(
            Channel(
                channel_id="joint_velocity",
                direction=Direction.READWRITE.value,
                quantity="angular_velocity",
                unit="rad/s",
                effect=HardwareEffect.ACTUATE.value,
                envelope=Envelope(declared=True, min_value=-6.28, max_value=6.28),
            ),
            Channel(
                channel_id="joint_force",
                direction=Direction.READWRITE.value,
                quantity="force",
                unit="N",
                effect=HardwareEffect.ACTUATE.value,
                envelope=Envelope(declared=True, min_value=-50.0, max_value=50.0),
            ),
            Channel(
                channel_id="joint_position",
                direction=Direction.READWRITE.value,
                quantity="angular_position",
                unit="rad",
                effect=HardwareEffect.ACTUATE.value,
                envelope=Envelope(declared=True, min_value=-3.14, max_value=3.14),
            ),
            Channel(
                channel_id="gripper",
                direction=Direction.READWRITE.value,
                quantity="position",
                unit="mm",
                effect=HardwareEffect.ACTUATE.value,
                envelope=Envelope(declared=True, min_value=0.0, max_value=100.0),
            ),
            Channel(
                channel_id="homed",
                direction=Direction.READ.value,
                quantity="state.homed",
                envelope=Envelope(declared=True),
            ),
        ),
    )


def _make_registry(context: HardwareContext) -> HardwareRegistry:
    """Build a minimal registry admitting *context*."""
    from leapflow.hardware.providers import HardwareContextProvider

    class _Single(HardwareContextProvider):
        kind = "test"

        def discover(self) -> list[HardwareContext]:
            return [context]

    settings = HardwareSettings(
        enabled=True,
        unverified_context_policy="allow",
        require_describe_before_write=False,
        envelope_grant=True,
    )
    registry = HardwareRegistry(settings, providers=(_Single(),))
    registry.load()
    return registry


def _make_tools(registry: HardwareRegistry, gate: Any = None) -> HardwareTools:
    return HardwareTools(registry, gate=gate, session_id=SESSION)


# ════════════════════════════════════════════════════════════════
# Registry-level check_safety_policy
# ════════════════════════════════════════════════════════════════


class TestRegistrySafetyCheck:
    """Unit tests for HardwareRegistry.check_safety_policy()."""

    def test_none_safety_passes(self) -> None:
        """hc.v0 devices (safety=None) always pass."""
        ctx = _robot_context(safety=None)
        registry = _make_registry(ctx)
        admitted = registry.context("robot_arm_1")
        assert admitted is not None
        ch = admitted.channel("joint_velocity")
        assert ch is not None
        ok, reason = registry.check_safety_policy(admitted, ch, 999.0)
        assert ok is True
        assert reason == ""

    def test_velocity_within_limit(self) -> None:
        ctx = _robot_context()
        registry = _make_registry(ctx)
        admitted = registry.context("robot_arm_1")
        assert admitted is not None
        ch = admitted.channel("joint_velocity")
        assert ch is not None
        ok, reason = registry.check_safety_policy(admitted, ch, 2.0)
        assert ok is True

    def test_velocity_exceeds_limit(self) -> None:
        ctx = _robot_context()
        registry = _make_registry(ctx)
        admitted = registry.context("robot_arm_1")
        assert admitted is not None
        ch = admitted.channel("joint_velocity")
        assert ch is not None
        ok, reason = registry.check_safety_policy(admitted, ch, 5.0)
        assert ok is False
        assert "velocity" in reason
        assert "3.140" in reason

    def test_negative_velocity_exceeds_limit(self) -> None:
        ctx = _robot_context()
        registry = _make_registry(ctx)
        admitted = registry.context("robot_arm_1")
        assert admitted is not None
        ch = admitted.channel("joint_velocity")
        assert ch is not None
        ok, reason = registry.check_safety_policy(admitted, ch, -5.0)
        assert ok is False
        assert "velocity" in reason

    def test_force_within_limit(self) -> None:
        ctx = _robot_context()
        registry = _make_registry(ctx)
        admitted = registry.context("robot_arm_1")
        assert admitted is not None
        ch = admitted.channel("joint_force")
        assert ch is not None
        ok, reason = registry.check_safety_policy(admitted, ch, 8.0)
        assert ok is True

    def test_force_exceeds_limit(self) -> None:
        ctx = _robot_context()
        registry = _make_registry(ctx)
        admitted = registry.context("robot_arm_1")
        assert admitted is not None
        ch = admitted.channel("joint_force")
        assert ch is not None
        ok, reason = registry.check_safety_policy(admitted, ch, 15.0)
        assert ok is False
        assert "force" in reason
        assert "10.000" in reason

    def test_unrelated_quantity_passes(self) -> None:
        """A channel whose quantity is not velocity/force is not checked."""
        ctx = _robot_context()
        registry = _make_registry(ctx)
        admitted = registry.context("robot_arm_1")
        assert admitted is not None
        ch = admitted.channel("gripper")
        assert ch is not None
        # gripper quantity is "position" -- not velocity or force
        ok, reason = registry.check_safety_policy(admitted, ch, 99.0)
        assert ok is True

    def test_interlock_flag_no_interlocks_refuses(self) -> None:
        """require_safety_interlock=True but no interlocks → refuse."""
        ctx = _robot_context(safety=_SAFETY_WITH_INTERLOCK, interlocks=())
        registry = _make_registry(ctx)
        admitted = registry.context("robot_arm_1")
        assert admitted is not None
        ch = admitted.channel("joint_velocity")
        assert ch is not None
        ok, reason = registry.check_safety_policy(admitted, ch, 1.0)
        assert ok is False
        assert "interlock" in reason.lower()

    def test_interlock_flag_with_interlocks_passes(self) -> None:
        """require_safety_interlock=True with interlocks → passes the registry check."""
        interlocks = (
            Interlock(
                interlock_id="homed_check",
                channel_id="homed",
                operator="eq",
                value=True,
            ),
        )
        ctx = _robot_context(safety=_SAFETY_WITH_INTERLOCK, interlocks=interlocks)
        registry = _make_registry(ctx)
        admitted = registry.context("robot_arm_1")
        assert admitted is not None
        ch = admitted.channel("joint_velocity")
        assert ch is not None
        # Velocity within limit; interlocks exist, so the registry check passes.
        ok, reason = registry.check_safety_policy(admitted, ch, 1.0)
        assert ok is True

    def test_non_numeric_value_passes(self) -> None:
        """A non-numeric value on a velocity channel is not safety-checked."""
        ctx = _robot_context()
        registry = _make_registry(ctx)
        admitted = registry.context("robot_arm_1")
        assert admitted is not None
        ch = admitted.channel("joint_velocity")
        assert ch is not None
        ok, reason = registry.check_safety_policy(admitted, ch, "fast")
        assert ok is True  # as_numeric returns None, so no check applies


# ════════════════════════════════════════════════════════════════
# Tools-level _write() safety integration
# ════════════════════════════════════════════════════════════════


class TestWriteSafetyIntegration:
    """Verify safety checks block writes via the full _write path."""

    @pytest.mark.asyncio
    async def test_velocity_violation_blocks_write(self) -> None:
        ctx = _robot_context()
        registry = _make_registry(ctx)
        tools = _make_tools(registry)
        result = await tools.hw_actuate(
            device_id="robot_arm_1",
            channel_id="joint_velocity",
            value=5.0,  # exceeds 3.14 limit
        )
        assert result["ok"] is False
        assert result["failure_code"] == "safety_policy_violation"
        assert result["side_effect_state"] == SIDE_EFFECT_NONE
        assert "Safety policy violation" in result["error"]

    @pytest.mark.asyncio
    async def test_force_violation_blocks_write(self) -> None:
        ctx = _robot_context()
        registry = _make_registry(ctx)
        tools = _make_tools(registry)
        result = await tools.hw_actuate(
            device_id="robot_arm_1",
            channel_id="joint_force",
            value=-20.0,  # exceeds 10.0 limit
        )
        assert result["ok"] is False
        assert result["failure_code"] == "safety_policy_violation"
        assert result["side_effect_state"] == SIDE_EFFECT_NONE

    @pytest.mark.asyncio
    async def test_within_limits_proceeds(self) -> None:
        """A write within safety limits proceeds to the approval phase."""
        gate = ScriptedHuman(ApprovalDecision.ALLOW)
        ctx = _robot_context()
        registry = _make_registry(ctx)
        tools = _make_tools(registry, gate=gate)
        result = await tools.hw_actuate(
            device_id="robot_arm_1",
            channel_id="joint_velocity",
            value=2.0,  # within 3.14 limit
        )
        # The command proceeds past safety; it either succeeds via mock transport
        # or reaches the approval gate. Either way, no safety_policy_violation.
        assert result.get("failure_code") != "safety_policy_violation"

    @pytest.mark.asyncio
    async def test_no_safety_policy_proceeds(self) -> None:
        """hc.v0 devices with safety=None skip safety checks entirely."""
        gate = ScriptedHuman(ApprovalDecision.ALLOW)
        ctx = _robot_context(safety=None)
        registry = _make_registry(ctx)
        tools = _make_tools(registry, gate=gate)
        result = await tools.hw_actuate(
            device_id="robot_arm_1",
            channel_id="joint_velocity",
            value=999.0,  # would exceed any limit, but no safety policy
        )
        # No safety_policy_violation — the write proceeds (envelope may catch it).
        assert result.get("failure_code") != "safety_policy_violation"


# ════════════════════════════════════════════════════════════════
# batch_actuate() safety integration
# ════════════════════════════════════════════════════════════════


class TestBatchActuateSafety:
    """Verify safety checks block batch writes."""

    @pytest.mark.asyncio
    async def test_batch_velocity_violation(self) -> None:
        ctx = _robot_context()
        registry = _make_registry(ctx)
        tools = _make_tools(registry)
        result = await tools.batch_actuate({
            "device_id": "robot_arm_1",
            "commands": [
                {"channel_id": "joint_position", "value": 1.0},  # ok
                {"channel_id": "joint_velocity", "value": 5.0},  # exceeds limit
            ],
        })
        assert result["ok"] is False
        assert result["failure_code"] == "safety_policy_violation"
        assert result["side_effect_state"] == SIDE_EFFECT_NONE
        assert "Command [1]" in result["error"]

    @pytest.mark.asyncio
    async def test_batch_force_violation(self) -> None:
        ctx = _robot_context()
        registry = _make_registry(ctx)
        tools = _make_tools(registry)
        result = await tools.batch_actuate({
            "device_id": "robot_arm_1",
            "commands": [
                {"channel_id": "joint_force", "value": -15.0},  # exceeds limit
            ],
        })
        assert result["ok"] is False
        assert result["failure_code"] == "safety_policy_violation"

    @pytest.mark.asyncio
    async def test_batch_within_limits(self) -> None:
        """A batch with all values within safety limits proceeds."""
        gate = ScriptedHuman(ApprovalDecision.ALLOW)
        ctx = _robot_context()
        registry = _make_registry(ctx)
        tools = _make_tools(registry, gate=gate)
        result = await tools.batch_actuate({
            "device_id": "robot_arm_1",
            "commands": [
                {"channel_id": "joint_velocity", "value": 1.0},
                {"channel_id": "joint_force", "value": 5.0},
            ],
        })
        assert result.get("failure_code") != "safety_policy_violation"


# ════════════════════════════════════════════════════════════════
# hw_estop emergency_decel_s integration
# ════════════════════════════════════════════════════════════════


class TestEstopEmergencyDecel:
    """Verify that hw_estop surfaces emergency_decel_s from SafetyPolicy."""

    @pytest.mark.asyncio
    async def test_estop_reports_decel_time(self) -> None:
        ctx = _robot_context()
        registry = _make_registry(ctx)
        tools = _make_tools(registry)
        result = await tools.hw_estop(device_id="robot_arm_1")
        assert result["ok"] is True
        assert result["emergency_decel_s"] == 0.5
        assert "note" in result
        assert "0.500" in result["note"]

    @pytest.mark.asyncio
    async def test_estop_no_safety_no_decel(self) -> None:
        """hc.v0 devices without SafetyPolicy have no emergency_decel_s."""
        ctx = _robot_context(safety=None)
        registry = _make_registry(ctx)
        tools = _make_tools(registry)
        result = await tools.hw_estop(device_id="robot_arm_1")
        assert result["ok"] is True
        assert "emergency_decel_s" not in result

    @pytest.mark.asyncio
    async def test_estop_zero_decel_not_reported(self) -> None:
        """A SafetyPolicy with emergency_decel_s=0 does not add the field."""
        safety = SafetyPolicy(emergency_decel_s=0.0)
        ctx = _robot_context(safety=safety)
        registry = _make_registry(ctx)
        tools = _make_tools(registry)
        result = await tools.hw_estop(device_id="robot_arm_1")
        assert result["ok"] is True
        assert "emergency_decel_s" not in result
