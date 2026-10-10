# Copyright (c) Alibaba, Inc. and its affiliates.
"""Contract tests for fail-closed degradation transitions."""

from __future__ import annotations

import pytest

from leapflow.hardware.context import DegradationPolicy, KinematicsDeclaration
from leapflow.hardware.degradation import DegradationCoordinator
from benchmarks.native.harness import make_channel, make_context, make_registry


@pytest.mark.asyncio
async def test_unknown_degradation_action_halts_and_latches() -> None:
    context = make_context(
        channels=(make_channel("sensor", writable=False),),
        degradation=DegradationPolicy(),
        transport_kind="simulated",
        transport_config={"values": {"sensor": 1.0}},
    )
    registry = make_registry(context)
    coordinator = DegradationCoordinator(registry, context)

    try:
        outcome = await coordinator.execute("unsupported", reason="sensor_loss")

        assert outcome.completed is True
        assert outcome.fallback_action == "halt"
        assert registry.is_device_degraded(context.device_id) is True
        assert registry.degradation_outcome(context.device_id) == outcome
    finally:
        await registry.close_all()


@pytest.mark.asyncio
async def test_continue_blind_requires_manual_clear() -> None:
    context = make_context(
        channels=(make_channel("sensor", writable=False),),
        degradation=DegradationPolicy(sensor_loss_policy="continue_blind"),
        transport_kind="simulated",
        transport_config={"values": {"sensor": 1.0}},
    )
    registry = make_registry(context)
    coordinator = DegradationCoordinator(registry, context)

    try:
        outcome = await coordinator.execute("continue_blind", reason="sensor_loss")

        assert outcome.completed is True
        assert outcome.requires_manual_clear is True
        assert registry.clear_degradation(context.device_id, recovery="healthy_read") is False
        assert registry.is_device_degraded(context.device_id) is True
    finally:
        await registry.close_all()


@pytest.mark.asyncio
async def test_declared_sensor_fallback_requires_verified_switch() -> None:
    context = make_context(
        channels=(
            make_channel("primary_sensor", writable=False),
            make_channel("backup_sensor", writable=False),
        ),
        degradation=DegradationPolicy(
            sensor_loss_policy="switch_sensor",
            sensor_fallbacks={"primary_sensor": "backup_sensor"},
        ),
        transport_kind="simulated",
        transport_config={"values": {"primary_sensor": 0.0, "backup_sensor": 1.0}},
    )
    registry = make_registry(context)
    coordinator = DegradationCoordinator(registry, context)
    switched: list[str] = []

    async def switcher(channel_id: str) -> bool:
        switched.append(channel_id)
        return channel_id == "backup_sensor"

    try:
        outcome = await coordinator.execute(
            "switch_sensor",
            reason="sensor_loss",
            failed_channel_id="primary_sensor",
            sensor_switcher=switcher,
        )

        assert outcome.completed is True
        assert outcome.active_channel_id == "backup_sensor"
        assert switched == ["backup_sensor"]
        assert registry.is_device_degraded(context.device_id) is False
    finally:
        await registry.close_all()


@pytest.mark.asyncio
async def test_safe_return_executes_and_verifies_declared_home_pose() -> None:
    context = make_context(
        channels=(make_channel("joint_0", quantity="joint_position", unit="rad"),),
        kinematics=KinematicsDeclaration(home_joint_targets={"joint_0": 1.25}),
        degradation=DegradationPolicy(safe_return_timeout_s=0.1),
        transport_kind="simulated",
        transport_config={"values": {"joint_0": 0.0}},
    )
    registry = make_registry(context)
    coordinator = DegradationCoordinator(registry, context)

    try:
        outcome = await coordinator.execute("safe_return", reason="sensor_loss")
        transport = registry.get_open_transport(context.device_id)
        reading = await transport.read("joint_0")

        assert outcome.completed is True
        assert outcome.fallback_action == ""
        assert reading.value == pytest.approx(1.25)
        assert registry.is_device_degraded(context.device_id) is True
    finally:
        await registry.close_all()


@pytest.mark.asyncio
async def test_missing_sensor_fallback_fails_closed_with_halt() -> None:
    context = make_context(
        channels=(make_channel("primary_sensor", writable=False),),
        degradation=DegradationPolicy(sensor_loss_policy="switch_sensor"),
        transport_kind="simulated",
        transport_config={"values": {"primary_sensor": 0.0}},
    )
    registry = make_registry(context)
    coordinator = DegradationCoordinator(registry, context)

    try:
        outcome = await coordinator.execute(
            "switch_sensor",
            reason="sensor_loss",
            failed_channel_id="primary_sensor",
        )

        assert outcome.completed is True
        assert outcome.fallback_action == "halt"
        assert registry.is_device_degraded(context.device_id) is True
    finally:
        await registry.close_all()
