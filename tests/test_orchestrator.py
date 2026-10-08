# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for MultiDeviceOrchestrator: multi-device orchestration primitives.

Covers SEQUENTIAL, PARALLEL, BARRIER modes, halt_all, affordance resolution,
and leader-follower teleoperation scenarios using mock transports.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from leapflow.robot.orchestrator import (
    DeviceOperation,
    ExecutionMode,
    MultiDeviceOrchestrator,
    OrchestrationStep,
)


# ════════════════════════════════════════════════════════════════
# Mock transport and registry
# ════════════════════════════════════════════════════════════════


@dataclass
class _MockReading:
    """Minimal reading returned by mock transport."""

    device_id: str = ""
    channel_id: str = ""
    value: Any = 0.0


@dataclass
class _MockWriteOutcome:
    """Minimal write outcome returned by mock transport."""

    ok: bool = True
    side_effect_state: str = "committed"


@dataclass
class _MockHaltStatus:
    """Minimal halt status returned by mock transport."""

    connected: bool = True
    halt_supported: bool = True


class MockTransport:
    """Simulates a HardwareTransport for orchestration tests.

    Tracks all reads, writes, and halts for assertion.  ``fail_write``
    causes the next write to return ``ok=False``.
    """

    def __init__(self, device_id: str = "dev_a") -> None:
        self.device_id = device_id
        self.reads: list[str] = []
        self.writes: list[tuple[str, Any]] = []
        self.halted: bool = False
        self.halt_count: int = 0
        self.fail_write: bool = False
        self.values: dict[str, float] = {}

    async def read(self, channel_id: str) -> _MockReading:
        self.reads.append(channel_id)
        return _MockReading(
            device_id=self.device_id,
            channel_id=channel_id,
            value=self.values.get(channel_id, 0.0),
        )

    async def write(self, channel_id: str, value: Any) -> _MockWriteOutcome:
        self.writes.append((channel_id, value))
        if self.fail_write:
            self.fail_write = False
            return _MockWriteOutcome(ok=False, side_effect_state="unknown")
        self.values[channel_id] = float(value)
        return _MockWriteOutcome(ok=True, side_effect_state="committed")

    async def halt(self) -> _MockHaltStatus:
        self.halted = True
        self.halt_count += 1
        return _MockHaltStatus()


class MockRegistry:
    """Simulates HardwareRegistry with pre-registered mock transports."""

    def __init__(self, transports: dict[str, MockTransport] | None = None) -> None:
        self._transports = transports or {}

    async def transport(self, device_id: str) -> MockTransport:
        t = self._transports.get(device_id)
        if t is None:
            raise RuntimeError(f"unknown device: {device_id}")
        return t

    async def read(self, device_id: str, channel_id: str) -> _MockReading:
        t = await self.transport(device_id)
        return await t.read(channel_id)


class MockCapabilityEntry:
    """Minimal CapabilityEntry for affordance resolution tests."""

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id


class MockCapabilityIndex:
    """Simulates CapabilityIndex returning pre-set entries."""

    def __init__(self, mapping: dict[str, list[MockCapabilityEntry]]) -> None:
        self._mapping = mapping

    def resolve(self, affordance: str) -> list[MockCapabilityEntry]:
        return list(self._mapping.get(affordance, []))


class MockHardwareTools:
    """Simulates HardwareTools.batch_actuate for approval-gated path."""

    def __init__(self, fail_device: str = "") -> None:
        self.calls: list[dict] = []
        self._fail_device = fail_device

    async def batch_actuate(self, params: dict) -> dict:
        self.calls.append(params)
        if params.get("device_id") == self._fail_device:
            return {"ok": False, "error": "approval_denied"}
        return {"ok": True, "side_effect_state": "committed"}


# ════════════════════════════════════════════════════════════════
# 1. SEQUENTIAL mode
# ════════════════════════════════════════════════════════════════


class TestSequentialExecution:
    """SEQUENTIAL mode: one-at-a-time, stop on first failure."""

    @pytest.fixture
    def transports(self):
        return {
            "arm_a": MockTransport("arm_a"),
            "arm_b": MockTransport("arm_b"),
        }

    @pytest.fixture
    def orchestrator(self, transports):
        return MultiDeviceOrchestrator(MockRegistry(transports))

    @pytest.mark.asyncio
    async def test_sequential_all_succeed(self, orchestrator, transports):
        """All operations execute in order when none fail."""
        step = OrchestrationStep(
            mode=ExecutionMode.SEQUENTIAL,
            operations=(
                DeviceOperation("arm_a", (("joint1", 1.0),)),
                DeviceOperation("arm_b", (("joint1", 2.0),)),
            ),
            label="seq_ok",
        )
        result = await orchestrator.execute_step(step)
        assert result.ok is True
        assert len(result.step_results) == 2
        assert result.halted is False
        assert transports["arm_a"].writes == [("joint1", 1.0)]
        assert transports["arm_b"].writes == [("joint1", 2.0)]

    @pytest.mark.asyncio
    async def test_sequential_stops_on_failure(self, orchestrator, transports):
        """Stops at first failure; second operation is never executed."""
        transports["arm_a"].fail_write = True
        step = OrchestrationStep(
            mode=ExecutionMode.SEQUENTIAL,
            operations=(
                DeviceOperation("arm_a", (("joint1", 1.0),)),
                DeviceOperation("arm_b", (("joint1", 2.0),)),
            ),
        )
        result = await orchestrator.execute_step(step)
        assert result.ok is False
        assert len(result.step_results) == 1
        assert transports["arm_b"].writes == []


# ════════════════════════════════════════════════════════════════
# 2. PARALLEL mode
# ════════════════════════════════════════════════════════════════


class TestParallelExecution:
    """PARALLEL mode: concurrent execution, halt_all on failure."""

    @pytest.fixture
    def transports(self):
        return {
            "arm_a": MockTransport("arm_a"),
            "arm_b": MockTransport("arm_b"),
        }

    @pytest.fixture
    def orchestrator(self, transports):
        return MultiDeviceOrchestrator(MockRegistry(transports))

    @pytest.mark.asyncio
    async def test_parallel_all_succeed(self, orchestrator, transports):
        """Both devices written concurrently when none fail."""
        step = OrchestrationStep(
            mode=ExecutionMode.PARALLEL,
            operations=(
                DeviceOperation("arm_a", (("joint1", 1.0),)),
                DeviceOperation("arm_b", (("joint1", 2.0),)),
            ),
        )
        result = await orchestrator.execute_step(step)
        assert result.ok is True
        assert not result.halted
        assert transports["arm_a"].writes == [("joint1", 1.0)]
        assert transports["arm_b"].writes == [("joint1", 2.0)]

    @pytest.mark.asyncio
    async def test_parallel_failure_triggers_halt_all(self, orchestrator, transports):
        """A failure in one device triggers halt_all on ALL participating devices."""
        transports["arm_a"].fail_write = True
        step = OrchestrationStep(
            mode=ExecutionMode.PARALLEL,
            operations=(
                DeviceOperation("arm_a", (("joint1", 1.0),)),
                DeviceOperation("arm_b", (("joint1", 2.0),)),
            ),
        )
        result = await orchestrator.execute_step(step)
        assert result.ok is False
        assert result.halted is True
        # Both devices must have been halted.
        assert transports["arm_a"].halted is True
        assert transports["arm_b"].halted is True


# ════════════════════════════════════════════════════════════════
# 3. BARRIER mode
# ════════════════════════════════════════════════════════════════


class TestBarrierExecution:
    """BARRIER mode: all devices reach target, or timeout → halt."""

    @pytest.fixture
    def transports(self):
        return {
            "arm_a": MockTransport("arm_a"),
            "arm_b": MockTransport("arm_b"),
        }

    @pytest.fixture
    def orchestrator(self, transports):
        return MultiDeviceOrchestrator(MockRegistry(transports))

    @pytest.mark.asyncio
    async def test_barrier_all_settle(self, orchestrator):
        """Barrier succeeds when all devices reach their targets immediately."""
        step = OrchestrationStep(
            mode=ExecutionMode.BARRIER,
            operations=(
                DeviceOperation("arm_a", (("joint1", 1.0),)),
                DeviceOperation("arm_b", (("joint1", 2.0),)),
            ),
            barrier_timeout_s=2.0,
        )
        result = await orchestrator.execute_step(step)
        assert result.ok is True
        assert not result.halted

    @pytest.mark.asyncio
    async def test_barrier_timeout_triggers_halt(self, transports):
        """Barrier timeout triggers halt_all when a device never reaches target."""
        # Override arm_b read to always return 0.0, simulating a device
        # that never reaches the commanded 999.0 target.
        original_read = transports["arm_b"].read

        async def _stuck_read(channel_id: str) -> _MockReading:
            return _MockReading(
                device_id="arm_b", channel_id=channel_id, value=0.0
            )

        transports["arm_b"].read = _stuck_read
        # Override registry.read for arm_b too.
        registry = MockRegistry(transports)
        orch = MultiDeviceOrchestrator(registry)

        step = OrchestrationStep(
            mode=ExecutionMode.BARRIER,
            operations=(
                DeviceOperation("arm_a", (("joint1", 1.0),)),
                DeviceOperation("arm_b", (("joint1", 999.0),)),
            ),
            barrier_timeout_s=0.15,
        )
        result = await orch.execute_step(step)
        assert result.ok is False
        assert result.halted is True
        assert transports["arm_a"].halted is True
        assert transports["arm_b"].halted is True


# ════════════════════════════════════════════════════════════════
# 4. halt_all
# ════════════════════════════════════════════════════════════════


class TestHaltAll:
    """halt_all: concurrent emergency stop of all devices."""

    @pytest.mark.asyncio
    async def test_halt_all_stops_all_devices(self):
        """Every device receives a halt() call concurrently."""
        ta = MockTransport("arm_a")
        tb = MockTransport("arm_b")
        registry = MockRegistry({"arm_a": ta, "arm_b": tb})
        orch = MultiDeviceOrchestrator(registry)

        halt_map = await orch.halt_all(("arm_a", "arm_b"))
        assert ta.halted is True
        assert tb.halted is True
        assert halt_map["arm_a"]["halted"] is True
        assert halt_map["arm_b"]["halted"] is True

    @pytest.mark.asyncio
    async def test_halt_all_deduplicates_ids(self):
        """Duplicate device_ids are halted only once."""
        ta = MockTransport("arm_a")
        registry = MockRegistry({"arm_a": ta})
        orch = MultiDeviceOrchestrator(registry)

        await orch.halt_all(("arm_a", "arm_a", "arm_a"))
        assert ta.halt_count == 1

    @pytest.mark.asyncio
    async def test_halt_all_empty(self):
        """Empty device list returns empty map."""
        orch = MultiDeviceOrchestrator(MockRegistry())
        result = await orch.halt_all(())
        assert result == {}

    @pytest.mark.asyncio
    async def test_halt_all_handles_transport_error(self):
        """halt_all does not raise when a device transport fails."""
        registry = MockRegistry({})  # No devices → transport() raises
        orch = MultiDeviceOrchestrator(registry)
        halt_map = await orch.halt_all(("missing_device",))
        assert halt_map["missing_device"]["halted"] is False
        assert "error" in halt_map["missing_device"]


# ════════════════════════════════════════════════════════════════
# 5. Affordance resolution
# ════════════════════════════════════════════════════════════════


class TestResolveByAffordance:
    """resolve_by_affordance routes to CapabilityIndex."""

    def test_resolve_returns_device_ids(self):
        index = MockCapabilityIndex({
            "grasp": [MockCapabilityEntry("arm_a"), MockCapabilityEntry("arm_b")],
        })
        orch = MultiDeviceOrchestrator(MockRegistry(), capability_index=index)
        result = orch.resolve_by_affordance("grasp")
        assert result == ("arm_a", "arm_b")

    def test_resolve_unknown_affordance(self):
        index = MockCapabilityIndex({})
        orch = MultiDeviceOrchestrator(MockRegistry(), capability_index=index)
        assert orch.resolve_by_affordance("fly") == ()

    def test_resolve_without_index(self):
        orch = MultiDeviceOrchestrator(MockRegistry())
        assert orch.resolve_by_affordance("grasp") == ()


# ════════════════════════════════════════════════════════════════
# 6. Leader-follower scenario
# ════════════════════════════════════════════════════════════════


class TestLeaderFollowerScenario:
    """Leader reads state, follower writes — a teleoperation pattern."""

    @pytest.mark.asyncio
    async def test_leader_follower_teleop(self):
        """Read from leader, write to follower in a two-step plan."""
        leader = MockTransport("leader")
        leader.values["joint1"] = 1.57
        follower = MockTransport("follower")
        registry = MockRegistry({"leader": leader, "follower": follower})
        orch = MultiDeviceOrchestrator(registry)

        # Step 1: read leader's state.
        leader_reading = await registry.read("leader", "joint1")
        target = leader_reading.value

        # Step 2: write to follower.
        step = OrchestrationStep(
            mode=ExecutionMode.SEQUENTIAL,
            operations=(
                DeviceOperation("follower", (("joint1", target),)),
            ),
            label="teleop_write",
        )
        result = await orch.execute_step(step)
        assert result.ok is True
        assert follower.writes == [("joint1", 1.57)]


# ════════════════════════════════════════════════════════════════
# 7. Approval-gated path (HardwareTools)
# ════════════════════════════════════════════════════════════════


class TestApprovalGatedPath:
    """Operations use batch_actuate when hardware_tools is provided."""

    @pytest.mark.asyncio
    async def test_tools_path_used_when_provided(self):
        tools = MockHardwareTools()
        registry = MockRegistry({"arm_a": MockTransport("arm_a")})
        orch = MultiDeviceOrchestrator(registry, hardware_tools=tools)

        step = OrchestrationStep(
            mode=ExecutionMode.SEQUENTIAL,
            operations=(
                DeviceOperation("arm_a", (("joint1", 1.0),)),
            ),
        )
        result = await orch.execute_step(step)
        assert result.ok is True
        assert len(tools.calls) == 1
        assert tools.calls[0]["device_id"] == "arm_a"

    @pytest.mark.asyncio
    async def test_tools_failure_propagates(self):
        tools = MockHardwareTools(fail_device="arm_a")
        registry = MockRegistry({"arm_a": MockTransport("arm_a")})
        orch = MultiDeviceOrchestrator(registry, hardware_tools=tools)

        step = OrchestrationStep(
            mode=ExecutionMode.SEQUENTIAL,
            operations=(DeviceOperation("arm_a", (("joint1", 1.0),)),),
        )
        result = await orch.execute_step(step)
        assert result.ok is False


# ════════════════════════════════════════════════════════════════
# 8. execute_plan
# ════════════════════════════════════════════════════════════════


class TestExecutePlan:
    """Multi-step plan execution."""

    @pytest.mark.asyncio
    async def test_plan_sequential_steps(self):
        ta = MockTransport("arm_a")
        registry = MockRegistry({"arm_a": ta})
        orch = MultiDeviceOrchestrator(registry)

        plan = (
            OrchestrationStep(
                mode=ExecutionMode.SEQUENTIAL,
                operations=(DeviceOperation("arm_a", (("j1", 1.0),)),),
                label="step1",
            ),
            OrchestrationStep(
                mode=ExecutionMode.SEQUENTIAL,
                operations=(DeviceOperation("arm_a", (("j2", 2.0),)),),
                label="step2",
            ),
        )
        result = await orch.execute_plan(plan)
        assert result.ok is True
        assert ta.writes == [("j1", 1.0), ("j2", 2.0)]

    @pytest.mark.asyncio
    async def test_plan_stops_on_failed_step(self):
        ta = MockTransport("arm_a")
        registry = MockRegistry({"arm_a": ta})
        orch = MultiDeviceOrchestrator(registry)

        plan = (
            OrchestrationStep(
                mode=ExecutionMode.SEQUENTIAL,
                operations=(DeviceOperation("arm_a", (("j1", 1.0),)),),
                label="step1",
            ),
            OrchestrationStep(
                mode=ExecutionMode.SEQUENTIAL,
                operations=(DeviceOperation("missing", (("j1", 1.0),)),),
                label="step2_fail",
            ),
            OrchestrationStep(
                mode=ExecutionMode.SEQUENTIAL,
                operations=(DeviceOperation("arm_a", (("j3", 3.0),)),),
                label="step3_never",
            ),
        )
        result = await orch.execute_plan(plan)
        assert result.ok is False
        # Step 3 should never have executed.
        assert ("j3", 3.0) not in ta.writes
