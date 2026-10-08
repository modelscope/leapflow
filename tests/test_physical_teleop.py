# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for physical trajectory types and teleop bridge.

Covers:
- PhysicalTrajectoryStep construction, serialization
- PhysicalTrajectory builder pattern, finalization, serialization
- LeaderFollowerBridge mock teleop loop
- KinestheticRecorder mock sampling
- Halt / stop behavior
- _parse_teach_start_args helper
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from leapflow.robot.trajectory import (
    PhysicalTrajectory,
    PhysicalTrajectoryStep,
    make_trajectory_id,
)
from leapflow.robot.teleop import (
    KinestheticRecorder,
    LeaderFollowerBridge,
    TeleopBridge,
    get_physical_session,
    register_physical_session,
    remove_physical_session,
)


# ---------------------------------------------------------------------------
# Fixtures: mock hardware registry
# ---------------------------------------------------------------------------

@dataclass
class _MockChannel:
    channel_id: str
    is_readable: bool = True
    is_writable: bool = False


@dataclass
class _MockContext:
    device_id: str
    channels: tuple[_MockChannel, ...] = ()


@dataclass
class _MockReading:
    channel_id: str
    value: float


@dataclass
class _MockBatchReading:
    device_id: str
    readings: tuple[_MockReading, ...]


class _MockTransport:
    """Minimal transport that records writes and returns preset readings."""

    kind = "mock"

    def __init__(self, positions: dict[str, float] | None = None) -> None:
        self._positions = positions or {}
        self.writes: list[tuple[str, Any]] = []

    async def read(self, channel_id: str) -> _MockReading:
        return _MockReading(channel_id=channel_id, value=self._positions.get(channel_id, 0.0))

    async def write(self, channel_id: str, value: Any) -> None:
        self.writes.append((channel_id, value))
        self._positions[channel_id] = float(value)


class MockRegistry:
    """Minimal mock of HardwareRegistry for teleop/kinesthetic tests."""

    def __init__(
        self,
        contexts: dict[str, _MockContext] | None = None,
        positions: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self._contexts = contexts or {}
        self._transports: dict[str, _MockTransport] = {}
        for device_id, pos in (positions or {}).items():
            self._transports[device_id] = _MockTransport(pos)

    def context(self, device_id: str) -> _MockContext | None:
        return self._contexts.get(device_id)

    async def transport(self, device_id: str) -> _MockTransport:
        if device_id not in self._transports:
            self._transports[device_id] = _MockTransport()
        return self._transports[device_id]

    async def read_batch(
        self, device_id: str, channel_ids: tuple[str, ...]
    ) -> _MockBatchReading:
        transport = await self.transport(device_id)
        readings = []
        for ch_id in channel_ids:
            r = await transport.read(ch_id)
            readings.append(r)
        return _MockBatchReading(device_id=device_id, readings=tuple(readings))

    def device_io(self, device_id: str) -> Any:
        return _NoopLock()

    def device_io_batch(self, device_id: str) -> Any:
        return _NoopLock()


class _NoopLock:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# PhysicalTrajectoryStep tests
# ---------------------------------------------------------------------------

class TestPhysicalTrajectoryStep:
    def test_construction_minimal(self) -> None:
        step = PhysicalTrajectoryStep(
            timestamp=1000.0,
            monotonic_at=500.0,
            joint_positions={"j1": 0.5, "j2": 1.0},
        )
        assert step.timestamp == 1000.0
        assert step.monotonic_at == 500.0
        assert step.joint_positions == {"j1": 0.5, "j2": 1.0}
        assert step.joint_velocities == {}
        assert step.gripper_state is None
        assert step.camera_frames == {}
        assert step.sensor_readings == {}
        assert step.action_source == "teleop"
        assert step.sequence == 0

    def test_construction_full(self) -> None:
        step = PhysicalTrajectoryStep(
            timestamp=1000.0,
            monotonic_at=500.0,
            joint_positions={"j1": 0.5},
            joint_velocities={"j1": 0.1},
            gripper_state=0.8,
            camera_frames={"cam0": "/tmp/frame_001.jpg"},
            sensor_readings={"force": 3.14},
            action_source="kinesthetic",
            sequence=42,
        )
        assert step.gripper_state == 0.8
        assert step.camera_frames == {"cam0": "/tmp/frame_001.jpg"}
        assert step.action_source == "kinesthetic"
        assert step.sequence == 42

    def test_to_dict_minimal(self) -> None:
        step = PhysicalTrajectoryStep(
            timestamp=100.0,
            monotonic_at=50.0,
            joint_positions={"j1": 1.0},
        )
        d = step.to_dict()
        assert d["timestamp"] == 100.0
        assert d["joint_positions"] == {"j1": 1.0}
        assert "joint_velocities" not in d  # empty → omitted
        assert "gripper_state" not in d  # None → omitted

    def test_to_dict_full(self) -> None:
        step = PhysicalTrajectoryStep(
            timestamp=100.0,
            monotonic_at=50.0,
            joint_positions={"j1": 1.0},
            joint_velocities={"j1": 0.5},
            gripper_state=0.3,
            camera_frames={"cam": "ref"},
            sensor_readings={"f": 1.0},
        )
        d = step.to_dict()
        assert d["joint_velocities"] == {"j1": 0.5}
        assert d["gripper_state"] == 0.3
        assert d["camera_frames"] == {"cam": "ref"}
        assert d["sensor_readings"] == {"f": 1.0}

    def test_frozen(self) -> None:
        step = PhysicalTrajectoryStep(
            timestamp=1.0, monotonic_at=1.0, joint_positions={"j": 0.0}
        )
        with pytest.raises(AttributeError):
            step.timestamp = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PhysicalTrajectory tests
# ---------------------------------------------------------------------------

class TestPhysicalTrajectory:
    def test_empty_trajectory(self) -> None:
        t = PhysicalTrajectory(
            trajectory_id="abc", device_id="robot.follower", goal="pick up"
        )
        assert t.sample_count == 0
        assert t.duration_s == 0.0
        assert t.steps == ()

    def test_with_step(self) -> None:
        t = PhysicalTrajectory(
            trajectory_id="t1", device_id="dev", goal="test"
        )
        step = PhysicalTrajectoryStep(
            timestamp=1.0, monotonic_at=1.0, joint_positions={"j": 0.5}
        )
        t2 = t.with_step(step)
        assert t.sample_count == 0  # original unchanged
        assert t2.sample_count == 1
        assert t2.steps[0] is step

    def test_finalized(self) -> None:
        t = PhysicalTrajectory(
            trajectory_id="t1",
            device_id="dev",
            goal="test",
            started_at=1000.0,
        )
        t2 = t.finalized(ended_at=1010.0)
        assert t2.ended_at == 1010.0
        assert t2.duration_s == 10.0
        assert t.ended_at == 0.0  # original unchanged

    def test_finalized_default_time(self) -> None:
        t = PhysicalTrajectory(
            trajectory_id="t1", device_id="dev", goal="", started_at=1.0
        )
        before = time.time()
        t2 = t.finalized()
        after = time.time()
        assert before <= t2.ended_at <= after

    def test_to_dict(self) -> None:
        step = PhysicalTrajectoryStep(
            timestamp=1.0, monotonic_at=1.0, joint_positions={"j": 0.1}
        )
        t = PhysicalTrajectory(
            trajectory_id="tid",
            device_id="dev",
            goal="g",
            steps=(step,),
            started_at=1.0,
            ended_at=2.0,
            metadata={"mode": "teleop"},
        )
        d = t.to_dict()
        assert d["trajectory_id"] == "tid"
        assert d["sample_count"] == 1
        assert d["duration_s"] == 1.0
        assert len(d["steps"]) == 1

    def test_make_trajectory_id(self) -> None:
        tid = make_trajectory_id()
        assert len(tid) == 16
        assert tid != make_trajectory_id()  # unique


# ---------------------------------------------------------------------------
# LeaderFollowerBridge tests
# ---------------------------------------------------------------------------

class TestLeaderFollowerBridge:
    @pytest.fixture
    def registry(self) -> MockRegistry:
        return MockRegistry(
            contexts={
                "leader": _MockContext(
                    "leader",
                    (_MockChannel("j1"), _MockChannel("j2")),
                ),
                "follower": _MockContext(
                    "follower",
                    (_MockChannel("j1"), _MockChannel("j2")),
                ),
            },
            positions={
                "leader": {"j1": 1.0, "j2": 2.0},
                "follower": {"j1": 0.0, "j2": 0.0},
            },
        )

    @pytest.mark.asyncio
    async def test_start_stop(self, registry: MockRegistry) -> None:
        bridge = LeaderFollowerBridge(
            registry,
            leader_device_id="leader",
            follower_device_id="follower",
            goal="test teleop",
            control_rate_hz=100.0,  # fast for test
        )
        assert not bridge.is_active
        await bridge.start()
        assert bridge.is_active

        # Let it run a few cycles.
        await asyncio.sleep(0.1)

        trajectory = await bridge.stop()
        assert not bridge.is_active
        assert trajectory.sample_count > 0
        assert trajectory.duration_s > 0.0
        assert trajectory.goal == "test teleop"
        assert trajectory.device_id == "follower"

        # Check step contents.
        step = trajectory.steps[0]
        assert step.action_source == "teleop"
        assert "j1" in step.joint_positions

    @pytest.mark.asyncio
    async def test_protocol_compliance(self, registry: MockRegistry) -> None:
        bridge = LeaderFollowerBridge(
            registry,
            leader_device_id="leader",
            follower_device_id="follower",
        )
        assert isinstance(bridge, TeleopBridge)

    @pytest.mark.asyncio
    async def test_joint_map(self, registry: MockRegistry) -> None:
        bridge = LeaderFollowerBridge(
            registry,
            leader_device_id="leader",
            follower_device_id="follower",
            joint_map={"j1": "motor_a", "j2": "motor_b"},
            control_rate_hz=100.0,
        )
        await bridge.start()
        await asyncio.sleep(0.05)
        trajectory = await bridge.stop()
        # The bridge should have written to the mapped channels.
        transport = await registry.transport("follower")
        written_channels = {ch for ch, _ in transport.writes}
        assert "motor_a" in written_channels or "motor_b" in written_channels

    @pytest.mark.asyncio
    async def test_double_start_noop(self, registry: MockRegistry) -> None:
        bridge = LeaderFollowerBridge(
            registry,
            leader_device_id="leader",
            follower_device_id="follower",
        )
        await bridge.start()
        await bridge.start()  # should be idempotent
        assert bridge.is_active
        await bridge.stop()


# ---------------------------------------------------------------------------
# KinestheticRecorder tests
# ---------------------------------------------------------------------------

class TestKinestheticRecorder:
    @pytest.fixture
    def registry(self) -> MockRegistry:
        return MockRegistry(
            contexts={
                "arm": _MockContext(
                    "arm",
                    (_MockChannel("j1"), _MockChannel("j2"), _MockChannel("j3")),
                ),
            },
            positions={"arm": {"j1": 0.1, "j2": 0.2, "j3": 0.3}},
        )

    @pytest.mark.asyncio
    async def test_record_and_stop(self, registry: MockRegistry) -> None:
        recorder = KinestheticRecorder(
            registry,
            device_id="arm",
            goal="grasp demo",
            sample_rate_hz=100.0,
        )
        assert not recorder.is_active
        await recorder.start()
        assert recorder.is_active

        await asyncio.sleep(0.1)

        trajectory = await recorder.stop()
        assert not recorder.is_active
        assert trajectory.sample_count > 0
        assert trajectory.duration_s > 0.0
        assert trajectory.goal == "grasp demo"
        assert trajectory.device_id == "arm"

        step = trajectory.steps[0]
        assert step.action_source == "kinesthetic"
        assert "j1" in step.joint_positions
        assert "j2" in step.joint_positions

    @pytest.mark.asyncio
    async def test_sequence_increments(self, registry: MockRegistry) -> None:
        recorder = KinestheticRecorder(
            registry,
            device_id="arm",
            sample_rate_hz=200.0,
        )
        await recorder.start()
        await asyncio.sleep(0.1)
        trajectory = await recorder.stop()

        sequences = [s.sequence for s in trajectory.steps]
        assert sequences == list(range(len(sequences)))

    @pytest.mark.asyncio
    async def test_double_start_noop(self, registry: MockRegistry) -> None:
        recorder = KinestheticRecorder(registry, device_id="arm")
        await recorder.start()
        await recorder.start()  # idempotent
        assert recorder.is_active
        await recorder.stop()


# ---------------------------------------------------------------------------
# Session tracking helpers
# ---------------------------------------------------------------------------

class TestSessionTracking:
    def test_register_get_remove(self) -> None:
        bridge = AsyncMock()
        register_physical_session("test_key", bridge)
        assert get_physical_session("test_key") is bridge
        removed = remove_physical_session("test_key")
        assert removed is bridge
        assert get_physical_session("test_key") is None

    def test_get_missing(self) -> None:
        assert get_physical_session("nonexistent") is None

    def test_remove_missing(self) -> None:
        assert remove_physical_session("nonexistent") is None


# ---------------------------------------------------------------------------
# _parse_teach_start_args tests
# ---------------------------------------------------------------------------

class TestParseTeachStartArgs:
    def test_empty(self) -> None:
        from leapflow.cli.commands.slash_handlers import _parse_teach_start_args

        result = _parse_teach_start_args("")
        assert result["mode"] == "gui"
        assert result["goal"] == ""

    def test_goal_only(self) -> None:
        from leapflow.cli.commands.slash_handlers import _parse_teach_start_args

        result = _parse_teach_start_args("pick up the cup")
        assert result["mode"] == "gui"
        assert result["goal"] == "pick up the cup"

    def test_teleop_mode(self) -> None:
        from leapflow.cli.commands.slash_handlers import _parse_teach_start_args

        result = _parse_teach_start_args(
            "--mode=teleop --leader=arm_leader --follower=arm_follower"
        )
        assert result["mode"] == "teleop"
        assert result["leader"] == "arm_leader"
        assert result["follower"] == "arm_follower"

    def test_kinesthetic_mode(self) -> None:
        from leapflow.cli.commands.slash_handlers import _parse_teach_start_args

        result = _parse_teach_start_args("--mode=kinesthetic --device=robot_arm")
        assert result["mode"] == "kinesthetic"
        assert result["device"] == "robot_arm"

    def test_mode_with_goal(self) -> None:
        from leapflow.cli.commands.slash_handlers import _parse_teach_start_args

        result = _parse_teach_start_args(
            "--mode=teleop --leader=l --follower=f pick up cup"
        )
        assert result["mode"] == "teleop"
        assert result["goal"] == "pick up cup"

    def test_space_separated_args(self) -> None:
        from leapflow.cli.commands.slash_handlers import _parse_teach_start_args

        result = _parse_teach_start_args("--mode kinesthetic --device arm")
        assert result["mode"] == "kinesthetic"
        assert result["device"] == "arm"
