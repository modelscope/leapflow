# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the realtime control loop module.

Covers:
- PIDJointController convergence and clamping
- ImpedanceController spring-damper behaviour
- TrajectoryTracker interpolation and progress
- RealtimeControlLoop session lifecycle
"""

from __future__ import annotations

import threading
import time
from typing import Any, Mapping
from unittest.mock import AsyncMock, MagicMock

import pytest

from leapflow.hardware.context import (
    Channel,
    Direction,
    Envelope,
    HardwareContext,
    HardwareEffect,
)
from leapflow.hardware.control_bus import (
    ControlBusConfig,
    ControlBusStats,
    ControlCommand,
    ControlPolicy,
    ControlState,
    HighFrequencyControlBus,
)
from leapflow.hardware.realtime import (
    ControlTelemetryEvent,
    ImpedanceController,
    PIDJointController,
    RealtimeControlLoop,
    TrajectoryTracker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    positions: Mapping[str, float],
    *,
    velocities: Mapping[str, float] | None = None,
    cycle_number: int = 0,
    cycle_dt_s: float = 0.01,
    timestamp: float | None = None,
) -> ControlState:
    """Factory for ControlState with sensible defaults."""
    return ControlState(
        timestamp=timestamp or time.monotonic(),
        joint_positions=positions,
        joint_velocities=velocities or {},
        sensor_readings={},
        cycle_number=cycle_number,
        cycle_dt_s=cycle_dt_s,
    )


def test_control_bus_refuses_commands_when_device_is_degraded() -> None:
    registry = MagicMock()
    registry.is_device_degraded.return_value = True
    bus = HighFrequencyControlBus(registry)
    bus._device_id = "arm"

    safe = bus._check_safety(
        _make_state({"joint_0": 0.0}),
        ControlCommand(joint_commands={"joint_0": 0.1}),
    )

    assert safe is False
    registry.context.assert_not_called()


def _control_context() -> HardwareContext:
    return HardwareContext(
        device_id="arm",
        channels=(
            Channel(
                channel_id="joint_0",
                direction=Direction.READWRITE.value,
                effect=HardwareEffect.ACTUATE.value,
                envelope=Envelope(
                    declared=True,
                    min_value=-1.0,
                    max_value=1.0,
                    max_rate=10.0,
                ),
            ),
            Channel(
                channel_id="mode",
                direction=Direction.WRITE.value,
                effect=HardwareEffect.CONFIGURE.value,
                envelope=Envelope(declared=True, allowed_values=("idle", "run")),
            ),
        ),
    )


def test_control_bus_enforces_envelope_without_device_safety_policy() -> None:
    registry = MagicMock()
    registry.is_device_degraded.return_value = False
    registry.context.return_value = _control_context()
    bus = HighFrequencyControlBus(registry)
    bus._device_id = "arm"

    safe = bus._check_safety(
        _make_state({"joint_0": 0.0}),
        ControlCommand(joint_commands={"joint_0": 1.5}),
    )

    assert safe is False


@pytest.mark.parametrize("channel_id,value", [("ghost", 0.1), ("mode", "run")])
def test_control_bus_rejects_unknown_or_non_actuation_channels(
    channel_id: str,
    value: Any,
) -> None:
    registry = MagicMock()
    registry.is_device_degraded.return_value = False
    registry.context.return_value = _control_context()
    bus = HighFrequencyControlBus(registry)
    bus._device_id = "arm"

    safe = bus._check_safety(
        _make_state({"joint_0": 0.0}),
        ControlCommand(joint_commands={channel_id: value}),
    )

    assert safe is False


def test_control_bus_enforces_declared_command_rate() -> None:
    registry = MagicMock()
    registry.is_device_degraded.return_value = False
    registry.context.return_value = _control_context()
    bus = HighFrequencyControlBus(registry)
    bus._device_id = "arm"
    bus._last_commands["joint_0"] = (0.0, time.monotonic())

    safe = bus._check_safety(
        _make_state({"joint_0": 0.0}),
        ControlCommand(joint_commands={"joint_0": 1.0}),
    )

    assert safe is False


# ---------------------------------------------------------------------------
# PIDJointController
# ---------------------------------------------------------------------------


class TestPIDJointController:
    """PID controller unit tests."""

    def test_protocol_conformance(self) -> None:
        """PIDJointController satisfies ControlPolicy Protocol."""
        pid = PIDJointController(("j1",))
        assert isinstance(pid, ControlPolicy)

    def test_policy_id(self) -> None:
        pid = PIDJointController(("j1",))
        assert pid.policy_id == "pid_joint"

    def test_convergence(self) -> None:
        """PID should drive position error toward zero over iterations."""
        pid = PIDJointController(("j1",), kp=5.0, ki=0.0, kd=0.0, max_output=10.0)
        pid.set_target({"j1": 1.0})

        position = 0.0
        dt = 0.01
        for i in range(200):
            state = _make_state({"j1": position}, cycle_dt_s=dt, cycle_number=i)
            cmd = pid.compute(state)
            # Simulate velocity-mode: position += velocity_command * dt
            position += cmd.joint_commands["j1"] * dt

        # After 200 iterations at 100 Hz with kp=5, error should be small.
        assert abs(position - 1.0) < 0.05, f"PID did not converge: pos={position}"

    def test_output_clamped(self) -> None:
        """Commands should be clamped to max_output."""
        pid = PIDJointController(("j1",), kp=1000.0, ki=0.0, kd=0.0, max_output=2.0)
        pid.set_target({"j1": 100.0})
        state = _make_state({"j1": 0.0})
        cmd = pid.compute(state)
        assert abs(cmd.joint_commands["j1"]) <= 2.0

    def test_position_mode(self) -> None:
        """In position mode, command = current + clamped_output."""
        pid = PIDJointController(
            ("j1",),
            kp=1.0,
            ki=0.0,
            kd=0.0,
            max_output=0.5,
            output_mode="position",
        )
        pid.set_target({"j1": 10.0})
        state = _make_state({"j1": 1.0})
        cmd = pid.compute(state)
        # output = kp * error = 1.0 * 9.0 → clamped to 0.5
        # command = current + clamped = 1.0 + 0.5 = 1.5
        assert cmd.joint_commands["j1"] == pytest.approx(1.5, abs=0.01)

    def test_reset_clears_integral(self) -> None:
        """reset() should zero the integral accumulator."""
        pid = PIDJointController(("j1",), kp=0.0, ki=10.0, kd=0.0, max_output=100.0)
        pid.set_target({"j1": 1.0})

        # Accumulate integral.
        for i in range(10):
            state = _make_state({"j1": 0.0}, cycle_dt_s=0.01, cycle_number=i)
            pid.compute(state)

        pid.reset()
        state = _make_state({"j1": 0.0}, cycle_dt_s=0.01)
        cmd = pid.compute(state)
        # After reset, integral contribution on first step = ki * error * dt = 10 * 1 * 0.01 = 0.1
        assert abs(cmd.joint_commands["j1"]) < 1.0

    def test_thread_safety_set_target(self) -> None:
        """set_target should not race with compute."""
        pid = PIDJointController(("j1",), kp=1.0, ki=0.0, kd=0.0)
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for _ in range(1000):
                    pid.set_target({"j1": 1.0})
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for i in range(1000):
                    state = _make_state({"j1": 0.0}, cycle_number=i)
                    pid.compute(state)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors

    def test_invalid_output_mode(self) -> None:
        with pytest.raises(ValueError, match="output_mode"):
            PIDJointController(("j1",), output_mode="torque")


# ---------------------------------------------------------------------------
# ImpedanceController
# ---------------------------------------------------------------------------


class TestImpedanceController:
    """Impedance controller unit tests."""

    def test_protocol_conformance(self) -> None:
        imp = ImpedanceController(("j1",))
        assert isinstance(imp, ControlPolicy)

    def test_policy_id(self) -> None:
        imp = ImpedanceController(("j1",))
        assert imp.policy_id == "impedance"

    def test_spring_restoring_force(self) -> None:
        """With only stiffness, controller should push toward reference."""
        imp = ImpedanceController(
            ("j1",),
            stiffness=100.0,
            damping=0.0,
            inertia=1.0,
        )
        imp.set_reference({"j1": 1.0})
        state = _make_state({"j1": 0.0}, velocities={"j1": 0.0})
        cmd = imp.compute(state)
        # F = K * (1.0 - 0.0) = 100 → a = 100 → v_cmd = 0 + 100 * 0.01 = 1.0
        assert cmd.joint_commands["j1"] > 0, "Should push toward reference"

    def test_damping_opposes_velocity(self) -> None:
        """Damping should slow down a moving joint."""
        imp = ImpedanceController(
            ("j1",),
            stiffness=0.0,
            damping=10.0,
            inertia=1.0,
        )
        imp.set_reference({"j1": 0.0}, velocity={"j1": 0.0})
        # Joint moving at v=5.0 with ref=0 and no stiffness → damping decelerates.
        state = _make_state({"j1": 0.0}, velocities={"j1": 5.0})
        cmd = imp.compute(state)
        # F = D * (0 - 5) = -50 → a = -50 → v_cmd = 5 + (-50)*0.01 = 4.5
        assert cmd.joint_commands["j1"] < 5.0

    def test_invalid_inertia(self) -> None:
        with pytest.raises(ValueError, match="inertia"):
            ImpedanceController(("j1",), inertia=0.0)

    def test_reset(self) -> None:
        imp = ImpedanceController(("j1",))
        imp.set_reference({"j1": 42.0})
        imp.reset()
        # After reset, reference should be 0.
        state = _make_state({"j1": 0.0}, velocities={"j1": 0.0})
        cmd = imp.compute(state)
        # K * (0 - 0) + D * (0 - 0) = 0 → v_cmd = 0
        assert abs(cmd.joint_commands["j1"]) < 0.01


# ---------------------------------------------------------------------------
# TrajectoryTracker
# ---------------------------------------------------------------------------


class TestTrajectoryTracker:
    """Trajectory tracker unit tests."""

    def _make_simple_trajectory(
        self,
    ) -> tuple[tuple[float, Mapping[str, float]], ...]:
        """Linear ramp from 0 to 1 over 1 second."""
        return (
            (0.0, {"j1": 0.0}),
            (0.5, {"j1": 0.5}),
            (1.0, {"j1": 1.0}),
        )

    def test_protocol_conformance(self) -> None:
        pid = PIDJointController(("j1",))
        tracker = TrajectoryTracker(pid, self._make_simple_trajectory())
        assert isinstance(tracker, ControlPolicy)

    def test_policy_id(self) -> None:
        pid = PIDJointController(("j1",))
        tracker = TrajectoryTracker(pid, self._make_simple_trajectory())
        assert tracker.policy_id == "trajectory_tracker:pid_joint"

    def test_interpolation_midpoint(self) -> None:
        """At t=0.25 the interpolated target should be ~0.25."""
        pid = PIDJointController(("j1",), kp=1.0, ki=0.0, kd=0.0)
        traj = self._make_simple_trajectory()
        tracker = TrajectoryTracker(pid, traj)

        t0 = 100.0  # arbitrary base time
        # First call sets start_time.
        state0 = _make_state({"j1": 0.0}, timestamp=t0)
        tracker.compute(state0)

        # At t=0.25 → target should be ~0.25
        state1 = _make_state({"j1": 0.0}, timestamp=t0 + 0.25)
        tracker.compute(state1)
        # The PID inner policy should now have target ≈ 0.25.
        with pid._lock:
            assert pid._targets["j1"] == pytest.approx(0.25, abs=0.01)

    def test_once_mode_completes(self) -> None:
        """In 'once' mode, tracker should report halt after trajectory ends."""
        pid = PIDJointController(("j1",))
        traj = self._make_simple_trajectory()
        tracker = TrajectoryTracker(pid, traj, loop_mode="once")

        t0 = 100.0
        state0 = _make_state({"j1": 0.0}, timestamp=t0)
        tracker.compute(state0)

        # Past end of trajectory.
        state_end = _make_state({"j1": 0.0}, timestamp=t0 + 2.0)
        cmd = tracker.compute(state_end)
        assert cmd.halt is True
        assert tracker.is_complete is True

    def test_hold_final_mode(self) -> None:
        """In 'hold_final' mode, tracker should hold last position."""
        pid = PIDJointController(("j1",), kp=1.0, ki=0.0, kd=0.0)
        traj = self._make_simple_trajectory()
        tracker = TrajectoryTracker(pid, traj, loop_mode="hold_final")

        t0 = 100.0
        tracker.compute(_make_state({"j1": 0.0}, timestamp=t0))

        # Past end.
        tracker.compute(_make_state({"j1": 0.0}, timestamp=t0 + 5.0))
        with pid._lock:
            assert pid._targets["j1"] == pytest.approx(1.0, abs=0.01)
        assert tracker.is_complete is True

    def test_loop_mode_wraps(self) -> None:
        """In 'loop' mode, the trajectory should repeat."""
        pid = PIDJointController(("j1",), kp=1.0, ki=0.0, kd=0.0)
        traj = self._make_simple_trajectory()
        tracker = TrajectoryTracker(pid, traj, loop_mode="loop")

        t0 = 100.0
        tracker.compute(_make_state({"j1": 0.0}, timestamp=t0))

        # At t=1.25 with duration 1.0 → effective = 0.25 → target ≈ 0.25
        tracker.compute(_make_state({"j1": 0.0}, timestamp=t0 + 1.25))
        with pid._lock:
            assert pid._targets["j1"] == pytest.approx(0.25, abs=0.05)

    def test_empty_trajectory_raises(self) -> None:
        pid = PIDJointController(("j1",))
        with pytest.raises(ValueError, match="at least one"):
            TrajectoryTracker(pid, ())

    def test_reset(self) -> None:
        pid = PIDJointController(("j1",))
        tracker = TrajectoryTracker(pid, self._make_simple_trajectory())
        state = _make_state({"j1": 0.0}, timestamp=100.0)
        tracker.compute(state)
        tracker.reset()
        assert tracker._start_time is None
        assert tracker.is_complete is False


# ---------------------------------------------------------------------------
# RealtimeControlLoop
# ---------------------------------------------------------------------------


class TestRealtimeControlLoop:
    """Lifecycle and orchestration tests."""

    def _make_mock_bus(self) -> MagicMock:
        """Create a mock HighFrequencyControlBus."""
        bus = MagicMock(spec=HighFrequencyControlBus)
        bus.is_running = False
        bus.async_start = AsyncMock()
        bus.async_stop = AsyncMock()
        bus.stats = ControlBusStats(cycles=42, overruns=1, mean_cycle_ms=9.5)
        bus._config = ControlBusConfig()
        return bus

    @pytest.mark.asyncio
    async def test_start_pid_returns_session_id(self) -> None:
        bus = self._make_mock_bus()
        loop = RealtimeControlLoop(bus)
        sid = await loop.start_pid("dev0", ("j1", "j2"), {"j1": 1.0, "j2": 0.5})
        assert isinstance(sid, str) and len(sid) == 12
        bus.async_start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_impedance(self) -> None:
        bus = self._make_mock_bus()
        loop = RealtimeControlLoop(bus)
        sid = await loop.start_impedance("dev0", ("j1",), {"j1": 0.5})
        assert isinstance(sid, str)

    @pytest.mark.asyncio
    async def test_start_trajectory(self) -> None:
        bus = self._make_mock_bus()
        loop = RealtimeControlLoop(bus)
        traj = ((0.0, {"j1": 0.0}), (1.0, {"j1": 1.0}))
        sid = await loop.start_trajectory("dev0", traj)
        assert isinstance(sid, str)
        # Verify the policy passed to bus is a TrajectoryTracker.
        call_args = bus.async_start.call_args
        policy = call_args.args[1] if len(call_args.args) > 1 else call_args[0][1]
        assert isinstance(policy, TrajectoryTracker)

    @pytest.mark.asyncio
    async def test_stop_returns_stats(self) -> None:
        bus = self._make_mock_bus()
        loop = RealtimeControlLoop(bus)
        await loop.start_pid("dev0", ("j1",), {"j1": 1.0})
        result = await loop.stop()
        assert result["status"] == "stopped"
        assert "stats" in result
        bus.async_stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_when_idle(self) -> None:
        bus = self._make_mock_bus()
        loop = RealtimeControlLoop(bus)
        result = await loop.stop()
        assert result["status"] == "no_session"

    @pytest.mark.asyncio
    async def test_status_idle(self) -> None:
        bus = self._make_mock_bus()
        loop = RealtimeControlLoop(bus)
        st = await loop.status()
        assert st["status"] == "idle"

    @pytest.mark.asyncio
    async def test_status_running(self) -> None:
        bus = self._make_mock_bus()
        bus.is_running = True
        loop = RealtimeControlLoop(bus)
        await loop.start_pid("dev0", ("j1",), {"j1": 1.0})
        st = await loop.status()
        assert st["status"] == "running"
        assert "session_id" in st
        assert "uptime_s" in st

    @pytest.mark.asyncio
    async def test_update_target_pid(self) -> None:
        bus = self._make_mock_bus()
        loop = RealtimeControlLoop(bus)
        await loop.start_pid("dev0", ("j1",), {"j1": 0.0})
        await loop.update_target({"j1": 2.0})
        # Verify the PID target was updated.
        session = loop._session
        assert session is not None
        policy = session.policy
        assert isinstance(policy, PIDJointController)
        with policy._lock:
            assert policy._targets["j1"] == 2.0

    @pytest.mark.asyncio
    async def test_update_target_no_session_raises(self) -> None:
        bus = self._make_mock_bus()
        loop = RealtimeControlLoop(bus)
        with pytest.raises(RuntimeError, match="No active"):
            await loop.update_target({"j1": 1.0})

    @pytest.mark.asyncio
    async def test_telemetry_event_structure(self) -> None:
        """ControlTelemetryEvent should be a well-formed frozen dataclass."""
        event = ControlTelemetryEvent(
            session_id="abc123",
            device_id="dev0",
            policy_id="pid_joint",
            cycles=100,
            mean_cycle_ms=9.8,
            max_jitter_ms=0.5,
            overruns=2,
            is_running=True,
            progress=0.5,
        )
        assert event.session_id == "abc123"
        assert event.progress == 0.5

    @pytest.mark.asyncio
    async def test_auto_stop_previous_session(self) -> None:
        """Starting a new session should stop the previous one."""
        bus = self._make_mock_bus()
        bus.is_running = True
        loop = RealtimeControlLoop(bus)
        await loop.start_pid("dev0", ("j1",), {"j1": 0.0})

        # Start a second session — the first should be stopped.
        bus.is_running = True  # Simulate still running.
        await loop.start_pid("dev0", ("j1",), {"j1": 1.0})
        # async_stop should have been called at least once for the first session.
        assert bus.async_stop.await_count >= 1
