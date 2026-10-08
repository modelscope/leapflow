# Copyright (c) Alibaba, Inc. and its affiliates.
"""Realtime control loop: higher-level orchestration on top of ControlBus.

While :class:`HighFrequencyControlBus` provides the raw cycle machinery (thread,
timing, transport I/O), this module adds:

1. Built-in control policies: PID joint controller, impedance controller
2. Policy chaining: System-1.5 neural policy → System-1 servo refiner
3. Trajectory tracking: follow a pre-planned joint trajectory at Hz rate
4. Watchdog: detect and recover from policy computation overruns
5. Telemetry: publish cycle stats to EventBus for LeapBoard display

This is the "System-1" layer in the System-1/1.5/2 hierarchy:

- System-2 (LLM, 1–5 s): high-level planning and reasoning
- System-1.5 (VLA, 5–30 Hz): learned policy inference
- System-1 (this module, 100–1000 Hz): servo-level closed-loop control
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from leapflow.hardware.control_bus import (
    ControlBusConfig,
    ControlCommand,
    ControlPolicy,
    ControlState,
    HighFrequencyControlBus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in control policies
# ---------------------------------------------------------------------------


@dataclass
class _PIDState:
    """Per-joint PID accumulator."""

    integral: float = 0.0
    prev_error: float = 0.0


class PIDJointController:
    """PID controller for joint position tracking.

    Implements :class:`ControlPolicy` Protocol.  One PID loop per joint,
    configurable gains.  Suitable for precise positioning tasks and as the
    System-1 refiner under a System-1.5 neural policy.
    """

    def __init__(
        self,
        joint_ids: tuple[str, ...],
        *,
        kp: float = 10.0,
        ki: float = 0.1,
        kd: float = 1.0,
        max_output: float = 6.28,
        output_mode: str = "velocity",
    ) -> None:
        if output_mode not in ("velocity", "position"):
            raise ValueError(f"output_mode must be 'velocity' or 'position', got {output_mode!r}")
        self._joint_ids = joint_ids
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._max_output = max_output
        self._output_mode = output_mode

        # Thread-safe target: written from async world, read from RT thread.
        self._lock = threading.Lock()
        self._targets: dict[str, float] = {jid: 0.0 for jid in joint_ids}

        # Per-joint PID state (only accessed from the RT thread).
        self._pid: dict[str, _PIDState] = {jid: _PIDState() for jid in joint_ids}

    @property
    def policy_id(self) -> str:
        return "pid_joint"

    def set_target(self, targets: Mapping[str, float]) -> None:
        """Set target positions for each joint.  Thread-safe."""
        with self._lock:
            for jid, val in targets.items():
                if jid in self._targets:
                    self._targets[jid] = val

    def compute(self, state: ControlState) -> ControlCommand:
        """PID computation for each joint."""
        with self._lock:
            targets = dict(self._targets)

        dt = state.cycle_dt_s
        if dt <= 0:
            dt = 1e-4  # guard against first-cycle zero dt

        commands: dict[str, float] = {}
        for jid in self._joint_ids:
            target = targets.get(jid, 0.0)
            current = state.joint_positions.get(jid, 0.0)
            error = target - current

            pid = self._pid[jid]
            pid.integral += error * dt
            derivative = (error - pid.prev_error) / dt if dt > 0 else 0.0
            pid.prev_error = error

            output = self._kp * error + self._ki * pid.integral + self._kd * derivative

            # Clamp output.
            output = max(-self._max_output, min(self._max_output, output))

            if self._output_mode == "velocity":
                commands[jid] = output
            else:
                # Position mode: command = current + clamped delta.
                commands[jid] = current + output

        return ControlCommand(joint_commands=commands)

    def reset(self) -> None:
        """Reset integral accumulators."""
        for pid in self._pid.values():
            pid.integral = 0.0
            pid.prev_error = 0.0


class ImpedanceController:
    """Impedance controller for compliant manipulation.

    Implements :class:`ControlPolicy` Protocol.  Models the robot end-effector
    as a virtual mass-spring-damper system::

        F = M * a_desired + D * (v - v_ref) + K * (x - x_ref)

    Useful for tasks requiring force control (insertion, polishing) and for
    safe human-robot interaction.
    """

    def __init__(
        self,
        joint_ids: tuple[str, ...],
        *,
        stiffness: float = 100.0,
        damping: float = 10.0,
        inertia: float = 1.0,
    ) -> None:
        if inertia <= 0:
            raise ValueError(f"inertia must be positive, got {inertia}")
        self._joint_ids = joint_ids
        self._stiffness = stiffness
        self._damping = damping
        self._inertia = inertia

        # Thread-safe reference: written from async world, read from RT thread.
        self._lock = threading.Lock()
        self._ref_pos: dict[str, float] = {jid: 0.0 for jid in joint_ids}
        self._ref_vel: dict[str, float] = {jid: 0.0 for jid in joint_ids}

    @property
    def policy_id(self) -> str:
        return "impedance"

    def set_reference(
        self,
        position: Mapping[str, float],
        velocity: Mapping[str, float] | None = None,
    ) -> None:
        """Set reference trajectory point.  Thread-safe."""
        with self._lock:
            for jid, val in position.items():
                if jid in self._ref_pos:
                    self._ref_pos[jid] = val
            if velocity is not None:
                for jid, val in velocity.items():
                    if jid in self._ref_vel:
                        self._ref_vel[jid] = val

    def compute(self, state: ControlState) -> ControlCommand:
        """Impedance control: compute torque from spring-damper model."""
        with self._lock:
            ref_pos = dict(self._ref_pos)
            ref_vel = dict(self._ref_vel)

        commands: dict[str, float] = {}
        for jid in self._joint_ids:
            x = state.joint_positions.get(jid, 0.0)
            v = state.joint_velocities.get(jid, 0.0)
            x_ref = ref_pos.get(jid, 0.0)
            v_ref = ref_vel.get(jid, 0.0)

            # F = K * (x_ref - x) + D * (v_ref - v)
            # a_desired = F / M  →  this becomes the velocity-change command
            force = self._stiffness * (x_ref - x) + self._damping * (v_ref - v)
            accel = force / self._inertia
            # Integrate acceleration over one dt to get velocity command.
            dt = state.cycle_dt_s if state.cycle_dt_s > 0 else 1e-4
            commands[jid] = v + accel * dt

        return ControlCommand(joint_commands=commands)

    def reset(self) -> None:
        """Reset reference to zeros."""
        with self._lock:
            for jid in self._ref_pos:
                self._ref_pos[jid] = 0.0
                self._ref_vel[jid] = 0.0


# ---------------------------------------------------------------------------
# Trajectory tracker
# ---------------------------------------------------------------------------


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between *a* and *b* at parameter *t* ∈ [0, 1]."""
    return a + (b - a) * t


class TrajectoryTracker:
    """Follows a pre-planned joint trajectory using PID or impedance control.

    Wraps a :class:`ControlPolicy` (typically :class:`PIDJointController`) and
    feeds it target positions from a time-indexed trajectory at each cycle.

    Implements :class:`ControlPolicy` Protocol so it can be directly passed to
    :class:`HighFrequencyControlBus`.
    """

    def __init__(
        self,
        inner_policy: ControlPolicy,
        trajectory: tuple[tuple[float, Mapping[str, float]], ...],
        *,
        loop_mode: str = "once",
    ) -> None:
        if not trajectory:
            raise ValueError("trajectory must contain at least one waypoint")
        if loop_mode not in ("once", "loop", "hold_final"):
            raise ValueError(
                f"loop_mode must be 'once', 'loop', or 'hold_final', got {loop_mode!r}"
            )
        self._inner = inner_policy
        self._trajectory = trajectory
        self._loop_mode = loop_mode
        self._duration = trajectory[-1][0] - trajectory[0][0] if len(trajectory) > 1 else 0.0

        # Runtime state — only touched from the RT thread.
        self._start_time: float | None = None
        self._complete = False

    @property
    def policy_id(self) -> str:
        return f"trajectory_tracker:{self._inner.policy_id}"

    def compute(self, state: ControlState) -> ControlCommand:
        """Interpolate trajectory at current time, set as target for inner policy."""
        if self._start_time is None:
            self._start_time = state.timestamp

        elapsed = state.timestamp - self._start_time
        traj = self._trajectory

        if self._duration <= 0:
            # Single waypoint — just hold it.
            targets = dict(traj[0][1])
            self._set_inner_target(targets)
            self._complete = True
            return self._inner.compute(state)

        # Effective elapsed after loop-mode handling.
        effective = elapsed
        if self._loop_mode == "loop" and self._duration > 0:
            effective = elapsed % self._duration + traj[0][0]
        elif self._loop_mode == "once":
            if elapsed >= self._duration:
                self._complete = True
                # Return halt once trajectory is complete.
                return ControlCommand(joint_commands={}, halt=True)
            effective = elapsed + traj[0][0]
        elif self._loop_mode == "hold_final":
            if elapsed >= self._duration:
                self._complete = True
                effective = traj[-1][0]
            else:
                effective = elapsed + traj[0][0]

        # Find the bracketing waypoints and interpolate.
        targets = self._interpolate(effective, traj)
        self._set_inner_target(targets)
        return self._inner.compute(state)

    def _set_inner_target(self, targets: dict[str, float]) -> None:
        """Feed interpolated targets to the inner policy."""
        if hasattr(self._inner, "set_target"):
            self._inner.set_target(targets)  # type: ignore[attr-defined]
        elif hasattr(self._inner, "set_reference"):
            self._inner.set_reference(targets)  # type: ignore[attr-defined]

    @staticmethod
    def _interpolate(
        t: float,
        traj: tuple[tuple[float, Mapping[str, float]], ...],
    ) -> dict[str, float]:
        """Linear interpolation between the two nearest waypoints."""
        # Before first waypoint.
        if t <= traj[0][0]:
            return dict(traj[0][1])
        # After last waypoint.
        if t >= traj[-1][0]:
            return dict(traj[-1][1])

        # Binary search for the bracket.
        lo, hi = 0, len(traj) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if traj[mid][0] <= t:
                lo = mid
            else:
                hi = mid

        t0, pos0 = traj[lo]
        t1, pos1 = traj[hi]
        dt = t1 - t0
        alpha = (t - t0) / dt if dt > 0 else 0.0

        result: dict[str, float] = {}
        all_keys = set(pos0.keys()) | set(pos1.keys())
        for k in all_keys:
            v0 = pos0.get(k, 0.0)
            v1 = pos1.get(k, 0.0)
            result[k] = _lerp(v0, v1, alpha)
        return result

    @property
    def progress(self) -> float:
        """0.0 to 1.0, fraction of trajectory completed."""
        if self._start_time is None or self._duration <= 0:
            return 0.0
        elapsed = time.monotonic() - self._start_time
        return min(1.0, max(0.0, elapsed / self._duration))

    @property
    def is_complete(self) -> bool:
        """Whether the trajectory has reached its end."""
        return self._complete

    def reset(self) -> None:
        """Reset tracker and inner policy state."""
        self._start_time = None
        self._complete = False
        self._inner.reset()


# ---------------------------------------------------------------------------
# Policy chain (System-1.5 → System-1)
# ---------------------------------------------------------------------------


class PolicyChain:
    """Chains a high-level policy with a low-level servo refiner.

    The outer policy (e.g. VLA at 10 Hz) produces coarse commands.  The inner
    policy (e.g. PID at 500 Hz) refines them into smooth servo-level control,
    interpolating between the outer policy's sparse updates.

    Implements :class:`ControlPolicy` Protocol.
    """

    def __init__(
        self,
        outer: ControlPolicy,
        inner: ControlPolicy,
        *,
        outer_rate_hz: float = 10.0,
    ) -> None:
        if outer_rate_hz <= 0:
            raise ValueError(f"outer_rate_hz must be positive, got {outer_rate_hz}")
        self._outer = outer
        self._inner = inner
        self._outer_rate_hz = outer_rate_hz

        # Cycle bookkeeping — only accessed from the RT thread.
        self._cycles_since_outer: int = 0
        self._outer_period_cycles: int = 1  # computed on first call
        self._first_call = True
        self._last_outer_command: ControlCommand | None = None

    @property
    def policy_id(self) -> str:
        return f"chain:{self._outer.policy_id}+{self._inner.policy_id}"

    def compute(self, state: ControlState) -> ControlCommand:
        """Every N cycles call outer for new target; every cycle call inner for servo."""
        if self._first_call:
            bus_hz = 1.0 / state.cycle_dt_s if state.cycle_dt_s > 0 else 100.0
            self._outer_period_cycles = max(1, round(bus_hz / self._outer_rate_hz))
            self._first_call = False
            self._cycles_since_outer = self._outer_period_cycles  # force first outer call

        self._cycles_since_outer += 1
        if self._cycles_since_outer >= self._outer_period_cycles:
            self._cycles_since_outer = 0
            outer_cmd = self._outer.compute(state)
            self._last_outer_command = outer_cmd
            if outer_cmd.halt:
                return outer_cmd
            # Feed outer targets to inner policy.
            if hasattr(self._inner, "set_target"):
                self._inner.set_target(outer_cmd.joint_commands)  # type: ignore[attr-defined]
            elif hasattr(self._inner, "set_reference"):
                self._inner.set_reference(outer_cmd.joint_commands)  # type: ignore[attr-defined]

        return self._inner.compute(state)

    def reset(self) -> None:
        """Reset both outer and inner policies."""
        self._outer.reset()
        self._inner.reset()
        self._cycles_since_outer = 0
        self._first_call = True
        self._last_outer_command = None


# ---------------------------------------------------------------------------
# Telemetry event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlTelemetryEvent:
    """Published to EventBus at configurable intervals for LeapBoard display."""

    session_id: str
    device_id: str
    policy_id: str
    cycles: int
    mean_cycle_ms: float
    max_jitter_ms: float
    overruns: int
    is_running: bool
    progress: float | None = None  # only for trajectory sessions


# ---------------------------------------------------------------------------
# RealtimeControlLoop  — high-level orchestrator
# ---------------------------------------------------------------------------


@dataclass
class _SessionInfo:
    """Internal bookkeeping for an active control session."""

    session_id: str
    device_id: str
    policy: ControlPolicy
    started_at: float = field(default_factory=time.monotonic)
    stopped_at: float | None = None
    config: ControlBusConfig | None = None


class RealtimeControlLoop:
    """High-level orchestrator for real-time control sessions.

    Wraps :class:`HighFrequencyControlBus` with:

    - Named control sessions (start → run → stop lifecycle)
    - Built-in policy selection (PID, impedance, trajectory, chain)
    - Telemetry publishing to EventBus
    - Graceful transition between policies (ramp down → switch → ramp up)
    """

    def __init__(
        self,
        bus: HighFrequencyControlBus,
        *,
        event_bus: Any = None,
        telemetry_interval_s: float = 1.0,
    ) -> None:
        self._bus = bus
        self._event_bus = event_bus
        self._telemetry_interval_s = telemetry_interval_s

        self._session: _SessionInfo | None = None
        self._telemetry_timer: threading.Timer | None = None
        self._lock = threading.Lock()

    # -- PID session -------------------------------------------------------

    async def start_pid(
        self,
        device_id: str,
        joint_ids: tuple[str, ...],
        targets: Mapping[str, float],
        *,
        config: ControlBusConfig | None = None,
        kp: float = 10.0,
        ki: float = 0.1,
        kd: float = 1.0,
        max_output: float = 6.28,
        output_mode: str = "velocity",
    ) -> str:
        """Start PID position tracking.  Returns session_id."""
        policy = PIDJointController(
            joint_ids, kp=kp, ki=ki, kd=kd,
            max_output=max_output, output_mode=output_mode,
        )
        policy.set_target(targets)
        return await self._start_session(device_id, policy, config)

    # -- Impedance session -------------------------------------------------

    async def start_impedance(
        self,
        device_id: str,
        joint_ids: tuple[str, ...],
        position: Mapping[str, float],
        *,
        config: ControlBusConfig | None = None,
        velocity: Mapping[str, float] | None = None,
        stiffness: float = 100.0,
        damping: float = 10.0,
        inertia: float = 1.0,
    ) -> str:
        """Start impedance control.  Returns session_id."""
        policy = ImpedanceController(
            joint_ids, stiffness=stiffness, damping=damping, inertia=inertia,
        )
        policy.set_reference(position, velocity)
        return await self._start_session(device_id, policy, config)

    # -- Trajectory session ------------------------------------------------

    async def start_trajectory(
        self,
        device_id: str,
        trajectory: tuple[tuple[float, Mapping[str, float]], ...],
        *,
        controller: str = "pid",
        config: ControlBusConfig | None = None,
        loop_mode: str = "once",
        **controller_kwargs: Any,
    ) -> str:
        """Follow a joint trajectory.  Returns session_id."""
        # Infer joint_ids from the first waypoint.
        joint_ids = tuple(trajectory[0][1].keys())

        inner: ControlPolicy
        if controller == "pid":
            inner = PIDJointController(joint_ids, **controller_kwargs)
        elif controller == "impedance":
            inner = ImpedanceController(joint_ids, **controller_kwargs)
        else:
            raise ValueError(f"Unknown controller type: {controller!r}")

        policy = TrajectoryTracker(inner, trajectory, loop_mode=loop_mode)
        return await self._start_session(device_id, policy, config)

    # -- Policy-chain session ----------------------------------------------

    async def start_policy_chain(
        self,
        device_id: str,
        outer_policy: ControlPolicy,
        *,
        inner_controller: str = "pid",
        outer_rate_hz: float = 10.0,
        config: ControlBusConfig | None = None,
        joint_ids: tuple[str, ...] | None = None,
        **inner_kwargs: Any,
    ) -> str:
        """Start a System-1.5 → System-1 policy chain.  Returns session_id."""
        if joint_ids is None:
            raise ValueError("joint_ids required for policy chain inner controller")

        inner: ControlPolicy
        if inner_controller == "pid":
            inner = PIDJointController(joint_ids, **inner_kwargs)
        elif inner_controller == "impedance":
            inner = ImpedanceController(joint_ids, **inner_kwargs)
        else:
            raise ValueError(f"Unknown inner_controller type: {inner_controller!r}")

        policy = PolicyChain(outer_policy, inner, outer_rate_hz=outer_rate_hz)
        return await self._start_session(device_id, policy, config)

    # -- Lifecycle ---------------------------------------------------------

    async def stop(self) -> dict[str, Any]:
        """Stop the current control session gracefully."""
        with self._lock:
            session = self._session
        if session is None:
            return {"status": "no_session"}

        self._stop_telemetry()
        await self._bus.async_stop()

        with self._lock:
            session.stopped_at = time.monotonic()
            stats = self._bus.stats.to_dict()
            result = {
                "status": "stopped",
                "session_id": session.session_id,
                "device_id": session.device_id,
                "policy_id": session.policy.policy_id,
                "duration_s": round(
                    (session.stopped_at - session.started_at), 3
                ),
                "stats": stats,
            }
            self._session = None
        return result

    async def status(self) -> dict[str, Any]:
        """Return current session status and statistics."""
        with self._lock:
            session = self._session
        if session is None:
            return {"status": "idle"}

        stats = self._bus.stats.to_dict()
        result: dict[str, Any] = {
            "status": "running" if self._bus.is_running else "stopped",
            "session_id": session.session_id,
            "device_id": session.device_id,
            "policy_id": session.policy.policy_id,
            "uptime_s": round(time.monotonic() - session.started_at, 3),
            "stats": stats,
        }
        # Include trajectory progress if applicable.
        if isinstance(session.policy, TrajectoryTracker):
            result["progress"] = round(session.policy.progress, 4)
            result["is_complete"] = session.policy.is_complete
        return result

    async def update_target(self, targets: Mapping[str, float]) -> None:
        """Update PID/impedance targets without stopping the loop."""
        with self._lock:
            session = self._session
        if session is None:
            raise RuntimeError("No active control session")

        policy = session.policy
        # Unwrap TrajectoryTracker — target updates go to the inner policy.
        if isinstance(policy, TrajectoryTracker):
            policy = policy._inner

        if isinstance(policy, PIDJointController):
            policy.set_target(targets)
        elif isinstance(policy, ImpedanceController):
            policy.set_reference(targets)
        elif isinstance(policy, PolicyChain):
            # Update the inner policy of the chain.
            inner = policy._inner
            if isinstance(inner, PIDJointController):
                inner.set_target(targets)
            elif isinstance(inner, ImpedanceController):
                inner.set_reference(targets)
        else:
            raise TypeError(
                f"Cannot update targets on policy {policy.policy_id!r}"
            )

    # -- Internal helpers --------------------------------------------------

    async def _start_session(
        self,
        device_id: str,
        policy: ControlPolicy,
        config: ControlBusConfig | None,
    ) -> str:
        """Common session start logic."""
        # Stop any running session first.
        if self._bus.is_running:
            await self.stop()

        session_id = uuid.uuid4().hex[:12]
        session = _SessionInfo(
            session_id=session_id,
            device_id=device_id,
            policy=policy,
            config=config,
        )
        with self._lock:
            self._session = session

        # Apply custom config to bus if provided.
        if config is not None:
            self._bus._config = config

        await self._bus.async_start(device_id, policy)
        self._start_telemetry(session)

        logger.info(
            "Realtime control session %s started: device=%s policy=%s",
            session_id,
            device_id,
            policy.policy_id,
        )
        return session_id

    def _start_telemetry(self, session: _SessionInfo) -> None:
        """Start periodic telemetry publishing."""
        if self._event_bus is None or self._telemetry_interval_s <= 0:
            return

        def _publish() -> None:
            if not self._bus.is_running:
                return
            stats = self._bus.stats
            progress = None
            if isinstance(session.policy, TrajectoryTracker):
                progress = session.policy.progress

            event = ControlTelemetryEvent(
                session_id=session.session_id,
                device_id=session.device_id,
                policy_id=session.policy.policy_id,
                cycles=stats.cycles,
                mean_cycle_ms=round(stats.mean_cycle_ms, 4),
                max_jitter_ms=round(stats.max_jitter_ms, 4),
                overruns=stats.overruns,
                is_running=self._bus.is_running,
                progress=progress,
            )
            try:
                self._event_bus.emit("control.telemetry", event)
            except Exception:  # noqa: BLE001 — telemetry must not break control
                pass

            # Reschedule.
            if self._bus.is_running:
                t = threading.Timer(self._telemetry_interval_s, _publish)
                t.daemon = True
                with self._lock:
                    self._telemetry_timer = t
                t.start()

        timer = threading.Timer(self._telemetry_interval_s, _publish)
        timer.daemon = True
        with self._lock:
            self._telemetry_timer = timer
        timer.start()

    def _stop_telemetry(self) -> None:
        """Cancel the telemetry timer."""
        with self._lock:
            timer = self._telemetry_timer
            self._telemetry_timer = None
        if timer is not None:
            timer.cancel()


__all__ = [
    "ControlTelemetryEvent",
    "ImpedanceController",
    "PIDJointController",
    "PolicyChain",
    "RealtimeControlLoop",
    "TrajectoryTracker",
]
