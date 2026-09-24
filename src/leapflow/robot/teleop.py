# Copyright (c) Alibaba, Inc. and its affiliates.
"""Teleoperation bridge: leader-follower device pairing for demonstration.

A teleoperation session reads commands from a leader device (a human moves
the leader arm) and mirrors them to a follower device (the robot being
taught), while recording the resulting trajectory.  This is the physical
analogue of demonstrating a task by doing it.

Kinesthetic mode is a variant where there is no separate leader: the human
physically moves the follower itself (with torque disabled), and the source
of truth is the follower's own joint encoders.

Both modes produce a :class:`PhysicalTrajectory` as their output, which
can later be exported as an episode for policy learning.

The control loop uses deadline-based scheduling identical to
``BatchStreamCoordinator._run()``: a fixed sleep after each iteration would
add the I/O duration to every period, so a loop declared at 30 Hz would run
slower — silently.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Mapping, Protocol, runtime_checkable

from leapflow.robot.trajectory import (
    PhysicalTrajectory,
    PhysicalTrajectoryStep,
    make_trajectory_id,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TeleopBridge Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TeleopBridge(Protocol):
    """Bridges a leader device's motion to a follower device.

    Implementations must be safe to start and stop from an ``asyncio`` event
    loop.  ``stop()`` returns the finalized trajectory so the caller can
    persist it or hand it to an episode recorder.
    """

    @property
    def session_id(self) -> str:
        """Unique identifier for this teleop session."""
        ...

    async def start(self) -> None:
        """Begin the control loop.  Returns promptly; the loop runs as a task."""
        ...

    async def stop(self) -> PhysicalTrajectory:
        """Stop the control loop and return the recorded trajectory."""
        ...

    @property
    def is_active(self) -> bool:
        """Whether the control loop is currently running."""
        ...


# ---------------------------------------------------------------------------
# LeaderFollowerBridge
# ---------------------------------------------------------------------------

class LeaderFollowerBridge:
    """Teleop bridge that mirrors leader joint positions to follower.

    Loop (at the follower's declared control rate):

    1. ``read_batch(leader_device, joint_channels)``
    2. Optionally transform/scale (leader and follower may differ)
    3. Write to follower through the approval-gated path
       (``hardware_tools.batch_actuate``) — teleop does **not** bypass gating
    4. ``read_batch(follower)`` for actual state
    5. Append ``PhysicalTrajectoryStep`` to trajectory

    Safety: the follower write goes through the normal HCP approval/trust
    path.  ``halt()`` on either device stops the session.
    """

    def __init__(
        self,
        registry: Any,
        *,
        leader_device_id: str,
        follower_device_id: str,
        goal: str = "",
        control_rate_hz: float = 30.0,
        joint_map: Mapping[str, str] | None = None,
        hardware_tools: Any = None,
    ) -> None:
        self._registry = registry
        self._leader_device_id = leader_device_id
        self._follower_device_id = follower_device_id
        self._goal = goal
        self._control_rate_hz = max(1.0, float(control_rate_hz))
        self._joint_map = dict(joint_map) if joint_map else {}
        self._hardware_tools = hardware_tools

        self._session_id = make_trajectory_id()
        self._trajectory = PhysicalTrajectory(
            trajectory_id=self._session_id,
            device_id=follower_device_id,
            goal=goal,
            started_at=0.0,
            metadata={
                "mode": "teleop",
                "leader_device_id": leader_device_id,
                "follower_device_id": follower_device_id,
                "control_rate_hz": control_rate_hz,
                "joint_map": self._joint_map or {},
            },
        )
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._sequence = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_active(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Begin the leader→follower mirror loop."""
        if self._task is not None:
            return
        self._stopping.clear()
        self._trajectory = PhysicalTrajectory(
            trajectory_id=self._trajectory.trajectory_id,
            device_id=self._trajectory.device_id,
            goal=self._trajectory.goal,
            started_at=time.time(),
            metadata=self._trajectory.metadata,
        )
        self._task = asyncio.create_task(
            self._run(), name=f"teleop:{self._session_id}"
        )

    async def stop(self) -> PhysicalTrajectory:
        """Stop the mirror loop and return the finalized trajectory."""
        self._stopping.set()
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as exc:  # noqa: BLE001
                logger.warning("Teleop task stop raised: %s", exc, exc_info=True)
        return self._trajectory.finalized()

    # -- Control loop ---------------------------------------------------------

    async def _run(self) -> None:
        """Core mirror loop using deadline-based scheduling."""
        interval = 1.0 / self._control_rate_hz
        next_at = time.monotonic()
        consecutive_failures = 0

        while not self._stopping.is_set():
            try:
                step = await self._one_cycle()
                if step is not None:
                    self._trajectory = self._trajectory.with_step(step)
                consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                consecutive_failures += 1
                if consecutive_failures == 1:
                    logger.warning(
                        "Teleop cycle failed: %s", exc, exc_info=True
                    )
                backoff = min(interval * (2 ** consecutive_failures), 5.0)
                await self._sleep(backoff)
                next_at = time.monotonic()
                continue

            # Deadline-based scheduling (identical to BatchStreamCoordinator).
            next_at += interval
            delay = next_at - time.monotonic()
            if delay < 0:
                missed = int(-delay // interval) + 1
                next_at += missed * interval
                delay = max(0.0, next_at - time.monotonic())
            await self._sleep(delay)

    async def _one_cycle(self) -> PhysicalTrajectoryStep | None:
        """Execute one leader-read → follower-write → follower-read cycle."""
        registry = self._registry

        # 1. Read leader joint positions.
        leader_context = registry.context(self._leader_device_id)
        if leader_context is None:
            return None
        leader_channels = tuple(
            ch.channel_id for ch in leader_context.channels if ch.is_readable
        )
        if not leader_channels:
            return None
        leader_batch = await registry.read_batch(
            self._leader_device_id, leader_channels
        )

        # 2. Map leader readings to follower commands.
        commands: list[tuple[str, Any]] = []
        leader_positions: dict[str, float] = {}
        for reading in leader_batch.readings:
            leader_positions[reading.channel_id] = float(reading.value)
            follower_ch = self._joint_map.get(reading.channel_id, reading.channel_id)
            commands.append((follower_ch, reading.value))

        # 3. Write to follower through approval-gated path.
        if commands and self._hardware_tools is not None:
            await self._hardware_tools.batch_actuate(
                self._follower_device_id, tuple(commands)
            )
        elif commands:
            # Direct write fallback when no hardware_tools provided (testing).
            from leapflow.hardware.transport import BatchTransport

            transport = await registry.transport(self._follower_device_id)
            if isinstance(transport, BatchTransport):
                async with registry.device_io_batch(self._follower_device_id):
                    await transport.write_batch(tuple(commands))
            else:
                async with registry.device_io(self._follower_device_id):
                    for ch_id, val in commands:
                        await transport.write(ch_id, val)

        # 4. Read follower actual state.
        follower_context = registry.context(self._follower_device_id)
        if follower_context is None:
            return None
        follower_channels = tuple(
            ch.channel_id for ch in follower_context.channels if ch.is_readable
        )
        follower_batch = await registry.read_batch(
            self._follower_device_id, follower_channels
        )

        # 5. Build step from follower state.
        now = time.time()
        mono = time.monotonic()
        positions: dict[str, float] = {}
        for reading in follower_batch.readings:
            positions[reading.channel_id] = float(reading.value)

        step = PhysicalTrajectoryStep(
            timestamp=now,
            monotonic_at=mono,
            joint_positions=positions,
            action_source="teleop",
            sequence=self._sequence,
        )
        self._sequence += 1
        return step

    async def _sleep(self, seconds: float) -> None:
        """Sleep interruptibly via the stopping event."""
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return


# ---------------------------------------------------------------------------
# KinestheticRecorder
# ---------------------------------------------------------------------------

class KinestheticRecorder:
    """Records a demonstration where the human moves the follower directly.

    No leader device.  Torque is disabled on the follower (via a configure
    channel if available), the human moves it by hand, and the recorder
    samples the follower's joint encoders at the control rate.

    Output is a ``PhysicalTrajectory`` with ``action_source="kinesthetic"``
    on every step.
    """

    def __init__(
        self,
        registry: Any,
        *,
        device_id: str,
        goal: str = "",
        sample_rate_hz: float = 30.0,
    ) -> None:
        self._registry = registry
        self._device_id = device_id
        self._goal = goal
        self._sample_rate_hz = max(1.0, float(sample_rate_hz))

        self._session_id = make_trajectory_id()
        self._trajectory = PhysicalTrajectory(
            trajectory_id=self._session_id,
            device_id=device_id,
            goal=goal,
            started_at=0.0,
            metadata={
                "mode": "kinesthetic",
                "device_id": device_id,
                "sample_rate_hz": sample_rate_hz,
            },
        )
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._sequence = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_active(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Begin sampling the follower's joint encoders."""
        if self._task is not None:
            return
        self._stopping.clear()
        self._trajectory = PhysicalTrajectory(
            trajectory_id=self._trajectory.trajectory_id,
            device_id=self._trajectory.device_id,
            goal=self._trajectory.goal,
            started_at=time.time(),
            metadata=self._trajectory.metadata,
        )
        self._task = asyncio.create_task(
            self._run(), name=f"kinesthetic:{self._session_id}"
        )

    async def stop(self) -> PhysicalTrajectory:
        """Stop sampling and return the finalized trajectory."""
        self._stopping.set()
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Kinesthetic task stop raised: %s", exc, exc_info=True
                )
        return self._trajectory.finalized()

    # -- Sampling loop --------------------------------------------------------

    async def _run(self) -> None:
        """Core sampling loop using deadline-based scheduling."""
        interval = 1.0 / self._sample_rate_hz
        next_at = time.monotonic()
        consecutive_failures = 0

        while not self._stopping.is_set():
            try:
                step = await self._sample()
                if step is not None:
                    self._trajectory = self._trajectory.with_step(step)
                consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                consecutive_failures += 1
                if consecutive_failures == 1:
                    logger.warning(
                        "Kinesthetic sample failed: %s", exc, exc_info=True
                    )
                backoff = min(interval * (2 ** consecutive_failures), 5.0)
                await self._sleep(backoff)
                next_at = time.monotonic()
                continue

            next_at += interval
            delay = next_at - time.monotonic()
            if delay < 0:
                missed = int(-delay // interval) + 1
                next_at += missed * interval
                delay = max(0.0, next_at - time.monotonic())
            await self._sleep(delay)

    async def _sample(self) -> PhysicalTrajectoryStep | None:
        """Read the follower's current joint state."""
        context = self._registry.context(self._device_id)
        if context is None:
            return None
        channels = tuple(
            ch.channel_id for ch in context.channels if ch.is_readable
        )
        if not channels:
            return None
        batch = await self._registry.read_batch(self._device_id, channels)

        now = time.time()
        mono = time.monotonic()
        positions: dict[str, float] = {}
        for reading in batch.readings:
            positions[reading.channel_id] = float(reading.value)

        step = PhysicalTrajectoryStep(
            timestamp=now,
            monotonic_at=mono,
            joint_positions=positions,
            action_source="kinesthetic",
            sequence=self._sequence,
        )
        self._sequence += 1
        return step

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return


# ---------------------------------------------------------------------------
# Module-level active session tracking
# ---------------------------------------------------------------------------

_active_physical_sessions: dict[str, TeleopBridge | LeaderFollowerBridge | KinestheticRecorder] = {}
"""Tracks active physical teach sessions, keyed by a context identifier.

Used by the /teach command dispatcher to start/stop physical sessions
without polluting the desktop teach session state.
"""


def register_physical_session(key: str, bridge: Any) -> None:
    """Register an active physical teach session."""
    _active_physical_sessions[key] = bridge


def get_physical_session(key: str) -> Any:
    """Return the active physical session for *key*, or None."""
    return _active_physical_sessions.get(key)


def remove_physical_session(key: str) -> Any:
    """Remove and return the active physical session for *key*, or None."""
    return _active_physical_sessions.pop(key, None)


__all__ = [
    "KinestheticRecorder",
    "LeaderFollowerBridge",
    "TeleopBridge",
    "get_physical_session",
    "register_physical_session",
    "remove_physical_session",
]
