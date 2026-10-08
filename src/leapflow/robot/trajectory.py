# Copyright (c) Alibaba, Inc. and its affiliates.
"""Physical demonstration trajectory types.

Parallel to the desktop ``TrajectoryStep`` (which captures GUI events), these
types capture physical robot state: joint positions, velocities, gripper
state, camera frames, and sensor readings.  Kept separate from the desktop
trajectory model so physical demonstration learning does not perturb the
existing GUI teach/learn regression.

Timebase convention follows ``Reading`` / ``HardwareEvent``:

- ``timestamp``: wall-clock (``time.time()`` epoch seconds).  The only clock
  that may be persisted, rendered, or correlated outside this process.
- ``monotonic_at``: ``time.monotonic()``.  The only clock that may be used
  for intervals, since wall-clock jumps fabricate rates no device produced.

``camera_frames`` stores file references, not inline bytes.  Frames are large
binary blobs that would inflate every serialization path; the ``FrameReading``
not-a-``Reading`` principle from HCP applies identically here.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# PhysicalTrajectoryStep
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysicalTrajectoryStep:
    """One timestep of a physical demonstration.

    Captures the complete observable state of a robot at a single instant:
    joint positions and velocities, gripper aperture, camera frame
    references, and arbitrary sensor readings.

    ``action_source`` classifies how this step was produced:

    - ``"teleop"``: a leader device was mirrored to the follower.
    - ``"kinesthetic"``: the human physically moved the robot.
    - ``"policy"``: a learned policy produced the action.
    - ``"manual"``: a single manual command (e.g. jog).
    """

    timestamp: float
    """Wall-clock (``time.time()`` epoch seconds)."""

    monotonic_at: float
    """``time.monotonic()`` — for interval computation only."""

    joint_positions: Mapping[str, float]
    """Channel-id → position (radians or meters, per channel declaration)."""

    joint_velocities: Mapping[str, float] = field(default_factory=dict)
    """Channel-id → velocity.  Empty when not sampled."""

    gripper_state: float | None = None
    """Gripper aperture, 0.0 = closed … 1.0 = open.  ``None`` = no gripper."""

    camera_frames: Mapping[str, str] = field(default_factory=dict)
    """Channel-id → frame file reference (path or URI, never inline bytes)."""

    sensor_readings: Mapping[str, Any] = field(default_factory=dict)
    """Arbitrary sensor values keyed by channel-id."""

    action_source: str = "teleop"
    """How this step was produced: ``teleop`` | ``kinesthetic`` | ``policy`` | ``manual``."""

    sequence: int = 0
    """Zero-based index within the trajectory, for gap detection."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON / NDJSON persistence."""
        payload: dict[str, Any] = {
            "timestamp": self.timestamp,
            "monotonic_at": self.monotonic_at,
            "joint_positions": dict(self.joint_positions),
            "sequence": self.sequence,
            "action_source": self.action_source,
        }
        if self.joint_velocities:
            payload["joint_velocities"] = dict(self.joint_velocities)
        if self.gripper_state is not None:
            payload["gripper_state"] = self.gripper_state
        if self.camera_frames:
            payload["camera_frames"] = dict(self.camera_frames)
        if self.sensor_readings:
            payload["sensor_readings"] = dict(self.sensor_readings)
        return payload


# ---------------------------------------------------------------------------
# PhysicalTrajectory
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysicalTrajectory:
    """A complete physical demonstration episode.

    Frozen so that a recorded trajectory is an immutable evidence artifact.
    Mutation uses the builder helpers ``with_step`` and ``finalized`` which
    return new instances — allocation is negligible compared to the cost of
    the physical sampling loop that produced the data.
    """

    trajectory_id: str
    """Unique identifier for this trajectory."""

    device_id: str
    """The follower device that was commanded (or recorded in kinesthetic mode)."""

    goal: str
    """Human-supplied description of what the demonstration achieves."""

    steps: tuple[PhysicalTrajectoryStep, ...] = ()
    """Ordered timesteps, oldest first."""

    started_at: float = 0.0
    """Wall-clock epoch when recording began."""

    ended_at: float = 0.0
    """Wall-clock epoch when recording ended.  0.0 while still recording."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Free-form metadata (leader device id, control rate, joint map, etc.)."""

    # -- Derived properties ---------------------------------------------------

    @property
    def duration_s(self) -> float:
        """Duration in seconds.  Zero while still recording."""
        if self.ended_at > self.started_at:
            return self.ended_at - self.started_at
        return 0.0

    @property
    def sample_count(self) -> int:
        """Number of recorded timesteps."""
        return len(self.steps)

    # -- Serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full trajectory for JSON persistence or export."""
        return {
            "trajectory_id": self.trajectory_id,
            "device_id": self.device_id,
            "goal": self.goal,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "sample_count": self.sample_count,
            "metadata": dict(self.metadata),
            "steps": [step.to_dict() for step in self.steps],
        }

    # -- Builder pattern (frozen, so returns new instances) --------------------

    def with_step(self, step: PhysicalTrajectoryStep) -> "PhysicalTrajectory":
        """Return a new trajectory with *step* appended."""
        return PhysicalTrajectory(
            trajectory_id=self.trajectory_id,
            device_id=self.device_id,
            goal=self.goal,
            steps=(*self.steps, step),
            started_at=self.started_at,
            ended_at=self.ended_at,
            metadata=self.metadata,
        )

    def finalized(self, ended_at: float | None = None) -> "PhysicalTrajectory":
        """Return a new trajectory with ``ended_at`` set."""
        return PhysicalTrajectory(
            trajectory_id=self.trajectory_id,
            device_id=self.device_id,
            goal=self.goal,
            steps=self.steps,
            started_at=self.started_at,
            ended_at=ended_at if ended_at is not None else time.time(),
            metadata=self.metadata,
        )


def make_trajectory_id() -> str:
    """Generate a short unique trajectory identifier."""
    return uuid.uuid4().hex[:16]


__all__ = [
    "PhysicalTrajectory",
    "PhysicalTrajectoryStep",
    "make_trajectory_id",
]
