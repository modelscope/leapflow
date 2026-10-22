# Copyright (c) Alibaba, Inc. and its affiliates.
"""Piper motion spaces, pyAgxArm feedback codec, and serialized command state.

Split out of ``piper.py`` so the transport keeps lifecycle, channel routing and
side-effect verdicts, while this module owns the vendor-facing encoding:

- **Two motion spaces.** Joint targets ride ``move_j``; end-effector pose
  targets ride ``move_l`` (straight-line, the default) or ``move_p``
  (point-to-point, the firmware solves IK). pyAgxArm selects the matching move
  mode on every move, so no move-mode state is tracked here. A batch that mixes
  both spaces has no single honest vendor command and is refused before any
  send (``mixed_motion_space``).
- **Hold semantics.** Every motion command is a full six-value vector; values
  the caller did not name are filled from the last command in the same space
  or, failing that, from current feedback. A command in one space invalidates
  the other space's cache -- after a pose move the last joint command no
  longer describes where the arm is, and holding it would jump the arm back.
  A send that raised leaves the arm in an unknown place, so both caches are
  dropped and the next command re-seeds from feedback.
- **Control-mode entry.** The firmware executes host commands only in CAN
  control mode. ``set_speed_percent`` sends the mode frame that selects CAN
  while keeping the move mode; ``enter_can`` repeats it until ``ctrl_mode``
  feedback confirms, so a dropped frame becomes a retry and a refusal (teach
  switch, latched stop) a typed failure instead of a silent no-op.
- **Units.** Declared channels use deg / mm; pyAgxArm uses rad / m. Pose
  orientation is fixed-axis X-Y-Z (roll, pitch, yaw); the SDK refuses roll/yaw
  outside [-pi, pi] and pitch outside [-pi/2, pi/2], so held feedback is
  wrapped/clamped onto those ranges before it is re-sent.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Iterable, Sequence

from leapflow.hardware.transport import TransportError

CTRL_MODE_STANDBY = 0x00
CTRL_MODE_CAN = 0x01
CTRL_MODE_TEACH = 0x02
CTRL_MODE_NAMES: dict[int, str] = {
    CTRL_MODE_STANDBY: "standby",
    CTRL_MODE_CAN: "can",
    CTRL_MODE_TEACH: "teach",
}
POSE_PATH_LINEAR = "linear"
POSE_PATH_POINT = "point"
POSE_PATHS: tuple[str, ...] = (POSE_PATH_LINEAR, POSE_PATH_POINT)

_MODE_POLL_S = 0.02
_MODE_RETRY_S = 0.15  # longer than pyAgxArm's 0.1 s identical-mode-frame throttle

SPACE_JOINT = "joint"
SPACE_POSE = "pose"

JOINT_CHANNELS: dict[str, int] = {f"joint.j{i}.position": i - 1 for i in range(1, 7)}
POSE_CHANNELS: dict[str, int] = {
    "ee.x": 0,
    "ee.y": 1,
    "ee.z": 2,
    "ee.rx": 3,
    "ee.ry": 4,
    "ee.rz": 5,
}
SPACE_CHANNELS: dict[str, dict[str, int]] = {
    SPACE_JOINT: JOINT_CHANNELS,
    SPACE_POSE: POSE_CHANNELS,
}

_MM_PER_M = 1000.0


# ── Value coercion ───────────────────────────────────────────────────────────


def attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read one attribute from an SDK feedback struct, tolerating absent fields."""
    value = getattr(obj, name, default)
    return default if value is None else value


def as_int(value: Any) -> int | None:
    """Coerce a raw feedback field (often an IntEnum) to int; booleans are not codes."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def numeric(value: Any) -> float | None:
    """Coerce a value to a finite float, rejecting bools/NaN/inf."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def enable_flag(value: Any) -> int | None:
    """Normalize an enable/disable command to 1/0, or None when not a flag."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)) and float(value).is_integer() and int(value) in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip() in ("0", "1"):
        return int(value.strip())
    return None


# ── Unit conversion ──────────────────────────────────────────────────────────


def _six(values: Any) -> list[float] | None:
    if values is None:
        return None
    try:
        numbers = [numeric(v) for v in values]
    except TypeError:
        return None
    if len(numbers) != 6 or any(n is None for n in numbers):
        return None
    return [float(n) for n in numbers]  # type: ignore[arg-type]


def joints_from_sdk(radians: Any) -> list[float] | None:
    """SDK joint angles (rad) -> declared degrees, or None when unusable."""
    values = _six(radians)
    return [math.degrees(v) for v in values] if values is not None else None


def joints_to_sdk(degrees: Sequence[float]) -> list[float]:
    return [math.radians(v) for v in degrees]


def pose_from_sdk(pose: Any) -> list[float] | None:
    """SDK flange pose (m, rad) -> declared (mm, deg), or None when unusable."""
    values = _six(pose)
    if values is None:
        return None
    return [v * _MM_PER_M for v in values[:3]] + [math.degrees(v) for v in values[3:]]


def pose_to_sdk(pose: Sequence[float]) -> list[float]:
    """Declared (mm, deg) -> SDK (m, rad) inside the ranges ``move_l``/``move_p`` accept."""
    roll, pitch, yaw = (math.radians((v + 180.0) % 360.0 - 180.0) for v in pose[3:])
    return [
        *(v / _MM_PER_M for v in pose[:3]),
        min(max(roll, -math.pi), math.pi),
        min(max(pitch, -math.pi / 2), math.pi / 2),
        min(max(yaw, -math.pi), math.pi),
    ]


# ── Feedback getters (cached by the SDK read thread; never touch the bus) ────


def sdk_message(getter: Callable[..., Any], *args: Any) -> Any:
    """The ``.msg`` body of a pyAgxArm feedback getter, or None when unavailable."""
    try:
        message = getter(*args)
    except Exception:  # noqa: BLE001 - a dead receive path must read as unavailable
        return None
    return attr(message, "msg") if message is not None else None


def read_joints(robot: Any) -> list[float] | None:
    """Six joint angles in degrees, or None when feedback is unavailable."""
    return joints_from_sdk(sdk_message(robot.get_joint_angles))


def read_pose(robot: Any) -> list[float] | None:
    """Flange pose (mm, deg) in the base frame, or None when unavailable."""
    return pose_from_sdk(sdk_message(robot.get_flange_pose))


def status_feedback(robot: Any) -> Any:
    """The arm-status message body, or None when unavailable."""
    return sdk_message(robot.get_arm_status)


def ctrl_mode(robot: Any) -> int | None:
    """The control mode the firmware reports (0x00 standby, 0x01 CAN, ...), if any."""
    return as_int(attr(status_feedback(robot), "ctrl_mode"))


def ctrl_mode_name(value: Any) -> str | None:
    """The declared name of a raw control mode, or None when there is no value."""
    mode = as_int(value)
    if mode is None:
        return None
    return CTRL_MODE_NAMES.get(mode, f"other(0x{mode:02X})")


def enable_status(robot: Any) -> list[bool] | None:
    """Per-motor driver enable flags, or None when any driver state is unavailable.

    Read from the driver states directly: the SDK's own list helper reports an
    absent driver state as ``False``, which would turn "unknown" into "disabled".
    """
    flags: list[bool] = []
    for joint in range(1, 7):
        state = sdk_message(robot.get_driver_states, joint)
        flag = attr(attr(state, "foc_status"), "driver_enable_status")
        if not isinstance(flag, bool):
            return None
        flags.append(flag)
    return flags


def firmware_version(robot: Any, timeout_s: float) -> str:
    """The main-controller software version (e.g. ``S-V1.9-0``); ``""`` when unanswered."""
    try:
        info = robot.get_firmware(timeout=timeout_s, min_interval=0.0)
    except Exception:  # noqa: BLE001 - probe metadata must never break on SDK quirks
        return ""
    version = info.get("software_version") if isinstance(info, dict) else None
    return version.strip() if isinstance(version, str) else ""


# ── Motion spaces ────────────────────────────────────────────────────────────


def motion_space(channel_id: str) -> str:
    """Return the motion space a channel commands, or ``""`` for non-motion."""
    for space, channels in SPACE_CHANNELS.items():
        if channel_id in channels:
            return space
    return ""


def motion_spaces(channel_ids: Iterable[str]) -> frozenset[str]:
    """Every motion space a batch touches; more than one is a mixed batch."""
    return frozenset(space for space in map(motion_space, channel_ids) if space)


class PiperMotion:
    """Serialized motion-command state for one arm.

    Every method that touches the arm is blocking and runs on a worker thread.
    The lock spans mode entry, vector fill, send, and cache advance, so
    concurrent writers can neither interleave vectors nor race a mode switch.
    """

    def __init__(self, *, speed_pct: int, pose_path: str, mode_timeout_s: float = 0.6) -> None:
        if pose_path not in POSE_PATHS:
            raise ValueError(f"pose_path must be one of {list(POSE_PATHS)}, got {pose_path!r}")
        self._lock = threading.Lock()
        self._speed_pct = min(100, max(1, int(speed_pct)))
        self._speed_sent = False
        self._pose_path = pose_path
        self._mode_timeout_s = max(0.1, float(mode_timeout_s))
        self._commanded: dict[str, list[float | None]] = {
            SPACE_JOINT: [None] * 6,
            SPACE_POSE: [None] * 6,
        }

    @property
    def speed_pct(self) -> int:
        """The speed rate (1-100 %) the next mode frame carries."""
        return self._speed_pct

    def set_speed(self, pct: int) -> int:
        """Stage a speed rate; the next command sends it before moving."""
        with self._lock:
            self._speed_pct = min(100, max(1, int(pct)))
            self._speed_sent = False
            return self._speed_pct

    @property
    def pose_path(self) -> str:
        """The path name (``linear`` / ``point``) later pose targets ride."""
        return self._pose_path

    def set_pose_path(self, name: str) -> str:
        """Select ``move_l`` / ``move_p`` for later pose targets."""
        with self._lock:
            self._pose_path = name
            return name

    def seed_joints(self, joints: list[float]) -> None:
        """Hold the current joints for the first command; sends nothing."""
        with self._lock:
            self._commanded[SPACE_JOINT] = list(joints)
            self._commanded[SPACE_POSE] = [None] * 6

    def send(
        self,
        robot: Any,
        space: str,
        feedback: list[float] | None,
        targets: list[tuple[int, float]],
        *, validate: Callable[[], None] | None = None,
    ) -> list[float] | None:
        """Send one full hold vector in ``space``; None when nothing could be held.

        A ``None`` return is decided before anything is sent: neither a prior
        command nor feedback existed for an axis the caller did not name.
        """
        with self._lock:
            vector = [
                commanded if commanded is not None else (feedback[i] if feedback else None)
                for i, commanded in enumerate(self._commanded[space])
            ]
            if any(value is None for value in vector):
                return None
            for index, value in targets:
                vector[index] = float(value)
            snapshot = [float(value) for value in vector]  # type: ignore[arg-type]
            try:
                if validate is not None:
                    validate()
                if ctrl_mode(robot) != CTRL_MODE_CAN:
                    self._enter_can(robot)
                elif not self._speed_sent:
                    robot.set_speed_percent(self._speed_pct)
                    self._speed_sent = True
                if validate is not None:
                    validate()
                if space == SPACE_JOINT:
                    robot.move_j(joints_to_sdk(snapshot))
                elif self._pose_path == POSE_PATH_POINT:
                    robot.move_p(pose_to_sdk(snapshot))
                else:
                    robot.move_l(pose_to_sdk(snapshot))
            except BaseException:
                self._commanded = {SPACE_JOINT: [None] * 6, SPACE_POSE: [None] * 6}
                self._speed_sent = False
                raise
            other = SPACE_POSE if space == SPACE_JOINT else SPACE_JOINT
            self._commanded[space] = list(snapshot)
            self._commanded[other] = [None] * 6
            return snapshot

    def enter_can(self, robot: Any) -> None:
        """Select CAN control (carrying the staged speed), verified, unconditionally."""
        with self._lock:
            self._enter_can(robot)

    def ensure_can(self, robot: Any) -> None:
        """Enter CAN control only when the arm reports something else.

        A gripper write does not go through the motion path, so it needs the
        same guarantee: outside CAN control the firmware drops the frame while
        the write would still report success. An arm already in CAN is left
        untouched, so an energized arm is never disturbed.
        """
        with self._lock:
            if ctrl_mode(robot) != CTRL_MODE_CAN:
                self._enter_can(robot)

    def _enter_can(self, robot: Any) -> None:
        deadline = time.monotonic() + self._mode_timeout_s
        while True:
            self._speed_sent = False
            robot.set_speed_percent(self._speed_pct)
            self._speed_sent = True
            window = min(deadline, time.monotonic() + _MODE_RETRY_S)
            while time.monotonic() < window:
                time.sleep(_MODE_POLL_S)
                if ctrl_mode(robot) == CTRL_MODE_CAN:
                    return
            if time.monotonic() >= deadline:
                seen = ctrl_mode_name(ctrl_mode(robot)) or "no feedback"
                raise TransportError(
                    f"the arm did not enter {ctrl_mode_name(CTRL_MODE_CAN)} mode within "
                    f"{self._mode_timeout_s:.1f}s (it reports {seen}); check the teach "
                    "switch, the pendant state, and whether an emergency stop is latched",
                    failure_code="piper_mode_switch_refused",
                )


__all__ = [
    "CTRL_MODE_CAN",
    "CTRL_MODE_NAMES",
    "CTRL_MODE_STANDBY",
    "CTRL_MODE_TEACH",
    "JOINT_CHANNELS",
    "POSE_CHANNELS",
    "POSE_PATHS",
    "SPACE_CHANNELS",
    "SPACE_JOINT",
    "SPACE_POSE",
    "PiperMotion",
    "as_int",
    "attr",
    "ctrl_mode",
    "ctrl_mode_name",
    "enable_flag",
    "enable_status",
    "firmware_version",
    "motion_space",
    "motion_spaces",
    "numeric",
    "pose_from_sdk",
    "pose_to_sdk",
    "read_joints",
    "read_pose",
    "sdk_message",
    "status_feedback",
]
