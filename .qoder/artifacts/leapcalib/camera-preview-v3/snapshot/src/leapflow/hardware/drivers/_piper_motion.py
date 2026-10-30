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
  the caller did not name are filled from the last command this driver *sent*
  in the same space or, failing that, from feedback read at send time. A
  command in one space voids the other space's hold -- after a pose move the
  last joint command no longer describes where the arm is, and holding it
  would jump the arm back. A send that raised leaves the arm in an unknown
  place, so both holds are dropped.

  A hold is a *commitment* -- "the arm is where I told it to be" -- and it is
  only true while this driver still owns the arm's posture. Anything that ends
  that ownership (a governed enable or reset, a disable, an emergency stop, a
  closed link) voids it through ``disown``. The distinction is not cosmetic: a
  joint frame read while the arm was being energised is a measurement of a
  transition, not a command, and holding it moved an axis the caller had never
  named by 2.2 degrees.
- **Control-mode entry.** The firmware executes host commands only in CAN
  control mode. ``set_speed_percent`` sends the mode frame that selects CAN;
  ``enter_can`` repeats it until ``ctrl_mode`` feedback confirms, so a dropped
  frame becomes a retry and a refusal (teach switch, latched stop) a typed
  failure instead of a silent no-op.
- **Move-mode assertion.** That same frame carries move mode ``0xFF`` -- the
  protocol's UNKNOWN -- and restores the previous mode only in this driver's
  cache, so the firmware stays UNKNOWN until the next mode frame. A target
  delivered while it is UNKNOWN is dropped without an error: the arm does not
  move, and the write still reads committed. Every send therefore asserts the
  mode its targets belong to and confirms the firmware latched it before the
  target frames go out. Nothing outside a command writes a mode frame; a
  stray speed frame cancels the motion it was meant to leave alone.
- **Target acceptance.** Latching the mode only says the firmware will listen;
  it does not say it will execute. The controller answers the posture itself
  through its status word, a frame or two later: a planning refusal (no IK, a
  singularity, a joint limit) leaves the arm exactly where the command found
  it, and a fault stops it mid-route. Either answer makes "committed" a lie,
  and afterwards neither is distinguishable from a slow arrival, so every send
  reads the status word over a short window and raises before the write can be
  reported as executed.
- **Units.** Declared channels use deg / mm; pyAgxArm uses rad / m. Pose
  orientation is fixed-axis X-Y-Z (roll, pitch, yaw); the SDK refuses roll/yaw
  outside [-pi, pi] and pitch outside [-pi/2, pi/2], so held feedback is
  wrapped/clamped onto those ranges before it is re-sent.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Callable, Iterable, Sequence

from leapflow.hardware.arm_status import (
    ARM_STATUS_NORMAL,
    LIMIT_FAULT_SHIFT,
    TARGET_REFUSALS,
    joint_faults,
    status_name,
)
from leapflow.hardware.transport import TransportError

logger = logging.getLogger(__name__)

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
# The firmware answers the target itself: a refusal or a fault shows up in the
# status word one or two CAN frames after the posture was sent. The window is the
# longest wait for that answer before the target counts as accepted.
_STATUS_WINDOW_S = 0.10
_STATUS_POLL_S = 0.01

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

MOVE_MODE_P = 0x00
MOVE_MODE_J = 0x01
MOVE_MODE_L = 0x02
MOVE_MODE_NAMES: dict[int, str] = {MOVE_MODE_P: "p", MOVE_MODE_J: "j", MOVE_MODE_L: "l"}

# One firmware request may be silently dropped before the SDK's fps estimate is
# positive, so the budget is spent on several short requests, paced so a reply
# that never comes cannot turn the retry into a bus flood.
FIRMWARE_REQUEST_S = 0.25
FIRMWARE_RETRY_PAUSE_S = 0.05


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


def move_mode(robot: Any) -> int | None:
    """The motion mode the firmware reports it has latched, if any."""
    return as_int(attr(status_feedback(robot), "mode_feedback"))


def arm_status(robot: Any) -> int | None:
    """The verdict code the firmware reports on the posture it was given, if any."""
    return as_int(attr(status_feedback(robot), "arm_status"))


def move_mode_name(value: Any) -> str:
    """The declared name of a raw motion mode, naming an unassigned code explicitly."""
    mode = as_int(value)
    if mode is None:
        return "no feedback"
    return MOVE_MODE_NAMES.get(mode, f"unknown(0x{mode:02X})")


def posture_text(space: str, vector: Sequence[float]) -> str:
    """The commanded posture as ``channel=value`` pairs, in axis order."""
    channels = sorted(SPACE_CHANNELS[space].items(), key=lambda item: item[1])
    return ", ".join(f"{channel_id}={vector[index]:.3f}" for channel_id, index in channels)


def sdk_move_mode(space: str, pose_path: str) -> tuple[str, int]:
    """The SDK mode name and firmware code the targets in ``space`` require."""
    if space == SPACE_JOINT:
        return "j", MOVE_MODE_J
    return ("p", MOVE_MODE_P) if pose_path == POSE_PATH_POINT else ("l", MOVE_MODE_L)


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
    """The main-controller software version (e.g. ``S-V1.9-0``); ``""`` when unanswered.

    The request is re-sent inside ``timeout_s`` because pyAgxArm's readiness
    predicate *discards* an answer that arrives while its own fps estimate is
    still ``<= 0`` and then waits out the rest of the window instead of asking
    again. The reply to the very first request after ``connect()`` usually beats
    that estimate, so a single request reports a healthy arm as unverifiable.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return ""
        try:
            info = robot.get_firmware(timeout=min(FIRMWARE_REQUEST_S, remaining), min_interval=0.0)
        except Exception:  # noqa: BLE001 - probe metadata must never break on SDK quirks
            info = None
        version = info.get("software_version") if isinstance(info, dict) else None
        if isinstance(version, str) and version.strip():
            return version.strip()
        time.sleep(min(FIRMWARE_RETRY_PAUSE_S, max(deadline - time.monotonic(), 0.0)))


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


class HoldRefused(TransportError):
    """A held axis cannot be commanded without moving it; nothing was sent.

    Raised before any frame goes out, so the transport may report this as a
    refusal (side-effect NONE) rather than as an unknown effect.
    """


class TargetRefused(TransportError):
    """The firmware planned the target and refused it; the arm did not move.

    Raised after the target frames were sent, so the frames did reach the arm --
    but its own answer is that the posture cannot be executed. That answer is
    what separates a refusal from an unknown effect: the arm is still where the
    command found it, and re-sending the same posture only asks again.
    """


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
        self._epoch = 0
        # The vector last *sent* in one space, with the control epoch it was
        # sent in. Empty until a command is sent; ``disown`` bumps the epoch
        # so a vector from before a discontinuity can never be held again.
        self._held: dict[str, tuple[int, list[float]]] = {}

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

    def disown(self, reason: str) -> None:
        """Void the held vectors: this driver no longer owns where the arm is.

        Called at every control discontinuity -- enable, reset, disable,
        emergency stop, closed link. Deliberately does not take the motion
        lock, because ``halt`` must never wait behind an in-flight send; a send
        stamps the vector it writes with the epoch it started in, so a bump
        that lands mid-send leaves that vector void instead of trusted.
        Epochs only ever increase, so concurrent bumps collapse harmlessly.
        """
        self._epoch += 1
        logger.info("piper: held command vector dropped after %s", reason)

    def send(
        self,
        robot: Any,
        space: str,
        feedback: list[float] | None,
        targets: list[tuple[int, float]],
        *, validate: Callable[[], None] | None = None,
        hold: Callable[[str, int, float], float] | None = None,
    ) -> list[float] | None:
        """Send one full hold vector in ``space``; None when nothing could be held.

        A ``None`` return is decided before anything is sent: neither a command
        this driver sent nor feedback existed for an axis the caller did not
        name. Feedback is the fallback rather than any cached number, because a
        value the arm was never commanded to cannot be a position it holds.

        ``hold`` sees every axis the caller did not name and either returns the
        value to send instead or raises ``HoldRefused``. The vendor clamps a
        commanded joint into its own limit map without saying so, so a hold
        outside the declared envelope would silently move that axis; deciding it
        here keeps what is sent equal to what the arm is told to do.

        The control and move modes these targets need are established and
        confirmed before the target frames leave (see the module docstring): a
        target the firmware silently drops is indistinguishable, afterwards,
        from an arm that is merely slow to arrive. The same is true of the
        posture itself, so the firmware's answer to it is read before the send
        may be reported as executed.
        """
        with self._lock:
            epoch = self._epoch
            held = self._held.get(space)
            committed = held[1] if held is not None and held[0] == epoch else None
            vector: list[float | None] = [
                committed[i] if committed is not None else (feedback[i] if feedback else None)
                for i in range(6)
            ]
            if any(value is None for value in vector):
                return None
            named = {index for index, _ in targets}
            if hold is not None:
                for index in range(6):
                    if index not in named:
                        vector[index] = hold(space, index, float(vector[index]))  # type: ignore[arg-type]
            for index, value in targets:
                vector[index] = float(value)
            snapshot = [float(value) for value in vector]  # type: ignore[arg-type]
            try:
                if validate is not None:
                    validate()
                if ctrl_mode(robot) != CTRL_MODE_CAN or not self._speed_sent:
                    self._enter_can(robot)
                self._assert_move_mode(robot, space)
                if validate is not None:
                    validate()
                if space == SPACE_JOINT:
                    robot.move_j(joints_to_sdk(snapshot))
                elif self._pose_path == POSE_PATH_POINT:
                    robot.move_p(pose_to_sdk(snapshot))
                else:
                    robot.move_l(pose_to_sdk(snapshot))
                self._confirm_target(robot, space, snapshot)
            except BaseException:
                self._held.clear()
                self._speed_sent = False
                raise
            # One entry replaces both spaces: the other space's hold no longer
            # describes where the arm is once this vector is executed.
            self._held = {space: (epoch, list(snapshot))}
            return snapshot

    def _confirm_target(self, robot: Any, space: str, snapshot: Sequence[float]) -> None:
        """Read the firmware's own answer to the posture just sent.

        A delivered target is not an executed one. The controller plans the
        posture and answers through its status word: a refusal leaves the arm
        exactly where the command found it, and a fault stops it on the way.
        Reporting either as committed lets a caller build on a posture the arm
        never took, so the answer is sampled for one short window before the
        send is allowed to succeed -- the reply is a frame or two behind the
        target, and a single read would evaluate the frame from before it.
        """
        deadline = time.monotonic() + _STATUS_WINDOW_S
        while True:
            feedback = status_feedback(robot)
            code = as_int(attr(feedback, "arm_status"))
            if code is not None and code != ARM_STATUS_NORMAL:
                self._raise_status(code, feedback, space, snapshot)
            if time.monotonic() >= deadline:
                return
            time.sleep(_STATUS_POLL_S)

    @staticmethod
    def _raise_status(code: int, feedback: Any, space: str, snapshot: Sequence[float]) -> None:
        """Raise the typed failure one status code implies for this posture."""
        posture = posture_text(space, snapshot)
        refusal = TARGET_REFUSALS.get(code)
        if refusal is not None:
            joints = joint_faults(as_int(attr(feedback, "err_code")), LIMIT_FAULT_SHIFT) or []
            named = f" (joints {', '.join(str(j) for j in joints)} out of limit)" if joints else ""
            raise TargetRefused(
                f"the arm refused the {space} target before moving: {refusal}{named}; it is "
                f"still where the command found it, so none of [{posture}] was executed",
                failure_code="piper_target_refused",
            )
        raise TransportError(
            f"the arm stopped carrying the command out ({status_name(code)}); "
            f"[{posture}] may have been partly executed",
            failure_code="piper_arm_fault",
        )

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

    def _assert_move_mode(self, robot: Any, space: str) -> None:
        """Put the firmware in the mode these targets belong to, confirmed by feedback.

        The speed frame parks the firmware in move mode UNKNOWN, and the SDK's
        own move call would only restore it as a side effect of the very command
        that depends on it. Asserting the mode here -- and waiting for the
        firmware to report it -- is what separates "the arm is slow to arrive"
        from "the arm never received a target it could act on".
        """
        name, expected = sdk_move_mode(space, self._pose_path)
        deadline = time.monotonic() + self._mode_timeout_s
        while True:
            robot.set_motion_mode(name)
            window = min(deadline, time.monotonic() + _MODE_RETRY_S)
            while time.monotonic() < window:
                time.sleep(_MODE_POLL_S)
                if move_mode(robot) == expected and ctrl_mode(robot) == CTRL_MODE_CAN:
                    return
            if time.monotonic() >= deadline:
                raise TransportError(
                    f"the arm did not adopt {name!r} motion mode within "
                    f"{self._mode_timeout_s:.1f}s (it reports "
                    f"{move_mode_name(move_mode(robot))} in "
                    f"{ctrl_mode_name(ctrl_mode(robot)) or 'no feedback'} control); "
                    "no target was sent",
                    failure_code="piper_move_mode_refused",
                )

__all__ = [
    "CTRL_MODE_CAN",
    "CTRL_MODE_NAMES",
    "CTRL_MODE_STANDBY",
    "CTRL_MODE_TEACH",
    "JOINT_CHANNELS",
    "MOVE_MODE_J",
    "MOVE_MODE_L",
    "MOVE_MODE_NAMES",
    "MOVE_MODE_P",
    "HoldRefused",
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
    "move_mode",
    "move_mode_name",
    "numeric",
    "sdk_move_mode",
    "pose_from_sdk",
    "pose_to_sdk",
    "read_joints",
    "read_pose",
    "sdk_message",
    "status_feedback",
]
