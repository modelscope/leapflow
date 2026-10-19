# Copyright (c) Alibaba, Inc. and its affiliates.
"""Piper motion spaces, SDK feedback codec, and serialized command state.

Split out of ``piper.py`` so the transport keeps lifecycle, channel routing and
side-effect verdicts, while this module owns the vendor-facing encoding:

- **Two motion spaces.** Joint targets ride ``JointCtrl`` under MOVE J;
  end-effector pose targets ride ``EndPoseCtrl`` under MOVE L (straight-line,
  the default) or MOVE P (point-to-point). The firmware holds one move mode at
  a time, so the command state tracks it and re-selects only on a change. A
  batch that mixes both spaces has no single honest vendor command and is
  refused before any send (``mixed_motion_space``).
- **Hold semantics.** Every motion command is a full six-value vector; values
  the caller did not name are filled from the last command in the same space
  or, failing that, from current feedback. A command in one space invalidates
  the other space's cache -- after a pose move the last joint command no
  longer describes where the arm is, and holding it would jump the arm back.
  A send that raised leaves the arm in an unknown place, so both caches are
  dropped and the next command re-seeds from feedback.
- **Units.** Pose feedback and commands are 0.001 mm (X/Y/Z) and 0.001 deg
  (RX/RY/RZ): the same uniform 1e-3 scale as joints, gripper stroke and effort.
"""

from __future__ import annotations

import threading
from typing import Any, Iterable

SCALE = 1000.0
"""SDK-native quantization: declared unit value * 1000 == raw integer."""

CTRL_MODE_CAN = 0x01
MOVE_P = 0x00
MOVE_J = 0x01
MOVE_L = 0x02
POSE_MOVE_MODES: dict[str, int] = {"linear": MOVE_L, "point": MOVE_P}

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
_POSE_FIELDS = ("X_axis", "Y_axis", "Z_axis", "RX_axis", "RY_axis", "RZ_axis")


# ── SDK struct codec ─────────────────────────────────────────────────────────


def attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read one attribute from an SDK feedback struct, tolerating absent fields."""
    value = getattr(obj, name, default)
    return default if value is None else value


def as_int(value: Any) -> int | None:
    """Coerce a raw feedback field to int; booleans and garbage are not angles."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def numeric(value: Any) -> float | None:
    """Coerce a commanded value to a finite float, rejecting bools/NaN/inf."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
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


def parse_joints(msg: Any) -> list[int] | None:
    """Six raw joint angles from ``GetArmJointMsgs``, or None when unusable."""
    state = attr(msg, "joint_state", msg)
    raw = [as_int(attr(state, f"joint_{i}")) for i in range(1, 7)]
    if any(value is None for value in raw):
        return None
    return [int(value) for value in raw]  # type: ignore[arg-type]


def parse_pose(msg: Any) -> list[int] | None:
    """Six raw pose values from ``GetArmEndPoseMsgs``, or None when unusable."""
    state = attr(msg, "end_pose", msg)
    raw = [as_int(attr(state, name)) for name in _POSE_FIELDS]
    if any(value is None for value in raw):
        return None
    return [int(value) for value in raw]  # type: ignore[arg-type]


def firmware_version(arm: Any) -> str:
    """Return the firmware string; ``-0x4AF`` means the query has no answer yet."""
    try:
        raw = arm.GetPiperFirmwareVersion()
    except Exception:  # noqa: BLE001 - probe metadata must never break on SDK quirks
        return ""
    if not isinstance(raw, str) or raw.startswith("-0x"):
        return ""
    return raw


def interface_version(arm: Any) -> str:
    try:
        raw = arm.GetCurrentInterfaceVersion()
    except Exception:  # noqa: BLE001
        return ""
    return str(raw) if raw is not None else ""


def status_feedback(arm: Any) -> Any:
    """Return the arm-status feedback struct, or None when unavailable."""
    try:
        msg = arm.GetArmStatus()
    except Exception:  # noqa: BLE001
        return None
    feedback = attr(msg, "arm_status", msg)
    if attr(feedback, "arm_status") is None and attr(feedback, "motion_status") is None:
        return None
    return feedback


def enable_status(arm: Any) -> list[bool] | None:
    """Return the per-motor enable list, or None when unavailable."""
    try:
        raw = arm.GetArmEnableStatus()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(raw, (list, tuple)):
        return None
    return [bool(item) for item in raw]


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
    The lock spans mode selection, vector fill, send, and cache advance, so
    concurrent writers can neither interleave vectors nor race a mode switch.
    """

    def __init__(self, *, speed_pct: int, pose_move_mode: int) -> None:
        self._lock = threading.Lock()
        self._speed_pct = speed_pct
        self._pose_move_mode = pose_move_mode
        self._mode: int | None = None
        self._commanded: dict[str, list[int | None]] = {
            SPACE_JOINT: [None] * 6,
            SPACE_POSE: [None] * 6,
        }

    @property
    def mode(self) -> int | None:
        """The move mode last selected on the arm; None when unknown."""
        return self._mode

    @property
    def speed_pct(self) -> int:
        """The speed rate (1-100 %) every ``ModeCtrl`` carries."""
        return self._speed_pct

    def set_speed(self, pct: int) -> int:
        """Change the speed rate; the next send re-selects the mode to apply it.

        The vendor carries the rate only on ``ModeCtrl``, so clearing the cached
        mode is what makes the change take effect without an extra frame now.
        """
        with self._lock:
            self._speed_pct = min(100, max(1, int(pct)))
            self._mode = None
            return self._speed_pct

    @property
    def pose_path(self) -> str:
        """The path name (``linear`` / ``point``) later pose targets ride."""
        return next(n for n, mode in POSE_MOVE_MODES.items() if mode == self._pose_move_mode)

    def set_pose_path(self, name: str) -> str:
        """Select MOVE L / MOVE P for later pose targets; the next pose send applies it."""
        with self._lock:
            self._pose_move_mode = POSE_MOVE_MODES[name]
            return name

    def select_joint_mode(self, arm: Any, joints: list[int]) -> None:
        """Bring-up: select CAN command + MOVE J and seed the joint hold vector."""
        with self._lock:
            self._select(arm, MOVE_J)
            self._commanded[SPACE_JOINT] = list(joints)
            self._commanded[SPACE_POSE] = [None] * 6

    def send(
        self,
        arm: Any,
        space: str,
        feedback: list[int] | None,
        targets: list[tuple[int, float]],
    ) -> list[int] | None:
        """Send one full hold vector in ``space``; None when nothing could be held.

        A ``None`` return is decided before anything is sent: neither a prior
        command nor feedback existed for an axis the caller did not name.
        """
        with self._lock:
            vector = [
                commanded if commanded is not None else (feedback[i] if feedback else None)
                for i, commanded in enumerate(self._commanded[space])
            ]
            if any(raw is None for raw in vector):
                return None
            for index, value in targets:
                vector[index] = int(round(value * SCALE))
            snapshot = [int(raw) for raw in vector]  # type: ignore[arg-type]
            mode = MOVE_J if space == SPACE_JOINT else self._pose_move_mode
            try:
                if self._mode != mode:
                    self._select(arm, mode)
                if space == SPACE_JOINT:
                    arm.JointCtrl(*snapshot)
                else:
                    arm.EndPoseCtrl(*snapshot)
            except BaseException:
                self._commanded = {SPACE_JOINT: [None] * 6, SPACE_POSE: [None] * 6}
                raise
            other = SPACE_POSE if space == SPACE_JOINT else SPACE_JOINT
            self._commanded[space] = list(snapshot)
            self._commanded[other] = [None] * 6
            return snapshot

    def _select(self, arm: Any, mode: int) -> None:
        # Unknown until the frame is accepted: a raising ModeCtrl must force a
        # re-select on the next command rather than trust a stale mode.
        self._mode = None
        arm.ModeCtrl(
            ctrl_mode=CTRL_MODE_CAN,
            move_mode=mode,
            move_spd_rate_ctrl=self._speed_pct,
        )
        self._mode = mode


__all__ = [
    "CTRL_MODE_CAN",
    "JOINT_CHANNELS",
    "MOVE_J",
    "MOVE_L",
    "MOVE_P",
    "POSE_CHANNELS",
    "POSE_MOVE_MODES",
    "SCALE",
    "SPACE_JOINT",
    "SPACE_POSE",
    "PiperMotion",
    "motion_space",
    "motion_spaces",
    "parse_joints",
    "parse_pose",
]
