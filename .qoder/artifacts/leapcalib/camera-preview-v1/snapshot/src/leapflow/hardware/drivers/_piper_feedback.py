# Copyright (c) Alibaba, Inc. and its affiliates.
"""Piper cached-feedback reads and their receive evidence.

Polling success is never receive evidence. Receipts are stamped at the SDK's
receive boundary, after the vendor parser has written the frame into its cache;
missing receipts fail closed. ``PiperFeedback`` reads the SDK caches for the
declared feedback channels and certifies each value with the receipts of the
CAN groups that carry it. Nothing here opens sockets, instantiates an SDK,
sends a frame, or infers reception from changing values.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, ContextManager, Mapping, Protocol, runtime_checkable

from leapflow.hardware.arm_status import COMM_FAULT_SHIFT, LIMIT_FAULT_SHIFT, joint_faults
from leapflow.hardware.drivers._piper_motion import (
    SPACE_CHANNELS,
    SPACE_JOINT,
    as_int,
    attr,
    ctrl_mode_name,
    enable_status,
    motion_space,
    numeric,
    read_joints,
    read_pose,
    sdk_message,
    status_feedback,
)
from leapflow.hardware.transport import TransportError


@dataclass(frozen=True)
class ReceiveReceipt:
    group: str
    sequence: int
    received_mono: float
    received_wall: float
    interval_s: float | None
    valid: bool

    def to_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class ReceiveSource(Protocol):
    """Receipts from the same decoder that supplies the sampled values."""

    def receipts(self) -> Mapping[str, ReceiveReceipt]: ...

    def capture(self) -> ContextManager: ...


class ReceiveLedger:
    """Thread-safe per-message ledger, fed at a decoder's receive boundary."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._rows: dict[str, ReceiveReceipt] = {}

    def record(self, group: str, *, valid: bool = True,
               mono: float | None = None, wall: float | None = None) -> None:
        mono = time.monotonic() if mono is None else mono
        wall = time.time() if wall is None else wall
        if not group or not math.isfinite(mono) or not math.isfinite(wall):
            raise ValueError("receipt needs a group and finite clocks")
        with self._lock:
            old = self._rows.get(group)
            if old is not None and mono <= old.received_mono:
                raise ValueError("receive clock must advance")
            self._rows[group] = ReceiveReceipt(
                group, old.sequence + 1 if old else 1, mono, wall,
                mono - old.received_mono if old else None, valid,
            )

    def receipts(self) -> Mapping[str, ReceiveReceipt]:
        with self._lock:
            return dict(self._rows)

    def capture(self) -> ContextManager:
        """Serialize a complete cache read with the decoder that updates it."""
        return self._lock


def attach_receive_ledger(robot: Any) -> ReceiveLedger:
    """Stamp every monitored frame pyAgxArm receives into a fresh ledger.

    pyAgxArm runs its packet callbacks in registration order on the read
    thread, and the arm/effector parsers register first (at construction and
    ``init_effector``), so a receipt is never stamped before its cache was
    updated: a reader can only see a receipt that is older than the cached
    value, which errs toward stale. A frame is valid when it is a data frame
    with the full 8-byte payload every monitored feedback message carries.
    The callback never raises: an exception there would close the SDK's bus.
    """
    ledger = ReceiveLedger()

    def on_frame(message: Any) -> None:
        try:
            group = f"0x{int(message.arbitration_id):03x}"
            if group not in CAN_GROUPS:
                return
            valid = not getattr(message, "is_remote_frame", False) and int(
                getattr(message, "dlc", 0)) == 8
            ledger.record(group, valid=valid)
        except Exception:  # noqa: BLE001 - must never break the SDK read thread
            return

    robot.get_context().register_parser_packet_fun(on_frame)
    return ledger


CAN_GROUPS = tuple(f"0x{i:03x}" for i in (
    *range(0x2A1, 0x2A9), *range(0x251, 0x257), *range(0x261, 0x267)
))

_REQUALIFY_POLL_S = 0.02
"""How often a bounded reception wait re-reads the ledger."""


def receipt_problem(rows: Mapping[str, Mapping], required: tuple[str, ...], *,
                    now: float, max_age_s: float, max_skew_s: float) -> str:
    """Reject absent, invalid, future, frozen or mixed-aged decoder groups."""
    if (not isinstance(rows, Mapping) or not required
            or not all(math.isfinite(v) for v in (now, max_age_s, max_skew_s))
            or max_age_s <= 0 or max_skew_s < 0):
        return "invalid receive contract"
    stamps = []
    for group in required:
        row = rows.get(group, {})
        if not isinstance(row, Mapping):
            return f"{group}: receive evidence unavailable"
        stamp, sequence = row.get("received_mono"), row.get("sequence")
        if (row.get("valid") is not True or not isinstance(stamp, (int, float))
                or isinstance(stamp, bool) or not math.isfinite(stamp)
                or not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= 0):
            return f"{group}: receive evidence unavailable"
        age = now - stamp
        if not 0 <= age <= max_age_s:
            return f"{group}: receive age {age:.3f}s"
        stamps.append(stamp)
    if stamps and max(stamps) - min(stamps) > max_skew_s:
        return "inter-group receive skew"
    return ""


GRIPPER_POSITION = "gripper.position"
GRIPPER_EFFORT = "gripper.effort"
ARM_ENABLED = "arm.enabled"
ARM_STATUS = "arm.status"
ARM_MOTION = "arm.motion"
ARM_CTRL_MODE = "arm.ctrl_mode"
ARM_TELEMETRY = "arm.telemetry"
GRIPPER_CHANNELS = (GRIPPER_POSITION, GRIPPER_EFFORT)
STATUS_CHANNELS = (ARM_STATUS, ARM_MOTION, ARM_CTRL_MODE)

_JOINT_GROUPS = tuple(f"0x{i:03x}" for i in range(0x2A5, 0x2A8))
_POSE_GROUPS = tuple(f"0x{i:03x}" for i in range(0x2A2, 0x2A5))
_GRIPPER_GROUPS = ("0x2a8",)
_STATUS_GROUPS = ("0x2a1", *(f"0x{i:03x}" for i in range(0x261, 0x267)))
_MOTION_STATUS_ARRIVED = 0x00
_MM_PER_M = 1000.0
_FAULT_FLAGS = (
    "driver_overcurrent",
    "driver_overheating",
    "driver_error_status",
    "collision_status",
    "stall_status",
    "motor_overheating",
    "voltage_too_low",
)


def receive_groups(channel_id: str) -> tuple[str, ...]:
    """The CAN feedback groups whose reception certifies ``channel_id``."""
    if channel_id == ARM_TELEMETRY:
        return CAN_GROUPS
    space = motion_space(channel_id)
    if space:
        return _JOINT_GROUPS if space == SPACE_JOINT else _POSE_GROUPS
    return _GRIPPER_GROUPS if channel_id in GRIPPER_CHANNELS else _STATUS_GROUPS


def status_value(feedback: Any) -> dict[str, Any]:
    """The arm-status codes LeapFlow reports, from one status message body.

    ``err_code`` is the firmware's 16-bit fault word: the high byte marks the
    joints whose angle is out of limit, the low byte the joints whose motor
    stopped answering. Reporting it decoded is what makes ``arm_status`` 0x05 --
    "joint communication error" -- name the joints that went silent instead of
    leaving an operator to decode a number the arm already knows.
    """
    status: dict[str, Any] = {
        key: as_int(attr(feedback, key))
        for key in ("ctrl_mode", "arm_status", "mode_feedback", "motion_status")
    }
    code = as_int(attr(feedback, "err_code"))
    status["error_code"] = code
    status["joint_limit_errors"] = joint_faults(code, LIMIT_FAULT_SHIFT)
    status["joint_communication_errors"] = joint_faults(code, COMM_FAULT_SHIFT)
    return status


class PiperFeedback:
    """Cached-feedback reads and receive evidence for one open arm.

    Every read comes from the SDK read thread's caches, so it never touches
    the bus; ``None`` marks feedback the SDK has not produced.
    """

    def __init__(self, robot: Any, gripper: Any, ledger: ReceiveLedger, *,
                 max_age_s: float, max_skew_s: float) -> None:
        self._robot = robot
        self._gripper = gripper
        self._ledger = ledger
        self._max_age_s = max_age_s
        self._max_skew_s = max_skew_s

    def problem(self, groups: tuple[str, ...] = CAN_GROUPS,
                receipts: Mapping[str, Mapping] | None = None) -> str:
        """Why ``groups`` are not freshly and coherently received; ``""`` when they are."""
        if receipts is None:
            receipts = self.receipts()
        return receipt_problem(receipts, groups, now=time.monotonic(),
                               max_age_s=self._max_age_s, max_skew_s=self._max_skew_s)

    def wait_qualified(self, timeout_s: float) -> str:
        """Poll for a coherent ledger for at most ``timeout_s``; ``""`` once it is.

        Reception is a state, not an instant. A capture that holds the
        interpreter -- a depth frame decoded and converted in the same process --
        delays the SDK's read thread long enough for one decoder group's last
        frame to sit a few hundred milliseconds behind the others, and the next
        single sample then reads that as a bus that is not carrying feedback. It
        is carrying it; it had not delivered all of it yet. Writes therefore
        establish freshness the same way reads do, within a bounded budget, and
        only a ledger that stays incoherent for the whole budget is a refusal.
        """
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            problem = self.problem()
            if not problem or time.monotonic() >= deadline:
                return problem
            time.sleep(min(_REQUALIFY_POLL_S, max(0.0, deadline - time.monotonic())))

    def require(self) -> None:
        """Raise unless every monitored group is freshly received."""
        problem = self.problem()
        if problem:
            raise TransportError(problem, failure_code="piper_unqualified_feedback")

    def receipts(self) -> dict[str, dict]:
        with self._ledger.capture():
            return {key: row.to_dict() for key, row in self._ledger.receipts().items()}

    def evidence(self, channel_id: str,
                 receipts: Mapping[str, Mapping] | None) -> tuple[float, float] | None:
        """(wall, monotonic) of the oldest group carrying ``channel_id``; None when stale."""
        groups = receive_groups(channel_id)
        if receipts is None or self.problem(groups, receipts):
            return None
        return (min(receipts[g]["received_wall"] for g in groups),
                min(receipts[g]["received_mono"] for g in groups))

    def joints(self) -> list[float] | None:
        return read_joints(self._robot)

    def space(self, space: str) -> list[float] | None:
        """Current feedback in one motion space (joint deg / pose mm+deg)."""
        return read_joints(self._robot) if space == SPACE_JOINT else read_pose(self._robot)

    def gripper(self) -> tuple[float | None, float | None]:
        """Gripper stroke (mm) and force (N), or Nones when unavailable."""
        state = sdk_message(self._gripper.get_gripper_status) if self._gripper is not None else None
        stroke, force = numeric(attr(state, "value")), numeric(attr(state, "force"))
        return (stroke * _MM_PER_M if stroke is not None else None, force)

    def poll(self, channel_ids: tuple[str, ...]) -> dict[str, Any]:
        """Read the requested feedback channels plus the receipts that certify them.

        The read runs under the ledger lock, so the receipts returned with the
        values (under ``"_receive"``) cannot advance mid-read.
        """
        wanted = set(channel_ids)
        values: dict[str, Any] = {}
        with self._ledger.capture():
            for space, channels in SPACE_CHANNELS.items():
                if wanted.isdisjoint(channels):
                    continue
                vector = self.space(space)
                for cid, index in channels.items():
                    values[cid] = vector[index] if vector is not None else None
            if not wanted.isdisjoint(GRIPPER_CHANNELS):
                stroke, force = self.gripper()
                values[GRIPPER_POSITION] = stroke
                values[GRIPPER_EFFORT] = force if stroke is not None else None
            if ARM_ENABLED in wanted:
                enabled = enable_status(self._robot)
                values[ARM_ENABLED] = all(enabled) if enabled is not None else None
            if ARM_TELEMETRY in wanted:
                values[ARM_TELEMETRY] = self._telemetry()
            if not wanted.isdisjoint(STATUS_CHANNELS):
                feedback = status_feedback(self._robot)
                status = status_value(feedback) if feedback is not None else None
                motion = status["motion_status"] if status else None
                values[ARM_STATUS] = status
                values[ARM_MOTION] = motion == _MOTION_STATUS_ARRIVED if motion is not None else None
                values[ARM_CTRL_MODE] = ctrl_mode_name(status["ctrl_mode"]) if status else None
            values["_receive"] = self.receipts()
        return values

    def _telemetry(self) -> dict[str, Any]:
        """Bounded vendor diagnostics through normal Reading/filter/store routing.

        ``torque`` is the SDK's current-derived estimate, not measured force;
        the field name and ``effort_source`` keep that limitation explicit.
        """
        motors = []
        for index in range(1, 7):
            fast = sdk_message(self._robot.get_motor_states, index)
            slow = sdk_message(self._robot.get_driver_states, index)
            foc = attr(slow, "foc_status")
            motors.append({
                "motor": index,
                "current_a": numeric(attr(fast, "current")),
                "estimated_effort_nm": numeric(attr(fast, "torque")),
                "speed_rad_s": numeric(attr(fast, "velocity")),
                "voltage_v": numeric(attr(slow, "vol")),
                "motor_temperature_c": numeric(attr(slow, "motor_temp")),
                "driver_temperature_c": numeric(attr(slow, "foc_temp")),
                "enabled": attr(foc, "driver_enable_status"),
                "faults": {name: attr(foc, name) for name in _FAULT_FLAGS},
            })
        receipts = self.receipts()
        return {"schema_version": 1, "motors": motors, "receive": receipts,
                "receive_qualified": not self.problem(CAN_GROUPS, receipts),
                "effort_source": "sdk_current_estimate"}
