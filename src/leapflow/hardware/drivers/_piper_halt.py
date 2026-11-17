# Copyright (c) Alibaba, Inc. and its affiliates.
"""Piper stop semantics: hold in place, and the policy for releasing torque.

pyAgxArm offers no hold primitive, and the two stops it does offer both let a
raised arm come down: ``electronic_emergency_stop()`` is documented to let a
raised arm "slowly descend with constant damping", and ``reset()`` /
``disable()`` release the drives so it falls at once. Neither is a halt under
the transport contract, so neither is reachable from one.

A halt here is a *position hold*: the joints the arm reports now are sent
back as its joint target, in the CAN control it is already in. The drives stay
powered, the firmware's own position loop stops the motion where it is, and a
held object stays in the closed gripper (the gripper is not touched). The
control mode is never changed by a halt: an arm in standby or teach mode is not
being driven from here, so commanding CAN control would itself be a motion.

Releasing torque is a separate, deliberate act. ``TorqueReleasePolicy`` decides
whether the governed paths that must release it (resuming a latched firmware
stop, ``arm.enable=0``) may do so: by default never, or -- when the device
profile declares a rest posture -- only while the arm demonstrably rests there.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from leapflow.hardware.arm_status import ARM_STATUS_EMERGENCY_STOP
from leapflow.hardware.drivers._piper_motion import (
    CTRL_MODE_CAN,
    JOINT_CHANNELS,
    MOVE_MODE_J,
    PiperMotion,
    as_int,
    attr,
    ctrl_mode_name,
    joints_to_sdk,
    motion_arrived,
    move_mode,
    numeric,
    read_joints,
    status_feedback,
)
from leapflow.hardware.transport import (
    STOP_EFFECT_DEVICE_LATCHED,
    STOP_EFFECT_HOLD,
    STOP_EFFECT_NONE,
    TransportStatus,
)

logger = logging.getLogger(__name__)

_POLL_S = 0.02
_MODE_RETRY_S = 0.15  # longer than pyAgxArm's identical-mode-frame throttle

HALT_LATCH_MOTION = "motion"
HALT_LATCH_ALWAYS = "always"
HALT_LATCH_POLICIES: tuple[str, ...] = (HALT_LATCH_MOTION, HALT_LATCH_ALWAYS)
"""When a halt refuses later targets until a governed resume.

``motion`` latches only a halt that interrupted something -- a write in flight
or an arm that had not arrived -- because that is the halt a caller must
acknowledge before it may move again. A halt of an idle arm (a shutdown sweep,
a lease expiry after the work finished) holds it without demanding an enable.
``always`` latches every halt.
"""

TORQUE_RELEASE_REFUSE = "refuse"
TORQUE_RELEASE_AT_REST = "at_rest"
TORQUE_RELEASE_POLICIES: tuple[str, ...] = (TORQUE_RELEASE_REFUSE, TORQUE_RELEASE_AT_REST)

_JOINT_NAMES = tuple(sorted(JOINT_CHANNELS, key=JOINT_CHANNELS.__getitem__))


@dataclass(frozen=True)
class HoldSpec:
    """How a halt is verified and whether it latches."""

    verify_s: float = 0.5
    tolerance_deg: float = 2.0
    latch: str = HALT_LATCH_MOTION

    def latches(self, *, in_flight: bool, arrived: bool | None) -> bool:
        """Whether this halt must refuse targets until a governed resume."""
        if self.latch == HALT_LATCH_ALWAYS:
            return True
        return in_flight or arrived is not True


@dataclass(frozen=True)
class HoldReport:
    """What a halt left the arm doing, and whether that was observed."""

    effect: str
    verified: bool | None
    delivered: bool
    detail: str
    joints_deg: tuple[float, ...] | None = None
    drift_deg: float | None = None

    def status(self, *, connected: bool, latched: bool) -> TransportStatus:
        """The transport-contract status of this halt."""
        detail = f"Piper halt: {self.detail}"
        if latched:
            detail += "; targets are refused until a governed arm.enable=1 resumes motion"
        return TransportStatus(
            connected=connected,
            halt_supported=True,
            halt_delivered=self.delivered,
            detail=detail,
            stop_effect=self.effect,
            stop_verified=self.verified,
            metadata={"latched": latched, "hold_joints_deg": list(self.joints_deg or ()),
                      "drift_deg": self.drift_deg},
        )


def halt_arm(robot: Any, motion: PiperMotion, spec: HoldSpec,
             in_flight: bool) -> tuple[HoldReport, bool]:
    """Blocking: hold in place, latching when the halt interrupted something. Never raises.

    ``in_flight`` says whether a write was in progress. The arm's own arrival
    flag is read *before* the hold, because the hold target itself reads as
    "not arrived" for a moment.
    """
    try:
        arrived = motion_arrived(robot)
    except Exception:  # noqa: BLE001 - an unreadable state latches conservatively
        arrived = None
    latched = spec.latches(in_flight=in_flight, arrived=arrived)
    reason = ""
    if latched:
        reason = "a halt that interrupted motion" if in_flight or arrived is False else "a halt"
    report = hold_in_place(robot, motion, spec, latch=reason)
    logger.warning("piper halt: effect=%s verified=%s latched=%s (%s)",
                   report.effect, report.verified, latched, report.detail)
    return report, latched


def hold_in_place(robot: Any, motion: PiperMotion, spec: HoldSpec, *,
                  latch: str = "") -> HoldReport:
    """Blocking: stop the arm where it stands, under power, and watch it stay. Never raises.

    Every command in flight is preempted first (``PiperMotion.preempt``), so no
    frame of it can follow the hold target; a non-empty ``latch`` refuses later
    targets from the same instant. What is sent depends only on what
    the arm itself reports:

    - its own emergency stop is latched -> nothing; the firmware already owns it;
    - it is not in CAN control (standby, teach) -> nothing; it is not being
      driven from here, and selecting CAN control would itself move it;
    - it is in CAN control -> its current joints as the joint target, after
      asserting joint move mode (a target in any other mode is dropped).
    """
    try:
        with motion.preempt("a halt", latch=latch):
            report = _send_hold(robot, motion)
    except Exception as exc:  # noqa: BLE001 - a halt never raises
        logger.error("piper: hold-in-place failed: %s", exc, exc_info=True)
        return HoldReport(STOP_EFFECT_NONE, False, False,
                          f"the hold target could not be sent ({exc}); the arm may still be "
                          "moving -- use the physical emergency-stop button if it is")
    if report.joints_deg is None or report.effect == STOP_EFFECT_DEVICE_LATCHED:
        return report
    return _verify(robot, report, spec)


def _send_hold(robot: Any, motion: PiperMotion) -> HoldReport:
    """Under the emit lock: decide from the arm's own state and send the hold target."""
    status = status_feedback(robot)
    joints = read_joints(robot)
    vector = tuple(joints) if joints is not None else None
    if as_int(attr(status, "arm_status")) == ARM_STATUS_EMERGENCY_STOP:
        return HoldReport(STOP_EFFECT_DEVICE_LATCHED, None, True,
                          "the arm's own emergency stop is already engaged; nothing was sent",
                          vector)
    if vector is None:
        return HoldReport(STOP_EFFECT_NONE, False, False,
                          "no joint feedback to hold the arm at, so no target was sent; check "
                          "the CAN link -- the physical emergency-stop button is the remaining stop")
    mode = as_int(attr(status, "ctrl_mode"))
    if mode != CTRL_MODE_CAN:
        return HoldReport(STOP_EFFECT_NONE, None, True,
                          f"the arm is in {ctrl_mode_name(mode) or 'an unreported'} mode, not "
                          "driven from here; nothing was sent", vector)
    confirmed = _joint_mode(robot, motion.mode_timeout_s)
    robot.move_j(joints_to_sdk(vector))
    detail = "the arm holds its current joints under power"
    if not confirmed:
        detail += "; joint move mode was not confirmed before the hold target"
    return HoldReport(STOP_EFFECT_HOLD, None, True, detail, vector)


def _joint_mode(robot: Any, timeout_s: float) -> bool:
    """Assert joint move mode and wait (bounded) for the firmware to report it."""
    deadline = time.monotonic() + timeout_s
    while move_mode(robot) != MOVE_MODE_J:
        if time.monotonic() >= deadline:
            return False
        robot.set_motion_mode("j")
        window = min(deadline, time.monotonic() + _MODE_RETRY_S)
        while time.monotonic() < window and move_mode(robot) != MOVE_MODE_J:
            time.sleep(_POLL_S)
    return True


def _verify(robot: Any, report: HoldReport, spec: HoldSpec) -> HoldReport:
    """Watch the joints for ``spec.verify_s``; the hold counts only if they stay."""
    target = report.joints_deg
    assert target is not None
    deadline = time.monotonic() + spec.verify_s
    drift: float | None = 0.0
    while True:
        joints = read_joints(robot)
        drift = (max(abs(a - b) for a, b in zip(joints, target))
                 if joints is not None else None)
        if drift is None or drift > spec.tolerance_deg or time.monotonic() >= deadline:
            break
        time.sleep(_POLL_S)
    if drift is None:
        verified, note = False, "joint feedback was lost while verifying the stop"
    elif drift > spec.tolerance_deg:
        verified, note = False, (f"the arm moved {drift:.2f} deg after the stop "
                                 f"(tolerance {spec.tolerance_deg:.2f} deg)")
    else:
        verified, note = True, f"it stayed within {spec.tolerance_deg:.2f} deg for {spec.verify_s:.2f}s"
    return HoldReport(report.effect, verified, report.delivered, f"{report.detail}; {note}",
                      target, drift)


@dataclass(frozen=True)
class TorqueReleasePolicy:
    """Whether a governed action may release the drives of a raised arm.

    ``refuse`` (the default) never lets software release torque: the operator
    supports the arm and releases it with the operator tool. ``at_rest`` allows
    it only while every declared rest joint is within ``tolerance_deg`` of the
    reading; ``None`` in ``rest_joints_deg`` marks a joint gravity cannot move.
    """

    mode: str = TORQUE_RELEASE_REFUSE
    rest_joints_deg: tuple[float | None, ...] = ()
    tolerance_deg: float = 5.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> TorqueReleasePolicy:
        """Parse ``torque_release`` / ``rest_joints_deg`` / ``rest_tolerance_deg``.

        Raises ``ValueError`` naming the offending key.
        """
        mode = str(config.get("torque_release") or TORQUE_RELEASE_REFUSE).strip().lower()
        if mode not in TORQUE_RELEASE_POLICIES:
            raise ValueError(f"torque_release must be one of {list(TORQUE_RELEASE_POLICIES)}, "
                             f"got {mode!r}")
        tolerance = numeric(config.get("rest_tolerance_deg", cls.tolerance_deg))
        if tolerance is None or tolerance <= 0:
            raise ValueError("rest_tolerance_deg must be a finite number > 0")
        raw = config.get("rest_joints_deg")
        rest: tuple[float | None, ...] = ()
        if raw is not None:
            if not isinstance(raw, Sequence) or isinstance(raw, str) or len(raw) != len(_JOINT_NAMES):
                raise ValueError(f"rest_joints_deg must list {len(_JOINT_NAMES)} values "
                                 "(null for a joint gravity cannot move)")
            values = [None if value is None else numeric(value) for value in raw]
            if any(value is None and item is not None for value, item in zip(values, raw)):
                raise ValueError("rest_joints_deg values must be finite numbers or null")
            rest = tuple(values)
        if mode == TORQUE_RELEASE_AT_REST and not any(value is not None for value in rest):
            raise ValueError("torque_release: at_rest needs rest_joints_deg naming at least one joint")
        return cls(mode=mode, rest_joints_deg=rest, tolerance_deg=tolerance)

    def refusal(self, joints: Sequence[float] | None, action: str) -> str:
        """Why ``action`` may not release torque now; ``""`` when it may."""
        lead = (f"{action} releases the arm's drives, and a raised arm falls when they let go; "
                "LeapFlow does not do that on its own")
        if self.mode == TORQUE_RELEASE_REFUSE:
            return (f"{lead}. Support the arm or lower it onto its rest, then release it with the "
                    "operator tool -- or declare transport.config.torque_release: at_rest with "
                    "rest_joints_deg so it is allowed once the arm rests there")
        if joints is None:
            return f"{lead}. There is no joint feedback to prove the arm rests; check the CAN link"
        away = [
            f"{name} at {value:.1f} deg (rest {rest:.1f} +/- {self.tolerance_deg:.1f})"
            for name, value, rest in zip(_JOINT_NAMES, joints, self.rest_joints_deg)
            if rest is not None and abs(value - rest) > self.tolerance_deg
        ]
        if away:
            return f"{lead}. The arm is not at its declared rest: {', '.join(away)}; lower it first"
        return ""


__all__ = [
    "HALT_LATCH_ALWAYS",
    "HALT_LATCH_MOTION",
    "HALT_LATCH_POLICIES",
    "TORQUE_RELEASE_AT_REST",
    "TORQUE_RELEASE_POLICIES",
    "TORQUE_RELEASE_REFUSE",
    "HoldReport",
    "HoldSpec",
    "TorqueReleasePolicy",
    "halt_arm",
    "hold_in_place",
]
