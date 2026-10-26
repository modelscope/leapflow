# Copyright (c) Alibaba, Inc. and its affiliates.
"""Piper link and power lifecycle on pyAgxArm: bring-up, probe state, energize.

Bring-up builds the model/firmware-specific SDK arm, stamps receive receipts,
connects, waits for complete feedback and checks the arm's own firmware
against the declared profile; it never energizes anything. ``energize`` is the
blocking body of the governed ``arm.enable=1`` write, in the order verified on
a Piper X (S-V1.9): resume a latched stop, enable every driver, enter CAN
control. Nothing here maps channels or validates envelopes; the transport owns
that.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Mapping, NamedTuple

from leapflow.hardware.arm_status import (
    ARM_STATUS_EMERGENCY_STOP,
    ARM_STATUS_NAMES,
    ARM_STATUS_NORMAL,
)
from leapflow.hardware.drivers._piper_feedback import (
    PiperFeedback,
    attach_receive_ledger,
    status_value,
)
from leapflow.hardware.drivers._piper_motion import (
    CTRL_MODE_CAN,
    PiperMotion,
    as_int,
    attr,
    ctrl_mode_name,
    enable_status,
    firmware_version,
    numeric,
    read_joints,
    read_pose,
    status_feedback,
)
from leapflow.hardware.transport import (
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_UNKNOWN,
    TransportError,
    WriteOutcome,
)

logger = logging.getLogger(__name__)

GRIPPER_EFFECTOR = "agx_gripper"
POLL_S = 0.02
_ENABLE_RETRY_S = 0.1
ENABLE_SETTLE_S = 1.0
"""How long an enable must hold before it counts as one.

A controller that has just been reset accepts CAN control and then, about half
one second later, reports a joint communication error and drops back to standby
with the drives released -- the arm sags away and every later command finds a
posture nobody commanded. That window is what this observation must outlast.
"""
_ENABLE_RESTART_S = 0.25
"""Pause before retrying an enable the controller dropped, so it can settle."""


class PiperLink(NamedTuple):
    """One connected arm: SDK handles, certified feedback and firmware."""

    robot: Any
    gripper: Any
    feedback: PiperFeedback
    firmware: str


class LinkSpec(NamedTuple):
    """Declared identity and bring-up budgets of one arm."""

    robot_model: str
    profile: str
    can_name: str
    max_age_s: float
    max_skew_s: float
    requalify_s: float
    warmup_timeout_s: float
    firmware_timeout_s: float


def wait_for(predicate: Callable[[], bool], until: float) -> bool:
    """Poll ``predicate`` until it holds or the monotonic deadline passes."""
    while True:
        if predicate():
            return True
        if time.monotonic() >= until:
            return False
        time.sleep(POLL_S)


def open_link(sdk: Any, spec: LinkSpec) -> PiperLink:
    """Blocking bring-up: build, connect, warm up, check firmware. Never leaks a link."""
    try:
        config = sdk.create_agx_arm_config(
            robot=spec.robot_model, comm="can",
            firmeware_version=spec.profile, channel=spec.can_name,
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise TransportError(
            f"pyAgxArm rejected robot={spec.robot_model!r} firmware_profile={spec.profile!r}: {exc}",
            failure_code="piper_invalid_config",
        ) from exc
    robot: Any = None
    try:
        robot = sdk.AgxArmFactory.create_arm(config)
        gripper = robot.init_effector(GRIPPER_EFFECTOR)
        feedback = PiperFeedback(robot, gripper, attach_receive_ledger(robot),
                                 max_age_s=spec.max_age_s, max_skew_s=spec.max_skew_s)
        robot.set_joint_limits_enabled(True)
        robot.connect()
        _warm_up(robot, spec)
        firmware = firmware_version(robot, spec.firmware_timeout_s)
        _check_firmware(sdk, spec, firmware)
        return PiperLink(robot, gripper, feedback, firmware)
    except BaseException:
        # A half-open CAN link must not outlive a failed open: disconnect
        # best-effort and surface the real failure.
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:  # noqa: BLE001 - cleanup must not mask the failure
                pass
        raise


def _warm_up(robot: Any, spec: LinkSpec) -> None:
    """Wait until every feedback group has produced a value; joins nothing."""
    deadline = time.monotonic() + spec.warmup_timeout_s
    while True:
        joints = read_joints(robot)
        if (joints is not None and read_pose(robot) is not None
                and status_feedback(robot) is not None
                and robot.is_connected() and robot.is_ok()):
            return
        if time.monotonic() >= deadline:
            raise TransportError(
                f"Piper arm on {spec.can_name!r} produced no joint/pose/status feedback within "
                f"{spec.warmup_timeout_s:.1f}s; check arm power, the CAN link and its bitrate",
                failure_code="piper_warmup_timeout",
            )
        time.sleep(POLL_S)


def _check_firmware(sdk: Any, spec: LinkSpec, firmware: str) -> None:
    """Refuse a declared profile the arm's own firmware contradicts."""
    resolve = getattr(sdk, "resolve_firmware_profile", None)
    if not firmware or not callable(resolve):
        raise TransportError(
            "cannot verify the arm firmware profile; check CAN reception and use a pyAgxArm "
            "version exposing resolve_firmware_profile",
            failure_code="piper_firmware_unavailable",
        )
    try:
        resolved = str(resolve(spec.robot_model, firmware))
    except (TypeError, ValueError):
        resolved = ""
    if resolved != spec.profile:
        raise TransportError(
            f"firmware {firmware!r} resolves to pyAgxArm profile {resolved or 'unknown'!r}, "
            f"but transport.config.firmware_profile is {spec.profile!r}; update the device profile",
            failure_code="piper_firmware_mismatch",
        )


def link_state(robot: Any) -> dict[str, Any]:
    """Blocking: link health and arm state for probe metadata. Never raises."""
    status = status_value(status_feedback(robot))

    def call(name: str, default: Any) -> Any:
        try:
            return getattr(robot, name)()
        except Exception:  # noqa: BLE001 - probe metadata must never raise
            return default

    return {
        "connected": bool(call("is_connected", False)),
        "healthy": bool(call("is_ok", False)),
        "comm_error": bool(call("has_comm_error", False)),
        "can_fps": float(numeric(call("get_fps", 0.0)) or 0.0),
        "ctrl_mode": status["ctrl_mode"],
        "ctrl_mode_name": ctrl_mode_name(status["ctrl_mode"]),
        "move_mode": status["mode_feedback"],
        "arm_status": status["arm_status"],
        "motion_status": status["motion_status"],
        "enabled_motors": enable_status(robot),
    }


def energize(robot: Any, feedback: PiperFeedback, motion: PiperMotion,
             timeout_s: float, settle_s: float = ENABLE_SETTLE_S,
             requalify_s: float = 0.0) -> WriteOutcome:
    """Blocking: resume a latched stop, energize, enter CAN control, and prove it holds.

    ``reset()`` clears a latched stop (and drops to standby), ``enable()`` is
    repeated until every driver reports enabled, then CAN control is selected
    and confirmed. Any step refused after a frame was sent reports COMMITTED:
    the arm state may already have changed.

    Those three steps are not the whole contract. A controller that has just
    been reset can accept CAN control and then, half a second later, report a
    joint communication error and drop back to standby with its drives released:
    the arm sags away from the posture the caller believes it holds. The motor
    enable flags do not catch this -- they keep reading enabled across an
    emergency stop -- so the only trustworthy evidence is the arm's own status.
    The enable is therefore *observed* for ``settle_s`` and counts only if the
    arm holds CAN control in a normal state throughout. A fault is transient
    (the controller clears it by itself), so the sequence is retried within the
    same ``timeout_s``; if it keeps faulting the caller is refused with the
    fault and the joints named, instead of an arm quietly falling over.

    The posture the arm held before this call is deliberately not carried over
    as a hold vector: while the arm is energized its frame jumps through states
    it was never in, so commanding one back would move axes nobody named.
    """
    problem = feedback.wait_qualified(requalify_s)
    if problem:
        return _outcome(SIDE_EFFECT_NONE, "fresh per-group CAN reception is required before "
                        f"enabling: {problem}", "piper_unqualified_feedback")
    motion.disown("a governed enable")
    deadline = time.monotonic() + timeout_s
    sent = False
    unstable = ""
    try:
        while True:
            sent = True
            if latched(robot):
                robot.reset()
                if not wait_for(lambda: not latched(robot), deadline):
                    return _outcome(SIDE_EFFECT_COMMITTED,
                                    "the arm still reports a latched emergency stop after reset(); "
                                    "release the physical stop button and retry",
                                    "piper_resume_refused")
            _enable_drivers(robot, feedback, deadline)
            # A fault left over from an earlier attempt makes the mode frame
            # pointless and the failure message misleading, so it is read first.
            unstable = _fault_reason(robot)
            if not unstable:
                motion.enter_can(robot)
                unstable = _unstable_reason(robot, settle_s)
            if not unstable:
                feedback.require()
                return WriteOutcome(ok=True, side_effect_state=SIDE_EFFECT_COMMITTED)
            if time.monotonic() + settle_s >= deadline:
                return _outcome(
                    SIDE_EFFECT_COMMITTED,
                    f"the arm would not stay energized: {unstable}; power-cycle the arm and check "
                    "the motor cable before retrying",
                    "piper_enable_unstable",
                )
            logger.warning("piper: enable did not hold (%s); retrying", unstable)
            time.sleep(_ENABLE_RESTART_S)
    except TransportError as exc:
        # Each step refuses with its own code -- resume, enable, mode switch,
        # feedback -- so the caller learns which one refused instead of a
        # blanket verdict that hides where the arm stopped cooperating.
        return _outcome(SIDE_EFFECT_COMMITTED if sent else SIDE_EFFECT_NONE, str(exc),
                        exc.failure_code)
    except Exception as exc:  # noqa: BLE001 - the reset/enable/mode frame may have landed
        return _outcome(SIDE_EFFECT_UNKNOWN, f"enable sequence failed: {exc}", "piper_send_failed")


def _enable_drivers(robot: Any, feedback: PiperFeedback, deadline: float) -> None:
    """Enable every motor, resending until the arm reports all six on."""
    while True:
        robot.enable()
        window = min(deadline, time.monotonic() + _ENABLE_RETRY_S)
        if wait_for(lambda: all(enable_status(robot) or [False]), window):
            feedback.require()
            return
        if time.monotonic() >= deadline:
            raise TransportError(
                "the arm accepted the enable command but not every motor reports enabled; "
                "check the emergency-stop button and the power switch",
                failure_code="piper_enable_refused",
            )


def _unstable_reason(robot: Any, settle_s: float) -> str:
    """Why the arm did not hold healthy CAN control for ``settle_s``; "" when it did."""
    deadline = time.monotonic() + max(settle_s, 0.0)
    while True:
        status = status_value(status_feedback(robot))
        # The fault word is read first: leaving CAN control is how a fault presents
        # itself, and naming the cause is what tells an operator what to look at.
        fault = _status_fault(status)
        if fault:
            return fault
        if status["ctrl_mode"] != CTRL_MODE_CAN:
            return ("it left CAN control ("
                    f"{ctrl_mode_name(status['ctrl_mode']) or 'no feedback'})")
        if time.monotonic() >= deadline:
            return ""
        time.sleep(POLL_S)


def _fault_reason(robot: Any) -> str:
    """The arm's own fault, named from its status word; "" when it reports none."""
    return _status_fault(status_value(status_feedback(robot)))


def _status_fault(status: Mapping[str, Any]) -> str:
    """Render one status word's fault, naming the joints it implicates."""
    code = status.get("arm_status")
    if not isinstance(code, int) or code == ARM_STATUS_NORMAL:
        return ""
    joints = status.get("joint_communication_errors")
    affected = f" (joints {', '.join(str(j) for j in joints)})" if joints else ""
    return f"the arm reports {ARM_STATUS_NAMES.get(code, f'fault 0x{code:02X}')}{affected}"


def latched(robot: Any) -> bool:
    """True while the firmware reports an emergency stop it has not been released from."""
    return as_int(attr(status_feedback(robot), "arm_status")) == ARM_STATUS_EMERGENCY_STOP


def _outcome(state: str, error: str, failure_code: str) -> WriteOutcome:
    return WriteOutcome(ok=False, side_effect_state=state, error=error, failure_code=failure_code)


__all__ = [
    "ARM_STATUS_EMERGENCY_STOP",
    "ENABLE_SETTLE_S",
    "GRIPPER_EFFECTOR",
    "LinkSpec",
    "PiperLink",
    "energize",
    "latched",
    "link_state",
    "open_link",
    "wait_for",
]
