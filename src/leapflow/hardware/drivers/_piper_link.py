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

import time
from typing import Any, Callable, NamedTuple

from leapflow.hardware.drivers._piper_feedback import (
    PiperFeedback,
    attach_receive_ledger,
    status_value,
)
from leapflow.hardware.drivers._piper_motion import (
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

GRIPPER_EFFECTOR = "agx_gripper"
ARM_STATUS_EMERGENCY_STOP = 0x01
POLL_S = 0.02
_ENABLE_RETRY_S = 0.1


class PiperLink(NamedTuple):
    """One connected arm: SDK handles, certified feedback, firmware and seed joints."""

    robot: Any
    gripper: Any
    feedback: PiperFeedback
    firmware: str
    joints: list[float]


class LinkSpec(NamedTuple):
    """Declared identity and bring-up budgets of one arm."""

    robot_model: str
    profile: str
    can_name: str
    max_age_s: float
    max_skew_s: float
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
        joints = _warm_up(robot, spec)
        firmware = firmware_version(robot, spec.firmware_timeout_s)
        _check_firmware(sdk, spec, firmware)
        return PiperLink(robot, gripper, feedback, firmware, joints)
    except BaseException:
        # A half-open CAN link must not outlive a failed open: disconnect
        # best-effort and surface the real failure.
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:  # noqa: BLE001 - cleanup must not mask the failure
                pass
        raise


def _warm_up(robot: Any, spec: LinkSpec) -> list[float]:
    deadline = time.monotonic() + spec.warmup_timeout_s
    while True:
        joints = read_joints(robot)
        if (joints is not None and read_pose(robot) is not None
                and status_feedback(robot) is not None
                and robot.is_connected() and robot.is_ok()):
            return joints
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
             timeout_s: float) -> WriteOutcome:
    """Blocking: resume a latched stop, energize, enter CAN control, verify each step.

    ``reset()`` clears a latched stop (and drops to standby), ``enable()`` is
    repeated until every driver reports enabled, then CAN control is selected
    and confirmed. Any step refused after a frame was sent reports COMMITTED:
    the arm state may already have changed.
    """
    problem = feedback.problem()
    if problem:
        return _outcome(SIDE_EFFECT_NONE, "fresh per-group CAN reception is required before "
                        f"enabling: {problem}", "piper_unqualified_feedback")
    joints = read_joints(robot)
    if joints is None:
        return _outcome(SIDE_EFFECT_NONE, "no joint feedback to seed the hold vector before "
                        "enabling", "piper_no_feedback")
    sent = False
    try:
        if _latched(robot):
            sent = True
            robot.reset()
            if not wait_for(lambda: not _latched(robot), time.monotonic() + timeout_s):
                return _outcome(SIDE_EFFECT_COMMITTED,
                                "the arm still reports a latched emergency stop after reset(); "
                                "release the physical stop button and retry",
                                "piper_resume_refused")
        feedback.require()
        deadline = time.monotonic() + timeout_s
        while True:
            sent = True
            robot.enable()
            window = min(deadline, time.monotonic() + _ENABLE_RETRY_S)
            if wait_for(lambda: all(enable_status(robot) or [False]), window):
                break
            if time.monotonic() >= deadline:
                return _outcome(SIDE_EFFECT_COMMITTED,
                                "the arm accepted the enable command but not every motor reports "
                                "enabled; check the emergency-stop button and the power switch",
                                "piper_enable_refused")
        motion.seed_joints(joints)
        feedback.require()
        motion.enter_can(robot)
        feedback.require()
    except TransportError as exc:
        code = exc.failure_code
        return _outcome(SIDE_EFFECT_COMMITTED if sent else SIDE_EFFECT_NONE, str(exc),
                        code if code == "piper_unqualified_feedback" else "piper_mode_switch_failed")
    except Exception as exc:  # noqa: BLE001 - the reset/enable/mode frame may have landed
        return _outcome(SIDE_EFFECT_UNKNOWN, f"enable sequence failed: {exc}", "piper_send_failed")
    return WriteOutcome(ok=True, side_effect_state=SIDE_EFFECT_COMMITTED)


def _latched(robot: Any) -> bool:
    return as_int(attr(status_feedback(robot), "arm_status")) == ARM_STATUS_EMERGENCY_STOP


def _outcome(state: str, error: str, failure_code: str) -> WriteOutcome:
    return WriteOutcome(ok=False, side_effect_state=state, error=error, failure_code=failure_code)


__all__ = [
    "ARM_STATUS_EMERGENCY_STOP",
    "GRIPPER_EFFECTOR",
    "LinkSpec",
    "PiperLink",
    "energize",
    "link_state",
    "open_link",
    "wait_for",
]
