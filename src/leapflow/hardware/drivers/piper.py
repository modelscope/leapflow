# Copyright (c) Alibaba, Inc. and its affiliates.
"""AgileX Piper 6-DOF arm transport for node-local LeapFlow hardware profiles.

The ``piper_sdk`` package is imported only when a declared Piper arm is opened,
so host-only installations stay dependency-free and the CAN-owning node (the
machine with the ``can0``-style interface at 1 Mbit/s) owns the vendor SDK --
the same node-local placement as the RealSense driver beside this module.

SDK facts this driver is built on (agilexrobotics/piper_sdk, interface V2 doc):

- Unit scale is a uniform 1e-3: joint angles in 0.001 deg, end-effector
  position in 0.001 mm and orientation in 0.001 deg, gripper stroke in
  0.001 mm, gripper effort in 0.001 N*m. Declared channels therefore use the
  SDK-native units (deg / mm / N_m) and conversion is an exact factor of 1000.
- Two motion spaces share one arm: ``joint.*`` targets ride ``JointCtrl``
  (MOVE J) and ``ee.*`` pose targets ride ``EndPoseCtrl`` (MOVE L or MOVE P,
  the firmware solves IK). Mode tracking, hold vectors, and the refusal of a
  batch that mixes both spaces live in ``_piper_motion``.
- ``JointCtrl``/``EndPoseCtrl`` each command six values as THREE CAN frames
  inside one vendor call. Per-frame outcomes are not observable through that
  API, so the honest ``WriteOutcome`` mapping is: no exception -> every frame
  was accepted by the socket -> COMMITTED; an exception from the send itself
  -> the frames may have landed partially -> UNKNOWN (which blocks replay
  exactly like COMMITTED). A refusal decided before any send (unknown channel,
  value outside the declared envelope, invalid value, mixed motion spaces) is
  the only case that provably reaches nothing.
- ``EnableArm(motor_num, enable_flag=0x02)`` energizes motors without motion and
  is reachable ONLY through the governed ``arm.enable`` write, which first
  resumes a latched emergency stop (``EmergencyStop(0x02)``) under the same
  approval. Bring-up never energizes anything: opens also happen for read-only
  observation, and a physical action must never ride a read path.
  ``DisableArm`` and ``ResetPiper`` release torque and drop the arm; neither is
  ever called by ``halt`` or by recovery: ``halt`` maps to the vendor hardware
  emergency stop ``EmergencyStop(0x01)`` (CAN 0x150).
- ``ModeCtrl`` carries both the control mode and the move mode. The Piper X
  firmware silently drops a CAN control-mode selection issued from standby, so
  ``transport.config.mode_switch`` selects the strategy: ``direct`` (one frame,
  the vendor default) or ``standby_then_can`` (the handshake the Piper X manual
  prescribes, verified against ``ctrl_mode`` feedback). The governed enable
  path re-asserts the mode before energizing for the same reason, and reports
  ``piper_mode_switch_failed`` rather than a success no motor followed.
- ``GetArmJointMsgs``/``GetArmEndPoseMsgs``/``GetArmGripperMsgs``/``GetArmStatus``
  return cached feedback from the SDK's own receive thread, so reads never
  touch the bus and a stalled thread is visible through ``isOk()``/
  ``GetCanFps()`` rather than as a plausible stale number.
"""

from __future__ import annotations

import asyncio
import importlib
import time
from typing import Any, Mapping

from leapflow.hardware.context import HardwareContext, Quality
from leapflow.hardware.drivers._piper_motion import (
    MODE_SWITCHES,
    MODE_SWITCH_DIRECT,
    POSE_MOVE_MODES,
    SCALE,
    SPACE_CHANNELS,
    SPACE_JOINT,
    SPACE_POSE,
    PiperMotion,
    as_int,
    attr,
    ctrl_mode_name,
    enable_flag,
    enable_status,
    firmware_version,
    interface_version,
    motion_space,
    motion_spaces,
    numeric,
    parse_joints,
    parse_pose,
    status_feedback,
)
from leapflow.hardware.transport import (
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_UNKNOWN,
    BatchReading,
    BatchWriteOutcome,
    Reading,
    TransportError,
    TransportStatus,
    WriteOutcome,
    worst_side_effect_state,
)

_SDK_MODULE = "piper_sdk"
_INTERFACE_CLASSES: dict[str, str] = {
    "legacy": "C_PiperInterface",
    "v2": "C_PiperInterface_V2",
    "v3": "C_PiperInterface_V3",
}

_GRIPPER_POSITION = "gripper.position"
_GRIPPER_EFFORT = "gripper.effort"
_ARM_ENABLE = "arm.enable"
_ARM_ENABLED = "arm.enabled"
_ARM_STATUS = "arm.status"
_ARM_MOTION = "arm.motion"
_ARM_CTRL_MODE = "arm.ctrl_mode"
_ARM_SPEED = "arm.speed_pct"
_ARM_POSE_PATH = "arm.pose_path"
_SETTINGS = (_ARM_SPEED, _ARM_POSE_PATH)  # staged for the next ModeCtrl; no frame now
_GRIPPER_CHANNELS = (_GRIPPER_POSITION, _GRIPPER_EFFORT)
_STATE_CHANNELS = (
    _ARM_ENABLE,
    _ARM_ENABLED,
    _ARM_STATUS,
    _ARM_MOTION,
    _ARM_CTRL_MODE,
    *_SETTINGS,
)


def _refused(error: str, failure_code: str) -> WriteOutcome:
    """A refusal decided before any send: nothing reached the arm."""
    return WriteOutcome(
        ok=False, side_effect_state=SIDE_EFFECT_NONE, error=error, failure_code=failure_code
    )


_NOT_WRITABLE = _refused("channel is not writable by the piper transport", "not_writable")
_NOT_OPEN = _refused("Piper transport is not open", "transport_not_open")

_STALE_PLACEHOLDERS: dict[str, Any] = {
    _ARM_ENABLED: False,
    _ARM_MOTION: False,
    _ARM_STATUS: {},
    _ARM_CTRL_MODE: "",
}
"""Value a STALE batch reading carries when feedback is unavailable (default 0.0)."""

_VENDOR_CALL = {SPACE_JOINT: "JointCtrl", SPACE_POSE: "EndPoseCtrl"}

_ARM_STATUS_EMERGENCY_STOP = 0x01
_MOTION_STATUS_ARRIVED = 0x00
_GRIPPER_CODE_ENABLE = 0x01
_ENABLE_FLAG_ON = 0x02
_ENABLE_FLAG_OFF = 0x01
_ESTOP_RESUME = 0x02
_ALL_MOTORS = 7

_WARMUP_POLL_S = 0.05
_ENABLE_POLL_S = 0.02


def _load_sdk() -> Any:
    """Import ``piper_sdk`` lazily; the host never needs it installed."""
    try:
        return importlib.import_module(_SDK_MODULE)
    except ImportError as exc:
        raise TransportError(
            "piper_sdk is not installed; install it in the node environment that owns the CAN interface",
            failure_code="piper_sdk_missing",
        ) from exc


def _status_value(feedback: Any) -> dict[str, Any]:
    return {key: attr(feedback, key) for key in ("ctrl_mode", "arm_status", "motion_status")}


class PiperArmTransport:
    """Drive one Piper arm through the node-local CAN interface.

    Brings the arm up with the vendor's canonical sequence (``ConnectPort`` ->
    warm-up on the receive thread -> ``ModeCtrl`` CAN-command/MOVE-J -> optional
    crash protection) without energizing anything; motors are energized only
    through the governed ``arm.enable`` write. It then exposes the declared
    channels: joints, end-effector pose, gripper stroke/effort, the guarded
    enable, speed and pose path, and readiness/status. Writes are bounded target
    updates the firmware interpolates; high-rate sessions ride ``BatchTransport``.
    """

    kind = "piper"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = dict(config or {})
        self._can_name = str(config.get("can_name") or "can0").strip()
        self._judge_flag = bool(config.get("judge_flag", False))
        self._can_auto_init = bool(config.get("can_auto_init", True))
        self._dh_is_offset = 1 if bool(config.get("dh_is_offset", True)) else 0
        self._interface = str(config.get("sdk_interface") or "v2").strip().lower()
        if self._interface not in _INTERFACE_CLASSES:
            raise TransportError(
                f"piper transport sdk_interface must be one of {sorted(_INTERFACE_CLASSES)}, "
                f"got {self._interface!r}",
                failure_code="piper_invalid_config",
            )
        pose_mode = str(config.get("pose_move_mode") or "linear").strip().lower()
        if pose_mode not in POSE_MOVE_MODES:
            raise TransportError(
                f"piper transport pose_move_mode must be one of {sorted(POSE_MOVE_MODES)}, "
                f"got {pose_mode!r}",
                failure_code="piper_invalid_config",
            )
        speed_pct = min(100, max(1, int(config.get("default_speed_pct") or 20)))
        mode_switch = str(config.get("mode_switch") or MODE_SWITCH_DIRECT).strip().lower()
        if mode_switch not in MODE_SWITCHES:
            raise TransportError(
                f"piper transport mode_switch must be one of {sorted(MODE_SWITCHES)}, "
                f"got {mode_switch!r}",
                failure_code="piper_invalid_config",
            )
        self._mode_switch = mode_switch
        self._mode_timeout_s = max(0.1, float(config.get("mode_switch_timeout_s") or 0.6))
        self._enable_timeout_s = max(0.1, float(config.get("enable_timeout_s") or 1.0))
        self._warmup_timeout_s = max(0.1, float(config.get("warmup_timeout_s") or 3.0))
        self._stale_after_s = max(0.0, float(config.get("stale_after_s") or 1.0))
        self._default_effort_raw = int(
            round(float(config.get("default_effort_n_m") or 0.5) * SCALE)
        )
        levels = config.get("crash_protection_levels")
        self._crash_levels: tuple[int, ...] = ()
        if levels is not None:
            parsed = tuple(int(level) for level in levels)
            if len(parsed) != 6 or any(level < 0 or level > 8 for level in parsed):
                raise TransportError(
                    "piper crash_protection_levels must be six integers in [0, 8]",
                    failure_code="piper_invalid_config",
                )
            self._crash_levels = parsed

        self._sdk: Any = None
        self._interface_cls: Any = None
        self._arm: Any = None
        self._context: HardwareContext | None = None
        self._connected = False
        self._sequence = 0
        # Mode tracking and both motion spaces' hold vectors. Seeded from joint
        # feedback at open so the first single-joint write holds the other five
        # joints at their current positions instead of jumping to zero.
        self._motion = PiperMotion(
            speed_pct=speed_pct,
            pose_move_mode=POSE_MOVE_MODES[pose_mode],
            mode_switch=mode_switch,
            mode_timeout_s=self._mode_timeout_s,
        )
        self._commanded_stroke_raw: int | None = None
        self._commanded_effort_raw = self._default_effort_raw
        self._last_feedback_mono = 0.0
        # Last seen vendor feedback time stamp, per stream. Freshness advances
        # only when the receive thread produces a NEW frame, never merely
        # because a cached getter returned a parseable value again.
        self._joint_stamp: float | None = None
        self._pose_stamp: float | None = None
        self._gripper_stamp: float | None = None
        self._firmware = ""
        self._interface_version = ""

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def open(self, context: HardwareContext) -> TransportStatus:
        """Run the canonical bring-up sequence once; idempotent afterwards."""
        if self._connected:
            return await self.probe()
        self._context = context
        self._sdk = _load_sdk()
        self._interface_cls = getattr(self._sdk, _INTERFACE_CLASSES[self._interface], None)
        if self._interface_cls is None:
            raise TransportError(
                f"installed piper_sdk has no class {_INTERFACE_CLASSES[self._interface]!r}; "
                "the SDK is older than the declared sdk_interface",
                failure_code="piper_interface_unavailable",
            )
        try:
            await asyncio.to_thread(self._bring_up)
        except TransportError:
            self._connected = False
            raise
        except Exception as exc:
            self._connected = False
            raise TransportError(
                f"failed to open Piper arm on {self._can_name!r}: {exc}",
                failure_code="piper_open_failed",
            ) from exc
        self._connected = True
        return await self.probe()

    def _bring_up(self) -> None:
        """Blocking bring-up: connect, warm up, select mode. Never leaks a link."""
        arm: Any = None
        try:
            arm = self._interface_cls(
                self._can_name,
                judge_flag=self._judge_flag,
                can_auto_init=self._can_auto_init,
                dh_is_offset=self._dh_is_offset,
                # Defense in depth: the SDK clamps both feedback and command frames
                # to the firmware limits, beside the declared envelopes governance
                # already enforces.
                start_sdk_joint_limit=True,
                start_sdk_gripper_limit=True,
            )
            arm.ConnectPort()
            deadline = time.monotonic() + self._warmup_timeout_s
            while True:
                joints = self._poll_joints(arm)
                fps = float(arm.GetCanFps() or 0.0)
                if arm.get_connect_status() and arm.isOk() and fps > 0.0 and joints is not None:
                    break
                if time.monotonic() >= deadline:
                    raise TransportError(
                        f"Piper arm on {self._can_name!r} produced no valid feedback within "
                        f"{self._warmup_timeout_s:.1f}s; the CAN link or the receive thread is down",
                        failure_code="piper_warmup_timeout",
                    )
                time.sleep(_WARMUP_POLL_S)
            self._arm = arm
            self._firmware = firmware_version(arm)
            self._interface_version = interface_version(arm)
            self._motion.select_joint_mode(arm, joints)
            if self._crash_levels:
                arm.CrashProtectionConfig(*self._crash_levels)
        except BaseException:
            # A half-open CAN link must not outlive a failed open: disconnect
            # best-effort, clear published state, and surface the real failure.
            self._arm = None
            if arm is not None:
                try:
                    arm.DisconnectPort(0.1)
                except Exception:  # noqa: BLE001 - cleanup must not mask the failure
                    pass
            raise

    async def close(self) -> TransportStatus:
        """Disconnect from the CAN port; never powers the arm down."""
        arm, self._arm = self._arm, None
        self._connected = False
        if arm is not None:
            try:
                await asyncio.to_thread(arm.DisconnectPort, 0.1)
            except Exception:  # noqa: BLE001 - close is idempotent even on a dead bus
                pass
        return TransportStatus(
            connected=False, halt_supported=True, detail="Piper transport closed"
        )

    async def probe(self) -> TransportStatus:
        """Report link health and the arm state cached by the SDK receive thread."""
        if self._arm is None:
            return TransportStatus(
                connected=False,
                halt_supported=True,
                detail="Piper transport is closed",
                metadata={"can_name": self._can_name, "sdk_interface": self._interface},
            )
        state = await asyncio.to_thread(self._probe_state)
        connected = self._connected and state["connected"] and state["healthy"]
        detail = "Piper arm link healthy" if connected else "Piper arm link degraded or closed"
        return TransportStatus(
            connected=connected,
            halt_supported=True,
            detail=detail,
            latency_ms=round(1000.0 / state["fps"], 3) if state["fps"] > 0 else 0.0,
            metadata={
                "can_name": self._can_name,
                "sdk_interface": self._interface,
                "interface_version": self._interface_version,
                "firmware_version": self._firmware,
                "can_fps": state["fps"],
                "ctrl_mode": state["ctrl_mode"],
                "ctrl_mode_name": ctrl_mode_name(state["ctrl_mode"]),
                "move_mode": self._motion.mode,
                "arm_status": state["arm_status"],
                "motion_status": state["motion_status"],
                "enabled_motors": state["enabled"],
                "crash_protection_levels": list(self._crash_levels),
            },
        )

    def _probe_state(self) -> dict[str, Any]:
        arm = self._arm
        assert arm is not None
        feedback = status_feedback(arm)
        status = _status_value(feedback) if feedback is not None else {}
        return {
            "connected": bool(arm.get_connect_status()),
            "healthy": bool(arm.isOk()),
            "fps": float(arm.GetCanFps() or 0.0),
            **{key: status.get(key) for key in ("ctrl_mode", "arm_status", "motion_status")},
            "enabled": enable_status(arm),
        }

    async def halt(self) -> TransportStatus:
        """Engage the vendor emergency stop. Lock-free, never raises.

        Deliberately NOT ``ResetPiper``/``DisableArm``: both drop power and let
        the arm fall. Resuming (``EmergencyStop(0x02)``) is a separate governed
        write on the ``arm.enable`` path, never part of a halt.
        """
        arm = self._arm
        if arm is None:
            return TransportStatus(
                connected=False,
                halt_supported=True,
                detail="Piper transport is closed; nothing to halt",
            )
        try:
            await asyncio.to_thread(arm.EmergencyStop, _ARM_STATUS_EMERGENCY_STOP)
        except Exception as exc:  # noqa: BLE001 - halt must never raise
            return TransportStatus(
                connected=self._connected,
                halt_supported=True,
                detail=f"Piper emergency stop command failed: {exc}",
            )
        return TransportStatus(
            connected=self._connected,
            halt_supported=True,
            detail=(
                "Piper emergency stop engaged (CAN 0x150/0x01); "
                "resume is a separate governed action"
            ),
        )

    # ── Reads ────────────────────────────────────────────────────────────────

    async def read(self, channel_id: str) -> Reading:
        """Read one channel from the SDK's cached feedback frames."""
        self._require_open(channel_id)
        if not self._known_channel(channel_id) or channel_id == _ARM_ENABLE:
            raise TransportError(f"unknown channel {channel_id!r}", failure_code="unknown_channel")
        values = await asyncio.to_thread(self._poll_values, (channel_id,))
        value = values.get(channel_id)
        if value is None:
            raise TransportError(
                f"Piper feedback for {channel_id!r} is unavailable",
                failure_code="piper_no_feedback",
            )
        return self._reading(channel_id, value)

    async def read_batch(self, channel_ids: tuple[str, ...]) -> BatchReading:
        """Read every requested channel from one coordinated feedback poll.

        Unavailable feedback reads as STALE rather than failing the batch, so
        one dead stream never blinds a control loop to the others.
        """
        self._require_open(channel_ids)
        unknown = [cid for cid in channel_ids if not self._known_channel(cid)]
        if unknown:
            raise TransportError(f"unknown channels {unknown}", failure_code="unknown_channel")
        observed_at = time.time()
        monotonic_at = time.monotonic()
        values = await asyncio.to_thread(self._poll_values, channel_ids)
        readings = tuple(
            self._reading(cid, values[cid], observed_at, monotonic_at)
            if values.get(cid) is not None
            else self._reading(
                cid,
                _STALE_PLACEHOLDERS.get(cid, 0.0),
                observed_at,
                monotonic_at,
                Quality.STALE.value,
            )
            for cid in channel_ids
        )
        return BatchReading(
            device_id=self._device_id,
            readings=readings,
            observed_at=observed_at,
            monotonic_at=monotonic_at,
        )

    def _poll_values(self, channel_ids: tuple[str, ...]) -> dict[str, Any]:
        """Blocking: poll only the needed streams; ``None`` marks unavailable feedback."""
        arm = self._arm
        assert arm is not None
        wanted = set(channel_ids)
        values: dict[str, Any] = {}
        for space, channels in SPACE_CHANNELS.items():
            if wanted.isdisjoint(channels):
                continue
            raw = self._poll_space(arm, space)
            for cid, index in channels.items():
                values[cid] = raw[index] / SCALE if raw is not None else None
        if not wanted.isdisjoint(_GRIPPER_CHANNELS):
            stroke, effort = self._poll_gripper(arm)
            values[_GRIPPER_POSITION] = stroke / SCALE if stroke is not None else None
            values[_GRIPPER_EFFORT] = (
                (effort or 0) / SCALE if stroke is not None and effort is not None else None
            )
        if _ARM_SPEED in wanted:
            values[_ARM_SPEED] = float(self._motion.speed_pct)
        if _ARM_POSE_PATH in wanted:
            values[_ARM_POSE_PATH] = self._motion.pose_path
        if _ARM_ENABLED in wanted:
            enabled = enable_status(arm)
            values[_ARM_ENABLED] = all(enabled) if enabled is not None else None
        if not wanted.isdisjoint((_ARM_STATUS, _ARM_MOTION, _ARM_CTRL_MODE)):
            feedback = status_feedback(arm)
            motion = attr(feedback, "motion_status") if feedback is not None else None
            values[_ARM_MOTION] = motion == _MOTION_STATUS_ARRIVED if motion is not None else None
            values[_ARM_STATUS] = _status_value(feedback) if feedback is not None else None
            reported = attr(feedback, "ctrl_mode") if feedback is not None else None
            values[_ARM_CTRL_MODE] = ctrl_mode_name(reported)
        return values

    # ── Writes ───────────────────────────────────────────────────────────────

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        """Write one channel, reporting whether the effect may have landed."""
        if channel_id == _ARM_ENABLE:
            return await self._write_enable(value)
        if channel_id in _SETTINGS:
            return self._write_setting(channel_id, value)
        space = motion_space(channel_id)
        if space:
            return (await self._write_motion(space, [(channel_id, value)]))[0]
        if channel_id == _GRIPPER_POSITION:
            return await self._write_gripper(stroke=value)
        if channel_id == _GRIPPER_EFFORT:
            return await self._write_gripper(effort=value)
        return _NOT_WRITABLE

    async def write_batch(self, commands: tuple[tuple[str, Any], ...]) -> BatchWriteOutcome:
        """Write channels in one bus pass: enable/settings, then motion, then gripper.

        Commands of one motion space merge into a single six-value vendor
        vector. A batch naming both joint and pose targets is refused whole
        before any send: the arm obeys one move mode, and executing either half
        would move it somewhere the caller never asked for.
        """
        spaces = motion_spaces(cid for cid, _ in commands)
        if len(spaces) > 1:
            refusal = _refused(
                "a batch may command joint targets or end-effector pose targets, not both",
                "mixed_motion_space",
            )
            return BatchWriteOutcome(
                ok=False,
                outcomes=tuple(refusal for _ in commands),
                side_effect_state=SIDE_EFFECT_NONE,
            )
        outcomes: list[WriteOutcome | None] = [None] * len(commands)
        for i, (cid, value) in enumerate(commands):
            if cid == _ARM_ENABLE:
                outcomes[i] = await self._write_enable(value)
            elif cid in _SETTINGS:
                outcomes[i] = self._write_setting(cid, value)
        for space in spaces:
            indexes = [i for i, (cid, _) in enumerate(commands) if motion_space(cid) == space]
            merged = await self._write_motion(space, [commands[i] for i in indexes])
            for i, outcome in zip(indexes, merged):
                outcomes[i] = outcome
        for i, (cid, value) in enumerate(commands):
            if cid == _GRIPPER_POSITION:
                outcomes[i] = await self._write_gripper(stroke=value)
            elif cid == _GRIPPER_EFFORT:
                outcomes[i] = await self._write_gripper(effort=value)
        final = [outcome if outcome is not None else _NOT_WRITABLE for outcome in outcomes]
        return BatchWriteOutcome(
            ok=all(o.ok for o in final),
            outcomes=tuple(final),
            side_effect_state=worst_side_effect_state(o.side_effect_state for o in final),
        )

    async def _write_enable(self, value: Any) -> WriteOutcome:
        """Guarded enable/disable: enter CAN mode, resume, energize -- or release torque."""
        flag = enable_flag(value)
        if flag is None:
            return _refused(f"arm.enable expects 0 or 1, got {value!r}", "invalid_value")
        arm = self._arm
        if arm is None or not self._connected:
            return _NOT_OPEN
        if flag:
            return await self._energize(arm)
        try:
            await asyncio.to_thread(arm.DisableArm, _ALL_MOTORS, _ENABLE_FLAG_OFF)
        except Exception as exc:  # noqa: BLE001 - the 0x471 frame may have landed
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_UNKNOWN,
                error=f"Disable command failed: {exc}",
                failure_code="piper_send_failed",
            )
        return WriteOutcome(ok=True, side_effect_state=SIDE_EFFECT_COMMITTED)

    async def _energize(self, arm: Any) -> WriteOutcome:
        """Enter CAN mode, resume a latched stop, energize, and verify the result.

        The Piper X firmware accepts an enable sent while the arm is in standby
        and changes nothing, so the mode selection runs first and the motor
        enable flags are confirmed from feedback before this write is called
        successful.
        """
        joints = await asyncio.to_thread(self._poll_joints, arm)
        if joints is None:
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error="no joint feedback to seed the hold vector before enabling",
                failure_code="piper_no_feedback",
            )
        try:
            await asyncio.to_thread(self._motion.select_joint_mode, arm, joints)
        except TransportError as exc:
            # The handshake sends frames before it can refuse: the control mode
            # may already have changed even though the arm cannot be driven.
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_COMMITTED,
                error=str(exc),
                failure_code="piper_mode_switch_failed",
            )
        except Exception as exc:  # noqa: BLE001 - a mode frame may have landed
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_UNKNOWN,
                error=f"Mode selection failed: {exc}",
                failure_code="piper_send_failed",
            )
        try:
            # A resume is NOT a no-op when nothing is latched: the firmware
            # answers 0x150/0x02 by dropping the control mode back to standby,
            # and an EnableArm issued there is silently ignored. So resume only
            # a latched stop, and walk back into CAN mode before energizing.
            if await asyncio.to_thread(self._estop_latched, arm):
                await asyncio.to_thread(arm.EmergencyStop, _ESTOP_RESUME)
                await asyncio.to_thread(self._motion.enter_can, arm)
                await asyncio.to_thread(self._motion.select_joint_mode, arm, joints)
            await asyncio.to_thread(arm.EnableArm, _ALL_MOTORS, _ENABLE_FLAG_ON)
        except TransportError as exc:
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_COMMITTED,
                error=str(exc),
                failure_code="piper_mode_switch_failed",
            )
        except Exception as exc:  # noqa: BLE001 - the 0x150/0x471 frame may have landed
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_UNKNOWN,
                error=f"Enable command failed: {exc}",
                failure_code="piper_send_failed",
            )
        if await asyncio.to_thread(self._await_enabled, arm) is False:
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_COMMITTED,
                error=(
                    "the arm accepted the enable command but no motor reports enabled; "
                    "check the emergency-stop button, the power switch, and that the "
                    "arm reports CAN mode"
                ),
                failure_code="piper_enable_refused",
            )
        return WriteOutcome(ok=True, side_effect_state=SIDE_EFFECT_COMMITTED)

    def _estop_latched(self, arm: Any) -> bool:
        """True when the arm reports a latched emergency stop in its status frame."""
        feedback = status_feedback(arm)
        if feedback is None:
            return False
        return as_int(attr(feedback, "arm_status")) == _ARM_STATUS_EMERGENCY_STOP

    def _await_enabled(self, arm: Any) -> bool | None:
        """Poll the motor enable flags; None when the firmware reports nothing."""
        deadline = time.monotonic() + self._enable_timeout_s
        while True:
            enabled = enable_status(arm)
            if enabled is None:
                return None
            if enabled and all(enabled):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(_ENABLE_POLL_S)

    def _write_setting(self, channel_id: str, value: Any) -> WriteOutcome:
        """Stage the speed rate or pose path the next motion applies; nothing is sent now."""
        if channel_id == _ARM_POSE_PATH:
            if not isinstance(value, str) or value not in POSE_MOVE_MODES:
                expected = ", ".join(sorted(POSE_MOVE_MODES))
                return _refused(f"arm.pose_path expects one of {expected}", "invalid_value")
            applied: Any = self._motion.set_pose_path(value)
        else:
            refusal = self._refusal(_ARM_SPEED, value)
            if refusal is not None:
                return refusal
            applied = float(self._motion.set_speed(int(round(numeric(value) or 1))))
        readback = self._reading(channel_id, applied)
        return WriteOutcome(ok=True, side_effect_state=SIDE_EFFECT_COMMITTED, readback=readback)

    async def _write_motion(
        self, space: str, commands: list[tuple[str, Any]]
    ) -> list[WriteOutcome]:
        """Send one hold vector in ``space`` covering every commanded channel.

        Uncommanded values are held (see ``_piper_motion``), so a single-axis
        write never lets the other five jump; one vendor call both sends the
        vector and defines the side-effect verdict for every channel in it. A
        refusal on any commanded channel refuses the whole atomic vector. The
        send runs on a worker thread so a stalled CAN write never blocks the
        event loop or delays ``halt``.
        """
        for channel_id, value in commands:
            refusal = self._refusal(channel_id, value)
            if refusal is not None:
                return [refusal] * len(commands)
        arm = self._arm
        if arm is None or not self._connected:
            return [_NOT_OPEN] * len(commands)
        channels = SPACE_CHANNELS[space]
        targets = [(channels[cid], float(numeric(value) or 0.0)) for cid, value in commands]
        try:
            snapshot = await asyncio.to_thread(self._send_motion, arm, space, targets)
        except Exception as exc:  # noqa: BLE001 - frames may have partially landed
            return [
                WriteOutcome(
                    ok=False,
                    side_effect_state=SIDE_EFFECT_UNKNOWN,
                    error=f"{_VENDOR_CALL[space]} send failed: {exc}",
                    failure_code="piper_send_failed",
                )
            ] * len(commands)
        if snapshot is None:
            return [
                WriteOutcome(
                    ok=False,
                    side_effect_state=SIDE_EFFECT_NONE,
                    error=f"no {space} feedback to hold the uncommanded axes at; "
                    "open the transport and wait for fresh feedback first",
                    failure_code="piper_no_feedback",
                )
            ] * len(commands)
        after = await asyncio.to_thread(self._poll_space, arm, space)
        outcomes: list[WriteOutcome] = []
        for cid, _value in commands:
            readback: Reading | None = None
            if after is not None:
                readback = self._reading(
                    cid, after[channels[cid]] / SCALE, monotonic_at=self._last_feedback_mono
                )
            outcomes.append(
                WriteOutcome(ok=True, side_effect_state=SIDE_EFFECT_COMMITTED, readback=readback)
            )
        return outcomes

    def _send_motion(
        self, arm: Any, space: str, targets: list[tuple[int, float]]
    ) -> list[int] | None:
        """Blocking: poll hold feedback, then send through the serialized state."""
        return self._motion.send(arm, space, self._poll_space(arm, space), targets)

    async def _write_gripper(self, stroke: Any = None, effort: Any = None) -> WriteOutcome:
        """Command the gripper stroke (0.001 mm) or effort (0.001 N*m)."""
        arm = self._arm
        if arm is None or not self._connected:
            return _NOT_OPEN
        channel_id = _GRIPPER_POSITION if stroke is not None else _GRIPPER_EFFORT
        value = stroke if stroke is not None else effort
        refusal = self._refusal(channel_id, value)
        if refusal is not None:
            return refusal
        number = numeric(value)
        assert number is not None  # _refusal rejects non-numerics above
        staged_stroke: int | None
        staged_effort = self._commanded_effort_raw
        if stroke is not None:
            staged_stroke = int(round(number * SCALE))
        else:
            staged_effort = int(round(number * SCALE))
            staged_stroke = self._commanded_stroke_raw
            if staged_stroke is None:
                stroke_fb, _ = await asyncio.to_thread(self._poll_gripper, arm)
                if stroke_fb is None:
                    return WriteOutcome(
                        ok=False,
                        side_effect_state=SIDE_EFFECT_NONE,
                        error="no gripper stroke command or feedback to pair with the effort",
                        failure_code="piper_no_feedback",
                    )
                staged_stroke = stroke_fb
        try:
            # The gripper bypasses the motion path, so the mode guard is this
            # write's own: in standby the firmware drops the frame while the
            # outcome would still read committed.
            await asyncio.to_thread(self._motion.ensure_can, arm)
        except TransportError as exc:
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_COMMITTED,
                error=str(exc),
                failure_code="piper_mode_switch_failed",
            )
        except Exception as exc:  # noqa: BLE001 - a mode frame may have landed
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_UNKNOWN,
                error=f"Mode selection failed: {exc}",
                failure_code="piper_send_failed",
            )
        try:
            await asyncio.to_thread(
                arm.GripperCtrl, staged_stroke, staged_effort, _GRIPPER_CODE_ENABLE, 0x00
            )
        except Exception as exc:  # noqa: BLE001 - the 0x159 frame may have landed
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_UNKNOWN,
                error=f"GripperCtrl send failed: {exc}",
                failure_code="piper_send_failed",
            )
        # Caches advance only after the frame was accepted: a refused or
        # uncertain send must not leak into the next write's stroke/effort pair.
        self._commanded_stroke_raw = staged_stroke
        self._commanded_effort_raw = staged_effort
        stroke_fb, effort_fb = await asyncio.to_thread(self._poll_gripper, arm)
        readback: Reading | None = None
        if stroke_fb is not None:
            raw = stroke_fb if stroke is not None else effort_fb
            readback = self._reading(
                channel_id, (raw or 0) / SCALE, monotonic_at=self._last_feedback_mono
            )
        return WriteOutcome(ok=True, side_effect_state=SIDE_EFFECT_COMMITTED, readback=readback)

    def _refusal(self, channel_id: str, value: Any) -> WriteOutcome | None:
        """Driver-level defense: refuse undeclared values before any send."""
        number = numeric(value)
        if number is None:
            return _refused(f"invalid value for {channel_id!r}: {value!r}", "invalid_value")
        channel = self._context.channel(channel_id) if self._context is not None else None
        envelope = channel.envelope if channel is not None else None
        if envelope is not None and envelope.declared and not envelope.contains(number):
            return _refused(
                f"{channel_id}={number} is outside the declared envelope "
                f"[{envelope.min_value}, {envelope.max_value}]",
                "value_out_of_range",
            )
        return None

    # ── Feedback polls (blocking; always run via to_thread) ──────────────────

    def _observe_stamp(self, msg: Any, last: float | None) -> float | None:
        """Advance the freshness clock only when vendor feedback advances.

        The vendor getters return whatever the receive thread last cached, so a
        successful parse does not mean a new sample. The frame's own
        ``time_stamp`` detects advancement; builds that omit it fall back to
        treating every successful parse as fresh.
        """
        raw = attr(msg, "time_stamp")
        stamp = raw if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
        if stamp is None or stamp != last:
            self._last_feedback_mono = time.monotonic()
            return None if stamp is None else float(stamp)
        return last

    def _poll_space(self, arm: Any, space: str) -> list[int] | None:
        return self._poll_joints(arm) if space == SPACE_JOINT else self._poll_pose(arm)

    def _poll_joints(self, arm: Any) -> list[int] | None:
        """Return the six raw joint angles, or None when feedback is unusable."""
        try:
            msg = arm.GetArmJointMsgs()
        except Exception:  # noqa: BLE001 - a dead receive thread must read as stale
            return None
        raw = parse_joints(msg)
        if raw is not None:
            self._joint_stamp = self._observe_stamp(msg, self._joint_stamp)
        return raw

    def _poll_pose(self, arm: Any) -> list[int] | None:
        """Return the raw end-effector pose (X, Y, Z, RX, RY, RZ), or None."""
        try:
            msg = arm.GetArmEndPoseMsgs()
        except Exception:  # noqa: BLE001 - a dead receive thread must read as stale
            return None
        raw = parse_pose(msg)
        if raw is not None:
            self._pose_stamp = self._observe_stamp(msg, self._pose_stamp)
        return raw

    def _poll_gripper(self, arm: Any) -> tuple[int | None, int | None]:
        try:
            msg = arm.GetArmGripperMsgs()
        except Exception:  # noqa: BLE001
            return (None, None)
        state = attr(msg, "gripper_state", msg)
        stroke = as_int(attr(state, "grippers_angle"))
        effort = as_int(attr(state, "grippers_effort"))
        if stroke is not None:
            self._gripper_stamp = self._observe_stamp(msg, self._gripper_stamp)
        return (stroke, effort)

    # ── Shared helpers ───────────────────────────────────────────────────────

    def _reading(
        self,
        channel_id: str,
        value: Any,
        observed_at: float | None = None,
        monotonic_at: float | None = None,
        quality: str | None = None,
    ) -> Reading:
        """Build a Reading with declared quantity/unit and staleness quality."""
        channel = self._context.channel(channel_id) if self._context is not None else None
        if quality is None:
            stale = (
                self._last_feedback_mono > 0.0
                and (time.monotonic() - self._last_feedback_mono) > self._stale_after_s
            )
            quality = Quality.STALE.value if stale else Quality.OK.value
        self._sequence += 1
        return Reading(
            device_id=self._device_id,
            channel_id=channel_id,
            value=value,
            quantity=channel.quantity if channel is not None else "",
            unit=channel.unit if channel is not None else "",
            observed_at=observed_at if observed_at is not None else time.time(),
            monotonic_at=monotonic_at if monotonic_at is not None else time.monotonic(),
            sequence=self._sequence,
            quality=quality,
        )

    def _known_channel(self, channel_id: str) -> bool:
        return (
            bool(motion_space(channel_id))
            or channel_id in _GRIPPER_CHANNELS
            or channel_id in _STATE_CHANNELS
        )

    def _require_open(self, channel_ids: tuple[str, ...] | str) -> None:
        label = channel_ids if isinstance(channel_ids, str) else list(channel_ids)
        if not self._connected or self._arm is None:
            raise TransportError(
                f"Piper transport for {label!r} is not open",
                failure_code="transport_not_open",
            )

    @property
    def _device_id(self) -> str:
        return self._context.device_id if self._context is not None else ""


def build_transport(config: Mapping[str, Any] | None = None) -> PiperArmTransport:
    """Entry-point factory for the optional Piper transport plugin."""
    return PiperArmTransport(config)


__all__ = ["PiperArmTransport", "build_transport"]
