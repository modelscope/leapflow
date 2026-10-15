# Copyright (c) Alibaba, Inc. and its affiliates.
"""AgileX Piper 6-DOF arm transport for node-local LeapFlow hardware profiles.

The ``piper_sdk`` package is imported only when a declared Piper arm is opened,
so host-only installations stay dependency-free and the CAN-owning node (the
machine with the ``can0``-style interface at 1 Mbit/s) owns the vendor SDK --
the same node-local placement as the RealSense driver beside this module.

SDK facts this driver is built on (agilexrobotics/piper_sdk, interface V2 doc):

- Unit scale is a uniform 1e-3: joint angles in 0.001 deg, gripper stroke in
  0.001 mm, gripper effort in 0.001 N*m. Declared channels therefore use the
  SDK-native units (deg / mm / N_m) and conversion is an exact factor of 1000.
- ``JointCtrl`` commands all six joints and is transmitted as THREE CAN frames
  (0x155/0x156/0x157) inside one vendor call. Per-frame outcomes are not
  observable through that API, so the honest ``WriteOutcome`` mapping is:
  no exception -> every frame was accepted by the socket -> COMMITTED;
  an exception from the send itself -> the frames may have landed partially
  -> UNKNOWN (which blocks replay exactly like COMMITTED). A refusal decided
  before any send (unknown channel, value outside the declared envelope,
  invalid value) is the only case that provably reaches nothing.
- ``EnableArm(motor_num, enable_flag=0x02)`` energizes motors without motion and
  is reachable ONLY through the governed ``arm.enable`` write, which first
  resumes a latched emergency stop (``EmergencyStop(0x02)``) under the same
  approval. Bring-up never energizes anything: opens also happen for read-only
  observation, and a physical action must never ride a read path.
  ``DisableArm`` and ``ResetPiper`` release torque and drop the arm; neither is
  ever called by ``halt`` or by recovery: ``halt`` maps to the vendor hardware
  emergency stop ``EmergencyStop(0x01)`` (CAN 0x150).
- ``GetArmJointMsgs``/``GetArmGripperMsgs``/``GetArmStatus`` return cached
  feedback from the SDK's own receive thread, so reads never touch the bus and
  a stalled thread is visible through ``isOk()``/``GetCanFps()`` rather than as
  a plausible stale number.
"""

from __future__ import annotations

import asyncio
import importlib
import threading
import time
from typing import Any, Mapping

from leapflow.hardware.context import HardwareContext, Quality
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

_JOINT_CHANNELS: dict[str, int] = {f"joint.j{i}.position": i for i in range(1, 7)}
_GRIPPER_POSITION = "gripper.position"
_GRIPPER_EFFORT = "gripper.effort"
_ARM_ENABLE = "arm.enable"
_ARM_ENABLED = "arm.enabled"
_ARM_STATUS = "arm.status"
_ARM_MOTION = "arm.motion"

_SCALE = 1000.0
"""SDK-native quantization: declared unit value * 1000 == raw integer."""

_CTRL_MODE_CAN = 0x01
_ARM_STATUS_EMERGENCY_STOP = 0x01
_MOTION_STATUS_ARRIVED = 0x00
_GRIPPER_CODE_ENABLE = 0x01
_ENABLE_FLAG_ON = 0x02
_ENABLE_FLAG_OFF = 0x01
_ESTOP_RESUME = 0x02
_ALL_MOTORS = 7

_WARMUP_POLL_S = 0.05


def _load_sdk() -> Any:
    """Import ``piper_sdk`` lazily; the host never needs it installed."""
    try:
        return importlib.import_module(_SDK_MODULE)
    except ImportError as exc:
        raise TransportError(
            "piper_sdk is not installed; install it in the node environment that owns the CAN interface",
            failure_code="piper_sdk_missing",
        ) from exc


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read one attribute from an SDK feedback struct, tolerating absent fields."""
    value = getattr(obj, name, default)
    return default if value is None else value


def _as_int(value: Any) -> int | None:
    """Coerce a raw feedback field to int; booleans and garbage are not angles."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _numeric(value: Any) -> float | None:
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


def _enable_flag(value: Any) -> int | None:
    """Normalize an enable/disable command to 1/0, or None when not a flag."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)) and float(value).is_integer() and int(value) in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip() in ("0", "1"):
        return int(value.strip())
    return None


def _firmware_version(arm: Any) -> str:
    """Return the firmware string; ``-0x4AF`` means the query has no answer yet."""
    try:
        raw = arm.GetPiperFirmwareVersion()
    except Exception:  # noqa: BLE001 - probe metadata must never break on SDK quirks
        return ""
    if not isinstance(raw, str) or raw.startswith("-0x"):
        return ""
    return raw


def _interface_version(arm: Any) -> str:
    try:
        raw = arm.GetCurrentInterfaceVersion()
    except Exception:  # noqa: BLE001
        return ""
    return str(raw) if raw is not None else ""


def _status_feedback(arm: Any) -> Any:
    """Return the arm-status feedback struct, or None when unavailable."""
    try:
        msg = arm.GetArmStatus()
    except Exception:  # noqa: BLE001
        return None
    feedback = _attr(msg, "arm_status", msg)
    if _attr(feedback, "arm_status") is None and _attr(feedback, "motion_status") is None:
        return None
    return feedback


def _enable_status(arm: Any) -> list[bool] | None:
    """Return the per-motor enable list, or None when unavailable."""
    try:
        raw = arm.GetArmEnableStatus()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(raw, (list, tuple)):
        return None
    return [bool(item) for item in raw]


class PiperArmTransport:
    """Drive one Piper arm through the node-local CAN interface.

    Brings the arm up with the vendor's canonical sequence (``ConnectPort`` ->
    warm-up on the receive thread -> ``ModeCtrl`` CAN-command/MOVE-J -> optional
    crash protection) without energizing anything; motors are energized only
    through the governed ``arm.enable`` write. It then exposes the declared
    channels: six joint positions, the gripper stroke/effort pair, the guarded
    enable, and the readiness/status reads. Writes are bounded target updates --
    the firmware interpolates -- so no host-side loop is needed for
    point-to-point motion; high-rate sessions ride ``BatchTransport`` and run
    through the node-local control bus like any other transport.
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
        self._speed_pct = min(100, max(1, int(config.get("default_speed_pct") or 20)))
        self._move_mode = int(config.get("move_mode") or 0x01)
        self._warmup_timeout_s = max(0.1, float(config.get("warmup_timeout_s") or 3.0))
        self._stale_after_s = max(0.0, float(config.get("stale_after_s") or 1.0))
        self._default_effort_raw = int(round(float(config.get("default_effort_n_m") or 0.5) * _SCALE))
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
        # Last commanded joint vector in SDK raw units. Seeded from feedback at
        # open so the first single-joint write holds the other five joints at
        # their current positions instead of jumping to zero.
        self._commanded_raw: list[int | None] = [None] * 6
        self._commanded_stroke_raw: int | None = None
        self._commanded_effort_raw = self._default_effort_raw
        self._last_feedback_mono = 0.0
        # Last seen vendor feedback time stamp, per stream. Freshness advances
        # only when the receive thread produces a NEW frame, never merely
        # because a cached getter returned a parseable value again.
        self._joint_stamp: float | None = None
        self._gripper_stamp: float | None = None
        self._send_lock = threading.Lock()
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
            self._firmware = _firmware_version(arm)
            self._interface_version = _interface_version(arm)
            arm.ModeCtrl(
                ctrl_mode=_CTRL_MODE_CAN,
                move_mode=self._move_mode,
                move_spd_rate_ctrl=self._speed_pct,
            )
            if self._crash_levels:
                arm.CrashProtectionConfig(*self._crash_levels)
            with self._send_lock:
                self._commanded_raw = list(joints)
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
        return TransportStatus(connected=False, halt_supported=True, detail="Piper transport closed")

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
                "arm_status": state["arm_status"],
                "motion_status": state["motion_status"],
                "enabled_motors": state["enabled"],
                "crash_protection_levels": list(self._crash_levels),
            },
        )

    def _probe_state(self) -> dict[str, Any]:
        arm = self._arm
        assert arm is not None
        feedback = _status_feedback(arm)
        return {
            "connected": bool(arm.get_connect_status()),
            "healthy": bool(arm.isOk()),
            "fps": float(arm.GetCanFps() or 0.0),
            "ctrl_mode": _attr(feedback, "ctrl_mode") if feedback is not None else None,
            "arm_status": _attr(feedback, "arm_status") if feedback is not None else None,
            "motion_status": _attr(feedback, "motion_status") if feedback is not None else None,
            "enabled": _enable_status(arm),
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
        if channel_id in _JOINT_CHANNELS:
            index = _JOINT_CHANNELS[channel_id]
            joints = await asyncio.to_thread(self._poll_joints, self._arm)
            if joints is None:
                raise TransportError(
                    f"Piper joint feedback for {channel_id!r} is unavailable",
                    failure_code="piper_no_feedback",
                )
            return self._reading(channel_id, joints[index - 1] / _SCALE)
        if channel_id in (_GRIPPER_POSITION, _GRIPPER_EFFORT):
            stroke_raw, effort_raw = await asyncio.to_thread(self._poll_gripper, self._arm)
            if stroke_raw is None:
                raise TransportError(
                    f"Piper gripper feedback for {channel_id!r} is unavailable",
                    failure_code="piper_no_feedback",
                )
            raw = stroke_raw if channel_id == _GRIPPER_POSITION else effort_raw
            return self._reading(channel_id, (raw or 0) / _SCALE)
        if channel_id == _ARM_ENABLED:
            enabled = await asyncio.to_thread(_enable_status, self._arm)
            if enabled is None:
                raise TransportError(
                    "Piper enable status is unavailable", failure_code="piper_no_feedback"
                )
            return self._reading(channel_id, all(enabled))
        if channel_id in (_ARM_STATUS, _ARM_MOTION):
            feedback = await asyncio.to_thread(_status_feedback, self._arm)
            if feedback is None:
                raise TransportError(
                    "Piper arm status is unavailable", failure_code="piper_no_feedback"
                )
            if channel_id == _ARM_MOTION:
                motion = _attr(feedback, "motion_status")
                value: Any = motion == _MOTION_STATUS_ARRIVED if motion is not None else False
            else:
                value = {
                    "ctrl_mode": _attr(feedback, "ctrl_mode"),
                    "arm_status": _attr(feedback, "arm_status"),
                    "motion_status": _attr(feedback, "motion_status"),
                }
            return self._reading(channel_id, value)
        raise TransportError(f"unknown channel {channel_id!r}", failure_code="unknown_channel")

    async def read_batch(self, channel_ids: tuple[str, ...]) -> BatchReading:
        """Read every requested channel from one coordinated feedback poll."""
        self._require_open(channel_ids)
        unknown = [cid for cid in channel_ids if not self._known_channel(cid)]
        if unknown:
            raise TransportError(f"unknown channels {unknown}", failure_code="unknown_channel")
        observed_at = time.time()
        monotonic_at = time.monotonic()
        needs = {
            "joints": any(cid in _JOINT_CHANNELS for cid in channel_ids),
            "gripper": any(cid in (_GRIPPER_POSITION, _GRIPPER_EFFORT) for cid in channel_ids),
            "status": any(cid in (_ARM_STATUS, _ARM_MOTION) for cid in channel_ids),
            "enabled": _ARM_ENABLED in channel_ids,
        }

        def _poll_all() -> tuple[Any, Any, Any]:
            assert self._arm is not None
            joints = self._poll_joints(self._arm) if needs["joints"] else None
            gripper = self._poll_gripper(self._arm) if needs["gripper"] else (None, None)
            feedback = (
                _status_feedback(self._arm) if (needs["status"] or needs["enabled"]) else None
            )
            return joints, gripper, feedback

        joints, gripper, feedback = await asyncio.to_thread(_poll_all)
        enabled = _enable_status(self._arm) if needs["enabled"] else None
        motion = _attr(feedback, "motion_status") if feedback is not None else None
        readings = [
            self._batch_reading(
                cid, joints, gripper, enabled, feedback, motion, observed_at, monotonic_at
            )
            for cid in channel_ids
        ]
        return BatchReading(
            device_id=self._device_id,
            readings=tuple(readings),
            observed_at=observed_at,
            monotonic_at=monotonic_at,
        )

    def _batch_reading(
        self,
        channel_id: str,
        joints: list[int] | None,
        gripper: tuple[int | None, int | None],
        enabled: list[bool] | None,
        feedback: Any,
        motion: int | None,
        observed_at: float,
        monotonic_at: float,
    ) -> Reading:
        """Build one batch reading; unavailable feedback reads as STALE, not fatal."""
        if channel_id in _JOINT_CHANNELS:
            index = _JOINT_CHANNELS[channel_id]
            if joints is None:
                return self._reading(channel_id, 0.0, observed_at, monotonic_at, Quality.STALE.value)
            return self._reading(channel_id, joints[index - 1] / _SCALE, observed_at, monotonic_at)
        if channel_id == _GRIPPER_POSITION:
            raw = gripper[0]
            if raw is None:
                return self._reading(channel_id, 0.0, observed_at, monotonic_at, Quality.STALE.value)
            return self._reading(channel_id, raw / _SCALE, observed_at, monotonic_at)
        if channel_id == _GRIPPER_EFFORT:
            raw = gripper[1]
            if raw is None:
                return self._reading(channel_id, 0.0, observed_at, monotonic_at, Quality.STALE.value)
            return self._reading(channel_id, raw / _SCALE, observed_at, monotonic_at)
        if channel_id == _ARM_ENABLED:
            if enabled is None:
                return self._reading(channel_id, False, observed_at, monotonic_at, Quality.STALE.value)
            return self._reading(channel_id, all(enabled), observed_at, monotonic_at)
        if channel_id == _ARM_MOTION:
            if motion is None:
                return self._reading(channel_id, False, observed_at, monotonic_at, Quality.STALE.value)
            return self._reading(
                channel_id, motion == _MOTION_STATUS_ARRIVED, observed_at, monotonic_at
            )
        if feedback is not None:  # _ARM_STATUS
            return self._reading(
                channel_id,
                {
                    "ctrl_mode": _attr(feedback, "ctrl_mode"),
                    "arm_status": _attr(feedback, "arm_status"),
                    "motion_status": _attr(feedback, "motion_status"),
                },
                observed_at,
                monotonic_at,
            )
        return self._reading(channel_id, {}, observed_at, monotonic_at, Quality.STALE.value)

    # ── Writes ───────────────────────────────────────────────────────────────

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        """Write one channel, reporting whether the effect may have landed."""
        if channel_id == _ARM_ENABLE:
            return await self._write_enable(value)
        if channel_id in _JOINT_CHANNELS:
            return (await self._write_joint_vector([(channel_id, value)]))[0]
        if channel_id == _GRIPPER_POSITION:
            return await self._write_gripper(stroke=value)
        if channel_id == _GRIPPER_EFFORT:
            return await self._write_gripper(effort=value)
        return WriteOutcome(
            ok=False,
            side_effect_state=SIDE_EFFECT_NONE,
            error=f"Piper channel {channel_id!r} is read-only",
            failure_code="not_writable",
        )

    async def write_batch(
        self, commands: tuple[tuple[str, Any], ...]
    ) -> BatchWriteOutcome:
        """Write channels in one bus pass: enable first, then joints, then gripper.

        All joint commands merge into a single ``JointCtrl`` vector (the vendor
        frame format commands six joints at once), so a full-arm batch costs one
        three-frame command rather than six.
        """
        outcomes: list[WriteOutcome | None] = [None] * len(commands)
        for i, (cid, value) in enumerate(commands):
            if cid == _ARM_ENABLE:
                outcomes[i] = await self._write_enable(value)
        joint_indexes = [i for i, (cid, _) in enumerate(commands) if cid in _JOINT_CHANNELS]
        if joint_indexes:
            joint_commands = [commands[i] for i in joint_indexes]
            merged = await self._write_joint_vector(joint_commands)
            for i, outcome in zip(joint_indexes, merged):
                outcomes[i] = outcome
        for i, (cid, value) in enumerate(commands):
            if cid == _GRIPPER_POSITION:
                outcomes[i] = await self._write_gripper(stroke=value)
            elif cid == _GRIPPER_EFFORT:
                outcomes[i] = await self._write_gripper(effort=value)
        final = [
            outcome
            if outcome is not None
            else WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error="channel not writable by the piper transport",
                failure_code="not_writable",
            )
            for outcome in outcomes
        ]
        return BatchWriteOutcome(
            ok=all(o.ok for o in final),
            outcomes=tuple(final),
            side_effect_state=worst_side_effect_state(o.side_effect_state for o in final),
        )

    async def _write_enable(self, value: Any) -> WriteOutcome:
        """Guarded enable/disable: resume + energize, or release torque (arm can fall)."""
        flag = _enable_flag(value)
        if flag is None:
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error=f"arm.enable expects 0 or 1, got {value!r}",
                failure_code="invalid_value",
            )
        arm = self._arm
        if arm is None or not self._connected:
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error="Piper transport is not open",
                failure_code="transport_not_open",
            )
        try:
            if flag:
                # Resume (0x02) is the only vendor path out of a latched
                # emergency stop and a no-op when none is latched, so enable
                # always runs resume-then-energize under one governed write.
                await asyncio.to_thread(arm.EmergencyStop, _ESTOP_RESUME)
                await asyncio.to_thread(arm.EnableArm, _ALL_MOTORS, _ENABLE_FLAG_ON)
            else:
                await asyncio.to_thread(arm.DisableArm, _ALL_MOTORS, _ENABLE_FLAG_OFF)
        except Exception as exc:  # noqa: BLE001 - the 0x150/0x471 frame may have landed
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_UNKNOWN,
                error=f"Enable/disable command failed: {exc}",
                failure_code="piper_send_failed",
            )
        return WriteOutcome(ok=True, side_effect_state=SIDE_EFFECT_COMMITTED)

    async def _write_joint_vector(
        self, commands: list[tuple[str, Any]]
    ) -> list[WriteOutcome]:
        """Send one hold-position ``JointCtrl`` vector covering the whole arm.

        The vendor command carries six joints across three CAN frames, so every
        write fills the uncompmanded joints from the last command or, failing
        that, from current feedback -- a single-joint write must never let the
        other five jump. One vendor call both sends the vector and defines the
        side-effect verdict for every channel in it.

        A refusal on any commanded joint refuses the whole vector: the vendor
        frame is atomic, so sending the remaining joints would move joints the
        caller never asked to move. The send runs on a worker thread -- a
        stalled CAN write must never block the event loop or delay ``halt``.
        """
        for channel_id, value in commands:
            refusal = self._refusal(channel_id, value)
            if refusal is not None:
                return [refusal] * len(commands)
        arm = self._arm
        if arm is None or not self._connected:
            return [self._not_open()] * len(commands)
        joints = await asyncio.to_thread(self._poll_joints, arm)
        targets = [(cid, _JOINT_CHANNELS[cid], _numeric(value)) for cid, value in commands]
        try:
            snapshot = await asyncio.to_thread(self._send_joint_vector, arm, joints, targets)
        except Exception as exc:  # noqa: BLE001 - frames may have partially landed
            return [
                WriteOutcome(
                    ok=False,
                    side_effect_state=SIDE_EFFECT_UNKNOWN,
                    error=f"JointCtrl send failed: {exc}",
                    failure_code="piper_send_failed",
                )
            ] * len(commands)
        if snapshot is None:
            return [
                WriteOutcome(
                    ok=False,
                    side_effect_state=SIDE_EFFECT_NONE,
                    error="no joint feedback to hold the uncommanded joints at; "
                    "open the transport and wait for fresh feedback first",
                    failure_code="piper_no_feedback",
                )
            ] * len(commands)
        joints_after = await asyncio.to_thread(self._poll_joints, arm)
        outcomes: list[WriteOutcome] = []
        for cid, index, _value in targets:
            readback: Reading | None = None
            if joints_after is not None:
                readback = self._reading(
                    cid,
                    joints_after[index - 1] / _SCALE,
                    monotonic_at=self._last_feedback_mono,
                )
            outcomes.append(
                WriteOutcome(ok=True, side_effect_state=SIDE_EFFECT_COMMITTED, readback=readback)
            )
        return outcomes

    def _send_joint_vector(
        self,
        arm: Any,
        joints: list[int] | None,
        targets: list[tuple[str, int, float]],
    ) -> list[int] | None:
        """Blocking hold-position send; runs on a worker thread, never the loop.

        The lock spans fill, send, and cache advance so concurrent writers
        cannot interleave vectors. A ``None`` return means no feedback existed
        to hold the uncompmanded joints at -- decided before anything was sent.
        """
        with self._send_lock:
            vector = [
                commanded if commanded is not None else (joints[index] if joints else None)
                for index, commanded in enumerate(self._commanded_raw)
            ]
            if any(raw is None for raw in vector):
                return None
            for _cid, index, numeric in targets:
                vector[index - 1] = int(round((numeric or 0.0) * _SCALE))
            snapshot = list(vector)
            arm.JointCtrl(*snapshot)
            # Cache advance is success-only: a failed or refused send must not
            # become the next write's hold vector.
            self._commanded_raw = snapshot
            return snapshot

    async def _write_gripper(self, stroke: Any = None, effort: Any = None) -> WriteOutcome:
        """Command the gripper stroke (0.001 mm) or effort (0.001 N*m)."""
        arm = self._arm
        if arm is None or not self._connected:
            return self._not_open()
        channel_id = _GRIPPER_POSITION if stroke is not None else _GRIPPER_EFFORT
        value = stroke if stroke is not None else effort
        refusal = self._refusal(channel_id, value)
        if refusal is not None:
            return refusal
        numeric = _numeric(value)
        assert numeric is not None  # _refusal rejects non-numerics above
        staged_stroke: int | None
        staged_effort = self._commanded_effort_raw
        if stroke is not None:
            staged_stroke = int(round(numeric * _SCALE))
        else:
            staged_effort = int(round(numeric * _SCALE))
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
                channel_id, (raw or 0) / _SCALE, monotonic_at=self._last_feedback_mono
            )
        return WriteOutcome(ok=True, side_effect_state=SIDE_EFFECT_COMMITTED, readback=readback)

    def _refusal(self, channel_id: str, value: Any) -> WriteOutcome | None:
        """Driver-level defense: refuse undeclared values before any send."""
        numeric = _numeric(value)
        if numeric is None:
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error=f"invalid value for {channel_id!r}: {value!r}",
                failure_code="invalid_value",
            )
        channel = self._context.channel(channel_id) if self._context is not None else None
        envelope = channel.envelope if channel is not None else None
        if envelope is not None and envelope.declared and not envelope.contains(numeric):
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                error=f"{channel_id}={numeric} is outside the declared envelope "
                f"[{envelope.min_value}, {envelope.max_value}]",
                failure_code="value_out_of_range",
            )
        return None

    def _not_open(self) -> WriteOutcome:
        return WriteOutcome(
            ok=False,
            side_effect_state=SIDE_EFFECT_NONE,
            error="Piper transport is not open",
            failure_code="transport_not_open",
        )

    # ── Feedback polls (blocking; always run via to_thread) ──────────────────

    def _observe_stamp(self, msg: Any, last: float | None) -> float | None:
        """Advance the freshness clock only when vendor feedback advances.

        The vendor getters return whatever the receive thread last cached, so a
        successful parse does not mean a new sample. The frame's own
        ``time_stamp`` detects advancement; builds that omit it fall back to
        treating every successful parse as fresh.
        """
        raw = _attr(msg, "time_stamp")
        stamp = raw if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
        if stamp is None or stamp != last:
            self._last_feedback_mono = time.monotonic()
            return None if stamp is None else float(stamp)
        return last

    def _poll_joints(self, arm: Any) -> list[int] | None:
        """Return the six raw joint angles, or None when feedback is unusable."""
        try:
            msg = arm.GetArmJointMsgs()
        except Exception:  # noqa: BLE001 - a dead receive thread must read as stale
            return None
        state = _attr(msg, "joint_state", msg)
        raw = [_as_int(_attr(state, f"joint_{i}")) for i in range(1, 7)]
        if any(value is None for value in raw):
            return None
        self._joint_stamp = self._observe_stamp(msg, self._joint_stamp)
        return [int(value) for value in raw]

    def _poll_gripper(self, arm: Any) -> tuple[int | None, int | None]:
        try:
            msg = arm.GetArmGripperMsgs()
        except Exception:  # noqa: BLE001
            return (None, None)
        state = _attr(msg, "gripper_state", msg)
        stroke = _as_int(_attr(state, "grippers_angle"))
        effort = _as_int(_attr(state, "grippers_effort"))
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
        return channel_id in _JOINT_CHANNELS or channel_id in (
            _GRIPPER_POSITION,
            _GRIPPER_EFFORT,
            _ARM_ENABLE,
            _ARM_ENABLED,
            _ARM_STATUS,
            _ARM_MOTION,
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
