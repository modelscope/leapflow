# Copyright (c) Alibaba, Inc. and its affiliates.
"""AgileX Piper-family 6-DOF arm transport (Piper X first) on the pyAgxArm SDK.

``pyAgxArm`` (agilexrobotics/pyAgxArm, the SDK the Piper X manual prescribes)
is imported only when a declared arm is opened, so host-only installations stay
dependency-free and the CAN-owning node (the machine with the ``can0``-style
interface at 1 Mbit/s) owns the vendor SDK -- the same node-local placement as
the RealSense driver beside this module.

SDK facts this driver is built on:

- The SDK is model- and firmware-aware: ``create_agx_arm_config(robot=...,
  firmeware_version=...)`` selects the codec, joint limits and kinematics for
  e.g. ``piper_x`` + ``v189``. A standard-Piper build drives a Piper X with the
  wrong limits, which the firmware answers with "target exceeds limit" on hold
  vectors that look reachable. Bring-up therefore checks the firmware's own
  version against the declared profile (``resolve_firmware_profile``) and
  refuses to open on a mismatch.
- SDK units are rad / m / N; declared channels use deg / mm / N. The SDK joint
  limit clamp is always on, so a value outside a firmware limit is silently
  rewritten before it reaches the arm: such a move executes something other
  than what was planned, and an axis nobody named moves by the difference. The
  driver therefore clamps a *held* axis itself inside the declared envelope and
  only within that channel's declared tolerance; beyond it the write is refused
  before any frame, because "hold" would otherwise be a lie.
- Two motion spaces share one arm: ``joint.*`` targets ride ``move_j`` and
  ``ee.*`` pose targets ride ``move_l`` or ``move_p`` (firmware IK). Hold
  vectors, CAN-mode entry, and the refusal of a batch that mixes both spaces
  live in ``_piper_motion``.
- A move commands six values as several CAN frames inside one vendor call.
  Per-frame outcomes are not observable, so the honest ``WriteOutcome`` mapping
  is: the target frames went out and the firmware reported no refusal ->
  COMMITTED; the firmware answered with a planning refusal -> NONE, because the
  arm is still where the command found it (``TargetRefused``); any other status
  code or an exception from the send itself -> UNKNOWN (which blocks replay
  exactly like COMMITTED). A refusal decided before any send (unknown channel,
  value outside the declared envelope, invalid value, mixed motion spaces,
  unqualified feedback) also reaches nothing.
- ``enable()`` energizes motors without motion and is reachable ONLY through
  the governed ``arm.enable`` write, which first resumes a latched stop
  (``reset()``) under the same approval. Bring-up never energizes anything:
  opens also happen for read-only observation, and a physical action must
  never ride a read path. ``halt`` maps to ``electronic_emergency_stop()`` (a
  damped stop, never a power drop); ``disable()`` drops torque and is only
  reachable as ``arm.enable=0``.
- Getters return the SDK read thread's cached frames, so reads never touch the
  bus. Freshness comes only from per-group receive receipts stamped by a
  packet callback (``_piper_feedback``), never from a getter returning a value.

Responsibilities are split by module: ``_piper_link`` (bring-up, probe state,
energize), ``_piper_motion`` (units, hold vectors, CAN-mode entry) and
``_piper_feedback`` (cached reads certified by receive receipts). This class
only maps declared channels onto them and owns the ``WriteOutcome`` verdicts.
"""

from __future__ import annotations

import asyncio
import importlib
import time
from typing import Any, Mapping

from leapflow.hardware.context import HardwareContext, Quality
from leapflow.hardware.drivers._piper_feedback import (
    ARM_CTRL_MODE,
    ARM_ENABLED,
    ARM_MOTION,
    ARM_STATUS,
    ARM_TELEMETRY,
    GRIPPER_CHANNELS,
    GRIPPER_EFFORT,
    GRIPPER_POSITION,
    STATUS_CHANNELS,
    PiperFeedback,
)
from leapflow.hardware.drivers._piper_link import (
    ENABLE_SETTLE_S,
    LinkSpec,
    energize,
    link_state,
    open_link,
)
from leapflow.hardware.drivers._piper_motion import (
    POSE_PATH_LINEAR,
    POSE_PATH_POINT,
    POSE_PATHS,
    SPACE_CHANNELS,
    SPACE_JOINT,
    HoldRefused,
    PiperMotion,
    TargetRefused,
    enable_flag,
    motion_space,
    motion_spaces,
    numeric,
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

_SDK_MODULE = "pyAgxArm"
_PIPER_MODELS = ("piper", "piper_h", "piper_l", "piper_x")
_LEGACY_KEYS = (
    "sdk_interface",
    "mode_switch",
    "judge_flag",
    "can_auto_init",
    "dh_is_offset",
    "crash_protection_levels",
    "default_effort_n_m",
)
"""piper_sdk-era keys; accepting them silently would hide a stale profile."""

_ARM_ENABLE = "arm.enable"
_ARM_SPEED = "arm.speed_pct"
_ARM_POSE_PATH = "arm.pose_path"
_SETTINGS = (_ARM_SPEED, _ARM_POSE_PATH)  # staged for the next move; no frame now

_CHANNEL_OF_INDEX: dict[str, dict[int, str]] = {
    space: {index: channel_id for channel_id, index in channels.items()}
    for space, channels in SPACE_CHANNELS.items()
}
"""Reverse of ``SPACE_CHANNELS``: a vector index back to its declared channel."""
_STATE_CHANNELS = (_ARM_ENABLE, ARM_ENABLED, *STATUS_CHANNELS, *_SETTINGS, ARM_TELEMETRY)


def _refused(error: str, failure_code: str) -> WriteOutcome:
    """A refusal decided before any send: nothing reached the arm."""
    return WriteOutcome(
        ok=False, side_effect_state=SIDE_EFFECT_NONE, error=error, failure_code=failure_code
    )


def _failed(state: str, error: str, failure_code: str) -> WriteOutcome:
    return WriteOutcome(ok=False, side_effect_state=state, error=error, failure_code=failure_code)


_NOT_WRITABLE = _refused("channel is not writable by the piper transport", "not_writable")
_NOT_OPEN = _refused("Piper transport is not open", "transport_not_open")

_STALE_PLACEHOLDERS: dict[str, Any] = {
    ARM_ENABLED: False,
    ARM_MOTION: False,
    ARM_STATUS: {},
    ARM_CTRL_MODE: "",
    ARM_TELEMETRY: {},
}
"""Value a STALE batch reading carries when feedback is unavailable (default 0.0)."""

_VENDOR_CALL = {SPACE_JOINT: "move_j", POSE_PATH_LINEAR: "move_l", POSE_PATH_POINT: "move_p"}
_MM_PER_M = 1000.0


def _load_sdk() -> Any:
    """Import ``pyAgxArm`` lazily; the host never needs it installed."""
    try:
        return importlib.import_module(_SDK_MODULE)
    except ImportError as exc:
        raise TransportError(
            "pyAgxArm is not installed; install it in the node environment that owns the CAN interface",
            failure_code="pyagxarm_missing",
        ) from exc


def _invalid_config(message: str) -> TransportError:
    return TransportError(message, failure_code="piper_invalid_config")


def _seconds(config: Mapping[str, Any], key: str, default: float, minimum: float) -> float:
    value = numeric(config.get(key, default))
    if value is None or value < minimum:
        raise _invalid_config(f"piper transport {key} must be a finite number >= {minimum}")
    return value


class PiperArmTransport:
    """Drive one Piper-family arm through the node-local CAN interface.

    Bring-up (``create_agx_arm_config`` -> arm + gripper -> receive ledger ->
    ``connect`` -> warm-up -> firmware/profile check) never energizes anything;
    motors are energized only through the governed ``arm.enable`` write. It then
    exposes the declared channels: joints, end-effector pose, gripper
    stroke/force, the guarded enable, speed and pose path, readiness/status and
    bounded telemetry. Writes are bounded target updates the firmware
    interpolates; high-rate sessions ride ``BatchTransport``.
    """

    kind = "piper"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = dict(config or {})
        legacy = sorted(key for key in _LEGACY_KEYS if key in config)
        if legacy:
            raise _invalid_config(
                f"piper transport keys {legacy} belong to the retired piper_sdk driver; remove "
                "them and declare robot/firmware_profile for pyAgxArm instead"
            )
        can_name = str(config.get("can_name") or "can0").strip()
        robot_model = str(config.get("robot") or "piper_x").strip().lower()
        if robot_model not in _PIPER_MODELS:
            raise _invalid_config(
                f"piper transport robot must be one of {list(_PIPER_MODELS)}, got {robot_model!r}"
            )
        pose_path = str(config.get("pose_move_mode") or POSE_PATH_LINEAR).strip().lower()
        if pose_path not in POSE_PATHS:
            raise _invalid_config(
                f"piper transport pose_move_mode must be one of {list(POSE_PATHS)}, got {pose_path!r}"
            )
        speed_pct = min(100, max(1, int(numeric(config.get("default_speed_pct")) or 20)))
        self._spec = LinkSpec(
            robot_model=robot_model,
            profile=str(config.get("firmware_profile") or "v189").strip().lower(),
            can_name=can_name,
            max_age_s=_seconds(config, "stale_after_s", 1.0, 0.01),
            max_skew_s=_seconds(config, "receive_max_skew_s", 0.15, 0.0),
            requalify_s=_seconds(config, "receive_requalify_s", 1.0, 0.0),
            warmup_timeout_s=_seconds(config, "warmup_timeout_s", 3.0, 0.1),
            firmware_timeout_s=_seconds(config, "firmware_timeout_s", 1.0, 0.0),
        )
        # One enable may have to resume a latch, energize and re-try a controller
        # that drops CAN control, so the budget covers several attempts.
        self._enable_timeout_s = _seconds(config, "enable_timeout_s", 6.0, 0.1)
        self._enable_settle_s = _seconds(config, "enable_settle_s", ENABLE_SETTLE_S, 0.0)
        default_force_n = _seconds(config, "default_gripper_force_n", 1.0, 0.0)

        self._robot: Any = None
        self._gripper: Any = None
        self._feedback: PiperFeedback | None = None
        self._context: HardwareContext | None = None
        self._connected = False
        self._sequence = 0
        self._firmware = ""
        # Hold vectors for both motion spaces, seeded from joint feedback at
        # open so the first single-joint write holds the other five joints.
        self._motion = PiperMotion(
            speed_pct=speed_pct,
            pose_path=pose_path,
            mode_timeout_s=_seconds(config, "mode_switch_timeout_s", 0.6, 0.1),
        )
        self._commanded_stroke_mm: float | None = None
        self._commanded_force_n = default_force_n

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def open(self, context: HardwareContext) -> TransportStatus:
        """Run the bring-up sequence once; idempotent afterwards."""
        if self._connected:
            return await self.probe()
        self._context = context
        sdk = _load_sdk()
        try:
            await asyncio.to_thread(self._bring_up, sdk)
        except TransportError:
            self._connected = False
            raise
        except Exception as exc:
            self._connected = False
            raise TransportError(
                f"failed to open Piper arm on {self._spec.can_name!r}: {exc}",
                failure_code="piper_open_failed",
            ) from exc
        self._connected = True
        return await self.probe()

    def _bring_up(self, sdk: Any) -> None:
        """Blocking: open the link and publish it only once it is fully qualified.

        No hold vector is seeded here: a held axis must come from a command this
        driver sent, never from a frame it happened to read.
        """
        link = open_link(sdk, self._spec)
        self._firmware = link.firmware
        self._robot, self._gripper, self._feedback = link.robot, link.gripper, link.feedback

    async def close(self) -> TransportStatus:
        """Disconnect from the CAN port; never powers the arm down."""
        robot, self._robot, self._gripper, self._feedback = self._robot, None, None, None
        self._connected = False
        if robot is not None:
            try:
                await asyncio.to_thread(robot.disconnect)
            except Exception:  # noqa: BLE001 - close is idempotent even on a dead bus
                pass
            self._motion.disown("the link was closed")
        return TransportStatus(connected=False, halt_supported=True, detail="Piper transport closed")

    async def probe(self) -> TransportStatus:
        """Report link health and the arm state cached by the SDK read thread."""
        identity = {"can_name": self._spec.can_name, "robot": self._spec.robot_model,
                    "firmware_profile": self._spec.profile}
        robot = self._robot
        if robot is None:
            return TransportStatus(connected=False, halt_supported=True,
                                   detail="Piper transport is closed", metadata=identity)
        state = await asyncio.to_thread(link_state, robot)
        link_up, healthy = state.pop("connected"), state.pop("healthy")
        connected = self._connected and link_up and healthy
        fps = state["can_fps"]
        return TransportStatus(
            connected=connected,
            halt_supported=True,
            detail="Piper arm link healthy" if connected else "Piper arm link degraded or closed",
            latency_ms=round(1000.0 / fps, 3) if fps > 0 else 0.0,
            metadata={**identity, "firmware_version": self._firmware,
                      "pose_path": self._motion.pose_path, **state},
        )

    async def halt(self) -> TransportStatus:
        """Engage the damped electronic emergency stop. Lock-free, never raises.

        Deliberately NOT ``disable()``: dropping torque lets the arm fall.
        Resuming (``reset()``) is a separate governed write on the
        ``arm.enable`` path, never part of a halt.
        """
        robot = self._robot
        if robot is None:
            return TransportStatus(connected=False, halt_supported=True,
                                   detail="Piper transport is closed; nothing to halt")
        # The stop ends this driver's ownership of the arm's posture; the entry
        # is taken before the frame goes out so nothing can be held across it.
        self._motion.disown("an emergency stop")
        try:
            await asyncio.to_thread(robot.electronic_emergency_stop)
        except Exception as exc:  # noqa: BLE001 - halt must never raise
            return TransportStatus(connected=self._connected, halt_supported=True,
                                   detail=f"Piper emergency stop command failed: {exc}")
        # The stop latches: re-asserting CAN control over a latched arm would
        # only make the latch look transient. Resume is a governed enable.
        return TransportStatus(
            connected=self._connected,
            halt_supported=True,
            detail="Piper damped emergency stop engaged; resume is a separate governed action",
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
        return self._reading(channel_id, value, receipts=values["_receive"])

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
            self._reading(cid, values[cid], receipts=values["_receive"])
            if values.get(cid) is not None
            else self._reading(cid, _STALE_PLACEHOLDERS.get(cid, 0.0), stale=True)
            for cid in channel_ids
        )
        return BatchReading(
            device_id=self._device_id,
            readings=readings,
            observed_at=observed_at,
            monotonic_at=monotonic_at,
        )

    def _poll_values(self, channel_ids: tuple[str, ...]) -> dict[str, Any]:
        """Blocking: feedback values with their receipts, plus the staged settings."""
        feedback = self._feedback
        assert feedback is not None
        values = feedback.poll(channel_ids)
        if _ARM_SPEED in channel_ids:
            values[_ARM_SPEED] = float(self._motion.speed_pct)
        if _ARM_POSE_PATH in channel_ids:
            values[_ARM_POSE_PATH] = self._motion.pose_path
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
        if channel_id == GRIPPER_POSITION:
            return await self._write_gripper(stroke=value)
        if channel_id == GRIPPER_EFFORT:
            return await self._write_gripper(force=value)
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
            return BatchWriteOutcome(ok=False, outcomes=tuple(refusal for _ in commands),
                                     side_effect_state=SIDE_EFFECT_NONE)
        if any(cid not in {*_SETTINGS, _ARM_ENABLE, *GRIPPER_CHANNELS} and not motion_space(cid)
               for cid, _ in commands):
            return BatchWriteOutcome(ok=False, outcomes=tuple(_NOT_WRITABLE for _ in commands),
                                     side_effect_state=SIDE_EFFECT_NONE)
        outcomes: list[WriteOutcome | None] = [None] * len(commands)
        blocked = _refused("an earlier command failed; remaining batch commands were not attempted",
                           "piper_batch_aborted")

        def failed() -> bool:
            return any(outcome is not None and not outcome.ok for outcome in outcomes)

        for i, (cid, value) in enumerate(commands):
            if cid == _ARM_ENABLE:
                outcomes[i] = blocked if failed() else await self._write_enable(value)
            elif cid in _SETTINGS:
                outcomes[i] = blocked if failed() else self._write_setting(cid, value)
        for space in spaces:
            indexes = [i for i, (cid, _) in enumerate(commands) if motion_space(cid) == space]
            merged = ([blocked] * len(indexes) if failed()
                      else await self._write_motion(space, [commands[i] for i in indexes]))
            for i, outcome in zip(indexes, merged):
                outcomes[i] = outcome
        for i, (cid, value) in enumerate(commands):
            if cid == GRIPPER_POSITION:
                outcomes[i] = blocked if failed() else await self._write_gripper(stroke=value)
            elif cid == GRIPPER_EFFORT:
                outcomes[i] = blocked if failed() else await self._write_gripper(force=value)
        final = [outcome if outcome is not None else _NOT_WRITABLE for outcome in outcomes]
        return BatchWriteOutcome(
            ok=all(o.ok for o in final),
            outcomes=tuple(final),
            side_effect_state=worst_side_effect_state(o.side_effect_state for o in final),
        )

    async def _write_enable(self, value: Any) -> WriteOutcome:
        """Guarded enable/disable: resume, energize, enter CAN -- or drop torque."""
        flag = enable_flag(value)
        if flag is None:
            return _refused(f"arm.enable expects 0 or 1, got {value!r}", "invalid_value")
        robot, feedback = self._robot, self._feedback
        if robot is None or feedback is None or not self._connected:
            return _NOT_OPEN
        if flag:
            return await asyncio.to_thread(energize, robot, feedback, self._motion,
                                           self._enable_timeout_s, self._enable_settle_s,
                                           self._spec.requalify_s)
        # Torque is about to drop, so the arm may end up anywhere -- including
        # by hand. Whatever vector was held stops describing it now.
        self._motion.disown("a governed disable")
        try:
            await asyncio.to_thread(robot.disable)
        except Exception as exc:  # noqa: BLE001 - the disable frame may have landed
            return _failed(SIDE_EFFECT_UNKNOWN, f"disable command failed: {exc}", "piper_send_failed")
        return WriteOutcome(ok=True, side_effect_state=SIDE_EFFECT_COMMITTED)

    def _write_setting(self, channel_id: str, value: Any) -> WriteOutcome:
        """Stage the speed rate or pose path the next move applies; nothing is sent now."""
        if channel_id == _ARM_POSE_PATH:
            if not isinstance(value, str) or value not in POSE_PATHS:
                return _refused(f"arm.pose_path expects one of {', '.join(POSE_PATHS)}",
                                "invalid_value")
            applied: Any = self._motion.set_pose_path(value)
        else:
            refusal = self._refusal(_ARM_SPEED, value)
            if refusal is not None:
                return refusal
            applied = float(self._motion.set_speed(int(round(numeric(value) or 1))))
        readback = self._reading(channel_id, applied)
        return WriteOutcome(ok=True, side_effect_state=SIDE_EFFECT_COMMITTED, readback=readback)

    async def _write_motion(self, space: str, commands: list[tuple[str, Any]]) -> list[WriteOutcome]:
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
        robot = self._robot
        if robot is None or not self._connected:
            return [_NOT_OPEN] * len(commands)
        problem = await self._wait_receive()
        if problem:
            return [_refused(f"fresh per-group CAN reception is required before motion: {problem}",
                             "piper_unqualified_feedback")] * len(commands)
        channels = SPACE_CHANNELS[space]
        targets = [(channels[cid], float(numeric(value) or 0.0)) for cid, value in commands]
        call = _VENDOR_CALL[space if space == SPACE_JOINT else self._motion.pose_path]
        try:
            snapshot = await asyncio.to_thread(self._send_motion, robot, space, targets)
        except (HoldRefused, TargetRefused) as exc:
            return [_refused(str(exc), exc.failure_code)] * len(commands)
        except TransportError as exc:
            return [_failed(SIDE_EFFECT_UNKNOWN, f"{call}: {exc}", exc.failure_code)] * len(commands)
        except Exception as exc:  # noqa: BLE001 - frames may have partially landed
            return [_failed(SIDE_EFFECT_UNKNOWN, f"{call} send failed: {exc}",
                            "piper_send_failed")] * len(commands)
        if snapshot is None:
            return [_refused(f"no {space} feedback to hold the uncommanded axes at; open the "
                             "transport and wait for fresh feedback first",
                             "piper_no_feedback")] * len(commands)
        captured = await asyncio.to_thread(self._poll_values, tuple(cid for cid, _ in commands))
        return [
            WriteOutcome(
                ok=True,
                side_effect_state=SIDE_EFFECT_COMMITTED,
                readback=(self._reading(cid, captured[cid], receipts=captured["_receive"])
                          if captured.get(cid) is not None else None),
            )
            for cid, _value in commands
        ]

    def _send_motion(self, robot: Any, space: str, targets: list[tuple[int, float]]) -> list[float] | None:
        """Blocking: read hold feedback, then send through the serialized state."""
        feedback = self._feedback.space(space) if self._feedback is not None else None
        return self._motion.send(robot, space, feedback, targets,
                                 validate=self._require_receive, hold=self._hold_value)

    def _hold_value(self, space: str, index: int, value: float) -> float:
        """Keep a held axis inside its declared envelope, or refuse to send it.

        Holding an axis means commanding it where it already is. The vendor
        clamps whatever it is given into its own firmware limit map, so a hold
        outside the envelope does not hold at all: the arm travels to the limit
        while the write reports success, and the caller never named that axis.

        Inside the channel's own declared tolerance the clamp is applied here,
        so what this driver sends is what the arm executes. Beyond it the vector
        is refused before any frame goes out and the message names the axis, its
        reading, and its limit -- the arm has to be brought inside the envelope
        by a command that names it.
        """
        channel_id = _CHANNEL_OF_INDEX[space][index]
        channel = self._context.channel(channel_id) if self._context is not None else None
        envelope = channel.envelope if channel is not None else None
        if envelope is None or not envelope.declared or envelope.contains(value):
            return value
        clamped = value
        if envelope.min_value is not None:
            clamped = max(clamped, envelope.min_value)
        if envelope.max_value is not None:
            clamped = min(clamped, envelope.max_value)
        drift = abs(clamped - value)
        if drift > float(envelope.tolerance or 0.0):
            unit = channel.unit if channel is not None else ""
            raise HoldRefused(
                f"{channel_id} stands at {value:.3f}{' ' + unit if unit else ''}, "
                f"{drift:.3f} outside its declared envelope "
                f"[{envelope.min_value}, {envelope.max_value}]; holding it would move an "
                "axis this command does not name, so name it and bring it inside first",
                failure_code="piper_hold_outside_envelope",
            )
        return clamped

    async def _write_gripper(self, stroke: Any = None, force: Any = None) -> WriteOutcome:
        """Command the gripper stroke (mm) or grip force (N); the other is held."""
        robot, gripper, feedback = self._robot, self._gripper, self._feedback
        if robot is None or gripper is None or feedback is None or not self._connected:
            return _NOT_OPEN
        channel_id = GRIPPER_POSITION if stroke is not None else GRIPPER_EFFORT
        value = stroke if stroke is not None else force
        refusal = self._refusal(channel_id, value)
        if refusal is not None:
            return refusal
        problem = await self._wait_receive()
        if problem:
            return _refused(f"fresh per-group CAN reception is required before gripper commands: "
                            f"{problem}", "piper_unqualified_feedback")
        number = numeric(value)
        assert number is not None  # _refusal rejects non-numerics above
        staged_stroke, staged_force = self._commanded_stroke_mm, self._commanded_force_n
        if stroke is not None:
            staged_stroke = number
        else:
            staged_force = number
            if staged_stroke is None:
                staged_stroke, _ = await asyncio.to_thread(feedback.gripper)
                if staged_stroke is None:
                    return _refused("no gripper stroke command or feedback to pair with the force",
                                    "piper_no_feedback")
        try:
            # The gripper bypasses the motion path, so the mode guard is this
            # write's own: outside CAN control the firmware drops the frame
            # while the outcome would still read committed.
            await asyncio.to_thread(self._motion.ensure_can, robot)
        except TransportError as exc:
            return _failed(SIDE_EFFECT_COMMITTED, str(exc), "piper_mode_switch_failed")
        except Exception as exc:  # noqa: BLE001 - a mode frame may have landed
            return _failed(SIDE_EFFECT_UNKNOWN, f"mode selection failed: {exc}", "piper_send_failed")
        problem = await self._wait_receive()
        if problem:
            return _failed(SIDE_EFFECT_COMMITTED,
                           f"feedback lost during gripper mode selection; target not sent: {problem}",
                           "piper_unqualified_feedback")
        try:
            await asyncio.to_thread(gripper.move_gripper_m, value=staged_stroke / _MM_PER_M,
                                    force=staged_force)
        except Exception as exc:  # noqa: BLE001 - the gripper frame may have landed
            return _failed(SIDE_EFFECT_UNKNOWN, f"move_gripper_m send failed: {exc}",
                           "piper_send_failed")
        # Caches advance only after the frame was accepted: a refused or
        # uncertain send must not leak into the next write's stroke/force pair.
        self._commanded_stroke_mm, self._commanded_force_n = staged_stroke, staged_force
        captured = await asyncio.to_thread(self._poll_values, (channel_id,))
        readback = (self._reading(channel_id, captured[channel_id], receipts=captured["_receive"])
                    if captured.get(channel_id) is not None else None)
        return WriteOutcome(ok=True, side_effect_state=SIDE_EFFECT_COMMITTED, readback=readback)

    # ── Shared helpers ───────────────────────────────────────────────────────

    def _receive_problem(self) -> str:
        feedback = self._feedback
        return feedback.problem() if feedback is not None else "Piper transport is not open"

    async def _wait_receive(self) -> str:
        """Establish fresh reception for a bounded time; ``""`` when the ledger coheres.

        A write is gated on feedback from the very groups it is about to command,
        and the gate used to be one sample. That sample is taken in an event loop
        that also decodes camera captures, and a capture starves the SDK's read
        thread long enough for one group to lag the others by half a second --
        which is a measurement of the capture, not of the bus. Nothing has been
        sent when this waits, so the wait costs only latency; a ledger that stays
        incoherent for the whole budget is still refused, unchanged.
        """
        feedback = self._feedback
        if feedback is None:
            return "Piper transport is not open"
        deadline = time.monotonic() + self._spec.requalify_s
        while True:
            problem = feedback.problem()
            if not problem or time.monotonic() >= deadline:
                return problem
            await asyncio.sleep(min(0.02, max(0.0, deadline - time.monotonic())))

    def _require_receive(self) -> None:
        problem = self._receive_problem()
        if problem:
            raise TransportError(problem, failure_code="piper_unqualified_feedback")

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

    def _reading(
        self,
        channel_id: str,
        value: Any,
        *,
        receipts: Mapping[str, Mapping] | None = None,
        stale: bool = False,
    ) -> Reading:
        """Build a Reading whose time and quality come from receive receipts.

        Settings are driver state, always current. Every feedback channel is
        OK only when all CAN groups that carry it were received recently and
        coherently; its timestamps are then those of the oldest such group.
        """
        channel = self._context.channel(channel_id) if self._context is not None else None
        observed_at, monotonic_at = time.time(), time.monotonic()
        quality = Quality.OK.value
        if channel_id not in _SETTINGS:
            evidence = (None if stale or self._feedback is None
                        else self._feedback.evidence(channel_id, receipts))
            if evidence is None:
                quality, monotonic_at = Quality.STALE.value, 0.0
            else:
                observed_at, monotonic_at = evidence
        self._sequence += 1
        return Reading(
            device_id=self._device_id,
            channel_id=channel_id,
            value=value,
            quantity=channel.quantity if channel is not None else "",
            unit=channel.unit if channel is not None else "",
            observed_at=observed_at,
            monotonic_at=monotonic_at,
            sequence=self._sequence,
            quality=quality,
        )

    def _known_channel(self, channel_id: str) -> bool:
        return (
            bool(motion_space(channel_id))
            or channel_id in GRIPPER_CHANNELS
            or channel_id in _STATE_CHANNELS
        )

    def _require_open(self, channel_ids: tuple[str, ...] | str) -> None:
        label = channel_ids if isinstance(channel_ids, str) else list(channel_ids)
        if not self._connected or self._robot is None:
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
