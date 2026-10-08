# Copyright (c) Alibaba, Inc. and its affiliates.
# Adapted from LeRobot (https://github.com/huggingface/lerobot).
"""Robot arm transport for LeapRobot: bridges a robot arm instance to the HCP six-method contract.

Robot arms expose a synchronous Python API -- ``connect()``, ``disconnect()``,
``get_observation()``, ``send_action()`` -- designed for direct control loops.
This transport wraps that API behind the asynchronous HCP contract so the rest
of the hardware stack (registry, streaming, tools, trust, audit) works without
knowing anything about the robot arm implementation.

The synchronous calls are dispatched to a thread via ``asyncio.to_thread`` so
the event loop is never blocked.  ``read_batch`` and ``write_batch`` exploit
the fact that the robot's native interface is already vectorized: one
``get_observation()`` returns all joint positions and camera frames at once,
and one ``send_action()`` commands all actuators atomically.

Implements three protocols:
- ``HardwareTransport`` (core six methods)
- ``BatchTransport`` (vectorized read/write for high-frequency control)
- ``FrameTransport`` (camera frame capture)
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Any, Callable, Mapping

from leapflow.hardware.context import (
    HardwareContext,
    HardwareEffect,
    Quality,
    Representation,
)
from leapflow.hardware.transport import (
    BatchReading,
    BatchWriteOutcome,
    FrameReading,
    Reading,
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_UNKNOWN,
    TransportError,
    TransportStatus,
    WriteOutcome,
)
from leapflow.hardware.transports.robot_channels import RobotChannelMapper

logger = logging.getLogger(__name__)

# Observation cache TTL in seconds.  Multiple reads within one control step
# reuse the same hardware observation rather than each triggering a bus poll.
_OBS_CACHE_TTL_S = 0.005  # 5 ms


class RobotArmTransport:
    """Wraps a robot arm instance as an HCP transport.

    Construction receives a factory callable rather than a live robot:
    ``open()`` calls the factory, ``close()`` calls ``disconnect()``.
    This ensures the transport owns the robot lifecycle and can recover
    from a disconnection by re-creating the instance.
    """

    kind: str = "robot_arm"

    def __init__(
        self,
        robot_factory: Callable[[], Any],
        channel_mapper: RobotChannelMapper | None = None,
    ) -> None:
        self._factory = robot_factory
        self._mapper = channel_mapper or RobotChannelMapper()
        self._robot: Any = None
        self._channels: dict[str, Any] = {}  # channel_id -> Channel
        self._connected = False
        self._context: HardwareContext | None = None
        self._sequence: dict[str, int] = {}

        # Observation cache: avoids redundant hardware polls within one step.
        self._obs_cache: dict[str, Any] | None = None
        self._obs_cache_mono: float = 0.0

        # Snapshot of last known observation for hold-position during writes.
        self._last_obs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # HardwareTransport (6 methods)
    # ------------------------------------------------------------------

    async def open(self, context: HardwareContext) -> TransportStatus:
        """Create the robot via factory, call connect(), populate channel map."""
        try:
            self._robot = self._factory()
            await asyncio.to_thread(self._robot.connect)
        except Exception as exc:
            self._connected = False
            raise TransportError(
                f"failed to connect robot arm: {exc}",
                failure_code="robot_arm_connect_failed",
            ) from exc

        self._context = context
        self._connected = True
        # Build channel lookup table from the declared context.
        self._channels = {ch.channel_id: ch for ch in context.channels}
        self._obs_cache = None
        self._obs_cache_mono = 0.0
        logger.info(
            "Robot arm transport opened: %d channels declared",
            len(self._channels),
        )
        return await self.probe()

    async def close(self) -> TransportStatus:
        """Disconnect the robot. Idempotent, never raises."""
        if self._robot is not None:
            try:
                await asyncio.to_thread(self._robot.disconnect)
            except Exception:
                logger.debug("Robot arm disconnect raised; suppressed", exc_info=True)
        self._connected = False
        self._robot = None
        self._obs_cache = None
        self._last_obs.clear()
        return TransportStatus(
            connected=False,
            halt_supported=True,
            detail="robot arm transport closed",
        )

    async def read(self, channel_id: str) -> Reading:
        """Read one channel by calling get_observation() and extracting the value.

        For motor channels: extract position/velocity from observation dict.
        For camera channels: raise TransportError (use read_frame instead).
        """
        self._require_open(channel_id)
        ch = self._channels.get(channel_id)
        if ch is None:
            raise TransportError(
                f"unknown channel {channel_id!r}",
                failure_code="unknown_channel",
            )
        if ch.representation == Representation.FRAME.value:
            raise TransportError(
                f"channel {channel_id!r} is a frame channel; use read_frame()",
                failure_code="frame_channel_not_readable",
            )
        obs = await self._get_observation()
        value = self._extract_value(channel_id, obs)
        seq = self._next_seq(channel_id)
        return Reading(
            device_id=self._context.device_id if self._context else "",
            channel_id=channel_id,
            value=value,
            quantity=ch.quantity,
            unit=ch.unit,
            sequence=seq,
            quality=Quality.OK.value,
        )

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        """Write one channel by building an action vector and calling send_action().

        Only ACTUATE channels (joint positions, gripper) are writable.
        The action vector is constructed with the current observation for
        non-targeted channels (hold position) and the commanded value for
        the target channel.
        """
        self._require_open(channel_id)
        ch = self._channels.get(channel_id)
        if ch is None:
            raise TransportError(
                f"unknown channel {channel_id!r}",
                failure_code="unknown_channel",
            )
        if not ch.is_writable:
            raise TransportError(
                f"channel {channel_id!r} is read-only",
                failure_code="channel_read_only",
            )
        # Build command dict for a single channel write.
        commands = {channel_id: value}
        # Refresh observation so hold-position uses current joint values,
        # not stale or empty data.
        try:
            await self._get_observation()
        except Exception:
            logger.debug(
                "Observation refresh before write failed; "
                "hold-position may use stale data",
                exc_info=True,
            )
        try:
            action = self._build_action(commands)
            await asyncio.to_thread(self._robot.send_action, action)
        except TransportError:
            raise
        except Exception as exc:
            # The action may have partially reached the device.
            return WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_UNKNOWN,
                error=f"send_action failed: {exc}",
                failure_code="robot_arm_send_action_failed",
            )

        # Invalidate observation cache so the next read picks up the new state.
        self._obs_cache = None

        # Readback if the channel requests verification.
        readback: Reading | None = None
        if ch.verify_after_write:
            try:
                readback = await self.read(channel_id)
            except Exception:
                logger.debug("Readback after write failed for %s", channel_id, exc_info=True)

        return WriteOutcome(
            ok=True,
            side_effect_state=SIDE_EFFECT_COMMITTED,
            readback=readback,
            settled=ch.envelope.settling_time_s <= 0.0,
        )

    async def probe(self) -> TransportStatus:
        """Check robot connection health. Non-blocking, returns from cached state."""
        connected = self._connected and self._robot is not None
        return TransportStatus(
            connected=connected,
            halt_supported=True,
            detail="robot arm transport" if connected else "robot arm disconnected",
            metadata={
                "channels": len(self._channels),
                "robot_type": type(self._robot).__name__ if self._robot else "none",
            },
        )

    async def halt(self) -> TransportStatus:
        """Emergency stop: send zero velocity to all actuators.

        Lock-free (as required by HCP). Constructs a zero-velocity action
        and sends it immediately without acquiring any lock.
        """
        if not self._connected or self._robot is None:
            return TransportStatus(
                connected=False,
                halt_supported=True,
                detail="halt: not connected",
            )
        try:
            zero_commands: dict[str, Any] = {}
            for cid, ch in self._channels.items():
                if ch.effect in HardwareEffect.writable():
                    # Send zero / current position to stop all motion.
                    zero_commands[cid] = 0.0
            if zero_commands:
                action = self._build_action(zero_commands, hold_position=False)
                # Dispatch to thread but do NOT hold any lock.
                await asyncio.to_thread(self._robot.send_action, action)
        except Exception as exc:
            logger.warning("Robot arm halt encountered error: %s", exc, exc_info=True)
            return TransportStatus(
                connected=self._connected,
                halt_supported=True,
                detail=f"halt attempted with error: {exc}",
            )
        return TransportStatus(
            connected=self._connected,
            halt_supported=True,
            detail="halted",
        )

    # ------------------------------------------------------------------
    # BatchTransport
    # ------------------------------------------------------------------

    async def read_batch(self, channel_ids: tuple[str, ...]) -> BatchReading:
        """Read multiple channels in one get_observation() call.

        This is the natural mode for robot arms: one call returns all values.
        Much more efficient than N sequential read() calls.
        """
        self._require_open("batch_read")
        obs = await self._get_observation()
        now_wall = time.time()
        now_mono = time.monotonic()
        readings: list[Reading] = []
        device_id = self._context.device_id if self._context else ""
        for cid in channel_ids:
            ch = self._channels.get(cid)
            if ch is None:
                raise TransportError(
                    f"unknown channel {cid!r} in batch read",
                    failure_code="unknown_channel",
                )
            if ch.representation == Representation.FRAME.value:
                raise TransportError(
                    f"frame channel {cid!r} cannot appear in read_batch; "
                    "use read_frame() instead",
                    failure_code="frame_channel_not_readable",
                )
            value = self._extract_value(cid, obs)
            seq = self._next_seq(cid)
            readings.append(
                Reading(
                    device_id=device_id,
                    channel_id=cid,
                    value=value,
                    quantity=ch.quantity,
                    unit=ch.unit,
                    observed_at=now_wall,
                    monotonic_at=now_mono,
                    sequence=seq,
                    quality=Quality.OK.value,
                )
            )
        return BatchReading(
            device_id=device_id,
            readings=tuple(readings),
            observed_at=now_wall,
            monotonic_at=now_mono,
        )

    async def write_batch(
        self, commands: tuple[tuple[str, Any], ...]
    ) -> BatchWriteOutcome:
        """Write multiple channels in one send_action() call.

        Builds the full action vector from commands, filling non-commanded
        channels with their current values (hold position).
        """
        self._require_open("batch_write")
        # Validate all channels first.
        cmd_dict: dict[str, Any] = {}
        for cid, val in commands:
            ch = self._channels.get(cid)
            if ch is None:
                raise TransportError(
                    f"unknown channel {cid!r} in batch write",
                    failure_code="unknown_channel",
                )
            if not ch.is_writable:
                raise TransportError(
                    f"channel {cid!r} is read-only",
                    failure_code="channel_read_only",
                )
            cmd_dict[cid] = val

        # Refresh observation so hold-position uses current joint values,
        # not stale or empty data.
        try:
            await self._get_observation()
        except Exception:
            logger.debug(
                "Observation refresh before batch write failed; "
                "hold-position may use stale data",
                exc_info=True,
            )
        try:
            action = self._build_action(cmd_dict)
            await asyncio.to_thread(self._robot.send_action, action)
        except TransportError:
            raise
        except Exception as exc:
            outcomes = tuple(
                WriteOutcome(
                    ok=False,
                    side_effect_state=SIDE_EFFECT_UNKNOWN,
                    error=f"send_action failed: {exc}",
                    failure_code="robot_arm_send_action_failed",
                )
                for _ in commands
            )
            return BatchWriteOutcome(
                ok=False,
                outcomes=outcomes,
                side_effect_state=SIDE_EFFECT_UNKNOWN,
            )

        # Invalidate cache.
        self._obs_cache = None

        outcomes = tuple(
            WriteOutcome(
                ok=True,
                side_effect_state=SIDE_EFFECT_COMMITTED,
                settled=True,
            )
            for _ in commands
        )
        return BatchWriteOutcome(
            ok=True,
            outcomes=outcomes,
            side_effect_state=SIDE_EFFECT_COMMITTED,
        )

    # ------------------------------------------------------------------
    # FrameTransport
    # ------------------------------------------------------------------

    async def read_frame(
        self,
        channel_id: str,
        *,
        max_width: int = 0,
        quality: int = 0,
        fps: float = 0.0,
    ) -> FrameReading:
        """Capture one frame from a camera channel.

        Calls get_observation() and extracts the image for the requested
        camera, encoding it as JPEG.
        """
        self._require_open(channel_id)
        ch = self._channels.get(channel_id)
        if ch is None:
            raise TransportError(
                f"unknown channel {channel_id!r}",
                failure_code="unknown_channel",
            )
        if ch.representation != Representation.FRAME.value:
            raise TransportError(
                f"channel {channel_id!r} is not a frame channel",
                failure_code="not_frame_channel",
            )
        obs = await self._get_observation()
        image_data = self._extract_frame(channel_id, obs)
        jpeg_bytes, width, height = self._encode_jpeg(
            image_data, max_width=max_width, quality=quality,
        )
        seq = self._next_seq(channel_id)
        return FrameReading(
            device_id=self._context.device_id if self._context else "",
            channel_id=channel_id,
            data=jpeg_bytes,
            media_type="image/jpeg",
            width=width,
            height=height,
            sequence=seq,
            quality=Quality.OK.value,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_observation(self) -> dict[str, Any]:
        """Thread-safe observation retrieval with caching.

        Cache the observation for a short window (5 ms) so that multiple
        reads within one control step don't each trigger a hardware poll.
        """
        now = time.monotonic()
        if self._obs_cache is not None and (now - self._obs_cache_mono) < _OBS_CACHE_TTL_S:
            return self._obs_cache

        if self._robot is None:
            raise TransportError(
                "robot instance is None",
                failure_code="transport_not_open",
            )
        obs = await asyncio.to_thread(self._robot.get_observation)
        if not isinstance(obs, dict):
            # Robot returns a dict; convert if needed.
            obs = dict(obs) if hasattr(obs, "items") else {}
        self._obs_cache = obs
        self._obs_cache_mono = time.monotonic()
        # Keep a snapshot for hold-position writes.
        self._last_obs.update(obs)
        return obs

    def _build_action(
        self,
        commands: dict[str, Any],
        *,
        hold_position: bool = True,
    ) -> Any:
        """Build a robot action tensor from channel commands.

        Non-commanded channels hold their current observed position when
        *hold_position* is True.  When False (halt), zeros are used.
        """
        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]

        # Collect all writable channels that map to action features.
        action_values: list[float] = []
        action_keys: list[str] = []

        for cid, ch in self._channels.items():
            if not ch.is_writable:
                continue
            # Only scalar/state channels participate in the action vector;
            # frame channels are never written.
            if ch.representation == Representation.FRAME.value:
                continue
            action_keys.append(cid)
            if cid in commands:
                action_values.append(float(commands[cid]))
            elif hold_position:
                # Use the feature key that maps to this channel.
                obs_val = self._resolve_obs_value(cid, self._last_obs) if self._last_obs else None
                if obs_val is not None:
                    action_values.append(float(obs_val))
                else:
                    raise TransportError(
                        f"Cannot hold position for channel {cid!r}: no observation "
                        "available. Read the device at least once before writing.",
                        failure_code="hold_position_no_observation",
                    )
            else:
                action_values.append(0.0)

        # Return as a torch tensor if available (expected format for many
        # robot SDKs), otherwise as a plain list.
        if torch is not None:
            return torch.tensor(action_values, dtype=torch.float32)
        return action_values

    def _extract_value(self, channel_id: str, obs: dict[str, Any]) -> Any:
        """Extract a scalar value for *channel_id* from the observation dict.

        Tries multiple key strategies:
        1. The channel_id itself (if the user named observation keys directly).
        2. The feature name derived from the channel_id (e.g. "joint.X.position" -> "X.position").
        3. Partial matching against observation keys.
        """
        # Strategy 1: direct key.
        if channel_id in obs:
            return _scalar(obs[channel_id])

        # Strategy 2: derive feature name from HCP channel_id.
        feature_key = self._channel_to_feature_key(channel_id)
        if feature_key and feature_key in obs:
            return _scalar(obs[feature_key])

        # Strategy 3: partial match -- find any obs key that contains the
        # base name of this channel.
        base = _channel_base_name(channel_id)
        suffix = _channel_suffix(channel_id)
        for key in obs:
            if base in key and (not suffix or suffix in key):
                return _scalar(obs[key])

        # No match found -- return None rather than fabricating a value.
        logger.debug(
            "No observation key found for channel %r (obs keys: %s)",
            channel_id, list(obs.keys()),
        )
        return None

    def _extract_frame(self, channel_id: str, obs: dict[str, Any]) -> Any:
        """Extract the raw image tensor/array for a camera channel."""
        # Derive the camera feature name.
        # channel_id is "camera.{name}", feature key is typically "{name}" or
        # "observation.images.{name}" in some robot SDKs.
        camera_name = channel_id.removeprefix("camera.")
        for candidate in (
            channel_id,
            camera_name,
            f"observation.images.{camera_name}",
            f"images.{camera_name}",
        ):
            if candidate in obs:
                return obs[candidate]
        # Fallback: search for any key containing the camera name and "image"/"cam".
        for key, val in obs.items():
            if camera_name in key:
                return val
        raise TransportError(
            f"no image data found for camera channel {channel_id!r}",
            failure_code="camera_data_missing",
        )

    def _resolve_obs_value(self, channel_id: str, obs: dict[str, Any]) -> Any:
        """Resolve a channel's value from the observation, for hold-position."""
        return self._extract_value(channel_id, obs)

    @staticmethod
    def _channel_to_feature_key(channel_id: str) -> str:
        """Convert an HCP channel_id to a probable robot feature key.

        ``"joint.shoulder_pan.position"`` -> ``"shoulder_pan.position"``
        ``"gripper.left"`` -> ``"left"``
        ``"camera.top"`` -> ``"top"``
        """
        parts = channel_id.split(".", 1)
        if len(parts) == 2 and parts[0] in ("joint", "gripper", "camera"):
            return parts[1]
        return channel_id

    @staticmethod
    def _encode_jpeg(
        image_data: Any,
        *,
        max_width: int = 0,
        quality: int = 0,
    ) -> tuple[bytes, int, int]:
        """Encode an image tensor/array as JPEG bytes.

        Returns (jpeg_bytes, width, height).
        """
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise TransportError(
                f"PIL/numpy required for frame encoding: {exc}",
                failure_code="frame_encoding_unavailable",
            ) from exc

        # Convert torch tensor to numpy if needed.
        if hasattr(image_data, "cpu"):
            arr = image_data.cpu().numpy()
        elif isinstance(image_data, np.ndarray):
            arr = image_data
        else:
            arr = np.asarray(image_data)

        # Handle CHW -> HWC conversion (common for torch tensors).
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        # Normalize float [0, 1] to uint8 [0, 255].
        if arr.dtype in (np.float32, np.float64):
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)

        img = Image.fromarray(arr)

        # Resize if requested.
        if max_width > 0 and img.width > max_width:
            ratio = max_width / img.width
            new_height = max(1, int(img.height * ratio))
            img = img.resize((max_width, new_height), Image.LANCZOS)

        width, height = img.size
        jpeg_quality = quality if quality > 0 else 85
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        return buf.getvalue(), width, height

    def _require_open(self, channel_id: str) -> None:
        """Raise TransportError when the transport is not connected."""
        if not self._connected or self._robot is None:
            raise TransportError(
                f"transport for {channel_id!r} is not open",
                failure_code="transport_not_open",
            )

    def _next_seq(self, channel_id: str) -> int:
        """Advance and return the sequence counter for a channel."""
        seq = self._sequence.get(channel_id, 0) + 1
        self._sequence[channel_id] = seq
        return seq


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _scalar(value: Any) -> Any:
    """Extract a Python scalar from a tensor/array if possible."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    return value


def _channel_base_name(channel_id: str) -> str:
    """Extract the base joint/device name from a channel_id.

    ``"joint.shoulder_pan.position"`` -> ``"shoulder_pan"``
    ``"gripper.left"`` -> ``"left"``
    """
    parts = channel_id.split(".")
    if len(parts) >= 3 and parts[0] == "joint":
        return parts[1]
    if len(parts) >= 2 and parts[0] in ("gripper", "camera"):
        return parts[1]
    return channel_id


def _channel_suffix(channel_id: str) -> str:
    """Extract the suffix (position/velocity) from a channel_id, or empty."""
    parts = channel_id.split(".")
    if len(parts) >= 3:
        return parts[-1]
    return ""


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def build_transport(config: Mapping[str, Any]) -> RobotArmTransport:
    """Factory function for the transport registry.

    Config keys:
    - robot_type (str): Robot type name (e.g. "koch_v1_1", "so100_follower")
    - robot_factory (callable): Optional factory callable for robot instantiation
    - robot_config (dict): Additional robot configuration passed to the robot constructor
    - default_sample_rate_hz (float): Default sampling rate for channel mapper
    - default_camera_fps (float): Default camera FPS for channel mapper

    The robot is NOT instantiated here -- only a factory callable is prepared.
    Actual instantiation happens in open().
    """
    robot_type = str(config.get("robot_type", ""))
    robot_config = dict(config.get("robot_config", {})) if isinstance(config.get("robot_config"), Mapping) else {}

    # Compat: aggregate top-level serial_port/cameras into robot_config so
    # configs written before the provider placed them inside robot_config
    # still reach the robot constructor.
    serial_port = config.get("serial_port")
    if serial_port is not None:
        robot_config.setdefault("serial_port", str(serial_port))
    cameras = config.get("cameras")
    if isinstance(cameras, Mapping):
        robot_config.setdefault("cameras", dict(cameras))

    default_sample_rate = float(
        config.get("default_sample_rate_hz", config.get("sample_rate_hz", 0.0)) or 0.0
    )
    default_camera_fps = float(config.get("default_camera_fps", 30.0) or 30.0)

    def _factory() -> Any:
        """Lazy-import robot SDK and construct the robot."""
        try:
            from leapflow.robot.base import make_robot  # type: ignore[import-untyped]
        except ImportError as exc:
            raise TransportError(
                "robot arm SDK is not installed; "
                "install it with: pip install leapflow[robot]",
                failure_code="robot_arm_not_installed",
            ) from exc
        kwargs: dict[str, Any] = {"robot_type": robot_type}
        if robot_config:
            kwargs.update(robot_config)
        return make_robot(**kwargs)

    mapper = RobotChannelMapper(
        default_sample_rate_hz=default_sample_rate,
        default_camera_fps=default_camera_fps,
    )
    return RobotArmTransport(robot_factory=_factory, channel_mapper=mapper)


__all__ = ["RobotArmTransport", "build_transport"]
