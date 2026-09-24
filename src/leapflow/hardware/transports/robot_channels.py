# Copyright (c) Alibaba, Inc. and its affiliates.
# Adapted from LeRobot (https://github.com/huggingface/lerobot).
"""Maps robot features to HCP channel declarations.

Robot arms describe their observation and action spaces as typed feature
dictionaries.  This module translates those declarations into HCP ``Channel``
instances so the rest of the hardware stack -- registry admission, streaming,
tools, trust, audit -- works without knowing anything about the specific
robot SDK.

This module has no hard dependencies on any specific robot SDK.  Nothing in
this file imports external robot packages at module level; the ``map_robot``
convenience method inspects a live robot instance through duck-typed attribute
access so that the mapper remains usable in environments where no robot SDK
is installed.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from leapflow.hardware.context import (
    Channel,
    Direction,
    Envelope,
    HardwareEffect,
    PrivacyTier,
    Representation,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic patterns for feature classification
# ---------------------------------------------------------------------------

_CAMERA_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|[._])cam(?:era)?(?:[._]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[._])image(?:[._]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[._])rgb(?:[._]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[._])depth(?:[._]|$)", re.IGNORECASE),
)

_GRIPPER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|[._])grip(?:per)?(?:[._]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[._])jaw(?:s)?(?:[._]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[._])tool(?:_?center)?(?:[._]|$)", re.IGNORECASE),
)

_VELOCITY_SUFFIX = re.compile(r"[._]velocity$", re.IGNORECASE)
_POSITION_SUFFIX = re.compile(r"[._]position$", re.IGNORECASE)

# Feature shapes with 3+ dimensions are likely image tensors.
_MIN_IMAGE_DIMS = 3


# ---------------------------------------------------------------------------
# Public helpers -- one channel factory per feature type
# ---------------------------------------------------------------------------


def infer_channel_type(feature_name: str, feature_shape: Any) -> str:
    """Infer whether a feature is motor/camera/gripper/other.

    Classification relies on the feature name and, when available, the
    dimensionality of its shape.  A shape with three or more dimensions
    (height x width x channels) is classified as camera regardless of name
    so that unnamed image observations are not lost.

    Returns one of ``"motor"``, ``"camera"``, ``"gripper"``, or ``"other"``.
    """
    name_lower = feature_name.lower()

    # Camera: name match or high-dimensional shape (image tensor).
    if any(p.search(feature_name) for p in _CAMERA_PATTERNS):
        return "camera"
    if _is_image_shape(feature_shape):
        return "camera"

    # Gripper / binary state.
    if any(p.search(feature_name) for p in _GRIPPER_PATTERNS):
        return "gripper"

    # Motor: anything with a position/velocity suffix, or a 1-D scalar
    # that is neither camera nor gripper.
    if _VELOCITY_SUFFIX.search(name_lower) or _POSITION_SUFFIX.search(name_lower):
        return "motor"

    # Fallback: scalar features are assumed to be motor-like joints when
    # they appear in the action space.  The caller may override this.
    if _is_scalar_shape(feature_shape):
        return "motor"

    return "other"


def motor_channel_pair(
    name: str,
    *,
    limits: tuple[float, float] | None = None,
    sample_rate_hz: float = 0.0,
) -> tuple[Channel, Channel]:
    """Return (position_channel, velocity_channel) for one motor.

    *name* is the bare joint name (e.g. ``"shoulder_pan"``), not the full
    feature key.  The returned channels carry LLM-readable descriptions so
    the agent can reason about the joint without external documentation.
    """
    pos_envelope = Envelope(
        declared=True,
        min_value=limits[0] if limits else None,
        max_value=limits[1] if limits else None,
        settling_time_s=0.1,
        reversible=True,
        notes=f"Angular position limits for joint '{name}'.",
    )
    pos_channel = Channel(
        channel_id=f"joint.{name}.position",
        direction=Direction.READWRITE.value,
        quantity="angular_position",
        unit="rad",
        effect=HardwareEffect.ACTUATE.value,
        representation=Representation.SCALAR.value,
        envelope=pos_envelope,
        sample_rate_hz=sample_rate_hz,
        verify_after_write=True,
        description=(
            f"Commanded angular position of the '{name}' joint in radians. "
            f"Writing to this channel actuates the motor to the target angle."
        ),
    )
    vel_channel = Channel(
        channel_id=f"joint.{name}.velocity",
        direction=Direction.READ.value,
        quantity="angular_velocity",
        unit="rad/s",
        effect=HardwareEffect.READ.value,
        representation=Representation.SCALAR.value,
        envelope=Envelope(declared=True, reversible=True),
        sample_rate_hz=sample_rate_hz,
        description=(
            f"Current angular velocity of the '{name}' joint in radians per "
            f"second. Read-only feedback from the motor encoder."
        ),
    )
    return pos_channel, vel_channel


def camera_channel(
    name: str,
    *,
    fps: float = 30.0,
    width: int = 0,
    height: int = 0,
) -> Channel:
    """Return one frame channel for a camera.

    The ``sample_rate_hz`` field carries the capture ceiling (frames per
    second) so that downstream consumers respect the hardware limit.
    """
    res_note = f" Resolution {width}x{height}." if width > 0 and height > 0 else ""
    return Channel(
        channel_id=f"camera.{name}",
        direction=Direction.READ.value,
        quantity="image",
        unit="frame",
        effect=HardwareEffect.READ.value,
        representation=Representation.FRAME.value,
        privacy=PrivacyTier.ENVIRONMENT.value,
        envelope=Envelope(declared=True),
        sample_rate_hz=fps,
        media_type="image/jpeg",
        description=(
            f"RGB image stream from the '{name}' camera at up to {fps:.0f} "
            f"fps.{res_note} Reading this channel observes the physical "
            f"environment."
        ),
    )


def gripper_channel(
    name: str,
    *,
    limits: tuple[float, float] | None = None,
) -> Channel:
    """Return one state channel for a gripper.

    When *limits* are not provided the envelope defaults to a binary
    open/close model with allowed values (0, 1).
    """
    if limits is not None:
        envelope = Envelope(
            declared=True,
            min_value=limits[0],
            max_value=limits[1],
            reversible=True,
            notes=f"Gripper '{name}' continuous range.",
        )
    else:
        envelope = Envelope(
            declared=True,
            allowed_values=(0, 1),
            reversible=True,
            notes=f"Gripper '{name}' binary open/close.",
        )
    return Channel(
        channel_id=f"gripper.{name}",
        direction=Direction.READWRITE.value,
        quantity="gripper_state",
        unit="",
        effect=HardwareEffect.ACTUATE.value,
        representation=Representation.STATE.value,
        envelope=envelope,
        verify_after_write=True,
        description=(
            f"State of the '{name}' gripper. Writing actuates the gripper; "
            f"reading returns its current position or open/close state."
        ),
    )


# ---------------------------------------------------------------------------
# Main mapper class
# ---------------------------------------------------------------------------


class RobotChannelMapper:
    """Extracts HCP channel declarations from robot arm configuration.

    Mapping rules
    -------------
    - Motor position feature -> 1 READWRITE channel (SCALAR, ACTUATE)
      ``channel_id = "joint.{name}.position"``
      quantity = ``"angular_position"``, unit = ``"rad"``
      Envelope: declared=True, min/max from motor limits if available,
      settling_time_s=0.1, reversible=True

    - Motor velocity feature -> 1 READ channel (SCALAR, READ)
      ``channel_id = "joint.{name}.velocity"``
      quantity = ``"angular_velocity"``, unit = ``"rad/s"``

    - Camera feature -> 1 READ channel (FRAME, READ)
      ``channel_id = "camera.{name}"``
      representation = FRAME, privacy_tier = ENVIRONMENT
      sample_rate_hz from camera config fps, or default 30

    - Gripper / binary state -> 1 READWRITE channel (STATE, ACTUATE)
      ``channel_id = "gripper.{name}"``
      Envelope: declared=True, allowed_values=(0, 1) or min/max
      reversible=True
    """

    def __init__(
        self,
        *,
        default_sample_rate_hz: float = 0.0,
        default_camera_fps: float = 30.0,
    ) -> None:
        self._default_sample_rate = default_sample_rate_hz
        self._default_camera_fps = default_camera_fps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map_features(
        self,
        observation_features: dict[str, Any],
        action_features: dict[str, Any],
        *,
        motor_limits: dict[str, tuple[float, float]] | None = None,
        camera_configs: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[Channel, ...]:
        """Return HCP channels derived from robot feature dictionaries.

        Observation and action features may overlap (e.g. the same joint
        appears in both).  Duplicate channel ids are deduplicated, with
        action features taking precedence because they carry write intent.
        """
        limits = motor_limits or {}
        cam_cfgs = camera_configs or {}
        channels: dict[str, Channel] = {}

        # Pass 1: observation features (read-only by default).
        for feat_name, feat_shape in observation_features.items():
            for ch in self._channels_for_feature(
                feat_name, feat_shape, is_action=False,
                motor_limits=limits, camera_configs=cam_cfgs,
            ):
                channels[ch.channel_id] = ch

        # Pass 2: action features override observations (promote to READWRITE).
        for feat_name, feat_shape in action_features.items():
            for ch in self._channels_for_feature(
                feat_name, feat_shape, is_action=True,
                motor_limits=limits, camera_configs=cam_cfgs,
            ):
                channels[ch.channel_id] = ch

        return tuple(channels.values())

    def map_robot(self, robot: Any) -> tuple[Channel, ...]:
        """Convenience: extract features from a connected Robot instance.

        Works by duck-typed attribute access so no specific robot SDK need
        be installed.  Raises ``TypeError`` when the object does not look like
        a robot with observation and action features.
        """
        obs = getattr(robot, "observation_features", None)
        act = getattr(robot, "action_features", None)
        if obs is None or act is None:
            raise TypeError(
                f"{type(robot).__name__} does not expose observation_features "
                "and action_features; expected a Robot instance"
            )
        if not isinstance(obs, dict) or not isinstance(act, dict):
            raise TypeError(
                "observation_features and action_features must be dicts, "
                f"got {type(obs).__name__} and {type(act).__name__}"
            )

        # Attempt to extract motor limits from the robot if available.
        motor_limits: dict[str, tuple[float, float]] | None = None
        raw_limits = getattr(robot, "motor_limits", None)
        if isinstance(raw_limits, dict):
            motor_limits = {
                str(k): (float(v[0]), float(v[1]))
                for k, v in raw_limits.items()
                if isinstance(v, (list, tuple)) and len(v) >= 2
            }

        # Attempt to extract camera configs.
        camera_configs: dict[str, dict[str, Any]] | None = None
        raw_cams = getattr(robot, "camera_configs", None)
        if isinstance(raw_cams, dict):
            camera_configs = {str(k): dict(v) for k, v in raw_cams.items() if isinstance(v, dict)}

        return self.map_features(
            obs, act,
            motor_limits=motor_limits,
            camera_configs=camera_configs,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _channels_for_feature(
        self,
        feature_name: str,
        feature_shape: Any,
        *,
        is_action: bool,
        motor_limits: dict[str, tuple[float, float]],
        camera_configs: dict[str, dict[str, Any]],
    ) -> list[Channel]:
        """Map a single feature to zero or more HCP channels."""
        kind = infer_channel_type(feature_name, feature_shape)
        base_name = _strip_suffixes(feature_name)

        if kind == "camera":
            cfg = camera_configs.get(base_name, {})
            fps = float(cfg.get("fps", self._default_camera_fps))
            width = int(cfg.get("width", 0))
            height = int(cfg.get("height", 0))
            return [camera_channel(base_name, fps=fps, width=width, height=height)]

        if kind == "gripper":
            lim = motor_limits.get(base_name)
            return [gripper_channel(base_name, limits=lim)]

        if kind == "motor":
            lim = motor_limits.get(base_name)
            pos_ch, vel_ch = motor_channel_pair(
                base_name,
                limits=lim,
                sample_rate_hz=self._default_sample_rate,
            )
            # Observation-only motors are demoted to read-only position.
            if not is_action:
                pos_ch = Channel(
                    channel_id=pos_ch.channel_id,
                    direction=Direction.READ.value,
                    quantity=pos_ch.quantity,
                    unit=pos_ch.unit,
                    effect=HardwareEffect.READ.value,
                    representation=pos_ch.representation,
                    envelope=pos_ch.envelope,
                    sample_rate_hz=pos_ch.sample_rate_hz,
                    description=pos_ch.description,
                )
            return [pos_ch, vel_ch]

        # Unknown feature type -- log and skip rather than crash.
        logger.debug(
            "Skipping unrecognised robot feature %r (shape=%r)",
            feature_name, feature_shape,
        )
        return []


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_image_shape(shape: Any) -> bool:
    """Return True when *shape* looks like an image tensor (>= 3 dims)."""
    if isinstance(shape, (list, tuple)) and len(shape) >= _MIN_IMAGE_DIMS:
        return True
    # numpy / torch shapes expose __len__.
    try:
        return len(shape) >= _MIN_IMAGE_DIMS
    except TypeError:
        return False


def _is_scalar_shape(shape: Any) -> bool:
    """Return True when *shape* describes a single scalar value."""
    if isinstance(shape, (list, tuple)):
        return len(shape) == 0 or shape == (1,) or shape == [1]
    try:
        return len(shape) <= 1
    except TypeError:
        # Bare int (e.g. shape=1) is scalar.
        return isinstance(shape, int) and shape <= 1


def _strip_suffixes(name: str) -> str:
    """Remove common trailing qualifiers to derive a base joint/device name.

    ``"shoulder_pan.position"`` -> ``"shoulder_pan"``
    ``"left_gripper_velocity"`` -> ``"left_gripper"``
    """
    for suffix in (".position", ".velocity", "_position", "_velocity"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


__all__ = [
    "RobotChannelMapper",
    "camera_channel",
    "gripper_channel",
    "infer_channel_type",
    "motor_channel_pair",
]
