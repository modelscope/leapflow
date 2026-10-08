# Copyright (c) Alibaba, Inc. and its affiliates.
# Adapted from LeRobot (https://github.com/huggingface/lerobot).
"""Robot arm context provider for LeapRobot: discovers robot arms from configuration.

A robot arm is described by its type (e.g. ``koch_v1_1``,
``so100_follower``) and its physical connections (serial ports, camera
indices).  This provider reads that configuration from the hardware
providers YAML block and produces a fully declared ``HardwareContext``
with channel declarations derived from the robot's feature description.

The robot is not instantiated by the provider -- only described.  The
transport factory in ``transports/robot_arm.py`` handles instantiation
during ``open()``.  This separation means ``discover()`` can run without
hardware attached, which is what makes offline ``hw_describe`` and test
assembly work.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping

from leapflow.hardware.context import (
    ContextProvenance,
    ContextSource,
    HardwareContext,
    TransportRef,
)
from leapflow.hardware.transports.robot_channels import RobotChannelMapper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device-class inference patterns
# ---------------------------------------------------------------------------

_ARM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"follower", re.IGNORECASE),
    re.compile(r"leader", re.IGNORECASE),
    re.compile(r"koch", re.IGNORECASE),
    re.compile(r"so10[01]", re.IGNORECASE),
    re.compile(r"aloha", re.IGNORECASE),
    re.compile(r"widow", re.IGNORECASE),
    re.compile(r"viper", re.IGNORECASE),
    re.compile(r"panda", re.IGNORECASE),
)

_MOBILE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"kiwi", re.IGNORECASE),
    re.compile(r"rover", re.IGNORECASE),
    re.compile(r"mobile", re.IGNORECASE),
)

_HUMANOID_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bg[1-9]\b", re.IGNORECASE),
    re.compile(r"\bh[1-9]\b", re.IGNORECASE),
    re.compile(r"reachy", re.IGNORECASE),
    re.compile(r"humanoid", re.IGNORECASE),
)


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------


class RobotArmContextProvider:
    """HardwareContextProvider that discovers robot arms from configuration.

    Configuration example (in profile hardware YAML)::

        hardware:
          providers:
            - type: robot_arm
              config:
                robot_type: so100_follower
                device_name: my_robot_arm
                display_name: "SO-100 Follower Arm"
                serial_port: /dev/ttyUSB0
                cameras:
                  top:
                    index: 0
                    width: 640
                    height: 480
                    fps: 30
                motor_limits:
                  shoulder_pan: [-3.14, 3.14]
                  shoulder_lift: [-1.57, 1.57]
                  elbow: [-3.14, 3.14]
                  wrist_1: [-3.14, 3.14]
                  wrist_2: [-3.14, 3.14]
                  wrist_3: [-3.14, 3.14]
                sample_rate_hz: 50.0
    """

    kind: str = "robot_arm"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = dict(config)

    def discover(self) -> tuple[HardwareContext, ...]:
        """Return one HardwareContext per configured robot arm.

        The context contains:

        - device_id: ``robot.{device_name}`` or ``robot.{robot_type}``
        - display_name: from config or robot_type
        - vendor: from config ``robot_type`` (generic)
        - model: robot_type
        - device_class: ``"robot_arm"`` (or inferred from type)
        - channels: from :meth:`RobotChannelMapper.map_features`
          (using motor_limits and camera configs from config)
        - transport_ref: ``TransportRef(kind="robot_arm", config={...})``
        - provenance: :attr:`ContextSource.IMPORTED`, notes about robot arm
        - halt_supported: ``True`` (zero-velocity stop)
        """
        robot_type = str(self._config.get("robot_type") or "").strip()
        if not robot_type:
            logger.warning("robot_arm provider: 'robot_type' is required; skipping")
            return ()

        device_name = str(
            self._config.get("device_name") or robot_type
        ).strip()
        device_id = f"robot.{device_name}"
        display_name = str(
            self._config.get("display_name") or robot_type
        ).strip()

        sample_rate_hz = float(self._config.get("sample_rate_hz") or 0.0)

        # Build synthetic feature dicts from configuration so the channel
        # mapper can produce declarations without a live robot.
        observation_features = _build_observation_features(self._config)
        action_features = _build_action_features(self._config)

        # Parse motor limits from config.
        motor_limits = _parse_motor_limits(self._config.get("motor_limits"))

        # Parse camera configs.
        camera_configs = _parse_camera_configs(self._config.get("cameras"))

        mapper = RobotChannelMapper(
            default_sample_rate_hz=sample_rate_hz,
            default_camera_fps=30.0,
        )
        channels = mapper.map_features(
            observation_features,
            action_features,
            motor_limits=motor_limits,
            camera_configs=camera_configs,
        )

        if not channels:
            logger.warning(
                "robot_arm provider: no channels derived for robot_type=%r; "
                "check motor_limits and cameras config",
                robot_type,
            )
            return ()

        # Build the transport config carrying everything the transport
        # factory needs to instantiate the robot at open() time.
        transport_config: dict[str, Any] = {
            "robot_type": robot_type,
            "robot_config": {},
        }
        if self._config.get("serial_port"):
            transport_config["robot_config"]["serial_port"] = str(self._config["serial_port"])
        if self._config.get("cameras"):
            transport_config["robot_config"]["cameras"] = dict(self._config["cameras"])
        if self._config.get("sample_rate_hz"):
            transport_config["default_sample_rate_hz"] = sample_rate_hz

        context = HardwareContext(
            device_id=device_id,
            display_name=display_name,
            device_class=_infer_device_class(robot_type),
            vendor=robot_type,
            model=robot_type,
            transport=TransportRef(kind="robot_arm", config=transport_config),
            channels=channels,
            halt_supported=True,
            notes=(
                f"Robot arm '{robot_type}' discovered from provider "
                f"configuration. Transport instantiates the robot on open()."
            ),
            provenance=ContextProvenance(
                source=ContextSource.IMPORTED.value,
                notes=(
                    "Mapped from robot type descriptor. Motor limits "
                    "and camera parameters are declared in the provider config, "
                    "not probed from hardware."
                ),
            ),
        )
        return (context,)


# ---------------------------------------------------------------------------
# Feature synthesis helpers
# ---------------------------------------------------------------------------


def _build_observation_features(config: Mapping[str, Any]) -> dict[str, Any]:
    """Synthesize observation features from provider config.

    When a real robot instance is not available (offline discovery),
    build a feature dictionary from the configured motor names and camera
    entries so the channel mapper can still produce declarations.

    Observation features include all motor positions (read) and all cameras.
    """
    features: dict[str, Any] = {}

    # Motor positions from motor_limits keys.
    motor_limits = config.get("motor_limits")
    if isinstance(motor_limits, Mapping):
        for motor_name in motor_limits:
            features[f"{motor_name}.position"] = (1,)

    # Camera observations from camera config.
    cameras = config.get("cameras")
    if isinstance(cameras, Mapping):
        for cam_name, cam_cfg in cameras.items():
            if isinstance(cam_cfg, Mapping):
                w = int(cam_cfg.get("width", 640))
                h = int(cam_cfg.get("height", 480))
                features[f"{cam_name}.image"] = (h, w, 3)
            else:
                features[f"{cam_name}.image"] = (480, 640, 3)

    return features


def _build_action_features(config: Mapping[str, Any]) -> dict[str, Any]:
    """Synthesize action features from provider config.

    Action features include all motor positions (write) -- the same joints
    that appear in observations but marked as actionable so the channel
    mapper promotes them to READWRITE.
    """
    features: dict[str, Any] = {}

    motor_limits = config.get("motor_limits")
    if isinstance(motor_limits, Mapping):
        for motor_name in motor_limits:
            features[f"{motor_name}.position"] = (1,)

    return features


def _parse_motor_limits(
    raw: Any,
) -> dict[str, tuple[float, float]] | None:
    """Parse motor limits from config into the format expected by the mapper."""
    if not isinstance(raw, Mapping):
        return None
    limits: dict[str, tuple[float, float]] = {}
    for name, bounds in raw.items():
        if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
            try:
                limits[str(name)] = (float(bounds[0]), float(bounds[1]))
            except (ValueError, TypeError):
                logger.warning(
                    "robot_arm provider: invalid motor limits for %r, skipping",
                    name,
                )
    return limits if limits else None


def _parse_camera_configs(
    raw: Any,
) -> dict[str, dict[str, Any]] | None:
    """Parse camera configs from the provider config block."""
    if not isinstance(raw, Mapping):
        return None
    configs: dict[str, dict[str, Any]] = {}
    for name, cfg in raw.items():
        if isinstance(cfg, Mapping):
            configs[str(name)] = dict(cfg)
    return configs if configs else None


def _infer_device_class(robot_type: str) -> str:
    """Infer the device class from the robot type name.

    Known patterns:

    - ``*follower*``, ``*leader*``, ``*koch*``, ``*so100*``, ``*so101*``
      -> ``"robot_arm"``
    - ``*kiwi*``, ``*rover*`` -> ``"mobile_robot"``
    - ``*g1*``, ``*h1*``, ``*reachy*`` -> ``"humanoid"``
    - otherwise -> ``"robot"``
    """
    if any(p.search(robot_type) for p in _ARM_PATTERNS):
        return "robot_arm"
    if any(p.search(robot_type) for p in _MOBILE_PATTERNS):
        return "mobile_robot"
    if any(p.search(robot_type) for p in _HUMANOID_PATTERNS):
        return "humanoid"
    return "robot"


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def build_provider(config: Mapping[str, Any]) -> RobotArmContextProvider:
    """Factory function for the provider registry."""
    return RobotArmContextProvider(config)


__all__ = ["RobotArmContextProvider", "build_provider"]
