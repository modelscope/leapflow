# Copyright (c) Alibaba, Inc. and its affiliates.
"""Robot base protocol and configuration for LeapRobot.

Defines the structural ``Robot`` Protocol that any physical robot must
satisfy to be usable with LeapFlow's hardware stack.  The Protocol
replaces the original ABC in accordance with LeapFlow's
"Protocol over ABC" principle.

Also provides a minimal ``RobotConfig`` frozen dataclass that replaces
the draccus-based configuration from the upstream project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, runtime_checkable

from typing import Protocol

from leapflow.robot.types import RobotAction, RobotObservation


# ---------------------------------------------------------------------------
# Robot Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Robot(Protocol):
    """Protocol for physical robot devices.

    Any object satisfying this protocol can be used with LeapFlow's
    hardware stack.  The seven core methods map directly to HCP
    Transport operations:

    - ``connect`` / ``disconnect`` → Transport ``open`` / ``close``
    - ``get_observation`` → Transport ``read`` / ``read_batch``
    - ``send_action`` → Transport ``write`` / ``write_batch``
    - ``calibrate`` → calibration lifecycle hook

    Properties ``observation_features`` and ``action_features`` declare the
    schema so the channel mapper can build the HCP channel set without
    connecting to hardware first.
    """

    @property
    def name(self) -> str:
        """Unique robot type identifier (e.g. ``"so100"``, ``"koch_v1.1"``)."""
        ...

    @property
    def observation_features(self) -> dict[str, Any]:
        """Structure and types of observations produced by the robot.

        Keys match what ``get_observation`` returns.  Values are either a
        Python type for scalar values or a shape tuple for array values.
        Must be callable regardless of connection state.
        """
        ...

    @property
    def action_features(self) -> dict[str, Any]:
        """Structure and types of actions expected by the robot.

        Keys match what ``send_action`` expects.  Values follow the same
        convention as ``observation_features``.  Must be callable
        regardless of connection state.
        """
        ...

    @property
    def is_connected(self) -> bool:
        """Whether the robot is currently connected."""
        ...

    def connect(self, calibrate: bool = True) -> None:
        """Establish communication with the robot.

        Args:
            calibrate: If ``True`` (default), automatically calibrate after
                connecting when the hardware requires it.
        """
        ...

    def disconnect(self) -> None:
        """Disconnect from the robot and release resources."""
        ...

    def get_observation(self) -> RobotObservation:
        """Retrieve the current observation from the robot.

        Returns:
            Flat dictionary whose structure matches
            :pyattr:`observation_features`.
        """
        ...

    def send_action(self, action: RobotAction) -> RobotAction:
        """Send an action command to the robot.

        Args:
            action: Dictionary of actuator commands whose structure
                matches :pyattr:`action_features`.

        Returns:
            The action actually applied (may be clipped by safety limits).
        """
        ...

    def calibrate(self) -> None:
        """Calibrate the robot if applicable; no-op otherwise."""
        ...


# ---------------------------------------------------------------------------
# RobotConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RobotConfig:
    """Minimal robot configuration replacing draccus-based config.

    This is intentionally lean — concrete robot implementations should
    subclass or compose their own config with additional fields.
    """

    robot_type: str
    """Registered robot type name (e.g. ``"so100"``, ``"moss_v1"``)."""

    id: str = "default"
    """Instance identifier distinguishing multiple robots of the same type."""

    calibration_dir: Path | None = None
    """Directory for calibration files.  ``None`` uses the default location."""

    device: str = "cpu"
    """Target device for tensor operations (``"cpu"``, ``"cuda"``, etc.)."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Vendor-specific or user-supplied configuration key-values."""


# ---------------------------------------------------------------------------
# Robot type registry
# ---------------------------------------------------------------------------

_ROBOT_TYPE_REGISTRY: dict[str, Any] = {}
"""Global mapping of robot type name to a callable that returns a Robot.

Factory callables receive ``(**kwargs) -> Robot``.  Registration happens
either explicitly via :func:`register_robot_type` or through lazy
discovery from ``leapflow.robot.drivers``.
"""

_DISCOVERY_DONE: bool = False


def register_robot_type(name: str, factory: Any) -> None:
    """Register a factory callable for a robot type name.

    The factory must accept keyword arguments and return an object
    satisfying the :class:`Robot` Protocol.
    """
    _ROBOT_TYPE_REGISTRY[name] = factory


def _discover_drivers() -> None:
    """One-shot import of known driver packages to populate the registry.

    Each driver module can call :func:`register_robot_type` at import time
    or expose a ``ROBOT_TYPES`` dict mapping type names to factory callables.
    Discovery is best-effort: an import failure skips the driver quietly
    because hardware dependencies are optional.
    """
    global _DISCOVERY_DONE
    if _DISCOVERY_DONE:
        return
    _DISCOVERY_DONE = True

    _driver_modules = (
        "leapflow.robot.drivers.feetech",
    )
    import importlib
    import logging

    _logger = logging.getLogger(__name__)
    for mod_path in _driver_modules:
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            _logger.debug("Robot driver %s not importable (optional dependency).", mod_path)
            continue
        # Convention: a driver may expose ROBOT_TYPES = {"name": callable}.
        types_map = getattr(mod, "ROBOT_TYPES", None)
        if isinstance(types_map, dict):
            for type_name, factory_fn in types_map.items():
                _ROBOT_TYPE_REGISTRY.setdefault(type_name, factory_fn)


def make_robot(robot_type: str, **kwargs: Any) -> Any:
    """Construct a robot instance by type name.

    This is the LeapRobot factory entry point that replaces the upstream
    ``make_robot`` from the open-source robot SDK.

    Lookup order:

    1. Explicit registrations via :func:`register_robot_type`.
    2. Lazy discovery from ``leapflow.robot.drivers.*`` sub-packages.
    3. A ``robot_factory`` key in *kwargs* (escape hatch for custom robots).

    Raises ``NotImplementedError`` only when no factory can be resolved.
    """
    # 1. Direct registry hit.
    factory = _ROBOT_TYPE_REGISTRY.get(robot_type)
    if factory is not None:
        return factory(**kwargs)

    # 2. Run driver auto-discovery and retry.
    _discover_drivers()
    factory = _ROBOT_TYPE_REGISTRY.get(robot_type)
    if factory is not None:
        return factory(**kwargs)

    # 3. Escape hatch: caller-supplied factory callable.
    custom_factory = kwargs.pop("robot_factory", None)
    if callable(custom_factory):
        return custom_factory(**kwargs)

    raise NotImplementedError(
        f"make_robot('{robot_type}'): no registered factory for this type.  "
        "Register one with register_robot_type(), declare ROBOT_TYPES in a "
        "driver module, or pass robot_factory in the transport config."
    )


__all__ = [
    "Robot",
    "RobotConfig",
    "make_robot",
    "register_robot_type",
]
