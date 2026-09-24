# Copyright (c) Alibaba, Inc. and its affiliates.
"""Control binding resolver: translates declarative interface bindings
into executable transport configurations.

When a hardware YAML declares ``control_bindings``, this module resolves
them to concrete ``TransportRef`` instances that the registry can use to
instantiate the appropriate transport.  The binding declaration is
transport-neutral — it says *what* to connect to, not *how* — and this
resolver maps the *what* to the *how*.

Supported binding types:

- ``local_bus``: serial port, CAN bus, or I2C — maps to ``robot_arm`` transport
- ``ros2_topics``: ROS2 topic/service/action — maps to ``ros2`` transport (when available)
- ``mcp_server``: MCP server endpoint — maps to ``mcp`` transport
- ``network``: TCP/UDP endpoint — maps to future network transport

When multiple bindings are declared, the resolver picks the highest-priority
available one (local_bus > mcp_server > ros2_topics > network), ensuring
the lowest-latency path is used when available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default priority — lower-latency transports are preferred.
# ---------------------------------------------------------------------------

_DEFAULT_PRIORITIES: Mapping[str, int] = {
    "local_bus": 100,
    "mcp_server": 80,
    "ros2_topics": 60,
    "network": 40,
}

# Binding type → transport kind that fulfils it.
_BINDING_TO_TRANSPORT: Mapping[str, str] = {
    "local_bus": "robot_arm",
    "ros2_topics": "ros2",
    "mcp_server": "mcp",
    "network": "network",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlBinding:
    """One declared control interface binding."""

    binding_type: str  # "local_bus", "ros2_topics", "mcp_server", "network"
    config: Mapping[str, Any]  # type-specific configuration
    priority: int = 0  # higher = preferred


@dataclass(frozen=True)
class ResolvedBinding:
    """A control binding resolved to a concrete transport configuration."""

    transport_ref: Any  # TransportRef
    binding: ControlBinding
    available: bool = True  # whether the transport kind is registered
    reason: str = ""  # why unavailable, if not available


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class ControlBindingResolver:
    """Resolves control_bindings declarations to TransportRef instances.

    Usage::

        resolver = ControlBindingResolver()
        bindings = resolver.parse(yaml_data.get("control_bindings", {}))
        resolved = resolver.resolve(bindings)
        if resolved and resolved.available:
            context = replace(context, transport=resolved.transport_ref)
    """

    # -- public API ---------------------------------------------------------

    def parse(self, raw: Mapping[str, Any]) -> tuple[ControlBinding, ...]:
        """Parse a control_bindings YAML section into typed bindings.

        Example YAML::

            control_bindings:
              local_bus:
                type: serial
                port: /dev/ttyUSB0
                baudrate: 1000000
              ros2_topics:
                joint_states: /robot/joint_states
                joint_commands: /robot/joint_commands
              mcp_server:
                url: "http://localhost:8080"

        Each top-level key is a binding type.  The value is a mapping of
        type-specific configuration.  An explicit ``priority`` key inside
        the config overrides the default ordering.
        """
        if not isinstance(raw, Mapping):
            return ()

        bindings: list[ControlBinding] = []
        for key, value in raw.items():
            key_str = str(key).strip()
            if not key_str or key_str not in _BINDING_TO_TRANSPORT:
                logger.debug(
                    "control_bindings: ignoring unknown binding type %r", key_str
                )
                continue
            if not isinstance(value, Mapping):
                logger.warning(
                    "control_bindings: expected a mapping for %r, got %s; skipping",
                    key_str,
                    type(value).__name__,
                )
                continue

            config = dict(value)
            # Allow an explicit priority override.
            explicit_priority = config.pop("priority", None)
            priority = (
                int(explicit_priority)
                if explicit_priority is not None
                else _DEFAULT_PRIORITIES.get(key_str, 0)
            )
            bindings.append(
                ControlBinding(
                    binding_type=key_str,
                    config=config,
                    priority=priority,
                )
            )

        # Sort descending by priority so the first match is best.
        bindings.sort(key=lambda b: b.priority, reverse=True)
        return tuple(bindings)

    def resolve(
        self,
        bindings: tuple[ControlBinding, ...],
        *,
        available_transports: frozenset[str] | None = None,
    ) -> ResolvedBinding | None:
        """Resolve bindings to the best available transport.

        Tries bindings in priority order (local_bus > mcp_server > ros2 > network).
        Returns the first one whose transport kind is available, or the first
        unavailable one (with ``available=False``) if none are available, or
        ``None`` if *bindings* is empty.
        """
        if not bindings:
            return None

        if available_transports is None:
            available_transports = self._discover_available_transports()

        first_unavailable: ResolvedBinding | None = None

        for binding in bindings:
            transport_kind = _BINDING_TO_TRANSPORT.get(binding.binding_type)
            if transport_kind is None:
                continue

            resolver = self._resolver_for(binding.binding_type)
            if resolver is None:
                continue

            transport_ref = resolver(binding.config)
            is_available = transport_kind in available_transports

            resolved = ResolvedBinding(
                transport_ref=transport_ref,
                binding=binding,
                available=is_available,
                reason="" if is_available else (
                    f"transport kind {transport_kind!r} is not registered"
                ),
            )

            if is_available:
                logger.debug(
                    "control_bindings: resolved %s -> transport %s",
                    binding.binding_type,
                    transport_kind,
                )
                return resolved

            if first_unavailable is None:
                first_unavailable = resolved

        # None available — return best-priority unavailable as advisory.
        return first_unavailable

    # -- private resolvers --------------------------------------------------

    def _resolver_for(self, binding_type: str) -> Any:
        """Return the resolver method for the given binding type, or None."""
        dispatch: dict[str, Any] = {
            "local_bus": self._resolve_local_bus,
            "ros2_topics": self._resolve_ros2,
            "mcp_server": self._resolve_mcp,
            "network": self._resolve_network,
        }
        return dispatch.get(binding_type)

    def _resolve_local_bus(self, config: Mapping[str, Any]) -> Any:
        """Map local_bus binding to robot_arm TransportRef.

        Supported sub-types:
        - ``serial`` → serial_port in robot_arm transport config
        - ``can`` → can_bus in robot_arm transport config
        - ``i2c`` → i2c in robot_arm transport config
        """
        from leapflow.hardware.context import TransportRef

        bus_type = str(config.get("type") or "serial").lower()
        transport_config: dict[str, Any] = {}

        if bus_type == "serial":
            transport_config["robot_config"] = {
                "serial_port": str(config.get("port") or ""),
            }
            baudrate = config.get("baudrate")
            if baudrate is not None:
                transport_config["robot_config"]["baudrate"] = int(baudrate)
        elif bus_type == "can":
            transport_config["robot_config"] = {
                "can_bus": str(config.get("interface") or ""),
                "can_channel": str(config.get("channel") or ""),
            }
            bitrate = config.get("bitrate")
            if bitrate is not None:
                transport_config["robot_config"]["can_bitrate"] = int(bitrate)
        elif bus_type == "i2c":
            transport_config["robot_config"] = {
                "i2c_bus": str(config.get("bus") or ""),
                "i2c_address": config.get("address", 0),
            }
        else:
            # Unknown sub-type — pass config through verbatim.
            transport_config["robot_config"] = dict(config)

        # Propagate robot_type if declared alongside.
        robot_type = config.get("robot_type")
        if robot_type:
            transport_config["robot_type"] = str(robot_type)

        return TransportRef(kind="robot_arm", config=transport_config)

    def _resolve_ros2(self, config: Mapping[str, Any]) -> Any:
        """Map ros2_topics binding to ros2 TransportRef.

        The config maps logical names to ROS2 topic paths::

            joint_states: /robot/joint_states
            joint_commands: /robot/joint_commands
        """
        from leapflow.hardware.context import TransportRef

        transport_config: dict[str, Any] = {"topics": dict(config)}

        # Hoist well-known keys that are transport-level rather than topic-level.
        for hoist_key in ("node_name", "namespace", "qos_depth"):
            value = config.get(hoist_key)
            if value is not None:
                transport_config[hoist_key] = value

        return TransportRef(kind="ros2", config=transport_config)

    def _resolve_mcp(self, config: Mapping[str, Any]) -> Any:
        """Map mcp_server binding to mcp TransportRef.

        Expected keys: ``url`` (required), plus optional auth/timeout.
        """
        from leapflow.hardware.context import TransportRef

        transport_config: dict[str, Any] = {}
        url = config.get("url")
        if url:
            transport_config["url"] = str(url)

        # Forward optional keys.
        for key in ("auth_token", "timeout_s", "headers"):
            value = config.get(key)
            if value is not None:
                transport_config[key] = value

        return TransportRef(kind="mcp", config=transport_config)

    def _resolve_network(self, config: Mapping[str, Any]) -> Any:
        """Map network binding to an appropriate transport.

        Produces a ``network`` transport kind.  The transport implementation
        is not yet available — the resolver emits a well-formed TransportRef
        so that ``available=False`` carries a complete description of what
        *would* be instantiated.
        """
        from leapflow.hardware.context import TransportRef

        transport_config: dict[str, Any] = {}
        for key in ("host", "port", "protocol", "timeout_s"):
            value = config.get(key)
            if value is not None:
                transport_config[key] = value

        return TransportRef(kind="network", config=transport_config)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _discover_available_transports() -> frozenset[str]:
        """Query the transport registry for currently available kinds.

        Uses a function-local import to avoid a module-level dependency on the
        transport sub-package (which may pull optional C extensions).
        """
        from leapflow.hardware.transports import available_transports

        return frozenset(available_transports())


__all__ = [
    "ControlBinding",
    "ControlBindingResolver",
    "ResolvedBinding",
]
