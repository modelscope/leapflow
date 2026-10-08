# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hardware context plugin -- a ToolPlugin, not a sibling subsystem.

Being an ordinary ``ToolPlugin`` is a deliberate structural choice, for two
reasons.

It inherits the whole engineering surface for free: discovery, topological
dependency injection, single-pass assembly, fiber lifecycle, hot reload,
sandboxing, manifest signing, the trust ledger, and usage tracking. None of it is
reimplemented here.

More importantly, it puts hardware on the governed path. Tools registered as
plugin metadata carry ``x_leapflow`` and therefore reach PCD disclosure and the
approval chain; a device exposed by bypassing that would execute physical commands
with no risk classification and no audit record. Reusing the plugin philosophy is
the governance decision, not a convenience.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from leapflow.plugins.protocol import ToolMetadata

logger = logging.getLogger(__name__)


class HardwareContextPlugin:
    """Exposes admitted hardware devices through the twelve generic tools."""

    def __init__(self) -> None:
        self._registry: Any = None
        self._gate: Any = None
        self._scope: Any = None
        self._session_id: str = ""
        self._tools: Any = None
        self._hw_tools: Any = None
        self._teardown_registered: bool = False
        self._hardware_trust_gate: Any = None
        self._event_bus: Any = None

        # LHPGateway: bridges hardware state to/from LLM context.
        self._lhp_gateway: Any = None
        # FleetManager: multi-node robot coordination.
        self._fleet_manager: Any = None
        self._fleet_tools: list[ToolMetadata] | None = None
        # HardwareHealthMonitor: centralized health supervision.
        self._health_monitor: Any = None

    @property
    def plugin_id(self) -> str:
        return "hardware_context"

    @property
    def category(self) -> str:
        return "hardware"

    @property
    def dependencies(self) -> list[str]:
        return [
            "hardware_registry",
            "hardware_approval_gate",
            "hardware_trust_gate",
            "effect_scope",
            "session_id",
            "event_bus",
        ]

    def bind_runtime(self, **deps: Any) -> None:
        """Receive the registry, the gate, and the scope that owns teardown.

        The registry is optional on purpose: with hardware disabled nothing binds
        it, ``tools`` stays empty, and the tool index is byte-identical to a build
        without this plugin. That property is what keeps the feature default-off and
        reversible, and it is also what keeps journey cassettes valid.

        The gate may be re-bound after assembly (daemon ``install_gate`` path).
        When that happens the *live* ``HardwareTools`` instance is patched
        in-place so that already-registered tool handlers see the new
        orchestrator without requiring re-assembly.
        """
        registry_changed = False
        if "hardware_registry" in deps:
            self._registry = deps.get("hardware_registry")
            registry_changed = True
        if "hardware_approval_gate" in deps:
            self._gate = deps.get("hardware_approval_gate")
            # Late-bind into the live HardwareTools so already-registered
            # handlers resolve to the new gate without re-assembly.
            if self._hw_tools is not None:
                self._hw_tools.set_gate(self._gate)
        if "hardware_trust_gate" in deps:
            self._hardware_trust_gate = deps.get("hardware_trust_gate")
        if "session_id" in deps:
            self._session_id = str(deps.get("session_id") or "")
        if "effect_scope" in deps:
            self._scope = deps.get("effect_scope")
            self._teardown_registered = False
        if "event_bus" in deps:
            self._event_bus = deps.get("event_bus")

        if registry_changed:
            self._tools = None
            self._hw_tools = None
            self._lhp_gateway = None
            self._fleet_manager = None
            self._fleet_tools = None
            self._health_monitor = None
            self._teardown_registered = False

        if self._registry is None:
            return

        self._register_trust_configs()
        self._register_teardown()
        self._init_lhp_gateway()
        self._init_fleet_manager()
        self._init_health_monitor()

    def _register_trust_configs(self) -> None:
        """Register per-device TrustConfig with the trust gate after admission.

        Idempotent: ``register_device`` is a no-op for an already-known device.
        Errors are swallowed per device so one bad config does not prevent the
        rest from being registered.
        """
        gate = self._hardware_trust_gate
        if gate is None or not hasattr(gate, "register_device"):
            return
        for ctx in self._registry.contexts():
            try:
                gate.register_device(ctx.device_id, ctx.trust_config)
            except Exception as exc:  # noqa: BLE001 - registration must not block startup
                logger.warning(
                    "Could not register trust config for %s: %s",
                    ctx.device_id,
                    exc,
                )

    def _register_teardown(self) -> None:
        """Close device connections when the owning scope unwinds.

        Registered through ``async_effect`` rather than ``effect``: ``close_all`` is
        a coroutine, and a coroutine handed to the synchronous variant is dropped
        without being awaited -- the connections would simply stay open.

        Guarded against double-registration: a re-bind that only updates the gate
        must not append a second teardown effect for the same registry.
        """
        if self._teardown_registered:
            return
        if self._scope is None:
            return
        register = getattr(self._scope, "async_effect", None)
        if register is None:
            logger.warning(
                "Hardware plugin received an effect scope without async_effect; "
                "device connections will not be closed on teardown"
            )
            return
        try:
            register(self._teardown)
            self._teardown_registered = True
        except (RuntimeError, ValueError) as exc:
            logger.warning("Could not register hardware teardown effect: %s", exc, exc_info=True)

    async def _teardown(self) -> None:
        """Shut down all hardware sub-components on scope unwind."""
        # Health monitor must stop before registry closes connections.
        if self._health_monitor is not None:
            try:
                await self._health_monitor.stop()
            except Exception:  # noqa: BLE001
                logger.debug("Health monitor stop failed", exc_info=True)

        # Fleet manager must stop before registry closes connections.
        if self._fleet_manager is not None:
            try:
                await self._fleet_manager.close()
            except Exception:  # noqa: BLE001
                logger.debug("Fleet manager close failed", exc_info=True)

        # Registry close_all.
        await self._registry.close_all()

    # -- LHPGateway initialization -----------------------------------------

    def _init_lhp_gateway(self) -> None:
        """Create LHPGateway when registry is available.

        The gateway is used by the ``hw_context_snapshot`` tool to provide
        structured hardware context to the LLM.
        """
        if self._lhp_gateway is not None:
            return
        if self._registry is None:
            return

        try:
            from leapflow.hardware.lhp_gateway import LHPGateway

            cap_index = getattr(self._registry, "_capability_index", None)
            env_source = None  # will be set when PhysicalEnvironmentSource starts

            self._lhp_gateway = LHPGateway(
                self._registry,
                capability_index=cap_index,
                environment_source=env_source,
            )
            logger.debug("LHPGateway initialized")
        except Exception:  # noqa: BLE001
            logger.debug("Failed to initialize LHPGateway", exc_info=True)

    # -- FleetManager initialization ---------------------------------------

    def _init_fleet_manager(self) -> None:
        """Create FleetManager when fleet nodes are configured.

        Fleet configuration is read from:
        1. Environment variable LEAPFLOW_FLEET_NODES (JSON array)
        2. Settings hardware.fleet_nodes (list of node configs)

        When no fleet nodes are configured, FleetManager is not created
        and no fleet tools appear in the tool index.
        """
        if self._fleet_manager is not None:
            return
        if self._registry is None:
            return

        fleet_configs = self._load_fleet_config()
        if not fleet_configs:
            return

        try:
            from leapflow.hardware.fleet import FleetManager, FleetNode

            self._fleet_manager = FleetManager(self._registry)

            # Store node configs for deferred async registration.
            # Actual node registration happens when tools are first accessed
            # or when an async caller explicitly starts the fleet.
            self._pending_fleet_nodes: list[Any] = []
            for node_cfg in fleet_configs:
                try:
                    node = FleetNode(
                        node_id=str(node_cfg.get("node_id", "")),
                        display_name=str(node_cfg.get("display_name", "")),
                        mcp_endpoint=str(node_cfg.get("mcp_endpoint", "")),
                    )
                    if node.node_id and node.mcp_endpoint:
                        self._pending_fleet_nodes.append(node)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Fleet: failed to parse node config %r: %s",
                        node_cfg, exc,
                    )

            logger.info(
                "FleetManager initialized with %d configured node(s)",
                len(fleet_configs),
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to initialize FleetManager", exc_info=True)

    async def _register_fleet_node_safe(self, node: Any) -> None:
        """Register a fleet node, swallowing errors."""
        try:
            await self._fleet_manager.register_node(node)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Fleet: failed to register node %s: %s",
                node.node_id, exc,
            )

    @staticmethod
    def _load_fleet_config() -> list[dict[str, Any]]:
        """Load fleet node configuration from env var or settings."""
        # 1. Environment variable (JSON array of node configs).
        env_raw = os.environ.get("LEAPFLOW_FLEET_NODES", "")
        if env_raw:
            try:
                configs = json.loads(env_raw)
                if isinstance(configs, list):
                    return [c for c in configs if isinstance(c, dict)]
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "LEAPFLOW_FLEET_NODES env var is not valid JSON; ignoring"
                )

        # 2. Settings-based config is handled by the caller that has
        #    access to Settings; this static method only covers env.
        return []

    # -- HardwareHealthMonitor initialization ------------------------------

    def _init_health_monitor(self) -> None:
        """Create HardwareHealthMonitor when registry is available.

        The monitor start is deferred until an async context is available
        (e.g. during daemon startup or first tool invocation).
        """
        if self._health_monitor is not None:
            return
        if self._registry is None:
            return

        try:
            from leapflow.hardware.health_monitor import HardwareHealthMonitor

            self._health_monitor = HardwareHealthMonitor(
                self._registry,
                event_bus=self._event_bus,
                fleet_manager=self._fleet_manager,
            )
            self._health_monitor_started = False

            logger.debug("HardwareHealthMonitor initialized (start deferred)")
        except Exception:  # noqa: BLE001
            logger.debug(
                "Failed to initialize HardwareHealthMonitor", exc_info=True
            )

    # -- Tool property (hw_context_snapshot + fleet tools) -----------------

    @property
    def tools(self) -> list[ToolMetadata]:
        """Return the tool set, empty until a registry is bound."""
        if self._registry is None:
            return []
        if self._tools is None:
            from leapflow.hardware.tools import HardwareTools, build_hardware_tools

            hw = HardwareTools(
                self._registry,
                gate=self._gate,
                session_id=self._session_id,
                hardware_trust_gate=self._hardware_trust_gate,
            )
            self._hw_tools = hw
            self._tools = build_hardware_tools(hw)

            # Append LHPGateway tool (hw_context_snapshot).
            if self._lhp_gateway is not None:
                self._tools.append(self._build_context_snapshot_tool())

            # Append fleet tools when FleetManager is available.
            if self._fleet_manager is not None:
                self._fleet_tools = self._build_fleet_tools()
                self._tools.extend(self._fleet_tools)

            # Append health monitor tool.
            if self._health_monitor is not None:
                self._tools.append(self._build_health_tool())

        return list(self._tools)

    def _build_context_snapshot_tool(self) -> ToolMetadata:
        """Build the hw_context_snapshot tool backed by LHPGateway."""
        async def _hw_context_snapshot(**kwargs: Any) -> dict[str, Any]:
            level = kwargs.get("level", "task_relevant")
            if self._lhp_gateway is None:
                return {"ok": False, "error": "LHPGateway not initialized"}
            try:
                snapshot = await self._lhp_gateway.snapshot(level)
                return {
                    "ok": True,
                    "text": snapshot.to_prompt_text(),
                    "data": snapshot.to_dict(),
                    "token_estimate": snapshot.token_estimate,
                }
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}

        return ToolMetadata(
            name="hw_context_snapshot",
            description=(
                "Get a structured snapshot of all hardware devices and their "
                "state. Use this before planning physical manipulation tasks "
                "to understand what devices are available and what they can do."
            ),
            handler=_hw_context_snapshot,
            parameters_schema={
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "enum": ["minimal", "task_relevant", "rich_context"],
                        "description": (
                            "Detail level: minimal=fast/cheap, "
                            "task_relevant=standard, rich_context=complete"
                        ),
                    },
                },
            },
            x_leapflow={
                "category": "hardware",
                "risk_level": "low",
                "mutates_state": False,
            },
        )

    def _build_fleet_tools(self) -> list[ToolMetadata]:
        """Build fleet management tools backed by FleetManager."""
        if self._fleet_manager is None:
            return []

        try:
            from leapflow.hardware.fleet import build_fleet_tools
            return build_fleet_tools(self._fleet_manager)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to build fleet tools", exc_info=True)
            return []

    def _build_health_tool(self) -> ToolMetadata:
        """Build the hw_health tool backed by HardwareHealthMonitor."""
        async def _hw_health(**kwargs: Any) -> dict[str, Any]:
            device_id = kwargs.get("device_id", "")
            if self._health_monitor is None:
                return {"ok": False, "error": "health monitor not initialized"}
            if device_id:
                return {
                    "ok": True,
                    "health": self._health_monitor.device_health(device_id),
                }
            return {"ok": True, "health": self._health_monitor.fleet_health()}

        return ToolMetadata(
            name="hw_health",
            description=(
                "Check hardware device health status. Without device_id, "
                "returns fleet-wide health summary. With device_id, returns "
                "health details for that specific device."
            ),
            handler=_hw_health,
            parameters_schema={
                "type": "object",
                "properties": {
                    "device_id": {
                        "type": "string",
                        "description": "Optional: specific device to check",
                    },
                },
            },
            x_leapflow={
                "category": "hardware",
                "risk_level": "low",
                "mutates_state": False,
            },
        )


plugin = HardwareContextPlugin()


__all__ = ["HardwareContextPlugin", "plugin"]
