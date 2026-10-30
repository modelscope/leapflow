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

import logging
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
        # FleetManager: multi-node robot coordination.  Node configs are parsed
        # here; registration happens in ``start_fleet`` once a loop is live.
        self._fleet_manager: Any = None
        self._fleet_tools: list[ToolMetadata] | None = None
        self._pending_fleet_nodes: list[Any] = []
        self._fleet_started: bool = False
        # Managers orphaned by a registry re-bind; closed by the next async
        # startup path (see ``start_fleet``) or by ``_teardown``.
        self._stale_fleet_managers: list[Any] = []
        # SkillService: governed node skills over the fleet (hw_skill_*). Its
        # lifetime follows the fleet manager it wraps.
        self._skill_service: Any = None
        self._stale_skill_services: list[Any] = []
        # HardwareHealthMonitor: centralized health supervision.
        self._health_monitor: Any = None
        self._health_monitor_started: bool = False
        self._stale_health_monitors: list[Any] = []

    def _adopt_reload_state(self, previous: Any) -> None:
        """Take ownership of live hardware supervisors from a retiring instance.

        Plugin reload creates a new plugin object while the daemon's registry,
        fleet connections and heartbeat remain live. Recreating those resources
        would leave the old heartbeat orphaned and the fresh manager unstarted.
        Transfer only long-lived runtime objects; tool handlers are rebuilt from
        the fresh class after dependency injection.
        """
        if getattr(previous, "plugin_id", "") != self.plugin_id:
            return
        # Seed the shared registry before DI is replayed so the identity check in
        # ``bind_runtime`` does not misclassify this ownership transfer as a
        # registry replacement and clear the resources adopted below.
        self._registry = getattr(previous, "_registry", None)
        self._fleet_manager = getattr(previous, "_fleet_manager", None)
        self._fleet_started = bool(getattr(previous, "_fleet_started", False))
        self._pending_fleet_nodes = list(getattr(previous, "_pending_fleet_nodes", ()))
        self._stale_fleet_managers = list(getattr(previous, "_stale_fleet_managers", ()))
        self._skill_service = getattr(previous, "_skill_service", None)
        self._stale_skill_services = list(getattr(previous, "_stale_skill_services", ()))
        self._health_monitor = getattr(previous, "_health_monitor", None)
        self._health_monitor_started = bool(getattr(previous, "_health_monitor_started", False))
        self._stale_health_monitors = list(getattr(previous, "_stale_health_monitors", ()))

        # The retiring instance's root-scope teardown may still run later. It
        # must not close resources whose ownership has moved to this instance.
        previous._fleet_manager = None
        previous._stale_fleet_managers = []
        previous._skill_service = None
        previous._stale_skill_services = []
        previous._health_monitor = None
        previous._health_monitor_started = False
        previous._stale_health_monitors = []

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
            new_registry = deps.get("hardware_registry")
            # Identity, not presence: a plugin hot reload re-applies every
            # last-bound dep (``ScopedPluginRegistry.reload``) with the *same*
            # registry. Treating that as a change dropped the live, started
            # FleetManager for a fresh one nobody ever starts -- fleet tools
            # then saw an empty fleet while the orphan kept supervising.
            registry_changed = new_registry is not self._registry
            self._registry = new_registry
        if "hardware_approval_gate" in deps:
            self._gate = deps.get("hardware_approval_gate")
            # Late-bind into the live HardwareTools so already-registered
            # handlers resolve to the new gate without re-assembly.
            if self._hw_tools is not None:
                self._hw_tools.set_gate(self._gate)
            if self._skill_service is not None:
                self._skill_service.set_gate(self._gate)
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
            # The previous manager may own a live heartbeat task; stash it so
            # an async path can close it. Clearing the reference here alone
            # would orphan a supervisor that keeps probing the old registry.
            if self._fleet_manager is not None:
                self._stale_fleet_managers.append(self._fleet_manager)
            if self._skill_service is not None:
                self._stale_skill_services.append(self._skill_service)
            if self._health_monitor is not None and self._health_monitor_started:
                self._stale_health_monitors.append(self._health_monitor)
            self._tools = None
            self._hw_tools = None
            self._lhp_gateway = None
            self._fleet_manager = None
            self._skill_service = None
            self._fleet_tools = None
            self._pending_fleet_nodes = []
            self._fleet_started = False
            self._health_monitor = None
            self._health_monitor_started = False
            self._teardown_registered = False

        if self._registry is None:
            return

        set_alert_orchestrator = getattr(self._registry, "set_alert_orchestrator", None)
        if callable(set_alert_orchestrator):
            set_alert_orchestrator(self._gate)
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
        for monitor in (*self._stale_health_monitors, self._health_monitor):
            if monitor is None:
                continue
            try:
                await monitor.stop()
            except Exception:  # noqa: BLE001
                logger.debug("Health monitor stop failed", exc_info=True)
        self._stale_health_monitors = []
        self._health_monitor_started = False

        # Skill services stop renewing run leases first, so every node halts
        # what this host was running before the fleet goes away.
        await self._close_skill_services((*self._stale_skill_services, self._skill_service))
        self._stale_skill_services = []

        # Fleet managers must stop before registry closes connections. Stale
        # managers (orphaned by a registry re-bind) are closed here too: their
        # heartbeat otherwise keeps probing a registry nobody owns.
        for manager in (*self._stale_fleet_managers, self._fleet_manager):
            if manager is None:
                continue
            try:
                await manager.close()
            except Exception:  # noqa: BLE001
                logger.debug("Fleet manager close failed", exc_info=True)
        self._stale_fleet_managers = []

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

            self._lhp_gateway = LHPGateway(self._registry)
            logger.debug("LHPGateway initialized")
        except Exception:  # noqa: BLE001
            logger.debug("Failed to initialize LHPGateway", exc_info=True)

    # -- FleetManager initialization ---------------------------------------

    def _init_fleet_manager(self) -> None:
        """Create FleetManager when fleet nodes are configured.

        Node configuration arrives through ``hardware_fleet_nodes`` in Settings
        (with ``LEAPFLOW_FLEET_NODES`` retained as a compatibility fallback).
        When no nodes are configured, FleetManager is not created and no fleet
        tools appear in the tool index.
        """
        if self._fleet_manager is not None:
            return
        if self._registry is None:
            return

        from leapflow.hardware.fleet import FleetManager
        from leapflow.hardware.fleet_config import configured_nodes_from_settings

        nodes = configured_nodes_from_settings(self._registry.settings.fleet_nodes)
        if not nodes:
            return

        try:
            self._fleet_manager = FleetManager(
                self._registry, permit_issuer=self._load_permit_issuer()
            )
            # Registration is deferred to ``start_fleet``: it speaks LHP and
            # needs a running loop, while plugin assembly is synchronous.
            self._pending_fleet_nodes = list(nodes)
            logger.info("FleetManager initialized with %d configured node(s)", len(nodes))
        except Exception:  # noqa: BLE001
            logger.debug("Failed to initialize FleetManager", exc_info=True)

    def _load_permit_issuer(self) -> Any | None:
        """The host permit issuer from the profile key, or ``None`` without one.

        Never generates a key: a node only accepts permits from a key its
        operator installed, so the key is created deliberately with
        ``leap hw permit-key``. Without it, ``permit_required`` nodes refuse
        writes with a result that names that command.
        """
        raw = str(getattr(self._registry.settings, "permit_issuer_key_path", "") or "")
        if not raw:
            return None
        from pathlib import Path

        from leapflow.hardware.lhp.permit import PermitIssuer

        path = Path(raw)
        if not path.exists():
            return None
        try:
            return PermitIssuer.from_key_file(path, auto_generate=False)
        except (OSError, ValueError) as exc:
            logger.warning("Permit issuer key at %s is unusable: %s", path, exc)
            return None

    @staticmethod
    async def _close_skill_services(services: tuple[Any, ...]) -> None:
        for service in services:
            if service is None:
                continue
            try:
                await service.close()
            except Exception:  # noqa: BLE001 -- cleanup must not block teardown
                logger.debug("Skill service close failed", exc_info=True)

    async def _close_stale_fleet_managers(self) -> None:
        """Close managers orphaned by a registry re-bind.

        A re-bind replaces the FleetManager synchronously, but closing one is
        async (stop heartbeat, unregister devices). Stashing keeps the
        reference alive until this path closes it; otherwise the old
        heartbeat task probes the old registry forever.
        """
        await self._close_skill_services(tuple(self._stale_skill_services))
        self._stale_skill_services = []
        stale, self._stale_fleet_managers = self._stale_fleet_managers, []
        for manager in stale:
            try:
                await manager.close()
            except Exception:  # noqa: BLE001 -- stale cleanup must not block startup
                logger.debug("Stale fleet manager close failed", exc_info=True)

    async def start_fleet(self) -> None:
        """Start hardware supervisors and register configured fleet nodes.

        Deferred from ``bind_runtime`` because registration speaks LHP and
        needs an event loop while assembly is synchronous; the runtime calls
        this once its loop is live (``cli/context.py``, daemon-mode init).

        Best-effort per node: an unreachable board logs and is skipped --
        starting the host runtime must never depend on an edge node being
        powered.  P2 adds retry/backoff and keep-last-degraded recovery.
        Idempotent.  Also closes any manager orphaned by a registry re-bind.
        """
        await self._close_stale_fleet_managers()
        await self._close_stale_health_monitors()
        if self._fleet_started or self._fleet_manager is None:
            await self._start_health_monitor()
            return
        self._fleet_started = True
        from leapflow.hardware.fleet_config import register_nodes_safely

        registered = await register_nodes_safely(self._fleet_manager, self._pending_fleet_nodes)
        self._pending_fleet_nodes = []
        await self._start_health_monitor()
        if not registered:
            return
        # Fleet admission happens after the initial runtime bind, so register
        # the newly materialized remote contexts with the per-device trust gate.
        self._register_trust_configs()
        try:
            await self._fleet_manager.start_heartbeat()
        except Exception:  # noqa: BLE001 -- supervision must not fail startup
            logger.warning("Fleet: heartbeat failed to start", exc_info=True)

    async def _close_stale_health_monitors(self) -> None:
        """Stop monitors orphaned by a registry replacement."""
        stale, self._stale_health_monitors = self._stale_health_monitors, []
        for monitor in stale:
            try:
                await monitor.stop()
            except Exception:  # noqa: BLE001 -- stale cleanup must not block startup
                logger.debug("Stale health monitor stop failed", exc_info=True)

    async def _start_health_monitor(self) -> None:
        """Start health supervision even when no fleet nodes are configured."""
        if self._health_monitor is None or self._health_monitor_started:
            return
        try:
            await self._health_monitor.start()
        except Exception:  # noqa: BLE001 -- supervision must not fail startup
            logger.warning("Hardware health monitor failed to start", exc_info=True)
            return
        self._health_monitor_started = True

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
            logger.debug("Failed to initialize HardwareHealthMonitor", exc_info=True)

    # -- Tool property (hw_context_snapshot + fleet tools) -----------------

    @property
    def tools(self) -> list[ToolMetadata]:
        """Return the tool set, empty until a registry is bound."""
        if self._registry is None:
            return []
        if self._tools is None:
            from pathlib import Path

            from leapflow.hardware.audit import HardwareAuditLog
            from leapflow.hardware.tools import HardwareTools, build_hardware_tools

            audit_path = str(getattr(self._registry.settings, "audit_log_path", "") or "")
            audit_log = HardwareAuditLog(Path(audit_path)) if audit_path else None
            hw = HardwareTools(
                self._registry,
                gate=self._gate,
                session_id=self._session_id,
                audit_log=audit_log,
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
                self._tools.extend(self._build_skill_tools(audit_log))

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
                "to understand which devices are available and their state; "
                "hw_skill_list shows what each workcell can do."
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
            from leapflow.hardware.fleet_tools import build_fleet_tools

            return build_fleet_tools(self._fleet_manager)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to build fleet tools", exc_info=True)
            return []

    def _build_skill_tools(self, audit_log: Any) -> list[ToolMetadata]:
        """Build the ``hw_skill_*`` tools over the fleet's skill nodes.

        The service is reused across rebuilds: it owns live run leases, and a
        replacement would silently stop renewing them.
        """
        from leapflow.hardware.skill_tools import SkillService, build_skill_tools

        if self._skill_service is None:
            self._skill_service = SkillService(
                self._fleet_manager,
                gate=self._gate,
                audit_log=audit_log,
                session_id=self._session_id,
                share_frames=bool(getattr(self._registry.settings, "share_frames", False)),
            )
        return build_skill_tools(self._skill_service)

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
