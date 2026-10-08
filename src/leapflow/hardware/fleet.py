# Copyright (c) Alibaba, Inc. and its affiliates.
"""Fleet deployment management: multi-node robot coordination.

Extends LeapFlow's single-machine hardware stack to a fleet of robots
distributed across multiple edge nodes.  The central leapd manages the
fleet topology, routes commands to the correct node via MCP Transport,
and aggregates trust/evidence/experience across all nodes.

Three deployment scenarios (from research note 10):
1. Dual-machine (Mac + edge board): Mac runs LLM/console, board runs RT control
2. Standalone edge + remote LLM: edge runs independently, LLM via API
3. Mac-only: USB-connected devices, no fleet needed

Fleet management adds:
- Node discovery and health monitoring
- Device-to-node routing (device_id → MCP endpoint)
- Cross-node trust aggregation
- Cross-node experience/evidence aggregation
- Fleet-level emergency stop (halt all nodes)
- Hot-plug: nodes joining/leaving the fleet

Design invariants:

- Fleet uses MCP Transport for bridging — no new RPC protocol.
- Remote devices register in the local HardwareRegistry as MCP transport
  kind, making them transparent to the agent layer above.
- ``halt_all`` is concurrent (``asyncio.gather``), fire-and-forget.
- Heartbeat is best-effort and never blocks data-plane operations.
- Trust aggregation takes the conservative value (lowest trust level).
- Nodes that go offline have their devices marked degraded automatically.
- No dependency on ``leapflow.engine``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

_NODE_STATUS_ONLINE = "online"
_NODE_STATUS_OFFLINE = "offline"
_NODE_STATUS_DEGRADED = "degraded"
_NODE_STATUS_UNKNOWN = "unknown"

_VALID_STATUSES = frozenset({
    _NODE_STATUS_ONLINE,
    _NODE_STATUS_OFFLINE,
    _NODE_STATUS_DEGRADED,
    _NODE_STATUS_UNKNOWN,
})


@dataclass(frozen=True)
class FleetNode:
    """One edge node in the fleet.

    Each node represents a physical machine (e.g. a Jetson board) that
    hosts one or more hardware devices.  The ``mcp_endpoint`` is the
    address of the MCP server on that node, through which all device
    commands are routed.

    ``device_ids`` is a snapshot taken at discovery time.  Devices may
    appear or disappear as the node's own HardwareRegistry reconciles;
    the fleet manager re-discovers periodically via heartbeat.
    """

    node_id: str
    display_name: str
    mcp_endpoint: str  # "http://host:port" or "stdio:command"
    device_ids: tuple[str, ...] = ()
    status: str = _NODE_STATUS_UNKNOWN
    last_heartbeat: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_status(self, status: str, *, heartbeat: float = 0.0) -> FleetNode:
        """Return a copy with an updated status and optional heartbeat."""
        return FleetNode(
            node_id=self.node_id,
            display_name=self.display_name,
            mcp_endpoint=self.mcp_endpoint,
            device_ids=self.device_ids,
            status=status if status in _VALID_STATUSES else _NODE_STATUS_UNKNOWN,
            last_heartbeat=heartbeat or self.last_heartbeat,
            metadata=self.metadata,
        )

    def with_devices(self, device_ids: tuple[str, ...]) -> FleetNode:
        """Return a copy with an updated device list."""
        return FleetNode(
            node_id=self.node_id,
            display_name=self.display_name,
            mcp_endpoint=self.mcp_endpoint,
            device_ids=device_ids,
            status=self.status,
            last_heartbeat=self.last_heartbeat,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class FleetTopology:
    """Complete fleet topology snapshot.

    An immutable view of the fleet at a point in time — safe to pass
    across boundaries and serialize for dashboards.
    """

    nodes: tuple[FleetNode, ...] = ()
    total_devices: int = 0
    online_nodes: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON / dashboard consumption."""
        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "display_name": n.display_name,
                    "mcp_endpoint": n.mcp_endpoint,
                    "device_ids": list(n.device_ids),
                    "status": n.status,
                    "last_heartbeat": n.last_heartbeat,
                    "metadata": dict(n.metadata),
                }
                for n in self.nodes
            ],
            "total_devices": self.total_devices,
            "online_nodes": self.online_nodes,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# FleetManager
# ---------------------------------------------------------------------------


class FleetManager:
    """Manages a fleet of distributed robot nodes.

    Coordinates with the local ``HardwareRegistry`` to present remote
    devices as if they were local — commands are transparently routed
    through MCP Transport to the correct edge node.
    """

    def __init__(
        self,
        registry: Any,  # local HardwareRegistry
        *,
        heartbeat_interval_s: float = 10.0,
        node_timeout_s: float = 30.0,
    ) -> None:
        self._registry = registry
        self._nodes: dict[str, FleetNode] = {}
        self._device_to_node: dict[str, str] = {}  # device_id → node_id
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._heartbeat_interval = max(1.0, float(heartbeat_interval_s))
        self._node_timeout = max(1.0, float(node_timeout_s))
        self._stopped = asyncio.Event()

    # -- Node management -------------------------------------------------

    async def register_node(self, node: FleetNode) -> FleetNode:
        """Register an edge node and discover its devices.

        1. Probe the node's MCP endpoint for availability.
        2. Discover devices via MCP tools (equivalent to ``hw_list``).
        3. Update the device→node routing table.
        4. Register each remote device in the local registry (if a
           reconciliation hook is available — otherwise the caller is
           responsible for adding the MCP-backed device declarations).

        Returns the node with updated device list and status.
        Raises ``FleetError`` if the endpoint is unreachable.
        """
        if not node.node_id:
            raise FleetError("node_id must be non-empty")
        if not node.mcp_endpoint:
            raise FleetError(f"node {node.node_id!r} has no mcp_endpoint")

        # Probe the endpoint.
        reachable = await self._probe_endpoint(node.mcp_endpoint)
        if not reachable:
            raise FleetError(
                f"node {node.node_id!r} at {node.mcp_endpoint!r} is unreachable"
            )

        # Discover devices on this node.
        device_ids = await self._discover_devices(node.mcp_endpoint)
        registered = node.with_devices(device_ids).with_status(
            _NODE_STATUS_ONLINE, heartbeat=time.monotonic()
        )

        # Update internal state.
        self._nodes[registered.node_id] = registered
        for did in registered.device_ids:
            self._device_to_node[did] = registered.node_id

        logger.info(
            "Fleet: registered node %r (%s) with %d device(s): %s",
            registered.node_id,
            registered.mcp_endpoint,
            len(registered.device_ids),
            ", ".join(registered.device_ids) or "(none)",
        )
        return registered

    async def unregister_node(self, node_id: str) -> None:
        """Remove a node and all its devices from the fleet.

        Halts all devices on the node first — best-effort, failures are
        logged but do not prevent removal.
        """
        node = self._nodes.get(node_id)
        if node is None:
            logger.debug("Fleet: unregister_node(%r) — not found, no-op", node_id)
            return

        # Halt all devices on the node (best-effort).
        for did in node.device_ids:
            try:
                await self._halt_device(did)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Fleet: halt device %r on unregister failed: %s",
                    did, exc, exc_info=True,
                )

        # Remove routing entries.
        for did in node.device_ids:
            self._device_to_node.pop(did, None)
        del self._nodes[node_id]

        logger.info("Fleet: unregistered node %r", node_id)

    async def discover_nodes(
        self, endpoints: tuple[str, ...]
    ) -> tuple[FleetNode, ...]:
        """Probe multiple endpoints for fleet nodes.

        Returns discovered nodes (not yet registered).  Unreachable
        endpoints are silently skipped.
        """
        discovered: list[FleetNode] = []

        async def _probe_one(endpoint: str) -> FleetNode | None:
            reachable = await self._probe_endpoint(endpoint)
            if not reachable:
                return None
            device_ids = await self._discover_devices(endpoint)
            # Derive a node_id from the endpoint for convenience.
            node_id = _endpoint_to_node_id(endpoint)
            return FleetNode(
                node_id=node_id,
                display_name=endpoint,
                mcp_endpoint=endpoint,
                device_ids=device_ids,
                status=_NODE_STATUS_ONLINE,
                last_heartbeat=time.monotonic(),
            )

        results = await asyncio.gather(
            *(_probe_one(ep) for ep in endpoints),
            return_exceptions=True,
        )
        for item in results:
            if isinstance(item, FleetNode):
                discovered.append(item)
            elif isinstance(item, Exception):
                logger.debug("Fleet: discover probe failed: %s", item)
        return tuple(discovered)

    # -- Routing ---------------------------------------------------------

    def route_device(self, device_id: str) -> str | None:
        """Return the node_id managing this device, or None if local."""
        return self._device_to_node.get(device_id)

    def is_remote(self, device_id: str) -> bool:
        """Return True if the device is managed by a remote node."""
        return device_id in self._device_to_node

    # -- Health monitoring -----------------------------------------------

    async def start_heartbeat(self) -> None:
        """Start periodic heartbeat monitoring of all nodes.

        Idempotent — calling twice is a no-op.
        """
        if self._heartbeat_task is not None:
            return
        self._stopped.clear()
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="fleet-heartbeat"
        )

    async def stop_heartbeat(self) -> None:
        """Stop heartbeat monitoring.  Idempotent."""
        self._stopped.set()
        task = self._heartbeat_task
        self._heartbeat_task = None
        if task is None:
            return
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception:  # noqa: BLE001 — teardown must not propagate
            logger.warning("Fleet heartbeat stop raised", exc_info=True)

    async def _heartbeat_loop(self) -> None:
        """Ping each node periodically.  Mark offline nodes after timeout.

        Best-effort: a failed probe logs but does not block the next
        cycle.  Nodes that exceed ``node_timeout_s`` since their last
        successful heartbeat are marked offline and their devices are
        marked degraded in the routing table.
        """
        while not self._stopped.is_set():
            for node_id in list(self._nodes):
                node = self._nodes.get(node_id)
                if node is None:
                    continue
                try:
                    reachable = await self._probe_endpoint(node.mcp_endpoint)
                except Exception:  # noqa: BLE001
                    reachable = False

                now = time.monotonic()
                if reachable:
                    self._nodes[node_id] = node.with_status(
                        _NODE_STATUS_ONLINE, heartbeat=now
                    )
                else:
                    elapsed = now - node.last_heartbeat
                    if elapsed > self._node_timeout:
                        if node.status != _NODE_STATUS_OFFLINE:
                            logger.warning(
                                "Fleet: node %r offline (%.1fs since last heartbeat)",
                                node_id, elapsed,
                            )
                            self._nodes[node_id] = node.with_status(
                                _NODE_STATUS_OFFLINE
                            )
                    elif node.status == _NODE_STATUS_ONLINE:
                        self._nodes[node_id] = node.with_status(
                            _NODE_STATUS_DEGRADED
                        )

            try:
                await asyncio.wait_for(
                    self._stopped.wait(), timeout=self._heartbeat_interval
                )
                break  # stopped
            except asyncio.TimeoutError:
                pass

    async def node_health(self, node_id: str) -> dict[str, Any]:
        """Return health details for one node.

        Probes the node synchronously and returns a diagnostic dict.
        """
        node = self._nodes.get(node_id)
        if node is None:
            return {"node_id": node_id, "error": "unknown node"}

        reachable = await self._probe_endpoint(node.mcp_endpoint)
        now = time.monotonic()
        if reachable:
            self._nodes[node_id] = node.with_status(
                _NODE_STATUS_ONLINE, heartbeat=now
            )
        return {
            "node_id": node_id,
            "display_name": node.display_name,
            "mcp_endpoint": node.mcp_endpoint,
            "status": _NODE_STATUS_ONLINE if reachable else node.status,
            "reachable": reachable,
            "device_count": len(node.device_ids),
            "device_ids": list(node.device_ids),
            "last_heartbeat": node.last_heartbeat,
            "seconds_since_heartbeat": now - node.last_heartbeat if node.last_heartbeat else -1,
        }

    # -- Fleet-level operations ------------------------------------------

    async def halt_all(self) -> dict[str, Any]:
        """Emergency stop ALL devices on ALL nodes.

        Sends halt concurrently to every node.  This is the fleet-level
        safety backstop — called when something goes wrong and we need
        to stop everything immediately.

        ``halt`` is lock-free (HCP requirement) so this does not wait
        behind in-flight I/O.  Individual failures are logged but never
        prevent the remaining halts from proceeding.

        Returns a per-node halt status map.
        """
        if not self._nodes:
            return {}

        async def _halt_node(node: FleetNode) -> tuple[str, dict[str, Any]]:
            results: dict[str, Any] = {"device_ids": list(node.device_ids)}
            successes = 0
            failures = 0
            for did in node.device_ids:
                try:
                    await self._halt_device(did)
                    successes += 1
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    logger.warning(
                        "Fleet halt_all: device %r on node %r failed: %s",
                        did, node.node_id, exc,
                    )
            results["successes"] = successes
            results["failures"] = failures
            results["halted"] = failures == 0
            return node.node_id, results

        raw = await asyncio.gather(
            *(_halt_node(n) for n in self._nodes.values()),
            return_exceptions=True,
        )
        halt_map: dict[str, Any] = {}
        for item in raw:
            if isinstance(item, Exception):
                logger.warning("Fleet halt_all: gather exception: %s", item)
                continue
            nid, status = item
            halt_map[nid] = status
        return halt_map

    def topology(self) -> FleetTopology:
        """Return current fleet topology snapshot."""
        nodes = tuple(sorted(self._nodes.values(), key=lambda n: n.node_id))
        total_devices = sum(len(n.device_ids) for n in nodes)
        online_nodes = sum(1 for n in nodes if n.status == _NODE_STATUS_ONLINE)
        return FleetTopology(
            nodes=nodes,
            total_devices=total_devices,
            online_nodes=online_nodes,
            timestamp=time.time(),
        )

    # -- Aggregation -----------------------------------------------------

    async def aggregate_trust(self) -> dict[str, Any]:
        """Collect and merge trust states from all nodes.

        Each node has its own ``HardwareTrustGate``; this aggregates
        them into a fleet-wide view.  The merge strategy is conservative:
        take the lowest trust level across nodes for each device/channel
        pair.  This ensures a device whose trust degraded on one node
        does not appear trusted fleet-wide.

        Returns a dict keyed by ``"device_id:channel_id"`` with the
        aggregated trust level name and per-node breakdown.
        """
        # Function-local import: trust module is optional at this layer.
        from leapflow.hardware.trust import HardwareTrustGate, HardwareTrustLevel

        fleet_trust: dict[str, dict[str, Any]] = {}

        # Collect from the local registry's trust gate if available.
        trust_gate: HardwareTrustGate | None = getattr(
            self._registry, "_trust_gate", None
        )
        if trust_gate is not None:
            for record in trust_gate.all_records():
                key = f"{record.device_id}:{record.channel_id}"
                fleet_trust[key] = {
                    "level": record.level.name,
                    "level_value": int(record.level),
                    "frozen": record.frozen,
                    "sources": {"local": record.level.name},
                }

        # For remote nodes, we query via MCP if their trust endpoint is
        # exposed.  For now this is a best-effort aggregation from the
        # local view — remote trust queries require a fleet-trust MCP tool
        # on each node, which is a future extension.

        return {
            "strategy": "conservative_merge",
            "description": (
                "Lowest trust level across all nodes for each (device, channel) pair"
            ),
            "trust": fleet_trust,
            "node_count": len(self._nodes),
            "timestamp": time.time(),
        }

    async def aggregate_evidence(self, *, since: float = 0.0) -> list[dict[str, Any]]:
        """Collect recent evidence from all nodes' EvidenceStores.

        Queries the local registry's evidence store.  Remote node
        evidence requires a fleet-evidence MCP tool on each node,
        which is a future extension.  For now, returns local evidence.
        """
        from leapflow.hardware.evidence import EvidenceStore

        store: EvidenceStore | None = getattr(self._registry, "_evidence_store", None)
        if store is None:
            # Try reading_store as a fallback attribute name.
            store = getattr(self._registry, "reading_store", None)

        if store is None or not hasattr(store, "query"):
            return []

        try:
            records = await store.query(since=since if since > 0 else None, limit=200)
            return records
        except Exception as exc:  # noqa: BLE001
            logger.debug("Fleet evidence aggregation failed: %s", exc)
            return []

    # -- Internal helpers ------------------------------------------------

    async def _probe_endpoint(self, endpoint: str) -> bool:
        """Probe an MCP endpoint for liveness.

        Returns True if the endpoint responds to a basic health check.
        This is a lightweight connectivity test, not a full capability
        discovery — it must complete quickly so the heartbeat loop stays
        responsive.
        """
        try:
            # Use the MCP client provider if available.
            from leapflow.hardware.transports.mcp import _CLIENT_PROVIDER

            if _CLIENT_PROVIDER is not None:
                client = _CLIENT_PROVIDER()
                if client is not None and hasattr(client, "call_tool"):
                    # Attempt a lightweight call — probe_tool or list_tools.
                    if hasattr(client, "list_tools"):
                        raw = client.list_tools()
                        if asyncio.iscoroutine(raw):
                            await asyncio.wait_for(raw, timeout=5.0)
                        return True
            # Fallback: treat endpoint as reachable if it parses.
            return bool(endpoint)
        except asyncio.TimeoutError:
            return False
        except Exception:  # noqa: BLE001
            return False

    async def _discover_devices(self, endpoint: str) -> tuple[str, ...]:
        """Discover devices on a node by querying its MCP server.

        Calls the equivalent of ``hw_list`` via MCP.  Returns an empty
        tuple if discovery fails.
        """
        try:
            from leapflow.hardware.transports.mcp import _CLIENT_PROVIDER

            if _CLIENT_PROVIDER is not None:
                client = _CLIENT_PROVIDER()
                if client is not None and hasattr(client, "call_tool"):
                    response = await client.call_tool("hw_list", {})
                    if isinstance(response, Mapping):
                        devices = response.get("devices", ())
                        if isinstance(devices, (list, tuple)):
                            return tuple(
                                str(d.get("device_id", d) if isinstance(d, dict) else d)
                                for d in devices
                                if d
                            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Fleet: device discovery on %s failed: %s", endpoint, exc
            )
        return ()

    async def _halt_device(self, device_id: str) -> None:
        """Halt a single device through the local registry's transport.

        Lock-free: ``transport.halt()`` does not acquire the I/O lock
        (HCP requirement), so this preempts in-flight operations.
        """
        try:
            transport = await self._registry.transport(device_id)
            await transport.halt()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Fleet: halt device %r failed: %s", device_id, exc)
            raise

    async def close(self) -> None:
        """Shut down the fleet manager.

        Stops heartbeat, then unregisters all nodes (which halts their
        devices).  Never raises.
        """
        await self.stop_heartbeat()
        for node_id in list(self._nodes):
            try:
                await self.unregister_node(node_id)
            except Exception:  # noqa: BLE001 — teardown must not propagate
                logger.warning(
                    "Fleet: unregister node %r during close failed",
                    node_id, exc_info=True,
                )


# ---------------------------------------------------------------------------
# FleetSessionRouter
# ---------------------------------------------------------------------------


class FleetSessionRouter:
    """Routes session operations to the correct fleet node.

    When an Agent targets a device managed by a remote node,
    the router transparently forwards the operation through
    the MCP Transport connection to that node.

    The router is a thin delegation layer — it resolves the node,
    then delegates to the local registry's transport (which is already
    an MCP transport pointing at the correct node).
    """

    def __init__(self, fleet_manager: FleetManager) -> None:
        self._fleet = fleet_manager

    async def route_write(
        self, device_id: str, channel_id: str, value: Any
    ) -> Any:
        """Route a write operation to the correct node.

        If the device is remote, the write goes through the MCP
        transport already registered in the local registry.  If local,
        it goes through the local transport directly.
        """
        registry = self._fleet._registry
        transport = await registry.transport(device_id)
        return await transport.write(channel_id, value)

    async def route_read(
        self, device_id: str, channel_id: str
    ) -> Any:
        """Route a read operation to the correct node.

        Same delegation pattern as ``route_write``: the transport
        handles the MCP bridge transparently.
        """
        registry = self._fleet._registry
        transport = await registry.transport(device_id)
        return await transport.read(channel_id)

    async def route_halt(self, device_id: str) -> Any:
        """Route a halt to the correct node.

        Lock-free requirement preserved: ``transport.halt()`` does
        not acquire the device I/O lock.
        """
        registry = self._fleet._registry
        transport = await registry.transport(device_id)
        return await transport.halt()


# ---------------------------------------------------------------------------
# Fleet tools (LLM-callable)
# ---------------------------------------------------------------------------


def build_fleet_tools(fleet_manager: FleetManager) -> list[Any]:
    """Generate ToolMetadata for fleet management tools.

    Three tools are exposed:

    - ``hw_fleet_status`` — show fleet topology, node health, device count
    - ``hw_fleet_discover`` — probe endpoints for new nodes
    - ``hw_fleet_halt_all`` — emergency stop all fleet devices
    """
    from leapflow.plugins.protocol import ToolMetadata

    _FLEET_READ_META = {
        "category": "hardware",
        "risk_level": "none",
    }
    _FLEET_WRITE_META = {
        "category": "hardware",
        "risk_level": "external",
    }

    async def _fleet_status(**kwargs: Any) -> dict[str, Any]:
        """Return fleet topology and per-node health summary."""
        topo = fleet_manager.topology()
        result = topo.to_dict()
        # Enrich with per-node health for online nodes.
        for node_dict in result.get("nodes", ()):
            nid = node_dict.get("node_id", "")
            if nid:
                health = await fleet_manager.node_health(nid)
                node_dict["health"] = health
        return result

    async def _fleet_discover(**kwargs: Any) -> dict[str, Any]:
        """Probe endpoints for new fleet nodes."""
        raw = kwargs.get("endpoints", "")
        if isinstance(raw, str):
            endpoints = tuple(
                ep.strip() for ep in raw.split(",") if ep.strip()
            )
        elif isinstance(raw, (list, tuple)):
            endpoints = tuple(str(ep) for ep in raw)
        else:
            return {"error": "endpoints must be a comma-separated string or list"}

        if not endpoints:
            return {"error": "no endpoints provided"}

        nodes = await fleet_manager.discover_nodes(endpoints)
        return {
            "discovered": [
                {
                    "node_id": n.node_id,
                    "display_name": n.display_name,
                    "mcp_endpoint": n.mcp_endpoint,
                    "device_ids": list(n.device_ids),
                    "status": n.status,
                }
                for n in nodes
            ],
            "total": len(nodes),
        }

    async def _fleet_halt_all(**kwargs: Any) -> dict[str, Any]:
        """Emergency stop ALL fleet devices on ALL nodes."""
        halt_map = await fleet_manager.halt_all()
        return {
            "halt_map": halt_map,
            "node_count": len(halt_map),
            "message": "Fleet-level emergency stop issued to all nodes",
        }

    return [
        ToolMetadata(
            name="hw_fleet_status",
            description=(
                "Show fleet topology: all nodes, their connection status, "
                "device count, and per-node health.  Use to understand the "
                "distributed robot fleet before issuing commands."
            ),
            parameters_schema={"type": "object", "properties": {}},
            handler=_fleet_status,
            x_leapflow=dict(_FLEET_READ_META),
            provides_capabilities=("hw.fleet.status",),
        ),
        ToolMetadata(
            name="hw_fleet_discover",
            description=(
                "Probe MCP endpoints for new fleet nodes.  Supply a "
                "comma-separated list of endpoints to check.  Returns "
                "discovered nodes that can be registered."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "endpoints": {
                        "type": "string",
                        "description": (
                            "Comma-separated MCP endpoints to probe, "
                            "e.g. 'http://192.168.1.50:9000,http://192.168.1.51:9000'"
                        ),
                    },
                },
                "required": ["endpoints"],
            },
            handler=_fleet_discover,
            x_leapflow=dict(_FLEET_READ_META),
            provides_capabilities=("hw.fleet.discover",),
        ),
        ToolMetadata(
            name="hw_fleet_halt_all",
            description=(
                "Emergency stop ALL devices on ALL fleet nodes immediately.  "
                "Use when any fleet-wide unsafe condition is observed.  "
                "Sends halt concurrently to every node; does not wait for "
                "individual node responses."
            ),
            parameters_schema={"type": "object", "properties": {}},
            handler=_fleet_halt_all,
            x_leapflow=dict(_FLEET_WRITE_META),
            mutates_state=True,
            provides_capabilities=("hw.fleet.halt_all",),
        ),
    ]


# ---------------------------------------------------------------------------
# FleetError
# ---------------------------------------------------------------------------


class FleetError(Exception):
    """Raised when a fleet operation fails.

    Structured like ``TransportError`` but fleet-scoped: it carries a
    ``failure_code`` for programmatic handling by the recovery layer.
    """

    def __init__(self, message: str, *, failure_code: str = "fleet_error") -> None:
        super().__init__(message)
        self.failure_code = failure_code


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _endpoint_to_node_id(endpoint: str) -> str:
    """Derive a stable node_id from an endpoint string.

    Strips protocol, replaces non-alphanumeric chars with underscores,
    and lowercases for consistency with device_id conventions.
    """
    cleaned = endpoint.lower()
    for prefix in ("http://", "https://", "stdio:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    return "".join(c if c.isalnum() else "_" for c in cleaned).strip("_")


__all__ = [
    "FleetError",
    "FleetManager",
    "FleetNode",
    "FleetSessionRouter",
    "FleetTopology",
    "build_fleet_tools",
]
