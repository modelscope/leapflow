# Copyright (c) Alibaba, Inc. and its affiliates.
"""Fleet deployment management: multi-node robot coordination.

Extends LeapFlow's single-machine hardware stack to a fleet of robots
distributed across multiple edge nodes.  The central leapd manages the
fleet topology, and every node conversation is held over LHP -- the
LeapFlow Hardware Protocol (``hardware/lhp/``): the node runs
``LhpNodeServer`` over its own ``HardwareRegistry`` and the host talks to
it with ``SocketLhpClient``.

Three deployment scenarios (from research note 10):
1. Dual-machine (Mac + edge board): Mac runs LLM/console, board runs RT control
2. Standalone edge + remote LLM: edge runs independently, LLM via API
3. Mac-only: USB-connected devices, no fleet needed

Fleet management adds:
- Node discovery and health monitoring
- Device-to-node routing (host-side device_id → node_id)
- Cross-node trust aggregation
- Cross-node experience/evidence aggregation
- Fleet-level emergency stop (halt all nodes)
- Hot-plug: nodes joining/leaving the fleet

Design invariants:

- Nodes are bridged over LHP; a remote device's transport kind is ``lhp``
  and the binding (endpoint, token, remote device id) travels in its
  ``TransportRef.config``.
- Remote devices register in the local HardwareRegistry as external-source
  declarations under the node's namespace, making them transparent to the
  agent layer above.
- Host-side identity is namespaced ``<node_id>__<device_id>``; the node
  keeps seeing its own node-local ids and never learns the prefix.
- A locally declared device always wins over a node's declaration of the
  same host-side id (LHP spec §8 merge order: providers first, external
  sources after, duplicates rejected rather than replaced).
- ``halt_all`` is concurrent (``asyncio.gather``), fire-and-forget.
- Heartbeat is best-effort and never blocks data-plane operations.
- Trust aggregation takes the conservative value (lowest trust level).
- Nodes that go offline have their devices marked degraded automatically.
- No dependency on ``leapflow.engine``.

Module split: node configuration and startup registration live in
``fleet_config.py``; the LLM-callable tools live in ``fleet_tools.py``.
This module keeps only the fleet model and ``FleetManager`` mechanics, and
is slightly above the 800-line soft limit because the manager's
registration, supervision, halt and aggregation paths share one routing
table and one node map -- splitting them would scatter that state.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from leapflow.hardware.context import HardwareContext, TransportRef
from leapflow.hardware.lhp import LhpRemoteError, SocketLhpClient, is_valid_device_id
from leapflow.hardware.attestation import confirmed_by

logger = logging.getLogger(__name__)

# How long a heartbeat probe may take before the node counts as unreachable.
# Short enough to stay invisible beside a 10s heartbeat interval, long enough
# for a busy node to answer a hello exchange.
_HEARTBEAT_PROBE_TIMEOUT_S = 5.0

# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

_NODE_STATUS_ONLINE = "online"
_NODE_STATUS_OFFLINE = "offline"
_NODE_STATUS_DEGRADED = "degraded"
_NODE_STATUS_UNKNOWN = "unknown"

_VALID_STATUSES = frozenset(
    {
        _NODE_STATUS_ONLINE,
        _NODE_STATUS_OFFLINE,
        _NODE_STATUS_DEGRADED,
        _NODE_STATUS_UNKNOWN,
    }
)


@dataclass(frozen=True)
class FleetNode:
    """One edge node in the fleet.

    Each node represents a physical machine (e.g. a Jetson board) that
    hosts one or more hardware devices and runs ``LhpNodeServer``.  The
    ``endpoint`` is the node's LHP binding address -- ``tcp://host:port``
    or ``unix:///path`` -- through which all device conversations are held.

    ``device_ids`` holds the *node-local* ids the node reported and the
    local registry admitted (host-side truth after namespacing); ids the
    registry rejected are not routable and deliberately absent.

    ``token`` is the shared secret for LHP hello-first authentication;
    empty means an open node, acceptable only on a trusted segment.

    ``declarations`` is the fingerprint the node's hello carried when its
    declarations were last admitted; empty for a node that does not send one.
    """

    node_id: str
    display_name: str
    endpoint: str  # "tcp://host:port" or "unix:///path"
    token: str = ""
    device_ids: tuple[str, ...] = ()
    status: str = _NODE_STATUS_UNKNOWN
    last_heartbeat: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    declarations: str = ""

    def host_device_id(self, device_id: str) -> str:
        """Return the host-side (namespaced) id for a node-local device id."""
        return f"{self.node_id}__{device_id}"

    def with_status(self, status: str, *, heartbeat: float = 0.0) -> FleetNode:
        """Return a copy with an updated status and optional heartbeat."""
        return replace(
            self,
            status=status if status in _VALID_STATUSES else _NODE_STATUS_UNKNOWN,
            last_heartbeat=heartbeat or self.last_heartbeat,
        )

    def with_devices(self, device_ids: tuple[str, ...]) -> FleetNode:
        """Return a copy with an updated device list."""
        return replace(self, device_ids=tuple(device_ids))


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
                    "endpoint": n.endpoint,
                    # Compatibility key: dashboards may still read the old
                    # field name (G4 decision in 04-open-questions.md).
                    "mcp_endpoint": n.endpoint,
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
# Talking to a node
# ---------------------------------------------------------------------------


async def fetch_node_declarations(
    endpoint: str, *, token: str = ""
) -> tuple[str, tuple[HardwareContext, ...], str]:
    """Fetch a node's identity and raw device declarations over LHP.

    Returns ``(reported_node_id, contexts, declarations_fingerprint)``.  The
    *reported* id comes from the node's hello answer; the caller decides whether
    to pin it (register: a mismatch refuses) or adopt it (discover: it is the
    authoritative name).  Declarations stay in the node's own id space;
    namespacing the ids is the caller's job.

    Module-level rather than a manager method because confirming a node's
    declaration is a read: an operator reviewing a declaration must not have to
    start a fleet that would halt the node's hardware when it shuts down.

    Raises ``FleetError`` when the node is unreachable, fails the exchange, or
    answers a declaration this host cannot read.  A machine-generated
    declaration that does not parse is a version mismatch, so the whole node is
    refused loudly rather than admitted partially.
    """
    client = SocketLhpClient(endpoint, token=token)
    try:
        hello = await client.connect()
        devices = await client.request("list", {})
    except (LhpRemoteError, ValueError, OSError) as exc:
        raise FleetError(
            f"node at {endpoint!r} unreachable: {exc}",
            failure_code="fleet_unreachable",
        ) from exc
    finally:
        await client.close()

    if not isinstance(devices, Mapping):
        raise FleetError(
            f"node at {endpoint!r} returned a malformed list payload",
            failure_code="fleet_malformed_list",
        )

    contexts: list[HardwareContext] = []
    raw_devices = devices.get("devices", ())
    if isinstance(raw_devices, (list, tuple)):
        for payload in raw_devices:
            if not isinstance(payload, Mapping):
                continue
            try:
                contexts.append(HardwareContext.from_mapping(dict(payload)))
            except (KeyError, TypeError, ValueError) as exc:
                raise FleetError(
                    f"node at {endpoint!r} declaration is unreadable: {exc}",
                    failure_code="fleet_bad_declaration",
                ) from exc

    reported = str(hello.get("node_id") or "")
    return reported, tuple(contexts), str(hello.get("declarations") or "")


# ---------------------------------------------------------------------------
# FleetManager
# ---------------------------------------------------------------------------


class FleetManager:
    """Manages a fleet of distributed robot nodes.

    Coordinates with the local ``HardwareRegistry`` to present remote
    devices as if they were local — commands are transparently routed
    through the ``lhp`` transport to the correct edge node, and node
    declarations enter the registry as external sources.
    """

    def __init__(
        self,
        registry: Any,  # local HardwareRegistry
        *,
        heartbeat_interval_s: float = 10.0,
        node_timeout_s: float = 30.0,
        permit_issuer: Any | None = None,
    ) -> None:
        self._registry = registry
        # Signs execution permits for nodes in ``permit_required`` mode. Shared
        # by every remote device's transport and by the skill service; absent,
        # such nodes refuse writes with a host-side message naming why.
        self._permit_issuer = permit_issuer
        self._nodes: dict[str, FleetNode] = {}
        self._device_to_node: dict[str, str] = {}  # device_id → node_id
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._heartbeat_interval = max(1.0, float(heartbeat_interval_s))
        self._node_timeout = max(1.0, float(node_timeout_s))
        self._stopped = asyncio.Event()

    @property
    def permit_issuer(self) -> Any | None:
        """The host's permit issuer, or ``None`` when permits cannot be minted."""
        return self._permit_issuer

    # -- Node management -------------------------------------------------

    async def register_node(self, node: FleetNode) -> FleetNode:
        """Register an edge node, reconcile its devices into the local registry.

        1. Validate the node_id -- it becomes a namespace prefix, so it must
           already satisfy the device-id charset.
        2. Fetch the node's device declarations over LHP; the hello exchange
           pins the node's identity.
        3. Relabel each declaration ``<node_id>__<device_id>`` with an ``lhp``
           transport binding, register the set as an external source, and
           reconcile so admission (including local-wins precedence) happens
           through the one real path.
        4. Route the *admitted* node-local ids and mark the node online.

        Returns the node with updated device list and status.
        Raises ``FleetError`` if the endpoint is unreachable, the node
        reports a different identity, or its declarations are malformed.
        """
        if not node.node_id or not is_valid_device_id(node.node_id):
            raise FleetError(
                f"node_id {node.node_id!r} must be non-empty and match [a-z0-9_]+",
                failure_code="fleet_bad_node_id",
            )
        if not node.endpoint:
            raise FleetError(
                f"node {node.node_id!r} has no endpoint",
                failure_code="fleet_no_endpoint",
            )

        reported, remote_contexts, declarations = await fetch_node_declarations(
            node.endpoint, token=node.token
        )
        if reported and reported != node.node_id:
            # Identity mismatch is never resolved automatically: adopting the
            # reported id could silently merge two boards into one namespace.
            raise FleetError(
                f"node at {node.endpoint!r} reports itself as {reported!r}, not {node.node_id!r}",
                failure_code="fleet_node_mismatch",
            )

        prefix = f"{node.node_id}__"
        local_contexts = tuple(
            replace(
                self._adjust_remote_provenance(remote, node.node_id),
                device_id=node.host_device_id(remote.device_id),
                transport=self._transport_ref(node, remote.device_id),
            )
            for remote in remote_contexts
        )
        self._registry.register_external_contexts(node.node_id, local_contexts)

        report = await self._registry.reconcile()
        admitted_local = tuple(
            did[len(prefix) :] for did in report.admitted if did.startswith(prefix)
        )
        dropped = tuple(did for did in report.rejected if did.startswith(prefix))
        if dropped:
            logger.warning(
                "Fleet: node %r declarations rejected at admission: %s",
                node.node_id,
                ", ".join(dropped),
            )

        registered = replace(
            node,
            device_ids=admitted_local,
            status=_NODE_STATUS_ONLINE,
            last_heartbeat=time.monotonic(),
            declarations=declarations,
        )
        # Re-registration replaces the previous snapshot: drop stale routing
        # entries for device ids this node no longer declares.
        previous = self._nodes.get(registered.node_id)
        if previous is not None:
            for local_did in previous.device_ids:
                self._device_to_node.pop(previous.host_device_id(local_did), None)
        self._nodes[registered.node_id] = registered
        for local_did in registered.device_ids:
            self._device_to_node[registered.host_device_id(local_did)] = registered.node_id

        logger.info(
            "Fleet: registered node %r (%s) with %d device(s): %s",
            registered.node_id,
            registered.endpoint,
            len(registered.device_ids),
            ", ".join(registered.device_ids) or "(none)",
        )
        return registered

    async def unregister_node(self, node_id: str) -> None:
        """Remove a node and all its devices from the fleet.

        Halts all devices on the node first -- best-effort, failures are
        logged but do not prevent removal.  Then drops the node's external
        source and reconciles, so its devices leave the local registry
        through the same path they arrived in.
        """
        node = self._nodes.get(node_id)
        if node is None:
            logger.debug("Fleet: unregister_node(%r) -- not found, no-op", node_id)
            return

        # Halt all devices on the node (best-effort).
        for local_did in node.device_ids:
            try:
                await self._halt_device(node.host_device_id(local_did))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Fleet: halt device %r on unregister failed: %s",
                    local_did,
                    exc,
                    exc_info=True,
                )

        # Remove routing entries.
        for local_did in node.device_ids:
            self._device_to_node.pop(node.host_device_id(local_did), None)
        del self._nodes[node_id]

        # Drop the declarations and converge the registry (never raises).
        self._registry.unregister_external_source(node_id)
        await self._registry.reconcile()

        logger.info("Fleet: unregistered node %r", node_id)

    async def discover_nodes(self, endpoints: tuple[str, ...]) -> tuple[FleetNode, ...]:
        """Probe multiple endpoints for fleet nodes.

        Returns discovered nodes (not yet registered).  Unreachable
        endpoints are silently skipped.  A reachable node's identity and
        device list come from its own hello/list answers; the derived
        ``_endpoint_to_node_id`` name is only a fallback for silent nodes.
        """
        discovered: list[FleetNode] = []

        async def _probe_one(endpoint: str) -> FleetNode | None:
            reported, remote_contexts, _ = await fetch_node_declarations(endpoint)
            node_id = reported or _endpoint_to_node_id(endpoint)
            return FleetNode(
                node_id=node_id,
                display_name=endpoint,
                endpoint=endpoint,
                device_ids=tuple(ctx.device_id for ctx in remote_contexts),
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
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="fleet-heartbeat")

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
            await self._heartbeat_sweep()
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self._heartbeat_interval)
                break  # stopped
            except asyncio.TimeoutError:
                pass

    async def _heartbeat_sweep(self) -> None:
        """Probe every node once, concurrently, and apply status transitions.

        Concurrent because an unreachable board costs a full probe timeout;
        probed serially, a few dark boards would delay supervision of the
        live ones past their own timeout window.  Every transition is logged
        so an outage and its recovery both leave evidence.

        A live node whose hello carries a different declarations fingerprint
        is re-registered: it restarted or re-scanned with changes, and without
        this the host would keep admitting the declarations it first saw --
        stale envelopes, and no trace of a camera plugged in since.
        """
        changed: list[str] = []
        node_ids = list(self._nodes)
        probes = await asyncio.gather(
            *(
                self._probe_endpoint(self._nodes[nid].endpoint, token=self._nodes[nid].token)
                for nid in node_ids
            ),
            return_exceptions=True,
        )
        now = time.monotonic()
        for node_id, probe in zip(node_ids, probes):
            node = self._nodes.get(node_id)
            if node is None:
                continue  # unregistered while the sweep was in flight
            if isinstance(probe, Mapping):
                if node.status in (_NODE_STATUS_OFFLINE, _NODE_STATUS_DEGRADED):
                    logger.info("Fleet: node %r back online", node_id)
                self._nodes[node_id] = node.with_status(_NODE_STATUS_ONLINE, heartbeat=now)
                declarations = str(probe.get("declarations") or "")
                if declarations and declarations != node.declarations:
                    changed.append(node_id)
                continue
            elapsed = now - node.last_heartbeat
            if elapsed > self._node_timeout:
                if node.status != _NODE_STATUS_OFFLINE:
                    logger.warning(
                        "Fleet: node %r offline (%.1fs since last heartbeat)",
                        node_id,
                        elapsed,
                    )
                    self._nodes[node_id] = node.with_status(_NODE_STATUS_OFFLINE)
            elif node.status == _NODE_STATUS_ONLINE:
                logger.info(
                    "Fleet: node %r degraded (heartbeat missed, %.1fs since last)",
                    node_id,
                    elapsed,
                )
                self._nodes[node_id] = node.with_status(_NODE_STATUS_DEGRADED)
        for node_id in changed:
            await self._refresh_node(node_id)

    async def _refresh_node(self, node_id: str) -> None:
        """Re-admit a node's changed declarations through ``register_node``.

        Re-registration replaces the routing snapshot and lets reconcile diff
        the declarations, so unchanged devices keep running and only altered
        ones are rebuilt.  A failure keeps the previous snapshot and is retried
        on the next sweep, because the stored fingerprint is left unchanged.
        """
        node = self._nodes.get(node_id)
        if node is None:
            return
        logger.info("Fleet: node %r declarations changed; re-registering", node_id)
        try:
            await self.register_node(node)
        except FleetError as exc:
            logger.warning("Fleet: re-registering node %r failed: %s", node_id, exc)

    async def node_health(self, node_id: str) -> dict[str, Any]:
        """Return health details for one node.

        Probes the node synchronously and returns a diagnostic dict.
        """
        node = self._nodes.get(node_id)
        if node is None:
            return {"node_id": node_id, "error": "unknown node"}

        reachable = await self._probe_endpoint(node.endpoint, token=node.token) is not None
        now = time.monotonic()
        if reachable:
            self._nodes[node_id] = node.with_status(_NODE_STATUS_ONLINE, heartbeat=now)
        return {
            "node_id": node_id,
            "display_name": node.display_name,
            "endpoint": node.endpoint,
            # Compatibility key -- see FleetTopology.to_dict (G4 decision).
            "mcp_endpoint": node.endpoint,
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
                    await self._halt_device(node.host_device_id(did))
                    successes += 1
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    logger.warning(
                        "Fleet halt_all: device %r on node %r failed: %s",
                        did,
                        node.node_id,
                        exc,
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
        from leapflow.hardware.trust import HardwareTrustGate

        fleet_trust: dict[str, dict[str, Any]] = {}

        # Collect from the local registry's trust gate if available.
        trust_gate: HardwareTrustGate | None = getattr(self._registry, "_trust_gate", None)
        if trust_gate is not None:
            for record in trust_gate.all_records():
                key = f"{record.device_id}:{record.channel_id}"
                fleet_trust[key] = {
                    "level": record.level.name,
                    "level_value": int(record.level),
                    "frozen": record.frozen,
                    "sources": {"local": record.level.name},
                }

        # Remote trust queries would need a dedicated LHP method on each
        # node; until then this is a best-effort aggregation from the local
        # view, and missing remote entries stay visible as such.

        return {
            "strategy": "conservative_merge",
            "description": ("Lowest trust level across all nodes for each (device, channel) pair"),
            "trust": fleet_trust,
            "node_count": len(self._nodes),
            "timestamp": time.time(),
        }

    async def aggregate_evidence(self, *, since: float = 0.0) -> list[dict[str, Any]]:
        """Collect recent evidence from all nodes' EvidenceStores.

        Queries the local registry's evidence store.  Remote node
        evidence would need a dedicated LHP method on each node; until
        then this returns local evidence only.
        """
        from leapflow.hardware.evidence import EvidenceStore

        store: EvidenceStore | None = getattr(self._registry, "evidence_store", None)
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

    async def _probe_endpoint(self, endpoint: str, *, token: str = "") -> Mapping[str, Any] | None:
        """Probe an LHP endpoint for liveness; return its hello, or None.

        One hello exchange with a short timeout: a lightweight connectivity
        test, not capability discovery -- it must complete quickly so the
        heartbeat loop stays responsive.  The hello is returned because its
        declarations fingerprint is how a sweep notices changed devices.
        Never raises.
        """
        client = SocketLhpClient(endpoint, token=token, timeout_s=_HEARTBEAT_PROBE_TIMEOUT_S)
        try:
            return await client.connect()
        except (LhpRemoteError, ValueError, OSError):
            return None
        finally:
            await client.close()

    def _transport_ref(self, node: FleetNode, remote_device_id: str) -> TransportRef:
        """Build the LHP binding for one remote device.

        ``remote_device_id`` is what the *node* calls the device; the host
        side calls it ``<node_id>__<device_id>``.  Both ids travel so the
        transport can address the right device on the node while every
        host-side artifact carries the namespaced id. The permit issuer rides
        along in-process only (declaration digests strip transport config).
        """
        return TransportRef(
            kind="lhp",
            config={
                "endpoint": node.endpoint,
                "token": node.token,
                "node_id": node.node_id,
                "remote_device_id": remote_device_id,
                "permit_issuer": self._permit_issuer,
            },
        )

    def _adjust_remote_provenance(self, context: Any, node_id: str) -> Any:
        """Strip self-reported verified status from remote contexts (P2-J).

        A node cannot be trusted to assert its own verification: a machine that
        can rewrite its declaration can also rewrite its claim that a human read
        it.  The host therefore lets the claim through only when it holds its own
        confirmation for ``<node>__<device>``, bound to the digest of the
        declaration the node is sending now.  A confirmation of some earlier
        declaration does not transfer to this one.

        The digest is computed over *this* context, in the node's own id space,
        so it matches the digest the node computes for the same declaration and
        an operator can compare the two.
        """
        provenance = getattr(context, "provenance", None)
        if provenance is None or not getattr(provenance, "is_verified", False):
            return context
        host_device_id = f"{node_id}__{context.device_id}"
        verifier = confirmed_by(self._verified_path(), host_device_id, context)
        if verifier:
            return replace(
                context,
                provenance=replace(provenance, verified_by=verifier),
            )
        logger.info(
            "Fleet: stripping self-reported verification for %s from node %s; "
            "this host holds no confirmation of that declaration",
            context.device_id,
            node_id,
        )
        return replace(
            context,
            provenance=replace(
                provenance,
                verified_by="",
                source=f"remote:{node_id}",
            ),
        )

    def _verified_path(self) -> str:
        """The host's confirmation record, as resolved by the active profile."""
        settings = getattr(self._registry, "settings", None)
        return str(getattr(settings, "verified_path", "") or "")

    async def _halt_device(self, device_id: str) -> None:
        """Halt a single device through the local registry's transport.

        Lock-free: ``transport.halt()`` does not acquire the I/O lock
        (HCP requirement), so this preempts in-flight operations.

        Raises ``FleetError`` when the transport could not deliver the halt.
        ``halt()`` never raises by contract, so without checking the delivery
        verdict a dead node would be reported as halted -- exactly the lie an
        emergency stop must not tell.

        The verdict is read from ``halt_delivered``, never inferred from
        ``connected``/``halt_supported``: a passive camera releases its capture and
        answers ``connected=False, halt_supported=False``, which is a stop that
        succeeded, while a closed link to an arm that may still be travelling
        answers the same shape for a stop that was never sent. A device declaring
        no halt support has no motion to arrest, so its stop is satisfied whatever
        the link did.

        The link is opened when none is open -- a stop has to reach the device --
        and the stop itself goes through the registry, which retires the link when
        the driver closed it.
        """
        try:
            await self._registry.transport(device_id)
            status = await self._registry.halt(device_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Fleet: halt device %r failed: %s", device_id, exc)
            raise
        if status is None:
            logger.debug("Fleet: no link for %r survived to carry the halt", device_id)
            return
        if status.halt_supported and not status.halt_delivered:
            raise FleetError(
                f"halt for {device_id!r} was not delivered: {status.detail or 'no link'}",
                failure_code="fleet_halt_undelivered",
            )

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
                    node_id,
                    exc_info=True,
                )


# ---------------------------------------------------------------------------
# FleetSessionRouter
# ---------------------------------------------------------------------------


class FleetSessionRouter:
    """Routes session operations to the correct fleet node.

    When an Agent targets a device managed by a remote node,
    the router transparently forwards the operation through
    the LHP transport connection to that node.

    The router is a thin delegation layer — it resolves the node,
    then delegates to the local registry's transport (which is already
    an ``lhp`` transport pointing at the correct node).
    """

    def __init__(self, fleet_manager: FleetManager) -> None:
        self._fleet = fleet_manager

    async def route_write(self, device_id: str, channel_id: str, value: Any) -> Any:
        """Route a write operation to the correct node.

        If the device is remote, the write goes through the ``lhp``
        transport already registered in the local registry.  If local,
        it goes through the local transport directly.
        """
        registry = self._fleet._registry
        transport = await registry.transport(device_id)
        return await transport.write(channel_id, value)

    async def route_read(self, device_id: str, channel_id: str) -> Any:
        """Route a read operation to the correct node.

        Same delegation pattern as ``route_write``: the transport
        handles the LHP bridge transparently.
        """
        registry = self._fleet._registry
        transport = await registry.transport(device_id)
        return await transport.read(channel_id)

    async def route_halt(self, device_id: str) -> Any:
        """Route a halt to the correct node.

        Lock-free requirement preserved: the registry's halt never acquires the
        device I/O lock. It does own the link cache, so a stop that closed the
        link leaves nothing behind for the next read to trip over.
        """
        registry = self._fleet._registry
        await registry.transport(device_id)
        return await registry.halt(device_id)


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
    for prefix in ("tcp://", "unix://", "http://", "https://", "stdio:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    return "".join(c if c.isalnum() else "_" for c in cleaned).strip("_")


__all__ = [
    "FleetError",
    "FleetManager",
    "FleetNode",
    "FleetSessionRouter",
    "FleetTopology",
]
