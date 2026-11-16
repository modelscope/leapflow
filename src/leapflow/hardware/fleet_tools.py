# Copyright (c) Alibaba, Inc. and its affiliates.
"""LLM-callable fleet tools backed by ``FleetManager``.

The agent-facing surface lives apart from the manager mechanics
(``fleet.py``) for the same reason ``tools.py`` lives apart from
``registry.py``: tool schemas and their governance metadata change with
the product surface, while routing and supervision change with the fabric.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from leapflow.hardware.fleet import FleetManager
    from leapflow.plugins.protocol import ToolMetadata

_FLEET_READ_META = {
    "category": "hardware",
    "risk_level": "none",
}
_FLEET_WRITE_META = {
    "category": "hardware",
    "risk_level": "external",
}


def build_fleet_tools(fleet_manager: "FleetManager") -> list["ToolMetadata"]:
    """Generate ToolMetadata for fleet management tools.

    Three tools are exposed:

    - ``hw_fleet_status`` — show fleet topology, node health, device count
    - ``hw_fleet_discover`` — probe endpoints for new nodes
    - ``hw_fleet_halt_all`` — emergency stop all fleet devices
    """
    from leapflow.plugins.protocol import ToolMetadata

    async def _fleet_status(**kwargs: Any) -> dict[str, Any]:
        """Return fleet topology and per-node health summary.

        Health probes run concurrently: each offline node costs a full probe
        timeout, and a serial loop would make one dark board stall the
        whole answer.
        """
        result = fleet_manager.topology().to_dict()
        node_dicts = [n for n in result.get("nodes", ()) if n.get("node_id")]
        healths = await asyncio.gather(
            *(fleet_manager.node_health(n["node_id"]) for n in node_dicts)
        )
        for node_dict, health in zip(node_dicts, healths):
            node_dict["health"] = health
        return result

    async def _fleet_discover(**kwargs: Any) -> dict[str, Any]:
        """Probe endpoints for new fleet nodes."""
        raw = kwargs.get("endpoints", "")
        if isinstance(raw, str):
            endpoints = tuple(ep.strip() for ep in raw.split(",") if ep.strip())
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
                    "endpoint": n.endpoint,
                    # Compatibility key -- see FleetTopology.to_dict (G4).
                    "mcp_endpoint": n.endpoint,
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
                "Probe LHP endpoints for new fleet nodes.  Supply a "
                "comma-separated list of endpoints to check.  Returns "
                "discovered nodes that can be registered."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "endpoints": {
                        "type": "string",
                        "description": (
                            "Comma-separated LHP endpoints to probe, "
                            "e.g. 'tcp://192.168.1.50:8767,tcp://192.168.1.51:8767'"
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


__all__ = ["build_fleet_tools"]
