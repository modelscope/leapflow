# Copyright (c) Alibaba, Inc. and its affiliates.
"""Fleet configuration and startup registration.

Kept apart from ``fleet.py`` (the manager mechanics) because node
configuration is an input boundary with its own concerns: tolerant parsing,
secret hygiene, and best-effort startup.  P2 grows this module with the
Settings-based double track and registration backoff; ``FleetManager`` stays
unaware of where its nodes were configured.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, Iterable, Mapping

from leapflow.hardware.fleet import FleetError, FleetNode

if TYPE_CHECKING:
    from leapflow.hardware.fleet import FleetManager

logger = logging.getLogger(__name__)

FLEET_NODES_ENV = "LEAPFLOW_FLEET_NODES"


def configured_nodes_from_env() -> tuple[FleetNode, ...]:
    """Parse ``LEAPFLOW_FLEET_NODES`` (JSON array) into node configurations.

    Entry keys: ``node_id``, ``display_name``, ``endpoint`` (preferred) or the
    legacy ``mcp_endpoint``, and ``token``.  Entries without a node id or an
    endpoint are skipped with a warning; id charset and reachability stay in
    ``register_node`` so validation has a single authority.  Malformed JSON
    disables fleet configuration for the process rather than failing startup --
    a profile typo must not take the daemon down.

    Diagnostics never echo a raw entry: entries carry the node's shared
    secret, and a log line is not a secret store.
    """
    raw = os.environ.get(FLEET_NODES_ENV, "")
    if not raw:
        return ()
    try:
        configs = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("%s is not valid JSON; ignoring", FLEET_NODES_ENV)
        return ()
    if not isinstance(configs, list):
        logger.warning("%s must be a JSON array; ignoring", FLEET_NODES_ENV)
        return ()

    nodes: list[FleetNode] = []
    for index, config in enumerate(configs):
        if not isinstance(config, Mapping):
            logger.warning(
                "Fleet: ignoring %s entry #%d: not a JSON object", FLEET_NODES_ENV, index
            )
            continue
        node = FleetNode(
            node_id=str(config.get("node_id") or ""),
            display_name=str(config.get("display_name") or ""),
            # ``endpoint`` is the LHP binding (tcp://host:port or unix:///path);
            # ``mcp_endpoint`` remains accepted so pre-LHP profiles keep loading.
            # Every key coalesces ``null`` to "": a templated JSON null means
            # "unset", and ``str(None)`` would turn an absent token into the
            # literal string "None" -- a wrong shared secret that fails hello
            # with a confusing auth error.
            endpoint=str(config.get("endpoint") or config.get("mcp_endpoint") or ""),
            token=str(config.get("token") or ""),
        )
        if not node.node_id or not node.endpoint:
            logger.warning(
                "Fleet: ignoring %s entry #%d without node_id/endpoint "
                "(node_id=%r, endpoint=%r)",
                FLEET_NODES_ENV,
                index,
                node.node_id,
                node.endpoint,
            )
            continue
        nodes.append(node)
    return tuple(nodes)


async def register_nodes_safely(
    manager: "FleetManager", nodes: Iterable[FleetNode]
) -> tuple[FleetNode, ...]:
    """Register nodes best-effort; a failing node is logged and skipped.

    Both startup paths (the daemon runtime and the in-process CLI read plane)
    share this so one unreachable board can never keep the reachable nodes --
    or the host's own devices -- from being usable.

    Registrations run concurrently: each unreachable node costs its full
    connect timeout, and a serial loop would add those costs up in front of
    daemon readiness.  Results keep the input order.
    """
    candidates = tuple(nodes)
    if not candidates:
        return ()
    results = await asyncio.gather(
        *(manager.register_node(node) for node in candidates),
        return_exceptions=True,
    )
    registered: list[FleetNode] = []
    for node, result in zip(candidates, results):
        if isinstance(result, FleetError):
            # Expected operational failure (board off, wrong token): visible
            # but not alarming.  P2 owns retry/backoff and re-registration.
            logger.warning("Fleet: node %r not registered: %s", node.node_id, result)
        elif isinstance(result, BaseException):
            logger.warning(
                "Fleet: node %r registration raised", node.node_id, exc_info=result
            )
        else:
            registered.append(result)
    return tuple(registered)


__all__ = ["FLEET_NODES_ENV", "configured_nodes_from_env", "register_nodes_safely"]
