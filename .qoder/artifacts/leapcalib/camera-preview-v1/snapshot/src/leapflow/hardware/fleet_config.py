# Copyright (c) Alibaba, Inc. and its affiliates.
"""Fleet configuration and startup registration.

Kept apart from ``fleet.py`` (the manager mechanics) because node
configuration is an input boundary with its own concerns: tolerant parsing,
secret hygiene, and best-effort startup.  P2 grows this module with the
Settings-based double track and registration backoff; ``FleetManager`` stays
unaware of where its nodes were configured.

It also owns the one thing an operator -- not the fleet -- writes: the
confirmation that a node's declaration is the one a human reviewed.  A node
cannot vouch for itself, so this record lives on the host, keyed by the
host-qualified device id and bound to the digest of what the node declared.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from leapflow.hardware.fleet import FleetError, FleetNode, fetch_node_declarations
from leapflow.hardware.attestation import declaration_digest, store_verification

if TYPE_CHECKING:
    from leapflow.hardware.fleet import FleetManager

logger = logging.getLogger(__name__)

FLEET_NODES_ENV = "LEAPFLOW_FLEET_NODES"


def configured_nodes_from_settings(raw: str = "") -> tuple[FleetNode, ...]:
    """Parse typed Settings fleet JSON, falling back to the legacy environment.

    A settings entry may name ``token_env`` instead of embedding a shared secret.
    Diagnostics intentionally never print the raw entry or resolved token.
    """
    source = "hardware_fleet_nodes"
    raw = str(raw or "").strip()
    if not raw:
        raw = os.environ.get(FLEET_NODES_ENV, "")
        source = FLEET_NODES_ENV
    if not raw:
        return ()
    try:
        configs = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("%s is not valid JSON; ignoring", source)
        return ()
    if not isinstance(configs, list):
        logger.warning("%s must be a JSON array; ignoring", source)
        return ()

    nodes: list[FleetNode] = []
    for index, config in enumerate(configs):
        if not isinstance(config, Mapping):
            logger.warning("Fleet: ignoring %s entry #%d: not a JSON object", source, index)
            continue
        token_env = str(config.get("token_env") or "").strip()
        token = os.environ.get(token_env, "") if token_env else str(config.get("token") or "")
        node = FleetNode(
            node_id=str(config.get("node_id") or ""),
            display_name=str(config.get("display_name") or ""),
            endpoint=str(config.get("endpoint") or config.get("mcp_endpoint") or ""),
            token=token,
        )
        if not node.node_id or not node.endpoint:
            logger.warning(
                "Fleet: ignoring %s entry #%d without node_id/endpoint (node_id=%r, endpoint=%r)",
                source,
                index,
                node.node_id,
                node.endpoint,
            )
            continue
        if token_env and not token:
            logger.warning("Fleet: node %r token_env %r is unset", node.node_id, token_env)
        nodes.append(node)
    return tuple(nodes)


def configured_nodes_from_env() -> tuple[FleetNode, ...]:
    """Compatibility wrapper for callers using ``LEAPFLOW_FLEET_NODES``."""
    return configured_nodes_from_settings()


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
            logger.warning("Fleet: node %r registration raised", node.node_id, exc_info=result)
        else:
            registered.append(result)
    return tuple(registered)


# ---------------------------------------------------------------------------
# Operator confirmation of node declarations
# ---------------------------------------------------------------------------


async def confirm_node_device(
    node: FleetNode,
    device_id: str,
    *,
    verifier: str,
    verified_path: Path | str,
) -> dict[str, Any]:
    """Record an operator's review of one device as the node now declares it.

    The node's own ``verified.json`` cannot stand in for this: a machine able
    to rewrite its declaration is also able to rewrite its claim that a human
    read it.  So the host fetches what the node declares *now*, binds that
    digest to the reviewing operator, and records it under the host-qualified
    id -- the exact entry ``FleetManager`` requires before it lets a remote
    device's own verification claim through.

    A read-only exchange on purpose: no fleet is started, so confirming a
    declaration cannot halt the node's hardware as a side effect.

    Raises ``FleetError`` when the node is unreachable or does not declare
    ``device_id``.
    """
    _, contexts, _ = await fetch_node_declarations(node.endpoint, token=node.token)
    declared = {context.device_id: context for context in contexts}
    context = declared.get(device_id)
    if context is None:
        raise FleetError(
            f"node {node.node_id!r} declares no device {device_id!r}",
            failure_code="fleet_unknown_device",
        )
    digest = declaration_digest(context)
    host_device_id = node.host_device_id(device_id)
    entry = store_verification(verified_path, host_device_id, verifier=verifier, digest=digest)
    return {
        "node_id": node.node_id,
        "device_id": device_id,
        "host_device_id": host_device_id,
        "declaration_digest": digest,
        **entry,
    }


__all__ = [
    "FLEET_NODES_ENV",
    "confirm_node_device",
    "configured_nodes_from_env",
    "configured_nodes_from_settings",
    "register_nodes_safely",
]
