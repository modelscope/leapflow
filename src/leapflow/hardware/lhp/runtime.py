# Copyright (c) Alibaba, Inc. and its affiliates.
"""Edge runtime: serve this machine's hardware registry as an LHP node.

This is the *node* side entry -- on a Jetson, ``python -m
leapflow.hardware.lhp --node-id jetson01`` and ``leap hw serve`` both land
here. It builds the node's own registry from the active profile with
persistence and streaming disabled (a node answers the wire protocol;
storing readings and owning sampling loops stay host-side concerns in P1),
then answers LHP requests until interrupted.

The listener is deliberately explicit: node id, bind address, and token all
come from flags or environment, never from discovery. Registration is the
host's act; this process only serves what it was told to serve, and the
ready line it prints is the operator's confirmation that a host can now
name this node.

Stopping is graceful on SIGINT/SIGTERM: the listener closes, live
connections are dropped, and node-side transports are released before the
process exits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from typing import Sequence

from leapflow.hardware.lhp.protocol import (
    LHP_DEFAULT_PORT,
    LHP_PROTOCOL_VERSION,
    is_valid_device_id,
)
from leapflow.hardware.lhp.server import LhpNodeServer

logger = logging.getLogger(__name__)

_NODE_ID_ENV = "LEAPFLOW_LHP_NODE_ID"
_TOKEN_ENV = "LEAPFLOW_LHP_TOKEN"


def resolve_endpoint_args(node_id: str, token: str) -> tuple[str, str]:
    """Resolve node id and token from flags with environment fallbacks.

    Shared by both entries (``python -m`` and ``leap hw serve``) so the
    environment variable names have exactly one definition.
    """
    resolved_id = str(node_id or "").strip() or os.environ.get(_NODE_ID_ENV, "").strip()
    resolved_token = str(token or "") or os.environ.get(_TOKEN_ENV, "")
    return resolved_id, resolved_token


async def serve_node(
    *,
    node_id: str,
    host: str = "127.0.0.1",
    port: int = LHP_DEFAULT_PORT,
    unix_path: str = "",
    token: str = "",
) -> int:
    """Serve the profile's hardware registry over LHP until interrupted.

    Returns a process exit code. A node with zero admitted devices still
    serves (an empty device list is a valid, diagnosable answer); the
    warning on stderr is there so the operator notices before the host does.
    """
    # Function-local imports: the wire-protocol package stays light, and the
    # registry/config dependency is visible exactly where it is taken.
    from dataclasses import replace

    from leapflow.config import load_config
    from leapflow.hardware.registry import HardwareRegistry, HardwareSettings

    settings = HardwareSettings.from_settings(load_config())
    # A node answers requests; it does not accumulate history or own sampling
    # loops in P1. Storage and streaming stay host-side concerns.
    registry = HardwareRegistry(
        replace(settings, persist_readings=False, stream_enabled=False)
    )
    registry.load()
    contexts = registry.contexts()
    if not contexts:
        sys.stderr.write(
            "warning: no devices admitted for this profile; serving an empty node. "
            "Declare devices (`leap hw list` shows the active profile's view) and retry.\n"
        )

    server = LhpNodeServer(registry, node_id=node_id, token=token)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # pragma: no cover - platform without loop signals
            pass
    try:
        if unix_path:
            endpoint = await server.start_unix(unix_path)
        else:
            bound_host, bound_port = await server.start(host=host, port=port)
            endpoint = f"tcp://{bound_host}:{bound_port}"
        ready = {
            "ok": True,
            "event": "lhp_node_ready",
            "protocol": LHP_PROTOCOL_VERSION,
            "node_id": node_id,
            "endpoint": endpoint,
            "device_ids": [context.device_id for context in contexts],
            "device_count": len(contexts),
            "token_required": bool(token),
        }
        print(json.dumps(ready, ensure_ascii=False), flush=True)
        logger.info("LHP node %r serving %d device(s) at %s", node_id, len(contexts), endpoint)
        await stop.wait()
    finally:
        await server.stop()
        await registry.close_all()
    logger.info("LHP node %r stopped", node_id)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m leapflow.hardware.lhp``."""
    parser = argparse.ArgumentParser(
        prog="python -m leapflow.hardware.lhp",
        description=(
            "Serve this machine's hardware devices as an LHP fabric node "
            "(the version-negotiated LeapFlow Hardware Protocol)."
        ),
    )
    parser.add_argument(
        "--node-id",
        default="",
        help=f"Node identity, [a-z0-9_]+ (env {_NODE_ID_ENV})",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="TCP bind address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=LHP_DEFAULT_PORT,
        help=f"TCP bind port (default: {LHP_DEFAULT_PORT}; 0 picks a free port)",
    )
    parser.add_argument(
        "--unix",
        default="",
        dest="unix_path",
        help="Bind a Unix-domain socket at this path instead of TCP",
    )
    parser.add_argument(
        "--token",
        default="",
        help=f"Shared secret required in hello (env {_TOKEN_ENV}); empty means an open node",
    )
    parser.add_argument("--verbose", action="store_true", help="Log at DEBUG level")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    node_id, token = resolve_endpoint_args(args.node_id, args.token)
    if not is_valid_device_id(node_id):
        parser.error(
            f"--node-id is required and must match [a-z0-9_]+ (got {args.node_id!r}); "
            f"or set {_NODE_ID_ENV}"
        )
    try:
        return asyncio.run(
            serve_node(
                node_id=node_id,
                host=args.host,
                port=args.port,
                unix_path=args.unix_path,
                token=token,
            )
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive interrupt
        return 130


__all__ = ["main", "resolve_endpoint_args", "serve_node"]
