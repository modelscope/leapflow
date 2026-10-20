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
from pathlib import Path
from typing import Any, Sequence

from leapflow.hardware.lhp.authority import WRITE_MODE_PERMIT_REQUIRED
from leapflow.hardware.lhp.protocol import (
    LHP_DEFAULT_PORT,
    LHP_PROTOCOL_VERSION,
    is_valid_device_id,
)
from leapflow.hardware.lhp.server import LhpNodeServer

logger = logging.getLogger(__name__)

_NODE_ID_ENV = "LEAPFLOW_LHP_NODE_ID"
_TOKEN_ENV = "LEAPFLOW_LHP_TOKEN"
_WRITE_MODE_ENV = "LEAPFLOW_LHP_WRITE_MODE"
_PERMIT_KEYS_ENV = "LEAPFLOW_LHP_PERMIT_ISSUER_KEYS"


def resolve_endpoint_args(node_id: str, token: str) -> tuple[str, str]:
    """Resolve node id and token from flags with environment fallbacks.

    Shared by both entries (``python -m`` and ``leap hw serve``) so the
    environment variable names have exactly one definition.
    """
    resolved_id = str(node_id or "").strip() or os.environ.get(_NODE_ID_ENV, "").strip()
    resolved_token = str(token or "") or os.environ.get(_TOKEN_ENV, "")
    return resolved_id, resolved_token


def _resolve_write_mode(write_mode: str) -> str:
    """Resolve write mode from flag or environment."""
    wm = str(write_mode or "").strip() or os.environ.get(_WRITE_MODE_ENV, "").strip()
    return wm or WRITE_MODE_PERMIT_REQUIRED


def _load_permit_verifier(key_paths: list[str], *, node_id: str) -> Any:
    """Load a PermitVerifier from issuer public key PEM files."""
    # Combine flag paths and environment paths
    env_paths = os.environ.get(_PERMIT_KEYS_ENV, "").strip()
    all_paths = list(key_paths)
    if env_paths:
        all_paths.extend(p.strip() for p in env_paths.split(os.pathsep) if p.strip())
    if not all_paths:
        return None
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    from leapflow.hardware.lhp.permit import PermitVerifier, key_id_from_public, public_key_from_pem

    trusted: dict[str, Ed25519PublicKey] = {}
    for path_str in all_paths:
        path = Path(path_str)
        if not path.exists():
            sys.stderr.write(f"warning: permit issuer key not found: {path}\n")
            continue
        try:
            pub = public_key_from_pem(path.read_bytes())
            kid = key_id_from_public(pub)
            trusted[kid] = pub
            logger.info("Loaded permit issuer key %s from %s", kid, path)
        except Exception as exc:
            sys.stderr.write(f"warning: failed to load key {path}: {exc}\n")
    if not trusted:
        return PermitVerifier(trusted, node_id=node_id)
    return PermitVerifier(trusted, node_id=node_id)


async def serve_node(
    *,
    node_id: str,
    host: str = "127.0.0.1",
    port: int = LHP_DEFAULT_PORT,
    unix_path: str = "",
    token: str = "",
    write_mode: str = WRITE_MODE_PERMIT_REQUIRED,
    permit_issuer_keys: list[str] | None = None,
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
    from leapflow.hardware.skills.runtime import SkillRuntime, SkillRuntimeSettings

    config = load_config()
    settings = HardwareSettings.from_settings(config)
    registry = HardwareRegistry(replace(settings, persist_readings=False, stream_enabled=False))
    registry.load()
    contexts = registry.contexts()
    if not contexts:
        sys.stderr.write(
            "warning: no devices admitted for this profile; serving an empty node. "
            "Declare devices (`leap hw list` shows the active profile's view) and retry.\n"
        )

    verifier = _load_permit_verifier(permit_issuer_keys or [], node_id=node_id)
    skill_runtime = SkillRuntime(registry, settings=SkillRuntimeSettings.from_settings(config))
    server = LhpNodeServer(
        registry,
        node_id=node_id,
        token=token,
        write_mode=write_mode,
        permit_verifier=verifier,
        skill_runtime=skill_runtime,
    )
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
            "skill_ids": list(skill_runtime.skill_ids),
            "workcell_count": len(skill_runtime.list().get("workcells", ())),
            "token_required": bool(token),
            "write_mode": write_mode,
            "trusted_issuers": list(getattr(verifier, "trusted_key_ids", ()) or ())
            if verifier is not None
            else [],
        }
        print(json.dumps(ready, ensure_ascii=False), flush=True)
        logger.info("LHP node %r serving %d device(s) at %s", node_id, len(contexts), endpoint)
        await stop.wait()
    finally:
        await server.stop()
        await skill_runtime.close()
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
    parser.add_argument("--host", default="127.0.0.1", help="TCP bind address (default: 127.0.0.1)")
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
    parser.add_argument(
        "--write-mode",
        default="",
        help=(
            f"Write authorization mode: permit_required (default), trusted_v1, "
            f"or observe_only (env {_WRITE_MODE_ENV})"
        ),
    )
    parser.add_argument(
        "--permit-issuer-key",
        action="append",
        default=[],
        dest="permit_issuer_keys",
        help=(
            f"Path to an Ed25519 public key PEM of a trusted permit issuer "
            f"(repeatable; env {_PERMIT_KEYS_ENV} with {os.pathsep!r} separator)"
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    node_id, token = resolve_endpoint_args(args.node_id, args.token)
    write_mode = _resolve_write_mode(args.write_mode)
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
                write_mode=write_mode,
                permit_issuer_keys=args.permit_issuer_keys,
            )
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive interrupt
        return 130


__all__ = ["main", "resolve_endpoint_args", "serve_node"]
