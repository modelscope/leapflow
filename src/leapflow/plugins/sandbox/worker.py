# Copyright (c) Alibaba, Inc. and its affiliates.
"""Sandbox worker entrypoint. Runs in an isolated subprocess.

Loads a plugin module, then serves tool invocation requests over stdin/stdout
using the SandboxRequest/SandboxResponse JSON-RPC protocol.

Security notes:
    - Runs as a separate process (crash isolation)
    - Communicates only via stdin/stdout (no shared memory)
    - A future enhancement can add resource limits (RLIMIT), seccomp, or
      a restricted import hook.

Usage (invoked by SandboxHost):
    python -m leapflow.plugins.sandbox.worker <plugin_module_path>
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import sys
from typing import Any, Callable, Dict

from leapflow.plugins.sandbox.protocol import SandboxRequest, SandboxResponse

logger = logging.getLogger(__name__)


async def _serve(plugin_module_path: str) -> None:
    """Load the plugin and serve requests from stdin."""
    handlers: Dict[str, Callable[..., Any]] = {}

    # Load the plugin module in this isolated process
    try:
        mod = importlib.import_module(plugin_module_path)
        plugin = getattr(mod, "plugin", None)
        if plugin is not None:
            for tool in plugin.tools:
                handlers[tool.name] = tool.handler
    except (ImportError, AttributeError) as exc:
        # Report load failure but keep serving (host will get errors on invoke)
        logger.error("Sandbox worker failed to load %s: %s", plugin_module_path, exc)

    # Serve loop: read line from stdin, process, write response to stdout
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break

        try:
            req = SandboxRequest.from_json(line.decode().strip())
        except (ValueError, json.JSONDecodeError):
            continue

        if req.method == "shutdown":
            break
        elif req.method == "ping":
            resp = SandboxResponse(request_id=req.request_id, ok=True, result="pong")
        elif req.method == "list_tools":
            resp = SandboxResponse(
                request_id=req.request_id, ok=True, result=list(handlers.keys())
            )
        elif req.method == "invoke_tool":
            resp = await _invoke(req, handlers)
        else:
            resp = SandboxResponse(
                request_id=req.request_id,
                ok=False,
                error=f"Unknown method: {req.method}",
            )

        sys.stdout.write(resp.to_json() + "\n")
        sys.stdout.flush()


async def _invoke(
    req: SandboxRequest, handlers: Dict[str, Callable[..., Any]]
) -> SandboxResponse:
    """Invoke a tool handler, catching all exceptions at the isolation boundary."""
    handler = handlers.get(req.tool_name)
    if handler is None:
        return SandboxResponse(
            request_id=req.request_id,
            ok=False,
            error=f"Tool not found: {req.tool_name}",
        )
    try:
        result = handler(**req.arguments)
        if asyncio.iscoroutine(result):
            result = await result
        return SandboxResponse(request_id=req.request_id, ok=True, result=result)
    except Exception as exc:  # noqa: BLE001 — isolation boundary must catch all plugin errors
        return SandboxResponse(
            request_id=req.request_id,
            ok=False,
            error=str(exc),
            error_type=type(exc).__name__,
        )


if __name__ == "__main__":
    module_path = sys.argv[1] if len(sys.argv) > 1 else ""
    asyncio.run(_serve(module_path))
