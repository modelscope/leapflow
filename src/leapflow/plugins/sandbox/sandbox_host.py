# Copyright (c) Alibaba, Inc. and its affiliates.
"""Sandbox host: manages worker subprocesses and proxies tool calls.

The host launches a worker subprocess per sandboxed plugin, then proxies
tool invocations to it via JSON-RPC. A SandboxedToolPlugin presents the
same interface as an in-process ToolPlugin, but its handlers marshal calls
to the subprocess.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from leapflow.plugins.protocol import ToolMetadata
from leapflow.plugins.sandbox.protocol import SandboxRequest, SandboxResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SandboxLimits:
    """Cold-path process and RPC limits for one sandbox worker."""

    invoke_timeout_s: float = 30.0
    shutdown_timeout_s: float = 3.0
    cpu_time_s: int = 0
    max_memory_bytes: int = 0

    def __post_init__(self) -> None:
        if self.invoke_timeout_s <= 0 or self.shutdown_timeout_s <= 0:
            raise ValueError("sandbox timeouts must be positive")
        if self.cpu_time_s < 0 or self.max_memory_bytes < 0:
            raise ValueError("sandbox resource limits cannot be negative")


class SandboxHost:
    """Manages a worker subprocess for one sandboxed plugin."""

    def __init__(
        self,
        plugin_module_path: str,
        *,
        invoke_timeout_s: float = 30.0,
        shutdown_timeout_s: float = 3.0,
        cpu_time_s: int = 0,
        max_memory_bytes: int = 0,
        python_paths: Sequence[str] = (),
    ) -> None:
        self._module_path = plugin_module_path
        self._python_paths = tuple(str(path) for path in python_paths if str(path))
        self._limits = SandboxLimits(
            invoke_timeout_s=float(invoke_timeout_s),
            shutdown_timeout_s=float(shutdown_timeout_s),
            cpu_time_s=int(cpu_time_s),
            max_memory_bytes=int(max_memory_bytes),
        )
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._lock = asyncio.Lock()  # serialize stdin/stdout access

    async def start(self) -> None:
        """Launch the worker subprocess."""
        env = dict(os.environ)
        env["LEAPFLOW_SANDBOX_CPU_TIME_S"] = str(self._limits.cpu_time_s)
        env["LEAPFLOW_SANDBOX_MAX_MEMORY_BYTES"] = str(self._limits.max_memory_bytes)
        if self._python_paths:
            current = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(
                [*self._python_paths, *([current] if current else [])]
            )
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "leapflow.plugins.sandbox.worker",
            self._module_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        logger.info(
            "Sandbox worker started for %s (pid %s)",
            self._module_path,
            self._proc.pid,
        )

    async def invoke(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> SandboxResponse:
        """Invoke a tool in the sandbox. Returns the response."""
        return await self._send_request(
            method="invoke_tool", tool_name=tool_name, arguments=arguments
        )

    async def ping(self) -> bool:
        """Health check the worker."""
        resp = await self._send_request(method="ping")
        return resp.ok

    async def list_tools(self) -> List[str]:
        """List tools available in the sandbox."""
        resp = await self._send_request(method="list_tools")
        return resp.result if resp.ok else []

    async def _send_request(
        self,
        method: str,
        tool_name: str = "",
        arguments: Optional[Dict[str, Any]] = None,
    ) -> SandboxResponse:
        """Send a request to the worker and wait for a response."""
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            return SandboxResponse(
                request_id="", ok=False, error="Sandbox not started"
            )

        req = SandboxRequest(
            request_id=str(uuid.uuid4()),
            method=method,
            tool_name=tool_name,
            arguments=arguments or {},
        )

        async with self._lock:
            try:
                self._proc.stdin.write((req.to_json() + "\n").encode())
                await self._proc.stdin.drain()
                line = await asyncio.wait_for(
                    self._proc.stdout.readline(), timeout=self._limits.invoke_timeout_s
                )
                if not line:
                    return SandboxResponse(
                        request_id=req.request_id,
                        ok=False,
                        error="Worker closed unexpectedly",
                    )
                return SandboxResponse.from_json(line.decode().strip())
            except asyncio.TimeoutError:
                await self._kill_worker()
                return SandboxResponse(
                    request_id=req.request_id,
                    ok=False,
                    error=f"Sandbox invoke timed out after {self._limits.invoke_timeout_s}s",
                )
            except (ConnectionResetError, OSError, ValueError) as exc:
                return SandboxResponse(
                    request_id=req.request_id,
                    ok=False,
                    error=f"Sandbox communication error: {exc}",
                )

    async def stop(self) -> None:
        """Shut down the worker subprocess gracefully, then forcibly if needed."""
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                req = SandboxRequest(request_id="shutdown", method="shutdown")
                self._proc.stdin.write((req.to_json() + "\n").encode())
                await self._proc.stdin.drain()
            await asyncio.wait_for(
                self._proc.wait(), timeout=self._limits.shutdown_timeout_s
            )
        except (asyncio.TimeoutError, ConnectionResetError, OSError):
            try:
                self._proc.kill()
                await self._proc.wait()
            except (ProcessLookupError, OSError):
                pass
        finally:
            self._proc = None
            logger.info("Sandbox worker stopped for %s", self._module_path)

    async def _kill_worker(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            proc.kill()
            await proc.wait()
        except (ProcessLookupError, OSError):
            pass
        finally:
            self._proc = None


class SandboxedToolPlugin:
    """A ToolPlugin whose handlers execute in a sandbox subprocess.

    Presents the same interface as an in-process plugin (plugin_id, category,
    tools, dependencies, bind_runtime) but each tool's handler proxies the
    call to the sandbox host.
    """

    def __init__(
        self,
        plugin_id: str,
        category: str,
        tool_metadatas: list,
        host: SandboxHost,
    ) -> None:
        self._plugin_id = plugin_id
        self._category = category
        self._host = host
        # Build tools with proxied handlers
        self._tools = [self._wrap_metadata(m) for m in tool_metadatas]

    def _wrap_metadata(self, meta: "ToolMetadata") -> "ToolMetadata":
        """Replace the handler with a sandbox-proxying handler."""
        from leapflow.plugins.protocol import ToolMetadata

        host = self._host

        async def _proxy_handler(**kwargs: Any) -> Any:
            resp = await host.invoke(meta.name, kwargs)
            if resp.ok:
                return resp.result
            return {"ok": False, "error": resp.error, "error_type": resp.error_type}

        return ToolMetadata(
            name=meta.name,
            description=meta.description,
            parameters_schema=meta.parameters_schema,
            handler=_proxy_handler,
            x_leapflow=meta.x_leapflow,
            mutates_state=meta.mutates_state,
        )

    @property
    def plugin_id(self) -> str:
        """Unique plugin identifier."""
        return self._plugin_id

    @property
    def category(self) -> str:
        """Tool category label."""
        return self._category

    @property
    def tools(self) -> list:
        """List of ToolMetadata with proxied handlers."""
        return self._tools

    @property
    def dependencies(self) -> list:
        """Sandboxed plugins have no host-side dependencies."""
        return []

    def bind_runtime(self, **deps: Any) -> None:
        """No-op: sandboxed plugins don't receive host runtime deps."""
