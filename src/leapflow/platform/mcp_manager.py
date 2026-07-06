"""MCP Server Manager — generalized MCP client for arbitrary servers.

Design (inspired by hermes tools/mcp_tool.py, generalized from CuaDriverClient):
- Daemon thread + asyncio event loop for bridging sync/async boundaries
- Multiple MCP server connections (stdio, HTTP/SSE future)
- Dynamic tool discovery and registration with ``mcp_`` prefix
- Safe environment for spawned processes (strip host secrets)
- Per-server RPC lock to prevent stdio wedge
- Reconnect with exponential backoff
- Protocol-first: McpServerConfig, McpToolSchema, McpServerManager Protocol
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_MAX_TOOL_WORKERS = 8
_DEFAULT_TOOL_TIMEOUT_S = 300.0
_RECONNECT_BASE_S = 2.0
_RECONNECT_CAP_S = 300.0
_MAX_RECONNECT_ATTEMPTS = 10


@dataclass(frozen=True)
class McpServerConfig:
    """Configuration for a single MCP server connection."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"
    tool_timeout_s: float = _DEFAULT_TOOL_TIMEOUT_S
    parallel_safe: bool = False
    stderr_log_path: Optional[str] = None


@dataclass(frozen=True)
class McpToolSchema:
    """Schema for a discovered MCP tool (OpenAI function-calling format)."""
    name: str
    original_name: str
    server_name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function-calling tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


@runtime_checkable
class McpServerManager(Protocol):
    """Protocol for managing MCP server connections (DIP)."""

    def get_tool_schemas(self) -> List[McpToolSchema]: ...
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any: ...
    def close(self) -> None: ...


def _sanitize_name(name: str) -> str:
    """Sanitize MCP server/tool names for use as function names."""
    return name.replace("-", "_").replace(".", "_").replace(" ", "_")


def _build_safe_env(server_config: McpServerConfig) -> Dict[str, str]:
    """Build safe environment for spawning MCP subprocess.

    Only inherits PATH, HOME, and XDG vars from host; adds explicit config env.
    """
    safe_keys = {
        "PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "TERM",
        "TMPDIR", "TMP", "TEMP",
    }
    safe_keys.update(k for k in os.environ if k.startswith("XDG_"))

    env = {k: v for k, v in os.environ.items() if k in safe_keys}
    env.update(server_config.env)
    return env


def mcp_schema_to_openai(input_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert MCP JSON Schema to OpenAI function parameters schema."""
    result: Dict[str, Any] = {"type": "object"}
    if "properties" in input_schema:
        result["properties"] = input_schema["properties"]
    if "required" in input_schema:
        result["required"] = input_schema["required"]
    if "additionalProperties" in input_schema:
        result["additionalProperties"] = input_schema["additionalProperties"]
    return result


class StdioMcpServer:
    """Manages a single stdio MCP server connection.

    Lifecycle:
    1. Spawn subprocess with safe environment
    2. Initialize MCP session (handshake)
    3. Discover tools
    4. Serve tool calls via RPC lock
    5. Reconnect on failure with backoff
    """

    def __init__(self, config: McpServerConfig) -> None:
        self._config = config
        self._tools: List[McpToolSchema] = []
        self._session: Any = None
        self._rpc_lock = asyncio.Lock()
        self._connected = False
        self._consecutive_failures = 0

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def tools(self) -> List[McpToolSchema]:
        return list(self._tools)

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> List[McpToolSchema]:
        """Connect to MCP server and discover tools."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            logger.warning("mcp_manager: 'mcp' package not installed; MCP servers unavailable")
            return []

        stderr_path = self._config.stderr_log_path
        if not stderr_path:
            log_dir = Path.home() / ".leapflow" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            stderr_path = str(log_dir / f"mcp-{_sanitize_name(self._config.name)}-stderr.log")

        server_params = StdioServerParameters(
            command=self._config.command,
            args=self._config.args,
            env=_build_safe_env(self._config),
        )

        try:
            errlog = open(stderr_path, "a")
            self._stdio_ctx = stdio_client(server_params, errlog=errlog)
            read_stream, write_stream = await self._stdio_ctx.__aenter__()

            self._session_ctx = ClientSession(read_stream, write_stream)
            self._session = await self._session_ctx.__aenter__()

            await self._session.initialize()
            self._connected = True
            self._consecutive_failures = 0

            self._tools = await self._discover_tools()
            logger.info("mcp_manager: connected to '%s', %d tools discovered",
                        self._config.name, len(self._tools))
            return self._tools

        except Exception as e:
            self._connected = False
            logger.error("mcp_manager: failed to connect to '%s': %s", self._config.name, e)
            return []

    async def _discover_tools(self) -> List[McpToolSchema]:
        """List tools from the connected MCP server."""
        if not self._session:
            return []
        try:
            result = await self._session.list_tools()
            schemas: List[McpToolSchema] = []
            for tool in result.tools:
                prefixed_name = f"mcp_{_sanitize_name(self._config.name)}_{_sanitize_name(tool.name)}"
                schemas.append(McpToolSchema(
                    name=prefixed_name,
                    original_name=tool.name,
                    server_name=self._config.name,
                    description=tool.description or "",
                    parameters=mcp_schema_to_openai(tool.inputSchema) if tool.inputSchema else {},
                ))
            return schemas
        except Exception as e:
            logger.warning("mcp_manager: tool discovery failed for '%s': %s", self._config.name, e)
            return []

    async def call_tool(self, original_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on this MCP server with RPC lock."""
        if not self._session or not self._connected:
            return {"ok": False, "error": f"MCP server '{self._config.name}' not connected"}

        async with self._rpc_lock:
            try:
                result = await asyncio.wait_for(
                    self._session.call_tool(original_name, arguments),
                    timeout=self._config.tool_timeout_s,
                )
                self._consecutive_failures = 0

                if hasattr(result, "content") and result.content:
                    parts = []
                    for part in result.content:
                        if hasattr(part, "text"):
                            parts.append(part.text)
                    return {"ok": True, "result": "\n".join(parts)}
                return {"ok": True, "result": str(result)}

            except asyncio.TimeoutError:
                self._consecutive_failures += 1
                return {"ok": False, "error": f"Tool '{original_name}' timed out after {self._config.tool_timeout_s}s"}
            except Exception as e:
                self._consecutive_failures += 1
                return {"ok": False, "error": f"Tool '{original_name}' failed: {e}"}

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        self._connected = False
        try:
            if hasattr(self, "_session_ctx") and self._session_ctx:
                await self._session_ctx.__aexit__(None, None, None)
            if hasattr(self, "_stdio_ctx") and self._stdio_ctx:
                await self._stdio_ctx.__aexit__(None, None, None)
        except Exception as e:
            logger.debug("mcp_manager: disconnect error for '%s': %s", self._config.name, e)
        finally:
            self._session = None


class McpManager:
    """Manages multiple MCP server connections with dynamic tool registration.

    Runs MCP servers on a daemon thread's event loop (same pattern as CuaDriverClient).
    Provides a sync-friendly facade for the engine.
    """

    def __init__(self) -> None:
        self._servers: Dict[str, StdioMcpServer] = {}
        self._tool_map: Dict[str, tuple[StdioMcpServer, str]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Start daemon thread with event loop if not already running."""
        if self._loop is not None and self._loop.is_running():
            return self._loop

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._started.set()
            loop.run_forever()

        self._thread = threading.Thread(target=_run_loop, daemon=True)
        self._thread.start()
        self._started.wait(timeout=5.0)
        assert self._loop is not None
        return self._loop

    def add_server(self, config: McpServerConfig) -> List[McpToolSchema]:
        """Add and connect an MCP server. Returns discovered tool schemas."""
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(
            self._add_server_async(config), loop
        )
        try:
            return future.result(timeout=30.0)
        except Exception as e:
            logger.error("mcp_manager: failed to add server '%s': %s", config.name, e)
            return []

    async def _add_server_async(self, config: McpServerConfig) -> List[McpToolSchema]:
        server = StdioMcpServer(config)
        tools = await server.connect()
        if tools:
            self._servers[config.name] = server
            for tool in tools:
                self._tool_map[tool.name] = (server, tool.original_name)
        return tools

    def get_tool_schemas(self) -> List[McpToolSchema]:
        """Return all discovered tool schemas across all servers."""
        schemas: List[McpToolSchema] = []
        for server in self._servers.values():
            schemas.extend(server.tools)
        return schemas

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Route a tool call to the appropriate MCP server."""
        entry = self._tool_map.get(tool_name)
        if entry is None:
            return {"ok": False, "error": f"Unknown MCP tool: {tool_name}"}
        server, original_name = entry
        return await server.call_tool(original_name, arguments)

    def call_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Synchronous wrapper for call_tool."""
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(
            self.call_tool(tool_name, arguments), loop
        )
        try:
            return future.result(timeout=_DEFAULT_TOOL_TIMEOUT_S)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def close(self) -> None:
        """Disconnect all servers and stop the event loop."""
        if self._loop and self._loop.is_running():
            for server in self._servers.values():
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        server.disconnect(), self._loop
                    )
                    future.result(timeout=5.0)
                except Exception:
                    pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._servers.clear()
        self._tool_map.clear()
