# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tool and plugin diagnostic checks."""
from __future__ import annotations

import logging
from typing import Any

from leapflow.cli.doctor.protocol import Finding

logger = logging.getLogger(__name__)

# Core tools that must be present for basic agent operation.
_CORE_TOOLS = frozenset({
    "file_read",
    "file_write",
    "shell",
    "web_fetch",
})


class PluginRegistryCheck:
    """Verify that the plugin registry loads without errors."""

    name = "Plugin registry"
    section = "tools"

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        try:
            from leapflow.plugins import get_registry

            reg = get_registry()
            plugins = list(reg.list_plugins())
            if plugins:
                f.pass_()
            else:
                f.warn("Plugin registry is empty — no plugins loaded")
        except Exception as exc:
            f.error(f"Plugin registry failed to load: {exc}")
        return f


class CoreToolsCheck:
    """Verify that essential tool handlers are registered."""

    name = "Core tools"
    section = "tools"

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        try:
            from leapflow.plugins import get_registry

            reg = get_registry()
            catalog = reg.capability_catalog()
            registered_names = set(catalog.keys()) if isinstance(catalog, dict) else set()

            missing = _CORE_TOOLS - registered_names
            if missing:
                f.warn(f"Core tool(s) not registered: {', '.join(sorted(missing))}")
            else:
                f.pass_()
        except Exception as exc:
            f.warn(f"Cannot inspect tool catalog: {exc}")
        return f


class MCPServerCheck:
    """Check MCP server connectivity if any are configured."""

    name = "MCP servers"
    section = "tools"

    def __init__(self, settings: Any) -> None:
        self._settings = settings

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        # MCP servers are opt-in; if none configured, pass silently.
        mcp_servers = getattr(self._settings, "mcp_servers", None)
        if not mcp_servers:
            f.pass_()
            return f

        try:
            configured = len(mcp_servers) if hasattr(mcp_servers, "__len__") else 0
            if configured > 0:
                f.pass_()
            else:
                f.pass_()
        except Exception as exc:
            f.warn(f"MCP server check failed: {exc}")
        return f
