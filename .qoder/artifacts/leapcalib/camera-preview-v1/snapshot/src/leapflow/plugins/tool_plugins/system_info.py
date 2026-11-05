# Copyright (c) Alibaba, Inc. and its affiliates.
"""System information plugin — current time and environment info."""

from __future__ import annotations

from leapflow.plugins.protocol import ToolMetadata
from leapflow.tools.system_tools import env_info, time_get


class SystemInfoPlugin:
    """Read-only system introspection tools (time, OS, Python version)."""

    @property
    def plugin_id(self) -> str:
        return "system_info"

    @property
    def category(self) -> str:
        return "general"

    @property
    def tools(self) -> list[ToolMetadata]:
        return [
            ToolMetadata(
                name="time_get",
                description="Get current date and time.",
                parameters_schema={"type": "object", "properties": {}},
                handler=time_get,
                provides_capabilities=("system.time",),
            ),
            ToolMetadata(
                name="env_info",
                description="Get system environment information (OS, Python version, cwd).",
                parameters_schema={"type": "object", "properties": {}},
                handler=env_info,
                provides_capabilities=("system.info",),
            ),
        ]

    @property
    def dependencies(self) -> list[str]:
        return []

    def bind_runtime(self, **deps: object) -> None:
        pass


# Module-level instance for plugin discovery
plugin = SystemInfoPlugin()
