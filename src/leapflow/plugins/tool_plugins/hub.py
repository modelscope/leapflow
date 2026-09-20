# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hub tools plugin — exposes Hub operations (push, pull, search, sync) as a ToolPlugin."""

from __future__ import annotations

from typing import Any

from leapflow.plugins.protocol import ToolMetadata


class HubToolsPlugin:
    """Agent-facing Hub skill management tools."""

    @property
    def plugin_id(self) -> str:
        return "hub"

    @property
    def category(self) -> str:
        return "hub"

    @property
    def tools(self) -> list[ToolMetadata]:
        from leapflow.tools.hub_tool import (
            hub_pull_tool,
            hub_push_tool,
            hub_search_tool,
            hub_sync_tool,
        )

        return [
            ToolMetadata(
                name="hub_push",
                description="Push a local skill to the ModelScope Hub for sharing or backup.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Name of the local skill to push",
                        },
                        "visibility": {
                            "type": "string",
                            "enum": ["private", "public", "internal"],
                            "description": "Repository visibility (default: private)",
                        },
                        "version": {
                            "type": "string",
                            "description": "Version string (default: auto-detect from skill)",
                        },
                    },
                    "required": ["skill_name"],
                },
                handler=hub_push_tool,
                x_leapflow={
                    "category": "hub",
                    "risk_level": "medium",
                    "schema_cost": "high",
                    "requires_approval": True,
                },
                mutates_state=True,
                execution_policy="external_side_effect",
                provides_capabilities=("hub.push",),
                requires_platform_capabilities=("file.ops",),
            ),
            ToolMetadata(
                name="hub_pull",
                description="Pull a skill from the ModelScope Hub to install locally.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "repo_id": {
                            "type": "string",
                            "description": "Repository identifier (e.g. 'owner/leapflow-skill-name')",
                        },
                        "version": {
                            "type": "string",
                            "description": "Specific version to pull (default: latest)",
                        },
                    },
                    "required": ["repo_id"],
                },
                handler=hub_pull_tool,
                x_leapflow={
                    "category": "hub",
                    "risk_level": "medium",
                    "schema_cost": "high",
                    "requires_approval": True,
                },
                mutates_state=True,
                execution_policy="external_side_effect",
                provides_capabilities=("hub.pull",),
                requires_platform_capabilities=("file.ops",),
            ),
            ToolMetadata(
                name="hub_search",
                description="Search for skills on the Hub by keyword or description.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Free-text search query for finding skills",
                        },
                    },
                    "required": ["query"],
                },
                handler=hub_search_tool,
                x_leapflow={
                    "category": "hub",
                    "risk_level": "read_only",
                    "schema_cost": "high",
                    "requires_approval": False,
                },
                provides_capabilities=("hub.search",),
            ),
            ToolMetadata(
                name="hub_sync",
                description="Preview or execute sync between local skills and Hub.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["full", "push-only", "pull-only"],
                            "description": "Sync mode (default: full)",
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "If true, only shows the plan (default: true)",
                        },
                    },
                },
                handler=hub_sync_tool,
                x_leapflow={
                    "category": "hub",
                    "risk_level": "medium",
                    "schema_cost": "high",
                    "requires_approval": True,
                },
                mutates_state=True,
                execution_policy="external_side_effect",
                provides_capabilities=("hub.sync",),
                requires_platform_capabilities=("file.ops",),
            ),
        ]

    @property
    def dependencies(self) -> list[str]:
        return []

    def bind_runtime(self, **deps: Any) -> None:
        pass


# Module-level instance for plugin discovery
plugin = HubToolsPlugin()
