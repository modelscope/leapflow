# Copyright (c) Alibaba, Inc. and its affiliates.
"""Skill discovery plugin — list and view learned skills."""

from __future__ import annotations

from leapflow.skills.discovery import skill_file_read, skill_list_files, skill_view, skills_list
from leapflow.plugins.protocol import ToolMetadata


class SkillDiscoveryPlugin:
    """Read-only tools for browsing the agent's learned skill library."""

    @property
    def plugin_id(self) -> str:
        return "skill_discovery"

    @property
    def category(self) -> str:
        return "read"

    @property
    def tools(self) -> list[ToolMetadata]:
        return [
            ToolMetadata(
                name="skills_list",
                description="List available learned skills. Use when user asks about capabilities or you need a specific skill.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Optional keyword filter"},
                        "category": {
                            "type": "string",
                            "description": "Filter by category (e.g. file-mgmt, apple)",
                        },
                        "source": {
                            "type": "string",
                            "description": "Filter by source: learned, manual, or hub",
                        },
                    },
                },
                handler=skills_list,
                x_leapflow={"category": "read", "plane": "task"},
                provides_capabilities=("skill.list",),
            ),
            ToolMetadata(
                name="skill_view",
                description="View the full content of a specific skill document.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Skill name to view"},
                    },
                    "required": ["name"],
                },
                handler=skill_view,
                x_leapflow={"category": "read", "plane": "task"},
                provides_capabilities=("skill.view",),
            ),
            ToolMetadata(
                name="skill_list_files",
                description="List all support files in a skill directory (configs, templates, examples). Use before skill_file_read to discover available files.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Skill name to list files for"},
                    },
                    "required": ["name"],
                },
                handler=skill_list_files,
                x_leapflow={"category": "skills", "risk_level": "none", "plane": "task"},
                provides_capabilities=("skill.list_files",),
            ),
            ToolMetadata(
                name="skill_file_read",
                description="Read an individual support file from a skill directory. Use skill_list_files first to discover available files.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Skill name"},
                        "file_path": {
                            "type": "string",
                            "description": "Relative path within the skill directory (no '..' or absolute paths)",
                        },
                    },
                    "required": ["name", "file_path"],
                },
                handler=skill_file_read,
                x_leapflow={"category": "skills", "risk_level": "none", "plane": "task"},
                provides_capabilities=("skill.file_read",),
            ),
        ]

    @property
    def dependencies(self) -> list[str]:
        return []

    def bind_runtime(self, **deps: object) -> None:
        pass


# Module-level instance for plugin discovery
plugin = SkillDiscoveryPlugin()
