# Copyright (c) Alibaba, Inc. and its affiliates.
"""Text utilities plugin — regex search and string replace.

Pilot migration: validates the full ToolPlugin pipeline.
"""

from __future__ import annotations

from leapflow.plugins.protocol import ToolMetadata
from leapflow.tools.text_tools import text_replace, text_search


class TextUtilsPlugin:
    """Pure in-memory text manipulation tools (no I/O, no state mutation)."""

    @property
    def plugin_id(self) -> str:
        return "text_utils"

    @property
    def category(self) -> str:
        return "general"

    @property
    def tools(self) -> list[ToolMetadata]:
        return [
            ToolMetadata(
                name="text_search",
                description="Search for a regex pattern in text.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to search in"},
                        "pattern": {"type": "string", "description": "Regex pattern to match"},
                    },
                    "required": ["text", "pattern"],
                },
                handler=text_search,
                x_leapflow={
                    "category": "general",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("text.search",),
            ),
            ToolMetadata(
                name="text_replace",
                description="Replace occurrences of a substring in text.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Original text"},
                        "old": {"type": "string", "description": "Substring to find"},
                        "new": {"type": "string", "description": "Replacement string"},
                        "count": {"type": "integer", "description": "Max replacements (0 = all)"},
                    },
                    "required": ["text", "old", "new"],
                },
                handler=text_replace,
                provides_capabilities=("text.replace",),
            ),
        ]

    @property
    def dependencies(self) -> list[str]:
        return []

    def bind_runtime(self, **deps: object) -> None:
        pass


# Module-level instance for plugin discovery
plugin = TextUtilsPlugin()
