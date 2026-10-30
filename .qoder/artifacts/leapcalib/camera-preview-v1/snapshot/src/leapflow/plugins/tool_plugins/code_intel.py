# Copyright (c) Alibaba, Inc. and its affiliates.
"""Code intelligence plugin — document symbols and repository map."""

from __future__ import annotations

from leapflow.tools.code_intel import code_intel
from leapflow.plugins.protocol import ToolMetadata
from leapflow.tools.repo_map import repo_map


class CodeIntelPlugin:
    """Read-only code analysis: AST symbols and project orientation."""

    @property
    def plugin_id(self) -> str:
        return "code_intel"

    @property
    def category(self) -> str:
        return "file"

    @property
    def tools(self) -> list[ToolMetadata]:
        return [
            ToolMetadata(
                name="code_intel",
                description=(
                    "Precise document symbols (outline) for a source file: classes, functions, and "
                    "methods with line ranges. Python uses an exact AST parse; other languages use a "
                    "keyword-prefix scan. Prefer over file_read mode=symbols for accurate navigation "
                    "before editing. Read-only."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Source file to analyze"},
                        "operation": {
                            "type": "string",
                            "enum": ["symbols"],
                            "description": "Analysis operation (default: symbols)",
                        },
                    },
                    "required": ["path"],
                },
                handler=code_intel,
                x_leapflow={
                    "category": "file",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("code.symbols",),
            ),
            ToolMetadata(
                name="repo_map",
                description=(
                    "Compact project orientation for a repository root: languages, detected test/lint "
                    "commands, top-level structure, entry points, manifest, and VCS branch. Call this "
                    "first when entering an unfamiliar codebase. Read-only."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Repository root (default: current dir)",
                        },
                    },
                },
                handler=repo_map,
                x_leapflow={
                    "category": "file",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("code.repo_map",),
            ),
        ]

    @property
    def dependencies(self) -> list[str]:
        return []

    def bind_runtime(self, **deps: object) -> None:
        pass


# Module-level instance for plugin discovery
plugin = CodeIntelPlugin()
