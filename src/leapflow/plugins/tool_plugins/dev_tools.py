# Copyright (c) Alibaba, Inc. and its affiliates.
"""Dev tools plugin — test runner and linter integration."""

from __future__ import annotations

from leapflow.tools.dev_tools import lint_check, test_run
from leapflow.plugins.protocol import ToolMetadata


class DevToolsPlugin:
    """Auto-detecting test and lint runners with structured results."""

    @property
    def plugin_id(self) -> str:
        return "dev_tools"

    @property
    def category(self) -> str:
        return "dev"

    @property
    def tools(self) -> list[ToolMetadata]:
        return [
            ToolMetadata(
                name="test_run",
                description=(
                    "Run the project's test suite and return structured results (framework, passed/"
                    "failed counts, failing tests). Auto-detects the runner (pytest/npm/go/cargo) or "
                    "uses a configured/explicit command; executes via the governed shell. ok=true means "
                    "the runner executed — see 'success' for pass/fail."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Explicit test command (optional; overrides auto-detect)",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Working directory (default: current dir)",
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Timeout seconds (default 120, max 120)",
                        },
                    },
                },
                handler=test_run,
                x_leapflow={
                    "category": "dev",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("dev.test",),
                requires_platform_capabilities=("shell.exec",),
            ),
            ToolMetadata(
                name="lint_check",
                description=(
                    "Run the project's linter and return a structured clean/issue result. Auto-detects "
                    "the linter (ruff/eslint/go vet/clippy) or uses a configured/explicit command; "
                    "executes via the governed shell. ok=true means the linter ran — see 'clean'."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Explicit lint command (optional; overrides auto-detect)",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Working directory (default: current dir)",
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Timeout seconds (default 120, max 120)",
                        },
                    },
                },
                handler=lint_check,
                x_leapflow={
                    "category": "dev",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("dev.lint",),
                requires_platform_capabilities=("shell.exec",),
            ),
        ]

    @property
    def dependencies(self) -> list[str]:
        return []

    def bind_runtime(self, **deps: object) -> None:
        pass


# Module-level instance for plugin discovery
plugin = DevToolsPlugin()
