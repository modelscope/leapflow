# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shell and terminal session plugin — one-shot commands and persistent sessions."""

from __future__ import annotations

from typing import Any

from leapflow.plugins.protocol import ToolMetadata


class ShellTerminalPlugin:
    """Shell execution and persistent terminal session tools (approval-gated)."""

    def __init__(self) -> None:
        self._approval_gate: Any = None

    @property
    def plugin_id(self) -> str:
        return "shell_terminal"

    @property
    def category(self) -> str:
        return "shell"

    @property
    def tools(self) -> list[ToolMetadata]:
        from leapflow.tools.shell_tools import shell_run
        from leapflow.tools.terminal_session import (
            terminal_close,
            terminal_list,
            terminal_open,
            terminal_read,
            terminal_send,
        )

        return [
            ToolMetadata(
                name="shell_run",
                description=(
                    "Execute a one-shot shell command with timeout protection. Runs in the "
                    "active workspace; paths resolving outside it are refused. Reach for a "
                    "structured tool first when one fits \u2014 web_fetch for anything over "
                    "HTTP(S), git_query/scm_sync for git, code_search/file_find/file_read for "
                    "the repo, config_get/config_set for LeapFlow's own settings \u2014 because "
                    "those report typed results, while a failed shell command can only be "
                    "diagnosed from its exit code and stderr. Every shell run counts as an "
                    "external side effect, so a failure stops the rest of the batch and is "
                    "not retried automatically."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"},
                        "cwd": {"type": "string", "description": "Working directory (optional)"},
                        "timeout": {
                            "type": "number",
                            "description": "Timeout in seconds (default: 30, max: 120)",
                        },
                    },
                    "required": ["command"],
                },
                handler=shell_run,
                x_leapflow={
                    "category": "shell",
                    "risk_level": "external",
                    "schema_cost": "low",
                    "requires_approval": True,
                    "mutates_state": True,
                    "effect_scope": "external",
                    "idempotency_scope": "session",
                },
                mutates_state=True,
                # Grounded in leapflow.domain.platform.Capability.SHELL_EXEC:
                # a host without shell execution cannot run this tool, which
                # environment-fit scoring uses to exclude it rather than fail
                # the call at runtime.
                requires_platform_capabilities=("shell.exec",),
                provides_capabilities=("shell.execute",),
            ),
            ToolMetadata(
                name="terminal_open",
                description=(
                    "Open a PERSISTENT shell session (REPL/dev server/watch), returning a session_id "
                    "for terminal_send/read/close. Disabled unless tools.terminal_session_enabled is "
                    "set. For one-shot commands use shell_run instead."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Optional initial command to run in the session",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Working directory (default: current dir)",
                        },
                        "shell": {
                            "type": "string",
                            "description": "Shell to launch (default: $SHELL or /bin/bash)",
                        },
                    },
                },
                handler=terminal_open,
                x_leapflow={
                    "category": "terminal",
                    "risk_level": "high",
                    "schema_cost": "medium",
                    "requires_approval": True,
                    "effect_scope": "external",
                },
                mutates_state=True,
                provides_capabilities=("shell.session_create",),
                requires_platform_capabilities=("shell.exec",),
            ),
            ToolMetadata(
                name="terminal_send",
                description="Send a line of input to a persistent terminal session and return output captured shortly after.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session id from terminal_open",
                        },
                        "input": {"type": "string", "description": "Line of input to send"},
                        "wait": {
                            "type": "number",
                            "description": "Seconds to wait for output before reading (default 0.3, max 10)",
                        },
                    },
                    "required": ["session_id"],
                },
                handler=terminal_send,
                x_leapflow={
                    "category": "terminal",
                    "risk_level": "high",
                    "schema_cost": "low",
                    "requires_approval": True,
                    "effect_scope": "external",
                },
                mutates_state=True,
                provides_capabilities=("shell.session_input",),
                requires_platform_capabilities=("shell.exec",),
            ),
            ToolMetadata(
                name="terminal_read",
                description="Drain buffered output from a persistent terminal session (optionally waiting briefly first).",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session id from terminal_open",
                        },
                        "wait": {
                            "type": "number",
                            "description": "Seconds to wait before draining (default 0, max 10)",
                        },
                    },
                    "required": ["session_id"],
                },
                handler=terminal_read,
                x_leapflow={
                    "category": "terminal",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("shell.session_output",),
                requires_platform_capabilities=("shell.exec",),
            ),
            ToolMetadata(
                name="terminal_close",
                description="Terminate a persistent terminal session and release its process group.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session id from terminal_open",
                        },
                    },
                    "required": ["session_id"],
                },
                handler=terminal_close,
                x_leapflow={
                    "category": "terminal",
                    "risk_level": "medium",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                mutates_state=True,
                provides_capabilities=("shell.session_destroy",),
                requires_platform_capabilities=("shell.exec",),
            ),
            ToolMetadata(
                name="terminal_list",
                description="List active persistent terminal sessions.",
                parameters_schema={"type": "object", "properties": {}},
                handler=terminal_list,
                x_leapflow={
                    "category": "terminal",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("shell.session_list",),
                requires_platform_capabilities=("shell.exec",),
            ),
        ]

    @property
    def dependencies(self) -> list[str]:
        return ["desktop_gate"]

    def bind_runtime(self, **deps: Any) -> None:
        if "desktop_gate" in deps:
            self._approval_gate = deps["desktop_gate"]


# Module-level instance for plugin discovery
plugin = ShellTerminalPlugin()
