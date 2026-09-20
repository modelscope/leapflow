# Copyright (c) Alibaba, Inc. and its affiliates.
"""File operations plugin — list, read, write, search, find, edit."""

from __future__ import annotations

from typing import Any

from leapflow.plugins.protocol import ToolMetadata


class FileOpsPlugin:
    """File system tools with approval-gate support for mutating operations."""

    def __init__(self) -> None:
        self._file_read_gate: Any = None
        self._file_write_gate: Any = None

    @property
    def plugin_id(self) -> str:
        return "file_ops"

    @property
    def category(self) -> str:
        return "file"

    @property
    def tools(self) -> list[ToolMetadata]:
        from leapflow.tools.file_operations import (
            code_search,
            edit_file,
            file_find,
            file_list,
            file_read,
            file_write,
        )

        return [
            ToolMetadata(
                name="file_list",
                description=(
                    "List files and directories at a given path. Use depth=1 or depth=2 to get a "
                    "recursive tree in one call instead of listing each sub-directory separately."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path (default: current dir)",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern for flat listing (default: *; ignored when depth > 0)",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Recursion depth: 0 = flat one-level listing (default), 1-5 = recursive tree skipping VCS/deps dirs",
                        },
                    },
                },
                handler=file_list,
                x_leapflow={
                    "category": "file",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("file.list",),
                requires_platform_capabilities=("file.ops",),
            ),
            ToolMetadata(
                name="file_read",
                description=(
                    "Read text file content with adaptive context governance. For large or unfamiliar files, "
                    "prefer mode='outline' or mode='symbols' first, then use mode='raw' "
                    "with start_line/max_lines for the specific range you actually need. "
                    "For LeapFlow's own settings, use config_list / config_get / config_set \u2014 "
                    "its config files are outside the workspace and not readable here."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"},
                        "max_lines": {
                            "type": "integer",
                            "description": "Max lines to return (default: 200)",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "1-based line to start reading from (default: 1)",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": "Max characters to read before line filtering (default bounded by runtime guard)",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["raw", "outline", "symbols"],
                            "description": "raw=exact lines, outline=headings/structure, symbols=class/function signatures",
                        },
                    },
                    "required": ["path"],
                },
                handler=file_read,
                x_leapflow={
                    "category": "file",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("file.read",),
                requires_platform_capabilities=("file.ops",),
            ),
            ToolMetadata(
                name="file_write",
                description="Write content to a file (overwrite or append).",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Target file path"},
                        "content": {"type": "string", "description": "Content to write"},
                        "mode": {
                            "type": "string",
                            "enum": ["overwrite", "append"],
                            "description": "Write mode (default: overwrite)",
                        },
                    },
                    "required": ["path", "content"],
                },
                handler=file_write,
                x_leapflow={
                    "category": "write",
                    "risk_level": "mutating",
                    "schema_cost": "low",
                    "requires_approval": True,
                },
                mutates_state=True,
                execution_policy="mutating_once",
                provides_capabilities=("file.write",),
                requires_platform_capabilities=("file.ops",),
            ),
            ToolMetadata(
                name="code_search",
                description=(
                    "Search file CONTENTS by regex pattern across a directory tree (ripgrep-backed). "
                    "Requires a regex pattern. NOT for listing or browsing directory contents \u2014 use file_list for that. "
                    "Prefer this over shell_run grep: faster, skips VCS/dependency/build dirs, "
                    "and returns structured path:line:column matches. Batch related lookups "
                    "into ONE call via `patterns` (OR-combined, single pass) instead of "
                    "issuing several separate searches. Use file_read for the surrounding "
                    "context of a hit."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for (REQUIRED \u2014 this tool searches file contents, not file names)",
                        },
                        "patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Additional regex patterns OR-combined with pattern into one search pass",
                        },
                        "path": {
                            "type": "string",
                            "description": "Base directory (default: current dir)",
                        },
                        "glob": {
                            "type": "string",
                            "description": "Filter files by glob, e.g. *.py",
                        },
                        "ignore_case": {
                            "type": "boolean",
                            "description": "Case-insensitive match (default: false)",
                        },
                        "multiline": {
                            "type": "boolean",
                            "description": "Let . span newlines / match across lines (default: false)",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max matches to return (default: 200)",
                        },
                        "context_lines": {
                            "type": "integer",
                            "description": "Lines of context before/after each match (default: 0, max 10)",
                        },
                    },
                    "required": [],
                },
                handler=code_search,
                x_leapflow={
                    "category": "file",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("file.search",),
                requires_platform_capabilities=("file.ops",),
            ),
            ToolMetadata(
                name="file_find",
                description=(
                    "Find files by a recursive glob pattern under a base path (e.g. '**/test_*.py' "
                    "or '*.md'). Prefer this over shell_run find; skips VCS/dependency/build dirs."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "glob": {
                            "type": "string",
                            "description": "Glob pattern, recursive (e.g. *.py, **/conftest.py)",
                        },
                        "path": {
                            "type": "string",
                            "description": "Base directory (default: current dir)",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max files to return (default: 500)",
                        },
                    },
                    "required": ["glob"],
                },
                handler=file_find,
                x_leapflow={
                    "category": "file",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("file.find",),
                requires_platform_capabilities=("file.ops",),
            ),
            ToolMetadata(
                name="edit_file",
                description=(
                    "Apply targeted, anchored search-replace edits to an EXISTING text file "
                    "(use file_write to create/overwrite). Each edit is {original_text, new_text, "
                    "replace_all?}; original_text must match exactly and uniquely (or set replace_all) "
                    "\u2014 a non-unique or missing anchor is rejected so files are never corrupted. Set "
                    "dry_run to preview. Alternatively pass a unified 'diff' to apply its hunks as "
                    "anchored edits. Far cheaper and safer than rewriting a whole file."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to edit"},
                        "edits": {
                            "type": "array",
                            "description": "List of edits, applied in order.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "original_text": {
                                        "type": "string",
                                        "description": "Exact text to replace (unique unless replace_all)",
                                    },
                                    "new_text": {
                                        "type": "string",
                                        "description": "Replacement text",
                                    },
                                    "replace_all": {
                                        "type": "boolean",
                                        "description": "Replace every occurrence (default: false)",
                                    },
                                },
                                "required": ["original_text", "new_text"],
                            },
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "Preview without writing (default: false)",
                        },
                        "diff": {
                            "type": "string",
                            "description": "Unified diff to apply (alternative to edits; each hunk applied as an anchored edit)",
                        },
                    },
                    "required": ["path"],
                },
                handler=edit_file,
                x_leapflow={
                    "category": "write",
                    "risk_level": "mutating",
                    "schema_cost": "medium",
                    "requires_approval": True,
                },
                mutates_state=True,
                execution_policy="mutating_idempotent",
                provides_capabilities=("file.edit",),
                requires_platform_capabilities=("file.ops",),
            ),
        ]

    @property
    def dependencies(self) -> list[str]:
        return ["file_read_gate", "file_write_gate"]

    def bind_runtime(self, **deps: Any) -> None:
        if "file_read_gate" in deps:
            self._file_read_gate = deps["file_read_gate"]
        if "file_write_gate" in deps:
            self._file_write_gate = deps["file_write_gate"]


# Module-level instance for plugin discovery
plugin = FileOpsPlugin()
