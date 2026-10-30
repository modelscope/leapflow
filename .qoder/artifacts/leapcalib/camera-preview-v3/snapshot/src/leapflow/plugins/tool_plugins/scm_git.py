# Copyright (c) Alibaba, Inc. and its affiliates.
"""SCM/Git plugin — structured git operations (sync, query, write)."""

from __future__ import annotations

from leapflow.plugins.protocol import ToolMetadata
from leapflow.tools.scm_tools import git_query, git_write, scm_sync


class ScmGitPlugin:
    """Typed git operations: sync (pull/push), read-only query, and mutating writes."""

    @property
    def plugin_id(self) -> str:
        return "scm_git"

    @property
    def category(self) -> str:
        return "scm"

    @property
    def tools(self) -> list[ToolMetadata]:
        return [
            ToolMetadata(
                name="scm_sync",
                description=(
                    "Run a typed git SCM action. Use this instead of shell_run for git pull/push/status. "
                    "For 'pull origin main then push', set action='pull_then_push', remote='origin', "
                    "pull_ref='main', and omit push_ref so LeapFlow pushes the current local branch."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["status", "pull", "push", "pull_then_push"],
                            "description": "Structured SCM action to run.",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Repository working directory (optional).",
                        },
                        "remote": {"type": "string", "description": "Git remote, default origin."},
                        "pull_ref": {
                            "type": "string",
                            "description": "Remote ref to pull, e.g. main.",
                        },
                        "push_ref": {
                            "type": "string",
                            "description": "Ref to push. Omit or use current_branch to push the current local branch.",
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Timeout in seconds (default/max 120).",
                        },
                    },
                    "required": ["action"],
                },
                handler=scm_sync,
                x_leapflow={
                    "category": "scm",
                    "risk_level": "high",
                    "schema_cost": "high",
                    "requires_approval": True,
                    "effect_scope": "external",
                    "idempotency_scope": "session",
                    "summary": "Typed git status/pull/push with explicit current-branch push semantics.",
                },
                mutates_state=True,
                provides_capabilities=("git.sync",),
                requires_platform_capabilities=("shell.exec",),
            ),
            ToolMetadata(
                name="git_query",
                description=(
                    "Read-only structured git inspection: action=diff|log|status|branch|show. "
                    "Prefer over shell_run for reading repo state — output is clipped, redacted, and "
                    "log/branch are parsed into structured fields. Use scm_sync for pull/push."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["diff", "log", "status", "branch", "show"],
                            "description": "Git read action",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Repository working directory (optional)",
                        },
                        "ref": {
                            "type": "string",
                            "description": "A single git ref (e.g. HEAD~1, a branch/commit); ranges not allowed",
                        },
                        "path": {
                            "type": "string",
                            "description": "Limit diff/log to this path (optional)",
                        },
                        "staged": {
                            "type": "boolean",
                            "description": "diff: show staged changes (default: false)",
                        },
                        "max_count": {
                            "type": "integer",
                            "description": "log: max entries (default 20, max 200)",
                        },
                        "stat": {
                            "type": "boolean",
                            "description": "log: include --stat (default: false)",
                        },
                    },
                    "required": ["action"],
                },
                handler=git_query,
                x_leapflow={
                    "category": "scm",
                    "risk_level": "read_only",
                    "schema_cost": "medium",
                    "requires_approval": False,
                },
                provides_capabilities=("git.query",),
                requires_platform_capabilities=("shell.exec",),
            ),
            ToolMetadata(
                name="git_write",
                description=(
                    "Mutating git actions: action=commit (message, stage_all), branch (create+switch), "
                    "checkout (switch; create=true for -b). Approval-gated. Use scm_sync for pull/push "
                    "and git_query for reads."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["commit", "branch", "checkout"],
                            "description": "Git write action",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Repository working directory (optional)",
                        },
                        "message": {
                            "type": "string",
                            "description": "commit: commit message (required for commit)",
                        },
                        "stage_all": {
                            "type": "boolean",
                            "description": "commit: stage all changes first (default: true)",
                        },
                        "name": {"type": "string", "description": "branch: new branch name"},
                        "ref": {
                            "type": "string",
                            "description": "checkout: ref/branch to switch to",
                        },
                        "create": {
                            "type": "boolean",
                            "description": "checkout: create the branch (-b) (default: false)",
                        },
                    },
                    "required": ["action"],
                },
                handler=git_write,
                x_leapflow={
                    "category": "scm",
                    "risk_level": "high",
                    "schema_cost": "medium",
                    "requires_approval": True,
                    "idempotency_scope": "session",
                },
                mutates_state=True,
                provides_capabilities=("git.write",),
                requires_platform_capabilities=("shell.exec",),
            ),
        ]

    @property
    def dependencies(self) -> list[str]:
        return []

    def bind_runtime(self, **deps: object) -> None:
        pass


# Module-level instance for plugin discovery
plugin = ScmGitPlugin()
