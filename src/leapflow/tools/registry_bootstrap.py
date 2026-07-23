"""Bootstrap general-purpose tools into the ToolBridge.

Provides TOOL_DEFINITIONS in OpenAI function calling schema and a bootstrap
function that registers all tools into an existing ToolBridge instance.
"""

from __future__ import annotations

from typing import Any, Dict, List

from leapflow.tools.file_operations import (
    code_search,
    edit_file,
    file_find,
    file_list,
    file_read,
    file_write,
)
from leapflow.tools.scm_tools import scm_sync, git_query
from leapflow.tools.code_intel import code_intel
from leapflow.tools.shell_tools import shell_run
from leapflow.tools.system_tools import env_info, time_get
from leapflow.tools.text_tools import text_replace, text_search
from leapflow.skills.discovery import skills_list, skill_view
from leapflow.tools.hub_tool import (
    HUB_BRIDGE_TOOLS,
    HUB_TOOL_DEFINITIONS,
    HUB_TOOL_HANDLERS,
)
from leapflow.tools.gateway_tool import (
    GATEWAY_BRIDGE_TOOLS,
    GATEWAY_TOOL_DEFINITIONS,
    GATEWAY_TOOL_HANDLERS,
    set_gateway_server as set_gateway_server,
)
from leapflow.tools.name_resolver import TOOL_NAME_ALIASES, ToolRegistry


# ─────────────────────────────────────────────────────────────────────
# OpenAI function calling schema definitions (for external consumers)
# ─────────────────────────────────────────────────────────────────────

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "file_list",
            "description": "List files and directories at a given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: current dir)"},
                    "pattern": {"type": "string", "description": "Glob pattern (default: *)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": (
                "Read text file content with adaptive context governance. For large or unfamiliar files, "
                "prefer mode='outline' or mode='symbols' first, then use mode='raw' "
                "with start_line/max_lines for the specific range you actually need. "
                "Do not probe `<workspace>/.leapflow/config.json`; LeapFlow uses structured "
                "config under `~/.leapflow/config/user.yaml`, "
                "`~/.leapflow/profiles/<profile>/config/*.yaml`, and optional "
                "`<workspace>/.leapflow/config.yaml`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "max_lines": {"type": "integer", "description": "Max lines to return (default: 200)"},
                    "start_line": {"type": "integer", "description": "1-based line to start reading from (default: 1)"},
                    "max_chars": {"type": "integer", "description": "Max characters to read before line filtering (default bounded by runtime guard)"},
                    "mode": {
                        "type": "string",
                        "enum": ["raw", "outline", "symbols"],
                        "description": "raw=exact lines, outline=headings/structure, symbols=class/function signatures",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Write content to a file (overwrite or append).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target file path"},
                    "content": {"type": "string", "description": "Content to write"},
                    "mode": {"type": "string", "enum": ["overwrite", "append"], "description": "Write mode (default: overwrite)"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_search",
            "description": (
                "Search file CONTENTS by regex across a directory tree (ripgrep-backed). "
                "Prefer this over shell_run grep: faster, skips VCS/dependency/build dirs, "
                "and returns structured path:line:column matches. Use file_read for the "
                "surrounding context of a hit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Base directory (default: current dir)"},
                    "glob": {"type": "string", "description": "Filter files by glob, e.g. *.py"},
                    "ignore_case": {"type": "boolean", "description": "Case-insensitive match (default: false)"},
                    "multiline": {"type": "boolean", "description": "Let . span newlines / match across lines (default: false)"},
                    "max_results": {"type": "integer", "description": "Max matches to return (default: 200)"},
                },
                "required": ["pattern"],
            },
            "x_leapflow": {"category": "file", "risk_level": "read_only", "schema_cost": "low", "requires_approval": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_find",
            "description": (
                "Find files by a recursive glob pattern under a base path (e.g. '**/test_*.py' "
                "or '*.md'). Prefer this over shell_run find; skips VCS/dependency/build dirs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "glob": {"type": "string", "description": "Glob pattern, recursive (e.g. *.py, **/conftest.py)"},
                    "path": {"type": "string", "description": "Base directory (default: current dir)"},
                    "max_results": {"type": "integer", "description": "Max files to return (default: 500)"},
                },
                "required": ["glob"],
            },
            "x_leapflow": {"category": "file", "risk_level": "read_only", "schema_cost": "low", "requires_approval": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Apply targeted, anchored search-replace edits to an EXISTING text file "
                "(use file_write to create/overwrite). Each edit is {original_text, new_text, "
                "replace_all?}; original_text must match exactly and uniquely (or set replace_all) "
                "— a non-unique or missing anchor is rejected so files are never corrupted. Set "
                "dry_run to preview. Far cheaper and safer than rewriting a whole file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "edits": {
                        "type": "array",
                        "description": "List of edits, applied in order.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original_text": {"type": "string", "description": "Exact text to replace (unique unless replace_all)"},
                                "new_text": {"type": "string", "description": "Replacement text"},
                                "replace_all": {"type": "boolean", "description": "Replace every occurrence (default: false)"},
                            },
                            "required": ["original_text", "new_text"],
                        },
                    },
                    "dry_run": {"type": "boolean", "description": "Preview without writing (default: false)"},
                },
                "required": ["path", "edits"],
            },
            "x_leapflow": {"category": "file", "risk_level": "mutating", "schema_cost": "medium", "requires_approval": True},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_intel",
            "description": (
                "Precise document symbols (outline) for a source file: classes, functions, and "
                "methods with line ranges. Python uses an exact AST parse; other languages use a "
                "keyword-prefix scan. Prefer over file_read mode=symbols for accurate navigation "
                "before editing. Read-only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Source file to analyze"},
                    "operation": {"type": "string", "enum": ["symbols"], "description": "Analysis operation (default: symbols)"},
                },
                "required": ["path"],
            },
            "x_leapflow": {"category": "file", "risk_level": "read_only", "schema_cost": "low", "requires_approval": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_run",
            "description": "Execute a shell command with timeout protection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "cwd": {"type": "string", "description": "Working directory (optional)"},
                    "timeout": {"type": "number", "description": "Timeout in seconds (default: 30, max: 120)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scm_sync",
            "description": (
                "Run a typed git SCM action. Use this instead of shell_run for git pull/push/status. "
                "For 'pull origin main then push', set action='pull_then_push', remote='origin', "
                "pull_ref='main', and omit push_ref so LeapFlow pushes the current local branch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "pull", "push", "pull_then_push"],
                        "description": "Structured SCM action to run.",
                    },
                    "cwd": {"type": "string", "description": "Repository working directory (optional)."},
                    "remote": {"type": "string", "description": "Git remote, default origin."},
                    "pull_ref": {"type": "string", "description": "Remote ref to pull, e.g. main."},
                    "push_ref": {
                        "type": "string",
                        "description": "Ref to push. Omit or use current_branch to push the current local branch.",
                    },
                    "timeout": {"type": "number", "description": "Timeout in seconds (default/max 120)."},
                },
                "required": ["action"],
            },
            "x_leapflow": {
                "category": "scm",
                "risk_level": "high",
                "schema_cost": "high",
                "requires_approval": True,
                "effect_scope": "external",
                "idempotency_scope": "session",
                "summary": "Typed git status/pull/push with explicit current-branch push semantics.",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_query",
            "description": (
                "Read-only structured git inspection: action=diff|log|status|branch|show. "
                "Prefer over shell_run for reading repo state — output is clipped, redacted, and "
                "log/branch are parsed into structured fields. Use scm_sync for pull/push."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["diff", "log", "status", "branch", "show"], "description": "Git read action"},
                    "cwd": {"type": "string", "description": "Repository working directory (optional)"},
                    "ref": {"type": "string", "description": "A single git ref (e.g. HEAD~1, a branch/commit); ranges not allowed"},
                    "path": {"type": "string", "description": "Limit diff/log to this path (optional)"},
                    "staged": {"type": "boolean", "description": "diff: show staged changes (default: false)"},
                    "max_count": {"type": "integer", "description": "log: max entries (default 20, max 200)"},
                    "stat": {"type": "boolean", "description": "log: include --stat (default: false)"},
                },
                "required": ["action"],
            },
            "x_leapflow": {"category": "scm", "risk_level": "read_only", "schema_cost": "medium", "requires_approval": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "time_get",
            "description": "Get current date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "env_info",
            "description": "Get system environment information (OS, Python version, cwd).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_search",
            "description": "Search for a regex pattern in text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to search in"},
                    "pattern": {"type": "string", "description": "Regex pattern to match"},
                },
                "required": ["text", "pattern"],
            },
            # Explicit metadata: pure in-memory regex search over caller-supplied
            # text, no I/O or state mutation. Declared explicitly rather than
            # relying on the "general" keyword fallback, which is intentionally
            # non-core by default (fail-closed) for anything not reviewed.
            "x_leapflow": {
                "category": "general",
                "risk_level": "read_only",
                "schema_cost": "low",
                "requires_approval": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_replace",
            "description": "Replace occurrences of a substring in text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Original text"},
                    "old": {"type": "string", "description": "Substring to find"},
                    "new": {"type": "string", "description": "Replacement string"},
                    "count": {"type": "integer", "description": "Max replacements (0 = all)"},
                },
                "required": ["text", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skills_list",
            "description": "List available learned skills. Use when user asks about capabilities or you need a specific skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Optional keyword filter"},
                    "category": {"type": "string", "description": "Filter by category (e.g. file-mgmt, apple)"},
                    "source": {"type": "string", "description": "Filter by source: learned, manual, or hub"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_view",
            "description": "View the full content of a specific skill document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Skill name to view"},
                },
                "required": ["name"],
            },
        },
    },
    # ── Memory tools (agent can actively search/add memory) ──
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search agent memory for relevant past experiences, observations, and facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords"},
                    "limit": {"type": "integer", "description": "Max results (default: 10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_add",
            "description": "Store a new observation or insight in memory for future reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "What to remember"},
                    "kind": {"type": "string", "enum": ["observation", "insight", "fact"], "description": "Memory type (default: observation)"},
                },
                "required": ["content"],
            },
        },
    },
    # ── Research ledger (durable long-task state; mechanism 5) ──
    {
        "type": "function",
        "function": {
            "name": "research_note",
            "description": (
                "Record a compact, structured note about the current task's state so it "
                "survives context compression on long / multi-step tasks. Use for durable "
                "findings, open questions still to resolve, decisions / excluded paths, and "
                "the immediate next step. One concise sentence per note."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["finding", "open_question", "resolved", "decision", "next_step"],
                        "description": "finding | open_question | resolved (closes a matching open question) | decision | next_step",
                    },
                    "text": {"type": "string", "description": "One concise sentence."},
                },
                "required": ["kind", "text"],
            },
        },
        "x_leapflow": {"category": "memory", "risk_level": "read_only", "requires_approval": False, "schema_cost": "medium"},
    },
    # ── Event-driven re-entry (S2) ──
    {
        "type": "function",
        "function": {
            "name": "schedule_reentry",
            "description": (
                "Register a re-entry so this task can resume later from its current "
                "orientation (findings / open questions / next step). Use when work must "
                "pause and continue after a delay (kind=time) or when a matching platform "
                "event arrives (kind=event), instead of finishing now. The research-ledger "
                "state is carried over automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["time", "event"],
                        "description": "time = resume after delay_seconds; event = resume when a matching platform event arrives",
                    },
                    "reason": {"type": "string", "description": "One concise sentence: what to continue and why (carried into the resumed turn)."},
                    "delay_seconds": {"type": "number", "description": "kind=time: seconds from now to resume."},
                    "event_match": {"type": "object", "description": "kind=event: match filter, e.g. platform / chat / keyword."},
                    "max_reentries": {"type": "integer", "description": "Max times this may resume (default 1)."},
                    "deadline_seconds": {"type": "number", "description": "Optional: abandon the re-entry after this many seconds."},
                },
                "required": ["kind", "reason"],
            },
        },
        "x_leapflow": {"category": "memory", "risk_level": "read_only", "requires_approval": False, "schema_cost": "medium"},
    },
    # ── Capability discovery (Tier 1 structural gate) ──
    # Lets the model expand a heavier tool category (hub/gateway/delegate) into
    # full native schemas on demand, instead of the runtime guessing from text
    # which categories are "probably" relevant to the current request.
    {
        "type": "function",
        "function": {
            "name": "capability_expand",
            "description": (
                "Fetch the full callable schema for every tool in a capability category "
                "(e.g. 'hub', 'gateway', 'delegate', 'file', 'memory', 'skill'). The compact "
                "tool index always lists every registered tool by name and a one-line summary, "
                "but only a static low-risk subset is directly callable each turn. If you need a "
                "tool from the index that is not yet callable, call capability_expand with its "
                "category first; the matching tools become callable in this turn. Never invent a "
                "tool name — expand the category instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Capability category name, e.g. hub, gateway, delegate"},
                },
                "required": ["category"],
            },
            "x_leapflow": {
                "category": "system",
                "risk_level": "read_only",
                "schema_cost": "low",
                "requires_approval": False,
            },
        },
    },
    # ── Subagent delegation ──
    {
        "type": "function",
        "function": {
            "name": "delegate_task",
            "description": (
                "Delegate a complex sub-task to an isolated subagent. "
                "The subagent gets a fresh context and restricted tool access. "
                "Use when a task is self-contained and can be solved independently."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "Clear description of the task to delegate"},
                    "context": {"type": "string", "description": "Relevant context for the subagent (optional)"},
                },
                "required": ["goal"],
            },
        },
    },
] + HUB_TOOL_DEFINITIONS + GATEWAY_TOOL_DEFINITIONS


# ─────────────────────────────────────────────────────────────────────
# ToolBridge registration table
# Conforms to bridge.register(name, description, parameters, handler)
# where parameters is Dict[str, str] (param_name -> type description)
# ─────────────────────────────────────────────────────────────────────

_BRIDGE_TOOLS = [
    {
        "name": "gp_file_list",
        "description": "List files and directories at a given path.",
        "parameters": {
            "path": "string (optional) — directory path to list (default: .)",
            "pattern": "string (optional) — glob pattern to filter (default: *)",
        },
        "handler": file_list,
    },
    {
        "name": "gp_file_read",
        "description": (
            "Read text file content with adaptive context governance. Use mode='outline'/'symbols' for large "
            "or unfamiliar files before reading raw ranges, to reduce context usage by default."
        ),
        "parameters": {
            "path": "string (required) — file path to read",
            "max_lines": "integer (optional) — max lines to return (default: 200)",
            "start_line": "integer (optional) — 1-based starting line (default: 1)",
            "max_chars": "integer (optional) — max characters to read before line filtering",
            "mode": "string (optional) — raw|outline|symbols (default: raw)",
        },
        "handler": file_read,
    },
    {
        "name": "gp_file_write",
        "description": "Write content to a file (overwrite or append).",
        "parameters": {
            "path": "string (required) — target file path",
            "content": "string (required) — content to write",
            "mode": "string (optional) — 'overwrite' (default) or 'append'",
        },
        "handler": file_write,
        "mutates_state": True,
    },
    {
        "name": "gp_code_search",
        "description": "Search file contents by regex across a directory tree (ripgrep-backed, structured results).",
        "parameters": {
            "pattern": "string (required) — regex pattern",
            "path": "string (optional) — base directory (default: .)",
            "glob": "string (optional) — filter files by glob, e.g. *.py",
            "ignore_case": "boolean (optional) — case-insensitive (default: false)",
            "multiline": "boolean (optional) — match across lines (default: false)",
            "max_results": "integer (optional) — max matches (default: 200)",
        },
        "handler": code_search,
    },
    {
        "name": "gp_file_find",
        "description": "Find files by recursive glob under a base path.",
        "parameters": {
            "glob": "string (required) — recursive glob, e.g. **/test_*.py",
            "path": "string (optional) — base directory (default: .)",
            "max_results": "integer (optional) — max files (default: 500)",
        },
        "handler": file_find,
    },
    {
        "name": "gp_edit_file",
        "description": "Apply anchored search-replace edits to an existing text file (dry_run supported).",
        "parameters": {
            "path": "string (required) — file path to edit",
            "edits": "array (required) — list of {original_text, new_text, replace_all?}",
            "dry_run": "boolean (optional) — preview without writing (default: false)",
        },
        "handler": edit_file,
        "mutates_state": True,
    },
    {
        "name": "gp_code_intel",
        "description": "Precise document symbols (classes/functions/methods with line ranges); Python via AST.",
        "parameters": {
            "path": "string (required) — source file to analyze",
            "operation": "string (optional) — 'symbols' (default)",
        },
        "handler": code_intel,
    },
    {
        "name": "gp_shell_run",
        "description": "Execute a shell command with timeout protection.",
        "parameters": {
            "command": "string (required) — shell command to execute",
            "cwd": "string (optional) — working directory",
            "timeout": "number (optional) — timeout in seconds (default: 30, max: 120)",
        },
        "handler": shell_run,
        "mutates_state": True,
    },
    {
        "name": "gp_scm_sync",
        "description": "Run typed git status/pull/push actions with explicit current-branch push semantics.",
        "parameters": {
            "action": "string (required) — status|pull|push|pull_then_push",
            "cwd": "string (optional) — repository working directory",
            "remote": "string (optional) — git remote, default origin",
            "pull_ref": "string (optional) — ref to pull, e.g. main",
            "push_ref": "string (optional) — ref to push; default current_branch",
            "timeout": "number (optional) — timeout in seconds (default/max 120)",
        },
        "handler": scm_sync,
        "mutates_state": True,
    },
    {
        "name": "gp_git_query",
        "description": "Read-only structured git inspection (diff/log/status/branch/show).",
        "parameters": {
            "action": "string (required) — diff|log|status|branch|show",
            "cwd": "string (optional) — repository working directory",
            "ref": "string (optional) — single git ref (no ranges)",
            "path": "string (optional) — limit diff/log to this path",
            "staged": "boolean (optional) — diff staged changes",
            "max_count": "integer (optional) — log max entries (default 20)",
            "stat": "boolean (optional) — log include --stat",
        },
        "handler": git_query,
    },
    {
        "name": "gp_time_get",
        "description": "Get current date and time.",
        "parameters": {},
        "handler": time_get,
    },
    {
        "name": "gp_env_info",
        "description": "Get system environment information (OS, Python version, cwd).",
        "parameters": {},
        "handler": env_info,
    },
    {
        "name": "gp_text_search",
        "description": "Search for a regex pattern in text.",
        "parameters": {
            "text": "string (required) — text to search in",
            "pattern": "string (required) — regex pattern to match",
        },
        "handler": text_search,
    },
    {
        "name": "gp_text_replace",
        "description": "Replace occurrences of a substring in text.",
        "parameters": {
            "text": "string (required) — original text",
            "old": "string (required) — substring to find",
            "new": "string (required) — replacement string",
            "count": "integer (optional) — max replacements (0 = all, default: 0)",
        },
        "handler": text_replace,
    },
    {
        "name": "gp_skills_list",
        "description": "List available learned skills. Use when user asks about capabilities or you need a specific skill.",
        "parameters": {
            "query": "string (optional) — keyword filter for skill names/descriptions",
            "category": "string (optional) — filter by category",
            "source": "string (optional) — filter by source (learned, manual, hub)",
        },
        "handler": skills_list,
    },
    {
        "name": "gp_skill_view",
        "description": "View the full content of a specific skill document.",
        "parameters": {
            "name": "string (required) — skill name to view",
        },
        "handler": skill_view,
    },
] + HUB_BRIDGE_TOOLS + GATEWAY_BRIDGE_TOOLS


# ─────────────────────────────────────────────────────────────────────
# Direct handler dispatch map (name → async handler function)
# Used by AgentEngine._unified_tool_loop for chat-mode tool execution.
# Maps BOTH the gp_-prefixed bridge names AND the unprefixed names from
# TOOL_DEFINITIONS so that native tool_calls (which use TOOL_DEFINITIONS
# names) resolve correctly.
# ─────────────────────────────────────────────────────────────────────

TOOL_HANDLERS: Dict[str, Any] = {t["name"]: t["handler"] for t in _BRIDGE_TOOLS}
# Add hub tool handlers
TOOL_HANDLERS.update(HUB_TOOL_HANDLERS)
# Add gateway tool handlers
TOOL_HANDLERS.update(GATEWAY_TOOL_HANDLERS)
# Add unprefixed aliases matching TOOL_DEFINITIONS names for native tool_calls
for _td in TOOL_DEFINITIONS:
    _func_name = _td.get("function", {}).get("name", "")
    if _func_name and _func_name not in TOOL_HANDLERS:
        _prefixed = f"gp_{_func_name}"
        if _prefixed in TOOL_HANDLERS:
            TOOL_HANDLERS[_func_name] = TOOL_HANDLERS[_prefixed]

# ─────────────────────────────────────────────────────────────────────
# Memory tool late-binding: handlers delegate to MemoryManager when
# installed, fail gracefully when not. Avoids import-time dependency.
# ─────────────────────────────────────────────────────────────────────

_memory_manager_ref: Any = None


def set_memory_manager(manager: Any) -> None:
    """Install MemoryManager reference for memory tool dispatch."""
    global _memory_manager_ref
    _memory_manager_ref = manager


async def _memory_search_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    if _memory_manager_ref is None:
        return {"ok": False, "error": "Memory system not initialized"}
    try:
        result = await _memory_manager_ref.handle_tool_call("memory_search", params)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _memory_add_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    if _memory_manager_ref is None:
        return {"ok": False, "error": "Memory system not initialized"}
    content = params.get("content", "")
    if content:
        try:
            from leapflow.security.threat_patterns import scan_for_threats, ThreatScope
            threats = scan_for_threats(content, scope=ThreatScope.STRICT, max_results=3)
            if any(t.severity >= 0.8 for t in threats):
                import logging as _log
                _log.getLogger(__name__).warning("memory_add: threat in content: %s",
                                                  [t.pattern_name for t in threats])
        except ImportError:
            pass
    try:
        result = await _memory_manager_ref.handle_tool_call("memory_add", params)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


TOOL_HANDLERS["memory_search"] = _memory_search_handler
TOOL_HANDLERS["memory_add"] = _memory_add_handler
TOOL_HANDLERS["gp_memory_search"] = _memory_search_handler
TOOL_HANDLERS["gp_memory_add"] = _memory_add_handler


# ────────────────────────────────────────────────────
# Research-ledger tool late-binding: delegates to the engine's per-task
# ResearchLedger when installed; fails gracefully when not.
# ────────────────────────────────────────────────────

_research_ledger_ref: Any = None


def set_research_ledger(ledger: Any) -> None:
    """Install the active ResearchLedger for research_note dispatch."""
    global _research_ledger_ref
    _research_ledger_ref = ledger


async def _research_note_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    if _research_ledger_ref is None:
        return {"ok": False, "error": "Research ledger not initialized"}
    ok = _research_ledger_ref.note(params.get("kind", ""), params.get("text", ""))
    if not ok:
        return {
            "ok": False,
            "error": "invalid note: kind must be one of finding|open_question|resolved|decision|next_step and text must be non-empty",
        }
    return {"ok": True, "open_questions": _research_ledger_ref.open_question_count}


TOOL_HANDLERS["research_note"] = _research_note_handler
TOOL_HANDLERS["gp_research_note"] = _research_note_handler


# ────────────────────────────────────────────────────
# Re-entry scheduling (S2) late-binding: delegates to the engine's
# _schedule_reentry when installed; gated by config + persisted engine-side.
# ────────────────────────────────────────────────────

_reentry_scheduler_ref: Any = None


def set_reentry_scheduler(scheduler: Any) -> None:
    """Install the engine's re-entry scheduler callable for schedule_reentry."""
    global _reentry_scheduler_ref
    _reentry_scheduler_ref = scheduler


async def _schedule_reentry_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    if _reentry_scheduler_ref is None:
        return {"ok": False, "error": "Re-entry scheduling not initialized"}
    try:
        result = _reentry_scheduler_ref(
            kind=str(params.get("kind", "time")),
            reason=str(params.get("reason", "")),
            delay_seconds=params.get("delay_seconds", 0.0),
            event_match=params.get("event_match") or {},
            max_reentries=params.get("max_reentries", 1),
            deadline_seconds=params.get("deadline_seconds", 0.0),
        )
        return result if isinstance(result, dict) else {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


TOOL_HANDLERS["schedule_reentry"] = _schedule_reentry_handler
TOOL_HANDLERS["gp_schedule_reentry"] = _schedule_reentry_handler


# ─────────────────────────────────────────────────────────────────────
# Subagent delegation (late-binding like memory tools)
# ─────────────────────────────────────────────────────────────────────

_subagent_manager_ref: Any = None


def set_subagent_manager(manager: Any) -> None:
    """Install SubagentManager reference for delegate_task dispatch."""
    global _subagent_manager_ref
    _subagent_manager_ref = manager


async def _delegate_task_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    if _subagent_manager_ref is None:
        return {"ok": False, "error": "Subagent system not configured"}
    try:
        from leapflow.engine.subagent import SubagentConfig, current_subagent_depth
        config = SubagentConfig(
            goal=params.get("goal", ""),
            context=params.get("context", ""),
            depth=current_subagent_depth() + 1,
        )
        result = await _subagent_manager_ref.delegate(config)
        return {"ok": result.status == "completed", "summary": result.summary, "status": result.status}
    except Exception as e:
        return {"ok": False, "error": str(e)}


TOOL_HANDLERS["delegate_task"] = _delegate_task_handler
TOOL_HANDLERS["gp_delegate_task"] = _delegate_task_handler


# ────────────────────────────────────────────────────────────────
# Capability discovery (Tier 1 structural gate): expand a category's tools
# into full native schemas on request, so the caller (engine.py) can merge
# them into the current turn's tools_kwarg instead of leaving the model to
# guess a tool name that was never disclosed.
# ────────────────────────────────────────────────────────────────

async def _capability_expand_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    from leapflow.engine.context_disclosure import build_capability_manifests

    category = str(params.get("category") or "").strip().lower()
    if not category:
        return {"ok": False, "error": "category is required"}
    manifests = build_capability_manifests(TOOL_DEFINITIONS)
    matched_names = {m.name for m in manifests if m.category == category}
    if not matched_names:
        available = sorted({m.category for m in manifests if m.category})
        return {
            "ok": False,
            "error": f"Unknown capability category: {category}",
            "available_categories": available,
        }
    expanded_tools = [
        td for td in TOOL_DEFINITIONS
        if td.get("function", {}).get("name") in matched_names
    ]
    return {"ok": True, "category": category, "expanded_tools": expanded_tools}


def _patch_capability_expand_categories() -> None:
    """Inject the real, current non-core category list into capability_expand's
    own description, computed from the live tool registry instead of a static
    hardcoded example list that would silently drift out of sync.
    """
    from leapflow.engine.context_disclosure import build_capability_manifests

    manifests = build_capability_manifests(TOOL_DEFINITIONS)
    non_core_categories = sorted({m.category for m in manifests if m.category and not m.is_core})
    for td in TOOL_DEFINITIONS:
        func = td.get("function", {})
        if func.get("name") != "capability_expand":
            continue
        categories_text = ", ".join(non_core_categories) or "none"
        func["description"] = (
            "Fetch the full callable schema for every tool in a capability category. "
            f"Current non-core categories that require expansion: {categories_text}. "
            "The compact tool index always lists every registered tool by name and a "
            "one-line summary tagged with its exact capability_expand category, but "
            "only a static low-risk subset is directly callable each turn. If you need "
            "a tool from the index that is not yet callable, call capability_expand "
            "with the exact category shown next to it; the matching tools become "
            "callable in this turn. Never invent a tool name — expand the category instead."
        )
        break


_patch_capability_expand_categories()


TOOL_HANDLERS["capability_expand"] = _capability_expand_handler
TOOL_HANDLERS["gp_capability_expand"] = _capability_expand_handler


TOOL_REGISTRY = ToolRegistry.from_definitions(
    TOOL_DEFINITIONS,
    TOOL_HANDLERS,
    bridge_tools=_BRIDGE_TOOLS,
    aliases=TOOL_NAME_ALIASES,
)


# ─────────────────────────────────────────────────────────────────────
# File access approval gates (Protocol-based, injectable)
# ─────────────────────────────────────────────────────────────────────

_file_read_gate: Any = None
_file_write_gate: Any = None


def set_file_read_gate(gate: Any) -> None:
    """Install a file-read approval gate."""
    global _file_read_gate
    _file_read_gate = gate


def get_file_read_gate() -> Any:
    return _file_read_gate


def set_file_write_gate(gate: Any) -> None:
    """Install a file-write approval gate."""
    global _file_write_gate
    _file_write_gate = gate


def get_file_write_gate() -> Any:
    return _file_write_gate


def bootstrap_tools(bridge: Any) -> int:
    """Register all general-purpose tools into a ToolBridge instance.

    Tools are prefixed with 'gp_' to avoid collision with built-in ToolBridge
    tools (file_list, shell) that delegate to ExecutionPort.

    Returns:
        Number of tools successfully registered.
    """
    registered = 0
    for tool in _BRIDGE_TOOLS:
        try:
            bridge.register(
                tool["name"],
                tool["description"],
                tool["parameters"],
                tool["handler"],
                mutates_state=tool.get("mutates_state", False),
            )
            registered += 1
        except Exception:
            # Skip tools that fail registration (e.g., incompatible bridge version)
            pass
    return registered
