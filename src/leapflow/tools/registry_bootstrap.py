"""Bootstrap general-purpose tools into the ToolBridge.

Provides TOOL_DEFINITIONS in OpenAI function calling schema and a bootstrap
function that registers all tools into an existing ToolBridge instance.
"""

from __future__ import annotations

from typing import Any, Dict, List

from leapflow.tools.file_operations import file_list, file_read, file_write
from leapflow.tools.shell_tools import shell_run
from leapflow.tools.system_tools import env_info, time_get
from leapflow.tools.text_tools import text_replace, text_search
from leapflow.skills.discovery import skills_list, skill_view
from leapflow.tools.hub_tool import (
    HUB_BRIDGE_TOOLS,
    HUB_TOOL_DEFINITIONS,
    HUB_TOOL_HANDLERS,
)


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
            "description": "Read the content of a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "max_lines": {"type": "integer", "description": "Max lines to return (default: 200)"},
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
] + HUB_TOOL_DEFINITIONS


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
        "description": "Read the content of a text file.",
        "parameters": {
            "path": "string (required) — file path to read",
            "max_lines": "integer (optional) — max lines to return (default: 200)",
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
] + HUB_BRIDGE_TOOLS


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
# Add unprefixed aliases matching TOOL_DEFINITIONS names for native tool_calls
for _td in TOOL_DEFINITIONS:
    _func_name = _td.get("function", {}).get("name", "")
    if _func_name and _func_name not in TOOL_HANDLERS:
        # Find handler by matching the gp_-prefixed version
        _prefixed = f"gp_{_func_name}"
        if _prefixed in TOOL_HANDLERS:
            TOOL_HANDLERS[_func_name] = TOOL_HANDLERS[_prefixed]


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
