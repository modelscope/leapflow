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
                "Do not probe `<workspace>/.leapflow/config.json`; LeapFlow uses `~/.leapflow/.env` "
                "and optional existing `./.env` project overrides."
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
        from leapflow.engine.subagent import SubagentConfig
        config = SubagentConfig(
            goal=params.get("goal", ""),
            context=params.get("context", ""),
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
