"""Slash-command registry and dispatch.

Single source of truth for all REPL commands.  The registry drives:
- Tab completion (via ``LeapApp``)
- ``/help`` categorized display
- Dispatch in the interactive loop
- Command alias resolution

Adding a new command = appending one ``CommandDef`` to ``COMMAND_REGISTRY``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple


class CommandEffect(str, Enum):
    """User-visible impact level for a slash command."""

    READ_ONLY = "read_only"
    SESSION = "session"
    HOST_CONTROL = "host_control"
    EXTERNAL = "external"
    DESTRUCTIVE = "destructive"


class CommandExecution(str, Enum):
    """Execution shape used by the TUI to explain command behavior."""

    INSTANT = "instant"
    SHORT_OPERATION = "short_operation"
    STREAMING = "streaming"
    LONG_RUNNING = "long_running"
    BACKGROUND = "background"


CommandRuntime = Literal["in_process", "daemon"]


@dataclass(frozen=True)
class CommandDef:
    """Definition of a single slash command."""

    name: str
    description: str
    category: str
    aliases: Tuple[str, ...] = ()
    args_hint: str = ""
    client_local: bool = False
    requires_host: bool = False
    requires_llm: bool = False
    effect: CommandEffect = CommandEffect.READ_ONLY
    execution: CommandExecution = CommandExecution.INSTANT

    def supports_runtime(self, runtime: CommandRuntime) -> bool:
        """All commands support all runtimes.

        Client-local commands are handled directly by the TUI without RPC.
        Engine-routed commands are dispatched through daemon RPC in daemon mode.
        """
        return True


# ── Registry (single source of truth) ────────────────────────────────

COMMAND_REGISTRY: Tuple[CommandDef, ...] = (
    # Session (client_local: handled directly by the TUI without daemon RPC)
    CommandDef("help", "Show available commands", "Session", aliases=("?", "帮助"), client_local=True),
    CommandDef("clear", "Clear screen and reset display", "Session", client_local=True),
    CommandDef("exit", "Quit LeapFlow", "Session", aliases=("quit", "q", "退出"), client_local=True),
    CommandDef("status", "Show session info, model, context, and platform", "Session"),
    CommandDef(
        "host",
        "Start, stop, or inspect CuaDriver OS control",
        "Session",
        args_hint="[status|start|stop|restart]",
        effect=CommandEffect.HOST_CONTROL,
        execution=CommandExecution.SHORT_OPERATION,
    ),

    # Task Control (client_local: queue state is TUI-side)
    CommandDef("cancel", "Cancel the currently running task", "Task Control", aliases=("abort",), client_local=True, effect=CommandEffect.SESSION, execution=CommandExecution.SHORT_OPERATION),
    CommandDef("skip", "Skip the current running task and continue the queue", "Task Control", client_local=True, effect=CommandEffect.SESSION, execution=CommandExecution.SHORT_OPERATION),
    CommandDef("pause", "Pause starting queued tasks", "Task Control", client_local=True, effect=CommandEffect.SESSION, execution=CommandExecution.INSTANT),
    CommandDef("resume", "Resume queued task execution", "Task Control", client_local=True, effect=CommandEffect.SESSION, execution=CommandExecution.INSTANT),
    CommandDef("queue", "Show or clear queued tasks", "Task Control", args_hint="[clear]", client_local=True, effect=CommandEffect.READ_ONLY),
    CommandDef("drop", "Remove a queued task by id", "Task Control", args_hint="<id>", client_local=True, effect=CommandEffect.SESSION),

    # Chat
    CommandDef("model", "Show or switch active model", "Chat", args_hint="[model_name]", requires_llm=True),
    CommandDef("config", "View or update runtime configuration", "Chat", args_hint="[show|list|keys|sources|get|set|unset|llm|secret] ...", effect=CommandEffect.SESSION, execution=CommandExecution.SHORT_OPERATION),
    CommandDef("usage", "Show token usage for current session", "Chat", requires_llm=True),
    CommandDef("orient", "Show the agent's unified orientation (layered) and pending re-entries", "Chat", effect=CommandEffect.READ_ONLY),

    # Teaching
    CommandDef("teach start", "Start teaching mode", "Teaching", aliases=("teach",), args_hint="[goal]"),
    CommandDef("teach stop", "Stop and distill skill", "Teaching"),
    CommandDef("teach status", "Show distillation progress", "Teaching"),
    CommandDef("teach pause", "Pause recording", "Teaching"),
    CommandDef("teach resume", "Resume recording", "Teaching"),
    CommandDef("teach discard", "Discard current recording", "Teaching"),
    CommandDef("teach skip", "Mark last n steps as noise", "Teaching", args_hint="[n]"),
    CommandDef("annotate", "Add annotation during teaching", "Teaching", args_hint="<text>"),

    # Skills & Tools
    CommandDef("skill", "List all skills", "Skills & Tools", aliases=("skill list",)),
    CommandDef("skill show", "Show skill details", "Skills & Tools", args_hint="<name>"),
    CommandDef("skill disable", "Disable a skill", "Skills & Tools", args_hint="<name>"),
    CommandDef("skill delete", "Delete a skill", "Skills & Tools", args_hint="<name>"),
    CommandDef("tool", "List available tools", "Skills & Tools"),
    CommandDef("run", "Execute a skill by trigger", "Skills & Tools", args_hint="<trigger>", requires_llm=True, execution=CommandExecution.STREAMING),

    # Hub
    CommandDef("hub push", "Push skill to hub", "Hub", args_hint="<skill>"),
    CommandDef("hub pull", "Pull skill from hub", "Hub", args_hint="<skill>"),
    CommandDef("hub search", "Search hub for skills", "Hub", args_hint="<query>"),

    # Gateway
    CommandDef("gateway", "Show connected platforms and gateway status", "Gateway", effect=CommandEffect.EXTERNAL),

    # App Connector
    CommandDef("app", "List supported external apps or open an app setup guide", "App Connector", args_hint="[platform]"),
    CommandDef("app list", "List supported external apps", "App Connector"),
    CommandDef("app status", "Show App Connector status", "App Connector", args_hint="[platform]"),
    CommandDef("app connect", "Connect a supported external app", "App Connector", args_hint="<platform> [--option value]", effect=CommandEffect.EXTERNAL, execution=CommandExecution.SHORT_OPERATION),
    CommandDef("app disconnect", "Disconnect an external app but keep configuration", "App Connector", args_hint="<platform>", effect=CommandEffect.EXTERNAL, execution=CommandExecution.SHORT_OPERATION),
    CommandDef("app remove", "Remove an app configuration", "App Connector", args_hint="<platform>", effect=CommandEffect.DESTRUCTIVE, execution=CommandExecution.SHORT_OPERATION),
    CommandDef("app events", "Inspect or control an app event source", "App Connector", args_hint="[status|start|stop] <platform>", effect=CommandEffect.EXTERNAL, execution=CommandExecution.SHORT_OPERATION),
    CommandDef("app actions", "List App Connector action domains", "App Connector", args_hint="<platform>"),

    # Scheduler
    CommandDef("arm", "Schedule a skill for timed execution", "Scheduler", args_hint="<skill> <cron>"),
    CommandDef("task", "List scheduled tasks", "Scheduler"),

    # Board & Monitors (LeapBoard) — one analysis target (current session),
    # rendered through a selectable template lens.
    CommandDef("board", "Analyze the current session; optionally pick a template lens", "Board", args_hint="[<template> | templates|refresh|pause|resume|stop|status]", effect=CommandEffect.SESSION, execution=CommandExecution.SHORT_OPERATION),
    CommandDef("board templates", "List, add, remove, or show board templates", "Board", args_hint="[list|add <path.yaml> [--name id] [--force]|remove <id>|show <id>]", effect=CommandEffect.SESSION, execution=CommandExecution.SHORT_OPERATION),
    CommandDef("board refresh", "Re-analyze the current session (or a watch by id) now", "Board", args_hint="[<id>]", effect=CommandEffect.SESSION, execution=CommandExecution.SHORT_OPERATION),
    CommandDef("board pause", "Pause session analysis (or a watch by id)", "Board", args_hint="[<id>]", effect=CommandEffect.SESSION),
    CommandDef("board resume", "Resume session analysis (or a watch by id)", "Board", args_hint="[<id>]", effect=CommandEffect.SESSION),
    CommandDef("board stop", "Stop the current session (or a watch by id)", "Board", args_hint="[<id>]", effect=CommandEffect.SESSION),
    CommandDef("board status", "Show watch state, recent findings, and templates", "Board"),
)

# ── Derived structures ───────────────────────────────────────────────

def _build_lookup() -> Dict[str, CommandDef]:
    lookup: Dict[str, CommandDef] = {}
    for cmd in COMMAND_REGISTRY:
        lookup[cmd.name] = cmd
        for alias in cmd.aliases:
            lookup[alias] = cmd
    return lookup


_COMMAND_LOOKUP = _build_lookup()


def resolve_command(text: str) -> Optional[CommandDef]:
    """Resolve user input to a CommandDef, handling aliases and multi-word commands.

    Tries longest match first: ``skill show foo`` matches ``skill show``
    before ``skill``.
    """
    words = text.strip().lower().split()
    for length in range(min(len(words), 3), 0, -1):
        key = " ".join(words[:length])
        if key in _COMMAND_LOOKUP:
            return _COMMAND_LOOKUP[key]
    return None


def commands_by_category(
    runtime: CommandRuntime | None = None,
    *,
    include_unsupported: bool = True,
) -> Dict[str, List[CommandDef]]:
    """Group commands by category for display."""
    groups: Dict[str, List[CommandDef]] = {}
    for cmd in COMMAND_REGISTRY:
        if runtime is not None and not include_unsupported and not cmd.supports_runtime(runtime):
            continue
        groups.setdefault(cmd.category, []).append(cmd)
    return groups


def completion_entries() -> List[Tuple[str, str]]:
    """Build the completion list for LeapApp's TextArea completer."""
    entries: List[Tuple[str, str]] = []
    for cmd in COMMAND_REGISTRY:
        entries.append((cmd.name, cmd.description))
    return entries
