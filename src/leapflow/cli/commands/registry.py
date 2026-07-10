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
    supports_in_process: bool = True
    supports_daemon: bool = False
    requires_host: bool = False
    requires_llm: bool = False
    effect: CommandEffect = CommandEffect.READ_ONLY
    execution: CommandExecution = CommandExecution.INSTANT

    def supports_runtime(self, runtime: CommandRuntime) -> bool:
        """Return whether this command can execute in the requested runtime."""
        if runtime == "daemon":
            return self.supports_daemon
        return self.supports_in_process


# ── Registry (single source of truth) ────────────────────────────────

COMMAND_REGISTRY: Tuple[CommandDef, ...] = (
    # Session
    CommandDef("help", "Show available commands", "Session", aliases=("?", "帮助"), supports_daemon=True),
    CommandDef("clear", "Clear screen and reset display", "Session", supports_daemon=True),
    CommandDef("status", "Show session info, model, context, and platform", "Session", supports_daemon=True),
    CommandDef(
        "host",
        "Start, stop, or inspect CuaDriver OS control",
        "Session",
        args_hint="[status|start|stop|restart]",
        supports_daemon=True,
        effect=CommandEffect.HOST_CONTROL,
        execution=CommandExecution.SHORT_OPERATION,
    ),
    CommandDef("exit", "Quit LeapFlow", "Session", aliases=("quit", "q", "退出"), supports_daemon=True),

    # Chat
    CommandDef("model", "Show or switch active model", "Chat", args_hint="[model_name]", supports_daemon=True, requires_llm=True),
    CommandDef("usage", "Show token usage for current session", "Chat", supports_daemon=True, requires_llm=True),

    # Teaching
    CommandDef("teach start", "Start teaching mode", "Teaching", aliases=("teach",), args_hint="[goal]"),
    CommandDef("teach stop", "Stop and distill skill", "Teaching", aliases=("stop", "done")),
    CommandDef("teach pause", "Pause recording", "Teaching"),
    CommandDef("teach resume", "Resume recording", "Teaching"),
    CommandDef("teach discard", "Discard current recording", "Teaching"),
    CommandDef("annotate", "Add annotation during teaching", "Teaching", args_hint="<text>"),
    CommandDef("skip", "Mark last n steps as noise", "Teaching", args_hint="[n]"),

    # Skills & Tools
    CommandDef("skills", "List all skills", "Skills & Tools", aliases=("skills list",)),
    CommandDef("skills show", "Show skill details", "Skills & Tools", args_hint="<name>"),
    CommandDef("skills disable", "Disable a skill", "Skills & Tools", args_hint="<name>"),
    CommandDef("skills delete", "Delete a skill", "Skills & Tools", args_hint="<name>"),
    CommandDef("tools", "List available tools", "Skills & Tools", supports_daemon=True),
    CommandDef("run", "Execute a skill by trigger", "Skills & Tools", args_hint="<trigger>", supports_daemon=True, requires_llm=True, execution=CommandExecution.STREAMING),

    # Hub
    CommandDef("hub push", "Push skill to hub", "Hub", args_hint="<skill>"),
    CommandDef("hub pull", "Pull skill from hub", "Hub", args_hint="<skill>"),
    CommandDef("hub search", "Search hub for skills", "Hub", args_hint="<query>"),

    # Shortcuts
    CommandDef("shortcut", "List shortcuts", "Shortcuts", aliases=("shortcut list",)),
    CommandDef("shortcut add", "Add a quick-reply shortcut", "Shortcuts", args_hint="<pattern> = <reply>"),
    CommandDef("shortcut remove", "Remove a shortcut", "Shortcuts", args_hint="<pattern>"),

    # Gateway
    CommandDef("gateway", "Show connected platforms and gateway status", "Gateway", effect=CommandEffect.EXTERNAL),

    # Scheduler
    CommandDef("arm", "Schedule a skill for timed execution", "Scheduler", args_hint="<skill> <cron>"),
    CommandDef("tasks", "List scheduled tasks", "Scheduler"),
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

    Tries longest match first: ``skills show foo`` matches ``skills show``
    before ``skills``.
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
