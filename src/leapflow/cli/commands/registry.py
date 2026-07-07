"""Slash-command registry and dispatch.

Single source of truth for all REPL commands.  The registry drives:
- Tab completion (via ``LeapInput``)
- ``/help`` categorized display
- Dispatch in the interactive loop
- Command alias resolution

Adding a new command = appending one ``CommandDef`` to ``COMMAND_REGISTRY``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CommandDef:
    """Definition of a single slash command."""

    name: str
    description: str
    category: str
    aliases: Tuple[str, ...] = ()
    args_hint: str = ""


# ── Registry (single source of truth) ────────────────────────────────

COMMAND_REGISTRY: Tuple[CommandDef, ...] = (
    # Session
    CommandDef("help", "Show available commands", "Session", aliases=("?", "帮助")),
    CommandDef("clear", "Clear screen and reset display", "Session"),
    CommandDef("status", "Show session info, model, context, and platform", "Session"),
    CommandDef("exit", "Quit LeapFlow", "Session", aliases=("quit", "q", "退出")),

    # Chat
    CommandDef("model", "Show or switch active model", "Chat", args_hint="[model_name]"),
    CommandDef("usage", "Show token usage for current session", "Chat"),

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
    CommandDef("tools", "List available tools", "Skills & Tools"),
    CommandDef("run", "Execute a skill by trigger", "Skills & Tools", args_hint="<trigger>"),

    # Hub
    CommandDef("hub push", "Push skill to hub", "Hub", args_hint="<skill>"),
    CommandDef("hub pull", "Pull skill from hub", "Hub", args_hint="<skill>"),
    CommandDef("hub search", "Search hub for skills", "Hub", args_hint="<query>"),

    # Shortcuts
    CommandDef("shortcut", "List shortcuts", "Shortcuts", aliases=("shortcut list",)),
    CommandDef("shortcut add", "Add a quick-reply shortcut", "Shortcuts", args_hint="<pattern> = <reply>"),
    CommandDef("shortcut remove", "Remove a shortcut", "Shortcuts", args_hint="<pattern>"),

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


def commands_by_category() -> Dict[str, List[CommandDef]]:
    """Group commands by category for display."""
    groups: Dict[str, List[CommandDef]] = {}
    for cmd in COMMAND_REGISTRY:
        groups.setdefault(cmd.category, []).append(cmd)
    return groups


def completion_entries() -> List[Tuple[str, str]]:
    """Build the completion list for LeapInput."""
    entries: List[Tuple[str, str]] = []
    for cmd in COMMAND_REGISTRY:
        full = f"/{cmd.name}" if cmd.args_hint else f"/{cmd.name}"
        entries.append((cmd.name, cmd.description))
    return entries
