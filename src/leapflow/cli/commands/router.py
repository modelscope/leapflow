"""Slash command routing primitives.

The router keeps parsing and result semantics independent from the TUI
implementation so in-process and daemon-backed sessions can share one command
contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from leapflow.cli.commands.registry import CommandDef, resolve_command

CommandRuntime = Literal["in_process", "daemon"]


@dataclass(frozen=True)
class CommandInvocation:
    """Parsed slash command invocation."""

    raw_text: str
    text: str
    command: CommandDef
    args: str
    runtime: CommandRuntime


@dataclass(frozen=True)
class CommandResult:
    """Structured result for short slash commands."""

    ok: bool
    title: str
    summary: str = ""
    details: tuple[str, ...] = field(default_factory=tuple)
    next_actions: tuple[str, ...] = field(default_factory=tuple)
    refresh_status: bool = False
    refresh_banner: bool = False


def render_command_result(console: object, result: CommandResult) -> None:
    """Render a structured command result with a consistent tone."""
    if result.ok:
        success = getattr(console, "success", None)
        if callable(success):
            success(result.title)
        else:
            getattr(console, "system")(result.title)
    else:
        getattr(console, "warning")(result.title)
    if result.summary:
        getattr(console, "system")(result.summary)
    for detail in result.details:
        getattr(console, "system")(detail)
    if result.next_actions:
        getattr(console, "system")("Next: " + " · ".join(result.next_actions))


class CommandRouter:
    """Resolve raw user input into a runtime-aware command invocation."""

    def __init__(self, runtime: CommandRuntime) -> None:
        self._runtime = runtime

    @property
    def runtime(self) -> CommandRuntime:
        """Return the runtime this router validates against."""
        return self._runtime

    def parse(self, raw_text: str) -> CommandInvocation | None:
        """Parse raw user input into a CommandInvocation when it is a command."""
        text = raw_text.lstrip("/") if raw_text.startswith("/") else raw_text
        command = resolve_command(text)
        if command is None:
            return None
        args = text[len(command.name):].strip()
        return CommandInvocation(
            raw_text=raw_text,
            text=text,
            command=command,
            args=args,
            runtime=self._runtime,
        )

    def unsupported_result(self, invocation: CommandInvocation) -> CommandResult | None:
        """Return a standard unsupported-runtime result when needed."""
        if invocation.command.supports_runtime(invocation.runtime):
            return None
        mode = "daemon" if invocation.runtime == "daemon" else "in-process"
        return CommandResult(
            ok=False,
            title=f"/{invocation.command.name} is not available in {mode} mode yet.",
            summary="The command is registered, but its execution backend has not reached parity in this runtime.",
            next_actions=("Use /help to see runtime support", "Try --no-daemon if this is a legacy command"),
        )
