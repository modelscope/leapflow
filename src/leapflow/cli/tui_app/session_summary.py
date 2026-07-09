"""Exit summary helpers for interactive TUI sessions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Protocol


class ConversationMessageLike(Protocol):
    role: str
    tool_name: str | None
    tool_calls_json: str | None


@dataclass
class SessionExitStats:
    """Small session-local counters for TUI exit reporting."""

    started_at: float = field(default_factory=time.monotonic)
    user_messages: int = 0
    assistant_messages: int = 0
    tool_calls: int = 0

    @property
    def message_count(self) -> int:
        return self.user_messages + self.assistant_messages

    @property
    def duration_s(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    def record_user_message(self) -> None:
        self.user_messages += 1

    def record_assistant_message(self) -> None:
        self.assistant_messages += 1

    def record_tool_calls(self, count: int) -> None:
        self.tool_calls += max(0, count)


def format_duration(seconds: float) -> str:
    """Format seconds as a compact human-readable duration."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def summarize_messages(
    messages: Iterable[ConversationMessageLike],
) -> tuple[int, int, int]:
    """Return total, user, and tool-call counts from stored messages."""
    total = 0
    user_messages = 0
    tool_calls = 0
    for message in messages:
        total += 1
        if message.role == "user":
            user_messages += 1
        if message.role == "tool" or message.tool_name or message.tool_calls_json:
            tool_calls += 1
    return total, user_messages, tool_calls


def build_exit_summary_lines(
    *,
    session_id: str,
    duration_s: float,
    message_count: int,
    user_messages: int,
    tool_calls: int,
    resume_command: str = "leap --resume",
) -> list[str]:
    """Build Hermes-style TUI exit summary lines."""
    normalized_session_id = session_id.strip()
    if not normalized_session_id:
        return ["Goodbye!"]

    return [
        "Resume this session with:",
        f"  {resume_command} {normalized_session_id}",
        "",
        f"Session:        {normalized_session_id}",
        f"Duration:       {format_duration(duration_s)}",
        (
            f"Messages:       {message_count} "
            f"({user_messages} user, {tool_calls} tool calls)"
        ),
    ]
