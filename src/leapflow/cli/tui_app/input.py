"""Slash-command completion for the TUI input area.

Provides interactive slash-command suggestions for the TextArea widget.
The completer keeps command discovery read-only and lightweight: it only uses
the static command registry entries already loaded by the CLI layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from prompt_toolkit.completion import Completer, Completion, ThreadedCompleter

if TYPE_CHECKING:
    from collections.abc import Iterable

    from prompt_toolkit.document import Document


_MAX_DESCRIPTION_WIDTH = 88


def _truncate_meta(text: str, *, width: int = _MAX_DESCRIPTION_WIDTH) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= width:
        return normalized
    return normalized[: max(0, width - 1)].rstrip() + "…"


class SlashCommandCompleter(Completer):
    """Complete slash commands with display text and descriptions.

    The returned ``Completion`` objects are intentionally prompt_toolkit-native:
    ``display`` renders the left command column and ``display_meta`` renders the
    right description column used by ``CompletionsMenu``.  This gives LeapFlow a
    Hermes-style slash menu without owning a custom focus or scroll state.
    """

    def __init__(self, commands: Sequence[tuple[str, str]]) -> None:
        self._commands = tuple(commands)

    @property
    def commands(self) -> tuple[tuple[str, str], ...]:
        """Return the immutable command catalog used by this completer."""
        return self._commands

    def get_completions(
        self,
        document: "Document",
        complete_event,
    ) -> "Iterable[Completion]":
        text = document.text_before_cursor.lstrip()
        if not text:
            return
        has_slash = text.startswith("/")
        if not has_slash and " " in text:
            return

        query = text.lstrip("/").lower()
        for command, description in self._commands:
            command_lower = command.lower()
            if query and not command_lower.startswith(query):
                continue
            completion = f"/{command}" if has_slash else command
            yield Completion(
                completion,
                start_position=-len(text),
                display=f"/{command}",
                display_meta=_truncate_meta(description),
            )


def build_completer(
    commands: Sequence[tuple[str, str]],
) -> ThreadedCompleter:
    """Create a threaded slash-command completer for the TextArea."""
    return ThreadedCompleter(SlashCommandCompleter(commands or []))
