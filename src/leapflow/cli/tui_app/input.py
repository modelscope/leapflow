"""Slash-command completion for the TUI input area.

Provides tab-completion for REPL slash commands, consumed by the
TextArea widget in the Application layout.  The completer runs in
a background thread (via ``ThreadedCompleter``) to keep the UI
responsive during completion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from prompt_toolkit.completion import Completer, Completion, ThreadedCompleter

if TYPE_CHECKING:
    from prompt_toolkit.document import Document


class _SlashCompleter(Completer):
    """Complete slash-style commands from a static command list.

    Matches both ``/help`` and bare ``help`` input styles so the user
    can type either form.
    """

    def __init__(self, commands: Sequence[tuple[str, str]]) -> None:
        self._commands = commands

    def get_completions(self, document: "Document", complete_event):
        text = document.text_before_cursor.lstrip()
        if not text:
            return
        has_slash = text.startswith("/")
        query = text.lstrip("/")
        if not query:
            for cmd, desc in self._commands:
                yield Completion(
                    f"/{cmd}",
                    start_position=-len(text),
                    display_meta=desc,
                )
            return
        for cmd, desc in self._commands:
            if cmd.startswith(query) and cmd != query:
                completion = f"/{cmd}" if has_slash else cmd
                yield Completion(
                    completion,
                    start_position=-len(text),
                    display_meta=desc,
                )


def build_completer(
    commands: Sequence[tuple[str, str]],
) -> ThreadedCompleter:
    """Create a threaded slash-command completer for the TextArea."""
    return ThreadedCompleter(_SlashCompleter(commands or []))
