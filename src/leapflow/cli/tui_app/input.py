"""prompt_toolkit-based input with history, multiline editing, and completion.

Replaces raw ``input()`` with a rich editing experience:
- Persistent history (``~/.leapflow/history``)
- Multiline editing (Shift+Enter or backslash-newline)
- In-prompt mode indicator with theme-aware colors
- Slash-command completion
- Vi/Emacs key binding selection via config
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Sequence, TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style

from leapflow.cli.tui_app.theme import Theme

if TYPE_CHECKING:
    from prompt_toolkit.document import Document

_HISTORY_FILE = "history"


class _SlashCompleter(Completer):
    """Complete slash-style commands from a static command list.

    Matches both ``/help`` and bare ``help`` input styles.
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
                yield Completion(f"/{cmd}", start_position=-len(text), display_meta=desc)
            return
        for cmd, desc in self._commands:
            if cmd.startswith(query) and cmd != query:
                completion = f"/{cmd}" if has_slash else cmd
                yield Completion(completion, start_position=-len(text), display_meta=desc)


def _build_style(theme: Theme) -> Style:
    """Build prompt_toolkit style from LeapFlow theme."""
    is_light = theme.name == "light"
    return Style.from_dict({
        "prompt": "bold ansibrightcyan" if not is_light else "bold ansicyan",
        "prompt.mode": "bold ansired" if not is_light else "bold ansired",
        "prompt.mode.paused": "bold ansiyellow",
        "prompt.mode.executing": "bold ansigreen",
        "": "" if not is_light else "ansiblack",
        "bottom-toolbar": "bg:ansibrightblack" if not is_light else "bg:ansigray ansiblack",
        "bottom-toolbar.text": "" if not is_light else "ansiblack",
    })


def _build_keybindings() -> KeyBindings:
    """Configure key bindings for multiline editing and navigation."""
    kb = KeyBindings()

    @kb.add(Keys.Escape, Keys.Enter)
    def _multiline_alt_enter(event):
        """Alt+Enter: insert newline (multiline mode)."""
        event.current_buffer.insert_text("\n")

    @kb.add(Keys.ControlD)
    def _eof(event):
        """Ctrl+D: raise EOFError (exit)."""
        event.app.exit(exception=EOFError)

    return kb


class LeapInput:
    """Interactive input manager wrapping ``prompt_toolkit.PromptSession``.

    Provides a rich editing experience with history persistence,
    multiline support, mode-aware prompts, and command completion.
    """

    def __init__(
        self,
        theme: Theme,
        *,
        data_dir: Optional[Path] = None,
        commands: Optional[Sequence[tuple[str, str]]] = None,
        editing_mode: str = "emacs",
    ) -> None:
        self._theme = theme
        data_dir = data_dir or Path(
            os.environ.get("LEAPFLOW_DATA_DIR", "~/.leapflow")
        ).expanduser()
        data_dir.mkdir(parents=True, exist_ok=True)

        history_path = data_dir / _HISTORY_FILE
        completer = _SlashCompleter(commands or _default_commands())

        mode_map = {
            "vi": EditingMode.VI,
            "vim": EditingMode.VI,
            "emacs": EditingMode.EMACS,
        }

        self._session: PromptSession = PromptSession(
            history=FileHistory(str(history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
            complete_while_typing=False,
            style=_build_style(theme),
            key_bindings=_build_keybindings(),
            editing_mode=mode_map.get(editing_mode.lower(), EditingMode.EMACS),
            enable_history_search=True,
            mouse_support=False,
            multiline=False,
        )

    async def prompt(
        self,
        mode: str = "idle",
        *,
        bottom_toolbar: Optional[Callable] = None,
    ) -> str:
        """Prompt the user for input.

        Args:
            mode: Current session mode (idle, learning, paused, executing).
            bottom_toolbar: Callable returning toolbar content.

        Returns:
            Stripped user input string.

        Raises:
            EOFError: On Ctrl+D.
            KeyboardInterrupt: On Ctrl+C.
        """
        prompt_tokens = self._build_prompt(mode)
        result = await self._session.prompt_async(
            prompt_tokens,
            bottom_toolbar=bottom_toolbar,
        )
        return result.strip()

    def _build_prompt(self, mode: str) -> FormattedText:
        """Build mode-aware prompt with theme colors."""
        if mode == "learning":
            return FormattedText([
                ("class:prompt.mode", " ● rec "),
            ])
        elif mode == "paused":
            return FormattedText([
                ("class:prompt.mode.paused", " ⏸ "),
            ])
        elif mode == "executing":
            return FormattedText([
                ("class:prompt.mode.executing", " ▶ "),
            ])
        return FormattedText([
            ("class:prompt", " ❯ "),
        ])


def _default_commands() -> list[tuple[str, str]]:
    """Default slash-style commands for completion."""
    return [
        ("help", "Show available commands"),
        ("exit", "Quit LeapFlow"),
        ("teach start", "Start teaching mode"),
        ("teach stop", "Stop and distill"),
        ("teach pause", "Pause recording"),
        ("teach resume", "Resume recording"),
        ("teach discard", "Discard recording"),
        ("teach save", "Save session for later"),
        ("skills", "List all skills"),
        ("skills list", "List all skills"),
        ("skills show", "Show skill details"),
        ("skills disable", "Disable a skill"),
        ("skills delete", "Delete a skill"),
        ("run", "Execute a skill"),
        ("hub push", "Push skill to hub"),
        ("hub pull", "Pull skill from hub"),
        ("hub search", "Search hub"),
    ]
