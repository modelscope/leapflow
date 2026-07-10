"""Rich console wrapper — the single output surface for the TUI.

Centralizes all visual output: markdown rendering, code highlighting,
tool status, error panels, session info, and system messages.
Components call methods on ``LeapConsole`` instead of printing directly.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme as RichTheme

from leapflow.cli.tui_app.command import TuiCommand, TuiCommandStatus
from leapflow.cli.tui_app.theme import ResolvedTheme, Theme

_COMMAND_CARD_MIN_SUMMARY = 32
_COMMAND_CARD_PADDING = 24


def _build_rich_theme(theme: Theme | ResolvedTheme) -> RichTheme:
    """Map LeapFlow theme to a Rich style dict."""
    return RichTheme({
        "leap.accent": theme.accent,
        "leap.dim": theme.text_dim,
        "leap.muted": theme.text_muted,
        "leap.success": theme.success,
        "leap.warning": theme.warning,
        "leap.error": theme.error,
        "leap.info": theme.info,
        "leap.border": theme.border,
        "leap.recording": theme.recording,
        "leap.executing": theme.executing,
        "leap.panel_title": theme.panel_title,
        "rule.line": theme.border,
    })


class LeapConsole:
    """Unified output surface wrapping ``rich.Console``.

    All TUI output flows through this class, ensuring consistent
    theming and preventing raw print() calls from breaking layout.
    """

    def __init__(self, theme: Theme | ResolvedTheme) -> None:
        self._theme = theme
        self._console = Console(
            theme=_build_rich_theme(theme),
            highlight=False,
            soft_wrap=True,
        )

    @property
    def theme(self) -> Theme | ResolvedTheme:
        return self._theme

    @property
    def width(self) -> int:
        return self._console.width

    @property
    def is_tty(self) -> bool:
        return self._console.is_terminal

    @property
    def raw(self) -> Console:
        """Direct access to the underlying Rich Console."""
        return self._console

    def print(self, *args, **kwargs) -> None:
        """Pass-through to rich.Console.print."""
        self._console.print(*args, **kwargs)

    def command_card(self, command: TuiCommand) -> None:
        """Render a compact lifecycle card for a queued or running command."""
        border_styles = {
            TuiCommandStatus.QUEUED: "leap.border",
            TuiCommandStatus.RUNNING: "leap.accent",
            TuiCommandStatus.DONE: "leap.success",
            TuiCommandStatus.FAILED: "leap.error",
        }
        title_styles = {
            TuiCommandStatus.QUEUED: "leap.muted",
            TuiCommandStatus.RUNNING: "leap.accent",
            TuiCommandStatus.DONE: "leap.success",
            TuiCommandStatus.FAILED: "leap.error",
        }
        summary_limit = max(_COMMAND_CARD_MIN_SUMMARY, self.width - _COMMAND_CARD_PADDING)
        body = Text(command.summary(limit=summary_limit), style="leap.muted")
        if command.error:
            body.append("\n")
            body.append(command.error, style="leap.error")
        if command.elapsed_s > 0:
            body.append("\n")
            body.append(f"elapsed: {command.elapsed_s:.1f}s", style="leap.dim")
        title = Text()
        title.append(command.label, style="bold")
        title.append(f" {command.status.value}", style=title_styles[command.status])
        self._console.print(Panel(
            body,
            title=title,
            title_align="left",
            border_style=border_styles[command.status],
            padding=(0, 1),
        ))

    def markdown(self, text: str, *, code_theme: str = "monokai", indent: int = 0) -> None:
        """Render markdown content with syntax-highlighted code blocks."""
        if not text.strip():
            return
        md = Markdown(
            text,
            code_theme=code_theme if self._theme.name == "dark" else "default",
        )
        renderable = Padding(md, (0, 0, 0, indent)) if indent > 0 else md
        self._console.print(renderable)

    def code(self, source: str, language: str = "python", *, title: str = "") -> None:
        """Render a standalone code block with syntax highlighting."""
        syntax = Syntax(
            source.rstrip(),
            language,
            theme="monokai" if self._theme.name == "dark" else "default",
            line_numbers=len(source.splitlines()) > 5,
            padding=(0, 1),
        )
        if title:
            self._console.print(Panel(syntax, title=title, border_style="leap.border"))
        else:
            self._console.print(syntax)

    def system(self, message: str, *, style: str = "leap.dim") -> None:
        """Print a system/info message in muted style."""
        self._console.print(f"  {message}", style=style)

    def success(self, message: str) -> None:
        self._console.print(f"  ✓ {message}", style="leap.success")

    def warning(self, message: str) -> None:
        self._console.print(f"  ⚠ {message}", style="leap.warning")

    def error(self, message: str) -> None:
        self._console.print(f"  ✗ {message}", style="leap.error")

    def error_panel(self, title: str, body: str) -> None:
        """Render a prominent error panel."""
        self._console.print(Panel(
            Text(body),
            title=title,
            border_style="leap.error",
            padding=(0, 1),
        ))

    def rule(self, title: str = "", *, style: Optional[str] = None) -> None:
        """Print a horizontal rule, optionally titled."""
        self._console.print(Rule(
            title=title,
            style=style or "leap.border",
        ))

    def tool_start(self, name: str, args_summary: str = "") -> None:
        """Announce a tool call starting."""
        label = Text()
        label.append("  ⚡ ", style="leap.accent")
        label.append(name, style="bold")
        if args_summary:
            label.append(f" {args_summary}", style="leap.dim")
        self._console.print(label)

    def tool_result(self, name: str, output: str, *, is_error: bool = False) -> None:
        """Display a tool call result, truncated if very long."""
        max_lines = 20
        lines = output.splitlines()
        truncated = len(lines) > max_lines
        display = "\n".join(lines[:max_lines])
        if truncated:
            display += f"\n  … ({len(lines) - max_lines} more lines)"

        style = "leap.error" if is_error else "leap.border"
        self._console.print(Panel(
            Syntax(display, "text", theme="monokai" if self._theme.name == "dark" else "default")
            if not is_error else Text(display),
            title=f"{'✗' if is_error else '↳'} {name}",
            border_style=style,
            padding=(0, 1),
        ))

    def thinking(self, text: str) -> None:
        """Display LLM thinking/reasoning content."""
        if not text.strip():
            return
        content = Text(text.strip(), style="leap.muted")
        self._console.print(Panel(
            content,
            title="💭 thinking",
            title_align="left",
            border_style="leap.border",
            padding=(0, 1),
        ))

    def response_label(self, elapsed_s: float, *, tool_count: int = 0) -> None:
        """Print the response attribution line with elapsed time."""
        from leapflow.cli.tui_app.stream import _format_elapsed

        label = Text()
        label.append("  ─ ", style="leap.border")
        label.append("LEAP", style="leap.accent")
        elapsed_str = _format_elapsed(elapsed_s)
        label.append(f"  {elapsed_str}", style="leap.dim")
        if tool_count > 0:
            label.append(f"  {tool_count} tool{'s' if tool_count != 1 else ''}", style="leap.dim")
        self._console.print(label)

    def session_info(
        self,
        *,
        model: str = "",
        platform: str = "",
        cwd: str = "",
        skill_count: int = 0,
        session_id: str = "",
    ) -> None:
        """Display compact session information after the banner."""
        info_parts: list[str] = []
        if model:
            info_parts.append(f"model: {model}")
        if platform:
            info_parts.append(f"platform: {platform}")
        if cwd:
            short_cwd = cwd.replace(os.path.expanduser("~"), "~")
            info_parts.append(f"cwd: {short_cwd}")
        if skill_count > 0:
            info_parts.append(f"skills: {skill_count}")

        if info_parts:
            self._console.print(
                f"  {' │ '.join(info_parts)}",
                style="leap.dim",
            )
        if session_id:
            self._console.print(
                f"  session: {session_id[:12]}",
                style="leap.muted",
            )

    def newline(self) -> None:
        self._console.print()

    def flush(self) -> None:
        """Ensure all buffered output is written."""
        if hasattr(sys.stdout, "flush"):
            sys.stdout.flush()
