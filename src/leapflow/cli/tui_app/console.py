"""Rich console wrapper — the single output surface for the TUI.

Centralizes all visual output: markdown rendering, code highlighting,
tool status, error panels, session info, and system messages.
Components call methods on ``LeapConsole`` instead of printing directly.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Mapping, Optional

from rich.console import Console
from rich.markdown import CodeBlock, Markdown
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


def _format_card_elapsed(seconds: float) -> str:
    """Format command-card elapsed time without consuming a body line."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    return f"{minutes}m{seconds - minutes * 60:.0f}s"


def _metadata_list(value: Any) -> list[str]:
    """Return compact string items from a metadata scalar or sequence."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    return [str(value)]


def _metadata_text(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    return "" if value is None else str(value)


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
        "leap.answer_border": theme.statusbar_dim,
        "leap.answer_title": f"bold {theme.statusbar_accent}",
        "leap.tool": theme.text_muted,
        "leap.tool_name": f"bold {theme.text_muted}",
        "markdown.h1": f"bold {theme.text}",
        "markdown.h2": f"bold {theme.accent_dim}",
        "markdown.h3": f"bold {theme.text_dim}",
        "markdown.h4": f"bold {theme.text_dim}",
        "markdown.h5": theme.text_dim,
        "markdown.h6": theme.text_dim,
        "markdown.strong": f"bold {theme.text}",
        "markdown.em": theme.text_dim,
        "markdown.hr": theme.border_dim,
        "markdown.item": theme.text,
        "markdown.code": f"bold {theme.info}",
        "rule.line": theme.border,
    })


class _TerminalBackgroundCodeBlock(CodeBlock):
    """Markdown code block that keeps the user's terminal background."""

    def __rich_console__(self, console, options):
        code = str(self.text).rstrip()
        yield Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            word_wrap=False,
            background_color="default",
            padding=1,
        )


class _TerminalBackgroundMarkdown(Markdown):
    """Rich Markdown variant with transparent fenced code blocks."""

    elements = {
        **Markdown.elements,
        "fence": _TerminalBackgroundCodeBlock,
        "code_block": _TerminalBackgroundCodeBlock,
    }


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
            TuiCommandStatus.BLOCKED: "leap.warning",
            TuiCommandStatus.FAILED: "leap.error",
            TuiCommandStatus.CANCELLED: "leap.warning",
            TuiCommandStatus.SKIPPED: "leap.warning",
        }
        title_styles = {
            TuiCommandStatus.QUEUED: "leap.muted",
            TuiCommandStatus.RUNNING: "leap.accent",
            TuiCommandStatus.DONE: "leap.success",
            TuiCommandStatus.BLOCKED: "leap.warning",
            TuiCommandStatus.FAILED: "leap.error",
            TuiCommandStatus.CANCELLED: "leap.warning",
            TuiCommandStatus.SKIPPED: "leap.warning",
        }
        summary_limit = max(_COMMAND_CARD_MIN_SUMMARY, self.width - _COMMAND_CARD_PADDING)
        body = Text(command.summary(limit=summary_limit), style="leap.muted")
        if command.error:
            body.append("\n")
            body.append(command.error, style="leap.error")
        title = Text()
        title.append(command.label, style="bold")
        title.append(f" {command.status.value}", style=title_styles[command.status])
        if command.elapsed_s > 0:
            title.append(f"  {_format_card_elapsed(command.elapsed_s)}", style="leap.dim")
        self._console.print(Panel(
            body,
            title=title,
            title_align="left",
            border_style=border_styles[command.status],
            padding=(0, 1),
        ))

    def markdown(
        self,
        text: str,
        *,
        code_theme: str = "monokai",
        indent: int = 0,
        margin_top: int = 0,
        margin_bottom: int = 0,
    ) -> None:
        """Render markdown content with optional visual spacing."""
        if not text.strip():
            return
        md = _TerminalBackgroundMarkdown(
            text,
            code_theme=code_theme if self._theme.name == "dark" else "default",
        )
        if indent > 0 or margin_top > 0 or margin_bottom > 0:
            renderable = Padding(md, (margin_top, 0, margin_bottom, indent))
        else:
            renderable = md
        self._console.print(renderable)

    def code(self, source: str, language: str = "python", *, title: str = "") -> None:
        """Render a standalone code block with syntax highlighting."""
        syntax = Syntax(
            source.rstrip(),
            language,
            theme="monokai" if self._theme.name == "dark" else "default",
            line_numbers=len(source.splitlines()) > 5,
            background_color="default",
            padding=(0, 1),
        )
        if title:
            self._console.print(Panel(syntax, title=title, border_style="leap.border"))
        else:
            self._console.print(syntax)

    def system(
        self,
        message: str,
        *,
        style: str = "leap.dim",
        margin_top: int = 0,
        margin_bottom: int = 0,
    ) -> None:
        """Print a system/info message in muted style with optional spacing."""
        for _ in range(max(0, margin_top)):
            self._console.print()
        self._console.print(f"  {message}", style=style)
        for _ in range(max(0, margin_bottom)):
            self._console.print()

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

    def permission_recovery_card(self, metadata: Mapping[str, Any]) -> None:
        """Render a copy-safe permission recovery card for App Connector failures."""
        platform = _metadata_text(metadata, "platform") or _metadata_text(metadata, "onboarding_platform")
        action = _metadata_text(metadata, "action")
        capability = _metadata_text(metadata, "capability")
        failure_code = _metadata_text(metadata, "failure_code")
        recoverability = _metadata_text(metadata, "recoverability")
        console_url = _metadata_text(metadata, "console_url")
        recovery_hint = _metadata_text(metadata, "recovery_hint")
        missing_scopes = _metadata_list(metadata.get("missing_scopes"))
        required_scopes = _metadata_list(metadata.get("required_scopes"))
        scope_relation = _metadata_text(metadata, "scope_relation") or "all_required"
        next_steps = _metadata_list(metadata.get("next_steps"))

        body = Text()
        if platform or action or capability:
            target = ".".join(part for part in (platform, action) if part)
            body.append("目标: ", style="leap.muted")
            body.append(target or capability or "平台能力", style="leap.error")
            if capability and capability != target:
                body.append(f"  capability={capability}", style="leap.muted")
            body.append("\n")
        if failure_code or recoverability:
            body.append("原因: ", style="leap.muted")
            body.append(failure_code or "authorization_required", style="leap.error")
            if recoverability:
                body.append(f"  recoverability={recoverability}", style="leap.muted")
            body.append("\n")
        scopes = missing_scopes or required_scopes
        if scopes:
            label = "缺失权限" if missing_scopes else "需要权限"
            if scope_relation == "one_of" and len(scopes) > 1:
                body.append(f"{label}（任选其中一项即可）:\n", style="leap.muted")
            else:
                body.append(f"{label}:\n", style="leap.muted")
            scope_style = "leap.error" if missing_scopes else "leap.warning"
            for scope in scopes:
                body.append(f"  - {scope}\n", style=scope_style)
        if console_url:
            body.append("开发者后台链接（可复制）:\n", style="leap.muted")
            body.append(f"  {console_url}\n", style="leap.info")
        if next_steps:
            body.append("下一步:\n", style="leap.muted")
            for step in next_steps[:5]:
                body.append(f"  {step}\n", style="leap.muted")
        elif recovery_hint:
            body.append("恢复提示: ", style="leap.muted")
            body.append(recovery_hint, style="leap.muted")
            body.append("\n")
        if not console_url:
            body.append("请在平台开发者后台补齐权限后重新发布/安装应用。\n", style="leap.muted")

        body.rstrip()
        if not body.plain.strip():
            body = Text("请在平台开发者后台补齐权限后重新发布/安装应用。", style="leap.muted")

        self._console.print(Panel(
            body,
            title=Text("🔐 权限恢复", style="bold"),
            title_align="left",
            border_style="leap.warning",
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
            Syntax(
                display,
                "text",
                theme="monokai" if self._theme.name == "dark" else "default",
                background_color="default",
            )
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

    def answer_label(self) -> None:
        """Print a clear warm boundary before the user-facing final answer."""
        title = Text(" LeapFlow ", style="leap.answer_title")
        self._console.print(Rule(
            title=title,
            style="leap.answer_border",
            align="left",
        ))

    def response_label(
        self,
        elapsed_s: float,
        *,
        tool_count: int = 0,
        command: TuiCommand | None = None,
    ) -> None:
        """Print the response attribution line with optional command status."""
        from leapflow.cli.tui_app.stream import _format_elapsed

        label = Text()
        label.append(" |--  ", style="leap.border")
        label.append("LEAP", style="leap.accent")
        if command is not None:
            command_styles = {
                TuiCommandStatus.DONE: "leap.success",
                TuiCommandStatus.BLOCKED: "leap.warning",
                TuiCommandStatus.FAILED: "leap.error",
                TuiCommandStatus.CANCELLED: "leap.warning",
                TuiCommandStatus.SKIPPED: "leap.warning",
            }
            command_style = command_styles.get(command.status, "leap.dim")
            label.append(f"  {command.label} {command.status.value}", style=command_style)
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
