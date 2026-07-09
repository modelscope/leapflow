"""LeapFlow Terminal UI — hybrid Application + Rich architecture.

Built on ``prompt_toolkit`` (Application layout, fixed input, key bindings)
and ``rich`` (Markdown rendering, panels, syntax highlighting, theming).
"""

from leapflow.cli.tui_app.app import LeapApp
from leapflow.cli.tui_app.console import LeapConsole
from leapflow.cli.tui_app.session_summary import (
    SessionExitStats,
    build_exit_summary_lines,
    format_duration,
    summarize_messages,
)
from leapflow.cli.tui_app.stream import StreamRenderer
from leapflow.cli.tui_app.theme import (
    ResolvedTheme,
    Theme,
    contrast_ratio,
    detect_theme,
    ensure_contrast,
    parse_hex_color,
    relative_luminance,
    resolve_theme,
)

__all__ = [
    "LeapApp",
    "LeapConsole",
    "SessionExitStats",
    "StreamRenderer",
    "build_exit_summary_lines",
    "ResolvedTheme",
    "Theme",
    "contrast_ratio",
    "detect_theme",
    "ensure_contrast",
    "format_duration",
    "parse_hex_color",
    "relative_luminance",
    "resolve_theme",
    "summarize_messages",
]
