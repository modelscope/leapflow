"""LeapFlow Terminal UI — rich interactive REPL experience.

Built on ``rich`` (output rendering) and ``prompt_toolkit`` (input handling).
Provides markdown rendering, syntax highlighting, streaming display,
multiline editing, history, and adaptive theming.
"""

from leapflow.cli.tui_app.theme import Theme, detect_theme
from leapflow.cli.tui_app.console import LeapConsole
from leapflow.cli.tui_app.input import LeapInput
from leapflow.cli.tui_app.stream import StreamRenderer

__all__ = [
    "Theme",
    "detect_theme",
    "LeapConsole",
    "LeapInput",
    "StreamRenderer",
]
