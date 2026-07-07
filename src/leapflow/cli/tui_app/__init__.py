"""LeapFlow Terminal UI — hybrid Application + Rich architecture.

Built on ``prompt_toolkit`` (Application layout, fixed input, key bindings)
and ``rich`` (Markdown rendering, panels, syntax highlighting, theming).
"""

from leapflow.cli.tui_app.app import LeapApp
from leapflow.cli.tui_app.console import LeapConsole
from leapflow.cli.tui_app.stream import StreamRenderer
from leapflow.cli.tui_app.theme import Theme, detect_theme

__all__ = [
    "LeapApp",
    "LeapConsole",
    "StreamRenderer",
    "Theme",
    "detect_theme",
]
