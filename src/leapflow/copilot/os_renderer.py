"""OS-native notification renderer for Copilot suggestions.

Platform-specific implementations:
- macOS: osascript + display notification (no external dependencies)
- Linux: notify-send (if available)
- Windows: PowerShell toast (if available)
- Fallback: stderr indicator

Design principle: "Quiet technology" — notifications are non-intrusive,
auto-dismiss, and respect system DND settings.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import sys
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from leapflow.copilot.types import PredictionCandidate

logger = logging.getLogger(__name__)


class OSNotificationRenderer:
    """Renders Copilot suggestions as OS-native notifications.

    Implements HintRenderer protocol (duck-typed):
    - show(candidate) -> display notification
    - dismiss() -> clear notification state
    - is_visible -> whether a notification is currently shown

    Falls back gracefully if OS notification is unavailable.
    """

    def __init__(
        self,
        *,
        app_name: str = "LeapFlow",
        subtitle: str = "Copilot Suggestion",
        auto_dismiss_s: float = 8.0,
        sound: bool = False,
    ) -> None:
        self._app_name = app_name
        self._subtitle = subtitle
        self._auto_dismiss_s = auto_dismiss_s
        self._sound = sound
        self._visible: bool = False
        self._current: Optional[PredictionCandidate] = None
        self._platform = platform.system().lower()
        self._dismiss_task: Optional[asyncio.Task] = None

    async def show(self, candidate: PredictionCandidate) -> None:
        """Display an OS notification for the suggestion."""
        self._visible = True
        self._current = candidate

        message = candidate.action_description
        title = f"{self._app_name} — {self._subtitle}"

        try:
            if self._platform == "darwin":
                await self._show_macos(title, message)
            elif self._platform == "linux":
                await self._show_linux(title, message)
            elif self._platform == "windows":
                await self._show_windows(title, message)
            else:
                self._show_fallback(title, message)
        except Exception:
            logger.debug("OS notification failed, using fallback", exc_info=True)
            self._show_fallback(title, message)

        # Schedule auto-dismiss
        if self._dismiss_task and not self._dismiss_task.done():
            self._dismiss_task.cancel()
        self._dismiss_task = asyncio.create_task(self._auto_dismiss())

    async def dismiss(self) -> None:
        """Dismiss current notification (clear internal state)."""
        if self._dismiss_task and not self._dismiss_task.done():
            self._dismiss_task.cancel()
        self._visible = False
        self._current = None

    @property
    def is_visible(self) -> bool:
        """Whether a notification is currently shown."""
        return self._visible

    # -- Platform-specific implementations --

    async def _show_macos(self, title: str, message: str) -> None:
        """macOS notification via osascript."""
        # Escape single quotes for AppleScript
        safe_title = title.replace("'", "'\\''")
        safe_message = message.replace("'", "'\\''")

        sound_clause = ' sound name "Submarine"' if self._sound else ""
        script = (
            f"display notification '{safe_message}' "
            f"with title '{safe_title}'{sound_clause}"
        )

        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", script,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()

        if proc.returncode != 0:
            logger.debug("osascript notification returned code %d", proc.returncode)

    async def _show_linux(self, title: str, message: str) -> None:
        """Linux notification via notify-send."""
        proc = await asyncio.create_subprocess_exec(
            "notify-send",
            "--app-name", self._app_name,
            "--expire-time", str(int(self._auto_dismiss_s * 1000)),
            title, message,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()

    async def _show_windows(self, title: str, message: str) -> None:
        """Windows notification via PowerShell toast."""
        ps_script = (
            "[Windows.UI.Notifications.ToastNotificationManager, "
            "Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null; "
            "$template = [Windows.UI.Notifications.ToastTemplateType]::ToastText02; "
            "$xml = [Windows.UI.Notifications.ToastNotificationManager]"
            "::GetTemplateContent($template); "
            '$text = $xml.GetElementsByTagName("text"); '
            f'$text[0].AppendChild($xml.CreateTextNode("{title}")) | Out-Null; '
            f'$text[1].AppendChild($xml.CreateTextNode("{message}")) | Out-Null; '
            "$toast = [Windows.UI.Notifications.ToastNotification]::new($xml); "
            "[Windows.UI.Notifications.ToastNotificationManager]"
            f'::CreateToastNotifier("{self._app_name}").Show($toast)'
        )
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-Command", ps_script,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()

    def _show_fallback(self, title: str, message: str) -> None:
        """Fallback: write to stderr with ANSI styling."""
        if sys.stderr.isatty():
            sys.stderr.write(f"\033[2m\U0001f4a1 {title}: {message}\033[0m\n")
        else:
            sys.stderr.write(f"[hint] {title}: {message}\n")
        sys.stderr.flush()

    async def _auto_dismiss(self) -> None:
        """Auto-dismiss after configured timeout."""
        try:
            await asyncio.sleep(self._auto_dismiss_s)
            self._visible = False
            self._current = None
        except asyncio.CancelledError:
            pass


class StderrHintRenderer:
    """Minimal stderr-based renderer — for headless/CI environments."""

    def __init__(self) -> None:
        self._visible: bool = False

    async def show(self, candidate: PredictionCandidate) -> None:
        """Display suggestion on stderr."""
        if sys.stderr.isatty():
            sys.stderr.write(
                f"\033[2m\U0001f4a1 Suggestion: {candidate.action_description} "
                f"(confidence={candidate.confidence:.2f})\033[0m\n"
            )
        else:
            sys.stderr.write(
                f"[hint] Suggestion: {candidate.action_description} "
                f"(confidence={candidate.confidence:.2f})\n"
            )
        sys.stderr.flush()
        self._visible = True

    async def dismiss(self) -> None:
        """Dismiss (clear state)."""
        self._visible = False

    @property
    def is_visible(self) -> bool:
        """Whether a hint is currently being shown."""
        return self._visible


class NoOpHintRenderer:
    """No-op renderer — suppresses all notifications."""

    async def show(self, candidate: "PredictionCandidate") -> None:
        pass

    async def dismiss(self) -> None:
        pass

    @property
    def is_visible(self) -> bool:
        return False


def create_renderer(
    mode: str = "auto", **kwargs
) -> "OSNotificationRenderer | StderrHintRenderer | NoOpHintRenderer":
    """Factory function for creating the appropriate renderer.

    Args:
        mode: "os" (native notifications), "stderr" (terminal),
              "log" (logger.info), "auto" (detect), "none" (no-op stderr)
    """
    if mode == "os":
        return OSNotificationRenderer(**kwargs)
    elif mode == "stderr":
        return StderrHintRenderer()
    elif mode == "log":
        from leapflow.copilot.renderer import LogHintRenderer
        return LogHintRenderer()
    elif mode == "auto":
        # Auto-detect: use OS notifications if not in CI/headless
        if sys.stderr.isatty() and platform.system() in ("Darwin", "Linux", "Windows"):
            return OSNotificationRenderer(**kwargs)
        return StderrHintRenderer()
    elif mode == "none":
        return NoOpHintRenderer()
    else:
        return StderrHintRenderer()
