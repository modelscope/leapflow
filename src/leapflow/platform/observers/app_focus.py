"""Application focus change observer (cross-platform).

Detects when the user switches between foreground applications.

Backends:
- macOS: NSWorkspace notification center (pyobjc)
- Linux: X11 _NET_ACTIVE_WINDOW property via subprocess (xdotool/xprop)
- Windows: pywin32 SetWinEventHook(EVENT_SYSTEM_FOREGROUND)
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

from leapflow.platform.protocol import EventTypes

if TYPE_CHECKING:
    from leapflow.platform.event_bus import EventBus

logger = logging.getLogger(__name__)


class AppFocusObserver:
    """Observes application foreground changes and publishes APP_FOCUS_CHANGE events.

    Automatically selects platform-specific backend.
    """

    def __init__(self, bus: "EventBus") -> None:
        self._bus = bus
        self._running = False
        self._impl: Optional[_FocusBackend] = None
        self._task: Optional[asyncio.Task[None]] = None

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start observing focus changes. Idempotent."""
        if self._running:
            return

        self._impl = _create_backend(sys.platform)
        if self._impl is None:
            logger.warning("No focus observer backend for platform: %s", sys.platform)
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("AppFocusObserver started (platform=%s)", sys.platform)

    async def stop(self) -> None:
        """Stop observing. Idempotent."""
        if not self._running:
            return

        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._impl is not None:
            self._impl.cleanup()
            self._impl = None

        logger.info("AppFocusObserver stopped")

    async def _poll_loop(self) -> None:
        """Polling loop that detects focus changes."""
        last_app_id: Optional[str] = None

        while self._running:
            try:
                info = self._impl.get_active_app() if self._impl else None
                if info is not None and info.get("app_id") != last_app_id:
                    last_app_id = info.get("app_id")
                    await self._emit(info)
            except Exception:
                logger.debug("Focus poll error", exc_info=True)

            await asyncio.sleep(0.5)

    async def _emit(self, info: Dict[str, Any]) -> None:
        """Emit focus change event through EventBus."""
        payload: Dict[str, Any] = {
            "bundle_id": info.get("app_id", ""),
            "app_name": info.get("app_name", ""),
            "pid": info.get("pid", 0),
            "window_title": info.get("window_title", ""),
            "ts": time.time(),
        }
        try:
            await self._bus.handle_event(EventTypes.APP_FOCUS_CHANGE, payload)
        except Exception:
            logger.error("Failed to emit APP_FOCUS_CHANGE event", exc_info=True)


# ══════════════════════════════════════════════════════════════════════
# Platform backends
# ══════════════════════════════════════════════════════════════════════


class _FocusBackend:
    """Base class for platform-specific focus detection."""

    def get_active_app(self) -> Optional[Dict[str, Any]]:
        """Return current foreground app info or None."""
        return None

    def cleanup(self) -> None:
        """Release platform resources."""
        pass


class _MacOSFocusBackend(_FocusBackend):
    """macOS backend using NSWorkspace (pyobjc)."""

    def get_active_app(self) -> Optional[Dict[str, Any]]:
        try:
            from AppKit import NSWorkspace
            ws = NSWorkspace.sharedWorkspace()
            app = ws.frontmostApplication()
            if app is None:
                return None
            return {
                "app_id": app.bundleIdentifier() or "",
                "app_name": app.localizedName() or "",
                "pid": app.processIdentifier(),
                "window_title": self._get_window_title(app.processIdentifier()),
            }
        except ImportError:
            # Fallback: use AppleScript
            return self._applescript_fallback()
        except Exception:
            logger.debug("macOS focus detection failed", exc_info=True)
            return None

    def _get_window_title(self, pid: int) -> str:
        """Get window title via AppleScript (best-effort)."""
        try:
            script = (
                'tell application "System Events" to get name of first window '
                f'of (first process whose unix id is {pid})'
            )
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=2,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    def _applescript_fallback(self) -> Optional[Dict[str, Any]]:
        """Fallback when pyobjc unavailable."""
        try:
            script = (
                'tell application "System Events" to get '
                '{bundle identifier, name, unix id} of first application process '
                'whose frontmost is true'
            )
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode != 0:
                return None
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                return {
                    "app_id": parts[0],
                    "app_name": parts[1],
                    "pid": int(parts[2]) if parts[2].isdigit() else 0,
                    "window_title": "",
                }
        except Exception:
            pass
        return None


class _LinuxFocusBackend(_FocusBackend):
    """Linux backend using xdotool/xprop (X11)."""

    def get_active_app(self) -> Optional[Dict[str, Any]]:
        try:
            # Get active window ID
            wid_result = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True, text=True, timeout=2,
            )
            if wid_result.returncode != 0:
                return None
            wid = wid_result.stdout.strip()

            # Get window name
            name_result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=2,
            )
            window_title = name_result.stdout.strip() if name_result.returncode == 0 else ""

            # Get PID
            pid_result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowpid"],
                capture_output=True, text=True, timeout=2,
            )
            pid = int(pid_result.stdout.strip()) if pid_result.returncode == 0 else 0

            # Get WM_CLASS for app identification
            class_result = subprocess.run(
                ["xprop", "-id", wid, "WM_CLASS"],
                capture_output=True, text=True, timeout=2,
            )
            app_id = ""
            app_name = ""
            if class_result.returncode == 0:
                # WM_CLASS(STRING) = "instance", "class"
                parts = class_result.stdout.split('"')
                if len(parts) >= 4:
                    app_id = parts[3]  # class name
                    app_name = parts[3]
                elif len(parts) >= 2:
                    app_id = parts[1]
                    app_name = parts[1]

            return {
                "app_id": app_id,
                "app_name": app_name,
                "pid": pid,
                "window_title": window_title,
            }
        except FileNotFoundError:
            logger.debug("xdotool not found — Linux focus detection unavailable")
            return None
        except Exception:
            logger.debug("Linux focus detection failed", exc_info=True)
            return None


class _WindowsFocusBackend(_FocusBackend):
    """Windows backend using ctypes."""

    def get_active_app(self) -> Optional[Dict[str, Any]]:
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]

            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return None

            # Window title
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            window_title = buf.value

            # PID
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

            # Process name from PID
            app_name = self._get_process_name(pid.value)

            return {
                "app_id": app_name,
                "app_name": app_name,
                "pid": pid.value,
                "window_title": window_title,
            }
        except Exception:
            logger.debug("Windows focus detection failed", exc_info=True)
            return None

    def _get_process_name(self, pid: int) -> str:
        """Get process executable name from PID."""
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return ""
            try:
                buf = ctypes.create_unicode_buffer(260)
                size = ctypes.c_uint(260)
                kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
                path = buf.value
                return path.split("\\")[-1] if path else ""
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return ""


def _create_backend(platform: str) -> Optional[_FocusBackend]:
    """Factory: select platform-appropriate backend."""
    if platform == "darwin":
        return _MacOSFocusBackend()
    elif platform.startswith("linux"):
        return _LinuxFocusBackend()
    elif platform == "win32":
        return _WindowsFocusBackend()
    return None
