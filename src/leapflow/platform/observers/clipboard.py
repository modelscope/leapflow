"""Clipboard change observer (cross-platform).

Monitors clipboard content via polling and emits CLIPBOARD_CHANGE events
when content changes. Uses content hashing for deduplication.

Backends:
- macOS: pbpaste subprocess (pyobjc NSPasteboard if available)
- Linux: xclip subprocess
- Windows: ctypes win32clipboard
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import subprocess
import sys
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

from leapflow.platform.protocol import EventTypes

if TYPE_CHECKING:
    from leapflow.platform.event_bus import EventBus

logger = logging.getLogger(__name__)


class ClipboardObserver:
    """Observes clipboard changes via polling and publishes CLIPBOARD_CHANGE events.

    Content hash comparison ensures only genuine changes trigger events.
    """

    def __init__(
        self,
        bus: "EventBus",
        poll_interval_s: float = 1.0,
    ) -> None:
        self._bus = bus
        self._poll_interval = poll_interval_s
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None
        self._last_hash: str = ""
        self._reader = _create_clipboard_reader(sys.platform)

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start clipboard polling. Idempotent."""
        if self._running:
            return

        if self._reader is None:
            logger.warning(
                "No clipboard reader for platform: %s", sys.platform
            )
            return

        # Capture initial state to avoid spurious first event
        initial = self._reader.read()
        if initial is not None:
            self._last_hash = _content_hash(initial)

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("ClipboardObserver started (interval=%.1fs)", self._poll_interval)

    async def stop(self) -> None:
        """Stop polling. Idempotent."""
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

        logger.info("ClipboardObserver stopped")

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                content = await asyncio.get_running_loop().run_in_executor(
                    None, self._reader.read  # type: ignore[union-attr]
                )
                if content is not None:
                    h = _content_hash(content)
                    if h != self._last_hash:
                        self._last_hash = h
                        await self._emit(content)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("Clipboard poll error", exc_info=True)

            await asyncio.sleep(self._poll_interval)

    async def _emit(self, content: str) -> None:
        """Emit clipboard change event."""
        # Determine content type heuristically
        content_type = "text"
        if content.startswith("/") or content.startswith("C:\\"):
            # Might be a file path
            content_type = "file"

        payload: Dict[str, Any] = {
            "text": content,
            "content_type": content_type,
            "source_app": "",  # Platform-specific enrichment possible
            "change_ts": time.time(),
        }
        try:
            await self._bus.handle_event(EventTypes.CLIPBOARD_CHANGE, payload)
        except Exception:
            logger.error("Failed to emit CLIPBOARD_CHANGE event", exc_info=True)


def _content_hash(content: str) -> str:
    """Fast hash for deduplication (not cryptographic)."""
    return hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()


# ══════════════════════════════════════════════════════════════════════
# Platform clipboard readers
# ══════════════════════════════════════════════════════════════════════


class _ClipboardReader:
    """Base class for clipboard reading."""

    def read(self) -> Optional[str]:
        """Read current clipboard text content. Returns None on error."""
        return None


class _MacOSClipboardReader(_ClipboardReader):
    """macOS clipboard via pbpaste."""

    def read(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["pbpaste"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass
        return None


class _LinuxClipboardReader(_ClipboardReader):
    """Linux clipboard via xclip or xsel."""

    def __init__(self) -> None:
        self._cmd: Optional[list[str]] = None
        # Detect available tool
        for cmd in [
            ["xclip", "-selection", "clipboard", "-o"],
            ["xsel", "--clipboard", "--output"],
        ]:
            try:
                subprocess.run(
                    cmd, capture_output=True, timeout=1,
                )
                self._cmd = cmd
                break
            except FileNotFoundError:
                continue

    def read(self) -> Optional[str]:
        if self._cmd is None:
            return None
        try:
            result = subprocess.run(
                self._cmd,
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass
        return None


class _WindowsClipboardReader(_ClipboardReader):
    """Windows clipboard via ctypes."""

    def read(self) -> Optional[str]:
        try:
            import ctypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

            CF_UNICODETEXT = 13

            if not user32.OpenClipboard(0):
                return None
            try:
                handle = user32.GetClipboardData(CF_UNICODETEXT)
                if not handle:
                    return None
                ptr = kernel32.GlobalLock(handle)
                if not ptr:
                    return None
                try:
                    return ctypes.wstring_at(ptr)  # type: ignore[attr-defined]
                finally:
                    kernel32.GlobalUnlock(handle)
            finally:
                user32.CloseClipboard()
        except Exception:
            return None


def _create_clipboard_reader(platform: str) -> Optional[_ClipboardReader]:
    """Factory: select platform-appropriate clipboard reader."""
    if platform == "darwin":
        return _MacOSClipboardReader()
    elif platform.startswith("linux"):
        reader = _LinuxClipboardReader()
        if reader._cmd is None:
            logger.warning(
                "No clipboard tool found (xclip/xsel) — ClipboardObserver disabled"
            )
            return None
        return reader
    elif platform == "win32":
        return _WindowsClipboardReader()
    return None
