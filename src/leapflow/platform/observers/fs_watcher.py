"""File system change observer using watchdog (cross-platform).

Backends:
- macOS: FSEvents (kqueue fallback)
- Linux: inotify
- Windows: ReadDirectoryChangesW

Events are debounced per-path within a configurable window (default 500ms).
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from leapflow.platform.protocol import EventTypes

if TYPE_CHECKING:
    from leapflow.platform.event_bus import EventBus

logger = logging.getLogger(__name__)


class FileSystemObserver:
    """Observes filesystem changes and publishes FS_CHANGE events.

    Uses the watchdog library for cross-platform FS monitoring.
    Debounces rapid changes to the same path within a configurable window.
    """

    def __init__(
        self,
        bus: "EventBus",
        watch_paths: Optional[list[str]] = None,
        debounce_ms: int = 500,
    ) -> None:
        self._bus = bus
        self._watch_paths = watch_paths or [str(Path.home())]
        self._debounce_ms = debounce_ms
        self._running = False
        self._observer: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Debounce state: path -> last_event_monotonic_time
        self._last_events: Dict[str, float] = {}

    @property
    def running(self) -> bool:
        return self._running

    def set_debounce_ms(self, ms: int) -> None:
        """Hot-reconfigure debounce window (e.g. for recording mode)."""
        self._debounce_ms = max(0, ms)

    async def start(self) -> None:
        """Start watching configured paths. Idempotent."""
        if self._running:
            return

        try:
            from watchdog.observers import Observer as WatchdogObserver
            from watchdog.events import FileSystemEventHandler, FileSystemEvent  # noqa: F401
        except ImportError:
            logger.warning(
                "watchdog not installed — FileSystemObserver disabled. "
                "Install with: pip install watchdog"
            )
            return

        self._loop = asyncio.get_running_loop()

        handler = _WatchdogHandler(self)
        self._observer = WatchdogObserver()

        for path_str in self._watch_paths:
            path = Path(path_str)
            if not path.exists():
                logger.warning("Watch path does not exist, skipping: %s", path_str)
                continue
            self._observer.schedule(handler, str(path), recursive=True)
            logger.debug("Watching: %s", path_str)

        self._observer.start()
        self._running = True
        logger.info(
            "FileSystemObserver started, watching %d paths", len(self._watch_paths)
        )

    async def stop(self) -> None:
        """Stop watching. Idempotent."""
        if not self._running:
            return

        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

        self._running = False
        self._last_events.clear()
        logger.info("FileSystemObserver stopped")

    def _should_debounce(self, path: str) -> bool:
        """Return True if this path event should be suppressed (within debounce window)."""
        now = time.monotonic()
        last = self._last_events.get(path, 0.0)
        if (now - last) * 1000 < self._debounce_ms:
            return True
        self._last_events[path] = now
        # Prune stale entries
        if len(self._last_events) > 1000:
            cutoff = now - (self._debounce_ms / 1000.0) * 2
            self._last_events = {
                k: v for k, v in self._last_events.items() if v > cutoff
            }
        return False

    def _on_fs_event(self, path: str, action: str, is_dir: bool) -> None:
        """Called from watchdog thread — schedules async event dispatch."""
        if self._should_debounce(path):
            return

        if self._loop is None or self._loop.is_closed():
            return

        payload: Dict[str, Any] = {
            "path": path,
            "action": action,
            "is_dir": is_dir,
        }
        # Schedule coroutine from watchdog's thread
        asyncio.run_coroutine_threadsafe(
            self._emit(payload), self._loop
        )

    async def _emit(self, payload: Dict[str, Any]) -> None:
        """Emit event through EventBus."""
        try:
            await self._bus.handle_event(EventTypes.FS_CHANGE, payload)
        except Exception:
            logger.error("Failed to emit FS_CHANGE event", exc_info=True)


class _WatchdogHandler:
    """Adapts watchdog events to FileSystemObserver._on_fs_event calls."""

    def __init__(self, observer: FileSystemObserver) -> None:
        self._observer = observer

    def dispatch(self, event: Any) -> None:
        """Called by watchdog for every FS event."""
        from watchdog.events import (
            EVENT_TYPE_CREATED,
            EVENT_TYPE_DELETED,
            EVENT_TYPE_MODIFIED,
            EVENT_TYPE_MOVED,
        )

        action_map = {
            EVENT_TYPE_CREATED: "created",
            EVENT_TYPE_DELETED: "deleted",
            EVENT_TYPE_MODIFIED: "modified",
            EVENT_TYPE_MOVED: "moved",
        }

        action = action_map.get(event.event_type, "modified")
        path = getattr(event, "dest_path", None) or event.src_path
        is_dir = event.is_directory

        self._observer._on_fs_event(path, action, is_dir)
