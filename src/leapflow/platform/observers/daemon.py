"""Observation daemon — manages all observer lifecycles for 24/7 resident observation.

ObservationDaemon is the single entry point for starting/stopping the entire
observation subsystem. It instantiates observers based on config and platform,
manages their independent lifecycles, and provides status reporting.
"""

from __future__ import annotations

import logging
from typing import Dict, List, TYPE_CHECKING

from leapflow.platform.observers import ObserverConfig
from leapflow.platform.observers.fs_watcher import FileSystemObserver
from leapflow.platform.observers.app_focus import AppFocusObserver
from leapflow.platform.observers.clipboard import ClipboardObserver
from leapflow.platform.observers.input_tap import InputTapObserver

if TYPE_CHECKING:
    from leapflow.platform.event_bus import EventBus

logger = logging.getLogger(__name__)

# Type alias for observer instances (duck-typed to Observer Protocol)
_ObserverInstance = FileSystemObserver | AppFocusObserver | ClipboardObserver | InputTapObserver


class ObservationDaemon:
    """Manages all observers lifecycle for 24/7 resident observation.

    Responsibilities:
    - Instantiate observers based on config and platform
    - Start/stop each observer independently (one failure doesn't block others)
    - Report per-observer running status
    """

    def __init__(self, bus: "EventBus", config: ObserverConfig | None = None) -> None:
        self._bus = bus
        self._config = config or ObserverConfig()
        self._observers: Dict[str, _ObserverInstance] = {}
        self._build_observers()

    def _build_observers(self) -> None:
        """Instantiate enabled observers based on config."""
        enabled = self._config.enabled

        if enabled.get("fs_watcher", True):
            self._observers["fs_watcher"] = FileSystemObserver(
                bus=self._bus,
                watch_paths=self._config.fs_watch_paths or None,
                debounce_ms=self._config.fs_debounce_ms,
            )

        if enabled.get("app_focus", True):
            self._observers["app_focus"] = AppFocusObserver(bus=self._bus)

        if enabled.get("clipboard", True):
            self._observers["clipboard"] = ClipboardObserver(
                bus=self._bus,
                poll_interval_s=self._config.clipboard_poll_interval_s,
            )

        if enabled.get("input_tap", False):
            self._observers["input_tap"] = InputTapObserver(
                bus=self._bus,
                throttle_ms=self._config.input_throttle_ms,
            )

    async def start(self) -> None:
        """Start all observers. Idempotent. One failure doesn't block others."""
        started: List[str] = []
        failed: List[str] = []

        for name, observer in self._observers.items():
            try:
                await observer.start()
                if observer.running:
                    started.append(name)
                else:
                    failed.append(name)
            except Exception:
                logger.error("Observer '%s' failed to start", name, exc_info=True)
                failed.append(name)

        logger.info(
            "ObservationDaemon started: %d active, %d failed — active=%s, failed=%s",
            len(started), len(failed), started, failed,
        )

    async def stop(self) -> None:
        """Stop all observers gracefully. Idempotent."""
        for name, observer in self._observers.items():
            try:
                await observer.stop()
            except Exception:
                logger.error("Observer '%s' failed to stop cleanly", name, exc_info=True)

        logger.info("ObservationDaemon stopped all observers")

    @property
    def status(self) -> Dict[str, bool]:
        """Per-observer running status."""
        return {name: obs.running for name, obs in self._observers.items()}

    @property
    def observer_names(self) -> List[str]:
        """List of configured observer names."""
        return list(self._observers.keys())
