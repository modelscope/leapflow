"""Observation daemon — manages all observer lifecycles for 24/7 resident observation.

ObservationDaemon is the single entry point for starting/stopping the entire
observation subsystem. It instantiates observers based on config and platform,
manages their independent lifecycles, and provides status reporting.

Supports ``RecordingProfile`` for high-fidelity mode during active recording:
tighter FS debounce, lower input throttle, and auto-enabling the input tap
observer.  Call ``apply_profile`` / ``reset_profile`` to switch.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

from leapflow.platform.observers import ObserverConfig, RecordingProfile
from leapflow.platform.observers.fs_watcher import FileSystemObserver
from leapflow.platform.observers.app_focus import AppFocusObserver
from leapflow.platform.observers.clipboard import ClipboardObserver
from leapflow.platform.observers.input_tap import InputTapObserver

if TYPE_CHECKING:
    from leapflow.platform.event_bus import EventBus

logger = logging.getLogger(__name__)

_ObserverInstance = FileSystemObserver | AppFocusObserver | ClipboardObserver | InputTapObserver


class ObservationDaemon:
    """Manages all observers lifecycle for 24/7 resident observation.

    Responsibilities:
    - Instantiate observers based on config and platform
    - Start/stop each observer independently (one failure doesn't block others)
    - Report per-observer running status
    - Apply/reset RecordingProfile for high-fidelity recording mode
    """

    def __init__(self, bus: "EventBus", config: ObserverConfig | None = None) -> None:
        self._bus = bus
        self._config = config or ObserverConfig()
        self._observers: Dict[str, _ObserverInstance] = {}
        self._active_profile: Optional[RecordingProfile] = None
        self._saved_fs_debounce: Optional[int] = None
        self._saved_input_throttle: Optional[int] = None
        self._profile_started_input_tap = False
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

    async def apply_profile(self, profile: RecordingProfile) -> None:
        """Switch to high-fidelity recording mode.

        Tightens observer parameters and optionally starts the input tap
        observer if the profile requests it and the observer exists.
        """
        if self._active_profile is not None:
            return

        self._active_profile = profile
        self._profile_started_input_tap = False

        fs = self._observers.get("fs_watcher")
        if isinstance(fs, FileSystemObserver):
            self._saved_fs_debounce = fs._debounce_ms
            fs.set_debounce_ms(profile.fs_debounce_ms)

        inp = self._observers.get("input_tap")
        if isinstance(inp, InputTapObserver):
            self._saved_input_throttle = inp._throttle_ms
            inp.set_throttle_ms(profile.input_throttle_ms)
            if profile.enable_input_tap and not inp.running:
                try:
                    await inp.start()
                    self._profile_started_input_tap = True
                except Exception:
                    logger.debug("input_tap auto-start failed during profile apply", exc_info=True)
        elif profile.enable_input_tap and "input_tap" not in self._observers:
            inp = InputTapObserver(bus=self._bus, throttle_ms=profile.input_throttle_ms)
            self._observers["input_tap"] = inp
            try:
                await inp.start()
                self._profile_started_input_tap = True
            except Exception:
                logger.debug("input_tap creation failed during profile apply", exc_info=True)

        logger.info(
            "recording_profile.applied fs_debounce=%dms input_throttle=%dms",
            profile.fs_debounce_ms, profile.input_throttle_ms,
        )

    async def reset_profile(self) -> None:
        """Restore observers to idle-mode parameters."""
        if self._active_profile is None:
            return

        fs = self._observers.get("fs_watcher")
        if isinstance(fs, FileSystemObserver) and self._saved_fs_debounce is not None:
            fs.set_debounce_ms(self._saved_fs_debounce)
            self._saved_fs_debounce = None

        inp = self._observers.get("input_tap")
        if isinstance(inp, InputTapObserver) and self._saved_input_throttle is not None:
            inp.set_throttle_ms(self._saved_input_throttle)
            self._saved_input_throttle = None

        if self._profile_started_input_tap and isinstance(inp, InputTapObserver):
            try:
                await inp.stop()
            except Exception:
                logger.debug("input_tap stop failed during profile reset", exc_info=True)
            self._profile_started_input_tap = False

        logger.info("recording_profile.reset")
        self._active_profile = None
