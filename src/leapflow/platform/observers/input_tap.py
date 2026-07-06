"""Keyboard and mouse input event observer (cross-platform).

Captures low-level input events and publishes UI_ACTION events.

Backends:
- macOS: Quartz CGEventTap (requires Accessibility permission)
- Linux: pynput (X11/Wayland)
- Windows: pynput

Events are throttled per action type within a configurable window (default 50ms)
to prevent high-frequency keyboard/scroll event flooding.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

from leapflow.platform.protocol import EventTypes

if TYPE_CHECKING:
    from leapflow.platform.event_bus import EventBus

logger = logging.getLogger(__name__)


class InputTapObserver:
    """Observes keyboard/mouse input and publishes UI_ACTION events.

    Requires elevated permissions on macOS (Accessibility).
    Uses pynput as cross-platform fallback.
    Throttles same-type events within configurable window.
    """

    def __init__(
        self,
        bus: "EventBus",
        throttle_ms: int = 50,
    ) -> None:
        self._bus = bus
        self._throttle_ms = throttle_ms
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._listeners: list[Any] = []
        # macOS Quartz: references for proper teardown
        self._quartz_tap: Any = None
        self._quartz_run_loop: Any = None
        # Throttle state: action_type -> last_emit_monotonic
        self._last_emit: Dict[str, float] = {}

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start capturing input. Idempotent."""
        if self._running:
            return

        self._loop = asyncio.get_running_loop()

        if sys.platform == "darwin":
            started = self._start_macos()
        else:
            started = self._start_pynput()

        if started:
            self._running = True
            logger.info("InputTapObserver started (platform=%s)", sys.platform)
        else:
            logger.warning("InputTapObserver failed to start")

    async def stop(self) -> None:
        """Stop capturing. Idempotent."""
        if not self._running:
            return

        # macOS: disable CGEventTap and stop CFRunLoop
        if self._quartz_tap is not None:
            try:
                from Quartz import CGEventTapEnable, CFRunLoopStop
                CGEventTapEnable(self._quartz_tap, False)
                if self._quartz_run_loop is not None:
                    CFRunLoopStop(self._quartz_run_loop)
            except Exception:
                logger.debug("Error stopping Quartz event tap", exc_info=True)
            self._quartz_tap = None
            self._quartz_run_loop = None

        for listener in self._listeners:
            try:
                if hasattr(listener, "stop"):
                    listener.stop()
            except Exception:
                logger.debug("Error stopping input listener", exc_info=True)

        self._listeners.clear()
        self._running = False
        self._last_emit.clear()
        logger.info("InputTapObserver stopped")

    def _should_throttle(self, action: str) -> bool:
        """Return True if event should be suppressed (within throttle window)."""
        now = time.monotonic()
        last = self._last_emit.get(action, 0.0)
        if (now - last) * 1000 < self._throttle_ms:
            return True
        self._last_emit[action] = now
        return False

    def _schedule_emit(self, payload: Dict[str, Any]) -> None:
        """Schedule async event emission from listener thread."""
        if self._loop is None or self._loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self._emit(payload), self._loop)

    async def _emit(self, payload: Dict[str, Any]) -> None:
        """Emit UI_ACTION event through EventBus."""
        try:
            await self._bus.handle_event(EventTypes.UI_ACTION, payload)
        except Exception:
            logger.error("Failed to emit UI_ACTION event", exc_info=True)

    # ── macOS: Quartz CGEventTap ──

    def _start_macos(self) -> bool:
        """Start macOS input tap using Quartz."""
        try:
            import Quartz
            from Quartz import (
                CGEventTapCreate,
                kCGSessionEventTap,
                kCGHeadInsertEventTap,
                kCGEventTapOptionListenOnly,
                CGEventMaskBit,
                kCGEventLeftMouseDown,
                kCGEventRightMouseDown,
                kCGEventKeyDown,
                kCGEventScrollWheel,
                CGEventTapEnable,
                CFMachPortCreateRunLoopSource,
                CFRunLoopAddSource,
                CFRunLoopGetCurrent,
                kCFRunLoopCommonModes,
            )
            import threading

            mask = (
                CGEventMaskBit(kCGEventLeftMouseDown)
                | CGEventMaskBit(kCGEventRightMouseDown)
                | CGEventMaskBit(kCGEventKeyDown)
                | CGEventMaskBit(kCGEventScrollWheel)
            )

            def callback(proxy: Any, event_type: int, event: Any, refcon: Any) -> Any:
                self._handle_quartz_event(event_type, event)
                return event

            tap = CGEventTapCreate(
                kCGSessionEventTap,
                kCGHeadInsertEventTap,
                kCGEventTapOptionListenOnly,
                mask,
                callback,
                None,
            )

            if tap is None:
                logger.warning(
                    "CGEventTap creation failed — Accessibility permission required"
                )
                return False

            source = CFMachPortCreateRunLoopSource(None, tap, 0)

            def run_loop_thread() -> None:
                run_loop = CFRunLoopGetCurrent()
                self._quartz_run_loop = run_loop
                CFRunLoopAddSource(run_loop, source, kCFRunLoopCommonModes)
                CGEventTapEnable(tap, True)
                Quartz.CFRunLoopRun()

            self._quartz_tap = tap
            thread = threading.Thread(target=run_loop_thread, daemon=True)
            thread.start()
            self._listeners.append(tap)
            return True

        except ImportError:
            logger.debug("Quartz not available, falling back to pynput")
            return self._start_pynput()
        except Exception:
            logger.debug("macOS input tap failed", exc_info=True)
            return self._start_pynput()

    def _handle_quartz_event(self, event_type: int, event: Any) -> None:
        """Process a Quartz CGEvent."""
        try:
            from Quartz import (
                kCGEventLeftMouseDown,
                kCGEventRightMouseDown,
                kCGEventKeyDown,
                kCGEventScrollWheel,
                CGEventGetLocation,
                CGEventGetIntegerValueField,
                kCGKeyboardEventKeycode,
                kCGScrollWheelEventDeltaAxis1,
            )

            if event_type in (kCGEventLeftMouseDown, kCGEventRightMouseDown):
                if self._should_throttle("click"):
                    return
                loc = CGEventGetLocation(event)
                payload: Dict[str, Any] = {
                    "action": "click",
                    "app_bundle_id": "",
                    "mouse_x": int(loc.x),
                    "mouse_y": int(loc.y),
                    "timestamp": time.time(),
                }
                self._schedule_emit(payload)

            elif event_type == kCGEventKeyDown:
                if self._should_throttle("key"):
                    return
                keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
                payload = {
                    "action": "type",
                    "app_bundle_id": "",
                    "key_code": int(keycode),
                    "timestamp": time.time(),
                }
                self._schedule_emit(payload)

            elif event_type == kCGEventScrollWheel:
                if self._should_throttle("scroll"):
                    return
                delta_y = CGEventGetIntegerValueField(
                    event, kCGScrollWheelEventDeltaAxis1
                )
                loc = CGEventGetLocation(event)
                payload = {
                    "action": "scroll",
                    "app_bundle_id": "",
                    "delta_y": int(delta_y),
                    "mouse_x": int(loc.x),
                    "mouse_y": int(loc.y),
                    "timestamp": time.time(),
                }
                self._schedule_emit(payload)

        except Exception:
            logger.debug("Quartz event processing error", exc_info=True)

    # ── Cross-platform: pynput ──

    def _start_pynput(self) -> bool:
        """Start using pynput (works on Linux/Windows, fallback on macOS)."""
        try:
            from pynput import mouse, keyboard
        except ImportError:
            logger.warning(
                "pynput not installed — InputTapObserver disabled. "
                "Install with: pip install pynput"
            )
            return False

        try:
            # Mouse listener
            mouse_listener = mouse.Listener(
                on_click=self._on_pynput_click,
                on_scroll=self._on_pynput_scroll,
            )
            mouse_listener.start()
            self._listeners.append(mouse_listener)

            # Keyboard listener
            key_listener = keyboard.Listener(
                on_press=self._on_pynput_key,
            )
            key_listener.start()
            self._listeners.append(key_listener)

            return True
        except Exception:
            logger.warning("pynput listener start failed", exc_info=True)
            return False

    def _on_pynput_click(
        self, x: int, y: int, button: Any, pressed: bool
    ) -> None:
        """Handle pynput mouse click."""
        if not pressed:
            return
        if self._should_throttle("click"):
            return

        payload: Dict[str, Any] = {
            "action": "click",
            "app_bundle_id": "",
            "mouse_x": int(x),
            "mouse_y": int(y),
            "timestamp": time.time(),
        }
        self._schedule_emit(payload)

    def _on_pynput_scroll(
        self, x: int, y: int, dx: int, dy: int
    ) -> None:
        """Handle pynput scroll."""
        if self._should_throttle("scroll"):
            return

        payload: Dict[str, Any] = {
            "action": "scroll",
            "app_bundle_id": "",
            "delta_x": dx,
            "delta_y": dy,
            "mouse_x": int(x),
            "mouse_y": int(y),
            "timestamp": time.time(),
        }
        self._schedule_emit(payload)

    def _on_pynput_key(self, key: Any) -> None:
        """Handle pynput key press."""
        if self._should_throttle("key"):
            return

        # Extract key information
        key_str = ""
        key_code = 0
        try:
            if hasattr(key, "char") and key.char:
                key_str = key.char
            elif hasattr(key, "vk"):
                key_code = key.vk
            key_str = key_str or str(key)
        except Exception:
            key_str = str(key)

        payload: Dict[str, Any] = {
            "action": "type",
            "app_bundle_id": "",
            "key_code": key_code,
            "char": key_str,
            "timestamp": time.time(),
        }
        self._schedule_emit(payload)
