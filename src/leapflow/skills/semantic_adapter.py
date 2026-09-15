# Copyright (c) Alibaba, Inc. and its affiliates.
"""Semantic Adapter — translation layer between LLM tools and platform ports.

This is the execution-side counterpart to the Recording pipeline's
EventNormalizer + ActionAbstractor. Where those translate raw OS signals
into semantic representations for learning, SemanticAdapter translates
LLM semantic intentions into platform-native operations for execution.

Architecture:
    LLM ToolCall → SemanticAdapter → ExecutionPort / PerceptionPort → RPC → OS

Addressing model (mirrors cua-driver):
    - list_windows supplies (pid, window_id) window targets.
    - observe_ui snapshots one window; every element carries an
      element_index. The snapshot is superseded by the next observation
      of the same window.
    - Action tools address elements by element_index from the latest
      snapshot; the adapter translates to the element_token the driver
      validates for staleness.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shlex
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from leapflow.domain.events import UIElement, UISnapshot

logger = logging.getLogger(__name__)


def _window_target(params: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Parse a (pid, window_id) target from tool params, or None if absent."""
    pid = params.get("pid")
    window_id = params.get("window_id")
    if pid is None or window_id is None:
        return None
    try:
        return int(pid), int(window_id)
    except (TypeError, ValueError):
        return None


def _launch_window_target(launch_result: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Extract (pid, window_id) from a launch_app response, or None.

    The driver's launch_app returns the launched app's pid and a windows
    array (same record shape as list_windows) precisely so callers can skip
    an extra discovery round-trip.
    """
    pid = launch_result.get("pid")
    windows = launch_result.get("windows")
    if not isinstance(pid, int) or pid <= 0 or not isinstance(windows, list):
        return None
    for record in windows:
        if isinstance(record, dict) and isinstance(record.get("window_id"), int):
            return pid, record["window_id"]
    return None


def _serialize_element(el: UIElement) -> Dict[str, Any]:
    """Compact LLM-facing element record; silent defaults are omitted."""
    record: Dict[str, Any] = {
        "element_index": el.element_index,
        "role": el.role,
    }
    if el.label:
        record["label"] = el.label
    if el.value:
        record["value"] = el.value
    if not el.enabled:
        record["enabled"] = False
    if el.selected is not None:
        record["selected"] = el.selected
    return record


def _snapshot_digest(snapshot: UISnapshot) -> str:
    """Fast content digest of a snapshot's elements for change detection."""
    content = "|".join(
        f"{el.role}:{el.label}:{el.value}" for el in snapshot.elements
    )
    return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:12]


@runtime_checkable
class PerceptionPort(Protocol):
    async def read_window_state(
        self, pid: int, window_id: int, query: str = ""
    ) -> UISnapshot: ...
    async def list_windows(self) -> Dict[str, Any]: ...
    async def get_clipboard(self) -> Dict[str, Any]: ...
    async def capture_screenshot(
        self, pid: Optional[int] = None, window_id: Optional[int] = None
    ) -> Dict[str, Any]: ...


@runtime_checkable
class ExecutionPort(Protocol):
    async def perform_ui_action(
        self, node_id: str, action: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]: ...
    async def launch_app(
        self, app_id: str, urls: Optional[List[str]] = None
    ) -> Dict[str, Any]: ...
    async def exec_shell(self, command: str) -> Dict[str, Any]: ...
    async def set_clipboard(self, text: str) -> Dict[str, Any]: ...
    async def type_text(self, text: str) -> Dict[str, Any]: ...
    async def send_shortcut(self, keys: str) -> Dict[str, Any]: ...
    async def activate_app(
        self, pid: int, window_id: Optional[int] = None
    ) -> Dict[str, Any]: ...
    async def list_apps(self) -> Dict[str, Any]: ...
    async def scroll(
        self,
        node_id: str,
        direction: str,
        amount: int = 3,
        pid: Optional[int] = None,
        window_id: Optional[int] = None,
    ) -> Dict[str, Any]: ...


class SemanticAdapter:
    """Translates LLM semantic tool calls into platform port operations.

    Keeps the latest window snapshot; action tools resolve element_index
    against it and address the driver via element_token, whose staleness
    the driver itself validates.
    """

    def __init__(
        self,
        perception: PerceptionPort,
        execution: ExecutionPort,
        *,
        settle_delay: float = 0.3,
        max_observed_elements: int = 120,
    ) -> None:
        self._perception = perception
        self._execution = execution
        self._settle_delay = settle_delay
        self._max_observed_elements = max_observed_elements
        self._last_snapshot: Optional[UISnapshot] = None

    # ═══════════════════════════════════════════════════════════════════
    # Perception tools (read-only)
    # ═══════════════════════════════════════════════════════════════════

    async def observe_ui(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Snapshot one window's actionable elements (indexed for actions)."""
        target = _window_target(params)
        if target is None:
            return {
                "ok": False,
                "error": "missing_window_target",
                "suggestion": "call list_windows first and pass the target's pid and window_id",
            }
        query = str(params.get("query", "") or "")

        pid, window_id = target
        snapshot = await self._perception.read_window_state(pid, window_id, query)
        self._last_snapshot = snapshot

        elements = snapshot.elements[: self._max_observed_elements]
        result: Dict[str, Any] = {
            "ok": True,
            "pid": pid,
            "window_id": window_id,
            "element_count": len(snapshot.elements),
            "elements": [_serialize_element(el) for el in elements],
        }
        if len(snapshot.elements) > len(elements):
            result["truncated"] = True
            result["suggestion"] = "pass query to filter elements of interest"
        self._attach_coverage(result, snapshot)
        return result

    async def list_windows(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List top-level windows — the source of pid/window_id targets."""
        result = await self._perception.list_windows()
        if isinstance(result, dict):
            return {"ok": True, **result}
        return {"ok": True, "windows": result}

    async def get_clipboard(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read current clipboard text."""
        result = await self._perception.get_clipboard()
        return {"ok": True, "text": result.get("text", ""), **result}

    async def read_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read the text content of an element from the latest snapshot."""
        element, error = self._resolve_element(params)
        if element is None:
            return error
        return {"ok": True, "text": element.value, "label": element.label}

    # ═══════════════════════════════════════════════════════════════════
    # Execution tools (state-changing)
    # ═══════════════════════════════════════════════════════════════════

    async def click(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Click an element by index, returning a post-action state hint."""
        element, error = self._resolve_element(params)
        if element is None:
            return error

        result = await self._execution.perform_ui_action(element.target, "press")

        if not result.get("ok"):
            error_info: Dict[str, Any] = {
                "ok": False,
                "error": f"click_failed: element {element.element_index} ({element.role} {element.label!r})",
            }
            if element.frame:
                error_info["frame"] = element.frame
            error_info["suggestion"] = (
                "click failed — re-observe_ui for a fresh snapshot, or try "
                "keyboard interaction (shortcut, type_text)"
            )
            return error_info

        state = await self._refresh_after_action()
        return {**result, "element_index": element.element_index, **state}

    async def type_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Type text into the currently focused element."""
        text = params.get("text", "")
        if not text:
            return {"ok": False, "error": "empty text"}
        return await self._execution.type_text(text)

    async def shortcut(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a keyboard shortcut."""
        keys = params.get("keys", "")
        if not keys:
            return {"ok": False, "error": "no keys specified"}
        return await self._execution.send_shortcut(keys)

    async def set_clipboard(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set clipboard text content."""
        text = params.get("text", "")
        return await self._execution.set_clipboard(text)

    async def switch_app(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Switch to an application and verify its window is readable."""
        app_id = params.get("app_id", "")
        if not app_id:
            return {"ok": False, "error": "app_id required"}

        launch_result = await self._execution.launch_app(app_id)
        if not launch_result.get("ok"):
            return {
                "ok": False,
                "error": f"launch_failed: app '{app_id}' not found or cannot be launched",
                "app_id": app_id,
                "suggestion": "Use list_apps(filter='...') to discover correct bundle_id",
            }

        target = _launch_window_target(launch_result)
        if target is None:
            return {
                "ok": True,
                "app_id": app_id,
                "verified": False,
                "suggestion": "call list_windows to pick the app's pid/window_id, then observe_ui",
            }

        pid, window_id = target
        await self._execution.activate_app(pid, window_id)

        for _ in range(10):
            await asyncio.sleep(0.5)
            try:
                snapshot = await self._perception.read_window_state(pid, window_id)
            except Exception:
                continue
            if snapshot.elements or not snapshot.degraded:
                self._last_snapshot = snapshot
                return {
                    "ok": True,
                    "app_id": app_id,
                    "pid": pid,
                    "window_id": window_id,
                    "element_count": len(snapshot.elements),
                }

        return {"ok": False, "error": "app_not_ready", "app_id": app_id}

    async def list_apps(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available applications, honoring the declared filter locally.

        The driver's list_apps takes no arguments; the tool's filter and
        running_only params are applied to the returned records here.
        """
        filter_str = str(params.get("filter", "") or "").lower()
        running_only = bool(params.get("running_only", False))
        result = await self._execution.list_apps()
        if not isinstance(result, dict):
            return {"ok": True, "apps": result}
        apps = result.get("apps")
        if not isinstance(apps, list) or (not filter_str and not running_only):
            return result

        def _keep(record: Dict[str, Any]) -> bool:
            if running_only and not record.get("running"):
                pid = record.get("pid")
                if not (isinstance(pid, int) and pid > 0):
                    return False
            if filter_str:
                name = str(record.get("name", "")).lower()
                bundle = str(record.get("bundle_id", "")).lower()
                if filter_str not in name and filter_str not in bundle:
                    return False
            return True

        return {**result, "apps": [r for r in apps if isinstance(r, dict) and _keep(r)]}

    async def open_url(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Open a URL in the default or specified browser."""
        url = params.get("url", "")
        if not url:
            return {"ok": False, "error": "url required"}
        if hasattr(self._execution, "open_url"):
            return await self._execution.open_url(url)
        app_id = params.get("app_id", "")
        cmd = f"open {shlex.quote(url)}"
        if app_id:
            cmd = f"open -a {shlex.quote(app_id)} {shlex.quote(url)}"
        return await self._execution.exec_shell(cmd)

    async def wait(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Wait for a specified duration (seconds)."""
        seconds = min(max(float(params.get("seconds", 1)), 0.1), 30.0)
        await asyncio.sleep(seconds)
        return {"ok": True, "waited": seconds}

    async def wait_until(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Poll the window until an element matching the condition appears."""
        condition = params.get("condition", "")
        target = _window_target(params) or self._current_target()
        timeout = min(max(float(params.get("timeout", 30)), 1.0), 180.0)
        poll_interval = min(max(float(params.get("poll_interval", 2)), 0.5), 10.0)

        if not condition:
            return {"ok": False, "error": "condition required"}
        if target is None:
            return {
                "ok": False,
                "error": "missing_window_target",
                "suggestion": "pass pid and window_id (from list_windows) or call observe_ui first",
            }
        pid, window_id = target

        condition_lower = condition.lower()
        elapsed = 0.0
        serialized: List[Dict[str, Any]] = []

        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            snapshot = await self._perception.read_window_state(pid, window_id)
            self._last_snapshot = snapshot
            serialized = [_serialize_element(el) for el in snapshot.elements[:20]]

            found = any(
                condition_lower in el.label.lower()
                or condition_lower in el.role.lower()
                for el in snapshot.elements
            )
            if found:
                return {
                    "ok": True,
                    "met": True,
                    "elapsed": round(elapsed, 1),
                    "elements": serialized,
                }

        return {
            "ok": True,
            "met": False,
            "elapsed": round(elapsed, 1),
            "elements": serialized,
            "timeout": True,
        }

    # ═══════════════════════════════════════════════════════════════════
    # Extended interaction tools
    # ═══════════════════════════════════════════════════════════════════

    _SCROLL_DIRECTIONS = ("up", "down", "left", "right")

    async def scroll(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Scroll an element (by index) or the target window's focused scroller."""
        direction = params.get("direction", "down")
        amount = min(max(int(params.get("amount", 3)), 1), 20)

        if direction not in self._SCROLL_DIRECTIONS:
            return {"ok": False, "error": f"invalid_direction: {direction} (use up/down/left/right)"}

        # The driver requires pid even on the targetless keystroke path.
        window = _window_target(params) or self._current_target()
        if window is None:
            return {
                "ok": False,
                "error": "missing_window_target",
                "suggestion": "call observe_ui(pid, window_id) first",
            }
        pid, window_id = window

        target = ""
        if params.get("element_index") is not None:
            element, error = self._resolve_element(params)
            if element is None:
                return error
            target = element.target

        await self._execution.scroll(
            target, direction, amount, pid=pid, window_id=window_id
        )
        state = await self._refresh_after_action()
        return {"ok": True, "direction": direction, "amount": amount, **state}

    async def select_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Select text in an element (focus it, then select all)."""
        element, error = self._resolve_element(params)
        if element is None:
            return error

        await self._execution.perform_ui_action(element.target, "press")
        await asyncio.sleep(self._settle_delay)
        await self._execution.send_shortcut("cmd+a")
        return {"ok": True, "element_index": element.element_index}

    async def right_click(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Right-click an element to open its context menu."""
        element, error = self._resolve_element(params)
        if element is None:
            return error

        result = await self._execution.perform_ui_action(element.target, "show_menu")
        await asyncio.sleep(self._settle_delay)

        snapshot = await self._resnapshot()
        if snapshot is None:
            return {**result, "element_index": element.element_index, "menu_items": []}
        menu_items = [
            _serialize_element(el) for el in snapshot.elements if "Menu" in el.role
        ]
        return {
            **result,
            "element_index": element.element_index,
            "menu_items": menu_items,
        }

    async def screenshot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Capture a screenshot for visual state verification.

        With a pid/window_id target (explicit or remembered) captures that
        window; otherwise captures the full desktop.
        """
        target = _window_target(params) or self._current_target()
        if target is None:
            result = await self._perception.capture_screenshot()
        else:
            result = await self._perception.capture_screenshot(
                pid=target[0], window_id=target[1]
            )
        return {"ok": True, "captured": True, **result}

    async def wait_until_stable(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Wait until the window stops changing (element digest stabilizes)."""
        timeout = min(max(float(params.get("timeout", 30)), 1.0), 180.0)
        poll_interval = min(max(float(params.get("poll_interval", 2)), 0.5), 10.0)
        target = _window_target(params) or self._current_target()
        if target is None:
            return {
                "ok": False,
                "error": "missing_window_target",
                "suggestion": "pass pid and window_id (from list_windows) or call observe_ui first",
            }
        pid, window_id = target

        elapsed = 0.0
        prev_digest = ""
        stable_count = 0

        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            snapshot = await self._perception.read_window_state(pid, window_id)
            digest = _snapshot_digest(snapshot)

            if digest == prev_digest:
                stable_count += 1
                if stable_count >= 2:
                    self._last_snapshot = snapshot
                    return {"ok": True, "stable": True, "elapsed": round(elapsed, 1)}
            else:
                stable_count = 0
                prev_digest = digest

        return {"ok": True, "stable": False, "elapsed": round(elapsed, 1), "timeout": True}

    # ═══════════════════════════════════════════════════════════════════
    # Snapshot management
    # ═══════════════════════════════════════════════════════════════════

    def _current_target(self) -> Optional[Tuple[int, int]]:
        if self._last_snapshot is None:
            return None
        return self._last_snapshot.pid, self._last_snapshot.window_id

    def describe_element(self, params: Dict[str, Any]) -> str:
        """Human-readable description of the element a call targets.

        Consumed by the policy gate: element_index params carry no
        semantics, so safety rules (e.g. send-button detection) need the
        resolved role + label.
        """
        if self._last_snapshot is None:
            return ""
        try:
            index = int(params.get("element_index"))
        except (TypeError, ValueError):
            return ""
        element = self._last_snapshot.find(index)
        if element is None:
            return ""
        parts = [element.role]
        if element.label:
            parts.append(element.label)
        if element.value:
            parts.append(f"value={element.value[:40]}")
        return " ".join(parts)

    def _resolve_element(
        self, params: Dict[str, Any]
    ) -> Tuple[Optional[UIElement], Dict[str, Any]]:
        """Resolve params['element_index'] against the latest snapshot.

        Returns (element, {}) on success or (None, structured_error).
        """
        if self._last_snapshot is None:
            return None, {
                "ok": False,
                "error": "no_snapshot",
                "suggestion": "call observe_ui(pid, window_id) first to index elements",
            }
        raw = params.get("element_index")
        try:
            index = int(raw)
        except (TypeError, ValueError):
            return None, {
                "ok": False,
                "error": f"invalid_element_index: {raw!r}",
                "suggestion": "pass the element_index of an element from observe_ui",
            }
        element = self._last_snapshot.find(index)
        if element is None:
            return None, {
                "ok": False,
                "error": f"element_not_found: {index}",
                "suggestion": "the snapshot may be stale — call observe_ui again",
            }
        return element, {}

    async def _resnapshot(self) -> Optional[UISnapshot]:
        """Re-observe the current window; None when no target is known."""
        target = self._current_target()
        if target is None:
            return None
        try:
            snapshot = await self._perception.read_window_state(*target)
        except Exception:
            return None
        self._last_snapshot = snapshot
        return snapshot

    async def _refresh_after_action(self) -> Dict[str, Any]:
        """Settle, re-snapshot, and produce a compact post-action state hint."""
        await asyncio.sleep(self._settle_delay)
        snapshot = await self._resnapshot()
        if snapshot is None:
            return {}
        state_hint = [
            f"{el.role}[{el.label}]" for el in snapshot.elements[:10] if el.label
        ]
        state: Dict[str, Any] = {
            "state_after": state_hint,
            "element_count": len(snapshot.elements),
        }
        self._attach_coverage(state, snapshot)
        return state

    @staticmethod
    def _attach_coverage(result: Dict[str, Any], snapshot: UISnapshot) -> None:
        """Surface the driver's blind-spot statements to the model.

        elements_complete=False or a coverage entry (e.g. browser page
        content not observable in window scope) means the model must not
        conclude an element is absent — it should fall back to screenshot
        pixels or app-appropriate tools.
        """
        if snapshot.degraded:
            result["degraded"] = True
            if snapshot.degraded_reason:
                result["degraded_reason"] = snapshot.degraded_reason
        if not snapshot.elements_complete:
            result["elements_complete"] = False
        if snapshot.coverage:
            result["coverage"] = snapshot.coverage
