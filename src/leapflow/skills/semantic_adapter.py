"""Semantic Adapter — translation layer between LLM tools and platform ports.

This is the execution-side counterpart to the Recording pipeline's
EventNormalizer + ActionAbstractor. Where those translate raw OS signals
into semantic representations for learning, SemanticAdapter translates
LLM semantic intentions into platform-native operations for execution.

Architecture:
    LLM ToolCall → SemanticAdapter → ExecutionPort / PerceptionPort → RPC → OS

Responsibilities:
    - UI tree summarization (raw AX tree → LLM-friendly element list)
    - Selector resolution (human-readable selector → node_id via cache)
    - Composite operations (switch_app = launch + activate + verify)
    - Input strategy selection (type_text via paste vs keystroke)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shlex
import time
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from leapflow.domain.events import UINode
from leapflow.skills.ui_selector import (
    UISelector,
    find_in_tree,
    parse_selector,
    resolve_selector_string,
)
from leapflow.skills.ui_summarizer import UIElement, UITreeSummarizer, summarize_tree

logger = logging.getLogger(__name__)


@runtime_checkable
class PerceptionPort(Protocol):
    async def read_ui_tree(self, app_id: Optional[str] = None) -> UINode: ...
    async def get_clipboard(self) -> Dict[str, Any]: ...
    async def capture_screenshot(self, region: str = "", app_id: str = "") -> Dict[str, Any]: ...


@runtime_checkable
class ExecutionPort(Protocol):
    async def perform_ui_action(
        self, node_id: str, action: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]: ...
    async def launch_app(self, app_id: str) -> Dict[str, Any]: ...
    async def exec_shell(self, command: str) -> Dict[str, Any]: ...
    async def set_clipboard(self, text: str) -> Dict[str, Any]: ...
    async def type_text(self, text: str, method: str = "paste") -> Dict[str, Any]: ...
    async def send_shortcut(self, keys: str) -> Dict[str, Any]: ...
    async def activate_app(self, app_id: str) -> Dict[str, Any]: ...
    async def list_apps(self, filter: str = "", running_only: bool = False) -> Dict[str, Any]: ...
    async def scroll(self, node_id: str, delta_x: int, delta_y: int) -> Dict[str, Any]: ...


class SemanticAdapter:
    """Translates LLM semantic tool calls into platform port operations.

    Manages a short-lived selector→node_id cache that's populated on each
    observe_ui() call and invalidated after a configurable TTL.
    """

    def __init__(
        self,
        perception: PerceptionPort,
        execution: ExecutionPort,
        *,
        cache_ttl: float = 5.0,
        settle_delay: float = 0.3,
        summarizer: Optional[UITreeSummarizer] = None,
    ) -> None:
        self._perception = perception
        self._execution = execution
        self._cache_ttl = cache_ttl
        self._settle_delay = settle_delay
        self._summarizer = summarizer or UITreeSummarizer()
        self._selector_cache: Dict[str, str] = {}
        self._last_tree: Optional[UINode] = None
        self._cache_ts: float = 0.0

    # ═══════════════════════════════════════════════════════════════════
    # Perception tools (read-only)
    # ═══════════════════════════════════════════════════════════════════

    async def observe_ui(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Observe the current UI state, returning a summarized element list."""
        app_id = params.get("app_id", "") or None
        focus_area = params.get("focus_area", "")

        tree = await self._perception.read_ui_tree(app_id)
        elements = self._summarizer.summarize(tree, focus_area=focus_area)

        self._update_cache(elements, tree)

        serialized = [
            {
                "selector": el.selector,
                "role": el.role,
                "label": el.label,
                **({"value": el.value} if el.value else {}),
                **({"actions": el.actions} if el.actions else {}),
                **({"path": el.path} if el.path else {}),
            }
            for el in elements
        ]

        return {
            "ok": True,
            "element_count": len(serialized),
            "elements": serialized,
        }

    async def get_clipboard(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read current clipboard text."""
        result = await self._perception.get_clipboard()
        return {"ok": True, "text": result.get("text", ""), **result}

    async def read_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read text content of a specific element."""
        selector_str = params.get("selector", "")
        node_id = await self._resolve_selector(selector_str)
        if not node_id:
            return {"ok": False, "error": f"element_not_found: {selector_str}"}

        if self._last_tree:
            node = self._find_node_by_id(self._last_tree, node_id)
            if node:
                return {"ok": True, "text": node.value, "label": node.label}

        return {"ok": True, "text": "", "note": "value not available"}

    # ═══════════════════════════════════════════════════════════════════
    # Execution tools (state-changing)
    # ═══════════════════════════════════════════════════════════════════

    _CLICK_ACTIONS = ("AXPress", "AXShowDefaultUI")

    async def click(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Click a UI element with fallback actions, returning post-action state hint."""
        selector_str = params.get("selector", "")
        node_id = await self._resolve_selector(selector_str)
        if not node_id:
            return {"ok": False, "error": f"element_not_found: {selector_str}"}

        result: Dict[str, Any] = {"ok": False}
        for action in self._CLICK_ACTIONS:
            result = await self._execution.perform_ui_action(node_id, action)
            if result.get("ok"):
                break

        self._invalidate_cache()

        if not result.get("ok"):
            node = self._find_node_by_id(self._last_tree, node_id) if self._last_tree else None
            error_info: Dict[str, Any] = {"ok": False, "error": f"click_failed: {selector_str}"}
            if node and node.frame:
                error_info["frame"] = node.frame
            error_info["suggestion"] = (
                "click failed — try keyboard interaction (shortcut, type_text) "
                "or a different selector"
            )
            return error_info

        await asyncio.sleep(self._settle_delay)
        try:
            tree = await self._perception.read_ui_tree(None)
            elements = self._summarizer.summarize(tree)
            self._update_cache(elements, tree)
            state_hint = [
                f"{el.role}[{el.label}]" for el in elements[:10] if el.label
            ]
            return {
                **result,
                "selector": selector_str,
                "state_after": state_hint,
                "element_count": len(elements),
            }
        except Exception:
            return {**result, "selector": selector_str}

    async def type_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Type text into the currently focused element."""
        text = params.get("text", "")
        method = params.get("method", "paste")
        if not text:
            return {"ok": False, "error": "empty text"}
        result = await self._execution.type_text(text, method)
        self._invalidate_cache()
        return result

    async def shortcut(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a keyboard shortcut."""
        keys = params.get("keys", "")
        if not keys:
            return {"ok": False, "error": "no keys specified"}
        result = await self._execution.send_shortcut(keys)
        self._invalidate_cache()
        return result

    async def set_clipboard(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set clipboard text content."""
        text = params.get("text", "")
        return await self._execution.set_clipboard(text)

    async def switch_app(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Switch to an application and verify it's in foreground."""
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

        await self._execution.activate_app(app_id)

        for _ in range(10):
            await asyncio.sleep(0.5)
            try:
                tree = await self._perception.read_ui_tree(app_id)
                if tree and (tree.children or tree.label):
                    self._invalidate_cache()
                    return {"ok": True, "app_id": app_id, "window_title": tree.label}
            except Exception:
                continue

        self._invalidate_cache()
        return {"ok": False, "error": "app_not_ready", "app_id": app_id}

    async def list_apps(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available applications on the system."""
        filter_str = params.get("filter", "")
        running_only = params.get("running_only", False)
        return await self._execution.list_apps(filter_str, running_only)

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
        """Poll UI until a condition appears met, or timeout.

        Checks if the condition string matches any element label or selector.
        Returns the current UI snapshot so the LLM can verify the condition.
        """
        condition = params.get("condition", "")
        app_id = params.get("app_id", "") or None
        timeout = min(max(float(params.get("timeout", 30)), 1.0), 180.0)
        poll_interval = min(max(float(params.get("poll_interval", 2)), 0.5), 10.0)

        if not condition:
            return {"ok": False, "error": "condition required"}

        condition_lower = condition.lower()
        elapsed = 0.0
        serialized: List[Dict[str, Any]] = []

        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            tree = await self._perception.read_ui_tree(app_id)
            elements = self._summarizer.summarize(tree)
            self._update_cache(elements, tree)

            serialized = [
                {"selector": el.selector, "role": el.role, "label": el.label}
                for el in elements[:20]
            ]

            found = any(
                condition_lower in el.label.lower()
                or condition_lower in el.selector.lower()
                for el in elements
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

    _SCROLL_DELTAS = {
        "down": (0, -1), "up": (0, 1),
        "left": (1, 0), "right": (-1, 0),
    }

    async def scroll(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Scroll a scrollable area in the given direction."""
        selector_str = params.get("selector", "")
        direction = params.get("direction", "down")
        amount = min(max(int(params.get("amount", 3)), 1), 20)

        if direction not in self._SCROLL_DELTAS:
            return {"ok": False, "error": f"invalid_direction: {direction} (use up/down/left/right)"}

        node_id = await self._resolve_selector(selector_str) if selector_str else None
        if not node_id:
            if not self._last_tree:
                tree = await self._perception.read_ui_tree(None)
                elements = self._summarizer.summarize(tree)
                self._update_cache(elements, tree)
            node_id = self._find_first_scrollable(self._last_tree) if self._last_tree else None

        unit_dx, unit_dy = self._SCROLL_DELTAS[direction]
        dx, dy = unit_dx * amount, unit_dy * amount

        await self._execution.scroll(node_id or "", dx, dy)
        self._invalidate_cache()
        await asyncio.sleep(self._settle_delay)

        tree = await self._perception.read_ui_tree(None)
        elements = self._summarizer.summarize(tree)
        self._update_cache(elements, tree)
        return {"ok": True, "direction": direction, "amount": amount, "element_count": len(elements)}

    async def select_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Select text in a UI element for subsequent copy."""
        selector_str = params.get("selector", "")
        method = params.get("method", "all")

        node_id = await self._resolve_selector(selector_str)
        if not node_id:
            return {"ok": False, "error": f"element_not_found: {selector_str}"}

        await self._execution.perform_ui_action(node_id, "AXPress")
        await asyncio.sleep(self._settle_delay)

        if method == "all":
            await self._execution.send_shortcut("cmd+a")
        else:
            await self._execution.perform_ui_action(node_id, "AXPress")

        self._invalidate_cache()
        return {"ok": True, "selector": selector_str, "method": method}

    async def right_click(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Right-click a UI element to open its context menu."""
        selector_str = params.get("selector", "")
        node_id = await self._resolve_selector(selector_str)
        if not node_id:
            return {"ok": False, "error": f"element_not_found: {selector_str}"}

        result = await self._execution.perform_ui_action(node_id, "AXShowMenu")
        self._invalidate_cache()
        await asyncio.sleep(self._settle_delay)

        tree = await self._perception.read_ui_tree(None)
        elements = self._summarizer.summarize(tree)
        self._update_cache(elements, tree)
        menu_items = [el for el in elements if "Menu" in el.role]

        return {
            **result,
            "selector": selector_str,
            "menu_items": [
                {"selector": el.selector, "label": el.label} for el in menu_items
            ],
        }

    async def screenshot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Capture a screenshot for visual state verification."""
        region = params.get("region", "")
        app_id = params.get("app_id", "")
        result = await self._perception.capture_screenshot(region=region, app_id=app_id)
        return {"ok": True, "captured": True, **result}

    async def wait_until_stable(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Wait until the UI stops changing (element digest stabilizes)."""
        timeout = min(max(float(params.get("timeout", 30)), 1.0), 180.0)
        poll_interval = min(max(float(params.get("poll_interval", 2)), 0.5), 10.0)
        app_id = params.get("app_id", "") or None

        elapsed = 0.0
        prev_digest = ""
        stable_count = 0

        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            tree = await self._perception.read_ui_tree(app_id)
            elements = self._summarizer.summarize(tree)
            digest = _elements_digest(elements)

            if digest == prev_digest:
                stable_count += 1
                if stable_count >= 2:
                    self._update_cache(elements, tree)
                    return {"ok": True, "stable": True, "elapsed": round(elapsed, 1)}
            else:
                stable_count = 0
                prev_digest = digest

        return {"ok": True, "stable": False, "elapsed": round(elapsed, 1), "timeout": True}

    # ═══════════════════════════════════════════════════════════════════
    # Cache management
    # ═══════════════════════════════════════════════════════════════════

    def _update_cache(self, elements: List[UIElement], tree: UINode) -> None:
        """Rebuild selector→node_id cache from fresh observation."""
        self._selector_cache.clear()
        for el in elements:
            if el.node_id:
                self._selector_cache[el.selector] = el.node_id
        self._last_tree = tree
        self._cache_ts = time.monotonic()

    def _invalidate_cache(self) -> None:
        """Mark cache as stale (actions may have changed UI state)."""
        self._cache_ts = 0.0

    @property
    def _cache_valid(self) -> bool:
        return (time.monotonic() - self._cache_ts) < self._cache_ttl

    async def _resolve_selector(self, selector_str: str) -> Optional[str]:
        """Resolve a selector string to a node_id, refreshing cache if needed."""
        if self._cache_valid and selector_str in self._selector_cache:
            return self._selector_cache[selector_str]

        if not self._cache_valid:
            tree = await self._perception.read_ui_tree(None)
            elements = self._summarizer.summarize(tree)
            self._update_cache(elements, tree)

        if selector_str in self._selector_cache:
            return self._selector_cache[selector_str]

        if self._last_tree:
            node_id = resolve_selector_string(self._last_tree, selector_str)
            if node_id:
                self._selector_cache[selector_str] = node_id
                return node_id

        return None

    def _find_node_by_id(self, node: UINode, target_id: str) -> Optional[UINode]:
        """DFS search for a node by id."""
        if node.node_id == target_id:
            return node
        for child in node.children:
            found = self._find_node_by_id(child, target_id)
            if found:
                return found
        return None

    def _find_first_scrollable(self, node: UINode) -> Optional[str]:
        """DFS for the first AXScrollArea node_id."""
        if node.role == "AXScrollArea":
            return node.node_id
        for child in node.children:
            found = self._find_first_scrollable(child)
            if found:
                return found
        return None


def _elements_digest(elements: List[UIElement]) -> str:
    """Compute a fast content digest of the element list for change detection."""
    content = "|".join(f"{el.selector}:{el.label}" for el in elements)
    return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:12]
