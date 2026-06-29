"""ToolBridge factory — constructs a bridge with semantic tools when perception is available.

This is the integration point where the SemanticAdapter layer is wired in.
When only ExecutionPort is available, falls back to the basic ToolBridge
(file ops + shell + launch_app + ui_action). When PerceptionPort is also
provided, the full semantic tool set is registered (observe_ui, click,
type_text, shortcut, switch_app, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from leapflow.skills.tool_executor import ToolBridge

if TYPE_CHECKING:
    from leapflow.engine.confirmation import IOProvider
    from leapflow.skills.action_policy import PolicyEngine


def build_tool_bridge(
    execution: Any,
    perception: Optional[Any] = None,
    *,
    policy: Optional["PolicyEngine"] = None,
    io: Optional["IOProvider"] = None,
) -> ToolBridge:
    """Construct a ToolBridge, optionally enriched with semantic UI tools.

    Args:
        execution: ExecutionPort implementation.
        perception: Optional PerceptionPort. When provided, enables
                    semantic UI tools (observe_ui, click, type_text, etc.)
                    via the SemanticAdapter translation layer.

    Returns:
        A fully configured ToolBridge ready for ReAct execution.
    """
    bridge = ToolBridge(execution, policy=policy, io=io)

    if perception is None:
        return bridge

    from leapflow.skills.semantic_adapter import SemanticAdapter

    adapter = SemanticAdapter(perception=perception, execution=execution)

    bridge.register(
        "observe_ui",
        "Observe current app UI. Returns a list of interactive elements with selectors.",
        {
            "app_id": "string (optional) — target app bundle ID, empty = frontmost",
            "focus_area": "string (optional) — focus hint (e.g. 'toolbar', 'sidebar')",
        },
        adapter.observe_ui,
    )
    bridge.register(
        "click",
        "Click a UI element by its selector (from observe_ui results)",
        {"selector": "string (required) — element selector, e.g. 'AXButton[label=Send]'"},
        adapter.click,
        mutates_state=True,
    )
    bridge.register(
        "type_text",
        "Type text into the currently focused element",
        {
            "text": "string (required) — text to type",
            "method": "string (optional) — 'paste' (default, best for CJK) or 'keystroke'",
        },
        adapter.type_text,
        mutates_state=True,
    )
    bridge.register(
        "shortcut",
        "Execute a keyboard shortcut",
        {"keys": "string (required) — shortcut keys, e.g. 'cmd+c', 'cmd+v', 'enter', 'cmd+t'"},
        adapter.shortcut,
        mutates_state=True,
    )
    bridge.register(
        "switch_app",
        "Switch to an app (launch if needed, activate, verify)",
        {"app_id": "string (required) — target app bundle ID"},
        adapter.switch_app,
        mutates_state=True,
    )
    bridge.register(
        "list_apps",
        "List available applications on this system. Use to discover correct bundle_id before switch_app.",
        {
            "filter": "string (optional) — filter by app name or bundle_id substring",
            "running_only": "boolean (optional, default=false) — only list currently running apps",
        },
        adapter.list_apps,
    )
    bridge.register(
        "open_url",
        "Open a URL in the default or specified browser",
        {
            "url": "string (required) — URL to open",
            "app_id": "string (optional) — browser bundle ID",
        },
        adapter.open_url,
        mutates_state=True,
    )
    bridge.register(
        "get_clipboard",
        "Read current clipboard text content",
        {},
        adapter.get_clipboard,
    )
    bridge.register(
        "set_clipboard",
        "Write text to the clipboard",
        {"text": "string (required) — text to place on clipboard"},
        adapter.set_clipboard,
        mutates_state=True,
    )
    bridge.register(
        "read_text",
        "Read the text content of a specific UI element",
        {"selector": "string (required) — element selector"},
        adapter.read_text,
    )
    bridge.register(
        "wait",
        "Wait for a specified duration before continuing",
        {"seconds": "number (required) — seconds to wait (0.1-30)"},
        adapter.wait,
        mutates_state=True,
        counts_as_progress=False,
    )
    bridge.register(
        "wait_until",
        "Wait until a UI condition is met (polls UI tree). Returns elements when found or on timeout.",
        {
            "condition": "string (required) — what to wait for (e.g. 'Send button', '发送')",
            "app_id": "string (optional) — app to observe",
            "timeout": "number (optional, default=30) — max seconds to wait",
            "poll_interval": "number (optional, default=2) — seconds between polls",
        },
        adapter.wait_until,
        mutates_state=True,
        counts_as_progress=False,
    )
    bridge.register(
        "wait_until_stable",
        "Wait until the UI stops changing (element set stabilizes across polls).",
        {
            "timeout": "number (optional, default=30) — max seconds to wait",
            "poll_interval": "number (optional, default=2) — seconds between polls",
            "app_id": "string (optional) — app to observe",
        },
        adapter.wait_until_stable,
        mutates_state=True,
        counts_as_progress=False,
    )
    bridge.register(
        "scroll",
        "Scroll a scrollable area. Use after observe_ui if target content is not visible.",
        {
            "selector": "string (optional) — scroll area selector, empty = first scrollable",
            "direction": "string (optional, default='down') — up/down/left/right",
            "amount": "number (optional, default=3) — scroll units (1-20)",
        },
        adapter.scroll,
        mutates_state=True,
    )
    bridge.register(
        "select_text",
        "Select text in a UI element (for subsequent cmd+c copy)",
        {
            "selector": "string (required) — element containing text to select",
            "method": "string (optional, default='all') — 'all' (select all) or 'word' (double-click)",
        },
        adapter.select_text,
        mutates_state=True,
    )
    bridge.register(
        "right_click",
        "Right-click a UI element to open its context menu. Returns visible menu items.",
        {
            "selector": "string (required) — element to right-click",
        },
        adapter.right_click,
        mutates_state=True,
    )
    bridge.register(
        "screenshot",
        "Capture a screenshot for visual verification. "
        "When app_id is provided, captures only that app's window (works across all displays).",
        {
            "app_id": "string (optional) — target app bundle ID for window-level capture",
            "region": "string (optional) — empty for full screen",
        },
        adapter.screenshot,
    )

    return bridge
