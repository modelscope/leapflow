"""Shared UI vocabulary — role classifications and action type mappings.

This module is the single source of truth for UI element semantics used by
both the Recording pipeline (EventNormalizer → ActionAbstractor) and the
Execution pipeline (SemanticAdapter → UITreeSummarizer). Keeping these
constants in one place ensures learn→run semantic coherence.

Architecture:
    Recording (forward):  raw AX role → classify → filter/weight in analysis
    Execution (reverse):  AX role → classify → filter/prioritize in summarizer
    Both share the same classification, preventing vocabulary drift.
"""

from __future__ import annotations

from typing import Dict, FrozenSet


# ═══════════════════════════════════════════════════════════════════════════
# Role classifications — what kind of UI element is this?
# ═══════════════════════════════════════════════════════════════════════════

INTERACTIVE_ROLES: FrozenSet[str] = frozenset({
    "AXButton", "AXTextField", "AXTextArea", "AXLink",
    "AXMenuItem", "AXCheckBox", "AXRadioButton", "AXPopUpButton",
    "AXTab", "AXSlider", "AXComboBox", "AXDisclosureTriangle",
    "AXIncrementor", "AXColorWell", "AXMenuButton",
})

LAYOUT_ROLES: FrozenSet[str] = frozenset({
    "AXGroup", "AXScrollArea", "AXSplitGroup", "AXLayoutArea",
    "AXList", "AXOutline", "AXTable", "AXRow", "AXColumn",
    "AXBrowser", "AXScrollBar", "AXRuler", "AXGrowArea",
    "AXMatte", "AXSplitter",
})

STRUCTURAL_ROLES: FrozenSet[str] = frozenset({
    "AXWindow", "AXSheet", "AXDialog", "AXToolbar",
    "AXMenuBar", "AXMenu",
})


# ═══════════════════════════════════════════════════════════════════════════
# ActionType ↔ Execution tool name mapping
#
# This bidirectional mapping connects the Recording vocabulary
# (ActionType enum values) with the Execution vocabulary (tool names
# registered in bridge_factory). This ensures that a SemanticAction
# recorded during learn can be mechanically translated into a tool call
# for execution, and vice versa.
# ═══════════════════════════════════════════════════════════════════════════

ACTION_TO_TOOL: Dict[str, str] = {
    "ui.click": "click",
    "ui.type": "type_text",
    "ui.shortcut": "shortcut",
    "clipboard.copy": "get_clipboard",
    "app.switch": "switch_app",
    "ui.scroll": "shortcut",
}

TOOL_TO_ACTION: Dict[str, str] = {
    "click": "ui.click",
    "type_text": "ui.type",
    "shortcut": "ui.shortcut",
    "get_clipboard": "clipboard.copy",
    "set_clipboard": "clipboard.copy",
    "switch_app": "app.switch",
    "open_url": "app.switch",
    "observe_ui": "ui.click",
}


def tool_name_for_action(action_name: str) -> str:
    """Map a Recording-side ActionType value to an Execution tool name."""
    return ACTION_TO_TOOL.get(action_name, "shell")


def action_name_for_tool(tool_name: str) -> str:
    """Map an Execution tool name back to a Recording-side action name."""
    return TOOL_TO_ACTION.get(tool_name, "unknown")
