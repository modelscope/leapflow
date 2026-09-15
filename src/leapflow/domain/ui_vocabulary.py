# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared UI vocabulary — ActionType ↔ tool name mappings.

This module connects the Recording vocabulary (ActionType enum values)
with the Execution vocabulary (tool names registered by the
``plugins.tool_plugins.desktop_semantic`` plugin for agent dispatch, and by the
``skills.tool_executor`` ExecutionToolset for the bounded skill executor),
keeping learn→run semantic coherence.

Role classification tables were retired with the tree summarizer: the
driver's get_window_state already returns the filtered, actionable-only
element list, so no execution-side role filtering remains.
"""

from __future__ import annotations

from typing import Dict


# ═══════════════════════════════════════════════════════════════════════════
# ActionType ↔ Execution tool name mapping
#
# This bidirectional mapping connects the Recording vocabulary
# (ActionType enum values) with the Execution vocabulary (tool names
# registered by ``plugins.tool_plugins.desktop_semantic`` — the agent dispatch
# surface — and the ``skills.tool_executor`` ExecutionToolset — the bounded
# skill executor). This ensures that a SemanticAction recorded during learn
# can be mechanically translated into a tool call for execution, and vice
# versa.
# ═══════════════════════════════════════════════════════════════════════════

ACTION_TO_TOOL: Dict[str, str] = {
    "ui.click": "click",
    "ui.type": "type_text",
    "ui.shortcut": "shortcut",
    "clipboard.copy": "get_clipboard",
    "app.switch": "switch_app",
    "ui.scroll": "scroll",
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
    "scroll": "ui.scroll",
}


def tool_name_for_action(action_name: str) -> str:
    """Map a Recording-side ActionType value to an Execution tool name."""
    return ACTION_TO_TOOL.get(action_name, "shell")


def action_name_for_tool(tool_name: str) -> str:
    """Map an Execution tool name back to a Recording-side action name."""
    return TOOL_TO_ACTION.get(tool_name, "unknown")
