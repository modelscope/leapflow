# Copyright (c) Alibaba, Inc. and its affiliates.
"""Desktop semantic tools plugin — exposes SemanticAdapter tools to the unified tool system.

Landing C: this plugin is the single registration site for semantic desktop
tools (the role the retired ToolBridge used to play). The plugin is discovered
at boot (contributes nothing until bound), and activates once
`bind_runtime(perception=..., execution=...)` is called with both ports. The
engine queries this plugin for schemas and handlers dynamically.

Architecture:
    ToolPluginRegistry discovers DesktopSemanticPlugin at boot
    → cli/context.py calls registry.bind_runtime(perception=P, execution=E)
    → plugin creates SemanticAdapter internally
    → engine queries plugin.get_semantic_schemas() / get_semantic_handlers()
    → SEMANTIC_TOOL_NAMES appear in the unified catalog only when active

The same entry list also feeds ``build_execution_toolset`` in
``skills.tool_executor``, so the bounded ReAct skill executor exposes exactly
the same semantic tools (with executor traits) as the unified loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from leapflow.plugins.protocol import ToolMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticToolEntry:
    """One semantic desktop tool: definition, handler, and executor traits.

    ``mutates_state`` / ``counts_as_progress`` drive the skill executor's
    early-stop heuristics; ``describer`` optionally resolves opaque params
    (e.g. element_index) into a human-readable target for the policy gate.
    """

    name: str
    description: str
    parameters: Dict[str, str]
    handler: Any
    mutates_state: bool = False
    counts_as_progress: Optional[bool] = None
    describer: Any = None

    def to_openai_schema(self) -> Optional[Dict[str, Any]]:
        """Convert this entry into an OpenAI function schema with x_leapflow metadata.

        Delegates to the shared converter in ``skills.semantic_schema`` so the
        unified-loop plugin and the bounded skill executor always produce
        identical schemas (single source of truth for registration-style
        parameter strings → JSON Schema). Returns None when the tool is
        outside the disclosable semantic tool set.
        """
        from leapflow.skills.semantic_schema import semantic_tool_to_openai

        return semantic_tool_to_openai(self)


class DesktopSemanticPlugin:
    """Dynamic provider for desktop semantic tools (observe_ui, click, etc.).

    Unlike static plugins, the tool set is only available when both perception
    and execution ports are bound via bind_runtime(). This mirrors the previous
    behavior where semantic tools were only registered when the perception
    port was present.
    """

    def __init__(self) -> None:
        self._adapter: Optional[Any] = None
        self._perception: Optional[Any] = None
        self._execution: Optional[Any] = None
        self._schemas: List[Dict[str, Any]] = []
        self._handlers: Dict[str, Any] = {}
        self._version: int = 0

    @property
    def plugin_id(self) -> str:
        return "desktop_semantic"

    @property
    def category(self) -> str:
        return "desktop"

    @property
    def tools(self) -> list[ToolMetadata]:
        # Desktop tools are dynamic — they don't participate in static registry assembly.
        # Schemas and handlers are provided via get_semantic_schemas/get_semantic_handlers
        # which the engine calls directly.
        return []

    @property
    def dependencies(self) -> list[str]:
        return ["perception", "execution"]

    def bind_runtime(self, **deps: Any) -> None:
        """Receive perception and execution ports; create SemanticAdapter when both available."""
        if "perception" in deps:
            self._perception = deps["perception"]
        if "execution" in deps:
            self._execution = deps["execution"]

        if self._perception is not None and self._execution is not None:
            self._build_adapter()
        else:
            # Either port missing — deactivate so the unified catalog and
            # handler table drop the desktop category entirely.
            self._deactivate()

    def _deactivate(self) -> None:
        if self._adapter is None and not self._schemas and not self._handlers:
            return
        self._adapter = None
        self._schemas = []
        self._handlers = {}
        self._version += 1
        logger.info("Desktop semantic plugin deactivated (perception/execution offline)")

    @property
    def active(self) -> bool:
        """Whether the plugin has an active SemanticAdapter (desktop available)."""
        return self._adapter is not None

    @property
    def version(self) -> int:
        """Monotonically increasing version — changes when adapter is rebuilt."""
        return self._version

    def get_semantic_schemas(self) -> List[Dict[str, Any]]:
        """OpenAI function-calling schemas for active semantic tools.

        Returns empty when desktop is offline (adapter not bound).
        The engine merges these into _unified_tool_catalog dynamically.
        """
        return self._schemas

    def get_semantic_handlers(self) -> Dict[str, Any]:
        """Handler map for active semantic tools.

        Returns empty when desktop is offline. The engine merges these
        into the per-turn handler table.
        """
        return dict(self._handlers)

    def _build_adapter(self) -> None:
        """Construct SemanticAdapter and populate schemas + handlers.

        Builds every artifact in locals first, then commits adapter, schemas,
        handlers, and the version bump together in one final block — a failure
        mid-build leaves the previous state fully intact instead of an active
        plugin with stale handlers and an unchanged version (which would keep
        the engine serving the previous schemas).
        """
        from leapflow.skills.semantic_adapter import SemanticAdapter

        adapter = SemanticAdapter(
            perception=self._perception,
            execution=self._execution,
        )

        # Build OpenAI schemas and handler map via the shared converter.
        schemas: List[Dict[str, Any]] = []
        handlers: Dict[str, Any] = {}
        for entry in build_semantic_tool_entries(adapter):
            schema = entry.to_openai_schema()
            if schema is None:
                continue
            schemas.append(schema)
            handlers[entry.name] = entry.handler
        schemas.sort(key=lambda s: s["function"]["name"])

        self._adapter = adapter
        self._schemas = schemas
        self._handlers = handlers
        self._version += 1
        logger.info(
            "Desktop semantic plugin activated: %d tools available",
            len(handlers),
        )


def build_semantic_tool_entries(adapter: Any) -> List[SemanticToolEntry]:
    """Build the entries for all semantic tools backed by one SemanticAdapter.

    Single source of truth for semantic tool registration: consumed by
    ``DesktopSemanticPlugin`` (unified-loop schemas/handlers) and by
    ``tool_executor.build_execution_toolset`` (bounded ReAct skill executor),
    so both surfaces expose exactly the same tool set and traits.
    """
    return [
        SemanticToolEntry(
            name="list_windows",
            description=(
                "List all top-level windows with pid, window_id, title, and per-window state "
                "(minimized, on-screen). Call this first to pick the pid and window_id that "
                "observe_ui and other window tools require."
            ),
            parameters={},
            handler=adapter.list_windows,
        ),
        SemanticToolEntry(
            name="observe_ui",
            description=(
                "Snapshot one window's actionable UI elements, each tagged with an element_index "
                "for click/right_click/read_text. Re-observe after actions — indices belong to one "
                "snapshot. Requires the window's pid and window_id from list_windows."
            ),
            parameters={
                "pid": "int (required) — target process ID from list_windows",
                "window_id": "int (required) — target window ID from list_windows",
                "query": "string (optional) — case-insensitive filter over roles/labels to shrink large windows",
            },
            handler=adapter.observe_ui,
        ),
        SemanticToolEntry(
            name="click",
            description="Click a UI element by its element_index (from the latest observe_ui snapshot)",
            parameters={"element_index": "int (required) — element_index from observe_ui"},
            handler=adapter.click,
            mutates_state=True,
            describer=adapter.describe_element,
        ),
        SemanticToolEntry(
            name="type_text",
            description="Type text into the currently focused element",
            parameters={"text": "string (required) — text to type"},
            handler=adapter.type_text,
            mutates_state=True,
        ),
        SemanticToolEntry(
            name="shortcut",
            description="Execute a keyboard shortcut",
            parameters={
                "keys": "string (required) — shortcut keys, e.g. 'cmd+c', 'cmd+v', 'enter', 'cmd+t'"
            },
            handler=adapter.shortcut,
            mutates_state=True,
        ),
        SemanticToolEntry(
            name="switch_app",
            description="Switch to an app (launch if needed, activate, verify)",
            parameters={"app_id": "string (required) — target app bundle ID"},
            handler=adapter.switch_app,
            mutates_state=True,
        ),
        SemanticToolEntry(
            name="list_apps",
            description=(
                "List available applications on this system. Use to discover correct bundle_id "
                "before switch_app."
            ),
            parameters={
                "filter": "string (optional) — filter by app name or bundle_id substring",
                "running_only": "boolean (optional, default=false) — only list currently running apps",
            },
            handler=adapter.list_apps,
        ),
        SemanticToolEntry(
            name="open_url",
            description="Open a URL in the default or specified browser",
            parameters={
                "url": "string (required) — URL to open",
                "app_id": "string (optional) — browser bundle ID",
            },
            handler=adapter.open_url,
            mutates_state=True,
        ),
        SemanticToolEntry(
            name="get_clipboard",
            description="Read current clipboard text content",
            parameters={},
            handler=adapter.get_clipboard,
        ),
        SemanticToolEntry(
            name="set_clipboard",
            description="Write text to the clipboard",
            parameters={"text": "string (required) — text to place on clipboard"},
            handler=adapter.set_clipboard,
            mutates_state=True,
        ),
        SemanticToolEntry(
            name="read_text",
            description="Read the text content of a specific UI element from the latest snapshot",
            parameters={"element_index": "int (required) — element_index from observe_ui"},
            handler=adapter.read_text,
        ),
        SemanticToolEntry(
            name="wait",
            description="Wait for a specified duration before continuing",
            parameters={"seconds": "number (required) — seconds to wait (0.1-30)"},
            handler=adapter.wait,
            mutates_state=True,
            counts_as_progress=False,
        ),
        SemanticToolEntry(
            name="wait_until",
            description=(
                "Wait until a UI condition is met (polls UI tree). Returns elements when found "
                "or on timeout."
            ),
            parameters={
                "condition": "string (required) — what to wait for (e.g. 'Send button', '发送')",
                "pid": "int (optional) — window's process ID, default = last observed window",
                "window_id": "int (optional) — window ID, default = last observed window",
                "timeout": "number (optional, default=30) — max seconds to wait",
                "poll_interval": "number (optional, default=2) — seconds between polls",
            },
            handler=adapter.wait_until,
            mutates_state=True,
            counts_as_progress=False,
        ),
        SemanticToolEntry(
            name="wait_until_stable",
            description="Wait until the UI stops changing (element set stabilizes across polls).",
            parameters={
                "timeout": "number (optional, default=30) — max seconds to wait",
                "poll_interval": "number (optional, default=2) — seconds between polls",
                "pid": "int (optional) — window's process ID, default = last observed window",
                "window_id": "int (optional) — window ID, default = last observed window",
            },
            handler=adapter.wait_until_stable,
            mutates_state=True,
            counts_as_progress=False,
        ),
        SemanticToolEntry(
            name="scroll",
            description=(
                "Scroll a scrollable area of a window. Omit element_index to scroll the window's "
                "focused/page scroller; pass one to scroll an exact element from the latest snapshot."
            ),
            parameters={
                "element_index": "int (optional) — scroll target from observe_ui, omit for focused scroller",
                "direction": "string (optional, default='down') — up/down/left/right",
                "amount": "number (optional, default=3) — scroll units (1-20)",
                "pid": "int (optional) — window's process ID, default = last observed window",
                "window_id": "int (optional) — window ID, default = last observed window",
            },
            handler=adapter.scroll,
            mutates_state=True,
        ),
        SemanticToolEntry(
            name="select_text",
            description="Select all text in a UI element (focus + select-all, for subsequent copy)",
            parameters={
                "element_index": "int (required) — element containing text, from observe_ui"
            },
            handler=adapter.select_text,
            mutates_state=True,
        ),
        SemanticToolEntry(
            name="right_click",
            description="Right-click a UI element to open its context menu. Returns visible menu items.",
            parameters={
                "element_index": "int (required) — element to right-click, from observe_ui"
            },
            handler=adapter.right_click,
            mutates_state=True,
            describer=adapter.describe_element,
        ),
        SemanticToolEntry(
            name="screenshot",
            description=(
                "Capture a screenshot for visual verification. With pid + window_id captures that "
                "window (works across all displays); defaults to the last observed window, or the "
                "full desktop when no window has been observed."
            ),
            parameters={
                "pid": "int (optional) — window's process ID from list_windows",
                "window_id": "int (optional) — window ID from list_windows",
            },
            handler=adapter.screenshot,
        ),
    ]


# Module-level instance for plugin discovery
plugin = DesktopSemanticPlugin()
