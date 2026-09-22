# Copyright (c) Alibaba, Inc. and its affiliates.
"""Memory & Research plugin — memory search/add and research ledger tools.

These tools have late-binding dependencies on engine internals (MemoryManager,
ResearchLedger) injected via bind_runtime().
"""

from __future__ import annotations

from typing import Any, Dict

from leapflow.plugins.protocol import ToolMetadata


def _active_workspace_root() -> str:
    """Return the current turn's workspace root from the tool execution context.

    Memory tools run inside ToolDispatchEngine._execute_tool_scoped, which installs the
    per-turn ToolExecutionContext. Reading it here scopes memory reads and tags
    writes to the active workspace (concurrency-safe via ContextVar).
    """
    try:
        from leapflow.tools.execution_context import current_tool_context

        ctx = current_tool_context()
    except LookupError:
        return ""
    return str(getattr(ctx, "workspace_root", "") or "")


class MemoryResearchPlugin:
    """Agent memory search/add and research-ledger note tools.

    Dependencies are late-bound because MemoryManager and ResearchLedger are
    created by the engine after the tool registry is assembled.
    """

    def __init__(self) -> None:
        self._memory_manager: Any = None
        self._research_ledger: Any = None

    @property
    def plugin_id(self) -> str:
        return "memory_research"

    @property
    def category(self) -> str:
        return "memory"

    @property
    def dependencies(self) -> list[str]:
        return ["memory_manager", "research_ledger"]

    def bind_runtime(self, **deps: Any) -> None:
        if "memory_manager" in deps:
            self._memory_manager = deps["memory_manager"]
        if "research_ledger" in deps:
            self._research_ledger = deps["research_ledger"]

    # ── Handlers ──

    async def _memory_search_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for memory_search tool."""
        if self._memory_manager is None:
            return {"ok": False, "error": "Memory system not initialized"}
        try:
            result = await self._memory_manager.handle_tool_call(
                "memory_search", params, workspace_root=_active_workspace_root()
            )
            return {"ok": True, "result": result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def _memory_add_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for memory_add tool."""
        if self._memory_manager is None:
            return {"ok": False, "error": "Memory system not initialized"}
        content = params.get("content", "")
        if content:
            try:
                from leapflow.security.threat_patterns import scan_for_threats, ThreatScope

                threats = scan_for_threats(content, scope=ThreatScope.STRICT, max_results=3)
                if any(t.severity >= 0.8 for t in threats):
                    import logging

                    logging.getLogger(__name__).warning(
                        "memory_add: threat in content: %s",
                        [t.pattern_name for t in threats],
                    )
            except ImportError:
                pass
        try:
            result = await self._memory_manager.handle_tool_call(
                "memory_add", params, workspace_root=_active_workspace_root()
            )
            return {"ok": True, "result": result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def _research_note_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for research_note tool."""
        if self._research_ledger is None:
            return {"ok": False, "error": "Research ledger not initialized"}
        ok = self._research_ledger.note(params.get("kind", ""), params.get("text", ""))
        if not ok:
            return {
                "ok": False,
                "error": "invalid note: kind must be one of finding|open_question|resolved|decision|next_step and text must be non-empty",
            }
        return {"ok": True, "open_questions": self._research_ledger.open_question_count}

    # ── Tool metadata ──

    @property
    def tools(self) -> list[ToolMetadata]:
        return [
            ToolMetadata(
                name="memory_search",
                description="Search agent memory for relevant past experiences, observations, and facts.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search keywords"},
                        "limit": {"type": "integer", "description": "Max results (default: 10)"},
                    },
                    "required": ["query"],
                },
                handler=self._memory_search_handler,
                x_leapflow={
                    "category": "memory",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("memory.search",),
            ),
            ToolMetadata(
                name="memory_add",
                description="Store a new observation or insight in memory for future reference.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "What to remember"},
                        "kind": {
                            "type": "string",
                            "enum": ["observation", "insight", "fact"],
                            "description": "Memory type (default: observation)",
                        },
                    },
                    "required": ["content"],
                },
                handler=self._memory_add_handler,
                x_leapflow={
                    "category": "write",
                    "risk_level": "medium",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("memory.add",),
            ),
            ToolMetadata(
                name="research_note",
                description=(
                    "Record a compact, structured note about the current task's state so it "
                    "survives context compression on long / multi-step tasks. Use for durable "
                    "findings, open questions still to resolve, decisions / excluded paths, and "
                    "the immediate next step. One concise sentence per note."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": [
                                "finding",
                                "open_question",
                                "resolved",
                                "decision",
                                "next_step",
                            ],
                            "description": "finding | open_question | resolved (closes a matching open question) | decision | next_step",
                        },
                        "text": {"type": "string", "description": "One concise sentence."},
                    },
                    "required": ["kind", "text"],
                },
                handler=self._research_note_handler,
                x_leapflow={
                    "category": "memory",
                    "risk_level": "read_only",
                    "schema_cost": "medium",
                    "requires_approval": False,
                },
                provides_capabilities=("memory.research_note",),
            ),
        ]


# Module-level instance for plugin discovery
plugin = MemoryResearchPlugin()
