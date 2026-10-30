# Copyright (c) Alibaba, Inc. and its affiliates.
"""Orchestration & System plugin — capability expansion, subagent delegation, re-entry scheduling.

These tools have late-binding dependencies on engine internals
(capability catalog provider, subagent manager, re-entry scheduler)
injected via bind_runtime().
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from leapflow.plugins.protocol import ToolMetadata


class OrchestrationPlugin:
    """Orchestration tools: capability discovery, task delegation, re-entry scheduling.

    All dependencies are late-bound because they reference engine components that
    are only available after the tool registry is assembled.
    """

    def __init__(self) -> None:
        self._capability_catalog_provider: Optional[Callable[[], List[Dict[str, Any]]]] = None
        self._subagent_manager: Any = None
        self._reentry_scheduler: Any = None

    @property
    def plugin_id(self) -> str:
        return "orchestration"

    @property
    def category(self) -> str:
        return "system"

    @property
    def dependencies(self) -> list[str]:
        return ["capability_catalog_provider", "subagent_manager", "reentry_scheduler"]

    def bind_runtime(self, **deps: Any) -> None:
        if "capability_catalog_provider" in deps:
            self._capability_catalog_provider = deps["capability_catalog_provider"]
        if "subagent_manager" in deps:
            self._subagent_manager = deps["subagent_manager"]
        if "reentry_scheduler" in deps:
            self._reentry_scheduler = deps["reentry_scheduler"]

    # ── Internal helpers ──

    def _capability_catalog(self) -> List[Dict[str, Any]]:
        """Resolve the live tool catalog for capability discovery."""
        if self._capability_catalog_provider is not None:
            try:
                catalog = self._capability_catalog_provider()
            except (RuntimeError, ValueError, TypeError):
                catalog = None
            if catalog:
                return list(catalog)
        # Fallback to static tool_definitions from the plugin registry
        from leapflow.plugins import get_registry

        return get_registry().tool_definitions

    # ── Handlers ──

    async def _capability_expand_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for capability_expand tool."""
        from leapflow.engine.context.context_disclosure import build_capability_manifests

        category = str(params.get("category") or "").strip().lower()
        if not category:
            return {"ok": False, "error": "category is required"}
        catalog = self._capability_catalog()
        manifests = build_capability_manifests(catalog)
        matched_names = {m.name for m in manifests if m.category == category}
        if not matched_names:
            available = sorted({m.category for m in manifests if m.category})
            return {
                "ok": False,
                "error": f"Unknown capability category: {category}",
                "available_categories": available,
            }
        expanded_tools = [
            td for td in catalog if td.get("function", {}).get("name") in matched_names
        ]
        return {"ok": True, "category": category, "expanded_tools": expanded_tools}

    async def _delegate_task_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for delegate_task tool."""
        if self._subagent_manager is None:
            return {"ok": False, "error": "Subagent system not configured"}
        try:
            from leapflow.engine.subagent import SubagentConfig, current_subagent_depth

            config = SubagentConfig(
                goal=params.get("goal", ""),
                context=params.get("context", ""),
                depth=current_subagent_depth() + 1,
            )
            result = await self._subagent_manager.delegate(config)
            return {
                "ok": result.status == "completed",
                "summary": result.summary,
                "status": result.status,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def _schedule_reentry_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for schedule_reentry tool."""
        if self._reentry_scheduler is None:
            return {"ok": False, "error": "Re-entry scheduling not initialized"}
        try:
            result = self._reentry_scheduler(
                kind=str(params.get("kind", "time")),
                reason=str(params.get("reason", "")),
                delay_seconds=params.get("delay_seconds", 0.0),
                event_match=params.get("event_match") or {},
                max_reentries=params.get("max_reentries", 1),
                deadline_seconds=params.get("deadline_seconds", 0.0),
            )
            return result if isinstance(result, dict) else {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Tool metadata ──

    @property
    def tools(self) -> list[ToolMetadata]:
        return [
            ToolMetadata(
                name="capability_expand",
                description=(
                    "Fetch the full callable schema for every tool in a capability category "
                    "(e.g. 'hub', 'gateway', 'desktop', 'delegate', 'file', 'memory', 'skill'). The compact "
                    "tool index always lists every registered tool by name and a one-line summary, "
                    "but only a static low-risk subset is directly callable each turn. If you need a "
                    "tool from the index that is not yet callable, call capability_expand with its "
                    "category first; the matching tools become callable in this turn. Never invent a "
                    "tool name \u2014 expand the category instead."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Capability category name, e.g. hub, gateway, delegate",
                        },
                    },
                    "required": ["category"],
                },
                handler=self._capability_expand_handler,
                x_leapflow={
                    "category": "system",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("system.capability_expand",),
            ),
            ToolMetadata(
                name="delegate_task",
                description=(
                    "Delegate a complex sub-task to an isolated subagent. "
                    "The subagent gets a fresh context and restricted tool access. "
                    "Use when a task is self-contained and can be solved independently."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": "Clear description of the task to delegate",
                        },
                        "context": {
                            "type": "string",
                            "description": "Relevant context for the subagent (optional)",
                        },
                    },
                    "required": ["goal"],
                },
                handler=self._delegate_task_handler,
                x_leapflow={
                    "category": "delegate",
                    "risk_level": "medium",
                    "schema_cost": "medium",
                    "requires_approval": False,
                },
                provides_capabilities=("system.delegate",),
            ),
            ToolMetadata(
                name="schedule_reentry",
                description=(
                    "Register a re-entry so this task can resume later from its current "
                    "orientation (findings / open questions / next step). Use when work must "
                    "pause and continue after a delay (kind=time) or when a matching platform "
                    "event arrives (kind=event), instead of finishing now. The research-ledger "
                    "state is carried over automatically."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["time", "event"],
                            "description": "time = resume after delay_seconds; event = resume when a matching platform event arrives",
                        },
                        "reason": {
                            "type": "string",
                            "description": "One concise sentence: what to continue and why (carried into the resumed turn).",
                        },
                        "delay_seconds": {
                            "type": "number",
                            "description": "kind=time: seconds from now to resume.",
                        },
                        "event_match": {
                            "type": "object",
                            "description": "kind=event: match filter, e.g. platform / chat / keyword.",
                        },
                        "max_reentries": {
                            "type": "integer",
                            "description": "Max times this may resume (default 1).",
                        },
                        "deadline_seconds": {
                            "type": "number",
                            "description": "Optional: abandon the re-entry after this many seconds.",
                        },
                    },
                    "required": ["kind", "reason"],
                },
                handler=self._schedule_reentry_handler,
                x_leapflow={
                    "category": "memory",
                    "risk_level": "read_only",
                    "schema_cost": "medium",
                    "requires_approval": False,
                },
                provides_capabilities=("system.schedule_reentry",),
            ),
        ]


# Module-level instance for plugin discovery
plugin = OrchestrationPlugin()
