# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bridge plugin — tool search and describe meta-tools.

These meta-tools let the LLM discover and inspect registered tools at runtime
via BM25-based search, without DisclosurePlanner reading user free-form text.
Both tools are read-only, low-cost, and available at PCD CORE level.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from leapflow.plugins.protocol import ToolMetadata

logger = logging.getLogger(__name__)


class BridgePlugin:
    """Bridge tools for LLM-driven capability discovery.

    Provides ``tool_search`` (BM25 search) and ``tool_describe`` (full schema
    inspection) as meta-tools the LLM can call to find and understand available
    tools.  The BM25 index is rebuilt lazily when the catalog changes.
    """

    def __init__(self) -> None:
        self._capability_catalog_provider: Optional[Callable[[], List[Dict[str, Any]]]] = None
        self._index: Any = None  # lazy ToolSearchIndex
        self._index_hash: int = 0

    @property
    def plugin_id(self) -> str:
        return "bridge"

    @property
    def category(self) -> str:
        return "bridge"

    @property
    def dependencies(self) -> list[str]:
        return ["capability_catalog_provider"]

    def bind_runtime(self, **deps: Any) -> None:
        if "capability_catalog_provider" in deps:
            self._capability_catalog_provider = deps["capability_catalog_provider"]

    # ── Internal helpers ──

    def _capability_catalog(self) -> List[Dict[str, Any]]:
        """Resolve the live tool catalog."""
        if self._capability_catalog_provider is not None:
            try:
                catalog = self._capability_catalog_provider()
            except (RuntimeError, ValueError, TypeError):
                catalog = None
            if catalog:
                return list(catalog)
        from leapflow.plugins import get_registry

        return get_registry().tool_definitions

    def _ensure_index(self) -> Any:
        """Lazily build / rebuild the search index when the catalog changes."""
        from leapflow.engine.tools.tool_search import (
            ToolSearchIndex,
            entries_from_tool_definitions,
        )

        catalog = self._capability_catalog()
        current_hash = hash(
            tuple(
                sorted(
                    str(td.get("function", {}).get("name", "")) for td in catalog
                )
            )
        )
        if self._index is None or current_hash != self._index_hash:
            entries = entries_from_tool_definitions(catalog)
            idx = ToolSearchIndex()
            idx.rebuild(entries)
            self._index = idx
            self._index_hash = current_hash
        return self._index

    # ── Handlers ──

    async def _tool_search_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search registered tools by BM25 relevance."""
        query = str(params.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "query is required"}
        max_results = int(params.get("max_results", 10))
        max_results = max(1, min(max_results, 30))

        try:
            index = self._ensure_index()
            results = index.search(query, max_results=max_results)
        except Exception as exc:
            logger.warning("tool_search index error: %s", exc, exc_info=True)
            return {"ok": False, "error": f"Search index error: {exc}"}

        return {
            "ok": True,
            "query": query,
            "results": results,
            "count": len(results),
            "hint": (
                "Use tool_describe(tool_name=...) to see the full schema "
                "of any result."
            ),
        }

    async def _tool_describe_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Describe a single tool's full schema."""
        tool_name = str(params.get("tool_name") or "").strip()
        if not tool_name:
            return {"ok": False, "error": "tool_name is required"}

        catalog = self._capability_catalog()
        for td in catalog:
            func = td.get("function", {})
            name = str(func.get("name") or td.get("name") or "")
            if name == tool_name:
                return {
                    "ok": True,
                    "tool": {
                        "name": name,
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                        "x_leapflow": (
                            func.get("x_leapflow")
                            or td.get("x_leapflow")
                            or {}
                        ),
                    },
                }
        return {
            "ok": False,
            "error": f"Tool '{tool_name}' not found in the registry.",
        }

    # ── Tool metadata ──

    @property
    def tools(self) -> list[ToolMetadata]:
        return [
            ToolMetadata(
                name="tool_search",
                description=(
                    "Search all registered tools by keyword relevance. Returns a "
                    "ranked list of matching tools with name, category, and summary. "
                    "Use this when you need a tool but are unsure of its exact name "
                    "or category."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Free-text search query describing the capability "
                                "needed."
                            ),
                        },
                        "max_results": {
                            "type": "integer",
                            "description": (
                                "Maximum results to return (default 10, max 30)."
                            ),
                        },
                    },
                    "required": ["query"],
                },
                handler=self._tool_search_handler,
                x_leapflow={
                    "category": "bridge",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "summary": "BM25 search over the tool registry by keyword.",
                    "requires_approval": False,
                },
                provides_capabilities=("bridge.tool_search",),
            ),
            ToolMetadata(
                name="tool_describe",
                description=(
                    "Get the full callable schema of a single tool by exact name. "
                    "Returns the tool's description, parameters, and metadata. "
                    "Use after tool_search to inspect a specific tool before "
                    "calling it."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Exact tool name to describe.",
                        },
                    },
                    "required": ["tool_name"],
                },
                handler=self._tool_describe_handler,
                x_leapflow={
                    "category": "bridge",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "summary": "Inspect the full schema of a registered tool.",
                    "requires_approval": False,
                },
                provides_capabilities=("bridge.tool_describe",),
            ),
        ]


# Module-level instance for plugin discovery
plugin = BridgePlugin()
