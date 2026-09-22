# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for BM25 tool search engine, budget-driven listing, and bridge plugin."""
from __future__ import annotations

import asyncio
import random


from leapflow.engine.tools.tool_search import (
    ListingLevel,
    ToolSearchIndex,
    _stem,
    _tokenize,
    entries_from_tool_definitions,
    render_tool_listing,
)


# ── Test data ──────────────────────────────────────────────────────────


def _sample_entries() -> list[dict]:
    """Representative tool entries for search tests."""
    return [
        {
            "name": "file_read",
            "category": "file",
            "summary": "Read the contents of a file from disk",
            "parameter_names": ["path", "encoding"],
        },
        {
            "name": "file_write",
            "category": "write",
            "summary": "Write content to a file on disk",
            "parameter_names": ["path", "content", "mode"],
        },
        {
            "name": "shell_exec",
            "category": "shell",
            "summary": "Execute a shell command in the terminal",
            "parameter_names": ["command", "timeout", "cwd"],
        },
        {
            "name": "memory_store",
            "category": "memory",
            "summary": "Store a key-value pair in agent memory",
            "parameter_names": ["key", "value", "namespace"],
        },
        {
            "name": "memory_recall",
            "category": "memory",
            "summary": "Recall a stored value from agent memory by key",
            "parameter_names": ["key", "namespace"],
        },
        {
            "name": "web_fetch",
            "category": "search",
            "summary": "Fetch and parse content from a web URL",
            "parameter_names": ["url", "selector"],
        },
        {
            "name": "git_diff",
            "category": "scm",
            "summary": "Show git diff of working tree or staged changes",
            "parameter_names": ["path", "staged"],
        },
        {
            "name": "capability_expand",
            "category": "system",
            "summary": "Fetch the full callable schema for every tool in a capability category",
            "parameter_names": ["category"],
        },
        {
            "name": "delegate_task",
            "category": "delegate",
            "summary": "Delegate a complex sub-task to an isolated subagent",
            "parameter_names": ["goal", "context"],
        },
        {
            "name": "tool_search",
            "category": "bridge",
            "summary": "Search all registered tools by keyword relevance",
            "parameter_names": ["query", "max_results"],
        },
    ]


def _build_index(entries: list[dict] | None = None) -> ToolSearchIndex:
    idx = ToolSearchIndex()
    idx.rebuild(entries or _sample_entries())
    return idx


def _make_tool_defs(count: int = 5) -> list[dict]:
    """Generate OpenAI-format tool definitions for listing tests."""
    defs = []
    for i in range(count):
        defs.append(
            {
                "type": "function",
                "function": {
                    "name": f"tool_{i:03d}",
                    "description": f"Description for tool {i} that does something useful",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param_a": {"type": "string", "description": "Param A"},
                            "param_b": {"type": "integer", "description": "Param B"},
                        },
                    },
                    "x_leapflow": {
                        "category": f"cat_{i % 3}",
                        "risk_level": "read_only",
                        "schema_cost": "low",
                    },
                },
            }
        )
    return defs


# ── Stemmer tests ──────────────────────────────────────────────────────


class TestStem:
    def test_short_words_unchanged(self) -> None:
        assert _stem("at") == "at"
        assert _stem("the") == "the"
        assert _stem("go") == "go"

    def test_plurals_simple(self) -> None:
        assert _stem("files") == "file"
        assert _stem("tools") == "tool"

    def test_plurals_ies(self) -> None:
        assert _stem("directories") == "directori"

    def test_plurals_sses(self) -> None:
        assert _stem("processes") == "process"

    def test_plurals_sibilant_es(self) -> None:
        assert _stem("boxes") == "box"
        assert _stem("patches") == "patch"

    def test_ing_suffix(self) -> None:
        assert _stem("reading") == "read"
        assert _stem("writing") == "writ"

    def test_ed_suffix(self) -> None:
        assert _stem("stored") == "stor"
        assert _stem("parsed") == "pars"

    def test_tion_suffix(self) -> None:
        assert _stem("execution") == "execu"

    def test_ment_suffix(self) -> None:
        assert _stem("management") == "manage"
        assert _stem("environment") == "environ"

    def test_ness_suffix(self) -> None:
        assert _stem("awareness") == "aware"

    def test_er_suffix(self) -> None:
        assert _stem("manager") == "manag"

    def test_ation_suffix(self) -> None:
        # "ation" → "ate": "configuration" → "configurate"
        assert _stem("configuration") == "configurate"


# ── Tokenize tests ─────────────────────────────────────────────────────


class TestTokenize:
    def test_basic(self) -> None:
        tokens = _tokenize("file read")
        assert "file" in tokens
        assert "read" in tokens

    def test_snake_case_splits(self) -> None:
        tokens = _tokenize("file_read_contents")
        assert "file" in tokens
        assert "read" in tokens
        assert "content" in tokens  # stemmed from "contents"

    def test_filters_single_char(self) -> None:
        tokens = _tokenize("a b c file")
        assert "file" in tokens
        assert "a" not in tokens

    def test_lowercases(self) -> None:
        tokens = _tokenize("FILE READ")
        assert "file" in tokens
        assert "read" in tokens

    def test_empty_string(self) -> None:
        assert _tokenize("") == []


# ── BM25 scoring tests ────────────────────────────────────────────────


class TestBM25Scoring:
    def test_relevant_results_first(self) -> None:
        idx = _build_index()
        results = idx.search("read file contents")
        assert len(results) > 0
        assert results[0]["name"] == "file_read"

    def test_memory_query(self) -> None:
        idx = _build_index()
        results = idx.search("store value in memory")
        names = [r["name"] for r in results]
        assert "memory_store" in names

    def test_shell_query(self) -> None:
        idx = _build_index()
        results = idx.search("execute shell command")
        assert results[0]["name"] == "shell_exec"

    def test_scores_are_positive(self) -> None:
        idx = _build_index()
        results = idx.search("file")
        for r in results:
            if r["score"] != "exact_match":
                assert r["score"] > 0

    def test_max_results_respected(self) -> None:
        idx = _build_index()
        results = idx.search("file", max_results=2)
        assert len(results) <= 2


# ── Gate token filter tests ────────────────────────────────────────────


class TestGateToken:
    def test_rare_term_gates(self) -> None:
        """Document must contain the highest-IDF query term."""
        idx = _build_index()
        # "subagent" is rare; only delegate_task mentions it
        results = idx.search("subagent")
        names = [r["name"] for r in results]
        assert "delegate_task" in names
        assert "file_read" not in names

    def test_common_term_allows_matches(self) -> None:
        """Common terms have low IDF and are not restrictive gates."""
        idx = _build_index()
        results = idx.search("file")
        assert len(results) >= 1


# ── Term coverage filter tests ─────────────────────────────────────────


class TestTermCoverage:
    def test_long_query_filters_poor_matches(self) -> None:
        """Queries with >= 4 terms require >= 50% term overlap."""
        idx = _build_index()
        results = idx.search("git diff staged working tree changes")
        names = [r["name"] for r in results]
        if names:
            assert "git_diff" in names

    def test_short_query_no_coverage_filter(self) -> None:
        """Queries with < 4 terms skip the coverage filter."""
        idx = _build_index()
        results = idx.search("key value")
        assert len(results) >= 1


# ── Exact name match tests ─────────────────────────────────────────────


class TestExactNameMatch:
    def test_exact_name_first(self) -> None:
        idx = _build_index()
        results = idx.search("file_read")
        assert results[0]["name"] == "file_read"
        assert results[0]["score"] == "exact_match"

    def test_name_with_hyphens(self) -> None:
        idx = _build_index()
        results = idx.search("file-read")
        assert results[0]["name"] == "file_read"
        assert results[0]["score"] == "exact_match"

    def test_name_with_spaces(self) -> None:
        idx = _build_index()
        results = idx.search("file read")
        assert results[0]["name"] == "file_read"
        assert results[0]["score"] == "exact_match"

    def test_exact_match_plus_bm25_results(self) -> None:
        idx = _build_index()
        results = idx.search("shell_exec")
        assert results[0]["name"] == "shell_exec"
        assert results[0]["score"] == "exact_match"


# ── Edge case tests ────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_query(self) -> None:
        idx = _build_index()
        assert idx.search("") == []

    def test_no_matching_results(self) -> None:
        idx = _build_index()
        results = idx.search("quantum_teleportation_device")
        assert results == []

    def test_empty_index(self) -> None:
        idx = ToolSearchIndex()
        idx.rebuild([])
        assert idx.search("anything") == []

    def test_rebuild_replaces_index(self) -> None:
        idx = _build_index()
        r1 = idx.search("file")
        idx.rebuild([{"name": "only_tool", "category": "test", "summary": "test"}])
        r2 = idx.search("file")
        assert len(r2) == 0 or r2 != r1

    def test_single_doc_index(self) -> None:
        idx = ToolSearchIndex()
        idx.rebuild([{"name": "alpha", "category": "a", "summary": "alpha tool"}])
        results = idx.search("alpha")
        assert len(results) == 1
        assert results[0]["name"] == "alpha"


# ── Budget-driven listing tests ────────────────────────────────────────


class TestRenderToolListing:
    def test_full_listing_small_catalog(self) -> None:
        defs = _make_tool_defs(3)
        text, level = render_tool_listing(defs, token_budget=5000)
        assert level == ListingLevel.FULL
        assert "tool_000" in text
        assert "tool_001" in text
        assert "tool_002" in text

    def test_degradation_with_tight_budget(self) -> None:
        defs = _make_tool_defs(50)
        _, level = render_tool_listing(defs, token_budget=50)
        assert level in (
            ListingLevel.NAMES_ONLY,
            ListingLevel.GROUPED,
            ListingLevel.NONE,
        )

    def test_none_listing_tiny_budget(self) -> None:
        defs = _make_tool_defs(200)
        text, level = render_tool_listing(defs, token_budget=10)
        assert level == ListingLevel.NONE
        assert "tool_search" in text

    def test_byte_stability(self) -> None:
        """Same input produces identical output."""
        defs = _make_tool_defs(10)
        text1, level1 = render_tool_listing(defs, token_budget=5000)
        text2, level2 = render_tool_listing(defs, token_budget=5000)
        assert text1 == text2
        assert level1 == level2

    def test_byte_stability_with_shuffled_input(self) -> None:
        """Shuffled input produces identical output (sorted categories + tools)."""
        defs = _make_tool_defs(10)
        text1, level1 = render_tool_listing(defs, token_budget=5000)
        shuffled = list(defs)
        random.seed(42)
        random.shuffle(shuffled)
        text2, level2 = render_tool_listing(shuffled, token_budget=5000)
        assert text1 == text2
        assert level1 == level2


# ── entries_from_tool_definitions tests ────────────────────────────────


class TestEntriesFromToolDefinitions:
    def test_extracts_fields(self) -> None:
        defs = _make_tool_defs(2)
        entries = entries_from_tool_definitions(defs)
        assert len(entries) == 2
        assert entries[0]["name"] == "tool_000"
        assert "param_a" in entries[0]["parameter_names"]
        assert "param_b" in entries[0]["parameter_names"]
        assert entries[0]["category"] != ""

    def test_empty_input(self) -> None:
        entries = entries_from_tool_definitions([])
        assert entries == []

    def test_roundtrip_with_search(self) -> None:
        """Entries built from tool_definitions can be searched."""
        defs = _make_tool_defs(5)
        entries = entries_from_tool_definitions(defs)
        idx = ToolSearchIndex()
        idx.rebuild(entries)
        results = idx.search("tool_002")
        assert results[0]["name"] == "tool_002"
        assert results[0]["score"] == "exact_match"


# ── Bridge plugin tests ────────────────────────────────────────────────


class TestBridgePlugin:
    def test_plugin_protocol(self) -> None:
        from leapflow.plugins.tool_plugins.bridge import plugin

        assert plugin.plugin_id == "bridge"
        assert plugin.category == "bridge"
        assert "capability_catalog_provider" in plugin.dependencies

    def test_tool_metadata(self) -> None:
        from leapflow.plugins.tool_plugins.bridge import plugin

        tools = plugin.tools
        assert len(tools) == 2
        by_name = {t.name: t for t in tools}

        ts = by_name["tool_search"]
        assert ts.x_leapflow["category"] == "bridge"
        assert ts.x_leapflow["risk_level"] == "read_only"
        assert ts.x_leapflow["schema_cost"] == "low"
        assert not ts.mutates_state

        td = by_name["tool_describe"]
        assert td.x_leapflow["category"] == "bridge"
        assert td.x_leapflow["risk_level"] == "read_only"
        assert td.x_leapflow["schema_cost"] == "low"
        assert not td.mutates_state

    def test_tools_are_pcd_core(self) -> None:
        from leapflow.engine.context.context_disclosure import CapabilityManifest
        from leapflow.plugins.tool_plugins.bridge import plugin

        for tool in plugin.tools:
            schema = tool.to_openai_schema()
            manifest = CapabilityManifest.from_tool_definition(schema)
            assert manifest.is_core, (
                f"{tool.name} should be PCD CORE "
                f"(risk={manifest.risk_level}, cost={manifest.schema_cost})"
            )

    def test_search_handler_empty_query(self) -> None:
        from leapflow.plugins.tool_plugins.bridge import plugin

        result = asyncio.run(plugin._tool_search_handler({"query": ""}))
        assert result["ok"] is False
        assert "required" in result["error"]

    def test_describe_handler_empty_name(self) -> None:
        from leapflow.plugins.tool_plugins.bridge import plugin

        result = asyncio.run(plugin._tool_describe_handler({"tool_name": ""}))
        assert result["ok"] is False
        assert "required" in result["error"]

    def test_describe_handler_not_found(self) -> None:
        from leapflow.plugins.tool_plugins.bridge import plugin

        plugin._capability_catalog_provider = lambda: _make_tool_defs(3)
        try:
            result = asyncio.run(
                plugin._tool_describe_handler({"tool_name": "nonexistent"})
            )
            assert result["ok"] is False
            assert "not found" in result["error"]
        finally:
            plugin._capability_catalog_provider = None

    def test_search_handler_with_catalog(self) -> None:
        from leapflow.plugins.tool_plugins.bridge import plugin

        plugin._capability_catalog_provider = lambda: _make_tool_defs(5)
        try:
            result = asyncio.run(
                plugin._tool_search_handler({"query": "tool_002"})
            )
            assert result["ok"] is True
            assert result["count"] > 0
            assert result["results"][0]["name"] == "tool_002"
        finally:
            plugin._capability_catalog_provider = None
            plugin._index = None
            plugin._index_hash = 0

    def test_describe_handler_with_catalog(self) -> None:
        from leapflow.plugins.tool_plugins.bridge import plugin

        plugin._capability_catalog_provider = lambda: _make_tool_defs(5)
        try:
            result = asyncio.run(
                plugin._tool_describe_handler({"tool_name": "tool_002"})
            )
            assert result["ok"] is True
            assert result["tool"]["name"] == "tool_002"
            assert "parameters" in result["tool"]
        finally:
            plugin._capability_catalog_provider = None
