# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the semantic desktop tool schema layer and registration plugin."""

from __future__ import annotations

import leapflow.plugins.tool_plugins.desktop_semantic as desktop_semantic_module
from leapflow.skills.semantic_adapter import SemanticAdapter
from leapflow.skills.semantic_schema import (
    DESKTOP_CATEGORY,
    SEMANTIC_TOOL_NAMES,
    parse_param_spec,
    semantic_requires_approval,
    semantic_tool_to_openai,
)
from leapflow.skills.tool_types import (
    ToolCall,
    ToolDefinition,
)
from leapflow.skills.tool_executor import (
    ExecutionToolset,
    build_execution_toolset,
)
from leapflow.plugins.tool_plugins.desktop_semantic import (
    DesktopSemanticPlugin,
    SemanticToolEntry,
    build_semantic_tool_entries,
)


def _definition(name: str, parameters: dict[str, str] | None = None) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"test {name}",
        parameters=parameters or {"target": "string (required) — element to act on"},
    )


def _adapter() -> SemanticAdapter:
    return SemanticAdapter(perception=object(), execution=object())


# ── Parameter spec parsing ─────────────────────────────────────────────


def test_parse_param_spec_required_with_description() -> None:
    spec = parse_param_spec("string (required) — text to type")
    assert spec.type == "string"
    assert spec.required is True
    assert spec.description == "text to type"


def test_parse_param_spec_optional_with_default() -> None:
    spec = parse_param_spec("number (optional, default=30) — max seconds to wait")
    assert spec.type == "number"
    assert spec.required is False
    assert spec.description == "max seconds to wait"


def test_parse_param_spec_no_flags() -> None:
    spec = parse_param_spec("boolean — whether to wait")
    assert spec.type == "boolean"
    assert spec.required is False
    assert spec.description == "whether to wait"


def test_parse_param_spec_no_description() -> None:
    spec = parse_param_spec("int (required)")
    assert spec.type == "integer"
    assert spec.required is True
    assert spec.description == ""


def test_parse_param_spec_empty_and_unknown() -> None:
    assert parse_param_spec("").required is False
    # Unrecognizable head falls back to string with the whole text as description.
    spec = parse_param_spec("??? weird")
    assert spec.type == "string"
    assert spec.description == "??? weird"


# ── Schema conversion ───────────────────────────────────────────────────


def test_semantic_tool_to_openai_mutating_metadata() -> None:
    schema = semantic_tool_to_openai(_definition("click"))
    assert schema is not None
    func = schema["function"]
    assert func["name"] == "click"
    assert func["parameters"]["properties"]["target"]["type"] == "string"
    assert func["parameters"]["required"] == ["target"]
    meta = schema["x_leapflow"]
    assert meta["category"] == DESKTOP_CATEGORY
    assert meta["risk_level"] == "medium"
    assert meta["schema_cost"] == "high"
    assert meta["requires_approval"] is True


def test_semantic_tool_to_openai_observation_metadata() -> None:
    for name in ("observe_ui", "list_apps", "screenshot"):
        schema = semantic_tool_to_openai(_definition(name))
        assert schema is not None
        meta = schema["x_leapflow"]
        assert meta["risk_level"] == "read_only"
        assert meta["requires_approval"] is False
        assert meta["schema_cost"] == "high"


def test_semantic_tool_to_openai_rejects_non_semantic() -> None:
    assert semantic_tool_to_openai(_definition("file_list")) is None
    assert semantic_tool_to_openai(_definition("gp_shell_run")) is None


# ── Approval classification ─────────────────────────────────────────────


def test_semantic_requires_approval_split() -> None:
    mutating = {"click", "type_text", "shortcut", "switch_app", "open_url",
                "set_clipboard", "scroll", "select_text", "right_click"}
    passive = SEMANTIC_TOOL_NAMES - mutating
    assert all(semantic_requires_approval(name) for name in mutating)
    assert all(not semantic_requires_approval(name) for name in passive)
    assert not semantic_requires_approval("file_list")


def test_semantic_name_set_is_complete() -> None:
    assert len(SEMANTIC_TOOL_NAMES) == 18


# ── Registration entries (single source of truth) ──────────────────────


def test_entries_cover_exactly_semantic_tool_names() -> None:
    entries = build_semantic_tool_entries(_adapter())
    assert {e.name for e in entries} == set(SEMANTIC_TOOL_NAMES)


def test_entry_traits_match_registration_contract() -> None:
    """Executor traits preserved from the original registration site."""
    entries = {e.name: e for e in build_semantic_tool_entries(_adapter())}

    # Mutating UI tools declare mutates_state; click/right_click carry a
    # describer so the policy gate can resolve element_index params.
    assert entries["click"].mutates_state is True
    assert callable(entries["click"].describer)
    assert entries["right_click"].mutates_state is True
    assert callable(entries["right_click"].describer)
    for name in ("type_text", "shortcut", "switch_app", "open_url",
                 "set_clipboard", "scroll", "select_text"):
        assert entries[name].mutates_state is True

    # Observation tools are read-only.
    for name in ("observe_ui", "list_apps", "list_windows", "screenshot"):
        assert entries[name].mutates_state is False

    # Wait tools mutate (clear dedup cache) but are not forward progress.
    for name in ("wait", "wait_until", "wait_until_stable"):
        assert entries[name].mutates_state is True
        assert entries[name].counts_as_progress is False

    # Every entry exposes an async-callable handler.
    for entry in entries.values():
        assert callable(entry.handler)


# ── Plugin lifecycle ────────────────────────────────────────────────────


def test_plugin_inactive_until_both_ports_bound() -> None:
    plugin = DesktopSemanticPlugin()
    assert plugin.active is False
    assert plugin.get_semantic_schemas() == []
    assert plugin.get_semantic_handlers() == {}

    plugin.bind_runtime(perception=object())  # execution still missing
    assert plugin.active is False


def test_plugin_activates_when_both_ports_bound() -> None:
    plugin = DesktopSemanticPlugin()
    plugin.bind_runtime(perception=object(), execution=object())
    assert plugin.active is True

    schemas = plugin.get_semantic_schemas()
    names = [item["function"]["name"] for item in schemas]
    assert names == sorted(SEMANTIC_TOOL_NAMES)
    # Handler table matches schema disclosure exactly — the two never disagree.
    assert set(plugin.get_semantic_handlers()) == set(names)


def test_plugin_deactivates_when_a_port_goes_away() -> None:
    plugin = DesktopSemanticPlugin()
    plugin.bind_runtime(perception=object(), execution=object())
    assert plugin.active is True

    plugin.bind_runtime(perception=None)  # single port cleared — offline
    assert plugin.active is False
    assert plugin.get_semantic_schemas() == []
    assert plugin.get_semantic_handlers() == {}


def test_plugin_schemas_carry_leapflow_metadata() -> None:
    plugin = DesktopSemanticPlugin()
    plugin.bind_runtime(perception=object(), execution=object())
    by_name = {
        item["function"]["name"]: item for item in plugin.get_semantic_schemas()
    }

    click = by_name["click"]
    assert click["x_leapflow"] == {
        "category": DESKTOP_CATEGORY,
        "risk_level": "medium",
        "schema_cost": "high",
        "requires_approval": True,
    }
    observe = by_name["observe_ui"]
    assert observe["x_leapflow"]["risk_level"] == "read_only"
    assert observe["x_leapflow"]["requires_approval"] is False
    # Required params parsed from registration strings.
    assert sorted(observe["function"]["parameters"]["required"]) == [
        "pid", "window_id",
    ]


async def test_plugin_handlers_dispatch_to_registered_tools(monkeypatch) -> None:
    """A semantic handler from the plugin executes when awaited with params."""
    calls: list[tuple[str, dict]] = []

    async def _click(params: dict) -> dict:
        calls.append(("click", dict(params)))
        return {"ok": True, "clicked": params.get("element_index")}

    def _fake_entries(adapter: object) -> list[SemanticToolEntry]:
        return [
            SemanticToolEntry(
                name="click",
                description="Click a UI element",
                parameters={"element_index": "int (required) — element_index"},
                handler=_click,
                mutates_state=True,
            )
        ]

    monkeypatch.setattr(
        desktop_semantic_module, "build_semantic_tool_entries", _fake_entries
    )
    plugin = DesktopSemanticPlugin()
    plugin.bind_runtime(perception=object(), execution=object())

    handler = plugin.get_semantic_handlers()["click"]
    result = await handler({"element_index": 5})
    assert result == {"ok": True, "clicked": 5}
    assert calls == [("click", {"element_index": 5})]


def test_plugin_bumps_version_on_state_change() -> None:
    plugin = DesktopSemanticPlugin()
    v0 = plugin.version
    plugin.bind_runtime(perception=object(), execution=object())
    assert plugin.version > v0
    plugin.bind_runtime(perception=None, execution=None)
    assert plugin.version > v0


# ── Skill executor factory ──────────────────────────────────────────────


def test_build_execution_toolset_merges_defaults_and_semantic_tools() -> None:
    toolset = build_execution_toolset(object(), perception=object())
    names = {t.name for t in toolset.tool_definitions()}

    # ExecutionPort defaults remain available to the ReAct loop.
    assert {"shell", "file_list", "file_move", "mkdir", "launch_app", "done"} <= names
    # Semantic tools registered with executor traits.
    assert SEMANTIC_TOOL_NAMES <= names
    assert toolset.is_mutating("click") is True
    assert toolset.is_mutating("observe_ui") is False
    assert toolset.is_progress("switch_app") is True
    assert toolset.is_progress("wait_until") is False


def test_build_execution_toolset_without_perception_has_no_semantic_tools() -> None:
    toolset = build_execution_toolset(object())
    names = {t.name for t in toolset.tool_definitions()}
    assert not (names & SEMANTIC_TOOL_NAMES)
    assert "shell" in names


def test_execution_toolset_register_is_open_for_extension() -> None:
    toolset = ExecutionToolset(object())

    async def _noop(params: dict) -> dict:
        return {"ok": True}

    toolset.register("custom_tool", "Custom", {}, _noop, mutates_state=True)
    assert "custom_tool" in toolset.handlers
    assert toolset.is_mutating("custom_tool") is True
    assert toolset.is_mutating("unknown_tool") is False


async def test_execution_toolset_unknown_tool_fails_cleanly() -> None:
    from leapflow.skills.tool_types import ToolCall

    toolset = build_execution_toolset(object(), perception=object())
    result = await toolset.dispatch(ToolCall(name="nope", params={}))
    assert result["ok"] is False
    assert "unknown_tool" in result["error"]
