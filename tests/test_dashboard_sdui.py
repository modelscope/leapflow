"""Hermetic tests for the dashboard SDUI core: ViewSpec, templates, intent."""

from __future__ import annotations

from leapflow.dashboard import (
    DashboardIntent,
    TemplateLibrary,
    normalize_viewspec,
    render_template,
    validate_viewspec,
)
from leapflow.dashboard.templates import bind_value, resolve_path


# ── ViewSpec catalog + validation + fallback ───────────────────────────────


def test_normalize_degrades_unknown_component_to_markdown() -> None:
    spec = normalize_viewspec({
        "title": "T",
        "root": [
            {"type": "Card", "children": [{"type": "Nonexistent", "props": {"x": 1}}]},
        ],
    })
    assert spec["schema_version"] == 1
    card = spec["root"][0]
    assert card["type"] == "Card"
    child = card["children"][0]
    assert child["type"] == "Markdown"
    assert "Unsupported component" in child["props"]["text"]


def test_normalize_keeps_valid_action_and_drops_invalid() -> None:
    spec = normalize_viewspec({
        "root": [
            {"type": "Button", "action": {"kind": "rpc", "name": "watch.pause", "params": {"id": "x"}}},
            {"type": "Button", "action": {"kind": "evil", "name": "hack"}},
        ],
    })
    assert spec["root"][0]["action"]["kind"] == "rpc"
    assert "action" not in spec["root"][1]


def test_validate_reports_unknown_type_and_bad_action() -> None:
    errors = validate_viewspec({
        "schema_version": 1,
        "root": [
            {"type": "Bogus"},
            {"type": "Button", "action": {"kind": "nope"}},
        ],
    })
    assert any("unknown component type" in e for e in errors)
    assert any("invalid action" in e for e in errors)


def test_validate_accepts_clean_spec() -> None:
    assert validate_viewspec({
        "schema_version": 1,
        "root": [{"type": "Card", "children": [{"type": "Markdown", "props": {"text": "hi"}}]}],
    }) == []


# ── Template binding ────────────────────────────────────────────────────────


def test_resolve_path_supports_dots_and_indices() -> None:
    data = {"a": {"b": [{"c": 7}]}}
    assert resolve_path(data, "a.b[0].c") == 7
    assert resolve_path(data, "a.missing") is None
    assert resolve_path(data, "a.b[5]") is None


def test_bind_value_full_and_interpolated() -> None:
    data = {"finding": {"title": "Spike", "score": 0.9}}
    assert bind_value("{{ finding.score }}", data) == 0.9  # full match preserves type
    assert bind_value("T: {{ finding.title }}", data) == "T: Spike"  # interpolation -> str


def test_bind_value_multiple_placeholders_interpolate() -> None:
    # Two placeholders + a literal must interpolate to a string, not be misread
    # as a single bogus dotted path (which previously resolved to None).
    data = {"observation": {"artifacts_included": 2, "artifact_count": 3}}
    assert (
        bind_value(
            "{{ observation.artifacts_included }}/{{ observation.artifact_count }}", data
        )
        == "2/3"
    )


def test_render_template_repeat_and_bind() -> None:
    template = {
        "template": "demo",
        "title": "Watch {{ name }}",
        "layout": [
            {
                "type": "Board",
                "children": [
                    {
                        "type": "FindingCard",
                        "repeat": "findings",
                        "as": "f",
                        "props": {"title": "{{ f.title }}", "bind": "f"},
                    }
                ],
            }
        ],
    }
    data = {"name": "AAPL", "findings": [{"title": "a"}, {"title": "b"}]}
    spec = render_template(template, data)
    assert spec["title"] == "Watch AAPL"
    cards = spec["root"][0]["children"]
    assert [c["props"]["title"] for c in cards] == ["a", "b"]
    assert cards[0]["props"]["data"] == {"title": "a"}  # bind -> data


def test_render_template_repeat_missing_list_yields_no_children() -> None:
    template = {"layout": [{"type": "Board", "children": [
        {"type": "FindingCard", "repeat": "nope", "props": {}}]}]}
    spec = render_template(template, {})
    assert spec["root"][0]["children"] == []


def test_template_library_generic_renders_session_analysis() -> None:
    lib = TemplateLibrary()
    assert "generic" in lib.names()
    spec = lib.render("generic", {"title": "Session Analysis", "analysis": {
        "story": "arc",
        "insights": [{"title": "i", "summary": "s", "severity": "notable"}],
        "action_items": [], "decisions": [], "open_questions": [],
        "entities": ["Alice"], "next_prompts": ["ask"],
    }, "observation": {"refresh_reason": "manual_refresh", "context_scope": "text_only"}, "artifact_context": []})
    assert spec["title"] == "Session Analysis"
    flat: list[dict] = []

    def _walk(nodes: list) -> None:
        for n in nodes:
            flat.append(n)
            _walk(n.get("children") or [])

    _walk(spec["root"])
    types = {n["type"] for n in flat}
    assert "StoryPanel" in types
    assert len([n for n in flat if n["type"] == "InsightCard"]) == 1


def test_template_library_unknown_falls_back_to_generic() -> None:
    lib = TemplateLibrary()
    spec = lib.render("does-not-exist", {"title": "F", "findings": []})
    assert spec["meta"]["template"] == "generic"


# ── DashboardIntent (dual entry) ────────────────────────────────────────────


def test_intent_from_args_first_token_is_template() -> None:
    assert DashboardIntent.from_args("finance").template == "finance"
    assert DashboardIntent.from_args("research extra tokens").template == "research"
    assert DashboardIntent.from_args("").template == ""


def test_intent_from_params_reads_template() -> None:
    assert DashboardIntent.from_params({"template": "research"}).template == "research"
    assert DashboardIntent.from_params({}).to_dict() == {"template": ""}
