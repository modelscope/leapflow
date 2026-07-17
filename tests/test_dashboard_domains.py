"""Hermetic tests for P4 domain templates and the custom-component escape hatch."""

from __future__ import annotations

from leapflow.dashboard import TemplateLibrary, normalize_viewspec, select_template
from leapflow.dashboard.viewspec import COMPONENT_TYPES


def _flatten(spec: dict) -> list[dict]:
    flat: list[dict] = []

    def _walk(nodes: list) -> None:
        for node in nodes:
            flat.append(node)
            _walk(node.get("children") or [])

    _walk(spec["root"])
    return flat


def _types(spec: dict) -> set[str]:
    return {n["type"] for n in _flatten(spec)}


_ANALYSIS = {
    "title": "Session Analysis",
    "analysis": {
        "story": "the arc",
        "insights": [{"title": "i", "summary": "s", "severity": "notable"}],
        "decisions": ["chose y"], "action_items": ["do x"], "open_questions": ["why?"],
        "entities": ["Alice"], "next_prompts": ["ask z"],
    },
    "observation": {"context_coverage_pct": 90}, "artifact_context": [],
}


def test_builtin_template_lenses_are_available() -> None:
    names = TemplateLibrary().names()
    for name in ("finance", "sentiment", "research", "generic"):
        assert name in names
    # Legacy watch-detail templates are gone; there is one target (the session).
    for gone in ("finance.market", "sentiment.topic", "research.paper", "session.analysis", "overview"):
        assert gone not in names


def test_select_template_returns_requested_lens_or_generic() -> None:
    names = TemplateLibrary().names()
    assert select_template("finance", names) == "finance"
    assert select_template("sentiment", names) == "sentiment"
    assert select_template("research", names) == "research"
    assert select_template("", names) == "generic"
    assert select_template("nope", names) == "generic"


def test_finance_lens_reframes_session_analysis() -> None:
    spec = TemplateLibrary().render("finance", _ANALYSIS)
    types = _types(spec)
    # Same session analysis, reframed as calls/actions/exposures.
    assert {"StoryPanel", "BarChart", "EntityGraph", "List"}.issubset(types)


def test_sentiment_lens_reframes_session_analysis() -> None:
    spec = TemplateLibrary().render("sentiment", _ANALYSIS)
    types = _types(spec)
    assert {"StoryPanel", "EntityGraph"}.issubset(types)
    assert len([n for n in _flatten(spec) if n["type"] == "InsightCard"]) == 1


def test_research_lens_reframes_session_analysis() -> None:
    spec = TemplateLibrary().render("research", _ANALYSIS)
    types = _types(spec)
    assert {"StoryPanel", "EntityGraph", "SuggestionChips"}.issubset(types)
    assert len([n for n in _flatten(spec) if n["type"] == "InsightCard"]) == 1


def test_custom_component_is_in_catalog_and_survives_normalize() -> None:
    assert "Custom" in COMPONENT_TYPES
    for component in ("BarChart", "PieChart", "LineChart", "Sparkline", "Timeline", "EntityGraph"):
        assert component in COMPONENT_TYPES
    spec = normalize_viewspec({"root": [{"type": "Custom", "props": {"render": "candlestick"}}]})
    assert spec["root"][0]["type"] == "Custom"
