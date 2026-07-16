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


def test_domain_templates_are_available() -> None:
    names = TemplateLibrary().names()
    for name in ("finance.market", "sentiment.topic", "research.paper", "session.analysis", "generic"):
        assert name in names


def test_select_template_maps_domains_to_templates() -> None:
    names = TemplateLibrary().names()
    assert select_template("finance", "", names) == "finance.market"
    assert select_template("sentiment", "", names) == "sentiment.topic"
    assert select_template("research", "", names) == "research.paper"


def test_finance_template_uses_custom_candlestick() -> None:
    lib = TemplateLibrary()
    spec = lib.render("finance.market", {
        "title": "AAPL", "watch": {"name": "AAPL", "finding_count": 2},
        "findings": [
            {"finding_id": "f1", "title": "volume spike", "summary": "3x", "severity": "alert",
             "payload": {"ohlc": [[1, 2, 3, 4]]}},
        ],
    })
    types = _types(spec)
    assert "Custom" in types  # escape-hatch candlestick component
    assert "PieChart" in types
    assert "Sparkline" in types
    assert "Stat" in types
    assert len([n for n in _flatten(spec) if n["type"] == "FindingCard"]) == 1
    custom = next(n for n in _flatten(spec) if n["type"] == "Custom")
    assert custom["props"]["render"] == "candlestick"
    assert custom["props"]["data"]  # findings bound in


def test_sentiment_template_binds_gauge_value() -> None:
    lib = TemplateLibrary()
    spec = lib.render("sentiment.topic", {
        "title": "BrandX", "watch": {"finding_count": 1},
        "findings": [{"finding_id": "s1", "title": "spike", "summary": "", "severity": "notable",
                      "payload": {"sentiment": 0.82}}],
    })
    gauge = next(n for n in _flatten(spec) if n["type"] == "Gauge")
    assert gauge["props"]["data"] == 0.82  # bound from findings[0].payload.sentiment
    assert "PieChart" in _types(spec)
    assert "LineChart" in _types(spec)


def test_research_template_binds_paper_link_action() -> None:
    lib = TemplateLibrary()
    spec = lib.render("research.paper", {
        "title": "arXiv cs.CL",
        "findings": [{"finding_id": "p1", "title": "A paper", "summary": "abstract", "severity": "info",
                      "payload": {"url": "http://arxiv.org/abs/x"}}],
    })
    cards = [n for n in _flatten(spec) if n["type"] == "FindingCard"]
    types = _types(spec)
    assert "PieChart" in types
    assert "Timeline" in types
    assert len(cards) == 1
    assert cards[0]["action"]["kind"] == "nav"
    assert cards[0]["action"]["params"]["url"] == "http://arxiv.org/abs/x"


def test_custom_component_is_in_catalog_and_survives_normalize() -> None:
    assert "Custom" in COMPONENT_TYPES
    for component in ("BarChart", "PieChart", "LineChart", "Sparkline", "Timeline", "EntityGraph"):
        assert component in COMPONENT_TYPES
    spec = normalize_viewspec({"root": [{"type": "Custom", "props": {"render": "candlestick"}}]})
    assert spec["root"][0]["type"] == "Custom"
