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
    "observation": {"refresh_reason": "manual_refresh", "context_scope": "text_only"}, "artifact_context": [],
}


def test_builtin_template_lenses_are_available() -> None:
    names = TemplateLibrary().names()
    for name in ("finance", "sentiment", "research", "generic", "hardware"):
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


# ════════════════════════════════════════════════════════════════
# The capability board renders from the producer, not from a hand-built payload
# ════════════════════════════════════════════════════════════════


async def test_the_capability_board_renders_the_real_producers_payload() -> None:
    """Driven by the production producer, because a synthetic payload proved nothing.

    The board was reported fixed once already on the strength of a hand-built payload
    fed straight into the renderer. That verified the renderer and nothing else: the
    producer set only ``evidence`` -- label/value pairs a person skims -- and never
    ``payload``, which is what the template binds to. Every panel resolved against an
    empty mapping and produced correct headings over no rows, with nothing anywhere
    reporting a fault.

    Requirements, plan steps, deltas and lifecycle results are lists of records that
    cannot be expressed as label/value evidence at all, which is precisely what the
    domain-private payload field exists for.
    """
    from types import SimpleNamespace

    from leapflow.dashboard.templates import TemplateLibrary, render_node
    from leapflow.monitor.capability_adaptation_producer import CapabilityAdaptationProducer

    record = {
        "record_id": "r-1",
        "phase": "executable",
        "environment": {"fingerprint_id": "env-7"},
        "plan": {
            "executable": True,
            "plan_id": "p-1",
            "missing_dependencies": [],
            "steps": [
                {
                    "tool_name": "pdf_read",
                    "plugin_id": "pdf",
                    "execution_policy": "read_only",
                    "requires_approval": False,
                }
            ],
        },
        "mutation": {"action": "install", "ok": True},
        "registry_version_before": 4,
        "registry_version_after": 5,
        "requirements": [{"capability": "pdf.read", "origin": "goal", "evidence": "user asked"}],
        "decision_delta": {"changed": [{"key": "pdf.read", "before": "none", "after": "pdf_read"}]},
        "observation_ids": ["o1", "o2", "o3"],
        "proposal": {"proposal_id": "prop-9", "status": "approved"},
        "policy_decision": {"action": "install", "autonomy_level": "candidate"},
        "governance_results": [
            {"action": "install", "plugin_id": "pdf", "trust_level": "candidate", "failure_streak": 0}
        ],
    }

    ctx = SimpleNamespace(
        spec=SimpleNamespace(watch_id="w1"),
        services=SimpleNamespace(capability_plan_store=SimpleNamespace(latest=lambda: record)),
    )
    findings = await CapabilityAdaptationProducer().observe(ctx)
    assert findings, "the producer must report on a stored plan"
    payload = dict(findings[0].payload)
    assert payload, "an empty payload leaves every panel on the board blank"

    template = TemplateLibrary().load("capability")
    data = {"title": "Capability", "capability_plan": payload, "findings": [], "watch": {}}
    stats: dict[str, object] = {}
    tables: dict[str, int] = {}

    def walk(node: object) -> None:
        for rendered in render_node(node, data):
            props = rendered.get("props", {})
            if rendered.get("type") == "Stat":
                stats[str(props.get("label"))] = props.get("value")
            if rendered.get("type") == "Table":
                tables[str(props.get("title"))] = len(props.get("data") or [])
            for child in rendered.get("children") or []:
                walk(child)

    for node in template.get("layout") or []:
        walk(node)

    blank = sorted(label for label, value in stats.items() if value in ("", None))
    assert not blank, f"these figures render empty: {blank}"
    assert stats["Loop phase"] == "executable"
    assert stats["Registry delta"] == "4 → 5", "a bare arrow means both versions were missing"
    assert stats["Observations"] == 3, "the count is derived, so it must be put in the payload"

    empty_tables = sorted(title for title, rows in tables.items() if rows == 0)
    assert not empty_tables, f"these tables render headings over no rows: {empty_tables}"
    assert tables == {
        "Requirements": 1,
        "Plan steps": 1,
        "Selection delta": 1,
        "Lifecycle timeline": 1,
    }
