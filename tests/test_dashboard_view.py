"""Hermetic tests for the dashboard view builder and WebSocket fan-out hub."""

from __future__ import annotations

from typing import Any

from leapflow.dashboard import (
    DashboardIntent,
    DashboardViewBuilder,
    TemplateLibrary,
    ViewHub,
    select_template,
)


class _FakeProvider:
    def __init__(self, watches: list[dict], findings: list[dict]) -> None:
        self._watches = watches
        self._findings = findings

    async def watches(self) -> list[dict[str, Any]]:
        return list(self._watches)

    async def findings(self, *, watch_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        items = [f for f in self._findings if f.get("watch_id") == watch_id] if watch_id else list(self._findings)
        return items[:limit]


def _flatten(spec: dict) -> list[dict]:
    flat: list[dict] = []

    def _walk(nodes: list) -> None:
        for node in nodes:
            flat.append(node)
            _walk(node.get("children") or [])

    _walk(spec["root"])
    return flat


def _session_provider() -> _FakeProvider:
    return _FakeProvider(
        watches=[{"watch_id": "s", "domain": "session", "state": "armed",
                  "last_run_at": 10.0, "next_due_at": 20.0, "run_count": 2}],
        findings=[
            {"finding_id": "s1", "watch_id": "s", "domain": "session", "title": "analysis",
             "severity": "notable", "payload": {
                 "story": "the arc",
                 "insights": [{"title": "i", "summary": "s", "severity": "notable"}],
                 "next_prompts": ["p"],
                 "observation_status": {"refresh_reason": "artifact_changed", "context_scope": "text_and_artifacts"},
                 "artifact_context": [{"name": "report.md", "status": "included"}],
             }},
            {"finding_id": "x1", "watch_id": "w", "domain": "finance", "title": "noise", "severity": "info"},
        ],
    )


# -- select_template: requested lens, else generic fallback -------------------


def test_select_template_returns_requested_or_generic_fallback() -> None:
    names = ["generic", "finance", "research"]
    assert select_template("finance", names) == "finance"
    assert select_template("", names) == "generic"
    assert select_template("unknown", names) == "generic"


# -- DashboardViewBuilder: one target (current session), template = lens ------


async def test_builder_default_template_renders_session_analysis() -> None:
    builder = DashboardViewBuilder(TemplateLibrary())
    spec = await builder.build(DashboardIntent(template=""), _session_provider())
    assert spec["title"] == "Session Analysis"
    types = {n["type"] for n in _flatten(spec)}
    assert {"StoryPanel", "BarChart", "EntityGraph", "Table"}.issubset(types)
    assert len([n for n in _flatten(spec) if n["type"] == "InsightCard"]) == 1


async def test_builder_named_template_reframes_same_session() -> None:
    builder = DashboardViewBuilder(TemplateLibrary())
    spec = await builder.build(DashboardIntent(template="finance"), _session_provider())
    types = {n["type"] for n in _flatten(spec)}
    # The finance lens renders the same session analysis, reframed.
    assert "StoryPanel" in types and "EntityGraph" in types


async def test_builder_unknown_template_falls_back_to_generic() -> None:
    builder = DashboardViewBuilder(TemplateLibrary())
    spec = await builder.build(DashboardIntent(template="does-not-exist"), _session_provider())
    assert spec["title"] == "Session Analysis"


async def test_builder_exposes_template_switcher_meta() -> None:
    # The web client renders its lens switcher from this meta (no hardcoding).
    builder = DashboardViewBuilder(TemplateLibrary())
    spec = await builder.build(DashboardIntent(template="finance"), _session_provider())
    assert spec["meta"]["active_template"] == "finance"
    assert {"generic", "finance", "sentiment", "research"}.issubset(set(spec["meta"]["templates"]))


# -- ViewHub fan-out ----------------------------------------------------------


async def test_view_hub_broadcast_and_unsubscribe() -> None:
    hub = ViewHub()
    queue = hub.subscribe("a")
    assert hub.broadcast({"type": "monitor.finding", "payload": {"x": 1}}) == 1
    assert (await queue.get())["type"] == "monitor.finding"
    hub.unsubscribe("a")
    assert hub.broadcast({"type": "x"}) == 0
    assert hub.subscriber_count == 0


async def test_view_hub_backpressure_drops_when_full() -> None:
    hub = ViewHub(maxsize=1)
    hub.subscribe("slow")
    assert hub.broadcast({"n": 1}) == 1
    assert hub.broadcast({"n": 2}) == 0  # queue full -> dropped, not blocked
    await hub.shutdown()
    assert hub.subscriber_count == 0
