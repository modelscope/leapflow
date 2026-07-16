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


def _finding_cards(spec: dict) -> list[dict]:
    flat: list[dict] = []

    def _walk(nodes: list) -> None:
        for node in nodes:
            flat.append(node)
            _walk(node.get("children") or [])

    _walk(spec["root"])
    return [n for n in flat if n["type"] == "FindingCard"]


# ── select_template convention ─────────────────────────────────────────────


def test_select_template_prefers_explicit_then_domain_then_generic() -> None:
    names = ["generic", "finance.market", "session.analysis"]
    assert select_template("finance", "", names) == "finance.market"
    assert select_template("", "session.analysis", names) == "session.analysis"
    assert select_template("unknown", "", names) == "generic"
    assert select_template("finance", "missing", names) == "finance.market"


# ── DashboardViewBuilder ────────────────────────────────────────────────────


async def test_builder_overview_renders_findings() -> None:
    provider = _FakeProvider(
        watches=[{"watch_id": "w1", "name": "M", "domain": "finance", "state": "armed"}],
        findings=[{"finding_id": "f1", "watch_id": "w1", "domain": "finance", "title": "spike", "severity": "alert"}],
    )
    builder = DashboardViewBuilder(TemplateLibrary())
    spec = await builder.build(DashboardIntent(action="home"), provider)
    assert spec["title"] == "LeapFlow Monitors"
    assert len(_finding_cards(spec)) == 1


async def test_builder_overview_lists_watch_lanes() -> None:
    provider = _FakeProvider(
        watches=[{"watch_id": "w1", "name": "Market", "domain": "finance",
                  "trigger": "every 5m", "state": "armed", "finding_count": 3}],
        findings=[],
    )
    builder = DashboardViewBuilder(TemplateLibrary())
    spec = await builder.build(DashboardIntent(action="home"), provider)

    flat: list[dict] = []

    def _walk(nodes: list) -> None:
        for node in nodes:
            flat.append(node)
            _walk(node.get("children") or [])

    _walk(spec["root"])
    watch_cards = [n for n in flat if n["type"] == "Card" and n.get("action", {}).get("name") == "openWatch"]
    assert len(watch_cards) == 1
    assert watch_cards[0]["action"]["params"]["target"] == "w1"


async def test_builder_watch_scopes_to_target() -> None:
    provider = _FakeProvider(
        watches=[{"watch_id": "abc123", "name": "Market", "domain": "finance", "state": "armed"}],
        findings=[{"finding_id": "f1", "watch_id": "abc123", "domain": "finance", "title": "t", "severity": "notable"}],
    )
    builder = DashboardViewBuilder(TemplateLibrary())
    spec = await builder.build(DashboardIntent(action="open", target="abc1"), provider)
    assert spec["title"] == "Market"
    assert len(_finding_cards(spec)) == 1


async def test_builder_session_uses_analysis_payload() -> None:
    provider = _FakeProvider(
        watches=[],
        findings=[
            {"finding_id": "s1", "watch_id": "s", "domain": "session", "title": "analysis",
             "severity": "notable", "payload": {
                 "story": "the arc",
                 "insights": [{"title": "i", "summary": "s", "severity": "notable"}],
                 "next_prompts": ["p"],
             }},
            {"finding_id": "x1", "watch_id": "w", "domain": "finance", "title": "noise", "severity": "info"},
        ],
    )
    builder = DashboardViewBuilder(TemplateLibrary())
    spec = await builder.build(DashboardIntent(action="session"), provider)
    assert spec["title"] == "Session Analysis"

    flat: list[dict] = []

    def _walk(nodes: list) -> None:
        for node in nodes:
            flat.append(node)
            _walk(node.get("children") or [])

    _walk(spec["root"])
    types = {n["type"] for n in flat}
    assert "StoryPanel" in types
    assert len([n for n in flat if n["type"] == "InsightCard"]) == 1  # from analysis payload


# ── ViewHub fan-out ─────────────────────────────────────────────────────────


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
