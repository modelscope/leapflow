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
    def __init__(
        self,
        watches: list[dict],
        findings: list[dict],
        signal_result: dict[str, Any] | None = None,
    ) -> None:
        self._watches = watches
        self._findings = findings
        self._signal_result = signal_result or {"metrics": {}, "signal_stream": []}

    async def watches(self) -> list[dict[str, Any]]:
        return list(self._watches)

    async def findings(self, *, watch_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        items = [f for f in self._findings if f.get("watch_id") == watch_id] if watch_id else list(self._findings)
        return items[:limit]

    async def signal_metrics(self) -> dict[str, Any]:
        return dict(self._signal_result)


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
    # Only visible (non-hidden) templates appear in the switcher.
    builder = DashboardViewBuilder(TemplateLibrary())
    spec = await builder.build(DashboardIntent(template="finance"), _session_provider())
    assert spec["meta"]["active_template"] == "finance"
    assert {"generic", "signals"}.issubset(set(spec["meta"]["templates"]))
    # Hidden templates must NOT appear in the switcher, and are exposed as a
    # client-side deny list so stale/custom metadata cannot re-render them.
    assert "finance" not in spec["meta"]["templates"]
    assert "research" not in spec["meta"]["templates"]
    assert "sentiment" not in spec["meta"]["templates"]
    assert {"finance", "research", "sentiment"}.issubset(set(spec["meta"]["hidden_templates"]))


async def test_builder_signals_template_renders_dense_operational_layout() -> None:
    builder = DashboardViewBuilder(TemplateLibrary())
    provider = _FakeProvider(
        watches=[
            {"watch_id": "abcdef123", "name": "Session", "domain": "session", "trigger": "every 2m", "state": "armed", "finding_count": 1},
            {"watch_id": "sig123", "name": "fs-observer", "domain": "signal", "trigger": "event:fs.*", "state": "done", "finding_count": 0},
        ],
        findings=[{"finding_id": "f1", "watch_id": "abcdef123", "domain": "session", "title": "analysis", "summary": "s", "severity": "notable"}],
        signal_result={
            "metrics": {
                "event_subscriber_count": 6,
                "active_trigger_count": 2,
                "active_watch_count": 1,
                "recent_findings_count": 1,
                "signal_noise_suppressed": 9,
                "signal_buffer_dropped": 3,
                "composite_source_dropped": 4,
                "reorder_buffer_pending": 2,
                "debounce_stats": {"sig123": 5},
                "trigger_stats": [{"watch_id": "sig123456", "pattern": "fs.*", "triggered": True, "last_event": "fs.modified"}],
            },
            "signal_stream": [
                {"event_type": "fs.modified", "source": "fs-old", "ts": 100.0},
                {"event_type": "gateway.message", "source": "gateway-new", "ts": 300.0},
                {"event_type": "clipboard.change", "source": "clipboard-mid", "ts": 200.0},
            ],
        },
    )

    spec = await builder.build(DashboardIntent(template="signals"), provider)

    assert spec["meta"]["active_template"] == "signals"
    flat = _flatten(spec)
    stats = [n for n in flat if n["type"] == "Stat"]
    labels = {n["props"]["label"] for n in stats}
    assert {
        "Subscribers",
        "Active triggers",
        "Active watches",
        "Stream events",
        "Recent findings",
        "Noise suppressed",
        "Buffer dropped",
        "Source dropped",
        "Reorder pending",
        "Debounced",
    }.issubset(labels)
    assert len(stats) >= 9
    signal_widget = next(
        n for n in flat
        if n["type"] == "Custom" and n["props"].get("render") == "signalTimeline"
    )
    stream = signal_widget["props"]["data"]
    assert signal_widget["props"]["max_items"] == 12
    assert [item["event_type"] for item in stream] == [
        "gateway.message",
        "clipboard.change",
        "fs.modified",
    ]
    assert [item["family"] for item in stream] == ["gateway", "clipboard", "fs"]
    assert stream[0]["source"] == "gateway-new"
    assert stream[0]["ts"] == 300.0
    assert len([n for n in flat if n["type"] == "BarChart"]) >= 2
    assert any(n["type"] == "Section" and n["props"].get("span") == 2 for n in flat)
    trigger_table = next(
        n for n in flat
        if n["type"] == "Table" and any(c.get("key") == "pattern" for c in n["props"].get("columns", []))
    )
    assert trigger_table["props"]["data"][0]["watch"] == "sig12345"


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


# ── An empty hardware board must say why ────────────────────────────────────
#
# Every panel on the hardware lens gates on the data it renders, which is what lets a new
# peripheral appear with no template edit. The cost is that *no* data renders *no* panels:
# the board showed a bare "Hardware" heading over an empty page, which answers none of the
# three questions a person actually has -- is it off, is it broken, or is nothing plugged
# in. Each branch below names the next step, because only the message can tell them apart.


def _fleet_spec(inventory: dict, digest: dict | None = None) -> dict:
    import asyncio

    from leapflow.dashboard.intent import DashboardIntent
    from leapflow.dashboard.service import DashboardViewBuilder

    class _Provider:
        async def watches(self) -> list:
            return []

        async def findings(self, *, watch_id: str = "", limit: int = 50) -> list:
            if digest is None:
                return []
            return [{"domain": "hardware", "severity": "info", "payload": digest}]

        async def signal_metrics(self) -> dict:
            return {}

        async def hardware_inventory(self) -> dict:
            return inventory

        async def hardware_device(self, device_id: str) -> dict:
            return {"ok": False, "code": "unknown_device"}

    builder = DashboardViewBuilder()
    return asyncio.run(builder.build(DashboardIntent(template="hardware"), _Provider()))


def _cards(spec: dict) -> list[tuple[str, str]]:
    """Return (title, body) for every Card that carries prose."""
    found: list[tuple[str, str]] = []

    def walk(nodes: list) -> None:
        for node in nodes:
            children = node.get("children") or []
            if node.get("type") == "Card":
                text = " ".join(
                    str((child.get("props") or {}).get("text", ""))
                    for child in children
                    if child.get("type") == "Markdown"
                )
                if text:
                    found.append((str((node.get("props") or {}).get("title", "")), text))
            walk(children)

    walk(spec.get("root") or [])
    return found


def _section_titles(spec: dict) -> list[str]:
    """Return every Section heading that survived its ``when`` gate."""
    titles: list[str] = []

    def walk(nodes: list) -> None:
        for node in nodes:
            if node.get("type") == "Section":
                title = str((node.get("props") or {}).get("title", ""))
                if title:
                    titles.append(title)
            walk(node.get("children") or [])

    walk(spec.get("root") or [])
    return titles


def test_a_disabled_hardware_subsystem_says_so_and_how_to_turn_it_on() -> None:
    spec = _fleet_spec({
        "ok": False, "code": "hardware_disabled",
        "error": "Hardware is disabled for this profile.",
    })

    cards = _cards(spec)
    assert cards, "a disabled subsystem rendered an empty page instead of an explanation"
    title, body = cards[0]
    assert "disabled" in title.lower()
    assert "hardware.enabled" in body, "the message must name the setting that changes it"
    assert "restart" in body.lower(), "hardware.* is restart-required and the message must say so"


def test_an_empty_bench_distinguishes_itself_from_a_broken_one() -> None:
    """Enabled with nothing attached is a different situation, and gets a different answer."""
    spec = _fleet_spec({"ok": True, "groups": [], "devices": [], "counts": {"devices": 0}})

    title, body = _cards(spec)[0]
    assert "no devices" in title.lower()
    assert "rescan" in body, "the next step for an empty bench is to scan again"
    assert "hardware.enabled" not in body, (
        "hardware is on here; telling the user to enable it would send them in a circle"
    )


def test_an_unreachable_subsystem_points_at_the_board_process() -> None:
    """The board is separately launched and long-lived, so its own age is a real cause."""
    spec = _fleet_spec({
        "ok": False, "code": "rpc_unavailable", "error": "This leapd build has no hardware RPC.",
    })

    title, body = _cards(spec)[0]
    assert "cannot reach" in title.lower()
    assert "This leapd build has no hardware RPC." in body, "the underlying error must survive"
    assert "leap board" in body


def test_a_bench_with_devices_gets_panels_and_no_notice() -> None:
    """The notice is for emptiness only: it must not sit above a page that has content."""
    spec = _fleet_spec({
        "ok": True,
        "counts": {"devices": 1, "channels": 2, "previewable": 1},
        "groups": [{"device_class": "camera", "devices": [{"device_id": "cam", "label": "Cam"}]}],
        "devices": [{"device_id": "cam"}],
    })

    assert _cards(spec) == [], "a populated bench must not be captioned as empty"
    titles = _section_titles(spec)
    assert "Attached devices" in titles


def test_the_fleet_renders_from_inventory_alone() -> None:
    """No synthetic mode flag: whatever data arrives, renders.

    This is the regression guard for the blank board. The panels used to gate on a ``fleet``
    key that only the newest builder supplied, so a board process serving fresh templates
    with older code produced a page with a title and nothing under it. Gating on the real
    data removes the whole class: an inventory with no digest still shows the devices.
    """
    spec = _fleet_spec({
        "ok": True,
        "counts": {"devices": 1, "channels": 2},
        "groups": [{"device_class": "camera", "devices": [{"device_id": "cam", "label": "Cam"}]}],
        "devices": [{"device_id": "cam"}],
    })

    assert "Attached devices" in _section_titles(spec)


def test_only_actionable_admission_notes_reach_the_board() -> None:
    """A V10 privacy warning is emitted per gated channel by design, not as a problem.

    Bound unfiltered, three cameras produced "Admission decisions: 3" under a heading that
    promises to explain why a device is not what it declared -- when nothing was demoted.
    """
    from leapflow.dashboard.service import _actionable_notes

    notes = _actionable_notes({"notes": [
        {"device_id": "cam", "rule": "V10", "outcome": "warning", "detail": "privacy"},
        {"device_id": "arm", "rule": "V4", "outcome": "demoted", "detail": "unverified"},
        {"device_id": "bad", "rule": "V2", "outcome": "rejected", "detail": "bad id"},
    ]})

    assert [n["outcome"] for n in notes] == ["demoted", "rejected"]
    assert _actionable_notes({}) == []
