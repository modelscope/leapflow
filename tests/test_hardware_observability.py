"""The physical-bench board: contract, derivation, wiring, and what it renders.

Grouped by the claim each assertion defends rather than by module, because the
failures worth catching here are connection failures. The board this replaces did
not exist; the board next to it, ``capability``, existed in full -- template,
producer, registered watch -- and rendered every value blank for want of one
dispatch branch. So the tests that matter most are the ones that follow data all
the way to a rendered node.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.dashboard.intent import DashboardIntent
from leapflow.dashboard.service import DashboardViewBuilder
from leapflow.dashboard.templates import TemplateLibrary
from leapflow.hardware.observability import (
    MAX_POINTS,
    MAX_SERIES,
    SERIES_SCHEMA_VERSION,
    WALL_CLOCK,
    ChannelSeries,
    HardwareObservationProducer,
    SeriesPoint,
    build_digest,
)
from leapflow.hardware.observability.series import clamp_series, decimate

_WALL = 1_787_000_000.0


# ════════════════════════════════════════════════════════════════
# Fakes shaped like the real registry surface
# ════════════════════════════════════════════════════════════════


def _envelope(**kwargs: Any) -> Any:
    defaults = {
        "declared": True,
        "min_value": 0.0,
        "max_value": 100.0,
        "quantization": 0.5,
        "notes": "",
    }
    return SimpleNamespace(**{**defaults, **kwargs})


def _channel(channel_id: str = "level", **kwargs: Any) -> Any:
    defaults = {
        "channel_id": channel_id,
        "quantity": "generic.level",
        "unit": "C",
        "sample_rate_hz": 10.0,
        "is_readable": True,
        "is_writable": True,
        "envelope": _envelope(),
    }
    return SimpleNamespace(**{**defaults, **kwargs})


def _context(channels: tuple[Any, ...] | None = None, **kwargs: Any) -> Any:
    chans = channels if channels is not None else (_channel(),)
    defaults = {
        "device_id": "bench",
        "display_name": "Bench node",
        "location": "lab-2",
        "halt_supported": True,
        "channels": chans,
        "writable_channels": tuple(c for c in chans if c.is_writable),
        "transport": SimpleNamespace(kind="mock"),
        "provenance": SimpleNamespace(verified_by="tester"),
    }
    return SimpleNamespace(**{**defaults, **kwargs})


def _event(kind: str = "threshold_exceeded", **kwargs: Any) -> Any:
    defaults = {
        "kind": kind,
        "device_id": "bench",
        "channel_id": "level",
        "detail": "left the declared range (0..100)",
        "value": 140.5,
        "unit": "C",
        "observed_at": _WALL,
    }
    return SimpleNamespace(**{**defaults, **kwargs})


class _Registry:
    """Answers the read-only questions the digest asks, and nothing else."""

    def __init__(
        self,
        *,
        contexts: tuple[Any, ...] | None = None,
        windows: list[dict[str, Any]] | None = None,
        events: tuple[Any, ...] = (),
        health: dict[str, Any] | None = None,
        store: Any | None = None,
        recorder: Any | None = None,
    ) -> None:
        self._contexts = contexts if contexts is not None else (_context(),)
        self._windows = windows if windows is not None else _windows()
        self._events = events
        self._health = health
        self.reading_store = store
        self.outcome_recorder = recorder
        self.read_calls = 0

    def contexts(self) -> tuple[Any, ...]:
        return self._contexts

    def opened_devices(self) -> tuple[str, ...]:
        return ("bench",)

    def channel_history(self, device_id: str, channel_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        return list(self._windows)

    def recent_events(self, device_id: str = "", limit: int = 10) -> tuple[Any, ...]:
        return self._events

    def stream_sources(self) -> tuple[Any, ...]:
        if self._health is None:
            return ()
        return (SimpleNamespace(health=dict(self._health), ring=None),)

    async def transport(self, device_id: str) -> Any:
        raise AssertionError("the digest must never open a transport")


def _windows(count: int = 20, breach_at: int | None = 7) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        high = 105.0 if index == breach_at else 42.0 + index
        rows.append({
            "ended_at": _WALL - 60 * (count - index),
            "mean_value": 40.0 + index,
            "min_value": 38.0 + index,
            "max_value": high,
            "samples": 600,
            "dropped": 0,
            "quality_worst": "ok",
        })
    return rows


def _producer_ctx(now: float = _WALL) -> Any:
    return SimpleNamespace(
        spec=SimpleNamespace(watch_id="w1", params={}),
        now=now,
        run_count=1,
        last_run_at=0.0,
        services=None,
        force=False,
    )


# ════════════════════════════════════════════════════════════════
# The payload contract is versioned, bounded, and states its clock
# ════════════════════════════════════════════════════════════════


def test_payload_declares_its_version_and_clock() -> None:
    """A renderer must be able to refuse a shape it does not understand.

    The clock is stated rather than assumed because ``leapflow.hardware`` carries two
    of them and only one can go on a time axis. A chart drawn from the other looks
    entirely normal while being wrong by decades, so the axis has to be able to
    check.
    """
    payload = build_digest(_Registry()).to_payload()
    assert payload["schema_version"] == SERIES_SCHEMA_VERSION
    assert payload["clock"] == WALL_CLOCK
    # Every x is a wall-clock epoch, not a value off the monotonic clock the
    # subsystem also carries. A wall-clock instant is decades larger than any
    # monotonic reading (seconds since boot), so it must sit above the live
    # monotonic clock -- a relative floor that never expires, unlike a hardcoded
    # epoch. A monotonic sample plotted on this axis would fail here while looking
    # entirely normal: the "wrong by decades" mistake the axis has to catch.
    monotonic_now = time.monotonic()
    assert all(point["x"] > monotonic_now for point in payload["series"][0]["points"])


def test_counts_are_precomputed_because_there_is_no_length_path() -> None:
    """The template resolver walks mapping keys and indices only.

    A template asking for ``.length`` gets ``None`` and renders an empty value, with
    nothing to indicate the path was never supported.
    """
    payload = build_digest(_Registry(events=(_event(),))).to_payload()
    assert payload["counts"]["devices"] == 1
    assert payload["counts"]["series"] == 1
    assert payload["counts"]["events"] == 1


def test_series_are_capped_worst_quality_first() -> None:
    """When the cap bites, a healthy channel is the one that goes.

    Dropping by name or arrival order would sometimes discard exactly the channel
    somebody opened the board to look at.
    """
    healthy = [
        ChannelSeries(id=f"dev.ok{i}", label=f"ok{i}", quality_worst="ok")
        for i in range(MAX_SERIES + 3)
    ]
    degraded = ChannelSeries(id="dev.bad", label="bad", quality_worst="saturated")
    kept = clamp_series([*healthy, degraded])
    assert len(kept) == MAX_SERIES
    assert "dev.bad" in {series.id for series in kept}


def test_long_series_are_decimated_not_truncated() -> None:
    """Both ends must survive thinning.

    Cutting the tail hides the present and cutting the head hides the baseline;
    either turns the chart into a claim about a window the axis does not describe.
    """
    points = tuple(SeriesPoint(x=float(i), y=float(i)) for i in range(MAX_POINTS * 3))
    thinned = decimate(points)
    assert len(thinned) == MAX_POINTS
    assert thinned[0] is points[0]
    assert thinned[-1] is points[-1]


def test_an_oversized_payload_is_reduced_to_fit() -> None:
    """The payload is persisted, pushed and ring-buffered, so it cannot be unbounded."""
    from leapflow.hardware.observability.series import MAX_PAYLOAD_BYTES, HardwareDigest

    fat = tuple(
        ChannelSeries(
            id=f"dev.c{index}",
            label=f"channel {index}",
            points=tuple(SeriesPoint(x=_WALL + i, y=float(i)) for i in range(MAX_POINTS)),
        )
        for index in range(MAX_SERIES)
    )
    payload = HardwareDigest(generated_at=_WALL, series=fat, devices=({"device_id": "d"},)).to_payload()
    import json

    assert len(json.dumps(payload).encode("utf-8")) <= MAX_PAYLOAD_BYTES


# ════════════════════════════════════════════════════════════════
# Derivation: every value comes from the declaration or the store
# ════════════════════════════════════════════════════════════════


def test_conformance_is_judged_on_the_window_extremes_not_the_mean() -> None:
    """An excursion that averages back inside the band still left it.

    The mean is precisely what hides it, which is why the storage tier keeps
    ``min``/``max`` at all. One injected window peaks at 105 against a declared
    maximum of 100 while its mean stays well inside.
    """
    digest = build_digest(_Registry())
    states = [row["state"] for row in digest.conformance]
    assert states.count("outside") == 1
    assert states.count("inside") == len(states) - 1


def test_near_the_boundary_is_distinguished_from_inside_it() -> None:
    """Approaching a limit and sitting inside one are different facts.

    A two-state view cannot express the difference, which is the reason somebody
    watches a trace rather than a boolean.
    """
    edge = _windows(count=3, breach_at=None)
    edge[1]["max_value"] = 99.0  # within 5% of the declared maximum of 100
    digest = build_digest(_Registry(windows=edge))
    assert "near" in {row["state"] for row in digest.conformance}


def test_conformance_is_unknown_without_a_declared_envelope() -> None:
    """An undeclared channel has no band, so no window can be judged against one."""
    undeclared = _context(channels=(_channel(envelope=_envelope(declared=False)),))
    digest = build_digest(_Registry(contexts=(undeclared,)))
    assert {row["state"] for row in digest.conformance} == {"unknown"}


def test_the_digest_never_touches_a_transport() -> None:
    """A board refresh must not become a reason the device bus is busy.

    The fake registry raises from ``transport``; reaching it at all is the failure.
    """
    digest = build_digest(_Registry(events=(_event(),)))
    assert digest.devices[0]["device_id"] == "bench"


def test_storage_health_reports_failures_not_only_successes() -> None:
    """``windows_written`` alone is a numerator with no denominator.

    A database that cannot be opened looks exactly like an idle bench, which is why
    the failure count is the only way the fault is ever noticed.
    """
    store = SimpleNamespace(
        raw_writes=120, windows_written=4, write_failures=3, rows_pruned=0, pending_channels=1
    )
    digest = build_digest(_Registry(store=store))
    assert digest.storage == {
        "persisting": True,
        "raw_writes": 120,
        "windows_written": 4,
        "write_failures": 3,
        "rows_pruned": 0,
        "pending_channels": 1,
    }


def test_learned_command_outcomes_reach_the_digest() -> None:
    """Until this panel, recalled experience only ever reached the model."""
    recorder = SimpleNamespace(
        recall=lambda **_: ({"command": "set level to 42 C", "outcome": "reached 41.6 C", "delta": 0.004},)
    )
    digest = build_digest(_Registry(recorder=recorder))
    assert digest.outcomes[0]["command"] == "set level to 42 C"
    assert digest.outcomes[0]["delta"] == pytest.approx(0.004)


def test_a_registry_that_cannot_answer_yields_a_partial_digest() -> None:
    """A missing section beats a missing board, and a raising watch stops the cycle."""

    class _Broken(_Registry):
        def recent_events(self, device_id: str = "", limit: int = 10) -> tuple[Any, ...]:
            raise RuntimeError("event ring unavailable")

    digest = build_digest(_Broken())
    assert digest.events == ()
    assert digest.series, "one broken section must not lose the rest"


# ════════════════════════════════════════════════════════════════
# The producer: severity drives whether anyone is told
# ════════════════════════════════════════════════════════════════


def _observe(registry: Any) -> Any:
    producer = HardwareObservationProducer(lambda: registry)
    findings = asyncio.run(producer.observe(_producer_ctx()))
    return findings


def test_an_envelope_event_is_an_alert() -> None:
    """Severity decides push versus persist, so a breach has to reach someone."""
    findings = _observe(_Registry(events=(_event("threshold_exceeded"),)))
    assert len(findings) == 1
    assert findings[0].severity.value == "alert"
    assert findings[0].payload["clock"] == WALL_CLOCK


def test_a_recovery_alone_does_not_raise_an_alert() -> None:
    """``settled`` is good news; colouring it like a breach teaches people to ignore
    the colour."""
    findings = _observe(_Registry(events=(_event("settled"),)))
    assert findings[0].severity.value == "info"


def test_unpersisted_windows_are_notable_on_their_own() -> None:
    """Dropped windows leave no trace in the data, so the count must speak up."""
    store = SimpleNamespace(raw_writes=1, windows_written=0, write_failures=2, pending_channels=0)
    findings = _observe(_Registry(store=store))
    assert findings[0].severity.value == "notable"
    assert "not being persisted" in findings[0].title


def test_a_cadence_shortfall_is_notable() -> None:
    """The stored series looks correct when a channel runs at two thirds of its rate."""
    health = {"channel_id": "level", "declared_hz": 10.0, "observed_hz": 6.4, "rate_ratio": 0.64}
    findings = _observe(_Registry(health=health))
    assert findings[0].severity.value == "notable"
    assert "behind declared rate" in findings[0].title


def test_jitter_within_a_fifth_is_not_reported_as_drift() -> None:
    """Scheduling jitter is not a defect; reporting it would bury the real thing."""
    health = {"channel_id": "level", "declared_hz": 10.0, "observed_hz": 9.5, "rate_ratio": 0.95}
    findings = _observe(_Registry(health=health))
    assert findings[0].severity.value == "info"


def test_no_devices_produces_no_finding() -> None:
    """Hardware is off by default, and a store full of "nothing happened" is noise."""
    assert _observe(_Registry(contexts=())) == []
    assert HardwareObservationProducer(lambda: None) and asyncio.run(
        HardwareObservationProducer(lambda: None).observe(_producer_ctx())
    ) == []


def test_a_raising_provider_does_not_fail_the_watch() -> None:
    """One panel is not worth stopping the monitor cycle for."""

    def _boom() -> Any:
        raise RuntimeError("registry gone")

    assert asyncio.run(HardwareObservationProducer(_boom).observe(_producer_ctx())) == []


def test_dedup_key_tracks_bench_state_not_the_trace() -> None:
    """A trace changes every cycle; deduping on it would notify on every cycle.

    Keyed on what a watcher reacts to -- degraded channels, live event kinds, whether
    persistence is failing -- so an unchanged bench stays quiet.
    """
    first = _observe(_Registry(events=(_event(),)))[0]
    shifted = _windows(count=20, breach_at=7)
    for row in shifted:
        row["mean_value"] += 5.0
    second = _observe(_Registry(windows=shifted, events=(_event(),)))[0]
    assert first.dedup_key == second.dedup_key

    changed = _observe(_Registry(events=(_event("stale"),)))[0]
    assert changed.dedup_key != first.dedup_key


# ════════════════════════════════════════════════════════════════
# Wiring: the board is reachable and actually renders the payload
# ════════════════════════════════════════════════════════════════


def test_the_hardware_template_is_discoverable() -> None:
    names = TemplateLibrary().names()
    assert "hardware" in names


def test_hardware_renders_the_finding_payload_not_the_session() -> None:
    """The assertion the ``capability`` board never had.

    Its template was valid, its producer registered, its watch armed -- and every
    value rendered blank, because the builder routed it to the session path where no
    ``capability_plan`` key exists. Following one real value to a rendered node is
    the only check that catches that.
    """
    finding = _observe(_Registry(events=(_event(),), store=SimpleNamespace(
        raw_writes=120, windows_written=4, write_failures=0, pending_channels=0)))[0]

    class _Provider:
        async def watches(self) -> list[dict[str, Any]]:
            return [{"domain": "hardware", "state": "watching", "name": "hardware-bench"}]

        async def findings(self, *, watch_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
            return [{"domain": "hardware", "severity": "alert", "payload": finding.payload}]

        async def signal_metrics(self) -> dict[str, Any]:
            return {"metrics": {}, "signal_stream": []}

        async def hardware_inventory(self) -> dict[str, Any]:
            # The fleet list is read live rather than taken from the cycle payload, so
            # the device count on the board comes from here and not from the digest.
            return {
                "ok": True,
                "groups": [{"device_class": "bench", "devices": [{"device_id": "dev"}]}],
                "devices": [{"device_id": "dev"}],
                "counts": {"devices": 1, "channels": 3, "previewable": 0},
                "notes": [],
            }

        async def hardware_device(self, device_id: str) -> dict[str, Any]:
            return {"ok": False, "code": "unknown_device"}

    spec = asyncio.run(DashboardViewBuilder().build(DashboardIntent(template="hardware"), _Provider()))
    stats = _nodes_of_type(spec, "Stat")
    values = {node["props"].get("label"): node["props"].get("value") for node in stats}
    assert values["Devices"] == 1
    assert values["Raw samples written"] == 120
    assert values["Watch state"] == "watching"


def _implemented_renderers() -> set[str]:
    """Return the component types the shipped frontend can actually draw.

    Read out of ``app.js`` rather than the catalog because the two are different
    things: a catalog entry with no renderer degrades to a card printing its own type
    name, which is how ``Heatmap`` sat in ``COMPONENT_CATALOG`` for months while every
    template asking for one got a fallback.
    """
    from pathlib import Path
    import re

    app_js = Path(__file__).parents[1] / "src" / "leapflow" / "dashboard" / "static" / "app.js"
    source = app_js.read_text(encoding="utf-8")
    block = source.split("const RENDERERS = {", 1)[1]
    return set(re.findall(r"^\s{4}([A-Za-z]+):", block, flags=re.MULTILINE))


def test_every_rendered_component_has_a_frontend_renderer() -> None:
    """Catalog membership is not implementation.

    ``Heatmap`` is in ``COMPONENT_CATALOG`` and had no renderer in ``app.js``, so a
    template asking for one got a fallback card printing its own type name. This
    asserts against the renderer table in the shipped JS, not the catalog.
    """
    implemented = _implemented_renderers()
    spec = TemplateLibrary().render("hardware", {"hardware": _rich_payload(), "observation": {}})
    used = {node["type"] for node in _walk(spec.get("root", []))}
    missing = sorted(used - implemented)
    assert not missing, f"hardware.yaml uses components with no frontend renderer: {missing}"


def test_no_shipped_template_references_an_unrenderable_component() -> None:
    """The general form of the check above, across every template in the package.

    Per-template coverage was the gap: ``hardware.yaml`` was checked and the rest were
    not, so a new lens could reference a catalog type with no renderer and ship looking
    valid. Rendered with empty data on purpose -- ``when:`` gates would otherwise hide
    most nodes, and an unrenderable component is a defect whether or not data happens to
    reach it today.
    """
    library = TemplateLibrary()
    implemented = _implemented_renderers()
    problems: list[str] = []
    for name in library.names():
        raw = library.load(name)
        assert raw is not None, f"{name}: template did not load"
        # Rendered without ``when`` gating so every declared node is inspected, not just
        # the ones a particular data shape happens to reveal.
        for node in _walk(_ungated(raw).get("layout") or []):
            ntype = str(node.get("type") or "")
            if ntype and ntype not in implemented:
                problems.append(f"{name}: {ntype}")
    assert not problems, (
        "templates reference components with no renderer in app.js: " + ", ".join(sorted(set(problems)))
    )


def _ungated(node: Any) -> Any:
    """Return *node* with every ``when``/``repeat`` directive stripped.

    So the walk sees the template's full component vocabulary rather than the subset a
    given payload unlocks.
    """
    if isinstance(node, dict):
        return {
            key: _ungated(value)
            for key, value in node.items()
            if key not in ("when", "repeat", "as")
        }
    if isinstance(node, list):
        return [_ungated(item) for item in node]
    return node


def test_the_board_never_carries_an_action_that_touches_a_device() -> None:
    """Refined from "no actions at all", which is no longer the invariant.

    The board now navigates: a fleet row opens that device's page. What must stay true
    is the reason the original rule existed -- a browser session is a weaker identity
    than the TUI process that holds the approval route, so nothing the board can click
    may reach a device without the approval chain. Navigation cannot; a write cannot be
    expressed here at all.

    Asserted over both action forms, because a per-row action lives inside ``props`` and
    the node-level check alone would not see it.
    """
    spec = TemplateLibrary().render("hardware", {"hardware": _rich_payload(), "observation": {}})
    offending: list[str] = []
    for node in _walk(spec.get("root", [])):
        for action in (node.get("action"), (node.get("props") or {}).get("row_action")):
            if isinstance(action, dict) and str(action.get("kind")) != "nav":
                offending.append(f"{node.get('type')}:{action.get('kind')}")
    assert not offending, (
        "the fleet board may only navigate; found non-nav actions on " + ", ".join(offending)
    )


def test_capability_now_receives_its_own_payload() -> None:
    """The pre-existing break the payload-domain table also fixes."""

    class _Provider:
        async def watches(self) -> list[dict[str, Any]]:
            return [{"domain": "capability_adaptation", "state": "armed"}]

        async def findings(self, *, watch_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
            return [{"domain": "capability_adaptation", "payload": {"phase": "executable"}}]

        async def signal_metrics(self) -> dict[str, Any]:
            return {"metrics": {}, "signal_stream": []}

    spec = asyncio.run(DashboardViewBuilder().build(DashboardIntent(template="capability"), _Provider()))
    values = {
        node["props"].get("label"): node["props"].get("value")
        for node in _nodes_of_type(spec, "Stat")
    }
    assert values.get("Loop phase") == "executable"


@pytest.mark.asyncio
async def test_the_hardware_producer_is_registered_only_when_hardware_is_enabled(
    tmp_path: Any,
) -> None:
    """Registration is conditional, and asserted on the registry the daemon built.

    Both directions matter: enabled must register, and disabled must not, or a
    profile with no devices runs a producer every cycle to conclude there is nothing
    to report.
    """
    from leapflow.daemon.monitor_coordinator import MonitorCoordinator
    from leapflow.storage.connection import LocalConnectionHolder

    async def _domains(*, hardware_enabled: bool) -> list[str]:
        holder = LocalConnectionHolder(tmp_path / f"hw-{hardware_enabled}.duckdb")
        ctx = SimpleNamespace(_db_holder=holder, event_bus=None, _hardware_registry=_Registry())
        bus = SimpleNamespace(emit_event=lambda *_a, **_k: None, emit=lambda *_a, **_k: None)
        settings = SimpleNamespace(
            scheduler_enabled=True,
            scheduler_tick_seconds=3600,
            scheduler_grace_seconds=120.0,
            workspace_root=str(tmp_path),
            hardware_enabled=hardware_enabled,
        )
        coordinator = MonitorCoordinator()
        await coordinator.start(ctx, bus, settings)
        try:
            manager = getattr(ctx, "monitors", None)
            assert manager is not None
            return manager.producers.domains()
        finally:
            await coordinator.stop()

    assert "hardware" in await _domains(hardware_enabled=True)
    assert "hardware" not in await _domains(hardware_enabled=False)


@pytest.mark.asyncio
async def test_a_hardware_watch_exists_to_invoke_the_producer(tmp_path: Any) -> None:
    """Registration without a watch naming the domain means the producer never runs."""
    from leapflow.daemon.monitor_coordinator import MonitorCoordinator
    from leapflow.monitor import MonitorManager
    from leapflow.storage.connection import LocalConnectionHolder

    holder = LocalConnectionHolder(tmp_path / "watches.duckdb")
    manager = MonitorManager(holder=holder, emit=lambda *_a, **_k: None, tick_seconds=3600)
    coordinator = MonitorCoordinator()
    coordinator._monitors = manager
    try:
        await coordinator._arm_default_watches()
        armed = {view.name: view for view in manager.list_watches()}
        assert "hardware-bench" in armed
        assert armed["hardware-bench"].domain == "hardware"
    finally:
        await manager.stop()


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════


def _walk(nodes: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        out.append(node)
        out.extend(_walk(node.get("children")))
    return out


def _nodes_of_type(spec: dict[str, Any], type_name: str) -> list[dict[str, Any]]:
    return [node for node in _walk(spec.get("root", [])) if node.get("type") == type_name]


def _rich_payload() -> dict[str, Any]:
    """A payload with every section populated, so no ``when`` gate hides a node."""
    store = SimpleNamespace(raw_writes=1, windows_written=1, write_failures=0, pending_channels=0)
    recorder = SimpleNamespace(recall=lambda **_: ({"command": "c", "outcome": "o", "delta": 0.1},))
    registry = _Registry(
        events=(_event(),),
        store=store,
        recorder=recorder,
        health={"channel_id": "level", "declared_hz": 10.0, "observed_hz": 9.9, "rate_ratio": 0.99},
    )
    return build_digest(registry, now=time.time()).to_payload()


# ════════════════════════════════════════════════════════════════
# CBAG: producer and digest share the same ALERT_KINDS set
# ════════════════════════════════════════════════════════════════


def test_producer_and_digest_share_alert_kinds() -> None:
    """Producer's push-severity set must be the same object as digest's row-severity set.

    CBAG: the two drifted the first time a kind was added: the board coloured
    the row as an alert while the producer still declined to push it. Keeping
    them as one shared constant prevents silent divergence.

    If this assertion fails, a new event kind was added to one copy but not
    the other, causing the push/colour to disagree.
    """
    from leapflow.hardware.observability.digest import ALERT_KINDS as digest_set
    from leapflow.hardware.observability.producer import _ALERT_EVENTS as producer_set

    assert producer_set is digest_set, (
        "ALERT_KINDS drift: the producer's push set and the digest's row-severity "
        "set must be the *same* object, not copies that can diverge. "
        f"producer={sorted(producer_set)}, digest={sorted(digest_set)}"
    )


def test_alert_kinds_contains_the_known_alert_event_types() -> None:
    """ALERT_KINDS must contain at least the four canonical alert kinds.

    CBAG: removing a kind from the set would silently suppress push
    notifications for that event type. This locks the known-good baseline.
    """
    from leapflow.hardware.observability.digest import ALERT_KINDS

    expected = {"threshold_exceeded", "rate_exceeded", "stale", "unreachable"}
    missing = expected - ALERT_KINDS
    assert not missing, (
        f"ALERT_KINDS is missing {sorted(missing)}; removing them would suppress "
        "push notifications for those event types"
    )


def test_producer_severity_matches_digest_event_severity() -> None:
    """For every alert kind, both the row colour and the push severity agree.

    CBAG: the producer decides push-vs-persist from event kinds. The digest
    assigns per-event severity for the board timeline. If they use different
    sets, the board colours a row as "alert" but the producer does not push it
    (or vice versa).
    """
    from leapflow.hardware.observability.digest import ALERT_KINDS, _event_severity

    for kind in ALERT_KINDS:
        assert _event_severity(kind) == "alert", (
            f"digest colours {kind!r} as {_event_severity(kind)!r}, not 'alert'; "
            "but the producer treats it as push-worthy — the two disagree"
        )


def test_digest_payload_clock_is_wall() -> None:
    """Every digest payload must declare ``clock=='wall'``.

    CBAG G16: a payload with the wrong clock would cause the chart to draw
    monotonic instants on a wall-clock axis, producing a correct-looking
    chart that is wrong by decades.
    """
    payload = build_digest(_Registry()).to_payload()
    assert payload["clock"] == WALL_CLOCK, (
        f"payload clock={payload['clock']!r}, expected 'wall'; G16 regression"
    )
    assert payload["schema_version"] == SERIES_SCHEMA_VERSION


def test_the_device_table_offers_buttons_rather_than_an_instruction() -> None:
    """"Select a row to open the device" is a sentence asking the reader to do the UI's job.

    Nothing about a table row says it is clickable, and a row that can do two things --
    open the device, preview it -- cannot say which one a click means. So the actions are
    buttons, and the Preview button appears only where there is something to preview,
    decided per row from the row's own data rather than from device knowledge in the
    template.
    """
    import pathlib

    import yaml

    import leapflow.dashboard.templates as templates_module

    source = pathlib.Path(templates_module.__file__).parent / "templates" / "hardware.yaml"
    spec = yaml.safe_load(source.read_text(encoding="utf-8"))

    def walk(node):
        if isinstance(node, dict):
            yield node
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)

    tables = [
        n["props"] for n in walk(spec)
        if n.get("type") == "Table" and "row_buttons" in (n.get("props") or {})
    ]
    assert tables, "the fleet device table must carry explicit row buttons"
    props = tables[0]

    assert "Select a row" not in str(props.get("caption", "")), (
        "the caption must not instruct the reader to discover a hidden affordance"
    )
    labels = [b["label"] for b in props["row_buttons"]]
    assert labels == ["Open", "Preview"]

    preview = props["row_buttons"][labels.index("Preview")]
    assert preview["require"] == "media", (
        "a Preview button on a device with nothing to preview is a button that cannot work"
    )
    # Both resolve their target from the row, because rows are expanded client-side and a
    # template placeholder has no row in scope when it is compiled.
    for button in props["row_buttons"]:
        assert button["param_from_row"]["device"] == "device_id"
        assert button["params"]["template"] == "hardware", "one lens, plus a target"
    assert preview["param_from_row"]["channel"] == "preview_channel", (
        "Preview must name the actual previewable channel, so it can start the target panel"
    )
