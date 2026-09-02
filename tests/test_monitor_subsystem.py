"""Hermetic tests for the domain-neutral monitor subsystem (Watch -> Finding).

No network, no LLM: uses a temporary DuckDB and a fake in-process producer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from leapflow.monitor import (
    EVENT_FINDING,
    Evidence,
    Finding,
    MonitorManager,
    ProducerRegistry,
    Severity,
    SuggestedAction,
    WatchSpec,
)
from leapflow.monitor.finding_store import FindingStore
from leapflow.monitor.types import ProducerContext
from leapflow.scheduler.coordinator import parse_trigger_expression
from leapflow.scheduler.triggers import create_trigger
from leapflow.scheduler.triggers.event import EventTrigger
from leapflow.storage.connection import LocalConnectionHolder


class _FakeProducer:
    """Deterministic producer returning a fixed list of findings per cycle."""

    def __init__(self, domain: str, findings: list[Finding]) -> None:
        self._domain = domain
        self._findings = findings
        self.calls = 0

    @property
    def domain(self) -> str:
        return self._domain

    async def observe(self, ctx: ProducerContext) -> list[Finding]:
        self.calls += 1
        return list(self._findings)


def _holder(tmp_path: Path) -> LocalConnectionHolder:
    return LocalConnectionHolder(tmp_path / "leap.duckdb")


# ── Contract serialization ────────────────────────────────────────────────


def test_finding_roundtrip_preserves_fields() -> None:
    finding = Finding(
        watch_id="w1",
        domain="finance",
        title="AAPL spike",
        summary="Unusual volume",
        severity=Severity.ALERT,
        score=0.87,
        evidence=(Evidence(kind="link", label="chart", url="http://x/y"),),
        tags=("volume", "equity"),
        suggested_actions=(SuggestedAction(name="drilldown", label="Open", kind="nav"),),
        payload={"ohlc": [[1, 2, 3, 4]]},
        dedup_key="aapl-2026-07-16",
    )
    restored = Finding.from_dict(finding.to_dict())
    assert restored.watch_id == "w1"
    assert restored.domain == "finance"
    assert restored.severity is Severity.ALERT
    assert restored.tags == ("volume", "equity")
    assert restored.evidence[0].url == "http://x/y"
    assert restored.suggested_actions[0].kind == "nav"
    assert restored.payload == {"ohlc": [[1, 2, 3, 4]]}
    assert restored.dedup_key == "aapl-2026-07-16"


def test_watchspec_params_roundtrip() -> None:
    spec = WatchSpec(
        name="ArXiv NLP",
        domain="research",
        trigger_expr="30m",
        source={"feed": "arxiv:cs.CL"},
        lens={"keywords": ["agent"]},
        sensitivity="alert",
        watch_id="wid",
    )
    restored = WatchSpec.from_params(spec.to_task_parameters())
    assert restored.name == "ArXiv NLP"
    assert restored.domain == "research"
    assert restored.trigger_expr == "30m"
    assert restored.source == {"feed": "arxiv:cs.CL"}
    assert restored.push_threshold() is Severity.ALERT


def test_severity_coerce_and_rank() -> None:
    assert Severity.coerce("alert") is Severity.ALERT
    assert Severity.coerce("bogus") is Severity.INFO
    assert Severity.ALERT.rank > Severity.NOTABLE.rank > Severity.INFO.rank


# ── FindingStore ────────────────────────────────────────────────────────────


def test_finding_store_crud_dedup_and_filters(tmp_path: Path) -> None:
    store = FindingStore(_holder(tmp_path))
    store.save(Finding(watch_id="w1", domain="d", title="a", severity=Severity.INFO, ts=100.0, dedup_key="k1"))
    store.save(Finding(watch_id="w1", domain="d", title="b", severity=Severity.ALERT, ts=200.0, dedup_key="k2"))
    store.save(Finding(watch_id="w2", domain="d", title="c", severity=Severity.NOTABLE, ts=150.0))

    assert store.exists_dedup("w1", "k1") is True
    assert store.exists_dedup("w1", "missing") is False

    all_w1 = store.list(watch_id="w1")
    assert [f.title for f in all_w1] == ["b", "a"]  # newest-first by ts

    alerts = store.list(min_severity=Severity.ALERT)
    assert [f.title for f in alerts] == ["b"]

    since = store.list(since=150.0)
    assert {f.title for f in since} == {"b", "c"}

    assert store.count() == 3
    assert store.count(watch_id="w1") == 2
    assert store.count(min_severity=Severity.NOTABLE) == 2

    store.delete_for_watch("w1")
    assert store.count(watch_id="w1") == 0


# ── MonitorManager lifecycle ────────────────────────────────────────────────


async def test_manager_arm_list_and_state_transitions(tmp_path: Path) -> None:
    manager = MonitorManager(holder=_holder(tmp_path))
    view = await manager.arm_watch(WatchSpec(name="Market", domain="finance", trigger_expr="5m"))
    assert view.domain == "finance"
    assert view.state == "armed"
    assert view.client_coupled is False
    assert view.to_dict()["client_coupled"] is False

    assert [v.watch_id for v in manager.list_watches()] == [view.watch_id]
    assert manager.has_active_watches() is True

    assert manager.pause_watch(view.watch_id).state == "suspended"
    assert manager.has_active_watches() is False
    assert manager.resume_watch(view.watch_id).state == "armed"

    muted = manager.set_muted(view.watch_id, True)
    assert muted.muted is True

    stopped = manager.stop_watch(view.watch_id)
    assert stopped.state == "done"
    assert manager.has_active_watches() is False
    assert manager.get_watch("nonexistent") is None


async def test_manager_run_once_persists_and_gates_push(tmp_path: Path) -> None:
    emitted: list[tuple[str, dict]] = []
    producers = ProducerRegistry()
    producers.register(_FakeProducer("finance", [
        Finding(watch_id="", domain="finance", title="quiet", severity=Severity.INFO, dedup_key="i"),
        Finding(watch_id="", domain="finance", title="move", severity=Severity.NOTABLE, dedup_key="n"),
        Finding(watch_id="", domain="finance", title="spike", severity=Severity.ALERT, dedup_key="a"),
    ]))
    manager = MonitorManager(
        holder=_holder(tmp_path),
        producers=producers,
        emit=lambda et, payload: emitted.append((et, payload)),
    )
    view = await manager.arm_watch(WatchSpec(name="M", domain="finance", sensitivity="notable"))

    result = await manager.run_watch_once(view.watch_id)
    assert result["ok"] is True
    assert result["findings"] == 3  # all persisted
    finding_events = [p for et, p in emitted if et == EVENT_FINDING]
    assert {p["title"] for p in finding_events} == {"move", "spike"}  # info gated out

    # Second cycle: dedup keys already present -> nothing new persisted/pushed.
    before = len(emitted)
    result2 = await manager.run_watch_once(view.watch_id)
    assert result2["findings"] == 0
    assert len(emitted) == before

    assert manager.finding_store.count(watch_id=view.watch_id) == 3


async def test_manager_muted_watch_persists_without_push(tmp_path: Path) -> None:
    emitted: list[tuple[str, dict]] = []
    producers = ProducerRegistry()
    producers.register(_FakeProducer("sentiment", [
        Finding(watch_id="", domain="sentiment", title="surge", severity=Severity.ALERT, dedup_key="s"),
    ]))
    manager = MonitorManager(
        holder=_holder(tmp_path),
        producers=producers,
        emit=lambda et, payload: emitted.append((et, payload)),
    )
    view = await manager.arm_watch(WatchSpec(name="S", domain="sentiment"))
    manager.set_muted(view.watch_id, True)
    emitted.clear()

    result = await manager.run_watch_once(view.watch_id)
    assert result["findings"] == 1
    assert [et for et, _ in emitted if et == EVENT_FINDING] == []  # muted -> no push


async def test_manager_unknown_domain_is_graceful(tmp_path: Path) -> None:
    manager = MonitorManager(holder=_holder(tmp_path))
    view = await manager.arm_watch(WatchSpec(name="X", domain="unregistered"))
    result = await manager.run_watch_once(view.watch_id)
    assert result["ok"] is False
    assert "no producer" in result["error"]


async def test_has_active_watches_excludes_client_coupled(tmp_path: Path) -> None:
    manager = MonitorManager(holder=_holder(tmp_path))
    coupled = await manager.arm_watch(WatchSpec(name="S", domain="session", client_coupled=True))
    assert coupled.client_coupled is True
    assert manager.has_active_watches() is False  # client-coupled must not keep leapd alive
    standalone = await manager.arm_watch(WatchSpec(name="F", domain="finance"))
    assert manager.has_active_watches() is True
    manager._task_store.update_state(standalone.watch_id, "executing")
    assert manager.has_active_watches() is True


async def test_sweep_client_coupled_watches_removes_only_session_watches(tmp_path: Path) -> None:
    manager = MonitorManager(holder=_holder(tmp_path))
    session = await manager.arm_watch(WatchSpec(name="S", domain="session", client_coupled=True))
    standalone = await manager.arm_watch(WatchSpec(name="F", domain="finance"))
    manager.finding_store.save(
        Finding(watch_id=session.watch_id, domain="session", title="stale", severity=Severity.INFO)
    )

    removed = manager.sweep_client_coupled_watches()

    assert removed == 1
    remaining = [v.watch_id for v in manager.list_watches()]
    assert remaining == [standalone.watch_id]  # standalone watch is durable
    assert manager.get_watch(session.watch_id) is None
    assert manager.finding_store.count(watch_id=session.watch_id) == 0


async def test_sweep_client_coupled_watches_is_noop_without_session_watches(tmp_path: Path) -> None:
    manager = MonitorManager(holder=_holder(tmp_path))
    await manager.arm_watch(WatchSpec(name="F", domain="finance"))
    assert manager.sweep_client_coupled_watches() == 0
    assert len(manager.list_watches()) == 1


# ── Event trigger expression tests ────────────────────────────────────────────


def test_parse_event_trigger_fs_change() -> None:
    """event:fs.change parses to trigger_type='event' with correct pattern."""
    trigger_type, trigger_config = parse_trigger_expression("event:fs.change")
    assert trigger_type == "event"
    assert trigger_config == {"event_pattern": "fs.change"}

    # Verify the config creates a valid EventTrigger
    trigger = create_trigger(trigger_type, trigger_config)
    assert isinstance(trigger, EventTrigger)
    assert trigger.event_pattern == "fs.change"


def test_parse_event_trigger_wildcard_pattern() -> None:
    """event:monitor.* parses correctly and supports glob matching."""
    trigger_type, trigger_config = parse_trigger_expression("event:monitor.*")
    assert trigger_type == "event"
    assert trigger_config == {"event_pattern": "monitor.*"}

    trigger = create_trigger(trigger_type, trigger_config)
    assert isinstance(trigger, EventTrigger)
    assert trigger.event_pattern == "monitor.*"
    # Glob matching works
    assert trigger.matches("monitor.cpu") is True
    assert trigger.matches("monitor.mem") is True
    assert trigger.matches("fs.change") is False


def test_parse_event_trigger_gateway_signal() -> None:
    """event:gateway.signal parses correctly."""
    trigger_type, trigger_config = parse_trigger_expression("event:gateway.signal")
    assert trigger_type == "event"
    assert trigger_config == {"event_pattern": "gateway.signal"}

    trigger = create_trigger(trigger_type, trigger_config)
    assert isinstance(trigger, EventTrigger)
    assert trigger.event_pattern == "gateway.signal"


def test_parse_event_trigger_empty_pattern_raises() -> None:
    """event: with empty pattern raises ValueError."""
    with pytest.raises(ValueError, match="event name"):
        parse_trigger_expression("event:")


def test_parse_event_trigger_whitespace_only_raises() -> None:
    """event: followed by whitespace only raises ValueError."""
    with pytest.raises(ValueError, match="event name"):
        parse_trigger_expression("event:   ")


def test_parse_event_trigger_case_insensitive_prefix() -> None:
    """The 'event:' prefix is recognized case-insensitively."""
    trigger_type, trigger_config = parse_trigger_expression("Event:ci.passed")
    assert trigger_type == "event"
    assert trigger_config == {"event_pattern": "ci.passed"}

    trigger_type2, trigger_config2 = parse_trigger_expression("EVENT:deploy.done")
    assert trigger_type2 == "event"
    assert trigger_config2 == {"event_pattern": "deploy.done"}


async def test_arm_watch_with_event_trigger_registers_bridge(tmp_path: Path) -> None:
    """arm_watch with event trigger registers the trigger in EventBridge."""
    manager = MonitorManager(holder=_holder(tmp_path))
    view = await manager.arm_watch(
        WatchSpec(name="FSWatch", domain="filesystem", trigger_expr="event:fs.change")
    )
    assert view.state == "armed"
    # Verify EventBridge has registered this watch
    assert manager.event_bridge.active_count == 1


async def test_arm_watch_with_event_trigger_glob_pattern(tmp_path: Path) -> None:
    """arm_watch with glob event pattern registers correctly."""
    manager = MonitorManager(holder=_holder(tmp_path))
    view = await manager.arm_watch(
        WatchSpec(name="AllMonitor", domain="monitor", trigger_expr="event:monitor.*")
    )
    assert view.state == "armed"
    assert manager.event_bridge.active_count == 1

    # Stop removes from bridge
    manager.stop_watch(view.watch_id)
    assert manager.event_bridge.active_count == 0


# ════════════════════════════════════════════════════════════════
# Default producers and watches are asserted by their effect, not their existence
# ════════════════════════════════════════════════════════════════


def _coordinator_with_manager(tmp_path: Path) -> tuple[object, MonitorManager]:
    """Build a real MonitorCoordinator around a temporary in-process manager."""
    from leapflow.daemon.monitor_coordinator import MonitorCoordinator

    holder = LocalConnectionHolder(tmp_path / "monitor.duckdb")
    manager = MonitorManager(holder=holder, emit=lambda *_a, **_k: None, tick_seconds=3600)
    coordinator = MonitorCoordinator()
    coordinator._monitors = manager
    return coordinator, manager


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "domain",
    ["session", "signal", "capability_adaptation", "plugin_health"],
)
async def test_every_default_producer_is_actually_registered(
    tmp_path: Path, domain: str
) -> None:
    """Drives the production ``start()`` and reads the registry it built.

    ``PluginHealthProducer`` was fully implemented, exported nowhere and registered
    nowhere, while its own docstring said it was registered. Instantiating the class
    in a test proves only that the class exists, and registering it in a test proves
    only that the test can -- so this asserts the registry that the coordinator
    itself populates. Deleting a ``producers.register(...)`` line turns this red.
    """
    from types import SimpleNamespace

    from leapflow.daemon.monitor_coordinator import MonitorCoordinator

    holder = LocalConnectionHolder(tmp_path / "monitor.duckdb")
    ctx = SimpleNamespace(_db_holder=holder, event_bus=None)
    bus = SimpleNamespace(emit_event=lambda *_a, **_k: None, emit=lambda *_a, **_k: None)
    settings = SimpleNamespace(
        scheduler_enabled=True,
        scheduler_tick_seconds=3600,
        scheduler_grace_seconds=120.0,
        workspace_root=str(tmp_path),
    )

    coordinator = MonitorCoordinator()
    await coordinator.start(ctx, bus, settings)
    manager = getattr(ctx, "monitors", None)
    assert manager is not None, "start() must build a manager for this test to mean anything"
    try:
        assert domain in manager.producers.domains(), f"no producer serves domain={domain!r}"
        assert manager.producers.resolve(domain) is not None
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_default_watches_cover_every_polled_default_domain(tmp_path: Path) -> None:
    """A registered producer with no watch naming its domain never runs.

    Registration and arming are two separate connections, and satisfying one reads
    as satisfying both. This asserts the pair.
    """
    coordinator, manager = _coordinator_with_manager(tmp_path)
    try:
        await coordinator._arm_default_watches()
        armed = {view.name: view for view in manager.list_watches()}
        assert "plugin-health" in armed, "the plugin_health producer has no watch to invoke it"
        assert armed["plugin-health"].domain == "plugin_health"
        # Polled, not event-driven: trust degradation is a trend between observations.
        assert armed["plugin-health"].trigger == "every 5m"
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_arming_defaults_twice_does_not_duplicate_an_interval_watch(
    tmp_path: Path,
) -> None:
    """Idempotency must hold for interval triggers, not just event ones.

    The previous dedup key rebuilt the label as ``f"event:{expr}"``, so an interval
    watch could never match its own entry and was re-armed on every daemon start.
    Each restart would have added another copy, each polling the same producer.
    """
    coordinator, manager = _coordinator_with_manager(tmp_path)
    try:
        await coordinator._arm_default_watches()
        first = [v.name for v in manager.list_watches()]
        await coordinator._arm_default_watches()
        second = [v.name for v in manager.list_watches()]
        assert sorted(first) == sorted(second)
        assert second.count("plugin-health") == 1
        assert second.count("fs-observer") == 1
    finally:
        await manager.stop()


def test_fit_to_frame_keeps_the_newest_prefix_within_the_byte_budget() -> None:
    """A batch reply is one JSON-RPC frame, so rows must be bounded by bytes.

    ``limit`` counts rows while the transport counts bytes; the two disagree, and an
    8 MiB batch of large findings overran the 4 MiB frame and took the whole Board
    down. The batch keeps the newest findings (findings are newest-first) up to the
    budget and drops the oldest tail rather than losing the present.
    """
    from leapflow.daemon.monitor_coordinator import _fit_to_frame

    findings = [{"id": i, "blob": "x" * 1000} for i in range(50)]
    budget = 5000
    kept = _fit_to_frame(findings, budget=budget)
    assert 0 < len(kept) < len(findings)
    assert [row["id"] for row in kept] == list(range(len(kept)))  # newest prefix
    import json

    encoded = sum(len(json.dumps(row).encode("utf-8")) for row in kept)
    assert encoded <= budget


def test_fit_to_frame_keeps_a_single_over_budget_finding_rather_than_none() -> None:
    """The newest state must never be silently empty.

    A lone finding larger than the whole budget is a producer defect the coordinator
    cannot fix by dropping neighbours it does not have. It is kept so the caller sees
    the current state; the transport layer still reports the frame overrun as a typed
    error rather than a crash.
    """
    from leapflow.daemon.monitor_coordinator import _fit_to_frame

    kept = _fit_to_frame([{"id": 0, "blob": "x" * 10000}], budget=1000)
    assert [row["id"] for row in kept] == [0]


@pytest.mark.asyncio
async def test_coordinator_findings_reply_stays_within_one_rpc_frame(tmp_path: Path) -> None:
    """End to end: many large findings must not produce an unreadable reply.

    Persists more oversized findings than one frame can carry, then asserts the
    coordinator's ``findings`` reply -- the exact RPC the Board calls -- encodes
    within the transport frame limit.
    """
    import json

    from leapflow.daemon._transport import RPC_STREAM_LIMIT

    coordinator, manager = _coordinator_with_manager(tmp_path)
    try:
        for index in range(12):
            manager.finding_store.save(
                Finding(
                    watch_id="hw",
                    domain="hardware",
                    title=f"snapshot {index}",
                    summary="bench",
                    severity=Severity.INFO,
                    ts=1_788_000_000.0 + index,
                    payload={"grid": [{"c": "ch", "x": float(n), "s": "inside"} for n in range(9000)]},
                    dedup_key=f"hw-{index}",
                )
            )
        reply = await coordinator.findings(watch_id="", limit=50)
        assert reply, "the newest finding must always be returned"
        encoded = len(json.dumps(reply, default=str).encode("utf-8"))
        assert encoded <= RPC_STREAM_LIMIT
    finally:
        await manager.stop()
