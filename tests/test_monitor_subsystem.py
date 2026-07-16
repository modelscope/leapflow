"""Hermetic tests for the domain-neutral monitor subsystem (Watch -> Finding).

No network, no LLM: uses a temporary DuckDB and a fake in-process producer.
"""

from __future__ import annotations

from pathlib import Path

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
