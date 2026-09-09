"""P2 collection layer: the tap, the sink, the trace store, and the four probes.

The probes exist for one reason: to record facts that no store retains. So the
tests that matter most here are the ones asserting a probe fires *where nothing
else would have noticed*, and that with no sink installed the system behaves
exactly as it did before the probes existed.

The four probe sites, and why each earns its keep:

* the plugin registry's version bump -- live conflicts and the version history are
  in memory only;
* the trust level transition -- only the *current* level is persisted, never the
  moment it moved or the direction;
* the world model's drive -- an intent proposed and **not admitted** writes no
  observation, so it exists nowhere at all;
* a lifecycle record opening -- the queue holds the item but not what it was
  opened for.
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from leapflow.domain.evolution_trace import EvolutionStage, EvolutionTrace
from leapflow.evolution import LedgerEvolutionSink
from leapflow.storage.evolution_trace_store import JsonEvolutionTraceStore
from leapflow.telemetry import evolution_tap


class _Collector:
    """Minimal structural sink."""

    def __init__(self) -> None:
        self.traces: list[EvolutionTrace] = []

    def record(self, trace: EvolutionTrace) -> None:
        self.traces.append(trace)


@pytest.fixture(autouse=True)
def _clean_sink():
    """No test may leak a sink: the tap is process-global."""
    evolution_tap.install_sink(None)
    yield
    evolution_tap.install_sink(None)


def _kinds(collector: _Collector) -> list[str]:
    return [t.kind for t in collector.traces]


# ── the tap ──────────────────────────────────────────────────────────────────


def test_no_sink_means_no_op_and_no_error():
    """The default state, and the only one an in-process CLI ever sees."""
    assert evolution_tap.is_enabled() is False
    evolution_tap.emit_trace(EvolutionStage.ACT, "anything", summary="s")  # must not raise


def test_a_broken_sink_cannot_break_the_caller():
    """Telemetry is never allowed an opinion about the operation it observes."""

    class _Broken:
        def record(self, trace):
            raise RuntimeError("sink exploded")

    evolution_tap.install_sink(_Broken())
    evolution_tap.emit_trace(EvolutionStage.ACT, "kind")  # must not raise


def test_install_is_last_wins_not_fan_out():
    """Two live sinks would double-count every fact."""
    first, second = _Collector(), _Collector()
    evolution_tap.install_sink(first)
    evolution_tap.install_sink(second)
    evolution_tap.emit_trace(EvolutionStage.ACT, "kind")
    assert not first.traces
    assert len(second.traces) == 1


# ── the sink ─────────────────────────────────────────────────────────────────


def test_record_does_no_io_and_flush_does():
    """The hot/cold split the probe contract depends on."""

    class _Store:
        def __init__(self):
            self.batches = []

        def append(self, traces):
            rows = list(traces)
            self.batches.append(rows)
            return len(rows)

    store = _Store()
    sink = LedgerEvolutionSink(store=store)
    sink.record(EvolutionTrace(stage=EvolutionStage.ACT, kind="a"))
    sink.record(EvolutionTrace(stage=EvolutionStage.ACT, kind="b"))
    assert store.batches == []  # nothing written yet
    assert sink.flush() == 2
    assert len(store.batches) == 1  # one write for the batch, not one per trace
    assert sink.flush() == 0  # drained


def test_overflow_drops_oldest_and_says_so():
    """A silent drop would make the panel quietly incomplete."""
    sink = LedgerEvolutionSink(store=None, buffer_size=2)
    for i in range(5):
        sink.record(EvolutionTrace(stage=EvolutionStage.ACT, kind=f"k{i}"))
    assert sink.stats["dropped"] == 3
    assert sink.stats["recorded"] == 5
    # Newest survive: a burst means churn, and where it ended up is what matters.
    assert [t.kind for t in sink.pending()] == ["k3", "k4"]


def test_flush_failure_loses_traces_rather_than_retrying_forever():
    class _Broken:
        def append(self, traces):
            raise OSError("disk gone")

    sink = LedgerEvolutionSink(store=_Broken())
    sink.record(EvolutionTrace(stage=EvolutionStage.ACT, kind="a"))
    assert sink.flush() == 0
    assert sink.pending() == ()  # drained, not retried
    assert sink.stats["dropped"] == 1


# ── the store ────────────────────────────────────────────────────────────────


def test_store_round_trip_is_newest_first(tmp_path):
    store = JsonEvolutionTraceStore(tmp_path / "t.json")
    store.append([{"trace_id": "a", "ts": 1.0}, {"trace_id": "b", "ts": 3.0}])
    store.append([{"trace_id": "c", "ts": 2.0}])
    assert [r["trace_id"] for r in store.list_traces()] == ["b", "c", "a"]


def test_store_trims_to_the_newest(tmp_path):
    """Retention is a count, because the newest traces are what an incident needs."""
    store = JsonEvolutionTraceStore(tmp_path / "t.json", max_traces=3)
    store.append([{"trace_id": f"t{i}", "ts": float(i)} for i in range(10)])
    kept = [r["trace_id"] for r in store.list_traces()]
    assert kept == ["t9", "t8", "t7"]
    assert store.count() == 3


def test_corrupt_store_reads_as_empty_rather_than_raising(tmp_path):
    path = tmp_path / "t.json"
    path.write_text("{ not json", encoding="utf-8")
    store = JsonEvolutionTraceStore(path)
    assert store.list_traces() == []
    assert store.append([{"trace_id": "a", "ts": 1.0}]) == 1


# ── probe: registry version bump ─────────────────────────────────────────────


def test_registry_version_bump_is_a_single_observed_point():
    """Every increment must be observable, not just the one method a probe sat in.

    ``notify_mutation`` -- the site the original design named -- is reached by only
    two scope-disposal callers, so a probe there would have missed plugin
    registration, assembly, publication and every unregister path.
    """
    from leapflow.plugins.registry import ToolPluginRegistry

    source = ToolPluginRegistry.__init__.__code__.co_consts  # touch to ensure import
    assert source is not None
    import inspect

    body = inspect.getsource(ToolPluginRegistry)
    # One statement increments the counter, inside the single bump method.
    assert body.count("self._version += 1") == 1
    assert "_bump_version" in body


def test_registry_mutation_emits_a_trace_and_marks_the_phase():
    """Boot composition and a later runtime change must be distinguishable.

    Otherwise every daemon start replays the initial plugin load and buries the
    rare real mutation under a boot log.
    """
    from leapflow.plugins.registry import ToolPluginRegistry

    collector = _Collector()
    evolution_tap.install_sink(collector)

    registry = ToolPluginRegistry()
    registry.notify_mutation()
    assert _kinds(collector) == ["registry_scope_disposed"]
    trace = collector.traces[0]
    assert trace.stage is EvolutionStage.ACT
    # Before assemble() this is composition, not evolution.
    assert trace.detail["phase"] == "composition"
    assert trace.correlation["registry_version"] == "1"

    registry._assembled = True
    registry.notify_mutation()
    assert collector.traces[-1].detail["phase"] == "runtime"


def test_registry_still_bumps_when_the_sink_is_broken():
    """The registry's own job must survive a telemetry failure."""
    from leapflow.plugins.registry import ToolPluginRegistry

    class _Broken:
        def record(self, trace):
            raise RuntimeError("boom")

    evolution_tap.install_sink(_Broken())
    registry = ToolPluginRegistry()
    before = registry.version
    registry.notify_mutation()
    assert registry.version == before + 1


# ── probe: trust transition ──────────────────────────────────────────────────


def test_trust_transition_is_emitted_where_the_flush_already_detects_it():
    """Only the current level is persisted; the move itself lives nowhere else."""
    from leapflow.engine.session_factory import _PersistingTrustLedger

    collector = _Collector()
    evolution_tap.install_sink(collector)

    ledger = _PersistingTrustLedger(candidate_at=2, verified_at=99, production_at=99)
    ledger.record_success("p")
    assert collector.traces == []  # no level change, no trace
    ledger.record_success("p")  # crosses candidate_at

    assert _kinds(collector) == ["trust_transition"]
    trace = collector.traces[0]
    assert trace.stage is EvolutionStage.LEARN
    assert trace.detail["from"] == "DRAFT"
    assert trace.detail["to"] == "CANDIDATE"
    assert trace.detail["frozen"] is False


def test_a_hard_failure_is_traced_as_frozen_even_at_an_unchanged_level():
    """DRAFT alone cannot say whether a plugin is new or disqualified."""
    from leapflow.engine.session_factory import _PersistingTrustLedger

    collector = _Collector()
    evolution_tap.install_sink(collector)

    ledger = _PersistingTrustLedger()
    ledger.record_failure("p", hard=True)  # already DRAFT; level does not move

    assert _kinds(collector) == ["trust_frozen"]
    assert collector.traces[0].detail["frozen"] is True
    assert collector.traces[0].detail["hard_failure"] is True


def test_the_pure_trust_ledger_gains_no_telemetry_dependency():
    """``PluginTrustLedger`` is a zero-dependency domain object and stays one."""
    import inspect

    from leapflow.learning import plugin_trust

    source = inspect.getsource(plugin_trust)
    assert "evolution_tap" not in source
    assert "emit_trace" not in source


# ── probe: the world model's unadmitted proposals ────────────────────────────


def test_unadmitted_intents_are_recorded_because_nothing_else_records_them():
    """The single most valuable trace: a proposal that entered no pipeline.

    It writes no observation, so without this the board shows an idle pipeline
    while the world model proposes on every session.
    """
    import asyncio
    from dataclasses import dataclass

    from leapflow.learning.world_model_driver import WorldModelEvolutionDriver

    @dataclass
    class _Intent:
        intent_id: str = "wmi-1"
        capability: str = "ui.chat.send"
        hypothesis: str = "no way to send a chat message"
        confidence: float = 0.8

        def to_dict(self):
            return {
                "intent_id": self.intent_id,
                "capability": self.capability,
                "hypothesis": self.hypothesis,
                "confidence": self.confidence,
            }

        def to_observation_result(self, **_kw):
            return {"error_type": "world_model_intent", "capability": self.capability}

    class _Teacher:
        async def grade_and_propose(self, trajectory, goal):
            return type("V", (), {"grades": (), "intents": (_Intent(),)})()

    class _RejectingIntake:
        """Mirrors a profile where ``world_model_intent`` is not an accepted kind."""

        def observe_result(self, *_a, **_kw):
            return None

        def requirements(self, **_kw):
            return ()

    collector = _Collector()
    evolution_tap.install_sink(collector)

    driver = WorldModelEvolutionDriver(teacher=_Teacher(), intake=_RejectingIntake())
    result = asyncio.run(driver.drive(trajectory=[{"step": 1}], goal="g"))

    assert result.admitted_observation_ids == ()
    assert _kinds(collector) == ["world_model_drive"]
    detail = collector.traces[0].detail
    assert detail["not_admitted_reason"]
    assert detail["intents"][0]["hypothesis"] == "no way to send a chat message"
    assert detail["admitted_observation_ids"] == []


def test_producer_surfaces_unadmitted_proposals_from_traces():
    """The payload must carry the model's reasoning, not just a count."""
    from leapflow.monitor.evolution_producer import EvolutionProducer

    traces = [
        {
            "trace_id": "t1",
            "stage": "observe",
            "kind": "world_model_drive",
            "summary": "teacher proposed 1, admitted 0",
            "detail": {
                "not_admitted_reason": "world_model_intent is not in accepted_evidence_kinds",
                "intents": [
                    {"capability": "ui.chat.send", "hypothesis": "no way to send", "confidence": 0.8}
                ],
            },
        }
    ]
    rows = EvolutionProducer._unadmitted(traces)
    assert rows == [
        {
            "capability": "ui.chat.send",
            "hypothesis": "no way to send",
            "confidence": "80%",
            "reason": "world_model_intent is not in accepted_evidence_kinds",
        }
    ]


def test_composition_traces_are_kept_out_of_the_live_feed():
    """A boot replay would bury the rare real mutation."""
    from leapflow.monitor.evolution_producer import EvolutionProducer

    traces = [
        {"trace_id": "a", "stage": "act", "kind": "registry_assembled",
         "summary": "boot", "detail": {"phase": "composition"}},
        {"trace_id": "b", "stage": "act", "kind": "registry_plugin_registered",
         "summary": "install", "detail": {"phase": "runtime"}},
        {"trace_id": "c", "stage": "learn", "kind": "trust_frozen",
         "summary": "frozen", "detail": {"phase": "runtime"}},
    ]
    feed = EvolutionProducer._trace_feed(traces)
    assert [row["summary"] for row in feed] == ["install", "frozen"]
    # A freeze is the one trace worth interrupting for.
    assert feed[-1]["severity"] == "alert"


def test_traces_participate_in_the_dedup_fingerprint():
    """A trust transition does not move the registry version.

    Without traces in the key the executor would skip the write and the live feed
    would freeze on the page while still looking current.
    """
    from leapflow.monitor.evolution_producer import EvolutionProducer as P

    base = {
        "summary": {"registry_version": 1, "registry_readable": True},
        "roster": [],
        "conflicts": [],
        "reachability": [],
        "episodes": [],
        "traces": [{"trace_id": "t1"}],
    }
    reference = P._fingerprint(base)
    grown = {**base, "traces": [{"trace_id": "t1"}, {"trace_id": "t2"}]}
    assert P._fingerprint(grown) != reference


# ── P4: fiber transitions recovered by snapshot diff ─────────────────────────


def test_fiber_transitions_need_a_baseline_before_reporting_anything():
    """The first cycle after a restart has nothing to compare against."""
    from leapflow.monitor.evolution_producer import EvolutionProducer

    producer = EvolutionProducer()
    assert producer._fiber_transitions({"a": "active"}, readable=True) == []
    # Second cycle, unchanged: still nothing to report.
    assert producer._fiber_transitions({"a": "active"}, readable=True) == []


def test_the_load_retry_path_is_visible_only_through_the_diff():
    """``LOADING -> FAILED -> LOADING`` bumps no registry version, so no probe sees it."""
    from leapflow.monitor.evolution_producer import EvolutionProducer

    producer = EvolutionProducer()
    producer._fiber_transitions({"p": "loading"}, readable=True)  # baseline
    failed = producer._fiber_transitions({"p": "failed"}, readable=True)
    retry = producer._fiber_transitions({"p": "loading"}, readable=True)

    assert failed == [{"plugin_id": "p", "from": "loading", "to": "failed", "kind": "moved"}]
    assert retry == [{"plugin_id": "p", "from": "failed", "to": "loading", "kind": "moved"}]


def test_appearance_and_disposal_are_distinguished_from_a_move():
    from leapflow.monitor.evolution_producer import EvolutionProducer

    producer = EvolutionProducer()
    producer._fiber_transitions({"a": "active"}, readable=True)
    rows = producer._fiber_transitions({"a": "active", "b": "pending"}, readable=True)
    assert rows == [{"plugin_id": "b", "from": "", "to": "pending", "kind": "appeared"}]

    rows = producer._fiber_transitions({"a": "active"}, readable=True)
    assert rows == [{"plugin_id": "b", "from": "pending", "to": "", "kind": "gone"}]


def test_an_unreadable_registry_fabricates_no_mass_event():
    """Diffing against an empty snapshot would report every plugin as disposed.

    And replacing the baseline with it would report every plugin as newly appeared
    on the next good cycle -- two fabricated mass events from one transient failure.
    """
    from leapflow.monitor.evolution_producer import EvolutionProducer

    producer = EvolutionProducer()
    producer._fiber_transitions({"a": "active", "b": "active"}, readable=True)

    assert producer._fiber_transitions({}, readable=False) == []
    # Baseline survived, so the next good cycle sees no change either.
    assert producer._fiber_transitions({"a": "active", "b": "active"}, readable=True) == []


def test_fiber_transitions_are_covered_by_the_fingerprint():
    """A retry leaves the roster identical, so only the delta can refresh the board."""
    from leapflow.monitor.evolution_producer import EvolutionProducer as P

    quiet = {
        "summary": {"registry_version": 1, "registry_readable": True},
        "roster": [], "conflicts": [], "reachability": [], "episodes": [], "traces": [],
    }
    noisy = {
        **quiet,
        "fiber_transitions": [
            {"plugin_id": "p", "from": "failed", "to": "loading", "kind": "moved"}
        ],
    }
    assert P._fingerprint(noisy) != P._fingerprint(quiet)


# ── P3: event-driven refresh ──────────────────────────────────────────────────


def test_the_sink_publishes_every_trace_it_buffers():
    """Publication is per-trace so a mutation can refresh the board immediately."""
    published: list = []
    sink = LedgerEvolutionSink(store=None, publish=published.append)
    sink.record(EvolutionTrace(stage=EvolutionStage.ACT, kind="registry_plugin_registered"))
    assert [t.kind for t in published] == ["registry_plugin_registered"]


def test_a_failing_publisher_cannot_lose_the_trace():
    """Buffering must survive a broken event bus: the durable record matters more."""

    def _broken(_trace):
        raise RuntimeError("bus down")

    sink = LedgerEvolutionSink(store=None, publish=_broken)
    sink.record(EvolutionTrace(stage=EvolutionStage.ACT, kind="k"))
    assert len(sink.pending()) == 1


def test_composition_traces_are_not_published_as_events():
    """Boot replays the plugin load; publishing it would fire the watch to say nothing.

    Asserted against the daemon's real publisher factory rather than a copy of its
    rule, because a second implementation of the filter could disagree with it.
    """
    import asyncio

    from leapflow.daemon.monitor_coordinator import MonitorCoordinator

    seen: list[str] = []

    class _Bus:
        async def handle_event(self, event_type, payload):
            seen.append(event_type)

    async def _drive():
        publisher = MonitorCoordinator()._make_evolution_publisher(
            SimpleNamespace(event_bus=_Bus())
        )
        assert publisher is not None
        publisher(
            EvolutionTrace(
                stage=EvolutionStage.ACT, kind="registry_assembled",
                detail={"phase": "composition"},
            )
        )
        publisher(
            EvolutionTrace(
                stage=EvolutionStage.ACT, kind="registry_plugin_registered",
                detail={"phase": "runtime"},
            )
        )
        await asyncio.sleep(0.05)  # let the scheduled coroutines run

    asyncio.run(_drive())
    assert seen == ["evolution.registry_plugin_registered"]


def test_the_publisher_is_absent_rather_than_broken_without_a_bus():
    """No event bus is a normal state (in-process CLI), not a failure to report."""
    from leapflow.daemon.monitor_coordinator import MonitorCoordinator

    coordinator = MonitorCoordinator()
    assert coordinator._make_evolution_publisher(SimpleNamespace()) is None
    assert coordinator._make_evolution_publisher(SimpleNamespace(event_bus=object())) is None


def test_the_live_watch_is_armed_alongside_the_polled_one():
    """State needs polling (it has no event); change needs events (polling is late).

    Both are armed on the same domain, and the content fingerprint makes the
    overlap free -- an unchanged framework dedups the second finding away.
    """
    from leapflow.daemon.monitor_coordinator import MonitorCoordinator

    entries = {
        name: (trigger, at_once)
        for name, domain, trigger, at_once in MonitorCoordinator._DEFAULT_WATCHES
        if domain == "framework_evolution"
    }
    assert entries == {
        # Runs at once: it reads live registry state, so its first answer is already
        # correct and a ten-minute blank board would be pure loss.
        "framework-evolution": ("10m", True),
        "framework-evolution-live": ("event:evolution.*", False),
    }


def test_only_live_state_producers_are_brought_forward():
    """An accumulating producer has nothing true to say before it has accumulated.

    Bringing every interval watch forward published an empty hardware digest that
    then stood as the newest finding until the next interval elapsed, so the board
    reported zero sample windows on a bench that was sampling.
    """
    from leapflow.daemon.monitor_coordinator import MonitorCoordinator

    at_once = {
        name for name, _d, _t, flag in MonitorCoordinator._DEFAULT_WATCHES if flag
    }
    assert at_once == {"framework-evolution"}
