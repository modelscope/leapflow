"""EvolutionProducer: the framework-evolution transparency panel.

The tests that matter most here are the negative ones. This producer reports
LeapFlow's own composition, so the failure modes are all about claiming more than
was measured:

* it must never report a pipeline segment as working because a module exists;
* it must distinguish "nothing observed" from "could not read";
* it must not fail the monitor cycle when the registry is unreachable;
* its dedup key must be a content fingerprint, or an unchanged framework either
  re-notifies every cycle or (if the clock leaks in) never dedups at all.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from leapflow.monitor.evolution_producer import (
    NO_EVIDENCE,
    NOT_ADMITTED,
    UNVERIFIABLE,
    WIRED,
    EvolutionProducer,
)
from leapflow.monitor.types import Severity, WatchSpec


def _ctx(now: float = 1000.0) -> SimpleNamespace:
    return SimpleNamespace(
        spec=WatchSpec(name="framework-evolution", domain="framework_evolution"),
        now=now,
        run_count=0,
        last_run_at=0.0,
        services=None,
        force=False,
    )


def _observe(producer: EvolutionProducer | None = None, **kwargs):
    return asyncio.run((producer or EvolutionProducer()).observe(_ctx(**kwargs)))


# ── fakes ─────────────────────────────────────────────────────────────────────


class _Tool:
    def __init__(self, name: str, provides: tuple[str, ...] = ()) -> None:
        self.name = name
        self.provides_capabilities = provides


class _Plugin:
    def __init__(self, plugin_id: str, tools: list[_Tool]) -> None:
        self.plugin_id = plugin_id
        self.tools = tools


class _Conflict:
    def __init__(self, tool_name: str, kept: str, rejected: str) -> None:
        self.tool_name = tool_name
        self.kept_plugin = kept
        self.rejected_plugin = rejected


class _Registry:
    def __init__(self, plugins: dict[str, _Plugin], *, version: int = 7, conflicts=()) -> None:
        self.plugins = plugins
        self.version = version
        self.conflicts = list(conflicts)
        self.tool_owners = {
            tool.name: pid for pid, plugin in plugins.items() for tool in plugin.tools
        }
        self.tool_handlers = {name: object() for name in self.tool_owners}


class _Level:
    def __init__(self, name: str) -> None:
        self.name = name


class _Trust:
    def __init__(self, levels: dict[str, str], frozen: set[str] | None = None) -> None:
        self._levels = levels
        self._frozen = frozen or set()

    def level(self, plugin_id: str) -> _Level:
        return _Level(self._levels.get(plugin_id, "DRAFT"))

    def is_frozen(self, plugin_id: str) -> bool:
        return plugin_id in self._frozen


class _Stats:
    def __init__(self, total_calls: int) -> None:
        self.total_calls = total_calls


class _Usage:
    def __init__(self, calls: dict[str, int]) -> None:
        self._calls = calls

    def stats_for_plugin(self, plugin_id: str):
        if plugin_id not in self._calls:
            return None
        return _Stats(self._calls[plugin_id])


def _install_registry(monkeypatch, registry, *, trust=None, usage=None, fibers=None):
    """Point the producer's lazy lookups at fakes.

    Patched on the producer module rather than the source packages because the
    producer imports them inside the call, which is what keeps ``leapflow.monitor``
    importable without the plugin subsystem.
    """
    import leapflow.plugins as plugins_pkg
    from leapflow.monitor import evolution_producer as mod

    monkeypatch.setattr(plugins_pkg, "get_registry", lambda: registry, raising=False)
    monkeypatch.setattr(
        mod.EvolutionProducer, "_trust_and_usage", staticmethod(lambda: (trust, usage))
    )
    monkeypatch.setattr(
        mod.EvolutionProducer, "_fiber_states", staticmethod(lambda: dict(fibers or {}))
    )
    # Stores are profile-scoped; a unit test has no profile layout.
    monkeypatch.setattr(
        mod.EvolutionProducer, "_json_store", staticmethod(lambda *a, **k: None)
    )


def _rows(finding) -> dict[str, dict]:
    return {row["key"]: row for row in finding.payload["reachability"]}


# ── the snapshot must be a live read, and must degrade honestly ───────────────


def test_registry_unreachable_reports_unverified_not_empty(monkeypatch):
    """A registry that cannot be read is not the same as a framework with no plugins."""
    import leapflow.plugins as plugins_pkg
    from leapflow.monitor import evolution_producer as mod

    def _boom():
        raise RuntimeError("no registry in this process")

    monkeypatch.setattr(plugins_pkg, "get_registry", _boom, raising=False)
    monkeypatch.setattr(mod.EvolutionProducer, "_json_store", staticmethod(lambda *a, **k: None))

    findings = _observe()
    assert len(findings) == 1
    payload = findings[0].payload
    assert payload["summary"]["registry_readable"] is False
    assert payload["roster"] == []
    assert "could not be verified" in payload["summary"]["headline"]
    # Unverifiable runtime state is worth surfacing, not filing silently.
    assert findings[0].severity is Severity.NOTABLE


def test_roster_and_topology_come_from_the_live_registry(monkeypatch):
    registry = _Registry(
        {
            "alpha": _Plugin("alpha", [_Tool("a_read", ("read.file",)), _Tool("a_write", ())]),
            "beta": _Plugin("beta", [_Tool("b_ping", ("net.ping",))]),
        },
        version=42,
    )
    _install_registry(
        monkeypatch,
        registry,
        trust=_Trust({"alpha": "VERIFIED", "beta": "DRAFT"}),
        fibers={"alpha": "active", "beta": "loading"},
    )

    payload = _observe()[0].payload
    roster = {row["plugin_id"]: row for row in payload["roster"]}
    assert roster["alpha"]["trust_level"] == "VERIFIED"
    assert roster["alpha"]["trust_class"] == "accruing"
    assert roster["alpha"]["fiber_state"] == "active"
    assert roster["alpha"]["tool_count"] == 2
    assert roster["beta"]["fiber_state"] == "loading"
    assert payload["summary"]["registry_version"] == 42
    assert payload["summary"]["tool_count"] == 3

    node_ids = {node["id"] for node in payload["topology"]["nodes"]}
    assert {"plugin:alpha", "tool:a_read", "capability:read.file"} <= node_ids
    assert {"source": "plugin:alpha", "target": "tool:a_read", "kind": "owns"} in payload[
        "topology"
    ]["edges"]


def test_tools_owned_by_another_plugin_are_excluded(monkeypatch):
    """The roster must match what the model can actually call, not what a plugin declares.

    Tool names are one global namespace arbitrated first-wins, so a losing
    challenger still *declares* the tool while ``tool_owners`` says otherwise.
    """
    registry = _Registry({"winner": _Plugin("winner", [_Tool("shared")])})
    registry.plugins["loser"] = _Plugin("loser", [_Tool("shared")])
    # tool_owners still credits the incumbent.
    _install_registry(monkeypatch, registry)

    roster = {row["plugin_id"]: row for row in _observe()[0].payload["roster"]}
    assert roster["winner"]["tool_count"] == 1
    assert roster["loser"]["tool_count"] == 0


def test_frozen_plugin_is_distinguished_from_draft(monkeypatch):
    """``DRAFT`` cannot say whether a plugin is new or permanently disqualified."""
    registry = _Registry({"new": _Plugin("new", []), "bad": _Plugin("bad", [])})
    _install_registry(
        monkeypatch,
        registry,
        trust=_Trust({"new": "DRAFT", "bad": "DRAFT"}, frozen={"bad"}),
    )

    finding = _observe()[0]
    roster = {row["plugin_id"]: row for row in finding.payload["roster"]}
    assert roster["new"]["trust_class"] == "new_unproven"
    assert roster["new"]["selectable"] == "yes"
    assert roster["bad"]["trust_class"] == "frozen"
    # The frozen-yet-selectable window is the whole point of the column.
    assert roster["bad"]["selectable"] == "no"
    assert finding.payload["summary"]["frozen_count"] == 1
    assert finding.severity is Severity.NOTABLE


def test_absent_trust_ledger_reports_unverified_never_a_guess(monkeypatch):
    """In-process runs bind no advisor; the roster must not invent a trust level."""
    _install_registry(monkeypatch, _Registry({"alpha": _Plugin("alpha", [])}), trust=None)

    finding = _observe()[0]
    assert finding.payload["roster"][0]["trust_level"] == "unverified"
    assert _rows(finding)["trust"]["status"] == UNVERIFIABLE


def test_conflicts_are_surfaced(monkeypatch):
    registry = _Registry(
        {"alpha": _Plugin("alpha", [])},
        conflicts=[_Conflict("dup_tool", "alpha", "beta")],
    )
    _install_registry(monkeypatch, registry)

    finding = _observe()[0]
    assert finding.payload["conflicts"] == [
        {"tool_name": "dup_tool", "kept_plugin": "alpha", "rejected_plugin": "beta"}
    ]
    assert finding.severity is Severity.NOTABLE


# ── reachability must never claim more than it measured ──────────────────────


def test_world_model_driver_is_unverifiable_without_a_durable_trace(monkeypatch):
    """The driver's own counts go to an in-memory pipeline observer, not a store.

    Absence of an admitted intent cannot tell "never ran" from "ran and the gate
    correctly refused it", so the only honest verdict is ``unverifiable`` -- never
    ``no_evidence``, which would read as a fault, and never ``wired``.
    """
    from leapflow.monitor import evolution_producer as mod

    class _Obs:
        def unresolved(self, **_kw):
            return [{"observation_id": "o1", "result": {"error_type": "unknown_tool"}}]

    _install_registry(monkeypatch, _Registry({}))
    monkeypatch.setattr(
        mod.EvolutionProducer,
        "_json_store",
        staticmethod(
            lambda layout_attr, *a, **k: _Obs()
            if layout_attr == "capability_observations_path"
            else None
        ),
    )

    row = _rows(_observe()[0])["world_model_driver"]
    assert row["status"] == UNVERIFIABLE
    assert "not persisted" in row["evidence"]
    assert "accepted_evidence_kinds" in row["next_step"]


def test_world_model_driver_is_wired_once_an_intent_is_admitted(monkeypatch):
    """An admitted intent is the one durable trace that the driver reached the pipeline."""
    from leapflow.monitor import evolution_producer as mod

    class _Obs:
        def unresolved(self, **_kw):
            return [
                {"observation_id": "o1", "result": {"error_type": "world_model_intent"}},
                {"observation_id": "o2", "result": {"error_type": "unknown_tool"}},
            ]

    _install_registry(monkeypatch, _Registry({}))
    monkeypatch.setattr(
        mod.EvolutionProducer,
        "_json_store",
        staticmethod(
            lambda layout_attr, *a, **k: _Obs()
            if layout_attr == "capability_observations_path"
            else None
        ),
    )

    row = _rows(_observe()[0])["world_model_driver"]
    assert row["status"] == WIRED
    assert "1 admitted" in row["evidence"]


def test_module_existence_is_never_reported_as_wired(monkeypatch):
    """The three awaiting-wiring segments have modules in the tree.

    Their presence must not read as evidence: a module with no caller is exactly
    the failure this panel exists to expose.
    """
    _install_registry(monkeypatch, _Registry({}))

    rows = _rows(_observe()[0])
    for key in ("effect_verification", "quarantine_feed", "reclamation"):
        assert rows[key]["status"] == NO_EVIDENCE, key
        assert rows[key]["next_step"], f"{key} must name what would close it"


def test_evidence_gate_reads_config_and_reports_not_admitted(monkeypatch):
    from leapflow.monitor import evolution_producer as mod

    _install_registry(monkeypatch, _Registry({}))
    monkeypatch.setattr(
        mod.EvolutionProducer,
        "_settings",
        staticmethod(lambda: SimpleNamespace(accepted_evidence_kinds=(), evolution_authorising_origins=())),
    )

    row = _rows(_observe()[0])["evidence_gate"]
    assert row["status"] == NOT_ADMITTED
    assert "accepted_evidence_kinds" in row["next_step"]


def test_evidence_gate_is_wired_once_world_model_intent_is_admitted(monkeypatch):
    from leapflow.monitor import evolution_producer as mod

    _install_registry(monkeypatch, _Registry({}))
    monkeypatch.setattr(
        mod.EvolutionProducer,
        "_settings",
        staticmethod(
            lambda: SimpleNamespace(
                accepted_evidence_kinds=("unknown_tool", "world_model_intent"),
                evolution_authorising_origins=("world_model",),
            )
        ),
    )

    rows = _rows(_observe()[0])
    assert rows["evidence_gate"]["status"] == WIRED
    assert rows["authorising_origins"]["status"] == WIRED


def test_unreadable_settings_are_unverifiable_not_no_evidence(monkeypatch):
    """"Could not read" and "nothing observed" are different answers."""
    from leapflow.monitor import evolution_producer as mod

    _install_registry(monkeypatch, _Registry({}))
    monkeypatch.setattr(mod.EvolutionProducer, "_settings", staticmethod(lambda: None))

    rows = _rows(_observe()[0])
    assert rows["evidence_gate"]["status"] == UNVERIFIABLE
    assert rows["authorising_origins"]["status"] == UNVERIFIABLE


def test_store_backed_segments_report_evidence_when_data_exists(monkeypatch):
    from leapflow.monitor import evolution_producer as mod

    class _Obs:
        def unresolved(self, **_kw):
            return [{"observation_id": "o1"}, {"observation_id": "o2"}]

    class _Queue:
        def list_items(self, **_kw):
            return [SimpleNamespace(status="PENDING"), SimpleNamespace(status="PROBATION")]

    class _Plans:
        def latest(self):
            return {"policy_decision": {"action": "propose"}}

    stores = {
        "capability_observations_path": _Obs(),
        "capability_proposal_queue_path": _Queue(),
        "capability_plans_path": _Plans(),
    }
    _install_registry(monkeypatch, _Registry({}))
    monkeypatch.setattr(
        mod.EvolutionProducer,
        "_json_store",
        staticmethod(lambda layout_attr, *a, **k: stores.get(layout_attr)),
    )

    rows = _rows(_observe()[0])
    assert rows["observations"]["status"] == WIRED
    assert "2 open" in rows["observations"]["evidence"]
    assert rows["lifecycle"]["status"] == WIRED
    assert "PENDING=1" in rows["lifecycle"]["evidence"]
    assert rows["policy"]["status"] == WIRED
    assert "propose" in rows["policy"]["evidence"]


def test_raising_store_is_unverifiable_and_does_not_fail_the_cycle(monkeypatch):
    from leapflow.monitor import evolution_producer as mod

    class _Broken:
        def unresolved(self, **_kw):
            raise OSError("disk gone")

        def list_items(self, **_kw):
            raise OSError("disk gone")

        def latest(self):
            raise OSError("disk gone")

    _install_registry(monkeypatch, _Registry({}))
    monkeypatch.setattr(mod.EvolutionProducer, "_json_store", staticmethod(lambda *a, **k: _Broken()))

    rows = _rows(_observe()[0])
    for key in ("observations", "lifecycle", "policy"):
        assert rows[key]["status"] == UNVERIFIABLE, key


# ── severity and dedup ───────────────────────────────────────────────────────


def test_quiet_state_is_info_not_an_alert(monkeypatch):
    """An idle pipeline and an unadmitted evidence kind are correct, quiet states."""
    _install_registry(
        monkeypatch,
        _Registry({"alpha": _Plugin("alpha", [_Tool("a")])}),
        trust=_Trust({"alpha": "PRODUCTION"}),
    )

    finding = _observe()[0]
    assert finding.severity is Severity.INFO
    assert any(row["status"] == NO_EVIDENCE for row in finding.payload["reachability"])


def test_dedup_key_is_stable_for_unchanged_state_across_cycles(monkeypatch):
    """The executor skips a duplicate dedup key, so an unchanged framework must
    produce the same key even though the clock advanced."""
    registry = _Registry({"alpha": _Plugin("alpha", [_Tool("a")])})
    _install_registry(monkeypatch, registry, trust=_Trust({"alpha": "VERIFIED"}))

    first = _observe(now=1000.0)[0]
    second = _observe(now=9999.0)[0]
    assert first.dedup_key == second.dedup_key
    # ...and the timestamp is still carried, so the finding is not undatable.
    assert first.ts == 1000.0
    assert second.ts == 9999.0


def test_dedup_key_changes_when_the_framework_changes(monkeypatch):
    registry = _Registry({"alpha": _Plugin("alpha", [_Tool("a")])})
    _install_registry(monkeypatch, registry, trust=_Trust({"alpha": "DRAFT"}))
    before = _observe()[0].dedup_key

    _install_registry(monkeypatch, registry, trust=_Trust({"alpha": "VERIFIED"}))
    assert _observe()[0].dedup_key != before


def test_registry_version_change_changes_the_dedup_key(monkeypatch):
    plugins = {"alpha": _Plugin("alpha", [_Tool("a")])}
    _install_registry(monkeypatch, _Registry(plugins, version=1))
    before = _observe()[0].dedup_key

    _install_registry(monkeypatch, _Registry(plugins, version=2))
    assert _observe()[0].dedup_key != before


# ── payload bounds and contract ──────────────────────────────────────────────


def test_payload_is_bounded(monkeypatch):
    """A producer that does not bound its payload pushes current findings out of the frame."""
    from leapflow.monitor import evolution_producer as mod

    plugins = {
        f"p{i}": _Plugin(f"p{i}", [_Tool(f"t{i}_{j}", (f"cap.{i}.{j}",)) for j in range(6)])
        for i in range(80)
    }
    _install_registry(monkeypatch, _Registry(plugins))

    payload = _observe()[0].payload
    assert len(payload["roster"]) <= mod._MAX_ROSTER
    assert len(payload["topology"]["nodes"]) <= mod._MAX_TOPOLOGY_NODES
    assert len(payload["topology"]["edges"]) <= mod._MAX_TOPOLOGY_EDGES
    assert len(payload["capability_map"]) <= mod._MAX_CAPABILITY_MAP


def test_capability_ownership_is_emitted_in_renderer_compatible_shapes(monkeypatch):
    """The shipped EntityGraph renderer is a badge cloud, not a graph.

    It reads ``props.data`` through ``asArray``, which returns ``[]`` for anything
    that is not a list, and shows each item's ``name``. A nodes/edges mapping
    therefore renders an empty panel and reports no fault -- so the producer must
    also emit the flat projections the view can actually bind.
    """
    registry = _Registry(
        {"alpha": _Plugin("alpha", [_Tool("a_read", ("read.file", "read.dir"))])}
    )
    _install_registry(monkeypatch, registry)

    payload = _observe()[0].payload

    # Badge cloud: a list whose items carry ``name``.
    assert isinstance(payload["capability_badges"], list)
    assert payload["capability_badges"] == [{"name": "read.dir"}, {"name": "read.file"}]

    # Table: one row per (capability, tool, plugin) with the keys the columns read.
    assert {"capability": "read.file", "tool": "a_read", "plugin": "alpha"} in payload[
        "capability_map"
    ]
    assert len(payload["capability_map"]) == 2

    # The general graph stays for a future real renderer.
    assert payload["topology"]["nodes"] and payload["topology"]["edges"]


def test_evolution_template_binds_only_shapes_its_renderers_read():
    """Guard the trap above at the template level, for this template.

    ``EntityGraph`` and ``Table`` both read ``props.data``; a template binding a
    mapping to either renders headings over nothing. Asserted here rather than
    only in the SDUI suite because the payload contract is this producer's.
    """
    from leapflow.dashboard.templates import TemplateLibrary

    raw = TemplateLibrary().load("evolution")
    assert raw is not None, "evolution template must ship"

    list_valued = {
        "evolution.reachability",
        "evolution.roster",
        "evolution.conflicts",
        "evolution.capability_map",
        "evolution.capability_badges",
        "evolution.timeline",
        "evolution.mutation_matrix",
        "evolution.trace_feed",
        "evolution.unadmitted",
        "evolution.fiber_transitions",
        "evolution.trust_mix",
        "evolution.reachability_mix",
        "evolution.provenance_mix",
        "evolution.reclaim_candidates",
        "evolution.summary.suggestions",
    }
    binds: list[tuple[str, str]] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            props = node.get("props")
            if node.get("type") in ("EntityGraph", "Table") and isinstance(props, dict):
                bind = props.get("bind")
                if isinstance(bind, str):
                    binds.append((str(node.get("type")), bind))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(raw)
    assert binds, "template must bind at least one data-driven panel"
    for component, bind in binds:
        assert bind in list_valued, f"{component} binds {bind!r}, which is not a list-valued key"


def test_roster_records_whether_a_plugin_was_ever_selected(monkeypatch):
    """``ever_used`` is a durable fact, chosen over a live call counter on purpose.

    A registered, unselectable, never-used artifact is the reclamation case, and
    that question survives between cycles. A raw counter would not: this finding
    dedups on a content fingerprint, so a per-tick metric either churns a row
    every cycle or freezes on the board while still looking current.
    """
    registry = _Registry({"used": _Plugin("used", []), "idle": _Plugin("idle", [])})
    _install_registry(monkeypatch, registry, usage=_Usage({"used": 12, "idle": 0}))

    roster = {row["plugin_id"]: row for row in _observe()[0].payload["roster"]}
    assert roster["used"]["ever_used"] == "yes"
    assert roster["idle"]["ever_used"] == "no"


def test_roster_renders_no_live_metric_columns(monkeypatch):
    """Guard against reintroducing a value that freezes while looking current.

    Any rendered field that changes every cycle must either be in the fingerprint
    (churning a row per tick) or be absent. Error rates and call counts belong to
    the ``plugin_health`` domain, which alerts on them directly.
    """
    _install_registry(
        monkeypatch,
        _Registry({"alpha": _Plugin("alpha", [])}),
        usage=_Usage({"alpha": 5}),
    )

    row = _observe()[0].payload["roster"][0]
    assert "error_rate" not in row
    assert "total_calls" not in row


def test_every_rendered_roster_field_is_covered_by_the_fingerprint(monkeypatch):
    """A rendered value left out of the fingerprint freezes on the page.

    The executor skips a finding whose dedup key already exists, so a field the
    board shows but the fingerprint ignores keeps its first-observed value
    forever while appearing live. Asserted by flipping each field in turn.
    """
    from leapflow.monitor.evolution_producer import EvolutionProducer as P

    base = {
        "plugin_id": "alpha",
        "fiber_state": "active",
        "trust_level": "DRAFT",
        "trust_class": "new_unproven",
        "selectable": True,
        "ever_used": False,
        "tool_count": 1,
    }
    payload = {
        "summary": {"registry_version": 1, "registry_readable": True},
        "roster": [base],
        "conflicts": [],
        "reachability": [{"key": "observations", "status": "no_evidence"}],
    }
    reference = P._fingerprint(payload)

    # ``trust_class`` and ``tool_count`` are derived from covered fields
    # (trust_level / the tool list), so flipping them alone is not required to
    # move the fingerprint; every independently-observed field must.
    for field, changed in (
        ("fiber_state", "disposed"),
        ("trust_level", "VERIFIED"),
        ("selectable", False),
        ("ever_used", True),
    ):
        variant = dict(payload)
        variant["roster"] = [{**base, field: changed}]
        assert P._fingerprint(variant) != reference, f"{field} is rendered but not fingerprinted"


def test_producer_declares_the_framework_evolution_domain():
    assert EvolutionProducer().domain == "framework_evolution"


def test_snapshot_only_state_is_declared_not_silently_empty(monkeypatch):
    """With no decision record there is no causal history; the view must say why."""
    _install_registry(monkeypatch, _Registry({}))

    payload = _observe()[0].payload
    assert payload["episodes"] == []
    assert payload["timeline"] == []
    assert payload["degraded"] is True
    assert payload["degraded_reason"]
    # Nothing has been closed, so there is no closure to qualify. Claiming an
    # L2 caveat here would attach a warning to an empty table.
    assert payload["summary"]["l2_only_closures"] is False


def test_unrebuildable_history_is_not_reported_as_no_activity(monkeypatch):
    """"Could not look" and "nothing to see" must not share one explanation.

    Both leave the timeline empty, so the only thing separating a fault from a
    quiet system is what the panel says about it. Reporting an unreadable store as
    "nothing has been recorded" would present a local defect as an absence of
    activity -- the same conflation the reachability rows exist to prevent.
    """
    from leapflow.monitor.evolution_producer import EvolutionProducer

    _install_registry(monkeypatch, _Registry({}))

    class _Broken(EvolutionProducer):
        def _episodes(self, ctx):
            return None

    class _Empty(EvolutionProducer):
        def _episodes(self, ctx):
            return ()

    broken = asyncio.run(_Broken().observe(_ctx()))[0].payload
    empty = asyncio.run(_Empty().observe(_ctx()))[0].payload

    assert broken["degraded"] is True and empty["degraded"] is True
    assert broken["degraded_kind"] == "unverifiable"
    assert empty["degraded_kind"] == "no_evidence"
    assert broken["degraded_reason"] != empty["degraded_reason"]
    assert "could not be read" in broken["degraded_reason"]
    # The fault must not be described as an absence of activity.
    assert "No capability decision has been recorded" not in broken["degraded_reason"]


def test_every_rendered_episode_field_is_covered_by_the_fingerprint():
    """Same freezing hazard as the roster, applied to the timeline.

    Only independently-observed fields need their own coverage:
    ``verification_tier`` is derived from ``gap_closure``, and ``outcome`` from
    ``gap_closure``/``mutation_action``, both of which are covered. A change to
    ``driver``/``capability``/``hypothesis`` can only arrive with a new decision
    record, which brings a new ``episode_id``.
    """
    from leapflow.monitor.evolution_producer import EvolutionProducer as P

    base = {
        "episode_id": "ep-1",
        "status": "committed",
        "gap_closure": "resolved",
        "mutation_action": "install",
        "trust_now": "DRAFT",
    }
    payload = {
        "summary": {"registry_version": 1, "registry_readable": True},
        "roster": [],
        "conflicts": [],
        "reachability": [],
        "episodes": [base],
    }
    reference = P._fingerprint(payload)

    for field, changed in (
        ("episode_id", "ep-2"),
        ("status", "aborted"),
        ("gap_closure", "reopened"),
        ("mutation_action", "rollback"),
        ("trust_now", "VERIFIED"),
    ):
        variant = dict(payload)
        variant["episodes"] = [{**base, field: changed}]
        assert P._fingerprint(variant) != reference, (
            f"{field} is rendered but not fingerprinted; the timeline would freeze"
        )

    # A new episode must refresh the board even if the framework itself is idle.
    grown = dict(payload)
    grown["episodes"] = [base, {**base, "episode_id": "ep-2"}]
    assert P._fingerprint(grown) != reference


def test_every_closed_vocabulary_the_payload_emits_is_translated():
    """Payload enums are rendered as table cells, so they need translating too.

    The client passes every string cell through its translator with a raw
    fallback, so an untranslated enum does not fail -- it renders English
    snake_case inside an otherwise localised page. The template-literal contract
    cannot catch this, because these words are *data*, and that gap is exactly how
    'declared_fitness' and 'no_evidence' reached five locales untranslated.

    Identifiers (plugin ids, tool names, capabilities) are correctly excluded:
    they are names, not vocabulary.
    """
    import sys

    sys.path.insert(0, "tests")
    from test_dashboard_i18n_static import _translation_tables

    from leapflow.domain.evolution_trace import (
        ABORTED,
        COMMITTED,
        CONFORMANCE,
        DECLARED_FITNESS,
        NOT_APPLICABLE,
        OBSERVED_EFFECT,
        OPEN,
        REOPENED,
        RESOLVED,
        STILL_OPEN,
    )
    from leapflow.monitor import evolution_producer as ep

    vocabulary = {
        # reachability
        ep.WIRED, ep.NO_EVIDENCE, ep.UNVERIFIABLE, ep.NOT_ADMITTED,
        # trust class + level
        ep._TRUST_CLASS_FROZEN, "unverified", *ep._TRUST_CLASS.values(), *ep._TRUST_CLASS,
        # rendered booleans
        ep._YES, ep._NO,
        # provenance: the distinction the board exists to report
        ep._BUILT_IN, ep._SELF_ACQUIRED,
        # episode + gap vocabularies
        COMMITTED, OPEN, ABORTED, RESOLVED, REOPENED, STILL_OPEN, NOT_APPLICABLE,
        CONFORMANCE, DECLARED_FITNESS, OBSERVED_EFFECT,
        # drivers the ledger can classify
        "world_model", "unknown_tool", "environment_probe", "manual", "unknown",
        # mutation actions
        "install", "reload", "disable", "remove", "rollback", "none",
    }
    tables = _translation_tables()
    assert tables, "no translation tables discovered"
    for locale, known in tables.items():
        if locale == "en":
            continue
        missing = sorted(word for word in vocabulary if word not in known)
        assert not missing, f"{locale} renders these payload enums untranslated: {missing}"


def test_rendered_flags_are_translatable_words_not_json_literals():
    """A boolean cell reaches the page as ``true``/``false`` in every language.

    The client's value translator only handles strings, so a raw bool bypasses it
    entirely. Emitting a vocabulary key keeps the column localisable.
    """
    from leapflow.monitor import evolution_producer as ep

    row = ep.EvolutionProducer()._roster_row("p", [], {}, None, None)
    assert row["selectable"] in (ep._YES, ep._NO)
    assert row["ever_used"] in (ep._YES, ep._NO)
    assert not isinstance(row["selectable"], bool)
    assert not isinstance(row["ever_used"], bool)


def test_no_section_renders_as_a_bare_title():
    """An empty panel under a heading reads as a load failure, not as 'nothing yet'.

    Guards the ``when``-on-section rule: putting the condition on the child instead
    left seven titled sections empty on a fresh profile.
    """
    from leapflow.dashboard.templates import TemplateLibrary

    for payload in ({}, {"summary": {}}, {"roster": [], "reachability": []}):
        spec = TemplateLibrary().render("evolution", {"evolution": payload})
        empty: list[str] = []

        def walk(nodes):
            for node in nodes:
                children = node.get("children") or []
                if node.get("type") == "Section" and not children:
                    empty.append(str((node.get("props") or {}).get("title")))
                walk(children)

        walk(spec["root"])
        assert not empty, f"sections rendered with a title and no content: {empty}"


def test_the_default_tab_is_never_blank():
    """Every visitor lands on the first tab; an empty pane reads as a broken page.

    The hazard is specific to tabs: with no causal history the other panes still
    have content, so the failure is invisible unless the first pane is checked on
    its own. Exercised across the states a fresh profile actually passes through.
    """
    from leapflow.dashboard.templates import TemplateLibrary

    payloads = (
        # Nothing at all but a summary: the state right after the first cycle.
        {"summary": {"active_plugins": 0}, "degraded": True},
        # A framework with plugins but no evolution yet -- the common case.
        {
            "summary": {"active_plugins": 17, "tool_count": 55},
            "degraded": True,
            "roster": [{"plugin_id": "a", "provenance": "built_in"}],
            "reachability": [{"stage": "s", "status": "no_evidence"}],
        },
    )
    for payload in payloads:
        spec = TemplateLibrary().render("evolution", {"evolution": payload})

        def first_tab(nodes):
            for node in nodes:
                if node.get("type") == "Tabs":
                    tabs = node.get("children") or []
                    return tabs[0] if tabs else None
                found = first_tab(node.get("children") or [])
                if found is not None:
                    return found
            return None

        tab = first_tab(spec["root"])
        assert tab is not None, "template no longer has tabs"
        assert tab.get("children"), (
            f"the default tab rendered empty for payload keys {sorted(payload)}"
        )
