# Copyright (c) Alibaba, Inc. and its affiliates.
"""Phase D: close the loop, and make it actually run in production.

The chain built in T1-T3 and C1-C3 was complete on paper and inert in fact.
``self.lifecycle_governor`` was never assigned anywhere, so the sweep resolved the
governor to ``None``, ``record_outcome`` was never called, and the degradation evidence
that the teacher prompt, the challenger identity, and the proposal path were all built to
consume was never produced. The suite was green throughout, because every unit test
constructs the governor itself -- which is exactly the shape of defect a unit test cannot
see: the wiring, not the logic.

With evidence flowing, the loop still needed closing. The teacher was shown the same
degradation every session and could only ever reach the same conclusion, having no way to
learn that its previous answer did not work.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.domain.adaptation_verdict import AdaptationVerdict
from leapflow.engine.engine import AgentEngine
from leapflow.learning.capability_observation import (
    CAPABILITY_DEGRADED,
    CapabilityEvidenceClassifier,
    CapabilityObservationService,
)
from leapflow.learning.degradation_sink import (
    build_degradation_sink,
    declared_capabilities_by_plugin,
)
from leapflow.learning.world_model_driver import WorldModelEvolutionDriver
from leapflow.plugins.lifecycle_governor import LifecycleGovernor
from leapflow.plugins.protocol import ToolMetadata
from leapflow.storage.capability_observation_store import JsonCapabilityObservationStore
from leapflow.storage.distilled_knowledge_store import JsonDistilledKnowledgeStore
from leapflow.world_model.trajectory_grader import (
    TeacherVerdict,
    _degraded_capability_section,
)


def _tool(name: str, *capabilities: str) -> ToolMetadata:
    return ToolMetadata(
        name=name,
        description=name,
        parameters_schema={"type": "object", "properties": {}},
        handler=lambda **kwargs: None,
        x_leapflow={"category": "chat", "risk_level": "read_only"},
        provides_capabilities=tuple(capabilities),
    )


def _registry(**plugins: tuple[ToolMetadata, ...]) -> Any:
    # First-wins, as the registry arbitrates it: the incumbent keeps the name and the
    # challenger is recorded as a conflict. A dict comprehension would let the last
    # declaration overwrite the first, which is the opposite of production.
    owners: dict[str, str] = {}
    for pid, tools in plugins.items():
        for t in tools:
            owners.setdefault(t.name, pid)
    handlers = {t.name: t.handler for tools in plugins.values() for t in tools}
    return SimpleNamespace(
        plugins={pid: SimpleNamespace(tools=list(tools)) for pid, tools in plugins.items()},
        tool_owners=owners,
        tool_handlers=handlers,
    )


class _Queue:
    def update(self, *args: Any, **kwargs: Any) -> None:
        return None


class _Outcomes:
    def __init__(self, streak: int = 0) -> None:
        self.streak = streak

    def add_outcome(self, **kwargs: Any) -> None:
        return None

    def failure_streak(self, plugin_id: str) -> int:
        return self.streak


@pytest.fixture
def wired(tmp_path):
    """Governor, observation service and knowledge store, wired as production wires them."""
    service = CapabilityObservationService(
        JsonCapabilityObservationStore(tmp_path / "obs.json"),
        classifier=CapabilityEvidenceClassifier.from_kinds(
            ["unknown_tool", CAPABILITY_DEGRADED]
        ),
    )
    knowledge = JsonDistilledKnowledgeStore(tmp_path / "dk.json")
    registry = _registry(chat_reply_v1=(_tool("chat_reply", "chat.reply"),))
    outcomes = _Outcomes()
    governor = LifecycleGovernor(
        proposal_queue=_Queue(),
        outcome_store=outcomes,
        degradation_sink=build_degradation_sink(
            intake=service,
            registry_provider=lambda: registry,
            knowledge_store=knowledge,
        ),
    )
    return SimpleNamespace(
        service=service, knowledge=knowledge, governor=governor, outcomes=outcomes
    )


def _record(governor: Any, *, ok: bool, failure_class: str = "") -> Any:
    return asyncio.run(
        governor.record_outcome(
            proposal_id="p1",
            plugin_id="chat_reply_v1",
            tool_name="chat_reply",
            ok=ok,
            failure_class=failure_class,
        )
    )


# ── D0: the translation from plugin health to capability evidence ──────────────


def test_plugin_health_becomes_capability_evidence(wired):
    """The governor holds no registry, so this translation is why the sink exists.

    A capability is what a rival can be built for and what knowledge attaches to; a
    plugin id is neither.
    """
    wired.outcomes.streak = 2
    _record(wired.governor, ok=False, failure_class="affordance_removed")

    facts = wired.service.degraded_capabilities()
    assert len(facts) == 1
    assert facts[0]["capability"] == "chat.reply"
    assert facts[0]["plugin_id"] == "chat_reply_v1"
    assert facts[0]["failure_class"] == "affordance_removed"


def test_capabilities_come_from_declarations_through_the_live_catalog():
    """Delegating to the resolver's builder keeps two filters this must not lose.

    A shadowed tool is not live (first-wins arbitration) and an unbound tool is not
    callable, so re-walking the registry would degrade a plugin under a capability it
    does not actually serve in this process.
    """
    registry = _registry(
        a=(_tool("shared", "chat.reply"),),
        b=(_tool("shared", "chat.react"),),  # loses the name to `a`
        c=(_tool("undeclared"),),  # declares nothing
    )
    declared = declared_capabilities_by_plugin(registry)

    assert declared.get("a") == ("chat.reply",)
    assert "b" not in declared, "a shadowed tool is not live"
    assert "c" not in declared, "a plugin declaring nothing cannot be degraded as one"


def test_a_plugin_declaring_no_capability_reports_nothing(tmp_path):
    """Inventing a name from the plugin id would put a fabricated capability in front
    of the teacher, which is worse than reporting nothing."""
    service = CapabilityObservationService(
        JsonCapabilityObservationStore(tmp_path / "obs.json"),
        classifier=CapabilityEvidenceClassifier.from_kinds([CAPABILITY_DEGRADED]),
    )
    sink = build_degradation_sink(
        intake=service,
        registry_provider=lambda: _registry(chat_reply_v1=(_tool("chat_reply"),)),
    )
    sink(plugin_id="chat_reply_v1", failure_streak=2, trust_level="DRAFT")

    assert service.degraded_capabilities() == ()


def test_the_registry_is_resolved_per_report_not_captured(tmp_path):
    """Plugins are installed and reloaded at runtime, so a snapshot goes stale."""
    service = CapabilityObservationService(
        JsonCapabilityObservationStore(tmp_path / "obs.json"),
        classifier=CapabilityEvidenceClassifier.from_kinds([CAPABILITY_DEGRADED]),
    )
    registries = [_registry(), _registry(late=(_tool("late_tool", "late.thing"),))]
    sink = build_degradation_sink(
        intake=service, registry_provider=lambda: registries[-1]
    )

    sink(plugin_id="late", failure_streak=2, trust_level="DRAFT")
    assert {f["capability"] for f in service.degraded_capabilities()} == {"late.thing"}


def test_a_broken_registry_does_not_break_governance(tmp_path):
    def _explode():
        raise RuntimeError("registry mid-reload")

    sink = build_degradation_sink(intake=object(), registry_provider=_explode)
    sink(plugin_id="anything", failure_streak=2, trust_level="DRAFT")  # must not raise


# ── D1: the feedback edge ──────────────────────────────────────────────────────


def test_the_teacher_is_shown_what_it_concluded_last_time(wired):
    """Without this the loop is open: the same evidence can only produce the same answer."""
    wired.outcomes.streak = 2
    _record(wired.governor, ok=False, failure_class="affordance_removed")
    wired.knowledge.record(
        AdaptationVerdict.create(
            "absorb", "chat.reply", "the send control moved to the toolbar"
        )
    )

    seen: list[Any] = []

    class _Teacher:
        async def grade_and_propose(self, trajectory, goal="", **kwargs):
            seen.append(kwargs.get("degraded_capabilities"))
            return TeacherVerdict()

    driver = WorldModelEvolutionDriver(
        teacher=_Teacher(), intake=wired.service, knowledge_store=wired.knowledge
    )
    asyncio.run(driver.drive([{"action": "a"}], "reply"))

    facts = seen[0]
    assert facts[0]["prior_action"] == "absorb"
    assert "moved to the toolbar" in facts[0]["prior_knowledge"]

    section = _degraded_capability_section(facts)
    assert "last time you answered 'absorb'" in section


def test_a_capability_with_no_prior_verdict_is_unchanged(wired):
    """Enrichment must not invent history where there is none."""
    wired.outcomes.streak = 2
    _record(wired.governor, ok=False)

    driver = WorldModelEvolutionDriver(
        teacher=SimpleNamespace(), intake=wired.service, knowledge_store=wired.knowledge
    )
    facts = driver._collect_degraded()
    assert facts and "prior_action" not in facts[0]


def test_prior_knowledge_is_history_not_a_verdict_on_the_verdict(wired):
    """Knowledge outliving a failure is evidence the adaptation did not resolve it.

    Not proof the judgement was wrong: the student may never have used it, the
    environment may have moved again, or this may be a different failure. Deciding
    which is what the teacher is for, so the prompt must not pre-empt it.
    """
    facts = (
        {
            "capability": "chat.reply",
            "plugin_id": "chat_reply_v1",
            "failure_streak": 2,
            "failure_class": "",
            "prior_action": "absorb",
            "prior_knowledge": "the control moved",
        },
    )
    section = _degraded_capability_section(facts)
    assert "last time you answered" in section
    # The section must not pre-empt the judgement. It carried the pre-Phase-B binary
    # ("an implementation that is wrong, or an environment that changed... report a gap
    # only for the former"), which contradicted the four-action prompt it sits inside --
    # and which suppresses the environment-upgrade case the whole design exists for.
    assert "Report a gap only" not in section
    assert "implementation that is\nwrong" not in section
    assert "which of the four actions the evidence" in section


# ── D2: recovery retires knowledge ─────────────────────────────────────────────


def test_recovery_retires_the_knowledge_that_described_the_failure(wired):
    """The one retirement neither supersession nor expiry covers.

    No newer verdict is coming precisely because there is no longer anything wrong, so
    without this the knowledge outlives the failure and misleads every later session.
    """
    wired.knowledge.record(
        AdaptationVerdict.create("absorb", "chat.reply", "the control is missing")
    )
    assert wired.knowledge.count() == 1

    wired.outcomes.streak = 0
    _record(wired.governor, ok=True)

    assert wired.knowledge.count() == 0, "a zero streak is the retirement signal"


def test_a_still_failing_capability_keeps_its_knowledge(wired):
    wired.knowledge.record(
        AdaptationVerdict.create("absorb", "chat.reply", "the control is missing")
    )
    wired.outcomes.streak = 2
    _record(wired.governor, ok=False)

    assert wired.knowledge.count() == 1


def test_retirement_without_a_knowledge_store_is_harmless(tmp_path):
    service = CapabilityObservationService(
        JsonCapabilityObservationStore(tmp_path / "obs.json"),
        classifier=CapabilityEvidenceClassifier.from_kinds([CAPABILITY_DEGRADED]),
    )
    sink = build_degradation_sink(
        intake=service,
        registry_provider=lambda: _registry(v1=(_tool("t", "chat.reply"),)),
    )
    sink(plugin_id="v1", failure_streak=0, trust_level="DRAFT")  # must not raise


# ── D3: the recommendation reaches the student ─────────────────────────────────


def _reader(store: Any) -> AgentEngine:
    engine = AgentEngine.__new__(AgentEngine)
    engine._knowledge_store = store
    engine._environment_fingerprint_id = ""
    engine._settings = SimpleNamespace(distilled_knowledge_limit=12)
    return engine


def test_a_rebind_target_tells_the_student_what_to_prefer(tmp_path):
    """Stored and never read by anyone is the failure mode this closes.

    The student would be told a problem exists without being told the answer that was
    already worked out.
    """
    store = JsonDistilledKnowledgeStore(tmp_path / "dk.json")
    store.record(
        AdaptationVerdict.create(
            "rebind", "chat.reply", "the app is now v3", target="chat_reply_v3"
        )
    )

    block = _reader(store)._distilled_knowledge_context()
    assert "Prefer chat_reply_v3." in block


def test_an_escalation_target_names_what_a_person_must_do(tmp_path):
    store = JsonDistilledKnowledgeStore(tmp_path / "dk.json")
    store.record(
        AdaptationVerdict.create(
            "escalate",
            "drive.upload",
            "uploading is refused",
            target="grant the drive.file scope",
        )
    )

    block = _reader(store)._distilled_knowledge_context()
    assert "This needs a person to: grant the drive.file scope." in block


def test_a_verdict_without_a_target_adds_no_hint(tmp_path):
    store = JsonDistilledKnowledgeStore(tmp_path / "dk.json")
    store.record(AdaptationVerdict.create("absorb", "chat.react", "it moved"))

    block = _reader(store)._distilled_knowledge_context()
    assert "- chat.react: it moved" in block
    assert "Prefer" not in block and "needs a person" not in block


# ── the whole loop, over two sessions ──────────────────────────────────────────


def test_two_sessions_close_the_loop(wired):
    """Session one distils; session two is told what session one concluded.

    This is the behaviour the phase exists for, and none of it happened before: the
    governor was never constructed, so no evidence was produced, so no verdict was
    reachable, so nothing was distilled and nothing could be reconsidered.
    """
    # Session one: the capability degrades and the teacher absorbs it.
    wired.outcomes.streak = 2
    _record(wired.governor, ok=False, failure_class="affordance_removed")

    verdicts = (
        AdaptationVerdict.create("absorb", "chat.reply", "the control moved to the toolbar"),
    )
    seen: list[Any] = []

    class _Teacher:
        def __init__(self, out) -> None:
            self.out = out

        async def grade_and_propose(self, trajectory, goal="", **kwargs):
            seen.append(kwargs.get("degraded_capabilities"))
            return TeacherVerdict(grades=(), verdicts=self.out)

    first = WorldModelEvolutionDriver(
        teacher=_Teacher(verdicts), intake=wired.service, knowledge_store=wired.knowledge
    )
    result = asyncio.run(first.drive([{"action": "a"}], "reply"))
    assert result.distilled == ("chat.reply",)
    assert seen[0] and "prior_action" not in seen[0][0], "nothing was known yet"

    # Session two: it still fails, and now the teacher sees its own previous answer.
    _record(wired.governor, ok=False, failure_class="affordance_removed")
    second = WorldModelEvolutionDriver(
        teacher=_Teacher(()), intake=wired.service, knowledge_store=wired.knowledge
    )
    asyncio.run(second.drive([{"action": "a"}], "reply"))
    assert seen[1][0]["prior_action"] == "absorb"

    # Session three: it works again, and the knowledge retires itself.
    wired.outcomes.streak = 0
    _record(wired.governor, ok=True)
    assert wired.knowledge.count() == 0


# ── the wiring itself, which is what unit tests cannot see ─────────────────────


def test_the_production_path_actually_builds_a_governor(tmp_path):
    """The defect this guards was reported as fixed while the wiring was absent.

    ``self.lifecycle_governor`` was read through ``getattr(..., None)`` and assigned
    nowhere, so the sweep ran with ``governor=None`` for the life of the product. Every
    test of the chain passed because every test constructed the governor itself -- so the
    only thing that can catch it is a test that goes through the production resolver.
    """
    from leapflow.cli.context import Context

    layout = SimpleNamespace(
        distilled_knowledge_path=tmp_path / "dk.json",
        capability_observations_path=tmp_path / "obs.json",
        capability_proposal_queue_path=tmp_path / "queue.json",
        plugin_outcomes_path=tmp_path / "outcomes.json",
    )
    context = Context.__new__(Context)
    context.settings = SimpleNamespace(
        profile_layout=layout,
        distilled_knowledge_ttl_s=0.0,
        accepted_evidence_kinds=("capability_degraded",),
        workspace_root=str(tmp_path),
    )

    governor = context._resolve_lifecycle_governor()

    assert governor is not None, "the sweep would run with governor=None"
    assert context._resolve_lifecycle_governor() is governor, "built once, not per sweep"
    # And the sink must be attached, or the chain is wired but silent.
    assert governor._degradation_sink is not None


def test_the_governor_uses_the_durable_trust_ledger(tmp_path):
    """A fresh ledger would give one process two divergent views of trust.

    The governor defaults to its own in-memory ``PluginTrustLedger`` when passed nothing,
    so the transitions it computed would land in a throwaway object while the persistent
    ledger the advisor reads stayed at DRAFT -- and no plugin could ever earn PRODUCTION,
    which is the whole of Progressive Trust.
    """
    from leapflow.cli.context import Context
    from leapflow.learning.plugin_advisor import (
        PluginAdvisor,
        get_default_advisor,
        set_default_advisor,
    )
    from leapflow.learning.plugin_trust import PluginTrustLedger

    previous = get_default_advisor()
    durable = PluginTrustLedger()
    set_default_advisor(PluginAdvisor(durable, SimpleNamespace()))
    try:
        context = Context.__new__(Context)
        assert context._process_trust_ledger() is durable
    finally:
        if previous is not None:
            set_default_advisor(previous)


def test_a_missing_profile_layout_degrades_instead_of_failing():
    """No durable place to record governance is the one legitimate reason to skip it."""
    from leapflow.cli.context import Context

    context = Context.__new__(Context)
    context.settings = SimpleNamespace(profile_layout=None)
    assert context._resolve_lifecycle_governor() is None
