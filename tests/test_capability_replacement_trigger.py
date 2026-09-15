# Copyright (c) Alibaba, Inc. and its affiliates.
"""T1-T3: the trigger for replacing an existing capability's implementation.

Before this, self-evolution only ever fired on a *missing* capability. An existing
provider that kept failing was handled by quarantine -- which disables it, creating a
gap, which then triggers generation. That ordering has three consequences the EVO-02
episode measured: an availability hole between disable and install, never any two
providers admissible at once, and the question "is the replacement actually better?"
never being asked, because the incumbent is gone by the time the rival arrives.

These tests cover the three pieces that make the other ordering possible:

* **T1** governance reports a failure that left the plugin *in service* -- the state
  between healthy and quarantined, which had no expression at all.
* **T2** the teacher is shown those facts and adjudicates. A failure count cannot tell
  a wrong implementation from a moved environment; both produce the same streak and
  want opposite actions.
* **T3** an admitted intent becomes a queued proposal, with an identity that lets a
  rival coexist with the incumbent it competes against.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.domain.evolution_intent import EvolutionIntent
from leapflow.learning.capability_gap_detector import CapabilityGapDetector
from leapflow.learning.capability_observation import (
    CAPABILITY_DEGRADED,
    EVIDENCE_SURVIVING_RESOLUTION,
    CapabilityEvidenceClassifier,
    CapabilityObservationService,
)
from leapflow.learning.world_model_driver import (
    CapabilityGapTeacher,
    WorldModelEvolutionDriver,
)
from leapflow.plugins.lifecycle_governor import LifecycleGovernor
from leapflow.storage.capability_observation_store import JsonCapabilityObservationStore


# ── T1: governance reports a still-serving failure ─────────────────────────────


class _Queue:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, proposal_id: str, **fields: Any) -> None:
        self.updates.append({"proposal_id": proposal_id, **fields})


class _Outcomes:
    def __init__(self, streak: int = 0) -> None:
        self.streak = streak
        self.added: list[dict[str, Any]] = []

    def add_outcome(self, **kwargs: Any) -> None:
        self.added.append(kwargs)

    def failure_streak(self, plugin_id: str) -> int:
        return self.streak


class _Actor:
    def __init__(self) -> None:
        self.disabled: list[str] = []

    async def disable(self, *, plugin_id: str) -> dict[str, Any]:
        self.disabled.append(plugin_id)
        return {"ok": True}


def _governor(streak: int, sink: Any = None, actor: Any = None) -> LifecycleGovernor:
    return LifecycleGovernor(
        proposal_queue=_Queue(),
        outcome_store=_Outcomes(streak),
        lifecycle_actor=actor,
        degradation_sink=sink,
    )


async def _record(governor: LifecycleGovernor, ok: bool, **kw: Any) -> Any:
    return await governor.record_outcome(
        proposal_id="p1", plugin_id="chat_reply_v1", tool_name="chat_reply_v1", ok=ok, **kw
    )


@pytest.mark.asyncio
async def test_a_failure_that_leaves_the_plugin_serving_is_reported():
    """The state that had no expression: failing, not disabled, still answering calls."""
    seen: list[dict[str, Any]] = []
    governor = _governor(streak=2, sink=lambda **kw: seen.append(kw))

    result = await _record(governor, ok=False)

    assert result.action != "quarantine", "two failures must not disable"
    assert seen == [{"plugin_id": "chat_reply_v1", "failure_streak": 2,
                     "trust_level": "DRAFT", "failure_class": ""}]


@pytest.mark.asyncio
async def test_a_quarantined_plugin_is_not_reported_as_degraded():
    """It is no longer serving, so there is nothing to build a rival *alongside*."""
    seen: list[dict[str, Any]] = []
    actor = _Actor()
    governor = _governor(streak=3, sink=lambda **kw: seen.append(kw), actor=actor)

    result = await _record(governor, ok=False)

    assert result.action == "quarantine"
    assert actor.disabled == ["chat_reply_v1"]
    assert seen == [], "a disabled plugin is a gap, not a degradation"


@pytest.mark.asyncio
async def test_a_success_reports_a_zero_streak_so_degradation_can_be_retired():
    """A health signal that only fires one way has no way back.

    Reporting only failures left a degradation record open forever: ``unresolved()``
    would grow monotonically and the teacher would keep being told a capability is
    failing long after it recovered, driving it to propose rivals for a healthy
    provider. A zero streak after a success is the retirement signal, and it is
    declarative -- the sink never has to infer recovery from an absence of reports.
    """
    seen: list[dict[str, Any]] = []
    await _record(_governor(streak=0, sink=lambda **kw: seen.append(kw)), ok=True)
    assert seen == [{"plugin_id": "chat_reply_v1", "failure_streak": 0,
                     "trust_level": "DRAFT", "failure_class": ""}]


@pytest.mark.asyncio
async def test_a_failing_sink_never_breaks_governance():
    """Reporting is advisory; governance drives trust and quarantine."""

    def explode(**_: Any) -> None:
        raise RuntimeError("sink down")

    result = await _record(_governor(streak=1, sink=explode), ok=False)
    assert result.action  # governance still produced a decision


@pytest.mark.asyncio
async def test_governance_works_with_no_sink_installed():
    result = await _record(_governor(streak=1), ok=False)
    assert result.action


# ── the retirement hazard: degradation must survive a met resolution ───────────


def test_degradation_evidence_is_not_retired_by_finding_the_incumbent(tmp_path):
    """The provider that exists *is* the thing being reported.

    ``resolve_capability`` retires evidence whose gap is closed. For ``unknown_tool``
    a provider existing closes it. For degradation it proves nothing -- and retiring it
    here would erase the record at the first resolution after it was written, which is
    the very next turn.
    """
    store = JsonCapabilityObservationStore(tmp_path / "obs.json")
    service = CapabilityObservationService(
        store,
        classifier=CapabilityEvidenceClassifier.from_kinds(
            ["unknown_tool", CAPABILITY_DEGRADED]
        ),
    )

    service.observe_result(
        {"error_type": CAPABILITY_DEGRADED, "capability": "chat.reply",
         "plugin_id": "chat_reply_v1", "failure_streak": 2}
    )
    service.observe_result(
        {"error_type": "unknown_tool", "original_tool_name": "send_chat"}
    )
    assert len(store.unresolved()) == 2

    # A resolution that found a provider for both names.
    service.resolve_capability("chat.reply", reason="incumbent serves it")
    service.resolve_capability("send_chat", reason="provider found")

    surviving = [
        str((r.get("result") or {}).get("error_type") or "") for r in store.unresolved()
    ]
    assert surviving == [CAPABILITY_DEGRADED], surviving
    assert CAPABILITY_DEGRADED in EVIDENCE_SURVIVING_RESOLUTION


# ── T2: the teacher adjudicates, and is given facts not a verdict ──────────────


def test_the_degradation_section_states_facts_without_deciding():
    """A streak cannot tell a wrong implementation from a moved environment.

    Both produce consecutive failures and want opposite actions -- rebuild versus
    rebind -- so the prompt must hand the teacher the observation and ask it to judge,
    not hand it a conclusion.

    This section originally asked the pre-Phase-B binary ("an implementation that is
    wrong, or an environment that changed... report a gap only for the former"), which
    both contradicted the four-action prompt it sits inside and suppressed the
    environment-upgrade case: on an upgrade the implementation was not written wrongly,
    so the honest answer to "is it wrong" is no and nothing would happen.
    """
    from leapflow.world_model.trajectory_grader import _degraded_capability_section

    section = _degraded_capability_section(
        [{"capability": "chat.reply", "plugin_id": "chat_reply_v1", "failure_streak": 2}]
    )

    assert "chat.reply" in section and "chat_reply_v1" in section
    assert "2 consecutive failure" in section
    # It must say the capability exists, so the question is which action -- not whether
    # the ability is absent.
    assert "already exists" in section
    # And it must point at the action space rather than pre-empting the judgement.
    assert "which of the four actions the evidence" in section
    assert "Report a gap only" not in section


def test_a_healthy_session_adds_nothing_to_the_prompt():
    from leapflow.world_model.trajectory_grader import _degraded_capability_section

    assert _degraded_capability_section(()) == ""
    assert _degraded_capability_section([{"capability": ""}]) == ""


class _Teacher:
    """Records the context it was handed."""

    def __init__(self, intents: tuple[EvolutionIntent, ...] = ()) -> None:
        self.intents = intents
        self.saw_degraded: Any = None

    async def grade_and_propose(self, trajectory, goal="", **kwargs):
        self.saw_degraded = kwargs.get("degraded_capabilities")
        return type("V", (), {"grades": (), "intents": self.intents})()


class _OldTeacher:
    """A teacher predating the extra context, to prove the contract stays open."""

    def __init__(self) -> None:
        self.called = False

    async def grade_and_propose(self, trajectory, goal=""):
        self.called = True
        return type("V", (), {"grades": (), "intents": ()})()


class _Intake:
    def __init__(self, admit: bool = True) -> None:
        self.admit = admit
        self.results: list[Any] = []

    def observe_result(self, result, **kwargs):
        self.results.append(result)
        return {"observation_id": "o1"} if self.admit else None

    def requirements(self, *, min_count: int = 1, limit: int = 50):
        return ()


@pytest.mark.asyncio
async def test_the_driver_passes_degradation_facts_to_the_teacher():
    teacher = _Teacher()
    facts = ({"capability": "chat.reply", "plugin_id": "chat_reply_v1", "failure_streak": 2},)
    driver = WorldModelEvolutionDriver(
        teacher=teacher, intake=_Intake(), degraded_capabilities=lambda: facts
    )

    await driver.drive([{"action": "reply"}], "reply in the thread")

    assert teacher.saw_degraded == facts


@pytest.mark.asyncio
async def test_a_teacher_that_predates_the_context_still_grades():
    """Losing the episode's grading over an unknown keyword would be a bad trade."""
    teacher = _OldTeacher()
    driver = WorldModelEvolutionDriver(
        teacher=teacher, intake=_Intake(), degraded_capabilities=lambda: ({"capability": "x"},)
    )

    await driver.drive([{"action": "a"}])

    assert teacher.called is True
    assert isinstance(teacher, CapabilityGapTeacher)


@pytest.mark.asyncio
async def test_unavailable_degradation_facts_degrade_grading_not_the_session():
    def explode():
        raise RuntimeError("store down")

    teacher = _Teacher()
    driver = WorldModelEvolutionDriver(
        teacher=teacher, intake=_Intake(), degraded_capabilities=explode
    )

    result = await driver.drive([{"action": "a"}])

    assert teacher.saw_degraded == ()
    assert result.proposed == 0


# ── T3: an admitted intent becomes a queued proposal ──────────────────────────


def _intent(capability: str = "chat.reply") -> EvolutionIntent:
    return EvolutionIntent.create(
        capability=capability,
        hypothesis="the current implementation keeps failing",
        confidence=0.8,
        expected_effect="the reply reaches the thread",
    )


@pytest.mark.asyncio
async def test_an_admitted_intent_reaches_the_proposal_queue():
    """The last hop: without it an intent becomes a requirement and stops there.

    Resolution reports the capability unmet and nothing turns that into an acquisition,
    which is why ``proposal_from_evolution_intent`` had no caller at all.
    """
    queued: list[Any] = []
    driver = WorldModelEvolutionDriver(
        teacher=_Teacher((_intent(),)),
        intake=_Intake(admit=True),
        proposal_sink=lambda proposal: queued.append(proposal) or proposal.proposal_id,
    )

    result = await driver.drive([{"action": "a"}])

    assert len(queued) == 1
    assert result.queued_proposal_ids == (queued[0].proposal_id,)
    assert result.to_dict()["queued"] == 1


@pytest.mark.asyncio
async def test_an_unadmitted_intent_is_never_queued():
    """The opt-in gate must not be bypassable through the proposal path."""
    queued: list[Any] = []
    driver = WorldModelEvolutionDriver(
        teacher=_Teacher((_intent(),)),
        intake=_Intake(admit=False),
        proposal_sink=lambda proposal: queued.append(proposal),
    )

    result = await driver.drive([{"action": "a"}])

    assert result.proposed == 1, "the teacher still proposed"
    assert result.admitted == 0
    assert queued == [], "not admitted must mean not queued"


@pytest.mark.asyncio
async def test_no_sink_means_no_proposals_and_no_error():
    driver = WorldModelEvolutionDriver(teacher=_Teacher((_intent(),)), intake=_Intake())
    result = await driver.drive([{"action": "a"}])
    assert result.queued_proposal_ids == ()


@pytest.mark.asyncio
async def test_one_failing_sink_call_does_not_stop_the_others():
    calls: list[str] = []

    def sink(proposal):
        calls.append(proposal.plugin_id)
        if len(calls) == 1:
            raise RuntimeError("queue full")
        return proposal.proposal_id

    driver = WorldModelEvolutionDriver(
        teacher=_Teacher((_intent("chat.reply"), _intent("chat.react"))),
        intake=_Intake(admit=True),
        proposal_sink=sink,
    )

    result = await driver.drive([{"action": "a"}])

    assert len(calls) == 2, "the second intent must still be attempted"
    assert len(result.queued_proposal_ids) == 1


# ── rival identity: the collision that would stop competition ─────────────────


def test_a_rival_gets_an_identity_that_can_coexist_with_the_incumbent():
    """A plugin id derived from the capability alone cannot compete with itself.

    Both a gap fill and a rival for ``chat.reply`` would be named
    ``chat_reply_plugin``. Installing the second collides with the first, so the two
    could never be admissible at the same time -- which is the entire purpose of
    proposing a rival.
    """
    detector = CapabilityGapDetector()
    intent = _intent()

    gap_fill = detector.proposal_from_evolution_intent(intent)
    rival = detector.proposal_from_evolution_intent(intent, incumbent="chat_reply_v1")

    assert gap_fill.plugin_id == "chat_reply_plugin"
    assert rival.plugin_id != gap_fill.plugin_id
    assert rival.proposed_tools[0].name != gap_fill.proposed_tools[0].name


def test_a_rival_records_what_it_replaces_for_the_approver():
    """An approver must see it competes with a named incumbent, not fills an empty slot."""
    rival = CapabilityGapDetector().proposal_from_evolution_intent(
        _intent(), incumbent="chat_reply_v1"
    )
    metadata = dict(rival.evidence[0].metadata)
    assert metadata["replaces"] == "chat_reply_v1"
    assert metadata["capability"] == "chat.reply"


def test_successive_rivals_for_one_capability_stay_distinct():
    """Otherwise the second rival overwrites the first and the trial has one arm."""
    detector = CapabilityGapDetector()
    first = detector.proposal_from_evolution_intent(_intent(), incumbent="chat_reply_v1")
    second = detector.proposal_from_evolution_intent(_intent(), incumbent="chat_reply_v1")
    assert first.plugin_id != second.plugin_id


@pytest.mark.asyncio
async def test_the_driver_marks_a_rival_only_when_the_capability_is_degraded():
    """Rival versus gap fill is a registry fact, never a reading of the hypothesis."""
    queued: list[Any] = []
    driver = WorldModelEvolutionDriver(
        teacher=_Teacher((_intent("chat.reply"), _intent("chat.react"))),
        intake=_Intake(admit=True),
        degraded_capabilities=lambda: (
            {"capability": "chat.reply", "plugin_id": "chat_reply_v1", "failure_streak": 2},
        ),
        proposal_sink=lambda proposal: queued.append(proposal) or proposal.proposal_id,
    )

    await driver.drive([{"action": "a"}])

    by_capability = {
        dict(p.evidence[0].metadata)["capability"]: dict(p.evidence[0].metadata)
        for p in queued
    }
    assert by_capability["chat.reply"]["replaces"] == "chat_reply_v1"
    assert "replaces" not in by_capability["chat.react"], "an absent provider is a gap"


def test_a_rival_cannot_widen_the_risk_ceiling():
    """More autonomy than filling a gap, so the clamp must still hold."""
    intent = EvolutionIntent.create(
        capability="chat.reply",
        hypothesis="needs shell access to work properly",
        confidence=0.9,
        max_risk_level="external",
    )
    rival = CapabilityGapDetector().proposal_from_evolution_intent(
        intent, incumbent="chat_reply_v1"
    )
    assert rival.risk_level != "external"
    assert dict(rival.evidence[0].metadata)["requested_max_risk_level"] == "external"


# ── the whole chain, against the real store ───────────────────────────────────


@pytest.mark.asyncio
async def test_the_trigger_chain_survives_the_real_observation_store(tmp_path):
    """T1 to T3 with the durable store in the middle, which is where it broke.

    Every unit above passes with a fake intake. The real store persists only an
    allow-listed set of payload keys, and ``plugin_id``/``failure_streak`` were not on
    it -- so the degradation facts arrived carrying ``None`` for both. The teacher then
    could not tell what would be replaced, and proposal identity fell back to the
    capability-derived name that collides with the incumbent. Nothing raised.
    """
    from leapflow.domain.evolution_intent import WORLD_MODEL_INTENT

    store = JsonCapabilityObservationStore(tmp_path / "obs.json")
    service = CapabilityObservationService(
        store,
        classifier=CapabilityEvidenceClassifier.from_kinds(
            ["unknown_tool", CAPABILITY_DEGRADED, WORLD_MODEL_INTENT]
        ),
    )
    declared = {"chat_reply_v1": ("chat.reply",)}

    def sink(
        *, plugin_id: str, failure_streak: int, trust_level: str, failure_class: str = ""
    ) -> None:
        for capability in declared.get(plugin_id, ()):
            service.observe_result(
                {
                    "error_type": CAPABILITY_DEGRADED,
                    "capability": capability,
                    "plugin_id": plugin_id,
                    "failure_streak": failure_streak,
                    "trust_level": trust_level,
                    "failure_class": failure_class,
                }
            )

    governor = LifecycleGovernor(
        proposal_queue=_Queue(), outcome_store=_Outcomes(2), degradation_sink=sink
    )
    await _record(governor, ok=False)

    # The service's own reader, so the filter and the environment tag are applied
    # once rather than re-derived by every consumer.
    degraded = service.degraded_capabilities

    # The incumbent must survive the round trip through the store.
    facts = degraded()
    assert facts and facts[0]["plugin_id"] == "chat_reply_v1", facts
    assert str(facts[0]["failure_streak"]) == "2", facts

    teacher = _Teacher((_intent("chat.reply"),))
    queued: list[Any] = []
    driver = WorldModelEvolutionDriver(
        teacher=teacher,
        intake=service,
        degraded_capabilities=degraded,
        proposal_sink=lambda proposal: queued.append(proposal) or proposal.proposal_id,
    )

    result = await driver.drive([{"action": "reply"}], "reply in the thread")

    assert result.admitted == 1 and len(result.queued_proposal_ids) == 1
    rival = queued[0]
    assert dict(rival.evidence[0].metadata)["replaces"] == "chat_reply_v1"

    # And it can coexist with what a gap fill for the same capability would be named.
    gap_fill = CapabilityGapDetector().proposal_from_evolution_intent(_intent("chat.reply"))
    assert rival.plugin_id != gap_fill.plugin_id

    # The degradation record is not erased by the incumbent still being found.
    service.resolve_capability("chat.reply", reason="incumbent serves it")
    assert degraded(), "degradation evidence must outlive a met resolution"


@pytest.mark.asyncio
async def test_a_partially_admitted_batch_queues_only_what_was_admitted():
    """The side door the opt-in gate exists to prevent.

    Collecting only the admitted observation *ids* was enough to count admissions and
    not enough to act on them: queueing then received every intent whenever any one of
    them was admitted, so a rejected hypothesis reached the proposal queue anyway. It is
    invisible today because the shipped intake accepts a kind wholesale, and becomes a
    real bypass the moment admission is decided per intent.
    """

    class _Selective:
        """Admits only the capability it was told to."""

        def __init__(self, allow: str) -> None:
            self.allow = allow
            self.seen: list[str] = []

        def observe_result(self, result, **kwargs):
            capability = str((result or {}).get("capability") or "")
            self.seen.append(capability)
            return {"observation_id": f"o-{capability}"} if capability == self.allow else None

        def requirements(self, *, min_count: int = 1, limit: int = 50):
            return ()

    intake = _Selective("chat.reply")
    queued: list[Any] = []
    driver = WorldModelEvolutionDriver(
        teacher=_Teacher((_intent("chat.reply"), _intent("chat.react"))),
        intake=intake,
        proposal_sink=lambda proposal: queued.append(proposal) or proposal.proposal_id,
    )

    result = await driver.drive([{"action": "a"}])

    assert intake.seen == ["chat.reply", "chat.react"], "both were offered to the gate"
    assert result.proposed == 2 and result.admitted == 1
    assert len(queued) == 1, "only the admitted intent may be queued"
    assert dict(queued[0].evidence[0].metadata)["capability"] == "chat.reply"


# ── A4-A6: the facts must be classified, filtered and environment-tagged ──────


def _degraded_service(tmp_path):
    from leapflow.learning.capability_observation import CAPABILITY_DEGRADED

    store = JsonCapabilityObservationStore(tmp_path / "obs.json")
    service = CapabilityObservationService(
        store,
        classifier=CapabilityEvidenceClassifier.from_kinds(
            ["unknown_tool", CAPABILITY_DEGRADED]
        ),
    )
    return store, service


def _env():
    from leapflow.domain.environment_fingerprint import EnvironmentFingerprint
    from leapflow.domain.platform import Capability, PlatformID, PlatformManifest

    return EnvironmentFingerprint.from_platform_manifest(
        PlatformManifest(PlatformID.DARWIN_15, "15.0", frozenset({Capability.FILE_OPS}))
    )


def _observe(service, capability: str, plugin: str, failure_class: str, environment=None):
    service.observe_result(
        {
            "error_type": CAPABILITY_DEGRADED,
            "capability": capability,
            "plugin_id": plugin,
            "failure_streak": 2,
            "failure_class": failure_class,
        },
        environment=environment,
    )


def test_a_retry_owned_failure_never_reaches_the_teacher(tmp_path):
    """A timeout is the retry layer's business, and the teacher has one lever: rebuild.

    Forwarding transients would ask a hindsight evaluator to adjudicate something that
    already resolved itself, and the only verdict that changes anything is the most
    expensive response in the system.
    """
    from leapflow.learning.capability_observation import RETRY_OWNED_FAILURE_CLASSES

    store, service = _degraded_service(tmp_path)
    _observe(service, "chat.reply", "chat_v1", "affordance_removed")
    _observe(service, "net.fetch", "flaky_v1", "timeout")

    assert len(store.unresolved()) == 2, "both are recorded"
    reaching = {f["capability"] for f in service.degraded_capabilities()}
    assert reaching == {"chat.reply"}, reaching
    assert "timeout" in RETRY_OWNED_FAILURE_CLASSES


def test_the_failure_class_survives_persistence(tmp_path):
    """It was not on the store's allow-list, so it was silently dropped -- twice now.

    Without it the fact reads "failed twice" and the retry-owned classes cannot be
    filtered at all, because the filter has nothing to filter on.
    """
    from leapflow.storage.capability_observation_store import _OBSERVATION_FIELDS

    assert "failure_class" in _OBSERVATION_FIELDS

    _, service = _degraded_service(tmp_path)
    _observe(service, "chat.reply", "chat_v1", "affordance_removed")
    fact = service.degraded_capabilities()[0]
    assert fact["failure_class"] == "affordance_removed"


def test_one_environment_change_is_recognisable_as_one_cause(tmp_path):
    """N capabilities bound to the same removed affordance are one change, not N.

    Without the fingerprint the teacher answers N times and can propose N rebuilds
    where the truth is one root cause and usually one rebind.
    """
    from leapflow.world_model.trajectory_grader import _degraded_capability_section

    _, service = _degraded_service(tmp_path)
    environment = _env()
    _observe(service, "chat.reply", "chat_v1", "affordance_removed", environment)
    _observe(service, "mail.send", "mail_v1", "affordance_removed", environment)

    facts = service.degraded_capabilities()
    assert len({f["environment"].get("fingerprint_id") for f in facts}) == 1

    section = _degraded_capability_section(facts)
    assert "one change rather than several" in section
    assert "affordance_removed" in section, "the class must be visible to the teacher"


def test_governance_reports_the_failure_class_it_was_given():
    """The streak alone cannot say what kind of failure it was."""
    import asyncio

    seen: list[dict[str, Any]] = []
    governor = _governor(streak=2, sink=lambda **kw: seen.append(kw))
    asyncio.run(
        governor.record_outcome(
            proposal_id="p1",
            plugin_id="chat_reply_v1",
            tool_name="chat_reply_v1",
            ok=False,
            failure_class="affordance_removed",
        )
    )
    assert seen[0]["failure_class"] == "affordance_removed"


# ── the fact the rebind/acquire choice is defined by ──────────────────────────


def test_the_absence_of_an_alternative_is_stated_not_omitted():
    """"No alternative exists" is the positive evidence for acquire.

    The action space defines ``rebind`` as "another installed capability already covers
    this" and ``acquire`` as "nothing does" -- and the teacher was shown neither. It saw a
    flat list of global capability names and had to guess. Measured on a real model:
    ``rebind`` on 3 of 3 trials of a unit whose candidate set had exactly one entry.

    Omitting the line would be worse than saying nothing, because silence reads as "not
    checked" rather than "checked and there are none".
    """
    from leapflow.world_model.trajectory_grader import _degraded_capability_section

    section = _degraded_capability_section(
        [{"capability": "chat.reply", "plugin_id": "chat_reply_v1",
          "failure_streak": 3, "alternatives": ()}]
    )
    assert "no other installed provider offers this capability" in section


def test_a_usable_alternative_is_named_and_an_unusable_one_is_qualified():
    from leapflow.world_model.trajectory_grader import _degraded_capability_section

    section = _degraded_capability_section(
        [{
            "capability": "chat.reply",
            "plugin_id": "chat_reply_v1",
            "failure_streak": 3,
            "alternatives": (
                {"plugin_id": "a", "tool_name": "chat_reply_v2", "fits_here": True,
                 "requires": ("app.chat.v2",)},
                {"plugin_id": "b", "tool_name": "chat_reply_v0", "fits_here": False,
                 "requires": ("app.chat.v0",)},
            ),
        }]
    )
    assert "chat_reply_v2" in section
    assert "chat_reply_v0 (needs app.chat.v0)" in section
    assert "one of these could take over" in section


def test_alternatives_that_all_misfit_say_so():
    """Existing but unusable is not the same as absent, and warrants a different action."""
    from leapflow.world_model.trajectory_grader import _degraded_capability_section

    section = _degraded_capability_section(
        [{
            "capability": "chat.reply", "plugin_id": "v1", "failure_streak": 3,
            "alternatives": (
                {"plugin_id": "b", "tool_name": "chat_reply_v0", "fits_here": False,
                 "requires": ("app.chat.v0",)},
            ),
        }]
    )
    assert "none of these can run in this environment" in section


def test_one_degradation_is_not_told_it_might_be_several():
    """The shared-environment hint asks the teacher to reconcile a single fact."""
    from leapflow.world_model.trajectory_grader import _degraded_capability_section

    one = _degraded_capability_section(
        [{"capability": "chat.reply", "plugin_id": "v1", "failure_streak": 2,
          "environment": {"fingerprint_id": "fp"}}]
    )
    assert "one change rather than several" not in one

    two = _degraded_capability_section(
        [
            {"capability": "chat.reply", "plugin_id": "v1", "failure_streak": 2,
             "environment": {"fingerprint_id": "fp"}},
            {"capability": "mail.send", "plugin_id": "v2", "failure_streak": 2,
             "environment": {"fingerprint_id": "fp"}},
        ]
    )
    assert "one change rather than several" in two


def test_an_unknown_environment_does_not_mark_every_alternative_unusable():
    """Undescribed must read as "cannot judge", not "nothing is available".

    The other reading would mark every alternative a misfit and push every verdict toward
    acquire -- the most expensive branch -- for the sole reason that the environment could
    not be described.
    """
    from leapflow.learning.degradation_sink import build_alternatives_provider
    from leapflow.plugins.protocol import ToolMetadata

    tool = ToolMetadata(
        name="chat_reply_v2",
        description="reply",
        parameters_schema={"type": "object", "properties": {}},
        handler=lambda **kwargs: None,
        x_leapflow={"category": "chat", "risk_level": "read_only"},
        provides_capabilities=("chat.reply",),
        requires_environment_affordances=("app.chat.v2",),
    )
    registry = SimpleNamespace(
        plugins={"v2": SimpleNamespace(tools=[tool])},
        tool_owners={"chat_reply_v2": "v2"},
        tool_handlers={"chat_reply_v2": tool.handler},
    )

    unknown = build_alternatives_provider(
        registry_provider=lambda: registry, affordances_provider=lambda: ()
    )("chat.reply", "v1")
    assert unknown and unknown[0]["fits_here"] is True

    absent = build_alternatives_provider(
        registry_provider=lambda: registry, affordances_provider=lambda: ("app.chat.v1",)
    )("chat.reply", "v1")
    assert absent and absent[0]["fits_here"] is False


def test_the_incumbent_is_not_offered_as_its_own_alternative():
    """Rebinding to the thing that is failing is not an option."""
    from leapflow.learning.degradation_sink import build_alternatives_provider
    from leapflow.plugins.protocol import ToolMetadata

    tool = ToolMetadata(
        name="chat_reply_v1",
        description="reply",
        parameters_schema={"type": "object", "properties": {}},
        handler=lambda **kwargs: None,
        x_leapflow={"category": "chat", "risk_level": "read_only"},
        provides_capabilities=("chat.reply",),
    )
    registry = SimpleNamespace(
        plugins={"v1": SimpleNamespace(tools=[tool])},
        tool_owners={"chat_reply_v1": "v1"},
        tool_handlers={"chat_reply_v1": tool.handler},
    )
    provider = build_alternatives_provider(registry_provider=lambda: registry)

    assert provider("chat.reply", "v1") == ()
    assert len(provider("chat.reply", "")) == 1
