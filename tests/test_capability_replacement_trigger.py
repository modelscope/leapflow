# Copyright (c) Alibaba, Inc. and its affiliates.
"""T1-T3: the trigger for replacing an existing capability's implementation.

Before this, self-evolution only ever fired on a *missing* capability. An existing
provider that kept failing was handled by quarantine -- which disables it, creating a
gap, which then triggers generation. That ordering has three consequences the EVO-02
episode measured: an availability hole between disable and install, never any two
providers admissible at once, and the question "is the replacement actually better?"
never being asked, because the incumbent is gone by the time the rival arrives.

These tests cover the pieces that make the other ordering possible:

* **T1** governance reports a failure that left the plugin *in service* -- the state
  between healthy and quarantined, which had no expression at all.
* **T2** the teacher is shown those facts and adjudicates. A failure count cannot tell
  a wrong implementation from a moved environment; both produce the same streak and
  want opposite actions.
* **T3** an admitted intent becomes a proposal whose identity lets a rival coexist with
  the incumbent it competes against. The end-to-end queueing now runs in the durable
  teacher worker (see ``test_durable_teacher.py``); here we hold the proposal-identity
  contract that makes coexistence possible.
"""

from __future__ import annotations

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


# ── T3: rival identity: the collision that would stop competition ─────────────


def _intent(capability: str = "chat.reply") -> EvolutionIntent:
    return EvolutionIntent.create(
        capability=capability,
        hypothesis="the current implementation keeps failing",
        confidence=0.8,
        expected_effect="the reply reaches the thread",
    )


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
