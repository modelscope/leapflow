# Copyright (c) Alibaba, Inc. and its affiliates.
"""P2(a): the evolution governance tier becomes reachable.

`AdaptiveEvolutionPolicy` and `LifecycleGovernor` implement trust, probation and
quarantine. These tests guard the production wiring after proposal and outcome
persistence moved onto the append-only evolution event stream; no profile-local
JSON queue may be required for the governance cycle.

These tests cover the two halves of the fix:

* **Protocols** -- the policy and governor now state what they require
  (`EvolutionProposalView`, `EvolutionLifecycleStore`, `OutcomeStore`) instead of
  binding to one concrete store, so they can be driven by the live chain.
* **A production filler** -- `plugin_propose` now opens a correlated lifecycle
  record, giving the governor something real to govern.

Also guards the distinction that must not be collapsed: the review vocabulary
(`draft | review | approved | rejected`) and the acquisition-lifecycle vocabulary
(`PENDING ... QUARANTINED`) are different concerns in different stores.
"""

from __future__ import annotations

import asyncio

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.learning.plugin_trust import PluginTrustLedger, PluginTrustLevel
from leapflow.plugins.adaptive_policy import AdaptiveEvolutionPolicy
from leapflow.plugins.evolution_contracts import (
    EvolutionLifecycleStore,
    EvolutionProposalView,
    OutcomeStore,
)
from leapflow.plugins.lifecycle_governor import LifecycleGovernor
from leapflow.storage.capability_proposal_queue import (
    CapabilityProposalItem,
    EvolutionCapabilityProposalStore,
)
from leapflow.storage.evolution_event_store import DuckDBEvolutionEventStore
from leapflow.storage.plugin_outcome_store import EvolutionPluginOutcomeStore


def _queue(tmp_path) -> EvolutionCapabilityProposalStore:
    events = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    return EvolutionCapabilityProposalStore(events, profile_id="profile-1")


def _requirement() -> CapabilityRequirement:
    return CapabilityRequirement.create(
        "chat.reply", "explicit_request", max_risk_level="read_only", requirement_id="r1"
    )


# ── the contracts are satisfied by the shipped stores ─────────────────────────


def test_shipped_types_satisfy_the_new_protocols(tmp_path):
    item = CapabilityProposalItem(proposal_id="p1", status="PENDING", requirements=())
    assert isinstance(item, EvolutionProposalView)
    queue = _queue(tmp_path)
    assert isinstance(queue, EvolutionLifecycleStore)
    assert isinstance(
        EvolutionPluginOutcomeStore(queue._event_store, profile_id="profile-1"),
        OutcomeStore,
    )


def test_policy_accepts_any_conforming_view():
    """The policy no longer requires the concrete queue item."""

    class _View:
        proposal_id = "p9"
        status = "PENDING"
        requirements: tuple = ()
        risk = {"risk_level": "read_only"}

    view = _View()
    assert isinstance(view, EvolutionProposalView)
    decision = AdaptiveEvolutionPolicy(autonomy_level="approve_to_install").decide(view)
    assert decision.action == "generate"


def test_governor_accepts_any_conforming_stores(tmp_path):
    """The governor can be driven by a non-default backing."""
    transitions: list[tuple] = []

    class _Store:
        def get(self, proposal_id):
            return None

        def update(self, proposal_id, **kwargs):
            transitions.append((proposal_id, kwargs.get("status")))
            return None

    class _Outcomes:
        def __init__(self):
            self.streak = 0

        def add_outcome(self, **kwargs):
            self.streak = 0 if kwargs.get("ok") else self.streak + 1
            return dict(kwargs)

        def failure_streak(self, plugin_id):
            return self.streak

    store, outcomes = _Store(), _Outcomes()
    assert isinstance(store, EvolutionLifecycleStore)
    assert isinstance(outcomes, OutcomeStore)
    governor = LifecycleGovernor(proposal_queue=store, outcome_store=outcomes)
    result = asyncio.run(governor.record_outcome(
        proposal_id="p1", plugin_id="pl1", tool_name="t1", ok=True
    ))
    assert result.action == "probation_execute"
    assert transitions == [("p1", "PROBATION")]


# ── the two vocabularies stay separate ────────────────────────────────────────


def test_review_and_lifecycle_vocabularies_are_distinct():
    """Collapsing these into one field would lose a whole dimension of state."""
    from leapflow.domain.plugin_proposal import ProposalStatus as ReviewStatus
    from leapflow.storage.capability_proposal_queue import ProposalStatus as LifecycleStatus
    from typing import get_args

    review = set(get_args(ReviewStatus))
    lifecycle = set(get_args(LifecycleStatus))
    assert review == {"draft", "review", "approved", "rejected"}
    assert {"PENDING", "INSTALLED", "PROBATION", "QUARANTINED"} <= lifecycle
    # They intentionally share no member: different concerns, different stores.
    assert review & lifecycle == set()


# ── the governance tier now runs end to end on the shipped stores ─────────────


def test_full_governance_cycle_on_the_shipped_stores(tmp_path):
    """PENDING -> generate decision -> outcomes -> quarantine, all real components."""
    queue = _queue(tmp_path)
    outcomes = EvolutionPluginOutcomeStore(
        queue._event_store, profile_id="profile-1"
    )
    trust = PluginTrustLedger()
    item = queue.enqueue(
        requirements=[_requirement()],
        risk={"risk_level": "read_only"},
        source="plugin_propose",
        metadata={"plugin_id": "chat_reply_plugin"},
    )
    assert item.status == "PENDING"

    # The policy can now decide on a record that a production caller created.
    decision = AdaptiveEvolutionPolicy(autonomy_level="approve_to_install").decide(
        item, trust_level=PluginTrustLevel.DRAFT
    )
    assert decision.action == "generate"
    queue.transition(item.proposal_id, "GENERATED", generated_code_ref="sha256:test")
    queue.transition(
        item.proposal_id,
        "APPROVED",
        proposal_approval_id="approval-content",
        mutation_approval_id="approval-mutation",
    )
    queue.transition(item.proposal_id, "INSTALLED", install_result={"ok": True})

    # The governor transitions that same installed record from execution outcomes.
    disabled: list[str] = []

    class _Actor:
        async def disable(self, *, plugin_id):
            disabled.append(plugin_id)
            return {"ok": True}

    governor = LifecycleGovernor(
        proposal_queue=queue, outcome_store=outcomes,
        lifecycle_actor=_Actor(), trust_ledger=trust, quarantine_after=3,
    )
    for _ in range(2):
        result = asyncio.run(governor.record_outcome(
            proposal_id=item.proposal_id, plugin_id="chat_reply_plugin",
            tool_name="chat_reply", ok=False,
        ))
        assert result.action == "probation_execute"
    final = asyncio.run(governor.record_outcome(
        proposal_id=item.proposal_id, plugin_id="chat_reply_plugin",
        tool_name="chat_reply", ok=False,
    ))
    assert final.action == "quarantine"
    assert final.failure_streak == 3
    assert disabled == ["chat_reply_plugin"]
    assert queue.get(item.proposal_id).status == "QUARANTINED"


def test_requarantined_record_remains_terminal_and_safety_memory_survives(tmp_path):
    """Re-proposing the same identity cannot erase a quarantine fact.

    ``proposal_id`` is a content hash of stable requirement identity. The event-sourced
    store returns the terminal record instead of resetting it to ``PENDING``; a genuinely
    new attempt must carry a new requirement identity and therefore preserves audit history.
    """
    queue = _queue(tmp_path)
    first = queue.enqueue(requirements=[_requirement()], source="plugin_propose")
    queue.transition(first.proposal_id, "GENERATED", generated_code_ref="sha256:test")
    queue.transition(
        first.proposal_id,
        "APPROVED",
        proposal_approval_id="approval-content",
        mutation_approval_id="approval-mutation",
    )
    queue.transition(first.proposal_id, "INSTALLED", install_result={"ok": True})
    queue.transition(first.proposal_id, "QUARANTINED")
    second = queue.enqueue(requirements=[_requirement()], source="plugin_propose")

    assert second.proposal_id == first.proposal_id      # content-addressed
    assert second.status == "QUARANTINED"                # terminal fact is preserved
    assert len(queue.list_items(limit=0)) == 1

    # Independent safety evidence remains queryable with the terminal proposal.
    outcomes = EvolutionPluginOutcomeStore(
        queue._event_store, profile_id="profile-1"
    )
    for _ in range(3):
        outcomes.add_outcome(plugin_id="p", tool_name="t", ok=False)
    assert outcomes.failure_streak("p") == 3

    trust = PluginTrustLedger()
    trust.record_failure("p", hard=True)
    assert PluginTrustLedger.load_state(trust.to_state()).is_frozen("p") is True


# ── plugin_propose now opens a lifecycle record (the production filler) ───────


def _plugin_with_stores(tmp_path):
    from leapflow.plugins.tool_plugins.self_management import SelfManagementPlugin

    plugin = SelfManagementPlugin()
    queue = _queue(tmp_path)
    plugin.bind_runtime(capability_lifecycle_store=queue)
    return plugin, queue


def _propose(plugin, **kwargs):
    handler = {t.name: t.handler for t in plugin.tools}["plugin_propose"]
    return asyncio.run(handler(**kwargs))


def test_plugin_propose_opens_a_correlated_lifecycle_record(tmp_path):
    plugin, queue = _plugin_with_stores(tmp_path)
    result = _propose(
        plugin,
        requested_capability="chat.reply",
        plugin_id="chat_reply_plugin",
        risk_level="read_only",
    )
    assert result["ok"] is True
    lifecycle_id = result["lifecycle_proposal_id"]
    assert lifecycle_id

    record = queue.get(lifecycle_id)
    assert record is not None
    assert record.status == "PENDING"                       # ready for the policy
    meta = dict(record.metadata)
    assert meta["plugin_id"] == result["proposal"]["plugin_id"]
    assert meta["review_proposal_id"] == result["proposal"]["proposal_id"]

    # The review proposal keeps its own, separate vocabulary.
    assert result["proposal"]["status"] == "draft"

    # ...and the policy can act on the record the tool just created.
    decision = AdaptiveEvolutionPolicy(autonomy_level="approve_to_install").decide(record)
    assert decision.action == "generate"


def test_propose_fails_closed_when_the_lifecycle_ledger_fails(tmp_path):
    """A proposal without its sole durable lifecycle record must not escape."""
    from leapflow.plugins.tool_plugins.self_management import SelfManagementPlugin

    class _Broken:
        def enqueue(self, **kwargs):
            raise OSError("disk on fire")

    plugin = SelfManagementPlugin()
    plugin.bind_runtime(capability_lifecycle_store=_Broken())
    result = _propose(plugin, requested_capability="chat.reply", risk_level="read_only")
    assert result["ok"] is False
    assert "persistence failed" in result["error"].lower()


def test_lifecycle_record_carries_the_declared_risk_ceiling(tmp_path):
    plugin, queue = _plugin_with_stores(tmp_path)
    result = _propose(
        plugin, requested_capability="chat.send", risk_level="medium",
    )
    record = queue.get(result["lifecycle_proposal_id"])
    assert dict(record.risk)["risk_level"] == "medium"
    assert dict(record.requirements[0])["max_risk_level"] == "medium"


# ── plugin_generate resolves canonical ids and review aliases (G3) ────────────


def test_generate_resolves_a_world_model_lifecycle_proposal(tmp_path):
    """Both a canonical lifecycle id and its review alias resolve from one store."""
    from leapflow.domain.capability_requirement import CapabilityRequirement

    plugin, queue = _plugin_with_stores(tmp_path)
    requirement = CapabilityRequirement.create(
        "chat.reply", "world_model", evidence="the send path silently no-ops",
        max_risk_level="read_only", requirement_id="req-wm-chat.reply",
    )
    item = queue.enqueue(
        requirements=(requirement,),
        source="world_model",
        observation_ids=("obs-1",),
        metadata={"plugin_id": "chat_reply_alt_plugin", "capability_summary": "reply via v2"},
    )

    source, plugin_id, description, provides = plugin._resolve_generation_source(
        item.proposal_id
    )
    assert source == "lifecycle"
    assert plugin_id == "chat_reply_alt_plugin"
    assert provides == ("chat.reply",)          # capability preserved for generation
    assert description                          # non-empty description derived

    # A review alias resolves through metadata on the same lifecycle record.
    review = _propose(plugin, requested_capability="chat.send", risk_level="read_only")
    assert plugin._resolve_generation_source(
        review["proposal"]["proposal_id"]
    )[0] == "review"

    # An unknown id resolves to nothing, so generation returns not-found.
    assert plugin._resolve_generation_source("prop-does-not-exist")[0] == ""
