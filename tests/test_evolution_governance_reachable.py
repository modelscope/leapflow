"""P2(a): the evolution governance tier becomes reachable.

`AdaptiveEvolutionPolicy` and `LifecycleGovernor` implement trust, probation and
quarantine. Everything they need was built and path-declared -- and orphaned:
`JsonCapabilityProposalQueue`, `JsonPluginOutcomeStore`, and both
`ProfileLayout` paths had no references outside their own modules, so nothing in
production could ever reach the machinery.

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
    JsonCapabilityProposalQueue,
)
from leapflow.storage.plugin_outcome_store import JsonPluginOutcomeStore


def _queue(tmp_path) -> JsonCapabilityProposalQueue:
    return JsonCapabilityProposalQueue(tmp_path / "lifecycle.json")


def _requirement() -> CapabilityRequirement:
    return CapabilityRequirement.create(
        "chat.reply", "explicit_request", max_risk_level="read_only", requirement_id="r1"
    )


# ── the contracts are satisfied by the shipped stores ─────────────────────────


def test_shipped_types_satisfy_the_new_protocols(tmp_path):
    item = CapabilityProposalItem(proposal_id="p1", status="PENDING", requirements=())
    assert isinstance(item, EvolutionProposalView)
    assert isinstance(_queue(tmp_path), EvolutionLifecycleStore)
    assert isinstance(JsonPluginOutcomeStore(tmp_path / "o.json"), OutcomeStore)


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
    outcomes = JsonPluginOutcomeStore(tmp_path / "outcomes.json")
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

    # The governor transitions that same record from execution outcomes.
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


def test_requarantined_record_is_reset_in_place_but_memory_survives_elsewhere(tmp_path):
    """Re-proposing after quarantine resets the record; the safety memory does not live there.

    ``proposal_id`` is a content hash of the requirement payload, so re-proposing
    the same capability reuses the id and resets it to ``PENDING`` -- the
    quarantine history is *overwritten* at the proposal layer rather than a second
    record being created. That is safe only because the trigger the governor
    actually consults lives elsewhere: the outcome store's ``failure_streak`` and
    the trust ledger's freeze both persist independently, so a re-proposed plugin
    does not get a clean slate where it matters.
    """
    queue = _queue(tmp_path)
    first = queue.enqueue(requirements=[_requirement()], source="plugin_propose")
    queue.update(first.proposal_id, status="QUARANTINED")
    second = queue.enqueue(requirements=[_requirement()], source="plugin_propose")

    assert second.proposal_id == first.proposal_id      # content-addressed
    assert second.status == "PENDING"                    # reset, ready to retry
    assert len(queue.list_items(limit=0)) == 1           # replaced, not appended

    # The memory that gates a retry survives the reset.
    outcomes = JsonPluginOutcomeStore(tmp_path / "outcomes.json")
    for _ in range(3):
        outcomes.add_outcome(plugin_id="p", tool_name="t", ok=False)
    assert outcomes.failure_streak("p") == 3

    trust = PluginTrustLedger()
    trust.record_failure("p", hard=True)
    assert PluginTrustLedger.load_state(trust.to_state()).is_frozen("p") is True


# ── plugin_propose now opens a lifecycle record (the production filler) ───────


def _plugin_with_stores(tmp_path):
    from leapflow.plugins.tool_plugins.self_management import SelfManagementPlugin
    from leapflow.storage.plugin_proposal_store import JsonPluginProposalStore

    plugin = SelfManagementPlugin()
    queue = _queue(tmp_path)
    plugin.bind_runtime(
        plugin_proposal_store=JsonPluginProposalStore(tmp_path / "review.json"),
        capability_lifecycle_store=queue,
    )
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


def test_propose_still_succeeds_when_the_lifecycle_ledger_fails(tmp_path):
    """Bookkeeping must never fail the proposal the caller asked for."""
    from leapflow.plugins.tool_plugins.self_management import SelfManagementPlugin
    from leapflow.storage.plugin_proposal_store import JsonPluginProposalStore

    class _Broken:
        def enqueue(self, **kwargs):
            raise OSError("disk on fire")

    plugin = SelfManagementPlugin()
    plugin.bind_runtime(
        plugin_proposal_store=JsonPluginProposalStore(tmp_path / "review.json"),
        capability_lifecycle_store=_Broken(),
    )
    result = _propose(plugin, requested_capability="chat.reply", risk_level="read_only")
    assert result["ok"] is True                 # the proposal survived
    assert result["lifecycle_proposal_id"] == ""  # ...and the failure is visible


def test_lifecycle_record_carries_the_declared_risk_ceiling(tmp_path):
    plugin, queue = _plugin_with_stores(tmp_path)
    result = _propose(
        plugin, requested_capability="chat.send", risk_level="medium",
    )
    record = queue.get(result["lifecycle_proposal_id"])
    assert dict(record.risk)["risk_level"] == "medium"
    assert dict(record.requirements[0])["max_risk_level"] == "medium"
