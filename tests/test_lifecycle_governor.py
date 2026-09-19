# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for adaptive lifecycle governance."""

from __future__ import annotations

import pytest

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.learning.plugin_trust import PluginTrustLedger, PluginTrustLevel
from leapflow.plugins.lifecycle_governor import LifecycleGovernor
from leapflow.storage.capability_proposal_queue import EvolutionCapabilityProposalStore
from leapflow.storage.evolution_event_store import DuckDBEvolutionEventStore
from leapflow.storage.plugin_outcome_store import EvolutionPluginOutcomeStore


class _Actor:
    def __init__(self) -> None:
        self.disabled: list[str] = []

    async def disable(self, *, plugin_id: str):
        self.disabled.append(plugin_id)
        return {"ok": True, "action": "disable", "plugin_id": plugin_id}


def _proposal(queue: EvolutionCapabilityProposalStore):
    requirement = CapabilityRequirement.create(
        "json.pretty",
        "explicit_request",
        max_risk_level="read_only",
        requirement_id="req-json-pretty",
    )
    item = queue.enqueue(
        requirements=(requirement,),
        risk={"risk_level": "read_only"},
        metadata={"plugin_id": "json_pretty_plugin"},
    )
    queue.transition(item.proposal_id, "GENERATED", generated_code_ref="sha256:test")
    queue.transition(
        item.proposal_id,
        "APPROVED",
        proposal_approval_id="approval-content",
        mutation_approval_id="approval-mutation",
    )
    return queue.transition(item.proposal_id, "INSTALLED", install_result={"ok": True})


@pytest.mark.asyncio
async def test_lifecycle_governor_promotes_verified_after_successes(tmp_path) -> None:
    events = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    queue = EvolutionCapabilityProposalStore(events, profile_id="profile-1")
    proposal = _proposal(queue)
    governor = LifecycleGovernor(
        proposal_queue=queue,
        outcome_store=EvolutionPluginOutcomeStore(events, profile_id="profile-1"),
        trust_ledger=PluginTrustLedger(candidate_at=1, verified_at=2, production_at=3),
        verified_at=PluginTrustLevel.VERIFIED,
    )

    await governor.record_outcome(
        proposal_id=proposal.proposal_id,
        plugin_id="json_pretty_plugin",
        tool_name="json_pretty",
        ok=True,
    )
    result = await governor.record_outcome(
        proposal_id=proposal.proposal_id,
        plugin_id="json_pretty_plugin",
        tool_name="json_pretty",
        ok=True,
    )

    assert result.action == "verify"
    assert queue.get(proposal.proposal_id).status == "VERIFIED"


@pytest.mark.asyncio
async def test_lifecycle_governor_quarantines_after_failure_streak(tmp_path) -> None:
    events = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    queue = EvolutionCapabilityProposalStore(events, profile_id="profile-1")
    proposal = _proposal(queue)
    actor = _Actor()
    governor = LifecycleGovernor(
        proposal_queue=queue,
        outcome_store=EvolutionPluginOutcomeStore(events, profile_id="profile-1"),
        lifecycle_actor=actor,
        quarantine_after=2,
    )

    await governor.record_outcome(
        proposal_id=proposal.proposal_id,
        plugin_id="json_pretty_plugin",
        tool_name="json_pretty",
        ok=False,
    )
    result = await governor.record_outcome(
        proposal_id=proposal.proposal_id,
        plugin_id="json_pretty_plugin",
        tool_name="json_pretty",
        ok=False,
    )

    assert result.action == "quarantine"
    assert actor.disabled == ["json_pretty_plugin"]
    assert queue.get(proposal.proposal_id).status == "QUARANTINED"


@pytest.mark.asyncio
async def test_internal_defect_immediately_freezes_and_quarantines(tmp_path) -> None:
    events = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    queue = EvolutionCapabilityProposalStore(events, profile_id="profile-1")
    proposal = _proposal(queue)
    actor = _Actor()
    trust = PluginTrustLedger(candidate_at=1, verified_at=2, production_at=3)
    governor = LifecycleGovernor(
        proposal_queue=queue,
        outcome_store=EvolutionPluginOutcomeStore(events, profile_id="profile-1"),
        lifecycle_actor=actor,
        trust_ledger=trust,
        quarantine_after=99,
    )

    result = await governor.record_outcome(
        proposal_id=proposal.proposal_id,
        plugin_id="json_pretty_plugin",
        tool_name="json_pretty",
        ok=False,
        failure_class="internal_defect",
    )

    assert result.action == "quarantine"
    assert trust.is_frozen("json_pretty_plugin") is True
    item = queue.get(proposal.proposal_id)
    assert item.status == "QUARANTINED"
    assert item.trust_state["frozen"] is True
    assert item.metadata["terminal_reason"] == "internal_defect"
