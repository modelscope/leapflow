# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for adaptive lifecycle governance."""

from __future__ import annotations

import pytest

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.learning.plugin_trust import PluginTrustLedger, PluginTrustLevel
from leapflow.plugins.lifecycle_governor import LifecycleGovernor
from leapflow.storage.capability_proposal_queue import JsonCapabilityProposalQueue
from leapflow.storage.plugin_outcome_store import JsonPluginOutcomeStore


class _Actor:
    def __init__(self) -> None:
        self.disabled: list[str] = []

    async def disable(self, *, plugin_id: str):
        self.disabled.append(plugin_id)
        return {"ok": True, "action": "disable", "plugin_id": plugin_id}


def _proposal(queue: JsonCapabilityProposalQueue):
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
    return queue.update(item.proposal_id, status="INSTALLED")


@pytest.mark.asyncio
async def test_lifecycle_governor_promotes_verified_after_successes(tmp_path) -> None:
    queue = JsonCapabilityProposalQueue(tmp_path / "proposals.json")
    proposal = _proposal(queue)
    governor = LifecycleGovernor(
        proposal_queue=queue,
        outcome_store=JsonPluginOutcomeStore(tmp_path / "outcomes.json"),
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
    queue = JsonCapabilityProposalQueue(tmp_path / "proposals.json")
    proposal = _proposal(queue)
    actor = _Actor()
    governor = LifecycleGovernor(
        proposal_queue=queue,
        outcome_store=JsonPluginOutcomeStore(tmp_path / "outcomes.json"),
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
