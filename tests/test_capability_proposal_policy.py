# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for adaptive proposal queue and policy decisions."""

from __future__ import annotations

import pytest

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.learning.plugin_trust import PluginTrustLevel
from leapflow.plugins.adaptive_loop import AdaptivePluginLoop
from leapflow.plugins.adaptive_policy import AdaptiveEvolutionPolicy
from leapflow.storage.capability_plan_store import JsonCapabilityPlanStore
from leapflow.storage.capability_proposal_queue import JsonCapabilityProposalQueue


def _req(risk: str = "external") -> CapabilityRequirement:
    return CapabilityRequirement.create(
        "json.pretty",
        "unknown_tool",
        max_risk_level=risk,  # type: ignore[arg-type]
        requirement_id="req-json-pretty",
    )


def test_proposal_queue_enqueues_and_updates_status(tmp_path) -> None:
    queue = JsonCapabilityProposalQueue(tmp_path / "proposals.json")

    item = queue.enqueue(
        requirements=(_req("read_only"),),
        environment={"fingerprint_id": "env-a"},
        observation_ids=("obs-1",),
        metadata={"plugin_id": "json_pretty_plugin"},
    )
    duplicate = queue.enqueue(
        requirements=(_req("read_only"),),
        environment={"fingerprint_id": "env-a"},
        observation_ids=("obs-1",),
    )
    updated = queue.update(item.proposal_id, status="GENERATED", generated_code_ref="code.py")

    assert duplicate.proposal_id == item.proposal_id
    assert updated is not None
    assert updated.status == "GENERATED"
    assert updated.generated_code_ref == "code.py"
    assert queue.active()[0].proposal_id == item.proposal_id


def test_adaptive_policy_requires_approval_for_generated_high_risk(tmp_path) -> None:
    queue = JsonCapabilityProposalQueue(tmp_path / "proposals.json")
    proposal = queue.enqueue(
        requirements=(_req("external"),),
        risk={"risk_level": "external"},
    )
    proposal = queue.update(proposal.proposal_id, status="GENERATED")

    decision = AdaptiveEvolutionPolicy(autonomy_level="trusted_autonomous").decide(
        proposal,
        trust_level=PluginTrustLevel.DRAFT,
        sandbox_validated=True,
    )

    assert decision.action == "request_approval"
    assert decision.requires_approval is True


@pytest.mark.asyncio
async def test_loop_applies_policy_install_through_actor(tmp_path) -> None:
    class Actor:
        async def install(self, **kwargs):
            return {"ok": True, "plugin_id": kwargs["plugin_id"], "action": "install"}

        async def disable(self, **kwargs):
            return {"ok": True}

        async def remove(self, **kwargs):
            return {"ok": True}

    queue = JsonCapabilityProposalQueue(tmp_path / "proposals.json")
    proposal = queue.enqueue(
        requirements=(_req("read_only"),),
        risk={"risk_level": "read_only"},
        metadata={"plugin_id": "json_pretty_plugin"},
    )
    proposal = queue.update(proposal.proposal_id, status="GENERATED")
    decision = AdaptiveEvolutionPolicy(autonomy_level="trusted_autonomous").decide(
        proposal,
        sandbox_validated=True,
    )
    loop = AdaptivePluginLoop(
        registry=object(),
        plan_store=JsonCapabilityPlanStore(tmp_path / "plans.json"),
        lifecycle_actor=Actor(),
    )

    result = await loop.apply_policy_decision(
        proposal,
        decision,
        proposal_queue=queue,
        generated_code="# code",
    )

    assert result["ok"] is True
    assert queue.get(proposal.proposal_id).status == "INSTALLED"
