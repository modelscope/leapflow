# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for adaptive proposal queue and policy decisions."""

from __future__ import annotations

import pytest

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.evolution.projection import EvolutionProjectionRunner
from leapflow.learning.plugin_trust import PluginTrustLevel
from leapflow.plugins.adaptive_policy import AdaptiveEvolutionPolicy
from leapflow.storage.capability_proposal_queue import EvolutionCapabilityProposalStore
from leapflow.storage.evolution_event_store import DuckDBEvolutionEventStore


def _req(risk: str = "external") -> CapabilityRequirement:
    return CapabilityRequirement.create(
        "json.pretty",
        "unknown_tool",
        max_risk_level=risk,  # type: ignore[arg-type]
        requirement_id="req-json-pretty",
    )


def test_proposal_queue_enqueues_and_updates_status(tmp_path) -> None:
    queue = EvolutionCapabilityProposalStore(
        DuckDBEvolutionEventStore(tmp_path / "events.duckdb"), profile_id="profile-1"
    )

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


def test_prepare_enqueue_does_not_publish_before_atomic_owner_commit(tmp_path) -> None:
    events = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    queue = EvolutionCapabilityProposalStore(events, profile_id="profile-1")

    item, event = queue.prepare_enqueue(
        requirements=(_req("read_only"),),
        environment={"fingerprint_id": "env-a"},
        occurred_at=10.0,
    )

    assert event is not None
    assert queue.get(item.proposal_id) is None
    events.append(event)
    assert queue.get(item.proposal_id) == item
    events.close()


@pytest.mark.asyncio
async def test_event_sourced_proposal_store_replays_the_latest_lifecycle_state(tmp_path) -> None:
    events = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    queue = EvolutionCapabilityProposalStore(events, profile_id="profile-1")
    item = queue.enqueue(
        requirements=(_req("read_only"),),
        environment={"fingerprint_id": "env-a", "session_id": "session-a"},
        metadata={"plugin_id": "json_pretty_plugin"},
    )
    queue.transition(item.proposal_id, "GENERATED", generated_code_ref="sha256:test")
    queue.transition(
        item.proposal_id,
        "APPROVED",
        proposal_approval_id="approval-content",
        mutation_approval_id="approval-mutation",
    )

    replayed = EvolutionCapabilityProposalStore(events, profile_id="profile-1")
    restored = replayed.get(item.proposal_id)

    assert restored is not None and restored.status == "APPROVED"
    assert restored.generated_code_ref == "sha256:test"
    assert replayed.active(limit=0) == [restored]
    assert len(events.read(profile_id="profile-1", proposal_id=item.proposal_id)) == 3
    projection = await EvolutionProjectionRunner(events).project_aggregate(
        profile_id="profile-1"
    )
    assert projection["proposals"][0]["status"] == "APPROVED"
    assert projection["mutation_matrix"][0]["lifecycle_status"] == "APPROVED"
    events.close()


def test_adaptive_policy_requires_approval_for_generated_high_risk(tmp_path) -> None:
    queue = EvolutionCapabilityProposalStore(
        DuckDBEvolutionEventStore(tmp_path / "events.duckdb"), profile_id="profile-1"
    )
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
