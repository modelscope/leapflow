# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for profile-scoped plugin proposal persistence."""
from __future__ import annotations

from leapflow.domain.plugin_proposal import BehaviorTestCase, GapEvidence, PluginProposal, ProposedToolSpec
from leapflow.storage.plugin_proposal_store import JsonPluginProposalStore


def test_json_plugin_proposal_store_round_trip(tmp_path) -> None:
    store = JsonPluginProposalStore(tmp_path / "proposals.json")
    proposal = PluginProposal.create(
        plugin_id="json_tools",
        capability_summary="Validate JSON",
        evidence=(GapEvidence.create("explicit", "Need JSON validation", confidence=0.8),),
        proposed_tools=(ProposedToolSpec(name="json_validate", description="Validate JSON"),),
        test_cases=(BehaviorTestCase.create("json_validate", arguments={"text": "{}"}, expected_subset={"ok": True}),),
    )

    store.save(proposal)
    loaded = store.get(proposal.proposal_id)

    assert loaded == proposal
    assert store.path.exists()
    assert store.list() == [proposal]
    assert loaded.test_cases[0].tool_name == "json_validate"


def test_json_plugin_proposal_store_update_status(tmp_path) -> None:
    store = JsonPluginProposalStore(tmp_path / "proposals.json")
    proposal = store.save(
        PluginProposal.create(plugin_id="p", capability_summary="capability")
    )

    updated = store.update_status(proposal.proposal_id, "approved")

    assert updated is not None
    assert updated.status == "approved"
    assert store.get(proposal.proposal_id).status == "approved"
