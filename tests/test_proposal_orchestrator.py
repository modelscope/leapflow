# Copyright (c) Alibaba, Inc. and its affiliates.
"""Contracts for the durable proposal policy and double-approval boundary."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from leapflow.daemon.approval_coordinator import ApprovalCoordinator, _DaemonApprovalGate
from leapflow.daemon.approval_route import approval_route
from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.evolution.artifact_store import ContentAddressedArtifactStore
from leapflow.plugins.adaptive_policy import AdaptiveEvolutionPolicy
from leapflow.plugins.proposal_orchestrator import ProposalOrchestrator
from leapflow.security.approval import ApprovalDecision, SessionAwareGate
from leapflow.security.orchestrator import ApprovalOrchestrator
from leapflow.storage.capability_proposal_queue import EvolutionCapabilityProposalStore
from leapflow.storage.evolution_event_store import DuckDBEvolutionEventStore


class _Gate:
    def __init__(self, decisions: list[bool | BaseException]) -> None:
        self._decisions = list(decisions)
        self.actions = []

    async def evaluate(self, action):
        self.actions.append(action)
        approved = self._decisions.pop(0)
        if isinstance(approved, BaseException):
            raise approved
        return SimpleNamespace(
            approved=approved,
            action=action,
            reason="user_approved" if approved else "user_denied",
            denial_message="denied" if not approved else "",
        )


class _RequestGate:
    def __init__(self, decisions: list[ApprovalDecision]) -> None:
        self._decisions = list(decisions)
        self.requests = []

    async def request_approval(self, request):
        self.requests.append(request)
        return self._decisions.pop(0)


def _queue(tmp_path: Path) -> tuple[EvolutionCapabilityProposalStore, str]:
    events = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    queue = EvolutionCapabilityProposalStore(events, profile_id="profile-1")
    item = queue.enqueue(
        requirements=(
            CapabilityRequirement.create(
                "chat.reply",
                "world_model",
                requirement_id="req-chat-reply",
                max_risk_level="read_only",
            ),
        ),
        risk={"risk_level": "read_only"},
        metadata={"plugin_id": "chat_reply_plugin"},
    )
    return queue, item.proposal_id


@pytest.mark.asyncio
async def test_generated_proposal_requires_two_distinct_approvals(tmp_path: Path) -> None:
    queue, proposal_id = _queue(tmp_path)
    gate = _Gate([True, True])
    orchestrator = ProposalOrchestrator(
        queue=queue,
        artifact_store=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        approval_gate=gate,
        policy=AdaptiveEvolutionPolicy(autonomy_level="generate_only"),
    )

    generated = orchestrator.register_generated(
        proposal_id,
        "plugin = object()\n",
        validation={"ok": True, "stage": "passed", "compatibility_ok": True},
    )
    content = await orchestrator.approve_content(proposal_id)
    mutation = await orchestrator.authorize_mutation(proposal_id)
    installed = orchestrator.record_installed(proposal_id, {"ok": True})

    assert generated.status == "GENERATED"
    assert content.approved and mutation.approved
    assert content.approval_id != mutation.approval_id
    assert installed.status == "INSTALLED"
    assert installed.proposal_approval_id == content.approval_id
    assert installed.mutation_approval_id == mutation.approval_id
    assert [action.metadata["approval_stage"] for action in gate.actions] == [
        "proposal_content",
        "plugin_mutation",
    ]
    assert orchestrator.generated_code(proposal_id) == "plugin = object()\n"


def test_missing_compatibility_evidence_fails_before_cas_admission(tmp_path: Path) -> None:
    queue, proposal_id = _queue(tmp_path)
    orchestrator = ProposalOrchestrator(
        queue=queue,
        artifact_store=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        approval_gate=None,
        policy=AdaptiveEvolutionPolicy(autonomy_level="generate_only"),
    )

    failed = orchestrator.register_generated(
        proposal_id,
        "plugin = object()\n",
        validation={"ok": True},
    )

    assert failed.status == "FAILED"
    assert failed.generated_code_ref == ""
    assert failed.metadata["terminal_reason"] == "static or compatibility validation failed"


@pytest.mark.asyncio
async def test_content_denial_is_terminal_and_blocks_mutation(tmp_path: Path) -> None:
    queue, proposal_id = _queue(tmp_path)
    orchestrator = ProposalOrchestrator(
        queue=queue,
        artifact_store=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        approval_gate=_Gate([False]),
        policy=AdaptiveEvolutionPolicy(autonomy_level="generate_only"),
    )
    orchestrator.register_generated(
        proposal_id,
        "plugin = object()\n",
        validation={"ok": True, "compatibility_ok": True}
    )

    approval = await orchestrator.approve_content(proposal_id)

    assert approval.approved is False
    assert queue.get(proposal_id).status == "REJECTED"
    with pytest.raises(PermissionError):
        await orchestrator.authorize_mutation(proposal_id)


@pytest.mark.asyncio
async def test_missing_gate_fails_closed_with_terminal_rejection(tmp_path: Path) -> None:
    queue, proposal_id = _queue(tmp_path)
    orchestrator = ProposalOrchestrator(
        queue=queue,
        artifact_store=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        approval_gate=None,
        policy=AdaptiveEvolutionPolicy(autonomy_level="generate_only"),
    )
    orchestrator.register_generated(
        proposal_id,
        "plugin = object()\n",
        validation={"ok": True, "compatibility_ok": True}
    )

    approval = await orchestrator.approve_content(proposal_id)

    assert approval.approved is False
    rejected = queue.get(proposal_id)
    assert rejected.status == "REJECTED"
    assert rejected.metadata["terminal_reason"] == "approval_gate_missing"


@pytest.mark.asyncio
async def test_mutation_denial_is_terminal(tmp_path: Path) -> None:
    queue, proposal_id = _queue(tmp_path)
    orchestrator = ProposalOrchestrator(
        queue=queue,
        artifact_store=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        approval_gate=_Gate([True, False]),
        policy=AdaptiveEvolutionPolicy(autonomy_level="generate_only"),
    )
    orchestrator.register_generated(proposal_id, "plugin = object()\n", validation={"ok": True, "compatibility_ok": True})
    await orchestrator.approve_content(proposal_id)

    approval = await orchestrator.authorize_mutation(proposal_id)

    assert approval.approved is False
    rejected = queue.get(proposal_id)
    assert rejected.status == "REJECTED"
    assert rejected.mutation_approval_id
    assert rejected.metadata["terminal_reason"] == "user_denied"


@pytest.mark.asyncio
async def test_gate_exception_fails_closed_and_records_reason(tmp_path: Path) -> None:
    queue, proposal_id = _queue(tmp_path)
    orchestrator = ProposalOrchestrator(
        queue=queue,
        artifact_store=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        approval_gate=_Gate([RuntimeError("route unavailable")]),
        policy=AdaptiveEvolutionPolicy(autonomy_level="generate_only"),
    )
    orchestrator.register_generated(proposal_id, "plugin = object()\n", validation={"ok": True, "compatibility_ok": True})

    approval = await orchestrator.approve_content(proposal_id)

    assert approval.approved is False
    rejected = queue.get(proposal_id)
    assert rejected.status == "REJECTED"
    assert rejected.metadata["terminal_reason"] == "approval_gate_error:RuntimeError"


@pytest.mark.asyncio
async def test_production_approval_orchestrator_supports_both_stages(tmp_path: Path) -> None:
    queue, proposal_id = _queue(tmp_path)
    request_gate = _RequestGate([ApprovalDecision.ALLOW_ONCE, ApprovalDecision.ALLOW_ONCE])
    orchestrator = ProposalOrchestrator(
        queue=queue,
        artifact_store=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        approval_gate=ApprovalOrchestrator(request_gate),
        policy=AdaptiveEvolutionPolicy(autonomy_level="generate_only"),
    )
    orchestrator.register_generated(proposal_id, "plugin = object()\n", validation={"ok": True, "compatibility_ok": True})

    content = await orchestrator.approve_content(proposal_id)
    mutation = await orchestrator.authorize_mutation(proposal_id)

    assert content.approved and mutation.approved
    assert content.approval_id != mutation.approval_id
    assert [
        request.action.metadata["approval_stage"] for request in request_gate.requests
    ] == ["proposal_content", "plugin_mutation"]


@pytest.mark.asyncio
async def test_daemon_route_emits_two_separate_approval_requests(tmp_path: Path) -> None:
    queue, proposal_id = _queue(tmp_path)
    coordinator = ApprovalCoordinator()
    chunks: asyncio.Queue = asyncio.Queue()
    gate = SessionAwareGate(_DaemonApprovalGate(coordinator))
    orchestrator = ProposalOrchestrator(
        queue=queue,
        artifact_store=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        approval_gate=ApprovalOrchestrator(gate),
        policy=AdaptiveEvolutionPolicy(autonomy_level="generate_only"),
    )
    orchestrator.register_generated(
        proposal_id,
        "plugin = object()\n",
        validation={"ok": True, "compatibility_ok": True},
    )
    token = approval_route.set((chunks, "request-1"))
    coordinator.register_route("request-1")
    stages: list[str] = []
    try:
        for operation in (
            orchestrator.approve_content(proposal_id),
            orchestrator.authorize_mutation(proposal_id),
        ):
            pending = asyncio.create_task(operation)
            chunk = await asyncio.wait_for(chunks.get(), timeout=1.0)
            approval = chunk.metadata["approval"]
            stages.append(approval["action"]["metadata"]["approval_stage"])
            resolved = await coordinator.resolve(
                approval["pending_id"], "allow_once", reason="journey approval"
            )
            assert resolved["ok"] is True
            assert (await pending).approved is True
    finally:
        coordinator.unregister_route("request-1")
        approval_route.reset(token)

    assert stages == ["proposal_content", "plugin_mutation"]
    item = queue.get(proposal_id)
    assert item is not None
    assert item.proposal_approval_id
    assert item.mutation_approval_id
    assert item.proposal_approval_id != item.mutation_approval_id


def test_terminal_resolution_states_record_reasons(tmp_path: Path) -> None:
    for target in ("SUPERSEDED", "EXPIRED", "NO_OP"):
        queue, proposal_id = _queue(tmp_path / target.lower())
        orchestrator = ProposalOrchestrator(
            queue=queue,
            artifact_store=ContentAddressedArtifactStore(tmp_path / "artifacts" / target.lower()),
            approval_gate=None,
            policy=AdaptiveEvolutionPolicy(autonomy_level="generate_only"),
        )
        if target == "SUPERSEDED":
            item = orchestrator.supersede(
                proposal_id, replacement_id="prop-new", reason="new evidence"
            )
            assert item.metadata["replacement_proposal_id"] == "prop-new"
        elif target == "EXPIRED":
            item = orchestrator.expire(proposal_id, reason="review window elapsed")
        else:
            item = orchestrator.record_noop(proposal_id, reason="capability already available")
        assert item.status == target
        assert item.metadata["terminal_reason"]
        assert queue.active(limit=0) == []


def test_queue_rejects_lifecycle_shortcuts(tmp_path: Path) -> None:
    queue, proposal_id = _queue(tmp_path)

    with pytest.raises(ValueError, match="PENDING -> INSTALLED"):
        queue.transition(proposal_id, "INSTALLED")


def test_expire_records_reason_in_metadata(tmp_path: Path) -> None:
    queue, proposal_id = _queue(tmp_path)
    orchestrator = ProposalOrchestrator(
        queue=queue,
        artifact_store=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        approval_gate=None,
        policy=AdaptiveEvolutionPolicy(autonomy_level="generate_only"),
    )

    item = orchestrator.expire(proposal_id, reason="ttl_exceeded")

    assert item.status == "EXPIRED"
    assert item.metadata["terminal_reason"] == "ttl_exceeded"
    assert "swept_at" in item.metadata


def test_supersede_records_reason_in_metadata(tmp_path: Path) -> None:
    queue, proposal_id = _queue(tmp_path)
    orchestrator = ProposalOrchestrator(
        queue=queue,
        artifact_store=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        approval_gate=None,
        policy=AdaptiveEvolutionPolicy(autonomy_level="generate_only"),
    )

    item = orchestrator.supersede(proposal_id, replacement_id="prop-new", reason="newer_proposal_exists")

    assert item.status == "SUPERSEDED"
    assert item.metadata["terminal_reason"] == "newer_proposal_exists"
    assert item.metadata["replacement_proposal_id"] == "prop-new"
    assert "swept_at" in item.metadata


def test_quarantine_to_probation_transition_allowed(tmp_path: Path) -> None:
    """QUARANTINED proposals can be transitioned back to PROBATION for re-trial."""
    queue, proposal_id = _queue(tmp_path)
    # Walk the proposal through PENDING -> GENERATED -> APPROVED -> INSTALLED -> QUARANTINED
    queue.transition(proposal_id, "GENERATED")
    queue.transition(proposal_id, "APPROVED")
    queue.transition(proposal_id, "INSTALLED")
    queue.transition(proposal_id, "QUARANTINED")
    item = queue.get(proposal_id)
    assert item.status == "QUARANTINED"

    # Now transition QUARANTINED -> PROBATION
    recovered = queue.transition(proposal_id, "PROBATION")
    assert recovered.status == "PROBATION"
