# Copyright (c) Alibaba, Inc. and its affiliates.
"""Durable orchestration for governed capability acquisition proposals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from leapflow.evolution.artifact_store import ContentAddressedArtifactStore
from leapflow.learning.plugin_trust import PluginTrustLevel
from leapflow.plugins.adaptive_policy import AdaptiveEvolutionPolicy
from leapflow.security.actions import ActionDescriptor
from leapflow.storage.capability_proposal_queue import CapabilityProposalItem


@runtime_checkable
class ProposalApprovalGate(Protocol):
    async def evaluate(self, action: ActionDescriptor) -> Any: ...


@dataclass(frozen=True)
class ProposalApproval:
    """One explicit approval result tied to a durable proposal transition."""

    approved: bool
    proposal_id: str
    stage: str
    approval_id: str = ""
    denial_message: str = ""


class ProposalOrchestrator:
    """Apply policy and two explicit approvals to one acquisition lifecycle."""

    def __init__(
        self,
        *,
        queue: Any,
        artifact_store: ContentAddressedArtifactStore,
        approval_gate: ProposalApprovalGate | None,
        policy: AdaptiveEvolutionPolicy,
    ) -> None:
        self._queue = queue
        self._artifacts = artifact_store
        self._approval_gate = approval_gate
        self._policy = policy

    def register_generated(
        self,
        proposal_id: str,
        code: str,
        *,
        validation: Mapping[str, Any],
    ) -> CapabilityProposalItem:
        """Persist validated source in CAS and advance PENDING to GENERATED."""
        proposal = self._required(proposal_id)
        if proposal.status in {"GENERATED", "APPROVED"} and proposal.generated_code_ref:
            if self._artifacts.get_text(proposal.generated_code_ref) == code:
                return proposal
            raise ValueError(f"proposal {proposal_id} already references a different artifact")
        if proposal.status != "PENDING":
            raise ValueError(f"proposal {proposal_id} is not pending")
        decision = self._policy.decide(proposal, trust_level=PluginTrustLevel.DRAFT)
        if not decision.allowed or decision.action != "generate":
            raise PermissionError(decision.reason)
        validation_payload = dict(validation)
        if not bool(validation_payload.get("ok", False)) or validation_payload.get(
            "compatibility_ok"
        ) is not True:
            return self._queue.transition(
                proposal_id,
                "FAILED",
                policy_decision=decision.to_dict(),
                metadata={
                    "validation": validation_payload,
                    "terminal_reason": "static or compatibility validation failed",
                },
            )
        artifact = self._artifacts.put_text(
            code,
            media_type="text/x-python; charset=utf-8",
            privacy_class="profile",
        )
        return self._queue.transition(
            proposal_id,
            "GENERATED",
            generated_code_ref=artifact.artifact_id,
            policy_decision=decision.to_dict(),
            metadata={"validation": validation_payload, "artifact": artifact.to_dict()},
        )

    async def approve_content(self, proposal_id: str) -> ProposalApproval:
        """Ask whether the generated implementation is acceptable in principle."""
        proposal = self._required(proposal_id)
        if proposal.status == "APPROVED" and proposal.proposal_approval_id:
            return ProposalApproval(
                True, proposal_id, "content", approval_id=proposal.proposal_approval_id
            )
        if proposal.status != "GENERATED":
            raise ValueError(f"proposal {proposal_id} has no generated artifact to approve")
        descriptor = ActionDescriptor.platform_action(
            "plugin_management",
            "approve_proposal_content",
            {
                "proposal_id": proposal_id,
                "artifact_id": proposal.generated_code_ref,
                "requirements": [dict(item) for item in proposal.requirements],
            },
            metadata={
                "effect": "write",
                "risk_level": "high",
                "category": "self_modification",
                "approval_stage": "proposal_content",
                "proposal_id": proposal_id,
            },
        )
        gate = self._approval_gate
        if gate is None:
            return self._reject(
                proposal_id,
                stage="content",
                approval_id=descriptor.action_id,
                reason="approval_gate_missing",
                denial_message="proposal approval blocked: no approval gate configured",
            )
        try:
            result = await gate.evaluate(descriptor)
        except Exception as exc:  # noqa: BLE001 - an unavailable gate must fail closed
            return self._reject(
                proposal_id,
                stage="content",
                approval_id=descriptor.action_id,
                reason=f"approval_gate_error:{type(exc).__name__}",
                denial_message="proposal approval failed closed",
            )
        approval_id = str(
            getattr(getattr(result, "action", None), "action_id", "")
            or descriptor.action_id
        )
        if not bool(getattr(result, "approved", False)):
            return self._reject(
                proposal_id,
                stage="content",
                approval_id=approval_id,
                reason=str(getattr(result, "reason", "") or "user_denied"),
                denial_message=str(getattr(result, "denial_message", "") or "approval denied"),
            )
        self._queue.transition(
            proposal_id,
            "APPROVED",
            proposal_approval_id=approval_id,
            policy_decision={
                "action": "request_approval",
                "reason": "generated proposal content approved",
                "approval_stage": "proposal_content",
            },
        )
        return ProposalApproval(True, proposal_id, "content", approval_id=approval_id)

    async def authorize_mutation(self, proposal_id: str) -> ProposalApproval:
        """Ask separately for the process-global install mutation."""
        proposal = self._required(proposal_id)
        if proposal.status == "APPROVED" and proposal.mutation_approval_id:
            return ProposalApproval(
                True, proposal_id, "mutation", approval_id=proposal.mutation_approval_id
            )
        if proposal.status != "APPROVED" or not proposal.proposal_approval_id:
            raise PermissionError("proposal content must be approved before installation")
        plugin_id = str(dict(proposal.metadata).get("plugin_id") or proposal_id)
        descriptor = ActionDescriptor.platform_action(
            "plugin_management",
            "install",
            {"proposal_id": proposal_id, "plugin_id": plugin_id},
            metadata={
                "effect": "write",
                "risk_level": "high",
                "category": "self_modification",
                "approval_stage": "plugin_mutation",
                "proposal_id": proposal_id,
            },
        )
        gate = self._approval_gate
        if gate is None:
            return self._reject(
                proposal_id,
                stage="mutation",
                approval_id=descriptor.action_id,
                reason="approval_gate_missing",
                denial_message="plugin mutation blocked: no approval gate configured",
            )
        try:
            result = await gate.evaluate(descriptor)
        except Exception as exc:  # noqa: BLE001 - an unavailable gate must fail closed
            return self._reject(
                proposal_id,
                stage="mutation",
                approval_id=descriptor.action_id,
                reason=f"approval_gate_error:{type(exc).__name__}",
                denial_message="plugin mutation approval failed closed",
            )
        approval_id = str(
            getattr(getattr(result, "action", None), "action_id", "")
            or descriptor.action_id
        )
        if not bool(getattr(result, "approved", False)):
            return self._reject(
                proposal_id,
                stage="mutation",
                approval_id=approval_id,
                reason=str(getattr(result, "reason", "") or "user_denied"),
                denial_message=str(getattr(result, "denial_message", "") or "approval denied"),
            )
        self._queue.update(proposal_id, mutation_approval_id=approval_id)
        return ProposalApproval(True, proposal_id, "mutation", approval_id=approval_id)

    def generated_code(self, proposal_id: str) -> str:
        proposal = self._required(proposal_id)
        if not proposal.generated_code_ref:
            raise ValueError(f"proposal {proposal_id} has no generated artifact")
        return self._artifacts.get_text(proposal.generated_code_ref)

    def record_installed(
        self,
        proposal_id: str,
        install_result: Mapping[str, Any],
    ) -> CapabilityProposalItem:
        proposal = self._required(proposal_id)
        if proposal.status == "INSTALLED":
            return proposal
        if proposal.status != "APPROVED" or not proposal.mutation_approval_id:
            raise PermissionError("plugin mutation approval is required before installation")
        target = "INSTALLED" if bool(install_result.get("ok")) else "FAILED"
        reason = "plugin installed" if target == "INSTALLED" else str(
            install_result.get("error") or "plugin installation failed"
        )
        return self._queue.transition(
            proposal_id,
            target,  # type: ignore[arg-type]
            install_result=dict(install_result),
            metadata={"terminal_reason": reason} if target == "FAILED" else {},
        )

    def supersede(self, proposal_id: str, *, replacement_id: str, reason: str) -> CapabilityProposalItem:
        """Close an uninstalled proposal in favor of a newer durable proposal."""
        return self._queue.transition(
            proposal_id,
            "SUPERSEDED",
            metadata={"terminal_reason": reason, "replacement_proposal_id": replacement_id},
        )

    def expire(self, proposal_id: str, *, reason: str) -> CapabilityProposalItem:
        """Close an uninstalled proposal whose review window has elapsed."""
        return self._queue.transition(
            proposal_id,
            "EXPIRED",
            metadata={"terminal_reason": reason},
        )

    def record_noop(self, proposal_id: str, *, reason: str) -> CapabilityProposalItem:
        """Close a proposal resolved without acquiring a new capability."""
        return self._queue.transition(
            proposal_id,
            "NO_OP",
            metadata={"terminal_reason": reason},
        )

    def _reject(
        self,
        proposal_id: str,
        *,
        stage: str,
        approval_id: str,
        reason: str,
        denial_message: str,
    ) -> ProposalApproval:
        approval_field = (
            {"proposal_approval_id": approval_id}
            if stage == "content"
            else {"mutation_approval_id": approval_id}
        )
        self._queue.transition(
            proposal_id,
            "REJECTED",
            **approval_field,
            policy_decision={
                "action": "reject",
                "reason": reason,
                "approval_stage": "proposal_content" if stage == "content" else "plugin_mutation",
            },
            metadata={"terminal_reason": reason},
        )
        return ProposalApproval(
            False,
            proposal_id,
            stage,
            approval_id=approval_id,
            denial_message=denial_message,
        )

    def _required(self, proposal_id: str) -> CapabilityProposalItem:
        proposal = self._queue.get(proposal_id)
        if proposal is None:
            raise KeyError(f"unknown capability proposal: {proposal_id}")
        return proposal


__all__ = ["ProposalApproval", "ProposalApprovalGate", "ProposalOrchestrator"]
