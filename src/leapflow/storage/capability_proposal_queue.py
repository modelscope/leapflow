# Copyright (c) Alibaba, Inc. and its affiliates.
"""Durable capability proposal queue for adaptive plugin evolution."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.domain.event_types import EvolutionEventType
from leapflow.domain.evolution_event import EvolutionContext, EvolutionEvent, content_hash

ProposalStatus = Literal[
    "PENDING",
    "GENERATED",
    "APPROVED",
    "INSTALLED",
    "PROBATION",
    "VERIFIED",
    "REJECTED",
    "FAILED",
    "QUARANTINED",
    "SUPERSEDED",
    "EXPIRED",
    "NO_OP",
]

_ACTIVE_STATUSES = {
    "PENDING",
    "GENERATED",
    "APPROVED",
    "INSTALLED",
    "PROBATION",
    "VERIFIED",
}
_CANCELLATION_STATUSES = frozenset({"REJECTED", "FAILED", "SUPERSEDED", "EXPIRED", "NO_OP"})
_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "PENDING": frozenset({"GENERATED", *_CANCELLATION_STATUSES}),
    "GENERATED": frozenset({"APPROVED", *_CANCELLATION_STATUSES}),
    "APPROVED": frozenset({"INSTALLED", *_CANCELLATION_STATUSES}),
    "INSTALLED": frozenset({"PROBATION", "QUARANTINED", "FAILED"}),
    "PROBATION": frozenset({"VERIFIED", "QUARANTINED", "FAILED"}),
    "VERIFIED": frozenset({"PROBATION", "QUARANTINED", "FAILED"}),
    "QUARANTINED": frozenset({"FAILED"}),
    "REJECTED": frozenset(),
    "FAILED": frozenset(),
    "SUPERSEDED": frozenset(),
    "EXPIRED": frozenset(),
    "NO_OP": frozenset(),
}


@dataclass(frozen=True)
class CapabilityProposalItem:
    """One queued adaptive evolution proposal."""

    proposal_id: str
    status: ProposalStatus
    requirements: tuple[Mapping[str, Any], ...]
    environment: Mapping[str, Any] = field(default_factory=dict)
    risk: Mapping[str, Any] = field(default_factory=dict)
    source: str = "runtime"
    observation_ids: tuple[str, ...] = ()
    policy_decision: Mapping[str, Any] = field(default_factory=dict)
    generated_code_ref: str = ""
    proposal_approval_id: str = ""
    mutation_approval_id: str = ""
    install_result: Mapping[str, Any] = field(default_factory=dict)
    test_results: tuple[Mapping[str, Any], ...] = ()
    trust_state: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "status": self.status,
            "requirements": [dict(item) for item in self.requirements],
            "environment": dict(self.environment),
            "risk": dict(self.risk),
            "source": self.source,
            "observation_ids": list(self.observation_ids),
            "policy_decision": dict(self.policy_decision),
            "generated_code_ref": self.generated_code_ref,
            "proposal_approval_id": self.proposal_approval_id,
            "mutation_approval_id": self.mutation_approval_id,
            "install_result": dict(self.install_result),
            "test_results": [dict(item) for item in self.test_results],
            "trust_state": dict(self.trust_state),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilityProposalItem":
        return cls(
            proposal_id=str(data.get("proposal_id") or ""),
            status=_coerce_status(data.get("status")),
            requirements=tuple(
                dict(item) for item in data.get("requirements") or () if isinstance(item, Mapping)
            ),
            environment=dict(data.get("environment") or {}),
            risk=dict(data.get("risk") or {}),
            source=str(data.get("source") or "runtime"),
            observation_ids=tuple(str(item) for item in data.get("observation_ids") or ()),
            policy_decision=dict(data.get("policy_decision") or {}),
            generated_code_ref=str(data.get("generated_code_ref") or ""),
            proposal_approval_id=str(data.get("proposal_approval_id") or ""),
            mutation_approval_id=str(data.get("mutation_approval_id") or ""),
            install_result=dict(data.get("install_result") or {}),
            test_results=tuple(
                dict(item) for item in data.get("test_results") or () if isinstance(item, Mapping)
            ),
            trust_state=dict(data.get("trust_state") or {}),
            created_at=float(data.get("created_at") or 0.0),
            updated_at=float(data.get("updated_at") or 0.0),
            metadata=dict(data.get("metadata") or {}),
        )


_STATUS_EVENT_TYPES: dict[str, str] = {
    "PENDING": EvolutionEventType.PROPOSAL_CREATED,
    "GENERATED": EvolutionEventType.PROPOSAL_GENERATED,
    "APPROVED": EvolutionEventType.PROPOSAL_APPROVED,
    "INSTALLED": EvolutionEventType.PLUGIN_INSTALLED,
    "PROBATION": EvolutionEventType.PLUGIN_PROBATION_STARTED,
    "VERIFIED": EvolutionEventType.PLUGIN_VERIFIED,
    "REJECTED": EvolutionEventType.PROPOSAL_REJECTED,
    "FAILED": EvolutionEventType.PROPOSAL_FAILED,
    "QUARANTINED": EvolutionEventType.PLUGIN_QUARANTINED,
    "SUPERSEDED": EvolutionEventType.PROPOSAL_SUPERSEDED,
    "EXPIRED": EvolutionEventType.PROPOSAL_EXPIRED,
    "NO_OP": EvolutionEventType.PROPOSAL_NO_OP,
}


class EvolutionCapabilityProposalStore:
    """Event-sourced capability proposal lifecycle used by production runtime."""

    def __init__(self, event_store: Any, *, profile_id: str) -> None:
        self._event_store = event_store
        self._profile_id = str(profile_id)
        self._lock = threading.RLock()

    def enqueue(
        self,
        *,
        requirements: Sequence[CapabilityRequirement | Mapping[str, Any]],
        environment: Mapping[str, Any] | None = None,
        risk: Mapping[str, Any] | None = None,
        source: str = "runtime",
        observation_ids: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> CapabilityProposalItem:
        with self._lock:
            item, event = self.prepare_enqueue(
                requirements=requirements,
                environment=environment,
                risk=risk,
                source=source,
                observation_ids=observation_ids,
                metadata=metadata,
            )
            if event is None:
                return item
            self._event_store.append(event)
            return self.get(item.proposal_id) or item

    def prepare_enqueue(
        self,
        *,
        requirements: Sequence[CapabilityRequirement | Mapping[str, Any]],
        environment: Mapping[str, Any] | None = None,
        risk: Mapping[str, Any] | None = None,
        source: str = "runtime",
        observation_ids: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        occurred_at: float | None = None,
    ) -> tuple[CapabilityProposalItem, EvolutionEvent | None]:
        """Build an idempotent creation event without writing it.

        Durable workers use this seam to commit the proposal in the same
        transaction as the teacher result. Ordinary callers can keep using
        :meth:`enqueue`, which appends the prepared event immediately.
        """
        req_payload = tuple(_requirement_dict(item) for item in requirements)
        proposal_id = _proposal_identity(req_payload, environment or {})
        with self._lock:
            existing = self.get(proposal_id)
            if existing is not None:
                return existing, None
            now = time.time() if occurred_at is None else float(occurred_at)
            item = CapabilityProposalItem(
                proposal_id=proposal_id,
                status="PENDING",
                requirements=req_payload,
                environment=dict(environment or {}),
                risk=dict(risk or {}),
                source=str(source or "runtime"),
                observation_ids=tuple(str(item) for item in observation_ids),
                created_at=now,
                updated_at=now,
                metadata=dict(metadata or {}),
            )
            return item, self._state_event(item, previous_status="")

    def get(self, proposal_id: str) -> CapabilityProposalItem | None:
        records = self._event_store.read(
            profile_id=self._profile_id,
            proposal_id=str(proposal_id),
            limit=5000,
        )
        for record in reversed(records):
            payload = record.event.to_dict()["payload"].get("proposal_state")
            if isinstance(payload, Mapping):
                return CapabilityProposalItem.from_dict(payload)
        return None

    def update(
        self,
        proposal_id: str,
        *,
        status: ProposalStatus | None = None,
        policy_decision: Mapping[str, Any] | None = None,
        generated_code_ref: str | None = None,
        proposal_approval_id: str | None = None,
        mutation_approval_id: str | None = None,
        install_result: Mapping[str, Any] | None = None,
        test_results: Sequence[Mapping[str, Any]] | None = None,
        trust_state: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CapabilityProposalItem | None:
        with self._lock:
            item = self.get(proposal_id)
            if item is None:
                return None
            target = _coerce_status(status or item.status)
            if target != item.status and target not in _ALLOWED_TRANSITIONS.get(
                item.status, frozenset()
            ):
                raise ValueError(
                    f"illegal proposal transition: {item.status} -> {target}"
                )
            updated = CapabilityProposalItem(
                proposal_id=item.proposal_id,
                status=target,
                requirements=item.requirements,
                environment=item.environment,
                risk=item.risk,
                source=item.source,
                observation_ids=item.observation_ids,
                policy_decision=dict(
                    policy_decision if policy_decision is not None else item.policy_decision
                ),
                generated_code_ref=(
                    item.generated_code_ref
                    if generated_code_ref is None
                    else str(generated_code_ref)
                ),
                proposal_approval_id=(
                    item.proposal_approval_id
                    if proposal_approval_id is None
                    else str(proposal_approval_id)
                ),
                mutation_approval_id=(
                    item.mutation_approval_id
                    if mutation_approval_id is None
                    else str(mutation_approval_id)
                ),
                install_result=dict(
                    install_result if install_result is not None else item.install_result
                ),
                test_results=tuple(
                    dict(result)
                    for result in (
                        test_results if test_results is not None else item.test_results
                    )
                ),
                trust_state=dict(
                    trust_state if trust_state is not None else item.trust_state
                ),
                created_at=item.created_at,
                updated_at=time.time(),
                metadata={**dict(item.metadata), **dict(metadata or {})},
            )
            return self._append_state(updated, previous_status=item.status)

    def transition(
        self,
        proposal_id: str,
        status: ProposalStatus,
        **changes: Any,
    ) -> CapabilityProposalItem:
        with self._lock:
            current = self.get(proposal_id)
            if current is None:
                raise KeyError(f"unknown capability proposal: {proposal_id}")
            target = _coerce_status(status)
            if target == current.status:
                updated = self.update(proposal_id, **changes) if changes else current
                if updated is None:
                    raise KeyError(f"unknown capability proposal: {proposal_id}")
                return updated
            if target not in _ALLOWED_TRANSITIONS.get(current.status, frozenset()):
                raise ValueError(
                    f"illegal proposal transition: {current.status} -> {target}"
                )
            updated = self.update(proposal_id, status=target, **changes)
            if updated is None:
                raise KeyError(f"unknown capability proposal: {proposal_id}")
            return updated

    def list_items(
        self, *, status: ProposalStatus | str = "", limit: int = 50
    ) -> list[CapabilityProposalItem]:
        latest: dict[str, CapabilityProposalItem] = {}
        cursor = 0
        while True:
            records = self._event_store.read(
                profile_id=self._profile_id,
                proposal_events_only=True,
                after_sequence=cursor,
                limit=5000,
            )
            if not records:
                break
            for record in records:
                payload = record.event.to_dict()["payload"].get("proposal_state")
                if isinstance(payload, Mapping):
                    item = CapabilityProposalItem.from_dict(payload)
                    latest[item.proposal_id] = item
            cursor = records[-1].sequence
            if len(records) < 5000:
                break
        items = list(latest.values())
        if status:
            items = [item for item in items if item.status == str(status)]
        items.sort(key=lambda item: item.updated_at or item.created_at, reverse=True)
        return items if limit <= 0 else items[:limit]

    def find_by_metadata(self, key: str, value: str) -> CapabilityProposalItem | None:
        target = str(value)
        for item in self.list_items(limit=0):
            if str(item.metadata.get(key) or "") == target:
                return item
        return None

    def active(self, *, limit: int = 50) -> list[CapabilityProposalItem]:
        items = [item for item in self.list_items(limit=0) if item.status in _ACTIVE_STATUSES]
        return items if limit <= 0 else items[:limit]

    def _append_state(
        self,
        item: CapabilityProposalItem,
        *,
        previous_status: str,
    ) -> CapabilityProposalItem:
        self._event_store.append(self._state_event(item, previous_status=previous_status))
        stored = self.get(item.proposal_id)
        return stored or item

    def _state_event(self, item: CapabilityProposalItem, *, previous_status: str) -> EvolutionEvent:
        event_type = (
            _STATUS_EVENT_TYPES[item.status]
            if item.status != previous_status
            else EvolutionEventType.PROPOSAL_UPDATED
        )
        state = item.to_dict()
        identity = dict(state)
        identity.pop("updated_at", None)
        first_requirement = dict(item.requirements[0]) if item.requirements else {}
        return EvolutionEvent.create(
            event_type,
            context=EvolutionContext(
                profile_id=self._profile_id,
                workspace_id=str(item.environment.get("workspace_id") or ""),
                session_id=str(item.environment.get("session_id") or ""),
                requirement_id=str(first_requirement.get("requirement_id") or ""),
                proposal_id=item.proposal_id,
                artifact_id=item.generated_code_ref,
                plugin_id=str(item.metadata.get("plugin_id") or ""),
                correlation_id=item.proposal_id,
            ),
            payload={
                "proposal_state": state,
                "previous_status": previous_status,
                "status": item.status,
                "reason": str(item.metadata.get("terminal_reason") or ""),
            },
            producer="proposal.orchestrator",
            privacy_class="profile",
            occurred_at=item.updated_at,
            dedup_key=(
                f"proposal.created:{item.proposal_id}"
                if not previous_status
                else f"proposal.state:{item.proposal_id}:{item.status}:{content_hash(identity)}"
            ),
        )


def _proposal_identity(
    requirements: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
) -> str:
    identity = [
        {
            "requirement_id": str(item.get("requirement_id") or ""),
            "capability": str(item.get("capability") or ""),
            "origin": str(item.get("origin") or ""),
            "max_risk_level": str(item.get("max_risk_level") or ""),
            "required_platform_capabilities": sorted(
                str(cap) for cap in (item.get("required_platform_capabilities") or [])
            ),
        }
        for item in requirements
    ]
    material = {
        "identity": identity,
        "environment": {
            "fingerprint_id": environment.get("fingerprint_id", ""),
            "platform_capabilities": environment.get("platform_capabilities", []),
            "workspace_markers": environment.get("workspace_markers", []),
        },
    }
    return "prop-" + content_hash(material)[:16]


def _coerce_status(value: Any) -> ProposalStatus:
    raw = str(value or "PENDING").upper()
    allowed = ProposalStatus.__args__  # type: ignore[attr-defined]
    return raw if raw in allowed else "PENDING"  # type: ignore[return-value]


def _requirement_dict(item: CapabilityRequirement | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(item, CapabilityRequirement):
        return item.to_dict()
    return dict(item)


__all__ = [
    "CapabilityProposalItem",
    "EvolutionCapabilityProposalStore",
    "ProposalStatus",
]
