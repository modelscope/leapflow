# Copyright (c) Alibaba, Inc. and its affiliates.
"""Durable capability proposal queue for adaptive plugin evolution."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from leapflow.domain.capability_requirement import CapabilityRequirement

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
]

_ACTIVE_STATUSES = {"PENDING", "GENERATED", "APPROVED", "INSTALLED", "PROBATION"}


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
    approval_id: str = ""
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
            "approval_id": self.approval_id,
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
            approval_id=str(data.get("approval_id") or ""),
            install_result=dict(data.get("install_result") or {}),
            test_results=tuple(
                dict(item) for item in data.get("test_results") or () if isinstance(item, Mapping)
            ),
            trust_state=dict(data.get("trust_state") or {}),
            created_at=float(data.get("created_at") or 0.0),
            updated_at=float(data.get("updated_at") or 0.0),
            metadata=dict(data.get("metadata") or {}),
        )


class JsonCapabilityProposalQueue:
    """Profile-scoped durable queue of adaptive evolution proposals."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

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
        """Create or return an active proposal for the requirement/environment pair."""
        req_payload = tuple(_requirement_dict(item) for item in requirements)
        proposal_id = self._proposal_id(req_payload, environment or {}, observation_ids)
        existing = self.get(proposal_id)
        if existing is not None and existing.status in _ACTIVE_STATUSES:
            return existing
        now = time.time()
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
        self._upsert(item)
        return item

    def get(self, proposal_id: str) -> CapabilityProposalItem | None:
        for item in self.list_items(limit=0):
            if item.proposal_id == proposal_id:
                return item
        return None

    def update(
        self,
        proposal_id: str,
        *,
        status: ProposalStatus | None = None,
        policy_decision: Mapping[str, Any] | None = None,
        generated_code_ref: str | None = None,
        approval_id: str | None = None,
        install_result: Mapping[str, Any] | None = None,
        test_results: Sequence[Mapping[str, Any]] | None = None,
        trust_state: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CapabilityProposalItem | None:
        item = self.get(proposal_id)
        if item is None:
            return None
        updated = CapabilityProposalItem(
            proposal_id=item.proposal_id,
            status=_coerce_status(status or item.status),
            requirements=item.requirements,
            environment=item.environment,
            risk=item.risk,
            source=item.source,
            observation_ids=item.observation_ids,
            policy_decision=dict(
                policy_decision if policy_decision is not None else item.policy_decision
            ),
            generated_code_ref=item.generated_code_ref
            if generated_code_ref is None
            else str(generated_code_ref),
            approval_id=item.approval_id if approval_id is None else str(approval_id),
            install_result=dict(
                install_result if install_result is not None else item.install_result
            ),
            test_results=tuple(
                dict(result)
                for result in (test_results if test_results is not None else item.test_results)
            ),
            trust_state=dict(trust_state if trust_state is not None else item.trust_state),
            created_at=item.created_at,
            updated_at=time.time(),
            metadata={**dict(item.metadata), **dict(metadata or {})},
        )
        self._upsert(updated)
        return updated

    def list_items(
        self, *, status: ProposalStatus | str = "", limit: int = 50
    ) -> list[CapabilityProposalItem]:
        payload = self._load_payload()
        items = [
            CapabilityProposalItem.from_dict(item)
            for item in payload.get("proposals", [])
            if isinstance(item, Mapping)
        ]
        if status:
            status_value = str(status)
            items = [item for item in items if item.status == status_value]
        items.sort(key=lambda item: item.updated_at or item.created_at, reverse=True)
        return items if limit <= 0 else items[:limit]

    def active(self, *, limit: int = 50) -> list[CapabilityProposalItem]:
        return [item for item in self.list_items(limit=0) if item.status in _ACTIVE_STATUSES][
            :limit
        ]

    def _upsert(self, item: CapabilityProposalItem) -> None:
        payload = self._load_payload()
        proposals = [entry for entry in payload.get("proposals", []) if isinstance(entry, Mapping)]
        proposals = [entry for entry in proposals if entry.get("proposal_id") != item.proposal_id]
        proposals.append(item.to_dict())
        payload["proposals"] = proposals
        self._write_payload(payload)

    def _proposal_id(
        self,
        requirements: Sequence[Mapping[str, Any]],
        environment: Mapping[str, Any],
        observation_ids: Sequence[str],
    ) -> str:
        material = {
            "requirements": [dict(item) for item in requirements],
            "environment": {
                "fingerprint_id": environment.get("fingerprint_id", ""),
                "platform_capabilities": environment.get("platform_capabilities", []),
                "workspace_markers": environment.get("workspace_markers", []),
            },
            "observation_ids": sorted(str(item) for item in observation_ids),
        }
        text = json.dumps(material, sort_keys=True, ensure_ascii=False, default=str)
        import hashlib

        return "prop-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _load_payload(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"version": 1, "proposals": []}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, Mapping) and isinstance(data.get("proposals"), list):
                return {"version": int(data.get("version") or 1), "proposals": data["proposals"]}
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {"version": 1, "proposals": []}
        return {"version": 1, "proposals": []}

    def _write_payload(self, payload: Mapping[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _coerce_status(value: Any) -> ProposalStatus:
    raw = str(value or "PENDING").upper()
    allowed = ProposalStatus.__args__  # type: ignore[attr-defined]
    return raw if raw in allowed else "PENDING"  # type: ignore[return-value]


def _requirement_dict(item: CapabilityRequirement | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(item, CapabilityRequirement):
        return item.to_dict()
    return dict(item)


__all__ = ["CapabilityProposalItem", "JsonCapabilityProposalQueue", "ProposalStatus"]
