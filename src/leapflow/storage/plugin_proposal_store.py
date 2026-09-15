# Copyright (c) Alibaba, Inc. and its affiliates.
"""Profile-scoped JSON store for plugin proposals.

The store intentionally uses the path supplied by ProfileLayout
(``profile_layout.plugin_proposals_path``). It does not infer profile roots or
assemble managed paths itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from leapflow.domain.plugin_proposal import (
    BehaviorTestCase,
    GapEvidence,
    PluginProposal,
    ProposalStatus,
    ProposedToolSpec,
)


class JsonPluginProposalStore:
    """Durable profile-local store for reviewable plugin proposals."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def list(self) -> list[PluginProposal]:
        return [self._proposal_from_dict(item) for item in self._load()]

    def get(self, proposal_id: str) -> PluginProposal | None:
        target = str(proposal_id or "")
        for proposal in self.list():
            if proposal.proposal_id == target:
                return proposal
        return None

    def save(self, proposal: PluginProposal) -> PluginProposal:
        items = [item for item in self._load() if item.get("proposal_id") != proposal.proposal_id]
        items.append(proposal.to_dict())
        self._save(items)
        return proposal

    def update_status(self, proposal_id: str, status: ProposalStatus) -> PluginProposal | None:
        proposal = self.get(proposal_id)
        if proposal is None:
            return None
        updated = PluginProposal(
            proposal_id=proposal.proposal_id,
            plugin_id=proposal.plugin_id,
            capability_summary=proposal.capability_summary,
            gap_type=proposal.gap_type,
            risk_level=proposal.risk_level,
            status=status,
            evidence=proposal.evidence,
            proposed_tools=proposal.proposed_tools,
            test_cases=proposal.test_cases,
            created_at=proposal.created_at,
        )
        return self.save(updated)

    def _load(self) -> list[dict[str, Any]]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (json.JSONDecodeError, OSError, ValueError):
            return []
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    def _save(self, items: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _proposal_from_dict(raw: dict[str, Any]) -> PluginProposal:
        evidence = tuple(
            GapEvidence.create(
                str(item.get("evidence_type") or "unknown"),
                str(item.get("summary") or ""),
                confidence=float(item.get("confidence") or 0.0),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in raw.get("evidence", [])
            if isinstance(item, dict)
        )
        tools = tuple(
            ProposedToolSpec(
                name=str(item.get("name") or "generated_tool"),
                description=str(item.get("description") or ""),
                risk_level=str(item.get("risk_level") or "read_only"),  # type: ignore[arg-type]
                mutates_state=bool(item.get("mutates_state", False)),
            )
            for item in raw.get("proposed_tools", [])
            if isinstance(item, dict)
        )
        tests = tuple(
            BehaviorTestCase.create(
                str(item.get("tool_name") or ""),
                arguments=dict(item.get("arguments") or {}),
                expected_subset=dict(item.get("expected_subset") or {}),
                description=str(item.get("description") or ""),
            )
            for item in raw.get("test_cases", [])
            if isinstance(item, dict)
        )
        return PluginProposal(
            proposal_id=str(raw.get("proposal_id") or ""),
            plugin_id=str(raw.get("plugin_id") or "generated_plugin"),
            capability_summary=str(raw.get("capability_summary") or ""),
            gap_type=str(raw.get("gap_type") or "tool_plugin"),  # type: ignore[arg-type]
            risk_level=str(raw.get("risk_level") or "read_only"),  # type: ignore[arg-type]
            status=str(raw.get("status") or "draft"),  # type: ignore[arg-type]
            evidence=evidence,
            proposed_tools=tools,
            test_cases=tests,
            created_at=float(raw.get("created_at") or 0.0),
        )
