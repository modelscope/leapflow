# Copyright (c) Alibaba, Inc. and its affiliates.
"""Domain types for capability gaps and plugin proposals.

These immutable records are the reviewable bridge between observing that
LeapFlow lacks a capability and asking the plugin generator to create one.
They deliberately contain no generation or installation side effects.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

GapType = Literal["tool_plugin", "gateway_adapter", "signal_source", "llm_provider", "unknown"]
ProposalStatus = Literal["draft", "review", "approved", "rejected"]
RiskLevel = Literal["read_only", "low", "medium", "high", "mutating", "external"]


def _freeze_metadata(metadata: dict[str, Any] | None = None) -> tuple[tuple[str, str], ...]:
    """Convert arbitrary metadata to a stable immutable string map."""
    if not metadata:
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in metadata.items()))


def _freeze_mapping(mapping: dict[str, Any] | None = None) -> tuple[tuple[str, Any], ...]:
    """Convert a dict into an immutable tuple while preserving JSON-like values."""
    if not mapping:
        return ()
    return tuple(sorted((str(key), value) for key, value in mapping.items()))


@dataclass(frozen=True)
class GapEvidence:
    """One structured observation that a capability is missing."""

    evidence_type: str
    summary: str
    confidence: float = 0.0
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @classmethod
    def create(
        cls,
        evidence_type: str,
        summary: str,
        *,
        confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> "GapEvidence":
        return cls(
            evidence_type=str(evidence_type),
            summary=str(summary),
            confidence=max(0.0, min(1.0, float(confidence))),
            metadata=_freeze_metadata(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_type": self.evidence_type,
            "summary": self.summary,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProposedToolSpec:
    """Reviewable sketch of one tool a generated plugin should expose."""

    name: str
    description: str
    risk_level: RiskLevel = "read_only"
    mutates_state: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level,
            "mutates_state": self.mutates_state,
        }


@dataclass(frozen=True)
class BehaviorTestCase:
    """One expected behavior check for a proposed plugin tool."""

    tool_name: str
    arguments: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    expected_subset: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    description: str = ""

    @classmethod
    def create(
        cls,
        tool_name: str,
        *,
        arguments: dict[str, Any] | None = None,
        expected_subset: dict[str, Any] | None = None,
        description: str = "",
    ) -> "BehaviorTestCase":
        return cls(
            tool_name=str(tool_name),
            arguments=_freeze_mapping(arguments),
            expected_subset=_freeze_mapping(expected_subset),
            description=str(description or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "expected_subset": dict(self.expected_subset),
            "description": self.description,
        }


@dataclass(frozen=True)
class PluginProposal:
    """A side-effect-free proposal for generating or installing a plugin."""

    proposal_id: str
    plugin_id: str
    capability_summary: str
    gap_type: GapType
    risk_level: RiskLevel
    status: ProposalStatus = "draft"
    evidence: tuple[GapEvidence, ...] = field(default_factory=tuple)
    proposed_tools: tuple[ProposedToolSpec, ...] = field(default_factory=tuple)
    test_cases: tuple[BehaviorTestCase, ...] = field(default_factory=tuple)
    created_at: float = field(default_factory=time.time)

    @classmethod
    def create(
        cls,
        *,
        plugin_id: str,
        capability_summary: str,
        gap_type: GapType = "tool_plugin",
        risk_level: RiskLevel = "read_only",
        evidence: tuple[GapEvidence, ...] = (),
        proposed_tools: tuple[ProposedToolSpec, ...] = (),
        test_cases: tuple[BehaviorTestCase, ...] = (),
        status: ProposalStatus = "draft",
    ) -> "PluginProposal":
        return cls(
            proposal_id=uuid.uuid4().hex[:12],
            plugin_id=plugin_id,
            capability_summary=capability_summary,
            gap_type=gap_type,
            risk_level=risk_level,
            status=status,
            evidence=evidence,
            proposed_tools=proposed_tools,
            test_cases=test_cases,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "plugin_id": self.plugin_id,
            "capability_summary": self.capability_summary,
            "gap_type": self.gap_type,
            "risk_level": self.risk_level,
            "status": self.status,
            "created_at": self.created_at,
            "evidence": [item.to_dict() for item in self.evidence],
            "proposed_tools": [item.to_dict() for item in self.proposed_tools],
            "test_cases": [item.to_dict() for item in self.test_cases],
        }
