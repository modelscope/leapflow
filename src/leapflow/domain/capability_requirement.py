"""Domain records for adaptive capability requirements.

A requirement describes what LeapFlow needs, not which concrete tool should be
used. It is intentionally immutable and side-effect free so it can be persisted,
shown to users, and fed into deterministic plugin resolution without coupling the
domain layer to the plugin registry or the engine loop.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from leapflow.domain.plugin_proposal import RiskLevel

RequirementOrigin = Literal[
    "unknown_tool",
    "explicit_request",
    "environment_probe",
    "task_contract",
    "world_model",
]
ApprovalMode = Literal["review_required", "autonomous_allowed"]


def _freeze_strs(values: tuple[str, ...] | list[str] | set[str] | str | None = None) -> tuple[str, ...]:
    """Normalize a string sequence into a stable tuple."""
    if not values:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(str(v) for v in values if str(v))


def _freeze_metadata(metadata: dict[str, Any] | None = None) -> tuple[tuple[str, str], ...]:
    """Convert arbitrary metadata to a stable immutable string map."""
    if not metadata:
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in metadata.items()))


@dataclass(frozen=True)
class CapabilityRequirement:
    """One structured capability need derived from runtime evidence."""

    requirement_id: str
    capability: str
    origin: RequirementOrigin
    evidence: str = ""
    required_platform_capabilities: tuple[str, ...] = field(default_factory=tuple)
    max_risk_level: RiskLevel = "external"
    approval_mode: ApprovalMode = "review_required"
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @classmethod
    def create(
        cls,
        capability: str,
        origin: RequirementOrigin,
        *,
        evidence: str = "",
        required_platform_capabilities: tuple[str, ...] | list[str] | set[str] | None = None,
        max_risk_level: RiskLevel = "external",
        approval_mode: ApprovalMode = "review_required",
        metadata: dict[str, Any] | None = None,
        requirement_id: str = "",
    ) -> "CapabilityRequirement":
        """Build a normalized requirement from structured evidence."""
        normalized = str(capability or "").strip()
        if not normalized:
            raise ValueError("capability is required")
        return cls(
            requirement_id=requirement_id or f"req-{uuid.uuid4().hex}",
            capability=normalized,
            origin=origin,
            evidence=str(evidence or ""),
            required_platform_capabilities=_freeze_strs(required_platform_capabilities),
            max_risk_level=max_risk_level,
            approval_mode=approval_mode,
            metadata=_freeze_metadata(metadata),
        )

    @property
    def allows_autonomous_approval(self) -> bool:
        """Return whether trusted governance may collapse repeated approvals.

        This does not install or execute anything automatically. It is only a
        declarative signal for later approval orchestration.
        """
        return self.approval_mode == "autonomous_allowed"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "requirement_id": self.requirement_id,
            "capability": self.capability,
            "origin": self.origin,
            "evidence": self.evidence,
            "required_platform_capabilities": list(self.required_platform_capabilities),
            "max_risk_level": self.max_risk_level,
            "approval_mode": self.approval_mode,
            "metadata": dict(self.metadata),
        }
