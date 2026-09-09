"""The world model's evolution proposal contract.

An ``EvolutionIntent`` is what the LLM-based world model emits when, given
privileged hindsight (goal + full trajectory + actual effects), it concludes that
LeapFlow lacks a capability it should have. It is the intended *first driver* of
capability self-evolution.

Three properties make it safe to let a language model author these:

* It is a **hypothesis, not an authorisation.** An intent carries no permission.
  It converts to an ordinary :class:`CapabilityRequirement` and then traverses the
  unchanged deterministic chain -- declared-fitness resolution, risk
  classification, approval, artifact validation, trust. The world model decides
  *what* to evolve and *why*; those components decide whether it is permitted and
  whether it worked.
* It is **declaration-driven.** Capability, target, and risk ceiling are explicit
  fields, never parsed out of prose, so no free-text inference reaches the
  governed pipeline.
* It is **evidence-linked.** ``evidence_ids`` ties the intent back to the
  observations and experiences that motivated it, so an intent can be audited,
  replayed, and (once acted on) retired.

The intent deliberately reuses the existing observation path rather than adding a
parallel one: :meth:`to_observation_result` renders the payload shape
``CapabilityObservationService``/``CapabilityGapDetector`` already consume, so a
world-model intent is governed by exactly the same machinery as an
``unknown_tool`` signal.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.domain.plugin_proposal import RiskLevel

#: Evidence kind carried by a world-model intent. Admit it through
#: ``CapabilityEvidenceClassifier`` to let the world model drive evolution; it is
#: intentionally absent from ``DEFAULT_ACCEPTED_EVIDENCE`` so shipped behaviour is
#: unchanged until an operator opts in.
WORLD_MODEL_INTENT = "world_model_intent"

#: Requirement origin recorded for anything derived from an intent.
WORLD_MODEL_ORIGIN = "world_model"

#: Risk ceiling applied to anything a model authored, unless a trusted caller
#: raises it explicitly. ``max_risk_level`` on a requirement is a *ceiling*, so a
#: larger value is more permissive -- which makes it partly an authorisation, not
#: merely a description. An intent may therefore only ever *narrow* the ceiling:
#: the effective value is the stricter of what the intent asked for and what the
#: trusted caller allows.
MODEL_AUTHORED_RISK_CEILING: RiskLevel = "read_only"

# Ascending permissiveness, matching ``RiskLevel``.
_RISK_ORDER: tuple[str, ...] = ("read_only", "low", "medium", "high", "mutating", "external")


def _risk_rank(level: str) -> int:
    """Rank a risk level, treating anything unknown as the most permissive.

    An unrecognised value must not read as *safe*, or a typo would silently widen
    the ceiling; ranking it highest means the clamp below always rejects it.
    """
    try:
        return _RISK_ORDER.index(str(level))
    except ValueError:
        return len(_RISK_ORDER) - 1


def _stricter(left: str, right: str) -> str:
    """Return whichever risk ceiling is more restrictive."""
    return left if _risk_rank(left) <= _risk_rank(right) else right


def _freeze(values: Any) -> tuple[str, ...]:
    if not values:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(str(v) for v in values if str(v))


@dataclass(frozen=True)
class EvolutionIntent:
    """One world-model hypothesis that a capability is missing or broken.

    ``confidence`` is the model's own calibration and is carried through to the
    requirement's metadata; it informs prioritisation and audit but must never be
    read as permission -- a high-confidence intent still passes every gate.
    """

    intent_id: str
    capability: str
    hypothesis: str
    confidence: float = 0.0
    target_affordance: str = ""
    rationale: str = ""
    expected_effect: str = ""
    max_risk_level: RiskLevel = "read_only"
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    required_platform_capabilities: tuple[str, ...] = field(default_factory=tuple)
    created_at: float = 0.0

    @classmethod
    def create(
        cls,
        capability: str,
        hypothesis: str,
        *,
        confidence: float = 0.0,
        target_affordance: str = "",
        rationale: str = "",
        expected_effect: str = "",
        max_risk_level: RiskLevel = "read_only",
        evidence_ids: Any = None,
        required_platform_capabilities: Any = None,
        intent_id: str = "",
        created_at: float | None = None,
    ) -> "EvolutionIntent":
        """Build a normalized intent.

        ``max_risk_level`` defaults to ``read_only``: a model-authored proposal
        starts at the lowest ceiling and must be widened deliberately, rather than
        inheriting the permissive domain default.
        """
        normalized = str(capability or "").strip()
        if not normalized:
            raise ValueError("capability is required")
        statement = str(hypothesis or "").strip()
        if not statement:
            raise ValueError("hypothesis is required")
        return cls(
            intent_id=intent_id or f"wmi-{uuid.uuid4().hex}",
            capability=normalized,
            hypothesis=statement,
            confidence=max(0.0, min(1.0, float(confidence))),
            target_affordance=str(target_affordance or ""),
            rationale=str(rationale or ""),
            expected_effect=str(expected_effect or ""),
            max_risk_level=max_risk_level,
            evidence_ids=_freeze(evidence_ids),
            required_platform_capabilities=_freeze(required_platform_capabilities),
            created_at=time.time() if created_at is None else float(created_at),
        )

    def effective_risk_ceiling(
        self, risk_ceiling: RiskLevel = MODEL_AUTHORED_RISK_CEILING
    ) -> str:
        """The ceiling actually applied: the stricter of the intent's and the caller's."""
        return _stricter(str(self.max_risk_level), str(risk_ceiling))

    def to_observation_result(
        self, *, risk_ceiling: RiskLevel = MODEL_AUTHORED_RISK_CEILING
    ) -> dict[str, Any]:
        """Render the payload the observation/detector path already consumes.

        Using the same shape as other structured evidence is what keeps the world
        model on the governed path instead of beside it. The emitted
        ``max_risk_level`` is clamped by ``risk_ceiling`` so the payload cannot
        widen its own permissions downstream.
        """
        effective = self.effective_risk_ceiling(risk_ceiling)
        payload: dict[str, Any] = {
            "error_type": WORLD_MODEL_INTENT,
            "origin": WORLD_MODEL_ORIGIN,
            "capability": self.capability,
            "evidence": self.hypothesis,
            "failure_code": "world_model_capability_gap",
            "recovery_hint": self.rationale or self.hypothesis,
            "confidence": self.confidence,
            "intent_id": self.intent_id,
            "max_risk_level": effective,
            "requirement_id": f"req-wm-{self.intent_id}",
            "suggestions": list(self.evidence_ids),
            "required_platform_capabilities": list(self.required_platform_capabilities),
            "target_affordance": self.target_affordance,
            "expected_effect": self.expected_effect,
        }
        if effective != str(self.max_risk_level):
            # Keep the model's request visible for audit even though it was denied.
            payload["requested_max_risk_level"] = str(self.max_risk_level)
        return payload

    def to_requirement(
        self, *, risk_ceiling: RiskLevel = MODEL_AUTHORED_RISK_CEILING
    ) -> CapabilityRequirement:
        """Convert directly to a requirement, bypassing the durable store.

        Prefer routing through ``CapabilityObservationService.observe_result`` so
        the intent is persisted and auditable; this direct conversion exists for
        callers that already hold the evidence trail. The risk ceiling is clamped
        exactly as in :meth:`to_observation_result`.
        """
        effective = self.effective_risk_ceiling(risk_ceiling)
        metadata: dict[str, Any] = {
            "evidence_kind": WORLD_MODEL_INTENT,
            "intent_id": self.intent_id,
            "confidence": self.confidence,
        }
        if effective != str(self.max_risk_level):
            metadata["requested_max_risk_level"] = str(self.max_risk_level)
        if self.target_affordance:
            metadata["target_affordance"] = self.target_affordance
        if self.expected_effect:
            metadata["expected_effect"] = self.expected_effect
        if self.evidence_ids:
            metadata["evidence_ids"] = ",".join(self.evidence_ids)
        return CapabilityRequirement.create(
            self.capability,
            WORLD_MODEL_ORIGIN,
            evidence=self.hypothesis,
            required_platform_capabilities=list(self.required_platform_capabilities),
            max_risk_level=effective,  # type: ignore[arg-type]
            metadata=metadata,
            requirement_id=f"req-wm-{self.intent_id}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "capability": self.capability,
            "hypothesis": self.hypothesis,
            "confidence": self.confidence,
            "target_affordance": self.target_affordance,
            "rationale": self.rationale,
            "expected_effect": self.expected_effect,
            "max_risk_level": self.max_risk_level,
            "evidence_ids": list(self.evidence_ids),
            "required_platform_capabilities": list(self.required_platform_capabilities),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvolutionIntent":
        return cls.create(
            str(payload.get("capability") or ""),
            str(payload.get("hypothesis") or ""),
            confidence=float(payload.get("confidence") or 0.0),
            target_affordance=str(payload.get("target_affordance") or ""),
            rationale=str(payload.get("rationale") or ""),
            expected_effect=str(payload.get("expected_effect") or ""),
            max_risk_level=payload.get("max_risk_level") or "read_only",
            evidence_ids=payload.get("evidence_ids"),
            required_platform_capabilities=payload.get("required_platform_capabilities"),
            intent_id=str(payload.get("intent_id") or ""),
            created_at=payload.get("created_at"),
        )


__all__ = [
    "MODEL_AUTHORED_RISK_CEILING",
    "WORLD_MODEL_INTENT",
    "WORLD_MODEL_ORIGIN",
    "EvolutionIntent",
]
