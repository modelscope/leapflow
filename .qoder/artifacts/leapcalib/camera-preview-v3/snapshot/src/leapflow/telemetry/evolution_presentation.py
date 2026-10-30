# Copyright (c) Alibaba, Inc. and its affiliates.
"""Read-only presentation projection for one runtime evolution trace.

The projection is intentionally smaller than :class:`EvolutionTrace`: it carries
only the stable identifiers and labels a live display needs. It is created after a
trace has been buffered, then published from the daemon event loop. It never
persists, approves, selects, or mutates anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from leapflow.domain.evolution_trace import EvolutionTrace


@dataclass(frozen=True)
class EvolutionPresentationEvent:
    """One safe, display-oriented view of a runtime evolution fact."""

    event_id: str
    episode_id: str
    stage: str
    kind: str
    ts: float
    correlation: Mapping[str, str]
    payload_ref: Mapping[str, str]
    summary: str
    evidence_level: str
    verification_tier: str
    side_effect_state: str

    @classmethod
    def from_trace(cls, trace: EvolutionTrace) -> "EvolutionPresentationEvent":
        """Project a trace without exposing its unbounded domain-private detail."""
        correlation = {str(key): str(value) for key, value in trace.correlation.items() if str(value)}
        detail = dict(trace.detail)
        episode_id = (
            correlation.get("record_id")
            or correlation.get("intent_id")
            or correlation.get("requirement_id")
            or correlation.get("observation_id")
            or correlation.get("lifecycle_proposal_id")
            or trace.trace_id
        )
        payload_ref = {
            key: value
            for key, value in correlation.items()
            if key in {
                "record_id",
                "intent_id",
                "requirement_id",
                "observation_id",
                "lifecycle_proposal_id",
                "plugin_id",
                "registry_version",
            }
        }
        return cls(
            event_id=trace.trace_id,
            episode_id=episode_id,
            stage=trace.stage.value,
            kind=trace.kind,
            ts=trace.ts,
            correlation=correlation,
            payload_ref=payload_ref,
            summary=trace.summary,
            evidence_level=str(detail.get("evidence_level") or "runtime_trace"),
            verification_tier=str(detail.get("verification_tier") or "not_recorded"),
            side_effect_state=str(detail.get("side_effect_state") or "not_recorded"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe event shape delivered to display clients."""
        return {
            "event_id": self.event_id,
            "episode_id": self.episode_id,
            "stage": self.stage,
            "kind": self.kind,
            "ts": self.ts,
            "correlation": dict(self.correlation),
            "payload_ref": dict(self.payload_ref),
            "summary": self.summary,
            "evidence_level": self.evidence_level,
            "verification_tier": self.verification_tier,
            "side_effect_state": self.side_effect_state,
        }


__all__ = ["EvolutionPresentationEvent"]
