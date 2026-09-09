"""Structured capability observations for adaptive plugin evolution.

The observation layer is intentionally side-effect free. It accepts structured
runtime evidence (currently unknown-tool results) and turns it into capability
requirements that a separate governance loop may review, plan, and mutate from.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.domain.environment_fingerprint import EnvironmentFingerprint
from leapflow.learning.capability_gap_detector import CapabilityGapDetector

# The evidence origin the observation layer has always accepted. Kept as the
# default so behaviour is unchanged unless a classifier is explicitly supplied.
DEFAULT_ACCEPTED_EVIDENCE = frozenset({"unknown_tool"})


@dataclass(frozen=True)
class CapabilityEvidenceClassifier:
    """Decide whether a structured tool result is capability-relevant evidence.

    The shipped observation layer hard-codes ``error_type == "unknown_tool"``,
    which is blind to a structural environment change under a still-present tool.
    This classifier makes the accepted ``error_type`` set explicit and
    configurable so an environment-aware source (interface-drift / affordance-loss
    signals) or the world-model teacher (``world_model_intent``) can feed the same
    governed pipeline, while the default set preserves today's behaviour exactly.

    The accepted set is driven by the ``accepted_evidence_kinds`` setting (see
    :meth:`from_settings`); it is never inferred from natural-language text.
    Widening it adds a *trigger*, never a permission: every admitted kind still
    traverses resolution, risk classification, approval, validation, and trust.
    """

    accepted: frozenset[str] = DEFAULT_ACCEPTED_EVIDENCE

    @classmethod
    def from_kinds(cls, kinds: Iterable[str] | None = None) -> "CapabilityEvidenceClassifier":
        """Build from an iterable of accepted error kinds (None -> default)."""
        if not kinds:
            return cls()
        return cls(accepted=frozenset(str(kind) for kind in kinds if str(kind)))

    @classmethod
    def from_settings(cls, settings: Any) -> "CapabilityEvidenceClassifier":
        """Build from a Settings-like object's ``accepted_evidence_kinds``.

        Returns the default (``unknown_tool`` only) when the setting is absent or
        empty, so an operator must opt in before any new trigger becomes live.
        """
        return cls.from_kinds(getattr(settings, "accepted_evidence_kinds", None))

    def accepts(self, result: Mapping[str, Any] | None) -> bool:
        return isinstance(result, Mapping) and str(result.get("error_type") or "") in self.accepted


@dataclass(frozen=True)
class CapabilityObservation:
    """One structured runtime signal relevant to plugin adaptation."""

    observed_at: float
    result: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"observed_at": self.observed_at, "result": dict(self.result)}


@dataclass
class CapabilityObservationBuffer:
    """Collect structured tool evidence and derive reviewable requirements."""

    detector: CapabilityGapDetector = field(default_factory=CapabilityGapDetector)
    # Optional evidence gate. ``None`` preserves the shipped behaviour (accept
    # only ``unknown_tool``); an explicit classifier widens the accepted set.
    classifier: CapabilityEvidenceClassifier | None = None
    _observations: list[CapabilityObservation] = field(default_factory=list)

    def add_result(self, result: Mapping[str, Any] | None) -> bool:
        """Record a structured tool result when it represents a capability gap."""
        if not self._accepts(result):
            return False
        self._observations.append(
            CapabilityObservation(observed_at=time.time(), result=dict(result or {}))
        )
        return True

    def _accepts(self, result: Mapping[str, Any] | None) -> bool:
        if self.classifier is not None:
            return self.classifier.accepts(result)
        return self._is_supported_signal(result)

    def extend_results(self, results: Sequence[Mapping[str, Any]]) -> int:
        """Record multiple tool results and return how many were accepted."""
        return sum(1 for result in results if self.add_result(result))

    def requirements(self, *, min_count: int = 1) -> tuple[CapabilityRequirement, ...]:
        """Return requirements derived from buffered structured evidence."""
        return self.detector.requirements_from_tool_results(
            tuple(observation.result for observation in self._observations),
            min_count=min_count,
        )

    def observations(self) -> tuple[CapabilityObservation, ...]:
        """Return an immutable snapshot of collected observations."""
        return tuple(self._observations)

    def clear(self) -> None:
        """Drop all buffered observations."""
        self._observations.clear()

    @staticmethod
    def _is_supported_signal(result: Mapping[str, Any] | None) -> bool:
        return isinstance(result, Mapping) and result.get("error_type") == "unknown_tool"


class CapabilityObservationService:
    """Bridge turn-local observations into durable, cross-turn requirements."""

    def __init__(
        self,
        store: Any,
        *,
        detector: CapabilityGapDetector | None = None,
        classifier: CapabilityEvidenceClassifier | None = None,
    ) -> None:
        self._store = store
        self._detector = detector or CapabilityGapDetector()
        # ``None`` preserves the shipped ``unknown_tool``-only gate; an explicit
        # classifier lets environment-derived evidence reach the durable store.
        self._classifier = classifier

    def _accepts(self, result: Mapping[str, Any] | None) -> bool:
        if self._classifier is not None:
            return self._classifier.accepts(result)
        return CapabilityObservationBuffer._is_supported_signal(result)

    def observe_result(
        self,
        result: Mapping[str, Any] | None,
        *,
        environment: EnvironmentFingerprint | Mapping[str, Any] | None = None,
        source: str = "runtime",
        session_id: str = "",
        turn_id: str = "",
        workspace_root: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Persist one structured observation, returning the stored record."""
        if not self._accepts(result):
            return None
        env_payload = (
            environment.to_dict()
            if isinstance(environment, EnvironmentFingerprint)
            else dict(environment or {})
        )
        return self._store.add_observation(
            result=dict(result or {}),
            environment=env_payload,
            source=source,
            session_id=session_id,
            turn_id=turn_id,
            workspace_root=workspace_root,
            metadata=dict(metadata or {}),
        )

    def flush_buffer(
        self,
        buffer: CapabilityObservationBuffer,
        *,
        environment: EnvironmentFingerprint | Mapping[str, Any] | None = None,
        source: str = "runtime",
        session_id: str = "",
        turn_id: str = "",
        workspace_root: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Persist every observation in a turn-local buffer."""
        records: list[dict[str, Any]] = []
        for observation in buffer.observations():
            record = self.observe_result(
                observation.result,
                environment=environment,
                source=source,
                session_id=session_id,
                turn_id=turn_id,
                workspace_root=workspace_root,
                metadata=metadata,
            )
            if record is not None:
                records.append(record)
        return tuple(records)

    def requirements(
        self, *, min_count: int = 1, limit: int = 50
    ) -> tuple[CapabilityRequirement, ...]:
        """Aggregate durable observations into reviewable requirements."""
        results = [
            record.get("result") or {}
            for record in self._store.unresolved(min_count=min_count, limit=limit)
        ]
        return self._detector.requirements_from_tool_results(results, min_count=1)

    def resolve_capability(
        self, capability: str, *, reason: str = "", limit: int = 50
    ) -> tuple[str, ...]:
        """Retire observations whose capability gap is now satisfied.

        Without this the observation lifecycle is write-only: ``unresolved()``
        filters on ``status == "open"``, so evidence that motivated a capability
        which now resolves keeps being reported, and any consumer sizing work from
        it would re-propose capabilities the system already has.

        Matching is done by running the same detector used to derive requirements,
        so an observation is retired only when it genuinely maps to the resolved
        capability -- never by string-matching the raw payload. Returns the ids of
        the observations retired.
        """
        target = str(capability or "").strip()
        if not target:
            return ()
        retired: list[str] = []
        for record in self._store.unresolved(min_count=1, limit=limit):
            observation_id = str(record.get("observation_id") or "")
            if not observation_id:
                continue
            derived = self._detector.requirements_from_tool_results(
                [record.get("result") or {}], min_count=1
            )
            if any(requirement.capability == target for requirement in derived):
                if self._store.mark_status(
                    observation_id, "resolved", reason=reason or f"{target} resolved"
                ):
                    retired.append(observation_id)
        return tuple(retired)


__all__ = [
    "CapabilityEvidenceClassifier",
    "CapabilityObservation",
    "CapabilityObservationBuffer",
    "CapabilityObservationService",
    "DEFAULT_ACCEPTED_EVIDENCE",
]
