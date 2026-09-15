# Copyright (c) Alibaba, Inc. and its affiliates.
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
from leapflow.domain.evolution_intent import WORLD_MODEL_INTENT
from leapflow.domain.environment_fingerprint import EnvironmentFingerprint
from leapflow.learning.capability_gap_detector import CapabilityGapDetector

# The evidence origin the observation layer has always accepted. Kept as the
# default so behaviour is unchanged unless a classifier is explicitly supplied.
DEFAULT_ACCEPTED_EVIDENCE = frozenset({"unknown_tool"})

#: Evidence that an *existing* provider of a capability is performing badly. Emitted
#: by lifecycle governance when a failure leaves the plugin still in service, so it
#: reports "what serves this capability is inadequate" rather than "nothing serves
#: it". Opt-in like every non-default kind.
CAPABILITY_DEGRADED = "capability_degraded"

#: Evidence kinds a successful resolution does **not** retire.
#:
#: Retirement means "the gap this evidence reported is closed". For ``unknown_tool``
#: that is exactly what a provider existing proves. For degradation it proves nothing:
#: the provider that exists is the thing being reported. Without this distinction a
#: degradation observation is retired on the very next turn -- resolution finds the
#: incumbent, calls the capability satisfied, and erases the record of it failing.
EVIDENCE_SURVIVING_RESOLUTION = frozenset({CAPABILITY_DEGRADED})

#: Failure classes the retry layer already owns, so degradation evidence carrying one
#: never reaches the teacher.
#:
#: A timeout or a dropped connection says nothing about whether the implementation is
#: right for this environment -- ``RecoveryAction.RETRY_WITH_BACKOFF`` handles it inside
#: the turn. Forwarding it anyway would ask a hindsight evaluator to adjudicate a
#: transient, and the only answer it could give that changes anything is "rebuild",
#: which is the most expensive response in the system applied to a problem that already
#: resolved itself.
#: Every member must be a class some classifier actually emits, and a test asserts it.
#: ``"rate_limit"`` was in here with no producer anywhere -- harmless, but it claimed to
#: filter something never seen, which is how a set like this stops being readable as a
#: statement about the system.
RETRY_OWNED_FAILURE_CLASSES = frozenset({"timeout", "connection_error", "transient"})


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
        """Build from ``evolution_enabled``, widened by ``accepted_evidence_kinds``.

        One switch, because two were one too many. ``accepted_evidence_kinds`` is a tuple of
        internal kind names surfaced under the key ``accepted.evidence_kinds`` -- a section
        that names nothing -- and a user who turned self-evolution on and then found nothing
        happened would have no way to guess that a second, differently-named setting also
        had to list ``world_model_intent``. So the switch admits it, and the tuple remains
        for the finer-grained case: structural kinds like ``interface_drift`` come from an
        environment probe rather than the world model and are opted into separately.

        Admission is a *trigger*, never a permission. Every admitted kind still traverses
        resolution, risk, approval, validation and trust unchanged.
        """
        kinds = tuple(getattr(settings, "accepted_evidence_kinds", None) or ())
        if getattr(settings, "evolution_enabled", False):
            # Widen, never replace. ``from_kinds`` treats a non-empty tuple as the whole
            # accepted set, so appending alone would have *dropped* ``unknown_tool`` --
            # turning self-evolution on would have silently disabled the trigger that was
            # already working, and the chain would have looked more capable while covering
            # less.
            kinds = (*DEFAULT_ACCEPTED_EVIDENCE, *kinds, WORLD_MODEL_INTENT)
        return cls.from_kinds(kinds)

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

    def degraded_capabilities(self, *, limit: int = 50) -> tuple[dict[str, Any], ...]:
        """The still-serving providers that are failing, as facts for the teacher.

        One reader for this evidence, so the two rules that make it usable live in one
        place instead of being re-derived by each consumer:

        * **Retry-owned failures are excluded.** A timeout says nothing about whether
          the implementation fits the environment, and the only verdict a hindsight
          evaluator could give that changes anything is "rebuild" -- the most expensive
          response in the system, applied to something that already resolved itself.
        * **The environment fingerprint travels with each fact.** One application
          upgrade breaks every capability bound to the old affordances, and without the
          fingerprint those arrive as N unrelated degradations. The teacher would then
          answer N times and could propose N rebuilds where the truth is one root cause
          and, usually, one rebind.

        Returns plain dicts rather than a domain type because this is a projection for a
        prompt, not a decision: nothing downstream should be able to act on it directly.
        """
        facts: list[dict[str, Any]] = []
        for record in self._store.unresolved(min_count=1, limit=limit):
            result = record.get("result") or {}
            if str(result.get("error_type") or "") != CAPABILITY_DEGRADED:
                continue
            if str(result.get("failure_class") or "") in RETRY_OWNED_FAILURE_CLASSES:
                continue
            capability = str(result.get("capability") or "").strip()
            if not capability:
                continue
            facts.append(
                {
                    "capability": capability,
                    "plugin_id": str(result.get("plugin_id") or ""),
                    "failure_streak": int(result.get("failure_streak") or 0),
                    "failure_class": str(result.get("failure_class") or ""),
                    "trust_level": str(result.get("trust_level") or ""),
                    # The environment the failures were seen in, so a compound change
                    # is recognisable as one transition rather than N coincidences.
                    "environment": dict(record.get("environment") or {}),
                }
            )
        return tuple(facts)

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

        Evidence in :data:`EVIDENCE_SURVIVING_RESOLUTION` is skipped: it reports that
        the *existing* provider is inadequate, so finding that provider does not
        address it. Retiring it here would delete the degradation record at the first
        resolution after it was written.
        """
        target = str(capability or "").strip()
        if not target:
            return ()
        retired: list[str] = []
        for record in self._store.unresolved(min_count=1, limit=limit):
            observation_id = str(record.get("observation_id") or "")
            if not observation_id:
                continue
            result = record.get("result") or {}
            if str(result.get("error_type") or "") in EVIDENCE_SURVIVING_RESOLUTION:
                continue
            derived = self._detector.requirements_from_tool_results([result], min_count=1)
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
    "CAPABILITY_DEGRADED",
    "DEFAULT_ACCEPTED_EVIDENCE",
    "EVIDENCE_SURVIVING_RESOLUTION",
]
