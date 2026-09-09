"""Verify an acquired capability by its effect, and reclaim what never works.

Two gaps this closes, both recorded by the EVO-02 experiments:

**Validation is not fitness (v0.5).** ``PluginValidator`` proves an artifact is
*conformant* -- it parses, imports, satisfies the Protocol, declares valid
metadata. It cannot prove the artifact *works*. Re-resolution afterwards only
proves *declared* fitness: a candidate says it provides the capability and its
declared affordances are present. A structurally perfect adapter targeting the
wrong thing passes both and is still useless.

**Retirement rode on declared fitness (v0.7).** Because the engine retires
observations whenever a resolution reports the requirement met, a wrongly-selected
tool retired the evidence for its own gap.

The fix is to make the *observed effect* the arbiter. This module is deliberately
deterministic: the world model decides what to evolve and why; whether it worked is
decided by comparing a declared expectation against an observed outcome. An
``EffectVerdict`` is designed to be fed straight into
``LifecycleGovernor.record_outcome``, so verification reuses the existing trust,
probation and quarantine machinery instead of adding a parallel one -- which is
also what makes reclamation fall out for free: an artifact that keeps failing
verification is quarantined and unregistered by the governor.

The residual case the governor cannot reach is an artifact that produces *no*
outcomes at all because nothing ever selects it (the over-risk artifact the EVO-02
episode installed and then refused). :class:`UnselectableArtifactReaper` handles
exactly that, conservatively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from leapflow.domain.capability_requirement import CapabilityRequirement

logger = logging.getLogger(__name__)

#: Result keys a tool may use to report what it observably did. This is the whole
#: declaration channel for confirmation: without one of these, a *successful* call is
#: ``unverifiable`` (we do not know whether the effect landed) while a *failed* call
#: still refutes. Several spellings are accepted because the convention post-dates
#: existing tools, and a tool that already says ``observed_effect`` should not have to
#: be rewritten to be verifiable.
OBSERVED_EFFECT_KEYS: tuple[str, ...] = ("observed_effect", "effect")

#: Reasons a verification can fail, kept as constants so callers can branch on
#: them without matching prose.
NO_OUTCOME = "no_outcome_observed"
EXECUTION_FAILED = "execution_failed"
EFFECT_ABSENT = "expected_effect_absent"
EFFECT_UNREPORTED = "tool_reported_no_effect"
VERIFIED = "effect_observed"
UNVERIFIABLE = "no_expected_effect_declared"


def observed_effect_from_result(result: Any) -> str:
    """Extract a tool's self-reported observed effect, if it declared one.

    Returns ``""`` when the tool said nothing, which the verifier reads as
    *unverifiable* rather than as failure. Deliberately does not synthesise prose from
    the rest of the payload: an invented description would be compared against the
    teacher's expectation and could confirm an acquisition that never worked.
    """
    if not isinstance(result, Mapping):
        return ""
    for key in OBSERVED_EFFECT_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


@dataclass(frozen=True)
class EffectVerdict:
    """Whether an acquired capability demonstrably did what it promised.

    ``verified is None`` means *unverifiable*, which is deliberately distinct from
    ``False``. Two situations produce it: the requirement declared no expected effect
    (``UNVERIFIABLE``), or the tool succeeded without reporting what it did
    (``EFFECT_UNREPORTED``). Neither is a failure, and treating either as one would
    quarantine healthy plugins over missing metadata.
    """

    plugin_id: str
    capability: str
    verified: bool | None
    reason: str
    expected_effect: str = ""
    observed_effect: str = ""

    @property
    def should_record_outcome(self) -> bool:
        """Only a decided verdict may drive trust or quarantine."""
        return self.verified is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "capability": self.capability,
            "verified": self.verified,
            "reason": self.reason,
            "expected_effect": self.expected_effect,
            "observed_effect": self.observed_effect,
        }


class CapabilityEffectVerifier:
    """Decide whether an acquired capability actually produced its effect.

    Deterministic by design. ``expected_effect`` is a declaration carried on the
    requirement (a world-model intent supplies it); the observed outcome comes from
    executing the tool. Matching is a containment test over normalized tokens, not
    a semantic judgement -- a semantic comparison belongs to the world model, and
    putting it here would let a model both propose a capability and certify its own
    work.
    """

    def __init__(self, *, min_token_overlap: int = 1) -> None:
        self._min_overlap = max(1, int(min_token_overlap))

    def verify(
        self,
        requirement: CapabilityRequirement,
        outcome: Mapping[str, Any] | None,
        *,
        plugin_id: str = "",
    ) -> EffectVerdict:
        """Compare the requirement's declared expectation against one outcome."""
        expected = str(dict(requirement.metadata).get("expected_effect") or "")
        capability = requirement.capability
        if outcome is None:
            return EffectVerdict(plugin_id, capability, None, NO_OUTCOME, expected)
        if not outcome.get("ok", False):
            # An execution failure is a decided negative regardless of what was
            # expected: the capability did not deliver.
            return EffectVerdict(
                plugin_id, capability, False, EXECUTION_FAILED, expected,
                observed_effect_from_result(outcome),
            )
        if not expected:
            return EffectVerdict(plugin_id, capability, None, UNVERIFIABLE, expected)

        observed = observed_effect_from_result(outcome)
        if not observed:
            # Absence of evidence, not evidence of absence. The call succeeded and the
            # tool simply said nothing about what it did, which is the normal state for
            # every handler written before the effect convention existed. Refuting here
            # would demote and eventually quarantine healthy plugins for a reporting
            # omission -- the exact failure the three-valued verdict exists to prevent.
            return EffectVerdict(
                plugin_id, capability, None, EFFECT_UNREPORTED, expected
            )
        if self._matches(expected, observed):
            return EffectVerdict(plugin_id, capability, True, VERIFIED, expected, observed)
        # The tool did report an effect and it is not the one that was expected: a
        # decided negative, and the case that catches a conformant adapter pointed at
        # the wrong thing.
        return EffectVerdict(plugin_id, capability, False, EFFECT_ABSENT, expected, observed)

    def _matches(self, expected: str, observed: str) -> bool:
        """Token-overlap containment. Empty observation never counts as a match."""
        expected_tokens = _tokens(expected)
        observed_tokens = _tokens(observed)
        if not expected_tokens or not observed_tokens:
            return False
        return len(expected_tokens & observed_tokens) >= min(
            self._min_overlap, len(expected_tokens)
        )


_STOPWORDS = frozenset(
    {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "of", "and", "it"}
)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
        if token and token not in _STOPWORDS
    )


#: Exclusions that will not be lifted by a change in the environment. An
#: environment-driven exclusion (a missing affordance) can become satisfiable when the
#: app is upgraded; a risk ceiling will not. Matched against the *scorer name* on the
#: excluded score component -- never against its prose, which is free to be reworded.
DURABLE_EXCLUSIONS: tuple[str, ...] = ("risk_cost",)


@dataclass(frozen=True)
class ReclamationCandidate:
    """A self-acquired artifact that has never been usable."""

    plugin_id: str
    reason: str
    resolutions_seen: int
    risk_excluded: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "reason": self.reason,
            "resolutions_seen": self.resolutions_seen,
            "risk_excluded": self.risk_excluded,
        }


class UnselectableArtifactReaper:
    """Find self-acquired artifacts that no admissible requirement can select.

    The case the governor cannot reach: an artifact that produces no outcomes
    because nothing ever selects it, so it accrues neither trust nor failures and
    is never quarantined. The EVO-02 episode installed exactly one -- an over-risk
    adapter that stayed registered for the rest of the run, holding a tool name it
    could never use.

    The predicate is deliberately conservative, because "unselectable" is only ever
    true *relative to the requirements seen so far*:

    * the plugin was installed by self-evolution (the caller supplies that set --
      a hand-installed plugin is never reaped);
    * it was **never** selected across every observed resolution;
    * every exclusion was a **durable** exclusion -- by default a risk-cap exclusion.
      An environment can change and make a tool viable again; a risk ceiling will not
      be lifted by the environment, so risk exclusion is the durable kind. Matching is
      by the excluded component's **scorer name** (``risk_cost``), not by its prose:
      keying off a human-readable reason would silently stop working the moment the
      resolver rewords it;
    * at least ``min_resolutions`` resolutions were observed, so a single unlucky
      requirement cannot condemn an artifact.

    The reaper only *reports*. Disabling or removing a plugin is a governed
    mutation and stays with the lifecycle actor.
    """

    def __init__(
        self,
        *,
        min_resolutions: int = 3,
        durable_exclusions: Sequence[str] = DURABLE_EXCLUSIONS,
    ) -> None:
        self._min_resolutions = max(1, int(min_resolutions))
        self._durable = frozenset(str(name) for name in durable_exclusions if str(name))

    def candidates(
        self,
        *,
        acquired_plugin_ids: Sequence[str],
        resolutions: Sequence[Mapping[str, Any]],
    ) -> tuple[ReclamationCandidate, ...]:
        """Return artifacts that every observed resolution refused on risk grounds.

        ``resolutions`` entries are expected to expose ``selected_plugin`` and a
        per-candidate ``exclusions`` mapping of ``plugin_id -> excluded scorer names``.
        """
        acquired = {str(pid) for pid in acquired_plugin_ids if str(pid)}
        if not acquired or len(resolutions) < self._min_resolutions:
            return ()

        seen: dict[str, int] = {pid: 0 for pid in acquired}
        risk_excluded: dict[str, int] = {pid: 0 for pid in acquired}
        selected_ever: set[str] = set()

        for resolution in resolutions:
            selected = str(resolution.get("selected_plugin") or "")
            if selected in acquired:
                selected_ever.add(selected)
            exclusions = dict(resolution.get("exclusions") or {})
            for plugin_id in acquired:
                reasons = exclusions.get(plugin_id)
                if reasons is None:
                    continue
                seen[plugin_id] += 1
                if any(str(reason) in self._durable for reason in reasons):
                    risk_excluded[plugin_id] += 1

        found: list[ReclamationCandidate] = []
        for plugin_id in sorted(acquired):
            if plugin_id in selected_ever:
                continue
            observed = seen[plugin_id]
            if observed < self._min_resolutions:
                continue
            # Every single exclusion must be the durable (risk) kind.
            if risk_excluded[plugin_id] != observed:
                continue
            found.append(
                ReclamationCandidate(
                    plugin_id=plugin_id,
                    reason="never selected; durably excluded in every observed resolution",
                    resolutions_seen=observed,
                    risk_excluded=risk_excluded[plugin_id],
                )
            )
        return tuple(found)


__all__ = [
    "DURABLE_EXCLUSIONS",
    "EFFECT_ABSENT",
    "EFFECT_UNREPORTED",
    "EXECUTION_FAILED",
    "NO_OUTCOME",
    "OBSERVED_EFFECT_KEYS",
    "UNVERIFIABLE",
    "VERIFIED",
    "CapabilityEffectVerifier",
    "EffectVerdict",
    "ReclamationCandidate",
    "UnselectableArtifactReaper",
    "observed_effect_from_result",
]
