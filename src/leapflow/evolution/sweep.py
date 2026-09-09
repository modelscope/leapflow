"""The cold-path co-evolution sweep: verify, govern, reclaim.

Three capabilities existed in the tree with no caller, which the evolution
dashboard reported as ``NO_EVIDENCE`` rows rather than treating the module's
presence as proof:

* ``CapabilityEffectVerifier`` -- closures rested on *declared* fitness (a
  candidate says it provides the capability and its affordances are present)
  instead of *observed* effect;
* ``QuarantineCandidateTracker`` -- trust demotion was live but quarantine had no
  feed, so a plugin could fail indefinitely without being disabled;
* ``UnselectableArtifactReaper`` -- an artifact no admissible requirement can
  select produces no outcomes, so it is never quarantined and stays registered as
  untracked residue.

This module is the single call site that closes all three. It runs on a **cold
path** (the session-end learning boundary, beside the world-model driver), never
inside a turn: governance is required to add no per-turn cost, and every step here
writes to a store or awaits a lifecycle actor.

Every step emits an evolution trace, because a transition that is not recorded is
not explainable, reproducible, or reversible. The sweep records its *no-op* and
*rejected* branches too -- "nothing to verify" and "no reclamation candidate" are
facts a reader needs in order to distinguish a quiet system from a switched-off
one.

Nothing here mutates the registry directly. Disabling a plugin stays with the
lifecycle actor, reached through ``LifecycleGovernor``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.domain.evolution_trace import EvolutionStage
from leapflow.learning.capability_effect_verifier import (
    CapabilityEffectVerifier,
    EffectVerdict,
    ReclamationCandidate,
    UnselectableArtifactReaper,
)
from leapflow.learning.outcome_governance_feed import (
    QuarantineCandidateTracker,
    drain_quarantine_candidates,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweepOutcome:
    """What one cold-path sweep observed and did.

    Reported into the session-end observability payload so a reader can tell a
    quiet sweep from an absent one -- the distinction the dashboard's
    ``NO_EVIDENCE`` rows exist to make.
    """

    verdicts: tuple[EffectVerdict, ...] = ()
    quarantined: tuple[Mapping[str, Any], ...] = ()
    reclamation: tuple[ReclamationCandidate, ...] = ()

    @property
    def verified(self) -> int:
        return sum(1 for v in self.verdicts if v.verified is True)

    @property
    def refuted(self) -> int:
        return sum(1 for v in self.verdicts if v.verified is False)

    @property
    def unverifiable(self) -> int:
        return sum(1 for v in self.verdicts if v.verified is None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect_verified": self.verified,
            "effect_refuted": self.refuted,
            "effect_unverifiable": self.unverifiable,
            "quarantined": len(self.quarantined),
            "reclamation_candidates": [c.plugin_id for c in self.reclamation],
        }


@dataclass
class CoevolutionSweep:
    """Runs effect verification, quarantine governance and reclamation.

    All collaborators are optional: a sweep with nothing wired emits the
    corresponding no-op traces and returns an empty outcome, which is what keeps
    this safe to call unconditionally at session end.
    """

    governor: Any = None
    tracker: QuarantineCandidateTracker | None = None
    verifier: CapabilityEffectVerifier = field(default_factory=CapabilityEffectVerifier)
    reaper: UnselectableArtifactReaper = field(default_factory=UnselectableArtifactReaper)
    proposal_ids: Mapping[str, str] = field(default_factory=dict)

    async def run(
        self,
        *,
        verifications: Sequence[tuple[CapabilityRequirement, Mapping[str, Any], str]] = (),
        acquired_plugin_ids: Sequence[str] = (),
        resolutions: Sequence[Mapping[str, Any]] = (),
    ) -> SweepOutcome:
        """Verify effects, drain quarantine candidates, then scan for residue.

        ``verifications`` is a sequence of ``(requirement, outcome, plugin_id)``:
        what an acquired capability was asked to do and what was observed. Order
        matters -- verification runs first so a refuted verdict can itself feed the
        quarantine drain in the same sweep.
        """
        verdicts = await self._verify(verifications)
        quarantined = await self._drain()
        reclamation = self._reclaim(acquired_plugin_ids, resolutions)
        return SweepOutcome(verdicts, quarantined, reclamation)

    # ── effect verification (L3 closure) ──────────────────────────────────

    async def _verify(
        self, verifications: Sequence[tuple[CapabilityRequirement, Mapping[str, Any], str]]
    ) -> tuple[EffectVerdict, ...]:
        if not verifications:
            self._emit(
                EvolutionStage.LEARN, "effect_verification",
                summary="nothing to verify this session",
                detail={"observed": 0, "no_op": True},
            )
            return ()

        verdicts: list[EffectVerdict] = []
        for requirement, outcome, plugin_id in verifications:
            try:
                verdict = self.verifier.verify(requirement, outcome, plugin_id=plugin_id)
            except (TypeError, ValueError, AttributeError):
                logger.debug("sweep: verification failed", exc_info=True)
                continue
            verdicts.append(verdict)
            self._emit(
                EvolutionStage.LEARN, "effect_verification",
                correlation={"plugin_id": verdict.plugin_id, "capability": verdict.capability},
                summary=f"{verdict.capability}: {verdict.reason}",
                detail=verdict.to_dict(),
            )
            # Only a decided verdict may move trust; "unverifiable" must not
            # quarantine a plugin for a missing declaration.
            if verdict.should_record_outcome:
                await self._record(verdict)
        return tuple(verdicts)

    async def _record(self, verdict: EffectVerdict) -> None:
        """Feed a decided verdict into trust/lifecycle governance."""
        if self.governor is None:
            return
        try:
            await self.governor.record_outcome(
                proposal_id=self.proposal_ids.get(verdict.plugin_id, ""),
                plugin_id=verdict.plugin_id,
                tool_name=verdict.plugin_id,
                ok=bool(verdict.verified),
                failure_class="" if verdict.verified else verdict.reason,
            )
        except Exception:  # noqa: BLE001 - governance must not break the sweep
            logger.debug("sweep: governance rejected a verdict", exc_info=True)

    # ── quarantine feed (cold-path drain) ─────────────────────────────────

    async def _drain(self) -> tuple[Mapping[str, Any], ...]:
        if self.tracker is None or self.governor is None:
            self._emit(
                EvolutionStage.ACT, "quarantine_drain",
                summary="quarantine feed not available",
                detail={"pending": 0, "no_op": True},
            )
            return ()
        pending = self.tracker.pending()
        if not pending:
            self._emit(
                EvolutionStage.ACT, "quarantine_drain",
                summary="no quarantine candidate this session",
                detail={"pending": 0, "no_op": True},
            )
            return ()
        handled = await drain_quarantine_candidates(
            self.tracker, self.governor, proposal_ids=self.proposal_ids
        )
        for entry in handled:
            plugin_id = str(entry.get("plugin_id") or "")
            detail = dict(entry)
            # Governance state is process-global because plugins are, so a streak can
            # be driven by one workspace and disable a plugin another was using. The
            # contributing workspaces travel with the decision so that is auditable
            # rather than mysterious.
            workspaces = self._contributing_workspaces(plugin_id)
            if workspaces:
                detail["contributing_workspaces"] = list(workspaces)
                detail["cross_workspace"] = len(workspaces) > 1
            self._emit(
                EvolutionStage.ACT, "quarantine_drain",
                correlation={"plugin_id": plugin_id},
                summary=f"{entry.get('plugin_id')}: {entry.get('action')}",
                detail=detail,
            )
        return handled

    @staticmethod
    def _contributing_workspaces(plugin_id: str) -> tuple[str, ...]:
        """Attribution lookup; absence is normal and must never break the drain."""
        try:
            from leapflow.evolution.observations import current_observations

            return current_observations().contributing_workspaces(plugin_id)
        except Exception:  # noqa: BLE001
            return ()

    # ── reclamation (residue) ─────────────────────────────────────────────

    def _reclaim(
        self,
        acquired_plugin_ids: Sequence[str],
        resolutions: Sequence[Mapping[str, Any]],
    ) -> tuple[ReclamationCandidate, ...]:
        try:
            found = self.reaper.candidates(
                acquired_plugin_ids=acquired_plugin_ids, resolutions=resolutions
            )
        except (TypeError, ValueError, AttributeError):
            logger.debug("sweep: reclamation scan failed", exc_info=True)
            return ()
        if not found:
            self._emit(
                EvolutionStage.LEARN, "reclamation",
                summary="no reclamation candidate",
                detail={
                    "acquired": len(acquired_plugin_ids),
                    "resolutions": len(resolutions),
                    "no_op": True,
                },
            )
            return ()
        for candidate in found:
            self._emit(
                EvolutionStage.LEARN, "reclamation",
                correlation={"plugin_id": candidate.plugin_id},
                summary=f"{candidate.plugin_id}: {candidate.reason}",
                detail=candidate.to_dict(),
            )
        return found

    @staticmethod
    def _emit(stage: EvolutionStage, kind: str, **kwargs: Any) -> None:
        """Record one sweep fact. Observability must never affect the observed."""
        try:
            from leapflow.telemetry.evolution_tap import emit_trace, is_enabled

            if is_enabled():
                emit_trace(stage, kind, **kwargs)
        except Exception:  # noqa: BLE001
            logger.debug("sweep: trace emission failed", exc_info=True)


__all__ = ["CoevolutionSweep", "SweepOutcome"]
