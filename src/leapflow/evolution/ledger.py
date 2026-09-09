"""Rebuild evolution episodes from the records the system already keeps.

No probe is needed for this. The adaptive loop already persists one decision
record per run carrying requirements, resolutions, the plan, the mutation, the
policy decision, the registry version on both sides, and the ids of the
observations that motivated it. That record is three of the five stages already;
this module supplies the two ends it never joined:

* **the cause** -- ``observation_ids`` reaches back into the observation store, so
  an episode can say what environment evidence set it in motion, and (since the
  requirement metadata now propagates them) carry the teacher's own hypothesis and
  confidence when the world model authored it;
* **the consequence** -- the observation's ``status`` says whether the gap the
  episode was supposed to close actually closed, stayed open, or *recurred*.

Reading rather than writing is the whole point of this stage. It ships a causal
timeline with no new probe, no new schema and no change to the evolution path; the
durable ledger only becomes necessary later, when live traces arrive from mutation
points that no store can reconstruct after the fact.

Two conclusions here are deliberately conservative:

* A retired observation is reported as ``DECLARED_FITNESS``, never as verified.
  The engine retires on re-resolution, which only proves a candidate *declared* it
  provides the capability -- the recorded v0.7 defect was exactly a wrongly
  selected tool retiring the evidence for its own gap.
* ``trust_now`` is a live read and is labelled as such. There is no history to
  reconstruct a "before" from, and inventing one would put a number on the board
  that never existed.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from leapflow.domain.evolution_trace import (
    ABORTED,
    COMMITTED,
    DECLARED_FITNESS,
    NOT_APPLICABLE,
    OPEN,
    REOPENED,
    RESOLVED,
    STILL_OPEN,
    EvolutionEpisode,
    EvolutionStage,
    EvolutionTrace,
)

logger = logging.getLogger(__name__)

#: An episode with no closing trace is abandoned rather than left open forever.
DEFAULT_EPISODE_TTL_S = 1800.0

#: How many observations to index. Bounded because the index is built per cycle.
_OBSERVATION_INDEX_LIMIT = 400

#: Policy actions that close an episode by deciding *not* to change anything.
#: Reported as committed on purpose: "why the framework did not evolve" is as much
#: a part of transparency as why it did.
_NO_CHANGE_ACTIONS = frozenset({"none", "observe_only"})

#: Evidence kind -> the driver class a reader recognises.
_DRIVER_BY_EVIDENCE = {
    "world_model_intent": "world_model",
    "unknown_tool": "unknown_tool",
    "interface_drift": "environment_probe",
    "affordance_removed": "environment_probe",
}


class EvolutionLedger:
    """Assemble recent evolution episodes from existing profile-scoped stores.

    Every store is optional. With only the plan store the timeline still renders,
    just without cause or consequence; with none of them ``recent_episodes``
    returns empty and the caller degrades to a live snapshot. A ledger that
    refused to work without every input would make the panel all-or-nothing.
    """

    def __init__(
        self,
        *,
        plan_store: Any,
        observation_store: Any = None,
        trust_ledger: Any = None,
        episode_ttl_s: float = DEFAULT_EPISODE_TTL_S,
    ) -> None:
        self._plans = plan_store
        self._observations = observation_store
        self._trust = trust_ledger
        self._ttl = max(0.0, float(episode_ttl_s))

    def recent_episodes(self, *, limit: int = 20, now: float = 0.0) -> tuple[EvolutionEpisode, ...]:
        """Return newest-first episodes rebuilt from decision records.

        Never raises: this feeds a transparency panel, and a ledger that fails
        would take the panel with it while reporting nothing about why.
        """
        try:
            records = list(self._plans.list_records(limit=max(1, int(limit))))
        except Exception:  # noqa: BLE001 - degraded timeline, not a fault
            logger.debug("evolution ledger: plan records unreadable", exc_info=True)
            return ()
        index = self._observation_index()
        episodes: list[EvolutionEpisode] = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            try:
                episodes.append(self._episode(record, index, now))
            except Exception:  # noqa: BLE001 - one bad record must not blank the timeline
                logger.debug("evolution ledger: record skipped", exc_info=True)
        return tuple(episodes)

    # ── observation index (the cause, and the consequence) ────────────────

    def _observation_index(self) -> dict[str, Mapping[str, Any]]:
        if self._observations is None:
            return {}
        try:
            records = self._observations.list_observations(limit=_OBSERVATION_INDEX_LIMIT)
        except Exception:  # noqa: BLE001
            logger.debug("evolution ledger: observations unreadable", exc_info=True)
            return {}
        return {
            str(record.get("observation_id") or ""): record
            for record in records
            if isinstance(record, Mapping) and record.get("observation_id")
        }

    # ── one episode ───────────────────────────────────────────────────────

    def _episode(
        self,
        record: Mapping[str, Any],
        index: Mapping[str, Mapping[str, Any]],
        now: float,
    ) -> EvolutionEpisode:
        record_id = str(record.get("record_id") or "")
        opened_at = float(record.get("created_at") or 0.0)
        observations = [
            index[obs_id]
            for obs_id in (str(item) for item in record.get("observation_ids") or ())
            if obs_id in index
        ]
        requirements = [r for r in record.get("requirements") or [] if isinstance(r, Mapping)]
        mutation = dict(record.get("mutation") or {})
        policy = dict(record.get("policy_decision") or {})
        proposal = dict(record.get("proposal") or {})

        before = int(record.get("registry_version_before") or -1)
        after = int(record.get("registry_version_after") or -1)
        capability = self._capability(requirements, observations)
        plugin_id = str(mutation.get("plugin_id") or "")
        mutation_action = str(mutation.get("action") or "")
        gap_closure = self._gap_closure(observations, mutation_action, after > before >= 0)
        declared = self._declared(requirements)

        traces = self._traces(
            record_id, record, observations, requirements, mutation, policy, before, after
        )
        status, closed_at = self._status(record, opened_at, mutation_action, after > before >= 0, now)

        return EvolutionEpisode(
            episode_id=f"ep-{record_id}" if record_id else f"ep-{int(opened_at)}",
            opened_at=opened_at,
            status=status,
            traces=traces,
            closed_at=closed_at,
            driver=self._driver(observations, requirements, record),
            capability=capability,
            intent_id=str(declared.get("intent_id") or ""),
            hypothesis=str(declared.get("hypothesis") or ""),
            confidence=self._confidence(declared),
            mutation_action=mutation_action or "none",
            plugin_id=plugin_id,
            registry_before=before,
            registry_after=after,
            # ``proposal.status`` on a decision record is the *acquisition
            # lifecycle* vocabulary. ``review_status`` is a different store's
            # answer to a different question ("should a human accept this"), so it
            # stays empty here rather than borrowing this value and conflating two
            # vocabularies the code explicitly warns must not be merged.
            lifecycle_status=str(proposal.get("status") or ""),
            policy_action=str(policy.get("action") or ""),
            autonomy_level=str(policy.get("autonomy_level") or ""),
            gap_closure=gap_closure,
            # Only ever declared fitness today: the engine retires an observation
            # when re-resolution reports the requirement met, which is not
            # evidence the capability works.
            verification_tier=DECLARED_FITNESS if gap_closure == RESOLVED else "",
            effect_verdict="",
            trust_at_decision=self._trust_at_decision(record, proposal),
            trust_now=self._trust_now(plugin_id),
            outcome=self._outcome(mutation_action, gap_closure, policy, after > before >= 0),
        )

    # ── stage traces ──────────────────────────────────────────────────────

    def _traces(
        self,
        record_id: str,
        record: Mapping[str, Any],
        observations: Sequence[Mapping[str, Any]],
        requirements: Sequence[Mapping[str, Any]],
        mutation: Mapping[str, Any],
        policy: Mapping[str, Any],
        before: int,
        after: int,
    ) -> tuple[EvolutionTrace, ...]:
        """Build one trace per stage the record can actually evidence."""
        traces: list[EvolutionTrace] = []
        base = {"record_id": record_id} if record_id else {}

        for observation in observations:
            result = dict(observation.get("result") or {})
            kind = str(result.get("error_type") or "environment_delta")
            traces.append(
                EvolutionTrace(
                    stage=EvolutionStage.OBSERVE,
                    kind=kind,
                    ts=float(observation.get("first_seen_at") or 0.0),
                    correlation={
                        **base,
                        "observation_id": str(observation.get("observation_id") or ""),
                    },
                    summary=str(result.get("evidence") or result.get("recovery_hint") or kind),
                    detail={
                        "occurrence_count": int(observation.get("occurrence_count") or 0),
                        "status": self._observation_status(observation),
                        "result": result,
                    },
                )
            )

        if requirements:
            traces.append(
                EvolutionTrace(
                    stage=EvolutionStage.ORIENT,
                    kind="capability_gap",
                    ts=float(record.get("created_at") or 0.0),
                    correlation={
                        **base,
                        "requirement_id": str(requirements[0].get("requirement_id") or ""),
                    },
                    summary=", ".join(
                        str(r.get("capability") or "") for r in requirements if r.get("capability")
                    ),
                    detail={"requirements": [dict(r) for r in requirements]},
                )
            )

        if policy or record.get("resolutions"):
            traces.append(
                EvolutionTrace(
                    stage=EvolutionStage.DECIDE,
                    kind="policy_decision" if policy else "resolution",
                    ts=float(record.get("created_at") or 0.0),
                    correlation=dict(base),
                    summary=str(policy.get("reason") or "capability resolved"),
                    detail={
                        "policy_decision": dict(policy),
                        # Kept whole: the per-candidate score components are the
                        # only record of why a candidate lost, which is the half of
                        # "decision transparency" a selected-only view drops.
                        "resolutions": [
                            dict(r) for r in record.get("resolutions") or [] if isinstance(r, Mapping)
                        ],
                    },
                )
            )

        if mutation:
            traces.append(
                EvolutionTrace(
                    stage=EvolutionStage.ACT,
                    kind="plugin_mutation",
                    ts=float(record.get("created_at") or 0.0),
                    correlation={
                        **base,
                        "plugin_id": str(mutation.get("plugin_id") or ""),
                        "registry_version": str(after),
                    },
                    summary=f"{mutation.get('action') or 'none'} {mutation.get('plugin_id') or ''}".strip(),
                    detail={
                        "mutation": dict(mutation),
                        "registry_before": before,
                        "registry_after": after,
                    },
                )
            )

        for observation in observations:
            status = self._observation_status(observation)
            reopened = self._reopened(observation)
            # A recurrence is recorded by flipping ``status`` back to ``open``, so
            # testing the status alone would skip the single most important
            # outcome there is: a gap that was closed and came back.
            if status == "open" and not reopened:
                continue
            traces.append(
                EvolutionTrace(
                    stage=EvolutionStage.LEARN,
                    kind="observation_reopened" if reopened else "observation_resolved",
                    ts=float(observation.get("last_seen_at") or 0.0),
                    correlation={
                        **base,
                        "observation_id": str(observation.get("observation_id") or ""),
                    },
                    summary=str(observation.get("status_reason") or status),
                    detail={"status": status, "reopened": reopened},
                )
            )

        governance = [g for g in record.get("governance_results") or [] if isinstance(g, Mapping)]
        for entry in governance:
            traces.append(
                EvolutionTrace(
                    stage=EvolutionStage.LEARN,
                    kind="governance_outcome",
                    ts=float(record.get("created_at") or 0.0),
                    correlation={**base, "plugin_id": str(entry.get("plugin_id") or "")},
                    summary=str(entry.get("action") or ""),
                    detail=dict(entry),
                )
            )

        return tuple(sorted(traces, key=lambda trace: (trace.ts, trace.stage.value)))

    # ── derivations ───────────────────────────────────────────────────────

    @staticmethod
    def _observation_status(observation: Mapping[str, Any]) -> str:
        """Read the lifecycle status, matching how the store itself defaults it.

        A newly written observation carries no ``status`` field at all; the store's
        own ``unresolved()`` treats that absence as open, so this must too or a
        fresh gap would read as closed.
        """
        return str(observation.get("status") or "open")

    @staticmethod
    def _reopened(observation: Mapping[str, Any]) -> bool:
        """Whether this observation was retired and then recurred.

        The store records a recurrence by flipping ``status`` back to ``open`` and
        writing a reason that says so, which is the only durable trace that a
        closed gap came back.
        """
        return "reopened" in str(observation.get("status_reason") or "").lower()

    def _gap_closure(
        self,
        observations: Sequence[Mapping[str, Any]],
        mutation_action: str,
        framework_changed: bool,
    ) -> str:
        if not observations:
            return NOT_APPLICABLE
        if any(self._reopened(observation) for observation in observations):
            return REOPENED
        statuses = {self._observation_status(observation) for observation in observations}
        if statuses == {"open"}:
            # The framework changed and the motivating gap is still open: the
            # acquisition ran and did not (yet) help. Worth distinguishing from an
            # episode that never acted at all.
            return STILL_OPEN if (framework_changed or mutation_action not in ("", "none")) else OPEN
        if "open" not in statuses:
            return RESOLVED
        return STILL_OPEN

    @staticmethod
    def _capability(
        requirements: Sequence[Mapping[str, Any]], observations: Sequence[Mapping[str, Any]]
    ) -> str:
        for requirement in requirements:
            capability = str(requirement.get("capability") or "")
            if capability:
                return capability
        for observation in observations:
            result = dict(observation.get("result") or {})
            capability = str(result.get("capability") or result.get("original_tool_name") or "")
            if capability:
                return capability
        return ""

    @staticmethod
    def _declared(requirements: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Pull the world model's own words out of the requirement metadata.

        These travel on the requirement because the declared-evidence path
        propagates them; without that they would exist only inside the intent and
        never reach anything durable.
        """
        for requirement in requirements:
            metadata = dict(requirement.get("metadata") or {})
            if metadata.get("intent_id") or metadata.get("hypothesis"):
                return {
                    "intent_id": metadata.get("intent_id"),
                    # The requirement carries the hypothesis as its evidence text.
                    "hypothesis": metadata.get("hypothesis") or requirement.get("evidence"),
                    "confidence": metadata.get("confidence"),
                }
        return {}

    @staticmethod
    def _confidence(declared: Mapping[str, Any]) -> float:
        try:
            return max(0.0, min(1.0, float(declared.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _driver(
        observations: Sequence[Mapping[str, Any]],
        requirements: Sequence[Mapping[str, Any]],
        record: Mapping[str, Any],
    ) -> str:
        """Classify what set this episode in motion, from declarations only."""
        for observation in observations:
            result = dict(observation.get("result") or {})
            driver = _DRIVER_BY_EVIDENCE.get(str(result.get("error_type") or ""))
            if driver:
                return driver
        for requirement in requirements:
            origin = str(requirement.get("origin") or "")
            if origin == "world_model":
                return "world_model"
            if origin == "explicit_request":
                return "manual"
            if origin:
                return origin
        return str(record.get("source") or "") or "unknown"

    def _status(
        self,
        record: Mapping[str, Any],
        opened_at: float,
        mutation_action: str,
        framework_changed: bool,
        now: float,
    ) -> tuple[str, float]:
        policy_action = str(dict(record.get("policy_decision") or {}).get("action") or "")
        if framework_changed:
            return COMMITTED, opened_at
        if policy_action in _NO_CHANGE_ACTIONS:
            return COMMITTED, opened_at
        if mutation_action and mutation_action != "none" and not framework_changed:
            # A mutation was attempted and the registry did not move: unresolved,
            # not committed. Left open so the operator sees an attempt that had no
            # effect rather than a clean conclusion.
            return (ABORTED if self._expired(opened_at, now) else OPEN), 0.0
        return (ABORTED if self._expired(opened_at, now) else OPEN), 0.0

    def _expired(self, opened_at: float, now: float) -> bool:
        return bool(now and self._ttl and (now - opened_at) > self._ttl)

    @staticmethod
    def _trust_at_decision(record: Mapping[str, Any], proposal: Mapping[str, Any]) -> str:
        trust_state = dict(proposal.get("trust_state") or {})
        level = str(trust_state.get("trust_level") or trust_state.get("level") or "")
        if level:
            return level
        for entry in record.get("governance_results") or []:
            if isinstance(entry, Mapping) and entry.get("trust_level"):
                return str(entry["trust_level"])
        return ""

    def _trust_now(self, plugin_id: str) -> str:
        """Live trust for the mutated plugin, or empty when it cannot be read.

        Named ``_now`` rather than ``_after`` on purpose: there is no stored
        history to reconstruct a before/after pair from, and presenting a live
        reading as an "after" would imply a comparison that was never made.
        """
        if not plugin_id or self._trust is None:
            return ""
        try:
            return str(self._trust.level(plugin_id).name)
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _outcome(
        mutation_action: str,
        gap_closure: str,
        policy: Mapping[str, Any],
        framework_changed: bool,
    ) -> str:
        """One phrase a reader can scan, covering the four interesting endings."""
        if gap_closure == REOPENED:
            return "regressed"
        if framework_changed and gap_closure == RESOLVED:
            return f"{mutation_action or 'changed'}; gap closed (declared fitness)"
        if framework_changed and gap_closure == STILL_OPEN:
            return f"{mutation_action or 'changed'}; gap still open"
        if framework_changed:
            return mutation_action or "changed"
        action = str(policy.get("action") or "")
        if action in _NO_CHANGE_ACTIONS:
            return f"no action ({action})"
        if mutation_action and mutation_action != "none":
            return f"{mutation_action} attempted; registry unchanged"
        return "no action"


__all__ = ["DEFAULT_EPISODE_TTL_S", "EvolutionLedger"]
