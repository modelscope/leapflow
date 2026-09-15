# Copyright (c) Alibaba, Inc. and its affiliates.
"""The world model as the first driver of capability self-evolution.

``TrajectoryGrader.grade_and_propose`` can emit an :class:`EvolutionIntent`, and
the observation pipeline can turn a declared intent into a governed
``CapabilityRequirement``. Nothing joined the two, so the world model could form a
capability hypothesis that no part of the system ever received. This driver is
that join, and it is deliberately the *only* one.

Where it runs, and why that is safe:

* **Cold path, once per episode.** It is invoked at the session-end learning
  boundary, after a trajectory is flushed -- never inside a turn. The teacher's own
  ``grading`` budget pool bounds how often it can spend an LLM call, so making the
  world model the first driver adds no per-turn cost.
* **Privileged context, not privileged authority.** The teacher sees the whole
  trajectory with actual outcomes (hindsight the acting policy never had), which is
  what lets it notice a capability was *missing* rather than merely used badly. It
  still only proposes: each intent is written as ordinary structured evidence and
  must pass the classifier, the detector, resolution, risk classification,
  approval, validation and trust exactly like an ``unknown_tool`` signal.
* **Opt-in.** Admission is decided by ``CapabilityEvidenceClassifier``. Until an
  operator adds ``world_model_intent`` to ``accepted_evidence_kinds``, intents are
  reported as *proposed but not admitted* and change nothing. The driver never
  writes around that gate.
* **Clamped.** Every intent is rendered with an explicit ``risk_ceiling``, so a
  model cannot widen the risk cap of the capability it is asking for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.domain.evolution_intent import (
    MODEL_AUTHORED_RISK_CEILING,
    EvolutionIntent,
)
from leapflow.domain.plugin_proposal import RiskLevel

logger = logging.getLogger(__name__)


@runtime_checkable
class CapabilityGapTeacher(Protocol):
    """A hindsight evaluator that can also propose capability gaps.

    Structural rather than a concrete import so the driver does not bind the
    learning layer to ``world_model``, and so a recorded or stub teacher can be
    substituted in tests and experiments.
    """

    async def grade_and_propose(
        self, trajectory: list[dict], goal: str = "", **kwargs: Any
    ) -> Any:
        """Return an object exposing ``grades`` and ``intents``.

        ``**kwargs`` keeps this structural contract open: the driver passes
        ``degraded_capabilities`` when it has any, and a teacher that predates that
        context stays conformant by ignoring it.
        """
        ...


@runtime_checkable
class EvidenceIntake(Protocol):
    """The governed intake an intent must pass through."""

    def observe_result(
        self, result: Mapping[str, Any] | None, **kwargs: Any
    ) -> dict[str, Any] | None:
        """Persist admitted evidence; return ``None`` when the gate rejects it."""
        ...

    def requirements(
        self, *, min_count: int = 1, limit: int = 50
    ) -> tuple[CapabilityRequirement, ...]:
        """Derive requirements from admitted evidence."""
        ...


@dataclass(frozen=True)
class WorldModelDriveResult:
    """What one world-model-driven evolution pass produced.

    ``proposed`` counts every intent the teacher formed; ``admitted`` counts those
    the evidence gate accepted. The two differ whenever the operator has not opted
    in, which is the normal default -- so a non-zero ``proposed`` with an empty
    ``admitted`` is a correct, quiet outcome, not a failure.
    """

    grades: tuple[Any, ...] = ()
    #: Everything the teacher concluded, across all four actions. ``intents`` below is
    #: the ``acquire`` subset, so the cheap verdicts stay visible instead of being
    #: dropped for not writing code -- three of the four change nothing except what the
    #: acting agent knows, which is the point of asking.
    verdicts: tuple[Any, ...] = ()
    #: Capabilities whose knowledge was written to the distilled store this session.
    #: The C1 channel's receipt: a session that adapted purely by teaching the next one
    #: something has this non-empty and everything else empty.
    distilled: tuple[str, ...] = ()
    intents: tuple[EvolutionIntent, ...] = ()
    admitted_observation_ids: tuple[str, ...] = ()
    requirements: tuple[CapabilityRequirement, ...] = field(default_factory=tuple)
    #: Proposals queued for governed acquisition. Empty when no sink is installed,
    #: which is the default: an intent then reaches a requirement and stops there.
    queued_proposal_ids: tuple[str, ...] = ()

    @property
    def proposed(self) -> int:
        return len(self.intents)

    @property
    def admitted(self) -> int:
        return len(self.admitted_observation_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graded_actions": len(self.grades),
            "proposed": self.proposed,
            "admitted": self.admitted,
            "queued": len(self.queued_proposal_ids),
            "capabilities": sorted({r.capability for r in self.requirements}),
            # Counted per action so a session that adapted purely by distilling
            # knowledge is distinguishable from one that did nothing.
            "distilled": list(self.distilled),
            "by_action": {
                action: sum(1 for v in self.verdicts if getattr(v, "action", "") == action)
                for action in ("absorb", "rebind", "acquire", "escalate")
            },
        }

class WorldModelEvolutionDriver:
    """Turn hindsight capability hypotheses into governed requirements."""

    def __init__(
        self,
        *,
        teacher: CapabilityGapTeacher,
        intake: EvidenceIntake,
        risk_ceiling: RiskLevel = MODEL_AUTHORED_RISK_CEILING,
        source: str = "world_model",
        degraded_capabilities: Any = None,
        proposal_sink: Any = None,
        knowledge_store: Any = None,
        alternatives_for: Any = None,
    ) -> None:
        self._teacher = teacher
        self._intake = intake
        self._risk_ceiling = risk_ceiling
        self._source = source
        # Facts the teacher needs in order to adjudicate a *replacement*: which
        # capabilities have a provider that keeps failing while still in service.
        # Injected as a callable so the driver does not bind to a store, and so a
        # deployment without governance wiring simply grades without them.
        self._degraded_capabilities = degraded_capabilities
        # Where an intent becomes a queued ``PluginProposal``. Without it an intent
        # reaches a requirement and stops: resolution reports the capability unmet and
        # nothing turns that into an acquisition. This is the last hop of the chain,
        # and it stays optional because queueing proposals is a governed, opt-in
        # capability rather than something grading should do by default.
        self._proposal_sink = proposal_sink
        # Where the cheap verdicts land. Three of the four actions change nothing except
        # what the acting agent knows, so without this they would be graded, traced, and
        # then thrown away -- the teacher would have judged correctly and the next
        # session would repeat the same mistake. Optional so a deployment without the
        # store still grades and still acquires.
        self._knowledge_store = knowledge_store
        # The other providers of a degraded capability, and whether each can run here.
        # Without it the teacher must choose between ``rebind`` ("another installed
        # capability covers this") and ``acquire`` ("nothing does") without being told
        # which is true -- the deciding fact for both.
        self._alternatives_for = alternatives_for

    async def drive(
        self,
        trajectory: Sequence[Mapping[str, Any]],
        goal: str = "",
        *,
        environment: Any = None,
        session_id: str = "",
        turn_id: str = "",
        workspace_root: str = "",
    ) -> WorldModelDriveResult:
        """Grade the episode, then submit any capability gap it revealed.

        Returns an empty result rather than raising: this runs on a learning
        boundary, and a failure to learn must never fail the session that produced
        the trajectory.
        """
        if not trajectory:
            return WorldModelDriveResult()
        degraded = self._collect_degraded()
        try:
            verdict = await self._teacher.grade_and_propose(
                list(trajectory), goal, degraded_capabilities=degraded
            )
        except TypeError:
            # A teacher that does not accept the newer context: grade without it
            # rather than lose the episode's grading entirely.
            try:
                verdict = await self._teacher.grade_and_propose(list(trajectory), goal)
            except Exception:  # noqa: BLE001 - teacher is advisory
                logger.debug("world_model_driver: teacher failed", exc_info=True)
                return WorldModelDriveResult()
        except Exception:  # noqa: BLE001 - teacher is advisory; never fail the session
            logger.debug("world_model_driver: teacher failed", exc_info=True)
            return WorldModelDriveResult()

        grades = tuple(getattr(verdict, "grades", ()) or ())
        verdicts = tuple(getattr(verdict, "verdicts", ()) or ())
        # Only ``acquire`` becomes an intent. The other three are conclusions about the
        # environment, and forwarding them into the acquisition path would turn a
        # recommendation to rebind into a request to write code.
        intents = tuple(getattr(verdict, "intents", ()) or ())
        # Distil before branching on ``intents``: a session whose every verdict was
        # ``absorb`` adapted the system, and it is the *only* thing that happened.
        distilled = self._distil(verdicts, environment)
        if not intents:
            # Still a real outcome: the teacher may have concluded the change is
            # absorbable, which is the cheapest and most common correct answer.
            return WorldModelDriveResult(
                grades=grades, verdicts=verdicts, distilled=distilled
            )

        admitted: list[str] = []
        # The intents the gate actually accepted, kept alongside their observation ids.
        # Collecting only the ids was enough to *count* admissions and not enough to
        # act on them: queueing then received every intent whenever any one of them was
        # admitted, so a rejected hypothesis reached the proposal queue through a side
        # door -- the exact bypass the opt-in gate exists to prevent.
        admitted_intents: list[EvolutionIntent] = []
        for intent in intents:
            try:
                record = self._intake.observe_result(
                    intent.to_observation_result(risk_ceiling=self._risk_ceiling),
                    environment=environment,
                    source=self._source,
                    session_id=session_id,
                    turn_id=turn_id,
                    workspace_root=workspace_root,
                )
            except (OSError, RuntimeError, TypeError, ValueError, AttributeError):
                logger.debug("world_model_driver: intake rejected an intent", exc_info=True)
                continue
            if record is not None:
                observation_id = str(record.get("observation_id") or "")
                if observation_id:
                    admitted.append(observation_id)
                    admitted_intents.append(intent)

        requirements: tuple[CapabilityRequirement, ...] = ()
        if admitted:
            try:
                requirements = self._intake.requirements(min_count=1)
            except (OSError, RuntimeError, TypeError, ValueError, AttributeError):
                logger.debug("world_model_driver: requirement derivation failed", exc_info=True)
        if intents and not admitted:
            logger.debug(
                "world_model_driver: %d intent(s) proposed but not admitted; add "
                "'world_model_intent' to accepted_evidence_kinds to enable",
                len(intents),
            )
        result = WorldModelDriveResult(
            grades=grades,
            verdicts=verdicts,
            distilled=distilled,
            intents=intents,
            admitted_observation_ids=tuple(admitted),
            requirements=requirements,
            queued_proposal_ids=(
                self._queue_proposals(admitted_intents, degraded)
                if admitted_intents
                else ()
            ),
        )
        self._trace_drive(result)
        return result

    def _distil(self, verdicts: Any, environment: Any) -> tuple[str, ...]:
        """Persist what each verdict concluded, returning the capabilities recorded.

        Contained: distillation improves the *next* session's context, so failing to
        write it must not fail this one. Returns capability names rather than entries
        because the caller reports counts and the entries live in the store.
        """
        if self._knowledge_store is None or not verdicts:
            return ()
        env = {}
        if environment is not None and hasattr(environment, "to_dict"):
            try:
                env = dict(environment.to_dict())
            except Exception:  # noqa: BLE001 - a fingerprint is context, not a gate
                env = {}
        try:
            stored = self._knowledge_store.record_all(verdicts, environment=env)
            return tuple(entry.capability for entry in stored)
        except Exception:  # noqa: BLE001 - distillation must never fail a session
            logger.debug("world_model_driver: distillation failed", exc_info=True)
            return ()

    def _collect_degraded(self) -> tuple[Mapping[str, Any], ...]:
        """Degradation facts for the teacher, filtered and environment-tagged.

        Prefers the intake's own reader when it has one, so the two rules that make
        this evidence usable -- retry-owned classes excluded, environment fingerprint
        attached -- are applied in one place rather than re-derived here. An explicit
        provider still wins, which is what lets an experiment substitute its own view.
        """
        provider = self._degraded_capabilities
        if provider is None:
            provider = getattr(self._intake, "degraded_capabilities", None)
        if provider is None:
            return ()
        try:
            facts = tuple(provider() or ())
        except Exception:  # noqa: BLE001 - missing context degrades grading, not the session
            logger.debug("world_model_driver: degradation facts unavailable", exc_info=True)
            return ()
        return self._with_alternatives(self._with_prior_verdicts(facts))

    def _with_prior_verdicts(
        self, facts: tuple[Mapping[str, Any], ...]
    ) -> tuple[Mapping[str, Any], ...]:
        """Attach what was concluded last time about each still-failing capability.

        This is the feedback edge, and without it the loop is open: the teacher would be
        shown the same degradation every session and could only ever reach the same
        conclusion, having no way to learn that its previous answer did not work.

        Deliberately stated as *fact*, not as a verdict on the verdict. Knowledge existing
        while the capability still fails is evidence that the previous adaptation did not
        resolve it -- not proof the judgement was wrong. The student may never have used
        the knowledge, or the environment may have moved again, or this may be a different
        failure. Which of those it is, is exactly what the teacher is for.
        """
        if self._knowledge_store is None or not facts:
            return facts
        enriched: list[Mapping[str, Any]] = []
        for fact in facts:
            capability = str(fact.get("capability") or "")
            try:
                prior = self._knowledge_store.for_capability(capability)
            except Exception:  # noqa: BLE001 - context, never a gate
                prior = None
            if prior is None:
                enriched.append(fact)
                continue
            merged = dict(fact)
            merged["prior_action"] = prior.action
            merged["prior_knowledge"] = prior.knowledge
            enriched.append(merged)
        return tuple(enriched)

    def _with_alternatives(
        self, facts: tuple[Mapping[str, Any], ...]
    ) -> tuple[Mapping[str, Any], ...]:
        """Attach the other providers of each degraded capability.

        Answers the question the action space is defined by. A teacher that cannot see
        whether an alternative exists is guessing between rebind and acquire, and the
        measured behaviour was exactly that.
        """
        if self._alternatives_for is None or not facts:
            return facts
        enriched: list[Mapping[str, Any]] = []
        for fact in facts:
            try:
                rows = tuple(
                    self._alternatives_for(
                        str(fact.get("capability") or ""), str(fact.get("plugin_id") or "")
                    )
                    or ()
                )
            except Exception:  # noqa: BLE001 - context, never a gate
                logger.debug("world_model_driver: alternatives unavailable", exc_info=True)
                enriched.append(fact)
                continue
            merged = dict(fact)
            merged["alternatives"] = rows
            enriched.append(merged)
        return tuple(enriched)

    def _queue_proposals(
        self,
        admitted_intents: Sequence[EvolutionIntent],
        degraded: Sequence[Mapping[str, Any]] = (),
    ) -> tuple[str, ...]:
        """Turn *admitted* intents into queued proposals, if a sink is installed.

        Takes only the intents the evidence gate accepted, never the full set: an
        intent the operator has not opted into must not become a queued acquisition by
        a side door. The proposal itself mutates nothing -- generation and installation
        remain separately approval-gated -- so queueing is the last *observation-only*
        step.

        An intent whose capability appears in ``degraded`` is queued as a *rival* to the
        named incumbent rather than as a gap fill. That is a factual lookup against the
        degradation record, not a reading of the hypothesis: the record exists precisely
        because a provider is installed and failing. Without the distinction the rival
        would be named after the capability alone, collide with the incumbent's own
        generated name, and never be able to coexist with the thing it competes against.

        Each proposal is built with the clamped risk ceiling, so a model cannot widen
        the risk cap of what it is asking to have built.
        """
        if self._proposal_sink is None or not admitted_intents:
            return ()
        incumbents = {
            str(item.get("capability") or ""): str(item.get("plugin_id") or "")
            for item in degraded or ()
            if item.get("capability")
        }
        queued: list[str] = []
        try:
            from leapflow.learning.capability_gap_detector import CapabilityGapDetector

            detector = CapabilityGapDetector()
        except Exception:  # noqa: BLE001
            logger.debug("world_model_driver: detector unavailable", exc_info=True)
            return ()
        for intent in admitted_intents:
            try:
                proposal = detector.proposal_from_evolution_intent(
                    intent,
                    risk_ceiling=self._risk_ceiling,
                    incumbent=incumbents.get(str(getattr(intent, "capability", "")), ""),
                )
                identifier = self._proposal_sink(proposal)
            except Exception:  # noqa: BLE001 - one bad intent must not stop the rest
                logger.debug("world_model_driver: proposal not queued", exc_info=True)
                continue
            queued.append(str(identifier or getattr(proposal, "proposal_id", "")))
        return tuple(q for q in queued if q)

    def _trace_drive(self, result: WorldModelDriveResult) -> None:
        """Emit what the teacher concluded, admitted or not.

        The highest-value probe in the system, because of the case it is the only
        record of: an intent that was *proposed and not admitted* writes no
        observation, so it exists nowhere durable and vanishes with the process. The
        board would otherwise show a silent, idle pipeline while the world model was
        in fact proposing on every session -- indistinguishable from a model that had
        nothing to say.

        Not admitting is a legitimate quiet outcome, not a failure: the evidence kind
        simply is not in ``accepted_evidence_kinds``. The trace says which it was so
        a reader can tell "switched off" from "nothing happening".
        """
        try:
            from leapflow.domain.evolution_trace import EvolutionStage
            from leapflow.telemetry.evolution_tap import emit_trace, is_enabled

            if not is_enabled():
                return
            intents = result.intents
            admitted = result.admitted_observation_ids
            emit_trace(
                EvolutionStage.OBSERVE,
                "world_model_drive",
                correlation={
                    "intent_ids": ",".join(
                        str(getattr(i, "intent_id", "")) for i in intents
                    ),
                },
                summary=(
                    f"teacher returned {len(result.verdicts)} verdict(s); "
                    f"{len(intents)} acquire, admitted {len(admitted)}"
                    if result.verdicts
                    else "teacher proposed nothing"
                ),
                detail={
                    # The model's own hypothesis, rationale, expected effect and
                    # confidence -- the only structured answer to "why should this
                    # evolve" that exists anywhere.
                    "verdicts": [
                        dict(v.to_dict()) if hasattr(v, "to_dict") else {}
                        for v in result.verdicts
                    ],
                    "intents": [self._intent_detail(i) for i in intents],
                    "admitted_observation_ids": list(admitted),
                    "queued_proposal_ids": list(result.queued_proposal_ids),
                    "graded": len(result.grades),
                    "requirements": len(result.requirements),
                    "not_admitted_reason": (
                        "world_model_intent is not in accepted_evidence_kinds"
                        if intents and not admitted
                        else ""
                    ),
                },
            )
        except Exception:  # noqa: BLE001 - the teacher is advisory; telemetry more so
            logger.debug("world_model_driver: evolution trace failed", exc_info=True)

    @staticmethod
    def _intent_detail(intent: Any) -> dict[str, Any]:
        """Serialise an intent defensively -- a teacher-authored object may be partial."""
        to_dict = getattr(intent, "to_dict", None)
        if callable(to_dict):
            try:
                return dict(to_dict())
            except Exception:  # noqa: BLE001
                pass
        return {
            key: getattr(intent, key, "")
            for key in ("intent_id", "capability", "hypothesis", "confidence")
        }


__all__ = [
    "CapabilityGapTeacher",
    "EvidenceIntake",
    "WorldModelDriveResult",
    "WorldModelEvolutionDriver",
]
