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

    async def grade_and_propose(self, trajectory: list[dict], goal: str = "") -> Any:
        """Return an object exposing ``grades`` and ``intents``."""
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
    intents: tuple[EvolutionIntent, ...] = ()
    admitted_observation_ids: tuple[str, ...] = ()
    requirements: tuple[CapabilityRequirement, ...] = field(default_factory=tuple)

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
            "capabilities": sorted({r.capability for r in self.requirements}),
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
    ) -> None:
        self._teacher = teacher
        self._intake = intake
        self._risk_ceiling = risk_ceiling
        self._source = source

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
        try:
            verdict = await self._teacher.grade_and_propose(list(trajectory), goal)
        except Exception:  # noqa: BLE001 - teacher is advisory; never fail the session
            logger.debug("world_model_driver: teacher failed", exc_info=True)
            return WorldModelDriveResult()

        grades = tuple(getattr(verdict, "grades", ()) or ())
        intents = tuple(getattr(verdict, "intents", ()) or ())
        if not intents:
            return WorldModelDriveResult(grades=grades)

        admitted: list[str] = []
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
            intents=intents,
            admitted_observation_ids=tuple(admitted),
            requirements=requirements,
        )
        self._trace_drive(result)
        return result

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
                    f"teacher proposed {len(intents)}, admitted {len(admitted)}"
                    if intents
                    else "teacher proposed nothing"
                ),
                detail={
                    # The model's own hypothesis, rationale, expected effect and
                    # confidence -- the only structured answer to "why should this
                    # evolve" that exists anywhere.
                    "intents": [self._intent_detail(i) for i in intents],
                    "admitted_observation_ids": list(admitted),
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
