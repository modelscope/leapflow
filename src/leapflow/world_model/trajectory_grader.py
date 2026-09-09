"""Trajectory Grader — OPD teacher role for train-free agentic learning.

Implements the "teacher-as-reward-model" pattern from On-Policy Distillation:
the LLM, given *full hindsight context* (goal + trajectory + outcomes), grades
each action step with an advantage signal and identifies forking actions where
the chosen path diverges materially from the optimal one.

Teacher/student asymmetry comes from *information context*, not model capability:
  - Teacher sees the complete trajectory including final outcomes.
  - Student (prediction loop) acts with only the current state visible.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from leapflow.world_model.budget import LearningBudgetController
    from leapflow.world_model.experience_store import ExperienceStore

from leapflow.domain.evolution_intent import EvolutionIntent
from leapflow.llm.base import LLMProvider
from leapflow.llm.message_builder import build_system_message, build_user_message_text
from leapflow.world_model._json_utils import extract_json_object

logger = logging.getLogger(__name__)

DEFAULT_GRADE_LABELS: tuple[str, ...] = (
    "optimal", "acceptable", "suboptimal", "harmful",
)

_GRADE_PROMPT = """\
You are evaluating an agent's action trajectory with FULL hindsight.

Goal: {goal}

Trajectory (chronological order):
{trajectory_text}

For each step, provide:
- advantage: float in [-1, 1]. Positive = better than average, negative = harmful.
- is_forking: true if a meaningfully different action here would have led to a \
very different outcome. These are critical decision points.
- grade_label: one of {grade_labels}.

Output JSON:
{{"grades": [{{"step": 1, "advantage": 0.3, "is_forking": false, \
"grade_label": "{example_label}"}}, ...]}}"""

# Appended when the teacher is also asked to propose capability gaps. Kept in the
# *same* call as grading so a proposal costs no additional budget token: the
# hindsight context needed to grade is exactly the context needed to notice that a
# capability is missing.
#
# The teacher is deliberately not asked for a risk level. An intent is a
# hypothesis, not an authorisation; the risk ceiling is imposed by the trusted
# caller (see ``EvolutionIntent`` / ``MODEL_AUTHORED_RISK_CEILING``).
_GAP_PROMPT_SECTION = """

Additionally, identify any capability the agent *lacked* -- cases where no
available action could have achieved the goal, as distinct from an available
action being chosen badly. Report only genuine gaps; report none if the agent had
what it needed and merely used it poorly.

Before reporting a gap, apply these two rules:
- Do NOT restate the task, the goal, or the episode name as a capability. A
  capability is a reusable ability such as "chat.reply", never a description of
  this particular attempt.
- If the episode failed for a reason that is not a missing capability -- a label
  was renamed, an element moved, a transient error, a wrong choice among
  available actions -- return an empty list. An invented capability is worse than
  a missed one, because it will be built.

For each gap provide:
- capability: a stable dotted capability name (e.g. "chat.reply").
- hypothesis: what is missing or broken, in one sentence.
- confidence: float in [0, 1].
- target_affordance: the environment affordance a new adapter should target, if visible.
- rationale: why the existing capabilities cannot serve this.
- expected_effect: what should observably happen once the capability exists.

Add to the JSON:
{{"capability_gaps": [{{"capability": "...", "hypothesis": "...", \
"confidence": 0.7, "target_affordance": "...", "rationale": "...", \
"expected_effect": "..."}}, ...]}}
Use an empty list when there is no genuine gap."""

#: A capability name is a short dotted path of identifier-like segments. Bounded
#: deliberately: a model asked for a capability sometimes answers with a sentence, and a
#: sentence must never become a requirement.
_CAPABILITY_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}(\.[a-z0-9][a-z0-9_]{0,31}){1,3}$")


def _is_capability_name(value: str) -> bool:
    """Whether a teacher-supplied string is shaped like a capability at all.

    Requires lowercase dotted structure with 2-4 segments. Rejects prose, bare words,
    paths, and anything long enough to be a description rather than a name.
    """
    return bool(value) and len(value) <= 96 and bool(_CAPABILITY_RE.match(value))


def _echoes_goal(capability: str, goal: str) -> bool:
    """Whether the capability is just the goal (or episode name) restated.

    A model handed a goal string will sometimes hand it straight back as the capability.
    Compared on alphanumerics only, so separator and case differences do not let an echo
    through.
    """
    def norm(value: str) -> str:
        return "".join(ch for ch in str(value).lower() if ch.isalnum())

    normalised_goal = norm(goal)
    return bool(normalised_goal) and norm(capability) == normalised_goal


@dataclass(frozen=True)
class ActionGrade:
    """Teacher's grading of a single action step."""

    experience_id: str
    advantage: float
    is_forking: bool
    grade_label: str


@dataclass(frozen=True)
class TeacherVerdict:
    """Everything one hindsight evaluation produced.

    ``grades`` distil into the experience store as advantage signal; ``intents``
    are capability hypotheses that may drive self-evolution. Both are derived from
    a single LLM call, so a verdict costs one ``grading`` budget token.
    """

    grades: tuple[ActionGrade, ...] = ()
    intents: tuple[EvolutionIntent, ...] = ()


class TrajectoryGrader:
    """Grades completed trajectories from a teacher perspective (full hindsight).

    After a session's prediction loop accumulates a trajectory buffer, this
    component performs a single LLM call to grade all steps, then writes the
    advantage/forking signals back into the ExperienceStore.
    """

    def __init__(
        self,
        llm: LLMProvider,
        experience_store: "ExperienceStore",
        budget: "LearningBudgetController",
        *,
        min_trajectory_length: int = 3,
        grade_labels: tuple[str, ...] = DEFAULT_GRADE_LABELS,
    ) -> None:
        self._llm = llm
        self._store = experience_store
        self._budget = budget
        self._min_len = min_trajectory_length
        self._grade_labels = grade_labels

    async def grade_trajectory(
        self,
        trajectory: List[dict],
        goal: str = "",
    ) -> List[ActionGrade]:
        """Grade a completed trajectory and persist advantage signals.

        Each element of *trajectory* must contain at minimum:
            experience_id, action_description, predicted_effect,
            actual_effect, delta
        """
        if len(trajectory) < self._min_len:
            return []
        if not self._budget.has_tokens("grading"):
            return []

        traj_text = self._format_trajectory(trajectory)
        payload = await self._call_teacher_raw(traj_text, goal, propose_gaps=False)
        self._budget.spend("grading")

        grades = self._persist_grades(trajectory, self._parse_grades(payload))
        return grades

    async def grade_and_propose(
        self,
        trajectory: List[dict],
        goal: str = "",
    ) -> "TeacherVerdict":
        """Grade the trajectory *and* propose capability gaps, in one LLM call.

        This is the teacher's full verdict: the advantage signal that distils into
        experience, plus any :class:`EvolutionIntent` describing a capability the
        agent lacked. Both come from the same hindsight context and the same
        single ``grading`` budget token, so proposing costs nothing beyond grading.

        The returned intents are hypotheses. They carry no authorisation and must
        still traverse the deterministic chain (declared-fitness resolution, risk
        classification, approval, artifact validation, trust) before anything is
        acquired.
        """
        if len(trajectory) < self._min_len:
            return TeacherVerdict((), ())
        if not self._budget.has_tokens("grading"):
            return TeacherVerdict((), ())

        traj_text = self._format_trajectory(trajectory)
        payload = await self._call_teacher_raw(traj_text, goal, propose_gaps=True)
        self._budget.spend("grading")

        grades = self._persist_grades(trajectory, self._parse_grades(payload))
        return TeacherVerdict(tuple(grades), self._parse_intents(payload, goal))

    async def _call_teacher_raw(
        self,
        trajectory_text: str,
        goal: str,
        *,
        propose_gaps: bool = False,
    ) -> dict:
        """Single LLM call: teacher evaluates with full hindsight.

        Returns the parsed JSON object so grades and capability gaps can both be
        derived from one response. A failed or unparseable call yields ``{}`` --
        the teacher is advisory, so it must never fail the caller.
        """
        labels_str = ", ".join(f'"{label}"' for label in self._grade_labels)
        prompt = _GRADE_PROMPT.format(
            goal=goal or "(not specified)",
            trajectory_text=trajectory_text,
            grade_labels=labels_str,
            example_label=self._grade_labels[1] if len(self._grade_labels) > 1 else self._grade_labels[0],
        )
        if propose_gaps:
            prompt += _GAP_PROMPT_SECTION.format()
        try:
            resp = await self._llm.achat(
                [build_system_message(
                    "You are a trajectory evaluator with perfect hindsight.",
                ),
                 build_user_message_text(prompt)],
                stream=False, enable_thinking=False,
            )
            return extract_json_object(resp.content or "") or {}
        except Exception:
            logger.debug("trajectory_grader.call_teacher failed", exc_info=True)
            return {}

    def _format_trajectory(self, trajectory: List[dict]) -> str:
        """Render trajectory steps into a numbered text block."""
        lines: list[str] = []
        for i, step in enumerate(trajectory, 1):
            lines.append(
                f"Step {i}: action={step.get('action_description', '?')}\n"
                f"  predicted: {step.get('predicted_effect', '?')}\n"
                f"  actual: {step.get('actual_effect', '?')}\n"
                f"  delta: {step.get('delta', '?')}"
            )
        return "\n".join(lines)

    def _parse_grades(self, payload: dict) -> List[ActionGrade]:
        """Parse the teacher payload into ActionGrade objects."""
        raw_grades = payload.get("grades", []) if isinstance(payload, dict) else []
        results: List[ActionGrade] = []
        for raw in raw_grades:
            if not isinstance(raw, dict):
                continue
            try:
                advantage = float(raw.get("advantage", 0))
            except (TypeError, ValueError):
                advantage = 0.0
            results.append(ActionGrade(
                experience_id="",
                advantage=max(-1.0, min(1.0, advantage)),
                is_forking=bool(raw.get("is_forking", False)),
                grade_label=str(raw.get("grade_label", "acceptable")),
            ))
        return results

    def _parse_intents(self, payload: dict, goal: str = "") -> tuple[EvolutionIntent, ...]:
        """Parse declared capability gaps into intents, skipping malformed entries.

        A gap without both a ``capability`` and a ``hypothesis`` is discarded: the
        capability name must be declared, never inferred from prose.

        Two further rejections exist because a live model was measured doing exactly
        this. Asked to diagnose an episode that failed for a *non-capability* reason,
        ``qwen3.7-plus`` returned the episode's own name as the capability on 3 of 3
        trials. Nothing downstream would have caught it -- the name is well-formed, so
        it would have become a requirement and the governed pipeline would have
        faithfully tried to build ``chat.cosmetic.example``.

        So a capability must *look* like a capability, and must not be a restatement of
        the goal. Neither check can catch a plausible-but-wrong capability; that is what
        validation, effect verification and quarantine are for. These catch the
        degenerate case, which is the one that produces pure noise.
        """
        raw_gaps = payload.get("capability_gaps", []) if isinstance(payload, dict) else []
        intents: List[EvolutionIntent] = []
        for raw in raw_gaps:
            if not isinstance(raw, dict):
                continue
            capability = str(raw.get("capability") or "").strip()
            if not _is_capability_name(capability):
                logger.debug(
                    "trajectory_grader: rejected non-capability name %r", capability
                )
                continue
            if _echoes_goal(capability, goal):
                logger.debug(
                    "trajectory_grader: rejected goal restatement %r", capability
                )
                continue
            try:
                confidence = float(raw.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            try:
                intents.append(EvolutionIntent.create(
                    capability,
                    str(raw.get("hypothesis") or ""),
                    confidence=confidence,
                    target_affordance=str(raw.get("target_affordance") or ""),
                    rationale=str(raw.get("rationale") or ""),
                    expected_effect=str(raw.get("expected_effect") or ""),
                ))
            except ValueError:
                logger.debug("trajectory_grader: discarded malformed capability gap %r", raw)
        return tuple(intents)

    def _persist_grades(
        self,
        trajectory: List[dict],
        grades: List[ActionGrade],
    ) -> List[ActionGrade]:
        """Write grading results back to ExperienceStore.

        Returns grades with populated experience_id fields.
        """
        if len(grades) != len(trajectory):
            logger.debug(
                "trajectory_grader: grade count (%d) != trajectory length (%d)",
                len(grades), len(trajectory),
            )

        populated: List[ActionGrade] = []
        for i, grade in enumerate(grades):
            if i >= len(trajectory):
                break
            exp_id = trajectory[i].get("experience_id", "")
            bound = ActionGrade(
                experience_id=exp_id,
                advantage=grade.advantage,
                is_forking=grade.is_forking,
                grade_label=grade.grade_label,
            )
            populated.append(bound)
            if not exp_id:
                continue
            self._store.update_advantage(
                experience_id=exp_id,
                advantage=grade.advantage,
                is_forking=grade.is_forking,
                grade_label=grade.grade_label,
            )
        return populated

    @staticmethod
    def trajectory_from_outcomes(
        outcomes: list,
    ) -> List[dict]:
        """Convert a list of PredictionOutcome objects to trajectory dicts."""
        result: List[dict] = []
        for o in outcomes:
            result.append({
                "experience_id": getattr(o, "experience_id", ""),
                "action_description": o.prediction.action_description,
                "predicted_effect": o.prediction.expected_effect,
                "actual_effect": o.actual_effect,
                "delta": o.delta,
            })
        return result
