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
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from leapflow.world_model.budget import LearningBudgetController
    from leapflow.world_model.experience_store import ExperienceStore

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


@dataclass(frozen=True)
class ActionGrade:
    """Teacher's grading of a single action step."""

    experience_id: str
    advantage: float
    is_forking: bool
    grade_label: str


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
        raw_grades = await self._call_teacher(traj_text, goal)
        self._budget.spend("grading")

        grades = self._persist_grades(trajectory, raw_grades)
        return grades

    async def _call_teacher(
        self,
        trajectory_text: str,
        goal: str,
    ) -> List[ActionGrade]:
        """Single LLM call: teacher grades with full hindsight."""
        labels_str = ", ".join(f'"{l}"' for l in self._grade_labels)
        prompt = _GRADE_PROMPT.format(
            goal=goal or "(not specified)",
            trajectory_text=trajectory_text,
            grade_labels=labels_str,
            example_label=self._grade_labels[1] if len(self._grade_labels) > 1 else self._grade_labels[0],
        )
        try:
            resp = await self._llm.achat(
                [build_system_message(
                    "You are a trajectory evaluator with perfect hindsight.",
                ),
                 build_user_message_text(prompt)],
                stream=False, enable_thinking=False,
            )
            return self._parse_grades(resp.content or "")
        except Exception:
            logger.debug("trajectory_grader.call_teacher failed", exc_info=True)
            return []

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

    def _parse_grades(self, response: str) -> List[ActionGrade]:
        """Parse teacher response into ActionGrade objects."""
        obj = extract_json_object(response)
        raw_grades = obj.get("grades", [])
        results: List[ActionGrade] = []
        for raw in raw_grades:
            if not isinstance(raw, dict):
                continue
            results.append(ActionGrade(
                experience_id="",
                advantage=max(-1.0, min(1.0, float(raw.get("advantage", 0)))),
                is_forking=bool(raw.get("is_forking", False)),
                grade_label=str(raw.get("grade_label", "acceptable")),
            ))
        return results

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
