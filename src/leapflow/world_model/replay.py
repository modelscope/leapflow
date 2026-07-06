"""Experience Replay Engine — off-policy learning from historical experiences.

Discovers cross-time, cross-application patterns by reflecting on stored
prediction experiences using LLM-guided batch analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from leapflow.world_model.budget import LearningBudgetController
    from leapflow.world_model.experience_store import ExperienceStore, ExperienceTuple

from leapflow.llm.base import LLMProvider
from leapflow.llm.message_builder import build_system_message, build_user_message_text
from leapflow.world_model._json_utils import extract_json_object

logger = logging.getLogger(__name__)

_REFLECT_PROMPT = """\
You are analyzing past prediction experiences to discover patterns.

Focus: {focus}

Experiences:
{experiences_formatted}

Questions:
1. What general causal rules can you infer from these observations?
2. Are there cross-application patterns (e.g., same shortcut works differently)?
3. What exceptions or edge cases are revealed?
4. What action-effect pairs are most uncertain and need more observation?

Output JSON:
{{
  "insights": [
    {{
      "type": "causal_transfer" | "pattern_abstract" | "edge_correction",
      "description": "...",
      "confidence": 0.8,
      "actionable": true,
      "causal_rule": {{
        "parent_channel": "...",
        "child_channel": "...",
        "app_scope": "...",
        "confidence": 0.85
      }}
    }}
  ]
}}"""

_DISTILL_PROMPT = """\
You are distilling accumulated knowledge from graded experiences.

Focus: forking actions and advantage-weighted patterns.

Graded experiences (sorted by |advantage|):
{experiences_formatted}

Tasks:
1. For high-advantage (positive) actions: extract reusable heuristic rules.
2. For low-advantage (negative) actions: identify what went wrong and propose \
a corrective rule.
3. For forking actions: describe the decision point and the optimal choice.

Output JSON:
{{
  "distilled_rules": [
    {{
      "type": "heuristic" | "correction" | "forking_insight",
      "description": "...",
      "confidence": 0.8,
      "actionable": true,
      "source_grade": "optimal" | "harmful" | "forking"
    }}
  ]
}}"""


@dataclass(frozen=True)
class ReplayInsight:
    """A single insight discovered during experience replay."""

    insight_type: str
    description: str
    source_experiences: List[str]
    confidence: float
    actionable: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExperienceReplayEngine:
    """Off-policy learning engine that extracts insights from stored experiences.

    Operates asynchronously during idle periods or at session boundaries,
    using LLM reflection to discover patterns invisible to online learning.
    """

    def __init__(
        self,
        llm: LLMProvider,
        experience_store: "ExperienceStore",
        budget: "LearningBudgetController",
        *,
        on_insight: Optional[Any] = None,
        regression_sample_size: int = 200,
    ) -> None:
        self._llm = llm
        self._store = experience_store
        self._budget = budget
        self._on_insight = on_insight
        self._regression_sample_size = regression_sample_size

    async def replay_session(
        self,
        *,
        max_batches: int = 3,
        batch_size: int = 5,
    ) -> List[ReplayInsight]:
        """Run a full replay session with multiple strategies."""
        if not self._budget.has_tokens("replay"):
            return []

        insights: List[ReplayInsight] = []

        # Strategy 1: High prediction error experiences
        high_delta = self._store.retrieve_high_delta(delta_min=0.5, limit=batch_size)
        if high_delta:
            batch = await self._reflect_batch(high_delta, focus="prediction_errors")
            insights.extend(batch)

        # Strategy 2: Cross-app pattern discovery
        if len(insights) < max_batches:
            cross_app = self._collect_cross_app_experiences(limit=batch_size)
            if cross_app:
                batch = await self._reflect_batch(cross_app, focus="transfer_rules")
                insights.extend(batch)

        self._budget.spend("replay")
        self._notify_insights(insights)

        for exp_batch in [high_delta]:
            for exp in (exp_batch or []):
                self._store.increment_replay_count(exp.experience_id)

        return insights

    async def replay_targeted(
        self,
        app_context: str,
        *,
        batch_size: int = 5,
    ) -> List[ReplayInsight]:
        """Run a targeted replay for a specific application context."""
        if not self._budget.has_tokens("replay"):
            return []

        experiences = self._store.retrieve_similar(
            "", app_context, limit=batch_size,
        )
        if not experiences:
            return []

        insights = await self._reflect_batch(experiences, focus=f"app:{app_context}")
        self._budget.spend("replay")
        self._notify_insights(insights)
        return insights

    async def _reflect_batch(
        self,
        experiences: List["ExperienceTuple"],
        focus: str,
    ) -> List[ReplayInsight]:
        """Use LLM to reflect on a batch of experiences."""
        formatted = "\n".join(
            f"{i+1}. [{e.app_context}] Action: {e.action_description}\n"
            f"   Predicted: {e.predicted_effect}\n"
            f"   Actual: {e.actual_effect}\n"
            f"   Delta: {e.delta:.2f}"
            for i, e in enumerate(experiences)
        )

        prompt = _REFLECT_PROMPT.format(
            focus=focus,
            experiences_formatted=formatted,
        )

        try:
            resp = await self._llm.achat(
                [build_system_message("You are a learning reflection engine."),
                 build_user_message_text(prompt)],
                stream=False, enable_thinking=False,
            )
            return self._parse_insights(resp.content or "", experiences)
        except Exception:
            logger.debug("replay.reflect_batch failed", exc_info=True)
            return []

    def _parse_insights(
        self,
        response: str,
        source_experiences: List["ExperienceTuple"],
    ) -> List[ReplayInsight]:
        """Parse LLM reflection output into structured insights."""
        try:
            obj = extract_json_object(response)
            raw_insights = obj.get("insights", [])
        except Exception:
            return []

        source_ids = [e.experience_id for e in source_experiences]
        results: List[ReplayInsight] = []

        for raw in raw_insights:
            if not isinstance(raw, dict):
                continue
            insight = ReplayInsight(
                insight_type=str(raw.get("type", "pattern_abstract")),
                description=str(raw.get("description", "")),
                source_experiences=source_ids,
                confidence=float(raw.get("confidence", 0.5)),
                actionable=bool(raw.get("actionable", False)),
                metadata={
                    k: v for k, v in raw.items()
                    if k not in ("type", "description", "confidence", "actionable")
                },
            )
            results.append(insight)

        return results

    def _collect_cross_app_experiences(
        self, limit: int = 10,
    ) -> List["ExperienceTuple"]:
        """Collect experiences across different apps for cross-app pattern analysis."""
        all_high = self._store.retrieve_high_delta(delta_min=0.3, limit=limit * 2)
        if not all_high:
            return []

        by_app: Dict[str, List["ExperienceTuple"]] = {}
        for exp in all_high:
            by_app.setdefault(exp.app_context, []).append(exp)

        if len(by_app) < 2:
            return []

        result: List["ExperienceTuple"] = []
        for app_exps in by_app.values():
            result.extend(app_exps[:limit // len(by_app) + 1])
        return result[:limit]

    async def self_distill(
        self,
        *,
        batch_size: int = 10,
    ) -> List[ReplayInsight]:
        """OPD self-distillation: extract rules from advantage-graded experiences.

        Prioritises forking actions and high-|advantage| experiences, using
        the distillation prompt to produce heuristic rules, corrections,
        and forking insights.

        Uses the dedicated ``distillation`` budget pool, independent of replay.
        """
        if not self._budget.has_tokens("distillation"):
            return []

        graded = self._collect_graded_experiences(limit=batch_size)
        if len(graded) < 2:
            return []

        formatted = "\n".join(
            f"{i+1}. [{e.app_context}] Action: {e.action_description}\n"
            f"   Predicted: {e.predicted_effect}\n"
            f"   Actual: {e.actual_effect}\n"
            f"   Delta: {e.delta:.2f}  Advantage: {e.advantage:.2f}  "
            f"Grade: {e.grade_label}  Forking: {e.is_forking}"
            for i, e in enumerate(graded)
        )

        prompt = _DISTILL_PROMPT.format(experiences_formatted=formatted)

        try:
            resp = await self._llm.achat(
                [build_system_message("You are a knowledge distillation engine."),
                 build_user_message_text(prompt)],
                stream=False, enable_thinking=False,
            )
            insights = self._parse_distilled(resp.content or "", graded)
        except Exception:
            logger.debug("replay.self_distill failed", exc_info=True)
            insights = []

        self._budget.spend("distillation")
        self._notify_insights(insights)
        return insights

    def detect_regression(
        self,
        recent_outcomes: list,
        *,
        window: int = 5,
        regression_threshold: float = 0.15,
    ) -> bool:
        """Detect if recent prediction accuracy has regressed.

        Compares the mean delta of the last *window* outcomes against the
        mean of all stored experiences. Returns True when the recent window
        is significantly worse (higher delta) than the historical baseline.
        """
        if len(recent_outcomes) < window:
            return False

        recent_deltas = [getattr(o, "delta", 0.5) for o in recent_outcomes[-window:]]
        recent_mean = sum(recent_deltas) / len(recent_deltas)

        all_exps = self._store.retrieve_high_delta(delta_min=0.0, limit=self._regression_sample_size)
        if len(all_exps) < window:
            return False

        hist_mean = sum(e.delta for e in all_exps) / len(all_exps)

        return recent_mean > hist_mean + regression_threshold

    def _collect_graded_experiences(
        self,
        limit: int = 10,
    ) -> List["ExperienceTuple"]:
        """Collect experiences that have been graded, sorted by |advantage|."""
        all_exps = self._store.retrieve_high_delta(delta_min=0.0, limit=limit * 3)
        graded = [e for e in all_exps if e.grade_label]
        graded.sort(key=lambda e: abs(e.advantage), reverse=True)
        return graded[:limit]

    def _parse_distilled(
        self,
        response: str,
        source_experiences: List["ExperienceTuple"],
    ) -> List[ReplayInsight]:
        """Parse distillation response into ReplayInsight objects."""
        try:
            obj = extract_json_object(response)
            raw_rules = obj.get("distilled_rules", [])
        except Exception:
            return []

        source_ids = [e.experience_id for e in source_experiences]
        results: List[ReplayInsight] = []
        for raw in raw_rules:
            if not isinstance(raw, dict):
                continue
            results.append(ReplayInsight(
                insight_type=str(raw.get("type", "heuristic")),
                description=str(raw.get("description", "")),
                source_experiences=source_ids,
                confidence=float(raw.get("confidence", 0.5)),
                actionable=bool(raw.get("actionable", True)),
                metadata={
                    k: v for k, v in raw.items()
                    if k not in ("type", "description", "confidence", "actionable")
                },
            ))
        return results

    def set_replay_priorities(self, grades: list) -> None:
        """Accept trajectory grades and mark high-value experiences for replay.

        High-|advantage| experiences (informative learning signals) get their
        replay count incremented, bringing them to the top of retrieval
        queries that sort by replay_count descending.
        """
        prioritized = 0
        for grade in grades:
            exp_id = getattr(grade, "experience_id", None)
            advantage = abs(getattr(grade, "advantage", 0.0))
            if exp_id and advantage > 0.3:
                try:
                    self._store.increment_replay_count(exp_id)
                    prioritized += 1
                except Exception:
                    logger.debug("replay.priority_update failed for %s", exp_id)
        logger.debug("set_replay_priorities: prioritized %d/%d grades", prioritized, len(grades))

    def _notify_insights(self, insights: List[ReplayInsight]) -> None:
        """Notify downstream consumers of discovered insights."""
        if not insights or self._on_insight is None:
            return
        for insight in insights:
            if insight.actionable:
                try:
                    self._on_insight(insight)
                except Exception:
                    logger.debug("replay.on_insight callback failed", exc_info=True)
