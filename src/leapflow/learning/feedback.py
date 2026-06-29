"""Feedback loop — detect re-executions of stored skills and auto-improve.

When the active learning observer detects a candidate matching an existing skill
at high confidence (>=0.85), this module computes a structural diff between the
stored skill and the new observation, then decides whether to auto-apply, suggest,
or skip the improvement.

Two-layer comparison baseline:
    1. Original trajectory episodes (preferred, via TrajectoryStore)
    2. StoredSkill fields (fallback when trajectory is unavailable)

Three-phase verdict pipeline:
    Phase 1: Structural diff (action-level + step-level)
    Phase 2: Heuristic rules (zero LLM cost)
    Phase 3: LLM refinement (optional, when heuristic confidence is low)
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from leapflow.domain.trajectory import Episode
from leapflow.learning.active_learning import _lcs
from leapflow.domain.skill_types import DistillationCandidate
from leapflow.storage.skill_library import SkillLibraryStore, StoredSkill

logger = logging.getLogger(__name__)


# ── Data models ──


@dataclass(frozen=True)
class TrajectoryDiff:
    """Structural difference between a stored skill and a new execution."""

    added_actions: List[str]
    removed_actions: List[str]
    shared_actions: List[str]
    added_steps: List[str]
    removed_steps: List[str]
    step_count_delta: int
    app_set_changed: bool
    new_apps: List[str]
    new_triggers: List[str]
    diff_hash: str


@dataclass(frozen=True)
class FeedbackVerdict:
    """Improvement assessment for a skill re-execution."""

    verdict: str  # improved | unchanged | regressed
    improvement_type: str  # additive | efficiency | structural | none
    confidence: float
    auto_apply: bool
    description: str
    llm_rationale: str = ""


# ── Evaluator ──


class FeedbackEvaluator:
    """Compares new skill observations against stored skills to drive improvement.

    Decision flow:
        1. Compute structural diff (action-level + step-level)
        2. Apply heuristic rules to determine verdict
        3. Optionally refine with LLM when heuristic confidence is low
    """

    def __init__(
        self,
        trajectory_store: Any,
        *,
        llm: Optional[Any] = None,
        auto_apply_min_confidence: float = 0.8,
        multi_observe_threshold: int = 2,
    ) -> None:
        self._traj_store = trajectory_store
        self._llm = llm
        self._auto_apply_min_confidence = auto_apply_min_confidence
        self._multi_observe_threshold = multi_observe_threshold

    def evaluate(
        self,
        skill: StoredSkill,
        new_candidate: DistillationCandidate,
        new_episode: Episode,
    ) -> Tuple[TrajectoryDiff, FeedbackVerdict]:
        """Synchronous heuristic-only evaluation."""
        diff = self._compute_diff(skill, new_candidate, new_episode)
        verdict = _heuristic_verdict(diff)
        return diff, verdict

    async def evaluate_async(
        self,
        skill: StoredSkill,
        new_candidate: DistillationCandidate,
        new_episode: Episode,
    ) -> Tuple[TrajectoryDiff, FeedbackVerdict]:
        """Full evaluation with optional LLM refinement."""
        diff = self._compute_diff(skill, new_candidate, new_episode)
        verdict = _heuristic_verdict(diff)
        if self.needs_llm_refinement(verdict):
            verdict = await self.llm_verdict(
                diff, skill, new_candidate, verdict
            )
        return diff, verdict

    def needs_llm_refinement(self, verdict: FeedbackVerdict) -> bool:
        return (
            self._llm is not None
            and verdict.verdict == "improved"
            and verdict.confidence < self._auto_apply_min_confidence
        )

    def should_auto_apply_with_history(
        self,
        verdict: FeedbackVerdict,
        diff: TrajectoryDiff,
        skill_id: str,
        store: SkillLibraryStore,
    ) -> bool:
        if verdict.auto_apply:
            return True
        if verdict.verdict != "improved":
            return False
        prior = store.count_by_diff_hash(skill_id, diff.diff_hash)
        return prior + 1 >= self._multi_observe_threshold

    # ── Diff computation ──

    def _compute_diff(
        self,
        skill: StoredSkill,
        candidate: DistillationCandidate,
        episode: Episode,
    ) -> TrajectoryDiff:
        orig_actions, orig_descs = self._load_original_episode_data(skill)
        new_actions = [a.action_name for a in episode.semantic_actions]
        new_descs = [a.description for a in episode.semantic_actions]

        shared = _lcs(orig_actions, new_actions)

        orig_counts = Counter(orig_actions)
        new_counts = Counter(new_actions)
        added = list((new_counts - orig_counts).elements())
        removed = list((orig_counts - new_counts).elements())

        orig_desc_set = set(orig_descs)
        new_desc_set = set(new_descs)

        hash_input = (
            "|".join(sorted(added)) + "||" + "|".join(sorted(removed))
        )
        diff_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]

        existing_triggers = set(skill.trigger_phrases)
        new_triggers = [
            t for t in candidate.trigger_phrases if t not in existing_triggers
        ]

        skill_app_set = set(skill.app_sequence)

        return TrajectoryDiff(
            added_actions=added,
            removed_actions=removed,
            shared_actions=shared,
            added_steps=[d for d in new_descs if d not in orig_desc_set],
            removed_steps=[d for d in orig_descs if d not in new_desc_set],
            step_count_delta=len(new_actions) - len(orig_actions),
            app_set_changed=skill_app_set != set(episode.app_sequence),
            new_apps=[
                a for a in episode.app_sequence if a not in skill_app_set
            ],
            new_triggers=new_triggers,
            diff_hash=diff_hash,
        )

    def _load_original_episode_data(
        self, skill: StoredSkill
    ) -> Tuple[List[str], List[str]]:
        """Load original action names and descriptions in a single query.

        Prefers original trajectory episodes; falls back to StoredSkill fields.
        """
        if skill.source_trajectory_id and self._traj_store is not None:
            episodes = self._traj_store.load_episodes(
                skill.source_trajectory_id
            )
            for ep in episodes:
                if ep.episode_id == skill.source_episode_id:
                    actions = [a.action_name for a in ep.semantic_actions]
                    descs = [a.description for a in ep.semantic_actions]
                    return actions, descs
        return list(skill.action_names), list(skill.steps)

    # ── LLM-enhanced verdict ──

    async def llm_verdict(
        self,
        diff: TrajectoryDiff,
        skill: StoredSkill,
        candidate: DistillationCandidate,
        heuristic: FeedbackVerdict,
    ) -> FeedbackVerdict:
        if self._llm is None:
            return heuristic
        from leapflow.llm.message_builder import (
            build_system_message,
            build_user_message_text,
        )

        prompt = (
            "Evaluate whether this change to a desktop automation skill is an improvement.\n\n"
            f"Original skill: {skill.title}\n"
            f"  Steps: {skill.steps}\n"
            f"  Apps: {skill.app_sequence}\n\n"
            f"New execution observation:\n"
            f"  Steps: {candidate.steps}\n\n"
            f"Structural diff:\n"
            f"  Added steps: {diff.added_steps}\n"
            f"  Removed steps: {diff.removed_steps}\n"
            f"  App set changed: {diff.app_set_changed}\n\n"
            'Return JSON: {"verdict": "improved|regressed|unchanged", '
            '"improvement_type": "additive|efficiency|structural|none", '
            '"confidence": 0.0-1.0, "auto_apply": true/false, '
            '"rationale": "..."}'
        )
        try:
            resp = await self._llm.achat(
                [
                    build_system_message(
                        "You are a skill improvement evaluator. Return ONLY JSON."
                    ),
                    build_user_message_text(prompt),
                ],
                stream=True,
                enable_thinking=False,
            )
            return _parse_llm_verdict(resp.content or "", heuristic)
        except Exception:
            logger.debug(
                "LLM feedback verdict failed; using heuristic", exc_info=True
            )
            return heuristic


# ── Heuristic rules ──


def _heuristic_verdict(diff: TrajectoryDiff) -> FeedbackVerdict:
    has_added = len(diff.added_actions) > 0
    has_removed = len(diff.removed_actions) > 0

    if not has_added and not has_removed:
        return FeedbackVerdict(
            verdict="unchanged",
            improvement_type="none",
            confidence=1.0,
            auto_apply=False,
            description="Execution matches stored skill exactly.",
        )

    if diff.app_set_changed:
        return FeedbackVerdict(
            verdict="improved",
            improvement_type="structural",
            confidence=0.4,
            auto_apply=False,
            description=(
                f"App context changed: +{diff.new_apps}. "
                f"+{len(diff.added_actions)} actions, "
                f"-{len(diff.removed_actions)} actions."
            ),
        )

    if has_added and not has_removed:
        return FeedbackVerdict(
            verdict="improved",
            improvement_type="additive",
            confidence=0.85,
            auto_apply=True,
            description=(
                f"Purely additive: +{len(diff.added_actions)} new actions "
                f"({', '.join(diff.added_actions[:3])})."
            ),
        )

    if not has_added and has_removed:
        return FeedbackVerdict(
            verdict="improved",
            improvement_type="efficiency",
            confidence=0.6,
            auto_apply=False,
            description=(
                f"Efficiency improvement: -{len(diff.removed_actions)} actions "
                f"({', '.join(diff.removed_actions[:3])})."
            ),
        )

    # has_added and has_removed
    return FeedbackVerdict(
        verdict="improved",
        improvement_type="structural",
        confidence=0.5,
        auto_apply=False,
        description=(
            f"Structural change: +{len(diff.added_actions)}, "
            f"-{len(diff.removed_actions)} actions."
        ),
    )


def _parse_llm_verdict(raw: str, fallback: FeedbackVerdict) -> FeedbackVerdict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return fallback
    try:
        data = json.loads(raw[start : end + 1])
        return FeedbackVerdict(
            verdict=str(data.get("verdict", fallback.verdict)),
            improvement_type=str(
                data.get("improvement_type", fallback.improvement_type)
            ),
            confidence=float(data.get("confidence", fallback.confidence)),
            auto_apply=bool(data.get("auto_apply", False)),
            description=fallback.description,
            llm_rationale=str(data.get("rationale", "")),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return fallback
