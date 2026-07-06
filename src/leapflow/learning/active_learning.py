"""Active learning — detect near-match skills and suggest updates.

Two-phase similarity pipeline:
    Phase 1: HeuristicSimilarityScorer fast-filter (zero LLM cost)
    Phase 2: LLMSimilarityScorer.refine() on narrow candidate set (optional)

Observer is registered as a pipeline callback; SkillMerger handles
user-approved updates.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set

if TYPE_CHECKING:
    from leapflow.analysis.consensus import MultiTrajectoryDistiller
    from leapflow.skills.activator import SkillActivator
    from leapflow.learning.doc_generator import SkillDocGenerator
    from leapflow.storage.skill_docs import SkillDocStore
    from leapflow.world_model.curiosity import CuriosityScore
    from leapflow.world_model.prediction import PredictionOutcome
    from leapflow.learning.feedback import (
        FeedbackEvaluator,
        FeedbackVerdict,
        TrajectoryDiff,
    )
    from leapflow.skills.registry import SkillRegistry

from leapflow.domain.trajectory import Episode
from leapflow.memory.providers.working import WorkingMemoryProvider
from leapflow.domain.skill_types import DistillationCandidate
from leapflow.storage.skill_library import (
    SkillExecution,
    SkillLibraryStore,
    SkillUpdateSuggestion,
    StoredSkill,
    deserialize_candidate,
    serialize_candidate,
)
from leapflow.learning.similarity import (
    HeuristicSimilarityScorer,
    LLMSimilarityScorer,
    SimilarityResult,
)

logger = logging.getLogger(__name__)

_MIN_ACTIVATION_CONFIDENCE = 0.6


# ── Observer ──


class ActiveLearningObserver:
    """Pipeline callback: compares new candidates against stored skills.

    Decision flow per candidate:
        score < heuristic_low  → new skill, save directly
        score >= heuristic_high → already exists, skip
        score in [low, high)   → ambiguous zone:
            with LLM  → refine then apply final decision
            without LLM → heuristic score is final
    """

    def __init__(
        self,
        skill_store: SkillLibraryStore,
        scorer: HeuristicSimilarityScorer,
        working_memory: WorkingMemoryProvider,
        *,
        llm_scorer: Optional[LLMSimilarityScorer] = None,
        feedback_evaluator: Optional["FeedbackEvaluator"] = None,
        skill_activator: Optional["SkillActivator"] = None,
        consensus_distiller: Optional["MultiTrajectoryDistiller"] = None,
        doc_generator: Optional["SkillDocGenerator"] = None,
        doc_store: Optional["SkillDocStore"] = None,
        skill_registry: Optional["SkillRegistry"] = None,
        llm: Optional[object] = None,
        execution: Optional[object] = None,
        heuristic_low: float = 0.3,
        heuristic_high: float = 0.75,
        final_low: float = 0.5,
        final_high: float = 0.85,
    ) -> None:
        self._store = skill_store
        self._scorer = scorer
        self._wm = working_memory
        self._llm_scorer = llm_scorer
        self._feedback = feedback_evaluator
        self._activator = skill_activator
        self._consensus = consensus_distiller
        self._doc_generator = doc_generator
        self._doc_store = doc_store
        self._skill_registry = skill_registry
        self._llm = llm
        self._execution = execution
        self._h_low = heuristic_low
        self._h_high = heuristic_high
        self._f_low = final_low
        self._f_high = final_high
        self._pending_activations: List[asyncio.Task] = []
        self._activated_keys: Set[str] = set()
        self._activated_trajectories: Set[str] = set()
        self._high_curiosity_apps: Set[str] = set()

    def on_candidates_ready(
        self,
        candidates: List[DistillationCandidate],
        episodes: List[Episode],
    ) -> None:
        """Synchronous entry point called by ImitationPipeline."""
        consolidated = self._consolidate_session_candidates(candidates, episodes)
        for candidate, episode in consolidated:
            self._process_candidate(candidate, episode)
        self._detect_repeated_patterns(episodes)

    def on_pattern_candidate(
        self, candidates: list,
    ) -> None:
        """Receive SkillCandidates from PatternMiner and process them.

        Bridges the PatternMiner → ActiveLearning gap: converts mined
        SkillCandidate objects into DistillationCandidate format and
        runs through the standard similarity pipeline.
        """
        for candidate in candidates:
            title = getattr(candidate, "title", "")
            steps = getattr(candidate, "steps", [])
            triggers = getattr(candidate, "trigger_phrases", [])
            confidence = getattr(candidate, "confidence", 0.5)

            if confidence < _MIN_ACTIVATION_CONFIDENCE or not title:
                continue

            dc = DistillationCandidate(
                title=title,
                trigger_phrases=list(triggers),
                steps=list(steps),
                parameters=[],
                pre_conditions=[],
                post_conditions=[],
                source_trajectory_id="",
                source_episode_id="",
                confidence=confidence,
            )

            synthetic_episode = Episode(
                trajectory_id="", start_idx=0, end_idx=0,
                app_sequence=[], semantic_actions=[],
                inferred_goal=title, confidence=confidence,
            )

            all_skills = self._store.load_all_active()
            if not all_skills:
                self._store.save_from_candidate(dc, [], [])
                logger.info("pattern_miner.new_skill title=%s conf=%.2f", title, confidence)
                continue

            matches = self._scorer.find_similar(dc, all_skills, threshold=self._h_low)
            if not matches:
                self._store.save_from_candidate(dc, [], [])
                logger.info("pattern_miner.new_skill title=%s conf=%.2f", title, confidence)
            elif matches[0].overall_score >= self._h_high:
                logger.debug(
                    "pattern_miner.skip_existing title=%s match=%s score=%.2f",
                    title, matches[0].stored_skill_title, matches[0].overall_score,
                )
            else:
                suggestion = self._build_suggestion(dc, matches[0], synthetic_episode)
                self._store.save_suggestion(suggestion)
                self._notify(suggestion)
                logger.info("pattern_miner.suggestion title=%s score=%.2f", title, matches[0].overall_score)

    def on_curiosity_signal(
        self, score: "CuriosityScore", outcome: "PredictionOutcome",
    ) -> None:
        """Callback triggered by PredictionLoop when curiosity is high.

        Integrates world model curiosity signals into the active learning
        workflow: high curiosity events are remembered, patterns with
        low novelty but high surprise trigger proactive skill review,
        and high-curiosity app contexts are accumulated for targeted
        replay at session end.
        """
        if score.total > 0.7:
            action_desc = outcome.prediction.action_description if outcome.prediction else "unknown"
            app_id = getattr(outcome.pre_snapshot, "app_bundle_id", "unknown") if outcome.pre_snapshot else "unknown"
            self._wm.remember_event(
                "curiosity_alert",
                f"High curiosity ({score.total:.2f}) for action: {action_desc}",
                {"delta": outcome.delta, "app": app_id},
            )
            self._high_curiosity_apps.add(app_id)

        if score.frequency_novelty < 0.3 and score.prediction_surprise > 0.5:
            self._wm.remember_event(
                "proactive_review_hint",
                f"Familiar pattern with surprising outcome "
                f"(fn={score.frequency_novelty:.2f}, ps={score.prediction_surprise:.2f}) — "
                f"skill may need updating",
            )

    def drain_high_curiosity_apps(self) -> Set[str]:
        """Return and clear the set of app contexts with high curiosity.

        Called at session end to trigger targeted replay for these domains.
        """
        apps = set(self._high_curiosity_apps)
        self._high_curiosity_apps.clear()
        return apps

    def _handle_regression(
        self,
        skill: "StoredSkill",
        candidate: DistillationCandidate,
        episode: Episode,
    ) -> None:
        """Respond to a skill regression: lower confidence and queue for replay.

        Rather than silently ignoring regression, this:
        1. Reduces the stored skill's confidence proportionally
        2. Marks the app context as high-curiosity (triggers targeted replay)
        3. Records a proactive-review event so the user is aware
        """
        _REGRESSION_PENALTY = 0.15
        new_conf = max(0.1, skill.confidence - _REGRESSION_PENALTY)
        if new_conf != skill.confidence:
            downgraded = StoredSkill(
                skill_id=skill.skill_id,
                title=skill.title,
                trigger_phrases=skill.trigger_phrases,
                steps=skill.steps,
                parameters=skill.parameters,
                pre_conditions=skill.pre_conditions,
                post_conditions=skill.post_conditions,
                app_sequence=skill.app_sequence,
                action_names=skill.action_names,
                source_trajectory_id=skill.source_trajectory_id,
                source_episode_id=skill.source_episode_id,
                confidence=new_conf,
                version=skill.version,
                status=skill.status,
                created_at=skill.created_at,
            )
            self._store.save_skill(downgraded)

        for app in episode.app_sequence:
            self._high_curiosity_apps.add(app)

        self._wm.remember_event(
            "skill_regression",
            f"Skill '{skill.title}' regressed (confidence {skill.confidence:.2f}"
            f" → {new_conf:.2f}). Queued for targeted replay.",
        )

    def _consolidate_session_candidates(
        self,
        candidates: List[DistillationCandidate],
        episodes: List[Episode],
    ) -> List[tuple[DistillationCandidate, Episode]]:
        """Merge candidates from the same trajectory into a single skill.

        A learn session produces one trajectory; splitting into multiple skills
        is almost always wrong. Merge by trajectory_id using LCS step alignment.
        """
        if len(candidates) <= 1:
            return list(zip(candidates, episodes))

        groups: Dict[str, List[tuple[DistillationCandidate, Episode]]] = {}
        for c, ep in zip(candidates, episodes):
            key = c.source_trajectory_id or ""
            groups.setdefault(key, []).append((c, ep))

        result: List[tuple[DistillationCandidate, Episode]] = []
        for key, group in groups.items():
            if not key or len(group) == 1:
                result.extend(group)
                continue
            merged_c, merged_ep = self._merge_candidate_group(group)
            result.append((merged_c, merged_ep))
            logger.info(
                "session_consolidation: merged %d candidates from trajectory=%s",
                len(group), key,
            )
        return result

    @staticmethod
    def _merge_candidate_group(
        group: List[tuple[DistillationCandidate, Episode]],
    ) -> tuple[DistillationCandidate, Episode]:
        """Merge multiple candidates from the same session into one."""
        best_idx = max(range(len(group)), key=lambda i: group[i][0].confidence)
        best_c, best_ep = group[best_idx]

        all_steps: List[str] = list(best_c.steps)
        all_triggers: List[str] = list(best_c.trigger_phrases)
        all_params: List[Dict[str, str]] = list(best_c.parameters)
        all_pre: List[str] = list(best_c.pre_conditions)
        all_apps: List[str] = list(best_ep.app_sequence)
        max_conf = best_c.confidence

        for i, (c, ep) in enumerate(group):
            if i == best_idx:
                continue
            all_steps = _merge_steps_lcs(all_steps, list(c.steps))
            all_triggers = _union_dedup(all_triggers, list(c.trigger_phrases))
            all_params = _merge_params(all_params, list(c.parameters))
            all_pre = _union_dedup(all_pre, list(c.pre_conditions))
            all_apps = _union_dedup(all_apps, list(ep.app_sequence))
            max_conf = max(max_conf, c.confidence)

        merged = DistillationCandidate(
            title=best_c.title,
            trigger_phrases=all_triggers,
            steps=all_steps,
            parameters=all_params,
            pre_conditions=all_pre,
            post_conditions=list(best_c.post_conditions),
            source_trajectory_id=best_c.source_trajectory_id,
            source_episode_id=best_c.source_episode_id,
            confidence=max_conf,
            recovery_events=list(best_c.recovery_events),
            anchor_candidates=list(best_c.anchor_candidates),
            procedure_graph=best_c.procedure_graph,
        )

        merged_ep = Episode(
            trajectory_id=best_ep.trajectory_id,
            start_idx=min(ep.start_idx for _, ep in group),
            end_idx=max(ep.end_idx for _, ep in group),
            app_sequence=all_apps,
            semantic_actions=[a for _, ep in group for a in ep.semantic_actions],
            inferred_goal=best_ep.inferred_goal,
            confidence=max_conf,
        )
        return merged, merged_ep

    def _detect_repeated_patterns(self, episodes: List[Episode]) -> None:
        """Proactive learning: detect repeated app sequences not matching any skill."""
        for ep in episodes:
            if not ep.app_sequence:
                continue
            key = ",".join(ep.app_sequence)
            count = self._wm.increment_pattern(key)
            if count == 3:
                all_skills = self._store.load_all_active()
                if not any(
                    set(s.app_sequence) == set(ep.app_sequence)
                    for s in all_skills
                ):
                    self._wm.remember_event(
                        "proactive_learn_hint",
                        f"Repeated pattern detected ({count}x): "
                        f"apps={ep.app_sequence}. "
                        f"Consider learning this as a skill.",
                    )

    def _process_candidate(
        self, candidate: DistillationCandidate, episode: Episode
    ) -> None:
        all_skills = self._store.load_all_active()
        app_seq = list(episode.app_sequence)
        action_names = [a.action_name for a in episode.semantic_actions]

        if not all_skills:
            self._store.save_from_candidate(candidate, app_seq, action_names)
            self._schedule_activation(candidate, episode)
            return

        matches = self._scorer.find_similar(
            candidate, all_skills, threshold=self._h_low
        )

        if not matches:
            self._store.save_from_candidate(candidate, app_seq, action_names)
            self._schedule_activation(candidate, episode)
            return

        best = matches[0]
        if best.overall_score >= self._h_high:
            if self._feedback is not None:
                self._evaluate_feedback(candidate, episode, best)
            else:
                logger.debug(
                    "active_learning.skip existing=%s score=%.2f",
                    best.stored_skill_title,
                    best.overall_score,
                )
            return

        # Consensus distillation requires LLM for human-readable output
        if self._consensus is not None and self._llm is not None:
            self._schedule_consensus_distill(candidate, episode, matches)
        elif self._llm_scorer is not None:
            self._schedule_llm_refine(candidate, episode, matches)
        else:
            self._apply_final_decision(
                candidate, episode, matches, app_seq, action_names
            )

    def _apply_final_decision(
        self,
        candidate: DistillationCandidate,
        episode: Episode,
        matches: List[SimilarityResult],
        app_seq: Optional[List[str]] = None,
        action_names: Optional[List[str]] = None,
    ) -> None:
        app_seq = app_seq or list(episode.app_sequence)
        action_names = action_names or [
            a.action_name for a in episode.semantic_actions
        ]
        best = matches[0]
        score = best.overall_score

        if score < self._f_low:
            self._store.save_from_candidate(candidate, app_seq, action_names)
            self._schedule_activation(candidate, episode)
        elif score >= self._f_high:
            logger.debug(
                "active_learning.skip_final existing=%s score=%.2f",
                best.stored_skill_title,
                score,
            )
        else:
            suggestion = self._build_suggestion(candidate, best, episode)
            self._store.save_suggestion(suggestion)
            self._notify(suggestion)

    def _schedule_llm_refine(
        self,
        candidate: DistillationCandidate,
        episode: Episode,
        matches: List[SimilarityResult],
    ) -> None:
        app_seq = list(episode.app_sequence)
        action_names = [a.action_name for a in episode.semantic_actions]
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._llm_refine(
                    candidate, episode, matches, app_seq, action_names
                )
            )
        except RuntimeError:
            self._apply_final_decision(
                candidate, episode, matches, app_seq, action_names
            )

    def _schedule_consensus_distill(
        self,
        candidate: DistillationCandidate,
        episode: Episode,
        matches: List[SimilarityResult],
    ) -> None:
        """Schedule consensus distillation using multiple trajectories."""
        app_seq = list(episode.app_sequence)
        action_names = [a.action_name for a in episode.semantic_actions]
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._consensus_distill(
                    candidate, episode, matches, app_seq, action_names
                )
            )
        except RuntimeError:
            # No event loop — fall back to LLM refine or heuristic
            if self._llm_scorer is not None:
                self._schedule_llm_refine(candidate, episode, matches)
            else:
                self._apply_final_decision(
                    candidate, episode, matches, app_seq, action_names
                )

    async def _consensus_distill(
        self,
        candidate: DistillationCandidate,
        episode: Episode,
        matches: List[SimilarityResult],
        app_seq: List[str],
        action_names: List[str],
    ) -> None:
        """Run multi-trajectory consensus distillation."""
        assert self._consensus is not None
        try:
            # Gather trajectory IDs: current + matched skill's source
            trajectory_ids: List[str] = []
            if episode.trajectory_id:
                trajectory_ids.append(episode.trajectory_id)

            best = matches[0]
            existing_skill = self._store.load_skill(best.stored_skill_id)
            if existing_skill and existing_skill.source_trajectory_id:
                trajectory_ids.append(existing_skill.source_trajectory_id)

            # Need at least 2 trajectories for consensus
            if len(trajectory_ids) < 2:
                # Fall back to LLM refine or heuristic
                if self._llm_scorer is not None:
                    refined = await self._llm_scorer.refine(candidate, matches)
                    self._apply_final_decision(
                        candidate, episode, refined, app_seq, action_names
                    )
                else:
                    self._apply_final_decision(
                        candidate, episode, matches, app_seq, action_names
                    )
                return

            consensus_candidate = await self._consensus.distill_consensus(trajectory_ids)

            if consensus_candidate is not None:
                # Use consensus result — treat as a new higher-quality candidate
                logger.info(
                    "consensus_distill.success trajectories=%d steps=%d confidence=%.2f",
                    len(trajectory_ids),
                    len(consensus_candidate.steps),
                    consensus_candidate.confidence,
                )
                # Save the consensus skill directly (it's already cleaned)
                self._store.save_from_candidate(
                    consensus_candidate, app_seq, action_names
                )
                self._schedule_activation(consensus_candidate, episode)
            else:
                # Consensus failed — fall back to normal flow
                if self._llm_scorer is not None:
                    refined = await self._llm_scorer.refine(candidate, matches)
                    self._apply_final_decision(
                        candidate, episode, refined, app_seq, action_names
                    )
                else:
                    self._apply_final_decision(
                        candidate, episode, matches, app_seq, action_names
                    )

        except Exception:
            logger.debug("Consensus distillation failed; falling back", exc_info=True)
            if self._llm_scorer is not None:
                try:
                    refined = await self._llm_scorer.refine(candidate, matches)
                    self._apply_final_decision(
                        candidate, episode, refined, app_seq, action_names
                    )
                except Exception:
                    self._apply_final_decision(
                        candidate, episode, matches, app_seq, action_names
                    )
            else:
                self._apply_final_decision(
                    candidate, episode, matches, app_seq, action_names
                )

    async def _llm_refine(
        self,
        candidate: DistillationCandidate,
        episode: Episode,
        matches: List[SimilarityResult],
        app_seq: List[str],
        action_names: List[str],
    ) -> None:
        try:
            assert self._llm_scorer is not None
            refined = await self._llm_scorer.refine(candidate, matches)
            self._apply_final_decision(
                candidate, episode, refined, app_seq, action_names
            )
        except Exception:
            logger.debug("LLM refine failed; using heuristic", exc_info=True)
            self._apply_final_decision(
                candidate, episode, matches, app_seq, action_names
            )

    def _build_suggestion(
        self,
        candidate: DistillationCandidate,
        match: SimilarityResult,
        episode: Episode,
    ) -> SkillUpdateSuggestion:
        return SkillUpdateSuggestion(
            suggestion_id=uuid.uuid4().hex[:16],
            existing_skill_id=match.stored_skill_id,
            existing_skill_title=match.stored_skill_title,
            new_candidate_json=serialize_candidate(candidate),
            similarity_score=match.overall_score,
            similarity_details={
                "action_sequence": match.action_sequence_score,
                "app_set": match.app_set_score,
                "goal_text": match.goal_text_score,
                "llm_score": match.llm_score,
                "llm_rationale": match.llm_rationale,
            },
            suggestion_type="update_existing",
            proposed_changes=_compute_changes(candidate, match),
            source_trajectory_id=episode.trajectory_id,
            source_episode_id=episode.episode_id,
        )

    def _notify(self, suggestion: SkillUpdateSuggestion) -> None:
        msg = (
            f"Skill update suggestion: '{suggestion.existing_skill_title}' "
            f"({suggestion.similarity_score:.0%} match). "
            f"Say 'review skill suggestions' to review."
        )
        self._wm.remember_event("skill_suggestion", msg)
        logger.info(
            "active_learning.suggestion skill=%s score=%.2f",
            suggestion.existing_skill_title,
            suggestion.similarity_score,
        )

    # ── Feedback loop ──

    def _evaluate_feedback(
        self,
        candidate: DistillationCandidate,
        episode: Episode,
        match: SimilarityResult,
    ) -> None:
        assert self._feedback is not None
        skill = self._store.load_skill(match.stored_skill_id)
        if skill is None:
            return

        diff, verdict = self._feedback.evaluate(skill, candidate, episode)

        if self._feedback.needs_llm_refinement(verdict):
            self._schedule_llm_feedback(
                candidate, episode, match, skill, diff, verdict
            )
        else:
            self._apply_feedback_verdict(
                candidate, episode, match, skill, diff, verdict
            )

    def _apply_feedback_verdict(
        self,
        candidate: DistillationCandidate,
        episode: Episode,
        match: SimilarityResult,
        skill: StoredSkill,
        diff: TrajectoryDiff,
        verdict: FeedbackVerdict,
    ) -> None:
        assert self._feedback is not None
        execution = SkillExecution(
            execution_id=uuid.uuid4().hex[:16],
            skill_id=skill.skill_id,
            trajectory_id=episode.trajectory_id,
            episode_id=episode.episode_id,
            similarity_score=match.overall_score,
            diff_hash=diff.diff_hash,
            diff_summary={
                "added_steps": diff.added_steps,
                "removed_steps": diff.removed_steps,
                "step_count_delta": diff.step_count_delta,
            },
            verdict=verdict.verdict,
        )

        if verdict.verdict == "unchanged":
            self._store.save_execution(execution)
            return

        if verdict.verdict == "regressed":
            self._store.save_execution(execution)
            self._handle_regression(skill, candidate, episode)
            logger.info("feedback.regressed skill=%s", skill.title)
            return

        # verdict == "improved"
        should_auto = self._feedback.should_auto_apply_with_history(
            verdict, diff, skill.skill_id, self._store
        )

        if should_auto:
            self._auto_apply_improvement(
                skill, candidate, episode, diff, verdict
            )
            execution.verdict = "auto_applied"
        else:
            suggestion = self._build_feedback_suggestion(
                candidate, match, episode, diff, verdict
            )
            self._store.save_suggestion(suggestion)
            self._notify_improvement(suggestion, verdict)
            execution.verdict = "suggested"

        self._store.save_execution(execution)

    def _schedule_llm_feedback(
        self,
        candidate: DistillationCandidate,
        episode: Episode,
        match: SimilarityResult,
        skill: StoredSkill,
        diff: TrajectoryDiff,
        heuristic_verdict: FeedbackVerdict,
    ) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._llm_feedback_refine(
                    candidate, episode, match, skill, diff, heuristic_verdict
                )
            )
        except RuntimeError:
            self._apply_feedback_verdict(
                candidate, episode, match, skill, diff, heuristic_verdict
            )

    async def _llm_feedback_refine(
        self,
        candidate: DistillationCandidate,
        episode: Episode,
        match: SimilarityResult,
        skill: StoredSkill,
        diff: TrajectoryDiff,
        heuristic_verdict: FeedbackVerdict,
    ) -> None:
        assert self._feedback is not None
        try:
            refined = await self._feedback.llm_verdict(
                diff, skill, candidate, heuristic_verdict
            )
            self._apply_feedback_verdict(
                candidate, episode, match, skill, diff, refined
            )
        except Exception:
            logger.debug(
                "LLM feedback refine failed; using heuristic", exc_info=True
            )
            self._apply_feedback_verdict(
                candidate, episode, match, skill, diff, heuristic_verdict
            )

    def _auto_apply_improvement(
        self,
        skill: StoredSkill,
        candidate: DistillationCandidate,
        episode: Episode,
        diff: TrajectoryDiff,
        verdict: FeedbackVerdict,
    ) -> None:
        candidate_apps = list(episode.app_sequence)
        new_action_names = [a.action_name for a in episode.semantic_actions]

        merged = StoredSkill(
            skill_id=skill.skill_id,
            title=skill.title,
            trigger_phrases=_union_dedup(
                skill.trigger_phrases, candidate.trigger_phrases
            ),
            steps=_merge_steps_lcs(skill.steps, candidate.steps),
            parameters=_merge_params(skill.parameters, candidate.parameters),
            pre_conditions=_union_dedup(
                skill.pre_conditions, candidate.pre_conditions
            ),
            post_conditions=list(skill.post_conditions),
            app_sequence=_union_dedup(skill.app_sequence, candidate_apps),
            action_names=_pick_longer(skill.action_names, new_action_names),
            source_trajectory_id=skill.source_trajectory_id,
            source_episode_id=skill.source_episode_id,
            confidence=(
                skill.confidence * skill.version + candidate.confidence
            )
            / (skill.version + 1),
            version=skill.version + 1,
            status="active",
            created_at=skill.created_at,
        )
        self._store.save_skill(merged)
        if self._activator:
            rec = self._store.load_parameterized_skill(skill.skill_id) if hasattr(self._store, 'load_parameterized_skill') else None
            if rec and rec.get("code"):
                self._activator.reactivate(skill.skill_id, rec["code"])
        self._wm.remember_event(
            "skill_auto_improved",
            f"Skill '{skill.title}' auto-improved to v{merged.version}: "
            f"{verdict.description}",
        )
        logger.info(
            "feedback.auto_applied skill=%s v%d type=%s",
            skill.title,
            merged.version,
            verdict.improvement_type,
        )

    def _build_feedback_suggestion(
        self,
        candidate: DistillationCandidate,
        match: SimilarityResult,
        episode: Episode,
        diff: TrajectoryDiff,
        verdict: FeedbackVerdict,
    ) -> SkillUpdateSuggestion:
        return SkillUpdateSuggestion(
            suggestion_id=uuid.uuid4().hex[:16],
            existing_skill_id=match.stored_skill_id,
            existing_skill_title=match.stored_skill_title,
            new_candidate_json=serialize_candidate(candidate),
            similarity_score=match.overall_score,
            similarity_details={
                "feedback_type": verdict.improvement_type,
                "feedback_description": verdict.description,
                "llm_rationale": verdict.llm_rationale,
                "diff_hash": diff.diff_hash,
            },
            suggestion_type="feedback_improvement",
            proposed_changes={
                "added_steps": diff.added_steps,
                "removed_steps": diff.removed_steps,
                "new_triggers": diff.new_triggers,
            },
            source_trajectory_id=episode.trajectory_id,
            source_episode_id=episode.episode_id,
        )

    def _notify_improvement(
        self,
        suggestion: SkillUpdateSuggestion,
        verdict: FeedbackVerdict,
    ) -> None:
        msg = (
            f"Feedback: skill '{suggestion.existing_skill_title}' can be improved "
            f"({verdict.improvement_type}). {verdict.description} "
            f"Say 'review skill suggestions' to review."
        )
        self._wm.remember_event("skill_feedback_suggestion", msg)

    # ── Skill activation ──

    def _schedule_activation(
        self, candidate: DistillationCandidate, episode: Episode
    ) -> None:
        """Schedule skill activation via SKILL.md (preferred) or codegen (fallback).

        When doc_generator is available, SKILL.md is the primary execution path —
        it supports runtime LLM reasoning without expensive upfront code generation.
        Codegen is only used as fallback when no doc path is configured.

        Only one activation is scheduled per source trajectory (session-level dedup).
        """
        if candidate.confidence < _MIN_ACTIVATION_CONFIDENCE:
            logger.info(
                "active_learning.skip_low_confidence title=%s conf=%.2f",
                candidate.title, candidate.confidence,
            )
            return

        tid = candidate.source_trajectory_id
        if tid and tid in self._activated_trajectories:
            logger.debug(
                "active_learning.skip_duplicate_activation trajectory=%s", tid,
            )
            return
        if tid:
            self._activated_trajectories.add(tid)

        has_doc_path = self._doc_generator is not None and self._doc_store is not None
        has_codegen = self._activator is not None
        if not has_doc_path and not has_codegen:
            return
        try:
            loop = asyncio.get_running_loop()
            if has_doc_path:
                task = loop.create_task(self._generate_skill_doc(candidate, episode))
                self._pending_activations.append(task)
            elif has_codegen:
                task = loop.create_task(self._activate_skill(candidate, episode))
                self._pending_activations.append(task)
        except RuntimeError:
            logger.debug(
                "active_learning.no_loop_for_activation; "
                "skill will activate at next startup"
            )

    async def _generate_skill_doc(
        self, candidate: DistillationCandidate, episode: Episode
    ) -> None:
        """Generate SKILL.md and register as an LLM-backed skill immediately.

        Guarantees that doc_store.save() is always called: if LLM generation
        fails or times out, a fallback document is built from the candidate.
        """
        from leapflow.utils.stream_progress import StreamProgressWriter

        assert self._doc_generator is not None
        assert self._doc_store is not None

        from leapflow.learning.doc_generator import DocGenContext

        _DIM = "\033[2m" if sys.stdout.isatty() else ""
        _RESET = "\033[0m" if sys.stdout.isatty() else ""

        print(f"{_DIM}  │   ├ Generating SKILL.md... [LLM]{_RESET}", flush=True)

        writer = StreamProgressWriter(prefix="  │   │ ")
        doc = None
        try:
            context = DocGenContext(
                existing_skill_names=self._doc_store.list_names(),
                episode=episode,
            )
            doc = await self._doc_generator.generate(
                candidate, context, on_chunk=writer,
            )
        except Exception:
            logger.debug("active_learning.doc_generation_failed", exc_info=True)
        finally:
            writer.finish()

        if doc is None:
            doc = self._build_fallback_doc(candidate, episode)
            print(f"{_DIM}  │   │ (using fallback document){_RESET}", flush=True)

        doc.provenance = self._build_provenance(candidate, episode)
        bundle = self._build_bundle_files(candidate, episode)
        saved_path = self._doc_store.save(doc, bundle=bundle)
        print(f"{_DIM}  │   ├ Saved: {saved_path / 'SKILL.md'}{_RESET}", flush=True)
        logger.info("active_learning.skill_doc_saved name=%s", doc.name)

        # Register immediately so the skill is usable within this session
        if self._skill_registry is not None and self._llm is not None:
            skill = self._doc_store.load_as_skill(
                doc.name, self._llm, execution=self._execution
            )
            if skill is not None:
                self._skill_registry.register(skill)
                self._activated_keys.add(candidate.title)
                self._wm.remember_event(
                    "skill_activated",
                    f"New skill '{skill.name}' is now executable (SKILL.md).",
                )
                print(f"{_DIM}  │   └ Skill registered: {skill.name}{_RESET}", flush=True)

    @staticmethod
    def _build_provenance(
        candidate: DistillationCandidate, episode: Episode
    ) -> "List[ProvenanceEntry]":
        """Build provenance entries linking SKILL.md to source trajectories."""
        import datetime

        from leapflow.learning.document import ProvenanceEntry

        entries: List[ProvenanceEntry] = []
        if episode.trajectory_id:
            notes = episode.inferred_goal or candidate.title
            entries.append(ProvenanceEntry(
                trajectory_id=episode.trajectory_id,
                date=datetime.date.today().isoformat(),
                notes=notes,
                reference=f"references/{episode.trajectory_id}_summary.md",
            ))
        return entries

    def _build_bundle_files(
        self, candidate: DistillationCandidate, episode: Episode
    ) -> "BundleFiles":
        """Build auxiliary bundle files from candidate and episode metadata."""
        from leapflow.storage.bundle_writer import BundleFiles

        anchors = None
        if candidate.anchor_candidates:
            anchors_dict = {}
            for ac in candidate.anchor_candidates:
                key = ac.element_label.lower().replace(" ", "_")
                entry: dict = {
                    "step": ac.step_index,
                    "label": ac.element_label,
                }
                if ac.element_role:
                    entry["role"] = ac.element_role
                if ac.app_bundle_id:
                    entry["app"] = ac.app_bundle_id
                anchors_dict[key] = entry
            anchors = {"anchors": anchors_dict}

        meta = {
            "author": "leapflow",
            "version": 1,
            "source": "learned",
            "confidence": candidate.confidence,
            "source_trajectory_id": candidate.source_trajectory_id,
            "source_episode_id": candidate.source_episode_id,
        }

        changelog = [f"Initial version from {episode.trajectory_id}"]

        summaries = {}
        if episode.trajectory_id:
            goal = episode.inferred_goal or candidate.title
            apps = ", ".join(episode.app_sequence) if episode.app_sequence else "unknown"
            summaries[episode.trajectory_id] = (
                f"# Trajectory Summary: {episode.trajectory_id}\n\n"
                f"Goal: {goal}\n"
                f"Apps: {apps}\n"
                f"Steps: {len(episode.semantic_actions)}\n"
            )

        return BundleFiles(
            anchors=anchors,
            meta=meta,
            changelog_entries=changelog,
            trajectory_summaries=summaries,
        )

    def _build_fallback_doc(
        self, candidate: DistillationCandidate, episode: Episode
    ) -> "SkillDocument":
        """Build a minimal SkillDocument from candidate fields without LLM."""
        from leapflow.learning.doc_generator import infer_allowed_tools
        from leapflow.learning.document import (
            ErrorHandlingEntry,
            ExampleDoc,
            ParameterDoc,
            SkillDocument,
            title_to_kebab,
        )

        name = title_to_kebab(candidate.title)
        triggers = candidate.trigger_phrases[:3] if candidate.trigger_phrases else []
        trigger_text = ", ".join(f'"{t}"' for t in triggers)
        description = (
            f"{candidate.title}. "
            f"Use when user says {trigger_text}."
            if triggers else candidate.title
        )

        parameters = [
            ParameterDoc(
                name=p.get("name", ""),
                type=p.get("type", "str"),
                required=bool(p.get("required", False)),
                default=p.get("default"),
                description=p.get("description", ""),
            )
            for p in candidate.parameters
            if isinstance(p, dict)
        ]

        allowed_tools = infer_allowed_tools(episode) if episode else ""

        error_handling = [
            ErrorHandlingEntry(
                pattern=r.pattern,
                signal=r.trigger_action,
                recovery=r.recovery_action,
            )
            for r in candidate.recovery_events
        ]

        return SkillDocument(
            name=name,
            description=description[:1024],
            goal=candidate.title,
            allowed_tools=allowed_tools or "Bash(*)",
            parameters=parameters,
            instructions=list(candidate.steps),
            preconditions=list(candidate.pre_conditions),
            postconditions=[],
            error_handling=error_handling,
            procedure_graph=candidate.procedure_graph,
            examples=[
                ExampleDoc(trigger=t, actions=candidate.steps[:3])
                for t in triggers[:2]
            ],
            metadata={
                "author": "leapflow",
                "version": 1,
                "source": "learned",
                "confidence": candidate.confidence,
            },
            source_trajectory_id=candidate.source_trajectory_id,
            source_episode_id=candidate.source_episode_id,
        )

    async def _activate_skill(
        self, candidate: DistillationCandidate, episode: Episode
    ) -> None:
        """Generate code and register skill in registry."""
        _DIM = "\033[2m" if sys.stdout.isatty() else ""
        _RESET = "\033[0m" if sys.stdout.isatty() else ""

        assert self._activator is not None
        try:
            print(f"{_DIM}  │   ├ Generating executable code... [LLM]{_RESET}", flush=True)
            skill = await self._activator.activate_candidate(candidate, episode)
            if skill:
                self._activated_keys.add(candidate.title)
                self._wm.remember_event(
                    "skill_activated",
                    f"New skill '{skill.name}' is now executable.",
                )
                print(f"{_DIM}  │   └ Skill activated: {skill.name}{_RESET}", flush=True)
            else:
                print(f"{_DIM}  │   └ Code generation failed (no valid output){_RESET}", flush=True)
        except Exception:
            print(f"{_DIM}  │   └ Activation error{_RESET}", flush=True)
            logger.debug("active_learning.activation_failed", exc_info=True)

    async def await_activations(self, timeout: float = 60.0) -> Set[str]:
        """Wait for all pending activation tasks and return activated candidate keys.

        Drains the pending task list and clears the activated-keys set so the
        next call starts fresh.  Safe to call even when no tasks are pending.
        Times out after ``timeout`` seconds to prevent LLM hangs from stalling
        the entire learn phase.
        """
        pending = self._pending_activations
        self._pending_activations = []
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "await_activations timed out after %.1fs (%d tasks pending)",
                    timeout, len(pending),
                )
                for task in pending:
                    if not task.done():
                        task.cancel()
        activated = self._activated_keys.copy()
        self._activated_keys.clear()
        self._activated_trajectories.clear()
        return activated


# ── Merger ──


class SkillMerger:
    """Merges an approved suggestion into the existing skill."""

    def apply(
        self, suggestion: SkillUpdateSuggestion, store: SkillLibraryStore
    ) -> StoredSkill:
        existing = store.load_skill(suggestion.existing_skill_id)
        if existing is None:
            raise ValueError(
                f"Skill {suggestion.existing_skill_id} not found"
            )
        candidate = deserialize_candidate(suggestion.new_candidate_json)
        candidate_apps = _extract_apps_from_candidate(candidate)

        merged = StoredSkill(
            skill_id=existing.skill_id,
            title=existing.title,
            trigger_phrases=_union_dedup(
                existing.trigger_phrases, candidate.trigger_phrases
            ),
            steps=_merge_steps_lcs(existing.steps, candidate.steps),
            parameters=_merge_params(existing.parameters, candidate.parameters),
            pre_conditions=_union_dedup(
                existing.pre_conditions, candidate.pre_conditions
            ),
            post_conditions=list(existing.post_conditions),
            app_sequence=_union_dedup(
                existing.app_sequence, candidate_apps
            ),
            action_names=_pick_longer(
                existing.action_names,
                [s.split()[0].lower() if s.strip() else "" for s in candidate.steps],
            ),
            source_trajectory_id=existing.source_trajectory_id,
            source_episode_id=existing.source_episode_id,
            confidence=(
                existing.confidence * existing.version + candidate.confidence
            )
            / (existing.version + 1),
            version=existing.version + 1,
            status="active",
            created_at=existing.created_at,
        )
        store.save_skill(merged)
        store.resolve_suggestion(suggestion.suggestion_id, "approved")
        logger.info(
            "skill_merger.applied skill=%s v%d",
            merged.skill_id,
            merged.version,
        )
        return merged


# ── Merge helpers ──


def _compute_changes(
    candidate: DistillationCandidate, match: SimilarityResult
) -> Dict:
    return {
        "new_steps": list(candidate.steps),
        "new_triggers": list(candidate.trigger_phrases),
        "candidate_title": candidate.title,
    }


def _extract_apps_from_candidate(c: DistillationCandidate) -> List[str]:
    apps: List[str] = []
    for cond in c.pre_conditions:
        if cond.endswith(" available"):
            apps.append(cond[: -len(" available")])
    return apps


def _union_dedup(a: Sequence[str], b: Sequence[str]) -> List[str]:
    """Ordered deduplication preserving insertion order."""
    seen: set[str] = set()
    result: List[str] = []
    for item in list(a) + list(b):
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _merge_params(
    a: Sequence[Dict[str, str]], b: Sequence[Dict[str, str]]
) -> List[Dict[str, str]]:
    """Merge parameter lists by name, deduplicating."""
    seen: set[str] = set()
    result: List[Dict[str, str]] = []
    for p in list(a) + list(b):
        name = p.get("name", "")
        if name and name not in seen:
            seen.add(name)
            result.append(dict(p))
    return result


def _pick_longer(a: List[str], b: List[str]) -> List[str]:
    return a if len(a) >= len(b) else b


def _merge_steps_lcs(existing: List[str], candidate: List[str]) -> List[str]:
    """Merge two step lists using LCS alignment.

    LCS elements are anchor points; non-LCS candidate steps are inserted
    at their relative positions between anchors.
    """
    lcs = _lcs(existing, candidate)
    if not lcs:
        return existing + [s for s in candidate if s not in existing]

    result: List[str] = []
    ei, ci = 0, 0

    for anchor in lcs:
        while ei < len(existing) and existing[ei] != anchor:
            result.append(existing[ei])
            ei += 1
        while ci < len(candidate) and candidate[ci] != anchor:
            if candidate[ci] not in result:
                result.append(candidate[ci])
            ci += 1
        result.append(anchor)
        ei += 1
        ci += 1

    while ei < len(existing):
        result.append(existing[ei])
        ei += 1
    while ci < len(candidate):
        if candidate[ci] not in result:
            result.append(candidate[ci])
        ci += 1

    return result


def _lcs(a: Sequence[str], b: Sequence[str]) -> List[str]:
    """Longest common subsequence of two string sequences."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return []
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    result: List[str] = []
    i, j = m, n
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            result.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    result.reverse()
    return result
