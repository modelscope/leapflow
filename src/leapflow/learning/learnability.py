"""Skill learnability assessment — decides if a recorded trajectory is worth distilling.

Architecture: Three-tier progressive assessment (L1 Rules → L2 VLM → L3 LLM).
- L1: Zero-cost rule-based pre-screening (< 100ms)
- L2: VLM video quality analysis (optional, 5-15s)
- L3: LLM event sequence semantic analysis (optional, 3-8s)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── Decision Enum ──


class LearnabilityDecision(Enum):
    """Outcome of learnability assessment."""

    LEARN = "learn"
    ASK = "ask"
    SKIP = "skip"


# ── Configuration ──


@dataclass(frozen=True)
class LearnabilityConfig:
    """All configurable parameters for learnability assessment."""

    min_steps: int = 3
    min_duration_s: float = 5.0
    max_idle_ratio: float = 0.80
    min_action_diversity: int = 2
    learn_threshold: float = 0.65
    ask_threshold: float = 0.40
    vlm_enabled: bool = True
    llm_enabled: bool = True
    rule_weight: float = 0.4
    vlm_weight: float = 0.3
    llm_weight: float = 0.3


# ── Input / Output Data ──


@dataclass(frozen=True)
class LearnabilityInput:
    """Input data for learnability assessment."""

    trajectory: Any  # Trajectory object (avoids circular import)
    goal: str = ""
    video_actions: List[Any] = field(default_factory=list)
    has_video: bool = False


@dataclass
class LearnabilityReport:
    """Assessment result with decision and reasoning."""

    decision: LearnabilityDecision
    score: float  # 0.0 - 1.0 综合分
    reason: str  # human-readable explanation
    rule_score: float = 0.0
    vlm_score: Optional[float] = None
    llm_score: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


# ── Protocol ──


@runtime_checkable
class LearnabilityAssessor(Protocol):
    """Protocol for learnability assessment implementations."""

    async def assess(self, input: LearnabilityInput) -> LearnabilityReport: ...


# ── L1: Rule-Based Assessor ──

_IDLE_THRESHOLD_S = 5.0  # Gap > 5s between steps counts as idle


class RuleBasedAssessor:
    """L1: Zero-cost rule-based pre-screening."""

    def __init__(self, config: LearnabilityConfig) -> None:
        self._config = config

    def assess(self, input: LearnabilityInput) -> tuple[float, Optional[str]]:
        """Returns (score, quick_reject_reason_or_None)."""
        traj = input.trajectory

        # Quick reject conditions (any one → immediate SKIP)
        if traj.step_count < self._config.min_steps:
            return 0.0, f"Too few steps ({traj.step_count} < {self._config.min_steps})"
        if traj.duration < self._config.min_duration_s:
            return 0.0, f"Too short ({traj.duration:.1f}s < {self._config.min_duration_s}s)"

        # Compute dimension scores
        scores: Dict[str, float] = {}
        scores["steps"] = min(1.0, traj.step_count / 10.0)  # normalize to ~10 steps
        scores["duration"] = min(1.0, traj.duration / 60.0)  # normalize to ~60s
        scores["diversity"] = self._action_diversity_score(traj)
        scores["idle_ratio"] = self._idle_ratio_score(traj)
        scores["continuity"] = self._continuity_score(traj)

        # Weighted average
        avg = sum(scores.values()) / len(scores) if scores else 0.0

        # Check idle ratio quick reject
        idle_ratio = self._compute_idle_ratio(traj)
        if idle_ratio > self._config.max_idle_ratio:
            return (
                avg * 0.3,
                f"Excessive idle time ({idle_ratio:.0%} > {self._config.max_idle_ratio:.0%})",
            )

        return avg, None

    # ── Private helpers ──

    def _action_diversity_score(self, traj: Any) -> float:
        """Score based on variety of distinct action types in the trajectory."""
        if not traj.steps:
            return 0.0
        action_types = {step.action.action_type for step in traj.steps}
        diversity = len(action_types)
        if diversity < self._config.min_action_diversity:
            return 0.2  # penalty for too-uniform actions
        # Normalize: 2 types → 0.4, 5+ types → 1.0
        return min(1.0, diversity / 5.0)

    def _idle_ratio_score(self, traj: Any) -> float:
        """Inverse of idle ratio — higher is better (less idle)."""
        idle_ratio = self._compute_idle_ratio(traj)
        return max(0.0, 1.0 - idle_ratio)

    def _continuity_score(self, traj: Any) -> float:
        """Score based on how continuous (gap-free) the operation sequence is."""
        if len(traj.steps) < 2:
            return 0.5  # neutral for single-step

        gaps = self._compute_gaps(traj)
        if not gaps:
            return 1.0

        # Count large gaps (> 10s)
        large_gaps = sum(1 for g in gaps if g > 10.0)
        total_intervals = len(gaps)

        if total_intervals == 0:
            return 1.0

        # Fewer large gaps → higher score
        gap_ratio = large_gaps / total_intervals
        return max(0.0, 1.0 - gap_ratio)

    def _compute_idle_ratio(self, traj: Any) -> float:
        """Raw idle ratio: fraction of total duration spent idle (> 5s gaps)."""
        if traj.duration <= 0 or len(traj.steps) < 2:
            return 0.0

        gaps = self._compute_gaps(traj)
        idle_time = sum(g for g in gaps if g > _IDLE_THRESHOLD_S)
        return idle_time / traj.duration

    def _compute_gaps(self, traj: Any) -> List[float]:
        """Compute time gaps between consecutive steps."""
        gaps: List[float] = []
        for i in range(1, len(traj.steps)):
            prev_ts = traj.steps[i - 1].action.timestamp
            curr_ts = traj.steps[i].action.timestamp
            gap = curr_ts - prev_ts
            if gap > 0:
                gaps.append(gap)
        return gaps


# ── Default Assessor (L1 + L2 + L3 combiner) ──


class DefaultLearnabilityAssessor:
    """Combines L1 Rules + L2 VLM + L3 LLM for comprehensive assessment."""

    def __init__(
        self,
        *,
        llm: Any = None,
        vlm: Any = None,
        config: Optional[LearnabilityConfig] = None,
    ) -> None:
        self._llm = llm
        self._vlm = vlm
        self._config = config or LearnabilityConfig()
        self._rules = RuleBasedAssessor(self._config)

    async def assess(self, input: LearnabilityInput) -> LearnabilityReport:
        """Run progressive assessment: L1 → L2 → L3."""
        # L1: Rules (always runs, zero cost)
        rule_score, quick_reject = self._rules.assess(input)

        if quick_reject and rule_score < self._config.ask_threshold:
            return LearnabilityReport(
                decision=LearnabilityDecision.SKIP,
                score=rule_score,
                reason=quick_reject,
                rule_score=rule_score,
            )

        # L2: VLM (optional)
        vlm_score: Optional[float] = None
        if self._config.vlm_enabled and self._vlm and input.has_video:
            vlm_score = await self._assess_vlm(input)

        # L3: LLM (optional)
        llm_score: Optional[float] = None
        if self._config.llm_enabled and self._llm:
            llm_score = await self._assess_llm(input)

        # Combine scores
        final_score = self._combine_scores(rule_score, vlm_score, llm_score)

        # Decision
        if final_score >= self._config.learn_threshold:
            decision = LearnabilityDecision.LEARN
            reason = "Operation pattern is learnable and generalizable"
        elif final_score >= self._config.ask_threshold:
            decision = LearnabilityDecision.ASK
            reason = "Uncertain learnability — user confirmation recommended"
        else:
            decision = LearnabilityDecision.SKIP
            reason = "Low learnability score — operation too simple or incomplete"

        return LearnabilityReport(
            decision=decision,
            score=final_score,
            reason=reason,
            rule_score=rule_score,
            vlm_score=vlm_score,
            llm_score=llm_score,
        )

    def _combine_scores(
        self, rule: float, vlm: Optional[float], llm: Optional[float]
    ) -> float:
        """Weighted combination with graceful handling of missing scores."""
        cfg = self._config
        total_weight = cfg.rule_weight
        weighted_sum = rule * cfg.rule_weight

        if vlm is not None:
            weighted_sum += vlm * cfg.vlm_weight
            total_weight += cfg.vlm_weight
        if llm is not None:
            weighted_sum += llm * cfg.llm_weight
            total_weight += cfg.llm_weight

        return weighted_sum / total_weight if total_weight > 0 else rule

    async def _assess_vlm(self, input: LearnabilityInput) -> float:
        """L2: VLM-based video quality assessment.

        Evaluates whether the video shows meaningful, learnable interactions.
        Returns score in [0.0, 1.0]. Returns 0.5 (neutral) on failure.
        """
        if not input.has_video or not input.video_actions:
            return 0.5  # No video signal available

        try:
            # Build summary from video_actions for VLM analysis
            action_summary = self._build_action_summary(input)
            messages = self._build_vlm_assessment_messages(action_summary, input)

            response = await self._vlm.achat(messages, stream=False)
            content = response.content if hasattr(response, 'content') else str(response)

            parsed = self._parse_json_response(content)
            if not parsed:
                return 0.5

            # Extract dimension scores
            interaction = float(parsed.get("interaction_visibility", 0.5))
            coherence = float(parsed.get("flow_coherence", 0.5))
            magnitude = float(parsed.get("content_change_magnitude", 0.5))
            reproducibility = float(parsed.get("reproducibility_score", 0.5))

            # Weighted combination
            score = (
                interaction * 0.3
                + coherence * 0.3
                + magnitude * 0.2
                + reproducibility * 0.2
            )
            return max(0.0, min(1.0, score))

        except Exception as exc:
            logger.warning("VLM learnability assessment failed: %s", exc)
            return 0.5  # graceful degradation

    def _build_action_summary(self, input: LearnabilityInput) -> str:
        """Build a text summary of video actions for VLM prompt."""
        actions = input.video_actions[:20]  # limit to first 20 actions
        lines: List[str] = []
        for i, action in enumerate(actions, 1):
            name = getattr(action, 'action_name', str(action))
            desc = getattr(action, 'description', '')
            app = getattr(action, 'app', '')
            confidence = getattr(action, 'confidence', 0)
            lines.append(f"{i}. [{app}] {name}: {desc} (confidence={confidence:.1f})")
        return "\n".join(lines) if lines else "(no video actions extracted)"

    def _build_vlm_assessment_messages(
        self, action_summary: str, input: LearnabilityInput
    ) -> list:
        """Build VLM prompt for learnability assessment."""
        traj = input.trajectory
        duration = traj.duration if hasattr(traj, 'duration') else 0
        step_count = traj.step_count if hasattr(traj, 'step_count') else 0
        goal = input.goal or "(not specified)"

        prompt = (
            "Analyze the following desktop operation recording and assess its "
            "learnability as a reusable skill.\n\n"
            "## Context\n"
            f"- User goal: {goal}\n"
            f"- Duration: {duration:.1f}s\n"
            f"- Event steps: {step_count}\n"
            f"- Video actions extracted: {len(input.video_actions)}\n\n"
            "## Video Action Summary\n"
            f"{action_summary}\n\n"
            "## Assessment Dimensions (score each 0.0 to 1.0)\n\n"
            "1. **interaction_visibility**: Are meaningful UI interactions visible? "
            "(clicking, typing, dragging, menu navigation vs. just scrolling/reading)\n"
            "2. **flow_coherence**: Do the actions form a logical, coherent workflow? "
            "(sequential steps toward a goal vs. random exploration)\n"
            "3. **content_change_magnitude**: Is there significant content change? "
            "(files created/moved, data transformed vs. no state change)\n"
            "4. **reproducibility_score**: Can this workflow be generalized and "
            "reproduced for similar tasks? (pattern-based vs. one-off unique operation)\n\n"
            "## Output (JSON only)\n"
            "```json\n"
            "{\n"
            '  "interaction_visibility": <float 0-1>,\n'
            '  "flow_coherence": <float 0-1>,\n'
            '  "content_change_magnitude": <float 0-1>,\n'
            '  "reproducibility_score": <float 0-1>,\n'
            '  "summary": "<one sentence describing what user did>"\n'
            "}\n"
            "```"
        )
        return [{"role": "user", "content": prompt}]

    def _parse_json_response(self, text: str) -> Optional[dict]:
        """Parse JSON from VLM/LLM response with fallbacks."""
        if not text:
            return None
        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Extract from markdown code block
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Find first {...} (supports one level of nesting)
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None

    async def _assess_llm(self, input: LearnabilityInput) -> float:
        """L3: LLM-based event sequence semantic analysis.

        Evaluates the semantic value of the event sequence for skill generation.
        Returns score in [0.0, 1.0]. Returns 0.5 (neutral) on failure.
        """
        if not self._llm:
            return 0.5

        try:
            event_transcript = self._build_event_transcript(input)
            messages = self._build_llm_assessment_messages(event_transcript, input)

            response = await self._llm.achat(messages, stream=False)
            content = response.content if hasattr(response, 'content') else str(response)

            parsed = self._parse_json_response(content)
            if not parsed:
                return 0.5

            # Extract dimension scores
            goal_clarity = float(parsed.get("goal_clarity", 0.5))
            completeness = float(parsed.get("step_completeness", 0.5))
            reproducibility = float(parsed.get("reproducibility", 0.5))
            complexity = float(parsed.get("complexity", 0.5))

            # Weighted combination
            score = (
                goal_clarity * 0.3 +
                completeness * 0.25 +
                reproducibility * 0.25 +
                complexity * 0.2
            )
            return max(0.0, min(1.0, score))

        except Exception as exc:
            logger.warning("LLM learnability assessment failed: %s", exc)
            return 0.5

    def _build_event_transcript(self, input: LearnabilityInput) -> str:
        """Build a concise event transcript from trajectory steps."""
        traj = input.trajectory
        steps = getattr(traj, 'steps', [])

        if not steps:
            return "(no events recorded)"

        # Show first 15 + last 5 steps (for long trajectories)
        max_head = 15
        max_tail = 5

        lines: List[str] = []
        show_steps = steps[:max_head]
        if len(steps) > max_head + max_tail:
            show_steps = steps[:max_head]
            lines_mid = f"... ({len(steps) - max_head - max_tail} more steps) ..."
            show_steps_tail = steps[-max_tail:]
        elif len(steps) > max_head:
            lines_mid = None
            show_steps_tail = steps[max_head:]
        else:
            lines_mid = None
            show_steps_tail = []

        for i, step in enumerate(show_steps, 1):
            action = getattr(step, 'action', step)
            action_type = getattr(action, 'action_type', getattr(action, 'event_type', 'unknown'))
            app = getattr(action, 'app_bundle_id', getattr(action, 'app', ''))
            payload = getattr(action, 'payload', {})
            # Extract key info from payload
            desc = ""
            if isinstance(payload, dict):
                if 'path' in payload:
                    desc = f"path={payload['path']}"
                elif 'text' in payload:
                    desc = f'text="{str(payload["text"])[:50]}"'
                elif 'url' in payload:
                    desc = f"url={payload['url']}"
            lines.append(f"{i}. [{app}] {action_type} {desc}".strip())

        if lines_mid:
            lines.append(lines_mid)

        if show_steps_tail:
            offset = len(steps) - len(show_steps_tail) + 1
            for i, step in enumerate(show_steps_tail, offset):
                action = getattr(step, 'action', step)
                action_type = getattr(action, 'action_type', getattr(action, 'event_type', 'unknown'))
                app = getattr(action, 'app_bundle_id', getattr(action, 'app', ''))
                lines.append(f"{i}. [{app}] {action_type}")

        return "\n".join(lines)

    def _build_llm_assessment_messages(self, event_transcript: str, input: LearnabilityInput) -> list:
        """Build LLM prompt for event sequence learnability assessment."""
        traj = input.trajectory
        duration = traj.duration if hasattr(traj, 'duration') else 0
        step_count = traj.step_count if hasattr(traj, 'step_count') else 0
        goal = input.goal or "(not specified by user)"

        # Compute basic stats
        steps = getattr(traj, 'steps', [])
        action_types: set = set()
        apps: set = set()
        for step in steps:
            action = getattr(step, 'action', step)
            action_types.add(getattr(action, 'action_type', getattr(action, 'event_type', 'unknown')))
            apps.add(getattr(action, 'app_bundle_id', getattr(action, 'app', '')))

        prompt = (
            "Analyze the following desktop operation event sequence and assess "
            "whether it represents a learnable, reusable skill pattern.\n\n"
            "## Context\n"
            f"- User goal: {goal}\n"
            f"- Total steps: {step_count}\n"
            f"- Duration: {duration:.1f}s\n"
            f"- Unique action types: {len(action_types)} ({', '.join(sorted(str(t) for t in action_types)[:8])})\n"
            f"- Applications involved: {len(apps)} ({', '.join(sorted(a for a in apps if a)[:5])})\n\n"
            "## Event Sequence\n"
            f"{event_transcript}\n\n"
            "## Assessment Dimensions (score each 0.0 to 1.0)\n\n"
            "1. **goal_clarity** (0-1): Is there a clear, inferable goal?\n"
            "   - 1.0: Crystal clear goal (e.g., \"organize files into categorized folders\")\n"
            "   - 0.5: Somewhat unclear but reasonable steps\n"
            "   - 0.0: No discernible goal, random actions\n\n"
            "2. **step_completeness** (0-1): Is the workflow complete from start to finish?\n"
            "   - 1.0: Complete workflow with clear beginning and end state\n"
            "   - 0.5: Mostly complete but may be missing a step\n"
            "   - 0.0: Incomplete, abandoned, or interrupted\n\n"
            "3. **reproducibility** (0-1): Can this be generalized to similar tasks?\n"
            "   - 1.0: Highly generalizable pattern (works on any similar input)\n"
            "   - 0.5: Partially generalizable (some steps are context-specific)\n"
            "   - 0.0: Completely one-off, cannot be reproduced\n\n"
            "4. **complexity** (0-1): Is this complex enough to be worth automating?\n"
            "   - 1.0: Multi-step cross-application workflow\n"
            "   - 0.5: Moderate (5+ meaningful steps)\n"
            "   - 0.0: Trivial (single action, not worth a skill)\n\n"
            "## Output (JSON only)\n"
            "```json\n"
            "{\n"
            '  "goal_clarity": <float 0-1>,\n'
            '  "step_completeness": <float 0-1>,\n'
            '  "reproducibility": <float 0-1>,\n'
            '  "complexity": <float 0-1>,\n'
            '  "inferred_goal": "<one sentence describing the inferred user goal>"\n'
            "}\n"
            "```"
        )

        return [{"role": "user", "content": prompt}]


__all__ = [
    "DefaultLearnabilityAssessor",
    "LearnabilityAssessor",
    "LearnabilityConfig",
    "LearnabilityDecision",
    "LearnabilityInput",
    "LearnabilityReport",
    "RuleBasedAssessor",
]
