"""Cross-trajectory consensus distillation.

When a user naturally performs the same task multiple times, each recording
will contain different noise (typos, detours, hesitations). The Longest
Common Subsequence across all trajectories converges on the essential steps,
filtering noise that varies between demonstrations.

    Demo 1: [A, B, X, C, D]    (X = mistake)
    Demo 2: [A, B, C, Y, D]    (Y = distraction)
    Demo 3: [A, Z, B, C, D]    (Z = hesitation)
    LCS(1,2,3) = [A, B, C, D]  ← clean skill
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from leapflow.domain.skill_types import DistillationCandidate

if TYPE_CHECKING:
    from leapflow.analysis.pipeline import ImitationPipeline

logger = logging.getLogger(__name__)


class MultiTrajectoryDistiller:
    """Extract consensus skills from multiple demonstrations of the same task."""

    def __init__(
        self,
        pipeline: "ImitationPipeline",
        *,
        min_trajectories: int = 2,
    ) -> None:
        self._pipeline = pipeline
        self._min_count = min_trajectories

    async def distill_consensus(
        self, trajectory_ids: List[str]
    ) -> Optional[DistillationCandidate]:
        if len(trajectory_ids) < self._min_count:
            return None

        all_episode_actions: List[List[str]] = []
        for tid in trajectory_ids:
            episodes = await self._pipeline.analyze(tid)
            if not episodes:
                continue
            best = max(episodes, key=lambda e: len(e.semantic_actions))
            action_names = [a.action_name for a in best.semantic_actions]
            all_episode_actions.append(action_names)

        if len(all_episode_actions) < self._min_count:
            return None

        consensus = all_episode_actions[0]
        for other in all_episode_actions[1:]:
            consensus = _lcs(consensus, other)

        if len(consensus) < 2:
            return None

        total = len(all_episode_actions)
        step_frequency: Dict[str, int] = {}
        for actions in all_episode_actions:
            for a in set(actions):
                step_frequency[a] = step_frequency.get(a, 0) + 1

        confidence = sum(
            step_frequency.get(s, 0) / total for s in consensus
        ) / len(consensus)

        logger.info(
            "consensus: %d trajectories → %d steps (confidence=%.2f)",
            len(trajectory_ids), len(consensus), confidence,
        )

        return DistillationCandidate(
            title=f"Consensus skill ({len(trajectory_ids)} demos)",
            trigger_phrases=[],
            steps=consensus,
            confidence=min(confidence, 0.95),
        )


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
