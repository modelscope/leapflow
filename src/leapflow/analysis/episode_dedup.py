"""Pre-distillation episode deduplication via structural fingerprinting.

Prevents redundant LLM calls by grouping structurally identical episodes
and selecting one representative per group for distillation.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from leapflow.domain.trajectory import Episode

logger = logging.getLogger(__name__)


def compute_episode_signature(episode: Episode) -> str:
    """Structural fingerprint: ordered app sequence + sorted action name set."""
    apps = "|".join(episode.app_sequence)
    actions = "|".join(sorted({a.action_name for a in episode.semantic_actions}))
    return f"{apps}::{actions}"


def deduplicate_episodes(episodes: List[Episode]) -> List[Episode]:
    """Select one representative per unique structural signature.

    The representative is the episode with the most semantic actions
    (richest signal for LLM distillation). Ties broken by confidence.
    """
    if not episodes:
        return []

    groups: Dict[str, List[Episode]] = {}
    for ep in episodes:
        sig = compute_episode_signature(ep)
        groups.setdefault(sig, []).append(ep)

    representatives: List[Episode] = []
    for group in groups.values():
        best = max(group, key=lambda e: (len(e.semantic_actions), e.confidence))
        representatives.append(best)

    if len(representatives) < len(episodes):
        logger.info(
            "episode_dedup: %d episodes → %d unique signatures",
            len(episodes),
            len(representatives),
        )

    return representatives
