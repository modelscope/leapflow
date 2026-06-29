"""Evolution memory provider — Ring 3 learning support for skill episodes.

Stores and retrieves skill execution episodes (actions, outcomes, rewards)
to enable experience-driven improvement. Implements generalization (LCS-based
pattern extraction from multiple episodes) and novelty checking.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from leapflow.memory.protocol import (
    MemoryEntry,
    MemoryKind,
    MemoryQuery,
    MemoryToolSchema,
    SignalDomain,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Decay formula (shared across providers)
# ──────────────────────────────────────────────────────────────────────

_DEFAULT_DECAY_LAMBDA: float = 1e-5


def _decay_score(
    semantic_weight: float,
    age_seconds: float,
    frequency: float,
    decay_lambda: float = _DEFAULT_DECAY_LAMBDA,
) -> float:
    """W = S * exp(-lambda * age) * log(1 + frequency)."""
    if semantic_weight <= 0 or frequency <= 0:
        return 0.0
    normalized_freq = 1.0 + math.log1p(frequency - 1.0)
    return semantic_weight * math.exp(-decay_lambda * age_seconds) * normalized_freq


# ──────────────────────────────────────────────────────────────────────
# Episode data structure
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SkillEpisode:
    """A recorded skill execution episode for experience replay."""

    episode_id: str
    skill_name: str
    actions: List[Dict[str, Any]]
    outcome: str
    reward: float
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ──────────────────────────────────────────────────────────────────────
# Provider
# ──────────────────────────────────────────────────────────────────────

class EvolutionMemoryProvider:
    """Experience store for skill evolution (Ring 3 learning loop).

    Tracks skill execution episodes to enable:
    - Experience-based planning (query similar past executions)
    - Reward-weighted action selection
    - Skill confidence calibration
    - Generalization: extract common patterns from repeated episodes (LCS)
    - Novelty detection: avoid storing duplicate experiences
    """

    _ACCEPTED_KINDS = frozenset({MemoryKind.SKILL_EPISODE, MemoryKind.SKILL_PATTERN, MemoryKind.PREDICTION})

    def __init__(
        self,
        *,
        max_episodes: int = 1000,
        generalization_threshold: int = 3,
        novelty_threshold: float = 0.3,
        decay_lambda: float = _DEFAULT_DECAY_LAMBDA,
    ) -> None:
        self._max_episodes = max_episodes
        self._generalization_threshold = generalization_threshold
        self._novelty_threshold = novelty_threshold
        self._decay_lambda = decay_lambda

        self._entries: Dict[str, MemoryEntry] = {}
        self._episodes: Dict[str, SkillEpisode] = {}
        # Skill-name index for fast lookup
        self._skill_index: Dict[str, List[str]] = {}
        # Stats tracking
        self._stats = {"episodes": 0, "patterns": 0, "generalization_attempts": 0}

    # ── Protocol properties ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return "evolution"

    # ── Protocol methods ──────────────────────────────────────────────

    async def initialize(self, **kwargs: Any) -> None:
        """No external resources needed for in-memory store."""

    async def shutdown(self) -> None:
        self._entries.clear()
        self._episodes.clear()
        self._skill_index.clear()

    def accepts(self, entry: MemoryEntry) -> bool:
        return entry.kind in self._ACCEPTED_KINDS

    async def insert(self, entry: MemoryEntry) -> str:
        """Store a skill episode or prediction entry."""
        self._entries[entry.entry_id] = entry
        self._evict_overflow()
        return entry.entry_id

    async def search(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Search by keywords (skill_name match) and context similarity with decay scoring."""
        now = time.time()
        keywords_lower = [k.lower() for k in query.keywords] if query.keywords else []
        results: List[MemoryEntry] = []

        for entry in self._entries.values():
            # Kind filter
            if query.kinds and entry.kind not in query.kinds:
                continue

            # Domain filter
            if query.domains and entry.domain not in query.domains:
                continue

            # Time range filter
            if query.time_range:
                t_min, t_max = query.time_range
                if not (t_min <= entry.timestamp <= t_max):
                    continue

            # Keyword/skill-name filter
            if keywords_lower:
                searchable = entry.content.lower()
                skill_name = str(entry.metadata.get("skill_name", "")).lower()
                searchable = f"{searchable} {skill_name}"
                if not any(kw in searchable for kw in keywords_lower):
                    continue

            # Compute decay-weighted score
            age = max(0.0, now - entry.timestamp)
            freq = float(entry.access_count)
            if keywords_lower:
                searchable = entry.content.lower()
                matched = sum(1 for kw in keywords_lower if kw in searchable)
                semantic = matched / len(keywords_lower)
            else:
                semantic = 1.0

            # Blend with reward if available
            reward = entry.metadata.get("reward", 0.0)
            if isinstance(reward, (int, float)) and reward > 0:
                semantic = max(semantic, reward)

            score = _decay_score(semantic, age, freq, self._decay_lambda)
            entry.score = score

            if entry.score >= query.min_score:
                results.append(entry)

        # Sort by score (reward-weighted) then recency
        results.sort(key=lambda e: (e.score, e.timestamp), reverse=True)
        return results[: query.limit]

    async def delete(self, entry_id: str) -> bool:
        if entry_id not in self._entries:
            return False
        entry = self._entries.pop(entry_id)
        # Clean up episode and index if applicable
        skill_name = entry.metadata.get("skill_name")
        if skill_name and skill_name in self._skill_index:
            ids = self._skill_index[skill_name]
            if entry_id in ids:
                ids.remove(entry_id)
        self._episodes.pop(entry_id, None)
        return True

    # ── Lifecycle hooks ────────────────────────────────────────────────

    def on_turn_start(self, turn: int, user_message: str) -> None:
        """No-op for evolution provider."""

    def on_inserted(self, entry: MemoryEntry) -> None:
        """No-op — evolution tracks its own inserts."""

    def on_accessed(self, entry: MemoryEntry) -> None:
        """Increment access count if the entry belongs to us."""
        if entry.entry_id in self._entries:
            self._entries[entry.entry_id].access_count += 1

    def get_tool_schemas(self) -> List[MemoryToolSchema]:
        """Expose read-only skill search tool to LLM."""
        return [MemoryToolSchema(
            name="memory_skills",
            description="Search learned skills and execution patterns.",
            parameters={
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Skill name or description keywords"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["keywords"],
            },
            provider_name="evolution",
        )]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Handle LLM tool call for skill search."""
        if tool_name == "memory_skills":
            import asyncio
            keywords = args.get("keywords", "").split()
            limit = int(args.get("limit", 5))
            mq = MemoryQuery(keywords=keywords, limit=limit)
            # Synchronous wrapper for the async search
            try:
                loop = asyncio.get_running_loop()
                # If we're inside an event loop, use run_until_complete workaround
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    results = pool.submit(
                        asyncio.run, self.search(mq)
                    ).result()
            except RuntimeError:
                results = asyncio.run(self.search(mq))
            return json.dumps({
                "results": [
                    {"content": e.content[:200], "kind": e.kind.value, "score": round(e.score, 3)}
                    for e in results
                ],
            }, ensure_ascii=False)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # ── Domain-specific methods ───────────────────────────────────────

    def record_episode(
        self,
        skill_name: str,
        actions: List[Dict[str, Any]],
        outcome: str,
        reward: float,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> SkillEpisode:
        """Record a complete skill execution episode.

        Creates both a MemoryEntry (for protocol-based access) and a
        SkillEpisode (for structured experience queries).
        After recording, runs novelty check and generalization check.
        """
        entry_id = uuid.uuid4().hex[:12]
        now = time.time()

        episode = SkillEpisode(
            episode_id=entry_id,
            skill_name=skill_name,
            actions=actions,
            outcome=outcome,
            reward=reward,
            context=context or {},
            timestamp=now,
        )
        self._episodes[entry_id] = episode

        # Create corresponding MemoryEntry
        entry = MemoryEntry(
            entry_id=entry_id,
            kind=MemoryKind.SKILL_EPISODE,
            domain=SignalDomain.SYSTEM,
            content=f"[{skill_name}] {outcome}",
            timestamp=now,
            score=reward,
            metadata={
                "skill_name": skill_name,
                "action_count": len(actions),
                "reward": reward,
                **(context or {}),
            },
        )
        self._entries[entry_id] = entry

        # Update skill index
        self._skill_index.setdefault(skill_name, []).append(entry_id)
        self._evict_overflow()
        self._stats["episodes"] += 1

        # Check generalization (only for successful outcomes)
        if reward > 0:
            self._check_generalization(skill_name)

        return episode

    def query_experience(
        self,
        skill_name: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        limit: int = 10,
    ) -> List[SkillEpisode]:
        """Query past episodes for a skill, optionally filtered by context similarity.

        Returns episodes sorted by reward (descending), capped at limit.
        """
        candidate_ids = self._skill_index.get(skill_name, [])
        episodes = [self._episodes[eid] for eid in candidate_ids if eid in self._episodes]

        if context:
            # Boost episodes with overlapping context keys
            def _context_overlap(ep: SkillEpisode) -> float:
                if not ep.context:
                    return 0.0
                overlap = sum(1 for k in context if k in ep.context)
                return overlap / max(1, len(context))

            episodes.sort(key=lambda ep: (ep.reward + _context_overlap(ep), ep.timestamp), reverse=True)
        else:
            episodes.sort(key=lambda ep: (ep.reward, ep.timestamp), reverse=True)

        return episodes[:limit]

    # ── Generalization (LCS-based pattern extraction) ──────────────────

    def generalize(self, skill_name: str) -> Optional[MemoryEntry]:
        """Extract a common pattern from multiple episodes of the same skill.

        Algorithm:
        1. Retrieve all successful episodes for this skill
        2. Extract tool/action sequences from each
        3. Compute multi-sequence LCS to find the common subsequence
        4. If LCS >= 2 actions, create a SKILL_PATTERN entry
        5. Return the pattern entry or None if insufficient data

        This implements the design doc's "generalize from skill_episodes"
        requirement using LCS (Longest Common Subsequence).
        """
        candidate_ids = self._skill_index.get(skill_name, [])
        if len(candidate_ids) < self._generalization_threshold:
            return None

        # Gather successful episodes
        successful_episodes: List[SkillEpisode] = []
        for eid in candidate_ids:
            ep = self._episodes.get(eid)
            if ep and ep.reward > 0:
                successful_episodes.append(ep)

        if len(successful_episodes) < self._generalization_threshold:
            return None

        # Extract action sequences (tool names)
        action_sequences: List[List[str]] = []
        for ep in successful_episodes:
            seq = [
                a.get("name", a.get("type", "unknown"))
                for a in ep.actions
                if isinstance(a, dict)
            ]
            if seq:
                action_sequences.append(seq)

        if len(action_sequences) < self._generalization_threshold:
            return None

        # Multi-sequence LCS
        common = action_sequences[0]
        for seq in action_sequences[1:]:
            common = self._lcs(common, seq)

        if len(common) < 2:
            return None

        # Build pattern entry
        pattern_content = json.dumps({
            "skill_name": skill_name,
            "common_actions": common,
            "sample_count": len(successful_episodes),
            "confidence": min(1.0, len(successful_episodes) / 10.0),
            "avg_reward": sum(ep.reward for ep in successful_episodes) / len(successful_episodes),
        }, ensure_ascii=False)

        pattern_entry = MemoryEntry(
            entry_id=f"pat_{uuid.uuid4().hex[:8]}",
            kind=MemoryKind.SKILL_PATTERN,
            domain=SignalDomain.SYSTEM,
            content=pattern_content,
            timestamp=time.time(),
            score=1.0,
            metadata={
                "skill_name": skill_name,
                "source_episodes": [ep.episode_id for ep in successful_episodes[:10]],
                "action_count": len(common),
            },
        )
        self._entries[pattern_entry.entry_id] = pattern_entry
        self._stats["patterns"] += 1
        logger.info(
            "evolution.generalized skill=%s episodes=%d pattern_actions=%d",
            skill_name, len(successful_episodes), len(common),
        )
        return pattern_entry

    def novelty_check(self, episode: SkillEpisode) -> bool:
        """Check if a new episode contains novel information.

        Compares the action sequence of the candidate episode against
        existing episodes of the same skill. Returns True if the episode
        is sufficiently different (novelty exceeds threshold).

        Algorithm: LCS-based sequence similarity. If similarity > (1 - novelty_threshold),
        the episode is considered a duplicate and returns False.
        """
        candidate_ids = self._skill_index.get(episode.skill_name, [])
        if not candidate_ids:
            return True  # First episode is always novel

        new_actions = [
            a.get("name", a.get("type", "unknown"))
            for a in episode.actions
            if isinstance(a, dict)
        ]
        if not new_actions:
            return True

        for eid in candidate_ids[-10:]:  # Check last 10 episodes
            existing = self._episodes.get(eid)
            if existing is None or existing.episode_id == episode.episode_id:
                continue
            existing_actions = [
                a.get("name", a.get("type", "unknown"))
                for a in existing.actions
                if isinstance(a, dict)
            ]
            if not existing_actions:
                continue

            similarity = self._sequence_similarity(new_actions, existing_actions)
            if similarity > (1.0 - self._novelty_threshold):
                return False  # Too similar, not novel

        return True

    # ── Internal ──────────────────────────────────────────────────────

    def _check_generalization(self, skill_name: str) -> None:
        """Check if enough episodes exist to trigger generalization."""
        self._stats["generalization_attempts"] += 1
        candidate_ids = self._skill_index.get(skill_name, [])
        successful_count = sum(
            1 for eid in candidate_ids
            if eid in self._episodes and self._episodes[eid].reward > 0
        )
        if successful_count >= self._generalization_threshold:
            # Check if we already have a pattern for this skill
            for entry in self._entries.values():
                if (entry.kind == MemoryKind.SKILL_PATTERN and
                        entry.metadata.get("skill_name") == skill_name):
                    return  # Already generalized
            self.generalize(skill_name)

    def _evict_overflow(self) -> None:
        """Remove oldest entries when capacity exceeded."""
        while len(self._entries) > self._max_episodes:
            oldest_id = min(self._entries, key=lambda eid: self._entries[eid].timestamp)
            entry = self._entries.pop(oldest_id)
            self._episodes.pop(oldest_id, None)
            skill_name = entry.metadata.get("skill_name")
            if skill_name and skill_name in self._skill_index:
                ids = self._skill_index[skill_name]
                if oldest_id in ids:
                    ids.remove(oldest_id)

    @staticmethod
    def _lcs(a: List[str], b: List[str]) -> List[str]:
        """Standard LCS (Longest Common Subsequence) via dynamic programming."""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        # Backtrack to reconstruct LCS
        result: List[str] = []
        i, j = m, n
        while i > 0 and j > 0:
            if a[i - 1] == b[j - 1]:
                result.append(a[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        return list(reversed(result))

    @staticmethod
    def _sequence_similarity(a: List[str], b: List[str]) -> float:
        """Compute sequence similarity based on LCS ratio."""
        if not a or not b:
            return 0.0
        lcs_len = len(EvolutionMemoryProvider._lcs(a, b))
        return lcs_len / max(len(a), len(b))
