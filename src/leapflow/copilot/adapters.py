"""Memory Bridge Adapters — connect Copilot prediction layers to the Memory system.

Each adapter implements a Copilot Protocol using a Memory provider as its backend,
enabling prediction tiers (L0-L3) to draw from persistent knowledge without direct
coupling between the two subsystems.

Design:
    - DIP: Copilot depends on Protocols; adapters fulfil those Protocols
    - SRP: One adapter = one Protocol bridged to one data source
    - OCP: New adapters require no changes to Copilot or Memory code
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

from leapflow.copilot.predictors.l0_hash import ContextHashHit
from leapflow.copilot.predictors.l2_embed import EmbeddingHit
from leapflow.copilot.predictors.l3_llm import RAGHit

if TYPE_CHECKING:
    from leapflow.copilot.predictors.l1_markov import L1MarkovPredictor
    from leapflow.memory.providers.episodic import EpisodicMemoryProvider
    from leapflow.memory.providers.semantic import SemanticMemoryProvider
    from leapflow.memory.providers.working import WorkingMemoryProvider
    from leapflow.world_model.experience_store import ExperienceStore

logger = logging.getLogger(__name__)

_COPILOT_PATTERN_KIND = "copilot_pattern"


# ────────────────────────────────────────────────────────────────────────────
# L0 Adapter: SemanticHashAdapter (ContextHashStore Protocol)
# ────────────────────────────────────────────────────────────────────────────


class SemanticHashAdapter:
    """ContextHashStore backed by SemanticMemoryProvider (DuckDB).

    Stores and retrieves context_hash → action observations as persistent
    memory entries of kind ``copilot_pattern``.  Enables L0 predictions to
    survive process restarts and accumulate over weeks of usage.
    """

    def __init__(self, semantic: "SemanticMemoryProvider") -> None:
        self._semantic = semantic

    async def query_by_hash(self, context_hash: str) -> List[ContextHashHit]:
        """Retrieve all recorded actions for a context hash from DuckDB."""
        try:
            hits = self._semantic.search_keywords(
                [context_hash], kinds=[_COPILOT_PATTERN_KIND], limit=20
            )
        except Exception as exc:
            logger.debug("SemanticHashAdapter.query_by_hash failed: %s", exc)
            return []

        results: List[ContextHashHit] = []
        for hit in hits:
            meta = hit.metadata
            action = meta.get("action", "")
            if not action:
                continue
            results.append(ContextHashHit(
                action=action,
                accept_count=int(meta.get("accept_count", 0)),
                total_count=int(meta.get("total_count", 1)),
                skill_id=meta.get("skill_id"),
            ))
        return results

    async def record_observation(
        self, context_hash: str, action: str, accepted: bool
    ) -> None:
        """Record a feedback observation, upserting the accept/total counts."""
        try:
            # Search for existing entry
            hits = self._semantic.search_keywords(
                [context_hash, action], kinds=[_COPILOT_PATTERN_KIND], limit=1
            )
            if hits:
                existing = hits[0]
                meta = dict(existing.metadata)
                meta["total_count"] = int(meta.get("total_count", 0)) + 1
                if accepted:
                    meta["accept_count"] = int(meta.get("accept_count", 0)) + 1
                self._semantic.update_metadata(existing.memory_id, meta)
            else:
                # Insert new entry
                self._semantic.insert_raw(
                    kind=_COPILOT_PATTERN_KIND,
                    content=f"[{context_hash}] {action}",
                    metadata={
                        "context_hash": context_hash,
                        "action": action,
                        "accept_count": 1 if accepted else 0,
                        "total_count": 1,
                    },
                )
        except Exception as exc:
            logger.debug("SemanticHashAdapter.record_observation failed: %s", exc)


# ────────────────────────────────────────────────────────────────────────────
# L1 Adapter: EpisodicSequenceAdapter (seed utility)
# ────────────────────────────────────────────────────────────────────────────


class EpisodicSequenceAdapter:
    """Seeds L1 Markov predictor with recent event sequences from EpisodicMemory.

    Not a Protocol adapter — a one-shot bootstrap utility called at startup to
    warm the Markov transition matrix from recently observed event patterns.
    """

    def __init__(self, episodic: "EpisodicMemoryProvider") -> None:
        self._episodic = episodic

    def seed_markov(self, predictor: "L1MarkovPredictor", lookback: int = 100) -> int:
        """Extract recent event sequences and feed them into the predictor.
    
        Uses the predictor's public ``import_state()`` API to inject transitions
        without accessing private internals.
    
        Returns the number of transitions seeded.
        """
        fragments = self._episodic.recent(limit=lookback)
        if len(fragments) < 2:
            return 0
    
        # Build a pseudo action_ring from recent event types
        action_sequence = [f.event_type for f in fragments]
    
        # Build transition counts via sliding window
        ngram_n = predictor.export_state().get("ngram_n", 3)
        transitions: dict[str, dict[str, int]] = {}
        totals: dict[str, int] = {}
    
        for i in range(ngram_n, len(action_sequence)):
            key = "\u2192".join(action_sequence[i - ngram_n: i])
            action = action_sequence[i]
            transitions.setdefault(key, {})[action] = (
                transitions.get(key, {}).get(action, 0) + 1
            )
            totals[key] = totals.get(key, 0) + 1
    
        if not transitions:
            return 0
    
        # Merge into the predictor via public API
        existing = predictor.export_state()
        existing_transitions = existing.get("transitions", {})
        existing_totals = existing.get("totals", {})
    
        # Merge new transitions into existing (additive)
        for key, actions in transitions.items():
            bucket = existing_transitions.setdefault(key, {})
            for act, count in actions.items():
                bucket[act] = bucket.get(act, 0) + count
            existing_totals[key] = existing_totals.get(key, 0) + totals[key]
    
        predictor.import_state({
            "ngram_n": ngram_n,
            "transitions": existing_transitions,
            "totals": existing_totals,
        })
    
        transitions_seeded = sum(totals.values())
        logger.info(
            "EpisodicSequenceAdapter seeded %d transitions from %d events",
            transitions_seeded, len(fragments),
        )
        return transitions_seeded


# ────────────────────────────────────────────────────────────────────────────
# L2 Adapter: ExperienceEmbedAdapter (EmbeddingSearchProvider Protocol)
# ────────────────────────────────────────────────────────────────────────────


class ExperienceEmbedAdapter:
    """EmbeddingSearchProvider backed by ExperienceStore prediction experiences.

    Bridges L2 embedding search to the world model's accumulated prediction
    experiences.  Lower delta (prediction error) implies higher relevance —
    the experience was accurately predicted, meaning the pattern is reliable.
    """

    def __init__(self, experience_store: "ExperienceStore") -> None:
        self._store = experience_store

    async def search_similar(
        self, query: str, *, limit: int = 5
    ) -> List[EmbeddingHit]:
        """Retrieve experiences semantically similar to the query context."""
        # Decompose query: first token as app_context, rest as action description
        parts = query.split(maxsplit=1)
        app_context = parts[0] if parts else ""
        action_desc = parts[1] if len(parts) > 1 else query

        try:
            experiences = self._store.retrieve_similar(
                action_desc, app_context, limit=limit
            )
        except Exception as exc:
            logger.debug("ExperienceEmbedAdapter.search_similar failed: %s", exc)
            return []

        hits: List[EmbeddingHit] = []
        for exp in experiences:
            # Low delta = high similarity (accurate past prediction = reliable pattern)
            similarity = max(0.0, 1.0 - exp.delta)
            hits.append(EmbeddingHit(
                action_description=exp.actual_effect or exp.action_description,
                similarity_score=similarity,
                experience_id=exp.experience_id,
            ))
        return hits


# ────────────────────────────────────────────────────────────────────────────
# L3 Adapter: MemoryRAGAdapter (RAGProvider Protocol)
# ────────────────────────────────────────────────────────────────────────────


class MemoryRAGAdapter:
    """RAGProvider combining WorkingMemory conversation + ExperienceStore experiences.

    Provides rich context for L3 LLM prediction by merging:
    1. Recent conversation turns (user intent / dialogue context)
    2. Similar past prediction experiences (what worked / failed before)
    """

    def __init__(
        self,
        working: "WorkingMemoryProvider",
        experience: "ExperienceStore",
    ) -> None:
        self._working = working
        self._experience = experience

    async def retrieve(self, query: str, *, limit: int = 3) -> List[RAGHit]:
        """Combine conversation context + experience retrieval."""
        hits: List[RAGHit] = []

        # 1. Recent conversation context (user intent signal)
        conversation_context = self._extract_conversation_context()
        if conversation_context:
            hits.append(RAGHit(
                content=f"## Recent Conversation\n{conversation_context}",
                relevance_score=0.9,
            ))

        # 2. Similar past experiences from world model
        try:
            parts = query.split(maxsplit=1)
            app = parts[0] if parts else ""
            action = parts[1] if len(parts) > 1 else query
            experiences = self._experience.retrieve_similar(action, app, limit=limit)
            for exp in experiences:
                relevance = max(0.0, 1.0 - exp.delta)
                hits.append(RAGHit(
                    content=(
                        f"Past: {exp.action_description} → {exp.actual_effect} "
                        f"(delta={exp.delta:.2f})"
                    ),
                    experience_id=exp.experience_id,
                    relevance_score=relevance,
                ))
        except Exception as exc:
            logger.debug("MemoryRAGAdapter experience retrieval failed: %s", exc)

        # Sort by relevance, cap at limit + 1 (conversation + N experiences)
        hits.sort(key=lambda h: -h.relevance_score)
        return hits[: limit + 1]

    def _extract_conversation_context(self) -> str:
        """Extract recent user/assistant messages as a compact context string."""
        messages = self._working.as_chat_messages()
        # Take last 4 messages (2 user-assistant pairs)
        recent = [
            m for m in messages[-6:]
            if isinstance(m.get("role"), str) and m["role"] in ("user", "assistant")
        ][-4:]
        if not recent:
            return ""
        lines: List[str] = []
        for msg in recent:
            role = msg.get("role", "?")
            content = str(msg.get("content", ""))[:120]
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)
