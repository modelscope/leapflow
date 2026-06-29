"""L0 Exact-Match Predictor — O(1) context-hash lookup.

Provides the fastest prediction path by matching the current ContextState hash
against historically observed action patterns.  Uses a Protocol-based store
abstraction so that any backend (DuckDB, Redis, in-memory) can be plugged in
without modifying prediction logic.

Thread-safety: The InMemoryContextHashStore is **not** thread-safe; wrap in
asyncio lock if shared across tasks.  In typical usage, a single event loop
owns the predictor exclusively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

from leapflow.copilot.types import (
    ContextState,
    FeedbackSignal,
    FeedbackType,
    PredictionCandidate,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Store Protocol + DTO
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContextHashHit:
    """Single match returned by a ContextHashStore query."""

    action: str
    accept_count: int
    total_count: int
    skill_id: Optional[str] = None

    @property
    def accept_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.accept_count / self.total_count


class ContextHashStore(Protocol):
    """Abstraction over the hash→action lookup backend.

    Implementations may target DuckDB (SkillLibrary), Redis, or a plain dict.
    """

    async def query_by_hash(self, context_hash: str) -> List[ContextHashHit]:
        """Return all recorded actions for the given context hash."""
        ...

    async def record_observation(
        self, context_hash: str, action: str, accepted: bool
    ) -> None:
        """Record a feedback observation for online learning."""
        ...


# ────────────────────────────────────────────────────────────────────────────
# InMemory implementation (dev / test)
# ────────────────────────────────────────────────────────────────────────────


class InMemoryContextHashStore:
    """Simple in-memory implementation of ContextHashStore.

    Suitable for unit tests and lightweight deployments.  All data is lost on
    process restart — use DuckDB-backed store for persistence.

    Supports capacity limits via ``max_entries``.  When exceeded, the entry
    with the lowest total_count is evicted (least-used eviction).
    """

    def __init__(self, max_entries: int = 10000) -> None:
        # context_hash → action → (accept_count, total_count)
        self._data: Dict[str, Dict[str, Tuple[int, int]]] = {}
        self._max_entries = max_entries

    async def query_by_hash(self, context_hash: str) -> List[ContextHashHit]:
        actions = self._data.get(context_hash)
        if not actions:
            return []
        return [
            ContextHashHit(
                action=action,
                accept_count=counts[0],
                total_count=counts[1],
            )
            for action, counts in actions.items()
        ]

    async def record_observation(
        self, context_hash: str, action: str, accepted: bool
    ) -> None:
        bucket = self._data.setdefault(context_hash, {})
        prev_accept, prev_total = bucket.get(action, (0, 0))
        bucket[action] = (
            prev_accept + (1 if accepted else 0),
            prev_total + 1,
        )
        # Evict least-used entries when over capacity
        if len(self._data) > self._max_entries:
            self._evict_least_used()

    def _evict_least_used(self) -> None:
        """Remove entries with the lowest total_count to stay within capacity."""
        # Compute total count per context_hash
        scored = [
            (ctx_hash, sum(c[1] for c in actions.values()))
            for ctx_hash, actions in self._data.items()
        ]
        scored.sort(key=lambda x: x[1])
        # Remove bottom 10% to avoid frequent evictions
        n_remove = max(1, len(scored) // 10)
        for ctx_hash, _ in scored[:n_remove]:
            del self._data[ctx_hash]
        logger.debug("InMemoryContextHashStore evicted %d entries", n_remove)


# ────────────────────────────────────────────────────────────────────────────
# L0 Predictor
# ────────────────────────────────────────────────────────────────────────────


class L0HashPredictor:
    """L0 Exact-Match Predictor — fastest path via context-hash lookup.

    Queries the configured ContextHashStore for actions historically associated
    with the current context hash.  Only candidates with accept_rate > 0.3 are
    emitted.

    Lifecycle:
      - Constructed once at startup with a store instance.
      - ``predict`` is called on every context update (< 5ms budget).
      - ``on_feedback`` updates the store's observation records.

    Usage::

        store = InMemoryContextHashStore()
        predictor = L0HashPredictor(store)
        candidates = await predictor.predict(context)
    """

    def __init__(self, store: ContextHashStore) -> None:
        self._store = store

    # ── PredictorLayer Protocol ────────────────────────────────────────────

    @property
    def layer_id(self) -> str:
        return "L0"

    @property
    def priority(self) -> int:
        return 0

    @property
    def timeout_ms(self) -> int:
        return 5

    async def predict(self, context: ContextState) -> List[PredictionCandidate]:
        """Lookup exact hash match in the store."""
        try:
            hits = await self._store.query_by_hash(context.context_hash)
        except Exception as exc:
            logger.error("L0 store query failed: %s", exc)
            return []

        candidates: List[PredictionCandidate] = []
        for hit in hits:
            if hit.accept_rate <= 0.3:
                continue
            candidates.append(
                PredictionCandidate(
                    action_description=hit.action,
                    confidence=min(hit.accept_rate, 0.99),
                    source_layer="L0",
                    context_hash=context.context_hash,
                    display_delay_ms=200,
                    skill_id=hit.skill_id,
                )
            )
        return candidates

    async def on_feedback(self, signal: FeedbackSignal) -> None:
        """Update accept/reject counts in the store."""
        accepted = signal.feedback_type == FeedbackType.ACCEPT
        try:
            await self._store.record_observation(
                signal.candidate.context_hash,
                signal.candidate.action_description,
                accepted,
            )
        except Exception as exc:
            logger.error("L0 feedback recording failed: %s", exc)
