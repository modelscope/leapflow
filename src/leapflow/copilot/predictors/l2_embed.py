"""L2 Embedding Retrieval Predictor — semantic similarity search.

Retrieves historically similar contexts via vector embedding nearest-neighbour
search.  Uses a Protocol-based provider so that any vector store backend
(DuckDB VSS, FAISS, Chroma, etc.) can be injected without modifying prediction
logic.

Thread-safety: Relies on the underlying EmbeddingSearchProvider being
async-safe.  The predictor itself holds no mutable state beyond configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol

from leapflow.copilot.types import (
    ContextState,
    FeedbackSignal,
    PredictionCandidate,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Provider Protocol + DTO
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EmbeddingHit:
    """A single result from vector similarity search."""

    action_description: str
    similarity_score: float
    experience_id: str = ""
    skill_id: Optional[str] = None


class EmbeddingSearchProvider(Protocol):
    """Abstraction over the vector similarity search backend.

    Implementations may target DuckDB VSS extension, FAISS, Chroma, or
    any other ANN index.
    """

    async def search_similar(
        self, query: str, *, limit: int = 5
    ) -> List[EmbeddingHit]:
        """Return the top-K most similar embedding hits for the query string."""
        ...


# ────────────────────────────────────────────────────────────────────────────
# L2 Predictor
# ────────────────────────────────────────────────────────────────────────────


class L2EmbeddingPredictor:
    """L2 Embedding Retrieval Predictor — semantic nearest-neighbour search.

    Constructs a textual query from the current context (app + window + recent
    actions), passes it to an EmbeddingSearchProvider, and returns candidates
    whose similarity exceeds a configurable threshold.

    Confidence is computed as ``similarity_score * decay_factor`` to avoid
    over-reliance on semantic proximity alone.

    Lifecycle:
      - Constructed once at startup with a provider instance.
      - ``predict`` is called when the engine schedules L2 (< 100ms budget).
      - ``on_feedback`` is a no-op for L2 (embedding index is updated externally).

    Usage::

        provider = MyEmbeddingStore(...)
        predictor = L2EmbeddingPredictor(provider)
        candidates = await predictor.predict(context)
    """

    def __init__(
        self,
        provider: EmbeddingSearchProvider,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        decay_factor: float = 0.8,
    ) -> None:
        self._provider = provider
        self._top_k = top_k
        self._threshold = similarity_threshold
        self._decay = decay_factor

    # ── PredictorLayer Protocol ────────────────────────────────────────────

    @property
    def layer_id(self) -> str:
        return "L2"

    @property
    def priority(self) -> int:
        return 2

    @property
    def timeout_ms(self) -> int:
        return 100

    async def predict(self, context: ContextState) -> List[PredictionCandidate]:
        """Search for semantically similar historical contexts."""
        query = self._build_query(context)

        try:
            hits = await self._provider.search_similar(
                query, limit=self._top_k
            )
        except Exception as exc:
            logger.error("L2 embedding search failed: %s", exc)
            return []

        candidates: List[PredictionCandidate] = []
        for hit in hits:
            if hit.similarity_score <= self._threshold:
                continue
            confidence = hit.similarity_score * self._decay
            candidates.append(
                PredictionCandidate(
                    action_description=hit.action_description,
                    confidence=min(confidence, 0.99),
                    source_layer="L2",
                    context_hash=context.context_hash,
                    display_delay_ms=800,
                    skill_id=hit.skill_id,
                    reasoning=f"similar to experience: {hit.experience_id}",
                )
            )
        return candidates

    async def on_feedback(self, signal: FeedbackSignal) -> None:
        """No-op — embedding index is updated externally by EvolutionLoop."""
        pass

    # ── Internal ───────────────────────────────────────────────────────────

    def _build_query(self, context: ContextState) -> str:
        """Construct a textual query from the operational context."""
        parts = [context.app_bundle, context.window_title]
        parts.extend(context.action_ring[-3:])
        return " ".join(p for p in parts if p)
