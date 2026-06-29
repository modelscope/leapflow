"""Embedding providers for semantic similarity in experience retrieval.

Abstracts the embedding source behind a Protocol so callers don't couple
to a specific model or library.  Two implementations ship out of the box:

  1. TFIDFEmbeddingProvider — zero external dependencies, deterministic
  2. LLMEmbeddingProvider  — delegates to the main LLMProvider (if it
     exposes an embedding endpoint)

Both are stateless beyond an LRU cache for repeated queries.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal contract for text → vector conversion."""

    def embed(self, text: str) -> List[float]: ...


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class TFIDFEmbeddingProvider:
    """Zero-dependency TF-IDF sparse-hash embedding.

    Maps tokens to a fixed-width vector via modular hashing, producing
    L2-normalised vectors suitable for cosine-similarity ranking.
    Good enough for short action-description strings without requiring
    any ML framework.
    """

    __slots__ = ("_dim", "_cache", "_cache_limit")

    def __init__(self, *, dim: int = 2048, cache_limit: int = 500) -> None:
        self._dim = dim
        self._cache: Dict[str, List[float]] = {}
        self._cache_limit = cache_limit

    def embed(self, text: str) -> List[float]:
        if text in self._cache:
            return self._cache[text]

        tokens = text.lower().split()
        tf = Counter(tokens)
        vec = [0.0] * self._dim
        for token, count in tf.items():
            idx = hash(token) % self._dim
            idf = 1.0 + 1.0 / (1.0 + len(token))
            vec[idx] += count * idf

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]

        if len(self._cache) < self._cache_limit:
            self._cache[text] = vec
        return vec


class LLMEmbeddingProvider:
    """Delegates embedding to the main LLM provider's embedding endpoint.

    Falls back to TFIDFEmbeddingProvider if the LLM doesn't support
    embeddings, keeping the caller unaware of the switch.
    """

    __slots__ = ("_llm", "_cache", "_cache_limit", "_fallback")

    def __init__(
        self,
        llm: "Any",
        *,
        cache_limit: int = 500,
        fallback: Optional[EmbeddingProvider] = None,
    ) -> None:
        self._llm = llm
        self._cache: Dict[str, List[float]] = {}
        self._cache_limit = cache_limit
        self._fallback = fallback or TFIDFEmbeddingProvider()

    def embed(self, text: str) -> List[float]:
        if text in self._cache:
            return self._cache[text]

        get_embedding = getattr(self._llm, "get_embedding", None)
        if get_embedding is None:
            return self._fallback.embed(text)

        try:
            vec = get_embedding(text)
            if vec and isinstance(vec, list):
                if len(self._cache) < self._cache_limit:
                    self._cache[text] = vec
                return vec
        except Exception:
            logger.debug("LLM embedding failed, using fallback", exc_info=True)

        return self._fallback.embed(text)
