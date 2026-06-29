"""Level 3: Semantic Cache — VLM extraction result reuse across sessions.

Caches VLM action extraction results keyed by visual content similarity,
enabling cross-session reuse for recurring visual patterns.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from leapflow.perception.types import VisualAction

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cached extraction result."""

    actions: List[VisualAction]
    created_at: float
    hit_count: int = 0


class SemanticCache:
    """Content-addressed cache for VLM frame-pair extraction results.

    Key insight: many frame pairs are visually similar across sessions
    (e.g., "open app X" always looks the same). Cache the extraction
    result to avoid redundant VLM calls.

    Cache key is derived from quantized frame embeddings + detected app,
    ensuring visually similar pairs share cache entries.
    """

    __slots__ = ("_store", "_max_size", "_ttl_s", "_hits", "_misses")

    def __init__(self, max_size: int = 1000, ttl_days: int = 7) -> None:
        self._store: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._ttl_s = ttl_days * 86400.0
        self._hits = 0
        self._misses = 0

    def get(self, cache_key: str) -> Optional[List[VisualAction]]:
        """Look up cached actions by key. Returns None on miss/expiry."""
        entry = self._store.get(cache_key)
        if entry is None:
            self._misses += 1
            return None

        if self._is_expired(entry):
            del self._store[cache_key]
            self._misses += 1
            return None

        entry.hit_count += 1
        self._hits += 1
        return entry.actions

    def put(self, cache_key: str, actions: List[VisualAction]) -> None:
        """Store extraction results."""
        if len(self._store) >= self._max_size and cache_key not in self._store:
            self._evict_oldest()

        self._store[cache_key] = CacheEntry(
            actions=actions,
            created_at=time.time(),
        )

    def build_key(
        self,
        embedding_a: Optional[List[float]],
        embedding_b: Optional[List[float]],
        app: str = "unknown",
    ) -> str:
        """Build a content-based cache key from frame embeddings and app.

        Quantizes embeddings to reduce key space (LSH-inspired).
        Falls back to empty-hash if embeddings unavailable.
        """
        a_bytes = self._quantize_embedding(embedding_a)
        b_bytes = self._quantize_embedding(embedding_b)
        raw = f"{app}:{a_bytes.hex()}:{b_bytes.hex()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(hit_rate, 1),
            "size": len(self._store),
        }

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0

    def _is_expired(self, entry: CacheEntry) -> bool:
        return (time.time() - entry.created_at) > self._ttl_s

    def _evict_oldest(self) -> None:
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k].created_at)
        del self._store[oldest_key]

    @staticmethod
    def _quantize_embedding(embedding: Optional[List[float]], bits: int = 64) -> bytes:
        """Quantize a float embedding to binary hash (LSH-style)."""
        if not embedding:
            return b"\x00" * (bits // 8)

        # Simple sign-based quantization: positive → 1, negative → 0
        n_bytes = bits // 8
        result = bytearray(n_bytes)
        for i in range(min(bits, len(embedding))):
            if embedding[i] > 0:
                result[i // 8] |= (1 << (7 - i % 8))
        return bytes(result)
