"""SpeculativePipeline — proactive prediction cache with tiered warming.

Implements the "predict-before-idle" strategy: when an action is observed,
the pipeline immediately runs fast layers (L0+L1) synchronously, then
asynchronously warms deeper layers (L2, L3) into a tiered cache.  By the
time the user pauses, predictions are already available from memory.

Cache architecture:
  - instant: L0+L1 results (< 5ms), highest display priority
  - warm:    L2 results (< 100ms), used if instant cache is empty
  - deep:    L3 results (< 3s), used only for complex contexts

Thread-safety: Designed for single asyncio event-loop execution.
Background tasks (L2/L3) are scheduled via ``asyncio.create_task``.

Memory control: LRU eviction when cache exceeds ``config.speculative_cache_size``.
TTL eviction when entries exceed ``config.cache_ttl_seconds``.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from leapflow.copilot.config import CopilotConfig
from leapflow.copilot.engine import PredictionEngine
from leapflow.copilot.types import (
    ContextState,
    PredictionCandidate,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Cache Entry
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class CacheEntry:
    """A single cache slot holding prediction results for a context snapshot."""

    candidates: List[PredictionCandidate]
    tier: str  # "instant" | "warm" | "deep"
    created_at: float
    context_hash: str


# ────────────────────────────────────────────────────────────────────────────
# SpeculativePipeline
# ────────────────────────────────────────────────────────────────────────────


class SpeculativePipeline:
    """Speculative prediction pipeline — proactive cache warming.

    When an action is observed, the pipeline:
      1. Immediately runs L0+L1 → stores in 'instant' cache.
      2. Asynchronously submits L2 → stores in 'warm' cache on completion.
      3. Conditionally submits L3 → stores in 'deep' cache on completion.
      4. Evicts expired/overflowing entries.

    Consumers (IdleDetector / SuggestionRenderer) call ``get_best()`` to
    retrieve the highest-priority cached prediction without triggering any
    new computation.

    Lifecycle:
      - Constructed once at startup with engine + config.
      - ``on_action_observed`` is called on every context update.
      - ``get_best`` is called during idle windows.
      - ``invalidate`` clears all cache (e.g. on major context shift).

    Usage::

        pipeline = SpeculativePipeline(engine, config)
        await pipeline.on_action_observed(context)
        # ... later, during idle ...
        best = pipeline.get_best(min_confidence=0.4)
    """

    def __init__(
        self,
        engine: PredictionEngine,
        config: CopilotConfig,
    ) -> None:
        self._engine = engine
        self._config = config
        # LRU-ordered cache: context_hash → {tier → CacheEntry}
        self._cache: OrderedDict[str, Dict[str, CacheEntry]] = OrderedDict()
        # Track pending async tasks to prevent over-scheduling
        self._pending_tasks: List[asyncio.Task] = []  # type: ignore[type-arg]
        # Previous context for unsupervised observation (observe path)
        self._prev_context: Optional[ContextState] = None

    # ── Public API ─────────────────────────────────────────────────────────

    async def on_action_observed(self, context: ContextState) -> None:
        """Trigger speculative prediction on a new action observation.

        Runs L0+L1 synchronously (instant tier), schedules L2/L3 asynchronously.
        """
        # Shallow-copy to avoid mutation by later events in async tasks
        context = copy.copy(context)

        # ── Unsupervised observation: record prev_context → current_action ──
        if self._prev_context is not None and context.action_ring:
            current_action = context.action_ring[-1]
            for layer in self._engine.layers:
                if hasattr(layer, "observe"):
                    try:
                        await layer.observe(self._prev_context, current_action)
                    except Exception:
                        logger.debug(
                            "observe failed for layer %s",
                            getattr(layer, "layer_id", "?"),
                        )
        self._prev_context = copy.copy(context)

        ctx_hash = context.context_hash

        # Clean up completed background tasks
        self._pending_tasks = [t for t in self._pending_tasks if not t.done()]

        # 1. Run L0 + L1 synchronously (instant tier)
        instant_candidates = await self._run_fast_layers(context)
        if instant_candidates:
            self._store_entry(
                ctx_hash,
                CacheEntry(
                    candidates=instant_candidates,
                    tier="instant",
                    created_at=time.time(),
                    context_hash=ctx_hash,
                ),
            )

        # 2. Schedule L2 asynchronously (warm tier)
        task_l2 = asyncio.create_task(
            self._warm_layer(context, layer_priority=2, tier="warm")
        )
        self._pending_tasks.append(task_l2)

        # 3. Conditionally schedule L3 (deep tier)
        if self._should_invoke_deep(context):
            task_l3 = asyncio.create_task(
                self._warm_layer(context, layer_priority=3, tier="deep")
            )
            self._pending_tasks.append(task_l3)

        # 4. Evict expired / overflowing entries
        self._evict()

    def get_best(
        self, min_confidence: float = 0.3
    ) -> Optional[PredictionCandidate]:
        """Retrieve the best cached prediction across all tiers.

        Priority order: instant > warm > deep.
        Returns None if no candidate meets the min_confidence threshold
        or all cached entries are stale.
        """
        now = time.time()
        ttl = self._config.cache_ttl_seconds

        # Iterate cache in reverse insertion order (most recent first)
        for ctx_hash in reversed(list(self._cache.keys())):
            tiers = self._cache[ctx_hash]
            # Check tiers in priority order
            for tier_name in ("instant", "warm", "deep"):
                entry = tiers.get(tier_name)
                if entry is None:
                    continue
                # Skip stale entries
                if now - entry.created_at > ttl:
                    continue
                # Find best candidate in this entry
                for candidate in sorted(
                    entry.candidates, key=lambda c: -c.confidence
                ):
                    if candidate.confidence >= min_confidence:
                        # Move to end (LRU touch)
                        self._cache.move_to_end(ctx_hash)
                        return candidate
        return None

    def invalidate(self) -> None:
        """Clear all cached predictions.

        Called on major context shifts or when the user starts a new
        workflow that invalidates all speculative state.
        """
        self._cache.clear()
        # Cancel pending background tasks
        for task in self._pending_tasks:
            if not task.done():
                task.cancel()
        self._pending_tasks.clear()
        logger.debug("SpeculativePipeline cache invalidated")

    @property
    def cache_size(self) -> int:
        """Current number of context hashes in cache."""
        return len(self._cache)

    # ── Internal ───────────────────────────────────────────────────────────

    async def _run_fast_layers(
        self, context: ContextState
    ) -> List[PredictionCandidate]:
        """Execute L0 and L1 layers synchronously (within their timeouts)."""
        results: List[PredictionCandidate] = []
        for layer in self._engine.layers:
            if layer.priority > 1:
                break  # Only L0, L1
            try:
                candidates = await asyncio.wait_for(
                    layer.predict(context),
                    timeout=layer.timeout_ms / 1000.0,
                )
                results.extend(candidates)
            except asyncio.TimeoutError:
                logger.debug(
                    "Fast layer %s timed out in pipeline", layer.layer_id
                )
            except Exception as exc:
                logger.error(
                    "Fast layer %s failed in pipeline: %s", layer.layer_id, exc
                )
        return results

    async def _warm_layer(
        self, context: ContextState, *, layer_priority: int, tier: str
    ) -> None:
        """Asynchronously execute a specific layer and store results in cache."""
        ctx_hash = context.context_hash
        results: List[PredictionCandidate] = []

        for layer in self._engine.layers:
            if layer.priority != layer_priority:
                continue
            try:
                candidates = await asyncio.wait_for(
                    layer.predict(context),
                    timeout=layer.timeout_ms / 1000.0,
                )
                results.extend(candidates)
            except asyncio.TimeoutError:
                logger.debug(
                    "Warm layer %s timed out in pipeline", layer.layer_id
                )
            except Exception as exc:
                logger.error(
                    "Warm layer %s failed in pipeline: %s", layer.layer_id, exc
                )

        if results:
            self._store_entry(
                ctx_hash,
                CacheEntry(
                    candidates=results,
                    tier=tier,
                    created_at=time.time(),
                    context_hash=ctx_hash,
                ),
            )

    def _should_invoke_deep(self, context: ContextState) -> bool:
        """Determine if L3 (deep/LLM) should be invoked.

        Heuristic: only invoke if context is complex enough and no high-confidence
        instant results are available.
        """
        # Check if instant cache already has high-confidence results
        ctx_hash = context.context_hash
        tiers = self._cache.get(ctx_hash, {})
        instant = tiers.get("instant")
        if instant and any(c.confidence > 0.8 for c in instant.candidates):
            return False

        # Complexity gate
        unique_apps = len(
            set(a.split(":")[1] for a in context.action_ring if ":" in a)
        )
        return unique_apps >= 2 or len(context.action_ring) >= 5

    def _store_entry(self, ctx_hash: str, entry: CacheEntry) -> None:
        """Store a cache entry, respecting LRU eviction limits."""
        if ctx_hash not in self._cache:
            self._cache[ctx_hash] = {}
        self._cache[ctx_hash][entry.tier] = entry
        # Move to end (most recently used)
        self._cache.move_to_end(ctx_hash)

    def _evict(self) -> None:
        """Evict expired and overflowing cache entries."""
        now = time.time()
        ttl = self._config.cache_ttl_seconds
        max_size = self._config.speculative_cache_size

        # TTL eviction
        expired_keys = []
        for ctx_hash, tiers in self._cache.items():
            all_stale = all(
                now - entry.created_at > ttl for entry in tiers.values()
            )
            if all_stale:
                expired_keys.append(ctx_hash)

        for key in expired_keys:
            del self._cache[key]

        # LRU eviction (remove oldest if over capacity)
        while len(self._cache) > max_size:
            self._cache.popitem(last=False)
