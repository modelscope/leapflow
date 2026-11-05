# Copyright (c) Alibaba, Inc. and its affiliates.
"""Discovery and selection registry for inference strategies.

Mirrors the transport factory table: strategies arrive from three sources
(built-in registration at import time, the ``leapflow.inference.strategies``
entry-point group, and runtime :meth:`InferenceStrategyRegistry.register`) and
are keyed by their ``strategy_id`` in one global namespace.  The registry is
first-wins on name collisions -- an installed package cannot hijack a built-in
id -- and thread-safe under an ``RLock`` so concurrent sessions can share one
instance.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from leapflow.robot.inference.strategy import ComputeBudget, InferenceStrategy

logger = logging.getLogger(__name__)

_EP_GROUP = "leapflow.inference.strategies"

__all__ = ["InferenceStrategyRegistry", "get_default_registry"]


class InferenceStrategyRegistry:
    """Discovers and manages physical AI inference strategies.

    Discovery sources (in priority order):
    1. Built-in strategies (registered at import time)
    2. Entry-point group 'leapflow.inference.strategies'
    3. Runtime registration via register()

    Thread-safe (RLock protected).
    """

    def __init__(self) -> None:
        self._strategies: dict[str, InferenceStrategy] = {}
        self._lock = threading.RLock()
        self._ep_scanned = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, strategy: InferenceStrategy) -> None:
        """Register a strategy under its ``strategy_id``.

        First-wins: a strategy whose id is already registered is skipped with a
        warning rather than overwriting the incumbent, so an out-of-tree plugin
        cannot silently replace a built-in path.
        """
        strategy_id = getattr(strategy, "strategy_id", "")
        if not strategy_id:
            raise ValueError("strategy must expose a non-empty 'strategy_id'")
        if not isinstance(strategy, InferenceStrategy):
            raise TypeError(
                f"object for {strategy_id!r} does not satisfy InferenceStrategy"
            )
        with self._lock:
            if strategy_id in self._strategies:
                logger.warning(
                    "inference strategy %r already registered; keeping incumbent",
                    strategy_id,
                )
                return
            self._strategies[strategy_id] = strategy
            logger.debug("inference strategy %r registered", strategy_id)

    def unregister(self, strategy_id: str) -> None:
        """Remove a strategy by id.  A no-op when the id is unknown."""
        with self._lock:
            self._strategies.pop(strategy_id, None)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, strategy_id: str) -> InferenceStrategy | None:
        """Return the strategy registered for *strategy_id*, or ``None``."""
        self._discover_entry_points()
        with self._lock:
            return self._strategies.get(strategy_id)

    def list_strategies(self) -> list[dict[str, Any]]:
        """Return a serialisable summary of every registered strategy.

        Each entry carries the id plus the declared compute profile, so a
        caller can inspect capabilities without instantiating anything heavy.
        """
        self._discover_entry_points()
        with self._lock:
            items = list(self._strategies.items())

        summary: list[dict[str, Any]] = []
        for strategy_id, strategy in items:
            profile = getattr(strategy, "compute_profile", None)
            entry: dict[str, Any] = {"strategy_id": strategy_id}
            if profile is not None:
                entry["compute_profile"] = {
                    "latency_range_ms": list(profile.latency_range_ms),
                    "supports_chunking": profile.supports_chunking,
                    "supports_ensemble": profile.supports_ensemble,
                    "supports_refinement": profile.supports_refinement,
                    "gpu_required": profile.gpu_required,
                    "max_chunk_size": profile.max_chunk_size,
                }
            summary.append(entry)
        return summary

    def select(
        self,
        *,
        budget: ComputeBudget | None = None,
        gpu_available: bool = False,
    ) -> InferenceStrategy | None:
        """Auto-select the best strategy matching the budget constraints.

        A strategy is a candidate when it satisfies both constraints:
        - its declared minimum latency fits within ``budget.max_latency_ms``;
        - it does not require a GPU that is unavailable in this process.

        Among the candidates the one with the lowest typical minimum latency
        wins (fastest declared path).  Returns ``None`` when nothing qualifies.
        """
        self._discover_entry_points()
        with self._lock:
            candidates = list(self._strategies.values())

        max_latency = budget.max_latency_ms if budget is not None else None

        best: InferenceStrategy | None = None
        best_latency = float("inf")
        for strategy in candidates:
            profile = getattr(strategy, "compute_profile", None)
            if profile is None:
                continue
            if profile.gpu_required and not gpu_available:
                continue
            typical_min = profile.latency_range_ms[0]
            if max_latency is not None and typical_min > max_latency:
                continue
            if typical_min < best_latency:
                best = strategy
                best_latency = typical_min
        return best

    # ------------------------------------------------------------------
    # Entry-point discovery
    # ------------------------------------------------------------------

    def _discover_entry_points(self) -> None:
        """Merge entry-point declared strategies into the table, once.

        Idempotent: guarded by ``_ep_scanned`` so the metadata scan runs only on
        first lookup.  Built-in and runtime registrations take precedence -- an
        entry-point whose id collides with an existing key is skipped.
        """
        with self._lock:
            if self._ep_scanned:
                return
            self._ep_scanned = True

        try:
            from importlib.metadata import entry_points
        except ImportError:  # defensive: should never happen on 3.11+
            return

        try:
            eps = entry_points(group=_EP_GROUP)
        except TypeError:
            # Ancient importlib.metadata (not reachable on >=3.11).
            eps = entry_points().get(_EP_GROUP, [])  # type: ignore[arg-type,union-attr]

        for ep in eps:
            try:
                factory = ep.load()
            except Exception as exc:  # noqa: BLE001 - one bad plugin must not break discovery
                logger.warning(
                    "inference strategy entry-point %r failed to load: %s",
                    ep.name, exc, exc_info=True,
                )
                continue
            try:
                strategy = factory() if callable(factory) else factory
            except Exception as exc:  # noqa: BLE001 - construction must not break discovery
                logger.warning(
                    "inference strategy entry-point %r failed to construct: %s",
                    ep.name, exc, exc_info=True,
                )
                continue
            try:
                self.register(strategy)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "inference strategy entry-point %r rejected: %s", ep.name, exc,
                )


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------

_default_registry: InferenceStrategyRegistry | None = None
_default_registry_lock = threading.Lock()


def get_default_registry() -> InferenceStrategyRegistry:
    """Return the process-wide default strategy registry.

    Lazily created on first access.  Built-in strategies are NOT
    auto-registered here (they require a policy path or server address
    at construction) -- the registry starts empty and strategies are
    registered on demand as policies are loaded.

    Sharing one registry across sessions lets a heavy local model that
    is loaded once be reused by every plugin instance in the process,
    without forcing callers through daemon-level dependency injection.
    """
    global _default_registry
    if _default_registry is None:
        with _default_registry_lock:
            if _default_registry is None:
                _default_registry = InferenceStrategyRegistry()
    return _default_registry
