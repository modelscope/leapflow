"""PredictionEngine — multi-layer cascade prediction scheduler.

Orchestrates all registered PredictorLayer instances, executing them according
to priority with per-layer timeout control.  Aggregates results via confidence
deduplication and multi-layer consensus boosting.

Thread-safety: Designed for single asyncio event-loop execution.  All layer
calls are awaited sequentially or via asyncio.wait_for; no thread-level
concurrency is introduced.

Error isolation: Individual layer failures (timeout or exception) never
propagate — the engine logs the error and continues with remaining layers.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional

from leapflow.copilot.config import CopilotConfig
from leapflow.copilot.types import (
    ContextState,
    FeedbackSignal,
    PredictionCandidate,
    PredictorLayer,
)

if TYPE_CHECKING:
    from leapflow.copilot.degradation import DegradationPolicy

logger = logging.getLogger(__name__)


class PredictionEngine:
    """Multi-layer cascade prediction engine.

    Coordinates multiple PredictorLayer instances sorted by priority.
    Lower priority values execute first (L0 before L1 before L2 before L3).

    Execution strategy:
      - Each layer is invoked with ``asyncio.wait_for`` bounded by its
        own ``timeout_ms``.
      - Timeout or exception in one layer does NOT affect others.
      - Results are aggregated: same action from multiple layers gets a
        consensus confidence boost.
      - Final output is sorted by confidence descending.

    Lifecycle:
      - Construct with initial layers + config.
      - Call ``predict(context)`` per context update.
      - Call ``dispatch_feedback(signal)`` to propagate user feedback to all layers.
      - Use ``register_layer`` / ``unregister_layer`` for hot-swapping.

    Usage::

        engine = PredictionEngine(layers=[l0, l1, l2, l3], config=config)
        candidates = await engine.predict(context)
        await engine.dispatch_feedback(feedback_signal)
    """

    def __init__(
        self,
        layers: List[PredictorLayer],
        config: CopilotConfig,
        degradation: Optional["DegradationPolicy"] = None,
    ) -> None:
        self._layers: List[PredictorLayer] = sorted(
            layers, key=lambda l: l.priority
        )
        self._config = config
        self._degradation = degradation

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def layers(self) -> List[PredictorLayer]:
        """Read-only view of registered layers (sorted by priority)."""
        return list(self._layers)

    async def predict(self, context: ContextState) -> List[PredictionCandidate]:
        """Execute all layers and return aggregated, deduplicated candidates.

        Each layer runs with its own timeout.  Timeout or failure in any layer
        is logged but does not prevent other layers from executing.

        Returns:
            Candidates sorted by confidence (descending), deduplicated by
            action_description with multi-layer consensus boosting.
        """
        all_candidates: List[PredictionCandidate] = []

        for layer in self._layers:
            # Skip disabled layers based on config
            if not self._is_layer_enabled(layer.layer_id):
                continue

            try:
                candidates = await asyncio.wait_for(
                    layer.predict(context),
                    timeout=layer.timeout_ms / 1000.0,
                )
                all_candidates.extend(candidates)

                # Early stop: if a layer produced very high confidence results
                if any(c.confidence > 0.9 for c in candidates):
                    break

            except asyncio.TimeoutError:
                logger.warning(
                    "Layer %s timed out (%dms)", layer.layer_id, layer.timeout_ms
                )
                continue
            except Exception as exc:
                logger.error(
                    "Layer %s prediction failed: %s", layer.layer_id, exc
                )
                continue

        return self._aggregate(all_candidates)

    async def dispatch_feedback(self, signal: FeedbackSignal) -> None:
        """Broadcast a feedback signal to all registered layers.

        Each layer's ``on_feedback`` is called independently; failures in
        one layer do not affect others.
        """
        for layer in self._layers:
            try:
                await layer.on_feedback(signal)
            except Exception as exc:
                logger.error(
                    "Layer %s feedback handling failed: %s",
                    layer.layer_id,
                    exc,
                )

    def register_layer(self, layer: PredictorLayer) -> None:
        """Dynamically register a new prediction layer.

        The layer is inserted in priority order.  If a layer with the same
        layer_id already exists, it is replaced.
        """
        # Remove existing with same id (if any)
        self._layers = [
            l for l in self._layers if l.layer_id != layer.layer_id
        ]
        self._layers.append(layer)
        self._layers.sort(key=lambda l: l.priority)
        logger.info(
            "Registered layer %s (priority=%d)", layer.layer_id, layer.priority
        )

    def unregister_layer(self, layer_id: str) -> None:
        """Dynamically unregister a prediction layer by its ID.

        No-op if the layer is not found.
        """
        before = len(self._layers)
        self._layers = [l for l in self._layers if l.layer_id != layer_id]
        if len(self._layers) < before:
            logger.info("Unregistered layer %s", layer_id)

    # ── Internal ───────────────────────────────────────────────────────────

    def _is_layer_enabled(self, layer_id: str) -> bool:
        """Check if a layer is enabled via config toggles and degradation level."""
        # Config-level toggle
        toggle_map = {
            "L0": self._config.l0_enabled,
            "L1": self._config.l1_enabled,
            "L2": self._config.l2_enabled,
            "L3": self._config.l3_enabled,
        }
        if not toggle_map.get(layer_id, True):
            return False
        # Degradation-level gate
        if self._degradation is not None:
            allowed = self._degradation.allowed_layers(self._degradation.current_level)
            if layer_id not in allowed:
                return False
        return True

    def _aggregate(
        self, candidates: List[PredictionCandidate]
    ) -> List[PredictionCandidate]:
        """Deduplicate by action_description with multi-layer consensus boosting.

        When multiple layers predict the same action, their confidences are
        combined using the independence assumption:
            P(combined) = 1 - ∏(1 - P_i)

        The highest-confidence candidate's metadata is preserved.
        """
        if not candidates:
            return []

        # Group by action_description
        grouped: Dict[str, List[PredictionCandidate]] = {}
        for c in candidates:
            grouped.setdefault(c.action_description, []).append(c)

        merged: List[PredictionCandidate] = []
        for action, group in grouped.items():
            if len(group) == 1:
                merged.append(group[0])
                continue

            # Multi-layer consensus — independence assumption
            combined_conf = 1.0 - math.prod(
                1.0 - c.confidence for c in group
            )
            best = max(group, key=lambda c: c.confidence)
            merged.append(
                PredictionCandidate(
                    action_description=action,
                    confidence=min(combined_conf, 0.99),
                    source_layer=best.source_layer,
                    context_hash=best.context_hash,
                    display_delay_ms=min(c.display_delay_ms for c in group),
                    is_destructive=best.is_destructive,
                    skill_id=best.skill_id,
                    parameters=best.parameters,
                    reasoning=f"consensus from {[c.source_layer for c in group]}",
                )
            )

        return sorted(merged, key=lambda c: -c.confidence)
