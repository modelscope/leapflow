"""L1 Markov Sequence Predictor — N-gram transition probability model.

Maintains a transition count matrix over action sequences (N-gram keys).
Given the most recent N actions, predicts the most likely next actions
based on historical transition frequencies.

Thread-safety: Not thread-safe.  Designed to run within a single asyncio
event loop.  For multi-worker scenarios, use external synchronisation.

Persistence: Call ``export_state()`` / ``import_state()`` to serialise
the transition matrix for checkpoint/restore cycles.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from leapflow.copilot.types import (
    ContextState,
    FeedbackSignal,
    FeedbackType,
    PredictionCandidate,
)

logger = logging.getLogger(__name__)


class L1MarkovPredictor:
    """L1 N-gram Sequence Predictor — transition probability over action rings.

    Maintains a frequency table keyed by the last N actions (joined with '→').
    The predict() method returns up to ``top_k`` next-action candidates whose
    transition probability exceeds ``min_prob``.

    Lifecycle:
      - Constructed once at startup with configurable N and thresholds.
      - ``predict`` is called on every context update (< 10ms budget).
      - ``on_feedback`` performs online update of the transition matrix.
      - ``prune(min_count)`` removes low-frequency entries to bound memory.

    Usage::

        predictor = L1MarkovPredictor(ngram_n=3)
        candidates = await predictor.predict(context)
    """

    def __init__(
        self,
        *,
        ngram_n: int = 3,
        top_k: int = 5,
        min_prob: float = 0.1,
        max_keys: int = 5000,
    ) -> None:
        self._n = ngram_n
        self._top_k = top_k
        self._min_prob = min_prob
        self._max_keys = max_keys
        # transition_counts[context_key][action] = count
        self._transitions: Dict[str, Dict[str, int]] = {}
        # total count per context_key (denominator for probability)
        self._totals: Dict[str, int] = {}

    # ── PredictorLayer Protocol ────────────────────────────────────────────

    @property
    def layer_id(self) -> str:
        return "L1"

    @property
    def priority(self) -> int:
        return 1

    @property
    def timeout_ms(self) -> int:
        return 10

    async def predict(self, context: ContextState) -> List[PredictionCandidate]:
        """Predict next actions based on recent N-gram transition probabilities."""
        key = self._make_key(context.action_ring)
        if key not in self._transitions:
            return []

        total = self._totals.get(key, 0)
        if total == 0:
            return []

        candidates: List[PredictionCandidate] = []
        sorted_actions = sorted(
            self._transitions[key].items(), key=lambda x: -x[1]
        )

        for action, count in sorted_actions[: self._top_k]:
            confidence = count / total
            if confidence < self._min_prob:
                break
            candidates.append(
                PredictionCandidate(
                    action_description=action,
                    confidence=confidence,
                    source_layer="L1",
                    context_hash=context.context_hash,
                    display_delay_ms=500,
                )
            )
        return candidates

    async def on_feedback(self, signal: FeedbackSignal) -> None:
        """Online update: record the actual action taken after this context."""
        ctx = signal.context_at_feedback
        if ctx is None:
            return

        key = self._make_key(ctx.action_ring)
        # Determine the action to record
        if signal.feedback_type == FeedbackType.ACCEPT:
            actual = signal.candidate.action_description
        elif signal.actual_action:
            actual = signal.actual_action
        else:
            actual = signal.candidate.action_description

        self._transitions.setdefault(key, {})[actual] = (
            self._transitions.get(key, {}).get(actual, 0) + 1
        )
        self._totals[key] = self._totals.get(key, 0) + 1

        # Auto-prune when key count exceeds threshold
        if len(self._transitions) > self._max_keys:
            self.prune(min_count=2)

    # ── Public utilities ───────────────────────────────────────────────────

    def prune(self, min_count: int = 2) -> int:
        """Remove entries with total count below threshold to prevent memory bloat.

        Returns:
            Number of keys removed.
        """
        to_remove = [
            key for key, total in self._totals.items() if total < min_count
        ]
        for key in to_remove:
            del self._transitions[key]
            del self._totals[key]
        if to_remove:
            logger.debug("L1 pruned %d low-frequency keys", len(to_remove))
        return len(to_remove)

    def export_state(self) -> Dict[str, Any]:
        """Serialise internal state for persistence.

        Returns:
            A JSON-serialisable dict containing the full transition matrix.
        """
        return {
            "ngram_n": self._n,
            "transitions": self._transitions,
            "totals": self._totals,
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """Restore internal state from a previously exported snapshot.

        Args:
            state: Dict produced by ``export_state()``.
        """
        self._n = state.get("ngram_n", self._n)
        self._transitions = state.get("transitions", {})
        self._totals = state.get("totals", {})
        logger.info(
            "L1 imported state: %d keys", len(self._transitions)
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _make_key(self, action_ring: List[str]) -> str:
        """Build the N-gram lookup key from the last N actions."""
        return "→".join(action_ring[-self._n :])
