"""Curiosity signal — composite intrinsic motivation for exploration.

Unifies three intrinsic motivation components from RL literature
(ICM prediction surprise, Bayesian information gain, count-based novelty)
into a single adaptive curiosity score, all computed without gradient updates.
"""

from __future__ import annotations

import math
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from leapflow.world_model.experience_store import ExperienceStore
    from leapflow.world_model.prediction import PredictionOutcome

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CuriosityConfig:
    """Weights and flags for curiosity signal composition."""

    alpha: float = 0.4     # Prediction Surprise weight
    beta: float = 0.3      # Information Gain weight
    gamma: float = 0.3     # Frequency Novelty weight
    auto_balance: bool = True

    # Maturity stage thresholds (event_count, experience_count)
    early_event_threshold: int = 100
    early_experience_threshold: int = 20
    middle_event_threshold: int = 500
    middle_experience_threshold: int = 100

    # Auto-balance weight presets per stage: (alpha, beta, gamma)
    early_weights: tuple = (0.2, 0.3, 0.5)
    middle_weights: tuple = (0.4, 0.4, 0.2)
    mature_weights: tuple = (0.6, 0.3, 0.1)

    # OPD advantage modulation strength ∈ [0, 1]
    advantage_modulation: float = 0.3


@dataclass(frozen=True)
class CuriosityScore:
    """Decomposed curiosity score with component breakdown."""

    total: float
    prediction_surprise: float
    information_gain: float
    frequency_novelty: float
    maturity_stage: str  # "early" | "middle" | "mature"


class CuriositySignal:
    """Computes a composite curiosity score from prediction outcomes.

    The signal drives exploration by combining:
    - Prediction Surprise (α): How wrong was the world model? (ICM analog)
    - Information Gain (β): How much did the causal graph's uncertainty change? (Bayesian)
    - Frequency Novelty (γ): How rare is this event pattern? (Count-based)

    Weights auto-balance based on learning maturity.
    """

    def __init__(
        self,
        config: CuriosityConfig,
        experience_store: "ExperienceStore",
        causal_graph: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._store = experience_store
        self._graph = causal_graph
        self._frequency_counter: Dict[str, int] = defaultdict(int)
        self._total_computations = 0

    def set_causal_graph(self, graph: Any) -> None:
        """Bind (or rebind) the session-scoped causal graph."""
        self._graph = graph

    def load_frequency_counter(self, counter: Dict[str, int]) -> None:
        """Restore frequency counter from persistent storage."""
        self._frequency_counter.update(counter)

    def compute(self, outcome: "PredictionOutcome") -> CuriosityScore:
        """Compute the composite curiosity score for a prediction outcome."""
        self._total_computations += 1

        ps = self._prediction_surprise(outcome)
        ig = self._information_gain(outcome)
        fn = self._frequency_novelty(outcome)

        alpha, beta, gamma = self._resolve_weights()
        total = alpha * ps + beta * ig + gamma * fn
        total = min(1.0, max(0.0, total))

        stage = self._maturity_stage()

        score = CuriosityScore(
            total=total,
            prediction_surprise=ps,
            information_gain=ig,
            frequency_novelty=fn,
            maturity_stage=stage,
        )

        logger.debug(
            "curiosity.compute total=%.3f ps=%.3f ig=%.3f fn=%.3f stage=%s",
            total, ps, ig, fn, stage,
        )
        return score

    def compute_with_trajectory_context(
        self,
        outcome: "PredictionOutcome",
        advantage: float = 0.0,
    ) -> CuriosityScore:
        """Compute curiosity modulated by OPD trajectory advantage.

        When the teacher has graded the trajectory:
        - Negative advantage (suboptimal actions) *amplifies* curiosity
          (the system should explore more in states where it performed poorly).
        - Positive advantage (already-good actions) *dampens* curiosity
          (less need to re-explore well-understood states).

        The modulation strength is controlled by ``CuriosityConfig.advantage_modulation``.
        """
        base = self.compute(outcome)
        modulation = self._config.advantage_modulation
        if advantage == 0.0 or modulation <= 0.0:
            return base

        modifier = 1.0 - modulation * advantage
        modifier = max(0.1, min(2.0, modifier))
        adjusted = min(1.0, max(0.0, base.total * modifier))
        return CuriosityScore(
            total=adjusted,
            prediction_surprise=base.prediction_surprise,
            information_gain=base.information_gain,
            frequency_novelty=base.frequency_novelty,
            maturity_stage=base.maturity_stage,
        )

    def _prediction_surprise(self, outcome: "PredictionOutcome") -> float:
        """ICM analog: direct use of prediction error δ."""
        return outcome.delta

    def _information_gain(self, outcome: "PredictionOutcome") -> float:
        """Bayesian analog: estimated entropy reduction from the observation.

        Computes H_before (current causal graph uncertainty) and estimates
        H_after by simulating how the observation would shift confidences
        toward certainty proportional to the prediction delta.
        """
        if self._graph is None:
            return 0.0

        events = getattr(self._graph, "events", {})
        if not events:
            return 0.0

        app_id = getattr(outcome.pre_snapshot, "app_bundle_id", "") if outcome.pre_snapshot else ""
        affected = [
            ev for ev in events.values()
            if hasattr(ev, "confidence") and hasattr(ev, "channel")
            and (not app_id or getattr(ev, "app_context", app_id) == app_id)
        ]

        if not affected:
            return 0.0

        sample = affected[-10:]
        h_before = 0.0
        h_after = 0.0
        for ev in sample:
            p = getattr(ev, "confidence", 0.5)
            h_before += _binary_entropy(p)
            p_updated = p + (1.0 - p) * outcome.delta * 0.5
            p_updated = max(0.01, min(0.99, p_updated))
            h_after += _binary_entropy(p_updated)

        n = len(sample)
        if n == 0:
            return 0.0

        gain = (h_before - h_after) / n
        return max(0.0, min(1.0, gain))

    def _frequency_novelty(self, outcome: "PredictionOutcome") -> float:
        """Count-based novelty blending action-pattern and causal-graph frequency.

        Action-level counter (app|action) tracks the agent's own experience;
        the CausalGraph's channel:event_type counter tracks raw event volume.
        Blending both avoids the stale-counter problem where the causal graph
        accumulates live data but the curiosity module only sees its own actions.
        """
        key = self._pattern_key(outcome)
        self._frequency_counter[key] += 1
        action_count = self._frequency_counter[key]
        action_novelty = 1.0 / math.sqrt(action_count)

        causal_novelty = self._causal_frequency_novelty(outcome)
        if causal_novelty < 0.0:
            return action_novelty
        return 0.6 * action_novelty + 0.4 * causal_novelty

    def _causal_frequency_novelty(self, outcome: "PredictionOutcome") -> float:
        """Derive novelty from the CausalGraph's live frequency counter.

        Returns a negative value if the graph has no frequency data (caller
        should fall back to action-only novelty).
        """
        if self._graph is None:
            return -1.0
        freq: dict = getattr(self._graph, "metadata", {}).get("frequency_counter", {})
        if not freq:
            return -1.0
        pre = outcome.pre_snapshot
        app = getattr(pre, "app_bundle_id", "") if pre else ""
        relevant = {k: v for k, v in freq.items() if not app or app.lower() in k.lower()} or freq
        if not relevant:
            return -1.0
        total = sum(relevant.values())
        return 1.0 / math.sqrt(total + 1)

    def _pattern_key(self, outcome: "PredictionOutcome") -> str:
        """Compress an outcome into a countable pattern key."""
        pre = outcome.pre_snapshot
        app = getattr(pre, "app_bundle_id", "unknown") if pre else "unknown"
        action = outcome.prediction.action_description[:50] if outcome.prediction else "unknown"
        return f"{app}|{action}"

    def _resolve_weights(self) -> Tuple[float, float, float]:
        """Resolve effective weights, optionally auto-balancing by maturity."""
        if not self._config.auto_balance:
            return self._config.alpha, self._config.beta, self._config.gamma
        stage = self._maturity_stage()
        cfg = self._config
        presets = {"early": cfg.early_weights, "middle": cfg.middle_weights, "mature": cfg.mature_weights}
        return presets.get(stage, cfg.mature_weights)

    def _maturity_stage(self) -> str:
        """Classify learning maturity based on accumulated data volume."""
        total_experiences = self._store.count()
        event_count = len(getattr(self._graph, "events", {})) if self._graph else 0
        cfg = self._config
        if event_count < cfg.early_event_threshold and total_experiences < cfg.early_experience_threshold:
            return "early"
        if event_count < cfg.middle_event_threshold and total_experiences < cfg.middle_experience_threshold:
            return "middle"
        return "mature"

    @property
    def frequency_snapshot(self) -> Dict[str, int]:
        """Return a copy of the frequency counter for persistence."""
        return dict(self._frequency_counter)


def _binary_entropy(p: float) -> float:
    """Binary entropy H(p) for a Bernoulli variable."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
