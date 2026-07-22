"""Adaptive prefix-commitment decision (mechanism 7, W2 slice 2).

Decides whether a task should *commit* to a stable, cacheable prompt prefix.
Prefix caching rewards stability, not mere length: committing pays off only when
a long horizon amortizes the first-call write cost against cheap cached reads.

This module owns the decision only (pure, testable). Enforcement -- freezing
disclosure at FULL, byte-stabilizing the tool payload, cache-aware compression,
layered layout, and provider breakpoints -- is a separate concern (W2 slice 3).

Design contract (aligns with the design doc 7.2):
- Commitment is monotonic: UNCOMMITTED -> COMMITTED, never back (7.2.6).
- Triggered by *expansion / long-horizon* signals, never by convergence: a task
  that is wrapping up must not newly commit a large prefix it cannot amortize.
- The commit predicate is a deterministic amortization inequality over a
  provider-neutral CachePriceModel; no natural-language fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, FrozenSet


class CommitmentStatus(str, Enum):
    """Lifecycle of the per-task prefix commitment (monotonic)."""

    UNCOMMITTED = "uncommitted"
    COMMITTED = "committed"


@dataclass(frozen=True)
class CachePriceModel:
    """Relative per-token prices for a provider's prefix cache.

    Prices are normalized so an uncached (miss) prompt token costs ``price_miss``
    (1.0 by default). ``price_read`` is a cached-read token (providers bill ~0.1x)
    and ``price_write`` is the first-materialization premium (auto-cache
    providers 1.0, Anthropic ~1.25 for the 5m TTL). Provided by the LLM adapter;
    the decision logic is provider-neutral and only consumes this model.
    """

    price_miss: float = 1.0
    price_read: float = 0.1
    price_write: float = 1.0


@dataclass(frozen=True)
class PrefixCommitmentConfig:
    """Thresholds for the commitment decision (7.2.2)."""

    commit_difficulty_threshold: float = 0.60
    min_prefix_tokens: int = 1024
    min_remaining_rounds: int = 3
    margin: float = 0.15
    # Expansion / long-horizon postures that make committing worthwhile. Note
    # this deliberately excludes converging/finalizing (near-end): see 7.2.2.
    expansion_postures: FrozenSet[str] = frozenset({"research", "expanding"})


@dataclass(frozen=True)
class PrefixCommitmentState:
    """Immutable snapshot of the current commitment (surfaced for observability)."""

    status: CommitmentStatus = CommitmentStatus.UNCOMMITTED
    committed_at_round: int = -1
    prefix_token_estimate: int = 0
    projected_savings: float = 0.0
    reason: str = ""

    @property
    def committed(self) -> bool:
        return self.status is CommitmentStatus.COMMITTED

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "committed": self.committed,
            "committed_at_round": self.committed_at_round,
            "prefix_token_estimate": self.prefix_token_estimate,
            "projected_savings": round(self.projected_savings, 2),
            "reason": self.reason,
        }


class PrefixCommitmentController:
    """Per-task controller: decides (once) whether to commit the prefix.

    Stateless w.r.t. the decision math (``should_commit`` is pure); holds only
    the monotonic commitment state, reset per task via :meth:`reset`.
    """

    def __init__(
        self,
        *,
        config: PrefixCommitmentConfig | None = None,
        price_model: CachePriceModel | None = None,
    ) -> None:
        self._config = config or PrefixCommitmentConfig()
        self._price = price_model or CachePriceModel()
        self._state = PrefixCommitmentState()

    @property
    def state(self) -> PrefixCommitmentState:
        return self._state

    @property
    def committed(self) -> bool:
        return self._state.committed

    def reset(self) -> None:
        """Clear commitment state at the start of a new task/turn."""
        self._state = PrefixCommitmentState()

    def projected_savings(
        self,
        *,
        remaining_rounds: int,
        est_full_prefix_tokens: int,
        est_pcd_prefix_tokens: int,
    ) -> float:
        """Effective-cost savings of committing vs churning over the horizon (7.2.3).

        Positive means committing is cheaper. ``cost_commit`` writes the stable
        prefix once then reads it cheaply; ``cost_nocommit`` re-encodes the
        churning prefix at the miss price every remaining round.
        """
        price = self._price
        rounds = max(1, remaining_rounds)
        cost_commit = (
            est_full_prefix_tokens * price.price_write
            + (rounds - 1) * est_full_prefix_tokens * price.price_read
        )
        cost_nocommit = rounds * est_pcd_prefix_tokens * price.price_miss
        return cost_nocommit - cost_commit * (1.0 + self._config.margin)

    def should_commit(
        self,
        *,
        difficulty: float,
        posture: str,
        remaining_rounds: int,
        est_full_prefix_tokens: int,
        est_pcd_prefix_tokens: int,
    ) -> bool:
        """Deterministic commit predicate (7.2.2 gates + 7.2.3 amortization)."""
        cfg = self._config
        if difficulty < cfg.commit_difficulty_threshold:
            return False
        if posture not in cfg.expansion_postures:
            return False
        if est_full_prefix_tokens < cfg.min_prefix_tokens:
            return False
        if remaining_rounds < cfg.min_remaining_rounds:
            return False
        return self.projected_savings(
            remaining_rounds=remaining_rounds,
            est_full_prefix_tokens=est_full_prefix_tokens,
            est_pcd_prefix_tokens=est_pcd_prefix_tokens,
        ) > 0.0

    def evaluate(
        self,
        *,
        difficulty: float,
        posture: str,
        round_number: int,
        remaining_rounds: int,
        est_full_prefix_tokens: int,
        est_pcd_prefix_tokens: int,
    ) -> PrefixCommitmentState:
        """Evaluate and (once) transition to COMMITTED. Monotonic (7.2.1)."""
        if self._state.committed:
            return self._state
        if self.should_commit(
            difficulty=difficulty,
            posture=posture,
            remaining_rounds=remaining_rounds,
            est_full_prefix_tokens=est_full_prefix_tokens,
            est_pcd_prefix_tokens=est_pcd_prefix_tokens,
        ):
            savings = self.projected_savings(
                remaining_rounds=remaining_rounds,
                est_full_prefix_tokens=est_full_prefix_tokens,
                est_pcd_prefix_tokens=est_pcd_prefix_tokens,
            )
            self._state = PrefixCommitmentState(
                status=CommitmentStatus.COMMITTED,
                committed_at_round=round_number,
                prefix_token_estimate=est_full_prefix_tokens,
                projected_savings=savings,
                reason=f"difficulty={difficulty:.2f} posture={posture} R={remaining_rounds}",
            )
        return self._state
