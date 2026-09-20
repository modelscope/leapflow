# Copyright (c) Alibaba, Inc. and its affiliates.
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

import hashlib
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, FrozenSet

logger = logging.getLogger(__name__)


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
    # Lowered from 1024 to 768 (P0-OPT-2) to allow earlier COMMITTED entry
    # in sessions whose stable prefix is large enough for amortization but
    # below the previous threshold.  The other gates (difficulty, posture,
    # remaining_rounds, projected_savings > 0) still prevent premature
    # commitment on short or trivial tasks.
    min_prefix_tokens: int = 768
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


@dataclass(frozen=True)
class CommitmentEnforcement:
    """Frozen snapshot of the disclosure state at the moment of cache commitment.

    Captures the exact disclosure level, tool set, and system-prompt identity
    so that subsequent turns can reproduce a byte-identical prefix.  The
    ``frozen_level`` field stores a :class:`DisclosureLevel` *value* (which is
    a plain ``str`` because ``DisclosureLevel`` is ``str, Enum``).  Keeping the
    type as ``str`` avoids a circular import between this module and
    ``context_disclosure``.
    """

    frozen_level: str
    """DisclosureLevel value (e.g. 'core', 'expanded', 'full')."""

    frozen_tool_names: tuple[str, ...]
    """Sorted tuple of tool names that were active at commitment time."""

    frozen_system_prompt_hash: str
    """SHA-256 hex digest of the system prompt at commitment time."""

    committed_at_turn: int
    """Turn index at which the enforcement was established."""


def _system_prompt_hash(system_prompt: str) -> str:
    """Compute a stable SHA-256 hex digest for a system prompt string."""
    return hashlib.sha256(system_prompt.encode("utf-8", errors="replace")).hexdigest()


class PrefixCommitmentController:
    """Per-task controller: decides (once) whether to commit the prefix.

    Stateless w.r.t. the decision math (``should_commit`` is pure); holds only
    the monotonic commitment state, reset per task via :meth:`reset`.

    **Enforcement lifecycle**:  After commitment, :meth:`enforce` freezes a
    snapshot of the current disclosure state.  :meth:`break_commitment` clears
    the enforcement without reverting ``CommitmentStatus`` (the commitment
    decision itself is still monotonic; only the *enforcement* is revocable so
    the planner can fall back to normal PCD when the prefix drifts).
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
        self._enforcement: CommitmentEnforcement | None = None

    # ── read-only accessors ───────────────────────────────────────────

    @property
    def state(self) -> PrefixCommitmentState:
        return self._state

    @property
    def committed(self) -> bool:
        return self._state.committed

    @property
    def enforcement(self) -> CommitmentEnforcement | None:
        """Return the active enforcement snapshot, or ``None`` if not enforced."""
        return self._enforcement

    # ── lifecycle ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear commitment state at the start of a new task/turn."""
        self._state = PrefixCommitmentState()
        self._enforcement = None

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

    # ── enforcement ───────────────────────────────────────────────────

    def enforce(
        self,
        current_level: str,
        current_tool_names: tuple[str, ...],
        system_prompt_hash: str,
        turn_index: int,
    ) -> CommitmentEnforcement | None:
        """Freeze the current disclosure snapshot once committed.

        On the first call after commitment, captures a
        :class:`CommitmentEnforcement` snapshot.  Subsequent calls while
        enforcement is active return the existing snapshot unchanged.

        Parameters are plain values (``str``, ``tuple``) rather than rich
        domain types to avoid a circular import with ``context_disclosure``.

        Returns:
            The active enforcement snapshot, or ``None`` if not yet committed.
        """
        if not self._state.committed:
            return None
        if self._enforcement is not None:
            return self._enforcement
        self._enforcement = CommitmentEnforcement(
            frozen_level=str(current_level),
            frozen_tool_names=tuple(sorted(current_tool_names)),
            frozen_system_prompt_hash=system_prompt_hash,
            committed_at_turn=turn_index,
        )
        logger.debug(
            "PrefixCommitment: enforced level=%s tools=%d turn=%d",
            current_level, len(current_tool_names), turn_index,
        )
        return self._enforcement

    def should_break_commitment(
        self,
        *,
        posture_changed: bool = False,
        tool_error: bool = False,
        slash_command: bool = False,
        transform_retry: bool = False,
    ) -> bool:
        """Return whether enforcement should be broken.

        Any structural disruption (posture shift, tool error, slash command,
        recovery-driven transform retry) means the stable prefix assumption
        no longer holds and the planner should fall back to normal PCD.
        """
        return posture_changed or tool_error or slash_command or transform_retry

    def break_commitment(self) -> None:
        """Clear enforcement without reverting the commitment decision.

        The ``CommitmentStatus`` remains COMMITTED (the decision is monotonic
        per the 7.2.6 contract), but the enforcement snapshot is discarded so
        :meth:`DisclosurePlanner.plan` will run normal PCD logic instead of
        freezing the disclosure level.  A new :meth:`enforce` call can
        re-establish enforcement if the prefix stabilizes again.
        """
        if self._enforcement is not None:
            logger.debug(
                "PrefixCommitment: enforcement broken (was level=%s turn=%d)",
                self._enforcement.frozen_level,
                self._enforcement.committed_at_turn,
            )
        self._enforcement = None

    def force_commit(self) -> None:
        """Force the controller into the committed state.

        Intended for session restoration / resume paths where the prior
        session was already committed.  The caller must follow up with
        :meth:`enforce` to re-establish the enforcement snapshot.
        """
        if self._state.committed:
            return
        self._state = PrefixCommitmentState(
            status=CommitmentStatus.COMMITTED,
            committed_at_round=-1,
            prefix_token_estimate=0,
            projected_savings=0.0,
            reason="force_commit (session restore)",
        )
        logger.debug("PrefixCommitment: force-committed for session restore")
