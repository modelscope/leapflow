"""Recovery decision types for the agent loop recovery subsystem.

A RecoveryDecision encapsulates what the coordinator decided to do about a
FailureEnvelope: which action to take, the retry semantics, the strategy
that produced the decision, and audit metadata.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from leapflow.engine.failure_envelope import FailureEnvelope

if TYPE_CHECKING:
    from leapflow.engine.interaction_request import InteractionRequest


class RecoveryAction(Enum):
    """Possible recovery actions the coordinator can prescribe."""

    RETRY_WITH_BACKOFF = "retry_with_backoff"
    TRANSFORM_AND_RETRY = "transform_and_retry"
    FAILOVER = "failover"
    HALT_CLEAN = "halt_clean"
    HALT_WITH_CHECKPOINT = "halt_with_checkpoint"
    ASK_USER = "ask_user"
    SKIP_AND_CONTINUE = "skip_and_continue"


@dataclass(frozen=True)
class BackoffConfig:
    """Backoff algorithm parameters for retry decisions."""

    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter_ratio: float = 0.5
    algorithm: str = "decorrelated_jitter"


@dataclass(frozen=True)
class RetrySemantics:
    """Describes how a retry interacts with the recovery budget."""

    consumes_retry_budget: bool = True
    resets_retry_count: bool = False
    backoff_config: BackoffConfig | None = None


@dataclass(frozen=True)
class RecoveryDecision:
    """Immutable record of a recovery decision.

    Produced by RecoveryCoordinator.evaluate() and consumed by the agent loop
    to determine the next control-flow action (retry, halt, ask, skip).
    """

    decision_id: str
    envelope: FailureEnvelope
    action: RecoveryAction
    reason: str
    strategy_key: str
    retry_semantics: RetrySemantics = field(default_factory=RetrySemantics)
    budget_cost: int = 0
    audit_metadata: tuple[tuple[str, Any], ...] = ()
    interaction: InteractionRequest | None = None
    transform_description: str = ""
    failover_target: str = ""

    @classmethod
    def create(
        cls,
        *,
        envelope: FailureEnvelope,
        action: RecoveryAction,
        reason: str,
        strategy_key: str,
        retry_semantics: RetrySemantics | None = None,
        budget_cost: int = 0,
        audit_metadata: dict[str, Any] | None = None,
        interaction: InteractionRequest | None = None,
        transform_description: str = "",
        failover_target: str = "",
    ) -> RecoveryDecision:
        """Factory that auto-generates decision_id and normalizes audit_metadata."""
        meta_tuple = tuple(sorted((audit_metadata or {}).items()))
        return cls(
            decision_id=uuid.uuid4().hex,
            envelope=envelope,
            action=action,
            reason=reason,
            strategy_key=strategy_key,
            retry_semantics=retry_semantics or RetrySemantics(),
            budget_cost=budget_cost,
            audit_metadata=meta_tuple,
            interaction=interaction,
            transform_description=transform_description,
            failover_target=failover_target,
        )

    @property
    def audit_metadata_dict(self) -> dict[str, Any]:
        """Reconstruct audit_metadata as a dict for serialization."""
        return dict(self.audit_metadata)

    @property
    def is_terminal(self) -> bool:
        """Whether this decision ends the current turn."""
        return self.action in (
            RecoveryAction.HALT_CLEAN,
            RecoveryAction.HALT_WITH_CHECKPOINT,
            RecoveryAction.ASK_USER,
        )

    @property
    def is_retry(self) -> bool:
        """Whether this decision involves retrying the failed operation."""
        return self.action in (
            RecoveryAction.RETRY_WITH_BACKOFF,
            RecoveryAction.TRANSFORM_AND_RETRY,
            RecoveryAction.FAILOVER,
        )
