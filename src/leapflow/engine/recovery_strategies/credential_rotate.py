# Copyright (c) Alibaba, Inc. and its affiliates.
"""Credential rotation recovery strategy.

Handles auth errors and rate limiting by rotating to alternate credentials
(API keys, tokens) from the credential pool.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from leapflow.engine.failure_envelope import FailureEnvelope
from leapflow.engine.recovery_budget import RecoveryBudget
from leapflow.engine.recovery_coordinator import RecoveryState
from leapflow.engine.recovery_decision import (
    RecoveryAction,
    RecoveryDecision,
    RetrySemantics,
)


@runtime_checkable
class CredentialAvailability(Protocol):
    """Inspector for whether alternate credentials remain worth rotating to.

    Structural (duck-typed) so the engine can inject the ``FailoverChain``
    without this module importing the ``llm`` layer.
    """

    def has_rotatable_credentials(self) -> bool: ...


class CredentialRotateStrategy:
    """Rotate credentials on authentication or rate-limit failures.

    When an API key is invalid, expired, or rate-limited, this strategy
    attempts to switch to alternate credentials from the configured pool.
    Credential rotation is a form of failover at the authentication level.
    """

    def __init__(self, credential_availability: CredentialAvailability | None = None) -> None:
        self._availability = credential_availability

    @property
    def key(self) -> str:
        return "credential_rotate"

    @property
    def priority(self) -> int:
        return 25

    @property
    def repeatable(self) -> bool:
        return False

    @property
    def applicable_sources(self) -> frozenset[str]:
        return frozenset({"llm"})

    @property
    def applicable_categories(self) -> frozenset[str]:
        return frozenset({"auth_error", "rate_limited", "billing"})

    def can_apply(self, envelope: FailureEnvelope, state: RecoveryState,
                  budget: RecoveryBudget | None = None) -> bool:
        """Applicable when rotation budget and a rotatable credential remain.

        Bows out when the pool reports no rotatable credential left (all keys
        dead or cooling down): rotating again would only loop, so
        ``ProviderFailoverStrategy`` should take over instead.
        """
        if budget is not None and not budget.can_rotate():
            return False
        if self._availability is not None and not self._availability.has_rotatable_credentials():
            return False
        return True

    def decide(self, envelope: FailureEnvelope, state: RecoveryState) -> RecoveryDecision:
        """Return FAILOVER decision for credential rotation."""
        return RecoveryDecision.create(
            envelope=envelope,
            action=RecoveryAction.FAILOVER,
            reason=f"Credential issue ({envelope.category}): rotating to alternate credentials",
            strategy_key=self.key,
            retry_semantics=RetrySemantics(
                consumes_retry_budget=True,
                resets_retry_count=True,
            ),
            budget_cost=1,
            audit_metadata={"trigger_category": envelope.category},
        )
