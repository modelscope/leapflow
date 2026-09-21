# Copyright (c) Alibaba, Inc. and its affiliates.
"""Recovery sub-package — error classification, failure envelopes, and recovery coordination."""
from __future__ import annotations

from leapflow.engine.recovery.error_classifier import ErrorCategory, ErrorClassifier
from leapflow.engine.recovery.failure_envelope import (
    FailureEnvelope,
    Recoverability,
    SideEffectState,
)
from leapflow.engine.recovery.interaction_request import (
    InteractionRequest,
    InteractionType,
    Severity,
    SuggestedAction,
    TimeoutBehavior,
)
from leapflow.engine.recovery.oneshot_guard import OneShotGuard
from leapflow.engine.recovery.recovery_audit import JsonlAuditSink, create_audit_entry
from leapflow.engine.recovery.recovery_budget import RecoveryBudget
from leapflow.engine.recovery.recovery_checkpoint import (
    InMemoryCheckpointStore,
    RecoveryCheckpoint,
)
from leapflow.engine.recovery.recovery_coordinator import (
    RecoveryCoordinator,
    RecoveryState,
    RecoveryStrategy,
)
from leapflow.engine.recovery.recovery_decision import (
    BackoffConfig,
    RecoveryAction,
    RecoveryDecision,
    RetrySemantics,
)
from leapflow.engine.recovery.strategies import default_strategies
from leapflow.engine.recovery.turn_recovery import TurnRecoveryState
from leapflow.engine.recovery.unified_classifier import UnifiedErrorClassifier

__all__ = [
    "BackoffConfig",
    "ErrorCategory",
    "ErrorClassifier",
    "FailureEnvelope",
    "InMemoryCheckpointStore",
    "InteractionRequest",
    "InteractionType",
    "JsonlAuditSink",
    "OneShotGuard",
    "Recoverability",
    "RecoveryAction",
    "RecoveryBudget",
    "RecoveryCheckpoint",
    "RecoveryCoordinator",
    "RecoveryDecision",
    "RecoveryState",
    "RecoveryStrategy",
    "RetrySemantics",
    "Severity",
    "SideEffectState",
    "SuggestedAction",
    "TimeoutBehavior",
    "TurnRecoveryState",
    "UnifiedErrorClassifier",
    "create_audit_entry",
    "default_strategies",
]
