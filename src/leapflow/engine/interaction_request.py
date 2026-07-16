"""Interaction request types for the agent loop recovery subsystem.

When automated recovery is insufficient (e.g., permissions, credentials,
ambiguous intent), the recovery coordinator emits an InteractionRequest
that the TUI or gateway surfaces to the user for resolution.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any


class InteractionType(Enum):
    """Classification of what the agent needs from the user."""

    APPROVAL = "approval"
    CLARIFICATION = "clarification"
    PARAMETER_CONFIRMATION = "parameter_confirmation"
    RECOVERY_GUIDANCE = "recovery_guidance"
    CREDENTIAL_SETUP = "credential_setup"
    PERMISSION_GRANT = "permission_grant"
    RETRY_CHOICE = "retry_choice"
    CONFLICT_RESOLUTION = "conflict_resolution"
    CONTINUE_AFTER_FIX = "continue_after_fix"
    DEGRADED_MODE_CONFIRM = "degraded_mode_confirm"


class Severity(Enum):
    """Visual and priority severity for an interaction request."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TimeoutBehavior(Enum):
    """What happens when an interaction request times out without user response."""

    CANCEL = "cancel"
    DEGRADE = "degrade"
    PERSIST = "persist"


@dataclass(frozen=True)
class SuggestedAction:
    """A suggested action the user can take to resolve an interaction request."""

    label: str
    command: str = ""
    description: str = ""
    is_default: bool = False


@dataclass(frozen=True)
class InteractionRequest:
    """Immutable interaction request emitted by the recovery coordinator.

    Represents a point where automated recovery cannot proceed without
    user input. The TUI or gateway layer surfaces this request and waits
    for user response before resuming the agent loop.
    """

    request_id: str
    interaction_type: InteractionType
    severity: Severity
    title: str
    description: str
    suggested_actions: tuple[SuggestedAction, ...] = ()
    context: tuple[tuple[str, Any], ...] = ()
    resumption_key: str = ""
    expires_at: float | None = None
    timeout_behavior: TimeoutBehavior = TimeoutBehavior.PERSIST

    @classmethod
    def create(
        cls,
        *,
        interaction_type: InteractionType,
        severity: Severity,
        title: str,
        description: str,
        suggested_actions: tuple[SuggestedAction, ...] = (),
        context: dict[str, Any] | None = None,
        resumption_key: str = "",
        expires_at: float | None = None,
        timeout_behavior: TimeoutBehavior = TimeoutBehavior.PERSIST,
    ) -> InteractionRequest:
        """Factory that auto-generates request_id and normalizes context."""
        ctx_tuple = tuple(sorted((context or {}).items()))
        return cls(
            request_id=uuid.uuid4().hex,
            interaction_type=interaction_type,
            severity=severity,
            title=title,
            description=description,
            suggested_actions=suggested_actions,
            context=ctx_tuple,
            resumption_key=resumption_key,
            expires_at=expires_at,
            timeout_behavior=timeout_behavior,
        )

    @property
    def context_dict(self) -> dict[str, Any]:
        """Reconstruct context as a dict for downstream consumers."""
        return dict(self.context)

    def to_json(self) -> dict[str, Any]:
        """Produce a JSON-serializable representation of this request."""
        return {
            "request_id": self.request_id,
            "interaction_type": self.interaction_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "suggested_actions": [
                {
                    "label": a.label,
                    "command": a.command,
                    "description": a.description,
                    "is_default": a.is_default,
                }
                for a in self.suggested_actions
            ],
            "context": self.context_dict,
            "resumption_key": self.resumption_key,
            "expires_at": self.expires_at,
            "timeout_behavior": self.timeout_behavior.value,
        }
