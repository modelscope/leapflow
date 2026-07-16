"""Unified error classifier bridging LLM, tool, and system failures into FailureEnvelope.

Provides a single classification entry point that produces FailureEnvelope instances
from any error source. Wraps the existing ErrorClassifier for LLM errors and adds
structured classification for tool results and system exceptions.
"""
from __future__ import annotations

import logging
from typing import Any

from leapflow.engine.error_classifier import ErrorCategory, ErrorClassifier
from leapflow.engine.failure_envelope import (
    FailureContext,
    FailureEnvelope,
    FailureSource,
    Recoverability,
    RecoveryHint,
    SideEffectState,
)

logger = logging.getLogger(__name__)

# Mapping from ErrorCategory to (Recoverability, default_failure_class)
_CATEGORY_RECOVERABILITY: dict[str, tuple[Recoverability, str]] = {
    ErrorCategory.TRANSIENT.value: (Recoverability.AUTO_RETRY, "transient"),
    ErrorCategory.RATE_LIMITED.value: (Recoverability.AUTO_RETRY, "rate_limited"),
    ErrorCategory.OVERLOADED.value: (Recoverability.AUTO_RETRY, "overloaded"),
    ErrorCategory.CONTEXT_OVERFLOW.value: (Recoverability.AUTO_RECOVER, "context_overflow"),
    ErrorCategory.PAYLOAD_TOO_LARGE.value: (Recoverability.AUTO_RECOVER, "payload_too_large"),
    ErrorCategory.FORMAT_ERROR.value: (Recoverability.AUTO_RECOVER, "format_error"),
    ErrorCategory.TOOL_FAILURE.value: (Recoverability.USER_FIXABLE, "tool_failure"),
    ErrorCategory.AUTH_ERROR.value: (Recoverability.AUTO_RETRY, "auth_error"),
    ErrorCategory.AUTH_PERMANENT.value: (Recoverability.NON_RECOVERABLE, "auth_permanent"),
    ErrorCategory.BILLING.value: (Recoverability.USER_FIXABLE, "billing"),
    ErrorCategory.CONTENT_BLOCKED.value: (Recoverability.NON_RECOVERABLE, "content_blocked"),
    ErrorCategory.MODEL_NOT_FOUND.value: (Recoverability.USER_FIXABLE, "model_not_found"),
    ErrorCategory.IMAGE_TOO_LARGE.value: (Recoverability.AUTO_RECOVER, "image_too_large"),
    ErrorCategory.SSL_ERROR.value: (Recoverability.NON_RECOVERABLE, "ssl_error"),
    ErrorCategory.PERMANENT.value: (Recoverability.NON_RECOVERABLE, "permanent"),
}

# Permission failure classes that indicate non-recoverable tool permission errors
_PERMISSION_FAILURE_CLASSES = frozenset({"authorization", "scope_denied"})
_PERMISSION_FAILURE_CODES = frozenset({"access_denied", "missing_scope", "platform_degraded"})


class UnifiedErrorClassifier:
    """Unified classification entry point producing FailureEnvelope from any error source.

    Bridges:
    - LLM/API exceptions -> FailureEnvelope (wraps existing ErrorClassifier)
    - Tool result dicts -> FailureEnvelope (new structured classification)
    - System exceptions -> FailureEnvelope (new)
    """

    def __init__(self, error_classifier: Any = None) -> None:
        """Accept existing ErrorClassifier instance for LLM error classification.

        Args:
            error_classifier: An ErrorClassifier instance. If None, a default is created.
        """
        if error_classifier is not None and isinstance(error_classifier, ErrorClassifier):
            self._classifier: ErrorClassifier = error_classifier
        else:
            self._classifier = ErrorClassifier()

    def classify_llm_error(
        self,
        exc: Exception,
        *,
        provider: str = "",
        model: str = "",
    ) -> FailureEnvelope:
        """Classify an LLM/API exception into a FailureEnvelope.

        Uses the existing ErrorClassifier internally to determine the category,
        then maps to the appropriate recoverability and constructs a FailureEnvelope.
        """
        category = self._classifier.classify(exc)
        category_str = category.value

        recoverability, failure_class = _CATEGORY_RECOVERABILITY.get(
            category_str, (Recoverability.NON_RECOVERABLE, "unknown")
        )

        # Build recovery hint from the classifier's friendly message
        friendly_msg = ErrorClassifier.friendly_message(category, str(exc)[:200])
        hint = RecoveryHint(hint_text=friendly_msg) if friendly_msg else None

        return FailureEnvelope.create(
            source=FailureSource.LLM,
            category=category_str,
            failure_class=failure_class,
            failure_code=f"llm_{category_str}",
            message=str(exc)[:500],
            recoverability=recoverability,
            side_effect_state=SideEffectState.NONE,
            context=FailureContext.from_dict_args(
                tool_name="",
                arguments={"provider": provider, "model": model} if provider or model else None,
            ),
            provider_hint=hint,
        )

    def classify_tool_result(
        self,
        result: dict[str, Any],
        *,
        tool_name: str = "",
        execution_policy: str = "read_only",
    ) -> FailureEnvelope | None:
        """Classify a tool result dict. Returns None if result is not a failure.

        Non-failure conditions (returns None):
        - ok=True
        - duplicate_suppressed=True
        - counts_as_failure=False
        """
        # Non-failure fast paths
        if result.get("ok", True) is True:
            return None
        if result.get("duplicate_suppressed", False):
            return None
        if result.get("counts_as_failure") is False:
            return None

        # Extract structured failure fields
        failure_class = str(result.get("failure_class") or "")
        failure_code = str(result.get("failure_code") or "")
        error_msg = str(result.get("error") or "")
        error_type = str(result.get("error_type") or "")
        retryable = bool(result.get("retryable", True))

        # Determine side-effect state based on execution policy
        side_effect_state = self._side_effect_state_from_policy(execution_policy)

        # Classification rules (priority order)

        # 1. Permission failures
        if failure_class in _PERMISSION_FAILURE_CLASSES or failure_code in _PERMISSION_FAILURE_CODES:
            return FailureEnvelope.create(
                source=FailureSource.TOOL,
                category="tool_permission",
                failure_class=failure_class or "authorization",
                failure_code=failure_code or "permission_denied",
                message=error_msg or "Permission denied",
                recoverability=Recoverability.NON_RECOVERABLE,
                side_effect_state=side_effect_state,
                context=FailureContext.from_dict_args(tool_name=tool_name),
            )

        # 2. Unknown tool with retryable flag
        if error_type == "unknown_tool" and retryable:
            return FailureEnvelope.create(
                source=FailureSource.TOOL,
                category="tool_unknown",
                failure_class="unknown_tool",
                failure_code=failure_code or "tool_not_found",
                message=error_msg or f"Unknown tool: {tool_name}",
                recoverability=Recoverability.AUTO_RECOVER,
                side_effect_state=SideEffectState.NONE,
                context=FailureContext.from_dict_args(tool_name=tool_name),
            )

        # 3. Timeout detection
        error_lower = error_msg.lower()
        if "timeout" in error_lower or "timed out" in error_lower:
            timeout_recoverability = (
                Recoverability.AUTO_RETRY
                if execution_policy == "read_only"
                else Recoverability.USER_FIXABLE
            )
            return FailureEnvelope.create(
                source=FailureSource.TOOL,
                category="tool_timeout",
                failure_class="timeout",
                failure_code=failure_code or "execution_timeout",
                message=error_msg,
                recoverability=timeout_recoverability,
                side_effect_state=side_effect_state,
                context=FailureContext.from_dict_args(tool_name=tool_name),
            )

        # 4. Generic tool failure
        recoverability = (
            Recoverability.AUTO_RETRY if retryable else Recoverability.USER_FIXABLE
        )
        return FailureEnvelope.create(
            source=FailureSource.TOOL,
            category="tool_failure",
            failure_class=failure_class or "tool_error",
            failure_code=failure_code or "execution_failed",
            message=error_msg or "Tool execution failed",
            recoverability=recoverability,
            side_effect_state=side_effect_state,
            context=FailureContext.from_dict_args(tool_name=tool_name),
        )

    def classify_system_error(self, exc: Exception) -> FailureEnvelope:
        """Classify a system-level exception (resource, timeout, etc.)."""
        msg = str(exc).lower()
        exc_type = type(exc).__name__

        # Timeout errors
        if isinstance(exc, (TimeoutError,)) or "timeout" in msg or "timed out" in msg:
            return FailureEnvelope.create(
                source=FailureSource.SYSTEM,
                category="system_timeout",
                failure_class="timeout",
                failure_code="system_timeout",
                message=str(exc)[:500],
                recoverability=Recoverability.AUTO_RETRY,
                side_effect_state=SideEffectState.NONE,
            )

        # Memory errors
        if isinstance(exc, MemoryError):
            return FailureEnvelope.create(
                source=FailureSource.SYSTEM,
                category="system_resource",
                failure_class="memory_error",
                failure_code="out_of_memory",
                message=str(exc)[:500],
                recoverability=Recoverability.NON_RECOVERABLE,
                side_effect_state=SideEffectState.UNKNOWN,
            )

        # OS/IO errors
        if isinstance(exc, OSError):
            return FailureEnvelope.create(
                source=FailureSource.SYSTEM,
                category="system_io",
                failure_class="os_error",
                failure_code=f"errno_{getattr(exc, 'errno', 'unknown')}",
                message=str(exc)[:500],
                recoverability=Recoverability.AUTO_RETRY,
                side_effect_state=SideEffectState.UNKNOWN,
            )

        # Connection errors (subclass of OSError but check explicitly)
        if "connection" in msg or "refused" in msg or "reset" in msg:
            return FailureEnvelope.create(
                source=FailureSource.SYSTEM,
                category="system_network",
                failure_class="connection_error",
                failure_code="connection_failed",
                message=str(exc)[:500],
                recoverability=Recoverability.AUTO_RETRY,
                side_effect_state=SideEffectState.NONE,
            )

        # Generic system error
        return FailureEnvelope.create(
            source=FailureSource.SYSTEM,
            category="system_unknown",
            failure_class=exc_type.lower(),
            failure_code="unclassified",
            message=str(exc)[:500],
            recoverability=Recoverability.USER_FIXABLE,
            side_effect_state=SideEffectState.UNKNOWN,
        )

    @staticmethod
    def _side_effect_state_from_policy(policy: str) -> SideEffectState:
        """Map execution policy to the appropriate side-effect state for failures."""
        if policy == "read_only":
            return SideEffectState.NONE
        if policy == "external_side_effect":
            return SideEffectState.UNKNOWN
        if policy in ("mutating_idempotent", "mutating_once"):
            return SideEffectState.PARTIAL
        return SideEffectState.UNKNOWN
