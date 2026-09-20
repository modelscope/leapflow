# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unified error classifier bridging LLM, tool, and system failures into FailureEnvelope.

Provides a single classification entry point that produces FailureEnvelope instances
from any error source. Wraps the existing ErrorClassifier for LLM errors and adds
structured classification for tool results and system exceptions.

Tool-result and system-error classification uses data-driven rule tables for
timeout/connection/network detection, mirroring the registry pattern in
error_classifier.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, FrozenSet, List

from leapflow.engine.error_classifier import ErrorCategory, ErrorClassifier
from leapflow.engine.failure_envelope import (
    FailureContext,
    FailureEnvelope,
    FailureSource,
    Recoverability,
    RecoveryHint,
    SideEffectState,
)
from leapflow.llm.credential_state import AllCredentialsExhausted

logger = logging.getLogger(__name__)

# Category for failures caused by a defect in LeapFlow rather than by a provider,
# a tool, or the environment. Kept distinct so it is never confused with a
# provider condition that recovery strategies could plausibly repair.
INTERNAL_DEFECT_CATEGORY = "internal_defect"

# Default mapping from ErrorCategory to (Recoverability, default_failure_class)
_DEFAULT_MAPPINGS: dict[str, tuple[Recoverability, str]] = {
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
    # Defects in LeapFlow itself. Registered so the mapping is discoverable, but
    # reached through exception type rather than message text (see below).
    INTERNAL_DEFECT_CATEGORY: (Recoverability.NON_RECOVERABLE, "internal_defect"),
}

# Exception types that always mean "LeapFlow has a bug", never "the provider or
# the network did something". They must bypass the provider taxonomy entirely:
# ``ErrorClassifier`` categorizes by substring on the exception message, so a
# mistyped attribute name that happened to contain "context" was classified as a
# context overflow and driven through compression, provider failover, and
# credential rotation before halting the turn with an unrelated reason. Matching
# on Python's own exception hierarchy is exact, unlike matching on message text.
_INTERNAL_DEFECT_TYPES: tuple[type[BaseException], ...] = (
    AttributeError,
    NameError,
    TypeError,
    IndexError,
    KeyError,
    ImportError,
    AssertionError,
    NotImplementedError,
)


class RecoverabilityRegistry:
    """Extensible registry for category -> recoverability mapping.

    Allows plugins and configuration to register new error categories
    without modifying this module.
    """

    def __init__(self) -> None:
        self._mappings: dict[str, tuple[Recoverability, str]] = dict(_DEFAULT_MAPPINGS)

    def register(self, category: str, recoverability: Recoverability,
                 failure_class: str = "") -> None:
        """Register or override a category mapping."""
        self._mappings[category] = (recoverability, failure_class or category)

    def get(self, category: str) -> tuple[Recoverability, str]:
        """Get recoverability for a category, defaulting to NON_RECOVERABLE."""
        return self._mappings.get(category, (Recoverability.NON_RECOVERABLE, "unknown"))

    def categories(self) -> list[str]:
        """List all registered categories."""
        return list(self._mappings.keys())

# Permission failure classes that indicate non-recoverable tool permission errors
_PERMISSION_FAILURE_CLASSES = frozenset({"authorization", "scope_denied"})
_PERMISSION_FAILURE_CODES = frozenset({"access_denied", "missing_scope", "platform_degraded"})


# ---------------------------------------------------------------------------
# Data-driven tool/system error message classification rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolMessageRule:
    """A keyword-based rule for classifying tool/system error messages."""
    category: str
    keywords: FrozenSet[str]
    description: str = ""


# Rules for classify_tool_result — timeout detection by substring
_TOOL_TIMEOUT_KEYWORDS: FrozenSet[str] = frozenset({"timeout", "timed out"})

# Rules for classify_system_error — ordered, first match wins
_SYSTEM_ERROR_RULES: List[ToolMessageRule] = [
    ToolMessageRule(
        category="system_timeout",
        keywords=frozenset({"timeout", "timed out"}),
        description="Timeout in system call",
    ),
    ToolMessageRule(
        category="system_network",
        keywords=frozenset({"connection", "refused", "reset"}),
        description="Network connectivity issues",
    ),
]


class UnifiedErrorClassifier:
    """Unified classification entry point producing FailureEnvelope from any error source.

    Bridges:
    - LLM/API exceptions -> FailureEnvelope (wraps existing ErrorClassifier)
    - Tool result dicts -> FailureEnvelope (new structured classification)
    - System exceptions -> FailureEnvelope (new)
    """

    def __init__(self, error_classifier: Any = None,
                 registry: RecoverabilityRegistry | None = None) -> None:
        """Accept existing ErrorClassifier instance for LLM error classification.

        Args:
            error_classifier: An ErrorClassifier instance. If None, a default is created.
            registry: Optional RecoverabilityRegistry for extensible category mapping.
        """
        if error_classifier is not None and isinstance(error_classifier, ErrorClassifier):
            self._classifier: ErrorClassifier = error_classifier
        else:
            self._classifier = ErrorClassifier()
        self._registry = registry or RecoverabilityRegistry()

    def classify_internal_defect(
        self,
        exc: Exception,
        *,
        provider: str = "",
        model: str = "",
    ) -> FailureEnvelope:
        """Classify an exception that indicates a defect in LeapFlow itself.

        Reported as non-recoverable on purpose: no retry, compression, failover,
        or credential rotation can fix a programming error, and attempting them
        wastes the turn and misattributes the cause. The envelope names the
        exception type so the halt message is actionable.
        """
        detail = f"{type(exc).__name__}: {exc}"
        logger.error("internal defect surfaced during an LLM call: %s", detail, exc_info=True)
        return FailureEnvelope.create(
            source=FailureSource.SYSTEM,
            category=INTERNAL_DEFECT_CATEGORY,
            failure_class="internal_defect",
            failure_code=f"internal_{type(exc).__name__.lower()}",
            message=detail[:500],
            recoverability=Recoverability.NON_RECOVERABLE,
            side_effect_state=SideEffectState.NONE,
            context=FailureContext.from_dict_args(
                tool_name="",
                arguments={"provider": provider, "model": model} if provider or model else None,
            ),
            provider_hint=RecoveryHint(
                hint_text=(
                    f"This is a defect in LeapFlow ({type(exc).__name__}), not a provider or "
                    "network problem. Retrying will not help; the traceback is in the daemon log."
                )
            ),
        )

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

        Exceptions that indicate a LeapFlow defect are routed away from the
        provider taxonomy first: the provider classifier reads message text, so a
        local bug would otherwise be assigned whatever provider condition its
        message happens to resemble.
        """
        if isinstance(exc, _INTERNAL_DEFECT_TYPES):
            return self.classify_internal_defect(exc, provider=provider, model=model)

        if isinstance(exc, AllCredentialsExhausted):
            return self._classify_credentials_exhausted(exc, provider=provider, model=model)

        category = self._classifier.classify(exc)
        category_str = category.value

        recoverability, failure_class = self._registry.get(category_str)

        # Build recovery hint from the classifier's friendly message
        friendly_msg = self._classifier.friendly_message(category, str(exc)[:200])
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

    def _classify_credentials_exhausted(
        self,
        exc: AllCredentialsExhausted,
        *,
        provider: str = "",
        model: str = "",
    ) -> FailureEnvelope:
        """Classify a fully-drained credential pool as admin-required.

        Reached by exception type, not message text: when every credential for
        every provider is dead or cooling down, no retry / failover / rotation
        can proceed — an operator must add or restore a key. Mapped to
        ``auth_permanent`` (permanent auth semantics) with ``ADMIN_REQUIRED``
        recoverability so the turn halts with an actionable reason instead of
        looping through recovery strategies that have nothing left to try.
        """
        logger.error("all LLM credentials exhausted: %s", exc)
        return FailureEnvelope.create(
            source=FailureSource.LLM,
            category=ErrorCategory.AUTH_PERMANENT.value,
            failure_class="auth_permanent",
            failure_code="llm_credentials_exhausted",
            message=str(exc)[:500],
            recoverability=Recoverability.ADMIN_REQUIRED,
            side_effect_state=SideEffectState.NONE,
            context=FailureContext.from_dict_args(
                tool_name="",
                arguments={"provider": provider or exc.provider, "model": model}
                if (provider or exc.provider or model) else None,
            ),
            provider_hint=RecoveryHint(
                hint_text=(
                    "All configured LLM API keys are unusable "
                    f"({exc.dead} revoked/billing-dead, {exc.cooling_down} rate-limited). "
                    "Add or restore a valid key, or wait for rate limits to reset."
                )
            ),
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
        retryable = bool(result.get("retryable") if result.get("retryable") is not None else True)

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

        # 3. Timeout detection (data-driven keyword set)
        error_lower = error_msg.lower()
        if any(kw in error_lower for kw in _TOOL_TIMEOUT_KEYWORDS):
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
        """Classify a system-level exception (resource, timeout, etc.).

        Uses a combination of type-based dispatch (for Python exception types)
        and data-driven keyword rules (for message-based classification).
        """
        msg = str(exc).lower()
        exc_type = type(exc).__name__

        # Type-based dispatch (most specific first)
        if isinstance(exc, TimeoutError):
            return FailureEnvelope.create(
                source=FailureSource.SYSTEM,
                category="system_timeout",
                failure_class="timeout",
                failure_code="system_timeout",
                message=str(exc)[:500],
                recoverability=Recoverability.AUTO_RETRY,
                side_effect_state=SideEffectState.NONE,
            )

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

        # Data-driven message classification (keyword rule table)
        for rule in _SYSTEM_ERROR_RULES:
            if any(kw in msg for kw in rule.keywords):
                if rule.category == "system_timeout":
                    return FailureEnvelope.create(
                        source=FailureSource.SYSTEM,
                        category=rule.category,
                        failure_class="timeout",
                        failure_code="system_timeout",
                        message=str(exc)[:500],
                        recoverability=Recoverability.AUTO_RETRY,
                        side_effect_state=SideEffectState.NONE,
                    )
                if rule.category == "system_network":
                    return FailureEnvelope.create(
                        source=FailureSource.SYSTEM,
                        category=rule.category,
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
