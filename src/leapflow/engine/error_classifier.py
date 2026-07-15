"""Error classification and recovery strategy for agent loops.

Enhanced taxonomy inspired by hermes-agent/error_classifier.py:
- Fine-grained HTTP status disambiguation (402/429/400/5xx)
- Structured ClassifiedError with recovery hints
- Provider-agnostic pattern matching
- Config-driven recovery strategies (OCP)
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    TRANSIENT = "transient"
    RATE_LIMITED = "rate_limited"
    OVERLOADED = "overloaded"
    CONTEXT_OVERFLOW = "context_overflow"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    FORMAT_ERROR = "format_error"
    TOOL_FAILURE = "tool_failure"
    AUTH_ERROR = "auth_error"
    AUTH_PERMANENT = "auth_permanent"
    BILLING = "billing"
    CONTENT_BLOCKED = "content_blocked"
    MODEL_NOT_FOUND = "model_not_found"
    IMAGE_TOO_LARGE = "image_too_large"
    SSL_ERROR = "ssl_error"
    PERMANENT = "permanent"


@dataclass(frozen=True)
class RecoveryStrategy:
    """What the agent loop should do when this category of error occurs."""
    retry: bool = False
    backoff: bool = False
    compress: bool = False
    inform_llm: bool = False
    should_fallback: bool = False
    should_rotate_credential: bool = False
    max_retries: int = 0
    base_delay: float = 1.0


@dataclass
class ClassifiedError:
    """Rich error classification with structured recovery hints."""
    category: ErrorCategory
    status_code: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    message: str = ""
    retryable: bool = True
    should_compress: bool = False
    should_fallback: bool = False
    should_rotate_credential: bool = False
    error_context: Dict[str, Any] = field(default_factory=dict)


def build_recovery_map(
    *,
    transient_max_retries: int = 3,
    rate_limit_base_delay: float = 5.0,
) -> Dict[ErrorCategory, RecoveryStrategy]:
    """Build recovery strategy map from config parameters (OCP-friendly)."""
    return {
        ErrorCategory.TRANSIENT: RecoveryStrategy(
            retry=True, backoff=True, max_retries=transient_max_retries,
        ),
        ErrorCategory.RATE_LIMITED: RecoveryStrategy(
            retry=True, backoff=True, max_retries=transient_max_retries + 2,
            base_delay=rate_limit_base_delay,
        ),
        ErrorCategory.OVERLOADED: RecoveryStrategy(
            retry=True, backoff=True, max_retries=transient_max_retries,
            base_delay=rate_limit_base_delay * 2,
        ),
        ErrorCategory.CONTEXT_OVERFLOW: RecoveryStrategy(
            retry=True, compress=True, max_retries=1,
        ),
        ErrorCategory.PAYLOAD_TOO_LARGE: RecoveryStrategy(
            retry=True, compress=True, max_retries=1,
        ),
        ErrorCategory.FORMAT_ERROR: RecoveryStrategy(retry=True, max_retries=2),
        ErrorCategory.TOOL_FAILURE: RecoveryStrategy(inform_llm=True),
        ErrorCategory.AUTH_ERROR: RecoveryStrategy(
            should_rotate_credential=True, retry=True, max_retries=1,
        ),
        ErrorCategory.AUTH_PERMANENT: RecoveryStrategy(),
        ErrorCategory.BILLING: RecoveryStrategy(
            should_rotate_credential=True, should_fallback=True,
        ),
        ErrorCategory.CONTENT_BLOCKED: RecoveryStrategy(should_fallback=True),
        ErrorCategory.MODEL_NOT_FOUND: RecoveryStrategy(should_fallback=True),
        ErrorCategory.IMAGE_TOO_LARGE: RecoveryStrategy(retry=True, max_retries=1),
        ErrorCategory.SSL_ERROR: RecoveryStrategy(),
        ErrorCategory.PERMANENT: RecoveryStrategy(),
    }


RECOVERY_MAP: Dict[ErrorCategory, RecoveryStrategy] = build_recovery_map()

# User-facing, actionable messages per error category. Surfaced by the agent
# loop so a failed turn explains *why* it failed and how to fix it, instead of
# a generic "I've reached my processing limit." fallback.
_FRIENDLY_MESSAGES: Dict[ErrorCategory, str] = {
    ErrorCategory.AUTH_ERROR: (
        "LLM authentication failed — the API key is missing or invalid. "
        "Set a valid LLM API key in the profile secret vault or use "
        "LEAPFLOW_LLM_API_KEY as a process override, then retry. "
        "Also verify llm.base_url / llm.model in profile config."
    ),
    ErrorCategory.AUTH_PERMANENT: (
        "LLM authentication failed — the API key is missing or invalid. "
        "Set a valid LLM API key in the profile secret vault or use "
        "LEAPFLOW_LLM_API_KEY as a process override, then retry. "
        "Also verify llm.base_url / llm.model in profile config."
    ),
    ErrorCategory.BILLING: (
        "LLM request rejected due to billing/quota limits \u2014 "
        "check your provider account balance or quota."
    ),
    ErrorCategory.MODEL_NOT_FOUND: (
        "The configured LLM model was not found \u2014 check LEAPFLOW_LLM_MODEL."
    ),
    ErrorCategory.RATE_LIMITED: (
        "LLM rate limit reached \u2014 please wait a moment and try again."
    ),
    ErrorCategory.OVERLOADED: (
        "The LLM provider is overloaded \u2014 please retry shortly."
    ),
    ErrorCategory.CONTEXT_OVERFLOW: (
        "The conversation exceeded the model's context window \u2014 "
        "start a new session or shorten the input."
    ),
    ErrorCategory.PAYLOAD_TOO_LARGE: (
        "The request payload was too large for the provider."
    ),
    ErrorCategory.CONTENT_BLOCKED: (
        "The request was blocked by the provider's content policy."
    ),
    ErrorCategory.SSL_ERROR: (
        "TLS/SSL error connecting to the LLM provider \u2014 "
        "check your network, proxy, or certificates."
    ),
}

_AUTH_KEYWORDS = ("api_key", "api key", "unauthorized", "forbidden", "401", "403")
_BILLING_KEYWORDS = ("insufficient_quota", "billing", "payment", "quota exceeded", "402")
_RATE_LIMIT_KEYWORDS = ("rate", "429", "too many", "throttl")
_OVERLOAD_KEYWORDS = ("overloaded", "503", "capacity", "server busy")
_CONTEXT_KEYWORDS = ("context", "token", "length", "maximum context", "max_tokens")
_CONTENT_POLICY_KEYWORDS = ("content_policy", "safety", "content filter", "moderation")


class ErrorClassifier:
    """Classifies errors into categories and provides recovery strategies.

    Classification pipeline (priority order):
    1. HTTP status code + message refinement
    2. Known keyword patterns
    3. SSL/transport errors
    4. Fallback to PERMANENT
    """

    def __init__(
        self, recovery_map: Optional[Dict[ErrorCategory, RecoveryStrategy]] = None
    ):
        self._map = recovery_map or RECOVERY_MAP

    def classify(self, exc: Exception) -> ErrorCategory:
        """Classify an LLM/network exception into a recovery category."""
        msg = str(exc).lower()

        status = self._extract_status_code(exc)
        if status is not None:
            category = self._classify_by_status(status, msg)
            if category is not None:
                return category

        return self._classify_by_message(msg)

    def classify_detailed(self, exc: Exception) -> ClassifiedError:
        """Classify with full context for advanced recovery logic."""
        status = self._extract_status_code(exc)
        category = self.classify(exc)
        recovery = self.get_recovery(category)

        return ClassifiedError(
            category=category,
            status_code=status,
            message=str(exc)[:500],
            retryable=recovery.retry,
            should_compress=recovery.compress,
            should_fallback=recovery.should_fallback,
            should_rotate_credential=recovery.should_rotate_credential,
        )

    def classify_tool_error(self, observation: Dict[str, Any]) -> ErrorCategory:
        """Classify a tool execution error from observation dict."""
        if observation.get("ok", True):
            return ErrorCategory.TRANSIENT
        error = str(observation.get("error", "")).lower()
        if "permission" in error or "access denied" in error:
            return ErrorCategory.TOOL_FAILURE
        if "timeout" in error or "timed out" in error:
            return ErrorCategory.TRANSIENT
        if "not found" in error:
            return ErrorCategory.TOOL_FAILURE
        if "rate" in error or "throttl" in error:
            return ErrorCategory.RATE_LIMITED
        return ErrorCategory.TOOL_FAILURE

    def get_recovery(self, category: ErrorCategory) -> RecoveryStrategy:
        return self._map.get(category, RecoveryStrategy())

    @staticmethod
    def friendly_message(category: ErrorCategory, detail: str = "") -> str:
        """Return a clear, actionable user-facing message for an error category.

        Lets the agent loop surface *why* a turn failed (with remediation
        guidance) instead of a generic fallback message.
        """
        base = _FRIENDLY_MESSAGES.get(category)
        if base:
            return base
        detail = (detail or "").strip()
        if detail:
            return f"LLM request failed ({category.value}): {detail[:200]}"
        return f"LLM request failed ({category.value})."

    @staticmethod
    def _extract_status_code(exc: Exception) -> Optional[int]:
        """Extract HTTP status code from common exception types."""
        for attr in ("status_code", "status", "code", "http_status"):
            val = getattr(exc, attr, None)
            if isinstance(val, int) and 100 <= val <= 599:
                return val
        msg = str(exc)
        for code in (400, 401, 402, 403, 404, 413, 422, 429, 500, 502, 503, 504):
            if str(code) in msg:
                return code
        return None

    @staticmethod
    def _classify_by_status(status: int, msg: str) -> Optional[ErrorCategory]:
        """Disambiguate errors by HTTP status + message content."""
        if status == 401:
            return ErrorCategory.AUTH_PERMANENT
        if status == 403:
            if any(kw in msg for kw in ("billing", "quota", "payment")):
                return ErrorCategory.BILLING
            return ErrorCategory.AUTH_PERMANENT
        if status == 402:
            if any(kw in msg for kw in ("try again", "resets at", "temporary")):
                return ErrorCategory.RATE_LIMITED
            return ErrorCategory.BILLING
        if status == 404:
            if "model" in msg:
                return ErrorCategory.MODEL_NOT_FOUND
            return ErrorCategory.PERMANENT
        if status == 413:
            return ErrorCategory.PAYLOAD_TOO_LARGE
        if status == 422:
            if any(kw in msg for kw in _CONTEXT_KEYWORDS):
                return ErrorCategory.CONTEXT_OVERFLOW
            return ErrorCategory.FORMAT_ERROR
        if status == 429:
            if "overloaded" in msg or "capacity" in msg:
                return ErrorCategory.OVERLOADED
            return ErrorCategory.RATE_LIMITED
        if status in (500, 502):
            if any(kw in msg for kw in _CONTEXT_KEYWORDS):
                return ErrorCategory.CONTEXT_OVERFLOW
            return ErrorCategory.TRANSIENT
        if status == 503:
            return ErrorCategory.OVERLOADED
        if status == 504:
            return ErrorCategory.TRANSIENT
        if 400 <= status < 500:
            if any(kw in msg for kw in ("content_policy", "safety", "blocked")):
                return ErrorCategory.CONTENT_BLOCKED
            if "image" in msg and ("large" in msg or "size" in msg):
                return ErrorCategory.IMAGE_TOO_LARGE
            return ErrorCategory.FORMAT_ERROR
        if status >= 500:
            return ErrorCategory.TRANSIENT
        return None

    @staticmethod
    def _classify_by_message(msg: str) -> ErrorCategory:
        """Classify by keyword patterns in error message."""
        if "ssl" in msg or "certificate" in msg:
            if "verify" in msg or "expired" in msg:
                return ErrorCategory.SSL_ERROR
            return ErrorCategory.TRANSIENT

        if "timeout" in msg or "timed out" in msg or "connection" in msg:
            return ErrorCategory.TRANSIENT

        if any(kw in msg for kw in _RATE_LIMIT_KEYWORDS):
            return ErrorCategory.RATE_LIMITED

        if any(kw in msg for kw in _OVERLOAD_KEYWORDS):
            return ErrorCategory.OVERLOADED

        if any(kw in msg for kw in _CONTEXT_KEYWORDS):
            return ErrorCategory.CONTEXT_OVERFLOW

        if any(kw in msg for kw in _BILLING_KEYWORDS):
            return ErrorCategory.BILLING

        if any(kw in msg for kw in _AUTH_KEYWORDS):
            return ErrorCategory.AUTH_ERROR

        if any(kw in msg for kw in _CONTENT_POLICY_KEYWORDS):
            return ErrorCategory.CONTENT_BLOCKED

        if "format" in msg or "json" in msg or "parse" in msg:
            return ErrorCategory.FORMAT_ERROR

        if "model" in msg and ("not found" in msg or "does not exist" in msg):
            return ErrorCategory.MODEL_NOT_FOUND

        return ErrorCategory.PERMANENT


def jittered_backoff(attempt: int, *, base: float = 1.0, cap: float = 60.0) -> float:
    """Decorrelated jitter backoff: random(0, min(cap, base * 2^attempt))."""
    delay = min(cap, base * (2**attempt))
    return random.uniform(0, delay)
