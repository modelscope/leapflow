# Copyright (c) Alibaba, Inc. and its affiliates.
"""Error classification and recovery strategy for agent loops.

Enhanced taxonomy inspired by hermes-agent/error_classifier.py:
- Fine-grained HTTP status disambiguation (402/429/400/5xx)
- Structured ClassifiedError with recovery hints
- Provider-agnostic pattern matching
- Config-driven recovery strategies (OCP)
- Data-driven classification via registry tables (no if-elif chains)
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

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
        "Set a valid LLM API key with `leap config llm key`, then retry. "
        "Also verify llm.base_url / llm.model in profile config."
    ),
    ErrorCategory.AUTH_PERMANENT: (
        "LLM authentication failed — the API key is missing or invalid. "
        "Set a valid LLM API key with `leap config llm key`, then retry. "
        "Also verify llm.base_url / llm.model in profile config."
    ),
    ErrorCategory.BILLING: (
        "LLM request rejected due to billing/quota limits \u2014 "
        "check your provider account balance or quota."
    ),
    ErrorCategory.MODEL_NOT_FOUND: (
        "The configured LLM model was not found — check llm.model in profile config."
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


# ---------------------------------------------------------------------------
# Data-Driven HTTP Status Code Classification
# ---------------------------------------------------------------------------

# Exact status code -> category (no message refinement needed)
_STATUS_CODE_CLASSIFICATION: Dict[int, ErrorCategory] = {
    401: ErrorCategory.AUTH_PERMANENT,
    413: ErrorCategory.PAYLOAD_TOO_LARGE,
    503: ErrorCategory.OVERLOADED,
    504: ErrorCategory.TRANSIENT,
}

# Status codes that require message-based refinement to determine category.
# Each entry maps: status -> list of (keywords_to_check, category_if_matched)
# with a final fallback category.
_StatusRefinement = Tuple[List[Tuple[Tuple[str, ...], ErrorCategory]], ErrorCategory]

_STATUS_REFINEMENT: Dict[int, _StatusRefinement] = {
    403: (
        [
            (("billing", "quota", "payment"), ErrorCategory.BILLING),
        ],
        ErrorCategory.AUTH_PERMANENT,
    ),
    402: (
        [
            (("try again", "resets at", "temporary"), ErrorCategory.RATE_LIMITED),
        ],
        ErrorCategory.BILLING,
    ),
    404: (
        [
            (("model",), ErrorCategory.MODEL_NOT_FOUND),
        ],
        ErrorCategory.PERMANENT,
    ),
    422: (
        [
            (("context", "token", "length", "maximum context", "max_tokens"), ErrorCategory.CONTEXT_OVERFLOW),
        ],
        ErrorCategory.FORMAT_ERROR,
    ),
    429: (
        [
            (("overloaded", "capacity"), ErrorCategory.OVERLOADED),
        ],
        ErrorCategory.RATE_LIMITED,
    ),
    500: (
        [
            (("context", "token", "length", "maximum context", "max_tokens"), ErrorCategory.CONTEXT_OVERFLOW),
        ],
        ErrorCategory.TRANSIENT,
    ),
    502: (
        [
            (("context", "token", "length", "maximum context", "max_tokens"), ErrorCategory.CONTEXT_OVERFLOW),
        ],
        ErrorCategory.TRANSIENT,
    ),
}

# Fallback ranges for codes not in the exact or refinement maps.
_STATUS_RANGE_CLASSIFICATION: List[Tuple[range, _StatusRefinement]] = [
    # 4xx fallback with message-based refinement
    (range(400, 500), (
        [
            (("content_policy", "safety", "blocked"), ErrorCategory.CONTENT_BLOCKED),
            (("image",), ErrorCategory.IMAGE_TOO_LARGE),  # refined further below
        ],
        ErrorCategory.FORMAT_ERROR,
    )),
    # 5xx fallback
    (range(500, 600), (
        [],
        ErrorCategory.TRANSIENT,
    )),
]


# ---------------------------------------------------------------------------
# Data-Driven Message Classification Rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MessageClassificationRule:
    """A pattern-based rule for classifying error messages.

    Rules are evaluated in registration order; the first match wins.
    """
    category: ErrorCategory
    keywords: FrozenSet[str]
    description: str = ""
    # Optional predicate for compound conditions that cannot be expressed
    # as simple keyword presence (e.g. require two keywords simultaneously).
    predicate: Optional[Callable[[str], bool]] = field(default=None, compare=False)


# Ordered rule table — first match wins.
_MESSAGE_RULES: List[MessageClassificationRule] = [
    # SSL/TLS (must precede transient to avoid "connection" match)
    MessageClassificationRule(
        category=ErrorCategory.SSL_ERROR,
        keywords=frozenset({"ssl", "certificate"}),
        description="SSL/TLS certificate or handshake failures",
        predicate=lambda msg: ("ssl" in msg or "certificate" in msg) and ("verify" in msg or "expired" in msg),
    ),
    # Transient: timeout/connection (check early to catch network issues)
    MessageClassificationRule(
        category=ErrorCategory.TRANSIENT,
        keywords=frozenset({"timeout", "timed out", "connection"}),
        description="Transient network/timeout errors",
    ),
    # Rate limiting
    MessageClassificationRule(
        category=ErrorCategory.RATE_LIMITED,
        keywords=frozenset({"rate", "429", "too many", "throttl"}),
        description="Rate limiting / throttling",
    ),
    # Overloaded
    MessageClassificationRule(
        category=ErrorCategory.OVERLOADED,
        keywords=frozenset({"overloaded", "503", "capacity", "server busy"}),
        description="Server overload / capacity",
    ),
    # Context overflow
    MessageClassificationRule(
        category=ErrorCategory.CONTEXT_OVERFLOW,
        keywords=frozenset({"context", "token", "length", "maximum context", "max_tokens"}),
        description="Context window overflow",
    ),
    # Billing
    MessageClassificationRule(
        category=ErrorCategory.BILLING,
        keywords=frozenset({"insufficient_quota", "billing", "payment", "quota exceeded", "402"}),
        description="Billing / quota failures",
    ),
    # Auth (recoverable — credential rotation may help)
    MessageClassificationRule(
        category=ErrorCategory.AUTH_ERROR,
        keywords=frozenset({"api_key", "api key", "unauthorized", "forbidden", "401", "403"}),
        description="Authentication / authorization errors",
    ),
    # Content policy
    MessageClassificationRule(
        category=ErrorCategory.CONTENT_BLOCKED,
        keywords=frozenset({"content_policy", "safety", "content filter", "moderation"}),
        description="Content policy violations",
    ),
    # Format / parse (explicit keywords only)
    MessageClassificationRule(
        category=ErrorCategory.FORMAT_ERROR,
        keywords=frozenset({"format", "json", "parse"}),
        description="Format / JSON parse errors",
    ),
    # Model not found (compound predicate)
    MessageClassificationRule(
        category=ErrorCategory.MODEL_NOT_FOUND,
        keywords=frozenset({"model"}),
        description="Model not found",
        predicate=lambda msg: "model" in msg and ("not found" in msg or "does not exist" in msg),
    ),
]

# Tool error classification rules (used by classify_tool_error)
_TOOL_ERROR_RULES: List[MessageClassificationRule] = [
    MessageClassificationRule(
        category=ErrorCategory.TOOL_FAILURE,
        keywords=frozenset({"permission", "access denied"}),
        description="Tool permission failures",
    ),
    MessageClassificationRule(
        category=ErrorCategory.TRANSIENT,
        keywords=frozenset({"timeout", "timed out"}),
        description="Tool timeout / transient failures",
    ),
    MessageClassificationRule(
        category=ErrorCategory.TOOL_FAILURE,
        keywords=frozenset({"not found"}),
        description="Tool resource not found",
    ),
    MessageClassificationRule(
        category=ErrorCategory.RATE_LIMITED,
        keywords=frozenset({"rate", "throttl"}),
        description="Tool rate limiting",
    ),
]


def register_message_rule(rule: MessageClassificationRule, *, priority: int = -1) -> None:
    """Register a custom classification rule.

    Args:
        rule: The classification rule to register.
        priority: Index at which to insert. -1 appends before the final fallback.
    """
    if priority < 0 or priority >= len(_MESSAGE_RULES):
        _MESSAGE_RULES.append(rule)
    else:
        _MESSAGE_RULES.insert(priority, rule)


def register_status_code(status_code: int, category: ErrorCategory) -> None:
    """Register or override a status code -> category mapping."""
    _STATUS_CODE_CLASSIFICATION[status_code] = category


# ---------------------------------------------------------------------------
# Classification Functions
# ---------------------------------------------------------------------------

def _classify_by_status(status: int, msg: str) -> Optional[ErrorCategory]:
    """Disambiguate errors by HTTP status + message content.

    Uses data-driven tables for lookup: exact match -> refinement rules -> range fallback.
    """
    # 1. Exact match (no refinement needed)
    if status in _STATUS_CODE_CLASSIFICATION:
        return _STATUS_CODE_CLASSIFICATION[status]

    # 2. Status codes requiring message refinement
    if status in _STATUS_REFINEMENT:
        refinements, fallback = _STATUS_REFINEMENT[status]
        for keywords, category in refinements:
            if any(kw in msg for kw in keywords):
                return category
        return fallback

    # 3. Range-based fallback with optional refinement
    for code_range, (refinements, fallback) in _STATUS_RANGE_CLASSIFICATION:
        if status in code_range:
            for keywords, category in refinements:
                if category == ErrorCategory.IMAGE_TOO_LARGE:
                    # Compound check: "image" AND ("large" or "size")
                    if "image" in msg and ("large" in msg or "size" in msg):
                        return category
                elif any(kw in msg for kw in keywords):
                    return category
            return fallback

    return None


def _classify_by_message(msg: str) -> ErrorCategory:
    """Classify by pattern rules in error message. First matching rule wins."""
    for rule in _MESSAGE_RULES:
        if rule.predicate is not None:
            if rule.predicate(msg):
                return rule.category
        elif any(kw in msg for kw in rule.keywords):
            return rule.category
    return ErrorCategory.PERMANENT


def _classify_tool_error_by_message(error: str) -> ErrorCategory:
    """Classify a tool error message using the tool error rule table."""
    lower = error.lower()
    for rule in _TOOL_ERROR_RULES:
        if any(kw in lower for kw in rule.keywords):
            return rule.category
    return ErrorCategory.TOOL_FAILURE


class ErrorClassifier:
    """Classifies errors into categories and provides recovery strategies.

    Classification pipeline (priority order):
    1. HTTP status code + message refinement
    2. Known keyword patterns (data-driven rule table)
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
            category = _classify_by_status(status, msg)
            if category is not None:
                return category

        return _classify_by_message(msg)

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
        error = str(observation.get("error", ""))
        return _classify_tool_error_by_message(error)

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


def jittered_backoff(attempt: int, *, base: float = 1.0, cap: float = 60.0) -> float:
    """Decorrelated jitter backoff: random(0, min(cap, base * 2^attempt))."""
    delay = min(cap, base * (2**attempt))
    return random.uniform(0, delay)
