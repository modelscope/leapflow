"""Error classification and recovery strategy for agent loops."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    TRANSIENT = "transient"
    RATE_LIMITED = "rate_limited"
    CONTEXT_OVERFLOW = "context_overflow"
    FORMAT_ERROR = "format_error"
    TOOL_FAILURE = "tool_failure"
    AUTH_ERROR = "auth_error"
    CONTENT_BLOCKED = "content_blocked"
    PERMANENT = "permanent"


@dataclass(frozen=True)
class RecoveryStrategy:
    retry: bool = False
    backoff: bool = False
    compress: bool = False
    inform_llm: bool = False
    max_retries: int = 0
    base_delay: float = 1.0


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
        ErrorCategory.CONTEXT_OVERFLOW: RecoveryStrategy(
            retry=True, compress=True, max_retries=1,
        ),
        ErrorCategory.FORMAT_ERROR: RecoveryStrategy(retry=True, max_retries=2),
        ErrorCategory.TOOL_FAILURE: RecoveryStrategy(inform_llm=True),
        ErrorCategory.AUTH_ERROR: RecoveryStrategy(),
        ErrorCategory.CONTENT_BLOCKED: RecoveryStrategy(),
        ErrorCategory.PERMANENT: RecoveryStrategy(),
    }


# Default recovery map (used when no config overrides are provided)
RECOVERY_MAP: Dict[ErrorCategory, RecoveryStrategy] = build_recovery_map()


# Auth-related keywords (specific enough to avoid false positives like "KeyError")
_AUTH_KEYWORDS = ("api_key", "api key", "auth", "unauthorized", "forbidden", "401", "403")


class ErrorClassifier:
    """Classifies errors into categories and provides recovery strategies."""

    def __init__(
        self, recovery_map: Optional[Dict[ErrorCategory, RecoveryStrategy]] = None
    ):
        self._map = recovery_map or RECOVERY_MAP

    def classify(self, exc: Exception) -> ErrorCategory:
        """Classify an LLM/network exception."""
        msg = str(exc).lower()
        if "timeout" in msg or "connection" in msg:
            return ErrorCategory.TRANSIENT
        if "rate" in msg or "429" in msg or "too many" in msg:
            return ErrorCategory.RATE_LIMITED
        if "context" in msg or "token" in msg or "length" in msg:
            return ErrorCategory.CONTEXT_OVERFLOW
        if any(kw in msg for kw in _AUTH_KEYWORDS):
            return ErrorCategory.AUTH_ERROR
        if "content" in msg or "policy" in msg or "blocked" in msg:
            return ErrorCategory.CONTENT_BLOCKED
        if "format" in msg or "json" in msg or "parse" in msg:
            return ErrorCategory.FORMAT_ERROR
        return ErrorCategory.PERMANENT

    def classify_tool_error(self, observation: Dict[str, Any]) -> ErrorCategory:
        """Classify a tool execution error from observation dict."""
        if observation.get("ok", True):
            return ErrorCategory.TRANSIENT  # not really an error
        error = str(observation.get("error", ""))
        if "permission" in error.lower() or "not found" in error.lower():
            return ErrorCategory.TOOL_FAILURE
        if "timeout" in error.lower():
            return ErrorCategory.TRANSIENT
        return ErrorCategory.TOOL_FAILURE

    def get_recovery(self, category: ErrorCategory) -> RecoveryStrategy:
        return self._map.get(category, RecoveryStrategy())


def jittered_backoff(attempt: int, *, base: float = 1.0, cap: float = 60.0) -> float:
    """Decorrelated jitter backoff: random(0, min(cap, base * 2^attempt))."""
    delay = min(cap, base * (2**attempt))
    return random.uniform(0, delay)
