"""Tests for the UnifiedErrorClassifier.

Covers:
- classify_llm_error produces correct FailureEnvelope for each ErrorCategory
- classify_tool_result returns None for non-failures
- classify_tool_result classifies permission failures, timeouts, unknown tools
- classify_tool_result maps execution_policy to SideEffectState
- classify_system_error handles common exceptions
"""
from __future__ import annotations

import pytest

from leapflow.engine.failure_envelope import (
    FailureEnvelope,
    FailureSource,
    Recoverability,
    SideEffectState,
)
from leapflow.engine.unified_classifier import UnifiedErrorClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classifier() -> UnifiedErrorClassifier:
    return UnifiedErrorClassifier()


# ===========================================================================
# classify_llm_error Tests
# ===========================================================================


class TestClassifyLlmError:
    def test_rate_limited(self, classifier: UnifiedErrorClassifier) -> None:
        exc = Exception("429 Too Many Requests: rate limit exceeded")
        env = classifier.classify_llm_error(exc, provider="openai", model="gpt-4")
        assert env.source == FailureSource.LLM
        assert env.category == "rate_limited"
        assert env.recoverability == Recoverability.AUTO_RETRY

    def test_transient(self, classifier: UnifiedErrorClassifier) -> None:
        exc = Exception("502 Bad Gateway: temporary server error")
        env = classifier.classify_llm_error(exc)
        assert env.source == FailureSource.LLM
        assert env.category == "transient"
        assert env.recoverability == Recoverability.AUTO_RETRY

    def test_context_overflow(self, classifier: UnifiedErrorClassifier) -> None:
        exc = Exception("422 maximum context length exceeded")
        env = classifier.classify_llm_error(exc)
        assert env.source == FailureSource.LLM
        assert env.category == "context_overflow"
        assert env.recoverability == Recoverability.AUTO_RECOVER

    def test_auth_error(self, classifier: UnifiedErrorClassifier) -> None:
        exc = Exception("unauthorized: invalid api_key")
        env = classifier.classify_llm_error(exc)
        assert env.source == FailureSource.LLM
        assert env.category == "auth_error"
        assert env.recoverability == Recoverability.AUTO_RETRY

    def test_auth_permanent(self, classifier: UnifiedErrorClassifier) -> None:
        exc = Exception("401 Unauthorized: invalid credentials")
        env = classifier.classify_llm_error(exc)
        assert env.source == FailureSource.LLM
        assert env.category == "auth_permanent"
        assert env.recoverability == Recoverability.NON_RECOVERABLE

    def test_billing(self, classifier: UnifiedErrorClassifier) -> None:
        exc = Exception("insufficient_quota: payment required")
        env = classifier.classify_llm_error(exc)
        assert env.source == FailureSource.LLM
        assert env.category == "billing"
        assert env.recoverability == Recoverability.USER_FIXABLE

    def test_content_blocked(self, classifier: UnifiedErrorClassifier) -> None:
        exc = Exception("400 content_policy violation: blocked")
        env = classifier.classify_llm_error(exc)
        assert env.source == FailureSource.LLM
        assert env.category == "content_blocked"
        assert env.recoverability == Recoverability.NON_RECOVERABLE

    def test_model_not_found(self, classifier: UnifiedErrorClassifier) -> None:
        exc = Exception("404 model 'gpt-5' does not exist")
        env = classifier.classify_llm_error(exc)
        assert env.source == FailureSource.LLM
        assert env.category == "model_not_found"
        assert env.recoverability == Recoverability.USER_FIXABLE

    def test_overloaded(self, classifier: UnifiedErrorClassifier) -> None:
        exc = Exception("503 server overloaded")
        env = classifier.classify_llm_error(exc)
        assert env.source == FailureSource.LLM
        assert env.category == "overloaded"
        assert env.recoverability == Recoverability.AUTO_RETRY

    def test_payload_too_large(self, classifier: UnifiedErrorClassifier) -> None:
        exc = Exception("413 Request Entity Too Large")
        env = classifier.classify_llm_error(exc)
        assert env.source == FailureSource.LLM
        assert env.category == "payload_too_large"
        assert env.recoverability == Recoverability.AUTO_RECOVER

    def test_format_error(self, classifier: UnifiedErrorClassifier) -> None:
        exc = Exception("422 invalid format json parse error")
        env = classifier.classify_llm_error(exc)
        assert env.source == FailureSource.LLM
        assert env.category in ("format_error", "context_overflow")
        # Both are AUTO_RECOVER

    def test_image_too_large(self, classifier: UnifiedErrorClassifier) -> None:
        exc = Exception("400 image size too large")
        env = classifier.classify_llm_error(exc)
        assert env.source == FailureSource.LLM
        assert env.category == "image_too_large"
        assert env.recoverability == Recoverability.AUTO_RECOVER

    def test_envelope_has_provider_hint(self, classifier: UnifiedErrorClassifier) -> None:
        exc = Exception("429 rate limit")
        env = classifier.classify_llm_error(exc)
        assert env.provider_hint is not None
        assert env.provider_hint.hint_text != ""

    def test_envelope_has_context_with_provider(self, classifier: UnifiedErrorClassifier) -> None:
        exc = Exception("timeout")
        env = classifier.classify_llm_error(exc, provider="dashscope", model="qwen-max")
        assert env.context.arguments_dict.get("provider") == "dashscope"
        assert env.context.arguments_dict.get("model") == "qwen-max"


# ===========================================================================
# classify_tool_result Tests — Non-failures
# ===========================================================================


class TestClassifyToolResultNonFailures:
    def test_ok_true_returns_none(self, classifier: UnifiedErrorClassifier) -> None:
        result = {"ok": True, "output": "success"}
        assert classifier.classify_tool_result(result) is None

    def test_ok_default_returns_none(self, classifier: UnifiedErrorClassifier) -> None:
        result = {"output": "success"}
        assert classifier.classify_tool_result(result) is None

    def test_duplicate_suppressed_returns_none(self, classifier: UnifiedErrorClassifier) -> None:
        result = {
            "ok": False,
            "duplicate_suppressed": True,
            "error": "duplicate execution",
        }
        assert classifier.classify_tool_result(result) is None

    def test_counts_as_failure_false_returns_none(self, classifier: UnifiedErrorClassifier) -> None:
        result = {
            "ok": False,
            "counts_as_failure": False,
            "error": "non-failure signal",
        }
        assert classifier.classify_tool_result(result) is None


# ===========================================================================
# classify_tool_result Tests — Failures
# ===========================================================================


class TestClassifyToolResultFailures:
    def test_permission_failure_by_class(self, classifier: UnifiedErrorClassifier) -> None:
        result = {
            "ok": False,
            "failure_class": "authorization",
            "failure_code": "missing_scope",
            "error": "Missing read scope for resource",
        }
        env = classifier.classify_tool_result(result, tool_name="feishu_send")
        assert env is not None
        assert env.source == FailureSource.TOOL
        assert env.category == "tool_permission"
        assert env.recoverability == Recoverability.NON_RECOVERABLE

    def test_permission_failure_by_code(self, classifier: UnifiedErrorClassifier) -> None:
        result = {
            "ok": False,
            "failure_class": "other",
            "failure_code": "access_denied",
            "error": "Access denied to endpoint",
        }
        env = classifier.classify_tool_result(result, tool_name="platform_action")
        assert env is not None
        assert env.category == "tool_permission"
        assert env.recoverability == Recoverability.NON_RECOVERABLE

    def test_scope_denied_failure(self, classifier: UnifiedErrorClassifier) -> None:
        result = {
            "ok": False,
            "failure_class": "scope_denied",
            "error": "Scope not granted",
        }
        env = classifier.classify_tool_result(result)
        assert env is not None
        assert env.category == "tool_permission"

    def test_unknown_tool_retryable(self, classifier: UnifiedErrorClassifier) -> None:
        result = {
            "ok": False,
            "error_type": "unknown_tool",
            "error": "Tool 'web_search' not found",
            "retryable": True,
        }
        env = classifier.classify_tool_result(result, tool_name="web_search")
        assert env is not None
        assert env.source == FailureSource.TOOL
        assert env.category == "tool_unknown"
        assert env.recoverability == Recoverability.AUTO_RECOVER

    def test_timeout_read_only(self, classifier: UnifiedErrorClassifier) -> None:
        result = {
            "ok": False,
            "error": "Execution timed out after 30s",
        }
        env = classifier.classify_tool_result(
            result, tool_name="web_search", execution_policy="read_only"
        )
        assert env is not None
        assert env.category == "tool_timeout"
        assert env.recoverability == Recoverability.AUTO_RETRY
        assert env.side_effect_state == SideEffectState.NONE

    def test_timeout_external_side_effect(self, classifier: UnifiedErrorClassifier) -> None:
        result = {
            "ok": False,
            "error": "Request timeout waiting for response",
        }
        env = classifier.classify_tool_result(
            result, tool_name="gateway_send", execution_policy="external_side_effect"
        )
        assert env is not None
        assert env.category == "tool_timeout"
        assert env.recoverability == Recoverability.USER_FIXABLE
        assert env.side_effect_state == SideEffectState.UNKNOWN

    def test_generic_failure_retryable(self, classifier: UnifiedErrorClassifier) -> None:
        result = {
            "ok": False,
            "error": "Connection refused to service",
            "retryable": True,
        }
        env = classifier.classify_tool_result(result, tool_name="shell_run")
        assert env is not None
        assert env.category == "tool_failure"
        assert env.recoverability == Recoverability.AUTO_RETRY

    def test_generic_failure_not_retryable(self, classifier: UnifiedErrorClassifier) -> None:
        result = {
            "ok": False,
            "error": "Invalid arguments",
            "retryable": False,
        }
        env = classifier.classify_tool_result(result, tool_name="file_write")
        assert env is not None
        assert env.category == "tool_failure"
        assert env.recoverability == Recoverability.USER_FIXABLE

    def test_execution_policy_read_only_side_effect(self, classifier: UnifiedErrorClassifier) -> None:
        result = {"ok": False, "error": "failed"}
        env = classifier.classify_tool_result(result, execution_policy="read_only")
        assert env is not None
        assert env.side_effect_state == SideEffectState.NONE

    def test_execution_policy_mutating_idempotent(self, classifier: UnifiedErrorClassifier) -> None:
        result = {"ok": False, "error": "failed"}
        env = classifier.classify_tool_result(result, execution_policy="mutating_idempotent")
        assert env is not None
        assert env.side_effect_state == SideEffectState.PARTIAL

    def test_execution_policy_external(self, classifier: UnifiedErrorClassifier) -> None:
        result = {"ok": False, "error": "failed"}
        env = classifier.classify_tool_result(result, execution_policy="external_side_effect")
        assert env is not None
        assert env.side_effect_state == SideEffectState.UNKNOWN


# ===========================================================================
# classify_system_error Tests
# ===========================================================================


class TestClassifySystemError:
    def test_timeout_error(self, classifier: UnifiedErrorClassifier) -> None:
        exc = TimeoutError("Connection timed out")
        env = classifier.classify_system_error(exc)
        assert env.source == FailureSource.SYSTEM
        assert env.category == "system_timeout"
        assert env.recoverability == Recoverability.AUTO_RETRY

    def test_memory_error(self, classifier: UnifiedErrorClassifier) -> None:
        exc = MemoryError("Cannot allocate memory")
        env = classifier.classify_system_error(exc)
        assert env.source == FailureSource.SYSTEM
        assert env.category == "system_resource"
        assert env.recoverability == Recoverability.NON_RECOVERABLE

    def test_os_error(self, classifier: UnifiedErrorClassifier) -> None:
        exc = OSError(28, "No space left on device")
        env = classifier.classify_system_error(exc)
        assert env.source == FailureSource.SYSTEM
        assert env.category == "system_io"
        assert env.recoverability == Recoverability.AUTO_RETRY

    def test_connection_error_from_message(self, classifier: UnifiedErrorClassifier) -> None:
        exc = Exception("Connection refused to localhost:8080")
        env = classifier.classify_system_error(exc)
        assert env.source == FailureSource.SYSTEM
        assert env.category == "system_network"
        assert env.recoverability == Recoverability.AUTO_RETRY

    def test_generic_exception(self, classifier: UnifiedErrorClassifier) -> None:
        exc = RuntimeError("Something unexpected happened")
        env = classifier.classify_system_error(exc)
        assert env.source == FailureSource.SYSTEM
        assert env.category == "system_unknown"
        assert env.recoverability == Recoverability.USER_FIXABLE

    def test_envelope_structure(self, classifier: UnifiedErrorClassifier) -> None:
        exc = TimeoutError("test timeout")
        env = classifier.classify_system_error(exc)
        assert isinstance(env, FailureEnvelope)
        assert len(env.envelope_id) == 32
        assert env.timestamp > 0
        assert env.message == "test timeout"
