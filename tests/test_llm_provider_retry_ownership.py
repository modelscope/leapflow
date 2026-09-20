# Copyright (c) Alibaba, Inc. and its affiliates.
"""The provider owns retry; the SDK must not add a second, hidden one.

``OpenAIChat`` implements a retry policy with backoff, and the recovery layer's
budgets assume ``timeout_s`` bounds *one* attempt. The OpenAI SDK retries twice by
default, so leaving that default in place multiplies the two policies: effective
attempts become ``max_retries * 3`` and a hard timeout surfaces after three times
the configured budget. Measured on a live endpoint before the fix: a 45s timeout
raised ``APITimeoutError`` after 137s (~3 x 45s), which is exactly the failure a
turn-level deadline cannot absorb.

These tests pin the single-owner contract without any network access.
"""

from __future__ import annotations

from leapflow.llm.openai_provider import OpenAIChat


def _provider(**kwargs) -> OpenAIChat:
    return OpenAIChat(
        api_key="sk-test-not-a-real-key",
        base_url="https://example.invalid/v1",
        model="test-model",
        **kwargs,
    )


def test_sdk_clients_do_not_retry_on_their_own():
    """Both SDK clients must be constructed with retries disabled."""
    provider = _provider()
    # ``max_retries`` is public on the SDK client and is what multiplies attempts.
    assert provider._async.max_retries == 0
    assert provider._sync.max_retries == 0


def test_provider_keeps_its_own_retry_policy():
    """Disabling SDK retry must not disable ours -- the policy just has one owner."""
    provider = _provider(max_retries=4)
    assert provider._max_retries == 4
    # Floor of one attempt: a zero would mean "never call the model".
    assert _provider(max_retries=0)._max_retries == 1


def test_configured_timeout_bounds_one_attempt():
    """The read timeout the caller asked for is the one the client carries."""
    provider = _provider(timeout_s=45.0)
    assert provider._async.timeout.read == 45.0
    # Connect/write/pool stay bounded independently so a stalled handshake cannot
    # consume the whole read budget.
    assert provider._async.timeout.connect == 30.0
