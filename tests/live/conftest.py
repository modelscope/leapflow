# Copyright (c) Alibaba, Inc. and its affiliates.
"""Fixtures for the live lane: credential gating, budget enforcement, CI summary.

The live lane speaks to a real provider, so two invariants dominate this module:

1. **Absence is a skip, never a failure.** Locally the credential env vars are
   unset; every live test must skip cleanly so ``pytest tests/live`` is a no-op
   for a developer without keys. The ``live_provider`` fixture is the single
   gate — a test that needs a provider requests it and inherits the skip.

2. **Cost is bounded per test and per suite.** Each test declares a call, token,
   and deadline budget through ``live_budget``. Usage is read from the provider's
   own ``usage`` dict (``total_tokens``), accumulated into a session-wide total,
   and asserted against ``LEAPFLOW_LIVE_TOKEN_BUDGET`` (default 75_000). The
   terminal summary prints the realised calls/tokens so a CI run always ends with
   its true cost on the record.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import pytest

from leapflow.llm.base import LLMProvider

# ── Re-export budget primitives from the shared harness module ───────────────
from tests._harness.live_budget import (
    LiveBudget,
    SuiteAccumulator,
    apply_terminal_summary,
    suite_token_budget,
)

# ── Credential environment ──────────────────────────────────────────────────
# The same trio production reads (leapflow.config._build_settings_from_env), so
# one set of CI secrets drives both the product and this lane.
_BASE_URL_ENV = "LEAPFLOW_LLM_BASE_URL"
_API_KEY_ENV = "LEAPFLOW_LLM_API_KEY"
_MODEL_ENV = "LEAPFLOW_LLM_MODEL"

# Keep the env-var name importable for the terminal summary.
_SUITE_BUDGET_ENV = "LEAPFLOW_LIVE_TOKEN_BUDGET"


class LiveCredentials:
    """Resolved provider coordinates for the live lane."""

    __slots__ = ("base_url", "api_key", "model")

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model


def _resolve_credentials() -> Optional[LiveCredentials]:
    """Return live credentials from the environment, or ``None`` if incomplete.

    All three variables must be present and non-empty; a partial set is treated
    as absent so a half-configured shell skips rather than fails mid-request.
    """
    base_url = os.getenv(_BASE_URL_ENV, "").strip()
    api_key = os.getenv(_API_KEY_ENV, "").strip()
    model = os.getenv(_MODEL_ENV, "").strip()
    if base_url and api_key and model:
        return LiveCredentials(base_url=base_url, api_key=api_key, model=model)
    return None


# ── Public fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def _suite_accumulator(pytestconfig: pytest.Config) -> SuiteAccumulator:
    """Session-scoped cost ledger, stashed on config for the terminal summary."""
    acc = SuiteAccumulator(token_budget=suite_token_budget())
    pytestconfig._leapflow_live_acc = acc  # type: ignore[attr-defined]
    return acc


@pytest.fixture
def live_credentials() -> LiveCredentials:
    """Live provider coordinates, or skip the test if any are missing."""
    creds = _resolve_credentials()
    if creds is None:
        pytest.skip(
            "live LLM credentials absent — set "
            f"{_BASE_URL_ENV}, {_API_KEY_ENV}, {_MODEL_ENV} to run the live lane"
        )
    return creds


@pytest.fixture
def live_provider(live_credentials: LiveCredentials) -> LLMProvider:
    """A real ``OpenAIChat`` bound to the credentialed endpoint.

    Retries are capped low: the live lane's own recovery test drives failover
    explicitly, and elsewhere a stuck endpoint should surface fast rather than
    burning the deadline on SDK-level retries.
    """
    from leapflow.llm.openai_provider import OpenAIChat

    return OpenAIChat(
        api_key=live_credentials.api_key,
        base_url=live_credentials.base_url,
        model=live_credentials.model,
        max_retries=2,
        timeout_s=30.0,
    )


@pytest.fixture
def live_budget(
    request: pytest.FixtureRequest, _suite_accumulator: SuiteAccumulator
) -> Callable[..., LiveBudget]:
    """Factory returning a :class:`LiveBudget` bound to the current test.

    Usage::

        def test_x(live_budget):
            budget = live_budget(max_calls=1, max_tokens=15_000, deadline_s=30)
    """

    def _make(*, max_calls: int, max_tokens: int, deadline_s: float) -> LiveBudget:
        return LiveBudget(
            name=request.node.name,
            max_calls=max_calls,
            max_tokens=max_tokens,
            deadline_s=deadline_s,
            _accumulator=_suite_accumulator,
        )

    return _make


# ── Terminal summary + suite-budget guard ───────────────────────────────────


def pytest_terminal_summary(
    terminalreporter: Any, exitstatus: int, config: pytest.Config
) -> None:
    """Print realised live-lane cost and fail the run if the suite budget blew.

    Delegates to :func:`apply_terminal_summary` (pure, pytest-free) so the
    logic is testable hermetically in ``test_live_budget.py``.
    """
    acc: Optional[SuiteAccumulator] = getattr(config, "_leapflow_live_acc", None)
    if acc is None:
        return

    apply_terminal_summary(
        acc,
        write_line=terminalreporter.write_line,
        write_line_red=lambda msg: terminalreporter.write_line(msg, red=True),
        set_exit_failed=lambda: setattr(
            terminalreporter._session, "exitstatus", pytest.ExitCode.TESTS_FAILED
        ),
        budget_env_name=_SUITE_BUDGET_ENV,
    )
