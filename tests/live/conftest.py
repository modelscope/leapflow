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
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import pytest

from leapflow.llm.base import ChunkCallback, LLMChatResponse, LLMProvider

# ── Credential environment ──────────────────────────────────────────────────
# The same trio production reads (leapflow.config._build_settings_from_env), so
# one set of CI secrets drives both the product and this lane.
_BASE_URL_ENV = "LEAPFLOW_LLM_BASE_URL"
_API_KEY_ENV = "LEAPFLOW_LLM_API_KEY"
_MODEL_ENV = "LEAPFLOW_LLM_MODEL"

# Total-suite ceiling, overridable so a nightly run on a pricier model can widen
# it deliberately rather than by editing code.
_SUITE_BUDGET_ENV = "LEAPFLOW_LIVE_TOKEN_BUDGET"
_DEFAULT_SUITE_TOKEN_BUDGET = 75_000


@dataclass(frozen=True)
class LiveCredentials:
    """Resolved provider coordinates for the live lane."""

    base_url: str
    api_key: str
    model: str


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


# ── Suite-wide cost accumulator ─────────────────────────────────────────────


@dataclass
class _SuiteAccumulator:
    """Running total of calls and tokens across every live test in a session."""

    token_budget: int
    calls: int = 0
    total_tokens: int = 0
    per_test: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add(self, test_name: str, *, calls: int, tokens: int) -> None:
        self.calls += calls
        self.total_tokens += tokens
        slot = self.per_test.setdefault(test_name, {"calls": 0, "tokens": 0})
        slot["calls"] += calls
        slot["tokens"] += tokens

    @property
    def budget_exceeded(self) -> bool:
        return self.total_tokens > self.token_budget


@pytest.fixture(scope="session")
def _suite_accumulator(pytestconfig: pytest.Config) -> _SuiteAccumulator:
    """Session-scoped cost ledger, stashed on config for the terminal summary."""
    acc = _SuiteAccumulator(token_budget=_suite_token_budget())
    pytestconfig._leapflow_live_acc = acc  # type: ignore[attr-defined]
    return acc


def _suite_token_budget() -> int:
    raw = os.getenv(_SUITE_BUDGET_ENV, "").strip()
    if not raw:
        return _DEFAULT_SUITE_TOKEN_BUDGET
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_SUITE_TOKEN_BUDGET
    return value if value > 0 else _DEFAULT_SUITE_TOKEN_BUDGET


# ── Per-test budget ─────────────────────────────────────────────────────────


class LiveBudgetExceeded(AssertionError):
    """A live test crossed its call, token, or wall-clock ceiling."""


@dataclass
class LiveBudget:
    """Hard per-test ceiling on provider calls, tokens, and wall-clock time.

    Every recorded call is checked immediately, so a runaway loop trips on the
    call that crosses the line rather than after the whole test drains its
    iteration budget. Usage is the provider's own ``total_tokens``; a provider
    that reports none contributes zero, which keeps the ceiling honest without
    inventing an estimate.
    """

    name: str
    max_calls: int
    max_tokens: int
    deadline_s: float
    _accumulator: _SuiteAccumulator
    calls: int = 0
    total_tokens: int = 0
    _started: float = field(default_factory=time.monotonic)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._started

    def record_usage(self, usage: Optional[Dict[str, Any]]) -> None:
        """Count one provider call and its tokens, then enforce every ceiling."""
        tokens = 0
        if usage:
            raw = usage.get("total_tokens", 0)
            if isinstance(raw, int) and raw > 0:
                tokens = raw
        self.calls += 1
        self.total_tokens += tokens
        self._accumulator.add(self.name, calls=1, tokens=tokens)

        if self.calls > self.max_calls:
            raise LiveBudgetExceeded(
                f"{self.name!r} made {self.calls} provider calls, past its ceiling "
                f"of {self.max_calls}. A turn stopped converging; investigate rather "
                "than raising the ceiling."
            )
        if self.total_tokens > self.max_tokens:
            raise LiveBudgetExceeded(
                f"{self.name!r} spent {self.total_tokens} tokens, past its ceiling of "
                f"{self.max_tokens}. Prompt growth, not a loop — trim the prompt "
                "rather than raising the ceiling."
            )
        self.check_deadline()

    def check_deadline(self) -> None:
        """Fail if the test has run past its wall-clock deadline."""
        if self.elapsed_s > self.deadline_s:
            raise LiveBudgetExceeded(
                f"{self.name!r} took {self.elapsed_s:.1f}s, over its "
                f"{self.deadline_s:.0f}s deadline."
            )

    def wrap(self, provider: LLMProvider) -> "_BudgetTrackingProvider":
        """Return a provider that records usage into this budget on every call."""
        return _BudgetTrackingProvider(provider, self)


class _BudgetTrackingProvider(LLMProvider):
    """Decorates a provider so every completion feeds the test's budget.

    Both entry points funnel through :meth:`LiveBudget.record_usage`. ``achat``
    carries a real ``usage`` dict; ``achat_stream`` yields raw text with no usage
    frame, so it records one call with zero tokens — accurate for the call count
    and honest about the missing token telemetry. Streaming tests that need token
    accounting use ``achat(stream=True, on_chunk=...)`` instead, which streams and
    still returns usage.
    """

    def __init__(self, inner: LLMProvider, budget: LiveBudget) -> None:
        self._inner = inner
        self._budget = budget

    async def achat(
        self,
        messages: List[Dict[str, Any]],
        *,
        stream: bool = True,
        enable_thinking: bool = False,
        on_chunk: ChunkCallback = None,
        **kwargs: Any,
    ) -> LLMChatResponse:
        resp = await self._inner.achat(
            messages,
            stream=stream,
            enable_thinking=enable_thinking,
            on_chunk=on_chunk,
            **kwargs,
        )
        self._budget.record_usage(getattr(resp, "usage", None))
        return resp

    async def achat_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        got_chunk = False
        async for chunk in self._inner.achat_stream(
            messages, enable_thinking=enable_thinking, **kwargs
        ):
            got_chunk = True
            yield chunk
        # Raw streaming has no usage frame; count the call with zero tokens.
        if got_chunk:
            self._budget.record_usage(None)


# ── Public fixtures ─────────────────────────────────────────────────────────


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
    request: pytest.FixtureRequest, _suite_accumulator: _SuiteAccumulator
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

    Runs after the session, so the total is the true bill for the run — visible
    in CI logs whether the tests passed or not. Only prints when the lane
    actually made calls, so it stays silent for the ordinary offline suite.
    """
    acc: Optional[_SuiteAccumulator] = getattr(config, "_leapflow_live_acc", None)
    if acc is None or acc.calls == 0:
        return

    write = terminalreporter.write_line
    write("")
    write("── live lane cost ──────────────────────────────────────────")
    for name, slot in sorted(acc.per_test.items()):
        write(f"  {name}: {slot['calls']} call(s), {slot['tokens']} token(s)")
    write(
        f"  TOTAL: {acc.calls} call(s), {acc.total_tokens} token(s) "
        f"(budget {acc.token_budget})"
    )
    if acc.budget_exceeded:
        terminalreporter.write_line(
            f"live suite spent {acc.total_tokens} tokens, over the "
            f"{acc.token_budget} suite budget ({_SUITE_BUDGET_ENV})",
            red=True,
        )
        # Turn a green run red: the tests may each pass while the lane as a whole
        # cost more than the operator sanctioned.
        terminalreporter._session.exitstatus = pytest.ExitCode.TESTS_FAILED
