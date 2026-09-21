# Copyright (c) Alibaba, Inc. and its affiliates.
"""Reusable live-lane budget enforcement primitives.

Extracted from ``tests/live/conftest.py`` so that hermetic budget unit tests
can exercise the accounting logic without importing a conftest as a module and
without requiring live LLM credentials.

Two classes form the public contract:

- :class:`SuiteAccumulator` — session-wide running total of calls and tokens.
- :class:`LiveBudget` — per-test ceiling (calls, tokens, wall-clock deadline)
  that records usage, checks limits on every call, and feeds the accumulator.

Both are pure data + arithmetic: no I/O, no provider imports, no pytest
fixtures.  The live conftest wraps them into fixtures and terminal hooks;
the unit test file exercises their boundary behavior directly.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from leapflow.llm.base import ChunkCallback, LLMChatResponse, LLMProvider


# ── Environment knobs ────────────────────────────────────────────────────────

_SUITE_BUDGET_ENV = "LEAPFLOW_LIVE_TOKEN_BUDGET"
_DEFAULT_SUITE_TOKEN_BUDGET = 75_000


def suite_token_budget() -> int:
    """Read the suite-wide token ceiling from the environment.

    Returns the default (75 000) when the variable is absent, empty, or
    non-positive so that a misconfigured shell degrades to the safe default
    rather than silently allowing unlimited spend.
    """
    raw = os.getenv(_SUITE_BUDGET_ENV, "").strip()
    if not raw:
        return _DEFAULT_SUITE_TOKEN_BUDGET
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_SUITE_TOKEN_BUDGET
    return value if value > 0 else _DEFAULT_SUITE_TOKEN_BUDGET


# ── Suite-wide cost accumulator ──────────────────────────────────────────────


@dataclass
class SuiteAccumulator:
    """Running total of calls and tokens across every live test in a session."""

    token_budget: int
    calls: int = 0
    total_tokens: int = 0
    per_test: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add(self, test_name: str, *, calls: int, tokens: int) -> None:
        """Record *calls* / *tokens* under *test_name* and update totals."""
        self.calls += calls
        self.total_tokens += tokens
        slot = self.per_test.setdefault(test_name, {"calls": 0, "tokens": 0})
        slot["calls"] += calls
        slot["tokens"] += tokens

    @property
    def budget_exceeded(self) -> bool:
        """True when the accumulated tokens exceed the suite ceiling."""
        return self.total_tokens > self.token_budget


# ── Per-test budget ──────────────────────────────────────────────────────────


class LiveBudgetExceeded(AssertionError):
    """A live test crossed its call, token, or wall-clock ceiling."""


@dataclass
class LiveBudget:
    """Hard per-test ceiling on provider calls, tokens, and wall-clock time.

    Every recorded call is checked immediately, so a runaway loop trips on the
    call that crosses the line rather than after the whole test drains its
    iteration budget.  Usage is the provider's own ``total_tokens``; a provider
    that reports none contributes zero, which keeps the ceiling honest without
    inventing an estimate.
    """

    name: str
    max_calls: int
    max_tokens: int
    deadline_s: float
    _accumulator: SuiteAccumulator
    calls: int = 0
    total_tokens: int = 0
    _started: float = field(default_factory=time.monotonic)

    # Optional clock override for hermetic tests.
    _clock: Any = field(default=None, repr=False)

    @property
    def elapsed_s(self) -> float:
        """Wall-clock seconds since budget creation."""
        now = self._clock() if self._clock is not None else time.monotonic()
        return now - self._started

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

    def wrap(self, provider: LLMProvider) -> BudgetTrackingProvider:
        """Return a provider wrapper that records usage into this budget."""
        return BudgetTrackingProvider(provider, self)


# ── Budget-tracking provider wrapper ─────────────────────────────────────────


class BudgetTrackingProvider(LLMProvider):
    """Decorates an :class:`LLMProvider` so every completion feeds a budget.

    Both entry points funnel through :meth:`LiveBudget.record_usage`.  ``achat``
    carries a real ``usage`` dict; ``achat_stream`` yields raw text with no usage
    frame, so it records one call with zero tokens — accurate for the call count
    and honest about the missing token telemetry.  Streaming tests that need
    token accounting use ``achat(stream=True, on_chunk=...)`` instead, which
    streams and still returns usage.
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


# ── Terminal summary helper ──────────────────────────────────────────────────


def apply_terminal_summary(
    acc: SuiteAccumulator,
    write_line: Callable[[str], Any],
    write_line_red: Callable[[str], Any],
    set_exit_failed: Callable[[], Any],
    budget_env_name: str = _SUITE_BUDGET_ENV,
) -> None:
    """Pure function that renders the live-lane cost summary.

    Extracted from the pytest terminal hook so it can be tested hermetically.
    Callers pass write helpers and a callback to mark the run failed; this
    function has no pytest dependency.
    """
    if acc.calls == 0:
        return

    write_line("")
    write_line("── live lane cost ──────────────────────────────────────────")
    for name, slot in sorted(acc.per_test.items()):
        write_line(f"  {name}: {slot['calls']} call(s), {slot['tokens']} token(s)")
    write_line(
        f"  TOTAL: {acc.calls} call(s), {acc.total_tokens} token(s) "
        f"(budget {acc.token_budget})"
    )
    if acc.budget_exceeded:
        write_line_red(
            f"live suite spent {acc.total_tokens} tokens, over the "
            f"{acc.token_budget} suite budget ({budget_env_name})"
        )
        set_exit_failed()
