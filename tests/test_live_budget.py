# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hermetic unit tests for live-lane budget enforcement.

These exercise :class:`LiveBudget`, :class:`SuiteAccumulator`,
:class:`BudgetTrackingProvider`, and :func:`apply_terminal_summary` without
real LLM tokens.  They live in the root test directory (not ``tests/live/``)
so they run in every normal PR check rather than being gated behind credentials.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List

import pytest

from leapflow.llm.base import ChunkCallback, LLMChatResponse, LLMProvider

from tests._harness.live_budget import (
    BudgetTrackingProvider,
    LiveBudget,
    LiveBudgetExceeded,
    SuiteAccumulator,
    apply_terminal_summary,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_budget(
    *,
    name: str = "test_budget",
    max_calls: int = 10,
    max_tokens: int = 5000,
    deadline_s: float = 60.0,
    token_budget: int = 100_000,
    clock: Any = None,
) -> LiveBudget:
    """Factory for a LiveBudget with a fresh SuiteAccumulator."""
    acc = SuiteAccumulator(token_budget=token_budget)
    budget = LiveBudget(
        name=name,
        max_calls=max_calls,
        max_tokens=max_tokens,
        deadline_s=deadline_s,
        _accumulator=acc,
        _clock=clock,
    )
    return budget


def _usage(total_tokens: int) -> Dict[str, Any]:
    """Build a minimal usage dict matching provider output shape."""
    return {"total_tokens": total_tokens}


# ── Fake provider for wrap() tests ───────────────────────────────────────────


class _FakeProvider(LLMProvider):
    """In-memory provider that returns canned responses for hermetic tests."""

    def __init__(
        self,
        *,
        response_content: str = "ok",
        usage_tokens: int = 100,
        stream_chunks: List[str] | None = None,
        raise_on_achat: BaseException | None = None,
    ) -> None:
        self._response_content = response_content
        self._usage_tokens = usage_tokens
        self._stream_chunks = stream_chunks if stream_chunks is not None else ["ch1", "ch2"]
        self._raise_on_achat = raise_on_achat
        self.achat_calls: int = 0
        self.stream_calls: int = 0

    async def achat(
        self,
        messages: List[Dict[str, Any]],
        *,
        stream: bool = True,
        enable_thinking: bool = False,
        on_chunk: ChunkCallback = None,
        **kwargs: Any,
    ) -> LLMChatResponse:
        self.achat_calls += 1
        if self._raise_on_achat is not None:
            raise self._raise_on_achat
        return LLMChatResponse(
            content=self._response_content,
            usage={"total_tokens": self._usage_tokens},
        )

    async def achat_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        self.stream_calls += 1
        for chunk in self._stream_chunks:
            yield chunk


# ═════════════════════════════════════════════════════════════════════════════
# 1. Per-test max_calls
# ═════════════════════════════════════════════════════════════════════════════


class TestMaxCallsBudget:
    def test_calls_within_limit_pass(self) -> None:
        """Recording calls up to exactly max_calls should not raise."""
        budget = _make_budget(max_calls=3)
        for _ in range(3):
            budget.record_usage(_usage(100))
        assert budget.calls == 3

    def test_calls_exceeding_limit_raises(self) -> None:
        """The call *after* max_calls must raise LiveBudgetExceeded."""
        budget = _make_budget(max_calls=2)
        budget.record_usage(_usage(10))
        budget.record_usage(_usage(10))
        with pytest.raises(LiveBudgetExceeded, match="provider calls"):
            budget.record_usage(_usage(10))

    def test_boundary_exactly_at_max_passes(self) -> None:
        """Exactly max_calls is within ceiling (> not >=)."""
        budget = _make_budget(max_calls=1)
        budget.record_usage(_usage(0))
        assert budget.calls == 1
        # Next call exceeds
        with pytest.raises(LiveBudgetExceeded):
            budget.record_usage(_usage(0))


# ═════════════════════════════════════════════════════════════════════════════
# 2. Per-test max_tokens
# ═════════════════════════════════════════════════════════════════════════════


class TestMaxTokensBudget:
    def test_tokens_within_limit_pass(self) -> None:
        """Tokens summing to exactly max_tokens should not raise."""
        budget = _make_budget(max_tokens=500, max_calls=100)
        budget.record_usage(_usage(250))
        budget.record_usage(_usage(250))
        assert budget.total_tokens == 500

    def test_tokens_exceeding_limit_raises(self) -> None:
        """The call whose cumulative tokens exceed max_tokens must raise."""
        budget = _make_budget(max_tokens=500, max_calls=100)
        budget.record_usage(_usage(400))
        with pytest.raises(LiveBudgetExceeded, match="tokens"):
            budget.record_usage(_usage(200))

    def test_none_usage_contributes_zero(self) -> None:
        """A provider returning no usage dict adds zero tokens."""
        budget = _make_budget(max_tokens=100, max_calls=100)
        budget.record_usage(None)
        assert budget.total_tokens == 0
        assert budget.calls == 1


# ═════════════════════════════════════════════════════════════════════════════
# 3. Deadline enforcement
# ═════════════════════════════════════════════════════════════════════════════


class TestDeadlineBudget:
    def test_within_deadline_passes(self) -> None:
        """When clock shows time within deadline, no error."""
        fake_time = [100.0]
        budget = _make_budget(deadline_s=10.0, max_calls=100, clock=lambda: fake_time[0])
        budget._started = 100.0
        fake_time[0] = 105.0  # 5s elapsed, within 10s deadline
        budget.record_usage(_usage(10))

    def test_past_deadline_raises(self) -> None:
        """When clock shows time past deadline, raises on next record_usage."""
        fake_time = [0.0]
        budget = _make_budget(deadline_s=5.0, max_calls=100, clock=lambda: fake_time[0])
        budget._started = 0.0
        fake_time[0] = 6.0  # 6s elapsed, over 5s deadline
        with pytest.raises(LiveBudgetExceeded, match="deadline"):
            budget.record_usage(_usage(0))

    def test_check_deadline_standalone(self) -> None:
        """check_deadline() can be called directly without recording usage."""
        fake_time = [0.0]
        budget = _make_budget(deadline_s=2.0, clock=lambda: fake_time[0])
        budget._started = 0.0
        fake_time[0] = 1.0
        budget.check_deadline()  # should not raise
        fake_time[0] = 3.0
        with pytest.raises(LiveBudgetExceeded, match="deadline"):
            budget.check_deadline()


# ═════════════════════════════════════════════════════════════════════════════
# 4. Suite accumulator total budget
# ═════════════════════════════════════════════════════════════════════════════


class TestSuiteAccumulator:
    def test_accumulates_across_tests(self) -> None:
        """Tokens from multiple test names accumulate into the total."""
        acc = SuiteAccumulator(token_budget=1000)
        acc.add("test_a", calls=1, tokens=300)
        acc.add("test_b", calls=2, tokens=400)
        assert acc.calls == 3
        assert acc.total_tokens == 700
        assert not acc.budget_exceeded

    def test_budget_exceeded_flag(self) -> None:
        """budget_exceeded goes True when total_tokens > token_budget."""
        acc = SuiteAccumulator(token_budget=500)
        acc.add("test_x", calls=1, tokens=501)
        assert acc.budget_exceeded

    def test_exactly_at_budget_not_exceeded(self) -> None:
        """Exactly at budget is not exceeded (> not >=)."""
        acc = SuiteAccumulator(token_budget=500)
        acc.add("test_y", calls=1, tokens=500)
        assert not acc.budget_exceeded

    def test_per_test_tracking(self) -> None:
        """per_test dict records calls and tokens per test name."""
        acc = SuiteAccumulator(token_budget=10_000)
        acc.add("test_a", calls=1, tokens=100)
        acc.add("test_a", calls=1, tokens=200)
        acc.add("test_b", calls=1, tokens=50)
        assert acc.per_test["test_a"] == {"calls": 2, "tokens": 300}
        assert acc.per_test["test_b"] == {"calls": 1, "tokens": 50}

    def test_suite_budget_propagation_from_live_budget(self) -> None:
        """LiveBudget.record_usage feeds into the suite accumulator."""
        acc = SuiteAccumulator(token_budget=200)
        budget = LiveBudget(
            name="test_prop",
            max_calls=100,
            max_tokens=10_000,
            deadline_s=999.0,
            _accumulator=acc,
        )
        budget.record_usage(_usage(150))
        budget.record_usage(_usage(60))
        assert acc.total_tokens == 210
        assert acc.budget_exceeded


# ═════════════════════════════════════════════════════════════════════════════
# 5. BudgetTrackingProvider via wrap()
# ═════════════════════════════════════════════════════════════════════════════


class TestBudgetWrapAchat:
    """Verify budget.wrap(provider) delegates achat and records usage once."""

    @pytest.mark.asyncio
    async def test_wrap_returns_budget_tracking_provider(self) -> None:
        """budget.wrap() returns a BudgetTrackingProvider instance."""
        budget = _make_budget(max_calls=5, max_tokens=10_000)
        fake = _FakeProvider(usage_tokens=200)
        wrapped = budget.wrap(fake)
        assert isinstance(wrapped, BudgetTrackingProvider)
        assert isinstance(wrapped, LLMProvider)

    @pytest.mark.asyncio
    async def test_achat_delegates_and_records_usage_once(self) -> None:
        """A single achat call delegates to inner and records usage exactly once."""
        budget = _make_budget(max_calls=10, max_tokens=10_000)
        fake = _FakeProvider(response_content="hello", usage_tokens=350)
        wrapped = budget.wrap(fake)

        resp = await wrapped.achat([{"role": "user", "content": "hi"}], stream=False)

        assert resp.content == "hello"
        assert resp.usage == {"total_tokens": 350}
        assert fake.achat_calls == 1
        assert budget.calls == 1
        assert budget.total_tokens == 350
        assert budget._accumulator.total_tokens == 350

    @pytest.mark.asyncio
    async def test_multiple_achat_calls_accumulate(self) -> None:
        """Successive calls accumulate in the budget."""
        budget = _make_budget(max_calls=10, max_tokens=10_000)
        fake = _FakeProvider(usage_tokens=100)
        wrapped = budget.wrap(fake)

        await wrapped.achat([{"role": "user", "content": "a"}], stream=False)
        await wrapped.achat([{"role": "user", "content": "b"}], stream=False)

        assert budget.calls == 2
        assert budget.total_tokens == 200


class TestBudgetWrapStream:
    """Verify streaming path retains chunk behavior and records usage once."""

    @pytest.mark.asyncio
    async def test_stream_yields_all_chunks(self) -> None:
        """achat_stream on the wrapper yields every chunk from the inner."""
        budget = _make_budget(max_calls=10, max_tokens=10_000)
        fake = _FakeProvider(stream_chunks=["alpha", "beta", "gamma"])
        wrapped = budget.wrap(fake)

        chunks: List[str] = []
        async for chunk in wrapped.achat_stream(
            [{"role": "user", "content": "stream"}]
        ):
            chunks.append(chunk)

        assert chunks == ["alpha", "beta", "gamma"]
        assert fake.stream_calls == 1

    @pytest.mark.asyncio
    async def test_stream_records_one_call_zero_tokens(self) -> None:
        """Streaming records one call with zero tokens (no usage frame)."""
        budget = _make_budget(max_calls=10, max_tokens=10_000)
        fake = _FakeProvider(stream_chunks=["x"])
        wrapped = budget.wrap(fake)

        async for _ in wrapped.achat_stream(
            [{"role": "user", "content": "stream"}]
        ):
            pass

        assert budget.calls == 1
        assert budget.total_tokens == 0

    @pytest.mark.asyncio
    async def test_empty_stream_records_nothing(self) -> None:
        """When the stream yields zero chunks, no usage is recorded."""
        budget = _make_budget(max_calls=10, max_tokens=10_000)
        fake = _FakeProvider(stream_chunks=[])
        wrapped = budget.wrap(fake)

        async for _ in wrapped.achat_stream(
            [{"role": "user", "content": "nothing"}]
        ):
            pass

        assert budget.calls == 0
        assert budget.total_tokens == 0


class TestBudgetWrapException:
    """Provider exceptions must not fabricate usage."""

    @pytest.mark.asyncio
    async def test_exception_does_not_record_usage(self) -> None:
        """When the inner provider raises, no usage is recorded."""
        budget = _make_budget(max_calls=10, max_tokens=10_000)
        fake = _FakeProvider(raise_on_achat=RuntimeError("boom"))
        wrapped = budget.wrap(fake)

        with pytest.raises(RuntimeError, match="boom"):
            await wrapped.achat([{"role": "user", "content": "fail"}], stream=False)

        assert budget.calls == 0
        assert budget.total_tokens == 0
        assert budget._accumulator.total_tokens == 0

    @pytest.mark.asyncio
    async def test_deadline_still_enforced_after_exception(self) -> None:
        """Even if a call fails, the deadline check still works on next call."""
        fake_time = [0.0]
        budget = _make_budget(
            max_calls=10, max_tokens=10_000, deadline_s=5.0,
            clock=lambda: fake_time[0],
        )
        budget._started = 0.0
        fake_good = _FakeProvider(usage_tokens=10)
        wrapped = budget.wrap(fake_good)

        # First call within deadline succeeds
        fake_time[0] = 2.0
        await wrapped.achat([{"role": "user", "content": "ok"}], stream=False)
        assert budget.calls == 1

        # Time advances past deadline; next call trips the deadline
        fake_time[0] = 6.0
        with pytest.raises(LiveBudgetExceeded, match="deadline"):
            await wrapped.achat([{"role": "user", "content": "late"}], stream=False)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Terminal summary (apply_terminal_summary)
# ═════════════════════════════════════════════════════════════════════════════


class TestApplyTerminalSummary:
    """Hermetic tests for the pure summary function."""

    def test_zero_calls_produces_no_output(self) -> None:
        """An accumulator with zero calls causes no output at all."""
        acc = SuiteAccumulator(token_budget=1000)
        lines: List[str] = []
        red_lines: List[str] = []
        failed = []

        apply_terminal_summary(
            acc,
            write_line=lines.append,
            write_line_red=red_lines.append,
            set_exit_failed=lambda: failed.append(True),
        )

        assert lines == []
        assert red_lines == []
        assert failed == []

    def test_within_budget_prints_summary_without_failure(self) -> None:
        """Non-zero calls within budget print the cost table but no red line."""
        acc = SuiteAccumulator(token_budget=10_000)
        acc.add("test_alpha", calls=2, tokens=500)
        acc.add("test_beta", calls=1, tokens=300)
        lines: List[str] = []
        red_lines: List[str] = []
        failed = []

        apply_terminal_summary(
            acc,
            write_line=lines.append,
            write_line_red=red_lines.append,
            set_exit_failed=lambda: failed.append(True),
        )

        # Summary rows printed
        assert any("test_alpha" in ln for ln in lines)
        assert any("test_beta" in ln for ln in lines)
        assert any("TOTAL" in ln for ln in lines)
        # No red / no failure
        assert red_lines == []
        assert failed == []

    def test_exceeded_budget_marks_failure_and_prints_red(self) -> None:
        """When budget_exceeded is True, a red line is printed and exit fails."""
        acc = SuiteAccumulator(token_budget=100)
        acc.add("expensive_test", calls=1, tokens=200)
        lines: List[str] = []
        red_lines: List[str] = []
        failed = []

        apply_terminal_summary(
            acc,
            write_line=lines.append,
            write_line_red=red_lines.append,
            set_exit_failed=lambda: failed.append(True),
        )

        # Red overage line
        assert len(red_lines) == 1
        assert "200" in red_lines[0]
        assert "100" in red_lines[0]
        # Failure callback invoked exactly once
        assert failed == [True]

    def test_per_test_rows_sorted_alphabetically(self) -> None:
        """Summary rows appear in sorted order of test name."""
        acc = SuiteAccumulator(token_budget=99_999)
        acc.add("test_zebra", calls=1, tokens=10)
        acc.add("test_alpha", calls=1, tokens=20)
        lines: List[str] = []

        apply_terminal_summary(
            acc,
            write_line=lines.append,
            write_line_red=lambda _: None,
            set_exit_failed=lambda: None,
        )

        # Find the two per-test rows
        test_rows = [ln for ln in lines if "test_" in ln and "TOTAL" not in ln]
        assert len(test_rows) == 2
        assert "test_alpha" in test_rows[0]
        assert "test_zebra" in test_rows[1]
