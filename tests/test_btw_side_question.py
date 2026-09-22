# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for /btw side question mechanism.

Covers:
- Command registry presence
- SideQuestionFiber isolation (no parent history writes)
- Handler empty-arg validation
- Usage attribution
- EventBus event emission
- Daemon-mode payload builder
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List

import pytest

from leapflow.llm.base import LLMChatResponse


# ════════════════════════════════════════════════════════════════
# Stubs and helpers
# ════════════════════════════════════════════════════════════════


class FakeLLMForBtw:
    """Minimal LLM provider that records calls and returns a canned response."""

    def __init__(self, reply: str = "42 is the answer.") -> None:
        self._reply = reply
        self.calls: List[Dict[str, Any]] = []

    async def achat(
        self,
        messages: List[Dict[str, Any]],
        *,
        stream: bool = True,
        enable_thinking: bool = False,
        on_chunk: Any = None,
        **kwargs: Any,
    ) -> LLMChatResponse:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return LLMChatResponse(
            content=self._reply,
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cached_tokens": 80,
            },
        )

    async def achat_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        yield self._reply


class FakeEventBus:
    """Event bus stub that records events."""

    def __init__(self) -> None:
        self.events: List[tuple[str, Dict[str, Any]]] = []

    async def handle_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.events.append((event_type, payload))


class FakeUsageTracker:
    """Stub usage tracker that records side question attributions."""

    def __init__(self) -> None:
        self.side_question_calls: List[Dict[str, int]] = []

    def record_side_question(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> None:
        self.side_question_calls.append({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
        })


def _make_fake_engine(
    *,
    llm: Any = None,
    system_prompt: str = "You are a helpful assistant.",
    session_id: str = "test-session-123",
    event_bus: Any = None,
    usage_tracker: Any = None,
) -> SimpleNamespace:
    """Build a minimal engine-like object for SideQuestionFiber."""
    return SimpleNamespace(
        _llm=llm or FakeLLMForBtw(),
        _last_system_prompt=system_prompt,
        _current_session_id=session_id,
        _event_bus=event_bus,
        _usage_tracker=usage_tracker,
    )


class FakeConsole:
    """Console stub that records output calls."""

    def __init__(self) -> None:
        self.warnings: List[str] = []
        self.markdowns: List[str] = []
        self.systems: List[str] = []

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def markdown(self, msg: str) -> None:
        self.markdowns.append(msg)

    def system(self, msg: str) -> None:
        self.systems.append(msg)


# ════════════════════════════════════════════════════════════════
# 1. Command registry
# ════════════════════════════════════════════════════════════════


class TestCommandRegistry:
    """Verify /btw is registered and resolvable."""

    def test_btw_in_registry(self) -> None:
        from leapflow.cli.commands.registry import COMMAND_REGISTRY

        names = [cmd.name for cmd in COMMAND_REGISTRY]
        assert "btw" in names

    def test_btw_alias_aside(self) -> None:
        from leapflow.cli.commands.registry import resolve_command

        cmd = resolve_command("aside hello")
        assert cmd is not None
        assert cmd.name == "btw"

    def test_btw_resolve(self) -> None:
        from leapflow.cli.commands.registry import resolve_command

        cmd = resolve_command("btw what is 2+2")
        assert cmd is not None
        assert cmd.name == "btw"
        assert cmd.category == "Interaction"

    def test_btw_properties(self) -> None:
        from leapflow.cli.commands.registry import COMMAND_REGISTRY

        btw = next(c for c in COMMAND_REGISTRY if c.name == "btw")
        assert btw.client_local is False
        assert btw.requires_llm is True
        assert btw.effect.value == "read_only"
        assert btw.execution.value == "streaming"
        assert btw.args_hint == "<question>"

    def test_btw_in_completion_entries(self) -> None:
        from leapflow.cli.commands.registry import completion_entries

        entries = completion_entries()
        names = [name for name, _ in entries]
        assert "btw" in names


# ════════════════════════════════════════════════════════════════
# 2. SideQuestionFiber
# ════════════════════════════════════════════════════════════════


class TestSideQuestionFiber:
    """Verify fiber isolation and behaviour."""

    @pytest.mark.asyncio
    async def test_fiber_returns_answer(self) -> None:
        """Fiber should yield the LLM response content."""
        from leapflow.engine.side_question import SideQuestionConfig, SideQuestionFiber

        engine = _make_fake_engine()
        config = SideQuestionConfig(question="What is 6*7?", parent_session_id="s1")
        fiber = SideQuestionFiber(engine, config)

        result = await fiber.run()
        assert result == "42 is the answer."

    @pytest.mark.asyncio
    async def test_fiber_does_not_write_to_parent(self) -> None:
        """Fiber must not call any store/memory method on the engine."""
        from leapflow.engine.side_question import SideQuestionConfig, SideQuestionFiber

        engine = _make_fake_engine()
        config = SideQuestionConfig(question="hello", parent_session_id="s1")
        fiber = SideQuestionFiber(engine, config)

        await fiber.run()

        # The engine has no _wm, _conversation_store — fiber must not try to access them
        assert not hasattr(engine, "_wm")
        assert not hasattr(engine, "_conversation_store")

    @pytest.mark.asyncio
    async def test_fiber_builds_minimal_messages(self) -> None:
        """The message list should be exactly [system, user]."""
        from leapflow.engine.side_question import SideQuestionConfig, SideQuestionFiber

        llm = FakeLLMForBtw()
        engine = _make_fake_engine(llm=llm, system_prompt="You are LeapFlow.")
        config = SideQuestionConfig(question="What time is it?", parent_session_id="s1")
        fiber = SideQuestionFiber(engine, config)

        await fiber.run()

        assert len(llm.calls) == 1
        messages = llm.calls[0]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are LeapFlow."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What time is it?"

    @pytest.mark.asyncio
    async def test_fiber_disables_tools_and_thinking(self) -> None:
        """achat must be called with tools=None, tool_choice=None, enable_thinking=False."""
        from leapflow.engine.side_question import SideQuestionConfig, SideQuestionFiber

        llm = FakeLLMForBtw()
        engine = _make_fake_engine(llm=llm)
        config = SideQuestionConfig(question="quick q", parent_session_id="s1")
        fiber = SideQuestionFiber(engine, config)

        await fiber.run()

        assert len(llm.calls) == 1
        kwargs = llm.calls[0]["kwargs"]
        assert "tools" in kwargs and kwargs["tools"] is None, (
            "tools must be explicitly set to None to disable tool calling"
        )
        assert "tool_choice" in kwargs and kwargs["tool_choice"] is None, (
            "tool_choice must be explicitly set to None"
        )

    @pytest.mark.asyncio
    async def test_fiber_uses_fallback_system_prompt(self) -> None:
        """When engine has no system prompt, a fallback should be used."""
        from leapflow.engine.side_question import SideQuestionConfig, SideQuestionFiber

        llm = FakeLLMForBtw()
        engine = _make_fake_engine(llm=llm, system_prompt="")
        config = SideQuestionConfig(question="hi", parent_session_id="s1")
        fiber = SideQuestionFiber(engine, config)

        await fiber.run()

        messages = llm.calls[0]["messages"]
        assert "helpful assistant" in messages[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_fiber_stream_yields_content(self) -> None:
        """run_stream() should yield at least one chunk."""
        from leapflow.engine.side_question import SideQuestionConfig, SideQuestionFiber

        engine = _make_fake_engine()
        config = SideQuestionConfig(question="test", parent_session_id="s1")
        fiber = SideQuestionFiber(engine, config)

        chunks = []
        async for chunk in fiber.run_stream():
            chunks.append(chunk)

        assert len(chunks) >= 1
        assert "".join(chunks) == "42 is the answer."

    @pytest.mark.asyncio
    async def test_fiber_handles_llm_error_gracefully(self) -> None:
        """LLM failure should yield an error message, not raise."""
        from leapflow.engine.side_question import SideQuestionConfig, SideQuestionFiber

        class FailingLLM:
            async def achat(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("LLM on fire")

        engine = _make_fake_engine(llm=FailingLLM())
        config = SideQuestionConfig(question="test", parent_session_id="s1")
        fiber = SideQuestionFiber(engine, config)

        chunks = []
        async for chunk in fiber.run_stream():
            chunks.append(chunk)

        output = "".join(chunks)
        assert "failed" in output.lower()


# ════════════════════════════════════════════════════════════════
# 3. EventBus emission
# ════════════════════════════════════════════════════════════════


class TestEventBusEmission:
    """Verify that fiber emits started/completed events."""

    @pytest.mark.asyncio
    async def test_emits_started_and_completed(self) -> None:
        from leapflow.engine.side_question import SideQuestionConfig, SideQuestionFiber

        bus = FakeEventBus()
        engine = _make_fake_engine(event_bus=bus)
        config = SideQuestionConfig(question="q", parent_session_id="s1")
        fiber = SideQuestionFiber(engine, config)

        await fiber.run()

        # Let event tasks complete
        await asyncio.sleep(0.05)

        event_types = [et for et, _ in bus.events]
        assert "side_question.started" in event_types
        assert "side_question.completed" in event_types

    @pytest.mark.asyncio
    async def test_completed_carries_usage(self) -> None:
        from leapflow.engine.side_question import SideQuestionConfig, SideQuestionFiber

        bus = FakeEventBus()
        engine = _make_fake_engine(event_bus=bus)
        config = SideQuestionConfig(question="q", parent_session_id="s1")
        fiber = SideQuestionFiber(engine, config)

        await fiber.run()
        await asyncio.sleep(0.05)

        completed = [p for et, p in bus.events if et == "side_question.completed"]
        assert len(completed) == 1
        assert completed[0]["prompt_tokens"] == 100
        assert completed[0]["completion_tokens"] == 20
        assert completed[0]["cached_tokens"] == 80


# ════════════════════════════════════════════════════════════════
# 4. Usage attribution
# ════════════════════════════════════════════════════════════════


class TestUsageAttribution:
    """Verify token usage is attributed to parent session."""

    @pytest.mark.asyncio
    async def test_usage_recorded_on_tracker(self) -> None:
        from leapflow.engine.side_question import SideQuestionConfig, SideQuestionFiber

        tracker = FakeUsageTracker()
        engine = _make_fake_engine(usage_tracker=tracker)
        config = SideQuestionConfig(question="q", parent_session_id="s1")
        fiber = SideQuestionFiber(engine, config)

        await fiber.run()

        assert len(tracker.side_question_calls) == 1
        assert tracker.side_question_calls[0]["prompt_tokens"] == 100

    @pytest.mark.asyncio
    async def test_usage_degrades_gracefully_without_method(self) -> None:
        """If tracker lacks record_side_question, fiber should not crash."""
        from leapflow.engine.side_question import SideQuestionConfig, SideQuestionFiber

        engine = _make_fake_engine(usage_tracker=SimpleNamespace())
        config = SideQuestionConfig(question="q", parent_session_id="s1")
        fiber = SideQuestionFiber(engine, config)

        # Should not raise
        result = await fiber.run()
        assert result == "42 is the answer."


# ════════════════════════════════════════════════════════════════
# 5. Handler validation
# ════════════════════════════════════════════════════════════════


class TestHandlerValidation:
    """Verify btw_handler.handle_btw input validation."""

    @pytest.mark.asyncio
    async def test_empty_args_shows_usage(self) -> None:
        from leapflow.cli.commands.btw_handler import handle_btw

        console = FakeConsole()
        ctx = SimpleNamespace(engine=_make_fake_engine())

        await handle_btw(ctx, console, "")

        assert len(console.warnings) == 1
        assert "Usage" in console.warnings[0]

    @pytest.mark.asyncio
    async def test_no_engine_shows_warning(self) -> None:
        from leapflow.cli.commands.btw_handler import handle_btw

        console = FakeConsole()
        ctx = SimpleNamespace(engine=None)

        await handle_btw(ctx, console, "What is 2+2?")

        assert len(console.warnings) == 1
        assert "engine" in console.warnings[0].lower()


# ════════════════════════════════════════════════════════════════
# 6. Daemon payload builder
# ════════════════════════════════════════════════════════════════


class TestBuildBtwPayload:
    """Verify build_btw_payload for daemon-mode execution."""

    @pytest.mark.asyncio
    async def test_empty_args_returns_error(self) -> None:
        from leapflow.cli.commands.btw_handler import build_btw_payload

        ctx = SimpleNamespace(engine=_make_fake_engine())
        payload = await build_btw_payload(ctx, "")
        assert payload["ok"] is False
        assert "Usage" in payload["message"]

    @pytest.mark.asyncio
    async def test_no_engine_returns_error(self) -> None:
        from leapflow.cli.commands.btw_handler import build_btw_payload

        ctx = SimpleNamespace(engine=None)
        payload = await build_btw_payload(ctx, "hello")
        assert payload["ok"] is False

    @pytest.mark.asyncio
    async def test_successful_payload(self) -> None:
        from leapflow.cli.commands.btw_handler import build_btw_payload

        ctx = SimpleNamespace(engine=_make_fake_engine())
        payload = await build_btw_payload(ctx, "What is 6*7?")

        assert payload["ok"] is True
        assert payload["view"] == "btw"
        assert payload["answer"] == "42 is the answer."
        assert payload["question"] == "What is 6*7?"
        assert "fiber_id" in payload
        assert payload["parent_session_id"] == "test-session-123"


# ════════════════════════════════════════════════════════════════
# 7. SideQuestionConfig
# ════════════════════════════════════════════════════════════════


class TestSideQuestionConfig:
    """Verify config dataclass properties."""

    def test_config_is_frozen(self) -> None:
        from leapflow.engine.side_question import SideQuestionConfig

        config = SideQuestionConfig(question="q", parent_session_id="s1")
        with pytest.raises(AttributeError):
            config.question = "modified"  # type: ignore[misc]

    def test_config_defaults(self) -> None:
        from leapflow.engine.side_question import SideQuestionConfig

        config = SideQuestionConfig(question="q", parent_session_id="s1")
        assert config.max_tokens == 2048
        assert config.disclosure_level == "CORE"
        assert config.fiber_id.startswith("btw-")

    def test_config_custom_values(self) -> None:
        from leapflow.engine.side_question import SideQuestionConfig

        config = SideQuestionConfig(
            question="q",
            parent_session_id="s1",
            max_tokens=512,
            disclosure_level="EXPANDED",
            fiber_id="btw-custom",
        )
        assert config.max_tokens == 512
        assert config.disclosure_level == "EXPANDED"
        assert config.fiber_id == "btw-custom"


# ════════════════════════════════════════════════════════════════
# 8. Dispatcher routing (command_execute)
# ════════════════════════════════════════════════════════════════


class TestDispatcherRouting:
    """Verify /btw routes through command_execute."""

    @pytest.mark.asyncio
    async def test_command_execute_routes_btw(self) -> None:
        """command_execute('btw', ...) should call build_btw_payload."""
        from leapflow.cli.commands.slash_handlers import command_execute

        ctx = SimpleNamespace(engine=_make_fake_engine())
        payload = await command_execute(ctx, "btw", "What is pi?")

        assert payload["ok"] is True
        assert payload["view"] == "btw"
        assert "pi" in payload["question"]
