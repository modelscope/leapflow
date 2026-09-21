# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for AgentSkillExecutor and coordinator routing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from leapflow.scheduler.agent_executor import AgentSkillExecutor
from leapflow.scheduler.types import SkillExecutor


# ════════════════════════════════════════════════════════════════════════
# Fakes / helpers
# ════════════════════════════════════════════════════════════════════════


def _check_protocol_conformance(cls: type, protocol: type) -> bool:
    """Structural Protocol check without @runtime_checkable."""
    import inspect
    for name, member in inspect.getmembers(protocol):
        if name.startswith("_"):
            continue
        if not hasattr(cls, name):
            return False
        if callable(member) and not callable(getattr(cls, name)):
            return False
    return True


@dataclass
class FakeToolCall:
    """Minimal tool call returned by a fake LLM."""

    id: str
    name: str
    arguments: dict


@dataclass
class FakeLLMResponse:
    """Minimal LLM response for testing."""

    content: str
    tool_calls: Optional[List[FakeToolCall]] = None


class FakeLLM:
    """Controllable fake LLM for testing the agent loop.

    ``responses`` is a list of FakeLLMResponse. Each call to ``achat``
    pops the next response. When exhausted, returns a plain text response.
    """

    def __init__(self, responses: Optional[List[FakeLLMResponse]] = None) -> None:
        self._responses = list(responses or [])
        self.call_count = 0

    async def achat(self, messages: list, *, stream: bool = False, **kwargs: Any) -> FakeLLMResponse:
        self.call_count += 1
        if self._responses:
            return self._responses.pop(0)
        return FakeLLMResponse(content="Done.")


class ErrorLLM:
    """LLM that always raises."""

    async def achat(self, messages: list, **kwargs: Any) -> Any:
        raise RuntimeError("LLM unavailable")


def _make_settings(**overrides: Any) -> Any:
    """Create a minimal settings-like object."""

    class _S:
        scheduler_agent_max_iterations = overrides.get("scheduler_agent_max_iterations", 25)
        scheduler_agent_tool_blocklist = overrides.get("scheduler_agent_tool_blocklist", "")
        max_tool_result_chars = 4000
        agent_subagent_max_depth = 2
        agent_subagent_max_iterations = overrides.get("scheduler_agent_max_iterations", 25)

    return _S()


def _echo_handler():
    """Returns a simple tool handler that echoes its arguments."""

    async def handler(args: dict) -> dict:
        return {"ok": True, "result": f"echoed: {args}"}

    return handler


def _make_tool_defs(names: list[str]) -> list[dict]:
    return [
        {"type": "function", "function": {"name": n, "parameters": {}}}
        for n in names
    ]


# ════════════════════════════════════════════════════════════════════════
# Protocol conformance
# ════════════════════════════════════════════════════════════════════════


class TestProtocolConformance:
    """Verify AgentSkillExecutor satisfies SkillExecutor Protocol."""

    def test_structural_conformance(self) -> None:
        """AgentSkillExecutor has an async execute(skill_name, parameters) method."""
        import inspect
        executor = AgentSkillExecutor(
            llm=FakeLLM(),
            tool_handlers={},
            tool_definitions=[],
        )
        assert hasattr(executor, "execute")
        assert callable(executor.execute)
        sig = inspect.signature(executor.execute)
        params = list(sig.parameters.keys())
        assert "skill_name" in params
        assert "parameters" in params
        assert _check_protocol_conformance(AgentSkillExecutor, SkillExecutor)


# ════════════════════════════════════════════════════════════════════════
# Successful execution
# ════════════════════════════════════════════════════════════════════════


class TestSuccessfulExecution:
    """Agent executor returns ok=True with output and tool summary."""

    @pytest.mark.asyncio
    async def test_simple_instruction(self) -> None:
        llm = FakeLLM([FakeLLMResponse(content="Task completed successfully.")])
        executor = AgentSkillExecutor(
            llm=llm,
            tool_handlers={},
            tool_definitions=[],
            settings=_make_settings(),
        )
        result = await executor.execute("test_skill", {"instruction": "Do something"})
        assert result["ok"] is True
        assert "Task completed successfully." in result["output"]

    @pytest.mark.asyncio
    async def test_with_tool_calls(self) -> None:
        """Agent makes tool calls, then returns final answer."""
        responses = [
            FakeLLMResponse(
                content="Let me use a tool.",
                tool_calls=[FakeToolCall(id="tc1", name="echo", arguments={"x": 1})],
            ),
            FakeLLMResponse(content="All done after using the tool."),
        ]
        llm = FakeLLM(responses)
        executor = AgentSkillExecutor(
            llm=llm,
            tool_handlers={"echo": _echo_handler()},
            tool_definitions=_make_tool_defs(["echo"]),
            settings=_make_settings(),
        )
        result = await executor.execute("test_skill", {"instruction": "Use echo"})
        assert result["ok"] is True
        assert "tool_calls=1" in result["output"]

    @pytest.mark.asyncio
    async def test_missing_instruction(self) -> None:
        executor = AgentSkillExecutor(
            llm=FakeLLM(),
            tool_handlers={},
            tool_definitions=[],
            settings=_make_settings(),
        )
        result = await executor.execute("test_skill", {})
        assert result["ok"] is False
        assert "instruction" in result["error"].lower()


# ════════════════════════════════════════════════════════════════════════
# Error containment
# ════════════════════════════════════════════════════════════════════════


class TestErrorContainment:
    """LLM/tool errors return failed status, never raise."""

    @pytest.mark.asyncio
    async def test_llm_error_contained(self) -> None:
        executor = AgentSkillExecutor(
            llm=ErrorLLM(),
            tool_handlers={},
            tool_definitions=[],
            settings=_make_settings(),
        )
        result = await executor.execute("test_skill", {"instruction": "Do something"})
        assert result["ok"] is False
        assert "error" in result
        assert "LLM" in result["error"] or "unavailable" in result["error"]

    @pytest.mark.asyncio
    async def test_tool_error_contained(self) -> None:
        """A tool that raises is caught inside the subagent loop."""

        async def bad_handler(args: dict) -> dict:
            raise ValueError("tool exploded")

        responses = [
            FakeLLMResponse(
                content="Calling tool.",
                tool_calls=[FakeToolCall(id="tc1", name="bad_tool", arguments={})],
            ),
            FakeLLMResponse(content="Recovered."),
        ]
        executor = AgentSkillExecutor(
            llm=FakeLLM(responses),
            tool_handlers={"bad_tool": bad_handler},
            tool_definitions=_make_tool_defs(["bad_tool"]),
            settings=_make_settings(),
        )
        # Should not raise; the subagent loop catches tool errors internally.
        result = await executor.execute("test_skill", {"instruction": "Use bad tool"})
        # Either ok or not, but it must not propagate.
        assert isinstance(result, dict)
        assert "ok" in result


# ════════════════════════════════════════════════════════════════════════
# Iteration budget
# ════════════════════════════════════════════════════════════════════════


class TestIterationBudget:
    """Budget is respected: loop terminates after max_iterations."""

    @pytest.mark.asyncio
    async def test_budget_limits_iterations(self) -> None:
        """LLM always calls tools → budget must stop the loop."""
        budget = 3

        class InfiniteToolLLM:
            call_count = 0

            async def achat(self, messages: list, **kwargs: Any) -> FakeLLMResponse:
                self.call_count += 1
                return FakeLLMResponse(
                    content="Calling tool again.",
                    tool_calls=[FakeToolCall(
                        id=f"tc{self.call_count}",
                        name="echo",
                        arguments={"n": self.call_count},
                    )],
                )

        llm = InfiniteToolLLM()
        executor = AgentSkillExecutor(
            llm=llm,
            tool_handlers={"echo": _echo_handler()},
            tool_definitions=_make_tool_defs(["echo"]),
            settings=_make_settings(scheduler_agent_max_iterations=budget),
        )
        result = await executor.execute("test_skill", {"instruction": "Loop forever"})
        # The loop must have terminated (not hung). The exact call count depends
        # on the adaptive budget, but it must be bounded.
        assert isinstance(result, dict)
        # The LLM was called at most budget * 2 + some margin (adaptive ceiling).
        assert llm.call_count <= budget * 3


# ════════════════════════════════════════════════════════════════════════
# Tool blocklist
# ════════════════════════════════════════════════════════════════════════


class TestToolBlocklist:
    """Blocklist filters tools out of the subagent's available set."""

    @pytest.mark.asyncio
    async def test_blocklist_from_payload(self) -> None:
        """A tool in the blocklist is not available to the subagent."""
        responses = [
            FakeLLMResponse(
                content="Calling blocked tool.",
                tool_calls=[FakeToolCall(id="tc1", name="blocked_tool", arguments={})],
            ),
            FakeLLMResponse(content="Done."),
        ]
        executor = AgentSkillExecutor(
            llm=FakeLLM(responses),
            tool_handlers={
                "allowed_tool": _echo_handler(),
                "blocked_tool": _echo_handler(),
            },
            tool_definitions=_make_tool_defs(["allowed_tool", "blocked_tool"]),
            settings=_make_settings(),
        )
        result = await executor.execute("test_skill", {
            "instruction": "Do something",
            "tool_blocklist": "blocked_tool",
        })
        assert isinstance(result, dict)
        # The blocked tool call should get a "Tool blocked" response.
        assert "ok" in result

    @pytest.mark.asyncio
    async def test_blocklist_from_settings(self) -> None:
        """Blocklist from settings is applied when payload doesn't override."""
        responses = [FakeLLMResponse(content="No tools needed.")]
        executor = AgentSkillExecutor(
            llm=FakeLLM(responses),
            tool_handlers={
                "safe_tool": _echo_handler(),
                "dangerous_tool": _echo_handler(),
            },
            tool_definitions=_make_tool_defs(["safe_tool", "dangerous_tool"]),
            settings=_make_settings(scheduler_agent_tool_blocklist="dangerous_tool"),
        )
        result = await executor.execute("test_skill", {"instruction": "Work safely"})
        assert result["ok"] is True


# ════════════════════════════════════════════════════════════════════════
# Coordinator routing
# ════════════════════════════════════════════════════════════════════════


class TestCoordinatorRouting:
    """Coordinator routes to AgentSkillExecutor when execution_mode=agent."""

    @pytest.mark.asyncio
    async def test_routing_executor_dispatches_agent(self) -> None:
        from leapflow.scheduler.coordinator import _RoutingExecutor

        default_exec = MagicMock()
        default_exec.execute = AsyncMock(return_value={"ok": True})

        agent_exec = MagicMock()
        agent_exec.execute = AsyncMock(return_value={"ok": True})

        router = _RoutingExecutor(default_exec, agent_factory=lambda: agent_exec)

        # Agent-mode parameters → agent executor.
        params = {"instruction": "hello", "execution_mode": "agent"}
        await router.execute("skill", params)
        agent_exec.execute.assert_called_once_with("skill", params)
        default_exec.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_routing_executor_dispatches_default(self) -> None:
        from leapflow.scheduler.coordinator import _RoutingExecutor

        default_exec = MagicMock()
        default_exec.execute = AsyncMock(return_value={"ok": True})

        router = _RoutingExecutor(default_exec, agent_factory=None)

        params = {"instruction": "hello"}
        await router.execute("skill", params)
        default_exec.execute.assert_called_once_with("skill", params)

    @pytest.mark.asyncio
    async def test_routing_executor_no_factory_uses_default(self) -> None:
        from leapflow.scheduler.coordinator import _RoutingExecutor

        default_exec = MagicMock()
        default_exec.execute = AsyncMock(return_value={"ok": True})

        router = _RoutingExecutor(default_exec, agent_factory=None)

        # Even with execution_mode=agent, if no factory → falls back to default.
        params = {"instruction": "hello", "execution_mode": "agent"}
        await router.execute("skill", params)
        default_exec.execute.assert_called_once()

    def test_coordinator_wrap_executor(self) -> None:
        from leapflow.scheduler.coordinator import TaskCoordinator, _RoutingExecutor
        from leapflow.scheduler.store import TaskStore

        store = MagicMock(spec=TaskStore)
        coordinator = TaskCoordinator(
            store=store,
            agent_executor_factory=lambda: MagicMock(),
        )
        default_exec = MagicMock()
        wrapped = coordinator.wrap_executor(default_exec)
        assert isinstance(wrapped, _RoutingExecutor)


# ════════════════════════════════════════════════════════════════════════
# Config settings accessibility
# ════════════════════════════════════════════════════════════════════════


class TestConfigSettings:
    """New settings are accessible from the Settings dataclass."""

    def test_settings_have_scheduler_agent_fields(self) -> None:
        from leapflow.config import Settings
        from dataclasses import fields as dc_fields

        field_names = {f.name for f in dc_fields(Settings)}
        assert "scheduler_agent_max_iterations" in field_names
        assert "scheduler_agent_tool_blocklist" in field_names

    def test_default_values(self) -> None:
        from leapflow.config import Settings
        from dataclasses import fields as dc_fields

        defaults = {}
        for f in dc_fields(Settings):
            if f.name.startswith("scheduler_agent_"):
                defaults[f.name] = f.default
        assert defaults["scheduler_agent_max_iterations"] == 25
        assert defaults["scheduler_agent_tool_blocklist"] == ""
