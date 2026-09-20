# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for the Waterfall Tool Execution Pipeline."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import pytest

from leapflow.domain.tool_pipeline import (
    AuditInterceptor,
    TimeoutInterceptor,
    TimeoutPipelineWrapper,
    ToolCallContext,
    ToolExecutionPipeline,
    ToolInterceptor,
    pause_tool_execution_timeout_for_human_decision,
    run_tool_with_timeout,
)


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════


class RecordingInterceptor:
    """Test interceptor that records invocation order."""

    def __init__(self, name: str, priority: int, *, short_circuit: Optional[Dict[str, Any]] = None) -> None:
        self._name = name
        self._priority = priority
        self._short_circuit = short_circuit
        self.before_calls: list[str] = []
        self.after_calls: list[tuple[str, Dict[str, Any]]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    async def before(self, context: ToolCallContext) -> Optional[Dict[str, Any]]:
        self.before_calls.append(context.tool_name)
        return self._short_circuit

    async def after(self, context: ToolCallContext, result: Dict[str, Any]) -> Dict[str, Any]:
        self.after_calls.append((context.tool_name, result))
        return result


class TransformInterceptor:
    """Test interceptor that transforms the result in after()."""

    def __init__(self, name: str, priority: int, transform_key: str, transform_value: Any) -> None:
        self._name = name
        self._priority = priority
        self._key = transform_key
        self._value = transform_value

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    async def before(self, context: ToolCallContext) -> Optional[Dict[str, Any]]:
        return None

    async def after(self, context: ToolCallContext, result: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(result)
        result[self._key] = self._value
        return result


async def echo_handler(context: ToolCallContext) -> Dict[str, Any]:
    """Simple handler that echoes the tool name and arguments."""
    return {"tool": context.tool_name, "args": context.arguments}


async def slow_handler(context: ToolCallContext) -> Dict[str, Any]:
    """Handler that sleeps for a configurable time."""
    delay = context.arguments.get("delay", 10.0)
    await asyncio.sleep(delay)
    return {"completed": True}


# ════════════════════════════════════════════════════════════════
# ToolCallContext tests
# ════════════════════════════════════════════════════════════════


class TestToolCallContext:
    """ToolCallContext construction and field access."""

    def test_context_basic_construction(self) -> None:
        ctx = ToolCallContext(tool_name="read_file", arguments={"path": "/tmp/test"})
        assert ctx.tool_name == "read_file"
        assert ctx.arguments == {"path": "/tmp/test"}
        assert ctx.metadata == {}
        assert ctx.annotations == {}

    def test_context_with_metadata(self) -> None:
        ctx = ToolCallContext(
            tool_name="write_file",
            arguments={"content": "hello"},
            metadata={"category": "file_ops", "risk_level": "high"},
        )
        assert ctx.metadata["category"] == "file_ops"

    def test_context_annotations_mutable(self) -> None:
        ctx = ToolCallContext(tool_name="test", arguments={})
        ctx.annotations["custom"] = 42
        assert ctx.annotations["custom"] == 42


# ════════════════════════════════════════════════════════════════
# ToolExecutionPipeline: basic execution
# ════════════════════════════════════════════════════════════════


class TestPipelineEmptyDirect:
    """Empty pipeline = direct handler call with zero overhead."""

    @pytest.mark.asyncio
    async def test_empty_pipeline_calls_handler_directly(self) -> None:
        pipeline = ToolExecutionPipeline()
        ctx = ToolCallContext(tool_name="echo", arguments={"x": 1})
        result = await pipeline.execute(ctx, echo_handler)
        assert result == {"tool": "echo", "args": {"x": 1}}

    @pytest.mark.asyncio
    async def test_empty_pipeline_interceptor_count_zero(self) -> None:
        pipeline = ToolExecutionPipeline()
        assert pipeline.interceptor_count == 0


# ════════════════════════════════════════════════════════════════
# ToolExecutionPipeline: interceptor ordering
# ════════════════════════════════════════════════════════════════


class TestPipelineOrdering:
    """Interceptors execute in priority order (before) / reverse (after)."""

    @pytest.mark.asyncio
    async def test_before_hooks_run_in_priority_order(self) -> None:
        pipeline = ToolExecutionPipeline()
        order: list[str] = []

        class OrderedInterceptor:
            def __init__(self, n: str, p: int) -> None:
                self._n, self._p = n, p

            @property
            def name(self) -> str:
                return self._n

            @property
            def priority(self) -> int:
                return self._p

            async def before(self, context: ToolCallContext) -> Optional[Dict[str, Any]]:
                order.append(f"before-{self._n}")
                return None

            async def after(self, context: ToolCallContext, result: Dict[str, Any]) -> Dict[str, Any]:
                order.append(f"after-{self._n}")
                return result

        pipeline.register(OrderedInterceptor("C", 30))
        pipeline.register(OrderedInterceptor("A", 10))
        pipeline.register(OrderedInterceptor("B", 20))

        ctx = ToolCallContext(tool_name="test", arguments={})
        await pipeline.execute(ctx, echo_handler)

        assert order == [
            "before-A", "before-B", "before-C",  # ascending priority
            "after-C", "after-B", "after-A",      # reverse priority
        ]

    @pytest.mark.asyncio
    async def test_after_hooks_run_in_reverse_priority(self) -> None:
        """after() hooks must run in REVERSE priority (highest priority first)."""
        pipeline = ToolExecutionPipeline()
        after_order: list[str] = []

        class AfterRecorder:
            def __init__(self, n: str, p: int) -> None:
                self._n, self._p = n, p

            @property
            def name(self) -> str:
                return self._n

            @property
            def priority(self) -> int:
                return self._p

            async def before(self, context: ToolCallContext) -> Optional[Dict[str, Any]]:
                return None

            async def after(self, context: ToolCallContext, result: Dict[str, Any]) -> Dict[str, Any]:
                after_order.append(self._n)
                return result

        pipeline.register(AfterRecorder("low", 5))
        pipeline.register(AfterRecorder("mid", 50))
        pipeline.register(AfterRecorder("high", 100))

        ctx = ToolCallContext(tool_name="test", arguments={})
        await pipeline.execute(ctx, echo_handler)

        # Reverse priority: high (100) → mid (50) → low (5)
        assert after_order == ["high", "mid", "low"]


# ════════════════════════════════════════════════════════════════
# ToolExecutionPipeline: short-circuit
# ════════════════════════════════════════════════════════════════


class TestPipelineShortCircuit:
    """before() returning non-None short-circuits execution."""

    @pytest.mark.asyncio
    async def test_short_circuit_skips_handler(self) -> None:
        pipeline = ToolExecutionPipeline()
        handler_called = [False]

        async def tracking_handler(ctx: ToolCallContext) -> Dict[str, Any]:
            handler_called[0] = True
            return {"handler": "ran"}

        blocker = RecordingInterceptor("blocker", priority=10, short_circuit={"blocked": True})
        pipeline.register(blocker)

        ctx = ToolCallContext(tool_name="test", arguments={})
        result = await pipeline.execute(ctx, tracking_handler)

        assert result == {"blocked": True}
        assert not handler_called[0]

    @pytest.mark.asyncio
    async def test_short_circuit_skips_lower_priority_before(self) -> None:
        """Only interceptors with before() already called get after()."""
        pipeline = ToolExecutionPipeline()

        first = RecordingInterceptor("first", priority=5)
        blocker = RecordingInterceptor("blocker", priority=10, short_circuit={"stopped": True})
        skipped = RecordingInterceptor("skipped", priority=20)

        pipeline.register(first)
        pipeline.register(blocker)
        pipeline.register(skipped)

        ctx = ToolCallContext(tool_name="test", arguments={})
        result = await pipeline.execute(ctx, echo_handler)

        assert result == {"stopped": True}
        assert len(first.before_calls) == 1
        assert len(blocker.before_calls) == 1
        assert len(skipped.before_calls) == 0  # never reached
        # first ran before() so it gets after()
        assert len(first.after_calls) == 1
        # blocker short-circuited — no after()
        assert len(blocker.after_calls) == 0
        # skipped never ran
        assert len(skipped.after_calls) == 0


# ════════════════════════════════════════════════════════════════
# ToolExecutionPipeline: registration/unregistration
# ════════════════════════════════════════════════════════════════


class TestPipelineRegistration:
    """Register and unregister interceptors."""

    def test_register_increases_count(self) -> None:
        pipeline = ToolExecutionPipeline()
        i = RecordingInterceptor("test", 10)
        pipeline.register(i)
        assert pipeline.interceptor_count == 1

    def test_register_duplicate_raises(self) -> None:
        pipeline = ToolExecutionPipeline()
        i1 = RecordingInterceptor("dup", 10)
        i2 = RecordingInterceptor("dup", 20)
        pipeline.register(i1)
        with pytest.raises(ValueError, match="Duplicate"):
            pipeline.register(i2)

    def test_unregister_removes_interceptor(self) -> None:
        pipeline = ToolExecutionPipeline()
        i = RecordingInterceptor("removable", 10)
        pipeline.register(i)
        assert pipeline.unregister("removable") is True
        assert pipeline.interceptor_count == 0

    def test_unregister_nonexistent_returns_false(self) -> None:
        pipeline = ToolExecutionPipeline()
        assert pipeline.unregister("ghost") is False

    @pytest.mark.asyncio
    async def test_unregistered_interceptor_not_called(self) -> None:
        pipeline = ToolExecutionPipeline()
        i = RecordingInterceptor("temp", 10)
        pipeline.register(i)
        pipeline.unregister("temp")

        ctx = ToolCallContext(tool_name="test", arguments={})
        await pipeline.execute(ctx, echo_handler)
        assert len(i.before_calls) == 0


# ════════════════════════════════════════════════════════════════
# ToolExecutionPipeline: result transformation
# ════════════════════════════════════════════════════════════════


class TestPipelineTransformation:
    """after() hooks can transform results."""

    @pytest.mark.asyncio
    async def test_after_transforms_result(self) -> None:
        pipeline = ToolExecutionPipeline()
        pipeline.register(TransformInterceptor("add_x", 10, "x", 42))
        pipeline.register(TransformInterceptor("add_y", 20, "y", "hello"))

        ctx = ToolCallContext(tool_name="echo", arguments={"a": 1})
        result = await pipeline.execute(ctx, echo_handler)

        # Handler produces {"tool": "echo", "args": {"a": 1}}
        # after runs in reverse: priority 20 first, then 10
        assert result["tool"] == "echo"
        assert result["x"] == 42
        assert result["y"] == "hello"


# ════════════════════════════════════════════════════════════════
# ToolInterceptor Protocol compliance
# ════════════════════════════════════════════════════════════════


class TestInterceptorProtocol:
    """ToolInterceptor is a runtime_checkable Protocol."""

    def test_recording_interceptor_satisfies_protocol(self) -> None:
        i = RecordingInterceptor("test", 10)
        assert isinstance(i, ToolInterceptor)

    def test_audit_interceptor_satisfies_protocol(self) -> None:
        i = AuditInterceptor()
        assert isinstance(i, ToolInterceptor)

    def test_timeout_interceptor_satisfies_protocol(self) -> None:
        i = TimeoutInterceptor()
        assert isinstance(i, ToolInterceptor)


# ════════════════════════════════════════════════════════════════
# AuditInterceptor tests
# ════════════════════════════════════════════════════════════════


class TestAuditInterceptor:
    """AuditInterceptor logs tool invocations and results."""

    @pytest.mark.asyncio
    async def test_audit_records_before_and_after(self) -> None:
        pipeline = ToolExecutionPipeline()
        audit = AuditInterceptor()
        pipeline.register(audit)

        ctx = ToolCallContext(tool_name="read_file", arguments={"path": "/tmp/x"})
        await pipeline.execute(ctx, echo_handler)

        log = audit.log
        assert len(log) == 2
        assert log[0]["phase"] == "before"
        assert log[0]["tool_name"] == "read_file"
        assert log[0]["arguments"] == {"path": "/tmp/x"}
        assert log[1]["phase"] == "after"
        assert log[1]["tool_name"] == "read_file"

    @pytest.mark.asyncio
    async def test_audit_never_short_circuits(self) -> None:
        pipeline = ToolExecutionPipeline()
        audit = AuditInterceptor()
        pipeline.register(audit)

        ctx = ToolCallContext(tool_name="test", arguments={})
        result = await pipeline.execute(ctx, echo_handler)
        # Handler still runs
        assert result["tool"] == "test"

    @pytest.mark.asyncio
    async def test_audit_does_not_modify_result(self) -> None:
        pipeline = ToolExecutionPipeline()
        audit = AuditInterceptor()
        pipeline.register(audit)

        ctx = ToolCallContext(tool_name="test", arguments={"k": "v"})
        result = await pipeline.execute(ctx, echo_handler)
        assert result == {"tool": "test", "args": {"k": "v"}}


# ════════════════════════════════════════════════════════════════
# TimeoutInterceptor tests
# ════════════════════════════════════════════════════════════════


class TestTimeoutInterceptor:
    """TimeoutInterceptor annotates context with timeout."""

    @pytest.mark.asyncio
    async def test_timeout_annotates_context(self) -> None:
        pipeline = ToolExecutionPipeline()
        timeout = TimeoutInterceptor(default_timeout=15.0)
        pipeline.register(timeout)

        ctx = ToolCallContext(tool_name="test", arguments={})
        await pipeline.execute(ctx, echo_handler)
        assert ctx.annotations["_timeout"] == 15.0

    @pytest.mark.asyncio
    async def test_timeout_uses_metadata_override(self) -> None:
        pipeline = ToolExecutionPipeline()
        timeout = TimeoutInterceptor(default_timeout=30.0)
        pipeline.register(timeout)

        ctx = ToolCallContext(tool_name="test", arguments={}, metadata={"timeout": 5.0})
        await pipeline.execute(ctx, echo_handler)
        assert ctx.annotations["_timeout"] == 5.0

    @pytest.mark.asyncio
    async def test_human_decision_pause_excludes_wait_from_execution_timeout(self) -> None:
        waiting = asyncio.Event()
        answered = asyncio.Event()
        context = ToolCallContext(tool_name="write_file", arguments={})

        async def handler(_: ToolCallContext) -> Dict[str, Any]:
            async with pause_tool_execution_timeout_for_human_decision():
                waiting.set()
                await answered.wait()
            await asyncio.sleep(0.005)
            return {"completed": True}

        task = asyncio.create_task(run_tool_with_timeout(handler(context), 0.02))
        await asyncio.wait_for(waiting.wait(), timeout=1.0)
        await asyncio.sleep(0.05)
        assert not task.done(), "human decision time must not consume the tool deadline"

        answered.set()
        assert await asyncio.wait_for(task, timeout=1.0) == {"completed": True}

    @pytest.mark.asyncio
    async def test_execution_still_times_out_after_human_decision_pause(self) -> None:
        waiting = asyncio.Event()
        answered = asyncio.Event()
        context = ToolCallContext(tool_name="write_file", arguments={})

        async def handler(_: ToolCallContext) -> Dict[str, Any]:
            async with pause_tool_execution_timeout_for_human_decision():
                waiting.set()
                await answered.wait()
            await asyncio.sleep(0.05)
            return {"completed": True}

        task = asyncio.create_task(run_tool_with_timeout(handler(context), 0.02))
        await asyncio.wait_for(waiting.wait(), timeout=1.0)
        await asyncio.sleep(0.05)
        assert not task.done(), "approval wait must remain unbounded"

        answered.set()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_timeout_wrapper_times_out(self) -> None:
        ctx = ToolCallContext(tool_name="slow", arguments={"delay": 10.0})
        ctx.annotations["_timeout"] = 0.05  # 50ms

        wrapped = TimeoutPipelineWrapper.wrap_handler(slow_handler, ctx)
        result = await wrapped(ctx)
        assert result["timed_out"] is True
        assert "error" in result

    @pytest.mark.asyncio
    async def test_timeout_wrapper_passes_on_fast_handler(self) -> None:
        ctx = ToolCallContext(tool_name="fast", arguments={"delay": 0.001})
        ctx.annotations["_timeout"] = 5.0

        wrapped = TimeoutPipelineWrapper.wrap_handler(slow_handler, ctx)
        result = await wrapped(ctx)
        assert result == {"completed": True}

    @pytest.mark.asyncio
    async def test_timeout_wrapper_no_annotation_returns_original(self) -> None:
        ctx = ToolCallContext(tool_name="test", arguments={})
        # No _timeout annotation
        wrapped = TimeoutPipelineWrapper.wrap_handler(echo_handler, ctx)
        assert wrapped is echo_handler  # same reference — no wrapping


# ════════════════════════════════════════════════════════════════
# ToolPluginRegistry integration
# ════════════════════════════════════════════════════════════════


class TestRegistryPipelineIntegration:
    """ToolPluginRegistry exposes a tool_pipeline property."""

    def test_registry_has_pipeline(self) -> None:
        from leapflow.plugins.registry import ToolPluginRegistry

        registry = ToolPluginRegistry()
        pipeline = registry.tool_pipeline
        assert isinstance(pipeline, ToolExecutionPipeline)
        assert pipeline.interceptor_count == 0

    def test_registry_pipeline_is_same_instance(self) -> None:
        from leapflow.plugins.registry import ToolPluginRegistry

        registry = ToolPluginRegistry()
        assert registry.tool_pipeline is registry.tool_pipeline

    @pytest.mark.asyncio
    async def test_registry_pipeline_accepts_interceptors(self) -> None:
        from leapflow.plugins.registry import ToolPluginRegistry

        registry = ToolPluginRegistry()
        audit = AuditInterceptor()
        registry.tool_pipeline.register(audit)
        assert registry.tool_pipeline.interceptor_count == 1

        ctx = ToolCallContext(tool_name="test", arguments={})
        result = await registry.tool_pipeline.execute(ctx, echo_handler)
        assert result["tool"] == "test"
        assert len(audit.log) == 2
