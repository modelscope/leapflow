# Copyright (c) Alibaba, Inc. and its affiliates.
"""Waterfall tool execution pipeline — composable interceptor chain.

Interceptors wrap tool execution with pre/post hooks, enabling pluggable
approval, audit, rate-limiting, timeout, caching, and transformation
without modifying engine dispatch logic.

Design:
    - Interceptors are sorted by priority (ascending: lower = earlier).
    - before() hooks run in priority order; a non-None return short-circuits.
    - after() hooks run in REVERSE priority order (innermost first).
    - Zero-interceptor pipeline has zero overhead (direct handler call).
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class _ToolExecutionDeadline:
    """Pauses one tool deadline while the tool awaits a human decision."""

    def __init__(self, loop: asyncio.AbstractEventLoop, timeout_s: float) -> None:
        self._loop = loop
        self._remaining_s = timeout_s
        self._scope: asyncio.Timeout | None = None
        self._pause_depth = 0

    def bind(self, scope: asyncio.Timeout) -> None:
        """Bind the active ``asyncio.timeout`` scope to this deadline."""
        self._scope = scope

    def pause(self) -> None:
        """Suspend deadline accounting until the matching resume call."""
        if self._pause_depth == 0:
            scope = self._scope
            if scope is None:
                return
            deadline = scope.when()
            self._remaining_s = max(0.0, (deadline or self._loop.time()) - self._loop.time())
            scope.reschedule(None)
        self._pause_depth += 1

    def resume(self) -> None:
        """Resume deadline accounting after the outermost human wait finishes."""
        if self._pause_depth == 0:
            return
        self._pause_depth -= 1
        if self._pause_depth == 0 and self._scope is not None:
            self._scope.reschedule(self._loop.time() + self._remaining_s)


_current_tool_execution_deadline: ContextVar[_ToolExecutionDeadline | None] = ContextVar(
    "leapflow_tool_execution_deadline", default=None,
)


@asynccontextmanager
async def pause_tool_execution_timeout_for_human_decision() -> AsyncIterator[None]:
    """Exclude human approval time from the active tool execution timeout.

    Outside an engine-managed tool invocation this is intentionally a no-op, so
    approval surfaces can use it without coupling themselves to a caller.
    """
    deadline = _current_tool_execution_deadline.get()
    if deadline is None:
        yield
        return
    deadline.pause()
    try:
        yield
    finally:
        deadline.resume()


async def run_tool_with_timeout(
    awaitable: Awaitable[Dict[str, Any]], timeout_s: float,
) -> Dict[str, Any]:
    """Run a tool with a deadline that pauses only for human decisions."""
    controller = _ToolExecutionDeadline(asyncio.get_running_loop(), timeout_s)
    token = _current_tool_execution_deadline.set(controller)
    try:
        async with asyncio.timeout(timeout_s) as scope:
            controller.bind(scope)
            return await awaitable
    finally:
        _current_tool_execution_deadline.reset(token)


@dataclass
class ToolCallContext:
    """Context passed through the waterfall pipeline.

    Carries tool identity, arguments, and extensible metadata/annotations
    that interceptors can read and write.
    """

    tool_name: str
    arguments: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    annotations: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ToolInterceptor(Protocol):
    """A composable interceptor in the tool execution waterfall.

    Implementations must provide:
    - name: unique identifier for registration/unregistration
    - priority: integer ordering (lower runs first in before(), last in after())
    - before(): pre-execution hook; return dict to short-circuit
    - after(): post-execution hook; may transform the result
    """

    @property
    def name(self) -> str:
        """Unique interceptor identifier."""
        ...

    @property
    def priority(self) -> int:
        """Ordering priority — lower values run earlier in before()."""
        ...

    async def before(self, context: ToolCallContext) -> Optional[Dict[str, Any]]:
        """Pre-execution hook.

        Return a dict to short-circuit (becomes the final result, skipping
        the handler and all lower-priority interceptors' before/after hooks).
        Return None to continue the pipeline.
        """
        ...

    async def after(self, context: ToolCallContext, result: Dict[str, Any]) -> Dict[str, Any]:
        """Post-execution hook. May inspect or transform the result."""
        ...


class ToolExecutionPipeline:
    """Ordered interceptor chain for tool execution.

    Interceptors are sorted by priority (ascending). The pipeline runs:
    1. All before() hooks in priority order (short-circuit on non-None return)
    2. The actual tool handler (the ``handler`` callable)
    3. All after() hooks in REVERSE priority order (innermost first)

    If no interceptors are registered, execute() calls the handler directly
    with zero overhead.
    """

    def __init__(self) -> None:
        self._interceptors: List[ToolInterceptor] = []

    @property
    def interceptor_count(self) -> int:
        """Number of registered interceptors."""
        return len(self._interceptors)

    def register(self, interceptor: ToolInterceptor) -> None:
        """Register an interceptor. Maintains sorted order by priority.

        Raises ValueError on duplicate name.
        """
        for existing in self._interceptors:
            if existing.name == interceptor.name:
                raise ValueError(f"Duplicate interceptor name: {interceptor.name!r}")
        self._interceptors.append(interceptor)
        self._interceptors.sort(key=lambda i: i.priority)

    def unregister(self, name: str) -> bool:
        """Remove an interceptor by name. Returns True if found and removed."""
        for idx, interceptor in enumerate(self._interceptors):
            if interceptor.name == name:
                self._interceptors.pop(idx)
                return True
        return False

    async def execute(
        self,
        context: ToolCallContext,
        handler: Callable[..., Awaitable[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Run the pipeline: before hooks → handler → after hooks.

        Args:
            context: The tool call context (tool_name, arguments, metadata).
            handler: The actual tool execution callable (async).

        Returns:
            The final result dict (possibly transformed by after hooks).

        Timeout: when ``context.annotations['timeout']`` is set (the engine
        passes the per-tool timeout there), the handler invocation receives an
        execution deadline. Human approval waits explicitly pause that deadline;
        a timeout still raises ``asyncio.TimeoutError`` so the caller's existing
        timeout handling stays authoritative. When no timeout is annotated the
        handler is called directly (no wrapping).
        """
        timeout = context.annotations.get("timeout")

        async def _run_handler() -> Dict[str, Any]:
            if timeout is not None:
                return await run_tool_with_timeout(handler(context), timeout)
            return await handler(context)

        if not self._interceptors:
            return await _run_handler()

        # Phase 1: before() hooks in priority order
        executed_before: List[ToolInterceptor] = []
        for interceptor in self._interceptors:
            short_circuit = await interceptor.before(context)
            if short_circuit is not None:
                # Short-circuit: run after() only for interceptors that
                # already ran their before() (excluding the one that short-circuited)
                result = short_circuit
                for prev in reversed(executed_before):
                    result = await prev.after(context, result)
                return result
            executed_before.append(interceptor)

        # Phase 2: execute the handler (timeout-wrapped when annotated)
        result = await _run_handler()

        # Phase 3: after() hooks in REVERSE priority order
        for interceptor in reversed(self._interceptors):
            result = await interceptor.after(context, result)

        return result


# ════════════════════════════════════════════════════════════════════════
# Built-in interceptor examples
# ════════════════════════════════════════════════════════════════════════


class AuditInterceptor:
    """Observability interceptor — logs tool invocations and results.

    Priority 100 (runs late in before, early in after) so it sees the
    final arguments and raw result before other transformations.
    """

    def __init__(self, *, log_level: int = logging.DEBUG) -> None:
        self._log_level = log_level
        self._log: List[Dict[str, Any]] = []  # in-memory audit trail

    @property
    def name(self) -> str:
        return "audit"

    @property
    def priority(self) -> int:
        return 100

    @property
    def log(self) -> List[Dict[str, Any]]:
        """Read-only access to the in-memory audit trail."""
        return list(self._log)

    async def before(self, context: ToolCallContext) -> Optional[Dict[str, Any]]:
        """Record tool invocation. Never short-circuits."""
        entry = {
            "phase": "before",
            "tool_name": context.tool_name,
            "arguments": context.arguments,
            "timestamp": time.time(),
        }
        self._log.append(entry)
        logger.log(
            self._log_level,
            "Audit: invoking %s with %d argument(s)",
            context.tool_name,
            len(context.arguments),
        )
        return None

    async def after(self, context: ToolCallContext, result: Dict[str, Any]) -> Dict[str, Any]:
        """Record tool result. Passes through unmodified."""
        entry = {
            "phase": "after",
            "tool_name": context.tool_name,
            "result_keys": list(result.keys()) if isinstance(result, dict) else None,
            "timestamp": time.time(),
        }
        self._log.append(entry)
        logger.log(
            self._log_level,
            "Audit: %s completed",
            context.tool_name,
        )
        return result


class TimeoutInterceptor:
    """Wraps handler execution with asyncio.wait_for timeout.

    Priority 10 (runs early in before, late in after) so the timeout
    encompasses all lower-priority interceptors' after-hooks as well.
    Note: the timeout is applied by wrapping the handler, not by modifying
    the pipeline flow. The before() hook stores the timeout; execute() applies it.
    """

    def __init__(self, default_timeout: float = 30.0) -> None:
        self._default_timeout = default_timeout

    @property
    def name(self) -> str:
        return "timeout"

    @property
    def priority(self) -> int:
        return 10

    async def before(self, context: ToolCallContext) -> Optional[Dict[str, Any]]:
        """Annotate context with timeout value. Never short-circuits."""
        timeout = context.metadata.get("timeout", self._default_timeout)
        context.annotations["_timeout"] = timeout
        return None

    async def after(self, context: ToolCallContext, result: Dict[str, Any]) -> Dict[str, Any]:
        """Pass-through (timeout is enforced at handler level)."""
        return result


class TimeoutPipelineWrapper:
    """A pipeline variant that applies TimeoutInterceptor's annotation.

    Use this to wrap the handler callable with asyncio.wait_for using
    the timeout stored in context.annotations by TimeoutInterceptor.
    """

    @staticmethod
    def wrap_handler(
        handler: Callable[..., Awaitable[Dict[str, Any]]],
        context: ToolCallContext,
    ) -> Callable[..., Awaitable[Dict[str, Any]]]:
        """Return a handler wrapped with timeout if annotated."""
        timeout = context.annotations.get("_timeout")
        if timeout is None:
            return handler

        async def _timed(ctx: ToolCallContext) -> Dict[str, Any]:
            try:
                return await asyncio.wait_for(handler(ctx), timeout=timeout)
            except asyncio.TimeoutError:
                return {
                    "error": f"Tool '{ctx.tool_name}' timed out after {timeout}s",
                    "timed_out": True,
                }

        return _timed
