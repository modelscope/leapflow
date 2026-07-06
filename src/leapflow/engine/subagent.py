"""Subagent isolation — delegated task execution with restricted context.

Design (inspired by hermes delegate_tool):
- Each subagent gets a fresh execution context (no parent message history)
- Tool restriction: blocked tools list + configurable enabled toolsets
- Memory isolation: no parent memory access, optional working memory only
- Session lineage: child session linked to parent via parent_session_id
- Summary budget: only summary flows back to parent (not full transcript)
- Recursion depth limit to prevent runaway delegation trees

Fits leapflow's architecture:
- Extends TaskScheduler with isolated execution contexts
- Emits SubagentCompleted/SubagentFailed events on EventBus
- Uses existing SkillRegistry with tool intersection
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_MAX_SPAWN_DEPTH = 2
_MAX_CONCURRENT_CHILDREN = 3
_SUMMARY_MAX_CHARS = 4000

DELEGATE_BLOCKED_TOOLS: FrozenSet[str] = frozenset({
    "delegate_task", "gp_delegate_task",
    "memory_write", "gp_memory_write",
    "send_message", "gp_send_message",
    "clarify", "gp_clarify",
})


@dataclass(frozen=True)
class SubagentConfig:
    """Configuration for a subagent execution context."""
    goal: str
    context: str = ""
    parent_session_id: Optional[str] = None
    allowed_tools: Optional[FrozenSet[str]] = None
    blocked_tools: FrozenSet[str] = DELEGATE_BLOCKED_TOOLS
    max_iterations: int = 15
    summary_max_chars: int = _SUMMARY_MAX_CHARS
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubagentResult:
    """Result from a subagent execution."""
    session_id: str
    goal: str
    summary: str
    status: str  # "completed" | "failed" | "cancelled"
    elapsed_s: float = 0.0
    tool_calls: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SubagentExecutor(Protocol):
    """Protocol for executing a subagent run (DIP)."""

    async def execute_subagent(self, config: SubagentConfig) -> SubagentResult:
        """Run a subagent with isolated context. Returns summary result."""
        ...


class SubagentManager:
    """Manages subagent lifecycle with isolation guarantees.

    Responsibilities:
    - Enforce depth limits and concurrent child limits
    - Create isolated execution contexts
    - Manage session lineage
    - Trim and summarize results for parent consumption
    """

    def __init__(
        self,
        *,
        executor: Optional[SubagentExecutor] = None,
        max_depth: int = _MAX_SPAWN_DEPTH,
        max_concurrent: int = _MAX_CONCURRENT_CHILDREN,
        on_complete: Optional[Callable[[SubagentResult], None]] = None,
    ) -> None:
        self._executor = executor
        self._max_depth = max_depth
        self._max_concurrent = max_concurrent
        self._on_complete = on_complete
        self._active: Dict[str, asyncio.Task[SubagentResult]] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def delegate(self, config: SubagentConfig) -> SubagentResult:
        """Delegate a task to a subagent with isolation.

        Enforces:
        - Depth limit
        - Concurrent child limit
        - Tool blocking
        - Summary truncation
        """
        if config.depth >= self._max_depth:
            return SubagentResult(
                session_id="",
                goal=config.goal,
                summary=f"Delegation depth limit ({self._max_depth}) reached.",
                status="failed",
                error="max_depth_exceeded",
            )

        if self._executor is None:
            return SubagentResult(
                session_id="",
                goal=config.goal,
                summary="Subagent executor not configured.",
                status="failed",
                error="no_executor",
            )

        session_id = f"sub_{uuid.uuid4().hex[:12]}"

        async with self._semaphore:
            t0 = time.monotonic()
            try:
                result = await self._executor.execute_subagent(config)
                result = self._trim_summary(result)
            except asyncio.CancelledError:
                result = SubagentResult(
                    session_id=session_id,
                    goal=config.goal,
                    summary="Subagent execution was cancelled.",
                    status="cancelled",
                    elapsed_s=time.monotonic() - t0,
                )
            except Exception as e:
                result = SubagentResult(
                    session_id=session_id,
                    goal=config.goal,
                    summary=f"Subagent failed: {e}",
                    status="failed",
                    elapsed_s=time.monotonic() - t0,
                    error=str(e),
                )

            if self._on_complete:
                try:
                    self._on_complete(result)
                except Exception as cb_err:
                    logger.debug("subagent.on_complete callback error: %s", cb_err)

            return result

    async def delegate_batch(
        self, configs: List[SubagentConfig]
    ) -> List[SubagentResult]:
        """Delegate multiple tasks concurrently (bounded by semaphore)."""
        tasks = [self.delegate(config) for config in configs]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    def cancel_all(self) -> int:
        """Cancel all active subagent tasks. Returns count cancelled."""
        cancelled = 0
        for task in self._active.values():
            if not task.done():
                task.cancel()
                cancelled += 1
        return cancelled

    def _trim_summary(self, result: SubagentResult) -> SubagentResult:
        """Ensure summary fits within parent's budget."""
        if len(result.summary) > _SUMMARY_MAX_CHARS:
            trimmed = result.summary[:_SUMMARY_MAX_CHARS - 50]
            trimmed += f"\n\n[... trimmed {len(result.summary) - _SUMMARY_MAX_CHARS + 50} chars]"
            return SubagentResult(
                session_id=result.session_id,
                goal=result.goal,
                summary=trimmed,
                status=result.status,
                elapsed_s=result.elapsed_s,
                tool_calls=result.tool_calls,
                error=result.error,
                metadata=result.metadata,
            )
        return result


def build_subagent_tool_filter(
    parent_tools: List[str],
    config: SubagentConfig,
) -> List[str]:
    """Compute the effective tool list for a subagent.

    Intersection of parent tools minus blocked tools, optionally filtered by allowed_tools.
    """
    available = set(parent_tools) - config.blocked_tools

    if config.allowed_tools is not None:
        available = available & config.allowed_tools

    if config.depth >= _MAX_SPAWN_DEPTH - 1:
        available -= {"delegate_task", "gp_delegate_task"}

    return sorted(available)
