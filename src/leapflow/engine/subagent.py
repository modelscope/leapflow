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
import contextvars
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_MAX_SPAWN_DEPTH = 2
_MAX_CONCURRENT_CHILDREN = 3
_SUMMARY_MAX_CHARS = 4000

# Depth of the subagent frame currently executing, propagated across the await
# chain so a nested delegate_task can compute its child's depth. 0 = top level.
#
# Safe by construction: set() and reset() (in SubagentManager.delegate below)
# execute inside the same call with no Task boundary in between --
# execute_subagent() (including EngineFrameSubagentExecutor's full-loop path,
# engine.py::_run_child_frame) resolves to a single value without ever
# yielding back through the parent engine's run_stream() generator. It
# therefore never crosses a per-chunk asyncio.create_task() boundary the way
# the daemon's leapd_approval_route once did (see the contract note on that
# ContextVar in daemon/service.py). If a future executor lets a subagent's
# progress stream back out through run_stream() before it completes, re-verify
# this invariant and, if violated, pin a shared contextvars.Context the same
# way server.py::_dispatch_stream does.
_current_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "leapflow_subagent_depth", default=0
)


def current_subagent_depth() -> int:
    """Depth of the subagent frame currently executing (0 = top-level turn)."""
    return _current_depth.get()


# delegate_task is intentionally NOT blocked here: recursion is gated by depth in
# build_subagent_tool_filter + SubagentManager (a child is only offered/allowed
# while it stays within max_depth). Blocking it outright would disable recursion.
DELEGATE_BLOCKED_TOOLS: FrozenSet[str] = frozenset({
    "memory_write", "gp_memory_write",
    "send_message", "gp_send_message",
    "clarify", "gp_clarify",
    "research_note", "gp_research_note",
    "schedule_reentry", "gp_schedule_reentry",
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
            depth_token = _current_depth.set(config.depth)
            try:
                result = await self._executor.execute_subagent(config)
                result = self._trim_summary(result, config.summary_max_chars)
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
            finally:
                _current_depth.reset(depth_token)

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

    def _trim_summary(self, result: SubagentResult, max_chars: int = _SUMMARY_MAX_CHARS) -> SubagentResult:
        """Ensure summary fits within parent's budget."""
        if len(result.summary) > max_chars:
            trimmed = result.summary[:max_chars - 50]
            trimmed += f"\n\n[... trimmed {len(result.summary) - max_chars + 50} chars]"
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


class DefaultSubagentExecutor:
    """Concrete SubagentExecutor that runs a lightweight tool loop in isolation.

    Creates a fresh message context with restricted tools and runs the
    standard LLM→tool loop until goal completion or budget exhaustion.
    """

    def __init__(
        self,
        *,
        llm: Any,
        tool_handlers: Dict[str, Any],
        tool_definitions: list,
        settings: Any = None,
    ) -> None:
        self._llm = llm
        self._tool_handlers = tool_handlers
        self._tool_definitions = tool_definitions
        self._settings = settings

    async def execute_subagent(self, config: SubagentConfig) -> SubagentResult:
        """Run isolated subagent with restricted tool access."""
        session_id = f"sub_{uuid.uuid4().hex[:12]}"
        t0 = time.monotonic()

        available_tools = build_subagent_tool_filter(
            list(self._tool_handlers.keys()), config,
            max_depth=(
                getattr(self._settings, "agent_subagent_max_depth", _MAX_SPAWN_DEPTH)
                if self._settings is not None else _MAX_SPAWN_DEPTH
            ),
        )
        filtered_handlers = {
            name: self._tool_handlers[name]
            for name in available_tools
            if name in self._tool_handlers
        }
        filtered_definitions = [
            td for td in self._tool_definitions
            if td.get("function", {}).get("name", "") in available_tools
        ]

        system_prompt = (
            f"You are a focused subagent. Complete this task:\n{config.goal}\n"
        )
        if config.context:
            system_prompt += f"\nContext:\n{config.context}\n"
        system_prompt += "\nProvide a clear, complete answer when done."

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": config.goal},
        ]

        tools_kwarg: dict[str, Any] = {}
        if filtered_definitions:
            tools_kwarg["tools"] = filtered_definitions

        content = ""
        tool_call_count = 0
        result_budget = (
            self._settings.max_tool_result_chars
            if self._settings is not None
            else _SUMMARY_MAX_CHARS
        )

        # Adaptive depth: a fresh elastic budget widened by an independent
        # difficulty signal, so a hard sub-task earns more iterations while a
        # simple one stays short (reuses the W1 budget + governance components).
        from leapflow.engine.budget import BudgetConfig, BudgetStatus, IterationBudget
        from leapflow.engine.context_control import (
            ContextGovernanceController,
            ToolEvidenceBuilder,
        )

        floor = config.max_iterations
        if self._settings is not None:
            cfg_iters = getattr(self._settings, "agent_subagent_max_iterations", 0)
            if cfg_iters > 0:
                floor = cfg_iters
        budget = IterationBudget.for_react(
            BudgetConfig(max_iterations=floor, iter_ceiling=floor * 2)
        )
        governance = ContextGovernanceController(
            evidence_builder=ToolEvidenceBuilder(max_content_chars=result_budget),
        )
        round_no = 0

        import json as _json_sub
        while True:
            if budget.consume() == BudgetStatus.EXHAUSTED:
                break
            round_no += 1
            try:
                resp = await self._llm.achat(
                    messages, stream=False, enable_thinking=False,
                    **tools_kwarg,
                )
            except Exception as exc:
                return SubagentResult(
                    session_id=session_id, goal=config.goal,
                    summary=f"LLM error: {exc}",
                    status="failed", elapsed_s=time.monotonic() - t0,
                    tool_calls=tool_call_count, error=str(exc),
                )

            content = (resp.content or "").strip()
            native_calls = getattr(resp, "tool_calls", None) or []

            if not native_calls:
                break

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.name, "arguments": _json_sub.dumps(tc.arguments, ensure_ascii=False)},
                }
                for tc in native_calls
            ]
            messages.append(assistant_msg)

            for tc in native_calls:
                handler = filtered_handlers.get(tc.name)
                if handler is None:
                    result_text = _json_sub.dumps({"ok": False, "error": f"Tool blocked: {tc.name}"})
                else:
                    try:
                        result = await handler(tc.arguments)
                        governance.compact_tool_result(tc.name, tc.arguments, result)
                        result_text = _json_sub.dumps(result, default=str, ensure_ascii=False)
                    except Exception as e:
                        result_text = _json_sub.dumps({"ok": False, "error": str(e)})
                result_text = result_text[:result_budget]
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})
                tool_call_count += 1

            # widen the frame's budget toward its difficulty (bounded by ceiling)
            difficulty = governance.snapshot(round_number=round_no).difficulty
            budget.retarget(budget.elastic_max(difficulty))

        return SubagentResult(
            session_id=session_id,
            goal=config.goal,
            summary=content[:config.summary_max_chars] or "(no output)",
            status="completed",
            elapsed_s=time.monotonic() - t0,
            tool_calls=tool_call_count,
        )


def build_subagent_tool_filter(
    parent_tools: List[str],
    config: SubagentConfig,
    *,
    max_depth: int = _MAX_SPAWN_DEPTH,
) -> List[str]:
    """Compute the effective tool list for a subagent.

    Intersection of parent tools minus blocked tools, optionally filtered by
    allowed_tools. delegate_task is offered only while a child would stay within
    the depth budget (child_depth = config.depth + 1 < max_depth).
    """
    available = set(parent_tools) - config.blocked_tools

    if config.allowed_tools is not None:
        available = available & config.allowed_tools

    if config.depth + 1 >= max_depth:
        available -= {"delegate_task", "gp_delegate_task"}

    return sorted(available)


class EngineFrameSubagentExecutor:
    """SubagentExecutor that runs the engine's full adaptive OODA loop on an
    isolated child frame (opt-in via ``agent.subagent_full_loop``).

    Where :class:`DefaultSubagentExecutor` runs a deliberately lightweight loop,
    this delegates to the engine's own ``_run_child_frame`` so the subagent gains
    the full loop (progressive disclosure, compression, recovery, research
    ledger) while staying state-isolated via the engine's per-frame state swap.
    Tool access is still restricted by :func:`build_subagent_tool_filter`, and
    recursion depth stays gated by the shared ``_current_depth`` contract.
    """

    def __init__(
        self,
        *,
        run_child: Callable[..., Any],
        tool_names: List[str],
        settings: Any = None,
    ) -> None:
        self._run_child = run_child
        self._tool_names = list(tool_names)
        self._settings = settings

    async def execute_subagent(self, config: SubagentConfig) -> SubagentResult:
        """Run an isolated subagent via the engine's full loop."""
        session_id = f"sub_{uuid.uuid4().hex[:12]}"
        t0 = time.monotonic()
        max_depth = (
            getattr(self._settings, "agent_subagent_max_depth", _MAX_SPAWN_DEPTH)
            if self._settings is not None else _MAX_SPAWN_DEPTH
        )
        available = build_subagent_tool_filter(self._tool_names, config, max_depth=max_depth)
        goal = config.goal
        if config.context:
            goal = f"{config.goal}\n\nContext:\n{config.context}"
        try:
            summary = await self._run_child(
                goal,
                depth=config.depth,
                tool_filter=frozenset(available),
                enable_thinking=False,
            )
        except Exception as exc:  # isolate subagent failure from the parent loop
            return SubagentResult(
                session_id=session_id, goal=config.goal,
                summary=f"Subagent error: {exc}", status="failed",
                elapsed_s=time.monotonic() - t0, error=str(exc),
            )
        return SubagentResult(
            session_id=session_id, goal=config.goal,
            summary=(summary or "")[:config.summary_max_chars],
            status="completed", elapsed_s=time.monotonic() - t0,
        )
