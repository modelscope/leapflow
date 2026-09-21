# Copyright (c) Alibaba, Inc. and its affiliates.
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
import collections
import contextvars
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Deque, Dict, FrozenSet, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_MAX_SPAWN_DEPTH = 2
_MAX_CONCURRENT_CHILDREN = 3
_SUMMARY_MAX_CHARS = 4000

# Depth of the subagent frame currently executing, propagated across the await
# chain so a nested delegate_task can compute its child's depth. 0 = top level.
#
# The ContextVar is set in SubagentManager.delegate() BEFORE the executor
# coroutine is wrapped in asyncio.create_task().  Python copies the current
# contextvars.Context at task-creation time, so the child task (and anything
# the executor await-chains into, including EngineFrameSubagentExecutor's
# engine.py::_run_child_frame) sees the correct depth.  The parent resets its
# own token in the finally block, which is safe because each Task owns an
# independent snapshot.
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


# ── Lifecycle events (frozen; safe to pass across asyncio tasks) ──


@dataclass(frozen=True)
class SubagentStarted:
    """Emitted when a subagent execution begins."""

    parent_session_id: str
    subagent_id: str
    goal: str
    depth: int
    timestamp: float = field(default_factory=time.time)

    @property
    def event_type(self) -> str:
        return "subagent.started"

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SubagentCompleted:
    """Emitted when a subagent finishes successfully."""

    parent_session_id: str
    subagent_id: str
    goal: str
    summary: str
    success: bool
    duration_s: float
    tool_calls: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def event_type(self) -> str:
        return "subagent.completed"

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SubagentFailed:
    """Emitted when a subagent execution fails or is cancelled."""

    parent_session_id: str
    subagent_id: str
    goal: str
    error: str
    duration_s: float
    status: str = "failed"  # "failed" | "cancelled"
    timestamp: float = field(default_factory=time.time)

    @property
    def event_type(self) -> str:
        return "subagent.failed"

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


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
    # Optional raw message list for persistence; None when the executor
    # handles persistence internally (e.g. EngineFrameSubagentExecutor).
    messages: Optional[List[Dict[str, Any]]] = None


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
        event_bus: Optional[Any] = None,
        conversation_store: Optional[Any] = None,
    ) -> None:
        self._executor = executor
        self._max_depth = max_depth
        self._max_concurrent = max_concurrent
        self._on_complete = on_complete
        self._event_bus = event_bus
        # Late-bound conversation store (DIP).  When present, the subagent's
        # message transcript is persisted after execution for auditability.
        # Absence degrades gracefully — no persistence, no crash.
        self._conversation_store = conversation_store
        self._active: Dict[str, asyncio.Task[SubagentResult]] = {}
        self._active_meta: Dict[str, Dict[str, Any]] = {}
        self._recent: Deque[Dict[str, Any]] = collections.deque(maxlen=50)
        self._total_delegated: int = 0
        self._total_completed: int = 0
        self._total_failed: int = 0
        self._total_duration: float = 0.0
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def _emit_event(self, event: Any) -> None:
        """Fire-and-forget an event on the bus; never fail the caller."""
        if self._event_bus is None:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._event_bus.handle_event(
                    event.event_type,
                    event.to_payload(),
                )
            )
        except Exception:
            logger.debug("subagent event emission suppressed", exc_info=True)

    async def delegate(self, config: SubagentConfig) -> SubagentResult:
        """Delegate a task to a subagent with isolation.

        Enforces:
        - Depth limit
        - Concurrent child limit
        - Tool blocking
        - Summary truncation

        The executor coroutine is wrapped in an ``asyncio.Task`` and registered
        in ``_active`` so that ``cancel_all()`` can cancel in-flight subagents.
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
        task_key = config.metadata.get("subagent_id") or session_id
        parent_sid = config.parent_session_id or ""

        self._emit_event(SubagentStarted(
            parent_session_id=parent_sid,
            subagent_id=session_id,
            goal=config.goal,
            depth=config.depth,
        ))
        self._total_delegated += 1
        self._active_meta[task_key] = {
            "subagent_id": session_id,
            "goal": config.goal[:200],
            "depth": config.depth,
            "parent_session_id": parent_sid,
            "start_time": time.time(),
        }

        async with self._semaphore:
            t0 = time.monotonic()
            # Set the depth ContextVar BEFORE creating the task so that the
            # task's copied context snapshot carries the correct value.
            depth_token = _current_depth.set(config.depth)
            task = asyncio.create_task(
                self._executor.execute_subagent(config),
                name=f"subagent:{task_key}",
            )
            self._active[task_key] = task
            try:
                raw_result = await task
                result = self._trim_summary(raw_result, config.summary_max_chars)
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
                self._active.pop(task_key, None)
                self._active_meta.pop(task_key, None)
                _current_depth.reset(depth_token)

            # Lifecycle events: completed vs failed/cancelled
            elapsed = result.elapsed_s or (time.monotonic() - t0)
            # Track stats and add to recent history
            self._total_duration += elapsed
            recent_entry: Dict[str, Any] = {
                "subagent_id": result.session_id or session_id,
                "goal": config.goal[:200],
                "depth": config.depth,
                "parent_session_id": parent_sid,
                "status": result.status,
                "duration_s": round(elapsed, 2),
                "tool_calls": result.tool_calls,
                "timestamp": time.time(),
            }
            if result.status == "completed":
                self._total_completed += 1
            else:
                self._total_failed += 1
                recent_entry["error"] = result.error or result.status
            self._recent.append(recent_entry)
            if result.status == "completed":
                self._emit_event(SubagentCompleted(
                    parent_session_id=parent_sid,
                    subagent_id=result.session_id or session_id,
                    goal=config.goal,
                    summary=result.summary[:200],
                    success=True,
                    duration_s=elapsed,
                    tool_calls=result.tool_calls,
                ))
            else:
                self._emit_event(SubagentFailed(
                    parent_session_id=parent_sid,
                    subagent_id=result.session_id or session_id,
                    goal=config.goal,
                    error=result.error or result.status,
                    duration_s=elapsed,
                    status=result.status,
                ))

            # Persist the subagent's conversation transcript when a store is
            # available and the executor exposed raw messages.  Engine-frame
            # subagents persist internally so result.messages is None for them.
            self._persist_conversation(
                session_id=result.session_id or session_id,
                parent_session_id=parent_sid,
                goal=config.goal,
                messages=result.messages,
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

    def has_active(self) -> bool:
        """Return True when at least one subagent task is in-flight.

        Designed for zero-cost prompt-assembly gating: callers skip all
        formatting work when this returns False.
        """
        return bool(self._active)

    def render_active_status(self) -> str:
        """Render a compact Markdown section describing in-flight subagents.

        Returns an empty string when no subagents are active (zero cost).
        The output is suitable for injection into the system-prompt volatile
        context at EXPANDED or FULL disclosure levels.
        """
        if not self._active:
            return ""
        lines: list[str] = ["## Active Delegated Tasks"]
        for task_key, task in self._active.items():
            task_name = getattr(task, "get_name", lambda: task_key)()
            # Extract goal from task name pattern "subagent:<key>"
            label = task_name.replace("subagent:", "") if task_name.startswith("subagent:") else task_key
            lines.append(f"- {label}: running")
        return "\n".join(lines)

    def cancel_all(self) -> int:
        """Cancel all active subagent tasks. Returns count cancelled."""
        cancelled = 0
        for task_key, task in list(self._active.items()):
            if not task.done():
                task.cancel()
                cancelled += 1
                logger.debug("cancel_all: cancelled subagent task %s", task_key)
        return cancelled

    def get_active_state(self) -> Dict[str, Any]:
        """Return a snapshot of current subagent state for dashboard/RPC.

        Returns a dict with four keys:
        - active: currently running subagents (list of dicts)
        - recent: last N completed/failed (list of dicts, newest first)
        - stats: aggregate counters (total_delegated, completed, failed, avg_duration, success_rate)
        - config: current configuration (max_depth, max_concurrent, summary_max_chars)
        """
        now = time.time()
        active_list = []
        for task_key, meta in self._active_meta.items():
            entry = dict(meta)
            entry["elapsed_s"] = round(now - meta.get("start_time", now), 2)
            active_list.append(entry)
        total_finished = self._total_completed + self._total_failed
        return {
            "active": active_list,
            "recent": list(reversed(self._recent)),  # newest first
            "stats": {
                "total_delegated": self._total_delegated,
                "completed": self._total_completed,
                "failed": self._total_failed,
                "avg_duration": (
                    round(self._total_duration / total_finished, 2)
                    if total_finished > 0 else 0.0
                ),
                "success_rate": (
                    round(self._total_completed / total_finished, 4)
                    if total_finished > 0 else 0.0
                ),
            },
            "config": {
                "max_depth": self._max_depth,
                "max_concurrent": self._max_concurrent,
                "summary_max_chars": _SUMMARY_MAX_CHARS,
            },
        }

    def _persist_conversation(
        self,
        *,
        session_id: str,
        parent_session_id: str,
        goal: str,
        messages: Optional[List[Dict[str, Any]]],
    ) -> None:
        """Persist the subagent's message transcript if a store is available.

        Graceful degradation: if *conversation_store* is ``None`` or *messages*
        is ``None`` (engine-frame path persists internally), this is a no-op.
        Persistence errors are logged and swallowed — never fail the caller.
        """
        if self._conversation_store is None or not messages:
            return
        try:
            store = self._conversation_store
            store.create_session(
                session_id,
                title=goal[:80].replace("\n", " ").strip() or "subagent",
                parent_session_id=parent_session_id or None,
                source="subagent",
            )
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                tc_raw = msg.get("tool_calls")
                store.append_message(
                    session_id,
                    role,
                    content,
                    tool_name=msg.get("tool_name"),
                    tool_call_id=msg.get("tool_call_id"),
                    tool_calls=tc_raw if isinstance(tc_raw, list) else None,
                )
        except Exception:
            logger.debug("subagent session persistence failed", exc_info=True)

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
                messages=result.messages,
            )
        return result


# Risk levels that are safe to execute without approval.
_SAFE_RISK_LEVELS: FrozenSet[str] = frozenset({"read_only", "none"})


class DefaultSubagentExecutor:
    """Concrete SubagentExecutor that runs a lightweight tool loop in isolation.

    Creates a fresh message context with restricted tools and runs the
    standard LLM→tool loop until goal completion or budget exhaustion.

    When a ``tool_pipeline`` is provided, every tool invocation goes through
    the same interceptor chain (approval, audit, timeout) the main agent
    uses — "one approval chain" per AGENTS.md. When absent, the executor
    degrades fail-closed: tools whose declared ``risk_level`` is not
    ``read_only``/``none`` are refused rather than silently executed.
    """

    def __init__(
        self,
        *,
        llm: Any,
        tool_handlers: Dict[str, Any],
        tool_definitions: list,
        settings: Any = None,
        tool_pipeline: Optional[Any] = None,
    ) -> None:
        self._llm = llm
        self._tool_handlers = tool_handlers
        self._tool_definitions = tool_definitions
        self._settings = settings
        self._tool_pipeline = tool_pipeline
        # Build a lookup from tool name → x_leapflow metadata for fail-closed
        # gating when the pipeline is absent.
        self._tool_risk: Dict[str, str] = {}
        for td in tool_definitions:
            fn = td.get("function", {})
            name = fn.get("name", "")
            x = fn.get("x_leapflow", {})
            if name and isinstance(x, dict):
                self._tool_risk[name] = x.get("risk_level", "mutating")

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
        from leapflow.engine.context.context_control import (
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
                        result = await self._execute_tool(
                            tc.name, tc.arguments, handler,
                        )
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
            messages=messages,
        )

    # ------------------------------------------------------------------
    # Tool execution — pipeline-gated or fail-closed
    # ------------------------------------------------------------------

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        handler: Any,
    ) -> Any:
        """Execute a single tool call through the pipeline or fail-closed."""
        pipeline = self._tool_pipeline

        if pipeline is not None and pipeline.interceptor_count > 0:
            # Route through the shared ToolExecutionPipeline so approval,
            # audit, and timeout interceptors apply identically to the main
            # agent loop.
            from leapflow.domain.tool_pipeline import ToolCallContext
            from leapflow.plugins.handler_invocation import invoke_tool_handler

            risk = self._tool_risk.get(tool_name, "mutating")
            ctx = ToolCallContext(
                tool_name=tool_name,
                arguments=arguments,
                metadata={"risk_level": risk, "source": "subagent"},
            )

            async def _invoke(c: ToolCallContext) -> Dict[str, Any]:
                return await invoke_tool_handler(handler, c.arguments)

            return await pipeline.execute(ctx, _invoke)

        # Fail-closed: no pipeline means no approval chain is available.
        # Only allow tools with a safe declared risk_level.
        risk_level = self._tool_risk.get(tool_name, "mutating")
        if risk_level not in _SAFE_RISK_LEVELS:
            return {
                "ok": False,
                "error": (
                    f"Tool '{tool_name}' (risk_level={risk_level}) blocked: "
                    "no approval pipeline available in subagent executor."
                ),
            }

        from leapflow.plugins.handler_invocation import invoke_tool_handler
        return await invoke_tool_handler(handler, arguments)


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
        available -= {"delegate_task"}

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
            child_result = await self._run_child(
                goal,
                depth=config.depth,
                tool_filter=frozenset(available),
                enable_thinking=False,
            )
            # _run_subagent_goal returns (summary, tool_calls); gracefully
            # handle the legacy str return for backward compatibility.
            if isinstance(child_result, tuple):
                summary_text, child_tool_calls = child_result
            else:
                summary_text = child_result
                child_tool_calls = 0
        except Exception as exc:  # isolate subagent failure from the parent loop
            return SubagentResult(
                session_id=session_id, goal=config.goal,
                summary=f"Subagent error: {exc}", status="failed",
                elapsed_s=time.monotonic() - t0, error=str(exc),
            )
        return SubagentResult(
            session_id=session_id, goal=config.goal,
            summary=(summary_text or "")[:config.summary_max_chars],
            status="completed", elapsed_s=time.monotonic() - t0,
            tool_calls=child_tool_calls,
        )
