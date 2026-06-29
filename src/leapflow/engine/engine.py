"""Main ReAct-style engine with routing, skills, and audit logging."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from leapflow.platform.protocol import HostRpc, Methods
from leapflow.config import Settings
from leapflow.engine.budget import BudgetConfig, BudgetStatus, IterationBudget
from leapflow.engine.context_compressor import CompressorConfig, ContextCompressor
from leapflow.engine.error_classifier import (
    ErrorCategory,
    ErrorClassifier,
    build_recovery_map,
    jittered_backoff,
)
from leapflow.engine.execution_trace import ExecutionMode, ExecutionTrace
from leapflow.engine.intent_classifier import Intent, IntentClassifier
from leapflow.engine.message_healer import MessageHealer
from leapflow.engine.shortcuts import ShortcutStore
from leapflow.engine.graph_planner import GraphPlanner
from leapflow.engine.scheduler import TaskScheduler
from leapflow.engine.session import SessionController
from leapflow.analysis.pipeline import ImitationPipeline
from leapflow.llm.base import LLMProvider
from leapflow.llm.message_builder import (
    build_assistant_message,
    build_system_message,
    build_user_message_text,
)
from leapflow.memory.providers.episodic import EpisodicMemoryProvider
from leapflow.memory.providers.semantic import SemanticMemoryProvider
from leapflow.memory.providers.working import WorkingMemoryProvider
from leapflow.memory.providers.evolution import EvolutionMemoryProvider
from leapflow.memory.manager import MemoryManager
from leapflow.prompts.templates import REACT_SYSTEM_TEMPLATE
from leapflow.learning.active_learning import SkillMerger
from leapflow.skills.builtin import app_launcher, clipboard_manager, file_organizer
from leapflow.storage.skill_library import SkillLibraryStore
from leapflow.skills.registry import Skill, SkillRegistry

logger = logging.getLogger(__name__)


def _log_progress(msg: str) -> None:
    """Print a persistent progress line to stderr (visible to user during `leap run`)."""
    sys.stderr.write(f"\033[2m\u2192 {msg}\033[0m\n")
    sys.stderr.flush()


def _show_indicator(msg: str) -> None:
    """Show a transient progress indicator on stderr (overwritten on next call)."""
    if not sys.stderr.isatty():
        return
    sys.stderr.write(f"\r\033[K\033[2m\u25cf {msg}\033[0m")
    sys.stderr.flush()


def _show_progress(phase: str, detail: str = "", step: int = 0, total: int = 0) -> None:
    """Show a structured progress indicator on stderr with optional step counter."""
    if not sys.stderr.isatty():
        return
    parts: list[str] = []
    if step and total:
        parts.append(f"[{step}/{total}]")
    parts.append(phase)
    if detail:
        parts.append(f"\u2014 {detail[:60]}")
    msg = " ".join(parts)
    sys.stderr.write(f"\r\033[K\033[2m\u25cf {msg}\033[0m")
    sys.stderr.flush()


def _clear_indicator() -> None:
    """Clear the transient progress indicator from stderr."""
    if not sys.stderr.isatty():
        return
    sys.stderr.write("\r\033[K")
    sys.stderr.flush()


def _print_tool_result(tool_name: str, result: Any, *, enabled: bool = True) -> None:
    """Print a brief tool result summary to stdout (visible to user)."""
    if not enabled:
        return
    if isinstance(result, dict):
        # Try to extract a meaningful summary
        if "error" in result:
            preview = f"error: {result['error']}"
        elif "output" in result:
            preview = str(result["output"])
        elif "result" in result:
            preview = str(result["result"])
        elif "entries" in result:
            preview = f"{len(result['entries'])} entries"
        elif "ok" in result:
            preview = "ok" if result["ok"] else "failed"
        else:
            preview = json.dumps(result, default=str, ensure_ascii=False)
    else:
        preview = str(result)
    # Truncate
    if len(preview) > 120:
        preview = preview[:117] + "..."
    sys.stdout.write(f"\033[2m  \u21b3 {tool_name}: {preview}\033[0m\n")
    sys.stdout.flush()


def _extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no json object")
    return json.loads(text[start : end + 1])


def _keywords_from_query(q: str) -> list[str]:
    toks = re.findall(r"[\w\-./]+|[\u4e00-\u9fff]+", q)
    return [t for t in toks if len(t) >= 2][:12]


@dataclass
class _LoopContext:
    """Mutable state carried across state machine transitions."""

    messages: List[Dict[str, Any]]
    last_content: str = ""
    last_action: Optional[Dict[str, Any]] = None
    last_observation: Any = None
    last_error: Optional[Exception] = None
    consecutive_failures: int = 0
    prefetch_done: bool = False  # track whether memory prefetch ran this loop


class AgentEngine:
    """Coordinates perception memory, LLM reasoning, RPC execution, and skills."""

    def __init__(
        self,
        settings: Settings,
        rpc: HostRpc,
        llm: LLMProvider,
        wm: WorkingMemoryProvider,
        lt: SemanticMemoryProvider,
        imm: EpisodicMemoryProvider,
        registry: SkillRegistry,
        classifier: IntentClassifier,
        imitation: Optional[ImitationPipeline] = None,
        skill_library: Optional[SkillLibraryStore] = None,
        graph_planner: Optional[GraphPlanner] = None,
        scheduler: Optional[TaskScheduler] = None,
        perception: Optional[Any] = None,
        execution: Optional[Any] = None,
        skill_activator: Optional[Any] = None,
        session: Optional[SessionController] = None,
        shortcuts: Optional["ShortcutStore"] = None,
        vlm: Optional[Any] = None,
        memory_manager: Optional[MemoryManager] = None,
        evolution: Optional[EvolutionMemoryProvider] = None,
        tool_bridge: Optional[Any] = None,
        skill_injector: Optional[Any] = None,
        skill_index: Optional[Any] = None,
    ) -> None:
        self._settings = settings
        self._rpc = rpc
        self._llm = llm
        self._vlm = vlm
        self._wm = wm
        self._lt = lt
        self._imm = imm
        self._registry = registry
        self._classifier = classifier
        self._shortcuts = shortcuts
        self._imitation = imitation
        self._skill_library = skill_library
        self._skill_merger = SkillMerger()
        self._graph_planner = graph_planner
        self._scheduler = scheduler
        self._perception = perception
        self._execution = execution
        self._activator = skill_activator
        self._session = session

        # Memory integration (MemoryManager + EvolutionProvider)
        self._memory_manager = memory_manager
        self._evolution = evolution

        # Skill index for compact prompt injection
        self._skill_index: Optional[Any] = skill_index

        # Pre-built ToolBridge with general-purpose tools registered
        self._tool_bridge = tool_bridge

        # Skill discovery (SkillInjector for slash commands)
        self._skill_injector = skill_injector

        # State-machine loop infrastructure (config-driven)
        self._budget_config = BudgetConfig(
            max_iterations=settings.react_max_iterations,
            soft_limit=settings.react_soft_limit,
            warning_threshold=settings.react_warning_threshold,
        )
        self._error_classifier = ErrorClassifier(
            recovery_map=build_recovery_map(
                transient_max_retries=settings.error_transient_max_retries,
                rate_limit_base_delay=settings.error_rate_limit_base_delay,
            )
        )
        self._compressor = ContextCompressor(CompressorConfig(
            threshold=settings.compress_threshold,
            keep_tail=settings.compress_keep_tail,
            max_output_chars=settings.max_tool_output_chars,
        ))
        self._healer = MessageHealer()

    async def run(self, user_text: str, *, enable_thinking: bool = False) -> str:
        """Entrypoint: simplified routing with unified tool loop as default path."""
        logger.info("audit.user_input chars=%s", len(user_text))

        # 1. Shortcut match (zero cost, exact keyword)
        if self._shortcuts:
            reply = self._shortcuts.match(user_text)
            if reply:
                self._wm.remember_chat(build_user_message_text(user_text))
                self._wm.remember_chat(build_assistant_message(reply))
                return reply

        # 2. Slash command (skill injection — zero-ambiguity activation)
        if user_text.startswith("/") and self._skill_injector:
            self._inject_pending_skill_reminder()
            self._wm.remember_chat(build_user_message_text(user_text))
            logger.debug("route.slash command=%s", user_text.split()[0])
            return await self._unified_tool_loop(user_text, enable_thinking=enable_thinking)

        self._inject_pending_skill_reminder()
        self._wm.remember_chat(build_user_message_text(user_text))

        # 3. Learn/teach command (special session mode switch)
        if self._is_learn_command(user_text):
            return await self._handle_learn_command(user_text)

        # 4. Everything else → unified tool loop (LLM decides tools vs direct response)
        logger.debug("route.unified user_text_len=%d", len(user_text))
        return await self._unified_tool_loop(user_text, enable_thinking=enable_thinking)

    async def run_stream(
        self, user_text: str, *, enable_thinking: bool = False
    ) -> AsyncIterator[str]:
        """Like run(), but yields text chunks for streamable responses."""
        logger.info("audit.user_input chars=%s", len(user_text))

        # 1. Shortcut match (zero cost)
        if self._shortcuts:
            reply = self._shortcuts.match(user_text)
            if reply:
                self._wm.remember_chat(build_user_message_text(user_text))
                self._wm.remember_chat(build_assistant_message(reply))
                yield reply
                return

        # 2. Slash command (skill injection)
        if user_text.startswith("/") and self._skill_injector:
            self._inject_pending_skill_reminder()
            self._wm.remember_chat(build_user_message_text(user_text))
            logger.debug("route.slash command=%s", user_text.split()[0])
            async for chunk in self._unified_tool_loop_stream(user_text, enable_thinking=enable_thinking):
                yield chunk
            return

        self._inject_pending_skill_reminder()
        self._wm.remember_chat(build_user_message_text(user_text))

        # 3. Learn/teach command (special session mode switch)
        if self._is_learn_command(user_text):
            result = await self._handle_learn_command(user_text)
            yield result
            return

        # 4. Everything else → unified tool loop (streaming)
        logger.debug("route.unified user_text_len=%d", len(user_text))
        async for chunk in self._unified_tool_loop_stream(user_text, enable_thinking=enable_thinking):
            yield chunk

    # ── Complex Task Handling (DAG path) ─────────────────────────────

    async def _handle_complex_task(self, user_goal: str) -> str:
        """Handle complex multi-step tasks via DAG planning and execution.

        Flow: GraphPlanner → TaskScheduler → summary report.
        Falls back to ReAct loop on planning failure.
        """
        assert self._graph_planner is not None
        assert self._scheduler is not None

        try:
            _log_progress("Building execution plan...")
            graph = await self._graph_planner.plan(user_goal)
            self._wm.remember_event(
                "dag_plan", graph.summary(), {"nodes": len(graph.nodes)}
            )
            node_names = [n.name for n in list(graph.nodes.values())[:10]]
            _log_progress(f"Execution plan ready: {len(graph.nodes)} steps — {', '.join(node_names)}")
            graph = await self._scheduler.execute_graph(graph)
            _log_progress("Plan execution complete")
            return graph.summary()
        except (ValueError, Exception) as e:
            _log_progress(f"DAG planning failed ({e}), falling back to ReAct loop")
            logger.warning("audit.dag_fallback reason=%s", e)
            return await self._fallback_react(user_goal)

    async def _fallback_react(self, user_text: str) -> str:
        """Fallback to ReAct loop when DAG planning/execution fails."""
        steps = await self._plan_steps(user_text)
        self._wm.remember_event("plan", " | ".join(steps), {"steps": steps})
        return await self._react_loop(user_text, steps)

    async def _plan_steps(self, user_goal: str) -> List[str]:
        """Generate flat step list via LLM for the ReAct loop."""
        catalog = self._registry.describe()
        messages = [
            build_system_message(
                "Return STRICT JSON: {\"steps\":[\"...\", ...]} with 3-7 steps for the goal. "
                f"Available skills:\n{catalog}"
            ),
            build_user_message_text(user_goal),
        ]
        try:
            resp = await self._llm.achat(messages, stream=False, enable_thinking=False)
            raw = (resp.content or "").strip()
            start = raw.find("{")
            end = raw.rfind("}")
            blob = raw[start : end + 1] if start != -1 and end != -1 else raw
            data = json.loads(blob)
            steps = [str(x) for x in list(data.get("steps") or [])]
            return steps[:10]
        except Exception:
            logger.debug("plan_steps failed", exc_info=True)
            return [user_goal]

    async def _react_loop(
        self,
        user_text: str,
        steps: List[str],
        *,
        enable_thinking: bool = False,
    ) -> str:
        """State-machine driven ReAct loop (async shell, sync semantics)."""
        budget = IterationBudget.for_react(self._budget_config)
        trace = ExecutionTrace()
        ctx = _LoopContext(messages=self._build_loop_messages(user_text, steps))
        state = ExecutionMode.PREPARING

        while state != ExecutionMode.COMPLETE:
            state = await self._loop_step(
                state, ctx, budget, trace,
                user_text=user_text,
                enable_thinking=enable_thinking,
            )

        # Fire-and-forget: emit learning signal to the evolution ring
        if trace.has_learning_signal:
            asyncio.create_task(self._emit_execution_trace(trace))

        # Sync conversation turn to long-term memory (non-blocking)
        if self._memory_manager and self._settings.memory_integration_enabled:
            asyncio.create_task(self._sync_turn_safe(ctx.messages))

        logger.info(
            "react_loop.complete steps=%d tokens=%d success=%s",
            trace.step_count, trace.total_tokens, trace.success,
        )
        return ctx.last_content or "Stopped after step budget."

    # ── State Machine Core ──────────────────────────────────────────────

    async def _loop_step(  # noqa: C901 (state machine dispatch)
        self,
        state: ExecutionMode,
        ctx: _LoopContext,
        budget: IterationBudget,
        trace: ExecutionTrace,
        *,
        user_text: str,
        enable_thinking: bool,
    ) -> ExecutionMode:
        """Execute one state transition. Returns the next state."""

        if state == ExecutionMode.PREPARING:
            return await self._state_preparing(ctx, budget, trace, user_text=user_text)

        if state == ExecutionMode.REASONING:
            return await self._state_reasoning(ctx, trace, enable_thinking=enable_thinking)

        if state == ExecutionMode.ROUTING:
            return self._state_routing(ctx, trace)

        if state == ExecutionMode.ACTING:
            return await self._state_acting(ctx, trace, user_text=user_text)

        if state == ExecutionMode.OBSERVING:
            return self._state_observing(ctx, budget, trace)

        if state == ExecutionMode.RECOVERING:
            return await self._state_recovering(ctx, budget, trace)

        # Fallback: unreachable unless enum extended
        trace.record(ExecutionMode.COMPLETE, error="invalid_state")
        return ExecutionMode.COMPLETE

    # ── State Handlers ──────────────────────────────────────────────────

    async def _state_preparing(
        self, ctx: _LoopContext, budget: IterationBudget, trace: ExecutionTrace,
        *, user_text: str = "",
    ) -> ExecutionMode:
        """Budget check + memory prefetch + message healing + compression."""
        status = budget.consume()

        if status == BudgetStatus.EXHAUSTED:
            trace.record(ExecutionMode.COMPLETE, error="budget_exhausted")
            ctx.last_content = self._budget_exhausted_response(ctx.messages)
            return ExecutionMode.COMPLETE

        # Prefetch memory context on first iteration (non-blocking with timeout)
        if not ctx.prefetch_done and self._memory_manager and self._settings.memory_integration_enabled:
            ctx.prefetch_done = True
            try:
                entries = await asyncio.wait_for(
                    self._memory_manager.prefetch(
                        user_text, limit=self._settings.memory_prefetch_limit,
                    ),
                    timeout=self._settings.memory_prefetch_timeout_s,
                )
                if entries:
                    context_lines = [e.content for e in entries if e.content]
                    if context_lines:
                        memory_block = (
                            "MEMORY_CONTEXT (relevant past experiences):\n"
                            + "\n".join(f"- {line}" for line in context_lines)
                        )
                        ctx.messages.append(build_user_message_text(memory_block))
                        logger.debug("memory.prefetch injected %d entries", len(context_lines))
            except asyncio.TimeoutError:
                logger.debug("memory.prefetch timed out (%.1fs)", self._settings.memory_prefetch_timeout_s)
            except Exception:
                logger.debug("memory.prefetch failed", exc_info=True)

        # Heal and compress
        ctx.messages = self._healer.heal(ctx.messages)
        ctx.messages = self._compressor.compress(ctx.messages)

        # Inject convergence hint near soft limit
        if status == BudgetStatus.SOFT_LIMIT:
            ctx.messages.append(build_user_message_text(
                "SYSTEM: Approaching iteration limit. Please converge and provide final answer."
            ))

        remaining = budget.remaining
        _log_progress(f"Thinking (budget {remaining} remaining)...")
        return ExecutionMode.REASONING

    async def _state_reasoning(
        self, ctx: _LoopContext, trace: ExecutionTrace, *, enable_thinking: bool
    ) -> ExecutionMode:
        """LLM call — the only await in the hot path."""
        t0 = time.perf_counter()
        try:
            resp = await self._llm.achat(
                ctx.messages + self._wm.as_chat_messages(),
                stream=False,
                enable_thinking=enable_thinking,
            )
            latency = (time.perf_counter() - t0) * 1000
            content = (resp.content or "").strip()
            tokens = getattr(resp, "usage_tokens", 0)
            trace.record(ExecutionMode.REASONING, tokens_used=tokens, latency_ms=latency)

            if resp.thinking_content:
                logger.debug("audit.thinking chars=%s", len(resp.thinking_content))

            ctx.last_content = content
            self._wm.remember_chat(build_assistant_message(content))
            ctx.messages.append(build_assistant_message(content))
            return ExecutionMode.ROUTING

        except Exception as exc:
            trace.record(ExecutionMode.RECOVERING, error=str(exc))
            ctx.last_error = exc
            return ExecutionMode.RECOVERING

    def _state_routing(self, ctx: _LoopContext, trace: ExecutionTrace) -> ExecutionMode:
        """Parse LLM output and decide: final answer, action, or raw text."""
        content = ctx.last_content

        try:
            obj = _extract_json_object(content)
        except Exception:
            # No parseable JSON — treat as final answer
            logger.info("audit.react_no_json; returning assistant text")
            trace.record(ExecutionMode.COMPLETE)
            return ExecutionMode.COMPLETE

        action = obj.get("action") or {}
        a_type = str(action.get("type", "")).strip()

        # Final answer
        if a_type == "answer" and str(action.get("name")) == "final":
            payload = action.get("payload") or {}
            ans = str(payload.get("text") or payload.get("content") or "").strip()
            ctx.last_content = ans or content
            trace.record(ExecutionMode.COMPLETE)
            return ExecutionMode.COMPLETE

        if not a_type:
            # No action type — treat content as final answer
            trace.record(ExecutionMode.COMPLETE)
            return ExecutionMode.COMPLETE

        # Prepare prediction loop (world model)
        action_name = str(action.get("name", a_type)).strip()
        predicted_effect = str(obj.get("predicted_effect", "")).strip()
        if predicted_effect:
            self._wm.remember_event(
                "react_prediction",
                f"action={action_name}, predicted={predicted_effect}",
                {"action": action_name},
            )

        ctx.last_action = action
        return ExecutionMode.ACTING

    async def _state_acting(
        self, ctx: _LoopContext, trace: ExecutionTrace, *, user_text: str
    ) -> ExecutionMode:
        """Execute the parsed action via skill/bridge."""
        action = ctx.last_action or {}
        action_name = str(action.get("name", action.get("type", ""))).strip()
        action_payload = action.get("payload") or {}

        payload_hint = json.dumps(action_payload, ensure_ascii=False)
        if len(payload_hint) > 200:
            payload_hint = payload_hint[:200] + "..."
        _log_progress(f"Executing action: {action_name} — {payload_hint}")

        # Prediction loop pre-snapshot
        a_type = str(action.get("type", "")).strip()
        predicted_effect = ""
        # Extract predicted_effect from original JSON if available
        try:
            obj = _extract_json_object(ctx.last_content)
            predicted_effect = str(obj.get("predicted_effect", "")).strip()
        except Exception:
            pass

        react_prediction = None
        pl = self._registry.prediction_loop
        if predicted_effect and pl is not None and pl.enabled:
            react_prediction = pl.create_from_react_prediction(
                action_desc=f"{a_type}:{action_name}",
                predicted_effect=predicted_effect,
            )
            await pl.capture_pre_snapshot()

        t0 = time.perf_counter()
        observation = await self._execute_action(action, user_text)
        latency = (time.perf_counter() - t0) * 1000

        # Prediction loop verification
        if react_prediction is not None and pl is not None:
            await pl.verify_prediction(react_prediction)

        trace.record(
            ExecutionMode.ACTING,
            action=action,
            observation=observation if isinstance(observation, dict) else {"result": str(observation)},
            latency_ms=latency,
        )
        ctx.last_observation = observation
        return ExecutionMode.OBSERVING

    def _state_observing(
        self, ctx: _LoopContext, budget: IterationBudget, trace: ExecutionTrace
    ) -> ExecutionMode:
        """Evaluate observation and feed back into messages."""
        observation = ctx.last_observation
        is_error = isinstance(observation, dict) and not observation.get("ok", True)

        if is_error:
            ctx.consecutive_failures += 1
            error_detail = (
                observation.get("error", "unknown error")
                if isinstance(observation, dict)
                else str(observation)
            )
            category = self._error_classifier.classify_tool_error(observation)
            max_tool_failures = self._settings.max_consecutive_tool_failures

            if category == ErrorCategory.PERMANENT or ctx.consecutive_failures >= max_tool_failures:
                _log_progress(f"{ctx.consecutive_failures} consecutive failures — stopping")
                trace.record(ExecutionMode.COMPLETE, error="max_tool_failures")
                ctx.last_content = self._error_response(observation)
                return ExecutionMode.COMPLETE

            _log_progress(f"Action failed ({ctx.consecutive_failures}/3): {error_detail}")
            budget.refund("tool_failure")
            error_obs = json.dumps(
                {"observation": observation, "recovery_hint": "Previous action failed. Try an alternative approach."},
                ensure_ascii=False,
            )
            ctx.messages.append(build_user_message_text(error_obs))
            self._wm.remember_event("react_error", str(error_detail)[:200], {})
            return ExecutionMode.PREPARING

        # Success
        ctx.consecutive_failures = 0
        obs_summary = str(observation)
        if len(obs_summary) > 300:
            obs_summary = obs_summary[:300] + "..."
        _log_progress(f"Observation: {obs_summary}")
        obs_text = json.dumps({"observation": observation}, ensure_ascii=False)
        ctx.messages.append(build_user_message_text(obs_text))
        self._wm.remember_event("react_observation", obs_text, {})
        return ExecutionMode.PREPARING

    async def _state_recovering(
        self, ctx: _LoopContext, budget: IterationBudget, trace: ExecutionTrace
    ) -> ExecutionMode:
        """Handle LLM/network errors with classified recovery."""
        exc = ctx.last_error
        if exc is None:
            trace.record(ExecutionMode.COMPLETE, error="unknown_recovery")
            return ExecutionMode.COMPLETE

        category = self._error_classifier.classify(exc)
        recovery = self._error_classifier.get_recovery(category)

        if recovery.retry and ctx.consecutive_failures < recovery.max_retries:
            ctx.consecutive_failures += 1
            if recovery.backoff:
                delay = jittered_backoff(ctx.consecutive_failures, base=recovery.base_delay)
                await asyncio.sleep(delay)
            if recovery.compress:
                ctx.messages = self._compressor.force_compress(ctx.messages)
            logger.warning(
                "react_loop.recovery category=%s attempt=%d",
                category.value, ctx.consecutive_failures,
            )
            return ExecutionMode.PREPARING

        # Unrecoverable
        trace.record(ExecutionMode.COMPLETE, error=str(exc))
        ctx.last_content = f"Error: {exc}"
        return ExecutionMode.COMPLETE

    # ── Unified Tool Loop (chat scenarios) ───────────────────────────────

    async def _unified_tool_loop(
        self, user_text: str, *, enable_thinking: bool = False
    ) -> str:
        """Unified chat+tool loop: LLM dynamically decides tools vs direct response.

        Uses native OpenAI tool_calls when provider supports them, falls back to
        text-based parsing. Injects working memory for multi-turn coherence.
        Reuses IterationBudget, ErrorClassifier, ContextCompressor, MessageHealer.
        """
        # Detect slash command → inject skill context
        if user_text.startswith("/"):
            slash_name = user_text.split()[0][1:]  # Remove leading /
            remaining = user_text[len(slash_name) + 1:].strip()
            if self._skill_injector:
                injection = self._skill_injector.build_injection_message(slash_name, remaining)
                if injection:
                    user_text = injection  # Replace user_text with skill injection

        from leapflow.prompts.templates import UNIFIED_SYSTEM_TEMPLATE
        from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS, TOOL_HANDLERS

        budget = IterationBudget.for_react(self._budget_config)
        trace = ExecutionTrace()

        # Build tool catalog text (for system prompt readability)
        tool_catalog = self._format_tool_catalog(TOOL_DEFINITIONS)

        # Build memory context from long-term memory
        memory_context = ""
        if self._memory_manager and self._settings.memory_integration_enabled:
            try:
                entries = await asyncio.wait_for(
                    self._memory_manager.prefetch(
                        user_text, limit=self._settings.memory_prefetch_limit,
                    ),
                    timeout=self._settings.memory_prefetch_timeout_s,
                )
                if entries:
                    memory_context = "## Recent Context\n" + "\n".join(
                        f"- [{e.kind.value}] {e.content[:100]}" for e in entries
                    )
            except Exception:
                pass

        # Build skill section (conditionally included)
        skill_section = ""
        if self._skill_index:
            entries = self._skill_index.get_entries()
            if entries:
                skill_index_text = self._skill_index.compact_index_text(entries)
                skill_section = (
                    "\n## Learned Skills\n"
                    "You have access to the following learned skills. "
                    "Use `gp_skills_list` to browse or `gp_skill_view` to read details:\n"
                    f"{skill_index_text}\n"
                )

        # Build system prompt
        system = UNIFIED_SYSTEM_TEMPLATE.format(
            tool_catalog=tool_catalog,
            skill_section=skill_section,
            memory_context=memory_context,
        )

        # Inject prior conversation turns from working memory for multi-turn coherence
        wm_history = self._wm.as_chat_messages()
        # Filter to keep only recent user/assistant exchanges (skip system events)
        prior_turns: List[Dict[str, Any]] = [
            m for m in wm_history
            if isinstance(m.get("role"), str) and m["role"] in ("user", "assistant")
        ]
        # Limit to last N turns to avoid overwhelming context
        max_prior_turns = 10
        prior_turns = prior_turns[-max_prior_turns:]

        messages: List[Dict[str, Any]] = [
            build_system_message(system),
            *prior_turns,
            build_user_message_text(user_text),
        ]

        content = ""
        use_native_tools = self._settings.native_tool_calling_enabled

        # Prepare tools kwarg for native tool calling
        tools_kwarg: Dict[str, Any] = {}
        if use_native_tools and TOOL_DEFINITIONS:
            tools_kwarg["tools"] = TOOL_DEFINITIONS

        # Loop: LLM response → parse → tool_call or final answer
        while not budget.exhausted:
            status = budget.consume()
            if status == BudgetStatus.EXHAUSTED:
                break

            # Heal + compress
            healed = self._healer.heal(messages)
            compressed = self._compressor.compress(healed)

            _show_progress("thinking", f"round {budget.used}")
            try:
                resp = await self._llm.achat(
                    compressed, stream=False, enable_thinking=enable_thinking,
                    **tools_kwarg,
                )
            except Exception as exc:
                _clear_indicator()
                category = self._error_classifier.classify(exc)
                recovery = self._error_classifier.get_recovery(category)
                if recovery.retry and budget.remaining > 0:
                    await asyncio.sleep(jittered_backoff(budget.used))
                    continue
                # If native tools caused the error, retry without them
                if use_native_tools and tools_kwarg:
                    logger.info("Native tool calling failed, falling back to text mode")
                    tools_kwarg = {}
                    use_native_tools = False
                    continue
                break
            _clear_indicator()

            content = (resp.content or "").strip()

            # Check for native tool_calls first (more reliable)
            native_calls = getattr(resp, "tool_calls", None) or []
            if native_calls:
                # Build assistant message with tool_calls metadata
                assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content}
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                    }
                    for tc in native_calls
                ]
                messages.append(assistant_msg)

                # Execute each native tool call
                for i, tc in enumerate(native_calls):
                    _show_progress("executing", tc.name, step=i + 1, total=len(native_calls))
                    tool_call_dict = {"name": tc.name, "arguments": tc.arguments}
                    result = await self._execute_general_tool(tool_call_dict, TOOL_HANDLERS)
                    _clear_indicator()
                    _print_tool_result(tc.name, result, enabled=self._settings.verbose_progress)
                    trace.record(
                        ExecutionMode.ACTING,
                        action=tool_call_dict,
                        observation=result if isinstance(result, dict) else {"result": str(result)},
                    )
                    result_text = json.dumps(result, default=str, ensure_ascii=False)[:3000]
                    # Append tool result in OpenAI format
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })

                self._wm.remember_chat(build_assistant_message(
                    content or f"[Called: {', '.join(tc.name for tc in native_calls)}]"
                ))

                if status == BudgetStatus.SOFT_LIMIT:
                    messages.append(build_user_message_text(
                        "SYSTEM: Approaching limit. Provide final answer now."
                    ))
                continue

            # Fallback: text-based tool call parsing
            self._wm.remember_chat(build_assistant_message(content))
            tool_call = self._parse_tool_call_from_content(content)

            if tool_call is None:
                # Pure text response — done (guard against empty content)
                trace.record(ExecutionMode.COMPLETE)
                if not content:
                    return "I processed your request but have no additional output."
                return content

            # Execute text-parsed tool
            messages.append(build_assistant_message(content))
            _show_progress("executing", tool_call['name'])
            result = await self._execute_general_tool(tool_call, TOOL_HANDLERS)
            _clear_indicator()
            _print_tool_result(tool_call['name'], result, enabled=self._settings.verbose_progress)
            trace.record(
                ExecutionMode.ACTING,
                action=tool_call,
                observation=result if isinstance(result, dict) else {"result": str(result)},
            )

            result_text = json.dumps(result, default=str, ensure_ascii=False)[:3000]
            messages.append(build_user_message_text(
                f"Tool result ({tool_call['name']}):\n{result_text}"
            ))

            if status == BudgetStatus.SOFT_LIMIT:
                messages.append(build_user_message_text(
                    "SYSTEM: Approaching limit. Provide final answer now."
                ))

        # Budget exhausted
        return content if content else "I've reached my processing limit."

    async def _unified_tool_loop_stream(
        self, user_text: str, *, enable_thinking: bool = False
    ) -> AsyncIterator[str]:
        """Streaming variant of _unified_tool_loop.

        Yields text chunks for the final response. Shows transient progress
        indicators on stderr for thinking and tool-execution phases.
        """
        # Reuse the same setup logic as _unified_tool_loop
        if user_text.startswith("/"):
            slash_name = user_text.split()[0][1:]
            remaining = user_text[len(slash_name) + 1:].strip()
            if self._skill_injector:
                injection = self._skill_injector.build_injection_message(slash_name, remaining)
                if injection:
                    user_text = injection

        from leapflow.prompts.templates import UNIFIED_SYSTEM_TEMPLATE
        from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS, TOOL_HANDLERS

        budget = IterationBudget.for_react(self._budget_config)
        trace = ExecutionTrace()

        tool_catalog = self._format_tool_catalog(TOOL_DEFINITIONS)

        # Memory context prefetch
        memory_context = ""
        if self._memory_manager and self._settings.memory_integration_enabled:
            try:
                entries = await asyncio.wait_for(
                    self._memory_manager.prefetch(
                        user_text, limit=self._settings.memory_prefetch_limit,
                    ),
                    timeout=self._settings.memory_prefetch_timeout_s,
                )
                if entries:
                    memory_context = "## Recent Context\n" + "\n".join(
                        f"- [{e.kind.value}] {e.content[:100]}" for e in entries
                    )
            except Exception:
                pass

        # Skill section
        skill_section = ""
        if self._skill_index:
            idx_entries = self._skill_index.get_entries()
            if idx_entries:
                skill_index_text = self._skill_index.compact_index_text(idx_entries)
                skill_section = (
                    "\n## Learned Skills\n"
                    "You have access to the following learned skills. "
                    "Use `gp_skills_list` to browse or `gp_skill_view` to read details:\n"
                    f"{skill_index_text}\n"
                )

        system = UNIFIED_SYSTEM_TEMPLATE.format(
            tool_catalog=tool_catalog,
            skill_section=skill_section,
            memory_context=memory_context,
        )

        # Prior conversation context
        wm_history = self._wm.as_chat_messages()
        prior_turns: List[Dict[str, Any]] = [
            m for m in wm_history
            if isinstance(m.get("role"), str) and m["role"] in ("user", "assistant")
        ]
        prior_turns = prior_turns[-10:]

        messages: List[Dict[str, Any]] = [
            build_system_message(system),
            *prior_turns,
            build_user_message_text(user_text),
        ]

        content = ""
        use_native_tools = self._settings.native_tool_calling_enabled

        tools_kwarg: Dict[str, Any] = {}
        if use_native_tools and TOOL_DEFINITIONS:
            tools_kwarg["tools"] = TOOL_DEFINITIONS

        # Loop: LLM -> parse -> tool_call or stream final answer
        while not budget.exhausted:
            status = budget.consume()
            if status == BudgetStatus.EXHAUSTED:
                break

            healed = self._healer.heal(messages)
            compressed = self._compressor.compress(healed)

            _show_progress("thinking", f"round {budget.used}")

            # When native tools are enabled we need stream=False for tool_calls
            if use_native_tools and tools_kwarg:
                try:
                    resp = await self._llm.achat(
                        compressed, stream=False, enable_thinking=enable_thinking,
                        **tools_kwarg,
                    )
                except Exception as exc:
                    _clear_indicator()
                    category = self._error_classifier.classify(exc)
                    recovery = self._error_classifier.get_recovery(category)
                    if recovery.retry and budget.remaining > 0:
                        await asyncio.sleep(jittered_backoff(budget.used))
                        continue
                    if use_native_tools and tools_kwarg:
                        logger.info("Native tool calling failed, falling back to text mode")
                        tools_kwarg = {}
                        use_native_tools = False
                        continue
                    break
                _clear_indicator()

                content = (resp.content or "").strip()
                native_calls = getattr(resp, "tool_calls", None) or []
                if native_calls:
                    assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content}
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                        }
                        for tc in native_calls
                    ]
                    messages.append(assistant_msg)

                    for i, tc in enumerate(native_calls):
                        _show_progress("executing", tc.name, step=i + 1, total=len(native_calls))
                        tool_call_dict = {"name": tc.name, "arguments": tc.arguments}
                        result = await self._execute_general_tool(tool_call_dict, TOOL_HANDLERS)
                        _clear_indicator()
                        _print_tool_result(tc.name, result, enabled=self._settings.verbose_progress)
                        trace.record(
                            ExecutionMode.ACTING,
                            action=tool_call_dict,
                            observation=result if isinstance(result, dict) else {"result": str(result)},
                        )
                        result_text = json.dumps(result, default=str, ensure_ascii=False)[:3000]
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_text,
                        })

                    self._wm.remember_chat(build_assistant_message(
                        content or f"[Called: {', '.join(tc.name for tc in native_calls)}]"
                    ))
                    if status == BudgetStatus.SOFT_LIMIT:
                        messages.append(build_user_message_text(
                            "SYSTEM: Approaching limit. Provide final answer now."
                        ))
                    continue

                # No tool calls - final answer; yield content
                self._wm.remember_chat(build_assistant_message(content))
                trace.record(ExecutionMode.COMPLETE)
                if not content:
                    yield "I processed your request but have no additional output."
                else:
                    yield content
                return

            else:
                # Non-native path: use stream=True for real-time output
                chunks_collected: list[str] = []
                try:
                    async for chunk in self._llm.achat_stream(
                        compressed, enable_thinking=enable_thinking,
                    ):
                        chunks_collected.append(chunk)
                except Exception as exc:
                    _clear_indicator()
                    category = self._error_classifier.classify(exc)
                    recovery = self._error_classifier.get_recovery(category)
                    if recovery.retry and budget.remaining > 0:
                        await asyncio.sleep(jittered_backoff(budget.used))
                        continue
                    break
                _clear_indicator()

                content = "".join(chunks_collected).strip()
                self._wm.remember_chat(build_assistant_message(content))
                tool_call = self._parse_tool_call_from_content(content)

                if tool_call is None:
                    # Final answer — yield chunks for real-time display
                    trace.record(ExecutionMode.COMPLETE)
                    if not content:
                        yield "I processed your request but have no additional output."
                    else:
                        yield content
                    return

                # Execute text-parsed tool
                messages.append(build_assistant_message(content))
                _show_progress("executing", tool_call['name'])
                result = await self._execute_general_tool(tool_call, TOOL_HANDLERS)
                _clear_indicator()
                _print_tool_result(tool_call['name'], result, enabled=self._settings.verbose_progress)
                trace.record(
                    ExecutionMode.ACTING,
                    action=tool_call,
                    observation=result if isinstance(result, dict) else {"result": str(result)},
                )
                result_text = json.dumps(result, default=str, ensure_ascii=False)[:3000]
                messages.append(build_user_message_text(
                    f"Tool result ({tool_call['name']}):\n{result_text}"
                ))
                if status == BudgetStatus.SOFT_LIMIT:
                    messages.append(build_user_message_text(
                        "SYSTEM: Approaching limit. Provide final answer now."
                    ))

        # Budget exhausted
        yield content if content else "I've reached my processing limit."


    # ── Unified Loop Helpers ───────────────────────────────────────────────

    @staticmethod
    def _format_tool_catalog(tool_definitions: List[Dict[str, Any]]) -> str:
        """Format available tools for the unified system prompt."""
        lines: List[str] = []
        for td in tool_definitions:
            func = td.get("function", {})
            name = func.get("name", td.get("name", "unknown"))
            desc = func.get("description", td.get("description", ""))
            params = ", ".join(
                func.get("parameters", {}).get("properties", {}).keys()
            )
            lines.append(f"- **{name}**({params}): {desc}")
        return "\n".join(lines)

    @staticmethod
    def _parse_tool_call_from_content(content: str) -> Optional[Dict[str, Any]]:
        """Extract tool call from LLM response content.

        Reuses the robust parser from tool_executor.
        """
        from leapflow.skills.tool_executor import _parse_tool_call

        call = _parse_tool_call(content)
        if call:
            return {"name": call.name, "arguments": call.params}
        return None

    async def _execute_general_tool(
        self, tool_call: Dict[str, Any], handlers: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a general-purpose tool via ToolBridge (preferred) or TOOL_HANDLERS fallback.

        Routing priority:
        1. ToolBridge dispatch (gp_-prefixed) — local Python GP tools, always available
        2. ToolBridge dispatch (exact name) — may route to ExecutionPort or semantic tools
        3. TOOL_HANDLERS dict (static fallback when no bridge)
        """
        from leapflow.skills.tool_executor import ToolCall as TC

        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        # Route through ToolBridge when available (single source of truth)
        if self._tool_bridge is not None:
            # Try gp_-prefixed first (local GP tools — always work, even offline)
            prefixed = f"gp_{name}"
            result = await self._tool_bridge.dispatch(TC(name=prefixed, params=args))
            if not (isinstance(result, dict) and "unknown_tool" in str(result.get("error", ""))):
                return result
            # Try exact name (may route to ExecutionPort or semantic adapter tools)
            result = await self._tool_bridge.dispatch(TC(name=name, params=args))
            if not (isinstance(result, dict) and "unknown_tool" in str(result.get("error", ""))):
                return result

        # Fallback: direct handler dispatch (e.g., when bridge not configured)
        handler = handlers.get(name)
        if handler is None:
            return {"ok": False, "error": f"Unknown tool: {name}"}
        try:
            return await handler(args)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Helpers ──────────────────────────────────────────────────────────

    def _build_loop_messages(self, user_text: str, steps: List[str]) -> List[Dict[str, Any]]:
        """Build initial messages for the ReAct loop."""
        skill_catalog = self._registry.describe_with_params()

        # Append memory tool descriptions if available
        memory_tools_desc = ""
        if self._memory_manager and self._settings.memory_integration_enabled:
            schemas = self._memory_manager.get_tool_schemas()
            if schemas:
                tool_lines = []
                for s in schemas:
                    tool_lines.append(f"  - {s.name}: {s.description}")
                memory_tools_desc = (
                    "\n\nMEMORY TOOLS (type=\"memory\"):\n"
                    + "\n".join(tool_lines)
                )

        system_prompt = REACT_SYSTEM_TEMPLATE.format(skill_catalog=skill_catalog)
        if memory_tools_desc:
            system_prompt += memory_tools_desc

        return [
            build_system_message(system_prompt),
            build_user_message_text(
                "GOAL:\n"
                f"{user_text}\n\n"
                "CONTEXT:\n"
                f"- plan_steps: {json.dumps(steps, ensure_ascii=False)}\n"
            ),
        ]

    def _budget_exhausted_response(self, messages: List[Dict[str, Any]]) -> str:
        """Generate response when budget is exhausted."""
        return "I've reached my reasoning step limit. Here's my best answer based on progress so far."

    @staticmethod
    def _error_response(observation: Any) -> str:
        """Format error observation as user-facing response."""
        if isinstance(observation, dict):
            return f"Action failed: {observation.get('error', 'unknown error')}"
        return f"Action failed: {observation}"

    async def _emit_execution_trace(self, trace: ExecutionTrace) -> None:
        """Fire-and-forget: emit trace as learning signal for the evolution ring."""
        try:
            logger.debug(
                "emit_trace steps=%d tokens=%d", trace.step_count, trace.total_tokens
            )
            # Write episode to evolution memory if available
            if self._evolution and self._settings.memory_integration_enabled:
                actions = [
                    {"state": e.state.value, **(e.action or {})}
                    for e in trace.entries
                    if e.state == ExecutionMode.ACTING and e.action
                ]
                outcome = "success" if trace.success else "failure"
                reward = 1.0 if trace.success else -0.5
                self._evolution.record_episode(
                    skill_name="react_loop",
                    actions=actions,
                    outcome=outcome,
                    reward=reward,
                    context={"steps": trace.step_count, "tokens": trace.total_tokens},
                )
                logger.debug("evolution.record_episode outcome=%s actions=%d", outcome, len(actions))
        except Exception:
            pass  # never fail the main loop

    async def _sync_turn_safe(self, messages: List[Dict[str, Any]]) -> None:
        """Non-blocking wrapper for MemoryManager.sync_turn."""
        try:
            assert self._memory_manager is not None
            await asyncio.wait_for(
                self._memory_manager.sync_turn(messages),
                timeout=self._settings.memory_prefetch_timeout_s,
            )
            logger.debug("memory.sync_turn completed")
        except asyncio.TimeoutError:
            logger.debug("memory.sync_turn timed out")
        except Exception:
            logger.debug("memory.sync_turn failed", exc_info=True)

    async def _try_trigger_match(self, user_text: str) -> Optional[str]:
        """Check if a learned skill directly matches the user's request.

        Returns the skill output if a high-confidence match is found,
        or None to fall through to the ReAct/DAG path.
        """
        matches = self._registry.find_by_trigger(user_text, threshold=0.5)
        if not matches:
            return None

        best = matches[0]
        if best.metadata.source not in ("distilled", "template"):
            return None
        if best.metadata.confidence < 0.6:
            return None

        logger.info(
            "audit.trigger_match skill=%s confidence=%.2f",
            best.name, best.metadata.confidence,
        )
        result = await self._registry.invoke(best.name, user_goal=user_text)
        if result.ok:
            return str(result.output)
        logger.warning(
            "audit.trigger_match_failed skill=%s error=%s",
            best.name, result.error,
        )
        return None

    async def _handle_simple_intent(self, intent: Intent, user_text: str) -> str:
        """Dispatch simple intents to dedicated handlers.

        .. deprecated::
            This method is no longer called from run()/run_stream().
            All routing now goes through _unified_tool_loop() by default.
            Kept for potential future use as tool-handler backends.
        """
        if intent.label == "conversational":
            if not self._settings.has_llm_credentials:
                return "LeapFlow ready. Configure LEAPFLOW_LLM_API_KEY to enable full conversations."
            # Route conversational intent through unified tool loop
            return await self._unified_tool_loop(user_text)
        if intent.label == "file_organize":
            if not self._settings.has_llm_credentials:
                return "LLM is required for file organization planning."
            return await file_organizer.run(
                self._rpc, self._llm, self._wm, self._lt, user_goal=user_text
            )
        if intent.label == "clipboard":
            if not self._settings.has_llm_credentials:
                data = await self._rpc.call(Methods.CLIPBOARD_GET, {})
                return str(data.get("text", "") or "(empty)")
            return await clipboard_manager.run(
                self._rpc, self._llm, self._wm, self._lt, user_goal=user_text
            )
        if intent.label in ("app_automation", "desktop_action"):
            if self._execution:
                return await self._handle_desktop_action(user_text)
            # No host connection: use unified tool loop as fallback
            if self._settings.has_llm_credentials:
                return await self._unified_tool_loop(user_text)
            if intent.label == "app_automation":
                return await app_launcher.run(self._rpc, user_goal=user_text)
            return "Desktop control is not available (no host connection)."
        if intent.label == "memory_recent":
            return await self._handle_memory_recent(user_text)
        if intent.label == "file_search":
            if not self._settings.has_llm_credentials:
                kws = _keywords_from_query(user_text)
                hits = self._lt.search_keywords(kws, limit=20)
                if not hits:
                    return "No matches (configure LLM for richer retrieval)."
                return "\n".join([f"- {h.content}" for h in hits[:20]])
            kws = _keywords_from_query(user_text)
            hits = self._lt.search_keywords(kws, limit=25)
            context = [{"content": h.content, "path": h.path, "score": h.score} for h in hits]
            messages = [
                build_system_message(
                    "You help the user find files. Use MEMORY_HITS; if insufficient, say what's missing."
                ),
                build_user_message_text(
                    f"Query:\n{user_text}\n\nMEMORY_HITS:\n{json.dumps(context, ensure_ascii=False)}"
                ),
            ]
            resp = await self._llm.achat(messages, stream=False, enable_thinking=False)
            return (resp.content or "").strip()

        if intent.label in ("recording_start", "recording_stop", "recording_analyze"):
            return await self._handle_recording_intent(intent, user_text)

        if intent.label in ("learn_start", "learn_stop", "learn_pause", "learn_resume", "learn_annotate"):
            return await self._handle_learn_intent(intent, user_text)

        if intent.label == "skill_list":
            return self._handle_skill_list()

        if intent.label == "skill_execute":
            return await self._handle_skill_execute(user_text)

        if intent.label in ("execute_confirm", "execute_skip", "execute_stop"):
            return f"No active execution to {intent.label.split('_')[1]}."

        if intent.label == "skill_review":
            return self._handle_skill_review()

        if intent.label == "skill_approve":
            return await self._handle_skill_approve(user_text)

        raise RuntimeError(f"Unhandled intent label: {intent.label}")

    async def _handle_desktop_action(self, user_text: str) -> str:
        if not self._execution:
            return "Desktop control is not available (no host connection)."
        if not self._settings.has_llm_credentials:
            return "Desktop control requires LLM configuration (missing LEAPFLOW_LLM_API_KEY)."

        from leapflow.skills.bridge_factory import build_tool_bridge
        from leapflow.skills.tool_executor import ToolUseSkillExecutor

        # Reuse pre-built bridge (with GP tools) or build a fresh one
        bridge = self._tool_bridge if self._tool_bridge else build_tool_bridge(self._execution, self._perception)
        from leapflow.engine.budget import BudgetConfig

        executor = ToolUseSkillExecutor(
            llm=self._llm,
            bridge=bridge,
            skill_content="",
            instructions=[user_text],
            vlm=self._vlm,
            skill_name="chat_desktop_action",
            step_timeout_s=120.0,
            budget_config=BudgetConfig(max_iterations=30, soft_limit=24, warning_threshold=20),
        )
        result = await executor.run(user_goal=user_text)
        self._wm.remember_event("desktop_action", result[:200], {})
        return result

    async def _handle_memory_recent(self, user_text: str) -> str:
        """Answer questions about recent activity using memory + optional LLM."""
        events = self._collect_recent_events()

        if not events:
            return "No recent activity records in memory."

        for f in self._imm.recent(limit=50):
            self._imm.touch(f.fragment_id)

        if self._settings.has_llm_credentials:
            return await self._synthesize_memory_answer(user_text, events)

        return self._format_recent_events(events)

    def _collect_recent_events(self) -> List[Dict[str, Any]]:
        """Gather events from immediate memory, dedup by (path, action)."""
        frags = self._imm.recent(limit=50)
        if not frags:
            hits = self._lt.recent_file_events(within_seconds=3600)
            return [
                {
                    "ts": h.created_at,
                    "time": datetime.fromtimestamp(h.created_at).strftime("%H:%M:%S"),
                    "type": h.kind,
                    "content": h.content,
                    "path": h.path or "",
                }
                for h in hits[:30]
            ]

        seen: Dict[str, Dict[str, Any]] = {}
        for f in frags:
            key = f"{f.event_type}:{f.path or f.content}"
            if key not in seen or f.created_at > seen[key]["ts"]:
                seen[key] = {
                    "ts": f.created_at,
                    "time": datetime.fromtimestamp(f.created_at).strftime("%H:%M:%S"),
                    "type": f.event_type,
                    "content": f.content,
                    "path": f.path or "",
                }
        result = sorted(seen.values(), key=lambda e: e["ts"], reverse=True)
        return result

    async def _synthesize_memory_answer(
        self, user_text: str, events: List[Dict[str, Any]]
    ) -> str:
        """Use LLM to answer the user's question based on collected events."""
        events_json = json.dumps(events, ensure_ascii=False)
        messages = [
            build_system_message(
                "You are LeapFlow's memory assistant. "
                "Given a list of recent system events (file changes, clipboard, app focus, etc.), "
                "answer the user's question accurately and concisely.\n"
                "Rules:\n"
                "- Filter events relevant to the user's question (time range, file type, etc.)\n"
                "- Skip obvious system/background noise (databases, caches, logs)\n"
                "- Include timestamps when the user asks for them\n"
                "- If no relevant events match, say so clearly\n"
                "- Answer in the same language as the user's question"
            ),
            build_user_message_text(
                f"Question: {user_text}\n\n"
                f"Recent events ({len(events)} total):\n{events_json}"
            ),
        ]
        try:
            resp = await self._llm.achat(messages, stream=False, enable_thinking=False)
            answer = (resp.content or "").strip()
            if answer:
                return answer
        except Exception:
            logger.warning("LLM synthesis failed for memory_recent", exc_info=True)
        return self._format_recent_events(events)

    @staticmethod
    def _format_recent_events(events: List[Dict[str, Any]]) -> str:
        """Fallback formatting when LLM is unavailable."""
        lines = [f"Recent activity ({len(events)} events):\n"]
        for e in events[:30]:
            lines.append(f"- {e['time']} [{e['type']}] {e['content']}")
        return "\n".join(lines)

    async def _handle_recording_intent(self, intent: Intent, user_text: str) -> str:
        """Handle recording-related intents (start/stop/analyze)."""
        if self._imitation is None:
            return "Imitation learning is not configured."

        if intent.label == "recording_start":
            tid = await self._imitation.start_recording()
            return f"Recording started. Trajectory ID: {tid}"

        if intent.label == "recording_stop":
            traj = await self._imitation.stop_recording()
            if traj is None:
                return "No active recording to stop."
            return (
                f"Recording stopped. Trajectory: {traj.trajectory_id}\n"
                f"Steps: {traj.step_count} | Duration: {traj.duration:.1f}s\n"
                f"Apps: {', '.join(traj.app_sequence) or 'none'}"
            )

        if intent.label == "recording_analyze":
            trajs = self._imitation.list_trajectories(limit=1)
            if not trajs:
                return "No trajectories found. Start a recording first."
            tid = trajs[0]["id"]
            candidates = await self._imitation.distill(tid)
            if not candidates:
                replay = self._imitation.format_trajectory(tid)
                return f"No skill candidates found.\n\nTrajectory replay:\n{replay}"
            lines = [f"Distilled {len(candidates)} skill candidate(s) from trajectory {tid}:\n"]
            for c in candidates:
                lines.append(f"  - {c.title} (confidence: {c.confidence:.2f})")
                lines.append(f"    Steps: {' → '.join(c.steps[:5])}")
                if c.trigger_phrases:
                    lines.append(f"    Triggers: {', '.join(c.trigger_phrases[:3])}")
            return "\n".join(lines)

        return "Unknown recording command."

    async def _handle_learn_intent(self, intent: Intent, user_text: str) -> str:
        if self._session is None:
            return "Session controller is not configured."

        if intent.label == "learn_start":
            try:
                session = await self._session.enter_learning(goal=user_text)
                return (
                    f"Learning started. Session: {session.session_id}\n"
                    f"Trajectory: {session.trajectory_id}\n"
                    "Perform the task you want me to learn. Say 'stop learning' when done."
                )
            except Exception as e:
                return f"Cannot start learning: {e}"

        if intent.label == "learn_stop":
            try:
                result = await self._session.exit_learning()
                lines = [
                    f"Learning stopped. Trajectory: {result.trajectory_id}",
                    f"Steps: {result.step_count} | Duration: {result.duration:.1f}s",
                ]
                if result.new_skills:
                    lines.append(f"New skills learned: {', '.join(result.new_skills)}")
                if result.suggestions > 0:
                    lines.append(f"Suggestions pending: {result.suggestions}")
                return "\n".join(lines)
            except Exception as e:
                return f"Cannot stop learning: {e}"

        if intent.label == "learn_pause":
            self._session.pause_learning()
            return "Learning paused. Say 'resume learning' to continue."

        if intent.label == "learn_resume":
            self._session.resume_learning()
            return "Learning resumed."

        if intent.label == "learn_annotate":
            self._session.annotate(user_text)
            return "Annotation added."

        return "Unknown learning command."

    def _handle_skill_list(self) -> str:
        skills = self._registry.list_all()
        if not skills:
            return "No skills registered."
        lines = [f"Registered skills ({len(skills)}):\n"]
        for s in skills:
            meta = s.metadata
            lines.append(
                f"  - {s.name} (v{meta.version}, {meta.confidence:.0%}) "
                f"— {s.description[:60]}"
            )
        return "\n".join(lines)

    async def _handle_skill_execute(self, user_text: str) -> str:
        if self._session is None:
            triggered = await self._try_trigger_match(user_text)
            return triggered or "No matching skill found."

        skill_name = self._session.find_skill(user_text)
        if skill_name is None:
            return "No matching skill found for your request."

        result = await self._session.execute_skill(skill_name)
        if result.ok:
            return f"Skill '{result.skill_name}' executed successfully.\n{result.output or ''}"
        return f"Skill '{result.skill_name}' failed: {result.error}"

    # ── Learn Command Detection ─────────────────────────────────────────

    # Patterns that indicate a genuine learn/teach session command.
    # Uses regex word-boundary checks to avoid false positives like
    # "learn about python" or "teaching methods for math".
    _LEARN_COMMAND_RE = re.compile(
        r"^(?:"
        r"(?:start\s+)?learn(?:ing)?(?:\s+(?:this|that|it|now))?$"
        r"|(?:start\s+)?teach(?:ing)?(?:\s+(?:this|that|it|me|now))?$"
        r"|stop\s+(?:learn|teach)(?:ing)?"
        r"|pause\s+learn(?:ing)?"
        r"|resume\s+learn(?:ing)?"
        r"|done\s+learn(?:ing)?"
        r"|finish\s+learn(?:ing)?"
        r"|end\s+learn(?:ing)?"
        r"|教(?:我|一下)?$"
        r"|学习(?:这个|一下)?$"
        r"|开始(?:学习|教学)"
        r"|停止学习|暂停学习|继续学习|结束学习"
        r"|watch\s+me"
        r")",
        re.IGNORECASE,
    )

    def _is_learn_command(self, text: str) -> bool:
        """Check if text is a learn/teach command that needs special session handling.

        Uses regex matching to avoid false positives like 'learn about python'
        or 'teach me how to cook' which should go through the unified tool loop.
        """
        stripped = text.strip()
        return bool(self._LEARN_COMMAND_RE.match(stripped))

    async def _handle_learn_command(self, user_text: str) -> str:
        """Route learn/teach commands through intent classifier for sub-intent dispatch."""
        intent = await self._classifier.classify(user_text)
        logger.debug("learn.classify label=%s reason=%s", intent.label, intent.reason)

        if intent.label in (
            "learn_start", "learn_stop", "learn_pause",
            "learn_resume", "learn_annotate",
        ):
            return await self._handle_learn_intent(intent, user_text)

        # Not actually a learn command after classification — fall through to unified loop
        return await self._unified_tool_loop(user_text)

    def _inject_pending_skill_reminder(self) -> None:
        if self._skill_library is None:
            return
        n = self._skill_library.count_pending()
        if n > 0:
            self._wm.remember_event(
                "skill_suggestion_reminder",
                f"[{n} skill update suggestion(s) pending review — "
                f"say 'review skill suggestions']",
            )

    def _handle_skill_review(self) -> str:
        if self._skill_library is None:
            return "Skill library is not configured."
        suggestions = self._skill_library.load_pending_suggestions(limit=10)
        if not suggestions:
            return "No pending skill update suggestions."
        lines = [f"Pending skill suggestions ({len(suggestions)}):\n"]
        for i, s in enumerate(suggestions, 1):
            details = s.similarity_details
            rationale = details.get("llm_rationale", "")
            changes = s.proposed_changes
            lines.append(
                f"  {i}. \"{s.existing_skill_title}\" "
                f"(similarity: {s.similarity_score:.0%})"
            )
            if rationale:
                lines.append(f"     LLM: {rationale}")
            new_steps = changes.get("new_steps", [])
            new_triggers = changes.get("new_triggers", [])
            if new_steps:
                lines.append(f"     +steps: {', '.join(new_steps[:3])}")
            if new_triggers:
                lines.append(f"     +triggers: {', '.join(new_triggers[:3])}")
        lines.append("\nSay 'approve <number>' or 'reject <number>' to act.")
        return "\n".join(lines)

    async def _handle_skill_approve(self, user_text: str) -> str:
        if self._skill_library is None:
            return "Skill library is not configured."
        suggestions = self._skill_library.load_pending_suggestions(limit=20)
        if not suggestions:
            return "No pending suggestions to approve or reject."

        action, indices = await self._parse_approval(user_text, suggestions)

        results: list[str] = []
        for idx in indices:
            if idx < 0 or idx >= len(suggestions):
                results.append(f"Index {idx + 1} out of range.")
                continue
            s = suggestions[idx]
            if action == "approve":
                merged = self._skill_merger.apply(s, self._skill_library)
                results.append(
                    f"Approved: \"{s.existing_skill_title}\" → v{merged.version}"
                )
            else:
                self._skill_library.resolve_suggestion(
                    s.suggestion_id, "rejected"
                )
                results.append(f"Rejected: \"{s.existing_skill_title}\"")
        return "\n".join(results)

    async def _parse_approval(
        self, user_text: str, suggestions: list
    ) -> tuple[str, list[int]]:
        text_lower = user_text.lower()
        is_approve = any(
            w in text_lower for w in ("approve", "accept", "yes", "批准", "接受")
        )
        is_reject = any(
            w in text_lower for w in ("reject", "deny", "no", "拒绝")
        )
        action = "approve" if is_approve else ("reject" if is_reject else "approve")

        if "all" in text_lower or "全部" in text_lower:
            return action, list(range(len(suggestions)))

        nums = re.findall(r"\d+", user_text)
        indices = [int(n) - 1 for n in nums if 0 < int(n) <= len(suggestions)]
        if not indices:
            indices = [0]
        return action, indices

    async def _execute_action(self, action: Dict[str, Any], user_goal: str) -> Any:
        a_type = str(action.get("type", "")).strip()
        name = str(action.get("name", "")).strip()
        payload = dict(action.get("payload") or {})

        # Memory tool interception: route memory_* calls to MemoryManager
        if (a_type == "memory" or name.startswith("memory_")) and self._memory_manager:
            tool_name = name if name.startswith("memory_") else f"memory_{name}"
            try:
                result = await self._memory_manager.handle_tool_call(tool_name, payload)
                logger.info("audit.memory_tool name=%s", tool_name)
                return {"ok": True, "result": result}
            except Exception as exc:
                return {"ok": False, "error": f"memory_tool_failed: {exc}"}

        if a_type == "skill":
            result = await self._registry.invoke(
                name, user_goal=user_goal, **payload,
            )
            if not result.ok:
                return {"ok": False, "error": result.error}
            logger.info("audit.skill name=%s ok", name)
            return {"ok": True, "result": result.output}

        if a_type == "bridge":
            method = str(payload.pop("method", "")).strip()
            if not method:
                return {"ok": False, "error": "missing_method"}
            pl = self._registry.prediction_loop
            if pl is not None and pl.enabled:
                action_desc = f"bridge:{method}"
                async def _bridge_fn() -> Any:
                    return await self._rpc.call(method, payload or None)
                output, _ = await pl.wrap_execution(action_desc, _bridge_fn)
                logger.info("audit.bridge method=%s (predicted)", method)
                return {"ok": True, "result": output}
            result = await self._rpc.call(method, payload or None)
            logger.info("audit.bridge method=%s", method)
            return {"ok": True, "result": result}

        if a_type == "tool":
            from leapflow.tools.registry_bootstrap import TOOL_HANDLERS
            tool_call_dict = {"name": name, "arguments": payload}
            result = await self._execute_general_tool(tool_call_dict, TOOL_HANDLERS)
            logger.info("audit.tool name=%s ok=%s", name, result.get("ok"))
            return result

        return {"ok": False, "error": f"unsupported_action:{a_type}"}


def build_default_registry(rpc: HostRpc, llm: LLMProvider, wm: WorkingMemoryProvider, lt: SemanticMemoryProvider) -> SkillRegistry:
    """Register built-in skills with closures (dependency injection)."""

    reg = SkillRegistry()

    async def _file_organizer(goal: str, **_kwargs: Any) -> str:
        return await file_organizer.run(rpc, llm, wm, lt, user_goal=goal)

    async def _clipboard(goal: str, **_kwargs: Any) -> str:
        return await clipboard_manager.run(rpc, llm, wm, lt, user_goal=goal)

    async def _app_launch(goal: str, **_kwargs: Any) -> str:
        return await app_launcher.run(rpc, user_goal=goal)

    reg.register(
        Skill(
            name="file_organizer",
            description="Organize PDFs/files using LLM plan + RPC file moves.",
            run=_file_organizer,
        )
    )
    reg.register(
        Skill(
            name="clipboard_manager",
            description="Summarize clipboard and store durable memory.",
            run=_clipboard,
        )
    )
    reg.register(
        Skill(
            name="app_launcher",
            description="Launch/activate apps and request simple automation actions.",
            run=_app_launch,
        )
    )
    return reg
