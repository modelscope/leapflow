# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tool execution engine — extracted from :class:`AgentEngine`.

Phase 5 refactor. This component owns all tool execution, dispatch, catalog
assembly, and tool-failure/guardrail evaluation. It holds a back-reference to
the owning engine so every access reads the engine's *live* mutable state,
preserving exact runtime semantics.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from leapflow.llm.message_builder import build_user_message_text
from leapflow.engine.context.context_disclosure import build_capability_manifests
from leapflow.engine.tools.execution_trace import ExecutionMode, ExecutionTrace
from leapflow.engine.tools.tool_concurrency import ToolCall as ConcurrentToolCall
from leapflow.engine.tools.tool_execution import ToolExecutionLedger, execution_policy_for
from leapflow.engine.recovery.recovery_audit import create_audit_entry
from leapflow.engine.recovery.recovery_decision import RecoveryAction, RecoveryDecision
from leapflow.engine.recovery.failure_envelope import Recoverability
from leapflow.tools.name_resolver import ToolResolution
from leapflow.engine._tool_helpers import _default_tool_registry, _normalize_tool_call
from leapflow.engine._message_helpers import (
    _truncate_result_for_budget,
    _tool_result_counts_as_failure,
    _tool_result_is_control_signal,
    _annotate_uncertain_effect,
    _should_stop_after_tool_result,
    _validate_tool_arguments,
    _skipped_after_failure_result,
    _show_progress,
    _clear_indicator,
    _print_tool_result,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from leapflow.engine.engine import AgentEngine

logger = logging.getLogger(__name__)


class ToolDispatchEngine:
    """Handles all tool execution, dispatch, and catalog management."""

    def __init__(self, engine: "AgentEngine") -> None:
        self._engine = engine

    def _check_guardrail(
        self,
        messages: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Run guardrail check. Returns 'halt' if loop should stop, else None."""
        if self._engine._guardrail is None:
            return None
        violation = self._engine._guardrail.check(messages)
        if not violation.violated:
            return None
        logger.warning("guardrail: %s", violation.reason)
        # Progress-aware: while the task is still advancing (stall counter at 0),
        # a detected repetition/domination is producing progress -> never halt,
        # and the finalize/diversify nudge is suppressed so legitimate batch or
        # sequential work on a long task is not cut short. Only when the task is
        # ALSO stalled does the guardrail escalate to a halt (or emit a nudge).
        #
        # The one exception is a ``progress_independent`` halt: it is raised only
        # when the violation is definitionally zero progress (the same tool
        # returned the same result N times), so it is honoured regardless of the
        # coarse global stall marker -- which a simple factual query may never
        # trip, leaving a genuine no-op loop to spin until the budget is spent.
        frame = self._engine._active_frame
        stalled = bool(frame is not None and getattr(frame, "stalled_rounds", 0) >= 1)
        if violation.severity == "halt" and (
            getattr(violation, "progress_independent", False) or stalled
        ):
            messages.append(
                build_user_message_text(
                    f"SYSTEM GUARDRAIL: {violation.reason}. {violation.suggestion}"
                )
            )
            return "halt"
        if not stalled:
            return None  # productive: neither halt nor nudge
        messages.append(
            build_user_message_text(f"SYSTEM WARNING: {violation.reason}. {violation.suggestion}")
        )
        return None
    def _evaluate_tool_failures(
        self,
        failed_items: List[tuple[str, Dict[str, Any]]],
        *,
        turn_id: int,
    ) -> Optional[str]:
        """Single recovery decision point for tool-result failures.

        A tool failure is an OBSERVATION for autonomous diagnosis: the failed
        result is already in the message history and is fed back to the LLM,
        which reasons about it and retries or changes approach on the next round.
        There is NO blanket count-based break — a task that fails then fixes keeps
        going; a genuinely stuck failure loop is bounded by the iteration budget,
        progress-based stall detection, and the progress-aware guardrail.

        Each failure is classified into a FailureEnvelope. The turn halts ONLY
        for a non-recoverable failure (e.g. permission denied), routed through
        the coordinator for the terminal decision + audit. Recoverable failures
        are fed back and audited as a zero-cost decision so they never spend the
        system recovery budget (reserved for infrastructure recovery). Returns a
        halt reason when the turn must stop, else None.
        """
        coordinator = self._engine._recovery_coordinator
        if coordinator is None:
            return None
        session_id = getattr(self._engine, "_current_session_id", "") or ""
        for tool_name, result in failed_items:
            if not isinstance(result, dict):
                continue
            envelope = self._engine._unified_classifier.classify_tool_result(
                result,
                tool_name=tool_name,
                execution_policy=result.get("execution_policy", "read_only"),
            )
            if envelope is None:
                continue
            if envelope.recoverability == Recoverability.NON_RECOVERABLE:
                decision = coordinator.evaluate(envelope)
                self._engine._audit_sink.record(
                    create_audit_entry(
                        envelope,
                        decision,
                        coordinator.budget,
                        session_id=session_id,
                        turn_id=turn_id,
                    )
                )
                return decision.reason or f"Non-recoverable tool failure ({envelope.category})"
            # Recoverable: fed back to the agent (zero-cost, no recovery budget spent).
            feedback = RecoveryDecision.create(
                envelope=envelope,
                action=RecoveryAction.SKIP_AND_CONTINUE,
                reason="Tool failure fed back to the agent for autonomous diagnosis and retry",
                strategy_key="tool_feedback",
                budget_cost=0,
            )
            self._engine._audit_sink.record(
                create_audit_entry(
                    feedback.envelope,
                    feedback,
                    coordinator.budget,
                    session_id=session_id,
                    turn_id=turn_id,
                )
            )
        return None
    def _tool_execution_metadata_with_focus(
        self,
        tool_name: str,
        arguments: Dict[str, Any] | None,
        result: Any,
    ) -> Dict[str, Any]:
        """Merge existing execution metadata with semantic-focus metadata."""
        metadata = self._tool_execution_metadata(result)
        metadata.update(self._engine._learning_bridge._tool_focus_metadata(tool_name, arguments, result))
        return metadata
    @staticmethod
    def _expand_tools_kwarg_full(
        tools_kwarg: Dict[str, Any], tool_definitions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Expand this turn's native tool schema to the full catalog.

        Structural failure-recovery gate: once an unknown_tool result proves
        that this turn's disclosed subset was insufficient, escalate to the
        full catalog immediately rather than guessing a smaller subset again.
        """
        return {"tools": list(tool_definitions)}
    @staticmethod
    def _merge_expanded_tool_schemas(
        tools_kwarg: Dict[str, Any],
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge capability_expand results into this turn's native tool schema.

        Tier 1 model-initiated discovery gate: when the model calls
        capability_expand and it succeeds, the returned tool schemas become
        callable for the rest of this turn.
        """
        additions: List[Dict[str, Any]] = []
        for item in results:
            result = item.get("result")
            if isinstance(result, dict) and result.get("ok") and result.get("expanded_tools"):
                additions.extend(result["expanded_tools"])
        if not additions:
            return tools_kwarg
        existing = list(tools_kwarg.get("tools") or [])
        existing_names = {td.get("function", {}).get("name") for td in existing}
        for td in additions:
            name = td.get("function", {}).get("name")
            if name and name not in existing_names:
                existing.append(td)
                existing_names.add(name)
        return {"tools": existing}
    def _compact_tool_result(
        self, tool_name: str, arguments: Dict[str, Any] | None, result: Any
    ) -> Any:
        """Return compact tool evidence for LLM replay."""
        return self._engine._context_governance_controller.compact_tool_result(tool_name, arguments, result)
    def _tool_context_metadata(
        self,
        tool_name: str,
        arguments: Dict[str, Any] | None,
        result: Any,
    ) -> Dict[str, Any]:
        """Return additional UI metadata from adaptive context handling."""
        metadata = self._engine._context_governance_controller.tool_metadata(tool_name, arguments, result)
        snapshot = self._engine._last_context_snapshot
        if snapshot:
            posture = snapshot.get("context_posture")
            if posture and posture != "baseline":
                metadata.setdefault("context_posture", posture)
            signal = snapshot.get("context_signal")
            if signal:
                metadata.setdefault("context_signal", signal)
            guidance = snapshot.get("context_guidance")
            if guidance:
                metadata.setdefault("context_guidance", guidance)
            disclosure_level = snapshot.get("disclosure_level")
            if disclosure_level:
                metadata.setdefault("disclosure_level", disclosure_level)
            disclosure_reason = snapshot.get("disclosure_reason")
            if disclosure_reason:
                metadata.setdefault("disclosure_reason", disclosure_reason)
            trace = snapshot.get("compression_trace")
            if isinstance(trace, dict) and trace.get("stages_applied"):
                metadata.setdefault("compression_stages", trace.get("stages_applied"))
                metadata.setdefault("compression_savings_ratio", trace.get("savings_ratio", 0.0))
                metadata.setdefault("compression_saved_tokens", trace.get("saved_tokens", 0))
                metadata.setdefault("compression_reason", trace.get("decision_reason", ""))
            if snapshot.get("forced_final_answer"):
                metadata.setdefault("context_posture", "finalizing")
        return metadata
    def _semantic_tool_schemas(self) -> List[Dict[str, Any]]:
        """Callable schemas for the semantic desktop tools from the desktop plugin.

        The plugin is a process singleton, so it is re-resolved from the tool
        registry on every read — a disabled/unregistered plugin (plugin_disable,
        fiber dispose) yields zero schemas immediately and never serves a
        stale cache entry. Cached on (plugin identity, version): identity makes
        a reloaded instance (version counter restarting at 0) always miss the
        predecessor's cache entry; version catches hot-swapped perception
        ports and re-activation of the same instance.
        """
        from leapflow.plugins import get_registry

        _plugin_registry = get_registry()

        dp = _plugin_registry.get_desktop_semantic_plugin()
        if dp is None or not dp.active:
            return []
        cache_key = (id(dp), dp.version)
        if self._engine._semantic_plugin_key != cache_key:
            self._engine._semantic_schemas = dp.get_semantic_schemas()
            self._engine._semantic_plugin_key = cache_key
        return self._engine._semantic_schemas
    def _unified_tool_catalog(self) -> List[Dict[str, Any]]:
        """Per-turn tool catalog: static registry plus live semantic schemas.

        Cached on (desktop plugin identity+version, static-registry size): the
        registry is append-only (session_search, platform schemas land after
        engine construction), so a length change invalidates exactly like a
        plugin disable or reload does.
        """
        from leapflow.plugins import get_registry

        _plugin_registry = get_registry()

        dp = _plugin_registry.get_desktop_semantic_plugin()
        dp_key = (id(dp), dp.version) if dp is not None else None
        cache_key = (dp_key, len(_plugin_registry.tool_definitions))
        if self._engine._unified_catalog_key != cache_key:
            self._engine._unified_catalog = (
                list(_plugin_registry.tool_definitions) + self._semantic_tool_schemas()
            )
            self._engine._unified_catalog_key = cache_key
            # Downstream caches are keyed on the catalog contents.
            self._engine._manifests_by_name = None
            self._engine._full_tools_tokens = None
        return self._engine._unified_catalog
    def _unified_tool_handlers(self) -> Dict[str, Any]:
        """Per-turn handler table: static handlers plus desktop semantic handlers.

        The desktop plugin is re-resolved from the tool registry on every read,
        so a disabled or reloaded plugin swaps the semantic handler entries on
        the very next call. Returns a fresh dict() copy of the plugin registry's
        handlers, giving each turn an isolated snapshot. Plugin reloads during a
        turn do not affect the turn in progress — it keeps using its own
        snapshot until completion. New turns starting after a reload pick up
        the new handlers.
        """
        from leapflow.plugins import get_registry

        _plugin_registry = get_registry()

        handlers: Dict[str, Any] = _plugin_registry.snapshot_handlers()
        dp = _plugin_registry.get_desktop_semantic_plugin()
        if dp is not None and dp.active:
            handlers.update(dp.get_semantic_handlers())
        return handlers
    async def _approve_desktop_action(self, name: str, args: Any) -> tuple[bool, str]:
        """Consult the desktop approval gate before a mutating semantic tool.

        Fail-closed: a missing gate or a failed evaluation blocks the action,
        mirroring the dangerous-command gate in shell_tools.
        """
        from leapflow.skills.semantic_schema import semantic_requires_approval

        if not semantic_requires_approval(name):
            return True, ""
        from leapflow.plugins import get_registry

        _plugin_registry = get_registry()

        gate = _plugin_registry.get_desktop_gate()
        if gate is None:
            return False, f"Desktop action '{name}' blocked: no approval gate configured"
        try:
            from leapflow.security.actions import ActionDescriptor

            payload = args if isinstance(args, dict) else {}
            result = await gate.evaluate(ActionDescriptor.platform_action("desktop", name, payload))
            if getattr(result, "approved", False):
                return True, ""
            message = str(
                getattr(result, "denial_message", "")
                or f"Desktop action '{name}' requires approval (denied)"
            )
            return False, message
        except Exception:
            logger.debug("desktop approval check failed", exc_info=True)
            return False, f"Desktop action '{name}' requires approval (denied)"
    @staticmethod
    def _format_tool_catalog(tool_definitions: List[Dict[str, Any]]) -> str:
        """Format available tools for the unified system prompt.

        Each non-core tool is annotated with its exact capability_expand category
        so the model never has to guess the category string — it reads it directly
        from the index, matching this turn's real manifest classification.
        """
        manifests = {m.name: m for m in build_capability_manifests(tool_definitions)}
        lines: List[str] = []
        for td in tool_definitions:
            func = td.get("function", {})
            name = func.get("name", td.get("name", "unknown"))
            desc = func.get("description", td.get("description", ""))
            params = ", ".join(func.get("parameters", {}).get("properties", {}).keys())
            manifest = manifests.get(name)
            tag = (
                f" [capability_expand category: {manifest.category}]"
                if manifest is not None and not manifest.is_core
                else ""
            )
            lines.append(f"- **{name}**({params}){tag}: {desc}")
        return "\n".join(lines)
    @staticmethod
    def _parse_tool_call_from_content(content: str) -> Optional[Dict[str, Any]]:
        """Extract tool call from LLM response content.

        Reuses the robust parser from tool_call_parser.
        """
        from leapflow.skills.tool_call_parser import parse_tool_call

        call = parse_tool_call(content)
        if call:
            return {"name": call.name, "arguments": call.params}
        return None
    async def _execute_tools_concurrent(
        self,
        native_calls: list,
        handlers: Dict[str, Any],
        *,
        trace: ExecutionTrace,
        messages: List[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        """Execute native tool calls respecting concurrency policy.

        Concurrent group runs via asyncio.gather; sequential group runs one-by-one.
        Results are appended to messages in OpenAI tool-result format and returned
        for streaming UI metadata.
        """
        result_budget = self._engine._effective_tool_result_budget()
        executed: list[Dict[str, Any]] = []
        original_names_by_id = {str(tc.id): str(tc.name) for tc in native_calls}

        tc_wrappers = [
            ConcurrentToolCall(
                id=tc.id,
                name=str(
                    _normalize_tool_call({"name": tc.name, "arguments": tc.arguments})["name"]
                ),
                arguments=tc.arguments,
            )
            for tc in native_calls
        ]

        if not self._engine._concurrency_policy or len(tc_wrappers) <= 1:
            for i, tc in enumerate(native_calls):
                original_name = str(tc.name)
                tool_call_dict = _normalize_tool_call(
                    {"name": original_name, "arguments": tc.arguments}
                )
                normalized_name = str(tool_call_dict["name"])
                self._engine._learning_bridge._emit_chat_event(
                    "tool_call",
                    {
                        "tool_name": normalized_name,
                        "arguments_summary": json.dumps(
                            tc.arguments, default=str, ensure_ascii=False
                        )[:300],
                    },
                )
                _show_progress("executing", normalized_name, step=i + 1, total=len(native_calls))
                result = await self._execute_tool_with_ledger(
                    tool_call_dict,
                    handlers,
                    tool_call_id=str(tc.id),
                )
                _clear_indicator()
                self._engine._learning_bridge._emit_chat_event(
                    "tool_result",
                    {
                        "tool_name": normalized_name,
                        "ok": bool(result.get("ok")) if isinstance(result, dict) else True,
                        "summary": json.dumps(result, default=str, ensure_ascii=False)[:300]
                        if isinstance(result, dict)
                        else str(result)[:300],
                    },
                )
                _print_tool_result(normalized_name, result, enabled=self._engine._settings.verbose_progress)
                trace.record(
                    ExecutionMode.ACTING,
                    action=tool_call_dict,
                    observation=result if isinstance(result, dict) else {"result": str(result)},
                )
                self._engine._learning_bridge._record_tool_focus(normalized_name, tc.arguments, result)
                result_payload = self._compact_tool_result(normalized_name, tc.arguments, result)
                result_text = _truncate_result_for_budget(result_payload, result_budget)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})
                self._engine._session_persistence._persist_message(
                    self._engine._current_session_id,
                    "tool",
                    result_text,
                    tool_name=normalized_name,
                    tool_call_id=str(tc.id),
                    metadata=self._tool_execution_metadata_with_focus(
                        normalized_name, tc.arguments, result
                    ),
                )
                executed.append(
                    {
                        "id": tc.id,
                        "name": normalized_name,
                        "original_tool_name": str(
                            tool_call_dict.get("original_tool_name") or original_name
                        ),
                        "arguments": tc.arguments,
                        "result": result,
                    }
                )
                if isinstance(result, dict) and _should_stop_after_tool_result(
                    normalized_name, result
                ):
                    for skipped_tc in native_calls[i + 1 :]:
                        skipped_call = _normalize_tool_call(
                            {"name": str(skipped_tc.name), "arguments": skipped_tc.arguments}
                        )
                        skipped_name = str(skipped_call["name"])
                        skipped_result = _skipped_after_failure_result(normalized_name, result)
                        self._append_skipped_tool_message(
                            skipped_tc.id,
                            skipped_name,
                            skipped_result,
                            messages=messages,
                            result_budget=result_budget,
                        )
                        executed.append(
                            {
                                "id": skipped_tc.id,
                                "name": skipped_name,
                                "original_tool_name": str(
                                    skipped_call.get("original_tool_name") or skipped_tc.name
                                ),
                                "arguments": skipped_tc.arguments,
                                "result": skipped_result,
                            }
                        )
                    logger.info(
                        "tool_concurrency: stopping remaining native tool calls after failed side effect from %s",
                        normalized_name,
                    )
                    break
            return executed

        concurrent, sequential = self._engine._concurrency_policy.partition(tc_wrappers)
        logger.info(
            "tool_concurrency.execute concurrent=%d sequential=%d",
            len(concurrent),
            len(sequential),
        )

        # Execute concurrent group via asyncio.gather, bounded so a large batch
        # does not fan out unbounded IO/subprocess load.
        if concurrent:
            max_parallel = max(1, int(getattr(self._engine._settings, "agent_max_parallel_tools", 8) or 8))
            _parallel_sem = asyncio.Semaphore(max_parallel)

            async def _run_one(ctc: ConcurrentToolCall) -> Dict[str, Any]:
                original_name = original_names_by_id.get(str(ctc.id), ctc.name)
                tool_call_dict = {
                    "name": ctc.name,
                    "arguments": ctc.arguments,
                    "original_tool_name": original_name,
                    "normalized_tool_name": ctc.name,
                }
                async with _parallel_sem:
                    return await self._execute_tool_with_ledger(
                        tool_call_dict,
                        handlers,
                        tool_call_id=str(ctc.id),
                    )

            gather_results = await asyncio.gather(
                *[_run_one(ctc) for ctc in concurrent],
                return_exceptions=True,
            )
            for ctc, result in zip(concurrent, gather_results):
                original_name = original_names_by_id.get(str(ctc.id), ctc.name)
                tool_call_dict = {
                    "name": ctc.name,
                    "arguments": ctc.arguments,
                    "original_tool_name": original_name,
                    "normalized_tool_name": ctc.name,
                }
                if isinstance(result, Exception):
                    error_result: Dict[str, Any] = {
                        "ok": False,
                        "error": f"{type(result).__name__}: {result}",
                    }
                    _print_tool_result(
                        ctc.name, error_result, enabled=self._engine._settings.verbose_progress
                    )
                    trace.record(
                        ExecutionMode.ACTING,
                        action=tool_call_dict,
                        observation=error_result,
                    )
                    result_payload = self._compact_tool_result(
                        ctc.name, ctc.arguments, error_result
                    )
                    result_text = _truncate_result_for_budget(result_payload, result_budget)
                else:
                    _print_tool_result(ctc.name, result, enabled=self._engine._settings.verbose_progress)
                    trace.record(
                        ExecutionMode.ACTING,
                        action=tool_call_dict,
                        observation=result if isinstance(result, dict) else {"result": str(result)},
                    )
                    result_payload = self._compact_tool_result(ctc.name, ctc.arguments, result)
                    result_text = _truncate_result_for_budget(result_payload, result_budget)
                effective_result = error_result if isinstance(result, Exception) else result
                self._engine._learning_bridge._record_tool_focus(ctc.name, ctc.arguments, effective_result)
                messages.append({"role": "tool", "tool_call_id": ctc.id, "content": result_text})
                self._engine._session_persistence._persist_message(
                    self._engine._current_session_id,
                    "tool",
                    result_text,
                    tool_name=ctc.name,
                    tool_call_id=str(ctc.id),
                    metadata=self._tool_execution_metadata_with_focus(
                        ctc.name, ctc.arguments, effective_result
                    ),
                )
                executed.append(
                    {
                        "id": ctc.id,
                        "name": ctc.name,
                        "original_tool_name": original_name,
                        "arguments": ctc.arguments,
                        "result": effective_result,
                    }
                )
                if isinstance(effective_result, dict) and _should_stop_after_tool_result(
                    ctc.name, effective_result
                ):
                    for skipped_ctc in sequential:
                        skipped_original = original_names_by_id.get(
                            str(skipped_ctc.id), skipped_ctc.name
                        )
                        skipped_result = _skipped_after_failure_result(
                            ctc.name, effective_result
                        )
                        self._append_skipped_tool_message(
                            skipped_ctc.id,
                            skipped_ctc.name,
                            skipped_result,
                            messages=messages,
                            result_budget=result_budget,
                        )
                        executed.append(
                            {
                                "id": skipped_ctc.id,
                                "name": skipped_ctc.name,
                                "original_tool_name": skipped_original,
                                "arguments": skipped_ctc.arguments,
                                "result": skipped_result,
                            }
                        )
                    logger.info(
                        "tool_concurrency: failed side effect returned from concurrent tool %s; skipping sequential group",
                        ctc.name,
                    )
                    return executed

        for i, ctc in enumerate(sequential):
            original_name = original_names_by_id.get(str(ctc.id), ctc.name)
            _show_progress("executing", ctc.name, step=i + 1, total=len(sequential))
            tool_call_dict = {
                "name": ctc.name,
                "arguments": ctc.arguments,
                "original_tool_name": original_name,
                "normalized_tool_name": ctc.name,
            }
            result = await self._execute_tool_with_ledger(
                tool_call_dict,
                handlers,
                tool_call_id=str(ctc.id),
            )
            _clear_indicator()
            _print_tool_result(ctc.name, result, enabled=self._engine._settings.verbose_progress)
            trace.record(
                ExecutionMode.ACTING,
                action=tool_call_dict,
                observation=result if isinstance(result, dict) else {"result": str(result)},
            )
            result_payload = self._compact_tool_result(ctc.name, ctc.arguments, result)
            result_text = _truncate_result_for_budget(result_payload, result_budget)
            self._engine._learning_bridge._record_tool_focus(ctc.name, ctc.arguments, result)
            messages.append({"role": "tool", "tool_call_id": ctc.id, "content": result_text})
            self._engine._session_persistence._persist_message(
                self._engine._current_session_id,
                "tool",
                result_text,
                tool_name=ctc.name,
                tool_call_id=str(ctc.id),
                metadata=self._tool_execution_metadata_with_focus(ctc.name, ctc.arguments, result),
            )
            executed.append(
                {
                    "id": ctc.id,
                    "name": ctc.name,
                    "original_tool_name": original_name,
                    "arguments": ctc.arguments,
                    "result": result,
                }
            )
            if isinstance(result, dict) and _should_stop_after_tool_result(ctc.name, result):
                for skipped_ctc in sequential[i + 1 :]:
                    skipped_original = original_names_by_id.get(
                        str(skipped_ctc.id), skipped_ctc.name
                    )
                    skipped_result = _skipped_after_failure_result(ctc.name, result)
                    self._append_skipped_tool_message(
                        skipped_ctc.id,
                        skipped_ctc.name,
                        skipped_result,
                        messages=messages,
                        result_budget=result_budget,
                    )
                    executed.append(
                        {
                            "id": skipped_ctc.id,
                            "name": skipped_ctc.name,
                            "original_tool_name": skipped_original,
                            "arguments": skipped_ctc.arguments,
                            "result": skipped_result,
                        }
                    )
                logger.info(
                    "tool_concurrency: stopping sequential native tool calls after failed side effect from %s",
                    ctc.name,
                )
                break
        return executed
    def _append_skipped_tool_message(
        self,
        tool_call_id: Any,
        tool_name: str,
        result: Dict[str, Any],
        *,
        messages: List[Dict[str, Any]],
        result_budget: int,
    ) -> None:
        """Append and persist a tool-result message for a call skipped by side-effect gating.

        The assistant message that opened this batch already advertised every
        ``tool_call_id`` it emitted. A call skipped after an earlier side-effect
        failure is never executed, but it still needs a matching ``role="tool"``
        message: without one the next request carries an assistant message with N
        tool_calls but fewer than N tool responses, and the provider rejects it
        with HTTP 400 ("insufficient tool messages following tool_calls message").
        The message is written to both the in-memory history and the durable
        transcript so a turn later rebuilt from persistence stays valid too.
        """
        result_text = _truncate_result_for_budget(result, result_budget)
        messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": result_text}
        )
        self._engine._session_persistence._persist_message(
            self._engine._current_session_id,
            "tool",
            result_text,
            tool_name=tool_name,
            tool_call_id=str(tool_call_id),
        )
    def _tool_execution_context(self) -> Any | None:
        """Build the tool context from the current task contract, if any."""
        contract = self._engine._current_task_contract
        if contract is None:
            return None
        from leapflow.tools.execution_context import ToolExecutionContext

        try:
            from leapflow.tools.shell_tools import _approval_gate

            orchestrator = _approval_gate
        except Exception:  # noqa: BLE001
            orchestrator = None

        return ToolExecutionContext.from_strings(
            workspace_root=contract.workspace_root,
            allowed_roots=contract.allowed_roots,
            session_id=str(self._engine._current_session_id or ""),
            task_id=contract.task_id,
            approval_bypass=getattr(self._engine._settings, "approval_bypass", False),
            orchestrator=orchestrator,
        )
    async def _execute_tool_scoped(
        self,
        tool_call: Dict[str, Any],
        handlers: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a tool with the current turn's workspace context installed."""
        from leapflow.tools.execution_context import reset_tool_context, set_tool_context

        token = set_tool_context(self._tool_execution_context())
        try:
            return await self._execute_general_tool(tool_call, handlers)
        finally:
            reset_tool_context(token)
    async def _execute_tool_with_ledger(
        self,
        tool_call: Dict[str, Any],
        handlers: Dict[str, Any],
        *,
        tool_call_id: str = "",
    ) -> Dict[str, Any]:
        """Execute a tool through the unified idempotency ledger."""
        original_name = str(tool_call.get("original_tool_name") or tool_call.get("name", ""))
        proposed_name = str(tool_call.get("name", ""))
        args = dict(tool_call.get("arguments") or {})
        registry = _default_tool_registry()
        resolution = registry.resolve(proposed_name, args)
        if not resolution.auto_executable or resolution.normalized_name is None:
            async def _run_unresolved() -> Dict[str, Any]:
                return await self._execute_tool_scoped(tool_call, handlers)

            return await self._engine._skill_dispatcher._execute_action_boundary(
                action_type="tool",
                action_name=proposed_name,
                arguments=args,
                execution_id=f"unresolved-{uuid.uuid4().hex}",
                execution_policy="external_side_effect",
                execute=_run_unresolved,
            )

        tool_name = resolution.normalized_name
        spec = registry.specs.get(tool_name)
        policy = execution_policy_for(tool_name, spec)
        if getattr(self._engine._settings, "agent_validate_tool_args", True):
            invalid_args = _validate_tool_arguments(spec, args)
            if invalid_args is not None:
                logger.info(
                    "tool_args_invalid: tool=%s missing=%s", tool_name, invalid_args.get("missing")
                )
                return invalid_args
        session_id = self._engine._current_session_id or "ephemeral"
        turn_id = self._engine._current_turn_id or f"turn-{self._engine._session_turn_count}"
        command_id = self._engine._current_command_id or turn_id
        normalized_call = {
            **tool_call,
            "name": tool_name,
            "arguments": args,
            "original_tool_name": original_name,
            "normalized_tool_name": tool_name,
        }
        record, existing = self._engine._tool_execution_ledger.reserve(
            session_id=session_id,
            turn_id=turn_id,
            command_id=command_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=args,
            policy=policy,
        )
        if existing is not None:
            if existing.status == "running":
                existing = await self._engine._tool_execution_ledger.wait_for_completion(
                    existing,
                    timeout_s=self._engine._tool_timeouts.get(tool_name, self._engine._default_tool_timeout_s),
                )
            duplicate = ToolExecutionLedger.duplicate_result(existing)
            duplicate.update(
                {
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "execution_policy": existing.policy,
                }
            )
            logger.info(
                "tool_idempotency: skipped duplicate tool=%s policy=%s key=%s",
                tool_name,
                existing.policy,
                existing.idempotency_key[:12],
            )
            return duplicate

        async def _execute_and_finalize() -> Dict[str, Any]:
            try:
                result = await self._execute_tool_scoped(normalized_call, handlers)
            except Exception as exc:
                failed_result: Dict[str, Any] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "retryable": True,
                    "execution_id": record.execution_id,
                    "idempotency_key": record.idempotency_key,
                    "execution_policy": policy,
                    "tool_call_id": tool_call_id,
                }
                _annotate_uncertain_effect(failed_result, policy)
                self._engine._tool_execution_ledger.complete(record, failed_result)
                raise
            if isinstance(result, dict):
                result_for_ledger: Dict[str, Any] = {
                    **result,
                    "execution_id": record.execution_id,
                    "idempotency_key": record.idempotency_key,
                    "execution_policy": policy,
                    "tool_call_id": tool_call_id,
                }
            else:
                result_for_ledger = {
                    "ok": True,
                    "result": result,
                    "execution_id": record.execution_id,
                    "idempotency_key": record.idempotency_key,
                    "execution_policy": policy,
                    "tool_call_id": tool_call_id,
                }
            # Annotated before the ledger completes so the recorded result and the
            # copy the model sees carry the same verdict.
            _annotate_uncertain_effect(result_for_ledger, policy)
            completed = self._engine._tool_execution_ledger.complete(record, result_for_ledger)
            result_for_ledger["execution_status"] = completed.status
            return result_for_ledger

        try:
            return await self._engine._skill_dispatcher._execute_action_boundary(
                action_type="tool",
                action_name=tool_name,
                arguments=args,
                execution_id=record.execution_id,
                execution_policy=policy,
                execute=_execute_and_finalize,
            )
        except Exception as exc:
            from leapflow.domain.evolution_event import ActionEvidenceUnavailable

            if not isinstance(exc, ActionEvidenceUnavailable):
                raise
            failed_result = {
                "ok": False,
                "error": str(exc),
                "failure_code": "evolution_evidence_unavailable",
                "retryable": True,
                "execution_id": record.execution_id,
                "idempotency_key": record.idempotency_key,
                "execution_policy": policy,
                "tool_call_id": tool_call_id,
                "counts_as_failure": False,
            }
            self._engine._tool_execution_ledger.complete(record, failed_result)
            return failed_result
    async def _execute_general_tool(
        self, tool_call: Dict[str, Any], handlers: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a general-purpose tool via registry handlers.

        Routing priority (Landing C):
        0. Semantic desktop tools — admitted only when this turn's handler
           table carries them, gated by the desktop approval gate when mutating
        1. Registry-merged handlers dict (includes plugin + semantic handlers)

        Security: untrusted tool results (MCP, web) are wrapped with delimiters.
        Secrets in error messages are redacted before returning to LLM.
        """
        from leapflow.security.redact import redact_sensitive_text
        from leapflow.skills.semantic_schema import SEMANTIC_TOOL_NAMES

        original_name = str(tool_call.get("original_tool_name") or tool_call.get("name", ""))
        proposed_name = str(tool_call.get("name", ""))
        args = tool_call.get("arguments", {})

        if proposed_name in SEMANTIC_TOOL_NAMES:
            if proposed_name not in handlers:
                return {
                    "ok": False,
                    "error": f"Desktop tool '{proposed_name}' is unavailable (perception offline)",
                }
            approved, denial = await self._approve_desktop_action(proposed_name, args)
            if not approved:
                return {"ok": False, "error": denial}
            name = proposed_name
        else:
            registry = _default_tool_registry()
            resolution = registry.resolve(proposed_name, args)
            if not resolution.auto_executable or resolution.normalized_name is None:
                return registry.unknown_result(
                    ToolResolution(
                        original_name=original_name,
                        normalized_name=resolution.normalized_name,
                        status=resolution.status,
                        confidence=resolution.confidence,
                        reason=resolution.reason,
                        suggestions=resolution.suggestions,
                        auto_executable=False,
                        risk_level=resolution.risk_level,
                    )
                )
            name = resolution.normalized_name

        result: Dict[str, Any]

        timeout = self._engine._tool_timeouts.get(name, self._engine._default_tool_timeout_s)
        t0 = time.perf_counter()

        try:
            handler = handlers.get(name)
            if handler is not None:
                # The execution deadline wraps each handler consistently, whether
                # plugins install pipeline interceptors or the direct path is used.
                from leapflow.domain.tool_pipeline import ToolCallContext, run_tool_with_timeout
                from leapflow.plugins import get_registry
                from leapflow.plugins.handler_invocation import invoke_tool_handler

                pipeline = get_registry().tool_pipeline
                if pipeline.interceptor_count > 0:

                    spec = _default_tool_registry().specs.get(name)
                    tool_metadata: Dict[str, Any] = {}
                    if spec is not None:
                        tool_metadata = {
                            "risk_level": spec.risk_level,
                            "mutates_state": spec.mutates_state,
                            "effect_scope": spec.effect_scope,
                            "idempotency_scope": spec.idempotency_scope,
                        }
                    call_ctx = ToolCallContext(
                        tool_name=name,
                        arguments=args,
                        metadata=tool_metadata,
                        annotations={"timeout": timeout},
                    )

                    async def _invoke_handler(ctx: ToolCallContext) -> Dict[str, Any]:
                        """Bridge the pipeline's context-based call to the ToolMetadata handler."""
                        return await invoke_tool_handler(handler, ctx.arguments)

                    result = await pipeline.execute(call_ctx, _invoke_handler)
                else:
                    result = await run_tool_with_timeout(
                        invoke_tool_handler(handler, args), timeout
                    )
            else:
                # No handler — tool is truly unknown
                missing_resolution = registry.resolve(original_name, args)
                return registry.unknown_result(missing_resolution)
        except asyncio.TimeoutError:
            duration = (time.perf_counter() - t0) * 1000
            self._engine._usage_tracker.record_tool_call(name, False, duration)
            return {"ok": False, "error": f"Tool '{name}' timed out after {timeout:.0f}s"}
        except Exception as e:
            duration = (time.perf_counter() - t0) * 1000
            self._engine._usage_tracker.record_tool_call(name, False, duration)
            error_msg = redact_sensitive_text(str(e), force=True)
            return {"ok": False, "error": error_msg}

        duration = (time.perf_counter() - t0) * 1000
        is_ok = not (isinstance(result, dict) and not result.get("ok", True))
        self._engine._usage_tracker.record_tool_call(name, is_ok, duration)

        return self._post_process_tool_result(name, result)
    @staticmethod
    def _post_process_tool_result(tool_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply security post-processing to tool results."""
        from leapflow.security.redact import redact_sensitive_text
        from leapflow.security.threat_patterns import is_untrusted_source, wrap_untrusted_result

        if not isinstance(result, dict):
            return result

        # Redact secrets from error messages
        error = result.get("error")
        if isinstance(error, str):
            result = {**result, "error": redact_sensitive_text(error, force=True)}

        # Wrap untrusted tool output with delimiters
        if is_untrusted_source(tool_name):
            for key in ("result", "output", "content"):
                val = result.get(key)
                if isinstance(val, str) and len(val) >= 32:
                    result = {**result, key: wrap_untrusted_result(val, source=tool_name)}
                    break

        return result
    @staticmethod
    def _tool_execution_metadata(result: Any) -> Dict[str, Any]:
        """Extract tool execution audit metadata for transcript rows."""
        if not isinstance(result, dict):
            return {}
        metadata: Dict[str, Any] = {}
        for key in (
            "execution_id",
            "idempotency_key",
            "execution_status",
            "execution_policy",
            "already_executed",
            "duplicate_suppressed",
            "execution_reused",
            "execution_skipped",
            "counts_as_failure",
            "counts_as_tool_attempt",
            "ui_hidden",
            "skipped_reason",
            "blocked_by_tool",
            "blocked_by_error",
            "tool_call_id",
            "path",
            "file_path",
            "bytes_written",
            "side_effect_uncertain",
        ):
            if key in result:
                metadata[key] = result[key]
        return metadata
    @staticmethod
    def _count_consecutive_tool_failures(messages: List[Dict[str, Any]]) -> int:
        """Count consecutive tool failures within the current user turn.

        Scans backwards from the tail, skipping interleaved assistant messages
        (which separate tool results across loop iterations). A tool success
        resets the counter to 0. Scanning stops at the current turn's ``user``
        message so stale failures from previous turns are never counted.
        """
        count = 0
        for msg in reversed(messages):
            role = msg.get("role", "")
            if role == "user":
                # Reached the current turn boundary — stop scanning.
                break
            if role != "tool":
                # Skip assistant messages interleaved between tool results.
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    if _tool_result_counts_as_failure(parsed):
                        count += 1
                        continue
                    if parsed.get("counts_as_failure") is False or _tool_result_is_control_signal(
                        parsed
                    ):
                        continue
            except (json.JSONDecodeError, ValueError):
                pass
            # Non-JSON or ok!=False — treat as success, reset
            return 0
        return count
