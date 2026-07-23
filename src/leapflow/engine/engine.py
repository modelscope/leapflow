"""Main ReAct-style engine with routing, skills, and audit logging."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union

from leapflow.platform.protocol import HostRpc, Methods
from leapflow.config import Settings
from leapflow.engine.budget import BudgetConfig, BudgetStatus, IterationBudget
from leapflow.engine.prefix_commitment import PrefixCommitmentController
from leapflow.engine.research_ledger import ResearchLedger
from leapflow.engine.agent_loop import AgentLoopFrame
from leapflow.engine.context_compressor import CompressorConfig, ContextCompressor
from leapflow.engine.context_control import (
    ContextBudgetEstimator,
    ContextGovernanceController,
    ContextPostureConfig,
    ContextWindowController,
    ToolEvidenceBuilder,
)
from leapflow.engine.context_disclosure import (
    DisclosureLevel,
    DisclosurePlanner,
    DisclosureRuntimeState,
    MemoryDisclosure,
    PromptAssemblyPlan,
    build_capability_manifests,
)
from leapflow.engine.error_classifier import (
    ErrorCategory,
    ErrorClassifier,
    build_recovery_map,
    jittered_backoff,
)
from leapflow.engine.execution_trace import ExecutionMode, ExecutionTrace
from leapflow.engine.intent_classifier import Intent, IntentClassifier
from leapflow.engine.message_healer import MessageHealer
from leapflow.engine.message_sanitizer import MessageSanitizer
from leapflow.engine.prompt_cache import CacheStrategy
from leapflow.engine.stale_stream import StaleStreamError, stale_guarded_stream, build_continuation_prompt
from leapflow.engine.turn_recovery import TurnRecoveryState
from leapflow.engine.turn_usage import TurnUsageTracker, cost_ceiling_exceeded, build_adaptive_learning_signal
from leapflow.engine.recovery_coordinator import RecoveryCoordinator
from leapflow.engine.recovery_budget import RecoveryBudget
from leapflow.engine.unified_classifier import UnifiedErrorClassifier
from leapflow.engine.recovery_decision import RecoveryAction, RecoveryDecision
from leapflow.engine.recovery_strategies import default_strategies
from leapflow.engine.recovery_audit import JsonlAuditSink, create_audit_entry
from leapflow.engine.failure_envelope import Recoverability
from leapflow.engine.recovery_checkpoint import RecoveryCheckpoint, InMemoryCheckpointStore
from leapflow.engine.tool_concurrency import (
    DefaultConcurrencyPolicy,
    ToolCall as ConcurrentToolCall,
    ToolConcurrencyPolicy,
)
from leapflow.engine.tool_execution import ToolExecutionLedger, execution_policy_for
from leapflow.engine.graph_planner import GraphPlanner
from leapflow.engine.scheduler import TaskScheduler
from leapflow.engine.session import SessionController, SessionMode
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
from leapflow.learning.active_learning import SkillMerger
from leapflow.skills.builtin import app_launcher, clipboard_manager, file_organizer
from leapflow.security.permission_failures import (
    is_permission_failure_payload,
    is_permission_hard_stop_payload,
)
from leapflow.storage.skill_library import SkillLibraryStore
from leapflow.storage.reentry_store import build_reentry_trigger
from leapflow.skills.registry import Skill, SkillRegistry
from leapflow.tools.name_resolver import ToolRegistry, ToolResolution

logger = logging.getLogger(__name__)

_TOOL_ARGS_PREVIEW_LIMIT = 160
_TOOL_RESULT_PREVIEW_LIMIT = 240
_TASK_CONTRACT_HEADING = "## Task Contract"


def _default_tool_registry() -> ToolRegistry:
    """Return the canonical runtime tool registry."""
    from leapflow.tools.registry_bootstrap import TOOL_REGISTRY

    return TOOL_REGISTRY


def _resolve_tool_name(tool_name: str, arguments: Dict[str, Any] | None = None) -> ToolResolution:
    """Resolve a tool name through the runtime registry."""
    return _default_tool_registry().resolve(tool_name, arguments or {})


def _normalize_tool_name(tool_name: str) -> str:
    """Return the canonical executable tool name when resolution is safe."""
    return _default_tool_registry().normalize_name(tool_name)


def _normalize_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Return a resolved tool call while preserving the original tool name."""
    original_name = str(tool_call.get("name", ""))
    arguments = tool_call.get("arguments") or {}
    resolution = _resolve_tool_name(original_name, arguments)
    if not resolution.auto_executable or resolution.normalized_name is None:
        return {**tool_call, **resolution.to_metadata()}
    return {
        **tool_call,
        "name": resolution.normalized_name,
        **resolution.to_metadata(),
    }


def _single_line_preview(value: Any, *, limit: int) -> str:
    """Return a compact single-line preview for UI metadata."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _tool_args_metadata(
    tool_name: str,
    arguments: Dict[str, Any] | None,
    *,
    original_tool_name: str | None = None,
) -> Dict[str, Any]:
    """Build safe, compact tool-start metadata for streaming UIs."""
    args = dict(arguments or {})
    original_name = original_tool_name or tool_name
    metadata: Dict[str, Any] = {
        "tool_name": tool_name,
        "original_tool_name": original_name,
        "normalized_tool_name": tool_name,
        "args_summary": _single_line_preview(args, limit=_TOOL_ARGS_PREVIEW_LIMIT),
    }
    resolution = _resolve_tool_name(original_name, args)
    metadata.update(resolution.to_metadata())
    metadata["tool_name"] = tool_name
    metadata["normalized_tool_name"] = tool_name
    if original_name != tool_name:
        metadata["resolved_from"] = original_name
    for key in ("command", "cmd", "path", "pattern", "query", "url"):
        value = args.get(key)
        if value:
            metadata[key] = _single_line_preview(value, limit=_TOOL_ARGS_PREVIEW_LIMIT)
    return metadata


def _tool_result_metadata(
    tool_name: str,
    arguments: Dict[str, Any] | None,
    result: Any,
    *,
    original_tool_name: str | None = None,
) -> Dict[str, Any]:
    """Build safe, compact tool-completion metadata for streaming UIs."""
    metadata = _tool_args_metadata(tool_name, arguments, original_tool_name=original_tool_name)
    if tool_name in {"platform_action", "gp_platform_action"} and arguments:
        for key in ("platform", "action"):
            value = arguments.get(key)
            if value:
                metadata[key] = _single_line_preview(value, limit=_TOOL_ARGS_PREVIEW_LIMIT)
    metadata["ok"] = True
    if isinstance(result, dict):
        metadata["ok"] = bool(result.get("ok", True))
        for key in ("exit_code", "path", "lines", "truncated", "bytes_written"):
            if key in result:
                metadata[key] = result[key]
        for key in (
            "error_type", "retryable", "resolution_status", "resolution_confidence",
            "already_executed", "duplicate_suppressed", "execution_reused", "execution_skipped",
            "counts_as_failure", "counts_as_tool_attempt", "ui_hidden", "skipped_reason",
            "blocked_by_tool", "blocked_by_error", "execution_id", "idempotency_key", "execution_status",
            "execution_policy", "tool_call_id",
        ):
            if key in result:
                metadata[key] = result[key]
        # App Connector authorization failure metadata
        for key in (
            "failure_class", "failure_code", "recoverability", "blocks_approval",
            "platform", "action", "capability", "missing_scopes", "required_scopes",
            "scope_relation", "scope_source", "console_url", "next_steps", "skip_approval",
        ):
            if key in result:
                metadata[key] = result[key]
        for key in ("suggestions", "available_tools"):
            value = result.get(key)
            if value:
                metadata[key] = value
        for key in ("stdout", "stderr", "content", "output", "error"):
            value = result.get(key)
            if value:
                metadata[f"{key}_preview"] = _single_line_preview(
                    value,
                    limit=_TOOL_RESULT_PREVIEW_LIMIT,
                )
        # App Connector recovery metadata for TUI transparency
        recovery_hint = result.get("recovery_hint")
        if recovery_hint:
            metadata["recovery_hint"] = _single_line_preview(recovery_hint, limit=_TOOL_RESULT_PREVIEW_LIMIT)
        onboarding_state = result.get("onboarding_state")
        if isinstance(onboarding_state, dict) and onboarding_state.get("stage"):
            metadata["onboarding_stage"] = str(onboarding_state["stage"])
            metadata["onboarding_platform"] = str(onboarding_state.get("platform_id") or "")
        if not any(key.endswith("_preview") for key in metadata):
            metadata["result_preview"] = _single_line_preview(
                result,
                limit=_TOOL_RESULT_PREVIEW_LIMIT,
            )
    else:
        metadata["result_preview"] = _single_line_preview(
            result,
            limit=_TOOL_RESULT_PREVIEW_LIMIT,
        )
    return metadata


def _is_retryable_unknown_tool_result(result: Any) -> bool:
    """Return whether a tool result can drive a one-shot name correction retry."""
    return isinstance(result, dict) and result.get("error_type") == "unknown_tool" and bool(
        result.get("retryable", False)
    )


def _has_completed_side_effect(results: List[Dict[str, Any]]) -> bool:
    """Return True if any result is a completed side-effect platform_action."""
    for item in results:
        result = item.get("result")
        if not isinstance(result, dict):
            continue
        if result.get("ok") and result.get("completed"):
            return True
    return False


def _unknown_tool_retry_prompt(result: Dict[str, Any]) -> str:
    """Build a compact structured correction prompt for a bad tool name."""
    suggestions = result.get("suggestions") or []
    available = result.get("available_tools") or []
    suggestions_text = ", ".join(str(item) for item in suggestions[:5]) or "none"
    available_text = ", ".join(str(item) for item in available[:12])
    return (
        "SYSTEM: The previous tool call used an unavailable tool name. "
        f"Original tool: {result.get('original_tool_name', '')}. "
        f"Resolution: {result.get('resolution_status', 'unknown')} "
        f"({result.get('resolution_reason', 'no match')}). "
        f"Suggested canonical tools: {suggestions_text}. "
        f"Available tools include: {available_text}. "
        "Retry once using an exact canonical tool name from the available list and valid arguments. "
        "Do not invent tool names, use aliases, or infer a tool from argument shape; answer without a tool if no exact tool fits."
    )


def _is_permission_failure_payload(payload: Dict[str, Any]) -> bool:
    """Return whether a tool-result payload represents an unresolved permission failure."""
    return is_permission_failure_payload(payload)


def _is_permission_hard_stop_payload(payload: Dict[str, Any]) -> bool:
    """Return whether a failed tool result must stop the current agent turn."""
    return is_permission_hard_stop_payload(payload)


_SIDE_EFFECT_STOP_POLICIES = frozenset({"external_side_effect", "mutating_once", "mutating_idempotent"})


def _tool_result_counts_as_failure(payload: Dict[str, Any]) -> bool:
    """Return whether a tool payload represents a real failed execution attempt."""
    if payload.get("counts_as_failure") is False:
        return False
    if _tool_result_is_control_signal(payload):
        return False
    return payload.get("ok") is False


def _tool_result_is_control_signal(payload: Dict[str, Any]) -> bool:
    """Return whether a tool payload is execution control metadata, not an attempt result."""
    return bool(payload.get("already_executed") or payload.get("duplicate_suppressed") or payload.get("execution_skipped"))


def _tool_failure_text(payload: Dict[str, Any]) -> str:
    """Return the most useful root-cause text from a failed tool payload."""
    for key in ("error", "stderr", "stdout", "message"):
        value = payload.get(key)
        if value:
            return str(value)
    return "unknown error"


def _should_stop_after_tool_result(tool_name: str, payload: Dict[str, Any]) -> bool:
    """Return whether a failed side-effect result must stop the current tool batch.

    Side-effect determination is policy-driven: the execution ledger injects an
    ``execution_policy`` (derived from registry metadata — risk level, mutation,
    idempotency) into every executed tool result, so a mutating/side-effecting
    tool is identified by its declared policy rather than a hardcoded tool-name
    list. This keeps the safety gate general and free of vendor-specific names.
    """
    if _is_permission_hard_stop_payload(payload):
        return True
    if not _tool_result_counts_as_failure(payload):
        return False
    return str(payload.get("execution_policy") or "") in _SIDE_EFFECT_STOP_POLICIES


def _validate_tool_arguments(spec: Any, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pre-execution argument check against a tool's declared required params.

    Returns a structured ``invalid_arguments`` result (for in-turn self-repair) if
    a required parameter key is absent, else ``None``. Presence-only (an empty but
    present value is the handler's concern) to avoid rejecting legitimately empty
    values. The result is marked non-failing and carries no execution_policy, so it
    neither trips the side-effect batch-stop gate nor penalizes failure budgets —
    the model simply sees the missing fields plus the accepted schema and retries.
    """
    if spec is None:
        return None
    required = getattr(spec, "required", frozenset()) or frozenset()
    if not required:
        return None
    missing = [name for name in required if name not in args]
    if not missing:
        return None
    accepted = sorted((getattr(spec, "parameters", frozenset()) or frozenset()) | set(required))
    tool_name = str(getattr(spec, "name", "") or "")
    return {
        "ok": False,
        "error": f"Invalid arguments for {tool_name}: missing required parameter(s): {', '.join(sorted(missing))}",
        "error_type": "invalid_arguments",
        "tool_name": tool_name,
        "missing": sorted(missing),
        "required": sorted(required),
        "accepted_parameters": accepted,
        "retryable": True,
        "counts_as_failure": False,
    }


def _head_tail_truncate(text: str, allow: int) -> str:
    """Keep the head and tail of a long string with an explicit elision marker.

    The tail of stdout/stderr/tracebacks/test output usually holds the actual
    error, so a naive head-only cut discards the most useful part.
    """
    if len(text) <= allow:
        return text
    keep = max(40, allow - 40)  # leave room for the marker
    head = (keep * 2) // 3
    tail = keep - head
    elided = len(text) - head - tail
    return f"{text[:head]}\n… [{elided} chars elided] …\n{text[-tail:]}"


def _truncate_result_for_budget(payload: Any, budget: int) -> str:
    """Serialize a tool result to JSON within ``budget``, preserving structure.

    Pass 1 – prune list fields (e.g. file_list entries): drop tail elements
    and annotate ``<key>_omitted`` so the LLM knows how many were removed.
    Pass 2 – shrink the largest string fields with head+tail truncation so
    the tail error / trace survives.  The final fallback emits a minimal
    valid-JSON sentinel; a raw string cut that leaves invalid JSON is never
    returned.  Never raises.
    """
    try:
        text = json.dumps(payload, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(payload)[:budget]
    if len(text) <= budget:
        return text
    if isinstance(payload, dict):
        shrunk = dict(payload)

        # Pass 1: prune list fields until the result fits.
        # This handles file_list / file_find payloads that carry many entries.
        for key in list(shrunk):
            v = shrunk[key]
            if not isinstance(v, list) or not v:
                continue
            orig_len = len(v)
            # Estimate target entry count from a small sample to minimise
            # iterations; then fine-tune with a tight while-loop.
            sample = json.dumps(v[:min(4, orig_len)], default=str, ensure_ascii=False)
            chars_per = max(1, len(sample) / min(4, orig_len))
            empty_payload = {**shrunk, key: [], key + "_omitted": orig_len}
            overhead = len(json.dumps(empty_payload, default=str, ensure_ascii=False))
            target = max(0, int((budget - overhead) / chars_per))
            shrunk[key] = v[:target]
            if target < orig_len:
                shrunk[key + "_omitted"] = orig_len - target
            # Fine-tune (estimation may be off by ±1 entry).
            while shrunk[key] and len(json.dumps(shrunk, default=str, ensure_ascii=False)) > budget:
                shrunk[key] = shrunk[key][:-1]
                shrunk[key + "_omitted"] = orig_len - len(shrunk[key])
            if len(json.dumps(shrunk, default=str, ensure_ascii=False)) <= budget:
                return json.dumps(shrunk, default=str, ensure_ascii=False)

        # Pass 2: shrink the largest string fields with head+tail truncation.
        while True:
            over = len(json.dumps(shrunk, default=str, ensure_ascii=False)) - budget
            if over <= 0:
                break
            candidates = [(k, v) for k, v in shrunk.items() if isinstance(v, str) and len(v) > 160]
            if not candidates:
                break
            key, value = max(candidates, key=lambda kv: len(kv[1]))
            allow = max(120, len(value) - over - 60)
            if allow >= len(value):
                break
            shrunk[key] = _head_tail_truncate(value, allow)

        text = json.dumps(shrunk, default=str, ensure_ascii=False)
        if len(text) <= budget:
            return text

        # Sentinel: emit minimal valid JSON rather than a raw string cut that
        # leaves the LLM with an unparseable fragment.
        sentinel = json.dumps({
            "ok": payload.get("ok"),
            "kind": payload.get("kind", ""),
            "truncated": True,
            "original_chars": len(text),
            "budget_chars": budget,
        }, default=str, ensure_ascii=False)
        return sentinel

    # Non-dict: hard string cut is unavoidable; the LLM sees a partial raw value.
    return text[:budget]


def _skipped_after_failure_result(blocking_tool: str, blocking_result: Dict[str, Any]) -> Dict[str, Any]:
    """Build a non-failure result for a tool skipped because an earlier side effect failed."""
    return {
        "ok": True,
        "execution_skipped": True,
        "skipped_reason": "previous_tool_failed",
        "blocked_by_tool": blocking_tool,
        "blocked_by_error": _tool_failure_text(blocking_result),
        "counts_as_failure": False,
        "counts_as_tool_attempt": False,
        "ui_hidden": True,
    }


def _permission_hard_stop_from_results(results: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Return the first hard-stop permission failure from native tool results."""
    for item in results:
        result = item.get("result") if isinstance(item, dict) else None
        if isinstance(result, dict) and _is_permission_hard_stop_payload(result):
            return result
    return None


def _build_permission_recovery_text(failure: Dict[str, Any]) -> str:
    """Render a deterministic permission-recovery message from a failure payload.

    This is the single authoritative renderer for authorization failures: it
    only cites scopes and links that are literally present in ``failure``,
    never invents, infers, or expands scope names, and only uses "one of"
    phrasing when ``scope_relation`` explicitly says so. Used both for the
    end-of-loop fallback and to override any free-text LLM answer that
    follows an unresolved permission failure.
    """
    platform = str(failure.get("platform") or "")
    capability = str(failure.get("capability") or "")
    where = f"`{platform}.{capability}`" if platform and capability else (capability or platform or "this action")
    missing_scopes: List[str] = [str(s) for s in (failure.get("missing_scopes") or []) if s]
    required_scopes: List[str] = [str(s) for s in (failure.get("required_scopes") or []) if s]
    scope_relation = str(failure.get("scope_relation") or "all_required")
    recovery_hint = str(failure.get("recovery_hint") or "")
    recoverability = str(failure.get("recoverability") or "")
    console_url = str(failure.get("console_url") or "")
    failure_code = str(failure.get("failure_code") or "")

    scopes = missing_scopes or required_scopes
    label = "Missing scope(s)" if missing_scopes else "Required scope(s)"

    lines: List[str] = [
        f"Authorization failed for {where}. "
        "The platform has denied access — this cannot be resolved by retrying."
    ]
    if scopes:
        quoted = ", ".join(f"`{s}`" for s in scopes)
        if scope_relation == "one_of" and len(scopes) > 1:
            lines.append(f"{label} (granting ANY ONE of the following is sufficient): {quoted}.")
        else:
            lines.append(f"{label}: {quoted}.")
    if recovery_hint and failure_code not in ("rate_limited",):
        lines.append(f"To fix: {recovery_hint}")
    elif recoverability == "admin_required":
        lines.append(
            "An administrator must grant the required permissions in the platform developer console "
            "and republish or reinstall the application."
        )
    if console_url:
        lines.append(f"Developer console: {console_url}")
    lines.append(
        "Do NOT retry this action. When informing the user, quote ONLY the scope name(s) listed above — "
        "never invent, guess, or add other scope names, and never claim they are interchangeable unless "
        "explicitly told they are."
    )
    return "\n".join(lines)


def _extract_recent_tool_failures(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return recent consecutive tool failure payloads, most recent first."""
    failures: List[Dict[str, Any]] = []
    for msg in reversed(messages[-24:]):
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        # Strip "Tool result (name):\n" prefix from text-mode tool messages
        if content.startswith("Tool result (") and ":\n" in content:
            content = content.split(":\n", 1)[1].strip()
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(payload, dict) or not _tool_result_counts_as_failure(payload):
            continue
        failures.append(payload)
        if len(failures) >= 3:
            break
    return failures


def _latest_turn_tool_result(messages: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Return the most recent tool-result payload within the current user turn.

    Scans backwards from the tail across both native (``role=="tool"``) and
    text-mode (``"Tool result (...):"``-prefixed user messages) tool-call
    conventions. Stops and returns ``None`` at the first genuine user message
    (the current turn's boundary) or non-JSON tool content.
    """
    for msg in reversed(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "tool":
            if not isinstance(content, str):
                return None
            try:
                payload = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                return None
            return payload if isinstance(payload, dict) else None
        if role == "user":
            text = str(content or "")
            if text.startswith("Tool result (") and ":\n" in text:
                body = text.split(":\n", 1)[1].strip()
                try:
                    payload = json.loads(body)
                except (json.JSONDecodeError, ValueError):
                    return None
                return payload if isinstance(payload, dict) else None
            # Reached the current turn's real user message boundary.
            return None
        # Skip interleaved assistant messages (preamble / tool_calls).
        continue
    return None


def _permission_override_message(messages: List[Dict[str, Any]]) -> str:
    """Return a deterministic override when the turn's last tool signal is an
    unresolved permission failure.

    Prevents the LLM's free-text final answer from paraphrasing, expanding,
    or fabricating scope names when the most recent tool call in this turn
    failed on authorization and was never followed by a successful retry.
    """
    payload = _latest_turn_tool_result(messages)
    if payload is None or not _is_permission_failure_payload(payload):
        return ""
    return _build_permission_recovery_text(payload)


def _last_tool_failures_recovery_message(messages: List[Dict[str, Any]]) -> str:
    """Build a user-facing message from the last consecutive tool failures.

    Called when the loop exits with no content due to hitting
    max_consecutive_tool_failures.  Returns "" when no useful failure context
    is available in the recent message history.
    """
    failures = _extract_recent_tool_failures(messages)
    if not failures:
        return ""

    last = failures[0]
    failure_code = str(last.get("failure_code") or "")
    error = str(last.get("error") or last.get("stderr") or last.get("stdout") or "")
    recovery_hint = str(last.get("recovery_hint") or "")
    available_actions: List[str] = list(last.get("available_action_names") or [])

    lines: List[str] = []

    # Authorization / permission failures — deterministic, no retry via LLM
    if _is_permission_failure_payload(last):
        lines.append(_build_permission_recovery_text(last))
    elif failure_code == "unknown_platform_action":
        platform = str(last.get("platform") or "")
        action = str(last.get("requested_action") or "")
        lines.append(f"`{platform}.{action}` is not a registered platform action.")
        if available_actions:
            actions_str = ", ".join(f"`{a}`" for a in available_actions[:10])
            lines.append(f"Registered actions for {platform}: {actions_str}.")
    elif failure_code == "wrong_action_namespace":
        action = str(last.get("requested_action") or "")
        lines.append(
            f"`{action}` is a platform management action — "
            "use `platform_connect` (not `platform_action`) for this."
        )
    elif failure_code == "unknown_platform":
        lines.append(error)
        platforms: List[str] = list(last.get("available_platforms") or [])
        if platforms:
            lines.append(f"Available platforms: {', '.join(platforms)}.")
    elif "Missing required fields" in error:
        lines.append(f"Action parameter incomplete: {error}. Please provide the missing field(s) and retry.")
    elif error:
        lines.append(f"Action failed: {error}")

    if recovery_hint and not any(recovery_hint[:50] in line for line in lines):
        lines.append(f"Hint: {recovery_hint}")

    if len(failures) > 1:
        lines.append(f"({len(failures)} consecutive tool failures in this turn)")

    return "\n".join(lines) if lines else ""


def _app_onboarding_recovery_message(messages: List[Dict[str, Any]]) -> str:
    """Build a useful final answer from recent App Connector recovery state."""
    for message in reversed(messages):
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if content.startswith("Tool result (") and ":\n" in content:
            content = content.split(":\n", 1)[1].strip()
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        state = payload.get("onboarding_state")
        if not isinstance(state, dict):
            continue
        platform = str(state.get("platform") or state.get("platform_id") or "the app")
        stage = str(state.get("stage") or "pending")
        hint = str(payload.get("recovery_hint") or state.get("last_error") or "")
        steps = payload.get("next_steps") or state.get("next_actions") or []
        lines = [
            f"App onboarding is paused for {platform} at stage `{stage}`.",
        ]
        if hint:
            lines.append(f"Reason: {hint}")
        if isinstance(steps, list) and steps:
            lines.append("Next steps:")
            lines.extend(f"- {step}" for step in steps[:4])
        lines.append("After completing the missing step, continue the same onboarding flow; LeapFlow will reuse the pending App Connector state.")
        return "\n".join(lines)
    return ""


def _estimate_text_tokens(text: str) -> int:
    """Approximate token count for status display when provider usage is absent."""
    if not text:
        return 0
    cjk_count = sum(
        1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f"
    )
    latin_chars = len(text) - cjk_count
    return max(1, cjk_count + latin_chars // 4)


def _estimate_message_tokens(message: Dict[str, Any]) -> int:
    """Approximate chat-message token cost, including small role overhead."""
    content = message.get("content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        content = "\n".join(parts)
    elif not isinstance(content, str):
        content = str(content)
    return 6 + _estimate_text_tokens(content)


def _estimate_prompt_tokens(messages: List[Dict[str, Any]]) -> int:
    """Approximate prompt token count for the exact message batch sent to the LLM."""
    if not messages:
        return 0
    return max(1, sum(_estimate_message_tokens(msg) for msg in messages) + 3)


def _log_progress(msg: str) -> None:
    """Print a persistent progress line to stderr (visible to user during `leap run`)."""
    if sys.stderr.isatty():
        sys.stderr.write(f"\033[2m\u2192 {msg}\033[0m\n")
    else:
        sys.stderr.write(f"→ {msg}\n")
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
    """Print a brief tool result summary to stdout (visible to user).

    Skips output when disabled or when stdout is not a TTY (e.g. daemon,
    CI/CD, piped output) to avoid polluting logs with ANSI escape codes.
    """
    if not enabled:
        return
    if not sys.stdout.isatty():
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
    if sys.stdout.isatty():
        sys.stdout.write(f"\033[2m  \u21b3 {tool_name}: {preview}\033[0m\n")
    else:
        sys.stdout.write(f"  ↳ {tool_name}: {preview}\n")
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


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """Typed event emitted during streaming execution.

    Event types (extensible via Literal union):
    - chunk: intermediate token fragment, safe to display immediately.
    - final: assembled complete response (full content).
    - tool_start: tool execution beginning (content = tool name).
    - tool_complete: tool execution finished (content = brief result).
    - thinking: reasoning/thinking phase indicator.
    - status: lifecycle status update.
    - approval_request: human approval request from a daemon-side action.
    - approval_response: human approval resolution notification.
    - error: error notification.
    """

    type: Literal[
        "chunk", "final", "tool_start", "tool_complete",
        "thinking", "status", "error", "approval_request", "approval_response",
    ]
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class _PromptAssembly:
    """Resolved prompt pieces for a unified-loop turn."""

    system: str
    plan: PromptAssemblyPlan
    prior_turns: List[Dict[str, Any]]


@dataclass(frozen=True)
class TaskContract:
    """Stable per-turn task contract that survives compression and retrieval drift."""

    task_id: str
    original_request: str
    workspace_root: str
    allowed_roots: tuple[str, ...]
    research_protocol: tuple[str, ...] = ()

    def render(self) -> str:
        """Render the contract as a compact system block."""
        lines = [
            "## Task Contract",
            f"- Task ID: {self.task_id}",
            f"- Original user request: {self.original_request}",
            f"- Workspace root: {self.workspace_root}",
            f"- Allowed roots: {', '.join(self.allowed_roots)}",
            (
                "- Treat relative project paths as relative to the workspace root; never infer `.` "
                "as the project root when a workspace root is provided."
            ),
            (
                "- LeapFlow workspace config is optional at `<workspace>/.leapflow/config.yaml`; "
                "runtime config is loaded from `~/.leapflow/config/user.yaml` and "
                "`~/.leapflow/profiles/<profile>/config/*.yaml`."
            ),
            (
                "- Preserve this task contract across summarization, compression, "
                "tool loops, and memory retrieval."
            ),
        ]
        if self.research_protocol:
            lines.append("- Research protocol:")
            lines.extend(f"  - {item}" for item in self.research_protocol)
        return "\n".join(lines)


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
        vlm: Optional[Any] = None,
        memory_manager: Optional[MemoryManager] = None,
        evolution: Optional[EvolutionMemoryProvider] = None,
        tool_bridge: Optional[Any] = None,
        skill_injector: Optional[Any] = None,
        skill_index: Optional[Any] = None,
        concurrency_policy: Optional[ToolConcurrencyPolicy] = None,
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
        self._imitation = imitation
        self._skill_library = skill_library
        self._skill_merger = SkillMerger(
            registry=registry,
            llm=llm,
            execution=execution,
        )
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

        # Tool concurrency policy (None = sequential fallback)
        self._concurrency_policy: Optional[ToolConcurrencyPolicy] = (
            concurrency_policy if concurrency_policy is not None else DefaultConcurrencyPolicy()
        )

        # Session persistence (injected by CLI)
        self._conversation_store: Optional[Any] = None
        self._current_session_id: Optional[str] = None
        self._current_turn_id: str = ""
        self._current_command_id: str = ""
        self._tool_execution_ledger = ToolExecutionLedger()

        self._current_request_id: str = ""

        # Memory context snapshot (frozen at session start for prefix cache stability)
        self._memory_context_snapshot: Optional[str] = None

        # Cancellation: tracks active task for interrupt support
        self._active_task: Optional[asyncio.Task] = None
        self._cancel_requested = False

        # Tool loop guardrails (injected by CLI)
        self._guardrail: Optional[Any] = None

        # Optional override for dynamic tool result budget (set by CLI wiring)
        self._tool_result_budget: Optional[int] = None

        # Per-turn usage tracking
        self._usage_tracker = TurnUsageTracker()

        # Per-tool timeout (seconds); can be overridden via set_tool_timeouts
        self._default_tool_timeout_s: float = 120.0
        self._tool_timeouts: Dict[str, float] = {}

        # Stale stream timeout
        self._stale_stream_timeout_s: float = 180.0

        # Evolution store for incremental persistence (injected by CLI)
        self._evolution_store: Optional[Any] = None

        # EventBus for learning signal emission (injected by CLI)
        self._event_bus: Optional[Any] = None

        # ExperienceStore bridge for world-model trajectory data (injected by CLI)
        self._experience_store: Optional[Any] = None

        # Model capability registry (injected by CLI)
        self._model_capabilities: Optional[Any] = None

        # Session-level counters (survive per-turn tracker resets)
        self._session_turn_count: int = 0
        self._last_context_tokens: int = 0

        # State-machine loop infrastructure (config-driven)
        self._budget_config = BudgetConfig(
            max_iterations=settings.agent_iter_floor,
            soft_limit=settings.react_soft_limit,
            warning_threshold=settings.react_warning_threshold,
            iter_ceiling=settings.agent_iter_ceiling,
            hard_cap=settings.agent_iter_hard_cap,
            scale_k=settings.agent_budget_scale_k,
        )
        # S3-L3: baseline difficulty weight, kept as the rollback/recompute anchor
        # so calibration never compounds and reset is exact.
        self._baseline_scale_k = settings.agent_budget_scale_k
        # S3-L4: calibrated finalize-posture threshold (None = use configured baseline).
        self._calibrated_finalizing_ratio: Optional[float] = None
        # S3 periodic re-calibration (opt-in): evolution store + root-turn counter.
        self._calibration_store: Optional[Any] = None
        self._turns_since_calibration = 0
        self._error_classifier = ErrorClassifier(
            recovery_map=build_recovery_map(
                transient_max_retries=settings.error_transient_max_retries,
                rate_limit_base_delay=settings.error_rate_limit_base_delay,
            )
        )
        self._compressor = self._new_compressor()
        self._context_controller = ContextWindowController(
            estimator=ContextBudgetEstimator(),
            hard_limit_ratio=settings.context_hard_limit_ratio,
            warning_ratio=settings.context_warning_ratio,
        )
        self._context_governance_controller = self._new_governance()
        self._prefix_commitment = PrefixCommitmentController()
        self._research_ledger = ResearchLedger()
        self._research_ledger_store: Optional[Any] = None
        self._reentry_store: Optional[Any] = None
        self._active_frame: Optional[AgentLoopFrame] = None
        self._full_tools_tokens: int | None = None
        self._last_context_snapshot: dict[str, Any] = {}
        self._last_disclosure_metadata: dict[str, Any] = {}
        self._current_task_contract: TaskContract | None = None
        self._disclosure_planner = DisclosurePlanner()
        # Tier 1 structural continuity gate: capability categories used by native
        # tool_calls in the most recently completed turn. Working memory only
        # stores a synthetic "[Called: ...]" summary (no structured tool_calls),
        # so this dedicated, reset-per-turn attribute is the actual source of
        # truth — never derived from re-parsing text.
        self._last_turn_tool_categories: frozenset[str] = frozenset()
        self._manifests_by_name: Dict[str, Any] | None = None
        self._healer = MessageHealer()

        # B2: Prompt cache optimization (None = disabled)
        self._cache_strategy: CacheStrategy | None = None

        # B4: Output sanitization (None = disabled)
        self._sanitizer: MessageSanitizer | None = None

        # Recovery coordinator infrastructure
        self._unified_classifier = UnifiedErrorClassifier(self._error_classifier)
        self._recovery_coordinator = RecoveryCoordinator()  # Re-created per turn
        self._checkpoint_store = InMemoryCheckpointStore()
        self._audit_sink = JsonlAuditSink()  # In-memory; path-based if layout available

        # Apply startup-time tool configuration derived from settings.
        self._configure_tool_defaults()

    def _configure_tool_defaults(self) -> None:
        """Wire settings-derived values into module-level tool defaults at start-up.

        Kept in its own method so child frames and test fixtures can call it
        without re-running the full ``__init__`` body.
        """
        try:
            from leapflow.tools.shell_tools import set_max_shell_timeout
            set_max_shell_timeout(self._settings.max_shell_timeout_s)
        except Exception:  # noqa: BLE001 - optional; defaults remain if import fails
            logger.debug("_configure_tool_defaults: shell_tools not available")

    # ── Optional strategy setters (config-driven) ────────────────────────

    def set_cache_strategy(self, strategy: CacheStrategy | None) -> None:
        """Configure prompt cache optimization strategy."""
        self._cache_strategy = strategy

    def set_sanitizer(self, sanitizer: MessageSanitizer | None) -> None:
        """Configure output message sanitizer."""
        self._sanitizer = sanitizer

    def reconfigure_host_backend(
        self,
        *,
        rpc: HostRpc,
        perception: Optional[Any],
        execution: Optional[Any],
        tool_bridge: Optional[Any],
    ) -> None:
        """Refresh host RPC and adapters without resetting chat/session state."""
        self._rpc = rpc
        self._perception = perception
        self._execution = execution
        self._tool_bridge = tool_bridge
        self._skill_merger = SkillMerger(
            registry=self._registry,
            llm=self._llm,
            execution=execution,
        )
        if self._settings.has_llm_credentials:
            self._scheduler = TaskScheduler(
                self._registry,
                rpc,
                graph_planner=self._graph_planner,
            )
        else:
            self._scheduler = None

    def reconfigure_runtime(
        self,
        *,
        settings: Settings,
        llm: LLMProvider,
        vlm: Optional[Any],
        classifier: IntentClassifier,
    ) -> None:
        """Refresh runtime LLM configuration without resetting session state."""
        self._settings = settings
        self._llm = llm
        self._vlm = vlm
        self._classifier = classifier
        self._compressor = self._new_compressor()
        self._context_controller = ContextWindowController(
            estimator=ContextBudgetEstimator(),
            hard_limit_ratio=settings.context_hard_limit_ratio,
            warning_ratio=settings.context_warning_ratio,
        )
        self._context_governance_controller = self._new_governance()
        self._skill_merger = SkillMerger(
            registry=self._registry,
            llm=llm,
            execution=self._execution,
        )
        if settings.has_llm_credentials:
            self._graph_planner = GraphPlanner(self._llm, self._registry)
            self._scheduler = TaskScheduler(
                self._registry,
                self._rpc,
                graph_planner=self._graph_planner,
            )
        else:
            self._graph_planner = None
            self._scheduler = None

    def set_tool_result_budget(self, budget: int) -> None:
        """Override per-tool result truncation budget (e.g. linked to model context)."""
        self._tool_result_budget = max(1, budget)

    def _effective_tool_result_budget(self) -> int:
        return self._tool_result_budget or self._settings.max_tool_result_chars

    async def _handle_api_error(
        self,
        classified: ErrorCategory,
        rec: Any,
        recovery: TurnRecoveryState,
        messages: list,
        budget: Any,
        *,
        use_native_tools: bool = False,
        tools_kwarg: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Legacy API error recovery dispatcher. Returns 'continue' to retry, else None.

        DEPRECATED: No longer called from main loops. Retained only for backward
        compat with any external subclass overrides. All recovery now flows through
        RecoveryCoordinator.evaluate().
        """
        if classified == ErrorCategory.CONTEXT_OVERFLOW and recovery.try_compress():
            messages[:] = self._compressor.force_compress(messages)
            logger.info("recovery: force_compress on context overflow")
            if budget.remaining > 0:
                return "continue"

        if classified == ErrorCategory.IMAGE_TOO_LARGE and recovery.try_multimodal_strip():
            self._strip_images_from_messages(messages)
            logger.info("recovery: stripped images from messages")
            if budget.remaining > 0:
                return "continue"

        if rec.should_fallback and recovery.try_provider_failover():
            llm = self._llm
            if hasattr(llm, '_failover'):
                llm._failover("recovery: provider failover")
            logger.info("recovery: provider failover triggered")
            if budget.remaining > 0:
                return "continue"

        if rec.should_rotate_credential and recovery.try_credential_rotate():
            logger.info("recovery: credential rotation requested")
            if budget.remaining > 0:
                return "continue"

        if classified == ErrorCategory.FORMAT_ERROR and recovery.try_disable_thinking():
            logger.info("recovery: disabled thinking mode")
            if budget.remaining > 0:
                return "continue"

        if rec.retry and budget.remaining > 0:
            if rec.backoff:
                await asyncio.sleep(jittered_backoff(budget.used, base=rec.base_delay))
            return "continue"

        return None

    _DEFAULT_LIVE_SIGNAL_KINDS = frozenset({
        "app.focus_change", "fs.change", "context.change", "intent.signal",
    })

    def _inject_live_signals(self, messages: list, watermark: list) -> None:
        """Inject high-priority WM events arrived since ``watermark[0]``.

        Uses a mutable watermark list (single-element) so the caller's
        timestamp advances after each injection, preventing duplicate
        signal messages across loop iterations.
        """
        since_ts = watermark[0]
        raw = getattr(self._settings, "live_signal_kinds", "")
        signal_kinds = frozenset(k.strip() for k in raw.split(",") if k.strip()) if raw else self._DEFAULT_LIVE_SIGNAL_KINDS
        recent = self._wm.get_events_since(since_ts)
        relevant = [
            e for e in recent
            if e.get("_event_kind") in signal_kinds
        ]
        if not relevant:
            return
        lines = []
        for ev in relevant[-5:]:
            text = ev.get("_event_text", "")
            if text:
                lines.append(str(text)[:120])
            else:
                lines.append(str(ev.get("content", ""))[:120])
        summary = "; ".join(lines)
        messages.append(build_system_message(f"[LIVE SIGNAL] {summary}"))
        watermark[0] = time.time()

    @staticmethod
    def _strip_images_from_messages(messages: list) -> None:
        """Remove image content parts from messages in-place (multimodal strip)."""
        for i, msg in enumerate(messages):
            content = msg.get("content")
            if isinstance(content, list):
                text_parts = [
                    p for p in content
                    if isinstance(p, dict) and p.get("type") != "image_url"
                    and p.get("type") != "input_image" and p.get("type") != "image"
                ]
                if len(text_parts) < len(content):
                    if text_parts:
                        messages[i] = {**msg, "content": text_parts}
                    else:
                        messages[i] = {**msg, "content": "[images removed to reduce context]"}

    def _execute_transform_decision(
        self,
        decision: "RecoveryDecision",
        messages: list,
    ) -> bool:
        """Execute a TRANSFORM_AND_RETRY decision. Returns True if transform succeeded.

        Handles different transform strategies:
        - context_compress: force-compress conversation history
        - multimodal_strip: remove image content from messages
        - native_to_text: disable native tool calling
        - thinking_disable: disable thinking mode (handled externally)
        """
        strategy_key = decision.strategy_key
        # Determine specific phase from audit_metadata if available
        phase = dict(decision.audit_metadata).get("phase", "")

        if strategy_key == "context_compress":
            if phase == "multimodal_to_text":
                self._strip_images_from_messages(messages)
            else:
                # Default: history_summarize and disclosure_shrink both use force_compress
                messages[:] = self._compressor.force_compress(messages)
            return True

        if strategy_key == "multimodal_strip":
            self._strip_images_from_messages(messages)
            return True

        if strategy_key == "native_to_text":
            # Handled by caller via tools_kwarg mutation
            return True

        if strategy_key == "thinking_disable":
            # Handled by caller via enable_thinking flag
            return True

        logger.warning("Unknown transform strategy: %s", strategy_key)
        return True

    def _check_guardrail(
        self,
        messages: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Run guardrail check. Returns 'halt' if loop should stop, else None."""
        if self._guardrail is None:
            return None
        violation = self._guardrail.check(messages)
        if not violation.violated:
            return None
        logger.warning("guardrail: %s", violation.reason)
        # Progress-aware: while the task is still advancing (stall counter at 0),
        # a detected repetition/domination is producing progress -> never halt,
        # and the finalize/diversify nudge is suppressed so legitimate batch or
        # sequential work on a long task is not cut short. Only when the task is
        # ALSO stalled does the guardrail escalate to a halt (or emit a nudge).
        frame = self._active_frame
        stalled = bool(frame is not None and getattr(frame, "stalled_rounds", 0) >= 1)
        if violation.severity == "halt" and stalled:
            messages.append(build_user_message_text(
                f"SYSTEM GUARDRAIL: {violation.reason}. {violation.suggestion}"
            ))
            return "halt"
        if not stalled:
            return None  # productive: neither halt nor nudge
        messages.append(build_user_message_text(
            f"SYSTEM WARNING: {violation.reason}. {violation.suggestion}"
        ))
        return None

    def _evaluate_tool_failures(
        self, failed_items: List[tuple[str, Dict[str, Any]]], *, turn_id: int,
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
        coordinator = self._recovery_coordinator
        if coordinator is None:
            return None
        session_id = getattr(self, "_current_session_id", "") or ""
        for tool_name, result in failed_items:
            if not isinstance(result, dict):
                continue
            envelope = self._unified_classifier.classify_tool_result(
                result, tool_name=tool_name,
                execution_policy=result.get("execution_policy", "read_only"),
            )
            if envelope is None:
                continue
            if envelope.recoverability == Recoverability.NON_RECOVERABLE:
                decision = coordinator.evaluate(envelope)
                self._audit_sink.record(create_audit_entry(
                    envelope, decision, coordinator.budget,
                    session_id=session_id, turn_id=turn_id,
                ))
                return decision.reason or f"Non-recoverable tool failure ({envelope.category})"
            # Recoverable: fed back to the agent (zero-cost, no recovery budget spent).
            feedback = RecoveryDecision.create(
                envelope=envelope,
                action=RecoveryAction.SKIP_AND_CONTINUE,
                reason="Tool failure fed back to the agent for autonomous diagnosis and retry",
                strategy_key="tool_feedback",
                budget_cost=0,
            )
            self._audit_sink.record(create_audit_entry(
                feedback.envelope, feedback, coordinator.budget,
                session_id=session_id, turn_id=turn_id,
            ))
        return None

    def set_tool_timeouts(self, timeouts: Dict[str, float]) -> None:
        """Set per-tool execution timeout overrides (seconds)."""
        self._tool_timeouts = dict(timeouts)

    def set_default_tool_timeout(self, timeout_s: float) -> None:
        self._default_tool_timeout_s = max(5.0, timeout_s)

    def set_stale_stream_timeout(self, timeout_s: float) -> None:
        self._stale_stream_timeout_s = max(30.0, timeout_s)

    def set_evolution_store(self, store: Any) -> None:
        """Inject evolution store for incremental episode persistence."""
        self._evolution_store = store

    def set_model_capabilities(self, registry: Any) -> None:
        """Inject model capability registry."""
        self._model_capabilities = registry

    def set_doc_store(self, doc_store: Any) -> None:
        """Inject SkillDocStore so SkillMerger can sync SKILL.md on approve."""
        self._skill_merger.set_doc_store(doc_store)

    def set_event_bus(self, event_bus: Any) -> None:
        """Inject EventBus for emitting learning signals (episode events)."""
        self._event_bus = event_bus

    def _emit_chat_event(self, sub_action: str, payload: Dict[str, Any]) -> None:
        """Emit a chat interaction event for trajectory recording during LEARNING.

        Only fires when the session is in LEARNING mode and an EventBus is available.
        The recorder's state machine ensures these events are only persisted as
        trajectory steps when recording is active.
        """
        if self._event_bus is None:
            return
        if self._session is None or self._session.mode != SessionMode.LEARNING:
            return
        from leapflow.domain.events import SystemEvent
        event = SystemEvent(
            event_type="chat.interaction",
            source="leapflow.engine",
            payload={"action": sub_action, **payload},
            timestamp=time.time(),
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._event_bus.handle_event(
                event.event_type, event.payload,
            ))
        except RuntimeError:
            pass

    def set_experience_store(self, store: Any) -> None:
        """Inject ExperienceStore for world-model trajectory bridge."""
        self._experience_store = store

    def set_conversation_store(self, store: Any) -> None:
        """Inject conversation persistence store."""
        self._conversation_store = store
        self._tool_execution_ledger.reset(store=store)

    def set_research_ledger_store(self, store: Any) -> None:
        """Inject the research-ledger persistence store (S1, optional).

        Wires the ledger change-listener so each note is persisted per session
        (durable Orient). Without a store, the ledger degrades gracefully to
        per-turn in-memory state.
        """
        self._research_ledger_store = store
        self._research_ledger.set_change_listener(self._persist_research_ledger)

    def set_reentry_store(self, store: Any) -> None:
        """Inject the re-entry store (S2, optional).

        Absent => ``schedule_reentry`` reports "not configured". Registration is
        additionally gated by ``agent_reentry_enabled`` (default off).
        """
        self._reentry_store = store

    def load_session(self, session_id: str) -> bool:
        """Resume a previous session by loading messages from DuckDB.

        Returns True if the session was found and messages loaded.
        """
        if not self._conversation_store:
            return False
        try:
            messages = self._conversation_store.get_messages(session_id, limit=500)
            if not messages:
                return False
            self._current_session_id = session_id
            for msg in messages:
                role = msg.role
                content = msg.content
                if role == "user":
                    self._wm.remember_chat(build_user_message_text(content))
                elif role == "assistant":
                    self._wm.remember_chat(build_assistant_message(content))
            logger.info("session.resume loaded %d messages from %s", len(messages), session_id)
            return True
        except Exception:
            logger.debug("session.resume failed", exc_info=True)
            return False

    def cancel(self) -> None:
        """Request cancellation of the active run/run_stream call.

        Thread-safe: can be called from signal handlers or other threads.
        Cancels the active asyncio task if one is tracked.
        """
        self._cancel_requested = True
        task = self._active_task
        if task is not None and not task.done():
            task.cancel()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_requested

    @property
    def model_capabilities(self) -> Optional[Any]:
        """Model capability registry (``ModelCapabilityRegistry``)."""
        return self._model_capabilities

    @property
    def usage_tracker(self) -> "TurnUsageTracker":
        """Current-turn usage accumulator."""
        return self._usage_tracker

    @property
    def turn_count(self) -> int:
        """Number of completed user turns in this session."""
        return self._session_turn_count

    @property
    def context_token_count(self) -> int:
        """Estimated provider-visible prompt tokens from the most recent API call."""
        return self._last_context_tokens

    @property
    def context_budget_snapshot(self) -> dict[str, Any]:
        """Last prompt-budget snapshot for status/daemon metadata."""
        return dict(self._last_context_snapshot)

    def _active_context_length(self) -> int:
        """Return the runtime context length for the active model/provider."""
        if self._model_capabilities is not None:
            try:
                return max(1, int(self._model_capabilities.resolve(self._settings.llm_model).context_length))
            except Exception:
                logger.debug("model capability lookup failed", exc_info=True)
        return max(1, int(self._settings.llm_context_length))

    def _begin_turn_context(self, user_text: str) -> None:
        """Reset turn-scoped state and build the stable task contract."""
        self._maybe_periodic_recalibration()
        self._memory_context_snapshot = None
        self._last_context_snapshot = {}
        self._last_disclosure_metadata = {}
        self._context_governance_controller.reset_turn_scope()
        self._prefix_commitment.reset()
        if self._research_ledger_store is not None and self._current_session_id:
            self._research_ledger.load_state(
                self._research_ledger_store.load(self._current_session_id)
            )
        else:
            self._research_ledger.reset()
        try:
            from leapflow.tools.registry_bootstrap import set_research_ledger, set_reentry_scheduler
            set_research_ledger(self._research_ledger)
            set_reentry_scheduler(self._schedule_reentry)
        except ImportError:
            pass
        self._current_task_contract = self._build_task_contract(user_text)
        self._current_turn_id = self._current_task_contract.task_id
        self._current_command_id = self._current_task_contract.task_id
        self._tool_execution_ledger.reset(store=self._conversation_store)
        try:
            from leapflow.tools.gateway_tool import reset_platform_action_scope
            reset_platform_action_scope()
        except ImportError:
            pass

    def _build_task_contract(self, user_text: str) -> TaskContract:
        workspace_root = (
            Path(getattr(self._settings, "workspace_root", Path.cwd()))
            .expanduser()
            .resolve()
        )
        protocol = self._research_protocol_for(user_text)
        return TaskContract(
            task_id=f"turn-{self._session_turn_count}",
            original_request=user_text.strip(),
            workspace_root=str(workspace_root),
            allowed_roots=(str(workspace_root),),
            research_protocol=protocol,
        )

    @staticmethod
    def _research_protocol_for(user_text: str) -> tuple[str, ...]:
        normalized = user_text.lower()
        architecture_tokens = (
            "architecture", "diagram", "design", "架构", "架构图", "系统设计", "框图",
        )
        if not any(token in normalized for token in architecture_tokens):
            return ()
        return (
            "Identify the active project root before reading files.",
            "Start from README, AGENTS, docs index, and top-level source layout.",
            "Use outlines, symbols, and bounded ranges before raw full-file reads.",
            "Cross-check entrypoints, core orchestration, representative modules, and tests.",
            "Produce a concise subsystem map and Mermaid architecture diagram grounded in evidence.",
        )

    def _task_scope_keywords(self, user_text: str) -> list[str]:
        keywords = _keywords_from_query(user_text)
        contract = self._current_task_contract
        if contract:
            workspace_name = Path(contract.workspace_root).name
            if workspace_name:
                keywords.append(workspace_name)
        deduped: list[str] = []
        seen: set[str] = set()
        for keyword in keywords:
            key = keyword.lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(keyword)
        return deduped[:12]

    def _task_contract_block(self) -> str:
        if not self._current_task_contract:
            return ""
        return self._current_task_contract.render()

    def _append_task_contract_to_system(self, system: str) -> str:
        block = self._task_contract_block()
        if not block:
            return system
        base = self._strip_task_contract_block(system)
        return f"{base.rstrip()}\n\n{block}\n" if base.strip() else f"{block}\n"

    @staticmethod
    def _strip_task_contract_block(content: str) -> str:
        marker = f"\n{_TASK_CONTRACT_HEADING}"
        if content.startswith(_TASK_CONTRACT_HEADING):
            return ""
        marker_index = content.find(marker)
        if marker_index == -1:
            return content
        return content[:marker_index].rstrip()

    def _ensure_task_contract_message(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        block = self._task_contract_block()
        if not block:
            return messages
        prepared: list[Dict[str, Any]] = []
        inserted = False
        for message in messages:
            if message.get("role") != "system":
                prepared.append(message)
                continue
            content = message.get("content", "")
            if not isinstance(content, str):
                prepared.append(message)
                continue
            base = self._strip_task_contract_block(content)
            if not inserted:
                updated = dict(message)
                updated["content"] = f"{base.rstrip()}\n\n{block}\n" if base.strip() else f"{block}\n"
                prepared.append(updated)
                inserted = True
            elif base.strip():
                updated = dict(message)
                updated["content"] = base
                prepared.append(updated)
        if inserted:
            return prepared
        return [build_system_message(block), *prepared]

    async def _assemble_unified_prompt(
        self,
        user_text: str,
        *,
        tool_definitions: List[Dict[str, Any]],
        enable_thinking: bool,
        slash_command: bool = False,
    ) -> _PromptAssembly:
        """Resolve progressive disclosure and build the system prompt."""
        from leapflow.prompts.templates import UNIFIED_SYSTEM_TEMPLATE

        runtime = DisclosureRuntimeState(
            enable_thinking=enable_thinking,
            native_tools_enabled=self._settings.native_tool_calling_enabled,
            slash_command=slash_command,
            context_posture=str(self._last_context_snapshot.get("context_posture") or "baseline"),
            recent_failure=bool(self._last_context_snapshot.get("forced_final_answer")),
            last_turn_tool_categories=self._recent_tool_categories(),
        )
        try:
            plan = self._disclosure_planner.plan(tool_definitions, runtime)
        except (TypeError, ValueError, RuntimeError) as exc:
            logger.warning("disclosure planning failed; falling back to full context: %s", exc)
            plan = DisclosurePlanner().full_plan(
                tool_definitions,
                runtime,
                "planner fallback preserved unified-loop behavior",
            )

        tool_catalog = self._format_tool_catalog(list(plan.catalog_definitions))
        memory_context = ""
        if plan.memory == MemoryDisclosure.SESSION_SUMMARY:
            memory_context = self._build_session_summary_context(max_messages=plan.max_prior_turns)
        elif plan.memory in {MemoryDisclosure.QUERY_RETRIEVAL, MemoryDisclosure.TASK_RETRIEVAL}:
            memory_context = await self._prefetch_and_freeze_memory(user_text)
        skill_section = self._build_skill_section(include_skills=plan.level != DisclosureLevel.CORE)
        app_connector_section = self._build_app_connector_section()
        system = UNIFIED_SYSTEM_TEMPLATE.format(
            tool_catalog=tool_catalog,
            app_connector_section=app_connector_section,
            skill_section=skill_section,
            memory_context=memory_context,
        )
        system = self._append_task_contract_to_system(system)
        self._last_disclosure_metadata = plan.metadata()
        prior_turns = self._prior_turns_for_plan(plan)
        return _PromptAssembly(system=system, plan=plan, prior_turns=prior_turns)

    def _recent_tool_categories(self) -> frozenset[str]:
        """Return capability categories used by native tool_calls in the prior turn.

        This is the Tier 1 continuity gate. It reads ``self._last_turn_tool_categories``,
        a dedicated attribute updated at the end of each completed turn by
        ``_record_tool_call_categories`` — never a re-reading of the user's free
        text, and never derived from working memory (which only stores a
        synthetic "[Called: ...]" summary string with no structured tool_calls).
        """
        return self._last_turn_tool_categories

    def _record_tool_call_categories(self, native_calls: list) -> None:
        """Update the Tier 1 continuity state from this turn's executed tool_calls.

        Accumulates into ``self._last_turn_tool_categories`` so a turn that makes
        several rounds of tool calls keeps every category it touched, not just
        the last round. Reset once per turn by the caller before the first round.
        """
        if self._manifests_by_name is None:
            from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS

            self._manifests_by_name = {
                m.name: m for m in build_capability_manifests(TOOL_DEFINITIONS)
            }
        categories = set(self._last_turn_tool_categories)
        for call in native_calls:
            name = str(getattr(call, "name", "") or "")
            manifest = self._manifests_by_name.get(name)
            if manifest and manifest.category not in {"system", "general"}:
                categories.add(manifest.category)
        self._last_turn_tool_categories = frozenset(categories)

    @staticmethod
    def _expand_tools_kwarg_full(tools_kwarg: Dict[str, Any], tool_definitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Expand this turn's native tool schema to the full catalog.

        Structural failure-recovery gate: once an unknown_tool result proves
        that this turn's disclosed subset was insufficient, escalate to the
        full catalog immediately rather than guessing a smaller subset again.
        """
        return {"tools": list(tool_definitions)}

    @staticmethod
    def _merge_expanded_tool_schemas(
        tools_kwarg: Dict[str, Any], results: List[Dict[str, Any]],
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

    def _build_session_summary_context(self, *, max_messages: int) -> str:
        """Return a structured local session summary without retrieval or extra LLM calls.

        Structured format preserves more signal per turn compared to a flat
        180-char single-line preview:
        - User turns: full first line up to 400 chars (preserves intent).
        - Assistant turns with tool calls: tool names + brief outcome.
        - Assistant prose turns: content preview up to 300 chars.
        """
        messages = self._wm.as_chat_messages()
        summary_lines: list[str] = []
        for message in messages[-max(0, max_messages):]:
            role = str(message.get("role") or "").strip()
            if role not in {"user", "assistant"}:
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(part.get("text", part)) if isinstance(part, dict) else str(part)
                    for part in content
                )
            elif not isinstance(content, str):
                content = str(content)

            if role == "user":
                # Preserve full user intent: first meaningful line, up to 400 chars.
                first_line = content.strip().split("\n")[0][:400]
                if first_line:
                    summary_lines.append(f"- [user] {first_line}")
            elif content.startswith("[Called:"):
                # Working-memory stores tool-calling turns as "[Called: t1, t2]"
                # summary strings.  Extract and preserve the tool list concisely.
                called_text = content[8:].rstrip("]").strip()[:200]
                summary_lines.append(f"- [assistant] called: {called_text}")
            else:
                # Assistant prose: single-line preview up to 300 chars.
                preview = _single_line_preview(content, limit=300)
                if preview:
                    summary_lines.append(f"- [assistant] {preview}")

        if not summary_lines:
            return ""
        return "\n## Recent Session Summary\n" + "\n".join(summary_lines) + "\n"

    def _build_skill_section(self, *, include_skills: bool) -> str:
        """Return compact learned-skill prompt text when the plan allows it."""
        if not include_skills or not self._skill_index:
            return ""
        entries = self._skill_index.get_entries()
        if not entries:
            return ""
        skill_index_text = self._skill_index.compact_index_text(entries)
        return (
            "\n## Learned Skills\n"
            "You have access to the following learned skills. "
            "Use `gp_skills_list` to browse or `gp_skill_view` to read details:\n"
            f"{skill_index_text}\n"
        )

    def _prior_turns_for_plan(self, plan: PromptAssemblyPlan) -> List[Dict[str, Any]]:
        """Return bounded prior conversation turns according to the disclosure plan."""
        wm_history = self._wm.as_chat_messages()
        prior_turns: List[Dict[str, Any]] = [
            message for message in wm_history
            if isinstance(message.get("role"), str) and message["role"] in ("user", "assistant")
        ]
        return prior_turns[-max(0, plan.max_prior_turns):]

    @staticmethod
    def _planned_enable_thinking(plan: PromptAssemblyPlan, requested: bool) -> bool:
        """Apply the plan-level reasoning gate to the provider request."""
        return requested and plan.reasoning.value != "off"

    def _planned_tools_kwarg(self, plan: PromptAssemblyPlan) -> Dict[str, Any]:
        """Return provider tool schemas only when the plan discloses native tools."""
        if plan.native_tools and plan.tool_definitions:
            return {"tools": list(plan.tool_definitions)}
        return {}

    def _prepare_llm_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Any = None,
        round_number: int = 0,
    ) -> List[Dict[str, Any]]:
        """Compress and hard-gate messages before sending them to the provider."""
        context_length = self._active_context_length()
        token_count = self._context_controller.estimator.estimate_messages(messages)
        prepared = self._compressor.compress(messages, token_count=token_count)
        if (
            getattr(self._settings, "agent_compression_writeback", False)
            and len(prepared) < len(messages)
        ):
            # E-3 (CL-8): persist the structural compression so append-only frozen
            # segments stay byte-stable across rounds -> continuous prefix-cache
            # reuse. The volatile notices appended below are NOT written back; the
            # recent raw tail is preserved by the compressor. Opt-in (default off).
            messages[:] = prepared
        prepared = self._ensure_task_contract_message(prepared)
        compression_trace = self._compressor.last_trace.as_dict()
        prepared = self._compressor.preflight_check(prepared, context_length=context_length)
        prepared = self._ensure_task_contract_message(prepared)
        if self._cache_strategy:
            prepared = self._cache_strategy.optimize(prepared)
            prepared = self._ensure_task_contract_message(prepared)
        decision = self._context_controller.prepare(
            prepared,
            tools=tools,
            context_length=context_length,
            compressor=self._compressor,
        )
        prepared = self._ensure_task_contract_message(decision.messages)
        compression_trace = self._compressor.last_trace.as_dict()
        warning = self._context_controller.warning_notice(
            decision.snapshot,
            round_number=round_number,
        )
        open_questions = self._ledger_open_questions()
        convergence = self._context_governance_controller.convergence_notice(
            round_number, open_questions=open_questions,
        )
        cost_notice = self._cost_ceiling_notice()
        for notice in (warning, convergence, cost_notice):
            if notice:
                prepared = [*prepared, build_user_message_text(notice)]
        ledger_block = self._research_ledger.render()
        if ledger_block:
            prepared = [*prepared, build_user_message_text(ledger_block)]
        prepared = self._ensure_task_contract_message(prepared)
        snapshot = self._context_controller.estimator.snapshot(
            prepared,
            tools=tools,
            context_length=context_length,
        )
        governance = self._context_governance_controller.snapshot(
            context_ratio=snapshot.ratio,
            round_number=round_number,
            open_questions=open_questions,
        ).as_dict()
        compressed = decision.compressed or bool(compression_trace.get("stages_applied"))
        self._last_context_tokens = snapshot.total_tokens
        self._last_context_snapshot = {
            "message_tokens": snapshot.message_tokens,
            "tool_schema_tokens": snapshot.tool_schema_tokens,
            "total_tokens": snapshot.total_tokens,
            "context_length": snapshot.context_length,
            "ratio": snapshot.ratio,
            "compressed": compressed,
            "forced_final_answer": decision.forced_final_answer,
            "compression_trace": compression_trace,
            "compression_reason": compression_trace.get("decision_reason", ""),
            "compression_savings_ratio": compression_trace.get("savings_ratio", 0.0),
            "compression_saved_tokens": compression_trace.get("saved_tokens", 0),
            "context_governance": governance,
            "difficulty": governance.get("difficulty", 0.0),
            "cumulative_effective_tokens": self._usage_tracker.summary().effective_prompt_tokens(),
            "open_questions": open_questions,
            "context_posture": governance.get("posture", "baseline"),
            "context_signal": governance.get("dominant_signal", ""),
            "context_guidance": governance.get("guidance", ""),
            "context_convergence_reason": governance.get("convergence_reason", ""),
            "disclosure": dict(self._last_disclosure_metadata),
            "disclosure_level": self._last_disclosure_metadata.get("level", ""),
            "disclosure_reason": self._last_disclosure_metadata.get("reason", ""),
        }
        if compressed:
            self._usage_tracker.mark_compression()
        return prepared

    def recalibrate_difficulty(self, store: Any) -> Any:
        """S3-L3: apply offline calibration (S3-L2) to the difficulty weight.

        Bounded, gated, and reversible: reads recent turn signals from the
        evolution store and — only when ``agent.calibration_enabled`` — installs a
        clamped ``scale_k`` derived from the *baseline* weight. Default-off, so
        budget behavior is byte-identical unless explicitly enabled. Returns the
        ``CalibrationResult`` for observability.
        """
        from leapflow.learning.difficulty_calibration import (
            CalibrationResult,
            apply_calibration,
            build_calibration_report_from_store,
        )

        enabled = bool(getattr(self._settings, "agent_calibration_enabled", False))
        if not enabled or store is None:
            return CalibrationResult(
                self._baseline_scale_k, self._budget_config.scale_k, False,
                "calibration disabled" if not enabled else "no evolution store",
            )
        try:
            report = build_calibration_report_from_store(store)
        except Exception:
            logger.debug("difficulty calibration: report build failed", exc_info=True)
            return CalibrationResult(
                self._baseline_scale_k, self._budget_config.scale_k, False, "report build failed",
            )
        result = apply_calibration(
            self._baseline_scale_k, report, enabled=True,
            min_confidence=float(getattr(self._settings, "agent_calibration_min_confidence", 0.3)),
        )
        if result.applied:
            self._budget_config = replace(self._budget_config, scale_k=result.effective_k)
            logger.info(
                "difficulty calibration applied: scale_k %.3f -> %.3f (%s)",
                self._baseline_scale_k, result.effective_k, result.reason,
            )
        return result

    def reset_calibration(self) -> None:
        """Revert any applied difficulty calibration to the configured baseline."""
        self._budget_config = replace(self._budget_config, scale_k=self._baseline_scale_k)

    def recalibrate_thresholds(self, store: Any) -> Any:
        """S3-L4: tune the finalize posture threshold from stored signals.

        Same bounded/gated/reversible contract as :meth:`recalibrate_difficulty`,
        applied to ``context_finalizing_ratio`` (clamped to a safe band) and
        derived from the configured baseline. Default-off; rebuilds the governance
        controller so subsequent frames observe the calibrated threshold.
        """
        from leapflow.learning.difficulty_calibration import (
            CalibrationResult,
            apply_calibration,
            build_threshold_report_from_store,
        )

        baseline = self._settings.context_finalizing_ratio
        current = self._calibrated_finalizing_ratio or baseline
        enabled = bool(getattr(self._settings, "agent_calibration_enabled", False))
        if not enabled or store is None:
            return CalibrationResult(
                baseline, current, False,
                "calibration disabled" if not enabled else "no evolution store",
            )
        try:
            report = build_threshold_report_from_store(store)
        except Exception:
            logger.debug("threshold calibration: report build failed", exc_info=True)
            return CalibrationResult(baseline, current, False, "report build failed")
        result = apply_calibration(
            baseline, report, enabled=True,
            min_confidence=float(getattr(self._settings, "agent_calibration_min_confidence", 0.3)),
            k_min=0.6, k_max=0.98,
        )
        if result.applied:
            self._calibrated_finalizing_ratio = result.effective_k
            self._context_governance_controller = self._new_governance()
            logger.info(
                "threshold calibration applied: finalizing_ratio %.3f -> %.3f (%s)",
                baseline, result.effective_k, result.reason,
            )
        return result

    def reset_threshold_calibration(self) -> None:
        """Revert any applied finalize-threshold calibration to the baseline."""
        self._calibrated_finalizing_ratio = None
        self._context_governance_controller = self._new_governance()

    def set_calibration_store(self, store: Any) -> None:
        """Install the evolution store used for periodic S3 re-calibration."""
        self._calibration_store = store

    def _maybe_periodic_recalibration(self) -> None:
        """S3-L3/L4 periodic re-calibration (opt-in via agent.calibration_interval_turns).

        The one-shot startup calibration already applies the learned adjustment;
        when a positive interval is set, re-run every N *root* turns so calibration
        tracks accumulating outcome data. Default 0 = one-shot only (no periodic).
        Bounded/gated/reversible like the underlying recalibration; never raises.
        """
        if not getattr(self._settings, "agent_calibration_enabled", False):
            return
        interval = int(getattr(self._settings, "agent_calibration_interval_turns", 0) or 0)
        if interval <= 0 or self._calibration_store is None:
            return
        self._turns_since_calibration += 1
        if self._turns_since_calibration < interval:
            return
        self._turns_since_calibration = 0
        try:
            self.recalibrate_difficulty(self._calibration_store)
            self.recalibrate_thresholds(self._calibration_store)
        except Exception:
            logger.debug("periodic recalibration failed", exc_info=True)

    def _widen_budget_for_difficulty(self, budget: IterationBudget) -> None:
        """Raise the elastic iteration cap to match the observed difficulty.

        Reads the difficulty produced by the most recent ``_prepare_llm_messages``
        governance snapshot and retargets the budget toward the difficulty-scaled
        ceiling. No-op for fixed budgets and for difficulty 0 (baseline floor).
        This is how a hard task earns a wider horizon while a simple task stays
        near the floor and relies on self-stop / answer-ready convergence.
        """
        difficulty = float(self._last_context_snapshot.get("difficulty", 0.0) or 0.0)
        budget.retarget(budget.elastic_max(difficulty))

    def _task_progress_marker(self) -> tuple:
        """Fingerprint of task progress for stall detection (P0).

        Combines the research-ledger shape (findings / open questions /
        decisions / next step) with governance evidence breadth (evidence count,
        distinct sources). A change between rounds means the task advanced; an
        unchanged marker across rounds indicates a stall. Works for ledger-using
        tasks and, via governance signals, for tasks that never call research_note.
        """
        d = self._research_ledger.as_dict()
        gov = self._last_context_snapshot.get("context_governance", {}) or {}
        return (
            len(d.get("findings", [])),
            len(d.get("open_questions", [])),
            len(d.get("decisions", [])),
            d.get("next_step", ""),
            int(gov.get("evidence_count", 0) or 0),
            int(gov.get("sources_seen", 0) or 0),
        )

    def _update_progress_and_stall(self, frame: AgentLoopFrame) -> None:
        """Advance the frame's stall counter: reset on progress, else increment."""
        marker = self._task_progress_marker()
        if marker == frame.progress_marker:
            frame.stalled_rounds += 1
        else:
            frame.stalled_rounds = 0
            frame.progress_marker = marker
            # Genuine progress: re-arm content-level recovery one-shots so a long
            # task can recover again later (e.g. multiple max_tokens continuations
            # or force-compressions across a long turn). Storm-prone infrastructure
            # one-shots stay strict (bounded by the RecoveryBudget instead).
            if frame.recovery is not None:
                frame.recovery.rearm_after_progress()

    def _within_resource_limits(self) -> bool:
        """Whether real resource budgets (cost) still allow continuation.

        The absolute iteration hard cap is enforced by the budget itself; this
        guards the *cost* ceiling when configured (0 disables). Context pressure
        is handled separately by the finalizing posture.
        """
        multiple = float(getattr(self._settings, "agent_cost_ceiling_context_multiple", 0.0) or 0.0)
        if multiple <= 0:
            return True
        effective = self._usage_tracker.summary().effective_prompt_tokens()
        ceiling = multiple * float(self._active_context_length() or 0)
        return ceiling <= 0 or effective < ceiling

    def _should_extend_budget(self, frame: AgentLoopFrame) -> bool:
        """Progress-gated continuation decision (P0).

        Extend the iteration budget past the elastic ceiling only when the task
        is *productively unfinished*: within resource limits, still making
        progress (not stalled), and not already signalled complete by the ledger
        (zero open questions). A stalled, complete, or resource-exhausted task is
        allowed to converge and stop — so a productive long task continues while a
        spinning one halts.
        """
        if not self._within_resource_limits():
            return False
        stall_rounds = int(getattr(self._settings, "agent_stall_rounds", 6) or 6)
        if frame.stalled_rounds >= stall_rounds:
            return False
        open_q = self._ledger_open_questions()
        if open_q is not None and open_q == 0:
            return False
        return True

    def _ledger_open_questions(self) -> int | None:
        """Ledger sufficiency signal for convergence: None when the ledger is
        inactive/empty (fall back to the marginal heuristic), else the current
        open-question count. A positive count suppresses early answer-ready
        convergence so a long task with tracked open work is never cut short.
        """
        ledger = self._research_ledger
        return None if ledger.is_empty else ledger.open_question_count

    def orientation_view(self, *, now: Optional[float] = None) -> Any:
        """Read-only unified orientation across immediate/working/long-term layers (S4-D1).

        Observe-only aggregation of existing state: the current research ledger
        forms the working layer (findings / open questions / next step). Changes
        no state; usable by dashboards, diagnostics, and future autonomy phases.
        """
        from leapflow.world_model.orientation import build_orientation_from_ledger

        return build_orientation_from_ledger(
            self._research_ledger.to_state(),
            now=now if now is not None else time.time(),
        )

    def _persist_research_ledger(self) -> None:
        """Persist the ledger for the active session (best-effort; S1 durable Orient).

        Fired as the ledger change-listener after each note. No-op when no store
        is wired or no session is established yet.
        """
        store = self._research_ledger_store
        session_id = self._current_session_id
        if store is None or not session_id:
            return
        store.save(session_id, self._research_ledger.to_state())

    def _schedule_reentry(
        self,
        *,
        kind: str = "time",
        reason: str = "",
        delay_seconds: Any = 0.0,
        event_match: Any = None,
        max_reentries: Any = 1,
        deadline_seconds: Any = 0.0,
    ) -> Dict[str, Any]:
        """Register a re-entry trigger seeded with the current orientation (S2 N2).

        Gated by ``agent_reentry_enabled`` (default off). Only persists a trigger
        (Orient snapshot = research-ledger state + task contract + reason); the
        actual wake-up dispatch is a later phase (N3+).
        """
        if not getattr(self._settings, "agent_reentry_enabled", False):
            return {"ok": False, "error": "re-entry is disabled (set agent.reentry_enabled=true)"}
        if self._reentry_store is None:
            return {"ok": False, "error": "re-entry store not configured"}
        contract = self._current_task_contract
        task_id = contract.task_id if contract else (self._current_session_id or "task")
        try:
            trigger = build_reentry_trigger(
                task_id=task_id,
                session_id=self._current_session_id or "",
                ledger_state=self._research_ledger.to_state(),
                task_contract=asdict(contract) if contract else {},
                continuation_summary=reason,
                kind=kind,
                delay_seconds=float(delay_seconds or 0.0),
                event_match=dict(event_match or {}),
                max_reentries=int(max_reentries or 1),
                deadline_seconds=float(deadline_seconds or 0.0),
            )
            self._reentry_store.save(trigger)
        except Exception as exc:
            return {"ok": False, "error": f"failed to schedule re-entry: {exc}"}
        return {
            "ok": True,
            "trigger_id": trigger.trigger_id,
            "kind": trigger.kind,
            "due_at": trigger.due_at,
            "note": "registered; wake-up dispatch activates in a later phase",
        }

    def _cost_ceiling_notice(self) -> str:
        """Soft finalize nudge when cumulative effective cost crosses the ceiling.

        Opt-in safety companion to the elastic iteration cap: bounds runaway cost
        on large-context long tasks. Soft (a nudge, not a hard stop) so no work is
        lost; the iteration ceiling remains the hard bound. Disabled by default
        (``agent_cost_ceiling_context_multiple`` = 0).
        """
        multiple = float(getattr(self._settings, "agent_cost_ceiling_context_multiple", 0.0) or 0.0)
        if multiple <= 0:
            return ""
        effective = self._usage_tracker.summary().effective_prompt_tokens()
        if not cost_ceiling_exceeded(
            effective_prompt_tokens=effective,
            context_length=self._active_context_length(),
            context_multiple=multiple,
        ):
            return ""
        return (
            "SYSTEM: Cumulative cost budget reached. Synthesize and provide the final "
            "answer now from the evidence already gathered; do not start new exploratory "
            "tool calls unless strictly required."
        )

    def _full_tool_schema_tokens(self) -> int:
        """Cached token estimate of the full tool catalog schema (static per process)."""
        if self._full_tools_tokens is None:
            from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS
            self._full_tools_tokens = self._context_controller.estimator.estimate_tools(TOOL_DEFINITIONS)
        return self._full_tools_tokens

    def _evaluate_prefix_commitment(self, budget: IterationBudget) -> None:
        """Evaluate the adaptive prefix-commitment decision (observe-only, W2 slice 2).

        Computes whether the task should commit to a stable, cacheable prefix and
        records the decision in the context snapshot for observability. Does not
        yet enforce (freeze disclosure / lock tools / cache-aware compression) --
        that is W2 slice 3. Reuses the token counts already produced by
        ``_prepare_llm_messages`` plus the post-retarget budget headroom, so it is
        cheap (no re-estimation of the message body) and changes no behavior.
        """
        snap = self._last_context_snapshot
        if not snap:
            return
        difficulty = float(snap.get("difficulty", 0.0) or 0.0)
        posture = str(snap.get("context_posture") or "baseline")
        message_tokens = int(snap.get("message_tokens", 0) or 0)
        disclosed_tool_tokens = int(snap.get("tool_schema_tokens", 0) or 0)
        est_full = message_tokens + self._full_tool_schema_tokens()
        est_pcd = message_tokens + disclosed_tool_tokens
        state = self._prefix_commitment.evaluate(
            difficulty=difficulty,
            posture=posture,
            round_number=budget.used,
            remaining_rounds=budget.remaining,
            est_full_prefix_tokens=est_full,
            est_pcd_prefix_tokens=est_pcd,
        )
        snap["prefix_commitment"] = state.as_dict()
        snap["prefix_committed"] = state.committed

    def _record_provider_usage(self, model: str, usage: Dict[str, Any]) -> None:
        """Prefer provider prompt usage when available and learn observed limits."""
        provider_prompt = int(usage.get("prompt_tokens", 0) or 0)
        if provider_prompt > 0:
            self._last_context_tokens = provider_prompt
            self._last_context_snapshot = {
                **self._last_context_snapshot,
                "provider_prompt_tokens": provider_prompt,
                "total_tokens": provider_prompt,
                "ratio": provider_prompt / max(1, int(self._last_context_snapshot.get("context_length") or 1)),
            }
        if self._model_capabilities and model and usage:
            self._model_capabilities.update_from_usage(model, usage)

    def _compact_tool_result(self, tool_name: str, arguments: Dict[str, Any] | None, result: Any) -> Any:
        """Return compact tool evidence for LLM replay."""
        return self._context_governance_controller.compact_tool_result(tool_name, arguments, result)

    def _tool_context_metadata(
        self,
        tool_name: str,
        arguments: Dict[str, Any] | None,
        result: Any,
    ) -> Dict[str, Any]:
        """Return additional UI metadata from adaptive context handling."""
        metadata = self._context_governance_controller.tool_metadata(tool_name, arguments, result)
        snapshot = self._last_context_snapshot
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

    async def run(self, user_text: str, *, enable_thinking: bool = False) -> str:
        """Entrypoint: simplified routing with unified tool loop as default path."""
        self._session_turn_count += 1
        logger.info("audit.user_input chars=%s", len(user_text))
        self._begin_turn_context(user_text)
        self._emit_chat_event("user_message", {"content": user_text[:500]})

        # 1. Slash command (skill injection — zero-ambiguity activation)
        if user_text.startswith("/") and self._skill_injector:
            self._inject_pending_skill_reminder()
            self._wm.remember_chat(build_user_message_text(user_text))
            logger.debug("route.slash command=%s", user_text.split()[0])
            return await self._unified_tool_loop(user_text, enable_thinking=enable_thinking)

        self._inject_pending_skill_reminder()
        self._wm.remember_chat(build_user_message_text(user_text))

        # 2. Teach command (special session mode switch)
        if self._is_teach_command(user_text):
            return await self._handle_learn_command(user_text)

        # 3. Everything else → unified tool loop (LLM decides tools vs direct response)
        logger.debug("route.unified user_text_len=%d", len(user_text))
        if not self._settings.has_llm_credentials:
            msg = self._error_classifier.friendly_message(ErrorCategory.AUTH_PERMANENT)
            self._wm.remember_chat(build_assistant_message(msg))
            return msg
        return await self._unified_tool_loop(user_text, enable_thinking=enable_thinking)

    async def run_stream(
        self, user_text: str, *, enable_thinking: bool = False, request_id: str = ""
    ) -> AsyncIterator[Union[str, StreamEvent]]:
        """Like run(), but yields text chunks for streamable responses.

        Yields:
            str: legacy plain-text chunks (teach commands).
            StreamEvent(type="chunk"): real-time token fragments.
            StreamEvent(type="final"): complete assembled response.
            StreamEvent(type="tool_call"): internal tool invocation (suppress display).
        """
        self._session_turn_count += 1
        self._current_request_id = request_id
        logger.info("audit.user_input chars=%s", len(user_text))
        self._begin_turn_context(user_text)
        self._emit_chat_event("user_message", {"content": user_text[:500]})

        # 1. Slash command (skill injection)
        if user_text.startswith("/") and self._skill_injector:
            self._inject_pending_skill_reminder()
            self._wm.remember_chat(build_user_message_text(user_text))
            logger.debug("route.slash command=%s", user_text.split()[0])
            async for chunk in self._unified_tool_loop_stream(user_text, enable_thinking=enable_thinking):
                yield chunk
            return

        self._inject_pending_skill_reminder()
        self._wm.remember_chat(build_user_message_text(user_text))

        # 2. Teach command (special session mode switch)
        if self._is_teach_command(user_text):
            result = await self._handle_learn_command(user_text)
            yield result
            return

        # 3. Everything else → unified tool loop (streaming)
        logger.debug("route.unified user_text_len=%d", len(user_text))
        if not self._settings.has_llm_credentials:
            msg = self._error_classifier.friendly_message(ErrorCategory.AUTH_PERMANENT)
            self._wm.remember_chat(build_assistant_message(msg))
            yield StreamEvent(type="final", content=msg)
            return
        async for chunk in self._unified_tool_loop_stream(user_text, enable_thinking=enable_thinking):
            yield chunk

    def _build_app_connector_section(self) -> str:
        """Return prompt-time app connector capabilities without classifying the user turn."""
        try:
            from leapflow.tools.gateway_tool import build_app_connector_prompt_section

            return build_app_connector_prompt_section()
        except Exception:
            logger.debug("app connector prompt section unavailable", exc_info=True)
            return ""

    # ── Unified Tool Loop (chat scenarios) ───────────────────────────────

    def _new_compressor(self) -> ContextCompressor:
        """Fresh context compressor (per engine, or per isolated child frame)."""
        ctx_len = self._settings.llm_context_length
        return ContextCompressor(CompressorConfig(
            token_budget=max(1, int(ctx_len * self._settings.context_hard_limit_ratio)),
            context_length=ctx_len,
            threshold=self._settings.compress_threshold,
            keep_tail=self._settings.compress_keep_tail,
            max_output_chars=self._settings.max_tool_output_chars,
        ))

    def _new_governance(self) -> ContextGovernanceController:
        """Fresh context-governance controller (per engine, or per child frame)."""
        ctx_len = self._settings.llm_context_length
        return ContextGovernanceController(
            evidence_builder=ToolEvidenceBuilder(
                max_content_chars=self._settings.tool_evidence_max_chars,
                context_length=ctx_len,
            ),
            repeated_read_limit=self._settings.repeated_read_limit,
            convergence_round=self._settings.long_task_convergence_round,
            convergence_round_ceiling=self._settings.convergence_round_ceiling,
            convergence_scale=self._settings.convergence_scale,
            posture_config=ContextPostureConfig(
                expanded_ratio=self._settings.context_expanded_ratio,
                finalizing_ratio=(
                    getattr(self, "_calibrated_finalizing_ratio", None)
                    or self._settings.context_finalizing_ratio
                ),
                expanded_evidence_threshold=self._settings.context_expanded_evidence_threshold,
                expanded_tool_call_threshold=self._settings.context_expanded_tool_call_threshold,
                research_source_threshold=self._settings.context_research_source_threshold,
                research_evidence_threshold=self._settings.context_research_evidence_threshold,
            ),
        )

    def _build_child_frame(
        self,
        user_text: str,
        *,
        depth: int,
        tool_filter: "frozenset[str] | None" = None,
        enable_thinking: bool = False,
        parent_session_id: Optional[str] = None,
    ) -> AgentLoopFrame:
        """Build an isolated child frame with fresh per-turn subsystems.

        A recursive subagent runs the same ``_run_agent_loop`` on this frame; the
        fresh budget/recovery/governance/ledger/commitment/usage/compressor keep
        its OODA loop from contaminating the parent frame's state.
        """
        return AgentLoopFrame(
            user_text=user_text,
            depth=depth,
            budget=IterationBudget.for_react(self._budget_config),
            recovery=TurnRecoveryState(),
            governance=self._new_governance(),
            ledger=ResearchLedger(),
            commitment=PrefixCommitmentController(),
            usage_tracker=TurnUsageTracker(),
            compressor=self._new_compressor(),
            tool_filter=tool_filter,
            enable_thinking=enable_thinking,
            parent_session_id=parent_session_id,
        )

    def _install_frame(self, frame: AgentLoopFrame) -> Dict[str, Any]:
        """Install a frame's per-turn subsystems as the engine's active state.

        Returns the previous per-turn state for restoration. This lets a child
        frame run the full loop on the shared engine while the parent frame's
        subsystems stay untouched (see ``_run_child_frame``).
        """
        saved: Dict[str, Any] = {
            "governance": self._context_governance_controller,
            "ledger": self._research_ledger,
            "commitment": self._prefix_commitment,
            "usage": self._usage_tracker,
            "compressor": self._compressor,
            "coordinator": self._recovery_coordinator,
            "snapshot": self._last_context_snapshot,
            "categories": self._last_turn_tool_categories,
            "frame": self._active_frame,
        }
        self._context_governance_controller = frame.governance
        self._research_ledger = frame.ledger
        self._prefix_commitment = frame.commitment
        self._usage_tracker = frame.usage_tracker
        self._compressor = frame.compressor
        if frame.recovery_coordinator is not None:
            self._recovery_coordinator = frame.recovery_coordinator
        self._last_context_snapshot = frame.last_context_snapshot
        self._last_turn_tool_categories = frame.last_turn_tool_categories
        self._active_frame = frame
        return saved

    def _restore_per_turn_state(self, saved: Dict[str, Any]) -> None:
        """Restore per-turn state previously saved by ``_install_frame``."""
        self._context_governance_controller = saved["governance"]
        self._research_ledger = saved["ledger"]
        self._prefix_commitment = saved["commitment"]
        self._usage_tracker = saved["usage"]
        self._compressor = saved["compressor"]
        self._recovery_coordinator = saved["coordinator"]
        self._last_context_snapshot = saved["snapshot"]
        self._last_turn_tool_categories = saved["categories"]
        self._active_frame = saved["frame"]

    async def _run_child_frame(self, frame: AgentLoopFrame) -> str:
        """Run a recursive subagent's isolated frame through the full loop.

        Swaps the engine's per-turn state to the child frame for the duration of
        the child loop, then restores the parent's state — so recursion is fully
        state-isolated without duplicating the loop body.
        """
        saved = self._install_frame(frame)
        try:
            return await self._run_agent_loop(frame)
        finally:
            self._restore_per_turn_state(saved)

    async def _run_subagent_goal(
        self,
        goal: str,
        *,
        depth: int,
        tool_filter: "frozenset[str] | None" = None,
        enable_thinking: bool = False,
    ) -> str:
        """Run a subagent goal as an isolated child frame through the full loop.

        Bridge for ``EngineFrameSubagentExecutor`` (opt-in full-loop subagents):
        the child frame's fresh subsystems + per-frame swap keep the subagent
        from contaminating the parent turn's state.
        """
        frame = self._build_child_frame(
            goal, depth=depth, tool_filter=tool_filter, enable_thinking=enable_thinking,
        )
        return await self._run_child_frame(frame)

    def _build_frame(
        self, user_text: str, enable_thinking: bool, budget: Any, recovery: Any,
    ) -> AgentLoopFrame:
        """Bundle per-frame state around the given budget/recovery.

        The root frame wraps the engine's (freshly reset) per-turn subsystems so
        loop-path reads through ``self._active_frame`` are byte-equivalent to the
        singletons; recursive subagents (later) build frames with fresh subsystems.
        """
        return AgentLoopFrame(
            user_text=user_text,
            enable_thinking=enable_thinking,
            budget=budget,
            recovery=recovery,
            governance=self._context_governance_controller,
            ledger=self._research_ledger,
            commitment=self._prefix_commitment,
            usage_tracker=self._usage_tracker,
            compressor=self._compressor,
        )

    def _build_root_frame(self, user_text: str, *, enable_thinking: bool = False) -> AgentLoopFrame:
        """Build the top-level (depth-0) agent-loop frame for a turn."""
        return self._build_frame(
            user_text, enable_thinking,
            IterationBudget.for_react(self._budget_config),
            TurnRecoveryState(),
        )

    async def _unified_tool_loop(
        self, user_text: str, *, enable_thinking: bool = False
    ) -> str:
        """Entry adapter: build the root frame and run the unified agent loop."""
        return await self._run_agent_loop(
            self._build_root_frame(user_text, enable_thinking=enable_thinking)
        )

    async def _run_agent_loop(self, frame: AgentLoopFrame) -> str:
        """Unified adaptive OODA loop over an isolated per-frame state.

        Per-frame execution state (budget, recovery) lives on ``frame`` so the
        same loop serves the top-level turn (root frame) and, in a later phase,
        recursive subagents (deeper frames with their own budget). Capabilities
        remain engine methods; the LLM dynamically decides tools vs direct reply.
        """
        user_text = frame.user_text
        enable_thinking = frame.enable_thinking
        budget = frame.budget
        recovery = frame.recovery
        self._active_frame = frame

        # Detect slash command → inject skill context
        if user_text.startswith("/"):
            slash_name = user_text.split()[0][1:]  # Remove leading /
            remaining = user_text[len(slash_name) + 1:].strip()
            if self._skill_injector:
                injection = self._skill_injector.build_injection_message(slash_name, remaining)
                if injection:
                    user_text = injection  # Replace user_text with skill injection

        from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS, TOOL_HANDLERS

        # A restricted frame (e.g. a subagent) is offered only its permitted
        # tools; the root frame (tool_filter=None) sees the full registry.
        tool_defs = TOOL_DEFINITIONS
        tool_handlers = TOOL_HANDLERS
        if frame.tool_filter is not None:
            tool_defs = [
                td for td in TOOL_DEFINITIONS
                if td.get("function", {}).get("name", "") in frame.tool_filter
            ]
            tool_handlers = {
                name: fn for name, fn in TOOL_HANDLERS.items() if name in frame.tool_filter
            }

        trace = ExecutionTrace()
        assembly = await self._assemble_unified_prompt(
            user_text,
            tool_definitions=tool_defs,
            enable_thinking=enable_thinking,
            slash_command=user_text.startswith("/"),
        )
        planned_enable_thinking = self._planned_enable_thinking(assembly.plan, enable_thinking)
        # Reset the Tier 1 continuity state now that this turn's plan has been
        # assembled from the *previous* turn's value; it accumulates fresh from
        # this turn's own tool_calls for the *next* turn's plan.
        self._last_turn_tool_categories = frozenset()

        messages: List[Dict[str, Any]] = [
            build_system_message(assembly.system),
            *assembly.prior_turns,
            build_user_message_text(user_text),
        ]

        content = ""
        fatal_error: Optional[str] = None
        # P3: Initialize recovery coordinator for this turn
        recovery_budget = RecoveryBudget(
            turn_deadline_s=self._settings.recovery_turn_deadline_s,
            total_recovery_actions=self._settings.recovery_total_actions,
            max_retry_per_category=self._settings.recovery_max_retry_per_category,
        )
        recovery_budget.start_deadline()
        self._recovery_coordinator = RecoveryCoordinator(
            strategies=default_strategies(),
            budget=recovery_budget,
        )
        self._recovery_coordinator.new_turn(turn_id=budget.used)
        use_native_tools = assembly.plan.native_tools
        result_budget = self._effective_tool_result_budget()
        unknown_tool_retry_used = False
        self._usage_tracker.reset()

        tools_kwarg: Dict[str, Any] = self._planned_tools_kwarg(assembly.plan)

        self._cancel_requested = False
        _signal_watermark = [time.time()]

        session_id = self._ensure_session_for_frame(frame, user_text)

        while not budget.exhausted:
            if self._cancel_requested:
                logger.info("unified_loop: cancelled by user")
                break

            status = budget.consume()
            if status == BudgetStatus.EXHAUSTED:
                # Progress-gated continuation: a productively-unfinished task
                # (open ledger work, still progressing, within resource limits)
                # extends past the elastic ceiling toward the hard cap instead of
                # terminating; a stalled/complete/over-budget task stops here.
                if budget.can_extend and self._should_extend_budget(frame):
                    budget.grant_extension(self._settings.agent_iter_extension_step)
                    if budget.status() == BudgetStatus.EXHAUSTED:
                        break  # absolute hard cap reached
                    logger.info(
                        "unified_loop: budget extended (progress-gated) to %d (stalled=%d)",
                        budget.effective_max, frame.stalled_rounds,
                    )
                    status = budget.status()
                else:
                    break

            self._inject_live_signals(messages, _signal_watermark)

            healed = self._healer.heal(messages)
            compressed = self._prepare_llm_messages(
                healed,
                tools=tools_kwarg.get("tools"),
                round_number=budget.used,
            )
            self._widen_budget_for_difficulty(budget)
            self._update_progress_and_stall(frame)
            self._evaluate_prefix_commitment(budget)

            try:
                resp = await self._llm.achat(
                    compressed, stream=False, enable_thinking=planned_enable_thinking,
                    **tools_kwarg,
                )
                recovery.record_api_success()
                usage = resp.usage or {}
                self._usage_tracker.record_api_call(
                    usage,
                    provider=getattr(self._llm, 'active_provider_name', ''),
                    model=resp.model or '',
                )
                provider_prompt = usage.get("prompt_tokens", 0)
                if provider_prompt > 0:
                    self._record_provider_usage(resp.model or '', usage)
            except Exception as exc:
                _clear_indicator()
                classified = self._error_classifier.classify(exc)
                category_str = classified.value if hasattr(classified, 'value') else str(classified)
                recovery.record_api_error(category_str)

                # Classify through unified coordinator and execute recovery
                envelope = self._unified_classifier.classify_llm_error(
                    exc, provider=getattr(self._llm, 'provider', ''),
                    model=getattr(self._llm, 'model', ''),
                )
                coordinator = self._recovery_coordinator
                try:
                    decision = coordinator.evaluate(envelope)
                except Exception as coord_exc:
                    logger.error("recovery_coordinator.evaluate() failed: %s", coord_exc)
                    fatal_error = f"Internal recovery error: {coord_exc}"
                    break
                self._audit_sink.record(create_audit_entry(
                    envelope, decision, coordinator.budget,
                    session_id=getattr(self, '_current_session_id', '') or '',
                    turn_id=budget.used,
                ))

                # Execute decision via coordinator
                if decision.action == RecoveryAction.RETRY_WITH_BACKOFF:
                    if decision.retry_semantics.backoff_config:
                        await asyncio.sleep(
                            jittered_backoff(budget.used, base=decision.retry_semantics.backoff_config.base_delay)
                        )
                    continue

                elif decision.action == RecoveryAction.TRANSFORM_AND_RETRY:
                    # Handle native_to_text locally (needs local var mutation)
                    if decision.strategy_key == "native_to_text":
                        tools_kwarg = {}
                        use_native_tools = False
                        transform_ok = True
                    else:
                        transform_ok = self._execute_transform_decision(decision, messages)
                    if transform_ok:
                        self._usage_tracker.mark_compression()
                    coordinator.on_strategy_outcome(decision.decision_id, transform_ok)
                    if not transform_ok:
                        fatal_error = f"Transform failed: {decision.reason}"
                        break
                    continue

                elif decision.action == RecoveryAction.FAILOVER:
                    if hasattr(self._llm, '_failover'):
                        self._llm._failover(f"recovery: {decision.reason}")
                    coordinator.on_strategy_outcome(decision.decision_id, True)
                    continue

                elif decision.action in (RecoveryAction.HALT_CLEAN, RecoveryAction.HALT_WITH_CHECKPOINT):
                    if decision.action == RecoveryAction.HALT_WITH_CHECKPOINT:
                        checkpoint = RecoveryCheckpoint(
                            session_id=getattr(self, '_current_session_id', '') or '',
                            turn_id=budget.used,
                            failure_envelope_data={
                                "message": envelope.message,
                                "category": envelope.category,
                                "failure_code": envelope.failure_code,
                                "source": envelope.source.value,
                            },
                            messages_snapshot=list(messages),
                            context_data={
                                "tools_kwarg_keys": list(tools_kwarg.keys()),
                                "use_native_tools": use_native_tools,
                                "budget_used": budget.used,
                            },
                        )
                        self._checkpoint_store.save(checkpoint)
                    fatal_error = decision.reason
                    self._audit_sink.update_outcome(
                        decision.decision_id, "failure",
                        reason="Terminal halt",
                    )
                    break

                else:
                    # ASK_USER, SKIP_AND_CONTINUE, or unknown
                    fatal_error = decision.reason
                    break
            _clear_indicator()

            content = (resp.content or "").strip()
            if self._sanitizer:
                content = self._sanitizer.sanitize(content)

            # Length continuation: if LLM hit max_tokens, attempt continuation
            finish = getattr(resp, 'finish_reason', None)
            if finish in ("length", "max_tokens") and recovery.try_length_continuation():
                logger.info("unified_loop: length continuation (finish_reason=%s)", finish)
                messages.append(build_assistant_message(content))
                messages.append(build_user_message_text(build_continuation_prompt(content)))
                continue

            native_calls = getattr(resp, "tool_calls", None) or []
            if native_calls:
                # Preamble exclusion: content alongside tool_calls is ephemeral
                # reasoning — exclude it from the message context to prevent
                # the next LLM turn from repeating it in the final answer.
                assistant_msg: Dict[str, Any] = {"role": "assistant", "content": ""}
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                    }
                    for tc in native_calls
                ]
                messages.append(assistant_msg)
                self._persist_message(session_id, "assistant", "", tool_calls=assistant_msg.get("tool_calls"))

                results = await self._execute_tools_concurrent(
                    native_calls, tool_handlers, trace=trace, messages=messages,
                )
                self._record_tool_call_categories(native_calls)
                tools_kwarg = self._merge_expanded_tool_schemas(tools_kwarg, results)

                permission_hard_stop = _permission_hard_stop_from_results(results)
                if permission_hard_stop:
                    logger.info(
                        "unified_loop: permission hard-stop after %s/%s",
                        permission_hard_stop.get("platform", "platform"),
                        permission_hard_stop.get("capability") or permission_hard_stop.get("action") or "action",
                    )
                    break

                retryable_unknown = next(
                    (
                        item.get("result")
                        for item in results
                        if _is_retryable_unknown_tool_result(item.get("result"))
                    ),
                    None,
                )
                if retryable_unknown and not unknown_tool_retry_used:
                    unknown_tool_retry_used = True
                    tools_kwarg = self._expand_tools_kwarg_full(tools_kwarg, tool_defs)
                    use_native_tools = bool(tools_kwarg)
                    messages.append(build_user_message_text(_unknown_tool_retry_prompt(retryable_unknown)))
                    continue

                halt_reason = self._evaluate_tool_failures(
                    [
                        (item.get("name") or "", item["result"])
                        for item in results
                        if isinstance(item.get("result"), dict) and _tool_result_counts_as_failure(item["result"])
                    ],
                    turn_id=budget.used,
                )
                if halt_reason:
                    fatal_error = halt_reason
                    break

                # Guardrail check after tool execution
                if self._check_guardrail(messages) == "halt":
                    break

                self._wm.remember_chat(build_assistant_message(
                    f"[Called: {', '.join(tc.name for tc in native_calls)}]"
                ))

                if status == BudgetStatus.SOFT_LIMIT and not self._should_extend_budget(frame):
                    messages.append(build_user_message_text(
                        "SYSTEM: Approaching limit. Provide final answer now."
                    ))
                elif _has_completed_side_effect(results):
                    messages.append(build_user_message_text(
                        "SYSTEM: Side-effect action completed (result has completed:true). "
                        "Do not re-invoke it with the same parameters. "
                        "If all user-requested actions are done, provide the final answer."
                    ))
                continue

            self._persist_message(session_id, "assistant", content)
            tool_call = self._parse_tool_call_from_content(content)

            if tool_call is None:
                self._wm.remember_chat(build_assistant_message(content))
                trace.record(ExecutionMode.COMPLETE)
                break

            # Text-mode preamble exclusion: only store call summary in WM,
            # not the natural language preamble that surrounds the tool_call tag.
            normalized_tool_call = _normalize_tool_call(tool_call)
            tool_name = str(normalized_tool_call["name"])
            self._wm.remember_chat(build_assistant_message(f"[Called: {tool_name}]"))

            messages.append(build_assistant_message(content))
            tool_arguments = normalized_tool_call.get("arguments")
            self._emit_chat_event("tool_call", {
                "tool_name": tool_name,
                "arguments_summary": json.dumps(tool_arguments, default=str, ensure_ascii=False)[:300] if tool_arguments else "",
            })
            _show_progress("executing", tool_name)
            result = await self._execute_tool_with_ledger(
                normalized_tool_call, tool_handlers, tool_call_id=f"text-{budget.used}",
            )
            _clear_indicator()
            self._emit_chat_event("tool_result", {
                "tool_name": tool_name,
                "ok": bool(result.get("ok")) if isinstance(result, dict) else True,
                "summary": json.dumps(result, default=str, ensure_ascii=False)[:300] if isinstance(result, dict) else str(result)[:300],
            })
            _print_tool_result(tool_name, result, enabled=self._settings.verbose_progress)
            trace.record(
                ExecutionMode.ACTING,
                action=normalized_tool_call,
                observation=result if isinstance(result, dict) else {"result": str(result)},
            )

            is_error = isinstance(result, dict) and _tool_result_counts_as_failure(result)
            if is_error:
                recovery.record_tool_failure()
            else:
                recovery.record_tool_success()
            result_payload = self._compact_tool_result(tool_name, tool_arguments, result)
            result_text = _truncate_result_for_budget(result_payload, result_budget)
            messages.append(build_user_message_text(
                f"Tool result ({tool_name}):\n{result_text}"
            ))
            self._persist_message(
                session_id, "tool", result_text,
                tool_name=tool_name, tool_call_id=f"text-{budget.used}",
                metadata=self._tool_execution_metadata(result),
            )

            if _is_permission_hard_stop_payload(result):
                logger.info(
                    "unified_loop: permission hard-stop after %s/%s",
                    result.get("platform", "platform"),
                    result.get("capability") or result.get("action") or tool_name,
                )
                break

            if _is_retryable_unknown_tool_result(result) and not unknown_tool_retry_used:
                unknown_tool_retry_used = True
                messages.append(build_user_message_text(_unknown_tool_retry_prompt(result)))
                continue

            if is_error:
                halt_reason = self._evaluate_tool_failures([(tool_name, result)], turn_id=budget.used)
                if halt_reason:
                    fatal_error = halt_reason
                    break

            if self._check_guardrail(messages) == "halt":
                break

            if status == BudgetStatus.SOFT_LIMIT and not self._should_extend_budget(self._active_frame):
                messages.append(build_user_message_text(
                    "SYSTEM: Approaching limit. Provide final answer now."
                ))

        # Turn-end learning/memory-sync are top-level-turn concerns; a recursive
        # child frame (subagent) must not pollute the parent's evolution/memory
        # (its result flows back via SubagentResult) nor leak background tasks.
        if getattr(self._active_frame, "is_root", True) and self._memory_manager and self._settings.memory_integration_enabled:
            asyncio.create_task(self._sync_turn_safe(messages))

        if getattr(self._active_frame, "is_root", True) and self._evolution is not None and content:
            asyncio.create_task(self._post_turn_review(messages, content))

        llm = self._llm
        if hasattr(llm, 'try_restore_primary'):
            llm.try_restore_primary()

        logger.info("turn_usage: %s", self._usage_tracker.format_log_line())

        if content:
            permission_override = _permission_override_message(messages)
            final = permission_override or content
            self._emit_chat_event("response", {"content": final[:500]})
            return final
        if fatal_error:
            self._emit_chat_event("response", {"content": fatal_error[:500]})
            return fatal_error
        fallback = (
            _app_onboarding_recovery_message(messages)
            or _last_tool_failures_recovery_message(messages)
            or self._budget_exhausted_response(messages)
        )
        self._emit_chat_event("response", {"content": fallback[:500]})
        return fallback

    async def _post_turn_review(
        self, messages: List[Dict[str, Any]], final_content: str
    ) -> None:
        """Background post-turn review: detect memorable patterns and persist episodes.

        Scans the turn's tool calls for interesting patterns (successes, failures)
        and records them as skill episodes for evolution learning. Delegates
        persistence, world-model bridging, and event emission to focused helpers.
        """
        try:
            tool_actions: List[Dict[str, Any]] = []
            for msg in messages:
                if msg.get("role") == "assistant":
                    for tc in (msg.get("tool_calls") or []):
                        fn = tc.get("function", {})
                        tool_actions.append({
                            "tool": fn.get("name", ""),
                            "args_preview": fn.get("arguments", "")[:100],
                        })

            if not tool_actions:
                return

            has_success = any(
                '"ok": true' in m.get("content", "") or '"ok":true' in m.get("content", "")
                for m in messages if m.get("role") in ("tool", "user")
            )
            has_failure = any(
                '"ok": false' in m.get("content", "") or '"ok":false' in m.get("content", "")
                for m in messages if m.get("role") in ("tool", "user")
            )

            reward = 0.5
            if has_success and not has_failure:
                reward = 1.0
            elif has_failure and not has_success:
                reward = -0.5

            skill_name = tool_actions[0]["tool"] if tool_actions else "unknown"
            episode_context = {"final_content_preview": final_content[:200]}
            episode_context.update(self._usage_tracker.to_learning_signal())
            episode_context.update(build_adaptive_learning_signal(self._last_context_snapshot or {}))
            episode = self._evolution.record_episode(
                skill_name=f"turn_{skill_name}",
                actions=tool_actions[:10],
                outcome="completed" if has_success else "mixed",
                reward=reward,
                context=episode_context,
            )

            self._persist_episode(episode)
            self._bridge_to_experience_store(episode, tool_actions, reward, has_success, has_failure)
            self._emit_episode_event(episode, reward)
        except Exception:
            logger.debug("post_turn_review failed", exc_info=True)

    def _persist_episode(self, episode: Any) -> None:
        """Incremental persistence: write episode to DuckDB immediately."""
        if self._evolution_store is None or episode is None:
            return
        try:
            self._evolution_store.save_episode(
                episode_id=episode.episode_id,
                skill_name=episode.skill_name,
                actions=episode.actions,
                outcome=episode.outcome,
                reward=episode.reward,
                context=episode.context,
                timestamp=episode.timestamp,
            )
        except Exception:
            logger.debug("evolution_store.save_episode failed", exc_info=True)

    def _bridge_to_experience_store(
        self, episode: Any, tool_actions: List[Dict[str, Any]],
        reward: float, has_success: bool, has_failure: bool,
    ) -> None:
        """Bridge tool-loop outcomes to ExperienceStore for world-model trajectory."""
        if self._experience_store is None or episode is None:
            return
        try:
            tool_names = ",".join(a.get("tool", "") for a in tool_actions[:3])
            self._experience_store.store(
                action_description=f"chat_tools:{tool_names}",
                app_context="",
                predicted_effect="",
                actual_effect=episode.outcome,
                delta=abs(reward),
                grade_label="helpful" if has_success and not has_failure else "mixed",
            )
        except Exception:
            logger.debug("experience_store.store failed", exc_info=True)

    def _emit_episode_event(self, episode: Any, reward: float) -> None:
        """Emit high-value episodes to EventBus for active learning consumption."""
        if episode is None or self._event_bus is None:
            return
        threshold = getattr(self._settings, "episode_emit_reward_threshold", 0.8)
        if abs(reward) < threshold:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._event_bus.handle_event(
                "learning.episode_recorded",
                {
                    "skill_name": episode.skill_name,
                    "reward": episode.reward,
                    "actions": [a.get("tool", "") for a in episode.actions[:5]],
                    "outcome": episode.outcome,
                },
            ))
        except RuntimeError:
            pass

    async def _unified_tool_loop_stream(
        self, user_text: str, *, enable_thinking: bool = False
    ) -> AsyncIterator[Union[str, StreamEvent]]:
        """Streaming variant of _unified_tool_loop.

        Yields StreamEvent objects for real-time token streaming and final
        responses. Shows transient progress indicators on stderr for thinking
        and tool-execution phases.
        """
        # Reuse the same setup logic as _unified_tool_loop
        if user_text.startswith("/"):
            slash_name = user_text.split()[0][1:]
            remaining = user_text[len(slash_name) + 1:].strip()
            if self._skill_injector:
                injection = self._skill_injector.build_injection_message(slash_name, remaining)
                if injection:
                    user_text = injection

        from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS, TOOL_HANDLERS

        budget = IterationBudget.for_react(self._budget_config)
        trace = ExecutionTrace()
        assembly = await self._assemble_unified_prompt(
            user_text,
            tool_definitions=TOOL_DEFINITIONS,
            enable_thinking=enable_thinking,
            slash_command=user_text.startswith("/"),
        )
        planned_enable_thinking = self._planned_enable_thinking(assembly.plan, enable_thinking)
        # Reset the Tier 1 continuity state now that this turn's plan has been
        # assembled from the *previous* turn's value; it accumulates fresh from
        # this turn's own tool_calls for the *next* turn's plan.
        self._last_turn_tool_categories = frozenset()

        messages: List[Dict[str, Any]] = [
            build_system_message(assembly.system),
            *assembly.prior_turns,
            build_user_message_text(user_text),
        ]

        content = ""
        fatal_error: Optional[str] = None
        turn_recovery = TurnRecoveryState()
        self._active_frame = self._build_frame(user_text, enable_thinking, budget, turn_recovery)
        # Initialize recovery coordinator for stream turn
        recovery_budget = RecoveryBudget(
            turn_deadline_s=self._settings.recovery_turn_deadline_s,
            total_recovery_actions=self._settings.recovery_total_actions,
            max_retry_per_category=self._settings.recovery_max_retry_per_category,
        )
        recovery_budget.start_deadline()
        self._recovery_coordinator = RecoveryCoordinator(
            strategies=default_strategies(),
            budget=recovery_budget,
        )
        self._recovery_coordinator.new_turn(turn_id=budget.used)
        use_native_tools = assembly.plan.native_tools
        result_budget = self._effective_tool_result_budget()
        unknown_tool_retry_used = False
        self._usage_tracker.reset()

        tools_kwarg: Dict[str, Any] = self._planned_tools_kwarg(assembly.plan)

        session_id = self._ensure_session(user_text)

        self._cancel_requested = False
        _signal_watermark = [time.time()]

        while not budget.exhausted:
            if self._cancel_requested:
                logger.info("unified_loop_stream: cancelled by user")
                break

            status = budget.consume()
            if status == BudgetStatus.EXHAUSTED:
                # Progress-gated continuation (mirrors _run_agent_loop): extend a
                # productively-unfinished task past the elastic ceiling toward the
                # hard cap; a stalled/complete/over-budget task stops here.
                if budget.can_extend and self._should_extend_budget(self._active_frame):
                    budget.grant_extension(self._settings.agent_iter_extension_step)
                    if budget.status() == BudgetStatus.EXHAUSTED:
                        break  # absolute hard cap reached
                    logger.info(
                        "unified_loop_stream: budget extended (progress-gated) to %d",
                        budget.effective_max,
                    )
                    status = budget.status()
                else:
                    break

            self._inject_live_signals(messages, _signal_watermark)

            healed = self._healer.heal(messages)
            compressed = self._prepare_llm_messages(
                healed,
                tools=tools_kwarg.get("tools") if use_native_tools else None,
                round_number=budget.used,
            )
            self._widen_budget_for_difficulty(budget)
            self._update_progress_and_stall(self._active_frame)
            self._evaluate_prefix_commitment(budget)

            content = ""

            if use_native_tools and tools_kwarg:
                try:
                    resp = await self._llm.achat(
                        compressed, stream=False, enable_thinking=planned_enable_thinking,
                        **tools_kwarg,
                    )
                    turn_recovery.record_api_success()
                    usage = resp.usage or {}
                    self._usage_tracker.record_api_call(
                        usage,
                        provider=getattr(self._llm, 'active_provider_name', ''),
                        model=resp.model or '',
                    )
                    provider_prompt = usage.get("prompt_tokens", 0)
                    if provider_prompt > 0:
                        self._record_provider_usage(resp.model or '', usage)
                except Exception as exc:
                    _clear_indicator()
                    turn_recovery.record_api_error()

                    # Classify through unified coordinator
                    envelope = self._unified_classifier.classify_llm_error(
                        exc, provider=getattr(self._llm, 'provider', ''),
                        model=getattr(self._llm, 'model', ''),
                    )
                    coordinator = self._recovery_coordinator
                    try:
                        decision = coordinator.evaluate(envelope)
                    except Exception as coord_exc:
                        logger.error("recovery_coordinator.evaluate() failed: %s", coord_exc)
                        yield StreamEvent(type="error", content=f"Internal recovery error: {coord_exc}")
                        break
                    self._audit_sink.record(create_audit_entry(
                        envelope, decision, coordinator.budget,
                        session_id=getattr(self, '_current_session_id', '') or '',
                        turn_id=budget.used,
                    ))

                    if decision.action == RecoveryAction.RETRY_WITH_BACKOFF:
                        if decision.retry_semantics.backoff_config:
                            await asyncio.sleep(
                                jittered_backoff(budget.used, base=decision.retry_semantics.backoff_config.base_delay)
                            )
                        continue
                    elif decision.action == RecoveryAction.TRANSFORM_AND_RETRY:
                        if decision.strategy_key == "native_to_text":
                            tools_kwarg = {}
                            use_native_tools = False
                        else:
                            self._execute_transform_decision(decision, messages)
                        coordinator.on_strategy_outcome(decision.decision_id, True)
                        continue
                    elif decision.action == RecoveryAction.FAILOVER:
                        if hasattr(self._llm, '_failover'):
                            self._llm._failover(f"recovery: {decision.reason}")
                        coordinator.on_strategy_outcome(decision.decision_id, True)
                        continue
                    else:
                        # Terminal: HALT_CLEAN, HALT_WITH_CHECKPOINT, ASK_USER
                        fatal_error = decision.reason
                        yield StreamEvent(type="error", content=decision.reason)
                        break
                _clear_indicator()

                content = (resp.content or "").strip()
                if self._sanitizer:
                    content = self._sanitizer.sanitize(content)

                # Length continuation for native tool path
                finish = getattr(resp, 'finish_reason', None)
                if finish in ("length", "max_tokens") and turn_recovery.try_length_continuation():
                    logger.info("unified_loop_stream: length continuation")
                    messages.append(build_assistant_message(content))
                    messages.append(build_user_message_text(build_continuation_prompt(content)))
                    continue

                native_calls = getattr(resp, "tool_calls", None) or []
                if native_calls:
                    # Preamble exclusion: content alongside tool_calls is ephemeral
                    # reasoning — exclude from context to prevent final-answer repetition.
                    assistant_msg: Dict[str, Any] = {"role": "assistant", "content": ""}
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                        }
                        for tc in native_calls
                    ]
                    messages.append(assistant_msg)
                    self._persist_message(
                        session_id, "assistant", "",
                        tool_calls=assistant_msg.get("tool_calls"),
                    )

                    for tc in native_calls:
                        resolved_call = _normalize_tool_call({"name": tc.name, "arguments": tc.arguments})
                        normalized_name = str(resolved_call["name"])
                        original_name = str(resolved_call.get("original_tool_name") or tc.name)
                        yield StreamEvent(
                            type="tool_start",
                            content=normalized_name,
                            metadata=_tool_args_metadata(
                                normalized_name,
                                tc.arguments,
                                original_tool_name=original_name,
                            ),
                        )
                    results = await self._execute_tools_concurrent(
                        native_calls, TOOL_HANDLERS, trace=trace, messages=messages
                    )
                    self._record_tool_call_categories(native_calls)
                    tools_kwarg = self._merge_expanded_tool_schemas(tools_kwarg, results)
                    result_by_id = {str(item.get("id")): item for item in results}
                    retryable_unknown = next(
                        (
                            item.get("result")
                            for item in results
                            if _is_retryable_unknown_tool_result(item.get("result"))
                        ),
                        None,
                    )
                    for tc in native_calls:
                        item = result_by_id.get(str(tc.id), {})
                        normalized_name = str(item.get("name") or _normalize_tool_name(tc.name))
                        original_name = str(item.get("original_tool_name") or tc.name)
                        yield StreamEvent(
                            type="tool_complete",
                            content=normalized_name,
                            metadata={
                                **_tool_result_metadata(
                                    normalized_name,
                                    tc.arguments,
                                    item.get("result"),
                                    original_tool_name=original_name,
                                ),
                                **self._tool_context_metadata(normalized_name, tc.arguments, item.get("result")),
                            },
                        )

                    permission_hard_stop = _permission_hard_stop_from_results(results)
                    if permission_hard_stop:
                        logger.info(
                            "unified_loop_stream: permission hard-stop after %s/%s",
                            permission_hard_stop.get("platform", "platform"),
                            permission_hard_stop.get("capability") or permission_hard_stop.get("action") or "action",
                        )
                        break

                    if retryable_unknown and not unknown_tool_retry_used:
                        unknown_tool_retry_used = True
                        tools_kwarg = self._expand_tools_kwarg_full(tools_kwarg, TOOL_DEFINITIONS)
                        use_native_tools = bool(tools_kwarg)
                        messages.append(build_user_message_text(_unknown_tool_retry_prompt(retryable_unknown)))
                        continue
                    halt_reason = self._evaluate_tool_failures(
                        [
                            (item.get("name") or "", item["result"])
                            for item in results
                            if isinstance(item.get("result"), dict) and _tool_result_counts_as_failure(item["result"])
                        ],
                        turn_id=budget.used,
                    )
                    if halt_reason:
                        fatal_error = halt_reason
                        break

                    if self._check_guardrail(messages) == "halt":
                        break

                    self._wm.remember_chat(build_assistant_message(
                        f"[Called: {', '.join(tc.name for tc in native_calls)}]"
                    ))
                    if status == BudgetStatus.SOFT_LIMIT and not self._should_extend_budget(self._active_frame):
                        messages.append(build_user_message_text(
                            "SYSTEM: Approaching limit. Provide final answer now."
                        ))
                    elif _has_completed_side_effect(results):
                        messages.append(build_user_message_text(
                            "SYSTEM: Side-effect action completed (result has completed:true). "
                            "Do not re-invoke it with the same parameters. "
                            "If all user-requested actions are done, provide the final answer."
                        ))
                    continue

            else:
                if self._settings.stream_output:
                    content_parts: list[str] = []
                    try:
                        _clear_indicator()
                        raw_stream = self._llm.achat_stream(
                            compressed, enable_thinking=planned_enable_thinking,
                        )
                        guarded = stale_guarded_stream(
                            raw_stream, timeout_s=self._stale_stream_timeout_s,
                        )
                        async for chunk in guarded:
                            content_parts.append(chunk)
                            yield StreamEvent(type="chunk", content=chunk)
                        turn_recovery.record_api_success()
                    except StaleStreamError as stale_exc:
                        _clear_indicator()
                        partial = stale_exc.partial_text or "".join(content_parts)
                        if partial.strip() and turn_recovery.try_length_continuation():
                            logger.warning("stale_stream: recovering with %d chars partial", len(partial))
                            content = partial.strip()
                            messages.append(build_assistant_message(content))
                            messages.append(build_user_message_text(
                                build_continuation_prompt(content)
                            ))
                            continue
                        yield StreamEvent(type="error", content=str(stale_exc))
                        break
                    except Exception as exc:
                        _clear_indicator()
                        turn_recovery.record_api_error()
                        # Classify through unified coordinator
                        envelope = self._unified_classifier.classify_llm_error(
                            exc, provider=getattr(self._llm, 'provider', ''),
                            model=getattr(self._llm, 'model', ''),
                        )
                        coordinator = self._recovery_coordinator
                        try:
                            decision = coordinator.evaluate(envelope)
                        except Exception as coord_exc:
                            logger.error("recovery_coordinator.evaluate() failed: %s", coord_exc)
                            yield StreamEvent(type="error", content=f"Internal recovery error: {coord_exc}")
                            break
                        self._audit_sink.record(create_audit_entry(
                            envelope, decision, coordinator.budget,
                            session_id=getattr(self, '_current_session_id', '') or '',
                            turn_id=budget.used,
                        ))
                        if decision.action == RecoveryAction.RETRY_WITH_BACKOFF:
                            if decision.retry_semantics.backoff_config:
                                await asyncio.sleep(
                                    jittered_backoff(budget.used, base=decision.retry_semantics.backoff_config.base_delay)
                                )
                            continue
                        elif decision.action == RecoveryAction.TRANSFORM_AND_RETRY:
                            self._execute_transform_decision(decision, messages)
                            coordinator.on_strategy_outcome(decision.decision_id, True)
                            continue
                        elif decision.action == RecoveryAction.FAILOVER:
                            if hasattr(self._llm, '_failover'):
                                self._llm._failover(f"recovery: {decision.reason}")
                            coordinator.on_strategy_outcome(decision.decision_id, True)
                            continue
                        else:
                            fatal_error = decision.reason
                            logger.error("unified_loop_stream: unrecoverable %s: %s", envelope.category, exc)
                            yield StreamEvent(type="error", content=decision.reason)
                            break

                    content = "".join(content_parts).strip()
                    if self._sanitizer:
                        content = self._sanitizer.sanitize(content)
                else:
                    try:
                        resp = await self._llm.achat(
                            compressed, stream=False, enable_thinking=planned_enable_thinking,
                        )
                        turn_recovery.record_api_success()
                        usage = resp.usage or {}
                        self._usage_tracker.record_api_call(
                            usage,
                            provider=getattr(self._llm, 'active_provider_name', ''),
                            model=resp.model or '',
                        )
                        provider_prompt = usage.get("prompt_tokens", 0)
                        if provider_prompt > 0:
                            self._last_context_tokens = provider_prompt
                    except Exception as exc:
                        _clear_indicator()
                        turn_recovery.record_api_error()
                        # Classify through unified coordinator
                        envelope = self._unified_classifier.classify_llm_error(
                            exc, provider=getattr(self._llm, 'provider', ''),
                            model=getattr(self._llm, 'model', ''),
                        )
                        coordinator = self._recovery_coordinator
                        try:
                            decision = coordinator.evaluate(envelope)
                        except Exception as coord_exc:
                            logger.error("recovery_coordinator.evaluate() failed: %s", coord_exc)
                            yield StreamEvent(type="error", content=f"Internal recovery error: {coord_exc}")
                            break
                        self._audit_sink.record(create_audit_entry(
                            envelope, decision, coordinator.budget,
                            session_id=getattr(self, '_current_session_id', '') or '',
                            turn_id=budget.used,
                        ))
                        if decision.action == RecoveryAction.RETRY_WITH_BACKOFF:
                            if decision.retry_semantics.backoff_config:
                                await asyncio.sleep(
                                    jittered_backoff(budget.used, base=decision.retry_semantics.backoff_config.base_delay)
                                )
                            continue
                        elif decision.action == RecoveryAction.TRANSFORM_AND_RETRY:
                            self._execute_transform_decision(decision, messages)
                            coordinator.on_strategy_outcome(decision.decision_id, True)
                            continue
                        elif decision.action == RecoveryAction.FAILOVER:
                            if hasattr(self._llm, '_failover'):
                                self._llm._failover(f"recovery: {decision.reason}")
                            coordinator.on_strategy_outcome(decision.decision_id, True)
                            continue
                        else:
                            fatal_error = decision.reason
                            logger.error("unified_loop_stream: unrecoverable %s: %s", envelope.category, exc)
                            yield StreamEvent(type="error", content=decision.reason)
                            break
                    _clear_indicator()
                    content = (resp.content or "").strip()
                    if self._sanitizer:
                        content = self._sanitizer.sanitize(content)

                    # Length continuation for non-stream path
                    finish = getattr(resp, 'finish_reason', None)
                    if finish in ("length", "max_tokens") and turn_recovery.try_length_continuation():
                        messages.append(build_assistant_message(content))
                        messages.append(build_user_message_text(build_continuation_prompt(content)))
                        continue

            self._persist_message(session_id, "assistant", content)
            tool_call = self._parse_tool_call_from_content(content)

            if tool_call is None:
                self._wm.remember_chat(build_assistant_message(content))
                trace.record(ExecutionMode.COMPLETE)
                if not content:
                    fallback = _app_onboarding_recovery_message(messages)
                    final_text = fallback or "I processed your request but have no additional output."
                    self._emit_chat_event("response", {"content": final_text[:500]})
                    yield StreamEvent(type="final", content=final_text)
                else:
                    permission_override = _permission_override_message(messages)
                    final_text = permission_override or content
                    self._emit_chat_event("response", {"content": final_text[:500]})
                    yield StreamEvent(type="final", content=final_text)
                return

            normalized_tool_call = _normalize_tool_call(tool_call)
            tool_name = str(normalized_tool_call["name"])
            original_tool_name = str(normalized_tool_call.get("original_tool_name", tool_name))
            self._wm.remember_chat(build_assistant_message(f"[Called: {tool_name}]"))

            messages.append(build_assistant_message(content))
            tool_arguments = normalized_tool_call.get("arguments")
            self._emit_chat_event("tool_call", {
                "tool_name": tool_name,
                "arguments_summary": json.dumps(tool_arguments, default=str, ensure_ascii=False)[:300] if tool_arguments else "",
            })
            yield StreamEvent(
                type="tool_start",
                content=tool_name,
                metadata=_tool_args_metadata(
                    tool_name,
                    tool_arguments,
                    original_tool_name=original_tool_name,
                ),
            )
            result = await self._execute_tool_with_ledger(
                normalized_tool_call, TOOL_HANDLERS, tool_call_id=f"text-{budget.used}",
            )
            _clear_indicator()
            self._emit_chat_event("tool_result", {
                "tool_name": tool_name,
                "ok": bool(result.get("ok")) if isinstance(result, dict) else True,
                "summary": json.dumps(result, default=str, ensure_ascii=False)[:300] if isinstance(result, dict) else str(result)[:300],
            })
            yield StreamEvent(
                type="tool_complete",
                content=tool_name,
                metadata={
                    **_tool_result_metadata(
                        tool_name,
                        tool_arguments,
                        result,
                        original_tool_name=original_tool_name,
                    ),
                    **self._tool_context_metadata(
                        tool_name,
                        tool_arguments,
                        result,
                    ),
                },
            )
            _print_tool_result(tool_name, result, enabled=self._settings.verbose_progress)
            trace.record(
                ExecutionMode.ACTING,
                action=normalized_tool_call,
                observation=result if isinstance(result, dict) else {"result": str(result)},
            )

            is_error = isinstance(result, dict) and _tool_result_counts_as_failure(result)
            if is_error:
                turn_recovery.record_tool_failure()
            else:
                turn_recovery.record_tool_success()

            result_payload = self._compact_tool_result(tool_name, tool_arguments, result)
            result_text = _truncate_result_for_budget(result_payload, result_budget)
            messages.append(build_user_message_text(
                f"Tool result ({tool_name}):\n{result_text}"
            ))
            self._persist_message(
                session_id, "tool", result_text,
                tool_name=tool_name, tool_call_id=f"text-{budget.used}",
                metadata=self._tool_execution_metadata(result),
            )

            if _is_permission_hard_stop_payload(result):
                logger.info(
                    "unified_loop_stream: permission hard-stop after %s/%s",
                    result.get("platform", "platform"),
                    result.get("capability") or result.get("action") or tool_name,
                )
                break

            if _is_retryable_unknown_tool_result(result) and not unknown_tool_retry_used:
                unknown_tool_retry_used = True
                messages.append(build_user_message_text(_unknown_tool_retry_prompt(result)))
                continue

            if is_error:
                halt_reason = self._evaluate_tool_failures([(tool_name, result)], turn_id=budget.used)
                if halt_reason:
                    fatal_error = halt_reason
                    break

            if self._check_guardrail(messages) == "halt":
                break

            if status == BudgetStatus.SOFT_LIMIT and not self._should_extend_budget(self._active_frame):
                messages.append(build_user_message_text(
                    "SYSTEM: Approaching limit. Provide final answer now."
                ))

        # Turn-end learning/memory-sync are top-level-turn concerns; a recursive
        # child frame (subagent) must not pollute the parent's evolution/memory
        # (its result flows back via SubagentResult) nor leak background tasks.
        if getattr(self._active_frame, "is_root", True) and self._memory_manager and self._settings.memory_integration_enabled:
            asyncio.create_task(self._sync_turn_safe(messages))

        if getattr(self._active_frame, "is_root", True) and self._evolution is not None and content:
            asyncio.create_task(self._post_turn_review(messages, content))

        llm = self._llm
        if hasattr(llm, 'try_restore_primary'):
            llm.try_restore_primary()

        logger.info("turn_usage: %s", self._usage_tracker.format_log_line())

        if content:
            permission_override = _permission_override_message(messages)
            final = permission_override or content
            self._emit_chat_event("response", {"content": final[:500]})
            yield StreamEvent(type="final", content=final)
        else:
            fallback = (
                _app_onboarding_recovery_message(messages)
                or _last_tool_failures_recovery_message(messages)
                or fatal_error
                or self._budget_exhausted_response(messages)
            )
            self._emit_chat_event("response", {"content": fallback[:500]})
            yield StreamEvent(type="final", content=fallback)


    # ── Unified Loop Helpers ───────────────────────────────────────────────

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
            params = ", ".join(
                func.get("parameters", {}).get("properties", {}).keys()
            )
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

        Reuses the robust parser from tool_executor.
        """
        from leapflow.skills.tool_executor import _parse_tool_call

        call = _parse_tool_call(content)
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
        result_budget = self._effective_tool_result_budget()
        executed: list[Dict[str, Any]] = []
        original_names_by_id = {str(tc.id): str(tc.name) for tc in native_calls}

        tc_wrappers = [
            ConcurrentToolCall(
                id=tc.id,
                name=str(_normalize_tool_call({"name": tc.name, "arguments": tc.arguments})["name"]),
                arguments=tc.arguments,
            )
            for tc in native_calls
        ]

        if not self._concurrency_policy or len(tc_wrappers) <= 1:
            for i, tc in enumerate(native_calls):
                original_name = str(tc.name)
                tool_call_dict = _normalize_tool_call({"name": original_name, "arguments": tc.arguments})
                normalized_name = str(tool_call_dict["name"])
                self._emit_chat_event("tool_call", {
                    "tool_name": normalized_name,
                    "arguments_summary": json.dumps(tc.arguments, default=str, ensure_ascii=False)[:300],
                })
                _show_progress("executing", normalized_name, step=i + 1, total=len(native_calls))
                result = await self._execute_tool_with_ledger(
                    tool_call_dict, handlers, tool_call_id=str(tc.id),
                )
                _clear_indicator()
                self._emit_chat_event("tool_result", {
                    "tool_name": normalized_name,
                    "ok": bool(result.get("ok")) if isinstance(result, dict) else True,
                    "summary": json.dumps(result, default=str, ensure_ascii=False)[:300] if isinstance(result, dict) else str(result)[:300],
                })
                _print_tool_result(normalized_name, result, enabled=self._settings.verbose_progress)
                trace.record(
                    ExecutionMode.ACTING,
                    action=tool_call_dict,
                    observation=result if isinstance(result, dict) else {"result": str(result)},
                )
                result_payload = self._compact_tool_result(normalized_name, tc.arguments, result)
                result_text = _truncate_result_for_budget(result_payload, result_budget)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})
                self._persist_message(
                    self._current_session_id, "tool", result_text,
                    tool_name=normalized_name, tool_call_id=str(tc.id),
                    metadata=self._tool_execution_metadata(result),
                )
                executed.append({
                    "id": tc.id,
                    "name": normalized_name,
                    "original_tool_name": str(tool_call_dict.get("original_tool_name") or original_name),
                    "arguments": tc.arguments,
                    "result": result,
                })
                if isinstance(result, dict) and _should_stop_after_tool_result(normalized_name, result):
                    for skipped_tc in native_calls[i + 1:]:
                        skipped_call = _normalize_tool_call({"name": str(skipped_tc.name), "arguments": skipped_tc.arguments})
                        skipped_name = str(skipped_call["name"])
                        executed.append({
                            "id": skipped_tc.id,
                            "name": skipped_name,
                            "original_tool_name": str(skipped_call.get("original_tool_name") or skipped_tc.name),
                            "arguments": skipped_tc.arguments,
                            "result": _skipped_after_failure_result(normalized_name, result),
                        })
                    logger.info(
                        "tool_concurrency: stopping remaining native tool calls after failed side effect from %s",
                        normalized_name,
                    )
                    break
            return executed

        concurrent, sequential = self._concurrency_policy.partition(tc_wrappers)
        logger.info(
            "tool_concurrency.execute concurrent=%d sequential=%d",
            len(concurrent),
            len(sequential),
        )

        # Execute concurrent group via asyncio.gather
        if concurrent:
            async def _run_one(ctc: ConcurrentToolCall) -> Dict[str, Any]:
                original_name = original_names_by_id.get(str(ctc.id), ctc.name)
                tool_call_dict = {
                    "name": ctc.name,
                    "arguments": ctc.arguments,
                    "original_tool_name": original_name,
                    "normalized_tool_name": ctc.name,
                }
                return await self._execute_tool_with_ledger(
                    tool_call_dict, handlers, tool_call_id=str(ctc.id),
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
                    _print_tool_result(ctc.name, error_result, enabled=self._settings.verbose_progress)
                    trace.record(
                        ExecutionMode.ACTING,
                        action=tool_call_dict,
                        observation=error_result,
                    )
                    result_payload = self._compact_tool_result(ctc.name, ctc.arguments, error_result)
                    result_text = _truncate_result_for_budget(result_payload, result_budget)
                else:
                    _print_tool_result(ctc.name, result, enabled=self._settings.verbose_progress)
                    trace.record(
                        ExecutionMode.ACTING,
                        action=tool_call_dict,
                        observation=result if isinstance(result, dict) else {"result": str(result)},
                    )
                    result_payload = self._compact_tool_result(ctc.name, ctc.arguments, result)
                    result_text = _truncate_result_for_budget(result_payload, result_budget)
                effective_result = error_result if isinstance(result, Exception) else result
                messages.append({"role": "tool", "tool_call_id": ctc.id, "content": result_text})
                self._persist_message(
                    self._current_session_id, "tool", result_text,
                    tool_name=ctc.name, tool_call_id=str(ctc.id),
                    metadata=self._tool_execution_metadata(effective_result),
                )
                executed.append({
                    "id": ctc.id,
                    "name": ctc.name,
                    "original_tool_name": original_name,
                    "arguments": ctc.arguments,
                    "result": effective_result,
                })
                if isinstance(effective_result, dict) and _should_stop_after_tool_result(ctc.name, effective_result):
                    for skipped_ctc in sequential:
                        skipped_original = original_names_by_id.get(str(skipped_ctc.id), skipped_ctc.name)
                        executed.append({
                            "id": skipped_ctc.id,
                            "name": skipped_ctc.name,
                            "original_tool_name": skipped_original,
                            "arguments": skipped_ctc.arguments,
                            "result": _skipped_after_failure_result(ctc.name, effective_result),
                        })
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
                tool_call_dict, handlers, tool_call_id=str(ctc.id),
            )
            _clear_indicator()
            _print_tool_result(ctc.name, result, enabled=self._settings.verbose_progress)
            trace.record(
                ExecutionMode.ACTING,
                action=tool_call_dict,
                observation=result if isinstance(result, dict) else {"result": str(result)},
            )
            result_payload = self._compact_tool_result(ctc.name, ctc.arguments, result)
            result_text = _truncate_result_for_budget(result_payload, result_budget)
            messages.append({"role": "tool", "tool_call_id": ctc.id, "content": result_text})
            self._persist_message(
                self._current_session_id, "tool", result_text,
                tool_name=ctc.name, tool_call_id=str(ctc.id),
                metadata=self._tool_execution_metadata(result),
            )
            executed.append({
                "id": ctc.id,
                "name": ctc.name,
                "original_tool_name": original_name,
                "arguments": ctc.arguments,
                "result": result,
            })
            if isinstance(result, dict) and _should_stop_after_tool_result(ctc.name, result):
                for skipped_ctc in sequential[i + 1:]:
                    skipped_original = original_names_by_id.get(str(skipped_ctc.id), skipped_ctc.name)
                    executed.append({
                        "id": skipped_ctc.id,
                        "name": skipped_ctc.name,
                        "original_tool_name": skipped_original,
                        "arguments": skipped_ctc.arguments,
                        "result": _skipped_after_failure_result(ctc.name, result),
                    })
                logger.info(
                    "tool_concurrency: stopping sequential native tool calls after failed side effect from %s",
                    ctc.name,
                )
                break
        return executed

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
            return await self._execute_general_tool(tool_call, handlers)

        tool_name = resolution.normalized_name
        spec = registry.specs.get(tool_name)
        policy = execution_policy_for(tool_name, spec)
        if getattr(self._settings, "agent_validate_tool_args", True):
            invalid_args = _validate_tool_arguments(spec, args)
            if invalid_args is not None:
                logger.info("tool_args_invalid: tool=%s missing=%s", tool_name, invalid_args.get("missing"))
                return invalid_args
        session_id = self._current_session_id or "ephemeral"
        turn_id = self._current_turn_id or f"turn-{self._session_turn_count}"
        command_id = self._current_command_id or turn_id
        normalized_call = {
            **tool_call,
            "name": tool_name,
            "arguments": args,
            "original_tool_name": original_name,
            "normalized_tool_name": tool_name,
        }
        record, existing = self._tool_execution_ledger.reserve(
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
                existing = await self._tool_execution_ledger.wait_for_completion(
                    existing,
                    timeout_s=self._tool_timeouts.get(tool_name, self._default_tool_timeout_s),
                )
            duplicate = ToolExecutionLedger.duplicate_result(existing)
            duplicate.update({
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "execution_policy": existing.policy,
            })
            logger.info(
                "tool_idempotency: skipped duplicate tool=%s policy=%s key=%s",
                tool_name, existing.policy, existing.idempotency_key[:12],
            )
            return duplicate

        try:
            result = await self._execute_general_tool(normalized_call, handlers)
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
            self._tool_execution_ledger.complete(record, failed_result)
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
        completed = self._tool_execution_ledger.complete(record, result_for_ledger)
        result_for_ledger["execution_status"] = completed.status
        return result_for_ledger

    async def _execute_general_tool(
        self, tool_call: Dict[str, Any], handlers: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a general-purpose tool via ToolBridge (preferred) or TOOL_HANDLERS fallback.

        Routing priority:
        1. ToolBridge dispatch (gp_-prefixed) — local Python GP tools, always available
        2. ToolBridge dispatch (exact name) — may route to ExecutionPort or semantic tools
        3. TOOL_HANDLERS dict (static fallback when no bridge)

        Security: untrusted tool results (MCP, web) are wrapped with delimiters.
        Secrets in error messages are redacted before returning to LLM.
        """
        from leapflow.skills.tool_executor import ToolCall as TC
        from leapflow.security.redact import redact_sensitive_text

        original_name = str(tool_call.get("original_tool_name") or tool_call.get("name", ""))
        proposed_name = str(tool_call.get("name", ""))
        args = tool_call.get("arguments", {})
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

        timeout = self._tool_timeouts.get(name, self._default_tool_timeout_s)
        t0 = time.perf_counter()

        try:
            # Route through ToolBridge when available (single source of truth)
            if self._tool_bridge is not None:
                prefixed = f"gp_{name}"
                result = await asyncio.wait_for(
                    self._tool_bridge.dispatch(TC(name=prefixed, params=args)),
                    timeout=timeout,
                )
                if not (isinstance(result, dict) and "unknown_tool" in str(result.get("error", ""))):
                    duration = (time.perf_counter() - t0) * 1000
                    is_ok = not (isinstance(result, dict) and not result.get("ok", True))
                    self._usage_tracker.record_tool_call(name, is_ok, duration)
                    return self._post_process_tool_result(name, result)
                result = await asyncio.wait_for(
                    self._tool_bridge.dispatch(TC(name=name, params=args)),
                    timeout=timeout,
                )
                if not (isinstance(result, dict) and "unknown_tool" in str(result.get("error", ""))):
                    duration = (time.perf_counter() - t0) * 1000
                    is_ok = not (isinstance(result, dict) and not result.get("ok", True))
                    self._usage_tracker.record_tool_call(name, is_ok, duration)
                    return self._post_process_tool_result(name, result)

            # Fallback: direct handler dispatch
            handler = handlers.get(name)
            if handler is None:
                missing_resolution = registry.resolve(original_name, args)
                return registry.unknown_result(missing_resolution)

            result = await asyncio.wait_for(handler(args), timeout=timeout)
        except asyncio.TimeoutError:
            duration = (time.perf_counter() - t0) * 1000
            self._usage_tracker.record_tool_call(name, False, duration)
            return {"ok": False, "error": f"Tool '{name}' timed out after {timeout:.0f}s"}
        except Exception as e:
            duration = (time.perf_counter() - t0) * 1000
            self._usage_tracker.record_tool_call(name, False, duration)
            error_msg = redact_sensitive_text(str(e), force=True)
            return {"ok": False, "error": error_msg}

        duration = (time.perf_counter() - t0) * 1000
        is_ok = not (isinstance(result, dict) and not result.get("ok", True))
        self._usage_tracker.record_tool_call(name, is_ok, duration)

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

    # ── Helpers ──────────────────────────────────────────────────────────

    def _budget_exhausted_response(self, messages: List[Dict[str, Any]]) -> str:
        """Response when the iteration hard cap is reached.

        When the research ledger shows unfinished work, surface the remaining
        open-question count and next step so the stop is informative and
        continuable (not a bare dead-stop); otherwise the plain notice.
        """
        base = "I've reached my reasoning step limit. Here's my best answer based on progress so far."
        led = self._research_ledger
        if led.is_empty or led.open_question_count == 0:
            return base
        d = led.as_dict()
        parts = [
            base,
            "",
            f"Note: {led.open_question_count} open question(s) remain — the task is not fully complete.",
        ]
        next_step = d.get("next_step", "")
        if next_step:
            parts.append(f"Suggested next step: {next_step}")
        return "\n".join(parts)

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
                    context={
                        "steps": trace.step_count,
                        "tokens": trace.total_tokens,
                        **build_adaptive_learning_signal(self._last_context_snapshot or {}),
                    },
                )
                logger.debug("evolution.record_episode outcome=%s actions=%d", outcome, len(actions))
        except Exception:
            pass  # never fail the main loop

    def _ensure_session_for_frame(self, frame: AgentLoopFrame, user_text: str) -> Optional[str]:
        """Resolve the persistence session for a loop frame (S4-E isolation).

        Root frames reuse the turn's conversation session; a recursive child
        frame (subagent) gets its *own* isolated ``sub_`` session so its
        transcript is persisted separately and never mixes into the parent
        turn's conversation.
        """
        if frame.is_root:
            return self._ensure_session(user_text)
        if not self._conversation_store or not self._settings.session_persistence_enabled:
            return None
        try:
            import uuid as _uuid
            child_session = f"sub_{_uuid.uuid4().hex[:12]}"
            title = user_text[:80].replace("\n", " ").strip() or "subagent"
            self._conversation_store.create_session(
                child_session, title=title, model=self._settings.llm_model, source="subagent",
            )
            return child_session
        except Exception:
            logger.debug("child session creation failed; skipping child persistence", exc_info=True)
            return None

    def _ensure_session(self, user_text: str) -> Optional[str]:
        """Create or reuse a conversation session. Returns session_id or None."""
        if not self._conversation_store or not self._settings.session_persistence_enabled:
            return None
        try:
            import uuid as _uuid
            if self._current_session_id is None:
                self._current_session_id = _uuid.uuid4().hex[:16]
                title = user_text[:80].replace("\n", " ").strip()
                self._conversation_store.create_session(
                    self._current_session_id, title=title,
                    model=self._settings.llm_model, source="cli",
                )
            self._persist_message(self._current_session_id, "user", user_text)
            return self._current_session_id
        except Exception:
            logger.debug("session.ensure failed", exc_info=True)
            return None

    @staticmethod
    def _tool_execution_metadata(result: Any) -> Dict[str, Any]:
        """Extract tool execution audit metadata for transcript rows."""
        if not isinstance(result, dict):
            return {}
        metadata: Dict[str, Any] = {}
        for key in (
            "execution_id", "idempotency_key", "execution_status", "execution_policy",
            "already_executed", "duplicate_suppressed", "execution_reused", "execution_skipped",
            "counts_as_failure", "counts_as_tool_attempt", "ui_hidden", "skipped_reason",
            "blocked_by_tool", "blocked_by_error", "tool_call_id", "path", "file_path",
            "bytes_written",
        ):
            if key in result:
                metadata[key] = result[key]
        return metadata

    def _persist_message(
        self, session_id: Optional[str], role: str, content: str,
        *, tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a message to conversation store (fire-and-forget)."""
        if not session_id or not self._conversation_store:
            return
        try:
            self._conversation_store.append_message(
                session_id, role, content[:8000],
                tool_name=tool_name, tool_call_id=tool_call_id,
                tool_calls=tool_calls, metadata=metadata,
            )
        except Exception:
            logger.debug("session.persist_message failed", exc_info=True)

    async def _prefetch_and_freeze_memory(self, user_text: str) -> str:
        """Prefetch memory context and freeze snapshot for session duration.

        Combines narrative memory (always-on MEMORY.md) with signal-based
        prefetch results into a unified context block.
        """
        if self._memory_context_snapshot is not None:
            return self._memory_context_snapshot

        if not self._memory_manager or not self._settings.memory_integration_enabled:
            self._memory_context_snapshot = ""
            return ""

        parts: list[str] = []

        # Layer 1: Narrative memory (MEMORY.md — always loaded, no timeout)
        narrative = self._memory_manager.get_provider("narrative")
        if narrative is not None and hasattr(narrative, "context_block"):
            try:
                block = narrative.context_block()
                if block:
                    parts.append(block)
            except Exception:
                logger.debug("narrative.context_block failed", exc_info=True)

        # Layer 2: Signal-based prefetch (DuckDB — timeout-bounded)
        try:
            entries = await asyncio.wait_for(
                self._memory_manager.prefetch(
                    user_text,
                    limit=self._settings.memory_prefetch_limit,
                    workspace_root=(
                        self._current_task_contract.workspace_root
                        if self._current_task_contract else ""
                    ),
                    task_id=(
                        self._current_task_contract.task_id
                        if self._current_task_contract else ""
                    ),
                    scope_keywords=self._task_scope_keywords(user_text),
                ),
                timeout=self._settings.memory_prefetch_timeout_s,
            )
            if entries:
                parts.append("## Recent Context\n" + "\n".join(
                    f"- [{e.kind.value}] {e.content[:500]}" for e in entries
                ))
        except asyncio.TimeoutError:
            logger.debug(
                "memory.prefetch timed out (%.1fs)", self._settings.memory_prefetch_timeout_s,
            )
        except Exception:
            logger.debug("memory.prefetch failed", exc_info=True)

        self._memory_context_snapshot = "\n\n".join(parts)
        return self._memory_context_snapshot

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
                    if parsed.get("counts_as_failure") is False or _tool_result_is_control_signal(parsed):
                        continue
            except (json.JSONDecodeError, ValueError):
                pass
            # Non-JSON or ok!=False — treat as success, reset
            return 0
        return count

    async def _try_trigger_match(self, user_text: str) -> Optional[str]:
        """Check if a learned skill directly matches the user's request.

        Returns the skill output if a high-confidence match is found,
        or None to fall through to the ReAct/DAG path.

        Enforces Progressive Trust: the ConfirmationHandler determines
        whether the skill requires user confirmation before execution.
        """
        matches = self._registry.find_by_trigger(user_text, threshold=0.5)
        if not matches:
            return None

        best = matches[0]
        if best.metadata.source not in ("distilled", "template"):
            return None
        if best.metadata.confidence < 0.6:
            return None

        from leapflow.engine.confirmation import ConfirmationHandler, ConfirmLevel

        handler = ConfirmationHandler(skill_store=self._skill_library)
        level = handler.determine_level(best)

        if level in (ConfirmLevel.STEP, ConfirmLevel.CONFIRM):
            logger.info(
                "audit.trigger_match_deferred skill=%s tier=%s (requires confirmation)",
                best.name, best.metadata.tier.name,
            )
            return None

        logger.info(
            "audit.trigger_match skill=%s confidence=%.2f level=%s",
            best.name, best.metadata.confidence, level.value,
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

    # Patterns that indicate a genuine teach session command.
    # Uses regex word-boundary checks to avoid false positives like
    # "teaching methods for math".
    _TEACH_COMMAND_RE = re.compile(
        r"^(?:"
        r"(?:start\s+)?teach(?:ing)?(?:\s+(?:this|that|it|me|now))?$"
        r"|stop\s+teach(?:ing)?"
        r"|pause\s+teach(?:ing)?"
        r"|resume\s+teach(?:ing)?"
        r"|done\s+teach(?:ing)?"
        r"|finish\s+teach(?:ing)?"
        r"|end\s+teach(?:ing)?"
        r"|教(?:我|一下)?$"
        r"|开始教学"
        r"|停止教学|暂停教学|继续教学|结束教学"
        r"|watch\s+me"
        r")",
        re.IGNORECASE,
    )

    def _is_teach_command(self, text: str) -> bool:
        """Check if text is a teach command that needs special session handling.

        Uses regex matching to avoid false positives like 'teach me how to cook'
        which should go through the unified tool loop.
        """
        stripped = text.strip()
        return bool(self._TEACH_COMMAND_RE.match(stripped))

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
            result = await self._execute_tool_with_ledger(
                tool_call_dict, TOOL_HANDLERS, tool_call_id=f"action-{name}",
            )
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
