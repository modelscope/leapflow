# Copyright (c) Alibaba, Inc. and its affiliates.
"""Pure helper functions for building, parsing, and formatting LLM messages.

Every function here is a module-level free function with no dependency on
AgentEngine state.  Extracted from ``engine.py`` to reduce file size.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Optional

from leapflow.security.permission_failures import (
    is_permission_failure_payload,
    is_permission_hard_stop_payload,
)
from leapflow.engine.tools.tool_execution import (
    effect_is_uncertain_on_failure,
    exit_code_from,
)
from leapflow.engine._tool_helpers import _resolve_tool_name

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TOOL_ARGS_PREVIEW_LIMIT = 160
_TOOL_RESULT_PREVIEW_LIMIT = 240
_TASK_CONTRACT_HEADING = "## Task Contract"

# Empty-response hardening: an LLM call that "succeeds" with empty content is a
# failure signal, never a valid answer. It gets one bounded retry with an
# explicit nudge; a second empty response produces a transparent degraded
# message instead of a fake-success filler.
_EMPTY_RESPONSE_RETRY_PROMPT = (
    "SYSTEM: Your previous reply was empty. Respond to the user's request now "
    "with substantive content. If you cannot help, say so explicitly."
)

_EMPTY_RESPONSE_DEGRADED_MESSAGE = (
    "The model returned an empty response twice, so no answer was produced for "
    "this turn. This is usually transient (e.g., provider or runtime warm-up "
    "right after startup) \u2014 please resend your message."
)

# Injected for the single tool-free round that runs when the loop stops before
# the model has written an answer (a detected repetition loop or an exhausted
# iteration budget). Breaking cold otherwise leaves the user with a generic
# "reasoning step limit" notice and none of the information the tools already
# returned; this asks the model to answer from what it has, with tools withheld
# so it cannot resume the loop.
_FORCED_FINALIZE_PROMPT = (
    "SYSTEM: No further tool calls are available for this turn. Do not attempt "
    "to call any tool. Answer the user's request directly and concisely using "
    "the information already gathered above. If part of it cannot be determined "
    "from what you have, say so plainly and state what would be needed \u2014 do "
    "not repeat an earlier tool call."
)

_SIDE_EFFECT_STOP_POLICIES = frozenset(
    {"external_side_effect", "mutating_once", "mutating_idempotent"}
)

# ---------------------------------------------------------------------------
# Preview / truncation helpers
# ---------------------------------------------------------------------------


def _single_line_preview(value: Any, *, limit: int, keep_tail: bool = False) -> str:
    """Return a compact single-line preview for UI metadata.

    ``keep_tail`` preserves both ends. Diagnostic text states its cause last — a
    traceback's final line, a compiler's error summary — so a head-only cut shows
    the least informative part of exactly the output a user needs to read.
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    if keep_tail:
        head = max(1, (limit - 1) * 2 // 5)
        tail = max(1, limit - 1 - head)
        return compact[:head] + "…" + compact[-tail:]
    return compact[: limit - 1] + "…"


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
            sample = json.dumps(v[: min(4, orig_len)], default=str, ensure_ascii=False)
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
        sentinel = json.dumps(
            {
                "ok": payload.get("ok"),
                "kind": payload.get("kind", ""),
                "truncated": True,
                "original_chars": len(text),
                "budget_chars": budget,
            },
            default=str,
            ensure_ascii=False,
        )
        return sentinel

    # Non-dict: hard string cut is unavoidable; the LLM sees a partial raw value.
    return text[:budget]


# ---------------------------------------------------------------------------
# Tool metadata builders
# ---------------------------------------------------------------------------


def _tool_args_metadata(
    tool_name: str,
    arguments: Dict[str, Any] | None,
    *,
    original_tool_name: str | None = None,
    tool_call_id: str = "",
) -> Dict[str, Any]:
    """Build safe, compact tool-start metadata for streaming UIs.

    ``tool_call_id`` is included so a UI can correlate a start with its own
    completion: a parallel batch emits several starts before any finishes, and
    without the id a renderer can only track "the last tool", which mislabels
    every line in the batch.
    """
    args = dict(arguments or {})
    original_name = original_tool_name or tool_name
    metadata: Dict[str, Any] = {
        "tool_name": tool_name,
        "original_tool_name": original_name,
        "normalized_tool_name": tool_name,
        "args_summary": _single_line_preview(args, limit=_TOOL_ARGS_PREVIEW_LIMIT),
    }
    if tool_call_id:
        metadata["tool_call_id"] = tool_call_id
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
    tool_call_id: str = "",
) -> Dict[str, Any]:
    """Build safe, compact tool-completion metadata for streaming UIs."""
    metadata = _tool_args_metadata(
        tool_name,
        arguments,
        original_tool_name=original_tool_name,
        tool_call_id=tool_call_id,
    )
    if tool_name in {"platform_action", "gp_platform_action"} and arguments:
        for key in ("platform", "action"):
            value = arguments.get(key)
            if value:
                metadata[key] = _single_line_preview(value, limit=_TOOL_ARGS_PREVIEW_LIMIT)
    metadata["ok"] = True
    if isinstance(result, dict):
        metadata["ok"] = bool(result.get("ok", True))
        exit_code = exit_code_from(result)
        if exit_code is not None:
            metadata["exit_code"] = exit_code
        for key in ("path", "lines", "truncated", "bytes_written"):
            if key in result:
                metadata[key] = result[key]
        for key in (
            "error_type",
            "retryable",
            "resolution_status",
            "resolution_confidence",
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
            "execution_id",
            "idempotency_key",
            "execution_status",
            "execution_policy",
            "tool_call_id",
            # Must reach the model: a failed side effect whose fate is unknown
            # needs verification, not a blind retry.
            "side_effect_uncertain",
            "retry_guidance",
        ):
            if key in result:
                metadata[key] = result[key]
        # App Connector authorization failure metadata
        for key in (
            "failure_class",
            "failure_code",
            "recoverability",
            "blocks_approval",
            "platform",
            "action",
            "capability",
            "missing_scopes",
            "required_scopes",
            "scope_relation",
            "scope_source",
            "console_url",
            "next_steps",
            "skip_approval",
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
                    # On failure these fields carry the diagnosis, and the cause is
                    # at the end of them.
                    keep_tail=metadata["ok"] is False and key in {"stderr", "error", "stdout"},
                )
        # App Connector recovery metadata for TUI transparency
        recovery_hint = result.get("recovery_hint")
        if recovery_hint:
            metadata["recovery_hint"] = _single_line_preview(
                recovery_hint, limit=_TOOL_RESULT_PREVIEW_LIMIT
            )
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


# ---------------------------------------------------------------------------
# Tool result classification
# ---------------------------------------------------------------------------


def _is_retryable_unknown_tool_result(result: Any) -> bool:
    """Return whether a tool result can drive a one-shot name correction retry."""
    return (
        isinstance(result, dict)
        and result.get("error_type") == "unknown_tool"
        and bool(result.get("retryable", False))
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


def _tool_result_counts_as_failure(payload: Dict[str, Any]) -> bool:
    """Return whether a tool payload represents a real failed execution attempt."""
    if payload.get("counts_as_failure") is False:
        return False
    if _tool_result_is_control_signal(payload):
        return False
    return payload.get("ok") is False


def _tool_result_is_control_signal(payload: Dict[str, Any]) -> bool:
    """Return whether a tool payload is execution control metadata, not an attempt result."""
    return bool(
        payload.get("already_executed")
        or payload.get("duplicate_suppressed")
        or payload.get("execution_skipped")
    )


def _tool_failure_text(payload: Dict[str, Any]) -> str:
    """Return the most useful root-cause text from a failed tool payload."""
    for key in ("error", "stderr", "stdout", "message"):
        value = payload.get(key)
        if value:
            return str(value)
    return "unknown error"


def _terminal_failure_text(decision: Any) -> str:
    """Render a terminal recovery decision for the user.

    When the decision carries an ``InteractionRequest``, its title, description,
    and suggested actions are what the user needs in order to act; the raw
    ``reason`` is written for the audit log. Falling back to ``reason`` alone
    (the previous behavior) told the user a turn had stopped without saying what
    to do about it.
    """
    interaction = getattr(decision, "interaction", None)
    if interaction is None:
        return str(getattr(decision, "reason", "") or "")

    lines = [str(interaction.title or "Input needed to continue")]
    if interaction.description:
        lines.append(str(interaction.description))
    for action in interaction.suggested_actions or ():
        label = str(getattr(action, "label", "") or "")
        command = str(getattr(action, "command", "") or "")
        entry = f"  - {label}" if label else "  -"
        if command:
            entry += f": {command}"
        lines.append(entry)
    return "\n".join(line for line in lines if line.strip())


def _interaction_metadata(decision: Any) -> Dict[str, Any]:
    """Return the structured InteractionRequest payload, or ``{}``.

    Carried on the stream event so the TUI/gateway can render a typed prompt and
    resume via ``resumption_key`` instead of parsing the message text.
    """
    interaction = getattr(decision, "interaction", None)
    if interaction is None:
        return {}
    return {
        "interaction": {
            "request_id": interaction.request_id,
            "interaction_type": getattr(
                interaction.interaction_type, "value", str(interaction.interaction_type)
            ),
            "severity": getattr(interaction.severity, "value", str(interaction.severity)),
            "title": interaction.title,
            "description": interaction.description,
            "suggested_actions": [
                {
                    "label": str(getattr(action, "label", "") or ""),
                    "command": str(getattr(action, "command", "") or ""),
                    "description": str(getattr(action, "description", "") or ""),
                    "is_default": bool(getattr(action, "is_default", False)),
                }
                for action in interaction.suggested_actions or ()
            ],
            "resumption_key": interaction.resumption_key,
            "timeout_behavior": getattr(
                interaction.timeout_behavior, "value", str(interaction.timeout_behavior)
            ),
            "context": interaction.context_dict,
        }
    }


def _annotate_uncertain_effect(payload: Dict[str, Any], policy: str) -> Dict[str, Any]:
    """Mark a failed side-effecting result whose effect may already have landed.

    A timeout or transport error on an outbound send does not mean the message
    was not delivered, so the model must verify before resending. Without this
    the failure reads as a plain "did not happen" and the natural next step is a
    blind retry that duplicates the effect. Batch-level protection already stops
    the rest of the batch (see ``_should_stop_after_tool_result``); this carries
    the same knowledge across turns, where the model decides what to do next.

    Advisory by design: only the tool's own state can settle whether the effect
    landed, so a hard block would also reject legitimate retries (e.g. resending
    after fixing an argument).
    """
    if not _tool_result_counts_as_failure(payload):
        return payload
    if not effect_is_uncertain_on_failure(policy):
        return payload
    payload["side_effect_uncertain"] = True
    payload["retry_guidance"] = (
        "This operation may already have taken effect despite the error. "
        "Verify the current state before retrying; do not simply repeat the call."
    )
    return payload


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


def _skipped_after_failure_result(
    blocking_tool: str, blocking_result: Dict[str, Any]
) -> Dict[str, Any]:
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
    where = (
        f"`{platform}.{capability}`"
        if platform and capability
        else (capability or platform or "this action")
    )
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


def _build_native_tool_assistant_message(
    native_calls: List[Any],
    *,
    thinking_content: Any = None,
) -> Dict[str, Any]:
    """Build a provider-valid assistant message that precedes tool results.

    ``reasoning_content`` is protocol continuation data for thinking-capable
    OpenAI-compatible providers such as DeepSeek. It is intentionally preserved
    verbatim only when the provider returned it, while the visible preamble stays
    excluded from the model context and durable transcript.
    """
    message: Dict[str, Any] = {"role": "assistant", "content": ""}
    if isinstance(thinking_content, str) and thinking_content:
        message["reasoning_content"] = thinking_content
    message["tool_calls"] = [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(call.arguments, ensure_ascii=False),
            },
        }
        for call in native_calls
    ]
    return message


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
    elif failure_code == "missing_required_fields" or "Missing required fields" in error:
        # TODO: migrate to failure_code-only once all producers emit
        # failure_code="missing_required_fields" instead of bare error text.
        lines.append(
            f"Action parameter incomplete: {error}. Please provide the missing field(s) and retry."
        )
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
        lines.append(
            "After completing the missing step, continue the same onboarding flow; LeapFlow will reuse the pending App Connector state."
        )
        return "\n".join(lines)
    return ""


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def _estimate_text_tokens(text: str) -> int:
    """Approximate token count for status display when provider usage is absent."""
    if not text:
        return 0
    cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f")
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


# ---------------------------------------------------------------------------
# Progress / display helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Misc utility
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no json object")
    return json.loads(text[start : end + 1])


def _keywords_from_query(q: str) -> list[str]:
    tokens: list[str] = []
    for segment in re.findall(r"[\u4e00-\u9fff]+|[\w\-./]+", q):
        if re.match(r"[\u4e00-\u9fff]", segment):
            if len(segment) == 1:
                tokens.append(segment)
            else:
                for i in range(len(segment) - 1):
                    tokens.append(segment[i : i + 2])
        elif len(segment) >= 2:
            tokens.append(segment)
    return tokens[:12]
