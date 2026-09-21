# Copyright (c) Alibaba, Inc. and its affiliates.
"""Adaptive context governance primitives for every agent interaction.

The module keeps context accounting, overflow prevention, exploration-ledger
tracking, and tool-result compaction independent from AgentEngine so all normal
interactions get long-task resilience without exposing a separate user mode.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence, runtime_checkable

from leapflow.engine.context.context_compressor import estimate_text_tokens as _estimate_text_tokens

logger = logging.getLogger(__name__)

_IMAGE_TOKEN_ESTIMATE = 1600
_MESSAGE_OVERHEAD_TOKENS = 8
_TOOL_SCHEMA_OVERHEAD_TOKENS = 12
_DEFAULT_HEAD_RATIO = 0.55
_DEFAULT_TAIL_RATIO = 0.25
_EVIDENCE_TOOLS = frozenset({
    "file_read", "gp_file_read",
    "file_list", "gp_file_list",
    "shell_run", "gp_shell_run",
})
_POSTURE_BASELINE = "baseline"
_POSTURE_EXPANDED = "expanded"
_POSTURE_RESEARCH = "research"
_POSTURE_EXPANDING = "expanding"
_POSTURE_CONVERGING = "converging"
_POSTURE_FINALIZING = "finalizing"


@runtime_checkable
class MessageCompressor(Protocol):
    """Protocol for components that can aggressively shrink chat messages."""

    def force_compress(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return a smaller message list while preserving recent intent."""
        ...


@dataclass(frozen=True)
class ContextBudgetSnapshot:
    """Observed prompt payload size before an LLM call."""

    message_tokens: int
    tool_schema_tokens: int
    total_tokens: int
    context_length: int
    ratio: float

    @property
    def percent(self) -> int:
        """Rounded utilization percentage."""
        return int(self.ratio * 100)


@dataclass(frozen=True)
class ContextBudgetDecision:
    """Result of preparing an LLM payload for the active context window."""

    messages: List[Dict[str, Any]]
    snapshot: ContextBudgetSnapshot
    compressed: bool = False
    forced_final_answer: bool = False
    notice: str = ""


# Estimator self-calibration. The character heuristic cannot match a real
# tokenizer, and the residual error scales with the window: on a 1M budget a 15%
# underestimate is ~150K tokens of invisible headroom loss, which either trips
# provider overflow or wastes the window. Rather than shipping a tokenizer per
# vendor (a dependency that goes stale exactly like the capability table), the
# estimator learns its own correction from provider-reported prompt_tokens.
_CALIBRATION_SMOOTHING = 0.25
_CALIBRATION_MIN_FACTOR = 0.5
_CALIBRATION_MAX_FACTOR = 2.5
_CALIBRATION_MIN_SAMPLE_TOKENS = 200


class ContextBudgetEstimator:
    """Estimate provider-visible prompt tokens including tool schemas.

    Self-calibrating: ``observe_actual`` feeds back a provider-reported prompt
    token count so the character heuristic converges on the active model and
    language mix. Uncalibrated behaviour is unchanged, so this only ever narrows
    the gap between the estimate and the number the provider actually charges.
    """

    def __init__(self) -> None:
        self._calibration: float = 1.0
        self._samples: int = 0

    @property
    def calibration_factor(self) -> float:
        """Current correction applied to raw heuristic estimates."""
        return self._calibration

    @property
    def calibration_samples(self) -> int:
        """How many provider observations have shaped the current factor."""
        return self._samples

    def observe_actual(self, *, estimated: int, actual: int) -> None:
        """Fold a provider-reported prompt token count into the calibration.

        ``estimated`` is a value this estimator produced, so it already carries
        the current factor. It is divided back out before comparing: measuring
        the corrected estimate against the actual would make the observed ratio
        approach 1.0 as soon as calibration started working, dragging the factor
        back toward its uncalibrated value. The comparison must always be against
        the raw heuristic.

        Uses an exponential moving average so a single outlier cannot swing the
        factor, and clamps the result: a wildly wrong factor would be worse than
        the raw heuristic, and the observed ratio can be distorted by
        provider-side prompt caching or injected system content.

        Tiny prompts are ignored — fixed per-request overhead dominates them and
        would bias the factor upward for every later estimate.
        """
        if estimated < _CALIBRATION_MIN_SAMPLE_TOKENS or actual <= 0:
            return
        raw = estimated / self._calibration if self._samples else float(estimated)
        if raw <= 0:
            return
        observed = actual / raw
        if not (_CALIBRATION_MIN_FACTOR <= observed <= _CALIBRATION_MAX_FACTOR):
            return
        if self._samples == 0:
            self._calibration = observed
        else:
            self._calibration += _CALIBRATION_SMOOTHING * (observed - self._calibration)
        self._calibration = min(
            _CALIBRATION_MAX_FACTOR, max(_CALIBRATION_MIN_FACTOR, self._calibration),
        )
        self._samples += 1

    def _calibrated(self, raw: int) -> int:
        if self._samples == 0 or raw <= 0:
            return raw
        return max(1, int(round(raw * self._calibration)))

    def estimate_messages(self, messages: Sequence[Dict[str, Any]]) -> int:
        """Estimate token usage for chat messages and tool-call envelopes."""
        if not messages:
            return 0
        total = 3
        for message in messages:
            total += _MESSAGE_OVERHEAD_TOKENS
            total += self._estimate_value(message.get("role", ""))
            total += self._estimate_value(message.get("content", ""))
            total += self._estimate_tool_calls(message.get("tool_calls", []))
            total += self._estimate_value(message.get("tool_call_id", ""))
        return max(1, self._calibrated(total))

    def estimate_tools(self, tools: Any) -> int:
        """Estimate token usage for native function/tool schemas."""
        if not tools:
            return 0
        try:
            text = json.dumps(tools, ensure_ascii=False, default=str, sort_keys=True)
        except (TypeError, ValueError):
            text = str(tools)
        return self._calibrated(_TOOL_SCHEMA_OVERHEAD_TOKENS + self._estimate_text(text))

    def snapshot(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        tools: Any = None,
        context_length: int,
    ) -> ContextBudgetSnapshot:
        """Build a complete prompt-budget snapshot."""
        safe_context = max(1, int(context_length or 1))
        message_tokens = self.estimate_messages(messages)
        tool_schema_tokens = self.estimate_tools(tools)
        total_tokens = message_tokens + tool_schema_tokens
        return ContextBudgetSnapshot(
            message_tokens=message_tokens,
            tool_schema_tokens=tool_schema_tokens,
            total_tokens=total_tokens,
            context_length=safe_context,
            ratio=total_tokens / safe_context,
        )

    def _estimate_value(self, value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return self._estimate_text(value)
        if isinstance(value, list):
            return sum(self._estimate_content_part(item) for item in value)
        if isinstance(value, dict):
            try:
                return self._estimate_text(json.dumps(value, ensure_ascii=False, default=str))
            except (TypeError, ValueError):
                return self._estimate_text(str(value))
        return self._estimate_text(str(value))

    def _estimate_content_part(self, item: Any) -> int:
        if not isinstance(item, dict):
            return self._estimate_value(item)
        part_type = str(item.get("type", ""))
        if part_type in {"image_url", "input_image", "image"}:
            return _IMAGE_TOKEN_ESTIMATE
        if part_type == "text":
            return self._estimate_text(str(item.get("text", "")))
        return self._estimate_value(item)

    def _estimate_tool_calls(self, tool_calls: Any) -> int:
        if not tool_calls:
            return 0
        return self._estimate_value(tool_calls)

    @staticmethod
    def _estimate_text(text: str) -> int:
        return _estimate_text_tokens(text)


class ContextWindowController:
    """Apply budget-aware hard gates before provider calls."""

    def __init__(
        self,
        *,
        estimator: ContextBudgetEstimator | None = None,
        hard_limit_ratio: float = 0.92,
        warning_ratio: float = 0.75,
    ) -> None:
        self._estimator = estimator or ContextBudgetEstimator()
        self._hard_limit_ratio = min(max(hard_limit_ratio, 0.50), 0.99)
        self._warning_ratio = min(max(warning_ratio, 0.10), self._hard_limit_ratio)

    @property
    def estimator(self) -> ContextBudgetEstimator:
        """Return the estimator used by this controller."""
        return self._estimator

    def prepare(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Any = None,
        context_length: int,
        compressor: MessageCompressor | None = None,
    ) -> ContextBudgetDecision:
        """Return messages that fit the active budget as safely as possible."""
        snapshot = self._estimator.snapshot(messages, tools=tools, context_length=context_length)
        if snapshot.ratio < self._hard_limit_ratio:
            return ContextBudgetDecision(messages=messages, snapshot=snapshot)

        compressed = False
        prepared = messages
        if compressor is not None:
            prepared = compressor.force_compress(prepared)
            compressed = True
            snapshot = self._estimator.snapshot(prepared, tools=tools, context_length=context_length)

        forced_final = False
        notice = ""
        if snapshot.ratio >= self._hard_limit_ratio:
            prepared = self._tail_preserving_drop(prepared)
            compressed = True
            forced_final = True
            notice = (
                "SYSTEM: Context budget is critically high after compression. "
                "Use the remaining evidence and provide the final answer now; "
                "do not call more exploratory tools unless absolutely required."
            )
            prepared.append({"role": "user", "content": notice})
            snapshot = self._estimator.snapshot(prepared, tools=tools, context_length=context_length)

        return ContextBudgetDecision(
            messages=prepared,
            snapshot=snapshot,
            compressed=compressed,
            forced_final_answer=forced_final,
            notice=notice,
        )

    def warning_notice(self, snapshot: ContextBudgetSnapshot, *, round_number: int) -> str:
        """Return a concise system notice when the payload is growing too large."""
        if snapshot.ratio < self._warning_ratio:
            return ""
        return (
            "SYSTEM: Context utilization is high "
            f"({snapshot.total_tokens:,}/{snapshot.context_length:,} estimated tokens, "
            f"round {round_number}). Prefer summaries, targeted reads, and final synthesis."
        )

    @staticmethod
    def _tail_preserving_drop(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(messages) <= 6:
            return messages
        head = messages[:1] if messages and messages[0].get("role") == "system" else []
        tail = messages[-5:]
        dropped = max(0, len(messages) - len(head) - len(tail))
        notice = {
            "role": "system",
            "content": (
                f"[Context hard gate: {dropped} older messages compacted out. "
                "Recent evidence and the current user request are authoritative.]"
            ),
        }
        return head + [notice] + tail


# Evidence scaling. The ceiling was low enough that a 1M window hit it almost
# immediately, so evidence stopped growing with the budget long before the
# window was under pressure.
_EVIDENCE_CEILING_CHARS = 24_000
_EVIDENCE_CONTEXT_DIVISOR = 32


class ToolEvidenceBuilder:
    """Convert verbose tool outputs into compact evidence for LLM replay.

    When ``context_length`` is provided, ``max_content_chars`` is raised
    proportionally so that larger context windows retain richer tool evidence.
    """

    def __init__(
        self,
        *,
        max_content_chars: int = 1200,
        max_items: int = 40,
        context_length: int = 0,
    ) -> None:
        if context_length > 0:
            adaptive = max(
                max_content_chars,
                min(_EVIDENCE_CEILING_CHARS, context_length // _EVIDENCE_CONTEXT_DIVISOR),
            )
            self._max_content_chars = max(200, adaptive)
        else:
            self._max_content_chars = max(200, max_content_chars)
        self._max_items = max(5, max_items)

    def build(self, tool_name: str, arguments: Dict[str, Any] | None, result: Any) -> Any:
        """Return a compact, JSON-serializable result preserving task evidence."""
        if not isinstance(result, dict):
            return self._compact_value(result)
        if tool_name == "platform_connect":
            return self._app_connector_evidence(result)
        if tool_name in {"platform_action", "gp_platform_action"}:
            return self._platform_action_evidence(arguments or {}, result)
        if result.get("ok") is False:
            return self._compact_error(result)
        if tool_name in {"file_read", "gp_file_read"}:
            return self._file_read_evidence(arguments or {}, result)
        if tool_name in {"file_list", "gp_file_list"}:
            return self._file_list_evidence(result)
        if tool_name in {"shell_run", "gp_shell_run"}:
            return self._shell_evidence(result)
        if tool_name in {"web_fetch", "gp_web_fetch"}:
            return self._web_fetch_evidence(result)
        return self._compact_mapping(result)

    def _file_read_evidence(self, arguments: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        content = str(result.get("content", ""))
        excerpt = self._head_tail(content, self._max_content_chars)
        evidence = {
            "ok": True,
            "kind": "file_read_evidence",
            "path": result.get("path") or arguments.get("path", ""),
            "lines": result.get("lines", 0),
            "truncated": bool(result.get("truncated", False)),
            "mode": result.get("mode") or arguments.get("mode") or "raw",
            "excerpt": excerpt,
        }
        for key in ("start_line", "end_line", "selected_lines", "outline"):
            if key in result:
                evidence[key] = result[key]
        return evidence

    def _file_list_evidence(self, result: Dict[str, Any]) -> Dict[str, Any]:
        entries = result.get("entries", [])
        tree = result.get("tree")

        if tree is not None:
            # Depth > 0: flatten the nested tree to compact indented path strings
            # so the LLM receives a readable structure without verbose dicts.
            flat: List[str] = []
            self._flatten_tree_nodes(tree, prefix="", out=flat, limit=self._max_items)
            return {
                "ok": True,
                "kind": "file_list_evidence",
                "path": result.get("path", ""),
                "depth": result.get("depth", 1),
                "tree": flat,
                "total_entries": result.get("total_entries", len(flat)),
                "truncated": bool(result.get("truncated", False)),
            }

        # Flat listing (depth=0): compact entry strings instead of verbose
        # {name, type, size} dicts — ~3× more entries per token budget.
        all_compact = [
            self._compact_entry(e) for e in entries
        ] if isinstance(entries, list) else []
        visible = all_compact[: self._max_items]
        return {
            "ok": True,
            "kind": "file_list_evidence",
            "path": result.get("path", ""),
            "entries": visible,
            "entry_count": result.get("entry_count", len(all_compact)),
            "truncated": bool(
                result.get("truncated", False) or
                (isinstance(entries, list) and len(entries) > len(visible))
            ),
        }

    @staticmethod
    def _compact_entry(entry: Any) -> str:
        """Render a {name, type, size} dict as a short token-efficient string."""
        if not isinstance(entry, dict):
            return str(entry)
        name = str(entry.get("name", ""))
        if entry.get("type") == "dir":
            return f"{name}/"
        size = entry.get("size")
        if size is not None:
            if size >= 1_000_000:
                return f"{name} ({size / 1_000_000:.1f}MB)"
            if size >= 1_000:
                return f"{name} ({size / 1_000:.0f}KB)"
            return f"{name} ({size}B)"
        return name

    def _flatten_tree_nodes(
        self, nodes: Any, *, prefix: str, out: List[str], limit: int
    ) -> None:
        """Recursively flatten a tree structure to indented compact strings."""
        if not isinstance(nodes, list):
            return
        for node in nodes:
            if len(out) >= limit:
                break
            if not isinstance(node, dict):
                continue
            name = str(node.get("name", ""))
            if node.get("type") == "dir":
                summary = node.get("summary", "")
                label = f"{prefix}{name}/" + (f" [{summary}]" if summary else "")
                out.append(label)
                children = node.get("children")
                if isinstance(children, list):
                    self._flatten_tree_nodes(
                        children, prefix=prefix + "  ", out=out, limit=limit
                    )
            else:
                out.append(prefix + self._compact_entry(node))

    def _shell_evidence(self, result: Dict[str, Any]) -> Dict[str, Any]:
        from leapflow.engine.tools.tool_execution import exit_code_from

        return {
            "ok": bool(result.get("ok", True)),
            "kind": "shell_evidence",
            # Read under either key: shell tools mirror subprocess' `returncode`,
            # so looking only for `exit_code` reported null on every success.
            "exit_code": exit_code_from(result),
            "stdout": self._head_tail(str(result.get("stdout", "")), self._max_content_chars),
            "stderr": self._head_tail(str(result.get("stderr", "")), self._max_content_chars // 2),
        }

    def _web_fetch_evidence(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compact a fetched page or payload down to task evidence.

        A single page routinely exceeds the whole turn's context budget, so the
        raw body must never pass through. Status, final URL and content type stay
        intact because they are what the model reasons about next; the body is
        head/tail trimmed, and links are capped rather than dropped so follow-up
        navigation is still possible.
        """
        evidence: Dict[str, Any] = {
            "ok": bool(result.get("ok", True)),
            "kind": "web_fetch_evidence",
            "status": result.get("status"),
            "final_url": result.get("final_url") or result.get("url", ""),
            "content_type": result.get("content_type", ""),
            "truncated": bool(result.get("truncated", False)),
        }
        for key in ("title", "select", "extractor", "note", "from_cache", "cache_path"):
            value = result.get(key)
            if value:
                evidence[key] = self._compact_value(value)
        if "data" in result:
            evidence["data"] = self._compact_value(result["data"])
        text = str(result.get("text") or "")
        if text:
            evidence["text"] = self._head_tail(text, self._max_content_chars)
        links = result.get("links")
        if isinstance(links, list) and links:
            evidence["links"] = [
                {"text": str(item.get("text", ""))[:120], "url": str(item.get("url", ""))}
                for item in links[: self._max_items]
                if isinstance(item, dict)
            ]
        return evidence

    def _platform_action_evidence(self, arguments: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Compact evidence for platform_action with strong completion markers."""
        if result.get("ok") is False:
            evidence = self._compact_error(result)
            for key in (
                "failure_class", "failure_code", "recoverability", "blocks_approval",
                "platform", "action", "capability", "missing_fields", "missing_scopes",
                "required_scopes", "scope_relation", "scope_source", "console_url",
                "next_steps", "recovery_hint", "expected_schema", "retryable", "skip_approval",
            ):
                if key in result:
                    evidence[key] = self._compact_value(result[key])
            for key in ("platform", "action"):
                if key not in evidence and arguments.get(key):
                    evidence[key] = self._compact_value(arguments[key])
            if result.get("llm_instruction"):
                evidence["llm_instruction"] = str(result["llm_instruction"])
            if result.get("platform_degraded"):
                evidence["platform_degraded"] = True
                evidence["degradation_reason"] = str(result.get("degradation_reason", ""))
            return evidence

        evidence: Dict[str, Any] = {
            "ok": True,
            "kind": "platform_action_evidence",
            "action": result.get("action") or arguments.get("action", ""),
            "platform": result.get("platform") or arguments.get("platform", ""),
        }
        if result.get("resource_id"):
            evidence["resource_id"] = str(result["resource_id"])
        if result.get("completed"):
            evidence["status"] = "COMPLETED"
            evidence["execution_note"] = str(result.get("execution_note") or "Done. Do not repeat.")
        if result.get("already_executed"):
            evidence["status"] = "ALREADY_EXECUTED"
            original = result.get("original_result")
            if isinstance(original, dict) and original.get("resource_id"):
                evidence["resource_id"] = str(original["resource_id"])
        if isinstance(result.get("data"), dict):
            evidence["data"] = self._compact_mapping(result["data"])
        return evidence

    def _compact_error(self, result: Dict[str, Any]) -> Dict[str, Any]:
        compact: Dict[str, Any] = {
            "ok": False,
            "error": self._head_tail(str(result.get("error", "unknown error")), self._max_content_chars),
        }
        # Preserve structured repair/recovery hints so the model can self-correct
        # in-turn: invalid_arguments -> missing/accepted_parameters; edit_file
        # anchor errors -> match_count; code_search -> error_type; etc.
        for key in ("error_type", "retryable", "missing", "accepted_parameters", "required", "match_count", "failure_code"):
            if key in result:
                compact[key] = self._compact_value(result[key])
        # Diagnostic output matters MOST on failure: preserve stdout/stderr
        # (head+tail so the tail traceback survives) and the exit code, so the
        # agent can see the actual error (e.g. a Python traceback from a failed
        # shell_run) and fix its cause instead of a bare "unknown error".
        for key in ("stdout", "stderr"):
            value = result.get(key)
            if value:
                compact[key] = self._head_tail(str(value), self._max_content_chars)
        for key in ("returncode", "exit_code"):
            if key in result:
                compact[key] = result[key]
        # Network failures explain themselves in the response: the status code is
        # the classification and the body says why (an API error payload, a rate
        # limit notice), so both must survive compaction for the model to act on.
        if "status" in result:
            compact["status"] = result["status"]
        body_excerpt = result.get("body_excerpt")
        if body_excerpt:
            compact["body_excerpt"] = self._head_tail(str(body_excerpt), self._max_content_chars // 2)
        return compact

    def _app_connector_evidence(self, result: Dict[str, Any]) -> Dict[str, Any]:
        ok = bool(result.get("ok", True))
        evidence: Dict[str, Any] = {
            "ok": ok,
            "kind": "app_connector_evidence",
        }
        if ok:
            # Success path: expose user-facing facts and onboarding state
            # (needed for empty-response recovery), but strip process noise
            # like setup_steps, setup_form, setup_guide, optional_settings.
            for key in ("platform", "status", "connected"):
                if key in result:
                    evidence[key] = self._compact_value(result[key])
            # Onboarding state is structural (used by recovery fallback),
            # not conversational noise — preserve it.
            for key in ("onboarding_state", "recovery_hint", "next_steps"):
                if key in result:
                    evidence[key] = self._compact_value(result[key])
            if isinstance(result.get("preflight_result"), dict):
                evidence["preflight_result"] = self._app_preflight_evidence(result["preflight_result"])
        else:
            # Failure path: retain recovery information for LLM retry logic.
            for key in (
                "platform",
                "stage",
                "onboarding_state",
                "recovery_hint",
                "next_steps",
                "error",
                "status",
                "connected",
            ):
                if key in result:
                    evidence[key] = self._compact_value(result[key])
            if isinstance(result.get("preflight_result"), dict):
                evidence["preflight_result"] = self._app_preflight_evidence(result["preflight_result"])
            if "required_fields" in result:
                evidence["required_fields"] = self._compact_value(result["required_fields"])
        return evidence

    def _app_check_evidence(self, checks: List[Any]) -> List[Dict[str, Any]]:
        return [
            {
                compact_key: check.get(compact_key)
                for compact_key in ("key", "kind", "status", "failure_code", "requires_approval", "auto_run")
                if isinstance(check, dict) and compact_key in check
            }
            for check in checks[: self._max_items]
        ]

    def _app_preflight_evidence(self, preflight: Dict[str, Any]) -> Dict[str, Any]:
        evidence: Dict[str, Any] = {}
        for key in ("ready", "stage", "backend_kind", "recoverable", "detail", "recovery_hint", "next_steps"):
            if key in preflight:
                evidence[key] = self._compact_value(preflight[key])
        checks = preflight.get("checks")
        if isinstance(checks, list):
            evidence["checks"] = self._app_check_evidence(checks)
        metadata = preflight.get("metadata")
        if isinstance(metadata, dict):
            evidence["metadata"] = {
                key: self._compact_value(metadata[key])
                for key in ("binary", "binary_path", "profile", "identity", "auth_status")
                if key in metadata
            }
        return evidence

    def _compact_mapping(self, result: Dict[str, Any]) -> Dict[str, Any]:
        compact: Dict[str, Any] = {}
        for key, value in result.items():
            compact[key] = self._compact_value(value)
        return compact

    def _compact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._head_tail(value, self._max_content_chars)
        if isinstance(value, list):
            return [self._compact_value(item) for item in value[: self._max_items]]
        if isinstance(value, dict):
            return {str(k): self._compact_value(v) for k, v in value.items()}
        return value

    @staticmethod
    def _head_tail(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        head = max(80, int(limit * _DEFAULT_HEAD_RATIO))
        tail = max(40, int(limit * _DEFAULT_TAIL_RATIO))
        omitted = len(text) - head - tail
        return f"{text[:head]}\n\n[... {omitted:,} chars omitted ...]\n\n{text[-tail:]}"


@dataclass(frozen=True)
class ContextPostureConfig:
    """Configurable thresholds for adaptive context-governance posture."""

    expanded_ratio: float = 0.60
    finalizing_ratio: float = 0.90
    expanded_evidence_threshold: int = 2
    expanded_tool_call_threshold: int = 3
    research_source_threshold: int = 3
    research_evidence_threshold: int = 5
    # Bidirectional posture: high-end expansion arm + low-end answer-ready arm.
    expand_difficulty_threshold: float = 0.55
    expand_context_ceiling: float = 0.70
    answer_ready_min_round: int = 2


@dataclass(frozen=True)
class DifficultyConfig:
    """Weights and saturation denominators for the continuous difficulty signal.

    Difficulty is a bounded [0, 1] estimate of how hard / long-horizon the active
    task is, derived only from structural exploration-ledger signals (breadth,
    evidence volume, friction, persistence, marginal growth, tool activity). It
    drives the elastic iteration budget and the expansion arm of the posture
    ladder. All denominators and weights are configurable; nothing here reads the
    user's free-form text.
    """

    d_sources: int = 6
    d_evidence: int = 10
    d_rounds: int = 12
    d_marginal: float = 1.0
    d_tool_calls: int = 8
    marginal_window: int = 3
    ema_alpha: float = 0.4
    w_breadth: float = 0.15
    w_volume: float = 0.15
    w_friction: float = 0.15
    w_persistence: float = 0.15
    w_marginal: float = 0.20
    w_activity: float = 0.20


@dataclass(frozen=True)
class ExplorationSnapshot:
    """Compact user-visible state for the adaptive exploration ledger."""

    posture: str = _POSTURE_BASELINE
    sources_seen: int = 0
    evidence_count: int = 0
    repeated_reads: int = 0
    tool_calls: int = 0
    dominant_signal: str = ""
    should_converge: bool = False
    convergence_reason: str = ""
    guidance: str = ""
    difficulty: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot for daemon/TUI metadata."""
        return {
            "posture": self.posture,
            "sources_seen": self.sources_seen,
            "evidence_count": self.evidence_count,
            "repeated_reads": self.repeated_reads,
            "tool_calls": self.tool_calls,
            "dominant_signal": self.dominant_signal,
            "should_converge": self.should_converge,
            "convergence_reason": self.convergence_reason,
            "guidance": self.guidance,
            "difficulty": self.difficulty,
        }


@dataclass
class ContextGovernanceController:
    """Adaptive exploration ledger for all interactions, not a separate mode."""

    evidence_builder: ToolEvidenceBuilder
    repeated_read_limit: int = 2
    convergence_round: int = 12
    # Difficulty-adaptive convergence: the effective round at which long-exploration
    # converging kicks in is scaled by observed difficulty.  A hard task earns more
    # exploration rounds; the ceiling prevents truly stuck tasks from running forever.
    convergence_round_ceiling: int = 40
    convergence_scale: float = 2.0
    checkpoint_interval: int = 15
    posture_config: ContextPostureConfig = field(default_factory=ContextPostureConfig)
    difficulty_config: DifficultyConfig = field(default_factory=DifficultyConfig)
    evidence_tools: frozenset[str] = _EVIDENCE_TOOLS
    research_source_threshold: int | None = None
    research_evidence_threshold: int | None = None

    def __post_init__(self) -> None:
        if self.research_source_threshold is not None or self.research_evidence_threshold is not None:
            self.posture_config = ContextPostureConfig(
                expanded_ratio=self.posture_config.expanded_ratio,
                finalizing_ratio=self.posture_config.finalizing_ratio,
                expanded_evidence_threshold=self.posture_config.expanded_evidence_threshold,
                expanded_tool_call_threshold=self.posture_config.expanded_tool_call_threshold,
                research_source_threshold=self.research_source_threshold or self.posture_config.research_source_threshold,
                research_evidence_threshold=self.research_evidence_threshold or self.posture_config.research_evidence_threshold,
                expand_difficulty_threshold=self.posture_config.expand_difficulty_threshold,
                expand_context_ceiling=self.posture_config.expand_context_ceiling,
                answer_ready_min_round=self.posture_config.answer_ready_min_round,
            )
        self._reads: dict[str, int] = {}
        self._sources_seen: set[str] = set()
        self._tool_counts: dict[str, int] = {}
        self._evidence_count = 0
        self._tool_failures = 0
        self._difficulty_prev = 0.0
        self._difficulty_ema_round = -1
        self._evidence_by_round: dict[int, int] = {}
        self._evidence_round_hwm = -1
        self._last_tool_name: str = ""
        self._last_tool_path: str = ""

    def reset_turn_scope(self) -> None:
        """Clear per-turn exploration state so posture never leaks across tasks."""
        self._reads.clear()
        self._sources_seen.clear()
        self._tool_counts.clear()
        self._evidence_count = 0
        self._tool_failures = 0
        self._difficulty_prev = 0.0
        self._difficulty_ema_round = -1
        self._evidence_by_round.clear()
        self._evidence_round_hwm = -1
        self._last_tool_name = ""
        self._last_tool_path = ""

    reset_task_scope = reset_turn_scope

    def _effective_convergence_round(self, difficulty: float) -> int:
        """Scale the convergence round with task difficulty, bounded by a ceiling.

        At difficulty=0 returns the base ``convergence_round`` unchanged.
        At difficulty=1 the effective round rises by ``convergence_scale`` ×
        the base, capped at ``convergence_round_ceiling``.  This gives hard
        tasks proportionally more exploration while still converging them
        eventually.
        """
        extension = round(self.convergence_round * max(0.0, min(1.0, difficulty)) * self.convergence_scale)
        return min(self.convergence_round_ceiling, self.convergence_round + extension)

    _WRITE_TOOLS = frozenset({
        "file_write", "gp_file_write",
        "text_replace", "gp_text_replace",
        "file_edit", "gp_file_edit",
    })

    def compact_tool_result(self, tool_name: str, arguments: Dict[str, Any] | None, result: Any) -> Any:
        """Return evidence and update the session exploration ledger.

        P0 convergence fixes:
        - evidence_count only increments for genuinely new sources (not re-reads)
        - Repeated reads beyond the limit are hard-gated: tool result is replaced
          with an instruction to use research_note, unless the immediately preceding
          tool call was a write/edit to the same path (post-edit verification is OK).
        """
        self._tool_counts[tool_name] = self._tool_counts.get(tool_name, 0) + 1
        if isinstance(result, dict) and result.get("ok") is False:
            self._tool_failures += 1

        gated_result: Any = None

        if tool_name in {"file_read", "gp_file_read"}:
            path = str((arguments or {}).get("path") or (result.get("path") if isinstance(result, dict) else ""))
            if path:
                key = str(Path(path).expanduser())
                # Reset read count if the previous tool was a write/edit to this path
                if self._last_tool_name in self._WRITE_TOOLS and self._last_tool_path == key:
                    self._reads[key] = 0
                is_new_source = key not in self._sources_seen
                self._reads[key] = self._reads.get(key, 0) + 1
                self._sources_seen.add(key)
                # Only count genuinely new sources as progress
                if is_new_source:
                    self._evidence_count += 1
                # Hard gate: block re-reads beyond the limit
                if self._reads[key] > self.repeated_read_limit:
                    gated_result = (
                        f"[REPEATED READ \u2014 this file was already read {self._reads[key]} times in this turn. "
                        f"Content was processed in earlier rounds but compressed. "
                        f"Use research_note to record your findings instead of re-reading. "
                        f"File: {path}]"
                    )
        elif tool_name in {"file_list", "gp_file_list"}:
            path = str((arguments or {}).get("path") or (result.get("path") if isinstance(result, dict) else ""))
            if path:
                key = str(Path(path).expanduser())
                is_new_source = key not in self._sources_seen
                self._reads[key] = self._reads.get(key, 0) + 1
                self._sources_seen.add(key)
                if is_new_source:
                    self._evidence_count += 1
                if self._reads[key] > self.repeated_read_limit:
                    gated_result = (
                        f"[REPEATED READ \u2014 this directory was already listed {self._reads[key]} times in this turn. "
                        f"Content was processed in earlier rounds but compressed. "
                        f"Use research_note to record your findings instead of re-reading. "
                        f"Path: {path}]"
                    )
        elif tool_name in self.evidence_tools:
            # shell_run and other non-path evidence tools: always new evidence
            self._evidence_count += 1

        # Track last tool for write-then-read exception
        self._last_tool_name = tool_name
        if tool_name in self._WRITE_TOOLS:
            write_path = str((arguments or {}).get("path") or "")
            self._last_tool_path = str(Path(write_path).expanduser()) if write_path else ""
        elif tool_name in {"file_read", "gp_file_read", "file_list", "gp_file_list"}:
            read_path = str((arguments or {}).get("path") or (result.get("path") if isinstance(result, dict) else ""))
            self._last_tool_path = str(Path(read_path).expanduser()) if read_path else ""
        else:
            self._last_tool_path = ""

        if gated_result is not None:
            return gated_result
        return self.evidence_builder.build(tool_name, arguments, result)

    def tool_metadata(self, tool_name: str, arguments: Dict[str, Any] | None, result: Any) -> Dict[str, Any]:
        """Build UX metadata about adaptive context handling."""
        metadata: Dict[str, Any] = {}
        if tool_name in self.evidence_tools:
            metadata["context_evidence"] = True
        if tool_name in {"file_read", "gp_file_read"}:
            path = str((arguments or {}).get("path") or "")
            if path:
                count = self._reads.get(str(Path(path).expanduser()), 0)
                metadata["read_count"] = count
                metadata["repeat_read"] = count > self.repeated_read_limit
        if tool_name in {"file_list", "gp_file_list"}:
            # Expose the same repeat_read signal for directories so the UI and
            # the agent loop can surface the same converging guidance as for
            # repeated file reads.
            path = str((arguments or {}).get("path") or (result.get("path") if isinstance(result, dict) else ""))
            if path:
                count = self._reads.get(str(Path(path).expanduser()), 0)
                metadata["read_count"] = count
                metadata["repeat_read"] = count > self.repeated_read_limit
        if isinstance(result, dict):
            if result.get("truncated"):
                metadata["tool_truncated"] = True
            if result.get("mode"):
                metadata["mode"] = result.get("mode")
        ledger = self.snapshot()
        if metadata and ledger.posture != _POSTURE_BASELINE:
            metadata["context_posture"] = ledger.posture
            metadata["context_signal"] = ledger.dominant_signal
            if ledger.guidance:
                metadata["context_guidance"] = ledger.guidance
        return metadata

    def _marginal_evidence(self, round_number: int) -> float:
        """Recent per-round growth in evidence volume (idempotent within a round).

        Records the current evidence count for ``round_number`` (only when the
        round advances, so out-of-order / stale peeks -- e.g. a round-0
        ``tool_metadata`` snapshot interleaved with authoritative round-N calls
        -- never overwrite an earlier round's baseline), then measures growth
        against the level ~one window of rounds earlier. A positive value means
        the task is still surfacing new evidence ("there is more to dig").
        """
        if round_number > self._evidence_round_hwm:
            self._evidence_by_round[round_number] = self._evidence_count
            self._evidence_round_hwm = round_number
        if not self._evidence_by_round:
            return 0.0
        window = max(1, self.difficulty_config.marginal_window)
        past_round = round_number - window
        earlier = [r for r in self._evidence_by_round if r <= past_round]
        baseline_round = max(earlier) if earlier else min(self._evidence_by_round)
        delta = self._evidence_count - self._evidence_by_round[baseline_round]
        return max(0.0, delta / window)

    def _compute_difficulty(
        self,
        *,
        round_number: int,
        sources_seen: int,
        tool_calls: int,
        repeated_reads: int,
        marginal: float,
    ) -> float:
        """Continuous [0, 1] task-difficulty estimate from structural signals only.

        EMA smoothing is applied at most once per round (idempotent to repeated
        snapshot() calls in the same round) so difficulty rises/falls smoothly.
        """
        cfg = self.difficulty_config
        breadth_n = min(sources_seen / cfg.d_sources, 1.0) if cfg.d_sources > 0 else 0.0
        volume_n = min(self._evidence_count / cfg.d_evidence, 1.0) if cfg.d_evidence > 0 else 0.0
        repeat_ratio = min(repeated_reads / max(1, self.repeated_read_limit), 1.0)
        fail_rate = self._tool_failures / max(1, tool_calls)
        friction_n = min(0.5 * repeat_ratio + 0.5 * fail_rate, 1.0)
        persistence_n = min(round_number / cfg.d_rounds, 1.0) if cfg.d_rounds > 0 else 0.0
        marginal_n = min(marginal / cfg.d_marginal, 1.0) if cfg.d_marginal > 0 else 0.0
        activity_n = min(tool_calls / cfg.d_tool_calls, 1.0) if cfg.d_tool_calls > 0 else 0.0
        raw = (
            cfg.w_breadth * breadth_n
            + cfg.w_volume * volume_n
            + cfg.w_friction * friction_n
            + cfg.w_persistence * persistence_n
            + cfg.w_marginal * marginal_n
            + cfg.w_activity * activity_n
        )
        raw = max(0.0, min(1.0, raw))
        if round_number > self._difficulty_ema_round:
            alpha = cfg.ema_alpha
            self._difficulty_prev = alpha * raw + (1.0 - alpha) * self._difficulty_prev
            self._difficulty_ema_round = round_number
        return self._difficulty_prev

    def snapshot(self, *, context_ratio: float = 0.0, round_number: int = 0, open_questions: int | None = None) -> ExplorationSnapshot:
        """Return the current adaptive-governance posture without exposing a mode.

        The posture ladder is bidirectional and difficulty-aware. Priority order,
        safety-first: (1) finalizing on context pressure, (2) converging on
        repeat-read loops, (3) EXPANDING when difficulty is high and context is
        healthy (a hard task earns a wider horizon), (4) converging on long
        low-difficulty exploration, then (5) research / expanded / baseline. A
        low-end "answer-ready" arm nudges early finalization for simple tasks
        without inflating disclosure.
        """
        cfg = self.posture_config
        repeated_reads = sum(1 for count in self._reads.values() if count > self.repeated_read_limit)
        tool_calls = sum(self._tool_counts.values())
        sources_seen = len(self._sources_seen)
        marginal = self._marginal_evidence(round_number)
        difficulty = self._compute_difficulty(
            round_number=round_number,
            sources_seen=sources_seen,
            tool_calls=tool_calls,
            repeated_reads=repeated_reads,
            marginal=marginal,
        )
        dominant_signal = ""
        posture = _POSTURE_BASELINE
        guidance = ""
        convergence_reason = ""

        expand_ok = (
            difficulty >= cfg.expand_difficulty_threshold
            and context_ratio < cfg.expand_context_ceiling
            and marginal > 0.0
        )

        if context_ratio >= cfg.finalizing_ratio:
            posture = _POSTURE_FINALIZING
            dominant_signal = "context-critical"
            convergence_reason = "context budget is critical"
            guidance = "finalize with existing evidence"
        elif repeated_reads > 0:
            posture = _POSTURE_CONVERGING
            dominant_signal = "repeat-read"
            convergence_reason = "repeat reads detected"
            guidance = "switch to complementary sources, outlines, symbols, or bounded ranges"
        elif expand_ok:
            posture = _POSTURE_EXPANDING
            dominant_signal = "high-difficulty"
            guidance = "broaden investigation, decompose, or delegate; iteration budget widened"
        elif round_number >= self._effective_convergence_round(difficulty):
            effective_round = self._effective_convergence_round(difficulty)
            posture = _POSTURE_CONVERGING
            dominant_signal = "long-exploration"
            convergence_reason = (
                f"exploration round limit reached (base {self.convergence_round}, "
                f"effective {effective_round} at difficulty {difficulty:.2f})"
            )
            guidance = "deduplicate evidence and prefer targeted reads"
        elif sources_seen >= cfg.research_source_threshold or self._evidence_count >= cfg.research_evidence_threshold:
            posture = _POSTURE_RESEARCH
            dominant_signal = "multi-source" if sources_seen >= cfg.research_source_threshold else "evidence-volume"
            guidance = "maintain research ledger and synthesize findings"
        elif context_ratio >= cfg.expanded_ratio or self._evidence_count >= cfg.expanded_evidence_threshold or tool_calls >= cfg.expanded_tool_call_threshold:
            posture = _POSTURE_EXPANDED
            dominant_signal = "context-growing" if context_ratio >= cfg.expanded_ratio else "tool-activity"
            guidance = "prefer outline, symbols, or range reads before raw content"

        should_converge = posture in {_POSTURE_CONVERGING, _POSTURE_FINALIZING}
        # Low-end symmetric arm: a low-difficulty turn that already gathered
        # evidence and is no longer surfacing new evidence should finalize early.
        # It keeps posture at baseline/expanded so disclosure stays minimal (a
        # simple task must never be pushed into full disclosure). When a research
        # ledger is active with unresolved open questions, this early convergence
        # is suppressed -- tracked open work means the task is not done, so a long
        # task is never cut short (open_questions: None = no ledger signal).
        if (
            not should_converge
            and posture in {_POSTURE_BASELINE, _POSTURE_EXPANDED}
            and self._evidence_count >= 1
            and marginal <= 0.0
            and difficulty < cfg.expand_difficulty_threshold
            and round_number >= cfg.answer_ready_min_round
            and (open_questions is None or open_questions == 0)
        ):
            should_converge = True
            convergence_reason = "answer-ready"
            guidance = "you likely have enough evidence; if the question is answered, provide the final answer now"

        return ExplorationSnapshot(
            posture=posture,
            sources_seen=sources_seen,
            evidence_count=self._evidence_count,
            repeated_reads=repeated_reads,
            tool_calls=tool_calls,
            dominant_signal=dominant_signal,
            should_converge=should_converge,
            convergence_reason=convergence_reason,
            guidance=guidance,
            difficulty=round(difficulty, 4),
        )

    def convergence_notice(self, round_number: int, *, open_questions: int | None = None) -> str:
        """Return a notice that nudges synthesis after excessive exploration."""
        snapshot = self.snapshot(round_number=round_number, open_questions=open_questions)
        if not snapshot.should_converge:
            return ""
        if snapshot.dominant_signal == "repeat-read":
            return (
                "SYSTEM: Adaptive context governance detected repeated reads. "
                "Do not reread the same raw source again. Pivot to complementary project evidence: "
                "directory outline, symbols, bounded line ranges, adjacent modules, tests, docs, or synthesize "
                "from the evidence already gathered if enough context exists."
            )
        if snapshot.convergence_reason == "answer-ready":
            return (
                "SYSTEM: You likely have enough evidence to answer. If the user's request is "
                "already addressed, provide the final answer now instead of gathering more."
            )
        reason = snapshot.convergence_reason or snapshot.dominant_signal or "context pressure"
        return (
            "SYSTEM: Adaptive context governance is converging "
            f"({reason}). Stop broad reading, deduplicate evidence already gathered, "
            "prefer targeted reads, and synthesize the final answer."
        )

    def checkpoint_notice(self, round_number: int) -> str | None:
        """Emit checkpoint instruction at configurable intervals during research."""
        interval = self.checkpoint_interval
        if interval <= 0 or round_number < interval:
            return None
        if round_number % interval != 0:
            return None
        # Only checkpoint in research/expanding posture, not simple Q&A
        snapshot = self.snapshot(round_number=round_number)
        if snapshot.posture in (_POSTURE_BASELINE, _POSTURE_EXPANDED):
            return None
        return (
            f"SYSTEM CHECKPOINT (round {round_number}): "
            "Persist your current findings NOW. "
            "(1) Call research_note to record key conclusions that should survive context compression. "
            "(2) Write intermediate results to a file if the task requires a deliverable. "
            "(3) If you can answer partially, provide it now. "
            "Remaining work should be declared as open_questions via research_note."
        )


LongTaskContextController = ContextGovernanceController
