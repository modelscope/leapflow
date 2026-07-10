"""Context-window control primitives for long-running agent tasks.

The module keeps context accounting, overflow prevention, and tool-result
compaction independent from AgentEngine so the execution loop can remain small
and policy can evolve without touching tool dispatch code.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence, runtime_checkable

logger = logging.getLogger(__name__)

_IMAGE_TOKEN_ESTIMATE = 1600
_MESSAGE_OVERHEAD_TOKENS = 8
_TOOL_SCHEMA_OVERHEAD_TOKENS = 12
_DEFAULT_HEAD_RATIO = 0.55
_DEFAULT_TAIL_RATIO = 0.25


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


class ContextBudgetEstimator:
    """Estimate provider-visible prompt tokens including tool schemas."""

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
        return max(1, total)

    def estimate_tools(self, tools: Any) -> int:
        """Estimate token usage for native function/tool schemas."""
        if not tools:
            return 0
        try:
            text = json.dumps(tools, ensure_ascii=False, default=str, sort_keys=True)
        except (TypeError, ValueError):
            text = str(tools)
        return _TOOL_SCHEMA_OVERHEAD_TOKENS + self._estimate_text(text)

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
        if not text:
            return 0
        cjk_count = sum(
            1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f"
        )
        latin_chars = len(text) - cjk_count
        return max(1, cjk_count + latin_chars // 4)


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


class ToolEvidenceBuilder:
    """Convert verbose tool outputs into compact evidence for LLM replay."""

    def __init__(self, *, max_content_chars: int = 1200, max_items: int = 40) -> None:
        self._max_content_chars = max(200, max_content_chars)
        self._max_items = max(5, max_items)

    def build(self, tool_name: str, arguments: Dict[str, Any] | None, result: Any) -> Any:
        """Return a compact, JSON-serializable result preserving task evidence."""
        if not isinstance(result, dict):
            return self._compact_value(result)
        if result.get("ok") is False:
            return self._compact_error(result)
        if tool_name in {"file_read", "gp_file_read"}:
            return self._file_read_evidence(arguments or {}, result)
        if tool_name in {"file_list", "gp_file_list"}:
            return self._file_list_evidence(result)
        if tool_name in {"shell_run", "gp_shell_run"}:
            return self._shell_evidence(result)
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
        compact_entries = entries[: self._max_items] if isinstance(entries, list) else []
        return {
            "ok": True,
            "kind": "file_list_evidence",
            "path": result.get("path", ""),
            "entries": compact_entries,
            "entry_count": result.get("entry_count", len(entries) if isinstance(entries, list) else 0),
            "truncated": bool(result.get("truncated", False) or (isinstance(entries, list) and len(entries) > len(compact_entries))),
        }

    def _shell_evidence(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ok": bool(result.get("ok", True)),
            "kind": "shell_evidence",
            "exit_code": result.get("exit_code"),
            "stdout": self._head_tail(str(result.get("stdout", "")), self._max_content_chars),
            "stderr": self._head_tail(str(result.get("stderr", "")), self._max_content_chars // 2),
        }

    def _compact_error(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ok": False,
            "error": self._head_tail(str(result.get("error", "unknown error")), self._max_content_chars),
        }

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


@dataclass
class LongTaskContextController:
    """Per-turn controller for exploration-heavy tasks."""

    evidence_builder: ToolEvidenceBuilder
    repeated_read_limit: int = 2
    convergence_round: int = 12

    def __post_init__(self) -> None:
        self._reads: dict[str, int] = {}

    def compact_tool_result(self, tool_name: str, arguments: Dict[str, Any] | None, result: Any) -> Any:
        """Return evidence and update exploration ledger."""
        if tool_name in {"file_read", "gp_file_read"}:
            path = str((arguments or {}).get("path") or (result.get("path") if isinstance(result, dict) else ""))
            if path:
                key = str(Path(path).expanduser())
                self._reads[key] = self._reads.get(key, 0) + 1
        return self.evidence_builder.build(tool_name, arguments, result)

    def tool_metadata(self, tool_name: str, arguments: Dict[str, Any] | None, result: Any) -> Dict[str, Any]:
        """Build UX metadata about long-task context handling."""
        metadata: Dict[str, Any] = {}
        evidence_tools = {
            "file_read", "gp_file_read",
            "file_list", "gp_file_list",
            "shell_run", "gp_shell_run",
        }
        if tool_name in evidence_tools:
            metadata["context_evidence"] = True
        if tool_name in {"file_read", "gp_file_read"}:
            path = str((arguments or {}).get("path") or "")
            if path:
                count = self._reads.get(str(Path(path).expanduser()), 0)
                metadata["read_count"] = count
                metadata["repeat_read"] = count > self.repeated_read_limit
        if isinstance(result, dict):
            if result.get("truncated"):
                metadata["tool_truncated"] = True
            if result.get("mode"):
                metadata["mode"] = result.get("mode")
        return metadata

    def convergence_notice(self, round_number: int) -> str:
        """Return a notice that nudges synthesis after excessive exploration."""
        if round_number < self.convergence_round:
            return ""
        return (
            "SYSTEM: You have explored for many rounds. Stop broad reading, "
            "deduplicate evidence already gathered, and synthesize the final answer."
        )
