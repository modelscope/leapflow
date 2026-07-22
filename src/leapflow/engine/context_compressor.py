"""Multi-stage context compression pipeline (Chain of Responsibility).

Four stages applied in order, each only activating when token budget is exceeded:
1. TrimStage: Truncate oversized tool outputs (zero LLM cost)
2. SummarizeStage: LLM-based summarization of middle turns (LLM cost)
3. ArchiveStage: Move early turns to long-term memory for retrieval (zero LLM cost)
4. DropStage: Force-drop oldest turns, keeping system + recent N (zero LLM cost)

Design principles:
- Each stage is independently configurable and skippable
- Stages respect cache boundaries (never compress system/stable prefix)
- Progressive: lighter stages first, heavier stages only if needed
- Message-pair integrity: tool-call/tool-result pairs are never split
- Summarization uses structured prompts with iterative updates
- All character/token thresholds adapt to the model's context window size
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

SummarizeFn = Callable[[str], Awaitable[str]]
ArchiveFn = Callable[[List[Dict[str, Any]]], Awaitable[None]]
TokenCountFn = Callable[[str], int]

# ── Adaptive scaling constants ────────────────────────────────────────
# These define how thresholds scale with context_length. They are NOT
# per-model magic numbers — the formulas produce smooth curves that
# work across 32K → 2M+ context windows.

_TRIM_CEILING_CHARS = 50_000
_TRIM_CONTEXT_DIVISOR = 50
_TRIM_BUDGET_ACTIVATION_RATIO = 0.15


def estimate_text_tokens(text: str) -> int:
    """CJK-aware token estimate for a single text string.

    CJK characters map roughly 1:1 to tokens; Latin text uses the
    standard ~4 characters per token heuristic.  Shared across the
    compression pipeline and budget estimator for consistency.
    """
    if not text:
        return 0
    cjk = sum(
        1 for ch in text
        if "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f"
    )
    latin = len(text) - cjk
    return max(1, cjk + latin // 4)


def adaptive_trim_chars(base: int, context_length: int) -> int:
    """Compute context-adaptive trim threshold in characters.

    Returns *at least* ``base`` chars, scaling up proportionally to the
    model's context window so that larger windows tolerate longer tool
    outputs without unnecessary truncation.
    """
    if context_length <= 0:
        return base
    context_chars = context_length * 4
    adaptive = min(_TRIM_CEILING_CHARS, context_chars // _TRIM_CONTEXT_DIVISOR)
    return max(base, adaptive)


@runtime_checkable
class CompressionStage(Protocol):
    """Protocol for a single compression stage."""

    @property
    def name(self) -> str:
        """Stage identifier for logging."""
        ...

    def should_apply(self, messages: List[Dict[str, Any]], token_count: int, budget: int) -> bool:
        """Whether this stage should activate given current state."""
        ...

    def apply(self, messages: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
        """Apply compression. Returns compressed message list."""
        ...


@dataclass
class CompressorConfig:
    """Configuration for the multi-stage compressor.

    Accepts legacy field names (threshold, keep_tail, max_output_chars) and
    maps them to current semantics.  When ``context_length`` is set, trim
    thresholds are scaled adaptively so that larger context windows tolerate
    bigger tool outputs without premature truncation.
    """

    token_budget: int = 128_000
    context_length: int = 0
    trim_threshold_chars: int = 2000
    trim_head_ratio: float = 0.25
    trim_tail_ratio: float = 0.1
    trim_ceiling_chars: int = _TRIM_CEILING_CHARS
    trim_budget_activation_ratio: float = _TRIM_BUDGET_ACTIVATION_RATIO
    summarize_threshold_messages: int = 16
    summarize_keep_recent: int = 6
    summarize_append_only: bool = True
    archive_threshold_messages: int = 24
    archive_keep_recent: int = 8
    drop_threshold_messages: int = 32
    drop_keep_recent: int = 4
    enabled_stages: List[str] = field(default_factory=lambda: ["trim", "summarize", "archive", "drop"])

    # Dedup: collapse identical tool results above this size
    dedup_min_chars: int = 200

    # LLM summarization callbacks (injected by orchestrator)
    summarize_fn: Optional[SummarizeFn] = field(default=None, repr=False)
    archive_fn: Optional[ArchiveFn] = field(default=None, repr=False)
    token_count_fn: Optional[TokenCountFn] = field(default=None, repr=False)

    # Legacy field aliases (backward compat with engine.py / tool_executor.py)
    threshold: int = 16
    keep_tail: int = 4
    max_output_chars: int = 2000

    def __post_init__(self) -> None:
        """Map legacy fields and apply adaptive scaling."""
        if self.max_output_chars != 2000 or self.trim_threshold_chars == 2000:
            self.trim_threshold_chars = self.max_output_chars
        if self.threshold != 16 or self.summarize_threshold_messages == 16:
            self.summarize_threshold_messages = self.threshold
        if self.keep_tail != 4 or self.summarize_keep_recent == 6:
            self.summarize_keep_recent = max(self.keep_tail, 4)
            self.drop_keep_recent = self.keep_tail

        self._base_trim_threshold = self.trim_threshold_chars
        self._apply_adaptive_scaling()

    def _apply_adaptive_scaling(self) -> None:
        """Scale thresholds proportionally to context window size."""
        self.trim_threshold_chars = adaptive_trim_chars(
            self._base_trim_threshold, self.context_length,
        )


class TrimStage:
    """Stage 1: Truncate oversized tool outputs + dedup identical results (zero LLM cost).

    Three passes:
    1. Dedup: collapse identical tool results (MD5, >dedup_min_chars)
    2. Truncate: head/tail trim for remaining oversized outputs
    3. Strip tool_call arguments over threshold

    Budget-aware: when context utilization is below ``budget_activation_ratio``
    only messages exceeding ``ceiling_chars`` are trimmed, preserving full tool
    output when the context window has plenty of room.
    """

    def __init__(
        self,
        *,
        max_chars: int = 2000,
        head_ratio: float = 0.25,
        tail_ratio: float = 0.1,
        dedup_min_chars: int = 200,
        ceiling_chars: int = _TRIM_CEILING_CHARS,
        budget_activation_ratio: float = _TRIM_BUDGET_ACTIVATION_RATIO,
    ) -> None:
        self._max_chars = max_chars
        self._ceiling_chars = ceiling_chars
        self._budget_activation_ratio = budget_activation_ratio
        self._head_size = max(100, int(max_chars * head_ratio))
        self._tail_size = max(50, int(max_chars * tail_ratio))
        self._dedup_min = dedup_min_chars

    @property
    def name(self) -> str:
        return "trim"

    def should_apply(self, messages: List[Dict[str, Any]], token_count: int, budget: int) -> bool:
        target = [
            msg for msg in messages
            if msg.get("role") in ("tool", "function", "assistant")
            and isinstance(msg.get("content"), str)
        ]
        if any(len(msg["content"]) > self._ceiling_chars for msg in target):
            return True
        if budget > 0 and token_count < budget * self._budget_activation_ratio:
            return False
        return any(len(msg["content"]) > self._max_chars for msg in target)

    def apply(self, messages: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
        result = self._dedup_tool_results(messages)
        result = self._truncate_outputs(result)
        return result

    def _dedup_tool_results(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse identical tool results to a placeholder."""
        seen_hashes: dict[str, int] = {}
        result: List[Dict[str, Any]] = []
        deduped = 0
        for msg in messages:
            content = msg.get("content", "")
            if (
                msg.get("role") in ("tool", "function")
                and isinstance(content, str)
                and len(content) > self._dedup_min
            ):
                h = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()
                if h in seen_hashes:
                    result.append({**msg, "content": "[Duplicate tool output — see earlier result]"})
                    deduped += 1
                    continue
                seen_hashes[h] = len(result)
            result.append(msg)
        if deduped:
            logger.debug("TrimStage: deduped %d identical tool results", deduped)
        return result

    def _truncate_outputs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Truncate oversized tool/assistant outputs with head+tail preservation."""
        effective_limit = min(self._max_chars, self._ceiling_chars)
        result: List[Dict[str, Any]] = []
        trimmed_count = 0
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            if (
                isinstance(content, str)
                and len(content) > effective_limit
                and role in ("tool", "function", "assistant")
            ):
                head = content[: self._head_size]
                tail = content[-self._tail_size:]
                trimmed_content = (
                    f"{head}\n\n"
                    f"[... trimmed {len(content) - self._head_size - self._tail_size} chars ...]\n\n"
                    f"{tail}"
                )
                result.append({**msg, "content": trimmed_content})
                trimmed_count += 1
            else:
                result.append(msg)
        if trimmed_count:
            logger.debug("TrimStage: trimmed %d oversized messages", trimmed_count)
        return result


_SUMMARY_PREFIX = (
    "[REFERENCE ONLY — Historical Context Summary]\n"
    "The following summarizes earlier conversation turns. "
    "The latest user message takes precedence over anything here. "
    "Treat historical sections as past context, not active instructions.\n\n"
)

_SUMMARY_END_MARKER = "\n[End of historical summary]"

_SUMMARIZE_SYSTEM_PROMPT = """\
You are a context compression agent. Summarize the conversation turns below \
into a structured, information-dense summary that preserves all actionable details.

Output format (use ALL applicable sections):
## Historical Task Snapshot
<verbatim latest unfulfilled user request from the turns>

## Completed Actions
<numbered list: N. ACTION target — outcome [tool: name]>

## Active State
<current working state: files modified, variables set, pending operations>

## Key Decisions
<important choices made and their rationale>

## Relevant Files
<files/paths mentioned or modified>

Rules:
- Preserve exact file paths, variable names, error messages, and code snippets
- Rewrite completed actions as past-tense facts
- Use the same language as the user's messages
- Redact any secrets (API keys, passwords, tokens)
- Stay within the budget — be concise but complete"""

_ITERATIVE_PROMPT = """\
You are updating an existing context summary with new conversation turns.

EXISTING SUMMARY:
{previous_summary}

NEW TURNS TO INCORPORATE:
{new_turns}

Merge the new information into the existing summary structure. \
Move completed tasks from "Active State" to "Completed Actions". \
Update all sections as needed. Preserve the same output format."""


def _log_archive_result(task: asyncio.Task[None]) -> None:
    """Callback for archive tasks — log failures instead of silencing them."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning(
            "ArchiveStage: background archive failed (%s: %s)",
            type(exc).__name__, exc,
        )


# ── Deterministic summary helpers ─────────────────────────────────────

_KEY_ARG_NAMES = (
    "path", "file", "command", "query", "action", "url", "chat_id", "pattern", "text",
)


def _compact_tool_args(args_str: str, limit: int = 120) -> str:
    """Extract key argument values from tool call JSON arguments."""
    if not args_str:
        return ""
    try:
        args = json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        return args_str[:limit]
    if not isinstance(args, dict):
        return str(args)[:limit]
    parts: list[str] = []
    for key in _KEY_ARG_NAMES:
        if key in args:
            parts.append(f"{key}={str(args[key])[:60]}")
    if not parts:
        for key, val in list(args.items())[:2]:
            parts.append(f"{key}={str(val)[:40]}")
    return ", ".join(parts[:3])


_KEY_RESULT_FIELDS = (
    "ok", "error", "path", "kind", "message_id", "resource_id", "id", "exit_code",
)


def _compact_tool_result(content: str, limit: int = 150) -> str:
    """Extract key information from a tool result for summaries."""
    if not content or content == "null":
        return "completed"
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            parts: list[str] = []
            for key in _KEY_RESULT_FIELDS:
                if key in data:
                    parts.append(f"{key}={str(data[key])[:60]}")
            if parts:
                return "; ".join(parts[:4])
    except (json.JSONDecodeError, TypeError):
        pass
    first_line = content.split("\n", 1)[0].strip()
    if first_line:
        return first_line[:limit]
    return content[:limit]


class SummarizeStage:
    """Stage 2: LLM-based summarization of middle turns.

    Features (inspired by hermes context_compressor):
    - Protects system prompt (head) and recent turns (tail)
    - Boundary alignment: never splits tool-call/tool-result pairs
    - LLM-based structured summarization with iterative updates
    - Deterministic fallback when LLM is unavailable
    - Anti-thrashing: skips if last compression saved <10%
    """

    def __init__(
        self,
        *,
        threshold_messages: int = 16,
        keep_recent: int = 6,
        summarize_fn: Optional[SummarizeFn] = None,
        summary_target_ratio: float = 0.2,
        append_only: bool = True,
    ) -> None:
        self._threshold = threshold_messages
        self._keep_recent = keep_recent
        self._summarize_fn = summarize_fn
        self._summary_target_ratio = summary_target_ratio
        self._append_only = append_only
        self._previous_summary: Optional[str] = None
        self._compression_count: int = 0
        self._last_savings_ratio: float = 1.0

    @property
    def name(self) -> str:
        return "summarize"

    def should_apply(self, messages: List[Dict[str, Any]], token_count: int, budget: int) -> bool:
        if len(messages) <= self._threshold:
            return False
        if token_count <= budget * 0.5:
            return False
        if self._compression_count >= 2 and self._last_savings_ratio < 0.10:
            logger.debug("SummarizeStage: skipped (anti-thrashing, last savings %.1f%%)",
                         self._last_savings_ratio * 100)
            return False
        return True

    def apply(self, messages: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
        if len(messages) <= self._keep_recent + 2:
            return messages

        pre_count = len(messages)

        head, tail, middle = self._partition(messages)
        if not middle:
            return messages

        summary_text = self._build_summary(middle)
        summary_msg = {
            "role": "user",
            "content": _SUMMARY_PREFIX + summary_text + _SUMMARY_END_MARKER,
            "_compressed_summary": True,
        }

        result = self._sanitize_tool_pairs(head + [summary_msg] + tail)

        self._previous_summary = summary_text
        self._compression_count += 1
        self._last_savings_ratio = 1.0 - (len(result) / max(pre_count, 1))
        logger.debug("SummarizeStage: compressed %d → %d messages (%.0f%% reduction)",
                      pre_count, len(result), self._last_savings_ratio * 100)
        return result

    def _partition(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split messages into head (system), tail (recent), middle (compressible).

        Append-only mode freezes already-produced summary segments into the head
        so each history window is summarized exactly once (no summary-of-summary
        drift; frozen segments stay byte-stable and cacheable). Only newly
        accumulated turns become the compressible middle.
        """
        head = [m for m in messages[:2] if m.get("role") == "system"] or messages[:1]
        head_count = len(head)

        frozen_count = 0
        if self._append_only:
            idx = head_count
            while idx < len(messages) and messages[idx].get("_compressed_summary"):
                frozen_count += 1
                idx += 1
        stable_count = head_count + frozen_count

        tail_start = max(stable_count, len(messages) - self._keep_recent)
        tail_start = self._align_boundary_backward(messages, tail_start)

        if self._append_only or self._compression_count > 0:
            protect_first_n = 0
        else:
            protect_first_n = min(2, len(messages) - head_count)

        middle_start = stable_count + protect_first_n
        middle_start = self._align_boundary_forward(messages, middle_start)

        if middle_start >= tail_start:
            return head + messages[head_count:stable_count], messages[stable_count:], []

        middle = messages[middle_start:tail_start]
        tail = messages[tail_start:]

        return head + messages[head_count:middle_start], tail, middle

    def _build_summary(self, middle: List[Dict[str, Any]]) -> str:
        """Build summary using LLM if available, else deterministic fallback."""
        if self._summarize_fn is not None:
            try:
                turns_text = self._format_turns_for_summary(middle)
                if self._previous_summary and not self._append_only:
                    prompt = _ITERATIVE_PROMPT.format(
                        previous_summary=self._previous_summary,
                        new_turns=turns_text,
                    )
                else:
                    prompt = (
                        f"{_SUMMARIZE_SYSTEM_PROMPT}\n\n"
                        f"CONVERSATION TURNS TO SUMMARIZE:\n{turns_text}"
                    )
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(asyncio.run, self._summarize_fn(prompt))
                        result = future.result(timeout=30.0)
                else:
                    result = loop.run_until_complete(self._summarize_fn(prompt))
                if result and len(result.strip()) > 20:
                    return result.strip()
            except Exception as exc:
                logger.warning("SummarizeStage: LLM summarization failed (%s), using fallback", exc)

        return self._deterministic_summary(middle)

    @staticmethod
    def _format_turns_for_summary(messages: List[Dict[str, Any]]) -> str:
        """Format messages as readable turns for the summarization prompt."""
        lines: List[str] = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))
            tool_calls = msg.get("tool_calls") or []
            if role == "assistant" and tool_calls:
                tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                lines.append(f"[Turn {i+1}] ASSISTANT: called tools: {', '.join(tool_names)}")
                if content:
                    lines.append(f"  Text: {content[:200]}")
            elif role == "tool":
                tool_id = msg.get("tool_call_id", "")
                lines.append(f"[Turn {i+1}] TOOL ({tool_id}): {content[:300]}")
            else:
                lines.append(f"[Turn {i+1}] {role.upper()}: {content[:400]}")
        return "\n".join(lines)

    @staticmethod
    def _deterministic_summary(middle: List[Dict[str, Any]]) -> str:
        """Zero-LLM-cost summary as fallback.

        Preserves tool names with key arguments, structured fields from
        tool results (paths, error codes, resource ids), and the full
        breadth of user messages so that the agent retains enough context
        to continue its task after compression.
        """
        summary_lines: List[str] = []
        for msg in middle:
            role = msg.get("role", "")
            content = str(msg.get("content", ""))
            tool_calls = msg.get("tool_calls") or []
            if role == "assistant" and tool_calls:
                parts = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "?")
                    args_summary = _compact_tool_args(func.get("arguments", ""))
                    parts.append(f"{name}({args_summary})" if args_summary else name)
                summary_lines.append(f"[assistant] called: {', '.join(parts)}")
                if content:
                    summary_lines.append(f"  → {content[:120]}")
            elif role in ("tool", "function"):
                excerpt = _compact_tool_result(content)
                summary_lines.append(f"[tool] {excerpt}")
            elif role == "user":
                summary_lines.append(f"[user] {content[:200]}")
            else:
                summary_lines.append(f"[{role}] {content[:120]}")
        return (
            f"## Historical Context ({len(middle)} messages)\n"
            + "\n".join(summary_lines[-30:])
        )

    @staticmethod
    def _align_boundary_forward(messages: List[Dict[str, Any]], idx: int) -> int:
        """If cut starts on a tool message, slide forward past orphan tool results."""
        while idx < len(messages) and messages[idx].get("role") == "tool":
            idx += 1
        return idx

    @staticmethod
    def _align_boundary_backward(messages: List[Dict[str, Any]], idx: int) -> int:
        """If cut lands mid tool-result run, walk back to include parent assistant."""
        while idx > 0 and messages[idx].get("role") == "tool":
            idx -= 1
        if idx > 0 and messages[idx].get("role") == "assistant" and messages[idx].get("tool_calls"):
            pass  # include the assistant with tool_calls in the tail
        return idx

    @staticmethod
    def _sanitize_tool_pairs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove orphaned tool results and strip orphaned tool_calls."""
        available_call_ids: set[str] = set()
        for msg in messages:
            for tc in msg.get("tool_calls") or []:
                call_id = tc.get("id", "") or tc.get("call_id", "")
                if call_id:
                    available_call_ids.add(call_id)

        result: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                if tool_call_id and tool_call_id not in available_call_ids:
                    continue
            elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                responded_ids: set[str] = set()
                for later in messages:
                    if later.get("role") == "tool":
                        responded_ids.add(later.get("tool_call_id", ""))

                kept_calls = [
                    tc for tc in msg["tool_calls"]
                    if (tc.get("id", "") or tc.get("call_id", "")) in responded_ids
                ]
                if not kept_calls and not msg.get("content"):
                    msg = {**msg, "content": "(tool call removed)", "tool_calls": []}
                elif len(kept_calls) != len(msg["tool_calls"]):
                    msg = {**msg, "tool_calls": kept_calls}
            result.append(msg)
        return result


class ArchiveStage:
    """Stage 3: Archive early turns to long-term memory for later retrieval.

    Moves conversation history to SemanticMemory — the information is not lost,
    just moved from context window to retrievable storage.
    """

    def __init__(
        self,
        *,
        threshold_messages: int = 24,
        keep_recent: int = 8,
        archive_fn: Optional[ArchiveFn] = None,
    ) -> None:
        self._threshold = threshold_messages
        self._keep_recent = keep_recent
        self._archive_fn = archive_fn

    @property
    def name(self) -> str:
        return "archive"

    def should_apply(self, messages: List[Dict[str, Any]], token_count: int, budget: int) -> bool:
        return len(messages) > self._threshold and token_count > budget * 0.8

    def apply(self, messages: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
        if len(messages) <= self._keep_recent + 2:
            return messages

        # Keep system head + recent tail
        head = messages[:1] if messages and messages[0].get("role") == "system" else []
        tail = messages[-self._keep_recent:]
        archived = messages[len(head) : -self._keep_recent]

        if not archived:
            return messages

        if self._archive_fn is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    task = asyncio.create_task(self._archive_fn(archived))
                    task.add_done_callback(_log_archive_result)
                else:
                    loop.run_until_complete(self._archive_fn(archived))
            except Exception as exc:
                logger.warning("ArchiveStage: archive_fn failed (%s)", exc)

        # Archive notification (actual archiving is async, handled externally)
        archive_notice = {
            "role": "system",
            "content": (
                f"[{len(archived)} earlier messages archived to long-term memory. "
                f"Use memory retrieval if you need earlier context.]"
            ),
        }

        logger.debug("ArchiveStage: archived %d messages to long-term memory", len(archived))
        return head + [archive_notice] + tail


class DropStage:
    """Stage 4: Force-drop oldest turns (last resort).

    Only keeps system prompt + most recent N turns.
    Information IS lost — this is the nuclear option.
    """

    def __init__(self, *, threshold_messages: int = 32, keep_recent: int = 4) -> None:
        self._threshold = threshold_messages
        self._keep_recent = keep_recent

    @property
    def name(self) -> str:
        return "drop"

    def should_apply(self, messages: List[Dict[str, Any]], token_count: int, budget: int) -> bool:
        return len(messages) > self._threshold or token_count > budget

    def apply(self, messages: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
        if len(messages) <= self._keep_recent + 1:
            return messages

        head = messages[:1] if messages and messages[0].get("role") == "system" else []
        tail = messages[-self._keep_recent:]
        dropped = len(messages) - len(head) - len(tail)

        drop_notice = {
            "role": "system",
            "content": f"[Context overflow: {dropped} messages dropped. Only recent context available.]",
        }

        logger.warning("DropStage: force-dropped %d messages (context overflow)", dropped)
        return head + [drop_notice] + tail


@dataclass(frozen=True)
class CompressionTrace:
    """Observable summary of a compression pass for UI and audit metadata."""

    stages_applied: List[str] = field(default_factory=list)
    stage_effects: List[Dict[str, Any]] = field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0
    messages_before: int = 0
    messages_after: int = 0
    savings_ratio: float = 0.0
    saved_tokens: int = 0
    forced: bool = False
    preflight_truncated_messages: int = 0
    decision_reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """Return a compact JSON-serializable representation."""
        return {
            "stages_applied": list(self.stages_applied),
            "stage_effects": [dict(effect) for effect in self.stage_effects],
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "messages_before": self.messages_before,
            "messages_after": self.messages_after,
            "savings_ratio": self.savings_ratio,
            "saved_tokens": self.saved_tokens,
            "forced": self.forced,
            "preflight_truncated_messages": self.preflight_truncated_messages,
            "decision_reason": self.decision_reason,
        }


class ContextCompressor:
    """Multi-stage context compression orchestrator.

    Applies compression stages in order (Chain of Responsibility).
    Each stage only activates if its condition is met.
    Respects cache boundaries — system messages are never compressed.
    """

    def __init__(self, config: CompressorConfig = CompressorConfig()) -> None:
        self._config = config
        self._stages: List[CompressionStage] = self._build_stages(config)
        self._last_trace = CompressionTrace()

    @property
    def last_trace(self) -> CompressionTrace:
        """Return the most recent compression trace."""
        return self._last_trace

    def reconfigure(self, *, token_budget: int = 0, context_length: int = 0) -> None:
        """Update runtime budget/context and rebuild affected stages.

        Only TrimStage is rebuilt — SummarizeStage and other stateful stages
        retain their iterative summary history and compression counters so
        that a hot-reload mid-session does not break summary continuity.
        """
        changed = False
        if token_budget > 0 and token_budget != self._config.token_budget:
            self._config.token_budget = token_budget
            changed = True
        if context_length > 0 and context_length != self._config.context_length:
            self._config.context_length = context_length
            self._config._apply_adaptive_scaling()
            changed = True
        if changed:
            new_trim = TrimStage(
                max_chars=self._config.trim_threshold_chars,
                head_ratio=self._config.trim_head_ratio,
                tail_ratio=self._config.trim_tail_ratio,
                dedup_min_chars=self._config.dedup_min_chars,
                ceiling_chars=self._config.trim_ceiling_chars,
                budget_activation_ratio=self._config.trim_budget_activation_ratio,
            )
            self._stages = [
                new_trim if stage.name == "trim" else stage
                for stage in self._stages
            ]
            logger.debug(
                "ContextCompressor reconfigured: budget=%d, context_length=%d, trim_chars=%d",
                self._config.token_budget,
                self._config.context_length,
                self._config.trim_threshold_chars,
            )

    def _build_stages(self, config: CompressorConfig) -> List[CompressionStage]:
        """Build stage chain from config."""
        stage_map: Dict[str, CompressionStage] = {
            "trim": TrimStage(
                max_chars=config.trim_threshold_chars,
                head_ratio=config.trim_head_ratio,
                tail_ratio=config.trim_tail_ratio,
                dedup_min_chars=config.dedup_min_chars,
                ceiling_chars=config.trim_ceiling_chars,
                budget_activation_ratio=config.trim_budget_activation_ratio,
            ),
            "summarize": SummarizeStage(
                threshold_messages=config.summarize_threshold_messages,
                keep_recent=config.summarize_keep_recent,
                summarize_fn=config.summarize_fn,
                append_only=config.summarize_append_only,
            ),
            "archive": ArchiveStage(
                threshold_messages=config.archive_threshold_messages,
                keep_recent=config.archive_keep_recent,
                archive_fn=config.archive_fn,
            ),
            "drop": DropStage(
                threshold_messages=config.drop_threshold_messages,
                keep_recent=config.drop_keep_recent,
            ),
        }
        return [stage_map[name] for name in config.enabled_stages if name in stage_map]

    def compress(self, messages: List[Dict[str, Any]], *, token_count: int = 0) -> List[Dict[str, Any]]:
        """Apply compression pipeline. Stages activate progressively as needed."""
        budget = self._config.token_budget

        if token_count <= 0:
            token_count = self._count_tokens(messages)
        tokens_before = token_count
        messages_before = len(messages)
        stages_applied: List[str] = []

        stage_effects: List[Dict[str, Any]] = []
        for stage in self._stages:
            if stage.should_apply(messages, token_count, budget):
                before_count = token_count
                before_messages = len(messages)
                messages = stage.apply(messages, budget)
                next_count = self._count_tokens(messages)
                changed = next_count != before_count or len(messages) != before_messages
                if changed:
                    stages_applied.append(stage.name)
                    stage_effects.append({
                        "stage": stage.name,
                        "tokens_before": before_count,
                        "tokens_after": next_count,
                        "messages_before": before_messages,
                        "messages_after": len(messages),
                    })
                token_count = next_count

        self._last_trace = self._build_trace(
            stages_applied=stages_applied,
            stage_effects=stage_effects,
            tokens_before=tokens_before,
            tokens_after=token_count,
            messages_before=messages_before,
            messages_after=len(messages),
            decision_reason="threshold-triggered" if stages_applied else "within-budget",
        )
        return messages

    def force_compress(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Force all stages regardless of thresholds (for overflow recovery)."""
        budget = self._config.token_budget
        original_tokens = self._count_tokens(messages)
        current_tokens = original_tokens
        messages_before = len(messages)
        stages_applied: List[str] = []
        stage_effects: List[Dict[str, Any]] = []
        for stage in self._stages:
            before_count = current_tokens
            before_messages = len(messages)
            messages = stage.apply(messages, budget)
            next_count = self._count_tokens(messages)
            changed = next_count != before_count or len(messages) != before_messages
            if changed:
                stages_applied.append(stage.name)
                stage_effects.append({
                    "stage": stage.name,
                    "tokens_before": before_count,
                    "tokens_after": next_count,
                    "messages_before": before_messages,
                    "messages_after": len(messages),
                })
            current_tokens = next_count
        tokens_after = self._count_tokens(messages)
        self._last_trace = self._build_trace(
            stages_applied=stages_applied,
            stage_effects=stage_effects,
            tokens_before=original_tokens,
            tokens_after=tokens_after,
            messages_before=messages_before,
            messages_after=len(messages),
            forced=True,
            decision_reason="hard-gate-overflow",
        )
        return messages

    @staticmethod
    def _build_trace(
        *,
        stages_applied: List[str],
        tokens_before: int,
        tokens_after: int,
        messages_before: int,
        messages_after: int,
        stage_effects: List[Dict[str, Any]] | None = None,
        forced: bool = False,
        preflight_truncated_messages: int = 0,
        decision_reason: str = "",
    ) -> CompressionTrace:
        saved = max(0, tokens_before - tokens_after)
        savings_ratio = saved / tokens_before if tokens_before > 0 else 0.0
        return CompressionTrace(
            stages_applied=stages_applied,
            stage_effects=stage_effects or [],
            tokens_before=max(0, tokens_before),
            tokens_after=max(0, tokens_after),
            messages_before=max(0, messages_before),
            messages_after=max(0, messages_after),
            savings_ratio=savings_ratio,
            saved_tokens=saved,
            forced=forced,
            preflight_truncated_messages=max(0, preflight_truncated_messages),
            decision_reason=decision_reason,
        )

    def _count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens using external tokenizer if available, else estimate.

        Always handles multimodal content parts and images regardless
        of whether a custom ``token_count_fn`` is set.
        """
        _IMAGE_TOKEN_ESTIMATE = 1600
        if self._config.token_count_fn is not None:
            fn = self._config.token_count_fn
            total = 0
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    total += fn(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                total += fn(part.get("text", ""))
                            elif part.get("type") in ("image_url", "input_image", "image"):
                                total += _IMAGE_TOKEN_ESTIMATE
                for tc in msg.get("tool_calls") or []:
                    args = tc.get("function", {}).get("arguments", "")
                    if isinstance(args, str):
                        total += fn(args)
            return total
        return self._estimate_tokens(messages)

    def preflight_check(
        self,
        messages: List[Dict[str, Any]],
        *,
        context_length: int = 128_000,
        huge_message_chars: int = 50_000,
    ) -> List[Dict[str, Any]]:
        """Preflight gate: detect and compress few-but-huge messages before LLM call.

        Handles cases where message count is low but individual messages are enormous
        (e.g., base64 images, massive file reads, large paste buffers). These won't
        trigger the standard count-based stages but will overflow the context window.

        Strategy: truncate any non-system message exceeding huge_message_chars,
        preserving head + tail with a truncation notice.
        """
        truncated_count = 0
        result = []
        for msg in messages:
            if msg.get("role") == "system":
                result.append(msg)
                continue

            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) <= huge_message_chars:
                result.append(msg)
                continue

            head_size = huge_message_chars // 3
            tail_size = huge_message_chars // 6
            truncated_content = (
                content[:head_size]
                + f"\n\n[... {len(content) - head_size - tail_size:,} chars truncated ...]\n\n"
                + content[-tail_size:]
            )
            new_msg = dict(msg, content=truncated_content)
            result.append(new_msg)
            truncated_count += 1
            logger.info(
                "preflight: truncated %s message from %d to %d chars",
                msg.get("role", "?"), len(content), len(truncated_content),
            )

        estimated_tokens = self._count_tokens(result)
        preflight_messages_after = len(result)
        if estimated_tokens > context_length * 0.9:
            result = self.force_compress(result)
            previous = self._last_trace
            self._last_trace = CompressionTrace(
                stages_applied=[*(['preflight'] if truncated_count else []), *previous.stages_applied],
                stage_effects=[
                    *([{
                        "stage": "preflight",
                        "tokens_before": self._count_tokens(messages),
                        "tokens_after": estimated_tokens,
                        "messages_before": len(messages),
                        "messages_after": preflight_messages_after,
                    }] if truncated_count else []),
                    *previous.stage_effects,
                ],
                tokens_before=max(previous.tokens_before, estimated_tokens),
                tokens_after=previous.tokens_after,
                messages_before=max(previous.messages_before, len(messages)),
                messages_after=previous.messages_after,
                savings_ratio=previous.savings_ratio,
                saved_tokens=previous.saved_tokens,
                forced=previous.forced,
                preflight_truncated_messages=truncated_count,
                decision_reason="huge-message-preflight+hard-gate-overflow" if truncated_count else previous.decision_reason,
            )
        elif truncated_count:
            tokens_before = self._count_tokens(messages)
            self._last_trace = self._build_trace(
                stages_applied=["preflight"],
                stage_effects=[{
                    "stage": "preflight",
                    "tokens_before": tokens_before,
                    "tokens_after": estimated_tokens,
                    "messages_before": len(messages),
                    "messages_after": len(result),
                }],
                tokens_before=tokens_before,
                tokens_after=estimated_tokens,
                messages_before=len(messages),
                messages_after=len(result),
                preflight_truncated_messages=truncated_count,
                decision_reason="huge-message-preflight",
            )

        return result

    @staticmethod
    def _estimate_tokens(messages: List[Dict[str, Any]]) -> int:
        """CJK-aware token estimation with multimodal support.

        Uses ``estimate_text_tokens`` for consistent CJK-aware counting
        and adds ~1600 tokens per image part.
        """
        _IMAGE_TOKEN_ESTIMATE = 1600
        total_tokens = 0
        image_count = 0

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_tokens += estimate_text_tokens(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type", "")
                    if part_type == "text":
                        total_tokens += estimate_text_tokens(part.get("text", ""))
                    elif part_type in ("image_url", "input_image", "image"):
                        image_count += 1

            for tc in msg.get("tool_calls") or []:
                args = tc.get("function", {}).get("arguments", "")
                if isinstance(args, str):
                    total_tokens += estimate_text_tokens(args)

        return total_tokens + (image_count * _IMAGE_TOKEN_ESTIMATE)
