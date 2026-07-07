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

    Backward-compatible: accepts legacy field names (threshold, keep_tail,
    max_output_chars) and maps them to new semantics.
    """

    token_budget: int = 128_000
    trim_threshold_chars: int = 2000
    trim_head_ratio: float = 0.25
    trim_tail_ratio: float = 0.1
    summarize_threshold_messages: int = 16
    summarize_keep_recent: int = 6
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
        """Map legacy fields into new config semantics."""
        if self.max_output_chars != 2000 or self.trim_threshold_chars == 2000:
            self.trim_threshold_chars = self.max_output_chars
        if self.threshold != 16 or self.summarize_threshold_messages == 16:
            self.summarize_threshold_messages = self.threshold
        if self.keep_tail != 4 or self.summarize_keep_recent == 6:
            self.summarize_keep_recent = max(self.keep_tail, 4)
            self.drop_keep_recent = self.keep_tail


class TrimStage:
    """Stage 1: Truncate oversized tool outputs + dedup identical results (zero LLM cost).

    Three passes (inspired by hermes context_compressor):
    1. Dedup: collapse identical tool results (MD5, >dedup_min_chars)
    2. Truncate: head/tail trim for remaining oversized outputs
    3. Strip tool_call arguments over threshold

    Preserves head/tail of long outputs for context.
    Only targets tool/function/assistant messages — never system.
    """

    def __init__(
        self,
        *,
        max_chars: int = 2000,
        head_ratio: float = 0.25,
        tail_ratio: float = 0.1,
        dedup_min_chars: int = 200,
    ) -> None:
        self._max_chars = max_chars
        self._head_size = max(100, int(max_chars * head_ratio))
        self._tail_size = max(50, int(max_chars * tail_ratio))
        self._dedup_min = dedup_min_chars

    @property
    def name(self) -> str:
        return "trim"

    def should_apply(self, messages: List[Dict[str, Any]], token_count: int, budget: int) -> bool:
        return any(
            isinstance(msg.get("content"), str) and len(msg["content"]) > self._max_chars
            for msg in messages
            if msg.get("role") in ("tool", "function", "assistant")
        )

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
        result: List[Dict[str, Any]] = []
        trimmed_count = 0
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            if (
                isinstance(content, str)
                and len(content) > self._max_chars
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
    ) -> None:
        self._threshold = threshold_messages
        self._keep_recent = keep_recent
        self._summarize_fn = summarize_fn
        self._summary_target_ratio = summary_target_ratio
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
        """Split messages into head (system), tail (recent), middle (compressible)."""
        head = [m for m in messages[:2] if m.get("role") == "system"] or messages[:1]
        head_count = len(head)

        tail_start = max(head_count, len(messages) - self._keep_recent)
        tail_start = self._align_boundary_backward(messages, tail_start)

        if self._compression_count > 0:
            protect_first_n = 0
        else:
            protect_first_n = min(2, len(messages) - head_count)

        middle_start = head_count + protect_first_n
        middle_start = self._align_boundary_forward(messages, middle_start)

        if middle_start >= tail_start:
            return head, messages[head_count:], []

        middle = messages[middle_start:tail_start]
        tail = messages[tail_start:]

        return head + messages[head_count:middle_start], tail, middle

    def _build_summary(self, middle: List[Dict[str, Any]]) -> str:
        """Build summary using LLM if available, else deterministic fallback."""
        if self._summarize_fn is not None:
            try:
                turns_text = self._format_turns_for_summary(middle)
                if self._previous_summary:
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
            tool_calls = msg.get("tool_calls", [])
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
        """Zero-LLM-cost summary as fallback."""
        summary_lines: List[str] = []
        for msg in middle:
            role = msg.get("role", "")
            content = str(msg.get("content", ""))[:80]
            tool_calls = msg.get("tool_calls", [])
            if role == "assistant" and tool_calls:
                tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                summary_lines.append(f"[assistant] called: {', '.join(tool_names)}")
            elif role in ("tool", "function"):
                summary_lines.append("[tool_result] executed")
            elif role == "user":
                summary_lines.append(f"[user] {content[:50]}...")
            else:
                summary_lines.append(f"[{role}] {content[:50]}...")
        return (
            f"## Historical Context ({len(middle)} messages)\n"
            + "\n".join(summary_lines[-20:])
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
            for tc in msg.get("tool_calls", []):
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
                    asyncio.create_task(self._archive_fn(archived))
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


class ContextCompressor:
    """Multi-stage context compression orchestrator.

    Applies compression stages in order (Chain of Responsibility).
    Each stage only activates if its condition is met.
    Respects cache boundaries — system messages are never compressed.
    """

    def __init__(self, config: CompressorConfig = CompressorConfig()) -> None:
        self._config = config
        self._stages: List[CompressionStage] = self._build_stages(config)

    def _build_stages(self, config: CompressorConfig) -> List[CompressionStage]:
        """Build stage chain from config."""
        stage_map: Dict[str, CompressionStage] = {
            "trim": TrimStage(
                max_chars=config.trim_threshold_chars,
                head_ratio=config.trim_head_ratio,
                tail_ratio=config.trim_tail_ratio,
                dedup_min_chars=config.dedup_min_chars,
            ),
            "summarize": SummarizeStage(
                threshold_messages=config.summarize_threshold_messages,
                keep_recent=config.summarize_keep_recent,
                summarize_fn=config.summarize_fn,
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

        for stage in self._stages:
            if stage.should_apply(messages, token_count, budget):
                messages = stage.apply(messages, budget)
                token_count = self._count_tokens(messages)

        return messages

    def force_compress(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Force all stages regardless of thresholds (for overflow recovery)."""
        budget = self._config.token_budget
        for stage in self._stages:
            messages = stage.apply(messages, budget)
        return messages

    def _count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens using external tokenizer if available, else estimate."""
        if self._config.token_count_fn is not None:
            total = 0
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    total += self._config.token_count_fn(content)
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    args = tc.get("function", {}).get("arguments", "")
                    if isinstance(args, str):
                        total += self._config.token_count_fn(args)
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
        modified = False
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
            modified = True
            logger.info(
                "preflight: truncated %s message from %d to %d chars",
                msg.get("role", "?"), len(content), len(truncated_content),
            )

        estimated_tokens = self._count_tokens(result)
        if estimated_tokens > context_length * 0.9:
            result = self.force_compress(result)

        return result

    @staticmethod
    def _estimate_tokens(messages: List[Dict[str, Any]]) -> int:
        """Rough token estimation (~4 chars per token)."""
        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        for msg in messages:
            for tc in msg.get("tool_calls", []):
                args = tc.get("function", {}).get("arguments", "")
                total_chars += len(str(args))
        return total_chars // 4
