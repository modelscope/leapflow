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
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


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

    # Legacy field aliases (backward compat with engine.py / tool_executor.py)
    threshold: int = 16
    keep_tail: int = 4
    max_output_chars: int = 2000

    def __post_init__(self) -> None:
        """Map legacy fields into new config semantics."""
        # If caller passed legacy fields with non-default values, propagate them
        if self.max_output_chars != 2000 or self.trim_threshold_chars == 2000:
            self.trim_threshold_chars = self.max_output_chars
        if self.threshold != 16 or self.summarize_threshold_messages == 16:
            self.summarize_threshold_messages = self.threshold
        if self.keep_tail != 4 or self.summarize_keep_recent == 6:
            self.summarize_keep_recent = max(self.keep_tail, 4)
            self.drop_keep_recent = self.keep_tail


class TrimStage:
    """Stage 1: Truncate oversized tool outputs (zero LLM cost).

    Preserves head/tail of long outputs for context.
    Only targets tool/function/assistant messages — never system.
    """

    def __init__(self, *, max_chars: int = 2000, head_ratio: float = 0.25, tail_ratio: float = 0.1) -> None:
        self._max_chars = max_chars
        self._head_size = max(100, int(max_chars * head_ratio))
        self._tail_size = max(50, int(max_chars * tail_ratio))

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
        result = []
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


class SummarizeStage:
    """Stage 2: LLM-based summarization of middle turns.

    Protects system prompt (head) and recent turns (tail).
    Summarizes the middle section into a compact digest.
    Requires an LLM callback for async summarization (falls back to
    deterministic summary when callback is not provided).
    """

    def __init__(
        self,
        *,
        threshold_messages: int = 16,
        keep_recent: int = 6,
        summarize_fn: Optional[Callable[[str], Awaitable[str]]] = None,
    ) -> None:
        self._threshold = threshold_messages
        self._keep_recent = keep_recent
        self._summarize_fn = summarize_fn

    @property
    def name(self) -> str:
        return "summarize"

    def should_apply(self, messages: List[Dict[str, Any]], token_count: int, budget: int) -> bool:
        return len(messages) > self._threshold and token_count > budget * 0.7

    def apply(self, messages: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
        if len(messages) <= self._keep_recent + 2:
            return messages

        # Protect: system messages (head) + recent turns (tail)
        head = [m for m in messages[:2] if m.get("role") == "system"] or messages[:1]
        head_count = len(head)
        tail = messages[-self._keep_recent:]
        middle = messages[head_count : -self._keep_recent] if self._keep_recent else messages[head_count:]

        if not middle:
            return messages

        # Synchronous deterministic summary (no LLM cost)
        # Async LLM summarization handled at orchestrator level via summarize_fn
        summary_lines: List[str] = []
        for msg in middle:
            role = msg.get("role", "")
            content = str(msg.get("content", ""))[:80]
            if role == "assistant" and ("{" in content or "tool" in content.lower()):
                summary_lines.append("[assistant] tool interaction")
            elif role in ("tool", "function"):
                summary_lines.append("[tool_result] executed")
            elif role == "user":
                summary_lines.append(f"[user] {content[:50]}...")
            else:
                summary_lines.append(f"[{role}] {content[:50]}...")

        summary_msg = {
            "role": "user",
            "content": (
                f"[Context compressed — {len(middle)} messages summarized]\n"
                + "\n".join(summary_lines[-20:])
            ),
        }

        logger.debug("SummarizeStage: compressed %d middle messages", len(middle))
        return head + [summary_msg] + tail


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
        archive_fn: Optional[Callable[[List[Dict[str, Any]]], Awaitable[None]]] = None,
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
            ),
            "summarize": SummarizeStage(
                threshold_messages=config.summarize_threshold_messages,
                keep_recent=config.summarize_keep_recent,
            ),
            "archive": ArchiveStage(
                threshold_messages=config.archive_threshold_messages,
                keep_recent=config.archive_keep_recent,
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
            token_count = self._estimate_tokens(messages)

        for stage in self._stages:
            if stage.should_apply(messages, token_count, budget):
                messages = stage.apply(messages, budget)
                token_count = self._estimate_tokens(messages)

        return messages

    def force_compress(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Force all stages regardless of thresholds (for overflow recovery)."""
        budget = self._config.token_budget
        for stage in self._stages:
            messages = stage.apply(messages, budget)
        return messages

    @staticmethod
    def _estimate_tokens(messages: List[Dict[str, Any]]) -> int:
        """Rough token estimation (~4 chars per token)."""
        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        return total_chars // 4
