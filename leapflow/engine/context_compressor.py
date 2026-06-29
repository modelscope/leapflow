"""Deterministic context compression for bounded token budgets."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompressorConfig:
    threshold: int = 16
    keep_tail: int = 4
    max_output_chars: int = 2000


class ContextCompressor:
    """Four-stage deterministic compression pipeline (zero LLM cost)."""

    def __init__(self, config: CompressorConfig = CompressorConfig()):
        self._config = config

    def compress(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compress messages if exceeding threshold."""
        if len(messages) <= self._config.threshold:
            return messages
        messages = self._trim_outputs(messages)
        messages = self._protect_and_summarize(messages)
        return messages

    def force_compress(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Force compression regardless of threshold (for context overflow recovery)."""
        messages = self._trim_outputs(messages)
        messages = self._protect_and_summarize(messages)
        return messages

    def _trim_outputs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 1: Truncate oversized tool outputs."""
        max_chars = self._config.max_output_chars
        # Preserve proportional head/tail when truncating
        head_size = max(100, max_chars // 4)
        tail_size = max(50, max_chars // 10)
        result = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > max_chars:
                head = content[:head_size]
                tail = content[-tail_size:]
                trimmed = (
                    f"{head}\n\n"
                    f"[... truncated {len(content) - head_size - tail_size} chars ...]\n\n"
                    f"{tail}"
                )
                result.append({**msg, "content": trimmed})
            else:
                result.append(msg)
        return result

    def _protect_and_summarize(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Stage 2-4: Protect head/tail, summarize middle."""
        if len(messages) <= 4:
            return messages

        # Head: system + first user (always protected)
        head = messages[:2]
        # Tail: recent interactions (protected)
        tail_size = min(self._config.keep_tail * 2, len(messages) - 2)
        tail = messages[-tail_size:] if tail_size > 0 else []
        # Middle: compressible region
        middle = messages[2:-tail_size] if tail_size > 0 else messages[2:]

        if not middle:
            return messages

        # Summarize middle into compact entries
        summary_lines = []
        for msg in middle:
            role = msg.get("role", "")
            content = str(msg.get("content", ""))[:100]
            if role == "assistant" and "{" in content:
                summary_lines.append(f"[assistant] tool_call: {content[:60]}...")
            elif "Tool result" in content or "tool" in role:
                ok = (
                    "ok"
                    if "ok" not in content or '"ok": true' in content.lower()
                    else "error"
                )
                summary_lines.append(f"[tool_result] → {ok}")
            else:
                summary_lines.append(f"[{role}] {content[:50]}...")

        summary_msg = {
            "role": "user",
            "content": (
                f"[Compressed context — {len(middle)} messages summarized]\n"
                + "\n".join(summary_lines)
            ),
        }

        return head + [summary_msg] + tail
