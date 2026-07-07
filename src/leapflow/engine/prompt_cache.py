"""Prompt cache optimization — reorganizes messages to maximize prefix cache hits.

Modern LLM APIs cache request prefixes automatically. This module ensures
the message structure is cache-friendly by separating stable content (system prompt,
tool schemas, persona) from dynamic content (conversation turns).
"""
from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class CacheStrategy(Protocol):
    """Protocol for prompt cache optimization strategies."""

    def optimize(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reorder/restructure messages to maximize cache prefix reuse."""
        ...


class PrefixCacheOptimizer:
    """Maximizes LLM prefix cache hits by organizing messages into stable/dynamic sections.

    Strategy:
    1. Identify stable prefix: system prompt + tools schema + persona instructions
    2. Identify frozen content: memory context snapshots (marked with _frozen_memory)
    3. Ensure stable prefix is always at the start (never interleaved with dynamic content)
    4. Mark cache boundary (for APIs that support explicit cache_control)
    5. Dynamic section: conversation turns in chronological order
    """

    def __init__(
        self,
        *,
        cache_marker_enabled: bool = True,
        stable_roles: frozenset[str] = frozenset({"system"}),
    ) -> None:
        self._cache_marker_enabled = cache_marker_enabled
        self._stable_roles = stable_roles

    def optimize(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reorganize messages for cache-friendliness.

        Groups system messages and frozen memory blocks at the start as stable prefix,
        followed by dynamic conversation turns.
        """
        if not messages:
            return messages

        stable: List[Dict[str, Any]] = []
        dynamic: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") in self._stable_roles:
                stable.append(msg)
            elif msg.get("_frozen_memory"):
                stable.append(msg)
            elif msg.get("_compressed_summary"):
                stable.append(msg)
            else:
                dynamic.append(msg)

        if self._cache_marker_enabled and stable:
            last_stable = {**stable[-1]}
            last_stable.setdefault("cache_control", {"type": "ephemeral"})
            stable[-1] = last_stable

        return stable + dynamic

    def estimate_cache_ratio(self, messages: List[Dict[str, Any]]) -> float:
        """Estimate what fraction of tokens are in the cacheable prefix."""
        if not messages:
            return 0.0
        stable_chars = sum(
            len(str(msg.get("content", "")))
            for msg in messages
            if msg.get("role") in self._stable_roles
        )
        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        return stable_chars / max(total_chars, 1)


class AnthropicCacheStrategy:
    """Anthropic-optimized caching: system + last N messages with cache_control.

    Places cache breakpoints on:
    1. System message (stable prefix — highest reuse)
    2. Last *breakpoints* non-system messages (conversation tail)

    This mirrors hermes prompt_caching.py ``system_and_3`` strategy,
    adapted for LeapFlow's async-first architecture.
    """

    def __init__(
        self,
        *,
        breakpoints: int = 3,
        cache_ttl: str = "5m",
    ) -> None:
        self._breakpoints = breakpoints
        self._marker = {"type": "ephemeral"}

    def optimize(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not messages:
            return messages

        import copy
        result = copy.deepcopy(messages)

        for msg in result:
            if msg.get("role") == "system":
                self._apply_marker(msg)

        non_system = [m for m in result if m.get("role") != "system"]
        for msg in non_system[-self._breakpoints:]:
            self._apply_marker(msg)

        return result

    def _apply_marker(self, msg: Dict[str, Any]) -> None:
        """Attach cache_control marker to a message."""
        content = msg.get("content")
        if content is None or content == "":
            msg["cache_control"] = self._marker
        elif isinstance(content, str):
            msg["content"] = [
                {"type": "text", "text": content, "cache_control": self._marker}
            ]
        elif isinstance(content, list) and content:
            last = {**content[-1], "cache_control": self._marker}
            msg["content"] = content[:-1] + [last]


class NoCacheStrategy:
    """No-op cache strategy — passes messages through unchanged."""

    def optimize(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return messages
