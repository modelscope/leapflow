# Copyright (c) Alibaba, Inc. and its affiliates.
"""Prompt cache optimization — reorganizes messages to maximize prefix cache hits.

Modern LLM APIs cache request prefixes automatically. This module ensures
the message structure is cache-friendly by separating stable content (system prompt,
tool schemas, persona) from dynamic content (conversation turns).
"""
from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

from leapflow.engine.context.context_disclosure import CacheBoundary

# ── System-prompt static/dynamic split anchors ────────────────────────────
# These are deterministic structural markers — no NL fitting.  They mirror
# the section layout of ``UNIFIED_SYSTEM_TEMPLATE`` in ``leapflow.prompts``.
_STATIC_TERMINAL_ANCHOR = "When finished with all tool calls"
_KNOWN_STATIC_HEADERS = frozenset({
    "## Capabilities",
    "## Tool Usage",
    "## Guidelines",
    "## Coding & Verification",
    "## Presentation Style",
})


@runtime_checkable
class CacheStrategy(Protocol):
    """Protocol for prompt cache optimization strategies.

    Implementations reorder / annotate messages to maximise prefix-cache
    reuse.  The optional *cache_boundary* parameter lets the caller signal
    whether the current turn has a committed, soft, or no cache boundary —
    implementations that do not use it can accept and ignore the default
    ``CacheBoundary.NONE``.
    """

    def optimize(
        self,
        messages: List[Dict[str, Any]],
        *,
        cache_boundary: CacheBoundary = CacheBoundary.NONE,
    ) -> List[Dict[str, Any]]:
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

    def optimize(
        self,
        messages: List[Dict[str, Any]],
        *,
        cache_boundary: CacheBoundary = CacheBoundary.NONE,
    ) -> List[Dict[str, Any]]:
        """Reorganize messages for cache-friendliness.

        Boundary-aware behavior (cold-path, once per round):

        * ``COMMITTED`` — the prefix is byte-frozen by commitment enforcement.
          Reordering would introduce non-determinism; instead the existing
          message order is preserved and only the stable-prefix tail marker is
          applied (harmless on auto-cache providers, beneficial on Anthropic).
        * ``SOFT`` — normal stable-prefix reordering (system / frozen-memory /
          compressed-summary first) plus marker, encouraging early prefix
          formation before commitment.
        * ``NONE`` — identical to ``SOFT`` (backward-compatible default).
        """
        if not messages:
            return messages

        # COMMITTED: preserve byte-stable order — no reordering.
        if cache_boundary is CacheBoundary.COMMITTED:
            return self._committed_passthrough(messages)

        # SOFT / NONE: reorder to maximize stable prefix length.
        stable: List[Dict[str, Any]] = []
        dynamic: List[Dict[str, Any]] = []

        for msg in messages:
            # Volatile-context messages (per-turn memory / knowledge /
            # semantic focus) must stay *outside* the stable prefix so
            # that the cacheable prefix bytes remain turn-invariant.
            if msg.get("_volatile_context"):
                dynamic.append(msg)
            elif msg.get("role") in self._stable_roles:
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

    # ── boundary-specific helpers ────────────────────────────────────

    def _committed_passthrough(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """COMMITTED path: return messages without reordering.

        The only mutation is the cache-control marker on the last message
        whose role is in *stable_roles* (when markers are enabled), which
        does not change ordering or content bytes.
        """
        result = list(messages)
        if self._cache_marker_enabled:
            # Mark the last stable-role message in its *original* position.
            for i in range(len(result) - 1, -1, -1):
                if result[i].get("role") in self._stable_roles:
                    result[i] = {**result[i]}
                    result[i].setdefault("cache_control", {"type": "ephemeral"})
                    break
        return result

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
    ) -> None:
        self._breakpoints = breakpoints
        self._marker = {"type": "ephemeral"}

    def optimize(
        self,
        messages: List[Dict[str, Any]],
        *,
        cache_boundary: CacheBoundary = CacheBoundary.NONE,
    ) -> List[Dict[str, Any]]:
        if not messages:
            return messages

        import copy
        result = copy.deepcopy(messages)

        # System messages: split static/dynamic when cache-aware.
        # Skip volatile-context messages — they change every turn and
        # would waste a cache breakpoint on non-reusable content.
        for msg in result:
            if msg.get("role") == "system":
                if msg.get("_volatile_context"):
                    continue
                if cache_boundary in (CacheBoundary.SOFT, CacheBoundary.COMMITTED):
                    self._apply_split_marker(msg)
                else:
                    self._apply_marker(msg)

        # Conversation tail breakpoints
        non_system = [m for m in result if m.get("role") != "system"]
        for msg in non_system[-self._breakpoints:]:
            self._apply_marker(msg)

        return result

    # ── static/dynamic system-prompt splitting ──────────────────────

    def _apply_split_marker(self, msg: Dict[str, Any]) -> None:
        """Split system content into static (cached) + dynamic (uncached)."""
        content = msg.get("content")
        if not isinstance(content, str):
            self._apply_marker(msg)
            return
        static, dynamic = self._split_system_prompt(content)
        if dynamic:
            msg["content"] = [
                {"type": "text", "text": static, "cache_control": self._marker},
                {"type": "text", "text": dynamic},
            ]
        else:
            self._apply_marker(msg)

    @staticmethod
    def _split_system_prompt(system_content: str) -> tuple[str, str]:
        """Split a formatted system prompt into (static_part, dynamic_part).

        The split is deterministic and based on structural anchors from
        ``UNIFIED_SYSTEM_TEMPLATE``.  The static part contains identity,
        capability, and guideline sections; the dynamic part contains memory
        context, session summaries, and active signals that change per turn.

        Returns:
            A ``(static, dynamic)`` tuple.  If no reliable split anchor is
            found, the entire content is returned as static with an empty
            dynamic part.
        """
        # Strategy 1: find the terminal anchor of the static template body
        anchor_pos = system_content.rfind(_STATIC_TERMINAL_ANCHOR)
        if anchor_pos >= 0:
            line_end = system_content.find("\n", anchor_pos)
            if line_end < 0:
                return (system_content, "")
            split_pos = line_end + 1
            # Skip blank lines between the static body and the dynamic part
            while split_pos < len(system_content) and system_content[split_pos] in ("\n", "\r", " "):
                split_pos += 1
            if split_pos >= len(system_content):
                return (system_content, "")
            return (system_content[:split_pos], system_content[split_pos:])

        # Strategy 2: find the first ## header not in the known static set
        lines = system_content.split("\n")
        char_offset = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## ") and stripped not in _KNOWN_STATIC_HEADERS:
                if char_offset > 0:
                    return (system_content[:char_offset], system_content[char_offset:])
            char_offset += len(line) + 1  # +1 for the \n

        # Fallback: entire content is static
        return (system_content, "")

    @staticmethod
    def _apply_tool_cache_marker(
        tool_definitions: List[Dict[str, Any]],
        cache_boundary: CacheBoundary,
    ) -> List[Dict[str, Any]]:
        """Mark the last tool definition with ``cache_control`` when committed.

        Only activates when *cache_boundary* is ``COMMITTED`` — i.e. the tool
        schema array is frozen and the provider can reliably cache it.
        Operates on a deep copy so the caller’s original definitions are
        never mutated.

        Returns:
            A (possibly copied) tool-definition list.
        """
        if cache_boundary is not CacheBoundary.COMMITTED or not tool_definitions:
            return tool_definitions
        import copy
        result = copy.deepcopy(tool_definitions)
        last = result[-1]
        # If tool def has nested "function", mark at top level for Anthropic API
        last["cache_control"] = {"type": "ephemeral"}
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

    def optimize(
        self,
        messages: List[Dict[str, Any]],
        *,
        cache_boundary: CacheBoundary = CacheBoundary.NONE,
    ) -> List[Dict[str, Any]]:
        return messages
