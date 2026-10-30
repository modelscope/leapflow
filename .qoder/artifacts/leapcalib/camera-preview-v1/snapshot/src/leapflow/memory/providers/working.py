# Copyright (c) Alibaba, Inc. and its affiliates.
"""Working memory provider — ring-buffer with token budgeting.

Implements MemoryProvider protocol while preserving chat-message semantics
used directly by the engine (remember_chat, as_chat_messages, remember_event).
"""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional

from leapflow.memory.protocol import MemoryEntry, MemoryKind, MemoryQuery


# ──────────────────────────────────────────────────────────────────────
# Token estimation helpers
# ──────────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Approximate token count handling both Latin and CJK text."""
    if not text:
        return 0
    cjk_count = sum(
        1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f"
    )
    latin_chars = len(text) - cjk_count
    return max(1, cjk_count + latin_chars // 4)


def _message_token_weight(msg: Dict[str, object]) -> int:
    """Compute approximate token cost of a single chat message."""
    content = msg.get("content", "")
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            else:
                parts.append(str(p))
        content = "\n".join(parts)
    elif not isinstance(content, str):
        content = str(content)
    base = 4
    return base + _estimate_tokens(content) + (2 if msg.get("role") else 0)


# ──────────────────────────────────────────────────────────────────────
# Provider
# ──────────────────────────────────────────────────────────────────────

_DEFAULT_WORKING_MEMORY_TOKENS = 8_192
_AUTO_WORKING_MEMORY_MIN_TOKENS = 16_384
_AUTO_WORKING_MEMORY_MAX_TOKENS = 131_072
_AUTO_WORKING_MEMORY_CONTEXT_DIVISOR = 8


def resolve_working_memory_max_tokens(configured_max_tokens: int, context_length: int) -> int:
    """Resolve a safe working-memory budget from an explicit or automatic setting.

    ``0`` means automatic sizing. The active conversation context cannot remain
    useful when its short-term history is fixed at a tiny legacy buffer while the
    selected model offers a much larger window. Explicit positive values remain
    an operator override; automatic values reserve up to one eighth of the model
    window, bounded so ordinary sessions do not grow without limit.
    """
    configured = int(configured_max_tokens or 0)
    if configured > 0:
        return configured
    context = max(1, int(context_length or 1))
    return min(
        _AUTO_WORKING_MEMORY_MAX_TOKENS,
        max(_AUTO_WORKING_MEMORY_MIN_TOKENS, context // _AUTO_WORKING_MEMORY_CONTEXT_DIVISOR),
    )


class WorkingMemoryProvider:
    """Token-budgeted ring buffer for conversation turns.

    Serves as the fastest memory tier — ephemeral, in-process, zero persistence.
    """

    _ACCEPTED_KINDS = frozenset({MemoryKind.CONVERSATION})

    def __init__(self, *, max_tokens: int = _DEFAULT_WORKING_MEMORY_TOKENS) -> None:
        self._max_tokens = max(1, int(max_tokens))
        self._items: Deque[Dict[str, object]] = deque()
        self._entries: Dict[str, MemoryEntry] = {}
        self._token_sum: int = 0
        self._evicted_messages: int = 0
        self._pattern_counts: Counter[str] = Counter()

    # ── Protocol properties ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return "working"

    # ── Protocol methods ──────────────────────────────────────────────

    async def initialize(self, **kwargs: Any) -> None:
        """No external resources needed."""

    async def shutdown(self) -> None:
        self.clear()

    def accepts(self, entry: MemoryEntry) -> bool:
        return entry.kind in self._ACCEPTED_KINDS

    async def insert(self, entry: MemoryEntry) -> str:
        """Insert a MemoryEntry as a synthetic chat message."""
        msg: Dict[str, object] = {
            "role": entry.metadata.get("role", "system"),
            "content": entry.content,
        }
        self.remember_chat(msg)
        self._entries[entry.entry_id] = entry
        return entry.entry_id

    async def search(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Keyword + domain + kind filter over buffered entries with decay scoring."""
        now = time.time()
        results: List[MemoryEntry] = []
        keywords_lower = [k.lower() for k in query.keywords] if query.keywords else []

        for entry in self._entries.values():
            # Domain filter
            if query.domains and entry.domain not in query.domains:
                continue

            # Kind filter
            if query.kinds and entry.kind not in query.kinds:
                continue

            # Time range filter
            if query.time_range:
                t_min, t_max = query.time_range
                if not (t_min <= entry.timestamp <= t_max):
                    continue

            # Keyword filter
            if keywords_lower:
                content_lower = entry.content.lower()
                if not any(kw in content_lower for kw in keywords_lower):
                    continue

            # Compute decay-weighted score: W = S * exp(-λ*age) * log(1+freq)
            age = max(0.0, now - entry.timestamp)
            freq = float(entry.access_count)
            if keywords_lower:
                matched = sum(1 for kw in keywords_lower if kw in entry.content.lower())
                semantic = matched / len(keywords_lower)
            else:
                semantic = 1.0

            if semantic > 0 and freq > 0:
                normalized_freq = 1.0 + math.log1p(freq - 1.0)
                entry.score = semantic * math.exp(-1e-5 * age) * normalized_freq
            else:
                entry.score = 0.0

            if entry.score >= query.min_score:
                results.append(entry)

        results.sort(key=lambda e: e.score, reverse=True)
        return results[:query.limit]

    async def delete(self, entry_id: str) -> bool:
        if entry_id not in self._entries:
            return False
        del self._entries[entry_id]
        return True

    # ── Lifecycle hooks (no-op for working memory) ─────────────────────

    def on_turn_start(self, turn: int, user_message: str) -> None:
        """No-op for working memory."""

    def on_inserted(self, entry: MemoryEntry) -> None:
        """No-op for working memory."""

    def on_accessed(self, entry: MemoryEntry) -> None:
        """No-op for working memory."""

    def get_tool_schemas(self) -> list:
        """Working memory is too ephemeral to expose as LLM tools."""
        return []

    # ── Engine-facing convenience methods ─────────────────────────────

    def remember_chat(self, message: Dict[str, object]) -> None:
        """Insert a chat-style message; evict from left if over budget."""
        w = _message_token_weight(message)
        self._items.append(message)
        self._token_sum += w
        self._evict_if_needed()

    _INTERNAL_KEYS = frozenset({"_event_ts", "_event_kind", "_event_text"})

    def remember_event(self, kind: str, text: str, metadata: Optional[Dict[str, object]] = None) -> None:
        """Store a compact system-side event as a synthetic system message."""
        ts = time.time()
        payload: Dict[str, object] = {
            "ts": ts,
            "kind": kind,
            "text": text,
            "metadata": metadata or {},
        }
        msg: Dict[str, object] = {
            "role": "system",
            "content": repr(payload),
            "_event_ts": ts,
            "_event_kind": kind,
            "_event_text": text,
        }
        self.remember_chat(msg)

    def get_events_since(self, since_ts: float) -> List[Dict[str, object]]:
        """Return system event messages recorded after ``since_ts``.

        Scans the item ring from newest to oldest and stops once messages
        are older than the threshold, keeping the scan O(recent) rather
        than O(total).
        """
        result: List[Dict[str, object]] = []
        for msg in reversed(self._items):
            ts = msg.get("_event_ts")
            if ts is None:
                continue
            if isinstance(ts, (int, float)) and ts > since_ts:
                result.append(msg)
            elif isinstance(ts, (int, float)):
                break
        result.reverse()
        return result

    def as_chat_messages(self) -> List[Dict[str, object]]:
        """Return a list copy suitable for LLM APIs.

        Internal metadata keys (_event_ts, etc.) are stripped so the
        output is safe to pass directly to LLM chat endpoints.
        """
        out: List[Dict[str, object]] = []
        for msg in self._items:
            if any(k in msg for k in self._INTERNAL_KEYS):
                out.append({k: v for k, v in msg.items() if k not in self._INTERNAL_KEYS})
            else:
                out.append(msg)
        return out

    @property
    def max_tokens(self) -> int:
        """Configured capacity available to this session's working memory."""
        return self._max_tokens

    @property
    def token_count(self) -> int:
        """Approximate token weight currently retained in the ring."""
        return self._token_sum

    @property
    def evicted_messages(self) -> int:
        """Number of messages evicted since construction or the last clear."""
        return self._evicted_messages

    def snapshot(self) -> Dict[str, int]:
        """Return safe observability data for context diagnostics."""
        return {
            "token_count": self._token_sum,
            "max_tokens": self._max_tokens,
            "message_count": len(self._items),
            "evicted_messages": self._evicted_messages,
        }

    def reconfigure_max_tokens(self, max_tokens: int) -> None:
        """Apply a live capacity change while preserving the newest context."""
        self._max_tokens = max(1, int(max_tokens))
        self._evict_if_needed()

    def clear(self) -> None:
        self._items.clear()
        self._entries.clear()
        self._token_sum = 0
        self._evicted_messages = 0
        self._pattern_counts.clear()

    def get_pattern_count(self, key: str) -> int:
        return self._pattern_counts[key]

    def increment_pattern(self, key: str) -> int:
        self._pattern_counts[key] += 1
        return self._pattern_counts[key]

    # ── Internal ──────────────────────────────────────────────────────

    def _evict_if_needed(self) -> None:
        while self._token_sum > self._max_tokens and self._items:
            old = self._items.popleft()
            self._token_sum -= _message_token_weight(old)
            self._evicted_messages += 1
