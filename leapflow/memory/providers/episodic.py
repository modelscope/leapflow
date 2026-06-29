"""Episodic memory provider — TTL-based buffer with decay-weighted retrieval.

Handles transient observations, events, and actions. Entries decay over time
and are automatically evicted when expired. Promotion to semantic storage
is triggered externally via touch().

Provides legacy-compatible ingest/recent/search methods for upstream consumers
(EventBus, Engine, StateSnapshotService) that operate on MemoryFragment objects.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional

from dataclasses import dataclass

from leapflow.memory.protocol import (
    MemoryEntry,
    MemoryKind,
    MemoryQuery,
    MemoryToolSchema,
    SignalDomain,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Configuration constants (overridable via constructor)
# ──────────────────────────────────────────────────────────────────────

_DEFAULT_TTL_SECONDS: float = 300.0
_DEFAULT_GC_INTERVAL: float = 30.0
_DEFAULT_MAX_ENTRIES: int = 200
_DEFAULT_DECAY_LAMBDA: float = 1e-5

# ──────────────────────────────────────────────────────────────────────
# Type aliases and domain types
# ──────────────────────────────────────────────────────────────────────

PromotionCallback = Callable[["MemoryFragment"], None]


@dataclass
class MemoryFragment:
    """Legacy-compatible fragment returned by ingest/recent/search."""

    fragment_id: str
    event_type: str
    content: str
    path: Optional[str]
    metadata: Dict[str, Any]
    created_at: float
    ttl: float = _DEFAULT_TTL_SECONDS
    referenced: bool = False

    @property
    def expired(self) -> bool:
        return time.time() - self.created_at > self.ttl

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


# ──────────────────────────────────────────────────────────────────────
# Inline decay formula
# ──────────────────────────────────────────────────────────────────────

def _decay_score(
    semantic_weight: float,
    age_seconds: float,
    frequency: float,
    decay_lambda: float = _DEFAULT_DECAY_LAMBDA,
) -> float:
    """W = S * exp(-lambda * age) * log(1 + frequency)."""
    if semantic_weight <= 0 or frequency <= 0:
        return 0.0
    normalized_freq = 1.0 + math.log1p(frequency - 1.0)
    return semantic_weight * math.exp(-decay_lambda * age_seconds) * normalized_freq


# ──────────────────────────────────────────────────────────────────────
# Provider
# ──────────────────────────────────────────────────────────────────────

class EpisodicMemoryProvider:
    """Time-decaying buffer for observations, events, and actions.

    Each entry carries a TTL; expired entries are garbage-collected automatically.
    Touching an entry promotes it via an optional callback (e.g. to semantic tier).
    """

    _ACCEPTED_KINDS = frozenset({
        MemoryKind.OBSERVATION,
        MemoryKind.EVENT,
        MemoryKind.ACTION,
    })

    def __init__(
        self,
        *,
        ttl: float = _DEFAULT_TTL_SECONDS,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        gc_interval: float = _DEFAULT_GC_INTERVAL,
        decay_lambda: float = _DEFAULT_DECAY_LAMBDA,
        on_promote: Optional[PromotionCallback] = None,
    ) -> None:
        self._ttl = ttl
        self._max_entries = max_entries
        self._gc_interval = gc_interval
        self._decay_lambda = decay_lambda
        self._on_promote = on_promote

        self._entries: Dict[str, MemoryEntry] = {}
        self._access_counts: Dict[str, int] = {}
        self._gc_task: Optional[asyncio.Task[None]] = None
        self._counter: int = 0

    # ── Protocol properties ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return "episodic"

    # ── Protocol methods ──────────────────────────────────────────────

    async def initialize(self, **kwargs: Any) -> None:
        """Start background GC loop."""
        self._start_gc()

    async def shutdown(self) -> None:
        """Stop GC and clear state."""
        self._stop_gc()
        self._entries.clear()
        self._access_counts.clear()

    def accepts(self, entry: MemoryEntry) -> bool:
        return entry.kind in self._ACCEPTED_KINDS

    async def insert(self, entry: MemoryEntry) -> str:
        """Ingest a new episodic entry."""
        self._entries[entry.entry_id] = entry
        self._access_counts.setdefault(entry.entry_id, 1)
        self._evict_overflow()
        logger.debug(
            "episodic.insert id=%s kind=%s domain=%s",
            entry.entry_id, entry.kind.value, entry.domain.value,
        )
        return entry.entry_id

    async def search(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Search with keyword + domain + time_range filters, scored by decay."""
        now = time.time()
        keywords_lower = [k.lower() for k in query.keywords] if query.keywords else []
        results: List[MemoryEntry] = []

        for entry in self._entries.values():
            # TTL check
            if now - entry.timestamp > self._ttl:
                continue

            # Kind filter
            if query.kinds and entry.kind not in query.kinds:
                continue

            # Domain filter
            if query.domains and entry.domain not in query.domains:
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

            # Compute decay-weighted score
            age = max(0.0, now - entry.timestamp)
            freq = float(self._access_counts.get(entry.entry_id, 1))
            # Semantic weight: fraction of keywords matched (or 1.0 if no keywords)
            if keywords_lower:
                matched = sum(1 for kw in keywords_lower if kw in entry.content.lower())
                semantic = matched / len(keywords_lower)
            else:
                semantic = 1.0
            entry.score = _decay_score(semantic, age, freq, self._decay_lambda)

            if entry.score >= query.min_score:
                results.append(entry)

        # Sort by score descending, truncate to limit
        results.sort(key=lambda e: e.score, reverse=True)
        return results[: query.limit]

    async def delete(self, entry_id: str) -> bool:
        if entry_id not in self._entries:
            return False
        del self._entries[entry_id]
        self._access_counts.pop(entry_id, None)
        return True

    # ── Lifecycle hooks ────────────────────────────────────────────────

    def on_turn_start(self, turn: int, user_message: str) -> None:
        """Trigger a GC sweep on turn boundaries."""
        self._gc_sweep()

    def on_inserted(self, entry: MemoryEntry) -> None:
        """No-op for episodic provider."""

    def on_accessed(self, entry: MemoryEntry) -> None:
        """Increment access count if the entry belongs to us."""
        if entry.entry_id in self._entries:
            self._access_counts[entry.entry_id] = (
                self._access_counts.get(entry.entry_id, 1) + 1
            )

    def on_promoted(self, entry: MemoryEntry, source_provider: str) -> None:
        """Accept promoted entries from working memory."""
        self._entries[entry.entry_id] = entry
        self._access_counts.setdefault(entry.entry_id, 1)
        self._evict_overflow()
        logger.debug("episodic.on_promoted from=%s id=%s", source_provider, entry.entry_id)

    def get_tool_schemas(self) -> list:
        """Expose recent events search tool to LLM."""
        return [MemoryToolSchema(
            name="memory_recent_events",
            description="Retrieve recent system events (file changes, clipboard, app focus, etc.).",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20, "description": "Max events to return"},
                },
                "required": [],
            },
            provider_name="episodic",
        )]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Handle LLM tool call for recent events."""
        import json as _json
        if tool_name == "memory_recent_events":
            limit = int(args.get("limit", 20))
            frags = self.recent(limit=limit)
            return _json.dumps({
                "events": [
                    {"type": f.event_type, "content": f.content[:200], "time": f.created_at}
                    for f in frags
                ],
            }, ensure_ascii=False)
        return _json.dumps({"error": f"Unknown tool: {tool_name}"})

    # ── Domain helpers ────────────────────────────────────────────────

    def touch(self, entry_id: str) -> Optional[MemoryEntry]:
        """Mark entry as referenced, increment access count, optionally promote."""
        entry = self._entries.get(entry_id)
        if entry is None:
            return None
        now = time.time()
        if now - entry.timestamp > self._ttl:
            return None
        self._access_counts[entry_id] = self._access_counts.get(entry_id, 1) + 1
        if self._on_promote:
            frag = self._entry_to_fragment(entry)
            self._on_promote(frag)
        return entry

    # ── Legacy interface (used by EventBus, Engine, StateSnapshot) ────

    def ingest(
        self,
        event_type: str,
        content: str,
        *,
        path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryFragment:
        """Insert a new fragment from an incoming system event (legacy API)."""
        self._counter += 1
        entry_id = f"imm_{self._counter}_{int(time.time() * 1000)}"
        now = time.time()
        entry = MemoryEntry(
            entry_id=entry_id,
            kind=MemoryKind.EVENT,
            domain=SignalDomain.SYSTEM,
            content=content,
            timestamp=now,
            score=1.0,
            metadata={
                "event_type": event_type,
                "path": path,
                **(metadata or {}),
            },
        )
        self._entries[entry_id] = entry
        self._access_counts.setdefault(entry_id, 1)
        self._evict_overflow()
        logger.debug("episodic.ingest type=%s path=%s", event_type, path)
        return self._entry_to_fragment(entry)

    def recent(self, limit: int = 20) -> List[MemoryFragment]:
        """Return most recent non-expired entries as MemoryFragments."""
        now = time.time()
        alive = [
            e for e in self._entries.values()
            if now - e.timestamp <= self._ttl
        ]
        alive.sort(key=lambda e: e.timestamp)
        return [self._entry_to_fragment(e) for e in alive[-limit:]]

    def search_fragments(
        self, keywords: List[str], limit: int = 10
    ) -> List[MemoryFragment]:
        """Simple keyword search over non-expired entries (legacy API)."""
        now = time.time()
        alive = [
            e for e in self._entries.values()
            if now - e.timestamp <= self._ttl
        ]
        if not keywords:
            alive.sort(key=lambda e: e.timestamp)
            return [self._entry_to_fragment(e) for e in alive[-limit:]]
        results: List[MemoryEntry] = []
        for entry in sorted(alive, key=lambda e: e.timestamp, reverse=True):
            text_lower = entry.content.lower()
            meta_str = str(entry.metadata.get("event_type", "")).lower()
            searchable = f"{text_lower} {meta_str}"
            if any(kw.lower() in searchable for kw in keywords):
                results.append(entry)
                if len(results) >= limit:
                    break
        return [self._entry_to_fragment(e) for e in results]

    def start_gc(self) -> None:
        """Start GC loop (legacy compatibility)."""
        self._start_gc()

    def stop_gc(self) -> None:
        """Stop GC loop (legacy compatibility)."""
        self._stop_gc()

    @property
    def active_count(self) -> int:
        now = time.time()
        return sum(1 for e in self._entries.values() if now - e.timestamp <= self._ttl)

    @property
    def size(self) -> int:
        return len(self._entries)

    # ── Internal ──────────────────────────────────────────────────────

    def _evict_overflow(self) -> None:
        """Remove oldest entries when capacity exceeded."""
        while len(self._entries) > self._max_entries:
            oldest_id = min(self._entries, key=lambda eid: self._entries[eid].timestamp)
            del self._entries[oldest_id]
            self._access_counts.pop(oldest_id, None)

    def _start_gc(self) -> None:
        if self._gc_task is None or self._gc_task.done():
            self._gc_task = asyncio.create_task(self._gc_loop())

    def _stop_gc(self) -> None:
        if self._gc_task and not self._gc_task.done():
            self._gc_task.cancel()

    async def _gc_loop(self) -> None:
        """Periodically sweep expired entries."""
        try:
            while True:
                await asyncio.sleep(self._gc_interval)
                self._gc_sweep()
        except asyncio.CancelledError:
            return

    def _gc_sweep(self) -> None:
        now = time.time()
        expired_ids = [
            eid for eid, e in self._entries.items() if now - e.timestamp > self._ttl
        ]
        for eid in expired_ids:
            del self._entries[eid]
            self._access_counts.pop(eid, None)
        if expired_ids:
            logger.debug("episodic.gc removed=%d remaining=%d", len(expired_ids), len(self._entries))

    def _entry_to_fragment(self, entry: MemoryEntry) -> MemoryFragment:
        """Convert a MemoryEntry to a MemoryFragment for legacy callers."""
        return MemoryFragment(
            fragment_id=entry.entry_id,
            event_type=str(entry.metadata.get("event_type", entry.kind.value)),
            content=entry.content,
            path=entry.metadata.get("path"),
            metadata=entry.metadata,
            created_at=entry.timestamp,
            ttl=self._ttl,
            referenced=self._access_counts.get(entry.entry_id, 1) > 1,
        )
