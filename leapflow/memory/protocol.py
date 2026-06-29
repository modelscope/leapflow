"""Memory subsystem protocol definitions.

Defines the universal interface that all memory providers must implement,
plus shared data types for entries, queries, signal domains, and memory kinds.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


class SignalDomain(str, Enum):
    """Signal source domain — extensible via string value."""
    VISION = "vision"
    FILESYSTEM = "fs"
    INPUT = "input"
    CLIPBOARD = "clipboard"
    NETWORK = "network"
    API = "api"
    DIALOG = "dialog"
    SYSTEM = "system"


class MemoryKind(str, Enum):
    """Semantic category of a memory entry."""
    OBSERVATION = "observation"
    ACTION = "action"
    SKILL_EPISODE = "skill_episode"
    SKILL_PATTERN = "skill_pattern"
    CONVERSATION = "conversation"
    PREDICTION = "prediction"
    EVENT = "event"
    FACT = "fact"
    USER_PREFERENCE = "user_pref"
    SESSION_SUMMARY = "session_summary"


@dataclass
class MemoryEntry:
    """A single memory record — universal across all providers."""
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    kind: MemoryKind = MemoryKind.OBSERVATION
    domain: SignalDomain = SignalDomain.SYSTEM
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 1
    ttl: Optional[float] = None  # None = never expires

    @property
    def expired(self) -> bool:
        """Whether this entry has exceeded its TTL."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    @property
    def age_seconds(self) -> float:
        """Seconds since creation."""
        return time.time() - self.timestamp


@dataclass
class MemoryQuery:
    """Structured query for memory search."""
    keywords: List[str] = field(default_factory=list)
    kinds: Optional[List[MemoryKind]] = None
    domains: Optional[List[SignalDomain]] = None
    time_range: Optional[Tuple[float, float]] = None
    limit: int = 20
    min_score: float = 0.0
    include_expired: bool = False
    cross_domain: bool = False  # Enable cross-domain correlation


@dataclass
class MemoryToolSchema:
    """Tool definition for LLM-exposed memory operations."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    provider_name: str = ""


@runtime_checkable
class MemoryProvider(Protocol):
    """Universal memory provider interface.

    All memory layers (working, episodic, semantic, evolution) implement
    this protocol, enabling uniform access through MemoryManager.

    Design principles:
    1. 6 core methods cover CRUD + retrieval + status
    2. Lifecycle hooks are optional (default no-op)
    3. Tool metadata is optionally exposed
    4. All IO operations are non-blocking
    """

    @property
    def name(self) -> str:
        """Unique provider identifier."""
        ...

    async def initialize(self, **kwargs: Any) -> None:
        """Initialize resources (DB connections, indices, etc.)."""
        ...

    async def shutdown(self) -> None:
        """Release resources gracefully."""
        ...

    async def insert(self, entry: MemoryEntry) -> str:
        """Store an entry. Returns entry_id."""
        ...

    async def search(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Search entries matching query criteria."""
        ...

    async def delete(self, entry_id: str) -> bool:
        """Remove an entry by ID. Returns True if found and deleted."""
        ...

    def accepts(self, entry: MemoryEntry) -> bool:
        """Whether this provider should handle the given entry kind."""
        ...

    # ══════════════ Lifecycle hooks (optional, default no-op) ══════════════

    def on_turn_start(self, turn: int, user_message: str) -> None:
        """Called at the start of each agent turn."""
        ...

    def on_promoted(self, entry: MemoryEntry, source_provider: str) -> None:
        """Called when an entry is promoted from another provider."""
        ...

    def on_inserted(self, entry: MemoryEntry) -> None:
        """Called after an entry is successfully inserted."""
        ...

    def on_accessed(self, entry: MemoryEntry) -> None:
        """Called when an entry is accessed/touched."""
        ...

    # ══════════════ Tool interface (optional) ══════════════

    def get_tool_schemas(self) -> List[MemoryToolSchema]:
        """Return tool schemas this provider exposes to the LLM."""
        ...

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Handle an LLM-initiated tool call."""
        ...
