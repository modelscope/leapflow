# Copyright (c) Alibaba, Inc. and its affiliates.
"""Memory manager — unified orchestrator for all memory providers.

Routes inserts to appropriate providers, aggregates search results,
manages lifecycle hooks, and exposes memory as LLM-callable tools.
"""
from __future__ import annotations

import inspect
import json
import logging
import math
import re as _re
from pathlib import Path
from typing import Any, Dict, List, Optional

from leapflow.memory.protocol import (
    MemoryEntry,
    MemoryKind,
    MemoryProvider,
    MemoryQuery,
    MemoryToolSchema,
    SignalDomain,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# CJK-aware tokenization for search
# ──────────────────────────────────────────────────────────────────────

def _tokenize_for_search(text: str) -> list[str]:
    """Split text into search tokens. CJK runs become overlapping bigrams."""
    tokens: list[str] = []
    for segment in _re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9_\-./]+', text):
        if _re.match(r'[\u4e00-\u9fff]', segment):
            # CJK: overlapping bigrams (or single char if length 1)
            if len(segment) == 1:
                tokens.append(segment)
            else:
                for i in range(len(segment) - 1):
                    tokens.append(segment[i:i+2])
        else:
            tokens.append(segment.lower())
    return tokens[:8]


# ──────────────────────────────────────────────────────────────────────
# Shared decay formula (avoids circular import)
# ──────────────────────────────────────────────────────────────────────

_DEFAULT_DECAY_LAMBDA: float = 1e-5


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


class MemoryManager:
    """Coordinates multiple MemoryProviders with intelligent routing.

    Responsibilities:
    - Provider registration and lifecycle management
    - Intelligent insert routing based on entry kind/ttl
    - Cross-provider search aggregation with decay scoring
    - Cross-domain correlation search within time windows
    - Lifecycle hook dispatch (on_turn_start, on_inserted, etc.)
    - LLM tool schema synthesis and tool call routing
    """

    def __init__(
        self,
        *,
        decay_lambda: float = _DEFAULT_DECAY_LAMBDA,
        cross_domain_window_s: float = 300.0,
    ) -> None:
        self._providers: Dict[str, MemoryProvider] = {}
        self._provider_order: List[str] = []
        self._initialized = False
        self._decay_lambda = decay_lambda
        self._cross_domain_window_s = cross_domain_window_s
        # Tool dispatch mapping: tool_name → provider_name
        self._tool_dispatch: Dict[str, str] = {}

    # ═══════════════ Provider management ═══════════════

    def add_provider(self, provider: MemoryProvider) -> None:
        """Register a provider. Order determines priority."""
        name = provider.name
        if name in self._providers:
            logger.warning("Provider %s already registered, skipping", name)
            return
        self._providers[name] = provider
        self._provider_order.append(name)
        self._rebuild_tool_dispatch()

    def remove_provider(self, name: str) -> None:
        """Remove a provider (for hot-unload or testing)."""
        self._providers.pop(name, None)
        if name in self._provider_order:
            self._provider_order.remove(name)
        self._rebuild_tool_dispatch()

    def get_provider(self, name: str) -> Optional[MemoryProvider]:
        """Get a provider by name."""
        return self._providers.get(name)

    @property
    def providers(self) -> List[MemoryProvider]:
        return list(self._providers.values())

    # ═══════════════ Lifecycle ═══════════════

    async def initialize_all(self, **kwargs: Any) -> None:
        """Initialize all registered providers."""
        for name in self._provider_order:
            try:
                await self._providers[name].initialize(**kwargs)
            except Exception as exc:
                logger.warning("memory.init_failed provider=%s error=%s", name, exc)
        self._initialized = True

    async def shutdown_all(self) -> None:
        """Shutdown all providers gracefully."""
        for name in self._provider_order:
            try:
                await self._providers[name].shutdown()
            except Exception as exc:
                logger.warning("memory.shutdown_failed provider=%s error=%s", name, exc)
        self._initialized = False

    def on_turn_start(self, turn: int, user_message: str) -> None:
        """Notify all providers of a new agent turn."""
        for name in self._provider_order:
            try:
                provider = self._providers[name]
                if hasattr(provider, "on_turn_start"):
                    provider.on_turn_start(turn, user_message)
            except Exception as exc:
                logger.debug("memory.on_turn_start provider=%s error=%s", name, exc)

    # ═══════════════ Core operations ═══════════════

    async def insert(self, entry: MemoryEntry, *, session_id: str = "") -> str:
        """Route entry to accepting providers and insert.

        Routing strategy:
        1. If a provider accepts() the entry → use it
        2. Otherwise fallback to first available provider
        After insert, fire on_inserted hook on all providers.

        When *session_id* is provided, the entry is tagged with that session
        so that session-scoped searches can later isolate it.
        """
        if session_id:
            entry.metadata = {**(entry.metadata or {}), "_session_id": session_id}

        inserted_by: Optional[str] = None
        entry_id = entry.entry_id

        for name in self._provider_order:
            provider = self._providers[name]
            if provider.accepts(entry):
                try:
                    # Pass session_id to providers that support it
                    insert_fn = provider.insert
                    sig = inspect.signature(insert_fn)
                    if "session_id" in sig.parameters:
                        entry_id = await insert_fn(entry, session_id=session_id)
                    else:
                        entry_id = await insert_fn(entry)
                    inserted_by = name
                    break
                except Exception as exc:
                    logger.warning("memory.insert_failed provider=%s error=%s", name, exc)

        # Fallback: insert to first available
        if inserted_by is None and self._providers:
            first = self._providers[self._provider_order[0]]
            try:
                insert_fn = first.insert
                sig = inspect.signature(insert_fn)
                if "session_id" in sig.parameters:
                    entry_id = await insert_fn(entry, session_id=session_id)
                else:
                    entry_id = await insert_fn(entry)
                inserted_by = self._provider_order[0]
            except Exception as exc:
                logger.warning("memory.insert_fallback_failed error=%s", exc)

        # Fire on_inserted lifecycle hook
        if inserted_by:
            self._fire_on_inserted(entry)

        return entry_id

    async def search(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Aggregate search results from all relevant providers.

        If query.cross_domain is True, delegates to search_cross_domain.
        Otherwise, queries all providers and merges by score.
        """
        if query.cross_domain:
            correlated = await self.search_cross_domain(
                " ".join(query.keywords),
                time_window_s=self._cross_domain_window_s,
                limit=query.limit,
            )
            return self._isolate_by_workspace(correlated, query.workspace_root)

        candidates = self._route_search(query)
        all_results: List[MemoryEntry] = []
        for name in candidates:
            try:
                results = await self._providers[name].search(query)
                all_results.extend(results)
            except Exception as exc:
                logger.debug("memory.search_failed provider=%s error=%s", name, exc)

        # Deduplicate by entry_id, sort by score desc
        seen: set = set()
        unique: List[MemoryEntry] = []
        for entry in sorted(all_results, key=lambda e: e.score, reverse=True):
            if entry.entry_id not in seen:
                seen.add(entry.entry_id)
                unique.append(entry)
        # Cross-workspace isolation: a daemon shared across projects must never
        # surface another workspace's tagged memories. Untagged entries are
        # global (user prefs, cross-project facts, legacy) and always pass.
        unique = self._isolate_by_workspace(unique, query.workspace_root)
        return unique[:query.limit]

    async def search_cross_domain(
        self,
        query: str,
        *,
        time_window_s: float = 300.0,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Cross-domain correlation: find related entries across signal domains within a time window.

        Algorithm:
        1. Search all providers with the query keywords
        2. Group results by timestamp clusters (within time_window_s)
        3. Boost entries that co-occur with entries from OTHER domains in the same window
        4. Return merged results sorted by correlation-boosted score

        This enables discovering patterns like: "a file change in domain FS happened
        right after a clipboard event in domain CLIPBOARD" — cross-modal correlation.
        """
        keywords = query.split()[:8] if query else []

        # Step 1: Gather entries from all providers (broad search)
        mq = MemoryQuery(
            keywords=keywords,
            limit=limit * 5,  # Over-fetch for correlation analysis
            min_score=0.0,
        )
        all_entries: List[MemoryEntry] = []
        for name in self._provider_order:
            try:
                results = await self._providers[name].search(mq)
                all_entries.extend(results)
            except Exception:
                pass

        if not all_entries:
            return []

        # Step 2: Deduplicate
        seen: set = set()
        unique: List[MemoryEntry] = []
        for e in all_entries:
            if e.entry_id not in seen:
                seen.add(e.entry_id)
                unique.append(e)

        # Step 3: Compute cross-domain correlation boost
        # For each entry, count how many entries from DIFFERENT domains
        # fall within the time window around it.
        for entry in unique:
            cross_count = 0
            for other in unique:
                if other.entry_id == entry.entry_id:
                    continue
                if other.domain == entry.domain:
                    continue  # Same domain doesn't count
                time_delta = abs(entry.timestamp - other.timestamp)
                if time_delta <= time_window_s:
                    cross_count += 1

            # Boost score: original score * (1 + correlation_factor)
            correlation_factor = min(cross_count * 0.2, 1.0)  # Cap at 2x boost
            entry.score = entry.score * (1.0 + correlation_factor)

        # Step 4: Sort by boosted score and return
        unique.sort(key=lambda e: e.score, reverse=True)
        return unique[:limit]

    async def prefetch(
        self,
        query_text: str,
        *,
        limit: int = 10,
        workspace_root: str = "",
        task_id: str = "",
        scope_keywords: List[str] | None = None,
        session_scope: str = "",
    ) -> List[MemoryEntry]:
        """Quick search for LLM context injection with optional project/task/session scope."""
        keywords = _tokenize_for_search(query_text)
        scope_terms = [term for term in (scope_keywords or []) if term]
        query = MemoryQuery(
            keywords=[*keywords, *scope_terms[:5]],
            limit=max(limit, limit * 3),
            workspace_root=workspace_root,
            task_id=task_id,
            scope_keywords=scope_terms,
            session_scope=session_scope,
        )
        entries = await self.search(query)
        if workspace_root or scope_terms or task_id:
            entries = self._scope_entries(
                entries,
                workspace_root=workspace_root,
                task_id=task_id,
                scope_keywords=scope_terms,
            )
        return entries[:limit]

    def _scope_entries(
        self,
        entries: List[MemoryEntry],
        *,
        workspace_root: str = "",
        task_id: str = "",
        scope_keywords: List[str] | None = None,
    ) -> List[MemoryEntry]:
        """Filter retrieved memories to the active project/task when scope is known."""
        if not entries:
            return []
        root = str(Path(workspace_root).expanduser().resolve()) if workspace_root else ""
        root_name = Path(root).name.lower() if root else ""
        terms = {term.lower() for term in (scope_keywords or []) if len(term) >= 2}
        if root_name:
            terms.add(root_name)
        if task_id:
            terms.add(str(task_id).lower())
        scoped: list[MemoryEntry] = []
        for entry in entries:
            if self._entry_matches_scope(entry, workspace_root=root, scope_terms=terms):
                scoped.append(entry)
        return scoped

    @staticmethod
    def _entry_matches_scope(entry: MemoryEntry, *, workspace_root: str, scope_terms: set[str]) -> bool:
        metadata = entry.metadata or {}
        path_values = [
            metadata.get("path"),
            metadata.get("file_path"),
            metadata.get("workspace_root"),
            metadata.get("project_root"),
            metadata.get("cwd"),
        ]
        has_path_scope = False
        for value in path_values:
            if not value:
                continue
            has_path_scope = True
            try:
                candidate = Path(str(value)).expanduser().resolve()
            except (OSError, RuntimeError, ValueError):
                candidate = Path(str(value))
            if workspace_root:
                try:
                    root_path = Path(workspace_root).expanduser().resolve()
                except (OSError, RuntimeError, ValueError):
                    root_path = Path(workspace_root)
                try:
                    in_workspace = candidate == root_path or candidate.is_relative_to(root_path)
                except (OSError, ValueError):
                    in_workspace = False
                if in_workspace:
                    return True
        if workspace_root and has_path_scope:
            return False
        haystack = " ".join([
            entry.content,
            json.dumps(metadata, ensure_ascii=False, default=str),
        ]).lower()
        return bool(scope_terms and any(term in haystack for term in scope_terms))

    @staticmethod
    def _normalize_workspace(root: str) -> str:
        """Resolve a workspace root to a canonical absolute path string."""
        if not root:
            return ""
        try:
            return str(Path(str(root)).expanduser().resolve())
        except (OSError, RuntimeError, ValueError):
            return str(root)

    @classmethod
    def _entry_workspace_tag(cls, entry: MemoryEntry) -> str:
        """Return the authoritative workspace tag on an entry, if any."""
        metadata = entry.metadata or {}
        raw = metadata.get("workspace_root") or metadata.get("project_root") or ""
        return cls._normalize_workspace(str(raw)) if raw else ""

    @classmethod
    def _isolate_by_workspace(
        cls, entries: List[MemoryEntry], workspace_root: str
    ) -> List[MemoryEntry]:
        """Drop entries whose workspace tag belongs to a different workspace.

        Entries without a workspace tag are treated as global (user prefs,
        cross-project facts, legacy records) and always pass. This enforces
        per-workspace isolation on a daemon shared across concurrent projects
        without hiding genuinely global knowledge.
        """
        root = cls._normalize_workspace(workspace_root)
        if not root:
            return entries
        kept: List[MemoryEntry] = []
        for entry in entries:
            tag = cls._entry_workspace_tag(entry)
            if not tag or tag == root:
                kept.append(entry)
        return kept

    @classmethod
    def tag_entry_workspace(cls, entry: MemoryEntry, workspace_root: str) -> MemoryEntry:
        """Tag an entry with the active workspace unless it already carries one."""
        root = cls._normalize_workspace(workspace_root)
        if root and not (entry.metadata or {}).get("workspace_root"):
            entry.metadata = {**(entry.metadata or {}), "workspace_root": root}
        return entry

    async def sync_turn(
        self, messages: List[Dict[str, Any]], *, workspace_root: str = "", session_id: str = ""
    ) -> None:
        """Background sync of conversation turn (fire-and-forget safe)."""
        # Extract last assistant message for storage
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                entry = MemoryEntry(
                    kind=MemoryKind.CONVERSATION,
                    domain=SignalDomain.SYSTEM,
                    content=msg["content"][:500],
                )
                self.tag_entry_workspace(entry, workspace_root)
                await self.insert(entry, session_id=session_id)
                break

    # ═══════════════ Promotion ═══════════════

    async def promote(
        self, entry: MemoryEntry, *, from_provider: str, to_provider: str
    ) -> None:
        """Cross-provider promotion with lifecycle hook.

        Moves/copies an entry from one provider tier to another (e.g. episodic → semantic).
        Fires on_promoted on the target provider.
        """
        target = self._providers.get(to_provider)
        if target is None:
            logger.warning("Promote target %s not found", to_provider)
            return
        try:
            if hasattr(target, "on_promoted"):
                target.on_promoted(entry, from_provider)
            await target.insert(entry)
            logger.debug(
                "memory.promoted entry=%s from=%s to=%s",
                entry.entry_id, from_provider, to_provider,
            )
        except Exception as exc:
            logger.warning("memory.promote_failed error=%s", exc)

    # ═══════════════ Tool interface ═══════════════

    def get_tool_schemas(self) -> List[MemoryToolSchema]:
        """Generate LLM tool schemas for memory operations.

        Collects schemas from all providers that implement get_tool_schemas(),
        plus adds the manager-level memory_search and memory_add tools.
        """
        schemas: List[MemoryToolSchema] = []

        # Manager-level tools (always available)
        schemas.append(MemoryToolSchema(
            name="memory_search",
            description="Search agent memory for relevant past experiences, observations, and facts.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords"},
                    "domain": {
                        "type": "string",
                        "enum": [d.value for d in SignalDomain],
                        "description": "Filter by signal domain",
                    },
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
            provider_name="__manager__",
        ))
        schemas.append(MemoryToolSchema(
            name="memory_add",
            description="Store a new observation or insight in memory for future reference.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "What to remember"},
                    "kind": {
                        "type": "string",
                        "enum": [k.value for k in MemoryKind if k not in (
                            MemoryKind.SKILL_EPISODE, MemoryKind.SKILL_PATTERN,
                        )],
                    },
                    "domain": {
                        "type": "string",
                        "enum": [d.value for d in SignalDomain],
                    },
                },
                "required": ["content"],
            },
            provider_name="__manager__",
        ))

        # Collect from providers
        for name in self._provider_order:
            provider = self._providers[name]
            if hasattr(provider, "get_tool_schemas"):
                try:
                    provider_schemas = provider.get_tool_schemas()
                    if provider_schemas:
                        schemas.extend(provider_schemas)
                except Exception:
                    pass

        return schemas

    def get_openai_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas in OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.parameters,
                },
            }
            for s in self.get_tool_schemas()
        ]

    async def handle_tool_call(
        self, tool_name: str, args: Dict[str, Any], *, workspace_root: str = ""
    ) -> str:
        """Route LLM tool calls to the appropriate handler.

        Manager-level tools (memory_search, memory_add) are handled directly.
        Provider-specific tools are dispatched to the owning provider.
        ``workspace_root`` scopes reads and tags writes to the active project
        so a shared daemon never mixes memories across workspaces.
        """
        # Manager-level tools
        if tool_name == "memory_search":
            return await self._handle_memory_search(args, workspace_root=workspace_root)
        if tool_name == "memory_add":
            return await self._handle_memory_add(args, workspace_root=workspace_root)

        # Provider-specific tools
        provider_name = self._tool_dispatch.get(tool_name)
        if provider_name:
            provider = self._providers.get(provider_name)
            if provider and hasattr(provider, "handle_tool_call"):
                try:
                    return provider.handle_tool_call(tool_name, args)
                except Exception as exc:
                    return json.dumps({"error": str(exc)})

        return json.dumps({"error": f"Unknown memory tool: {tool_name}"})

    # ═══════════════ Internal routing ═══════════════

    def _route_search(self, query: MemoryQuery) -> List[str]:
        """Determine which providers are candidates for a query."""
        if not query.kinds:
            return list(self._provider_order)

        candidates: set = set()
        for kind in query.kinds:
            if kind in (MemoryKind.SKILL_EPISODE, MemoryKind.SKILL_PATTERN):
                candidates.add("evolution")
            elif kind == MemoryKind.CONVERSATION:
                candidates.add("working")
            elif kind == MemoryKind.EVENT:
                candidates.add("episodic")
                candidates.add("working")
            else:
                candidates.add("semantic")
        # Semantic is always a fallback
        candidates.add("semantic")
        return [n for n in self._provider_order if n in candidates]

    def _fire_on_inserted(self, entry: MemoryEntry) -> None:
        """Fire on_inserted lifecycle hook on all providers."""
        for name in self._provider_order:
            try:
                provider = self._providers[name]
                if hasattr(provider, "on_inserted"):
                    provider.on_inserted(entry)
            except Exception:
                pass

    def _rebuild_tool_dispatch(self) -> None:
        """Rebuild the tool_name → provider_name mapping."""
        self._tool_dispatch.clear()
        for name in self._provider_order:
            provider = self._providers[name]
            if hasattr(provider, "get_tool_schemas"):
                try:
                    schemas = provider.get_tool_schemas()
                    if schemas:
                        for s in schemas:
                            self._tool_dispatch[s.name] = name
                except Exception:
                    pass

    async def _handle_memory_search(
        self, args: Dict[str, Any], *, workspace_root: str = ""
    ) -> str:
        """Handle manager-level memory_search tool call."""
        query_text = args.get("query", "")
        domain_str = args.get("domain")
        limit = int(args.get("limit", 10))

        domains = [SignalDomain(domain_str)] if domain_str else None
        mq = MemoryQuery(
            keywords=_tokenize_for_search(query_text),
            domains=domains,
            limit=limit,
            workspace_root=workspace_root,
        )
        results = await self.search(mq)
        return json.dumps({
            "results": [
                {
                    "content": e.content[:200],
                    "kind": e.kind.value,
                    "domain": e.domain.value,
                    "score": round(e.score, 3),
                }
                for e in results
            ]
        }, ensure_ascii=False)

    async def _handle_memory_add(
        self, args: Dict[str, Any], *, workspace_root: str = ""
    ) -> str:
        """Handle manager-level memory_add tool call."""
        content = args.get("content", "")
        if not content:
            return json.dumps({"error": "content is required"})

        kind_str = args.get("kind", "observation")
        domain_str = args.get("domain")

        try:
            kind = MemoryKind(kind_str)
        except ValueError:
            kind = MemoryKind.OBSERVATION

        domain = SignalDomain(domain_str) if domain_str else SignalDomain.SYSTEM

        entry = MemoryEntry(
            kind=kind,
            domain=domain,
            content=content[:2200],  # Safety cap per design doc
        )
        self.tag_entry_workspace(entry, workspace_root)
        entry_id = await self.insert(entry)
        return json.dumps({"success": True, "id": entry_id})
