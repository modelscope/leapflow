"""Semantic memory provider — DuckDB-backed persistent storage with domain support.

Serves as the long-term knowledge store. Accepts all memory kinds as the
final persistence tier. Supports structured keyword queries with decay-weighted
scoring and signal-domain filtering.

Provides legacy-compatible methods (insert by kind/content, search_keywords,
recent_file_events, etc.) for upstream consumers (Engine, ExperienceStore).
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import duckdb

from leapflow.memory.protocol import (
    MemoryEntry,
    MemoryKind,
    MemoryQuery,
    SignalDomain,
)
from leapflow.storage.connection import ConnectionHolder

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Inline decay formula
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


# ──────────────────────────────────────────────────────────────────────
# Legacy-compatible MemoryHit dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MemoryHit:
    """Retrieved memory row with a decay-weighted score."""

    memory_id: str
    kind: str
    content: str
    path: Optional[str]
    metadata: Dict[str, Any]
    created_at: float
    score: float


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _parse_metadata(raw: Any) -> Dict[str, Any]:
    """Safely parse metadata from JSON string or dict."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
            return dict(loaded) if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


# ──────────────────────────────────────────────────────────────────────
# Provider
# ──────────────────────────────────────────────────────────────────────

class SemanticMemoryProvider:
    """DuckDB-backed persistent memory with signal-domain indexing.

    Accepts all MemoryKinds — acts as the universal long-term store.
    """

    def __init__(self, source: Union[ConnectionHolder, Path]) -> None:
        self._owns_holder = isinstance(source, Path)
        self._source = source
        self._con: Optional[duckdb.DuckDBPyConnection] = None

    # ── Protocol properties ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return "semantic"

    # ── Protocol methods ──────────────────────────────────────────────

    async def initialize(self, **kwargs: Any) -> None:
        """Open DB connection, ensure schema exists, migrate if needed."""
        if self._owns_holder:
            from leapflow.storage.duckdb_connect import connect as safe_connect
            self._con = safe_connect(self._source)  # type: ignore[arg-type]
        else:
            self._con = self._source.connection  # type: ignore[union-attr]
        self._init_schema()

    async def shutdown(self) -> None:
        """Close DB connection only if we own it."""
        if self._owns_holder and self._con:
            self._con.close()
        self._con = None

    def accepts(self, entry: MemoryEntry) -> bool:
        # Semantic tier accepts everything — it's the final store
        return True

    async def insert(self, entry: MemoryEntry) -> str:
        """Persist an entry to DuckDB. Returns entry_id."""
        from leapflow.storage.write_buffer import execute_with_retry
        con = self._connection()
        now = time.time()
        meta_json = json.dumps(entry.metadata, ensure_ascii=False)
        path = entry.metadata.get("path")
        execute_with_retry(
            con,
            """
            INSERT INTO leap_memory (id, kind, domain, content, path, metadata, created_at, accessed_at, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                entry.entry_id,
                entry.kind.value,
                entry.domain.value,
                entry.content,
                path,
                meta_json,
                entry.timestamp,
                now,
                1,
            ],
        )
        return entry.entry_id

    async def search(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Structured keyword search with domain/kind/time filtering and decay scoring."""
        con = self._connection()
        now = time.time()

        # Build WHERE clauses dynamically
        conditions: List[str] = []
        params: List[Any] = []

        # Keyword filter (AND semantics)
        keywords = [k.strip() for k in query.keywords if k.strip()] if query.keywords else []
        if keywords:
            for _ in keywords:
                conditions.append("content ILIKE ?")
                params.append(f"%{_}%")

        # Kind filter
        if query.kinds:
            placeholders = ",".join(["?"] * len(query.kinds))
            conditions.append(f"kind IN ({placeholders})")
            params.extend(k.value for k in query.kinds)

        # Domain filter
        if query.domains:
            placeholders = ",".join(["?"] * len(query.domains))
            conditions.append(f"domain IN ({placeholders})")
            params.extend(d.value for d in query.domains)

        # Time range filter
        if query.time_range:
            t_min, t_max = query.time_range
            conditions.append("created_at >= ? AND created_at <= ?")
            params.extend([t_min, t_max])

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""
            SELECT id, kind, domain, content, metadata, created_at, accessed_at, access_count
            FROM leap_memory
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT {int(query.limit * 3)}
        """

        rows = con.execute(sql, params).fetchall()

        # Score and filter
        n_kws = len(keywords) if keywords else 1
        entries: List[MemoryEntry] = []
        for rid, kind, domain, content, metadata, created_at, accessed_at, access_count in rows:
            age = max(0.0, now - float(created_at))
            freq = float(max(1, int(access_count)))

            # Compute semantic weight from keyword overlap
            if keywords:
                content_lower = str(content).lower()
                matched = sum(1 for k in keywords if k.lower() in content_lower)
                semantic = matched / n_kws
            else:
                semantic = 1.0

            score = _decay_score(semantic, age, freq)
            if score < query.min_score:
                continue

            meta = _parse_metadata(metadata)
            entry = MemoryEntry(
                entry_id=str(rid),
                kind=MemoryKind(kind),
                domain=SignalDomain(domain) if domain else SignalDomain.SYSTEM,
                content=str(content),
                timestamp=float(created_at),
                score=score,
                metadata=meta,
            )
            entries.append(entry)

        entries.sort(key=lambda e: e.score, reverse=True)
        return entries[: query.limit]

    async def delete(self, entry_id: str) -> bool:
        """Remove an entry by ID."""
        from leapflow.storage.write_buffer import execute_with_retry
        con = self._connection()
        count = con.execute(
            "SELECT COUNT(*) FROM leap_memory WHERE id = ?", [entry_id]
        ).fetchone()[0]
        if count > 0:
            execute_with_retry(
                con, "DELETE FROM leap_memory WHERE id = ?", [entry_id]
            )
            return True
        return False

    # ── Additional public methods ─────────────────────────────────────

    def touch(self, entry_id: str) -> None:
        """Increment access count and update accessed_at timestamp."""
        from leapflow.storage.write_buffer import execute_with_retry
        con = self._connection()
        now = time.time()
        execute_with_retry(
            con,
            "UPDATE leap_memory SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
            [now, entry_id],
        )

    def prune(
        self,
        *,
        max_age_days: float = 90.0,
        min_score: float = 0.001,
        protected_kinds: Sequence[str] = ("skill_episode", "prediction", "skill", "prediction_experience"),
    ) -> int:
        """Remove old, low-value rows. Protected kinds are exempt."""
        from leapflow.storage.write_buffer import execute_with_retry
        con = self._connection()
        cutoff = time.time() - max_age_days * 86400
        placeholders = ",".join(["?"] * len(protected_kinds))
        count_row = con.execute(
            f"""
            SELECT COUNT(*) FROM leap_memory
            WHERE created_at < ?
              AND access_count <= 1
              AND kind NOT IN ({placeholders})
            """,
            [cutoff] + list(protected_kinds),
        ).fetchone()
        deleted = count_row[0] if count_row else 0
        if deleted > 0:
            execute_with_retry(
                con,
                f"""
                DELETE FROM leap_memory
                WHERE created_at < ?
                  AND access_count <= 1
                  AND kind NOT IN ({placeholders})
                """,
                [cutoff] + list(protected_kinds),
            )
            logger.info("semantic.prune deleted=%d cutoff_days=%.0f", deleted, max_age_days)
        return deleted

    def close(self) -> None:
        """Close DB connection (sync convenience)."""
        if self._con:
            self._con.close()
            self._con = None

    # ── Legacy interface (Engine, ExperienceStore, Skills) ────────────

    def insert_raw(
        self,
        kind: str,
        content: str,
        *,
        path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """Insert a row using the legacy (kind, content) signature."""
        from leapflow.storage.write_buffer import execute_with_retry
        con = self._ensure_connection()
        mid = memory_id or str(uuid.uuid4())
        now = time.time()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        domain = SignalDomain.FILESYSTEM.value if "file" in kind else SignalDomain.SYSTEM.value
        execute_with_retry(
            con,
            """
            INSERT INTO leap_memory (id, kind, domain, content, path, metadata, created_at, accessed_at, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [mid, kind, domain, content, path, meta_json, now, now, 1],
        )
        return mid

    def upsert_raw(
        self,
        kind: str,
        content: str,
        *,
        path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """Insert or update a row by primary key.

        Useful for singleton state blobs (e.g. Markov state) that must
        be overwritten on each session rather than duplicated.
        """
        from leapflow.storage.write_buffer import execute_with_retry
        con = self._ensure_connection()
        mid = memory_id or str(uuid.uuid4())
        now = time.time()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        domain = SignalDomain.FILESYSTEM.value if "file" in kind else SignalDomain.SYSTEM.value
        execute_with_retry(con, "DELETE FROM leap_memory WHERE id = ?", [mid])
        execute_with_retry(
            con,
            """
            INSERT INTO leap_memory (id, kind, domain, content, path, metadata, created_at, accessed_at, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [mid, kind, domain, content, path, meta_json, now, now, 1],
        )
        return mid

    def search_keywords(
        self,
        keywords: Sequence[str],
        *,
        kinds: Optional[Iterable[str]] = None,
        limit: int = 20,
    ) -> List[MemoryHit]:
        """Keyword search returning MemoryHit objects (legacy API)."""
        con = self._ensure_connection()
        now = time.time()
        kws = [k.strip() for k in keywords if k.strip()]
        if not kws:
            return []

        like_clauses = " AND ".join(["content ILIKE ?" for _ in kws])
        where_kinds = ""
        kind_params: List[Any] = []
        if kinds:
            ks = list(kinds)
            placeholders = ",".join(["?"] * len(ks))
            where_kinds = f" AND kind IN ({placeholders})"
            kind_params = list(ks)

        params: List[Any] = [f"%{k}%" for k in kws] + kind_params
        rows = con.execute(
            f"""
            SELECT id, kind, content, path, metadata, created_at, accessed_at, access_count
            FROM leap_memory
            WHERE {like_clauses}{where_kinds}
            ORDER BY created_at DESC
            LIMIT {int(limit)}
            """,
            params,
        ).fetchall()

        hits: List[MemoryHit] = []
        n_kws = len(kws)
        for rid, k, content, path, meta_raw, created_at, accessed_at, access_count in rows:
            meta = _parse_metadata(meta_raw)
            age = max(0.0, now - float(created_at))
            content_lower = str(content).lower()
            matched = sum(1 for kw in kws if kw.lower() in content_lower)
            semantic = matched / n_kws if n_kws else 1.0
            freq = float(max(1, int(access_count)))
            s = _decay_score(semantic, age, freq)
            hits.append(
                MemoryHit(
                    memory_id=str(rid),
                    kind=str(k),
                    content=str(content),
                    path=str(path) if path else None,
                    metadata=meta,
                    created_at=float(created_at),
                    score=s,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    def recent_file_events(self, within_seconds: float = 3600.0) -> List[MemoryHit]:
        """Return recent file-event rows."""
        con = self._ensure_connection()
        now = time.time()
        since = now - float(within_seconds)
        rows = con.execute(
            """
            SELECT id, kind, content, path, metadata, created_at, accessed_at, access_count
            FROM leap_memory
            WHERE kind IN ('file_event', 'file_change') AND created_at >= ?
            ORDER BY created_at DESC
            LIMIT 200
            """,
            [since],
        ).fetchall()
        hits: List[MemoryHit] = []
        for rid, k, content, path, meta_raw, created_at, accessed_at, access_count in rows:
            meta = _parse_metadata(meta_raw)
            age = max(0.0, now - float(created_at))
            freq = float(max(1, int(access_count)))
            hits.append(
                MemoryHit(
                    memory_id=str(rid),
                    kind=str(k),
                    content=str(content),
                    path=str(path) if path else None,
                    metadata=meta,
                    created_at=float(created_at),
                    score=_decay_score(1.0, age, freq),
                )
            )
        return hits

    def get_by_id(self, memory_id: str) -> Optional[MemoryHit]:
        """Fetch a single row by primary key."""
        con = self._ensure_connection()
        row = con.execute(
            """
            SELECT id, kind, content, path, metadata, created_at, accessed_at, access_count
            FROM leap_memory WHERE id = ?
            """,
            [memory_id],
        ).fetchone()
        if not row:
            return None
        rid, k, content, path, meta_raw, created_at, accessed_at, access_count = row
        meta = _parse_metadata(meta_raw)
        now = time.time()
        age = max(0.0, now - float(created_at))
        freq = float(max(1, int(access_count)))
        return MemoryHit(
            memory_id=str(rid),
            kind=str(k),
            content=str(content),
            path=str(path) if path else None,
            metadata=meta,
            created_at=float(created_at),
            score=_decay_score(1.0, age, freq),
        )

    def query_by_kind(
        self,
        kind: str,
        *,
        where_metadata: Optional[str] = None,
        order_by: str = "created_at DESC",
        limit: int = 100,
    ) -> List[MemoryHit]:
        """Query rows by kind with optional metadata filter."""
        con = self._ensure_connection()
        now = time.time()
        sql = (
            "SELECT id, kind, content, path, metadata, created_at, accessed_at, access_count "
            "FROM leap_memory WHERE kind = ?"
        )
        params: List[Any] = [kind]
        if where_metadata:
            sql += f" AND {where_metadata}"
        sql += f" ORDER BY {order_by} LIMIT {int(limit)}"
        rows = con.execute(sql, params).fetchall()
        hits: List[MemoryHit] = []
        for rid, k, content, path, meta_raw, created_at, accessed_at, access_count in rows:
            meta = _parse_metadata(meta_raw)
            age = max(0.0, now - float(created_at))
            freq = float(max(1, int(access_count)))
            hits.append(
                MemoryHit(
                    memory_id=str(rid), kind=str(k), content=str(content),
                    path=str(path) if path else None, metadata=meta,
                    created_at=float(created_at),
                    score=_decay_score(1.0, age, freq),
                )
            )
        return hits

    def count_by_kind(self, kind: str) -> int:
        """Count rows of a specific kind."""
        con = self._ensure_connection()
        result = con.execute(
            "SELECT COUNT(*) FROM leap_memory WHERE kind = ?", [kind]
        ).fetchone()
        return int(result[0]) if result else 0

    def update_metadata(self, memory_id: str, metadata: Dict[str, Any]) -> None:
        """Update the metadata JSON for a given row."""
        from leapflow.storage.write_buffer import execute_with_retry
        con = self._ensure_connection()
        meta_json = json.dumps(metadata, ensure_ascii=False)
        execute_with_retry(
            con,
            "UPDATE leap_memory SET metadata = ? WHERE id = ?",
            [meta_json, memory_id],
        )

    # ── Lifecycle hooks ────────────────────────────────────────────────

    def on_turn_start(self, turn: int, user_message: str) -> None:
        """No-op for semantic provider."""

    def on_inserted(self, entry: MemoryEntry) -> None:
        """No-op for semantic provider."""

    def on_accessed(self, entry: MemoryEntry) -> None:
        """Increment access count in DB for the entry."""
        if self._con is not None:
            try:
                self.touch(entry.entry_id)
            except Exception:
                pass

    def on_promoted(self, entry: MemoryEntry, source_provider: str) -> None:
        """Accept promoted entries from episodic tier.

        Converts EVENT kind to OBSERVATION for long-term storage.
        """
        if entry.kind == MemoryKind.EVENT:
            entry = MemoryEntry(
                entry_id=entry.entry_id,
                kind=MemoryKind.OBSERVATION,
                domain=entry.domain,
                content=entry.content,
                timestamp=entry.timestamp,
                score=entry.score,
                metadata=entry.metadata,
                access_count=entry.access_count,
            )
        try:
            from leapflow.storage.write_buffer import execute_with_retry
            con = self._ensure_connection()
            meta_json = json.dumps(entry.metadata, ensure_ascii=False)
            execute_with_retry(
                con,
                """
                INSERT INTO leap_memory (id, kind, domain, content, path, metadata, created_at, accessed_at, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    entry.entry_id,
                    entry.kind.value,
                    entry.domain.value,
                    entry.content,
                    entry.metadata.get("path"),
                    meta_json,
                    entry.timestamp,
                    time.time(),
                    entry.access_count,
                ],
            )
            logger.debug("semantic.on_promoted from=%s id=%s", source_provider, entry.entry_id)
        except Exception as exc:
            logger.debug("semantic.on_promoted failed: %s", exc)

    def get_tool_schemas(self) -> list:
        """Expose memory_search and memory_add tools to LLM."""
        from leapflow.memory.protocol import MemoryToolSchema
        return [
            MemoryToolSchema(
                name="memory_search_semantic",
                description="Search long-term memory by keywords, domain, or kind.",
                parameters={
                    "type": "object",
                    "properties": {
                        "keywords": {"type": "string", "description": "Search keywords"},
                        "domain": {"type": "string", "enum": [d.value for d in SignalDomain]},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["keywords"],
                },
                provider_name="semantic",
            ),
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Handle LLM tool call for semantic search."""
        if tool_name == "memory_search_semantic":
            import asyncio
            keywords = args.get("keywords", "").split()
            domain_str = args.get("domain")
            limit = int(args.get("limit", 10))
            domains = [SignalDomain(domain_str)] if domain_str else None
            mq = MemoryQuery(keywords=keywords, domains=domains, limit=limit)
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    results = pool.submit(asyncio.run, self.search(mq)).result()
            except RuntimeError:
                results = asyncio.run(self.search(mq))
            return json.dumps({
                "results": [
                    {"content": e.content[:200], "kind": e.kind.value, "score": round(e.score, 3)}
                    for e in results
                ],
            }, ensure_ascii=False)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # ── Internal ──────────────────────────────────────────────────────

    def _connection(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            raise RuntimeError("SemanticMemoryProvider not initialized — call initialize() first")
        return self._con

    def _ensure_connection(self) -> duckdb.DuckDBPyConnection:
        """Ensure DB is connected, auto-initializing if needed (for legacy callers)."""
        if self._con is None:
            if self._owns_holder:
                from leapflow.storage.duckdb_connect import connect as safe_connect
                self._con = safe_connect(self._source)  # type: ignore[arg-type]
            else:
                self._con = self._source.connection  # type: ignore[union-attr]
            self._init_schema()
        return self._con

    def _init_schema(self) -> None:
        """Create or migrate the schema with domain + path columns."""
        con = self._connection()
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS leap_memory (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                domain TEXT NOT NULL DEFAULT 'system',
                content TEXT NOT NULL,
                path TEXT,
                metadata TEXT,
                created_at DOUBLE NOT NULL,
                accessed_at DOUBLE NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 1
            );
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_lm_created ON leap_memory(created_at);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_lm_kind ON leap_memory(kind);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_lm_domain ON leap_memory(domain);")

        # Migrations: add columns if table existed without them
        for col in ["domain", "path"]:
            try:
                con.execute(f"ALTER TABLE leap_memory ADD COLUMN {col} TEXT DEFAULT 'system'" if col == "domain" else f"ALTER TABLE leap_memory ADD COLUMN {col} TEXT")
            except duckdb.CatalogException:
                pass  # Column already exists
