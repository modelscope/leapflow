"""DuckDB-indexed cache manager for profile/workspace/session scopes."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import duckdb

from leapflow.layout import CacheLayout


class CacheScope(str, Enum):
    """Supported cache isolation scopes."""

    PROFILE = "profile"
    WORKSPACE = "workspace"
    SESSION = "session"


@dataclass(frozen=True)
class CacheEntry:
    """Metadata for one cached file or object."""

    entry_id: str
    profile_id: str
    scope: str
    category: str
    source: str
    path: Path
    size_bytes: int
    workspace_id: str = ""
    session_id: str = ""
    content_hash: str = ""
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float | None = None
    expires_at: float | None = None
    sensitive: bool = False
    syncable: bool = True
    owner_component: str = "cache"
    metadata: dict[str, Any] = field(default_factory=dict)


class CacheManager:
    """Manage cache paths, index entries, and cleanup policies."""

    def __init__(self, layout: CacheLayout, *, profile_id: str) -> None:
        self._layout = layout
        self._profile_id = profile_id
        self._layout.ensure()
        self._init_schema()

    @property
    def index_path(self) -> Path:
        return self._layout.index_path

    def path(
        self,
        *,
        scope: CacheScope | str,
        category: str,
        workspace_id: str = "",
        session_id: str = "",
        source: str = "default",
    ) -> Path:
        scope_value = scope.value if isinstance(scope, CacheScope) else str(scope)
        cache_path = self._layout.category_dir(
            scope=scope_value,
            category=category,
            workspace_id=workspace_id,
            session_id=session_id,
        ) / source
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    def register(
        self,
        *,
        path: Path,
        scope: CacheScope | str,
        category: str,
        source: str,
        workspace_id: str = "",
        session_id: str = "",
        content_hash: str = "",
        expires_at: float | None = None,
        sensitive: bool = False,
        syncable: bool = True,
        owner_component: str = "cache",
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        entry = self._build_entry(
            path=path,
            scope=scope,
            category=category,
            source=source,
            workspace_id=workspace_id,
            session_id=session_id,
            content_hash=content_hash,
            expires_at=expires_at,
            sensitive=sensitive,
            syncable=syncable,
            owner_component=owner_component,
            metadata=metadata,
        )
        self.register_many((entry,))
        return entry

    def register_many(self, entries: Iterable[CacheEntry]) -> tuple[CacheEntry, ...]:
        """Register multiple cache entries using one DuckDB connection."""
        items = tuple(entries)
        if not items:
            return ()
        with self._connect() as con:
            con.executemany(
                """
                INSERT OR REPLACE INTO cache_entry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [_entry_params(entry) for entry in items],
            )
        return items

    def register_directory(
        self,
        *,
        root: Path,
        scope: CacheScope | str,
        category: str,
        source: str,
        workspace_id: str = "",
        session_id: str = "",
        expires_at: float | None = None,
        sensitive: bool = False,
        syncable: bool = True,
        owner_component: str = "cache",
        metadata: dict[str, Any] | None = None,
        suffixes: Iterable[str] | None = None,
    ) -> list[CacheEntry]:
        suffix_filter = {suffix.lower() for suffix in suffixes or ()}
        entries: list[CacheEntry] = []
        if not root.exists():
            return entries
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if suffix_filter and path.suffix.lower() not in suffix_filter:
                continue
            entries.append(self._build_entry(
                path=path,
                scope=scope,
                category=category,
                source=source,
                workspace_id=workspace_id,
                session_id=session_id,
                expires_at=expires_at,
                sensitive=sensitive,
                syncable=syncable,
                owner_component=owner_component,
                metadata=metadata,
            ))
        self.register_many(entries)
        return entries

    def cleanup_quota(
        self,
        *,
        scope: str,
        max_bytes: int,
        category: str | None = None,
        workspace_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        entries = self.list_entries(
            scope=scope,
            category=category,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        total_size = sum(entry.size_bytes for entry in entries)
        if total_size <= max_bytes:
            return 0
        removed = 0
        to_delete_ids: list[str] = []
        for entry in sorted(entries, key=lambda item: item.last_accessed_at or item.created_at):
            if total_size <= max_bytes:
                break
            if self._is_managed_path(entry.path):
                entry.path.unlink(missing_ok=True)
            total_size -= entry.size_bytes
            to_delete_ids.append(entry.entry_id)
            removed += 1
        if to_delete_ids:
            with self._connect() as con:
                con.executemany(
                    "DELETE FROM cache_entry WHERE id = ?",
                    [(entry_id,) for entry_id in to_delete_ids],
                )
        return removed

    def touch(self, entry_id: str) -> None:
        with self._connect() as con:
            con.execute(
                "UPDATE cache_entry SET last_accessed_at = ? WHERE id = ?",
                [time.time(), entry_id],
            )

    def cleanup_expired(self, *, now: float | None = None) -> int:
        cutoff = now or time.time()
        removed = 0
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, path FROM cache_entry WHERE expires_at IS NOT NULL AND expires_at <= ?",
                [cutoff],
            ).fetchall()
            for entry_id, raw_path in rows:
                path = Path(str(raw_path))
                if self._is_managed_path(path):
                    path.unlink(missing_ok=True)
                con.execute("DELETE FROM cache_entry WHERE id = ?", [entry_id])
                removed += 1
        return removed

    def list_entries(
        self,
        *,
        scope: str | None = None,
        category: str | None = None,
        workspace_id: str | None = None,
        session_id: str | None = None,
    ) -> list[CacheEntry]:
        clauses = ["profile_id = ?"]
        params: list[Any] = [self._profile_id]
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        if workspace_id is not None:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        where = " AND ".join(clauses)
        with self._connect() as con:
            rows = con.execute(f"SELECT * FROM cache_entry WHERE {where}", params).fetchall()
        return [_entry_from_row(row) for row in rows]

    def _init_schema(self) -> None:
        self._layout.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entry (
                    id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    workspace_id TEXT,
                    session_id TEXT,
                    scope TEXT NOT NULL,
                    category TEXT NOT NULL,
                    source TEXT NOT NULL,
                    path TEXT NOT NULL,
                    content_hash TEXT,
                    size_bytes BIGINT NOT NULL,
                    created_at DOUBLE NOT NULL,
                    last_accessed_at DOUBLE,
                    expires_at DOUBLE,
                    sensitive BOOLEAN NOT NULL,
                    syncable BOOLEAN NOT NULL,
                    owner_component TEXT NOT NULL,
                    metadata_json TEXT
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_cache_scope ON cache_entry(profile_id, scope)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_cache_expiry ON cache_entry(expires_at)")

    def _build_entry(
        self,
        *,
        path: Path,
        scope: CacheScope | str,
        category: str,
        source: str,
        workspace_id: str = "",
        session_id: str = "",
        content_hash: str = "",
        expires_at: float | None = None,
        sensitive: bool = False,
        syncable: bool = True,
        owner_component: str = "cache",
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        scope_value = scope.value if isinstance(scope, CacheScope) else str(scope)
        resolved = path.expanduser().resolve()
        size = resolved.stat().st_size if resolved.exists() and resolved.is_file() else 0
        entry_id = content_hash or _entry_id(resolved)
        now = time.time()
        return CacheEntry(
            entry_id=entry_id,
            profile_id=self._profile_id,
            workspace_id=workspace_id,
            session_id=session_id,
            scope=scope_value,
            category=category,
            source=source,
            path=resolved,
            content_hash=content_hash,
            size_bytes=size,
            created_at=now,
            last_accessed_at=now,
            expires_at=expires_at,
            sensitive=sensitive,
            syncable=syncable,
            owner_component=owner_component,
            metadata=dict(metadata or {}),
        )

    def _connect(self):
        return duckdb.connect(str(self._layout.index_path))

    def _is_managed_path(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self._layout.root.resolve())
            return True
        except ValueError:
            return False


def _entry_id(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()


def _entry_params(entry: CacheEntry) -> list[Any]:
    return [
        entry.entry_id,
        entry.profile_id,
        entry.workspace_id,
        entry.session_id,
        entry.scope,
        entry.category,
        entry.source,
        str(entry.path),
        entry.content_hash,
        entry.size_bytes,
        entry.created_at,
        entry.last_accessed_at,
        entry.expires_at,
        entry.sensitive,
        entry.syncable,
        entry.owner_component,
        json.dumps(entry.metadata, ensure_ascii=False),
    ]


def _entry_from_row(row: tuple[Any, ...]) -> CacheEntry:
    return CacheEntry(
        entry_id=str(row[0]),
        profile_id=str(row[1]),
        workspace_id=str(row[2] or ""),
        session_id=str(row[3] or ""),
        scope=str(row[4]),
        category=str(row[5]),
        source=str(row[6]),
        path=Path(str(row[7])),
        content_hash=str(row[8] or ""),
        size_bytes=int(row[9] or 0),
        created_at=float(row[10] or 0.0),
        last_accessed_at=row[11],
        expires_at=row[12],
        sensitive=bool(row[13]),
        syncable=bool(row[14]),
        owner_component=str(row[15] or "cache"),
        metadata=json.loads(row[16] or "{}"),
    )
