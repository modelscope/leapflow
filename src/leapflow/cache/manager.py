# Copyright (c) Alibaba, Inc. and its affiliates.
"""DuckDB-indexed cache manager for profile/workspace/session scopes."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import duckdb

from leapflow.layout import CacheLayout

logger = logging.getLogger(__name__)

# Per-process state keyed by index file, shared by every CacheManager instance on it.
# Callers such as the web cache build a manager per call, so per-instance state would
# neither serialize their writes nor spare them the schema DDL on every construction.
_REGISTRY_LOCK = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}
_SCHEMA_READY: set[tuple[str, int]] = set()

# Secondary indexes created by earlier releases and dropped on open. On a table of a
# few thousand rows a scan costs well under a millisecond, while an ART index turns
# every update of its columns into a delete plus insert that leaves a dead row group
# behind at each checkpoint -- the index grew to hundreds of megabytes that way.
_LEGACY_INDEXES = ("idx_cache_expiry", "idx_cache_scope")

# Column order of ``cache_entry``; ``_entry_params`` produces values in this order.
_COLUMNS = (
    "id",
    "profile_id",
    "workspace_id",
    "session_id",
    "scope",
    "category",
    "source",
    "path",
    "content_hash",
    "size_bytes",
    "created_at",
    "last_accessed_at",
    "expires_at",
    "sensitive",
    "syncable",
    "owner_component",
    "metadata_json",
)
# Columns refreshed when an already-indexed entry is registered again. ``id`` is the
# key and ``created_at`` is when the artifact first appeared, so neither is rewritten.
_REFRESH_COLUMNS = tuple(column for column in _COLUMNS if column not in ("id", "created_at"))
_INSERT_SQL = f"INSERT INTO cache_entry VALUES ({', '.join('?' * len(_COLUMNS))})"
_UPDATE_SQL = (
    "UPDATE cache_entry SET "
    + ", ".join(f"{column} = ?" for column in _REFRESH_COLUMNS)
    + " WHERE id = ?"
)


def _path_lock(index_path: Path) -> threading.RLock:
    key = str(index_path)
    with _REGISTRY_LOCK:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


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
        self._connect_lock = _path_lock(layout.index_path)
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

    def private_path(self, *, category: str, source: str) -> Path:
        """Layout-owned profile cache directory for private evidence.

        Refuse symlink ancestors before creating anything, and never repair a
        foreign-owned directory. This does not alter permissions of shared roots.
        """
        for component in (category, source):
            if not component or Path(component).name != component or component in (".", ".."):
                raise ValueError("private cache category/source must be path components")
        directory = self._layout.category_dir(scope="profile", category=category) / source
        if not directory.is_absolute():
            raise ValueError("private cache requires an absolute layout root")
        for ancestor in (*reversed(directory.parents), directory):
            if ancestor.is_symlink():
                raise ValueError("private cache refuses symlink ancestors")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("private cache directory must be owner-controlled")
        os.chmod(directory, 0o700)
        return directory

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
        entry = self.build_entry(
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
        """Register multiple cache entries using one DuckDB connection.

        Known entries are updated in place and only unknown ones are inserted. An
        upsert (``INSERT OR REPLACE``) is a delete plus an insert in DuckDB, and since
        each short-lived connection checkpoints on close, every re-registration left a
        dead row group in the file: a few thousand entries grew the index to 700 MB and
        each subsequent open/close cost over a CPU-second. Held under the per-file lock
        so two threads cannot both find an id missing and both insert it; another
        process cannot interleave because DuckDB locks the file for the connection.
        """
        items = tuple(entries)
        if not items:
            return ()
        # Last registration of an id wins, as it did under the previous upsert.
        latest = {entry.entry_id: entry for entry in items}
        with self._connect_lock, self._connect() as con:
            existing = {
                str(row[0])
                for row in con.execute(
                    "SELECT id FROM cache_entry WHERE list_contains(?, id)",
                    [list(latest)],
                ).fetchall()
            }
            updates = [_refresh_params(e) for e in latest.values() if e.entry_id in existing]
            inserts = [_entry_params(e) for e in latest.values() if e.entry_id not in existing]
            if updates:
                con.executemany(_UPDATE_SQL, updates)
            if inserts:
                con.executemany(_INSERT_SQL, inserts)
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
            entries.append(self.build_entry(
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

    def cleanup_expired(
        self,
        *,
        now: float | None = None,
        category: str | None = None,
        keep: Iterable[Path] = (),
        min_idle_s: float = 0.0,
    ) -> int:
        """Delete expired entries and their managed files; return how many were removed.

        ``category`` limits the sweep to one owner's artifacts. ``keep`` spares paths the
        owner is still writing, whose recorded expiry can trail the file by a refresh
        interval. ``min_idle_s`` spares any file written to more recently than that,
        whoever is writing it: the index is shared across processes, so a row can be
        stale for a file another process has just appended to, and only the file itself
        knows. A file that cannot be deleted keeps its row, so the next sweep retries it
        instead of leaving an unindexed file nobody will ever expire.
        """
        cutoff = time.time() if now is None else now
        kept = {str(Path(path).expanduser().resolve()) for path in keep}
        idle_cutoff = cutoff - min_idle_s if min_idle_s > 0 else None
        sql = "SELECT id, path FROM cache_entry WHERE expires_at IS NOT NULL AND expires_at <= ?"
        params: list[Any] = [cutoff]
        if category is not None:
            sql += " AND category = ?"
            params.append(category)
        with self._connect_lock, self._connect() as con:
            removed: list[str] = []
            for entry_id, raw_path in con.execute(sql, params).fetchall():
                if str(raw_path) in kept:
                    continue
                path = Path(str(raw_path))
                if self._is_managed_path(path):
                    if idle_cutoff is not None and _modified_after(path, idle_cutoff):
                        continue
                    try:
                        path.unlink(missing_ok=True)
                    except OSError as exc:
                        logger.warning("Could not delete expired cache file %s: %s", path, exc)
                        continue
                removed.append(str(entry_id))
            if removed:
                con.executemany("DELETE FROM cache_entry WHERE id = ?", [(entry_id,) for entry_id in removed])
        return len(removed)

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
        """Create the table and drop legacy indexes, once per index file per process.

        Keyed by inode as well as path, so a file that was deleted or replaced (for
        example by an offline compaction) is initialized again rather than assumed.
        """
        index_path = self._layout.index_path
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect_lock:
            key = _schema_key(index_path)
            if key is not None and key in _SCHEMA_READY:
                return
            self._create_schema()
            key = _schema_key(index_path)
            if key is not None:
                _SCHEMA_READY.add(key)

    def _create_schema(self) -> None:
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
            for index_name in _LEGACY_INDEXES:
                con.execute(f"DROP INDEX IF EXISTS {index_name}")

    def build_entry(
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
        """Describe ``path`` as an entry without writing it, for ``register_many``.

        Lets a caller that indexes several files at once pay for one connection rather
        than one per file.
        """
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
        """Create a DuckDB connection, serialized to protect against concurrent writes.

        Deliberately one short-lived connection per operation, an intended exception to
        the ``ConnectionHolder`` rule in ``leapflow.storage.connection``: the index is
        shared by every LeapFlow process on the profile (daemon, CLI, doctor, node), and
        DuckDB holds an exclusive file lock for as long as a read-write connection is
        open, so a long-lived connection in one process would lock the others out. The
        cost of that choice is bounded by keeping writes infrequent and in place.
        """
        with self._connect_lock:
            return duckdb.connect(str(self._layout.index_path))

    def _is_managed_path(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self._layout.root.resolve())
            return True
        except ValueError:
            return False


def _modified_after(path: Path, moment: float) -> bool:
    try:
        return path.stat().st_mtime > moment
    except OSError:
        return False


def _schema_key(index_path: Path) -> tuple[str, int] | None:
    try:
        return (str(index_path), index_path.stat().st_ino)
    except OSError:
        return None


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


def _refresh_params(entry: CacheEntry) -> list[Any]:
    values = dict(zip(_COLUMNS, _entry_params(entry)))
    return [values[column] for column in _REFRESH_COLUMNS] + [entry.entry_id]


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
