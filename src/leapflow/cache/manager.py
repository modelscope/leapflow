# Copyright (c) Alibaba, Inc. and its affiliates.
"""DuckDB-indexed cache manager for profile/workspace/session scopes."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import stat
import threading
import time
import uuid
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
_VALIDITY_PROCESS = uuid.uuid4().hex
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

    def persist_session_content(
        self, content: bytes, *, workspace_id: str, session_id: str,
        category: str, source: str, suffix: str, expires_at: float,
        monotonic_deadline: float, max_session_bytes: int, metadata: dict[str, Any],
    ) -> CacheEntry:
        """Atomically reserve quota and index a unique private sensitive artifact.

        No eviction, shared-content authority, or lifetime extension. A process
        restart invalidates these transient references. DuckDB's exclusive
        connection and the in-process lock serialize reservation and publication.
        """
        if (not content or len(content) > max_session_bytes
                or not math.isfinite(expires_at) or expires_at <= time.time()
                or not math.isfinite(monotonic_deadline) or monotonic_deadline <= time.monotonic()):
            raise ValueError("session evidence is empty, expired, or exceeds its quota")
        if suffix not in (".jpg", ".png", ".webp"):
            raise ValueError("unsupported session evidence extension")
        directory = self._private_session_directory(workspace_id, session_id, category, source)
        filename = uuid.uuid4().hex + suffix
        try:
            with self._connect_lock, self._connect() as con:
                used = con.execute(
                    "SELECT COALESCE(SUM(size_bytes), 0) FROM cache_entry WHERE profile_id = ? "
                    "AND scope = 'session' AND workspace_id = ? AND session_id = ?",
                    [self._profile_id, workspace_id, session_id],
                ).fetchone()[0]
                if used + len(content) > max_session_bytes:
                    raise ValueError("session evidence quota full; no artifact was admitted or evicted")
                fd = os.open(filename, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(content)
                        handle.flush()
                        os.fsync(handle.fileno())
                    path = self._layout.category_dir(
                        scope="session", category=category, workspace_id=workspace_id,
                        session_id=session_id,
                    ) / source / filename
                    entry = CacheEntry(
                        entry_id=uuid.uuid4().hex, profile_id=self._profile_id,
                        path=path, scope="session", category=category, source=source,
                        workspace_id=workspace_id, session_id=session_id, size_bytes=len(content),
                        content_hash=hashlib.sha256(content).hexdigest(), expires_at=expires_at,
                        sensitive=True, syncable=False, owner_component=source,
                        metadata={**metadata, "_validity_process": _VALIDITY_PROCESS,
                                  "_monotonic_deadline": monotonic_deadline},
                    )
                    if time.monotonic() >= monotonic_deadline or time.time() >= expires_at:
                        raise ValueError("session evidence expired during persistence")
                    con.execute(_INSERT_SQL, _entry_params(entry))
                except (OSError, ValueError, duckdb.Error):
                    os.unlink(filename, dir_fd=directory)
                    raise
                return entry
        finally:
            os.close(directory)

    def discard_session_content(self, entry: CacheEntry) -> None:
        """Remove only this process's newly retained artifact after cancellation."""
        root = self._layout.session_dir(entry.workspace_id, entry.session_id)
        if (entry.profile_id != self._profile_id or entry.scope != "session"
                or entry.metadata.get("_validity_process") != _VALIDITY_PROCESS
                or not entry.path.is_relative_to(root)):
            raise ValueError("cannot discard foreign session content")
        with self._connect_lock, self._connect() as con:
            row = con.execute("SELECT path FROM cache_entry WHERE id = ? AND profile_id = ?",
                              [entry.entry_id, self._profile_id]).fetchone()
            if row is not None:
                if str(row[0]) != str(entry.path):
                    raise ValueError("session content changed before discard")
                self._unlink_session_file(entry.path)
                con.execute("DELETE FROM cache_entry WHERE id = ?", [entry.entry_id])

    @staticmethod
    def _unlink_session_file(path: Path) -> None:
        """Unlink via pinned directories; final links are removed, never followed."""
        directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in path.parts[1:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = child
            try:
                os.unlink(path.name, dir_fd=directory)
            except FileNotFoundError:
                pass
        except FileNotFoundError:
            pass
        finally:
            os.close(directory)

    def _private_session_directory(
        self, workspace_id: str, session_id: str, category: str, source: str,
    ) -> int:
        """Create/open owned directories descriptor-relatively, never follow links."""
        for part in (workspace_id, session_id, category, source):
            if not part or Path(part).name != part or part in (".", ".."):
                raise ValueError("session evidence ownership must contain safe path components")
        path = self._layout.category_dir(
            scope="session", category=category, workspace_id=workspace_id, session_id=session_id,
        ) / source
        if not path.is_absolute() or os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
            raise ValueError("secure session evidence storage unavailable on this platform")
        parent = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in path.parts[1:]:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=parent)
                except FileExistsError:
                    pass
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                os.close(parent)
                parent = child
            info = os.fstat(parent)
            if info.st_uid != os.getuid():
                raise ValueError("private session directory is not owner-controlled")
            os.fchmod(parent, 0o700)
            return parent
        except (OSError, ValueError):
            os.close(parent)
            raise

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
        sql = "SELECT id, path, expires_at, metadata_json FROM cache_entry WHERE profile_id = ? AND expires_at IS NOT NULL"
        params: list[Any] = [self._profile_id]
        monotonic_now = time.monotonic()
        if category is not None:
            sql += " AND category = ?"
            params.append(category)
        with self._connect_lock, self._connect() as con:
            removed: list[str] = []
            for entry_id, raw_path, expiry, metadata_json in con.execute(sql, params).fetchall():
                metadata = json.loads(metadata_json or "{}")
                transient = metadata.get("_validity_process") == _VALIDITY_PROCESS
                deadline = metadata.get("_monotonic_deadline")
                monotonic_expired = transient and (
                    type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= monotonic_now
                )
                if expiry > cutoff and not monotonic_expired:
                    continue
                if str(raw_path) in kept:
                    continue
                path = Path(str(raw_path))
                if transient or self._is_managed_path(path):
                    if not path.is_relative_to(self._layout.root):
                        continue
                    if idle_cutoff is not None and _modified_after(path, idle_cutoff):
                        continue
                    try:
                        if transient:
                            self._unlink_session_file(path)
                        else:
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

    def resolve_session_entry(self, entry_id: str, *, workspace_id: str,
                              session_id: str, max_bytes: int) -> CacheEntry:
        """Resolve an opaque artifact ID within its caller's session, never a path."""
        if not workspace_id or not session_id or max_bytes < 1:
            raise ValueError("artifact requires an active workspace/session and byte budget")
        with self._connect_lock, self._connect() as con:
            row = con.execute(
                "SELECT * FROM cache_entry WHERE id = ? AND profile_id = ? AND scope = 'session' "
                "AND workspace_id = ? AND session_id = ?",
                [entry_id, self._profile_id, workspace_id, session_id],
            ).fetchone()
        if row is None:
            raise ValueError("artifact unavailable in this session")
        entry = _entry_from_row(row)
        root = self._layout.session_dir(workspace_id, session_id)
        if (entry.expires_at is None or not math.isfinite(entry.expires_at)
                or entry.expires_at <= time.time()):
            raise ValueError("artifact expired or lacks a lifetime")
        if "_monotonic_deadline" in entry.metadata:
            deadline = entry.metadata["_monotonic_deadline"]
            if (entry.metadata.get("_validity_process") != _VALIDITY_PROCESS
                    or type(deadline) not in (int, float) or not math.isfinite(deadline)
                    or deadline <= time.monotonic()):
                raise ValueError("transient artifact expired or its producing runtime restarted")
        if (entry.size_bytes < 1 or entry.size_bytes > max_bytes or entry.syncable or not entry.sensitive
                or not entry.content_hash):
            raise ValueError("artifact is not admitted sensitive, checksum-bound session evidence")
        if not entry.path.is_relative_to(root) or any(p.is_symlink() for p in (entry.path, *entry.path.parents)):
            raise ValueError("artifact escaped its managed session root")
        return entry

    def read_session_entry(self, entry: CacheEntry, *, max_bytes: int) -> bytes:
        """Revalidate ownership/TTL after approval and consume a bounded regular file."""
        current = self.resolve_session_entry(entry.entry_id, workspace_id=entry.workspace_id,
                                             session_id=entry.session_id, max_bytes=max_bytes)
        if current != entry:
            raise ValueError("artifact changed while awaiting approval")
        fd = self._open_session_file(entry.path)
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("artifact is not a regular file")
            content = handle.read(max_bytes + 1)
        if len(content) > max_bytes or len(content) != entry.size_bytes:
            raise ValueError("artifact content changed or exceeded its budget")
        if hashlib.sha256(content).hexdigest() != entry.content_hash:
            raise ValueError("artifact checksum differs from admitted content")
        current = self.resolve_session_entry(entry.entry_id, workspace_id=entry.workspace_id,
                                             session_id=entry.session_id, max_bytes=max_bytes)
        if current != entry:
            raise ValueError("artifact changed during its read")
        return content

    @staticmethod
    def _open_session_file(path: Path) -> int:
        """Open relative to pinned directory descriptors, refusing symlink/FIFO races."""
        if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
            raise ValueError("Secure session artifact reads unavailable on this platform")
        directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in path.parts[1:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = child
            return os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        finally:
            os.close(directory)

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
        if scope_value == CacheScope.SESSION.value and content_hash:
            # Content deduplication must not merge independent session evidence
            # or distinct files. Unhashed streaming entries retain their path ID;
            # persisted legacy IDs remain readable without rewriting references.
            basis = json.dumps([self._profile_id, workspace_id, session_id, str(resolved), content_hash])
            entry_id = hashlib.sha256(basis.encode("utf-8")).hexdigest()
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
