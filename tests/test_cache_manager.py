# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import annotations

from leapflow.cache.manager import CacheManager, CacheScope
from leapflow.layout import build_layout


def test_cache_entry_id_is_stable_across_manager_instances(tmp_path) -> None:
    profile_layout = build_layout(tmp_path / "leap-home").ensure(profile_id="default")
    cache_path = profile_layout.cache.category_dir(
        scope="workspace",
        category="video",
        workspace_id="ws-test",
    ) / "segment.mp4"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"video")

    first = CacheManager(profile_layout.cache, profile_id="default").register(
        path=cache_path,
        scope=CacheScope.WORKSPACE,
        category="video",
        source="segment",
        workspace_id="ws-test",
    )
    second = CacheManager(profile_layout.cache, profile_id="default").register(
        path=cache_path,
        scope=CacheScope.WORKSPACE,
        category="video",
        source="segment",
        workspace_id="ws-test",
    )

    assert first.entry_id == second.entry_id
    assert len(first.entry_id) == 64


def test_cache_register_directory_and_quota_cleanup(monkeypatch, tmp_path) -> None:
    profile_layout = build_layout(tmp_path / "leap-home").ensure(profile_id="default")
    manager = CacheManager(profile_layout.cache, profile_id="default")
    root = profile_layout.cache.session_dir("ws-test", "session-1") / "video" / "recording"
    root.mkdir(parents=True)
    old_segment = root / "old.mp4"
    new_segment = root / "new.mp4"
    ignored = root / "notes.txt"
    old_segment.write_bytes(b"a" * 10)
    new_segment.write_bytes(b"b" * 10)
    ignored.write_text("ignore", encoding="utf-8")

    connect_calls: list[str] = []
    original_connect = manager._connect

    def counted_connect():
        connect_calls.append("connect")
        return original_connect()

    monkeypatch.setattr(manager, "_connect", counted_connect)

    entries = manager.register_directory(
        root=root,
        scope=CacheScope.SESSION,
        category="video",
        source="recording",
        workspace_id="ws-test",
        session_id="session-1",
        sensitive=True,
        syncable=False,
        owner_component="perception.video",
        suffixes=(".mp4",),
    )

    assert len(entries) == 2
    assert len(connect_calls) == 1
    assert all(entry.sensitive is True and entry.syncable is False for entry in entries)
    assert all(entry.owner_component == "perception.video" for entry in entries)

    connect_calls.clear()
    removed = manager.cleanup_quota(
        scope=CacheScope.SESSION.value,
        category="video",
        workspace_id="ws-test",
        session_id="session-1",
        max_bytes=10,
    )

    assert len(connect_calls) == 2
    remaining = manager.list_entries(
        scope=CacheScope.SESSION.value,
        category="video",
        workspace_id="ws-test",
        session_id="session-1",
    )
    assert removed == 1
    assert len(remaining) == 1
    assert ignored.exists()


def _segment(profile_layout, name: str = "seg.ndjson"):
    path = profile_layout.cache.session_dir("ws-test", "session-1") / "raw" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _register(manager: CacheManager, path, *, expires_at: float) -> None:
    manager.register(
        path=path,
        scope=CacheScope.SESSION,
        category="raw",
        source=path.name,
        workspace_id="ws-test",
        session_id="session-1",
        expires_at=expires_at,
    )


def test_re_registration_refreshes_in_place_and_keeps_created_at(tmp_path) -> None:
    """A known entry is updated, not replaced: one row, latest size/TTL, first sighting kept."""
    profile_layout = build_layout(tmp_path / "leap-home").ensure(profile_id="default")
    manager = CacheManager(profile_layout.cache, profile_id="default")
    path = _segment(profile_layout)

    _register(manager, path, expires_at=100.0)
    created_at = manager.list_entries()[0].created_at
    for size in range(2, 12):
        path.write_bytes(b"x" * size)
        _register(manager, path, expires_at=100.0 + size)

    entries = manager.list_entries()
    assert len(entries) == 1
    assert entries[0].size_bytes == 11
    assert entries[0].expires_at == 111.0
    assert entries[0].created_at == created_at, (
        "created_at is when the artifact first appeared; a refresh must not reset it"
    )


def test_repeated_registration_does_not_grow_the_index_file(tmp_path) -> None:
    """Each short-lived connection checkpoints on close.

    Under ``INSERT OR REPLACE`` every re-registration left a dead row group behind, so
    a few thousand segments refreshed once a second grew the index to 700 MB and each
    open/close cost over a CPU-second. An in-place update must leave the file flat.
    """
    profile_layout = build_layout(tmp_path / "leap-home").ensure(profile_id="default")
    manager = CacheManager(profile_layout.cache, profile_id="default")
    paths = [_segment(profile_layout, f"seg{index}.ndjson") for index in range(5)]

    def _refresh_all(round_index: int) -> None:
        manager.register_many(
            manager.build_entry(
                path=path,
                scope=CacheScope.SESSION,
                category="raw",
                source=path.name,
                workspace_id="ws-test",
                session_id="session-1",
                expires_at=1000.0 + round_index,
                metadata={"round": round_index},
            )
            for path in paths
        )

    for round_index in range(10):
        _refresh_all(round_index)
    settled = manager.index_path.stat().st_size
    for round_index in range(10, 60):
        _refresh_all(round_index)

    growth = manager.index_path.stat().st_size - settled
    assert growth <= 256 * 1024, f"index grew {growth} bytes over 50 in-place refreshes"
    assert len(manager.list_entries()) == len(paths)


def test_legacy_secondary_indexes_are_dropped_and_data_kept(tmp_path) -> None:
    """An index written by an earlier release loses its ART indexes on first open."""
    import duckdb

    profile_layout = build_layout(tmp_path / "leap-home").ensure(profile_id="default")
    manager = CacheManager(profile_layout.cache, profile_id="default")
    path = _segment(profile_layout)
    _register(manager, path, expires_at=100.0)
    with duckdb.connect(str(manager.index_path)) as con:
        con.execute("CREATE INDEX idx_cache_scope ON cache_entry(profile_id, scope)")
        con.execute("CREATE INDEX idx_cache_expiry ON cache_entry(expires_at)")

    from leapflow.cache import manager as manager_module

    manager_module._SCHEMA_READY.clear()
    reopened = CacheManager(profile_layout.cache, profile_id="default")

    with duckdb.connect(str(reopened.index_path)) as con:
        indexes = con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()
    assert indexes == []
    assert [entry.path for entry in reopened.list_entries()] == [path.resolve()]


def test_cleanup_expired_is_limited_to_the_requested_category(tmp_path) -> None:
    profile_layout = build_layout(tmp_path / "leap-home").ensure(profile_id="default")
    manager = CacheManager(profile_layout.cache, profile_id="default")
    raw = _segment(profile_layout, "raw.ndjson")
    other = _segment(profile_layout, "other.json")
    _register(manager, raw, expires_at=100.0)
    manager.register(
        path=other,
        scope=CacheScope.SESSION,
        category="web_fetch",
        source=other.name,
        workspace_id="ws-test",
        session_id="session-1",
        expires_at=100.0,
    )

    assert manager.cleanup_expired(now=200.0, category="raw") == 1

    assert not raw.exists()
    assert other.exists(), "a sweep scoped to one owner must not expire another's artifacts"
    assert [entry.category for entry in manager.list_entries()] == ["web_fetch"]


def test_cleanup_expired_spares_kept_paths(tmp_path) -> None:
    """A segment still being written can carry a stale expiry; its owner names it to keep."""
    profile_layout = build_layout(tmp_path / "leap-home").ensure(profile_id="default")
    manager = CacheManager(profile_layout.cache, profile_id="default")
    live = _segment(profile_layout, "live.ndjson")
    done = _segment(profile_layout, "done.ndjson")
    _register(manager, live, expires_at=100.0)
    _register(manager, done, expires_at=100.0)

    # Unresolved on purpose: the index stores resolved paths.
    unresolved = live.parent / ".." / live.parent.name / live.name
    assert manager.cleanup_expired(now=200.0, keep=[unresolved]) == 1

    assert live.exists() and not done.exists()
    assert [entry.path.name for entry in manager.list_entries()] == ["live.ndjson"]


def test_cleanup_expired_honours_an_explicit_epoch(tmp_path) -> None:
    """``now=0.0`` is a moment, not "unset"; it must not fall back to the wall clock."""
    profile_layout = build_layout(tmp_path / "leap-home").ensure(profile_id="default")
    manager = CacheManager(profile_layout.cache, profile_id="default")
    path = _segment(profile_layout)
    _register(manager, path, expires_at=100.0)

    assert manager.cleanup_expired(now=0.0) == 0
    assert manager.cleanup_expired(now=99.0) == 0
    assert path.exists()
    assert manager.cleanup_expired(now=100.0) == 1


def test_cleanup_expired_spares_recently_written_files_when_asked(tmp_path) -> None:
    import os
    import time

    profile_layout = build_layout(tmp_path / "leap-home").ensure(profile_id="default")
    manager = CacheManager(profile_layout.cache, profile_id="default")
    fresh = _segment(profile_layout, "fresh.ndjson")
    idle = _segment(profile_layout, "idle.ndjson")
    an_hour_ago = time.time() - 3600.0
    os.utime(idle, (an_hour_ago, an_hour_ago))
    _register(manager, fresh, expires_at=100.0)
    _register(manager, idle, expires_at=100.0)

    assert manager.cleanup_expired(min_idle_s=600.0) == 1

    assert fresh.exists() and not idle.exists()
    assert [entry.path.name for entry in manager.list_entries()] == ["fresh.ndjson"]


def test_cleanup_expired_keeps_the_row_of_a_file_it_could_not_delete(monkeypatch, tmp_path) -> None:
    """Dropping the row would leave an unindexed file that nothing would ever expire."""
    from pathlib import Path

    profile_layout = build_layout(tmp_path / "leap-home").ensure(profile_id="default")
    manager = CacheManager(profile_layout.cache, profile_id="default")
    path = _segment(profile_layout)
    _register(manager, path, expires_at=100.0)

    def refuse(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError("read-only")

    monkeypatch.setattr(Path, "unlink", refuse)
    assert manager.cleanup_expired(now=200.0) == 0
    monkeypatch.undo()

    assert len(manager.list_entries()) == 1
    assert manager.cleanup_expired(now=200.0) == 1
    assert not path.exists()


def test_schema_is_initialized_once_per_index_file(monkeypatch, tmp_path) -> None:
    """Callers like the web cache build a manager per call; that must not cost DDL each time."""
    profile_layout = build_layout(tmp_path / "leap-home").ensure(profile_id="default")
    calls: list[str] = []
    original = CacheManager._create_schema

    def counted(self: CacheManager) -> None:
        calls.append("schema")
        original(self)

    monkeypatch.setattr(CacheManager, "_create_schema", counted)
    for _ in range(5):
        CacheManager(profile_layout.cache, profile_id="default")

    assert calls == ["schema"]
