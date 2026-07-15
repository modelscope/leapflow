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


def test_cache_register_directory_and_quota_cleanup(tmp_path) -> None:
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
    assert all(entry.sensitive is True and entry.syncable is False for entry in entries)
    assert all(entry.owner_component == "perception.video" for entry in entries)

    removed = manager.cleanup_quota(
        scope=CacheScope.SESSION.value,
        category="video",
        workspace_id="ws-test",
        session_id="session-1",
        max_bytes=10,
    )

    remaining = manager.list_entries(
        scope=CacheScope.SESSION.value,
        category="video",
        workspace_id="ws-test",
        session_id="session-1",
    )
    assert removed == 1
    assert len(remaining) == 1
    assert ignored.exists()
