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
