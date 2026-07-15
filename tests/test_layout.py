from __future__ import annotations

from leapflow.layout import build_layout, workspace_id_for_path


def test_layout_history_workspace_helpers_and_descriptors(tmp_path) -> None:
    layout = build_layout(tmp_path / "leap-home")
    profile_layout = layout.ensure(profile_id="default")
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    assert profile_layout.history_dir == profile_layout.root / "history"
    assert profile_layout.tui_history_path == profile_layout.history_dir / "tui_history"
    assert layout.workspace_config_path(workspace_root) == workspace_root / ".leapflow" / "config.yaml"
    assert layout.workspace_manifest_path(workspace_root) == workspace_root / ".leapflow" / "workspace.yaml"

    workspace_manifest = layout.write_workspace_manifest(workspace_root)
    workspace_id = workspace_id_for_path(workspace_root)
    cache_manifest = profile_layout.cache.write_workspace_manifest(workspace_id, workspace_root)

    assert workspace_manifest.exists()
    assert cache_manifest.exists()
    assert profile_layout.cache.workspace_manifest_path(workspace_id) == cache_manifest

    history_descriptor = layout.describe_path(profile_layout.tui_history_path)
    assert history_descriptor.category == "history"
    assert history_descriptor.owner_component == "tui"
    assert history_descriptor.syncable is False

    mcp_descriptor = layout.describe_path(layout.mcp_servers_path)
    assert mcp_descriptor.category == "mcp_config"
    assert mcp_descriptor.scope == "global"

    session_video = profile_layout.cache.session_dir(workspace_id, "session-1") / "video" / "recording.mp4"
    video_descriptor = layout.describe_path(session_video)
    assert video_descriptor.category == "cache_sensitive"
    assert video_descriptor.scope == "session"
    assert video_descriptor.syncable is False
