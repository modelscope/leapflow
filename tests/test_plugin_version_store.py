# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for profile-scoped plugin version store."""
from __future__ import annotations

from leapflow.storage.plugin_version_store import PluginVersionStore


def test_plugin_version_store_records_active_and_versions(tmp_path) -> None:
    source = tmp_path / "plugin_a.py"
    source.write_text("VALUE = 'v0'\n", encoding="utf-8")
    store = PluginVersionStore(tmp_path / "versions")

    entry = store.record_source("plugin_a", source, version="v0")

    assert entry["version"] == "v0"
    assert store.active("plugin_a")["version"] == "v0"
    assert store.source_for("plugin_a", "v0").exists()
    assert [item["version"] for item in store.versions("plugin_a")] == ["v0"]


def test_plugin_version_store_rollback_copies_snapshot(tmp_path) -> None:
    source = tmp_path / "plugin_a.py"
    source.write_text("VALUE = 'v0'\n", encoding="utf-8")
    store = PluginVersionStore(tmp_path / "versions")
    store.record_source("plugin_a", source, version="v0")
    source.write_text("VALUE = 'v1'\n", encoding="utf-8")
    store.record_source("plugin_a", source, version="v1")

    entry = store.rollback("plugin_a", "v0", source)

    assert entry["version"] == "v0"
    assert source.read_text(encoding="utf-8") == "VALUE = 'v0'\n"
    assert store.active("plugin_a")["metadata"]["rollback"] is True
