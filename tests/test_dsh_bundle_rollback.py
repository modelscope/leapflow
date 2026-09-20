# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for DSH directory-level bundle snapshot and rollback."""
from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import pytest

from leapflow.storage.plugin_version_store import PluginVersionStore


# ── helpers ───────────────────────────────────────────────────────

def _make_bundle(tmp_path: Path, plugin_id: str = "my_dsh") -> tuple[Path, Path]:
    """Create a wrapper .py file and a bundle directory with several files."""
    wrapper = tmp_path / "plugins" / f"{plugin_id}.py"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("# wrapper v1\nimport json\n", encoding="utf-8")

    bundle = tmp_path / "dsh" / plugin_id
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "index.js").write_text("module.exports = {};", encoding="utf-8")
    (bundle / "package.json").write_text('{"name":"test"}', encoding="utf-8")
    sub = bundle / "lib"
    sub.mkdir()
    (sub / "helper.js").write_text("exports.help = true;", encoding="utf-8")
    return wrapper, bundle


def _content_sha256(wrapper: Path, bundle: Path) -> str:
    """Reproduce the combined SHA-256 used by record_bundle."""
    h = hashlib.sha256()
    h.update(wrapper.read_bytes())
    for p in sorted(bundle.rglob("*")):
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


# ── tests ─────────────────────────────────────────────────────────

def test_bundle_snapshot_and_restore_roundtrip(tmp_path: Path) -> None:
    """record_bundle → delete originals → rollback_bundle restores everything."""
    wrapper, bundle = _make_bundle(tmp_path)
    store = PluginVersionStore(tmp_path / "versions")

    entry = store.record_bundle("my_dsh", wrapper, bundle, version="v1")

    assert entry["version"] == "v1"
    assert entry["is_bundle"] is True
    assert entry["bundle_sha256"]
    assert Path(entry["snapshot_path"]).exists()

    # Capture original content for later comparison.
    orig_wrapper = wrapper.read_text(encoding="utf-8")
    orig_index = (bundle / "index.js").read_text(encoding="utf-8")
    orig_helper = (bundle / "lib" / "helper.js").read_text(encoding="utf-8")

    # Delete originals.
    wrapper.unlink()
    import shutil
    shutil.rmtree(bundle)
    assert not wrapper.exists()
    assert not bundle.exists()

    # Rollback.
    result = store.rollback_bundle("my_dsh", "v1", wrapper, bundle)
    assert result["ok"] is True
    assert result["bundle"] is True

    # Verify everything is restored.
    assert wrapper.read_text(encoding="utf-8") == orig_wrapper
    assert (bundle / "index.js").read_text(encoding="utf-8") == orig_index
    assert (bundle / "lib" / "helper.js").read_text(encoding="utf-8") == orig_helper


def test_rollback_bundle_restores_both_wrapper_and_dir(tmp_path: Path) -> None:
    """Modify both wrapper and bundle after recording, rollback restores original."""
    wrapper, bundle = _make_bundle(tmp_path)
    store = PluginVersionStore(tmp_path / "versions")

    store.record_bundle("my_dsh", wrapper, bundle, version="v1")

    # Modify both wrapper and bundle content.
    wrapper.write_text("# modified wrapper\n", encoding="utf-8")
    (bundle / "index.js").write_text("MODIFIED", encoding="utf-8")
    (bundle / "lib" / "helper.js").write_text("MODIFIED HELPER", encoding="utf-8")
    (bundle / "new_file.txt").write_text("should disappear", encoding="utf-8")

    result = store.rollback_bundle("my_dsh", "v1", wrapper, bundle)
    assert result["ok"] is True

    assert wrapper.read_text(encoding="utf-8") == "# wrapper v1\nimport json\n"
    assert (bundle / "index.js").read_text(encoding="utf-8") == "module.exports = {};"
    assert (bundle / "lib" / "helper.js").read_text(encoding="utf-8") == "exports.help = true;"
    # The extra file added after recording should be gone (dir was replaced).
    assert not (bundle / "new_file.txt").exists()


def test_rollback_bundle_with_sha256_mismatch_fails(tmp_path: Path) -> None:
    """Tampering with the stored archive triggers an integrity error."""
    wrapper, bundle = _make_bundle(tmp_path)
    store = PluginVersionStore(tmp_path / "versions")

    entry = store.record_bundle("my_dsh", wrapper, bundle, version="v1")
    archive_path = Path(entry["snapshot_path"])

    # Tamper: rewrite the archive with different content.
    tampered_wrapper = tmp_path / "tampered.py"
    tampered_wrapper.write_text("# TAMPERED\n", encoding="utf-8")
    tampered_bundle = tmp_path / "tampered_bundle"
    tampered_bundle.mkdir()
    (tampered_bundle / "bad.js").write_text("BAD", encoding="utf-8")
    import io
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(tampered_wrapper), arcname=f"wrapper/{tampered_wrapper.name}")
        tar.add(str(tampered_bundle), arcname="bundle")
    archive_path.write_bytes(buf.getvalue())

    with pytest.raises(RuntimeError, match="integrity check failed"):
        store.rollback_bundle("my_dsh", "v1", wrapper, bundle)


def test_rollback_bundle_missing_version_raises_key_error(tmp_path: Path) -> None:
    """Attempting to rollback to a non-existent bundle version raises KeyError."""
    store = PluginVersionStore(tmp_path / "versions")
    wrapper = tmp_path / "w.py"
    bundle = tmp_path / "b"

    with pytest.raises(KeyError, match="Bundle version not found"):
        store.rollback_bundle("ghost", "v99", wrapper, bundle)


def test_record_bundle_updates_active_and_index(tmp_path: Path) -> None:
    """Recording two bundle versions tracks both and points active to the latest."""
    wrapper, bundle = _make_bundle(tmp_path)
    store = PluginVersionStore(tmp_path / "versions")

    store.record_bundle("my_dsh", wrapper, bundle, version="v1")
    # Modify and record v2.
    (bundle / "index.js").write_text("v2 content", encoding="utf-8")
    wrapper.write_text("# wrapper v2\n", encoding="utf-8")
    store.record_bundle("my_dsh", wrapper, bundle, version="v2")

    versions = store.versions("my_dsh")
    version_ids = [v["version"] for v in versions]
    assert "v1" in version_ids
    assert "v2" in version_ids

    active = store.active("my_dsh")
    assert active is not None
    assert active["version"] == "v2"
    assert active["is_bundle"] is True
