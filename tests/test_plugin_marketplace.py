# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the plugin marketplace (discover, verify, install external plugins)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from leapflow.plugins.marketplace import (
    MarketplaceClient,
    MarketplaceSource,
    PluginManifest,
)
from leapflow.plugins.marketplace.client import LocalDirectorySource


pytestmark = pytest.mark.unit


SAMPLE_CODE = b"plugin = None  # a trivial demo plugin\n"


def _make_manifest(checksum: str = "") -> PluginManifest:
    return PluginManifest(
        name="demo",
        version="1.0.0",
        author="test",
        description="A demo plugin",
        entry_point="demo",
        checksum_sha256=checksum,
    )


def _seed_marketplace(root: Path, code: bytes = SAMPLE_CODE) -> PluginManifest:
    """Create a fake marketplace directory with one sample plugin.

    Layout:
        <root>/demo/manifest.json
        <root>/demo/demo.py
    """
    checksum = PluginManifest.compute_checksum(code)
    manifest = _make_manifest(checksum)
    plugin_dir = root / manifest.name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "manifest.json").write_text(manifest.to_json())
    (plugin_dir / f"{manifest.entry_point}.py").write_bytes(code)
    return manifest


# ---------------------------------------------------------------------------
# PluginManifest
# ---------------------------------------------------------------------------


class TestPluginManifest:
    def test_manifest_json_roundtrip(self) -> None:
        manifest = PluginManifest(
            name="demo",
            version="2.3.4",
            author="alice",
            description="round trip",
            entry_point="demo_entry",
            plugin_type="tool",
            source_url="file:///tmp/demo",
            checksum_sha256="deadbeef",
            requires_sandbox=True,
            dependencies=["other"],
            min_leapflow_version="0.0.9",
        )
        restored = PluginManifest.from_json(manifest.to_json())
        assert restored == manifest

    def test_manifest_checksum_verification(self) -> None:
        checksum = PluginManifest.compute_checksum(SAMPLE_CODE)
        manifest = _make_manifest(checksum)
        assert manifest.verify_checksum(SAMPLE_CODE) is True
        assert manifest.verify_checksum(b"tampered content") is False

    def test_manifest_checksum_empty_is_unverifiable(self) -> None:
        # An empty declared checksum cannot be verified against anything.
        manifest = _make_manifest(checksum="")
        assert manifest.verify_checksum(SAMPLE_CODE) is False

    def test_manifest_tolerates_extra_keys(self) -> None:
        raw = {
            "name": "demo",
            "version": "1.0.0",
            "author": "test",
            "description": "d",
            "entry_point": "demo",
            "unknown_future_field": "ignored",
            "another_extra": [1, 2, 3],
        }
        manifest = PluginManifest.from_json(json.dumps(raw))
        assert manifest.name == "demo"
        assert manifest.entry_point == "demo"
        assert not hasattr(manifest, "unknown_future_field")


# ---------------------------------------------------------------------------
# LocalDirectorySource
# ---------------------------------------------------------------------------


class TestLocalDirectorySource:
    def test_local_source_is_marketplace_source(self, tmp_path: Path) -> None:
        source = LocalDirectorySource(tmp_path)
        assert isinstance(source, MarketplaceSource)

    def test_local_source_lists_manifests(self, tmp_path: Path) -> None:
        _seed_marketplace(tmp_path)
        source = LocalDirectorySource(tmp_path)
        manifests = source.list_manifests()
        assert len(manifests) == 1
        assert manifests[0].name == "demo"

    def test_local_source_missing_root_is_empty(self, tmp_path: Path) -> None:
        source = LocalDirectorySource(tmp_path / "does_not_exist")
        assert source.list_manifests() == []

    def test_local_source_fetch_code(self, tmp_path: Path) -> None:
        manifest = _seed_marketplace(tmp_path)
        source = LocalDirectorySource(tmp_path)
        code = source.fetch_code(manifest)
        assert code == SAMPLE_CODE

    def test_local_source_fetch_missing_code_returns_none(self, tmp_path: Path) -> None:
        source = LocalDirectorySource(tmp_path)
        # A manifest whose code file was never written.
        assert source.fetch_code(_make_manifest()) is None


# ---------------------------------------------------------------------------
# MarketplaceClient
# ---------------------------------------------------------------------------


class TestMarketplaceClient:
    def test_client_discover(self, tmp_path: Path) -> None:
        _seed_marketplace(tmp_path)
        client = MarketplaceClient(
            LocalDirectorySource(tmp_path), install_dir=tmp_path / "installed"
        )
        manifests = client.discover()
        assert [m.name for m in manifests] == ["demo"]

    def test_client_install_success(self, tmp_path: Path) -> None:
        _seed_marketplace(tmp_path)
        install_dir = tmp_path / "installed"
        client = MarketplaceClient(LocalDirectorySource(tmp_path), install_dir=install_dir)
        result = client.install("demo")
        assert result["ok"] is True
        assert result["name"] == "demo"
        assert result["version"] == "1.0.0"
        assert result["requires_sandbox"] is True
        installed = Path(result["installed_path"])
        assert installed.exists()
        assert installed.read_bytes() == SAMPLE_CODE

    def test_client_install_checksum_mismatch_refused(self, tmp_path: Path) -> None:
        # Seed a manifest with a checksum that does not match the code on disk.
        plugin_dir = tmp_path / "demo"
        plugin_dir.mkdir(parents=True)
        bad_manifest = _make_manifest(checksum="0" * 64)
        (plugin_dir / "manifest.json").write_text(bad_manifest.to_json())
        (plugin_dir / "demo.py").write_bytes(SAMPLE_CODE)

        install_dir = tmp_path / "installed"
        client = MarketplaceClient(LocalDirectorySource(tmp_path), install_dir=install_dir)
        result = client.install("demo")
        assert result["ok"] is False
        assert "Checksum mismatch" in result["error"]
        # Nothing must be written on an integrity failure.
        assert not (install_dir / "demo.py").exists()

    def test_client_install_unknown_plugin(self, tmp_path: Path) -> None:
        _seed_marketplace(tmp_path)
        client = MarketplaceClient(
            LocalDirectorySource(tmp_path), install_dir=tmp_path / "installed"
        )
        result = client.install("nonexistent")
        assert result["ok"] is False
        assert "not found" in result["error"]

    def test_client_install_skips_verify_when_disabled(self, tmp_path: Path) -> None:
        # verify=False bypasses the integrity gate even with a bad checksum.
        plugin_dir = tmp_path / "demo"
        plugin_dir.mkdir(parents=True)
        bad_manifest = _make_manifest(checksum="0" * 64)
        (plugin_dir / "manifest.json").write_text(bad_manifest.to_json())
        (plugin_dir / "demo.py").write_bytes(SAMPLE_CODE)

        client = MarketplaceClient(
            LocalDirectorySource(tmp_path), install_dir=tmp_path / "installed"
        )
        result = client.install("demo", verify=False)
        assert result["ok"] is True

    def test_client_uninstall(self, tmp_path: Path) -> None:
        _seed_marketplace(tmp_path)
        install_dir = tmp_path / "installed"
        client = MarketplaceClient(LocalDirectorySource(tmp_path), install_dir=install_dir)

        client.install("demo")
        assert (install_dir / "demo.py").exists()

        result = client.uninstall("demo")
        assert result["ok"] is True
        assert not (install_dir / "demo.py").exists()

    def test_client_uninstall_not_installed(self, tmp_path: Path) -> None:
        _seed_marketplace(tmp_path)
        client = MarketplaceClient(
            LocalDirectorySource(tmp_path), install_dir=tmp_path / "installed"
        )
        result = client.uninstall("demo")
        assert result["ok"] is False
        assert "not installed" in result["error"]
