# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for Ed25519 signing and verification in the Plugin Marketplace."""

from __future__ import annotations

from pathlib import Path

import pytest

from leapflow.plugins.marketplace import MarketplaceClient, PluginManifest
from leapflow.plugins.marketplace.client import LocalDirectorySource


pytestmark = pytest.mark.unit

SAMPLE_CODE = b"plugin = None  # a trivial demo plugin\n"


def _base_manifest() -> PluginManifest:
    return PluginManifest(
        name="demo",
        version="1.0.0",
        author="test",
        description="A demo plugin",
        entry_point="demo",
    )


def _seed_signed_marketplace(
    root: Path, code: bytes, private_key_hex: str
) -> PluginManifest:
    """Create a marketplace directory with a signed plugin."""
    manifest = _base_manifest().sign(code, private_key_hex)
    plugin_dir = root / manifest.name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "manifest.json").write_text(manifest.to_json())
    (plugin_dir / f"{manifest.entry_point}.py").write_bytes(code)
    return manifest


# ---------------------------------------------------------------------------
# Manifest signing / verification
# ---------------------------------------------------------------------------


class TestManifestSigning:
    def test_manifest_sign_and_verify_ok(self) -> None:
        """Sign then verify with correct pubkey succeeds."""
        priv, pub = PluginManifest.generate_keypair()
        manifest = _base_manifest()
        signed = manifest.sign(SAMPLE_CODE, priv)

        assert signed.signature != ""
        assert signed.signer_pubkey == pub
        assert signed.checksum_sha256 != ""
        assert signed.verify_signature(SAMPLE_CODE, {pub}) is True

    def test_manifest_verify_fails_wrong_pubkey(self) -> None:
        """Signed by A, verify with B's pubkey -> False."""
        priv_a, _pub_a = PluginManifest.generate_keypair()
        _priv_b, pub_b = PluginManifest.generate_keypair()

        signed = _base_manifest().sign(SAMPLE_CODE, priv_a)
        # Verify with B's pubkey — should fail
        assert signed.verify_signature(SAMPLE_CODE, {pub_b}) is False

    def test_manifest_verify_fails_tampered_code(self) -> None:
        """Sign valid, tamper code -> False."""
        priv, pub = PluginManifest.generate_keypair()
        signed = _base_manifest().sign(SAMPLE_CODE, priv)

        tampered_code = b"tampered content\n"
        assert signed.verify_signature(tampered_code, {pub}) is False

    def test_manifest_verify_fails_tampered_metadata(self) -> None:
        """Sign valid, change entry_point -> False."""
        priv, pub = PluginManifest.generate_keypair()
        signed = _base_manifest().sign(SAMPLE_CODE, priv)

        # Create a tampered manifest with different entry_point
        tampered = PluginManifest(
            name=signed.name,
            version=signed.version,
            author=signed.author,
            description=signed.description,
            entry_point="evil_entry",  # changed!
            plugin_type=signed.plugin_type,
            source_url=signed.source_url,
            checksum_sha256=signed.checksum_sha256,
            requires_sandbox=signed.requires_sandbox,
            dependencies=list(signed.dependencies),
            min_leapflow_version=signed.min_leapflow_version,
            signature=signed.signature,
            signer_pubkey=signed.signer_pubkey,
        )
        assert tampered.verify_signature(SAMPLE_CODE, {pub}) is False

    def test_manifest_without_signature_returns_false(self) -> None:
        """No signature -> verify returns False."""
        _priv, pub = PluginManifest.generate_keypair()
        manifest = _base_manifest()
        assert manifest.verify_signature(SAMPLE_CODE, {pub}) is False

    def test_signed_manifest_json_roundtrip(self) -> None:
        """Signature fields survive JSON serialization."""
        priv, pub = PluginManifest.generate_keypair()
        signed = _base_manifest().sign(SAMPLE_CODE, priv)

        restored = PluginManifest.from_json(signed.to_json())
        assert restored.signature == signed.signature
        assert restored.signer_pubkey == signed.signer_pubkey
        assert restored.verify_signature(SAMPLE_CODE, {pub}) is True

    def test_generate_keypair_produces_valid_hex(self) -> None:
        """Keypair generation returns valid 32-byte hex strings."""
        priv, pub = PluginManifest.generate_keypair()
        assert len(bytes.fromhex(priv)) == 32
        assert len(bytes.fromhex(pub)) == 32


# ---------------------------------------------------------------------------
# Client install with signature verification
# ---------------------------------------------------------------------------


class TestClientSignatureVerification:
    def test_client_install_rejects_untrusted_signer(self, tmp_path: Path) -> None:
        """trusted_pubkeys=set(other_pubkey), manifest signed by us -> refused."""
        priv, _pub = PluginManifest.generate_keypair()
        _other_priv, other_pub = PluginManifest.generate_keypair()

        _seed_signed_marketplace(tmp_path, SAMPLE_CODE, priv)

        install_dir = tmp_path / "installed"
        client = MarketplaceClient(LocalDirectorySource(tmp_path), install_dir=install_dir)

        # Only trust the other key, not ours
        result = client.install("demo", trusted_pubkeys={other_pub})
        assert result["ok"] is False
        assert "Signature verification failed" in result["error"]
        assert not (install_dir / "demo.py").exists()

    def test_client_install_accepts_trusted_signer(self, tmp_path: Path) -> None:
        """trusted_pubkeys includes ours -> accepted."""
        priv, pub = PluginManifest.generate_keypair()
        _seed_signed_marketplace(tmp_path, SAMPLE_CODE, priv)

        install_dir = tmp_path / "installed"
        client = MarketplaceClient(LocalDirectorySource(tmp_path), install_dir=install_dir)

        result = client.install("demo", trusted_pubkeys={pub})
        assert result["ok"] is True
        assert result["name"] == "demo"
        assert (install_dir / "demo.py").exists()

    def test_client_install_without_trusted_pubkeys_skips_verification(
        self, tmp_path: Path
    ) -> None:
        """Without trusted_pubkeys, signature verification is optional (backward compat)."""
        priv, _pub = PluginManifest.generate_keypair()
        _seed_signed_marketplace(tmp_path, SAMPLE_CODE, priv)

        install_dir = tmp_path / "installed"
        client = MarketplaceClient(LocalDirectorySource(tmp_path), install_dir=install_dir)

        # No trusted_pubkeys -> install succeeds regardless of signature
        result = client.install("demo")
        assert result["ok"] is True

    def test_client_install_unsigned_plugin_fails_when_pubkeys_required(
        self, tmp_path: Path
    ) -> None:
        """Unsigned plugin + trusted_pubkeys set -> refused."""
        _priv, pub = PluginManifest.generate_keypair()

        # Seed unsigned manifest
        manifest = _base_manifest()
        checksum = PluginManifest.compute_checksum(SAMPLE_CODE)
        unsigned = PluginManifest(
            name=manifest.name,
            version=manifest.version,
            author=manifest.author,
            description=manifest.description,
            entry_point=manifest.entry_point,
            checksum_sha256=checksum,
        )
        plugin_dir = tmp_path / "demo"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "manifest.json").write_text(unsigned.to_json())
        (plugin_dir / "demo.py").write_bytes(SAMPLE_CODE)

        install_dir = tmp_path / "installed"
        client = MarketplaceClient(LocalDirectorySource(tmp_path), install_dir=install_dir)

        result = client.install("demo", trusted_pubkeys={pub})
        assert result["ok"] is False
        assert "Signature verification failed" in result["error"]
