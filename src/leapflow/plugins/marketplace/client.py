"""Marketplace client: discover, download, verify, and install plugins.

Prototype uses a local directory as the marketplace source. The
MarketplaceSource abstraction allows swapping for an HTTP source later
without changing the client logic.

Installation flow:
    1. discover() → list available PluginManifests
    2. install(name) → download code, verify checksum, write to install dir
    3. The installed plugin is then loaded via the standard registry
       (sandboxed if requires_sandbox=True)

Security:
    - Checksum verification before install (integrity)
    - Ed25519 signature verification when trusted_pubkeys provided (authenticity)
    - requires_sandbox defaults True (untrusted execution isolation)
    - Install is a deliberate action requiring approval (not automatic)
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable

from leapflow.plugins.marketplace.manifest import PluginManifest

logger = logging.getLogger(__name__)


@runtime_checkable
class MarketplaceSource(Protocol):
    """Abstraction over where plugins are discovered/fetched from."""

    def list_manifests(self) -> List[PluginManifest]: ...
    def fetch_code(self, manifest: PluginManifest) -> Optional[bytes]: ...


class LocalDirectorySource:
    """A marketplace source backed by a local directory.

    Directory layout:
        <root>/
            <plugin_name>/
                manifest.json
                <entry_point>.py
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def list_manifests(self) -> List[PluginManifest]:
        manifests = []
        if not self._root.exists():
            return manifests
        for plugin_dir in self._root.iterdir():
            manifest_file = plugin_dir / "manifest.json"
            if manifest_file.exists():
                try:
                    manifests.append(PluginManifest.from_json(manifest_file.read_text()))
                except (ValueError, OSError) as exc:
                    logger.warning("Bad manifest in %s: %s", plugin_dir, exc)
        return manifests

    def fetch_code(self, manifest: PluginManifest) -> Optional[bytes]:
        code_file = self._root / manifest.name / f"{manifest.entry_point}.py"
        if not code_file.exists():
            return None
        try:
            return code_file.read_bytes()
        except OSError:
            return None


class MarketplaceClient:
    """Discovers and installs plugins from a MarketplaceSource."""

    def __init__(self, source: MarketplaceSource, install_dir: Path) -> None:
        self._source = source
        self._install_dir = Path(install_dir)

    def discover(self) -> List[PluginManifest]:
        """List all available plugins from the source."""
        return self._source.list_manifests()

    def resolve_manifest(self, name: str) -> Dict[str, Any] | None:
        """Return one manifest as a plain mapping for compatibility assessment."""
        from dataclasses import asdict

        manifest = next((item for item in self._source.list_manifests() if item.name == name), None)
        return asdict(manifest) if manifest is not None else None

    def install(
        self,
        name: str,
        *,
        verify: bool = True,
        trusted_pubkeys: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Download, verify, and install a plugin by name.

        Args:
            name: Plugin identifier to install.
            verify: When True, verify SHA-256 checksum (integrity).
            trusted_pubkeys: When provided, Ed25519 signature verification
                is MANDATORY.  The manifest's signer_pubkey must be in
                this set and the signature must be valid over the canonical
                payload.  If verification fails, install is refused.

        Returns a result dict with ok/error and the installed path.
        Does NOT auto-load the plugin — that's a separate approval-gated step.
        """
        manifests = {m.name: m for m in self._source.list_manifests()}
        manifest = manifests.get(name)
        if manifest is None:
            return {"ok": False, "error": f"Plugin '{name}' not found in marketplace"}

        code = self._source.fetch_code(manifest)
        if code is None:
            return {"ok": False, "error": f"Failed to fetch code for '{name}'"}

        # Integrity verification
        if verify and manifest.checksum_sha256:
            if not manifest.verify_checksum(code):
                return {
                    "ok": False,
                    "error": f"Checksum mismatch for '{name}' — refusing to install (integrity failure)",
                }

        # Authenticity verification (Ed25519 signature)
        if trusted_pubkeys is not None:
            if not manifest.verify_signature(code, trusted_pubkeys):
                return {
                    "ok": False,
                    "error": (
                        f"Signature verification failed for '{name}' — "
                        "refusing to install (authenticity failure)"
                    ),
                }

        # Write to install directory
        try:
            self._install_dir.mkdir(parents=True, exist_ok=True)
            target = self._install_dir / f"{manifest.entry_point}.py"
            target.write_bytes(code)
        except OSError as exc:
            return {"ok": False, "error": f"Install write failed: {exc}"}

        logger.info("Installed plugin '%s' v%s to %s", name, manifest.version, target)
        return {
            "ok": True,
            "name": name,
            "version": manifest.version,
            "installed_path": str(target),
            "requires_sandbox": manifest.requires_sandbox,
        }

    def uninstall(self, name: str) -> Dict[str, Any]:
        """Remove an installed plugin's files."""
        manifests = {m.name: m for m in self._source.list_manifests()}
        manifest = manifests.get(name)
        entry = manifest.entry_point if manifest else name
        target = self._install_dir / f"{entry}.py"
        if target.exists():
            try:
                target.unlink()
                return {"ok": True, "name": name}
            except OSError as exc:
                return {"ok": False, "error": str(exc)}
        return {"ok": False, "error": f"Plugin '{name}' not installed"}
