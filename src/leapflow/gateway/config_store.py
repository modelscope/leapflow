"""Gateway configuration persistence (``gateway.yaml``).

Reads and writes platform configurations. Manifest-declared secret fields are
stored in the profile secret vault and referenced from ``gateway.yaml`` as
``secret://`` refs; non-secret options remain inline.
Thread-safe via atomic write (write to temp file then ``os.replace``).
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from leapflow.gateway.credential_vault import CredentialVault, _ensure_file_permissions
from leapflow.gateway.manifest import PlatformManifest

logger = logging.getLogger(__name__)

_CONFIG_VERSION = 1


# ═══════════════════════════════════════════════════════════════
# Config domain types
# ═══════════════════════════════════════════════════════════════

@dataclass
class PlatformConfig:
    """Runtime configuration for a single platform."""

    enabled: bool = True
    credentials: Dict[str, str] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    configured_at: str = ""
    configured_by: str = "conversation"


@dataclass
class GatewayConfig:
    """Full gateway configuration read from / written to disk."""

    version: int = _CONFIG_VERSION
    platforms: Dict[str, PlatformConfig] = field(default_factory=dict)
    auto_connect: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Config store
# ═══════════════════════════════════════════════════════════════

class GatewayConfigStore:
    """Reads and writes gateway configuration with vault-backed credential refs."""

    def __init__(self, config_path: Path, vault: CredentialVault) -> None:
        self._config_path = config_path
        self._vault = vault
        self._manifests: Dict[str, PlatformManifest] = {}

    def set_manifests(self, manifests: Dict[str, PlatformManifest]) -> None:
        """Provide manifest lookup for secret ref decisions."""
        self._manifests = manifests

    # ── Read ─────────────────────────────────────────────────

    def load(self) -> GatewayConfig:
        """Load gateway config from disk.  Returns empty config if missing."""
        if not self._config_path.exists():
            return GatewayConfig()
        try:
            import yaml

            raw = yaml.safe_load(
                self._config_path.read_text(encoding="utf-8"),
            ) or {}
        except Exception:
            logger.warning(
                "Failed to parse gateway.yaml, using defaults", exc_info=True,
            )
            return GatewayConfig()

        config = GatewayConfig(version=raw.get("version", _CONFIG_VERSION))
        for pid, pdata in raw.get("platforms", {}).items():
            config.platforms[pid] = PlatformConfig(
                enabled=pdata.get("enabled", True),
                credentials=pdata.get("credentials", {}),
                options=pdata.get("options", {}),
                configured_at=pdata.get("configured_at", ""),
                configured_by=pdata.get("configured_by", "manual"),
            )
        config.auto_connect = raw.get("auto_connect", [])
        return config

    # ── Write (atomic) ───────────────────────────────────────

    def save(self, config: GatewayConfig) -> None:
        """Atomically write gateway config to disk."""
        import yaml

        raw: Dict[str, Any] = {
            "version": config.version,
            "platforms": {},
            "auto_connect": config.auto_connect,
        }
        for pid, pc in config.platforms.items():
            raw["platforms"][pid] = {
                "enabled": pc.enabled,
                "credentials": pc.credentials,
                "options": pc.options,
                "configured_at": pc.configured_at,
                "configured_by": pc.configured_by,
            }

        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._config_path.parent),
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                yaml.safe_dump(
                    raw, fh, default_flow_style=False, allow_unicode=True,
                )
            os.replace(tmp_path, str(self._config_path))
            _ensure_file_permissions(self._config_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ── Platform-level convenience ───────────────────────────

    def save_platform(
        self,
        platform_id: str,
        credentials: Dict[str, str],
        options: Dict[str, Any],
        manifest: PlatformManifest,
    ) -> None:
        """Save or update a single platform's configuration."""
        config = self.load()

        secret_keys = frozenset(
            c.key for c in manifest.credentials if c.secret
        )
        stored_credentials = self._vault.store_credentials(platform_id, credentials, secret_keys)

        config.platforms[platform_id] = PlatformConfig(
            enabled=True,
            credentials=stored_credentials,
            options=options,
            configured_at=datetime.now(timezone.utc).isoformat(),
            configured_by="conversation",
        )
        if platform_id not in config.auto_connect:
            config.auto_connect.append(platform_id)

        self.save(config)

    def load_platform_credentials(
        self,
        platform_id: str,
        manifest: PlatformManifest,
    ) -> Optional[Dict[str, str]]:
        """Load and resolve credentials for a platform.

        Environment variables ``LEAPFLOW_<PLATFORM>_<KEY>`` (uppercased)
        take precedence over file-stored refs, enabling container and
        CI/CD deployments without touching ``gateway.yaml``.

        Returns ``None`` if the platform is not configured (neither file
        nor env vars provide credentials).
        """
        config = self.load()
        pc = config.platforms.get(platform_id)

        if pc is None:
            creds = self._load_from_env(platform_id, manifest)
            return creds if creds else None
        if not pc.credentials:
            creds = self._load_from_env(platform_id, manifest)
            if creds:
                return creds
            has_required_credentials = any(c.required for c in manifest.credentials)
            return None if has_required_credentials else {}

        secret_keys = frozenset(
            c.key for c in manifest.credentials if c.secret
        )
        result = self._vault.load_credentials(platform_id, pc.credentials, secret_keys)

        env_overrides = self._load_from_env(platform_id, manifest)
        if env_overrides:
            result.update(env_overrides)

        return result

    @staticmethod
    def _load_from_env(
        platform_id: str,
        manifest: PlatformManifest,
    ) -> Dict[str, str]:
        """Check for ``LEAPFLOW_<PLATFORM>_<KEY>`` environment overrides."""
        prefix = f"LEAPFLOW_{platform_id.upper()}_"
        overrides: Dict[str, str] = {}
        for cred in manifest.credentials:
            env_key = prefix + cred.key.upper()
            env_val = os.environ.get(env_key)
            if env_val:
                overrides[cred.key] = env_val
        return overrides

    def remove_platform(self, platform_id: str) -> None:
        """Remove a platform from configuration."""
        config = self.load()
        config.platforms.pop(platform_id, None)
        if platform_id in config.auto_connect:
            config.auto_connect.remove(platform_id)
        self.save(config)
