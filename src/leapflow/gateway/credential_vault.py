"""Gateway credential refs backed by the unified LeapFlow secret vault.

Gateway config files store only ``secret://`` references for manifest-declared
secret fields. Actual credential values live in the active profile secret vault.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict
import logging

from leapflow.security.secrets import FernetSecretVault, secret_ref

logger = logging.getLogger(__name__)


def _ensure_file_permissions(path: Path) -> None:
    """Apply private permissions to gateway config files when present."""
    if path.exists():
        from leapflow.security.secrets import _ensure_private_permissions
        _ensure_private_permissions(path)


class CredentialVault:
    """Stores gateway secret fields as refs in the unified profile vault."""

    def __init__(self, secrets_dir: Path) -> None:
        self._vault = FernetSecretVault(secrets_dir / "vault.json", secrets_dir / "vault.key")

    def store_credentials(
        self,
        platform_id: str,
        credentials: Dict[str, str],
        secret_keys: frozenset,
    ) -> Dict[str, str]:
        """Persist secret fields and return config-safe credentials containing refs."""
        result: Dict[str, str] = {}
        for key, value in credentials.items():
            if key in secret_keys and value:
                ref = secret_ref("profile", "gateway", platform_id, key)
                self._vault.set(ref, value, metadata={"owner": "gateway", "platform": platform_id, "key": key})
                result[f"{key}_ref"] = ref
            elif key not in secret_keys:
                result[key] = value
        return result

    def load_credentials(
        self,
        platform_id: str,
        stored: Dict[str, str],
        secret_keys: frozenset,
    ) -> Dict[str, str]:
        """Resolve config-safe credential refs from the unified vault."""
        result: Dict[str, str] = {}
        for key, value in stored.items():
            if key.endswith("_ref"):
                continue
            if key not in secret_keys:
                result[key] = value
        for key in secret_keys:
            ref = stored.get(f"{key}_ref")
            if not isinstance(ref, str) or not ref.startswith("secret://"):
                continue
            resolved = self._vault.get(ref)
            if resolved:
                result[key] = resolved
            else:
                logger.warning("Missing gateway credential ref for %s.%s: %s", platform_id, key, ref)
        return result
