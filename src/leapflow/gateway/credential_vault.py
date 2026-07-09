"""Credential encryption for gateway platform credentials.

Defence layers implemented here:

1. Fernet (AES-128-CBC + HMAC-SHA256) encryption at rest for secret fields
2. File permissions ``0600`` on ``gateway.yaml`` and key file
3. Graceful degradation to base64 if ``cryptography`` is not installed

Additional layers (file-read denial, tool-result sanitisation, log
redaction) are implemented at their respective trust boundaries.
"""
from __future__ import annotations

import base64
import logging
import os
import stat
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_FERNET_PREFIX = "enc:fernet:"
_B64_PREFIX = "enc:b64:"
_ENC_PREFIXES = (_FERNET_PREFIX, _B64_PREFIX)


def _ensure_file_permissions(path: Path) -> None:
    """Set file permissions to 0600 (owner read/write only)."""
    if path.exists() and os.name != "nt":
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


class CredentialVault:
    """Encrypts and decrypts platform credentials at rest.

    Only fields declared ``secret: true`` in the platform manifest are
    encrypted.  Non-secret fields (e.g. ``app_id``) remain plaintext so
    users can inspect ``gateway.yaml`` without tools.
    """

    def __init__(self, profile_dir: Path) -> None:
        self._key_path = profile_dir / ".credential_key"
        self._fernet: Optional[Any] = None
        self._fallback_mode = False

    # ── Key management ───────────────────────────────────────

    def _ensure_key(self) -> None:
        """Load or generate the encryption key (lazy)."""
        if self._fernet is not None or self._fallback_mode:
            return
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            logger.warning(
                "cryptography package not installed; credentials will use "
                "base64 encoding only.  Install 'cryptography' for AES encryption.",
            )
            self._fallback_mode = True
            return

        if self._key_path.exists():
            key = self._key_path.read_bytes().strip()
        else:
            key = Fernet.generate_key()
            self._key_path.parent.mkdir(parents=True, exist_ok=True)
            self._key_path.write_bytes(key)
            _ensure_file_permissions(self._key_path)

        self._fernet = Fernet(key)

    # ── Single-value operations ──────────────────────────────

    def encrypt_value(self, plaintext: str) -> str:
        """Encrypt a single credential value.  Returns prefixed ciphertext."""
        self._ensure_key()
        if self._fernet is not None:
            token = self._fernet.encrypt(plaintext.encode("utf-8"))
            return _FERNET_PREFIX + token.decode("ascii")
        return _B64_PREFIX + base64.b64encode(
            plaintext.encode("utf-8"),
        ).decode("ascii")

    def decrypt_value(self, stored: str) -> str:
        """Decrypt a single stored credential value.

        If decryption fails (corrupted key file or tampered ciphertext),
        a clear error is logged with recovery guidance.
        """
        if stored.startswith(_FERNET_PREFIX):
            self._ensure_key()
            if self._fernet is None:
                raise RuntimeError(
                    "Fernet-encrypted value present but cryptography not installed",
                )
            token = stored[len(_FERNET_PREFIX):].encode("ascii")
            try:
                return self._fernet.decrypt(token).decode("utf-8")
            except Exception as exc:
                logger.error(
                    "Credential decryption failed (key file may be corrupted "
                    "or replaced).  Recovery: delete %s and reconfigure the "
                    "platform via 'gateway_connect'.  Error: %s",
                    self._key_path,
                    type(exc).__name__,
                )
                raise
        if stored.startswith(_B64_PREFIX):
            encoded = stored[len(_B64_PREFIX):]
            return base64.b64decode(encoded).decode("utf-8")
        return stored

    # ── Batch operations (dict-level) ────────────────────────

    def encrypt_credentials(
        self,
        credentials: Dict[str, str],
        secret_keys: frozenset,
    ) -> Dict[str, str]:
        """Encrypt secret fields in a credentials dict.  Idempotent."""
        result: Dict[str, str] = {}
        for key, value in credentials.items():
            if (
                key in secret_keys
                and value
                and not value.startswith(_ENC_PREFIXES)
            ):
                result[key] = self.encrypt_value(value)
            else:
                result[key] = value
        return result

    def decrypt_credentials(
        self,
        stored: Dict[str, str],
        secret_keys: frozenset,
    ) -> Dict[str, str]:
        """Decrypt secret fields from stored credentials."""
        result: Dict[str, str] = {}
        for key, value in stored.items():
            if key in secret_keys and value.startswith(_ENC_PREFIXES):
                result[key] = self.decrypt_value(value)
            else:
                result[key] = value
        return result
