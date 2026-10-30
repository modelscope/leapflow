# Copyright (c) Alibaba, Inc. and its affiliates.
"""Plugin manifest format for marketplace distribution.

Supports Ed25519 signing for authenticity guarantees (in addition to
SHA-256 checksum for integrity).  Authors generate an Ed25519 keypair,
sign the manifest payload, and publish the public key.  Clients verify
signatures against a set of pre-configured trusted public keys.

Crypto backend: ``cryptography`` (Ed25519 — asymmetric, no shared-secret
distribution required).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Set
import hashlib
import json

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from cryptography.exceptions import InvalidSignature


@dataclass(frozen=True)
class PluginManifest:
    """Metadata describing a distributable plugin.

    A manifest is the contract between a plugin author and the marketplace.
    It carries enough information to discover, verify, and install a plugin.
    """
    name: str                       # unique plugin identifier
    version: str                    # semver
    author: str
    description: str
    entry_point: str                # module path within the package, e.g. "my_plugin"
    plugin_type: str = "tool"       # "tool" | "active_signal_source" | "gateway" | "llm"
    source_url: str = ""            # where to download the code (file:// or https://)
    checksum_sha256: str = ""       # integrity verification
    requires_sandbox: bool = True   # untrusted by default
    dependencies: List[str] = field(default_factory=list)  # other plugin names
    min_leapflow_version: str = ""
    signature: str = ""             # hex-encoded Ed25519 signature
    signer_pubkey: str = ""         # hex-encoded Ed25519 public key

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: str) -> "PluginManifest":
        raw = json.loads(data)
        # tolerate extra keys
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def verify_checksum(self, content: bytes) -> bool:
        """Verify content matches the declared checksum."""
        if not self.checksum_sha256:
            return False
        actual = hashlib.sha256(content).hexdigest()
        return actual == self.checksum_sha256

    @staticmethod
    def compute_checksum(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    # ------------------------------------------------------------------
    # Ed25519 signing / verification
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_payload(name: str, version: str, entry_point: str, checksum: str) -> bytes:
        """Build the deterministic bytes to sign/verify.

        Concatenates name, version, entry_point, and checksum_sha256
        separated by '|' — deterministic and order-stable.
        """
        return "|".join([name, version, entry_point, checksum]).encode("utf-8")

    def sign(self, code: bytes, private_key_hex: str) -> "PluginManifest":
        """Return a new manifest with Ed25519 signature over (metadata || checksum).

        The checksum is computed from *code*; the signature covers the
        canonical payload (name|version|entry_point|checksum_sha256).

        Args:
            code: The raw plugin source bytes to compute the checksum over.
            private_key_hex: Hex-encoded 32-byte Ed25519 private seed.

        Returns:
            A new PluginManifest with checksum_sha256, signature, and
            signer_pubkey populated.
        """
        checksum = self.compute_checksum(code)
        seed = bytes.fromhex(private_key_hex)
        private_key = Ed25519PrivateKey.from_private_bytes(seed)
        public_key = private_key.public_key()

        payload = self._canonical_payload(self.name, self.version, self.entry_point, checksum)
        sig = private_key.sign(payload)

        pubkey_bytes = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)

        # frozen dataclass — use object.__setattr__ replacement via new instance
        return PluginManifest(
            name=self.name,
            version=self.version,
            author=self.author,
            description=self.description,
            entry_point=self.entry_point,
            plugin_type=self.plugin_type,
            source_url=self.source_url,
            checksum_sha256=checksum,
            requires_sandbox=self.requires_sandbox,
            dependencies=list(self.dependencies),
            min_leapflow_version=self.min_leapflow_version,
            signature=sig.hex(),
            signer_pubkey=pubkey_bytes.hex(),
        )

    def verify_signature(self, code: bytes, trusted_pubkeys: Set[str]) -> bool:
        """Verify Ed25519 signature was made by a trusted signer.

        Args:
            code: The raw plugin source bytes (to recompute checksum).
            trusted_pubkeys: Set of hex-encoded public keys considered
                trusted.  If the manifest's signer_pubkey is not in this
                set, verification fails immediately.

        Returns:
            True only if the signature is valid AND the signer is trusted.
        """
        if not self.signature or not self.signer_pubkey:
            return False

        if self.signer_pubkey not in trusted_pubkeys:
            return False

        checksum = self.compute_checksum(code)
        payload = self._canonical_payload(self.name, self.version, self.entry_point, checksum)

        try:
            pubkey_bytes = bytes.fromhex(self.signer_pubkey)
            public_key = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
            public_key.verify(bytes.fromhex(self.signature), payload)
        except (InvalidSignature, ValueError):
            return False

        return True

    @staticmethod
    def generate_keypair() -> tuple[str, str]:
        """Generate a new Ed25519 keypair for plugin signing.

        Returns:
            (private_key_hex, public_key_hex) — 32-byte seed and 32-byte
            public key, both hex-encoded.
        """
        private_key = Ed25519PrivateKey.generate()
        seed = private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        pub = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return seed.hex(), pub.hex()
