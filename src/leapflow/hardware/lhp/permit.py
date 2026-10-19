# Copyright (c) Alibaba, Inc. and its affiliates.
"""LHP v2 execution permit: Ed25519-signed, single-use write authorization.

An ``ExecutionPermit`` is the host's cryptographic proof that a human approved
a specific physical command. The node verifies it before any write precheck so
that a forged, expired, replayed, or misaddressed permit is rejected with
``side_effect_state=NONE`` -- the command provably never reached the device.

Design invariants:

- **Canonical digest**: ``request_digest`` is a deterministic sha256 of the
  method and every request parameter except the permit itself, so host and
  node derive it independently from what actually crosses the wire. A permit
  for ``write(joint.shoulder, 1.5)`` does not authorize
  ``write(joint.shoulder, 2.0)`` or ``write_batch(...)``. Whole-task methods
  (``skill.execute``) bind a content digest of the approved plan instead.

- **Single-use by default**: ``max_uses=1`` with nonce replay rejection.
  A captured permit cannot be replayed even within its TTL.

- **Audience binding**: the permit names the node it was issued for.
  A permit for ``jetson01`` is invalid on ``jetson02``.

- **No engine dependency**: this module imports only ``cryptography`` (already
  a project dependency for marketplace signing) and the stdlib.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
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

logger = logging.getLogger(__name__)

# ── Canonical action digest ──────────────────────────────────────────────

PERMIT_VERSION = 1
"""Permit envelope version; incremented on incompatible changes."""

MAX_PERMIT_TTL_S = 60.0
"""Upper bound for a permit's lifetime; longer values are clamped."""

DEFAULT_PERMIT_TTL_S = 10.0
"""Default TTL when no explicit value is configured."""

MAX_CLOCK_SKEW_S = 2.0
"""Maximum tolerated clock difference between host and node."""

MAX_NONCE_CACHE_SIZE = 4096
"""Upper bound for the replay-rejection LRU."""


PERMIT_PARAM = "permit"
"""The request parameter carrying the permit; never part of the digest it signs."""


def request_digest(method: str, params: Mapping[str, Any]) -> str:
    """Return the sha256 hex digest of one request as it crosses the wire.

    The parameters are normalized through a JSON round trip first (tuples become
    lists, keys become strings), so the host hashing the Python mapping it is
    about to send and the node hashing the decoded frame it received agree byte
    for byte. Batch commands keep their order: a reordered batch is a different
    action.
    """
    body = {str(key): value for key, value in params.items() if key != PERMIT_PARAM}
    normalized = json.loads(json.dumps(body, default=str))
    canonical = json.dumps(
        {"method": str(method), "params": normalized},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Permit data type ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExecutionPermit:
    """A signed, single-use authorization for a specific physical command."""

    v: int
    issuer: str  # human-readable issuer label
    key_id: str  # sha256(pub)[:16]
    audience: str  # node_id
    session_id: str
    workspace: str
    approval_id: str
    action_digest: str  # sha256 hex of canonical action
    issued_at: float  # unix timestamp
    expires_at: float  # unix timestamp
    nonce: str  # 16-byte hex
    max_uses: int = 1
    signature: str = ""  # hex-encoded Ed25519 signature

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": self.v,
            "issuer": self.issuer,
            "key_id": self.key_id,
            "audience": self.audience,
            "session_id": self.session_id,
            "workspace": self.workspace,
            "approval_id": self.approval_id,
            "action_digest": self.action_digest,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "nonce": self.nonce,
            "max_uses": self.max_uses,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecutionPermit:
        return cls(
            v=int(data.get("v") or PERMIT_VERSION),
            issuer=str(data.get("issuer") or ""),
            key_id=str(data.get("key_id") or ""),
            audience=str(data.get("audience") or ""),
            session_id=str(data.get("session_id") or ""),
            workspace=str(data.get("workspace") or ""),
            approval_id=str(data.get("approval_id") or ""),
            action_digest=str(data.get("action_digest") or ""),
            issued_at=float(data.get("issued_at") or 0.0),
            expires_at=float(data.get("expires_at") or 0.0),
            nonce=str(data.get("nonce") or ""),
            max_uses=int(data.get("max_uses") or 1),
            signature=str(data.get("signature") or ""),
        )

    def signable_payload(self) -> bytes:
        """Return the deterministic bytes that are signed/verified."""
        obj = {
            "v": self.v,
            "issuer": self.issuer,
            "key_id": self.key_id,
            "audience": self.audience,
            "session_id": self.session_id,
            "workspace": self.workspace,
            "approval_id": self.approval_id,
            "action_digest": self.action_digest,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "nonce": self.nonce,
            "max_uses": self.max_uses,
        }
        return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ── Permit rejection reasons ─────────────────────────────────────────────

REJECT_INVALID_SIGNATURE = "permit_invalid"
REJECT_EXPIRED = "permit_expired"
REJECT_REPLAYED = "permit_replayed"
REJECT_AUDIENCE_MISMATCH = "permit_audience_mismatch"
REJECT_DIGEST_MISMATCH = "permit_digest_mismatch"
REJECT_PRE_BOOT = "permit_pre_boot"


# ── Key utilities ────────────────────────────────────────────────────────


def key_id_from_public(pub: Ed25519PublicKey) -> str:
    """Derive the 16-hex-char key id from an Ed25519 public key."""
    raw = pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]


def generate_keypair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Generate a fresh Ed25519 keypair."""
    private = Ed25519PrivateKey.generate()
    return private, private.public_key()


def save_private_key(private: Ed25519PrivateKey, path: Path) -> None:
    """Write an Ed25519 private key in PEM format with 0600 permissions."""
    pem = private.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pem)
    os.chmod(path, 0o600)


def load_private_key(path: Path) -> Ed25519PrivateKey:
    """Load an Ed25519 private key from a PEM file."""
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    pem = path.read_bytes()
    key = load_pem_private_key(pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"key at {path} is not Ed25519")
    return key


def public_key_from_pem(pem_bytes: bytes) -> Ed25519PublicKey:
    """Load an Ed25519 public key from PEM bytes."""
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    key = load_pem_public_key(pem_bytes)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("key is not Ed25519")
    return key


def public_key_pem(pub: Ed25519PublicKey) -> bytes:
    """Export an Ed25519 public key as PEM bytes."""
    return pub.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)


# ── Issuer (host side) ───────────────────────────────────────────────────


class PermitIssuer:
    """Signs execution permits with an Ed25519 private key.

    The private key is stored in the profile layout at
    ``hardware/lhp_permit_issuer.key`` (0600).  On first use the key is
    generated and persisted; subsequent invocations reuse it.
    """

    def __init__(
        self,
        private_key: Ed25519PrivateKey,
        *,
        issuer_label: str = "leapflow_host",
        ttl_s: float = DEFAULT_PERMIT_TTL_S,
    ) -> None:
        self._private = private_key
        self._public = private_key.public_key()
        self._key_id = key_id_from_public(self._public)
        self._issuer = issuer_label
        self._ttl_s = min(max(1.0, float(ttl_s)), MAX_PERMIT_TTL_S)

    @property
    def key_id(self) -> str:
        return self._key_id

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self._public

    @classmethod
    def from_key_file(
        cls,
        path: Path,
        *,
        issuer_label: str = "leapflow_host",
        ttl_s: float = DEFAULT_PERMIT_TTL_S,
        auto_generate: bool = True,
    ) -> PermitIssuer:
        """Load or generate the permit issuer key at *path*."""
        if path.exists():
            private = load_private_key(path)
        elif auto_generate:
            private, _ = generate_keypair()
            save_private_key(private, path)
            logger.info(
                "Generated new LHP permit issuer key at %s (key_id=%s)",
                path,
                key_id_from_public(private.public_key()),
            )
        else:
            raise FileNotFoundError(f"permit issuer key not found at {path}")
        return cls(private, issuer_label=issuer_label, ttl_s=ttl_s)

    def issue(
        self,
        *,
        audience: str,
        session_id: str,
        workspace: str,
        approval_id: str,
        digest: str,
    ) -> ExecutionPermit:
        """Issue a signed single-use permit."""
        now = time.time()
        nonce = os.urandom(16).hex()
        permit = ExecutionPermit(
            v=PERMIT_VERSION,
            issuer=self._issuer,
            key_id=self._key_id,
            audience=audience,
            session_id=session_id,
            workspace=workspace,
            approval_id=approval_id,
            action_digest=digest,
            issued_at=now,
            expires_at=now + self._ttl_s,
            nonce=nonce,
            max_uses=1,
        )
        sig = self._private.sign(permit.signable_payload())
        return ExecutionPermit(**{**permit.to_dict(), "signature": sig.hex()})


# ── Verifier (node side) ─────────────────────────────────────────────────


@dataclass
class PermitRejection:
    """Structured rejection from the verifier; always side_effect=NONE."""

    code: str
    message: str


class PermitVerifier:
    """Verifies execution permits against a set of trusted public keys.

    The nonce replay cache is bounded and keyed by expiry time so that
    entries are evicted naturally without requiring a background task.
    After a node restart the cache is empty; permits issued before the
    boot time are rejected outright (``permit_pre_boot``).
    """

    def __init__(
        self,
        trusted_keys: Mapping[str, Ed25519PublicKey],
        *,
        node_id: str,
        boot_time: float | None = None,
        max_clock_skew_s: float = MAX_CLOCK_SKEW_S,
    ) -> None:
        self._trusted: dict[str, Ed25519PublicKey] = dict(trusted_keys)
        self._node_id = node_id
        self._boot_time = boot_time if boot_time is not None else time.time()
        self._max_skew = max(0.0, float(max_clock_skew_s))
        # nonce -> expires_at; OrderedDict for LRU eviction
        self._seen_nonces: OrderedDict[str, float] = OrderedDict()

    @property
    def has_trusted_keys(self) -> bool:
        return bool(self._trusted)

    @property
    def trusted_key_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._trusted))

    def verify(
        self,
        permit: ExecutionPermit,
        *,
        expected_digest: str,
    ) -> PermitRejection | None:
        """Return None on success, or a PermitRejection on failure."""
        now = time.time()

        # 1. Audience
        if permit.audience != self._node_id:
            return PermitRejection(
                REJECT_AUDIENCE_MISMATCH,
                f"permit audience {permit.audience!r} does not match node {self._node_id!r}",
            )

        # 2. Action digest
        if permit.action_digest != expected_digest:
            return PermitRejection(
                REJECT_DIGEST_MISMATCH,
                "permit action digest does not match the requested command",
            )

        # 3. Pre-boot rejection (restart invalidates all prior permits)
        if permit.issued_at < self._boot_time - self._max_skew:
            return PermitRejection(
                REJECT_PRE_BOOT,
                "permit was issued before node boot; start a new session",
            )

        # 4. Expiry with clock tolerance
        if now > permit.expires_at + self._max_skew:
            return PermitRejection(
                REJECT_EXPIRED,
                f"permit expired at {permit.expires_at:.1f}, now {now:.1f}",
            )

        # 5. Signature
        pub = self._trusted.get(permit.key_id)
        if pub is None:
            return PermitRejection(
                REJECT_INVALID_SIGNATURE,
                f"key_id {permit.key_id!r} is not trusted on this node",
            )
        try:
            pub.verify(bytes.fromhex(permit.signature), permit.signable_payload())
        except (InvalidSignature, ValueError):
            return PermitRejection(
                REJECT_INVALID_SIGNATURE,
                "permit signature verification failed",
            )

        # 6. Nonce replay
        self._evict_expired(now)
        if permit.nonce in self._seen_nonces:
            return PermitRejection(
                REJECT_REPLAYED,
                "permit nonce has already been consumed",
            )
        self._seen_nonces[permit.nonce] = permit.expires_at
        if len(self._seen_nonces) > MAX_NONCE_CACHE_SIZE:
            self._seen_nonces.popitem(last=False)

        return None

    def _evict_expired(self, now: float) -> None:
        """Remove nonces whose permits have expired beyond the skew window."""
        cutoff = now - self._max_skew
        while self._seen_nonces:
            oldest_nonce, oldest_expiry = next(iter(self._seen_nonces.items()))
            if oldest_expiry < cutoff:
                self._seen_nonces.pop(oldest_nonce)
            else:
                break


__all__ = [
    "DEFAULT_PERMIT_TTL_S",
    "ExecutionPermit",
    "MAX_CLOCK_SKEW_S",
    "MAX_NONCE_CACHE_SIZE",
    "MAX_PERMIT_TTL_S",
    "PERMIT_VERSION",
    "PermitIssuer",
    "PermitRejection",
    "PermitVerifier",
    "REJECT_AUDIENCE_MISMATCH",
    "REJECT_DIGEST_MISMATCH",
    "REJECT_EXPIRED",
    "REJECT_INVALID_SIGNATURE",
    "REJECT_PRE_BOOT",
    "REJECT_REPLAYED",
    "PERMIT_PARAM",
    "request_digest",
    "generate_keypair",
    "key_id_from_public",
    "load_private_key",
    "public_key_from_pem",
    "public_key_pem",
    "save_private_key",
]
