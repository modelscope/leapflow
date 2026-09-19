# Copyright (c) Alibaba, Inc. and its affiliates.
"""Content-addressed storage for large evolution artifacts.

The event store keeps bounded JSON facts. Generated source, model responses,
validation reports, and recordings live here and are referenced by digest. Writes
are atomic and immutable: writing identical bytes is idempotent; conflicting bytes
cannot share an address by construction.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ArtifactRef:
    """Stable reference to one immutable artifact."""

    digest: str
    size_bytes: int
    media_type: str
    privacy_class: str
    relative_path: str

    @property
    def artifact_id(self) -> str:
        return f"sha256:{self.digest}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "privacy_class": self.privacy_class,
            "relative_path": self.relative_path,
        }


class ArtifactIntegrityError(RuntimeError):
    """Raised when bytes at a content address do not match its digest."""


class ContentAddressedArtifactStore:
    """Profile-scoped immutable SHA-256 artifact store."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).expanduser().resolve()

    @property
    def root(self) -> Path:
        return self._root

    def put_bytes(
        self,
        content: bytes,
        *,
        media_type: str = "application/octet-stream",
        privacy_class: str = "system",
    ) -> ArtifactRef:
        """Store bytes atomically and return their content address."""
        payload = bytes(content)
        digest = hashlib.sha256(payload).hexdigest()
        relative = Path(digest[:2]) / digest
        target = self._root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            self._verify_path(target, digest)
        else:
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{digest}.", suffix=".tmp", dir=str(target.parent)
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                # Another writer may have won with the same digest; replacing it is
                # harmless because the bytes are content-addressed, but avoid a write
                # when possible so readers never see needless inode churn.
                if target.exists():
                    temporary.unlink(missing_ok=True)
                    self._verify_path(target, digest)
                else:
                    temporary.replace(target)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        return ArtifactRef(
            digest=digest,
            size_bytes=len(payload),
            media_type=str(media_type or "application/octet-stream"),
            privacy_class=str(privacy_class or "system"),
            relative_path=relative.as_posix(),
        )

    def put_text(
        self,
        content: str,
        *,
        media_type: str = "text/plain; charset=utf-8",
        privacy_class: str = "system",
    ) -> ArtifactRef:
        return self.put_bytes(
            str(content).encode("utf-8"),
            media_type=media_type,
            privacy_class=privacy_class,
        )

    def put_json(
        self,
        value: Mapping[str, Any] | list[Any],
        *,
        privacy_class: str = "system",
    ) -> ArtifactRef:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        return self.put_text(
            text + "\n",
            media_type="application/json",
            privacy_class=privacy_class,
        )

    def resolve(self, ref: ArtifactRef | str) -> Path:
        """Resolve a digest/ref below the store root without accepting a path."""
        digest = (
            ref.digest if isinstance(ref, ArtifactRef) else str(ref).removeprefix("sha256:")
        ).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("artifact reference must be a SHA-256 digest")
        return self._root / digest[:2] / digest

    def get_bytes(self, ref: ArtifactRef | str) -> bytes:
        """Read an artifact and verify that its immutable address is still valid."""
        path = self.resolve(ref)
        payload = path.read_bytes()
        expected = ref.digest if isinstance(ref, ArtifactRef) else str(ref).removeprefix("sha256:")
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected.lower():
            raise ArtifactIntegrityError(
                f"artifact digest mismatch: expected {expected.lower()}, got {actual}"
            )
        return payload

    def get_text(self, ref: ArtifactRef | str, *, encoding: str = "utf-8") -> str:
        return self.get_bytes(ref).decode(encoding)

    @staticmethod
    def _verify_path(path: Path, expected_digest: str) -> None:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_digest:
            raise ArtifactIntegrityError(
                f"artifact digest mismatch: expected {expected_digest}, got {actual}"
            )


__all__ = ["ArtifactIntegrityError", "ArtifactRef", "ContentAddressedArtifactStore"]
