"""Unified secret vault for LeapFlow credentials."""
from __future__ import annotations

import json
import os
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


_SECRET_PREFIX = "secret://"
_FERNET_PREFIX = "enc:fernet:"


@dataclass(frozen=True)
class SecretRecord:
    """Encrypted secret plus metadata."""

    ref: str
    value: str
    metadata: dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": self.ref,
            "value": self.value,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SecretRecord":
        return cls(
            ref=str(data.get("ref") or ""),
            value=str(data.get("value") or ""),
            metadata={str(k): str(v) for k, v in dict(data.get("metadata") or {}).items()},
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
        )


@runtime_checkable
class SecretVault(Protocol):
    """Secret storage protocol."""

    def get(self, ref: str) -> str | None: ...
    def set(self, ref: str, value: str, metadata: dict[str, str] | None = None) -> None: ...
    def delete(self, ref: str) -> None: ...
    def resolve_config_refs(self, value: object) -> object: ...


class FernetSecretVault:
    """Fernet-backed JSON secret vault."""

    def __init__(self, vault_path: Path, key_path: Path) -> None:
        self._vault_path = vault_path
        self._key_path = key_path
        self._fernet = None

    def get(self, ref: str) -> str | None:
        self._validate_ref(ref)
        record = self._load().get(ref)
        if record is None:
            return None
        return self._decrypt(record.value)

    def set(self, ref: str, value: str, metadata: dict[str, str] | None = None) -> None:
        self._validate_ref(ref)
        records = self._load()
        now = time.time()
        existing = records.get(ref)
        records[ref] = SecretRecord(
            ref=ref,
            value=self._encrypt(value),
            metadata=dict(metadata or {}),
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        self._save(records)

    def delete(self, ref: str) -> None:
        self._validate_ref(ref)
        records = self._load()
        records.pop(ref, None)
        self._save(records)

    def resolve_config_refs(self, value: object) -> object:
        if isinstance(value, str) and value.startswith(_SECRET_PREFIX):
            return self.get(value) or ""
        if isinstance(value, dict):
            return {key: self.resolve_config_refs(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.resolve_config_refs(item) for item in value]
        return value

    def rotate_key(self) -> None:
        records = self._load(decrypt=True)
        if self._key_path.exists():
            self._key_path.unlink()
        self._fernet = None
        rotated = {
            ref: SecretRecord(
                ref=record.ref,
                value=self._encrypt(record.value),
                metadata=record.metadata,
                created_at=record.created_at,
                updated_at=time.time(),
            )
            for ref, record in records.items()
        }
        self._save(rotated)

    def _load(self, *, decrypt: bool = False) -> dict[str, SecretRecord]:
        try:
            raw = json.loads(self._vault_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        records: dict[str, SecretRecord] = {}
        for ref, payload in raw.items():
            if not isinstance(payload, dict):
                continue
            record = SecretRecord.from_dict({"ref": ref, **payload})
            if decrypt:
                record = SecretRecord(
                    ref=record.ref,
                    value=self._decrypt(record.value),
                    metadata=record.metadata,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                )
            records[record.ref] = record
        return records

    def _save(self, records: dict[str, SecretRecord]) -> None:
        self._vault_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {ref: record.to_dict() for ref, record in sorted(records.items())}
        for record in payload.values():
            record.pop("ref", None)
        self._vault_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _ensure_private_permissions(self._vault_path)

    def _encrypt(self, value: str) -> str:
        fernet = self._ensure_fernet()
        token = fernet.encrypt(value.encode("utf-8")).decode("ascii")
        return _FERNET_PREFIX + token

    def _decrypt(self, value: str) -> str:
        if not value.startswith(_FERNET_PREFIX):
            raise RuntimeError("Secret vault contains unsupported plaintext or weakly encoded data")
        token = value[len(_FERNET_PREFIX):].encode("ascii")
        return self._ensure_fernet().decrypt(token).decode("utf-8")

    def _ensure_fernet(self):
        if self._fernet is not None:
            return self._fernet
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise RuntimeError("cryptography is required for persistent secret storage") from exc
        if self._key_path.exists():
            key = self._key_path.read_bytes().strip()
        else:
            key = Fernet.generate_key()
            self._key_path.parent.mkdir(parents=True, exist_ok=True)
            self._key_path.write_bytes(key)
            _ensure_private_permissions(self._key_path)
        self._fernet = Fernet(key)
        return self._fernet

    @staticmethod
    def _validate_ref(ref: str) -> None:
        if not ref.startswith(_SECRET_PREFIX):
            raise ValueError("Secret references must start with secret://")


def _ensure_private_permissions(path: Path) -> None:
    if os.name != "nt" and path.exists():
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def secret_ref(scope: str, *parts: str) -> str:
    """Build a normalized secret reference."""
    clean_parts = [part.strip("/") for part in parts if part.strip("/")]
    return _SECRET_PREFIX + "/".join([scope.strip("/"), *clean_parts])
