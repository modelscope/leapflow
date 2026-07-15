"""Unified secret vault for LeapFlow credentials."""
from __future__ import annotations

import json
import os
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from leapflow.layout import PathLayout, ProfileLayout


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
    def list_refs(self) -> tuple[str, ...]: ...
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

    def list_refs(self) -> tuple[str, ...]:
        return tuple(sorted(self._load().keys()))

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
        secret_scope(ref)


def _ensure_private_permissions(path: Path) -> None:
    if os.name != "nt" and path.exists():
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


class ScopedSecretResolver:
    """Resolve secret refs across global and active profile vaults."""

    def __init__(self, layout: PathLayout, profile_layout: ProfileLayout) -> None:
        self._global_vault = FernetSecretVault(layout.global_secrets.vault_path, layout.global_secrets.key_path)
        self._profile_vault = FernetSecretVault(profile_layout.secrets.vault_path, profile_layout.secrets.key_path)

    def vault_for_ref(self, ref: str) -> FernetSecretVault:
        scope = secret_scope(ref)
        if scope == "global":
            return self._global_vault
        if scope == "profile":
            return self._profile_vault
        raise ValueError(f"Unsupported secret scope: {scope}")

    def get(self, ref: str) -> str | None:
        return self.vault_for_ref(ref).get(ref)

    def set(self, ref: str, value: str, metadata: dict[str, str] | None = None) -> None:
        self.vault_for_ref(ref).set(ref, value, metadata=metadata)

    def list_refs(self) -> tuple[str, ...]:
        refs = [*self._global_vault.list_refs(), *self._profile_vault.list_refs()]
        return tuple(sorted(set(refs)))

    def delete(self, ref: str) -> None:
        self.vault_for_ref(ref).delete(ref)


def secret_scope(ref: str) -> str:
    """Return the scope component from a normalized secret ref."""
    if not ref.startswith(_SECRET_PREFIX):
        raise ValueError("Secret references must start with secret://")
    rest = ref[len(_SECRET_PREFIX):]
    scope = rest.split("/", 1)[0]
    if scope not in {"global", "profile"}:
        raise ValueError("Secret references must use global or profile scope")
    return scope


def secret_ref(scope: str, *parts: str) -> str:
    """Build a normalized secret reference."""
    normalized_scope = scope.strip("/")
    if normalized_scope not in {"global", "profile"}:
        raise ValueError("Secret scope must be 'global' or 'profile'")
    clean_parts = [part.strip("/") for part in parts if part.strip("/")]
    if not clean_parts:
        raise ValueError("Secret references require at least one path part")
    return _SECRET_PREFIX + "/".join([normalized_scope, *clean_parts])
