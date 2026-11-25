# Copyright (c) Alibaba, Inc. and its affiliates.
"""Approval grant and audit stores."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from leapflow.layout import ApprovalLayout
from leapflow.security.actions import ActionDescriptor
from leapflow.utils.file_lock import lock_fd, unlock_fd


class ApprovalScope(str, Enum):
    """How long an approval decision applies."""

    ONCE = "once"
    TURN = "turn"
    SESSION = "session"
    PROFILE = "profile"
    GLOBAL = "global"


@dataclass(frozen=True)
class ApprovalGrant:
    """A reusable approval or deny decision."""

    key: str
    scope: str
    decision: str
    action_kind: str
    effect: str
    resource: str = ""
    reason: str = ""
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    binding: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: float | None = None) -> bool:
        return self.expires_at is not None and (now or time.time()) >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApprovalGrant":
        return cls(
            key=str(data.get("key") or ""),
            scope=str(data.get("scope") or ApprovalScope.SESSION.value),
            decision=str(data.get("decision") or "allow"),
            action_kind=str(data.get("action_kind") or ""),
            effect=str(data.get("effect") or ""),
            resource=str(data.get("resource") or ""),
            reason=str(data.get("reason") or ""),
            created_at=float(data.get("created_at") or time.time()),
            expires_at=data.get("expires_at"),
            binding=dict(data.get("binding") or {}),
        )


@runtime_checkable
class ApprovalGrantStore(Protocol):
    """Storage abstraction for reusable approval decisions."""

    def get(self, key: str) -> ApprovalGrant | None: ...
    def put(self, grant: ApprovalGrant) -> None: ...
    def list(self) -> list[ApprovalGrant]: ...


@runtime_checkable
class RevocableApprovalGrantStore(ApprovalGrantStore, Protocol):
    """Atomic consent writes bound to the revocation state observed before prompting."""

    def revision(self, binding: dict[str, Any]) -> tuple[str, ...]: ...
    def put_if_current(self, grant: ApprovalGrant, revision: tuple[str, ...]) -> bool: ...
    def revoke_where(
        self, predicate: Callable[[ApprovalGrant], bool], *, binding: dict[str, Any],
    ) -> list[ApprovalGrant]: ...


class InMemoryApprovalGrantStore:
    """Session-local grant store."""

    def __init__(self) -> None:
        self._grants: dict[str, ApprovalGrant] = {}
        self._revocations: dict[str, tuple[dict[str, Any], str]] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> ApprovalGrant | None:
        with self._lock:
            grant = self._grants.get(key)
            if grant is not None and grant.is_expired():
                self._grants.pop(key, None)
                return None
            return grant

    def put(self, grant: ApprovalGrant) -> None:
        with self._lock:
            self._grants[grant.key] = grant

    def list(self) -> list[ApprovalGrant]:
        with self._lock:
            now = time.time()
            expired = [key for key, grant in self._grants.items() if grant.is_expired(now)]
            for key in expired:
                self._grants.pop(key, None)
            return list(self._grants.values())

    def remove_where(self, predicate: Callable[[ApprovalGrant], bool]) -> list[ApprovalGrant]:
        with self._lock:
            removed = [grant for grant in self.list() if predicate(grant)]
            for grant in removed:
                self._grants.pop(grant.key, None)
            return removed

    def revision(self, binding: dict[str, Any]) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(revision for selector, revision in self._revocations.values()
                                if all(binding.get(key) == value for key, value in selector.items())))

    def put_if_current(self, grant: ApprovalGrant, revision: tuple[str, ...]) -> bool:
        with self._lock:
            if self.revision(grant.binding) != revision:
                return False
            self.put(grant)
            return True

    def revoke_where(
        self, predicate: Callable[[ApprovalGrant], bool], *, binding: dict[str, Any],
    ) -> list[ApprovalGrant]:
        if not binding:
            raise ValueError("Revocation requires a nonempty binding selector")
        with self._lock:
            removed = self.remove_where(predicate)
            key = json.dumps(binding, sort_keys=True, separators=(",", ":"))
            self._revocations[key] = (dict(binding), uuid.uuid4().hex)
            return removed


class JsonApprovalGrantStore(InMemoryApprovalGrantStore):
    """Profile-local JSON grant store used before DuckDB approval tables exist."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path
        self._stamp: tuple[int, int, int] | None = None
        self._load()

    def _file_stamp(self) -> tuple[int, int, int] | None:
        try:
            stat = self._path.stat()
            return stat.st_mtime_ns, stat.st_size, stat.st_ino
        except FileNotFoundError:
            return None

    def _refresh(self) -> None:
        if self._file_stamp() != self._stamp:
            self._load()

    def get(self, key: str) -> ApprovalGrant | None:
        with self._lock:
            self._refresh()
            return super().get(key)

    def list(self) -> list[ApprovalGrant]:
        with self._lock:
            self._refresh()
            return super().list()

    def put(self, grant: ApprovalGrant) -> None:
        self._mutate(lambda: super(JsonApprovalGrantStore, self).put(grant))

    def remove_where(self, predicate: Callable[[ApprovalGrant], bool]) -> list[ApprovalGrant]:
        return self._mutate(lambda: super(JsonApprovalGrantStore, self).remove_where(predicate))

    def revision(self, binding: dict[str, Any]) -> tuple[str, ...]:
        with self._lock:
            self._refresh()
            return super().revision(binding)

    def put_if_current(self, grant: ApprovalGrant, revision: tuple[str, ...]) -> bool:
        def commit() -> bool:
            if super(JsonApprovalGrantStore, self).revision(grant.binding) != revision:
                return False
            super(JsonApprovalGrantStore, self).put(grant)
            return True

        return self._mutate(commit)

    def revoke_where(
        self, predicate: Callable[[ApprovalGrant], bool], *, binding: dict[str, Any],
    ) -> list[ApprovalGrant]:
        def revoke() -> list[ApprovalGrant]:
            removed = super(JsonApprovalGrantStore, self).remove_where(predicate)
            if not binding:
                raise ValueError("Revocation requires a nonempty binding selector")
            key = json.dumps(binding, sort_keys=True, separators=(",", ":"))
            self._revocations[key] = (dict(binding), uuid.uuid4().hex)
            return removed

        return self._mutate(revoke)

    def _mutate(self, operation: Callable[[], Any]) -> Any:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = ApprovalLayout(self._path.parent).grants_lock_path
            with lock_path.open("a+b") as handle:
                lock_fd(handle)
                try:
                    self._load(strict=True)
                    result = operation()
                    self._save()
                    return result
                except (OSError, ValueError, TypeError):
                    self._load()
                    raise
                finally:
                    unlock_fd(handle)

    def _load(self, *, strict: bool = False) -> None:
        # Capture the stamp before reading so a racing replacement forces another refresh.
        self._stamp = self._file_stamp()
        self._grants.clear()
        self._revocations.clear()
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            revocations = {}
            if isinstance(payload, dict):
                records = payload.get("revocations")
                if not isinstance(records, list):
                    raise ValueError("Invalid approval revocations")
                for record in records:
                    if not isinstance(record, dict):
                        raise ValueError("Invalid approval revocation")
                    selector, revision = record.get("binding"), record.get("revision")
                    if not isinstance(selector, dict) or not selector or not isinstance(revision, str) or not revision:
                        raise ValueError("Invalid approval revocation")
                    key = json.dumps(selector, sort_keys=True, separators=(",", ":"))
                    revocations[key] = (selector, revision)
                payload = payload.get("grants")
            if not isinstance(payload, list):
                raise ValueError("Approval grants must be a list")
            loaded = {}
            for item in payload:
                if not isinstance(item, dict):
                    raise ValueError("Invalid approval grant")
                grant = ApprovalGrant.from_dict(item)
                if grant.key and not grant.is_expired():
                    loaded[grant.key] = grant
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as exc:
            if strict:
                raise ValueError("Approval grant store is unreadable; refusing to overwrite it") from exc
            return
        self._grants = loaded
        self._revocations = revocations

    def _save(self) -> None:
        grants = [grant.to_dict() for grant in super().list()]
        payload = ({"grants": grants, "revocations": [
            {"binding": binding, "revision": revision} for binding, revision in self._revocations.values()
        ]} if self._revocations else grants)
        with tempfile.TemporaryDirectory(dir=self._path.parent) as directory:
            temporary = Path(directory) / "grants.json"
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, self._path)
        self._stamp = self._file_stamp()


class ApprovalAuditLog:
    """Append-only JSONL audit trail for approval decisions."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._entries: list[dict[str, Any]] = []

    @property
    def entries(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._entries)

    def get(self, action_id: str) -> dict[str, Any] | None:
        """Resolve a unique durable approval without accepting a caller-supplied log."""
        if not action_id:
            return None
        entries = list(self._entries)
        if self._path is not None and self._path.exists():
            entries = []
            with self._path.open("rb") as handle:
                for line in iter(lambda: handle.readline(65537), b""):
                    if len(line) > 65536:
                        raise ValueError("Approval audit entry exceeds its bound")
                    entry = json.loads(line)
                    if not isinstance(entry, dict):
                        raise ValueError("Invalid approval audit entry")
                    if entry.get("action_id") == action_id:
                        entries.append(entry)
                        if len(entries) > 1:
                            return None
        matches = [entry for entry in entries if entry.get("action_id") == action_id]
        return dict(matches[0]) if len(matches) == 1 else None

    def record(
        self,
        *,
        action: ActionDescriptor,
        decision: str,
        risk_level: str,
        risk_reasons: tuple[str, ...] = (),
        scope: str = ApprovalScope.ONCE.value,
        actor: str = "user",
        reason: str = "",
    ) -> None:
        entry = {
            "ts": time.time(),
            "action_id": action.action_id,
            "session_id": action.session_id,
            "turn_id": action.turn_id,
            "tool_call_id": action.tool_call_id,
            "action_kind": action.kind,
            "effect": action.effect,
            "resource": action.resource,
            "decision": decision,
            "scope": scope,
            "actor": actor,
            "risk_level": risk_level,
            "risk_reasons": list(risk_reasons),
            "reason": reason,
            "detail": action.detail[:500],
        }
        binding = action.metadata.get("approval_binding")
        if isinstance(binding, dict):
            entry["binding"] = dict(binding)
        self._entries.append(entry)
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def grant_key(action: ActionDescriptor, scope: ApprovalScope | str) -> str:
    """Build a grant key that avoids coarse category-wide approvals."""
    scope_value = scope.value if isinstance(scope, ApprovalScope) else str(scope)
    session = action.session_id if scope_value in {ApprovalScope.SESSION.value, ApprovalScope.TURN.value} else ""
    turn = action.turn_id if scope_value == ApprovalScope.TURN.value else ""
    return ":".join(
        part for part in (
            scope_value,
            session,
            turn,
            action.kind,
            action.effect,
            action.signature(),
        ) if part
    )
