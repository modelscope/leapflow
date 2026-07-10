"""Approval grant and audit stores."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from leapflow.security.actions import ActionDescriptor


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
        )


@runtime_checkable
class ApprovalGrantStore(Protocol):
    """Storage abstraction for reusable approval decisions."""

    def get(self, key: str) -> ApprovalGrant | None: ...
    def put(self, grant: ApprovalGrant) -> None: ...
    def list(self) -> list[ApprovalGrant]: ...


class InMemoryApprovalGrantStore:
    """Session-local grant store."""

    def __init__(self) -> None:
        self._grants: dict[str, ApprovalGrant] = {}

    def get(self, key: str) -> ApprovalGrant | None:
        grant = self._grants.get(key)
        if grant is not None and grant.is_expired():
            self._grants.pop(key, None)
            return None
        return grant

    def put(self, grant: ApprovalGrant) -> None:
        self._grants[grant.key] = grant

    def list(self) -> list[ApprovalGrant]:
        now = time.time()
        expired = [key for key, grant in self._grants.items() if grant.is_expired(now)]
        for key in expired:
            self._grants.pop(key, None)
        return list(self._grants.values())


class JsonApprovalGrantStore(InMemoryApprovalGrantStore):
    """Profile-local JSON grant store used before DuckDB approval tables exist."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path
        self._load()

    def put(self, grant: ApprovalGrant) -> None:
        super().put(grant)
        self._save()

    def _load(self) -> None:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return
        if not isinstance(payload, list):
            return
        for item in payload:
            if isinstance(item, dict):
                grant = ApprovalGrant.from_dict(item)
                if grant.key and not grant.is_expired():
                    self._grants[grant.key] = grant

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [grant.to_dict() for grant in self.list()]
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class ApprovalAuditLog:
    """Append-only JSONL audit trail for approval decisions."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._entries: list[dict[str, Any]] = []

    @property
    def entries(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._entries)

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
