"""Structured action descriptors for human approval decisions."""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ActionKind(str, Enum):
    """High-level action families that may require approval."""

    SHELL_COMMAND = "shell.command"
    FILE_WRITE = "file.write"
    FILE_DELETE = "file.delete"
    GATEWAY_SEND = "gateway.send"
    SCHEDULER_ARM = "scheduler.arm"
    SKILL_EXECUTE = "skill.execute"
    SKILL_PROMOTE = "skill.promote"
    APP_INSTALL = "app.install"
    RUNTIME_CONFIGURE = "runtime.configure"
    EXTERNAL_ACTION = "external.action"


class ActionEffect(str, Enum):
    """Observable effect of an action."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    SEND = "send"
    DELETE = "delete"
    CONFIGURE = "configure"
    SCHEDULE = "schedule"
    PROMOTE = "promote"


class ActionOrigin(str, Enum):
    """Where an action originated."""

    AGENT_TOOL = "agent_tool"
    SKILL = "skill"
    SCHEDULER = "scheduler"
    GATEWAY = "gateway"
    DAEMON = "daemon"
    USER = "user"


@dataclass(frozen=True)
class ActionDescriptor:
    """A normalized description of an operation before it mutates the world."""

    kind: str
    summary: str
    detail: str
    effect: str
    resource: str = ""
    origin: str = ActionOrigin.AGENT_TOOL.value
    action_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = ""
    turn_id: str = ""
    tool_call_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def shell(
        cls,
        command: str,
        *,
        cwd: str | None = None,
        origin: str = ActionOrigin.AGENT_TOOL.value,
        metadata: dict[str, Any] | None = None,
    ) -> "ActionDescriptor":
        merged = dict(metadata or {})
        if cwd:
            merged["cwd"] = cwd
        return cls(
            kind=ActionKind.SHELL_COMMAND.value,
            summary=_summarize_shell(command),
            detail=command,
            effect=ActionEffect.EXECUTE.value,
            resource=str(cwd or "shell"),
            origin=origin,
            metadata=merged,
        )

    @classmethod
    def file_write(
        cls,
        path: str,
        content: str,
        *,
        mode: str = "overwrite",
        metadata: dict[str, Any] | None = None,
    ) -> "ActionDescriptor":
        merged = dict(metadata or {})
        merged.update({"mode": mode, "bytes": len(content.encode("utf-8"))})
        preview = content[:500]
        return cls(
            kind=ActionKind.FILE_WRITE.value,
            summary=f"Write file: {path}",
            detail=preview,
            effect=ActionEffect.WRITE.value,
            resource=path,
            metadata=merged,
        )

    @classmethod
    def gateway_send(
        cls,
        platform: str,
        chat_id: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "ActionDescriptor":
        merged = dict(metadata or {})
        merged.update({"platform": platform, "chat_id": chat_id})
        return cls(
            kind=ActionKind.GATEWAY_SEND.value,
            summary=f"Send message to {platform}/{chat_id}",
            detail=text,
            effect=ActionEffect.SEND.value,
            resource=f"{platform}:{chat_id}",
            metadata=merged,
        )

    def signature(self) -> str:
        """Return a stable signature suitable for session/profile grants."""
        payload = {
            "kind": self.kind,
            "effect": self.effect,
            "resource": _normalize_resource(self.resource),
            "detail": _normalize_detail(self.kind, self.detail),
            "origin": self.origin,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionDescriptor":
        return cls(
            kind=str(data.get("kind") or ActionKind.EXTERNAL_ACTION.value),
            summary=str(data.get("summary") or "Action"),
            detail=str(data.get("detail") or ""),
            effect=str(data.get("effect") or ActionEffect.EXECUTE.value),
            resource=str(data.get("resource") or ""),
            origin=str(data.get("origin") or ActionOrigin.AGENT_TOOL.value),
            action_id=str(data.get("action_id") or uuid.uuid4().hex),
            session_id=str(data.get("session_id") or ""),
            turn_id=str(data.get("turn_id") or ""),
            tool_call_id=str(data.get("tool_call_id") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


def _summarize_shell(command: str) -> str:
    lowered = command.lower()
    if "<<" in command:
        return "Run script via heredoc"
    if "curl" in lowered or "wget" in lowered:
        return "Run shell command with network access"
    return "Run shell command"


def _normalize_resource(resource: str) -> str:
    return resource.replace("\\", "/").strip().lower()


def _normalize_detail(kind: str, detail: str) -> str:
    text = re.sub(r"\s+", " ", detail.strip())
    if kind == ActionKind.GATEWAY_SEND.value:
        return "<message>"
    return text[:4000]
