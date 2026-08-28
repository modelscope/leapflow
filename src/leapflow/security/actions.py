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
    FILE_READ = "file.read"
    FILE_WRITE = "file.write"
    FILE_DELETE = "file.delete"
    GATEWAY_SEND = "gateway.send"
    PLATFORM_ACTION = "platform.action"
    SCHEDULER_ARM = "scheduler.arm"
    SKILL_EXECUTE = "skill.execute"
    SKILL_PROMOTE = "skill.promote"
    APP_INSTALL = "app.install"
    RUNTIME_CONFIGURE = "runtime.configure"
    NETWORK_FETCH = "network.fetch"
    WORKSPACE_ESCAPE = "workspace.escape"
    EXTERNAL_ACTION = "external.action"
    # A tool provided by an external MCP server. First-class rather than folded into
    # external.action for the same reason the device kinds are: classifiers dispatch on
    # ``kind``, and an unrecognized one only ever reaches the generic fallback, which
    # makes the tier an accident of that fallback's value instead of a decision.
    #
    # It is its own kind rather than a per-capability one because the MCP protocol does
    # not tell us what a tool does. What we know is where it came from, and that is the
    # honest basis for assessing it.
    MCP_TOOL = "mcp.tool"
    # Physical device operations. Each effect class is its own first-class kind
    # rather than a metadata field on one "device.write" kind, because
    # DefaultRiskClassifier and its peers dispatch on ``kind``: an unrecognized
    # one only ever reaches the generic fallback, which would make the tier an
    # accident of the fallback's value instead of a decision. The classes differ
    # by risk profile, not by device type -- actuating carries kinetic energy,
    # dispensing consumes an irreversible resource, configuring has thermal or
    # mechanical inertia, and reading has no effect at all.
    DEVICE_READ = "device.read"
    DEVICE_CONFIGURE = "device.configure"
    DEVICE_ACTUATE = "device.actuate"
    DEVICE_DISPENSE = "device.dispense"
    DEVICE_ESTOP = "device.estop"


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
    def workspace_escape(
        cls,
        path: str,
        *,
        operation: str,
        effect: str = ActionEffect.READ.value,
        detail: str = "",
        origin: str = ActionOrigin.AGENT_TOOL.value,
        metadata: dict[str, Any] | None = None,
    ) -> "ActionDescriptor":
        """Describe a tool reaching a path outside the active workspace.

        A first-class kind rather than a free-form string: ``DefaultRiskClassifier``
        dispatches on ``kind``, so an unrecognized one only ever reached the
        generic fallback. That made the tier an accident of the fallback's value
        instead of a decision, and gave a read the same weight as a write.

        *effect* is the operation's real effect (read / write / execute), which is
        what separates listing a sibling repo from writing into it.
        """
        merged = dict(metadata or {})
        merged.update({"operation": operation, "workspace_escape": True})
        return cls(
            kind=ActionKind.WORKSPACE_ESCAPE.value,
            summary=f"Allow {operation} outside the workspace: {path}",
            detail=detail or f"{operation} wants to access {path}, which is outside the active workspace.",
            effect=effect,
            resource=path,
            origin=origin,
            metadata=merged,
        )

    @classmethod
    def device(
        cls,
        *,
        kind: str,
        device_id: str,
        channel_id: str,
        quantity: str = "",
        value: Any = None,
        unit: str = "",
        envelope_band: str = "",
        location: str = "",
        reversible: bool = False,
        origin: str = ActionOrigin.AGENT_TOOL.value,
        metadata: dict[str, Any] | None = None,
    ) -> "ActionDescriptor":
        """Describe one physical device operation.

        ``resource`` is ``<device_id>:<channel_id>@<envelope_band>``, and that
        composition is the whole grant contract. Scoping reuse to the channel
        *and its declared band* is what makes "allow writes to this channel" a
        usable consent instead of a prompt per microlitre -- while ensuring that
        widening the declared envelope invalidates the narrower grant it was
        given under, rather than silently inheriting it. The commanded value is
        deliberately excluded (see ``_normalize_detail``); values outside the
        band never reach approval at all, because the risk classifier
        hardline-denies them first.

        ``location`` is included in the summary on purpose: in the physical
        world, *which* machine is safety information, and the summary is what a
        human reads before consenting.
        """
        merged = dict(metadata or {})
        merged.update(
            {
                "device_id": device_id,
                "channel_id": channel_id,
                "envelope_band": envelope_band,
                "reversible": reversible,
            }
        )
        if quantity:
            merged["quantity"] = quantity
        where = f" at {location}" if location else ""
        rendered = _render_device_value(value, unit)
        action_word = kind.rsplit(".", 1)[-1]
        resource = f"{device_id}:{channel_id}"
        if envelope_band:
            resource = f"{resource}@{envelope_band}"
        return cls(
            kind=kind,
            summary=f"{action_word.capitalize()} {device_id}.{channel_id}{where}",
            detail=(
                f"{action_word} {device_id}.{channel_id}"
                f"{f' to {rendered}' if rendered else ''}"
                f"{where}."
            ),
            effect=_DEVICE_EFFECTS.get(kind, ActionEffect.EXECUTE.value),
            resource=resource,
            origin=origin,
            metadata=merged,
        )

    @classmethod
    def mcp_tool(
        cls,
        *,
        server: str,
        tool: str,
        arguments: Any = None,
        description: str = "",
        read_only: bool = False,
        origin: str = ActionOrigin.AGENT_TOOL.value,
        metadata: dict[str, Any] | None = None,
    ) -> "ActionDescriptor":
        """Describe a call to a tool supplied by an external MCP server.

        ``resource`` is ``<server>:<tool>`` and the arguments are deliberately kept out
        of the grant identity (see ``_normalize_detail``): scoping consent to the tool
        rather than the payload is what makes "allow this tool for the session" a usable
        decision instead of a prompt per distinct argument -- the same reasoning already
        applied to ``network.fetch``.

        The server name leads the summary because it is the trust boundary. Which server
        a tool came from is the only thing a person can actually judge; the tool's own
        description was written by that server and cannot vouch for itself.
        """
        merged = dict(metadata or {})
        merged.update({"server": server, "tool": tool, "read_only": read_only})
        summary = f"Run MCP tool {tool} from server {server}"
        detail = f"MCP server {server!r} tool {tool!r}"
        if description.strip():
            # Truncated because this text is persisted to the approval audit log, and an
            # MCP description is attacker-controlled input of unbounded length.
            detail = f"{detail}: {' '.join(description.split())[:400]}"
        return cls(
            kind=ActionKind.MCP_TOOL.value,
            summary=summary,
            detail=detail,
            effect=ActionEffect.READ.value if read_only else ActionEffect.EXECUTE.value,
            resource=f"{server}:{tool}",
            origin=origin,
            metadata=merged,
        )

    @classmethod
    def file_read(
        cls,
        path: str,
        *,
        mode: str = "raw",
        metadata: dict[str, Any] | None = None,
    ) -> "ActionDescriptor":
        merged = dict(metadata or {})
        merged.update({"mode": mode})
        return cls(
            kind=ActionKind.FILE_READ.value,
            summary=f"Read file: {path}",
            detail=f"Read local file content from {path}",
            effect=ActionEffect.READ.value,
            resource=path,
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

    @classmethod
    def platform_action(
        cls,
        platform: str,
        action: str,
        payload: dict[str, Any],
        *,
        backend_kind: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "ActionDescriptor":
        """Describe a platform action before backend execution."""
        merged = dict(metadata or {})
        merged.update({
            "platform": platform,
            "action": action,
            "backend_kind": backend_kind,
        })
        effect = str(merged.get("effect") or _effect_for_platform_action(action))
        detail = json.dumps(payload, ensure_ascii=False, sort_keys=True)[:1000]
        return cls(
            kind=ActionKind.PLATFORM_ACTION.value,
            summary=f"Run {platform}.{action}",
            detail=detail,
            effect=effect,
            resource=f"{platform}:{action}",
            metadata=merged,
        )

    @classmethod
    def network_fetch(
        cls,
        url: str,
        *,
        origin: str,
        method: str = "GET",
        metadata: dict[str, Any] | None = None,
    ) -> "ActionDescriptor":
        """Describe an outbound HTTP read before it leaves the machine.

        ``resource`` is the origin rather than the full URL so a session grant
        means "this host is trusted for now" instead of expiring on the next
        path or query string, which would turn progressive trust into a prompt
        per request.
        """
        merged = dict(metadata or {})
        merged.update({"url": url, "method": method, "origin": origin})
        return cls(
            kind=ActionKind.NETWORK_FETCH.value,
            summary=f"Fetch {method} {origin}",
            detail=url,
            effect=ActionEffect.READ.value,
            resource=origin,
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


def _effect_for_platform_action(action: str) -> str:
    lowered = action.lower()
    if any(token in lowered for token in ("send", "reply", "message")):
        return ActionEffect.SEND.value
    if any(token in lowered for token in ("delete", "remove")):
        return ActionEffect.DELETE.value
    if any(token in lowered for token in ("create", "update", "write", "append", "approve")):
        return ActionEffect.WRITE.value
    return ActionEffect.READ.value


def _normalize_resource(resource: str) -> str:
    return resource.replace("\\", "/").strip().lower()


_DEVICE_KIND_PREFIX = "device."

_DEVICE_EFFECTS: dict[str, str] = {
    ActionKind.DEVICE_READ.value: ActionEffect.READ.value,
    ActionKind.DEVICE_CONFIGURE.value: ActionEffect.CONFIGURE.value,
    ActionKind.DEVICE_ACTUATE.value: ActionEffect.EXECUTE.value,
    ActionKind.DEVICE_DISPENSE.value: ActionEffect.EXECUTE.value,
    ActionKind.DEVICE_ESTOP.value: ActionEffect.EXECUTE.value,
}


def _render_device_value(value: Any, unit: str) -> str:
    """Render a commanded value for a human reading an approval prompt.

    Kept deliberately dumb: no rounding, no unit conversion. The prompt must
    show what will actually be sent, and this text is also persisted to the
    approval audit log.
    """
    if value is None:
        return ""
    rendered = str(value)
    return f"{rendered} {unit}".strip() if unit else rendered


def _normalize_detail(kind: str, detail: str) -> str:
    text = re.sub(r"\s+", " ", detail.strip())
    if kind in {ActionKind.GATEWAY_SEND.value, ActionKind.PLATFORM_ACTION.value}:
        return "<platform-payload>"
    if kind == ActionKind.NETWORK_FETCH.value:
        # Collapsed on purpose: the grant is scoped by origin (the resource), so
        # keeping the full URL here would mint a separate grant per path and
        # query string and re-prompt for every request to an approved host.
        return "<network-target>"
    if kind == ActionKind.MCP_TOOL.value:
        # Same reasoning: the grant is scoped by server:tool, so the arguments must not
        # enter the key or every distinct payload would re-prompt for a tool the user
        # already approved. The description is excluded too -- it is attacker-controlled
        # text from the server, and letting it shape grant identity would let a server
        # invalidate its own grants by rewording itself.
        return "<mcp-invocation>"
    if kind.startswith(_DEVICE_KIND_PREFIX):
        # Same reasoning as network.fetch, one step further: the grant is scoped
        # by device:channel (the resource) plus the declared envelope band, so
        # the commanded value must not enter the key or every distinct setpoint
        # would mint its own grant and re-prompt. Values outside the band never
        # reach approval -- the risk classifier hardline-denies them first, which
        # is what makes a band-wide consent safe rather than open-ended.
        return "<device-command>"
    return text[:4000]
