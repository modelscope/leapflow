"""JSON-RPC 2.0 protocol types and LeapService interface for leapd.

The protocol layer defines:

1. **Message types** — JSON-RPC request, response, notification, and error
2. **LeapService** — the abstract service interface that both in-process
   and daemon modes implement
3. **Method registry** — ``domain.action`` naming convention

Wire format: newline-delimited JSON over Unix socket.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Protocol, runtime_checkable


class ErrorCode(IntEnum):
    """Standard JSON-RPC 2.0 error codes plus application-specific ones."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Application-specific (reserved range: -32000 to -32099)
    DATABASE_LOCKED = -32001
    SESSION_NOT_FOUND = -32002
    SKILL_NOT_FOUND = -32003
    CANCELLED = -32004


@dataclass(frozen=True)
class RpcError:
    """JSON-RPC error object."""
    code: int
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d


@dataclass(frozen=True)
class RpcRequest:
    """JSON-RPC 2.0 request."""
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    jsonrpc: str = "2.0"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RpcRequest:
        return cls(
            method=data["method"],
            params=data.get("params", {}),
            id=data.get("id", uuid.uuid4().hex[:12]),
        )


@dataclass(frozen=True)
class RpcResponse:
    """JSON-RPC 2.0 response (success or error)."""
    id: str
    result: Optional[Any] = None
    error: Optional[RpcError] = None
    jsonrpc: str = "2.0"

    def to_json(self) -> str:
        d: Dict[str, Any] = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            d["error"] = self.error.to_dict()
        else:
            d["result"] = self.result
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def success(cls, request_id: str, result: Any = None) -> RpcResponse:
        return cls(id=request_id, result=result)

    @classmethod
    def fail(cls, request_id: str, code: int, message: str, data: Any = None) -> RpcResponse:
        return cls(id=request_id, error=RpcError(code=code, message=message, data=data))


@dataclass(frozen=True)
class RpcNotification:
    """JSON-RPC 2.0 notification (no id, no response expected)."""
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    jsonrpc: str = "2.0"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass(frozen=True)
class StreamChunk:
    """A single streaming chunk for daemon engine events."""
    request_id: str
    content: str
    done: bool = False
    event_type: Literal[
        "chunk",
        "final",
        "tool_start",
        "tool_complete",
        "thinking",
        "status",
        "error",
        "approval_request",
        "approval_response",
    ] = "chunk"
    metadata: Optional[Dict[str, Any]] = None

    def to_notification(self) -> RpcNotification:
        params: Dict[str, Any] = {
            "id": self.request_id,
            "content": self.content,
            "done": self.done,
            "event_type": self.event_type,
        }
        if self.metadata:
            params["metadata"] = self.metadata
        return RpcNotification(method="stream.chunk", params=params)


# ══════════════════════════════════════════════════════════════════════
# LeapService — the contract between client and daemon
# ══════════════════════════════════════════════════════════════════════

@runtime_checkable
class LeapService(Protocol):
    """Service interface for leapd operations.

    Both in-process mode and daemon mode implement this protocol.
    Client code is agnostic to whether it talks to a local object
    or a remote daemon over Unix socket.
    """

    async def signal_record(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Record a signal (observation, action, event)."""
        ...

    async def memory_search(self, query: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memory across all providers."""
        ...

    async def memory_insert(self, content: str, kind: str = "fact", **kwargs: Any) -> str:
        """Insert a memory entry. Returns entry_id."""
        ...

    async def session_create(self, **kwargs: Any) -> Dict[str, Any]:
        """Create a new conversation session."""
        ...

    async def session_resume(self, session_id: str) -> Dict[str, Any]:
        """Resume an existing session."""
        ...

    async def engine_chat(self, message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Chat with the engine (streaming). Yields StreamChunks."""
        ...

    async def engine_cancel(self) -> bool:
        """Cancel the currently running engine task."""
        ...

    async def skill_execute(self, skill_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a skill by name."""
        ...

    async def scheduler_arm(self, task_config: Dict[str, Any]) -> str:
        """Arm a scheduled task. Returns task_id."""
        ...

    async def status(self) -> Dict[str, Any]:
        """Return daemon status (uptime, connections, db path, etc.)."""
        ...

    async def host_status(self) -> Dict[str, Any]:
        """Return host backend status."""
        ...

    async def host_start(self) -> Dict[str, Any]:
        """Start the host backend if available."""
        ...

    async def host_stop(self) -> Dict[str, Any]:
        """Stop the host backend and keep the daemon runtime alive."""
        ...

    async def host_restart(self) -> Dict[str, Any]:
        """Restart the host backend."""
        ...

    async def tools_list(self) -> Dict[str, Any]:
        """Return available tool groups for slash-command rendering."""
        ...

    async def usage_summary(self) -> Dict[str, Any]:
        """Return token usage for the current daemon session."""
        ...

    async def model_info(self, model_name: str = "") -> Dict[str, Any]:
        """Return active model information and switch guidance."""
        ...

    async def app_command(self, args: str = "") -> Dict[str, Any]:
        """Return an App Connector slash-command payload."""
        ...

    async def command_execute(self, name: str, args: str = "") -> Dict[str, Any]:
        """Execute any engine-routed slash command and return structured result."""
        ...

    async def approval_status(self) -> Dict[str, Any]:
        """Return pending approval requests."""
        ...

    async def approval_resolve(
        self,
        pending_id: str,
        decision: str,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Resolve a pending approval request."""
        ...

    async def approval_cancel(self, pending_id: str, reason: str = "cancelled") -> Dict[str, Any]:
        """Cancel a pending approval request."""
        ...

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        ...

    # ── Gateway ──────────────────────────────────────────────────────

    async def gateway_connect(
        self,
        platform: str,
        credentials: Dict[str, str],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Connect a platform via the gateway."""
        ...

    async def gateway_disconnect(self, platform: str) -> Dict[str, Any]:
        """Disconnect a platform."""
        ...

    async def gateway_status(self) -> List[Dict[str, Any]]:
        """Return status of all gateway platforms."""
        ...

    async def gateway_send(
        self,
        platform: str,
        chat_id: str,
        text: str,
        thread_id: str = "",
    ) -> Dict[str, Any]:
        """Send a message to a connected platform conversation."""
        ...


# ══════════════════════════════════════════════════════════════════════
# Method registry — maps JSON-RPC method names to LeapService methods
# ══════════════════════════════════════════════════════════════════════

METHOD_REGISTRY: Dict[str, str] = {
    "signal.record": "signal_record",
    "memory.search": "memory_search",
    "memory.insert": "memory_insert",
    "session.create": "session_create",
    "session.resume": "session_resume",
    "engine.chat": "engine_chat",
    "engine.cancel": "engine_cancel",
    "skill.execute": "skill_execute",
    "scheduler.arm": "scheduler_arm",
    "daemon.status": "status",
    "daemon.shutdown": "shutdown",
    "host.status": "host_status",
    "host.start": "host_start",
    "host.stop": "host_stop",
    "host.restart": "host_restart",
    "tools.list": "tools_list",
    "usage.summary": "usage_summary",
    "model.info": "model_info",
    "app.command": "app_command",
    "command.execute": "command_execute",
    "approval.status": "approval_status",
    "approval.resolve": "approval_resolve",
    "approval.cancel": "approval_cancel",
    "gateway.connect": "gateway_connect",
    "gateway.disconnect": "gateway_disconnect",
    "gateway.status": "gateway_status",
    "gateway.send": "gateway_send",
}
