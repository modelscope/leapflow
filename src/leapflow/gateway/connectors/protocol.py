"""App connector protocols for platform action execution.

This module defines the platform-neutral contract used by REST, CLI, and
future MCP backends. Platform-specific packages provide action specs; backend
implementations only execute those specs and return normalized results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Mapping, Protocol, runtime_checkable


class BackendKind(str, Enum):
    """Supported execution backend kinds."""

    REST = "rest"
    CLI = "cli"
    MCP = "mcp"


@dataclass(frozen=True)
class ActionSpec:
    """Declarative description of one platform action."""

    name: str
    backend_kind: str
    description: str = ""
    effect: str = "execute"
    schema: Mapping[str, Any] = field(default_factory=dict)
    backend_config: Mapping[str, Any] = field(default_factory=dict)
    risk_level: str = "medium"
    output_policy: str = "summary"


@dataclass(frozen=True)
class ActionResult:
    """Backend-normalized action execution result."""

    ok: bool
    data: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""
    resource_id: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionPreview:
    """Side-effect-free preview used by approval UI before execution."""

    ok: bool
    summary: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass(frozen=True)
class BackendEvent:
    """Backend-normalized inbound event."""

    event_id: str
    event_type: str
    platform_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    checkpoint: str = ""


@dataclass(frozen=True)
class EventSourceStatus:
    """Lifecycle status for a backend event source."""

    ok: bool
    backend_kind: str
    detail: str = ""
    checkpoint: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendStatus:
    """Status of a backend for one configured platform/profile."""

    ok: bool
    backend_kind: str
    detail: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ExecutionBackend(Protocol):
    """Protocol implemented by all execution forms: REST, CLI, MCP, etc."""

    kind: str

    async def status(self) -> BackendStatus:
        """Return backend availability and authentication status."""
        ...

    async def authenticate(self, payload: Mapping[str, Any]) -> BackendStatus:
        """Run backend-specific authentication or return guidance."""
        ...

    async def execute(
        self,
        spec: ActionSpec,
        payload: Mapping[str, Any],
    ) -> ActionResult:
        """Execute a registered action spec with validated payload."""
        ...

    async def preview(
        self,
        spec: ActionSpec,
        payload: Mapping[str, Any],
    ) -> ActionPreview:
        """Return a side-effect-free preview for approval UI."""
        ...


@runtime_checkable
class BackendEventSource(Protocol):
    """Protocol implemented by CLI, REST, webhook, or MCP event sources."""

    platform_id: str
    backend_kind: str

    async def start(self, *, checkpoint: str = "") -> EventSourceStatus:
        """Start receiving events from the backend."""
        ...

    async def stop(self) -> EventSourceStatus:
        """Stop receiving events from the backend."""
        ...

    async def events(self) -> AsyncIterator[BackendEvent]:
        """Yield normalized inbound events."""
        ...

    async def status(self) -> EventSourceStatus:
        """Return current event source health."""
        ...


@runtime_checkable
class AppConnector(Protocol):
    """Platform-level connector that exposes actions through one backend."""

    platform_id: str

    def action_specs(self) -> Mapping[str, ActionSpec]:
        """Return available actions keyed by domain.operation."""
        ...

    def action_spec(self, action: str) -> ActionSpec | None:
        """Return one registered action spec, if available."""
        ...

    async def preview_action(
        self,
        action: str,
        payload: Mapping[str, Any],
    ) -> ActionPreview:
        """Return a side-effect-free action preview."""
        ...

    def event_source(self) -> BackendEventSource | None:
        """Return the inbound event source, if this connector supports one."""
        ...

    async def execute_action(
        self,
        action: str,
        payload: Mapping[str, Any],
    ) -> ActionResult:
        """Validate and execute a platform action."""
        ...
