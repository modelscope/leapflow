"""App connector protocols for platform action execution.

This module defines the platform-neutral contract used by REST, CLI, and
future MCP backends. Platform-specific packages provide action specs; backend
implementations only execute those specs and return normalized results.

It also defines the event normalisation contract used by the gateway consumer
loop to classify raw backend events into domain types (Message, Callback,
Signal, Lifecycle, Ignored).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, Mapping, Protocol, Sequence, runtime_checkable


class BackendKind(str, Enum):
    """Supported execution backend kinds."""

    REST = "rest"
    CLI = "cli"
    MCP = "mcp"


@dataclass(frozen=True)
class ActionAuthSpec:
    """Platform-neutral authorization contract for an action."""

    identities: tuple[str, ...] = ()
    scopes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    resource_fields: tuple[str, ...] = ()
    recovery: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionFailure:
    """Structured action failure used for recovery and approval gating."""

    failure_class: str
    failure_code: str
    message: str
    recoverability: str = "retryable"
    retryable: bool = True
    recovery_hint: str = ""
    next_steps: tuple[str, ...] = ()
    required_scopes: tuple[str, ...] = ()
    missing_scopes: tuple[str, ...] = ()
    requested_scopes: tuple[str, ...] = ()
    granted_scopes: tuple[str, ...] = ()
    identity: str = ""
    console_url: str = ""
    capability: str = ""
    blocks_approval: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict)
    # ── Scope authority metadata (permission recovery contract) ──────────
    # scope_relation describes how the listed scopes combine:
    #   "all_required" — every listed scope must be granted (conjunction)
    #   "one_of"        — any single listed scope is sufficient (disjunction)
    # scope_source describes where the scope list came from, in descending
    # trust order:
    #   "authoritative" — extracted from the upstream API's own error payload
    #                      (e.g. lark-cli typed PermissionError.MissingScopes)
    #   "declared"       — derived from this action's own manifest/action-pack
    #                      auth.scopes contract (feishu.yaml, etc.)
    #   "unverified"     — inferred from free-text/heuristic matching; must
    #                      never be surfaced as a concrete scope list to users
    scope_relation: str = "all_required"
    scope_source: str = "declared"

    def as_dict(self) -> dict[str, Any]:
        """Return safe non-empty fields for tool results and UI metadata."""
        data: dict[str, Any] = {
            "failure_class": self.failure_class,
            "failure_code": self.failure_code,
            "recoverability": self.recoverability,
            "retryable": self.retryable,
            "blocks_approval": self.blocks_approval,
        }
        optional: dict[str, Any] = {
            "recovery_hint": self.recovery_hint,
            "next_steps": list(self.next_steps),
            "required_scopes": list(self.required_scopes),
            "missing_scopes": list(self.missing_scopes),
            "requested_scopes": list(self.requested_scopes),
            "granted_scopes": list(self.granted_scopes),
            "identity": self.identity,
            "console_url": self.console_url,
            "capability": self.capability,
        }
        for key, value in optional.items():
            if value:
                data[key] = value
        if data.get("required_scopes") or data.get("missing_scopes"):
            data["scope_relation"] = self.scope_relation
            data["scope_source"] = self.scope_source
        return data


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
    capability: str = ""
    auth: ActionAuthSpec = field(default_factory=ActionAuthSpec)
    recovery: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionResult:
    """Backend-normalized action execution result."""

    ok: bool
    data: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""
    resource_id: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)
    failure: ActionFailure | None = None


@dataclass(frozen=True)
class ActionPreview:
    """Side-effect-free preview used by approval UI before execution."""

    ok: bool
    summary: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""
    failure: ActionFailure | None = None


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
class ActionDiscovery(Protocol):
    """Protocol for backends that support dynamic command discovery.

    Implementations run ``--help`` or equivalent introspection to discover
    available commands, converting them into draft :class:`ActionSpec`
    objects with conservative safety defaults (``risk_level="high"``).
    """

    async def discover_actions(
        self,
        *,
        groups: Sequence[str] = (),
    ) -> list[ActionSpec]:
        """Discover available actions, optionally scoped to command groups.

        When *groups* is empty, performs a full top-level discovery.
        When *groups* is provided, only discovers commands under those
        specific command groups (e.g. ``["im", "calendar"]``).
        """
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


# ═══════════════════════════════════════════════════════════════
# Event classification types (Orient stage)
# ═══════════════════════════════════════════════════════════════

class EventKind(str, Enum):
    """Classification of a backend event for routing."""

    MESSAGE = "message"
    CALLBACK = "callback"
    SIGNAL = "signal"
    LIFECYCLE = "lifecycle"
    IGNORED = "ignored"


@dataclass(frozen=True)
class InboundCallback:
    """Platform callback event (card button click, form submit, etc.).

    ``reply_token`` may have platform-specific TTL and usage limits
    (e.g. Feishu: 30 min, max 2 updates).
    """

    source: "MessageSource"
    callback_id: str
    action_type: str
    action_value: Mapping[str, Any] = field(default_factory=dict)
    original_message_id: str = ""
    reply_token: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class EventClassification:
    """Result of classifying a BackendEvent."""

    kind: EventKind
    message: "InboundMessage | None" = None
    callback: InboundCallback | None = None
    raw_event: BackendEvent | None = None


@runtime_checkable
class PlatformEventNormalizer(Protocol):
    """Classifies raw backend events into domain types.

    Each platform provides a concrete implementation that maps its
    event schema to the shared ``EventClassification`` type.
    """

    platform_id: str

    def classify(self, event: BackendEvent) -> EventClassification:
        """Classify and normalise a backend event."""
        ...

    def is_self_message(self, event: BackendEvent, bot_id: str) -> bool:
        """Return True if the event was produced by the bot itself."""
        ...
