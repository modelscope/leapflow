"""Gateway protocol types and adapter interface.

Defines the contract between platform adapters and the gateway server.
All domain types are immutable (frozen dataclass) for safety at trust
boundaries.  ``PlatformAdapter`` uses ``typing.Protocol`` — no base-class
inheritance — so adapters satisfy the contract via structural subtyping.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)


# ═══════════════════════════════════════════════════════════════
# Inbound types (platform → gateway)
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MessageSource:
    """Deterministic identity for message origin.

    Platform-specific normalisation (WhatsApp JID, Feishu union_id, …)
    stays in the adapter layer — this type is platform-agnostic.
    """

    platform: str
    chat_id: str
    chat_type: str = "dm"
    user_id: str = ""
    user_name: str = ""
    thread_id: str = ""
    scope_id: str = ""
    profile: str = "default"


@dataclass(frozen=True)
class MediaAttachment:
    """A media item attached to a message."""

    url: str
    media_type: str = ""
    filename: str = ""
    size_bytes: int = 0


@dataclass(frozen=True)
class InboundMessage:
    """Platform-normalised inbound message."""

    source: MessageSource
    text: str
    message_id: str
    media: Tuple[MediaAttachment, ...] = ()
    reply_to_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════
# Outbound types (gateway → platform)
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SendTarget:
    """Where to send a message."""

    platform: str
    chat_id: str
    thread_id: str = ""
    reply_to_id: str = ""


@dataclass(frozen=True)
class OutboundContent:
    """Content to send to a platform.

    ``metadata`` serves as an extensibility escape-hatch: platform-specific
    data, WebUI control hints (cards, forms, buttons) — anything that the
    core doesn't need to understand.
    """

    text: str
    media: Tuple[MediaAttachment, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SendResult:
    """Result of a send operation."""

    ok: bool
    message_id: str = ""
    error: str = ""


# ═══════════════════════════════════════════════════════════════
# Platform adapter contract
# ═══════════════════════════════════════════════════════════════

MessageHandler = Callable[[InboundMessage], Awaitable[None]]


@runtime_checkable
class PlatformAdapter(Protocol):
    """Contract for a platform connection.

    Each adapter manages its own connection lifecycle.
    The gateway sets ``on_message`` before calling ``connect()``.

    Capability flags are declared as class-level attributes.  Callers
    read them via ``getattr()`` to determine platform-specific behaviour
    without ``isinstance`` checks.
    """

    @property
    def platform_id(self) -> str: ...

    # ── Capability flags (class-level declarations) ──────────
    supports_async_delivery: bool
    splits_long_messages: bool
    max_message_length: int

    # ── Message handler (set by server before connect) ───────
    @property
    def on_message(self) -> Optional[MessageHandler]: ...

    @on_message.setter
    def on_message(self, handler: MessageHandler) -> None: ...

    # ── Lifecycle ────────────────────────────────────────────
    async def connect(self, *, is_reconnect: bool = False) -> None:
        """Establish connection.  Raises on failure.

        When *is_reconnect* is ``True``, preserve server-side message
        queues (e.g. Telegram offset, Feishu subscription).
        """
        ...

    async def disconnect(self) -> None:
        """Cleanly disconnect from the platform."""
        ...

    # ── Messaging ────────────────────────────────────────────
    async def send(
        self,
        target: SendTarget,
        content: OutboundContent,
    ) -> SendResult:
        """Send a message to the platform."""
        ...


# ═══════════════════════════════════════════════════════════════
# Runtime status
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PlatformStatus:
    """Runtime status of a known platform."""

    platform_id: str
    connected: bool
    connected_since: float = 0.0
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
