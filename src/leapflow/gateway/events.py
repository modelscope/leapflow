"""Gateway event types for EventBus integration.

Gateway publishes events; MemoryManager, Copilot, and AgentEngine can
subscribe independently.  All types are frozen dataclasses — safe to
pass across asyncio tasks.

The current ``EventBus`` is perception-specific (``handle_event``).
These types define the contract for a future generalised pub/sub layer.
For Phase 1, ``GatewayServer`` uses an optional callback to notify
interested subscribers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Tuple

from leapflow.gateway.protocol import MessageSource


@dataclass(frozen=True)
class GatewayMessageReceived:
    """Inbound message from any external platform."""

    source: MessageSource
    session_key: str
    text: str
    media_urls: Tuple[str, ...] = ()
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class GatewaySessionCreated:
    """New session established through gateway."""

    session_key: str
    source: MessageSource
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class GatewaySessionEnded:
    """Session ended (reset / timeout / explicit close)."""

    session_key: str
    reason: str
    timestamp: float = field(default_factory=time.time)
