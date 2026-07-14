"""Trigger policy for inbound IM messages.

Controls which inbound messages activate the agent's Decide stage.
Analogous to Attention Layer 0 foreground gating in desktop perception.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from leapflow.gateway.protocol import InboundMessage


class TriggerMode(str, Enum):
    """Trigger activation mode for inbound messages."""

    MENTION_ONLY = "mention_only"
    ALL = "all"
    KEYWORD = "keyword"
    MANUAL = "manual"


@dataclass(frozen=True)
class TriggerPolicy:
    """Immutable policy that decides whether an inbound message
    should activate agent processing.
    """

    mode: TriggerMode = TriggerMode.MENTION_ONLY
    allowed_chats: frozenset[str] = frozenset()
    blocked_chats: frozenset[str] = frozenset()
    blocked_users: frozenset[str] = frozenset()
    keywords: tuple[str, ...] = ()
    max_events_per_minute: int = 30
    cooldown_per_chat_s: float = 1.0

    def should_activate(
        self,
        message: InboundMessage,
        *,
        rate_tracker: _RateTracker | None = None,
    ) -> bool:
        """Evaluate whether a message should trigger agent processing."""
        source = message.source

        if source.chat_id in self.blocked_chats:
            return False
        if source.user_id in self.blocked_users:
            return False
        if self.allowed_chats and source.chat_id not in self.allowed_chats:
            return False

        if rate_tracker is not None and not rate_tracker.allow(
            source.chat_id,
            max_per_minute=self.max_events_per_minute,
            cooldown_s=self.cooldown_per_chat_s,
        ):
            return False

        if self.mode is TriggerMode.ALL:
            return True
        if self.mode is TriggerMode.MANUAL:
            return False
        if self.mode is TriggerMode.KEYWORD:
            text_lower = message.text.lower()
            return any(kw.lower() in text_lower for kw in self.keywords)

        return self._is_mention_or_dm(message)

    def _is_mention_or_dm(self, message: InboundMessage) -> bool:
        """Check if the message is a DM or contains a bot mention.

        Mention detection is the normalizer's responsibility — the
        normalizer sets ``metadata["bot_mentioned"]`` during event
        classification.  TriggerPolicy only reads that flag.
        """
        if message.source.chat_type in ("dm", "p2p", "private"):
            return True

        metadata = message.metadata or {}
        return bool(metadata.get("bot_mentioned"))


class _RateTracker:
    """Sliding-window rate limiter per chat (not persisted)."""

    def __init__(self) -> None:
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._last_pass: dict[str, float] = {}

    def allow(
        self,
        chat_id: str,
        *,
        max_per_minute: int,
        cooldown_s: float,
    ) -> bool:
        now = time.monotonic()

        if cooldown_s > 0:
            last = self._last_pass.get(chat_id, 0.0)
            if now - last < cooldown_s:
                return False

        if max_per_minute > 0:
            window = self._windows[chat_id]
            cutoff = now - 60.0
            window[:] = [t for t in window if t > cutoff]
            if len(window) >= max_per_minute:
                return False
            window.append(now)

        self._last_pass[chat_id] = now
        return True
