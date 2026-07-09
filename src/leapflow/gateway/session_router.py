"""Structured session routing for gateway messages.

``SessionKey`` is an immutable domain type that replaces simple string
concatenation.  Code logic accesses fields by name (``key.platform``,
``key.chat_id``); ``__str__()`` produces a colon-separated form **only**
for serialisation (DuckDB storage, dict keys, logs).

``build_session_key()`` is the single source of truth for routing —
platform-specific normalisation (WhatsApp JID, Feishu union_id, …)
stays in the adapter layer.
"""
from __future__ import annotations

from dataclasses import dataclass

from leapflow.gateway.protocol import MessageSource

_SINGLE_CHAR_UNSAFE = frozenset("/\\:")
_SUBSTRING_UNSAFE = ("..",)


@dataclass(frozen=True)
class SessionKey:
    """Structured session identifier.

    ``frozen=True`` makes it hashable — usable as a dict key directly via
    ``__hash__``.  ``__str__()`` is reserved for serialisation only.
    """

    profile: str
    platform: str
    chat_type: str
    chat_id: str
    thread_id: str = ""
    user_id: str = ""

    def __post_init__(self) -> None:
        for field_name in ("profile", "platform", "chat_type", "chat_id", "thread_id", "user_id"):
            value = getattr(self, field_name)
            normalized = "" if value is None else str(value)
            object.__setattr__(self, field_name, normalized)

        for field_name in ("profile", "platform", "chat_type", "chat_id", "thread_id", "user_id"):
            field_val = getattr(self, field_name)
            if not field_val:
                continue
            if any(c in field_val for c in _SINGLE_CHAR_UNSAFE):
                raise ValueError(
                    f"Unsafe characters in session key field {field_name}: {field_val!r}",
                )
            if any(sub in field_val for sub in _SUBSTRING_UNSAFE):
                raise ValueError(
                    f"Unsafe substring in session key field {field_name}: {field_val!r}",
                )

    def __str__(self) -> str:
        parts = [self.profile, self.platform, self.chat_type, self.chat_id]
        if self.thread_id:
            parts.append(self.thread_id)
        if self.user_id:
            parts.append(self.user_id)
        return ":".join(parts)


def build_session_key(
    source: MessageSource,
    *,
    per_user_in_group: bool = True,
) -> SessionKey:
    """Build a deterministic ``SessionKey`` from a message source.

    Group chats are isolated per-user by default (each user gets their
    own session).  DMs are keyed by ``chat_id`` (1:1 with user).
    Threads append ``thread_id`` to the parent key.
    """
    user_id = ""
    if (
        source.chat_type in ("group", "channel")
        and per_user_in_group
        and source.user_id
    ):
        user_id = source.user_id

    return SessionKey(
        profile=source.profile,
        platform=source.platform,
        chat_type=source.chat_type,
        chat_id=source.chat_id,
        thread_id=source.thread_id,
        user_id=user_id,
    )
