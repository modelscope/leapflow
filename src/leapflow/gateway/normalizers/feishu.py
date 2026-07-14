"""Feishu/Lark event normalizer.

Maps the flat NDJSON output of ``lark-cli event consume`` into the
shared ``EventClassification`` types.  lark-cli already resolves
@-mentions inline and unwraps message content, so the normalizer
works with pre-processed fields.

NDJSON fields (lark-cli ``ImMessageReceiveOutput``):

    type, event_id, timestamp, message_id, create_time,
    chat_id, chat_type, message_type, sender_id, content
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Mapping

from leapflow.gateway.connectors.protocol import (
    BackendEvent,
    EventClassification,
    EventKind,
    InboundCallback,
)
from leapflow.gateway.protocol import InboundMessage, MediaAttachment, MessageSource

logger = logging.getLogger(__name__)

_AT_ALL_PATTERN = re.compile(r"@(?:all(?:\b|$)|所有人)", re.IGNORECASE)

_MESSAGE_EVENT_TYPES = frozenset({
    "im.message.receive_v1",
})

_CALLBACK_EVENT_TYPES = frozenset({
    "card.action.trigger",
})

_SIGNAL_EVENT_TYPES = frozenset({
    "im.message.reaction.created_v1",
    "im.message.reaction.deleted_v1",
    "im.message.read_v1",
})

_LIFECYCLE_EVENT_TYPES = frozenset({
    "im.chat.member.bot.added_v1",
    "im.chat.member.bot.deleted_v1",
    "im.chat.updated_v1",
    "im.chat.disbanded_v1",
})

_MEDIA_MESSAGE_TYPES = frozenset({"image", "file", "audio", "video", "media", "sticker"})


class FeishuEventNormalizer:
    """Normalizes Feishu backend events into domain types.

    Satisfies the ``PlatformEventNormalizer`` protocol via structural
    subtyping (no base class).
    """

    platform_id = "feishu"

    def __init__(
        self,
        *,
        bot_id: str = "",
        bot_name: str = "",
        profile: str = "default",
    ) -> None:
        self._bot_id = bot_id
        self._bot_name = bot_name
        self._profile = profile

    @property
    def bot_id(self) -> str:
        return self._bot_id

    @bot_id.setter
    def bot_id(self, value: str) -> None:
        self._bot_id = value

    @property
    def bot_name(self) -> str:
        return self._bot_name

    @bot_name.setter
    def bot_name(self, value: str) -> None:
        self._bot_name = value

    def is_self_message(self, event: BackendEvent, bot_id: str) -> bool:
        """Check if the event was produced by the bot itself."""
        sender = _get_sender_id(event.payload)
        return bool(bot_id and sender and sender == bot_id)

    def classify(self, event: BackendEvent) -> EventClassification:
        """Classify a Feishu backend event into a domain type."""
        event_type = event.event_type

        if event_type in _MESSAGE_EVENT_TYPES:
            if self.is_self_message(event, self._bot_id):
                return EventClassification(kind=EventKind.IGNORED)
            message = self._to_inbound_message(event)
            if message is None:
                return EventClassification(kind=EventKind.IGNORED, raw_event=event)
            return EventClassification(kind=EventKind.MESSAGE, message=message)

        if event_type in _CALLBACK_EVENT_TYPES:
            callback = self._to_inbound_callback(event)
            return EventClassification(
                kind=EventKind.CALLBACK, callback=callback, raw_event=event,
            )

        if event_type in _SIGNAL_EVENT_TYPES:
            return EventClassification(kind=EventKind.SIGNAL, raw_event=event)

        if event_type in _LIFECYCLE_EVENT_TYPES:
            return EventClassification(kind=EventKind.LIFECYCLE, raw_event=event)

        return EventClassification(kind=EventKind.SIGNAL, raw_event=event)

    def _to_inbound_message(self, event: BackendEvent) -> InboundMessage | None:
        """Convert a message event to InboundMessage.

        lark-cli emits **flat** NDJSON — fields are top-level, not
        nested under ``event.message`` / ``event.sender``.
        """
        payload = event.payload
        content = str(payload.get("content", ""))
        message_type = str(payload.get("message_type", "text"))

        media = self._extract_media(payload, message_type)

        if not content and not media:
            return None

        chat_type_raw = str(payload.get("chat_type", ""))
        chat_type = "dm" if chat_type_raw in ("p2p", "private", "") else "group"

        bot_mentioned = self._detect_bot_mention(content) if content else False

        source = MessageSource(
            platform="feishu",
            chat_id=str(payload.get("chat_id", "")),
            chat_type=chat_type,
            user_id=str(payload.get("sender_id", "")),
            profile=self._profile,
        )

        create_time = payload.get("create_time") or payload.get("timestamp", "")
        try:
            timestamp = int(create_time) / 1000.0 if create_time else time.time()
        except (ValueError, TypeError):
            timestamp = time.time()

        if not content and media:
            content = f"[{message_type}]"

        metadata: dict[str, Any] = {
            "event_id": event.event_id,
            "message_type": message_type,
        }
        if bot_mentioned:
            metadata["bot_mentioned"] = True

        message_id = str(payload.get("message_id") or payload.get("id", ""))
        parent_id = str(payload.get("parent_id") or payload.get("root_id") or "")

        return InboundMessage(
            source=source,
            text=content,
            message_id=message_id,
            reply_to_id=parent_id or None,
            media=tuple(media),
            metadata=metadata,
            timestamp=timestamp,
        )

    @staticmethod
    def _extract_media(
        payload: Mapping[str, Any], message_type: str,
    ) -> list[MediaAttachment]:
        """Extract media attachments from non-text message types.

        Returns attachment metadata with ``file_key`` for deferred
        download via ``im.download_resource``.
        """
        if message_type == "text":
            return []

        attachments: list[MediaAttachment] = []
        message_id = str(payload.get("message_id") or payload.get("id", ""))

        if message_type not in _MEDIA_MESSAGE_TYPES:
            return []

        file_key = str(payload.get("file_key") or payload.get("image_key") or "")
        filename = str(payload.get("file_name") or payload.get("filename") or "")
        size = 0
        try:
            size = int(payload.get("file_size") or payload.get("size") or 0)
        except (ValueError, TypeError):
            pass

        if file_key:
            attachments.append(MediaAttachment(
                url=f"feishu://resource/{message_id}/{file_key}",
                media_type=message_type,
                filename=filename,
                size_bytes=size,
            ))

        return attachments

    def _to_inbound_callback(self, event: BackendEvent) -> InboundCallback:
        """Parse a card.action.trigger payload into InboundCallback.

        Feishu card action payloads (flat NDJSON from lark-cli) contain:
        - open_message_id: the card message that was clicked
        - open_chat_id: chat where the card lives
        - action: {tag, value, ...} — the button/form action
        - token: reply token for updating the card
        """
        payload = event.payload
        action = payload.get("action", {})
        if isinstance(action, str):
            action = {}

        action_value = action.get("value", {})
        if isinstance(action_value, str):
            action_value = {"raw": action_value}

        chat_id = str(payload.get("open_chat_id", ""))
        user_id = str(payload.get("open_id", ""))
        chat_type_raw = str(payload.get("chat_type", ""))
        chat_type = "dm" if chat_type_raw in ("p2p", "private", "") else "group"

        source = MessageSource(
            platform="feishu",
            chat_id=chat_id,
            chat_type=chat_type,
            user_id=user_id,
            profile=self._profile,
        )

        return InboundCallback(
            source=source,
            callback_id=str(event.event_id),
            action_type=str(action.get("tag", "button")),
            action_value=action_value,
            original_message_id=str(payload.get("open_message_id", "")),
            reply_token=str(payload.get("token", "")),
            metadata={
                "event_type": event.event_type,
                "operator_name": str(payload.get("operator_name", "")),
            },
        )

    def _detect_bot_mention(self, content: str) -> bool:
        """Detect bot @-mention or @all in pre-rendered content text.

        lark-cli resolves mentions inline as ``@DisplayName``.
        ``@all`` / ``@所有人`` are treated as a bot mention since
        they address everyone in the chat including the bot.
        """
        if _AT_ALL_PATTERN.search(content):
            return True
        if not self._bot_name:
            return False
        return f"@{self._bot_name}" in content


def _get_sender_id(payload: Mapping[str, Any]) -> str:
    """Extract sender open_id from flat or nested event payload."""
    sender_id = payload.get("sender_id")
    if isinstance(sender_id, str):
        return sender_id
    if isinstance(sender_id, dict):
        return str(sender_id.get("open_id", ""))
    sender = payload.get("sender")
    if isinstance(sender, dict):
        sid = sender.get("sender_id")
        if isinstance(sid, dict):
            return str(sid.get("open_id", ""))
        if isinstance(sid, str):
            return sid
    return ""
