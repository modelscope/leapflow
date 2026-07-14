"""Telegram event normalizer.

Maps raw Telegram update payloads (from a ``BackendEventSource``)
into the shared ``EventClassification`` types.
"""
from __future__ import annotations

import time
from typing import Any, Mapping

from leapflow.gateway.connectors.protocol import (
    BackendEvent,
    EventClassification,
    EventKind,
)
from leapflow.gateway.protocol import InboundMessage, MediaAttachment, MessageSource


class TelegramEventNormalizer:
    """Normalizes Telegram BackendEvents into domain types.

    Satisfies the ``PlatformEventNormalizer`` protocol via structural subtyping.
    """

    platform_id = "telegram"

    def __init__(self, *, bot_id: str = "", bot_name: str = "", profile: str = "default") -> None:
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
        payload = event.payload
        raw_message = payload.get("message") or payload.get("edited_message") or {}
        if not isinstance(raw_message, dict):
            return False
        from_user = raw_message.get("from", {})
        if isinstance(from_user, dict):
            sender_id = str(from_user.get("id", ""))
            return bool(bot_id and sender_id == bot_id)
        return False

    def classify(self, event: BackendEvent) -> EventClassification:
        payload = event.payload
        if payload.get("message") or payload.get("edited_message"):
            if self.is_self_message(event, self._bot_id):
                return EventClassification(kind=EventKind.IGNORED)
            message = self._to_inbound_message(event)
            if message is None:
                return EventClassification(kind=EventKind.IGNORED, raw_event=event)
            return EventClassification(kind=EventKind.MESSAGE, message=message)

        if payload.get("callback_query"):
            return EventClassification(kind=EventKind.CALLBACK, raw_event=event)

        if payload.get("my_chat_member") or payload.get("chat_member"):
            return EventClassification(kind=EventKind.LIFECYCLE, raw_event=event)

        if payload.get("message_reaction") or payload.get("message_reaction_count"):
            return EventClassification(kind=EventKind.SIGNAL, raw_event=event)

        return EventClassification(kind=EventKind.SIGNAL, raw_event=event)

    def _to_inbound_message(self, event: BackendEvent) -> InboundMessage | None:
        payload = event.payload
        raw_message = payload.get("message") or payload.get("edited_message")
        if not isinstance(raw_message, dict):
            return None

        text = str(raw_message.get("text") or raw_message.get("caption") or "")
        media = self._extract_media(raw_message)

        if not text and not media:
            return None

        chat = raw_message.get("chat", {})
        user = raw_message.get("from", {})
        if not isinstance(chat, dict):
            chat = {}
        if not isinstance(user, dict):
            user = {}

        chat_id = str(chat.get("id", ""))
        chat_type_raw = str(chat.get("type", "private"))
        chat_type = "dm" if chat_type_raw == "private" else "group"
        user_id = str(user.get("id", ""))
        user_name = str(user.get("username") or user.get("first_name") or "")

        bot_mentioned = self._detect_bot_mention(raw_message)

        if not text and media:
            text = f"[{media[0].media_type}]"

        metadata: dict[str, Any] = {
            "update_id": str(payload.get("update_id", "")),
        }
        if bot_mentioned:
            metadata["bot_mentioned"] = True

        return InboundMessage(
            source=MessageSource(
                platform="telegram",
                chat_id=chat_id,
                chat_type=chat_type,
                user_id=user_id,
                user_name=user_name,
                profile=self._profile,
            ),
            text=text,
            message_id=str(raw_message.get("message_id", "")),
            media=tuple(media),
            metadata=metadata,
            timestamp=float(raw_message.get("date", time.time())),
        )

    @staticmethod
    def _extract_media(raw_message: Mapping[str, Any]) -> list[MediaAttachment]:
        """Extract media attachments from a Telegram message."""
        attachments: list[MediaAttachment] = []
        for media_key, media_type in (
            ("photo", "image"), ("document", "file"),
            ("audio", "audio"), ("video", "video"),
            ("voice", "audio"), ("sticker", "sticker"),
        ):
            media_data = raw_message.get(media_key)
            if media_data is None:
                continue
            if media_key == "photo" and isinstance(media_data, list) and media_data:
                best = media_data[-1] if media_data else {}
                file_id = str(best.get("file_id", "")) if isinstance(best, dict) else ""
                if file_id:
                    attachments.append(MediaAttachment(
                        url=f"telegram://file/{file_id}",
                        media_type=media_type,
                        size_bytes=int(best.get("file_size", 0)) if isinstance(best, dict) else 0,
                    ))
            elif isinstance(media_data, dict):
                file_id = str(media_data.get("file_id", ""))
                if file_id:
                    attachments.append(MediaAttachment(
                        url=f"telegram://file/{file_id}",
                        media_type=media_type,
                        filename=str(media_data.get("file_name", "")),
                        size_bytes=int(media_data.get("file_size", 0)),
                    ))
        return attachments

    def _detect_bot_mention(self, raw_message: Mapping[str, Any]) -> bool:
        """Detect @bot mention or @all-equivalent in Telegram message entities."""
        entities = raw_message.get("entities", [])
        if not isinstance(entities, list):
            return False
        text = str(raw_message.get("text", ""))
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            if ent.get("type") == "mention":
                offset = int(ent.get("offset", 0))
                length = int(ent.get("length", 0))
                mentioned = text[offset:offset + length].lstrip("@")
                if mentioned.lower() in ("all", "everyone"):
                    return True
                if self._bot_name and mentioned.lower() == self._bot_name.lower():
                    return True
            if ent.get("type") == "text_mention":
                user = ent.get("user", {})
                if isinstance(user, dict):
                    uid = str(user.get("id", ""))
                    if self._bot_id and uid == self._bot_id:
                        return True
        return False
