"""DingTalk event normalizer.

Maps DingTalk webhook/stream callback payloads (from a ``BackendEventSource``)
into the shared ``EventClassification`` types.
"""
from __future__ import annotations

import re
import time
from typing import Any, Mapping

from leapflow.gateway.connectors.protocol import (
    BackendEvent,
    EventClassification,
    EventKind,
)
from leapflow.gateway.protocol import InboundMessage, MessageSource


_AT_ALL_PATTERN = re.compile(r"@(?:all(?:\b|$)|所有人)", re.IGNORECASE)


class DingTalkEventNormalizer:
    """Normalizes DingTalk BackendEvents into domain types.

    Satisfies the ``PlatformEventNormalizer`` protocol via structural subtyping.
    """

    platform_id = "dingtalk"

    def __init__(
        self, *, bot_id: str = "", bot_name: str = "", robot_code: str = "", profile: str = "default",
    ) -> None:
        self._bot_id = bot_id
        self._bot_name = bot_name
        self._robot_code = robot_code
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
        sender_id = str(payload.get("senderStaffId") or payload.get("senderId") or "")
        if bot_id and sender_id == bot_id:
            return True
        robot = str(payload.get("robotCode") or "")
        if self._robot_code and robot == self._robot_code:
            is_robot_msg = payload.get("isFromRobot") or payload.get("robotCode")
            if is_robot_msg:
                return True
        return False

    def classify(self, event: BackendEvent) -> EventClassification:
        payload = event.payload
        event_type = event.event_type

        if event_type in ("chat_member_join", "chat_member_leave", "org_suite_relay"):
            return EventClassification(kind=EventKind.LIFECYCLE, raw_event=event)

        text = self._extract_text(payload)
        if text:
            if self.is_self_message(event, self._bot_id):
                return EventClassification(kind=EventKind.IGNORED)
            message = self._to_inbound_message(event, text)
            if message is None:
                return EventClassification(kind=EventKind.IGNORED, raw_event=event)
            return EventClassification(kind=EventKind.MESSAGE, message=message)

        if payload.get("actionCardActionIds") or payload.get("actionCardCallbackUrl"):
            return EventClassification(kind=EventKind.CALLBACK, raw_event=event)

        return EventClassification(kind=EventKind.SIGNAL, raw_event=event)

    def _to_inbound_message(self, event: BackendEvent, text: str) -> InboundMessage | None:
        payload = event.payload
        chat_id = str(
            payload.get("conversationId")
            or payload.get("conversation_id")
            or payload.get("chat_id")
            or ""
        )
        chat_type = "group" if payload.get("conversationType") == "2" else "dm"
        user_id = str(payload.get("senderStaffId") or payload.get("senderId") or "")
        user_name = str(payload.get("senderNick") or payload.get("senderName") or "")
        message_id = str(payload.get("msgId") or payload.get("message_id") or "")

        bot_mentioned = (
            bool(payload.get("isInAtList"))
            or bool(payload.get("isAtAll"))
            or self._detect_bot_mention(text)
        )

        metadata: dict[str, Any] = {
            "robot_code": self._robot_code,
        }
        if bot_mentioned:
            metadata["bot_mentioned"] = True

        create_time = payload.get("createAt") or payload.get("create_time")
        try:
            timestamp = int(create_time) / 1000.0 if create_time else time.time()
        except (ValueError, TypeError):
            timestamp = time.time()

        return InboundMessage(
            source=MessageSource(
                platform="dingtalk",
                chat_id=chat_id,
                chat_type=chat_type,
                user_id=user_id,
                user_name=user_name,
                profile=self._profile,
            ),
            text=text,
            message_id=message_id,
            metadata=metadata,
            timestamp=timestamp,
        )

    def _detect_bot_mention(self, content: str) -> bool:
        if _AT_ALL_PATTERN.search(content):
            return True
        if not self._bot_name:
            return False
        return f"@{self._bot_name}" in content

    @staticmethod
    def _extract_text(payload: Mapping[str, Any]) -> str:
        text = payload.get("text")
        if isinstance(text, dict):
            return str(text.get("content") or "")
        if text:
            return str(text)
        content = payload.get("content")
        if isinstance(content, dict):
            return str(content.get("text") or content.get("content") or "")
        return str(payload.get("msgContent") or "")
