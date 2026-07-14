"""Tests for FeishuEventNormalizer — Feishu event classification and mapping."""
from __future__ import annotations

import pytest

from leapflow.gateway.connectors.protocol import BackendEvent, EventKind
from leapflow.gateway.normalizers.feishu import FeishuEventNormalizer


def _make_message_event(
    sender_id: str = "ou_user1",
    chat_id: str = "oc_chat1",
    chat_type: str = "group",
    content: str = "hello",
    event_id: str = "ev_1",
    message_id: str = "om_1",
    message_type: str = "text",
) -> BackendEvent:
    """Build a flat NDJSON event matching lark-cli im.message.receive_v1 output."""
    return BackendEvent(
        event_id=event_id,
        event_type="im.message.receive_v1",
        platform_id="feishu",
        payload={
            "type": "im.message.receive_v1",
            "event_id": event_id,
            "message_id": message_id,
            "chat_id": chat_id,
            "chat_type": chat_type,
            "message_type": message_type,
            "sender_id": sender_id,
            "content": content,
            "create_time": "1720000000000",
        },
    )


def test_classify_text_message() -> None:
    """im.message.receive_v1 with text content → MESSAGE + InboundMessage."""
    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    result = normalizer.classify(_make_message_event())

    assert result.kind == EventKind.MESSAGE
    assert result.message is not None
    assert result.message.text == "hello"
    assert result.message.source.chat_id == "oc_chat1"
    assert result.message.source.chat_type == "group"
    assert result.message.source.user_id == "ou_user1"
    assert result.message.source.platform == "feishu"
    assert result.message.message_id == "om_1"
    assert result.message.metadata["event_id"] == "ev_1"
    assert result.message.metadata["message_type"] == "text"


def test_self_message_ignored() -> None:
    """Message from bot's own open_id → IGNORED."""
    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    result = normalizer.classify(_make_message_event(sender_id="ou_bot1"))

    assert result.kind == EventKind.IGNORED


def test_self_message_check_with_empty_bot_id() -> None:
    """When bot_id is empty, no messages are filtered as self."""
    normalizer = FeishuEventNormalizer(bot_id="")
    result = normalizer.classify(_make_message_event(sender_id="ou_user1"))

    assert result.kind == EventKind.MESSAGE


def test_dm_chat_type_mapping() -> None:
    """p2p chat_type maps to 'dm'."""
    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    result = normalizer.classify(_make_message_event(chat_type="p2p"))

    assert result.kind == EventKind.MESSAGE
    assert result.message is not None
    assert result.message.source.chat_type == "dm"


def test_group_chat_type() -> None:
    """group chat_type stays 'group'."""
    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    result = normalizer.classify(_make_message_event(chat_type="group"))

    assert result.message is not None
    assert result.message.source.chat_type == "group"


def test_empty_content_returns_ignored() -> None:
    """Message with empty content → IGNORED (no InboundMessage produced)."""
    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    result = normalizer.classify(_make_message_event(content=""))

    assert result.kind == EventKind.IGNORED


def test_reaction_classified_as_signal() -> None:
    """im.message.reaction.created_v1 → SIGNAL."""
    event = BackendEvent(
        event_id="ev_r1",
        event_type="im.message.reaction.created_v1",
        platform_id="feishu",
        payload={"sender_id": "ou_user1"},
    )
    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    result = normalizer.classify(event)

    assert result.kind == EventKind.SIGNAL
    assert result.raw_event is event


def test_lifecycle_event() -> None:
    """im.chat.member.bot.added_v1 → LIFECYCLE."""
    event = BackendEvent(
        event_id="ev_l1",
        event_type="im.chat.member.bot.added_v1",
        platform_id="feishu",
        payload={},
    )
    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    result = normalizer.classify(event)

    assert result.kind == EventKind.LIFECYCLE


def test_callback_event() -> None:
    """card.action.trigger → CALLBACK."""
    event = BackendEvent(
        event_id="ev_c1",
        event_type="card.action.trigger",
        platform_id="feishu",
        payload={"sender_id": "ou_user1"},
    )
    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    result = normalizer.classify(event)

    assert result.kind == EventKind.CALLBACK
    assert result.raw_event is event


def test_unknown_event_type_becomes_signal() -> None:
    """Unrecognized event types become SIGNAL (annotate, don't discard)."""
    event = BackendEvent(
        event_id="ev_u1",
        event_type="some.future.event_v3",
        platform_id="feishu",
        payload={},
    )
    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    result = normalizer.classify(event)

    assert result.kind == EventKind.SIGNAL


def test_bot_mention_detection() -> None:
    """Bot name in content → metadata['bot_mentioned'] = True."""
    normalizer = FeishuEventNormalizer(bot_id="ou_bot1", bot_name="LeapBot")
    result = normalizer.classify(
        _make_message_event(content="@LeapBot 你好"),
    )

    assert result.kind == EventKind.MESSAGE
    assert result.message is not None
    assert result.message.metadata.get("bot_mentioned") is True


def test_no_mention_without_bot_name() -> None:
    """Without bot_name, no mention detection occurs."""
    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    result = normalizer.classify(
        _make_message_event(content="@SomeBot hello"),
    )

    assert result.kind == EventKind.MESSAGE
    assert result.message is not None
    assert "bot_mentioned" not in result.message.metadata


def test_bot_id_setter() -> None:
    """bot_id can be updated after construction."""
    normalizer = FeishuEventNormalizer(bot_id="")
    result1 = normalizer.classify(_make_message_event(sender_id="ou_bot1"))
    assert result1.kind == EventKind.MESSAGE

    normalizer.bot_id = "ou_bot1"
    result2 = normalizer.classify(_make_message_event(sender_id="ou_bot1"))
    assert result2.kind == EventKind.IGNORED


def test_timestamp_parsing() -> None:
    """create_time in milliseconds is converted to seconds."""
    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    result = normalizer.classify(_make_message_event())

    assert result.message is not None
    assert abs(result.message.timestamp - 1720000000.0) < 1.0


def test_is_self_message_standalone() -> None:
    """is_self_message works independently of classify."""
    normalizer = FeishuEventNormalizer()
    event = _make_message_event(sender_id="ou_bot1")

    assert normalizer.is_self_message(event, "ou_bot1") is True
    assert normalizer.is_self_message(event, "ou_other") is False
    assert normalizer.is_self_message(event, "") is False


def test_nested_sender_id_extraction() -> None:
    """Nested sender_id structures (V2 envelope fallback) are handled."""
    event = BackendEvent(
        event_id="ev_n1",
        event_type="im.message.receive_v1",
        platform_id="feishu",
        payload={
            "sender": {"sender_id": {"open_id": "ou_nested"}},
            "message": {
                "chat_id": "oc_1", "chat_type": "group",
                "message_type": "text", "content": "nested test",
                "message_id": "om_n1",
            },
        },
    )
    normalizer = FeishuEventNormalizer(bot_id="ou_nested")
    assert normalizer.is_self_message(event, "ou_nested") is True
