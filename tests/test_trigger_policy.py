"""Tests for TriggerPolicy — inbound message gating."""
from __future__ import annotations

import pytest

from leapflow.gateway.protocol import InboundMessage, MessageSource
from leapflow.gateway.trigger_policy import TriggerMode, TriggerPolicy, _RateTracker


def _msg(
    text: str = "hello",
    chat_type: str = "group",
    chat_id: str = "oc_1",
    user_id: str = "ou_1",
    metadata: dict | None = None,
) -> InboundMessage:
    return InboundMessage(
        source=MessageSource(
            platform="feishu",
            chat_id=chat_id,
            chat_type=chat_type,
            user_id=user_id,
        ),
        text=text,
        message_id="om_1",
        metadata=metadata or {},
    )


def test_mention_only_group_no_mention() -> None:
    """Group message without mention → not activated."""
    policy = TriggerPolicy(mode=TriggerMode.MENTION_ONLY)
    assert policy.should_activate(_msg(text="regular message")) is False


def test_mention_only_dm_always_activates() -> None:
    """DM messages always activate in mention_only mode."""
    policy = TriggerPolicy(mode=TriggerMode.MENTION_ONLY)
    assert policy.should_activate(_msg(chat_type="dm")) is True
    assert policy.should_activate(_msg(chat_type="p2p")) is True
    assert policy.should_activate(_msg(chat_type="private")) is True


def test_mention_only_with_bot_mentioned_metadata() -> None:
    """Message with bot_mentioned metadata flag activates in mention_only."""
    policy = TriggerPolicy(mode=TriggerMode.MENTION_ONLY)
    assert policy.should_activate(
        _msg(text="@LeapBot help", metadata={"bot_mentioned": True}),
    ) is True
    assert policy.should_activate(_msg(text="hello there")) is False


def test_mention_only_with_metadata_flag() -> None:
    """Message with bot_mentioned metadata activates."""
    policy = TriggerPolicy(mode=TriggerMode.MENTION_ONLY)
    assert policy.should_activate(
        _msg(metadata={"bot_mentioned": True}),
    ) is True


def test_all_mode_always_activates() -> None:
    """All mode activates for every message."""
    policy = TriggerPolicy(mode=TriggerMode.ALL)
    assert policy.should_activate(_msg()) is True


def test_manual_mode_never_activates() -> None:
    """Manual mode never activates."""
    policy = TriggerPolicy(mode=TriggerMode.MANUAL)
    assert policy.should_activate(_msg()) is False
    assert policy.should_activate(_msg(chat_type="dm")) is False


def test_keyword_mode() -> None:
    """Keyword mode matches case-insensitively."""
    policy = TriggerPolicy(mode=TriggerMode.KEYWORD, keywords=("help", "urgent"))
    assert policy.should_activate(_msg(text="I need HELP")) is True
    assert policy.should_activate(_msg(text="this is urgent please")) is True
    assert policy.should_activate(_msg(text="just chatting")) is False


def test_blocked_chats() -> None:
    """Messages from blocked chats are rejected."""
    policy = TriggerPolicy(mode=TriggerMode.ALL, blocked_chats=frozenset({"oc_blocked"}))
    assert policy.should_activate(_msg(chat_id="oc_blocked")) is False
    assert policy.should_activate(_msg(chat_id="oc_allowed")) is True


def test_blocked_users() -> None:
    """Messages from blocked users are rejected."""
    policy = TriggerPolicy(mode=TriggerMode.ALL, blocked_users=frozenset({"ou_bad"}))
    assert policy.should_activate(_msg(user_id="ou_bad")) is False
    assert policy.should_activate(_msg(user_id="ou_good")) is True


def test_allowed_chats_whitelist() -> None:
    """When allowed_chats is set, only those chats pass."""
    policy = TriggerPolicy(mode=TriggerMode.ALL, allowed_chats=frozenset({"oc_vip"}))
    assert policy.should_activate(_msg(chat_id="oc_vip")) is True
    assert policy.should_activate(_msg(chat_id="oc_other")) is False


def test_allowed_users_whitelist() -> None:
    """When allowed_users is set, only those users pass."""
    policy = TriggerPolicy(mode=TriggerMode.ALL, allowed_users=frozenset({"ou_vip"}))
    assert policy.should_activate(_msg(user_id="ou_vip")) is True
    assert policy.should_activate(_msg(user_id="ou_other")) is False


def test_allowed_users_empty_means_all_allowed() -> None:
    """Empty allowed_users means no user filtering."""
    policy = TriggerPolicy(mode=TriggerMode.ALL, allowed_users=frozenset())
    assert policy.should_activate(_msg(user_id="ou_anyone")) is True


def test_string_mode_via_enum_value() -> None:
    """TriggerMode can be created from string value."""
    assert TriggerMode("mention_only") is TriggerMode.MENTION_ONLY
    assert TriggerMode("all") is TriggerMode.ALL
    assert TriggerMode("keyword") is TriggerMode.KEYWORD
    assert TriggerMode("manual") is TriggerMode.MANUAL


def test_invalid_mode_raises() -> None:
    """Invalid mode string raises ValueError."""
    with pytest.raises(ValueError):
        TriggerMode("invalid_mode")


def test_rate_tracker_cooldown() -> None:
    """Rate tracker enforces per-chat cooldown."""
    tracker = _RateTracker()
    assert tracker.allow("chat1", max_per_minute=100, cooldown_s=1.0) is True
    assert tracker.allow("chat1", max_per_minute=100, cooldown_s=1.0) is False


def test_rate_tracker_separate_chats() -> None:
    """Rate limits are per-chat, not global."""
    tracker = _RateTracker()
    assert tracker.allow("chat1", max_per_minute=100, cooldown_s=1.0) is True
    assert tracker.allow("chat2", max_per_minute=100, cooldown_s=1.0) is True
