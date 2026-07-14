"""Tests for GatewayServer consumer loop — BackendEvent routing."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest

from leapflow.gateway.connectors.lark_event_source import BotIdentity
from leapflow.gateway.connectors.protocol import BackendEvent, EventSourceStatus
from leapflow.gateway.normalizers.feishu import FeishuEventNormalizer
from leapflow.gateway.protocol import InboundMessage
from leapflow.gateway.server import GatewayServer
from leapflow.gateway.trigger_policy import TriggerMode, TriggerPolicy


class FakeEventSource:
    """Fake BackendEventSource that yields pre-configured events."""

    platform_id = "feishu"
    backend_kind = "fake"

    def __init__(self, events: list[BackendEvent]) -> None:
        self._events = list(events)
        self._started = False

    async def start(self, *, checkpoint: str = "") -> EventSourceStatus:
        self._started = True
        return EventSourceStatus(ok=True, backend_kind=self.backend_kind)

    async def stop(self) -> EventSourceStatus:
        self._started = False
        return EventSourceStatus(ok=True, backend_kind=self.backend_kind)

    async def events(self) -> AsyncIterator[BackendEvent]:
        for event in self._events:
            yield event

    async def status(self) -> EventSourceStatus:
        return EventSourceStatus(ok=self._started, backend_kind=self.backend_kind)


class FakeAdapter:
    """Minimal adapter for consumer loop tests."""

    platform_id = "feishu"
    supports_async_delivery = True
    splits_long_messages = False
    max_message_length = 4000

    def __init__(
        self,
        event_source: FakeEventSource,
        bot_identity: BotIdentity | None = None,
    ) -> None:
        self._event_source = event_source
        self.bot_identity = bot_identity or BotIdentity()
        self.on_message = None

    def event_source(self) -> FakeEventSource:
        return self._event_source

    async def connect(self, *, is_reconnect: bool = False) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def send(self, target: Any, content: Any) -> Any:
        pass


def _make_feishu_event(
    sender_id: str = "ou_user1",
    content: str = "hello",
    event_id: str = "ev_1",
    chat_type: str = "group",
) -> BackendEvent:
    return BackendEvent(
        event_id=event_id,
        event_type="im.message.receive_v1",
        platform_id="feishu",
        payload={
            "type": "im.message.receive_v1",
            "event_id": event_id,
            "message_id": f"om_{event_id}",
            "chat_id": "oc_chat1",
            "chat_type": chat_type,
            "message_type": "text",
            "sender_id": sender_id,
            "content": content,
            "create_time": "1720000000000",
        },
    )


@pytest.mark.asyncio
async def test_consumer_loop_routes_messages(tmp_path: Any) -> None:
    """BackendEvent → normalizer → _on_inbound_message → handler."""
    received: list[InboundMessage] = []

    async def handler(msg: InboundMessage, session_key: str) -> None:
        received.append(msg)

    server = GatewayServer(tmp_path)
    server.set_message_handler(handler)

    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    server.register_normalizer("feishu", normalizer)
    server.register_trigger_policy("feishu", TriggerPolicy(mode=TriggerMode.ALL))

    source = FakeEventSource([_make_feishu_event()])
    adapter = FakeAdapter(source)
    server._adapters["feishu"] = adapter

    result = await server.start_platform_events("feishu")
    assert result["ok"] is True

    await asyncio.sleep(0.1)
    await server.stop_platform_events("feishu")

    assert len(received) == 1
    assert received[0].text == "hello"
    assert received[0].source.platform == "feishu"


@pytest.mark.asyncio
async def test_consumer_loop_filters_self_messages(tmp_path: Any) -> None:
    """Self-message → IGNORED → not routed to handler."""
    received: list[InboundMessage] = []

    async def handler(msg: InboundMessage, session_key: str) -> None:
        received.append(msg)

    server = GatewayServer(tmp_path)
    server.set_message_handler(handler)

    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    server.register_normalizer("feishu", normalizer)
    server.register_trigger_policy("feishu", TriggerPolicy(mode=TriggerMode.ALL))

    source = FakeEventSource([_make_feishu_event(sender_id="ou_bot1")])
    adapter = FakeAdapter(source)
    server._adapters["feishu"] = adapter

    await server.start_platform_events("feishu")
    await asyncio.sleep(0.1)
    await server.stop_platform_events("feishu")

    assert len(received) == 0


@pytest.mark.asyncio
async def test_trigger_policy_mention_only(tmp_path: Any) -> None:
    """Non-mention messages in mention_only mode → not routed to handler."""
    received: list[InboundMessage] = []
    emitted_events: list[object] = []

    async def handler(msg: InboundMessage, session_key: str) -> None:
        received.append(msg)

    async def on_event(event: object) -> None:
        emitted_events.append(event)

    server = GatewayServer(tmp_path, on_event=on_event)
    server.set_message_handler(handler)

    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    server.register_normalizer("feishu", normalizer)
    server.register_trigger_policy("feishu", TriggerPolicy(mode=TriggerMode.MENTION_ONLY))

    source = FakeEventSource([
        _make_feishu_event(content="regular message in group"),
    ])
    adapter = FakeAdapter(source)
    server._adapters["feishu"] = adapter

    await server.start_platform_events("feishu")
    await asyncio.sleep(0.1)
    await server.stop_platform_events("feishu")

    assert len(received) == 0
    assert len(emitted_events) > 0


@pytest.mark.asyncio
async def test_trigger_policy_dm_activates_in_mention_only(tmp_path: Any) -> None:
    """DM messages activate even in mention_only mode."""
    received: list[InboundMessage] = []

    async def handler(msg: InboundMessage, session_key: str) -> None:
        received.append(msg)

    server = GatewayServer(tmp_path)
    server.set_message_handler(handler)

    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    server.register_normalizer("feishu", normalizer)
    server.register_trigger_policy("feishu", TriggerPolicy(mode=TriggerMode.MENTION_ONLY))

    source = FakeEventSource([_make_feishu_event(chat_type="p2p")])
    adapter = FakeAdapter(source)
    server._adapters["feishu"] = adapter

    await server.start_platform_events("feishu")
    await asyncio.sleep(0.1)
    await server.stop_platform_events("feishu")

    assert len(received) == 1


@pytest.mark.asyncio
async def test_consumer_task_cancelled_on_stop(tmp_path: Any) -> None:
    """stop_platform_events cancels the consumer task cleanly."""
    server = GatewayServer(tmp_path)

    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    server.register_normalizer("feishu", normalizer)
    server.register_trigger_policy("feishu", TriggerPolicy(mode=TriggerMode.ALL))

    source = FakeEventSource([_make_feishu_event()])
    adapter = FakeAdapter(source)
    server._adapters["feishu"] = adapter

    await server.start_platform_events("feishu")
    assert "feishu" in server._consumer_tasks

    await server.stop_platform_events("feishu")
    assert "feishu" not in server._consumer_tasks


@pytest.mark.asyncio
async def test_signal_events_emitted(tmp_path: Any) -> None:
    """Signal events (reactions) are emitted but not routed to handler."""
    received: list[InboundMessage] = []
    emitted: list[object] = []

    async def handler(msg: InboundMessage, session_key: str) -> None:
        received.append(msg)

    async def on_event(event: object) -> None:
        emitted.append(event)

    server = GatewayServer(tmp_path, on_event=on_event)
    server.set_message_handler(handler)

    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    server.register_normalizer("feishu", normalizer)
    server.register_trigger_policy("feishu", TriggerPolicy(mode=TriggerMode.ALL))

    reaction_event = BackendEvent(
        event_id="ev_r1",
        event_type="im.message.reaction.created_v1",
        platform_id="feishu",
        payload={"sender_id": "ou_user1"},
    )
    source = FakeEventSource([reaction_event])
    adapter = FakeAdapter(source)
    server._adapters["feishu"] = adapter

    await server.start_platform_events("feishu")
    await asyncio.sleep(0.1)
    await server.stop_platform_events("feishu")

    assert len(received) == 0
    assert any(isinstance(e, BackendEvent) for e in emitted)


@pytest.mark.asyncio
async def test_multiple_events_in_sequence(tmp_path: Any) -> None:
    """Multiple valid messages — first passes; rest hit cooldown."""
    received: list[InboundMessage] = []

    async def handler(msg: InboundMessage, session_key: str) -> None:
        received.append(msg)

    server = GatewayServer(tmp_path)
    server.set_message_handler(handler)

    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    server.register_normalizer("feishu", normalizer)
    server.register_trigger_policy("feishu", TriggerPolicy(mode=TriggerMode.ALL))

    events = [
        _make_feishu_event(event_id="ev_1", content="msg1"),
        _make_feishu_event(event_id="ev_2", content="msg2"),
        _make_feishu_event(event_id="ev_3", content="msg3"),
    ]
    source = FakeEventSource(events)
    adapter = FakeAdapter(source)
    server._adapters["feishu"] = adapter

    await server.start_platform_events("feishu")
    await asyncio.sleep(0.2)
    await server.stop_platform_events("feishu")

    assert len(received) >= 1
    assert received[0].text == "msg1"


@pytest.mark.asyncio
async def test_all_events_pass_with_zero_cooldown(tmp_path: Any) -> None:
    """With cooldown=0 all rapid events from the same chat are routed."""
    received: list[InboundMessage] = []

    async def handler(msg: InboundMessage, session_key: str) -> None:
        received.append(msg)

    server = GatewayServer(tmp_path)
    server.set_message_handler(handler)

    normalizer = FeishuEventNormalizer(bot_id="ou_bot1")
    server.register_normalizer("feishu", normalizer)
    server.register_trigger_policy(
        "feishu",
        TriggerPolicy(mode=TriggerMode.ALL, cooldown_per_chat_s=0),
    )

    events = [
        _make_feishu_event(event_id="ev_1", content="msg1"),
        _make_feishu_event(event_id="ev_2", content="msg2"),
        _make_feishu_event(event_id="ev_3", content="msg3"),
    ]
    source = FakeEventSource(events)
    adapter = FakeAdapter(source)
    server._adapters["feishu"] = adapter

    await server.start_platform_events("feishu")
    await asyncio.sleep(0.2)
    await server.stop_platform_events("feishu")

    assert len(received) == 3
    assert [m.text for m in received] == ["msg1", "msg2", "msg3"]


# ── Text chunking and formatting tests ────────────────────────


def test_chunk_text_short_message() -> None:
    """Short messages are returned as a single chunk."""
    from leapflow.gateway.server import _chunk_text

    assert _chunk_text("hello", 100) == ["hello"]
    assert _chunk_text("", 100) == []


def test_chunk_text_long_message() -> None:
    """Long messages are split at paragraph boundaries."""
    from leapflow.gateway.server import _chunk_text

    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = _chunk_text(text, 30)
    assert len(chunks) >= 2
    assert "".join(chunks) == text.replace("\n\n", "")  # stripped whitespace


def test_chunk_text_respects_max_len() -> None:
    """Every chunk respects max_len boundary."""
    from leapflow.gateway.server import _chunk_text

    text = "A" * 500
    chunks = _chunk_text(text, 100)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == text


def test_has_rich_formatting_detects_code() -> None:
    """Code blocks and inline code trigger rich format detection."""
    from leapflow.gateway.server import _has_rich_formatting

    assert _has_rich_formatting("```python\nprint(1)\n```") is True
    assert _has_rich_formatting("Use `foo` and **bar** to do things") is True
    assert _has_rich_formatting("just plain text here") is False


def test_has_rich_formatting_detects_headers_and_lists() -> None:
    """Headers + lists trigger rich format detection."""
    from leapflow.gateway.server import _has_rich_formatting

    md = "# Title\n\n- item one\n- item two"
    assert _has_rich_formatting(md) is True
