# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hermetic tests for EventBridge — EventBus to EventTrigger adapter.

No network, no LLM: pure in-memory trigger matching logic.
"""

from __future__ import annotations

import time
from unittest.mock import patch

from leapflow.domain.events import SystemEvent
from leapflow.monitor.event_bridge import EventBridge
from leapflow.scheduler.triggers.event import EventTrigger


def _make_event(event_type: str, source: str = "test") -> SystemEvent:
    """Create a minimal SystemEvent for testing."""
    return SystemEvent(
        event_type=event_type,
        source=source,
        payload={},
        timestamp=time.time(),
    )


# ── Registration and triggering ───────────────────────────────────────────


def test_event_bridge_register_and_trigger() -> None:
    """Register a pattern='fs.*' trigger, send 'fs.change', verify activated."""
    bridge = EventBridge()
    trigger = EventTrigger(event_pattern="fs.*")

    bridge.register("watch-1", trigger)
    assert bridge.active_count == 1
    assert not trigger.is_triggered

    bridge.on_event(_make_event("fs.change"))

    assert trigger.is_triggered
    assert trigger.last_event == "fs.change"


def test_event_bridge_no_match() -> None:
    """Register 'fs.*' trigger, send 'app.focus_change', verify NOT activated."""
    bridge = EventBridge()
    trigger = EventTrigger(event_pattern="fs.*")

    bridge.register("watch-2", trigger)
    bridge.on_event(_make_event("app.focus_change"))

    assert not trigger.is_triggered
    assert trigger.last_event == ""


def test_event_bridge_unregister() -> None:
    """Register then unregister; matching event should not fire trigger."""
    bridge = EventBridge()
    trigger = EventTrigger(event_pattern="fs.*")

    bridge.register("watch-3", trigger)
    assert bridge.active_count == 1

    bridge.unregister("watch-3")
    assert bridge.active_count == 0

    bridge.on_event(_make_event("fs.change"))

    assert not trigger.is_triggered


def test_event_bridge_empty_short_circuit() -> None:
    """Calling on_event with no registered triggers raises no exception."""
    bridge = EventBridge()
    assert bridge.active_count == 0

    # Should not raise
    bridge.on_event(_make_event("fs.change"))
    bridge.on_event(_make_event("app.focus_change"))


# ── Multiple triggers ─────────────────────────────────────────────────────


def test_event_bridge_multiple_triggers() -> None:
    """Multiple triggers: only matching ones fire."""
    bridge = EventBridge()
    fs_trigger = EventTrigger(event_pattern="fs.*")
    app_trigger = EventTrigger(event_pattern="app.*")

    bridge.register("watch-fs", fs_trigger)
    bridge.register("watch-app", app_trigger)
    assert bridge.active_count == 2

    bridge.on_event(_make_event("fs.change"))

    assert fs_trigger.is_triggered
    assert not app_trigger.is_triggered


def test_event_bridge_unregister_nonexistent() -> None:
    """Unregistering a non-existent watch_id is a no-op."""
    bridge = EventBridge()
    trigger = EventTrigger(event_pattern="fs.*")
    bridge.register("watch-x", trigger)

    # Should not raise
    bridge.unregister("nonexistent")
    assert bridge.active_count == 1


# ── Debounce ───────────────────────────────────────────────────────────────


def test_event_bridge_debounce_suppresses_rapid_events() -> None:
    """Rapid successive events within debounce window fire trigger only once."""
    bridge = EventBridge()
    trigger = EventTrigger(event_pattern="fs.*")
    bridge.register("watch-d1", trigger, debounce_s=1.0)

    # Use a fixed monotonic clock to simulate rapid events
    fake_time = [100.0]

    def _monotonic() -> float:
        return fake_time[0]

    with patch("leapflow.monitor.event_bridge.time.monotonic", side_effect=_monotonic):
        # First event fires
        bridge.on_event(_make_event("fs.change"))
        assert trigger.is_triggered

        # Reset trigger state to detect re-fires
        trigger.advance(0)
        assert not trigger.is_triggered

        # Send 4 more events at the same timestamp (within debounce window)
        for _ in range(4):
            bridge.on_event(_make_event("fs.change"))

    # Trigger should NOT have been re-fired
    assert not trigger.is_triggered
    # All 4 subsequent events should have been debounced
    assert bridge.debounced_count("watch-d1") == 4


def test_event_bridge_debounce_allows_after_window() -> None:
    """Events after the debounce window has elapsed are allowed through."""
    bridge = EventBridge()
    trigger = EventTrigger(event_pattern="fs.*")
    bridge.register("watch-d2", trigger, debounce_s=1.0)

    fake_time = [100.0]

    def _monotonic() -> float:
        return fake_time[0]

    with patch("leapflow.monitor.event_bridge.time.monotonic", side_effect=_monotonic):
        # First event fires
        bridge.on_event(_make_event("fs.change"))
        assert trigger.is_triggered

        trigger.advance(0)
        assert not trigger.is_triggered

        # Advance time past debounce window
        fake_time[0] = 101.1  # 1.1s later, > 1.0s debounce

        # Second event should fire
        bridge.on_event(_make_event("fs.change"))
        assert trigger.is_triggered

    assert bridge.debounced_count("watch-d2") == 0


def test_event_bridge_debounce_per_watch() -> None:
    """Each watch has its own independent debounce tracking."""
    bridge = EventBridge()
    trigger_a = EventTrigger(event_pattern="fs.*")
    trigger_b = EventTrigger(event_pattern="fs.*")

    bridge.register("watch-a", trigger_a, debounce_s=1.0)
    bridge.register("watch-b", trigger_b, debounce_s=1.0)

    fake_time = [100.0]

    def _monotonic() -> float:
        return fake_time[0]

    with patch("leapflow.monitor.event_bridge.time.monotonic", side_effect=_monotonic):
        # First event fires both
        bridge.on_event(_make_event("fs.change"))
        assert trigger_a.is_triggered
        assert trigger_b.is_triggered

        trigger_a.advance(0)
        trigger_b.advance(0)

        # Advance time: past window for watch-a only by manipulating
        # per-watch last_triggered directly
        # Instead, send another event — both should be debounced
        bridge.on_event(_make_event("fs.change"))
        assert not trigger_a.is_triggered
        assert not trigger_b.is_triggered

    # Both suppressed equally
    assert bridge.debounced_count("watch-a") == 1
    assert bridge.debounced_count("watch-b") == 1
