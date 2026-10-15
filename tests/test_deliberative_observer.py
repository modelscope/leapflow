# Copyright (c) Alibaba, Inc. and its affiliates.
"""HardwareDeliberativeObserver protocol and registry wiring tests.

Covers:
- HardwareDeliberativeObserver is a runtime_checkable Protocol
- A class implementing on_deliberative_event satisfies the Protocol
- Registry forwards deliberative events to the observer
- Registry does NOT forward reflex events to the observer
- Registry does NOT forward record events to the observer
- Registry does NOT forward non-HardwareEvent types to the observer
- Observer exception does not break publish_event
- Observer can be set to None (disabled)
"""

from __future__ import annotations

from typing import Any

from leapflow.hardware.stream import (
    HardwareDeliberativeObserver,
    HardwareEvent,
    HardwareEventClass,
)


# ── Stubs ───────────────────────────────────────────────────────────────


class _RecordingObserver:
    """Captures every deliberative event it receives."""

    def __init__(self) -> None:
        self.received: list[HardwareEvent] = []

    def on_deliberative_event(self, event: HardwareEvent) -> None:
        self.received.append(event)


class _RaisingObserver:
    """Raises on every call to verify fault isolation."""

    def on_deliberative_event(self, event: HardwareEvent) -> None:
        raise RuntimeError("observer boom")


def _make_event(kind: str, event_class: str = "") -> HardwareEvent:
    return HardwareEvent(
        kind=kind,
        device_id="sensor",
        channel_id="temp",
        quantity="temperature",
        detail="test",
        event_class=event_class,
    )


# ── Protocol tests ────────────────────────────────────────────────────


class TestHardwareDeliberativeObserverProtocol:
    def test_protocol_is_runtime_checkable(self) -> None:
        observer = _RecordingObserver()
        assert isinstance(observer, HardwareDeliberativeObserver)

    def test_plain_object_not_protocol(self) -> None:
        assert not isinstance(object(), HardwareDeliberativeObserver)


# ── Registry wiring tests ────────────────────────────────────────────


class _MinimalRegistry:
    """Stub registry with just enough to test deliberative observer wiring."""

    def __init__(self) -> None:
        self.recorded_events: list[Any] = []
        self._recent_events: list[Any] = []
        self._event_emitter = None
        self._deliberative_observer = None

    def record_event(self, event: Any) -> None:
        self.recorded_events.append(event)


def _make_registry() -> _MinimalRegistry:
    from leapflow.hardware.registry import HardwareRegistry

    reg = _MinimalRegistry()
    reg.publish_event = HardwareRegistry.publish_event.__get__(reg, type(reg))
    reg._ensure_classified = HardwareRegistry._ensure_classified
    reg._notify_deliberative_observer = HardwareRegistry._notify_deliberative_observer.__get__(reg, type(reg))
    reg.set_deliberative_observer = HardwareRegistry.set_deliberative_observer.__get__(reg, type(reg))
    return reg


class TestRegistryDeliberativeWiring:
    def test_deliberative_event_forwarded(self) -> None:
        reg = _make_registry()
        observer = _RecordingObserver()
        reg.set_deliberative_observer(observer)
        event = _make_event("quality_degraded", event_class=HardwareEventClass.DELIBERATIVE)
        reg.publish_event(event)
        assert len(observer.received) == 1
        assert observer.received[0].kind == "quality_degraded"

    def test_reflex_event_not_forwarded(self) -> None:
        reg = _make_registry()
        observer = _RecordingObserver()
        reg.set_deliberative_observer(observer)
        event = _make_event("threshold_exceeded", event_class=HardwareEventClass.REFLEX)
        reg.publish_event(event)
        assert len(observer.received) == 0

    def test_record_event_not_forwarded(self) -> None:
        reg = _make_registry()
        observer = _RecordingObserver()
        reg.set_deliberative_observer(observer)
        event = _make_event("settled", event_class=HardwareEventClass.RECORD)
        reg.publish_event(event)
        assert len(observer.received) == 0

    def test_unclassified_deliberative_gets_classified_and_forwarded(self) -> None:
        """quality_degraded with no explicit class gets classified as deliberative."""
        reg = _make_registry()
        observer = _RecordingObserver()
        reg.set_deliberative_observer(observer)
        # No explicit event_class — the registry should classify it
        event = _make_event("quality_degraded")
        reg.publish_event(event)
        assert len(observer.received) == 1
        assert observer.received[0].event_class == HardwareEventClass.DELIBERATIVE

    def test_stale_event_forwarded(self) -> None:
        """stale is also deliberative."""
        reg = _make_registry()
        observer = _RecordingObserver()
        reg.set_deliberative_observer(observer)
        event = _make_event("stale")
        reg.publish_event(event)
        assert len(observer.received) == 1

    def test_non_hardware_event_not_forwarded(self) -> None:
        reg = _make_registry()
        observer = _RecordingObserver()
        reg.set_deliberative_observer(observer)
        # A plain dict is not a HardwareEvent
        reg.publish_event({"kind": "something", "event_class": "deliberative"})
        assert len(observer.received) == 0

    def test_no_observer_no_error(self) -> None:
        """publish_event works fine with no observer set."""
        reg = _make_registry()
        event = _make_event("quality_degraded")
        # Should not raise
        reg.publish_event(event)

    def test_observer_exception_does_not_break_publish(self) -> None:
        """A raising observer must not prevent the event from reaching the emitter."""
        reg = _make_registry()
        raised_observer = _RaisingObserver()
        reg.set_deliberative_observer(raised_observer)
        emitted: list[Any] = []
        reg._event_emitter = lambda e: emitted.append(e)
        event = _make_event("quality_degraded")
        # Should not raise despite the observer raising
        reg.publish_event(event)
        # The event should still reach the emitter
        assert len(emitted) == 1

    def test_set_observer_to_none_disables(self) -> None:
        reg = _make_registry()
        observer = _RecordingObserver()
        reg.set_deliberative_observer(observer)
        reg.set_deliberative_observer(None)
        event = _make_event("quality_degraded", event_class=HardwareEventClass.DELIBERATIVE)
        reg.publish_event(event)
        assert len(observer.received) == 0
