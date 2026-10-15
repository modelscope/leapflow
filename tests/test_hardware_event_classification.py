# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hardware event classification (D-X4 three-way split) tests.

Covers:
- HardwareEventClass enum values
- classify_hardware_event for every EventKind
- Unknown kinds default to REFLEX (fail-closed)
- HardwareEvent.event_class field and resolved_event_class property
- to_payload includes event_class
- Registry.publish_event classifies unclassified events
- Pre-classified events are not reclassified
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.hardware.stream import (
    EventKind,
    HardwareEvent,
    HardwareEventClass,
    classify_hardware_event,
)


# ── Classification mapping ──────────────────────────────────────────────


class TestClassifyHardwareEvent:
    """Verify every known EventKind maps to the expected behavioural class."""

    @pytest.mark.parametrize(
        "kind,expected_class",
        [
            # Reflex (safety-critical, no LLM)
            (EventKind.THRESHOLD_EXCEEDED, HardwareEventClass.REFLEX),
            (EventKind.RATE_EXCEEDED, HardwareEventClass.REFLEX),
            (EventKind.UNREACHABLE, HardwareEventClass.REFLEX),
            # Deliberative (may trigger reentry)
            (EventKind.QUALITY_DEGRADED, HardwareEventClass.DELIBERATIVE),
            (EventKind.STALE, HardwareEventClass.DELIBERATIVE),
            # Record (informational)
            (EventKind.SETTLED, HardwareEventClass.RECORD),
            (EventKind.SAMPLE_LOSS, HardwareEventClass.RECORD),
            (EventKind.CALIBRATION_STARTED, HardwareEventClass.RECORD),
            (EventKind.CALIBRATION_COMPLETED, HardwareEventClass.RECORD),
            (EventKind.CALIBRATION_FAILED, HardwareEventClass.RECORD),
            (EventKind.CALIBRATION_EXPIRED, HardwareEventClass.RECORD),
        ],
    )
    def test_known_kinds(self, kind: str, expected_class: str) -> None:
        assert classify_hardware_event(kind) == expected_class

    def test_unknown_kind_defaults_to_reflex(self) -> None:
        """Unknown kinds are fail-closed: assume safety-critical."""
        assert classify_hardware_event("some_future_kind") == HardwareEventClass.REFLEX

    def test_empty_kind_defaults_to_reflex(self) -> None:
        assert classify_hardware_event("") == HardwareEventClass.REFLEX


# ── HardwareEvent field and property ─────────────────────────────────────


class TestHardwareEventClassField:
    def test_event_class_default_empty(self) -> None:
        event = HardwareEvent(
            kind="threshold_exceeded",
            device_id="heater",
            channel_id="power",
            quantity="watts",
            detail="over",
        )
        assert event.event_class == ""

    def test_event_class_explicit(self) -> None:
        event = HardwareEvent(
            kind="threshold_exceeded",
            device_id="heater",
            channel_id="power",
            quantity="watts",
            detail="over",
            event_class=HardwareEventClass.REFLEX,
        )
        assert event.event_class == HardwareEventClass.REFLEX

    def test_resolved_event_class_from_kind(self) -> None:
        """When event_class is empty, resolved_event_class computes from kind."""
        event = HardwareEvent(
            kind="settled",
            device_id="pump",
            channel_id="flow",
            quantity="mL/s",
            detail="done",
        )
        assert event.resolved_event_class == HardwareEventClass.RECORD

    def test_resolved_event_class_prefers_explicit(self) -> None:
        """When event_class is set, resolved_event_class returns it."""
        event = HardwareEvent(
            kind="settled",
            device_id="pump",
            channel_id="flow",
            quantity="mL/s",
            detail="done",
            event_class=HardwareEventClass.DELIBERATIVE,
        )
        assert event.resolved_event_class == HardwareEventClass.DELIBERATIVE


# ── Payload includes event_class ─────────────────────────────────────────


class TestPayloadEventClass:
    def test_to_payload_includes_event_class(self) -> None:
        event = HardwareEvent(
            kind="rate_exceeded",
            device_id="heater",
            channel_id="power",
            quantity="watts",
            detail="fast",
            observed_at=time.time(),
        )
        payload = event.to_payload()
        assert payload["event_class"] == HardwareEventClass.REFLEX
        assert payload["kind"] == "rate_exceeded"

    def test_to_payload_with_explicit_class(self) -> None:
        event = HardwareEvent(
            kind="settled",
            device_id="pump",
            channel_id="flow",
            quantity="mL/s",
            detail="done",
            event_class=HardwareEventClass.RECORD,
        )
        payload = event.to_payload()
        assert payload["event_class"] == HardwareEventClass.RECORD


# ── Registry classification wiring ───────────────────────────────────────


class _MinimalRegistry:
    """Stub registry with just enough to test publish_event classification."""

    def __init__(self) -> None:
        self.recorded_events: list[Any] = []
        self.emitted_events: list[Any] = []
        self._recent_events: list[Any] = []
        self._event_emitter = None
        self._deliberative_observer = None

    def record_event(self, event: Any) -> None:
        self.recorded_events.append(event)

    def set_event_emitter(self, emit: Any) -> None:
        self._event_emitter = emit


class TestRegistryClassificationWiring:
    def _make_registry(self) -> _MinimalRegistry:
        """Create a minimal registry with publish_event from the real class."""
        from leapflow.hardware.registry import HardwareRegistry

        reg = _MinimalRegistry()
        # Bind the real publish_event and _ensure_classified methods
        reg.publish_event = HardwareRegistry.publish_event.__get__(reg, type(reg))
        reg._ensure_classified = HardwareRegistry._ensure_classified
        reg._notify_deliberative_observer = HardwareRegistry._notify_deliberative_observer.__get__(reg, type(reg))
        return reg

    def test_unclassified_event_gets_classified(self) -> None:
        reg = self._make_registry()
        event = HardwareEvent(
            kind="threshold_exceeded",
            device_id="heater",
            channel_id="power",
            quantity="watts",
            detail="hot",
        )
        assert event.event_class == ""
        reg.publish_event(event)
        recorded = reg.recorded_events[-1]
        assert recorded.event_class == HardwareEventClass.REFLEX

    def test_preclassified_event_not_reclassified(self) -> None:
        reg = self._make_registry()
        event = HardwareEvent(
            kind="settled",
            device_id="pump",
            channel_id="flow",
            quantity="mL/s",
            detail="done",
            event_class=HardwareEventClass.DELIBERATIVE,  # intentionally "wrong"
        )
        reg.publish_event(event)
        recorded = reg.recorded_events[-1]
        # Should keep the pre-set class, not overwrite with RECORD
        assert recorded.event_class == HardwareEventClass.DELIBERATIVE

    def test_non_hardware_event_passes_through(self) -> None:
        reg = self._make_registry()
        fake_event = SimpleNamespace(kind="custom", device_id="x")
        reg.publish_event(fake_event)
        assert reg.recorded_events[-1] is fake_event

    def test_emitter_receives_classified_event(self) -> None:
        reg = self._make_registry()
        emitted: list[Any] = []
        reg._event_emitter = lambda e: emitted.append(e)
        event = HardwareEvent(
            kind="stale",
            device_id="sensor",
            channel_id="temp",
            quantity="C",
            detail="old",
        )
        reg.publish_event(event)
        assert emitted
        assert emitted[0].event_class == HardwareEventClass.DELIBERATIVE


# ── Enum values ──────────────────────────────────────────────────────────


class TestHardwareEventClassEnum:
    def test_reflex_value(self) -> None:
        assert HardwareEventClass.REFLEX == "reflex"

    def test_deliberative_value(self) -> None:
        assert HardwareEventClass.DELIBERATIVE == "deliberative"

    def test_record_value(self) -> None:
        assert HardwareEventClass.RECORD == "record"
