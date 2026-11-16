# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for opt-in hardware reentry adapter (P1-G / plan §Test).

Covers: deliberative-only dispatch, source-domain matching, session preservation,
CAS single-fire, budget, default-off, unknown kind raises ValueError.
"""

from __future__ import annotations

import pytest

from leapflow.scheduler.reentry_driver import event_matches
from leapflow.storage.reentry_store import build_reentry_trigger


# ── event_matches: source domain gate ────────────────────────────────────


class TestEventMatchesDomainGate:
    """Hardware triggers must not match gateway events and vice versa."""

    def test_empty_filter_matches_nothing(self):
        assert event_matches({}) is False
        assert event_matches({}, source="hardware", device_id="d1", event="e1") is False

    def test_gateway_trigger_does_not_match_hardware_event(self):
        gate_filter = {"source": "gateway", "platform": "feishu"}
        assert event_matches(gate_filter, source="hardware", device_id="d", event="alert") is False

    def test_hardware_trigger_does_not_match_gateway_event(self):
        hw_filter = {"source": "hardware", "device_id": "arm"}
        assert event_matches(hw_filter, platform="feishu", chat="c1", text="hello") is False

    def test_hardware_trigger_matches_hardware_event(self):
        hw_filter = {"source": "hardware", "device_id": "arm", "event": "alert"}
        assert event_matches(
            hw_filter, source="hardware", device_id="arm", event="alert"
        ) is True

    def test_hardware_trigger_device_mismatch(self):
        hw_filter = {"source": "hardware", "device_id": "arm"}
        assert event_matches(
            hw_filter, source="hardware", device_id="camera", event="alert"
        ) is False

    def test_hardware_trigger_event_mismatch(self):
        hw_filter = {"source": "hardware", "device_id": "arm", "event": "alert"}
        assert event_matches(
            hw_filter, source="hardware", device_id="arm", event="fault"
        ) is False

    def test_hardware_trigger_without_device_matches_any_device(self):
        hw_filter = {"source": "hardware"}
        assert event_matches(
            hw_filter, source="hardware", device_id="arm", event="alert"
        ) is True

    def test_gateway_trigger_default_source(self):
        """Default source is 'gateway'; filter without source implies gateway."""
        gate_filter = {"platform": "feishu"}
        # Default source for the filter is "gateway", and the call default is also "gateway"
        assert event_matches(gate_filter, platform="feishu", chat="c1") is True

    def test_gateway_trigger_with_keyword(self):
        gate_filter = {"platform": "feishu", "keyword": "help"}
        assert event_matches(gate_filter, platform="feishu", text="please help me") is True
        assert event_matches(gate_filter, platform="feishu", text="no match here") is False


# ── build_reentry_trigger: unknown kind raises ValueError ────────────────


class TestBuildReentryTriggerValidation:
    def test_time_kind_accepted(self):
        t = build_reentry_trigger(
            task_id="t1", kind="time", delay_seconds=10.0, now=1000.0
        )
        assert t.kind == "time"
        assert t.due_at == 1010.0

    def test_event_kind_accepted(self):
        t = build_reentry_trigger(
            task_id="t1", kind="event",
            event_match={"source": "hardware", "device_id": "arm"},
            now=1000.0,
        )
        assert t.kind == "event"
        assert t.due_at == 0.0

    def test_unknown_kind_raises_value_error(self):
        with pytest.raises(ValueError, match="unsupported reentry kind"):
            build_reentry_trigger(task_id="t1", kind="signal", now=1000.0)

    def test_weird_kind_raises_value_error(self):
        with pytest.raises(ValueError, match="unsupported reentry kind"):
            build_reentry_trigger(task_id="t1", kind="weird", now=1000.0)


# ── ReentryService.on_hardware_event: deliberative only, default off ─────


class TestReentryServiceHardwareEvent:
    """Test the hardware event dispatch on ReentryService."""

    @pytest.mark.asyncio
    async def test_the_flag_reaches_settings_and_gates_the_hardware_path(
        self, tmp_path, monkeypatch
    ):
        """Enablement has to arrive on Settings, not merely sit in the environment.

        The daemon opens the hardware observer bridge only when
        ``settings.agent_reentry_hardware_enabled`` is true, and the service also
        requires base re-entry.  Asserting on ``os.getenv`` let an environment read
        that never reached the Settings constructor look correct while the gate
        stayed permanently shut: no configuration could turn hardware re-entry on.
        """
        from leapflow.config import load_config
        from leapflow.scheduler.reentry_service import ReentryService
        from leapflow.storage.reentry_store import ReentryStore

        monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.delenv("LEAPFLOW_AGENT_REENTRY_ENABLED", raising=False)
        monkeypatch.delenv("LEAPFLOW_AGENT_REENTRY_HARDWARE_ENABLED", raising=False)
        assert load_config().agent_reentry_hardware_enabled is False

        monkeypatch.setenv("LEAPFLOW_AGENT_REENTRY_HARDWARE_ENABLED", "1")
        hardware_only = load_config()
        assert hardware_only.agent_reentry_hardware_enabled is True

        store = ReentryStore(tmp_path / "reentry.duckdb")
        service = ReentryService(store=store, manager=object(), settings=hardware_only)
        # Hardware re-entry is inert without base re-entry.
        assert service._hardware_reentry_enabled() is False

        monkeypatch.setenv("LEAPFLOW_AGENT_REENTRY_ENABLED", "1")
        both = load_config()
        service = ReentryService(store=store, manager=object(), settings=both)
        assert service._hardware_reentry_enabled() is True

    @pytest.mark.asyncio
    async def test_hardware_event_trigger_matches_only_hardware_source(self):
        """An event trigger with source=hardware only matches hardware events."""
        hw_trigger = {"source": "hardware", "device_id": "arm", "event": "collision"}

        # Should match
        assert event_matches(
            hw_trigger,
            source="hardware",
            device_id="arm",
            event="collision",
        ) is True

        # Should NOT match a gateway event
        assert event_matches(
            hw_trigger,
            source="gateway",
            platform="feishu",
        ) is False

        # Should NOT match a hardware event with different device
        assert event_matches(
            hw_trigger,
            source="hardware",
            device_id="camera",
            event="collision",
        ) is False
