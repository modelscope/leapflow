"""Tests for Phase 2: alert policy, metrics exporter, and calibration events.

Covers the three sub-items delivered together:

2.3 HardwareAlertPolicy
    - Rule loading from settings
    - Consecutive-hit gating
    - Channel filter matching
    - estop dispatches halt without approval
    - Other actions route through ApprovalOrchestrator

2.6 HardwareMetricsExporter
    - Default off (build_exporter returns None)
    - collect() returns store, stream, registry, and policy metrics
    - render_prometheus() produces valid exposition format
    - Zero overhead when disabled

IC-6 Calibration lifecycle events + board
    - EventKind has all four calibration variants
    - calibration_failed/expired are notable severity in digest
    - hardware.yaml template validates against SDUI component catalog
    - Calibration Health section renders when data is present
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.hardware.alert_policy import (
    DEFAULT_CONSECUTIVE,
    AlertRule,
    HardwareAlertPolicy,
    build_alert_policy,
    load_alert_policies,
)
from leapflow.hardware.observability.exporter import (
    ALERT_POLICY_FIRED_TOTAL,
    DEVICES_ADMITTED,
    DROPPED_TOTAL,
    OBSERVED_HZ,
    RAW_WRITES_TOTAL,
    SAMPLES_TOTAL,
    STREAM_SOURCES_ACTIVE,
    WRITE_FAILURES_TOTAL,
    HardwareMetricsExporter,
    MetricSample,
    build_exporter,
)
from leapflow.hardware.stream import EventKind, HardwareEvent


# ════════════════════════════════════════════════════════════════
# 2.3 HardwareAlertPolicy
# ════════════════════════════════════════════════════════════════


class TestAlertRuleMatching:
    def test_rule_matches_any_channel_when_no_filter(self) -> None:
        rule = AlertRule(event_kind="threshold_exceeded", action="hw_estop")
        assert rule.matches("threshold_exceeded", "dev1", "ch1")
        assert rule.matches("threshold_exceeded", "dev2", "ch99")
        assert not rule.matches("rate_exceeded", "dev1", "ch1")

    def test_rule_matches_specific_channel(self) -> None:
        rule = AlertRule(
            event_kind="threshold_exceeded",
            action="hw_estop",
            channel_filter="dev1.level",
        )
        assert rule.matches("threshold_exceeded", "dev1", "level")
        assert not rule.matches("threshold_exceeded", "dev1", "knob")
        assert not rule.matches("threshold_exceeded", "dev2", "level")

    def test_rule_matches_bare_channel_id(self) -> None:
        rule = AlertRule(
            event_kind="rate_exceeded",
            action="notify",
            channel_filter="level",
        )
        assert rule.matches("rate_exceeded", "dev1", "level")
        assert rule.matches("rate_exceeded", "dev2", "level")
        assert not rule.matches("rate_exceeded", "dev1", "knob")


class TestAlertRuleFromDict:
    def test_basic_parse(self) -> None:
        rule = AlertRule.from_dict({
            "event_kind": "threshold_exceeded",
            "action": "hw_estop",
            "channel_filter": "dev1.level",
            "require_consecutive": 5,
        })
        assert rule.event_kind == "threshold_exceeded"
        assert rule.action == "hw_estop"
        assert rule.channel_filter == "dev1.level"
        assert rule.require_consecutive == 5

    def test_defaults(self) -> None:
        rule = AlertRule.from_dict({"event_kind": "stale", "action": "alert"})
        assert rule.channel_filter == ""
        assert rule.require_consecutive == DEFAULT_CONSECUTIVE

    def test_consecutive_clamped_to_one(self) -> None:
        rule = AlertRule.from_dict({
            "event_kind": "x",
            "action": "y",
            "require_consecutive": 0,
        })
        assert rule.require_consecutive == 1


class TestLoadAlertPolicies:
    def test_empty_settings(self) -> None:
        settings = SimpleNamespace()
        assert load_alert_policies(settings) == ()

    def test_non_list_ignored(self) -> None:
        settings = SimpleNamespace(hardware_alert_policies="not a list")
        assert load_alert_policies(settings) == ()

    def test_valid_rules_loaded(self) -> None:
        settings = SimpleNamespace(hardware_alert_policies=[
            {"event_kind": "threshold_exceeded", "action": "hw_estop"},
            {"event_kind": "rate_exceeded", "action": "notify", "require_consecutive": 5},
        ])
        rules = load_alert_policies(settings)
        assert len(rules) == 2
        assert rules[0].action == "hw_estop"
        assert rules[1].require_consecutive == 5

    def test_malformed_entries_skipped(self) -> None:
        settings = SimpleNamespace(hardware_alert_policies=[
            "not a dict",
            {"event_kind": "", "action": "hw_estop"},  # missing event_kind
            {"event_kind": "stale", "action": "alert"},  # valid
        ])
        rules = load_alert_policies(settings)
        assert len(rules) == 1
        assert rules[0].event_kind == "stale"


class TestBuildAlertPolicy:
    def test_returns_none_when_no_rules(self) -> None:
        settings = SimpleNamespace()
        assert build_alert_policy(settings) is None

    def test_returns_policy_when_rules_present(self) -> None:
        settings = SimpleNamespace(hardware_alert_policies=[
            {"event_kind": "threshold_exceeded", "action": "hw_estop"},
        ])
        policy = build_alert_policy(settings)
        assert policy is not None
        assert len(policy.rules) == 1


class TestConsecutiveGating:
    def _event(self, kind: str = "threshold_exceeded") -> HardwareEvent:
        return HardwareEvent(
            kind=kind,
            device_id="dev1",
            channel_id="level",
            quantity="generic.level",
            detail="test",
            observed_at=time.time(),
        )

    def test_fires_only_after_consecutive_hits(self) -> None:
        rule = AlertRule(event_kind="threshold_exceeded", action="hw_estop", require_consecutive=3)
        policy = HardwareAlertPolicy((rule,))

        policy.evaluate(self._event())
        assert policy.fired_count == 0
        policy.evaluate(self._event())
        assert policy.fired_count == 0
        policy.evaluate(self._event())
        # Third consecutive hit should fire (no registry, so estop logs error)
        assert policy.fired_count == 1

    def test_different_kinds_tracked_independently(self) -> None:
        rule_a = AlertRule(event_kind="threshold_exceeded", action="hw_estop", require_consecutive=2)
        rule_b = AlertRule(event_kind="rate_exceeded", action="notify", require_consecutive=2)
        policy = HardwareAlertPolicy((rule_a, rule_b))

        policy.evaluate(self._event("threshold_exceeded"))
        policy.evaluate(self._event("rate_exceeded"))
        assert policy.fired_count == 0

        policy.evaluate(self._event("threshold_exceeded"))
        assert policy.fired_count == 1  # threshold hit twice

    def test_reset_channel_clears_counters(self) -> None:
        rule = AlertRule(event_kind="threshold_exceeded", action="hw_estop", require_consecutive=3)
        policy = HardwareAlertPolicy((rule,))

        policy.evaluate(self._event())
        policy.evaluate(self._event())
        assert policy.fired_count == 0

        policy.reset_channel("dev1", "level")
        policy.evaluate(self._event())
        assert policy.fired_count == 0  # counter was reset, only 1 now


class TestEstopDispatch:
    @pytest.mark.asyncio
    async def test_estop_calls_transport_halt(self) -> None:
        halted = []

        class _Transport:
            async def halt(self):
                halted.append(True)
                return SimpleNamespace(halt_supported=True)

        class _Registry:
            async def transport(self, device_id: str):
                return _Transport()

        rule = AlertRule(event_kind="threshold_exceeded", action="hw_estop", require_consecutive=1)
        policy = HardwareAlertPolicy((rule,), registry=_Registry())

        event = HardwareEvent(
            kind="threshold_exceeded",
            device_id="dev1",
            channel_id="level",
            quantity="q",
            detail="d",
            observed_at=time.time(),
        )
        policy.evaluate(event)

        # Let the fire-and-forget task run
        await asyncio.sleep(0.05)
        assert halted


class TestApprovalDispatch:
    @pytest.mark.asyncio
    async def test_non_estop_routes_through_orchestrator(self) -> None:
        evaluated = []

        class _Orchestrator:
            async def evaluate(self, action: Any):
                evaluated.append(action)
                return SimpleNamespace(approved=True)

        rule = AlertRule(event_kind="stale", action="notify_operator", require_consecutive=1)
        policy = HardwareAlertPolicy((rule,), orchestrator=_Orchestrator())

        event = HardwareEvent(
            kind="stale",
            device_id="dev1",
            channel_id="level",
            quantity="q",
            detail="d",
            observed_at=time.time(),
        )
        policy.evaluate(event)

        await asyncio.sleep(0.05)
        assert len(evaluated) == 1
        assert evaluated[0].kind == "device.alert.notify_operator"


# ════════════════════════════════════════════════════════════════
# 2.6 HardwareMetricsExporter
# ════════════════════════════════════════════════════════════════


class TestBuildExporter:
    def test_default_off(self) -> None:
        settings = SimpleNamespace()
        assert build_exporter(settings) is None

    def test_explicit_false(self) -> None:
        settings = SimpleNamespace(hardware_metrics_export_enabled=False)
        assert build_exporter(settings) is None

    def test_enabled(self) -> None:
        settings = SimpleNamespace(hardware_metrics_export_enabled=True)
        exporter = build_exporter(settings)
        assert exporter is not None
        assert isinstance(exporter, HardwareMetricsExporter)


class TestMetricSample:
    def test_prometheus_line_no_labels(self) -> None:
        sample = MetricSample(name="m", kind="gauge", value=42.0)
        assert sample.prometheus_line() == "m 42.0"

    def test_prometheus_line_with_labels(self) -> None:
        sample = MetricSample(
            name="m",
            kind="counter",
            value=7.0,
            labels=(("device_id", "dev1"), ("channel_id", "ch1")),
        )
        assert sample.prometheus_line() == 'm{device_id="dev1",channel_id="ch1"} 7.0'


class TestExporterCollect:
    def test_store_metrics(self) -> None:
        store = SimpleNamespace(
            write_failures=3,
            raw_writes=100,
            windows_written=50,
            rows_pruned=10,
        )
        exporter = HardwareMetricsExporter(reading_store=store)
        samples = exporter.collect()
        names = {s.name for s in samples}
        assert WRITE_FAILURES_TOTAL in names
        assert RAW_WRITES_TOTAL in names

        wf = next(s for s in samples if s.name == WRITE_FAILURES_TOTAL)
        assert wf.value == 3.0
        assert wf.kind == "counter"

    def test_stream_metrics(self) -> None:
        health = {
            "device_id": "dev1",
            "channel_id": "level",
            "samples": 1000,
            "dropped": 5,
            "events_paced_out": 2,
            "skipped_slots": 1,
            "observed_hz": 9.8,
            "rate_ratio": 0.98,
        }
        source = SimpleNamespace(health=health)
        registry = SimpleNamespace(stream_sources=(source,), contexts=lambda: ())
        exporter = HardwareMetricsExporter(registry=registry)
        samples = exporter.collect()
        names = {s.name for s in samples}
        assert SAMPLES_TOTAL in names
        assert DROPPED_TOTAL in names
        assert OBSERVED_HZ in names
        assert STREAM_SOURCES_ACTIVE in names

        active = next(s for s in samples if s.name == STREAM_SOURCES_ACTIVE)
        assert active.value == 1.0

    def test_registry_device_count(self) -> None:
        registry = SimpleNamespace(
            contexts=lambda: ("ctx1", "ctx2"),
            stream_sources=(),
        )
        exporter = HardwareMetricsExporter(registry=registry)
        samples = exporter.collect()
        dev = next(s for s in samples if s.name == DEVICES_ADMITTED)
        assert dev.value == 2.0

    def test_alert_policy_metric(self) -> None:
        policy = SimpleNamespace(fired_count=7)
        exporter = HardwareMetricsExporter(alert_policy=policy)
        samples = exporter.collect()
        ap = next(s for s in samples if s.name == ALERT_POLICY_FIRED_TOTAL)
        assert ap.value == 7.0

    def test_render_prometheus(self) -> None:
        store = SimpleNamespace(write_failures=1, raw_writes=2, windows_written=3, rows_pruned=0)
        exporter = HardwareMetricsExporter(reading_store=store)
        text = exporter.render_prometheus()
        assert "# HELP" in text
        assert "# TYPE" in text
        assert WRITE_FAILURES_TOTAL in text
        assert text.endswith("\n")

    def test_empty_collect_no_crash(self) -> None:
        exporter = HardwareMetricsExporter()
        samples = exporter.collect()
        assert samples == []


# ════════════════════════════════════════════════════════════════
# IC-6 Calibration lifecycle events
# ════════════════════════════════════════════════════════════════


class TestCalibrationEventKinds:
    def test_all_four_kinds_exist(self) -> None:
        assert EventKind.CALIBRATION_STARTED == "calibration_started"
        assert EventKind.CALIBRATION_COMPLETED == "calibration_completed"
        assert EventKind.CALIBRATION_FAILED == "calibration_failed"
        assert EventKind.CALIBRATION_EXPIRED == "calibration_expired"

    def test_calibration_event_type_uses_hw_prefix(self) -> None:
        event = HardwareEvent(
            kind=EventKind.CALIBRATION_STARTED,
            device_id="dev1",
            channel_id="level",
            quantity="generic.level",
            detail="calibration initiated",
        )
        assert event.event_type == "hw.calibration_started"


class TestCalibrationEventSeverity:
    """Calibration events should be classified correctly in the digest."""

    def test_failed_is_notable(self) -> None:
        from leapflow.hardware.observability.digest import _event_severity

        assert _event_severity("calibration_failed") == "notable"

    def test_expired_is_notable(self) -> None:
        from leapflow.hardware.observability.digest import _event_severity

        assert _event_severity("calibration_expired") == "notable"

    def test_started_is_info(self) -> None:
        from leapflow.hardware.observability.digest import _event_severity

        assert _event_severity("calibration_started") == "info"

    def test_completed_is_info(self) -> None:
        from leapflow.hardware.observability.digest import _event_severity

        assert _event_severity("calibration_completed") == "info"


class TestCalibrationDashboard:
    """The hardware.yaml template must validate against the SDUI component catalog."""

    def test_hardware_template_validates(self) -> None:
        from leapflow.dashboard.templates import TemplateLibrary
        from leapflow.dashboard.viewspec import validate_viewspec

        lib = TemplateLibrary()
        # Render with enough data to show every section
        payload = {
            "hardware": {
                "schema_version": 1,
                "counts": {"devices": 1, "series": 1, "events": 1, "outcomes": 1},
                "devices": [{"label": "d", "transport_kind": "mock"}],
                "series": [{"id": "d.ch", "label": "l", "points": [{"x": 1, "y": 2}]}],
                "events": [{"title": "e", "summary": "s", "severity": "info", "x": 1}],
                "conformance_mix": [{"label": "inside", "value": 10}],
                "sampling": [{"channel_id": "ch", "declared_hz": 10, "observed_hz": 9.9}],
                "outcomes": [{"channel_id": "ch", "command": "c", "outcome": "o", "delta": 0.1}],
                "calibration": [
                    {
                        "channel_id": "d.level",
                        "state": "valid",
                        "calibrated_at": "2026-08-01T00:00:00Z",
                        "days_since": 30,
                        "residual": 0.02,
                        "next_recal_due": "2026-11-01T00:00:00Z",
                    }
                ],
                "storage": {"write_failures": 0, "raw_writes": 100},
            },
            "observation": {"watch_state": "armed"},
        }
        spec = lib.render("hardware", payload)
        errors = validate_viewspec(spec)
        assert not errors, f"hardware.yaml validation errors: {errors}"

    def test_calibration_section_present(self) -> None:
        """The rendered spec must contain the calibration section."""
        from leapflow.dashboard.templates import TemplateLibrary

        lib = TemplateLibrary()
        payload = {
            "hardware": {
                "counts": {"devices": 0, "series": 0, "events": 0, "outcomes": 0},
                "calibration": [
                    {
                        "channel_id": "d.level",
                        "state": "expired",
                        "calibrated_at": "",
                        "days_since": 999,
                        "residual": 0.0,
                        "next_recal_due": "",
                    }
                ],
                "storage": {},
            },
            "observation": {},
            # No mode flag is needed. Every fleet panel gates on the data it renders --
            # this one on ``hardware.calibration`` -- so supplying the data is what makes
            # it appear. An earlier revision gated the section on a synthetic ``fleet``
            # key instead, which is how a board served by a process that predated the key
            # rendered a bare title and nothing else.
        }
        spec = lib.render("hardware", payload)
        # Walk the tree to find the calibration Section
        found = False
        for node in _walk(spec.get("root", [])):
            if (
                node.get("type") == "Section"
                and "alibration" in str(node.get("props", {}).get("title", ""))
            ):
                found = True
                break
        assert found, "Calibration Health section not found in rendered hardware template"


# ════════════════════════════════════════════════════════════════
# Integration: alert policy wired into stream dispatch
# ════════════════════════════════════════════════════════════════


class TestStreamAlertPolicyIntegration:
    """Verify that _dispatch calls alert_policy.evaluate."""

    def test_dispatch_calls_policy(self) -> None:
        evaluated = []

        class _Policy:
            def evaluate(self, event: Any) -> None:
                evaluated.append(event)

            def reset_channel(self, device_id: str, channel_id: str) -> None:
                pass

        from leapflow.hardware.context import (
            HC_VERSION,
            Channel,
            ContextProvenance,
            Direction,
            Envelope,
            HardwareContext,
            TransportRef,
        )
        from leapflow.hardware.stream import HardwareStreamSource

        context = HardwareContext(
            device_id="dev",
            hc_version=HC_VERSION,
            halt_supported=True,
            transport=TransportRef(kind="mock", config={}),
            channels=(
                Channel(
                    channel_id="ch",
                    direction=Direction.READ.value,
                    quantity="q",
                    unit="u",
                    sample_rate_hz=10.0,
                    envelope=Envelope(declared=True, min_value=0, max_value=100),
                ),
            ),
            provenance=ContextProvenance(verified_by="test"),
        )

        source = HardwareStreamSource(
            None, context, context.channels[0], alert_policy=_Policy()
        )

        event = HardwareEvent(
            kind=EventKind.THRESHOLD_EXCEEDED,
            device_id="dev",
            channel_id="ch",
            quantity="q",
            detail="d",
            observed_at=time.time(),
        )
        source._dispatch([event], None)
        assert len(evaluated) == 1
        assert evaluated[0].kind == EventKind.THRESHOLD_EXCEEDED

    def test_dispatch_resets_on_settled(self) -> None:
        resets = []

        class _Policy:
            def evaluate(self, event: Any) -> None:
                pass

            def reset_channel(self, device_id: str, channel_id: str) -> None:
                resets.append((device_id, channel_id))

        from leapflow.hardware.context import (
            HC_VERSION,
            Channel,
            ContextProvenance,
            Direction,
            Envelope,
            HardwareContext,
            TransportRef,
        )
        from leapflow.hardware.stream import HardwareStreamSource

        context = HardwareContext(
            device_id="dev",
            hc_version=HC_VERSION,
            halt_supported=True,
            transport=TransportRef(kind="mock", config={}),
            channels=(
                Channel(
                    channel_id="ch",
                    direction=Direction.READ.value,
                    quantity="q",
                    unit="u",
                    sample_rate_hz=10.0,
                    envelope=Envelope(declared=True, min_value=0, max_value=100),
                ),
            ),
            provenance=ContextProvenance(verified_by="test"),
        )

        source = HardwareStreamSource(
            None, context, context.channels[0], alert_policy=_Policy()
        )

        settled_event = HardwareEvent(
            kind=EventKind.SETTLED,
            device_id="dev",
            channel_id="ch",
            quantity="q",
            detail="recovered",
            observed_at=time.time(),
        )
        source._dispatch([settled_event], None)
        assert ("dev", "ch") in resets


# ════════════════════════════════════════════════════════════════
# Helper
# ════════════════════════════════════════════════════════════════


def _walk(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten a ViewSpec tree into a list of all nodes."""
    result: list[dict[str, Any]] = []
    for node in nodes:
        result.append(node)
        children = node.get("children", [])
        if isinstance(children, list):
            result.extend(_walk(children))
    return result
