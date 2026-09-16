# Copyright (c) Alibaba, Inc. and its affiliates.
"""Signal flow metrics collection for real-time observability.

Aggregates health metrics from EventBus, EventBridge, buffers, and monitors
into a single snapshot. No persistence — pure in-memory queries.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass(frozen=True)
class TriggerStat:
    """Stats for a single event trigger pattern."""

    watch_id: str
    pattern: str
    triggered: bool
    last_event: str


@dataclass(frozen=True)
class SignalMetricsSnapshot:
    """Point-in-time signal flow health snapshot."""

    timestamp: float
    event_subscriber_count: int
    active_trigger_count: int
    trigger_stats: List[Dict[str, Any]]
    debounce_stats: Dict[str, int]  # watch_id -> debounced_count
    signal_buffer_dropped: int
    reorder_buffer_pending: int
    composite_source_dropped: int
    active_watch_count: int
    recent_findings_count: int
    signal_noise_seen: int = 0
    signal_noise_passed: int = 0
    signal_noise_suppressed: int = 0
    signal_noise_by_reason: Dict[str, int] | None = None
    signal_noise_by_family: Dict[str, int] | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize snapshot to plain dict for RPC transport."""
        return asdict(self)


class SignalMetricsCollector:
    """Gather real-time signal flow metrics from various components."""

    def collect(
        self,
        *,
        event_bus: Any = None,
        monitor_manager: Any = None,
        signal_buffer: Any = None,
        reorder_buffer: Any = None,
        composite_source: Any = None,
        signal_noise_gate: Any = None,
    ) -> SignalMetricsSnapshot:
        """Collect current metrics snapshot (synchronous, fast)."""
        # EventBus stats
        subscriber_count = (
            len(getattr(event_bus, "_subscribers", []))
            if event_bus
            else 0
        )

        # EventBridge stats
        bridge = (
            getattr(monitor_manager, "_event_bridge", None)
            if monitor_manager
            else None
        )
        trigger_count = getattr(bridge, "active_count", 0) if bridge else 0
        triggers = getattr(bridge, "_triggers", {}) if bridge else {}
        debounce: Dict[str, int] = {}
        trigger_stats_list: List[Dict[str, Any]] = []

        if bridge:
            for watch_id, trigger in triggers.items():
                trigger_stats_list.append(
                    {
                        "watch_id": watch_id,
                        "pattern": getattr(trigger, "event_pattern", ""),
                        "triggered": getattr(trigger, "is_triggered", False),
                        "last_event": getattr(trigger, "last_event", ""),
                    }
                )
            debounce_counts = getattr(bridge, "_debounced_count", {})
            debounce = dict(debounce_counts)

        # Buffer stats
        buffer_dropped = (
            getattr(signal_buffer, "dropped_count", 0) if signal_buffer else 0
        )
        reorder_pending = (
            getattr(reorder_buffer, "pending_count", 0) if reorder_buffer else 0
        )
        source_dropped = (
            getattr(composite_source, "drop_count", 0) if composite_source else 0
        )
        noise_stats = {}
        if signal_noise_gate is not None:
            try:
                raw_stats = signal_noise_gate if isinstance(signal_noise_gate, dict) else getattr(signal_noise_gate, "stats", {})
                noise_stats = dict(raw_stats) if isinstance(raw_stats, dict) else {}
            except Exception:
                noise_stats = {}

        # Monitor stats
        active_watches = 0
        findings_count = 0
        if monitor_manager:
            try:
                watches = monitor_manager.list_watches()
                active_watches = len(
                    [w for w in watches if w.state in ("armed", "watching", "due")]
                )
            except Exception:
                pass
            try:
                findings = monitor_manager.list_findings(limit=100)
                findings_count = len(findings)
            except Exception:
                pass

        return SignalMetricsSnapshot(
            timestamp=time.time(),
            event_subscriber_count=subscriber_count,
            active_trigger_count=trigger_count,
            trigger_stats=trigger_stats_list,
            debounce_stats=debounce,
            signal_buffer_dropped=buffer_dropped,
            reorder_buffer_pending=reorder_pending,
            composite_source_dropped=source_dropped,
            active_watch_count=active_watches,
            recent_findings_count=findings_count,
            signal_noise_seen=int(noise_stats.get("seen", 0) or 0),
            signal_noise_passed=int(noise_stats.get("passed", 0) or 0),
            signal_noise_suppressed=int(noise_stats.get("suppressed", 0) or 0),
            signal_noise_by_reason=dict(noise_stats.get("by_reason", {}) or {}),
            signal_noise_by_family=dict(noise_stats.get("by_family", {}) or {}),
        )
