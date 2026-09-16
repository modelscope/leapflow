# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for daemon MonitorCoordinator signal-noise boundary behavior."""
from __future__ import annotations

import time
from dataclasses import dataclass

from leapflow.daemon.monitor_coordinator import MonitorCoordinator
from leapflow.domain.events import SystemEvent
from leapflow.monitor.signal_noise import SignalNoiseConfig, SignalNoiseGate


@dataclass
class _NotificationBus:
    events: list[tuple[str, dict]]

    def emit(self, notification) -> None:
        self.events.append((notification.event_type, notification.payload))


def _event(event_type: str, source: str) -> SystemEvent:
    return SystemEvent(event_type=event_type, source=source, payload={"path": source}, timestamp=time.time())


def test_monitor_signal_subscriber_filters_noise_before_bridge_and_stream() -> None:
    coordinator = MonitorCoordinator()
    coordinator._signal_noise_gate = SignalNoiseGate(
        SignalNoiseConfig(workspace_root="/repo", same_source_cooldown_s=0.0)
    )
    notifications = _NotificationBus([])
    bridged: list[str] = []
    subscriber = coordinator._make_monitor_signal_subscriber(
        lambda event: bridged.append(event.source),
        notifications,
    )

    noisy = _event(
        "fs.change",
        "/Users/jason/Library/Application Support/Qoder/User/globalStorage/state.vscdb-shm",
    )
    clean = _event("fs.change", "/repo/src/app.py")

    subscriber(noisy)
    subscriber(clean)

    assert bridged == ["/repo/src/app.py"]
    assert coordinator.get_signal_stream() == [
        {"event_type": "fs.change", "source": "/repo/src/app.py", "ts": clean.timestamp}
    ]
    assert notifications.events == [
        ("signal.stream", {"event_type": "fs.change", "source": "/repo/src/app.py", "ts": clean.timestamp})
    ]
    assert coordinator.signal_noise_stats["seen"] == 2
    assert coordinator.signal_noise_stats["passed"] == 1
    assert coordinator.signal_noise_stats["suppressed"] == 1


def test_monitor_signal_subscriber_keeps_gateway_events() -> None:
    coordinator = MonitorCoordinator()
    coordinator._signal_noise_gate = SignalNoiseGate(SignalNoiseConfig(workspace_root="/repo"))
    notifications = _NotificationBus([])
    bridged: list[str] = []
    subscriber = coordinator._make_monitor_signal_subscriber(
        lambda event: bridged.append(event.event_type),
        notifications,
    )
    event = _event("gateway.signal", "lark")

    subscriber(event)

    assert bridged == ["gateway.signal"]
    assert coordinator.get_signal_stream() == [
        {"event_type": "gateway.signal", "source": "lark", "ts": event.timestamp}
    ]
    assert coordinator.signal_noise_stats["suppressed"] == 0
