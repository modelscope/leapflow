# Copyright (c) Alibaba, Inc. and its affiliates.
"""End-to-end tests for event-driven Watch full chain.

Verifies the complete pipeline:
  EventBus emit → EventBridge match → EventTrigger activate →
  MonitorManager recognises DUE → Producer executes → Finding persisted.

No network, no LLM: uses a temporary DuckDB and a fake in-process producer.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from leapflow.domain.events import SystemEvent
from leapflow.monitor import (
    EVENT_FINDING,
    Finding,
    MonitorManager,
    ProducerRegistry,
    Severity,
    WatchSpec,
)
from leapflow.monitor.types import ProducerContext
from leapflow.storage.connection import LocalConnectionHolder


# ── Helpers ──────────────────────────────────────────────────────────────────


class _MockProducer:
    """Deterministic producer returning a fixed finding per cycle."""

    def __init__(self, domain: str, findings: list[Finding]) -> None:
        self._domain = domain
        self._findings = findings
        self.calls = 0

    @property
    def domain(self) -> str:
        return self._domain

    async def observe(self, ctx: ProducerContext) -> list[Finding]:
        self.calls += 1
        return list(self._findings)


def _holder(tmp_path: Path) -> LocalConnectionHolder:
    return LocalConnectionHolder(tmp_path / "leap.duckdb")


def _make_event(event_type: str, source: str = "test") -> SystemEvent:
    """Create a minimal SystemEvent for testing."""
    return SystemEvent(
        event_type=event_type,
        source=source,
        payload={},
        timestamp=time.time(),
    )


# ── 1. Full chain E2E ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_driven_watch_full_chain(tmp_path: Path) -> None:
    """Full chain: event → bridge → trigger → run_watch_once → finding persisted."""
    emitted: list[tuple[str, dict]] = []
    producers = ProducerRegistry()
    producer = _MockProducer("filesystem", [
        Finding(
            watch_id="",
            domain="filesystem",
            title="file changed",
            severity=Severity.ALERT,
            dedup_key="f1",
        ),
    ])
    producers.register(producer)

    manager = MonitorManager(
        holder=_holder(tmp_path),
        producers=producers,
        emit=lambda et, payload: emitted.append((et, payload)),
    )

    # Step 1: arm watch with event trigger
    view = await manager.arm_watch(
        WatchSpec(name="FSWatch", domain="filesystem", trigger_expr="event:fs.*")
    )
    assert view.state == "armed"
    assert manager.event_bridge.active_count == 1

    # Step 2: deliver a matching event through the bridge
    manager.event_bridge.on_event(_make_event("fs.change"))

    # Step 3: verify the trigger is activated
    trigger = manager.event_bridge._triggers[view.watch_id]
    assert trigger.is_triggered is True
    assert trigger.last_event == "fs.change"

    # Step 4: manually run the watch cycle (simulates scheduler tick)
    result = await manager.run_watch_once(view.watch_id)
    assert result["ok"] is True
    assert result["findings"] == 1

    # Step 5: verify finding was pushed and persisted
    finding_events = [p for et, p in emitted if et == EVENT_FINDING]
    assert len(finding_events) == 1
    assert finding_events[0]["title"] == "file changed"

    assert manager.finding_store.count(watch_id=view.watch_id) == 1
    assert producer.calls == 1


# ── 2. Debounce suppression ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_driven_watch_debounce_suppression(tmp_path: Path) -> None:
    """Rapid events within debounce window fire trigger only once."""
    producers = ProducerRegistry()
    producers.register(_MockProducer("filesystem", [
        Finding(watch_id="", domain="filesystem", title="x", severity=Severity.INFO),
    ]))
    manager = MonitorManager(
        holder=_holder(tmp_path),
        producers=producers,
    )

    view = await manager.arm_watch(
        WatchSpec(name="FSWatch", domain="filesystem", trigger_expr="event:fs.change")
    )

    # Use a fixed monotonic clock so all events land within the debounce window
    fake_time = [100.0]

    def _monotonic() -> float:
        return fake_time[0]

    with patch("leapflow.monitor.event_bridge.time.monotonic", side_effect=_monotonic):
        # First event fires the trigger
        manager.event_bridge.on_event(_make_event("fs.change"))
        trigger = manager.event_bridge._triggers[view.watch_id]
        assert trigger.is_triggered is True

        # Send 9 more rapid events — all should be debounced
        for _ in range(9):
            manager.event_bridge.on_event(_make_event("fs.change"))

    # Trigger was only activated once (the first)
    assert trigger.is_triggered is True  # still set from the first fire

    # 9 events were suppressed by debounce
    assert manager.event_bridge.debounced_count(view.watch_id) == 9


# ── 3. No match, no trigger ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_driven_watch_no_match_no_trigger(tmp_path: Path) -> None:
    """Non-matching event does not activate trigger."""
    manager = MonitorManager(holder=_holder(tmp_path))
    view = await manager.arm_watch(
        WatchSpec(name="FSWatch", domain="filesystem", trigger_expr="event:fs.*")
    )

    # Send an unrelated event type
    manager.event_bridge.on_event(_make_event("app.focus_change"))

    # Trigger should NOT be activated
    trigger = manager.event_bridge._triggers[view.watch_id]
    assert trigger.is_triggered is False
    assert trigger.last_event == ""


# ── 4. Latency under 200ms ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_driven_watch_latency_under_200ms(tmp_path: Path) -> None:
    """Pure internal path (no LLM) completes within 200ms budget."""
    producers = ProducerRegistry()
    producers.register(_MockProducer("perf", [
        Finding(watch_id="", domain="perf", title="fast", severity=Severity.ALERT),
    ]))
    manager = MonitorManager(
        holder=_holder(tmp_path),
        producers=producers,
    )

    t0 = time.monotonic()

    view = await manager.arm_watch(
        WatchSpec(name="PerfWatch", domain="perf", trigger_expr="event:perf.*")
    )
    manager.event_bridge.on_event(_make_event("perf.tick"))
    result = await manager.run_watch_once(view.watch_id)

    elapsed_ms = (time.monotonic() - t0) * 1000

    assert result["ok"] is True
    assert result["findings"] == 1
    assert elapsed_ms < 200, f"Latency {elapsed_ms:.1f}ms exceeds 200ms budget"


# ── 5. Stop unregisters from bridge ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_driven_watch_stop_unregisters(tmp_path: Path) -> None:
    """Stopping a watch unregisters its trigger from EventBridge."""
    manager = MonitorManager(holder=_holder(tmp_path))

    view = await manager.arm_watch(
        WatchSpec(name="FSWatch", domain="filesystem", trigger_expr="event:fs.*")
    )
    assert manager.event_bridge.active_count == 1

    # Stop the watch
    stopped = manager.stop_watch(view.watch_id)
    assert stopped.state == "done"
    assert manager.event_bridge.active_count == 0

    # Sending a matching event should have no effect (trigger already removed)
    manager.event_bridge.on_event(_make_event("fs.change"))

    # No trigger should be in the bridge anymore
    assert view.watch_id not in manager.event_bridge._triggers
