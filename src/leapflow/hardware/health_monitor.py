# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hardware health monitor: centralized device health supervision.

Aggregates health signals from multiple sources:
- HardwareStreamSource / BatchStreamCoordinator: comm_loss, sensor_loss events
- HighFrequencyControlBus: overrun, safety_violation stats
- HardwareRegistry: device probe results
- FleetManager: node heartbeat status

Executes DegradationPolicy actions based on aggregated health:
- halt: stop device immediately
- hold_position: stop sending new commands
- safe_return: plan trajectory to home position (if kinematics available)
- switch_sensor: attempt to use backup sensor (if available)
- continue_blind: log warning, continue without the degraded sensor

Design:
- Subscribes to EventBus for hardware events (``hw.*``, ``control.*``)
- Periodically polls device health via registry probe
- When a device's health crosses a threshold, executes the DegradationPolicy
  action declared in its HardwareContext
- Best-effort: all monitoring is non-blocking, and failures are logged but
  never propagate to the caller or block the data plane
- No dependency on ``leapflow.engine``
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from leapflow.hardware.degradation import DegradationCoordinator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Health status constants
# ---------------------------------------------------------------------------

_HEALTH_OK = "ok"
_HEALTH_WARNING = "warning"
_HEALTH_DEGRADED = "degraded"
_HEALTH_CRITICAL = "critical"
_HEALTH_OFFLINE = "offline"

_VALID_HEALTH_STATES = frozenset({
    _HEALTH_OK,
    _HEALTH_WARNING,
    _HEALTH_DEGRADED,
    _HEALTH_CRITICAL,
    _HEALTH_OFFLINE,
})


@dataclass
class DeviceHealthRecord:
    """Mutable health record for one device.

    Tracks recent failures, degradation state, and the timestamp of the
    last successful probe.  Updated by the poll loop and by event handlers.
    """

    device_id: str
    status: str = _HEALTH_OK
    last_probe_ok: bool = True
    last_probe_time: float = 0.0
    consecutive_probe_failures: int = 0
    comm_loss_events: int = 0
    sensor_loss_events: int = 0
    degradation_policy_active: str = ""  # which policy action is in effect
    degradation_reason: str = ""
    last_event_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for tools / dashboard consumption."""
        return {
            "device_id": self.device_id,
            "status": self.status,
            "last_probe_ok": self.last_probe_ok,
            "last_probe_time": self.last_probe_time,
            "consecutive_probe_failures": self.consecutive_probe_failures,
            "comm_loss_events": self.comm_loss_events,
            "sensor_loss_events": self.sensor_loss_events,
            "degradation_policy_active": self.degradation_policy_active,
            "degradation_reason": self.degradation_reason,
            "last_event_time": self.last_event_time,
        }


# ---------------------------------------------------------------------------
# HardwareHealthMonitor
# ---------------------------------------------------------------------------


class HardwareHealthMonitor:
    """Centralized health monitoring and degradation policy executor.

    Subscribes to EventBus for hardware events and periodically
    polls device health.  When a device's health crosses a threshold,
    executes the DegradationPolicy action declared in its HardwareContext.

    All operations are best-effort: monitoring failures are logged but
    never propagate.  The monitor does not gate the data plane; it only
    issues actions (halt, hold_position) and reports state.
    """

    def __init__(
        self,
        registry: Any,
        *,
        event_bus: Any = None,
        fleet_manager: Any = None,
        poll_interval_s: float = 5.0,
    ) -> None:
        self._registry = registry
        self._event_bus = event_bus
        self._fleet_manager = fleet_manager
        self._poll_interval = max(1.0, float(poll_interval_s))

        # Per-device health tracking.
        self._records: dict[str, DeviceHealthRecord] = {}
        self._poll_task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()
        self._started = False

        # EventBus subscription ids for cleanup.
        self._subscriptions: list[Any] = []

    # -- Public lifecycle ---------------------------------------------------

    async def start(self) -> None:
        """Start health monitoring (subscribe to events + start poll loop).

        Idempotent: calling twice is a no-op.
        """
        if self._started:
            return

        self._stopped.clear()
        self._subscribe_events()

        # Seed initial health records from the registry.
        self._seed_records()

        self._poll_task = asyncio.create_task(
            self._poll_loop(), name="hw-health-monitor"
        )
        self._started = True
        logger.info(
            "HardwareHealthMonitor started (poll_interval=%.1fs, devices=%d)",
            self._poll_interval,
            len(self._records),
        )

    async def stop(self) -> None:
        """Stop health monitoring.  Idempotent.  Never raises."""
        if not self._started:
            return

        self._stopped.set()
        self._unsubscribe_events()

        task = self._poll_task
        self._poll_task = None
        if task is not None:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception:  # noqa: BLE001 — teardown must not propagate
                logger.warning("Health monitor poll stop raised", exc_info=True)

        self._started = False
        logger.info("HardwareHealthMonitor stopped")

    # -- Health queries -----------------------------------------------------

    def device_health(self, device_id: str) -> dict[str, Any]:
        """Return current health assessment for one device."""
        record = self._records.get(device_id)
        if record is None:
            return {"device_id": device_id, "status": "unknown", "error": "not tracked"}
        return record.to_dict()

    def fleet_health(self) -> dict[str, Any]:
        """Return health assessment for all devices (local + fleet)."""
        devices = {did: rec.to_dict() for did, rec in self._records.items()}

        # Aggregate fleet node health if available.
        fleet_nodes: dict[str, Any] = {}
        if self._fleet_manager is not None:
            try:
                topo = self._fleet_manager.topology()
                for node in topo.nodes:
                    fleet_nodes[node.node_id] = {
                        "status": node.status,
                        "device_count": len(node.device_ids),
                        "last_heartbeat": node.last_heartbeat,
                    }
            except Exception:  # noqa: BLE001
                logger.debug("Fleet topology query failed", exc_info=True)

        status_counts: dict[str, int] = {}
        for rec in self._records.values():
            status_counts[rec.status] = status_counts.get(rec.status, 0) + 1

        return {
            "total_devices": len(devices),
            "status_counts": status_counts,
            "devices": devices,
            "fleet_nodes": fleet_nodes,
            "timestamp": time.time(),
        }

    # -- Degradation policy execution ---------------------------------------

    async def _evaluate_health(self, device_id: str) -> None:
        """Evaluate device health and execute degradation policy if needed.

        Called after each probe cycle or significant event.  Determines the
        device's overall health status and, if degraded, looks up the
        DegradationPolicy in its HardwareContext to decide what to do.
        """
        record = self._records.get(device_id)
        if record is None:
            return

        # Determine health status from the record.
        prev_status = record.status
        new_status = self._compute_health_status(record)
        record.status = new_status

        # If status worsened, check degradation policy.
        if new_status in (_HEALTH_DEGRADED, _HEALTH_CRITICAL) and prev_status == _HEALTH_OK:
            await self._apply_degradation_policy(device_id, record)
        elif new_status == _HEALTH_OFFLINE:
            await self._execute_halt(device_id, reason="device offline (probe failed)")

    @staticmethod
    def _compute_health_status(record: DeviceHealthRecord) -> str:
        """Determine overall health status from the record's counters."""
        if record.consecutive_probe_failures >= 5:
            return _HEALTH_OFFLINE
        if record.consecutive_probe_failures >= 3:
            return _HEALTH_CRITICAL
        if record.degradation_policy_active:
            return _HEALTH_DEGRADED
        if record.comm_loss_events > 0 or record.sensor_loss_events > 0:
            return _HEALTH_WARNING
        if not record.last_probe_ok:
            return _HEALTH_WARNING
        return _HEALTH_OK

    async def _apply_degradation_policy(
        self, device_id: str, record: DeviceHealthRecord
    ) -> None:
        """Look up and execute the DegradationPolicy for a device."""
        context = self._get_device_context(device_id)
        if context is None:
            logger.warning(
                "HealthMonitor: no context for device %s; defaulting to halt",
                device_id,
            )
            await self._execute_halt(device_id, reason="no context available")
            return

        degradation = getattr(context, "degradation", None)
        if degradation is None:
            # No degradation policy declared; default to halt (fail-closed).
            await self._execute_halt(device_id, reason="no degradation policy declared")
            return

        # Choose action based on the dominant failure mode.
        if record.comm_loss_events > record.sensor_loss_events:
            policy_action = getattr(degradation, "comm_loss_policy", "halt") or "halt"
            reason = "comm_loss"
        else:
            policy_action = getattr(degradation, "sensor_loss_policy", "halt") or "halt"
            reason = "sensor_loss"

        record.degradation_reason = reason

        outcome = await DegradationCoordinator(self._registry, context).execute(
            policy_action,
            reason=reason,
        )
        record.degradation_policy_active = outcome.action
        if outcome.error:
            record.degradation_reason = f"{reason}: {outcome.error}"

    async def _execute_halt(self, device_id: str, reason: str) -> None:
        """Halt a device immediately via the registry transport."""
        logger.warning(
            "HealthMonitor: halting device %s (reason: %s)", device_id, reason,
        )
        try:
            transport = await self._registry.transport(device_id)
            await transport.halt()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "HealthMonitor: halt failed for %s: %s",
                device_id, exc, exc_info=True,
            )
        self._emit_event("device.halted", {
            "device_id": device_id,
            "reason": reason,
        })

    async def _execute_hold_position(self, device_id: str) -> None:
        """Mark a device for hold-position (no new commands)."""
        logger.warning(
            "HealthMonitor: hold_position on device %s", device_id,
        )
        # The hold_position behavior is enforced by the write path checking
        # is_device_degraded() — we only need to log and emit the event.
        # The actual degradation_triggered flag is set by HardwareStreamSource.
        self._emit_event("device.hold_position", {"device_id": device_id})

    async def _execute_safe_return(self, device_id: str) -> None:
        """Plan trajectory to home position using kinematics declaration.

        If kinematics are not available, falls back to hold_position.
        """
        context = self._get_device_context(device_id)
        kinematics = getattr(context, "kinematics", None) if context else None
        if kinematics is None:
            logger.warning(
                "HealthMonitor: safe_return requested for %s but no kinematics; "
                "falling back to hold_position",
                device_id,
            )
            await self._execute_hold_position(device_id)
            return

        # For now, safe_return is treated as hold_position + logged intent.
        # A full trajectory planner to the home position requires the RT
        # control loop, which is a future extension.
        logger.info(
            "HealthMonitor: safe_return initiated for %s "
            "(currently implemented as hold_position + intent)",
            device_id,
        )
        await self._execute_hold_position(device_id)
        self._emit_event("device.safe_return_requested", {
            "device_id": device_id,
            "kinematics_available": True,
        })

    async def _execute_switch_sensor(
        self, device_id: str, channel_id: str
    ) -> None:
        """Attempt to use backup sensor channel if available.

        Looks for alternative readable channels on the device and logs the
        switch attempt.  Actual sensor switching depends on the device's
        transport supporting channel remapping.
        """
        logger.info(
            "HealthMonitor: switch_sensor requested for %s (channel=%s)",
            device_id, channel_id or "(auto)",
        )
        context = self._get_device_context(device_id)
        if context is None:
            return

        # Find readable channels that could serve as backup.
        backup_channels = [
            ch.channel_id
            for ch in context.channels
            if ch.is_readable and ch.channel_id != channel_id
            and getattr(ch, "representation", "") != "frame"
        ]
        if backup_channels:
            logger.info(
                "HealthMonitor: available backup sensors for %s: %s",
                device_id, backup_channels,
            )
        else:
            logger.warning(
                "HealthMonitor: no backup sensors available for %s; "
                "continuing with degraded sensor",
                device_id,
            )

        self._emit_event("device.switch_sensor", {
            "device_id": device_id,
            "channel_id": channel_id,
            "backup_channels": backup_channels,
        })

    # -- Event handling (EventBus subscribers) ------------------------------

    def _on_hardware_event(self, event: Any) -> None:
        """Handle a hardware event from EventBus.

        Events processed:
        - hw.degradation.*: stream source detected comm/sensor loss
        - control.telemetry: control bus stats (overruns, violations)
        - control.hierarchy.escalation: control layer escalation
        """
        if not isinstance(event, dict):
            return

        device_id = event.get("device_id", "")
        if not device_id:
            return

        record = self._records.get(device_id)
        if record is None:
            record = DeviceHealthRecord(device_id=device_id)
            self._records[device_id] = record

        record.last_event_time = time.monotonic()

        event_type = event.get("type", "")
        if "comm_loss" in str(event_type):
            record.comm_loss_events += 1
        elif "sensor_loss" in str(event_type):
            record.sensor_loss_events += 1

    def _on_control_event(self, event: Any) -> None:
        """Handle control telemetry / escalation events."""
        if not isinstance(event, dict):
            return

        device_id = event.get("device_id", "")
        if not device_id:
            return

        record = self._records.get(device_id)
        if record is None:
            return

        record.last_event_time = time.monotonic()

    # -- EventBus subscription management -----------------------------------

    def _subscribe_events(self) -> None:
        """Subscribe to relevant EventBus topics."""
        if self._event_bus is None:
            return

        subscribe = getattr(self._event_bus, "on", None) or getattr(
            self._event_bus, "subscribe", None
        )
        if subscribe is None:
            return

        topics = [
            ("hw.degradation", self._on_hardware_event),
            ("hw.comm_loss", self._on_hardware_event),
            ("hw.sensor_loss", self._on_hardware_event),
            ("control.telemetry", self._on_control_event),
            ("control.hierarchy.escalation", self._on_control_event),
        ]
        for topic, handler in topics:
            try:
                sub = subscribe(topic, handler)
                if sub is not None:
                    self._subscriptions.append(sub)
            except Exception:  # noqa: BLE001 — subscription failure is non-fatal
                logger.debug(
                    "HealthMonitor: failed to subscribe to %s", topic, exc_info=True
                )

    def _unsubscribe_events(self) -> None:
        """Clean up EventBus subscriptions."""
        if self._event_bus is None:
            return

        unsubscribe = getattr(self._event_bus, "off", None) or getattr(
            self._event_bus, "unsubscribe", None
        )
        for sub in self._subscriptions:
            if unsubscribe is not None:
                try:
                    unsubscribe(sub)
                except Exception:  # noqa: BLE001
                    pass
        self._subscriptions.clear()

    # -- Poll loop ----------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Periodic poll: probe all devices, check fleet nodes, assess health.

        Best-effort: a failed probe for one device does not prevent checking
        the rest.  The loop runs until ``stop()`` is called.
        """
        while not self._stopped.is_set():
            for device_id in list(self._records):
                try:
                    await self._probe_device(device_id)
                    await self._evaluate_health(device_id)
                except Exception:  # noqa: BLE001 — poll must not crash
                    logger.debug(
                        "HealthMonitor: poll failed for %s",
                        device_id, exc_info=True,
                    )

            # Check fleet node health if a fleet manager is available.
            if self._fleet_manager is not None:
                await self._poll_fleet_nodes()

            try:
                await asyncio.wait_for(
                    self._stopped.wait(), timeout=self._poll_interval
                )
                break  # stopped
            except asyncio.TimeoutError:
                pass

    async def _probe_device(self, device_id: str) -> None:
        """Probe a device for liveness via the registry."""
        record = self._records.get(device_id)
        if record is None:
            return

        try:
            transport = await self._registry.transport(device_id)
            probe_result = await transport.probe()
            record.last_probe_ok = bool(
                probe_result.ok if hasattr(probe_result, "ok") else probe_result
            )
            if record.last_probe_ok:
                record.consecutive_probe_failures = 0
            else:
                record.consecutive_probe_failures += 1
        except Exception:  # noqa: BLE001 — probe failure is a data point, not an error
            record.last_probe_ok = False
            record.consecutive_probe_failures += 1

        record.last_probe_time = time.monotonic()

    async def _poll_fleet_nodes(self) -> None:
        """Check fleet node health and update device records accordingly.

        Devices on offline fleet nodes are marked degraded.
        """
        if self._fleet_manager is None:
            return

        try:
            topo = self._fleet_manager.topology()
        except Exception:  # noqa: BLE001
            return

        for node in topo.nodes:
            if node.status == "offline":
                for did in node.device_ids:
                    record = self._records.get(did)
                    if record is not None and record.status != _HEALTH_OFFLINE:
                        record.status = _HEALTH_CRITICAL
                        record.degradation_reason = f"fleet node {node.node_id} offline"

    # -- Internal helpers ---------------------------------------------------

    def _seed_records(self) -> None:
        """Initialize health records from the registry's known devices."""
        try:
            for context in self._registry.contexts():
                did = context.device_id
                if did not in self._records:
                    self._records[did] = DeviceHealthRecord(
                        device_id=did,
                        last_probe_time=time.monotonic(),
                    )
        except Exception:  # noqa: BLE001
            logger.debug("HealthMonitor: failed to seed records", exc_info=True)

    def _get_device_context(self, device_id: str) -> Any:
        """Look up a device's HardwareContext from the registry."""
        try:
            return self._registry.context(device_id)
        except Exception:  # noqa: BLE001
            return None

    def _emit_event(self, topic: str, payload: dict[str, Any]) -> None:
        """Best-effort event emission to EventBus."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(f"hw.health.{topic}", payload)
        except Exception:  # noqa: BLE001 — telemetry must not break monitoring
            pass


__all__ = [
    "DeviceHealthRecord",
    "HardwareHealthMonitor",
]
