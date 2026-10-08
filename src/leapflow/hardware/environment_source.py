# Copyright (c) Alibaba, Inc. and its affiliates.
"""Physical environment source: observes the robot workspace through HCP.

Produces ``EnvironmentObservation`` events that feed the evolution engine's
capability gap detector.  Three observation types:

- **snapshot**: complete workspace state — all devices, their health,
  capabilities, current sensor readings summary, and camera frame metadata.
  Emitted periodically (configurable interval, default 30s) and on demand.

- **delta**: structural changes since last snapshot — devices that appeared
  or disappeared, affordances gained or lost, channels that changed direction
  or effect, health transitions (OK→STALE, connected→disconnected).

- **outcome**: operation result feedback — sourced from EvidenceStore's
  recent verdicts, aggregated as success/failure rates per device/affordance.

Unlike ``LeapSpaceEnvironmentSource`` which polls an external agent runner,
this source observes LeapFlow's own hardware stack directly, making it
zero-latency and always-available when hardware is enabled.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_SOURCE_ID = "physical_hardware"
_DEFAULT_SNAPSHOT_INTERVAL_S = 30.0
_DEFAULT_POLL_INTERVAL_S = 5.0
_OUTCOME_WINDOW_S = 300.0  # 5-minute lookback for operation verdicts


class PhysicalEnvironmentSource:
    """EnvironmentSource that observes the physical robot workspace.

    Satisfies the EnvironmentSource Protocol:
    - source_id: "physical_hardware"
    - start(emit): begin periodic observation loop
    - stop(): cancel observation loop

    Dependencies (injected via constructor):
    - registry: HardwareRegistry — device enumeration and health
    - evidence_store: EvidenceStore (optional) — operation success rates
    - capability_index: CapabilityIndex (optional) — affordance inventory
    """

    def __init__(
        self,
        registry: Any,
        *,
        evidence_store: Any = None,
        capability_index: Any = None,
        snapshot_interval_s: float = _DEFAULT_SNAPSHOT_INTERVAL_S,
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        self._registry = registry
        self._evidence_store = evidence_store
        self._capability_index = capability_index
        self._snapshot_interval_s = max(1.0, float(snapshot_interval_s))
        self._poll_interval_s = max(0.1, float(poll_interval_s))
        self._task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()
        self._last_snapshot: dict[str, Any] | None = None

    # ── Protocol surface ──

    @property
    def source_id(self) -> str:
        return _SOURCE_ID

    async def start(self, emit: Any) -> None:
        """Begin periodic observation. Returns promptly; loop runs as internal task."""
        if self._task is not None:
            return
        self._stopped.clear()
        self._task = asyncio.create_task(
            self._run(emit), name="physical-environment-source"
        )

    async def stop(self) -> None:
        """Stop observation. Idempotent."""
        self._stopped.set()
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=3.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception:  # noqa: BLE001 – teardown must not propagate
            logger.warning("Physical environment source stop raised", exc_info=True)

    # ── Core observation loop ──

    async def _run(self, emit: Any) -> None:
        """Core observation loop.

        Every poll_interval_s:
        1. Scan registry for device changes → emit delta if changed
        2. Check stream health for degradation events

        Every snapshot_interval_s:
        3. Build full workspace snapshot → emit snapshot
        4. Query evidence_store for recent verdicts → emit outcome
        """
        # Function-local imports: hardware must not create a top-level dependency
        # on domain types so the import graph stays clean.
        from leapflow.domain.environment_signal import EnvironmentObservation

        start_mono = time.monotonic()
        next_poll_at = start_mono
        next_snapshot_at = start_mono  # first cycle always takes a full snapshot

        while not self._stopped.is_set():
            now_mono = time.monotonic()

            try:
                current = await self.take_snapshot()
            except Exception:  # noqa: BLE001 – one failed scan must not kill the loop
                logger.warning("Physical environment snapshot failed", exc_info=True)
                await self._sleep(self._poll_interval_s)
                next_poll_at = time.monotonic() + self._poll_interval_s
                continue

            wall = current.get("timestamp", time.time())

            if now_mono >= next_snapshot_at:
                # ── Full snapshot cycle ──
                iface = self._to_interface_snapshot(current, wall)
                await self._emit_safe(emit, EnvironmentObservation.snapshot(iface))
                await self._emit_outcomes(emit, wall)
                next_snapshot_at = now_mono + self._snapshot_interval_s
            else:
                # ── Delta-only cycle ──
                if self._last_snapshot is not None:
                    delta_dict = await self.compute_delta(self._last_snapshot, current)
                    if delta_dict is not None:
                        prev_wall = self._last_snapshot.get("timestamp", wall)
                        prev_iface = self._to_interface_snapshot(
                            self._last_snapshot, prev_wall
                        )
                        curr_iface = self._to_interface_snapshot(current, wall)
                        delta_obs = EnvironmentObservation.between(prev_iface, curr_iface)
                        if delta_obs is not None:
                            await self._emit_safe(emit, delta_obs)

            self._last_snapshot = current

            # Deadline-based scheduling (matches HardwareStreamSource pattern).
            next_poll_at += self._poll_interval_s
            delay = next_poll_at - time.monotonic()
            if delay < 0:
                missed = int(-delay // self._poll_interval_s) + 1
                next_poll_at += missed * self._poll_interval_s
                delay = max(0.0, next_poll_at - time.monotonic())
            await self._sleep(delay)

    # ── Public observation builders ──

    async def take_snapshot(self) -> dict[str, Any]:
        """Build a complete workspace snapshot (also used on demand).

        Returns a dict suitable for EnvironmentObservation(kind="snapshot"):
        {
            "source": "physical_hardware",
            "devices": [
                {
                    "device_id": "robot.arm",
                    "device_class": "robot_arm",
                    "connected": true,
                    "health": "ok",
                    "affordances": ["grasp", "place", "push"],
                    "channels": {"joint.shoulder.position": {"value": 0.5, ...}},
                    "kinematics": {"dof": 6, "chain_type": "serial"},
                },
                ...
            ],
            "capability_summary": {
                "total_affordances": 4,
                "available_affordances": ["grasp", "place", "push", "pour"],
                "device_count": 2,
            },
            "timestamp": 1695000000.0,
        }
        """
        devices: list[dict[str, Any]] = []
        all_affordances: set[str] = set()

        for context in self._registry.contexts():
            device_info = self._device_to_dict(context)
            devices.append(device_info)
            all_affordances.update(device_info.get("affordances", ()))

        # Merge capability_index affordances when available.
        if self._capability_index is not None:
            try:
                all_affordances.update(self._capability_index.all_affordances())
            except Exception:  # noqa: BLE001 – observation must not fail
                pass

        return {
            "source": _SOURCE_ID,
            "devices": devices,
            "capability_summary": {
                "total_affordances": len(all_affordances),
                "available_affordances": sorted(all_affordances),
                "device_count": len(devices),
            },
            "timestamp": time.time(),
        }

    async def compute_delta(
        self, previous: dict[str, Any], current: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Compare two snapshots and return structural delta, or None if unchanged.

        Detects:
        - device_added / device_removed
        - affordance_gained / affordance_lost
        - health_changed (ok→stale, connected→disconnected)
        - channel_changed (direction, effect, or representation changed)

        Returns None when nothing structurally changed (value changes are
        not structural — they're handled by the streaming/event system).
        """
        prev_devs = {d["device_id"]: d for d in previous.get("devices", ())}
        curr_devs = {d["device_id"]: d for d in current.get("devices", ())}

        prev_fps = {did: self._fingerprint_from_dict(d) for did, d in prev_devs.items()}
        curr_fps = {did: self._fingerprint_from_dict(d) for did, d in curr_devs.items()}

        added = sorted(set(curr_fps) - set(prev_fps))
        removed = sorted(set(prev_fps) - set(curr_fps))

        health_changed: list[dict[str, str]] = []
        affordance_gained: list[str] = []
        affordance_lost: list[str] = []
        channel_changed: list[dict[str, Any]] = []

        for did in sorted(set(prev_fps) & set(curr_fps)):
            pf, cf = prev_fps[did], curr_fps[did]

            # Health transitions.
            if pf.get("health") != cf.get("health"):
                health_changed.append(
                    {"device_id": did, "before": pf["health"], "after": cf["health"]}
                )
            if pf.get("connected") != cf.get("connected"):
                health_changed.append(
                    {
                        "device_id": did,
                        "before": "connected" if pf["connected"] else "disconnected",
                        "after": "connected" if cf["connected"] else "disconnected",
                    }
                )

            # Affordance changes.
            prev_aff = set(pf.get("affordances", ()))
            curr_aff = set(cf.get("affordances", ()))
            for a in sorted(curr_aff - prev_aff):
                affordance_gained.append(a)
            for a in sorted(prev_aff - curr_aff):
                affordance_lost.append(a)

            # Channel structural changes (added/removed channels).
            prev_ch = set(pf.get("channels", ()))
            curr_ch = set(cf.get("channels", ()))
            if prev_ch != curr_ch:
                channel_changed.append(
                    {
                        "device_id": did,
                        "added": sorted(curr_ch - prev_ch),
                        "removed": sorted(prev_ch - curr_ch),
                    }
                )

        if not any(
            (added, removed, health_changed, affordance_gained,
             affordance_lost, channel_changed)
        ):
            return None

        return {
            "source": _SOURCE_ID,
            "device_added": added,
            "device_removed": removed,
            "health_changed": health_changed,
            "affordance_gained": affordance_gained,
            "affordance_lost": affordance_lost,
            "channel_changed": channel_changed,
            "timestamp": time.time(),
        }

    async def collect_outcomes(self) -> dict[str, Any] | None:
        """Query recent operation verdicts from EvidenceStore.

        Returns None if no evidence_store or no recent operations.

        Returns:
        {
            "source": "physical_hardware",
            "period_s": 300,
            "devices": {
                "robot.arm": {
                    "total_operations": 15,
                    "success_rate": 0.87,
                    "recent_failures": ["position_deviation > tolerance on joint.elbow"],
                },
            },
        }
        """
        if self._evidence_store is None:
            return None

        device_outcomes: dict[str, dict[str, Any]] = {}
        window_days = _OUTCOME_WINDOW_S / 86400.0

        for context in self._registry.contexts():
            did = context.device_id
            try:
                rate = await self._evidence_store.success_rate(
                    did, window_days=window_days
                )
            except Exception:  # noqa: BLE001 – a query failure is diagnostic, not fatal
                logger.debug("Evidence query failed for %s", did, exc_info=True)
                continue

            total = rate.get("total", 0)
            if total == 0:
                continue

            # Best-effort query for recent failure details.
            failures: list[str] = []
            try:
                records = await self._evidence_store.query(
                    device_id=did,
                    verdict_status="failure",
                    since=time.time() - _OUTCOME_WINDOW_S,
                    limit=5,
                )
                for rec in records:
                    detail = rec.get("verdict_detail") or ""
                    if detail:
                        failures.append(detail)
            except Exception:  # noqa: BLE001
                pass

            device_outcomes[did] = {
                "total_operations": total,
                "success_rate": rate.get("success_rate", 0.0),
                "recent_failures": failures,
            }

        if not device_outcomes:
            return None

        return {
            "source": _SOURCE_ID,
            "period_s": _OUTCOME_WINDOW_S,
            "devices": device_outcomes,
        }

    def _device_fingerprint(self, context: Any) -> dict[str, Any]:
        """Extract structural fingerprint of a device for delta comparison.

        Only structural attributes (not values): device_id, channels list,
        affordances, kinematics, transport kind, halt_supported.
        """
        channels: list[str] = [ch.channel_id for ch in context.channels]

        affordances: list[str] = []
        cap = getattr(context, "capabilities", None)
        if cap is not None:
            affordances = list(getattr(cap, "affordances", ()) or ())

        kin = getattr(context, "kinematics", None)
        kin_key = ""
        if kin is not None:
            kin_key = f"{getattr(kin, 'chain_type', '')}:{getattr(kin, 'dof', 0)}"

        return {
            "device_id": context.device_id,
            "device_class": getattr(context, "device_class", ""),
            "channels": sorted(channels),
            "affordances": sorted(affordances),
            "kinematics": kin_key,
            "transport_kind": getattr(context.transport, "kind", ""),
            "halt_supported": bool(context.halt_supported),
        }

    # ── Internal helpers ──

    def _device_to_dict(self, context: Any) -> dict[str, Any]:
        """Build a device dict for the snapshot payload."""
        channels: dict[str, dict[str, Any]] = {}
        for ch in context.channels:
            summary = self._channel_summary(context.device_id, ch.channel_id)
            entry: dict[str, Any] = {
                "direction": ch.direction,
                "effect": ch.effect,
                "unit": ch.unit,
                "quality": "unknown",
            }
            if summary and summary.get("samples", 0) > 0:
                entry["value"] = summary.get("latest")
                entry["quality"] = summary.get("quality", "ok")
            channels[ch.channel_id] = entry

        affordances: list[str] = []
        cap = getattr(context, "capabilities", None)
        if cap is not None:
            affordances = list(getattr(cap, "affordances", ()) or ())

        kin_dict: dict[str, Any] | None = None
        kin = getattr(context, "kinematics", None)
        if kin is not None:
            kin_dict = {
                "dof": getattr(kin, "dof", 0),
                "chain_type": getattr(kin, "chain_type", ""),
            }

        health = self._device_health(context.device_id)

        return {
            "device_id": context.device_id,
            "device_class": getattr(context, "device_class", ""),
            "connected": True,  # admitted by the registry ⇒ structurally present
            "health": health,
            "affordances": sorted(affordances),
            "channels": channels,
            "kinematics": kin_dict,
        }

    def _device_health(self, device_id: str) -> str:
        """Derive device health from recent hardware events.

        Returns "ok", "stale", "degraded", or "unreachable".
        """
        try:
            events = self._registry.recent_events(device_id=device_id, limit=5)
        except Exception:  # noqa: BLE001
            return "ok"
        if not events:
            return "ok"
        # Walk newest-first (events are stored oldest-first, so reverse).
        for event in reversed(events):
            kind = getattr(event, "kind", "")
            if kind == "unreachable":
                return "unreachable"
            if kind == "stale":
                return "stale"
            if kind == "quality_degraded":
                return "degraded"
            if kind == "settled":
                return "ok"
        return "ok"

    def _channel_summary(
        self, device_id: str, channel_id: str
    ) -> dict[str, Any] | None:
        """Get channel reading summary from the registry's stream sources."""
        try:
            return self._registry.channel_summary(device_id, channel_id)
        except Exception:  # noqa: BLE001 – never fail an observation read
            return None

    @staticmethod
    def _fingerprint_from_dict(device: dict[str, Any]) -> dict[str, Any]:
        """Extract structural fingerprint from a snapshot device dict."""
        raw_channels = device.get("channels")
        channels = sorted(raw_channels.keys()) if isinstance(raw_channels, dict) else []
        return {
            "device_id": device.get("device_id", ""),
            "device_class": device.get("device_class", ""),
            "channels": channels,
            "affordances": sorted(device.get("affordances", ())),
            "connected": device.get("connected", True),
            "health": device.get("health", "ok"),
        }

    def _to_interface_snapshot(
        self, snapshot: dict[str, Any], observed_at: float
    ) -> Any:
        """Convert a snapshot dict to an InterfaceSnapshot for domain observation.

        Each device becomes an InterfaceElement (name=device_id, role=device_class,
        enabled=connected).  Affordances map directly to the snapshot's affordance
        tuple.  The version encodes the structural composition so that
        ``EnvironmentObservation.between()`` detects changes correctly.
        """
        from leapflow.domain.environment_signal import InterfaceElement, InterfaceSnapshot

        devices = snapshot.get("devices", ())
        elements: list[InterfaceElement] = []
        affordances: set[str] = set()

        for dev in devices:
            elements.append(
                InterfaceElement(
                    name=dev.get("device_id", ""),
                    role=dev.get("device_class", "") or "device",
                    enabled=bool(dev.get("connected", True)),
                )
            )
            affordances.update(dev.get("affordances", ()))

        cap_summary = snapshot.get("capability_summary", {})
        affordances.update(cap_summary.get("available_affordances", ()))

        # Encode the structural shape as a version so the domain delta catches
        # topology changes (device count, total channel count).
        total_channels = sum(
            len(d.get("channels", {})) if isinstance(d.get("channels"), dict) else 0
            for d in devices
        )
        version = f"{len(devices)}d.{total_channels}ch"

        return InterfaceSnapshot.create(
            source_id=_SOURCE_ID,
            app_id="physical_workspace",
            version=version,
            affordances=tuple(sorted(affordances)),
            elements=tuple(elements),
            data=cap_summary,
            observed_at=observed_at,
            provenance={
                "kind": "hardware_registry",
                "device_count": str(len(devices)),
            },
        )

    async def _emit_outcomes(self, emit: Any, wall: float) -> None:
        """Collect and emit per-device outcome observations."""
        from leapflow.domain.environment_signal import EnvironmentObservation

        outcomes = await self.collect_outcomes()
        if outcomes is None:
            return

        for did, stats in outcomes.get("devices", {}).items():
            total = stats.get("total_operations", 0)
            sr = stats.get("success_rate", 0.0)
            if total == 0:
                outcome_str = "UNKNOWN"
            elif sr >= 0.5:
                outcome_str = "PASS"
            else:
                outcome_str = "FAIL"

            obs = EnvironmentObservation.task_outcome(
                source_id=_SOURCE_ID,
                task_id=did,
                outcome=outcome_str,
                capability=f"hardware.{did}",
                observed_at=wall,
                provenance={
                    "kind": "hardware_evidence",
                    "total_operations": str(total),
                    "success_rate": f"{sr:.2f}",
                    "period_s": str(outcomes.get("period_s", _OUTCOME_WINDOW_S)),
                },
            )
            await self._emit_safe(emit, obs)

    @staticmethod
    async def _emit_safe(emit: Any, observation: Any) -> None:
        """Emit an observation, logging but never propagating errors."""
        if emit is None:
            return
        try:
            await emit(observation)
        except Exception:  # noqa: BLE001 – a sink failure must not kill the loop
            logger.warning(
                "Physical environment observation emit failed", exc_info=True
            )

    async def _sleep(self, seconds: float) -> None:
        """Sleep respecting the stop event, matching HardwareStreamSource pattern."""
        try:
            await asyncio.wait_for(self._stopped.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass


__all__ = ["PhysicalEnvironmentSource"]
