# Copyright (c) Alibaba, Inc. and its affiliates.
"""LHP Gateway: hardware state as LLM context (the uplink).

Raw device state -> structured context snapshots at three PCD levels. The
LLM never sees raw sensor readings; it sees a curated, resolution-appropriate
summary that fits its context budget.

There is no downlink here: the LLM acts through governed tools, and anything
richer than a single write runs as a node-local skill (``hw_skill_run``), which
plans, gates and executes next to the device.

PCD integration: the uplink respects Progressive Context Disclosure --
  a simple positioning task gets Level 0 (device list + basic state),
  while a complex manipulation gets Level 2 (full kinematics + scene
  graph + force readings).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token estimation constant
# ---------------------------------------------------------------------------
_CHARS_PER_TOKEN = 4

# Snapshot freshness threshold for synchronous prompt_context().
_SNAPSHOT_FRESHNESS_S = 5.0

# Complexity keywords used by recommend_level() heuristics.
_COMPLEX_KEYWORDS = frozenset(
    {
        "assemble",
        "assembly",
        "insert",
        "thread",
        "screw",
        "solder",
        "calibrate",
        "align",
        "complex",
        "multi-step",
        "coordinated",
        "bimanual",
        "dual-arm",
        "force-controlled",
        "compliant",
    }
)
_SIMPLE_KEYWORDS = frozenset(
    {
        "move",
        "go",
        "position",
        "home",
        "park",
        "stop",
        "halt",
        "wait",
        "status",
        "check",
        "read",
        "observe",
        "look",
        "scan",
    }
)


# ===================================================================
# PCD Level
# ===================================================================


class PCDLevel(str, Enum):
    """Progressive Context Disclosure level for hardware context."""

    MINIMAL = "minimal"  # Level 0: device names, connection status
    TASK_RELEVANT = "task_relevant"  # Level 1: + joint positions, gripper, health
    RICH_CONTEXT = "rich_context"  # Level 2: + kinematics, safety, full readings, frames


# ===================================================================
# Uplink IR: Hardware -> LLM
# ===================================================================


@dataclass(frozen=True)
class DeviceSnapshot:
    """One device's state at a specific PCD level.

    Fields are progressively populated according to the requested level:
    - Level 0 (minimal): device_id, device_class, display_name, status, connected, halt_supported
    - Level 1 (task_relevant): + joint_positions, gripper_state, health
    - Level 2 (rich_context): + kinematics, safety_limits, full_readings, trust_level, degradation_status
    """

    device_id: str
    device_class: str
    display_name: str
    status: str  # "connected", "disconnected", "degraded"

    # Level 0 (always present)
    connected: bool = True
    halt_supported: bool = True

    # Level 1 (task_relevant+)
    joint_positions: Mapping[str, float] = field(default_factory=dict)
    gripper_state: float | None = None
    health: str = ""

    # Level 2 (rich_context)
    kinematics: Mapping[str, Any] = field(default_factory=dict)
    safety_limits: Mapping[str, Any] = field(default_factory=dict)
    full_readings: Mapping[str, Any] = field(default_factory=dict)
    trust_level: str = ""
    degradation_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        result: dict[str, Any] = {
            "device_id": self.device_id,
            "device_class": self.device_class,
            "display_name": self.display_name,
            "status": self.status,
            "connected": self.connected,
            "halt_supported": self.halt_supported,
        }
        if self.joint_positions:
            result["joint_positions"] = dict(self.joint_positions)
        if self.gripper_state is not None:
            result["gripper_state"] = self.gripper_state
        if self.health:
            result["health"] = self.health
        if self.kinematics:
            result["kinematics"] = dict(self.kinematics)
        if self.safety_limits:
            result["safety_limits"] = dict(self.safety_limits)
        if self.full_readings:
            result["full_readings"] = dict(self.full_readings)
        if self.trust_level:
            result["trust_level"] = self.trust_level
        if self.degradation_status:
            result["degradation_status"] = self.degradation_status
        return result


@dataclass(frozen=True)
class HardwareContextSnapshot:
    """Structured hardware context for LLM consumption.

    Not a raw dump -- a curated summary at the requested PCD level.
    Designed to be injected into the system prompt or a tool result
    without overwhelming the context window.
    """

    level: str  # PCDLevel value
    devices: tuple[DeviceSnapshot, ...]
    timestamp: float
    token_estimate: int  # rough estimate of tokens this will consume

    def to_prompt_text(self) -> str:
        """Render as LLM-readable text for prompt injection.

        Structured Markdown format designed for LLM comprehension.
        """
        lines: list[str] = []
        lines.append("## Hardware Context")
        lines.append("")
        level_label = {
            PCDLevel.MINIMAL.value: "Minimal (L0)",
            PCDLevel.TASK_RELEVANT.value: "Task-Relevant (L1)",
            PCDLevel.RICH_CONTEXT.value: "Rich (L2)",
        }.get(self.level, self.level)
        lines.append(f"**Detail level**: {level_label}  ")
        lines.append(f"**Devices**: {len(self.devices)}  ")

        lines.append("")

        for dev in self.devices:
            lines.append(f"### {dev.display_name or dev.device_id}")
            lines.append(f"- **Class**: {dev.device_class or 'unknown'}")
            lines.append(f"- **Status**: {dev.status}")
            if not dev.connected:
                lines.append("- **⚠ DISCONNECTED**")

            # Level 1+ details
            if dev.joint_positions:
                pos_str = ", ".join(f"{k}={v:.3f}" for k, v in dev.joint_positions.items())
                lines.append(f"- **Joints**: {pos_str}")
            if dev.gripper_state is not None:
                lines.append(f"- **Gripper**: {dev.gripper_state:.2f}")
            if dev.health:
                lines.append(f"- **Health**: {dev.health}")

            # Level 2 details
            if dev.kinematics:
                dof = dev.kinematics.get("dof", "?")
                chain = dev.kinematics.get("chain_type", "?")
                lines.append(f"- **Kinematics**: {dof}-DOF {chain}")
            if dev.safety_limits:
                lines.append(f"- **Safety**: {_compact_safety(dev.safety_limits)}")
            if dev.trust_level:
                lines.append(f"- **Trust**: {dev.trust_level}")
            if dev.degradation_status:
                lines.append(f"- **Degradation**: {dev.degradation_status}")
            if dev.full_readings:
                lines.append("- **Readings**:")
                for ch_id, reading in dev.full_readings.items():
                    if isinstance(reading, dict):
                        val = reading.get("latest", reading.get("value", "?"))
                        unit = reading.get("unit", "")
                        quality = reading.get("quality", "")
                        q_tag = f" [{quality}]" if quality and quality != "ok" else ""
                        lines.append(f"  - {ch_id}: {val} {unit}{q_tag}")
                    else:
                        lines.append(f"  - {ch_id}: {reading}")

            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "level": self.level,
            "devices": [d.to_dict() for d in self.devices],
            "timestamp": self.timestamp,
            "token_estimate": self.token_estimate,
        }


@dataclass(frozen=True)
class HardwareContextDelta:
    """Changes since the last snapshot, for incremental context updates."""

    previous_timestamp: float
    current_timestamp: float
    devices_added: tuple[str, ...] = ()
    devices_removed: tuple[str, ...] = ()
    state_changes: tuple[Mapping[str, Any], ...] = ()  # per-device diffs
    is_structural: bool = False  # True if devices were added or removed

    def to_prompt_text(self) -> str:
        """Render delta as concise LLM-readable text."""
        lines: list[str] = []
        elapsed = self.current_timestamp - self.previous_timestamp
        lines.append(f"## Hardware Delta (Δ{elapsed:.1f}s)")
        lines.append("")

        if self.devices_added:
            lines.append(f"**Added**: {', '.join(self.devices_added)}")
        if self.devices_removed:
            lines.append(f"**Removed**: {', '.join(self.devices_removed)}")
        if self.is_structural:
            lines.append("**⚠ Structural change** (devices added or removed)")

        for change in self.state_changes:
            dev_id = change.get("device_id", "?")
            lines.append(
                f"- **{dev_id}**: ",
            )
            for key, val in change.items():
                if key != "device_id":
                    lines.append(f"  - {key}: {val}")

        if not (self.devices_added or self.devices_removed or self.state_changes):
            lines.append("No changes detected.")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "previous_timestamp": self.previous_timestamp,
            "current_timestamp": self.current_timestamp,
            "devices_added": list(self.devices_added),
            "devices_removed": list(self.devices_removed),
            "state_changes": [dict(c) for c in self.state_changes],
            "is_structural": self.is_structural,
        }


# ===================================================================
# LHPGateway
# ===================================================================


class LHPGateway:
    """Hardware state -> LLM context at the requested PCD level.

    Dependencies:
    - registry: HardwareRegistry
    - environment_source: PhysicalEnvironmentSource (optional, for snapshot reuse)
    """

    def __init__(self, registry: Any, *, environment_source: Any = None) -> None:
        self._registry = registry
        self._environment_source = environment_source
        self._last_snapshot: HardwareContextSnapshot | None = None

    # ------------------------------------------------------------------
    # Uplink: Hardware -> LLM
    # ------------------------------------------------------------------

    async def snapshot(self, level: str = PCDLevel.TASK_RELEVANT.value) -> HardwareContextSnapshot:
        """Build a hardware context snapshot at the requested PCD level.

        Level 0 (minimal):       ~50 tokens per device
        Level 1 (task_relevant): ~150 tokens per device
        Level 2 (rich_context):  ~400 tokens per device
        """
        # Reuse PhysicalEnvironmentSource data when available to avoid
        # a duplicate registry scan.
        env_data: dict[str, Any] | None = None
        if self._environment_source is not None:
            try:
                env_data = await self._environment_source.take_snapshot()
            except Exception:  # noqa: BLE001 – fallback to direct scan
                logger.debug("Environment source snapshot failed, using direct scan", exc_info=True)

        device_snapshots: list[DeviceSnapshot] = []
        for context in self._registry.contexts():
            env_device = _find_env_device(env_data, context.device_id) if env_data else None
            ds = await self._build_device_snapshot(context, level, env_device=env_device)
            device_snapshots.append(ds)

        snap = HardwareContextSnapshot(
            level=level,
            devices=tuple(device_snapshots),
            timestamp=time.time(),
            token_estimate=0,
        )
        # Replace with computed token estimate (frozen, so rebuild).
        token_est = self._estimate_tokens(snap)
        snap = HardwareContextSnapshot(
            level=snap.level,
            devices=snap.devices,
            timestamp=snap.timestamp,
            token_estimate=token_est,
        )
        self._last_snapshot = snap
        return snap

    async def delta(self) -> HardwareContextDelta | None:
        """Compute delta since last snapshot. None if nothing changed."""
        if self._last_snapshot is None:
            return None

        previous = self._last_snapshot
        current = await self.snapshot(previous.level)

        prev_ids = {d.device_id for d in previous.devices}
        curr_ids = {d.device_id for d in current.devices}

        added = tuple(sorted(curr_ids - prev_ids))
        removed = tuple(sorted(prev_ids - curr_ids))

        # Compute per-device state changes for devices present in both.
        prev_map = {d.device_id: d for d in previous.devices}
        curr_map = {d.device_id: d for d in current.devices}
        changes: list[dict[str, Any]] = []

        for did in sorted(prev_ids & curr_ids):
            pd, cd = prev_map[did], curr_map[did]
            diff: dict[str, Any] = {"device_id": did}
            changed = False
            if pd.status != cd.status:
                diff["status"] = f"{pd.status} → {cd.status}"
                changed = True
            if pd.health != cd.health and cd.health:
                diff["health"] = f"{pd.health} → {cd.health}"
                changed = True
            if pd.connected != cd.connected:
                diff["connected"] = cd.connected
                changed = True
            if changed:
                changes.append(diff)

        is_structural = bool(added or removed)

        if not (added or removed or changes):
            return None

        return HardwareContextDelta(
            previous_timestamp=previous.timestamp,
            current_timestamp=current.timestamp,
            devices_added=added,
            devices_removed=removed,
            state_changes=tuple(changes),
            is_structural=is_structural,
        )

    def prompt_context(self, level: str = PCDLevel.TASK_RELEVANT.value) -> str:
        """Synchronous: return the latest snapshot as prompt-injectable text.

        Called by PromptAssembler during prompt construction.
        Uses cached snapshot if fresh enough (< 5s old).
        """
        snap = self._last_snapshot
        if snap is not None and snap.level == level:
            age = time.time() - snap.timestamp
            if age < _SNAPSHOT_FRESHNESS_S:
                return snap.to_prompt_text()

        # No fresh snapshot available; return a minimal fallback from cache
        # or an empty string.  The caller should have called snapshot() first
        # in the async setup path.
        if snap is not None:
            return snap.to_prompt_text()
        return ""

    # ------------------------------------------------------------------
    # PCD Level Selection
    # ------------------------------------------------------------------

    def recommend_level(self, task_description: str = "") -> str:
        """Suggest PCD level based on task complexity.

        Simple tasks ("move to position") -> Level 0
        Standard manipulation ("pick up the cup") -> Level 1
        Complex tasks ("assemble the parts") -> Level 2

        Uses keyword heuristics, not LLM inference.
        """
        if not task_description:
            return PCDLevel.TASK_RELEVANT.value

        lower = task_description.lower()
        words = set(lower.split())

        # Check complex keywords first (higher priority).
        if words & _COMPLEX_KEYWORDS or any(kw in lower for kw in _COMPLEX_KEYWORDS):
            return PCDLevel.RICH_CONTEXT.value

        # Check simple keywords.
        if words & _SIMPLE_KEYWORDS and not (words & _COMPLEX_KEYWORDS):
            return PCDLevel.MINIMAL.value

        # Default for standard manipulation tasks.
        return PCDLevel.TASK_RELEVANT.value

    # ------------------------------------------------------------------
    # Internal: device snapshot building
    # ------------------------------------------------------------------

    async def _build_device_snapshot(
        self,
        context: Any,
        level: str,
        *,
        env_device: dict[str, Any] | None = None,
    ) -> DeviceSnapshot:
        """Build one device's snapshot at the requested level.

        Reads from the HardwareContext declaration and optionally merges
        live data from the PhysicalEnvironmentSource snapshot.
        """
        device_id = context.device_id
        device_class = getattr(context, "device_class", "") or ""
        display_name = getattr(context, "display_name", "") or device_id

        # Determine connection status.
        connected = True
        health_str = ""
        if env_device is not None:
            connected = bool(env_device.get("connected", True))
            health_str = str(env_device.get("health", "ok"))

        if not connected:
            status = "disconnected"
        elif health_str in ("degraded", "stale", "unreachable"):
            status = "degraded"
        else:
            status = "connected"

        halt_supported = bool(getattr(context, "halt_supported", False))

        # Level 0: basic info only.
        if level == PCDLevel.MINIMAL.value:
            return DeviceSnapshot(
                device_id=device_id,
                device_class=device_class,
                display_name=display_name,
                status=status,
                connected=connected,
                halt_supported=halt_supported,
            )

        # Level 1+: joint positions, gripper state, health.
        # Collect joint positions from channel summaries.
        joint_positions: dict[str, float] = {}
        gripper_state: float | None = None
        if env_device is not None:
            raw_channels = env_device.get("channels") or {}
            if isinstance(raw_channels, dict):
                for ch_id, ch_data in raw_channels.items():
                    if isinstance(ch_data, dict) and ch_data.get("value") is not None:
                        try:
                            val = float(ch_data["value"])
                        except (TypeError, ValueError):
                            continue
                        if "gripper" in ch_id.lower():
                            gripper_state = val
                        else:
                            joint_positions[ch_id] = val
        else:
            # Try reading from registry's channel summaries.
            for ch in context.channels:
                if not ch.is_readable or ch.is_media:
                    continue
                summary = self._channel_summary(device_id, ch.channel_id)
                if summary and summary.get("samples", 0) > 0:
                    val = summary.get("latest")
                    if val is not None:
                        try:
                            fval = float(val)
                        except (TypeError, ValueError):
                            continue
                        if "gripper" in ch.channel_id.lower():
                            gripper_state = fval
                        else:
                            joint_positions[ch.channel_id] = fval

        if level == PCDLevel.TASK_RELEVANT.value:
            return DeviceSnapshot(
                device_id=device_id,
                device_class=device_class,
                display_name=display_name,
                status=status,
                connected=connected,
                halt_supported=halt_supported,
                joint_positions=joint_positions,
                gripper_state=gripper_state,
                health=health_str or "ok",
            )

        # Level 2: full kinematics, safety, readings, trust, degradation.
        kin_dict: Mapping[str, Any] = {}
        kin = getattr(context, "kinematics", None)
        if kin is not None:
            kin_dict = (
                kin.to_dict()
                if hasattr(kin, "to_dict")
                else {
                    "dof": getattr(kin, "dof", 0),
                    "chain_type": getattr(kin, "chain_type", ""),
                }
            )

        safety_dict: Mapping[str, Any] = {}
        safety = getattr(context, "safety", None)
        if safety is not None:
            safety_dict = safety.to_dict() if hasattr(safety, "to_dict") else {}

        # Full channel readings.
        full_readings: dict[str, Any] = {}
        for ch in context.channels:
            if not ch.is_readable or ch.is_media:
                continue
            summary = self._channel_summary(device_id, ch.channel_id)
            if summary and summary.get("samples", 0) > 0:
                full_readings[ch.channel_id] = summary
            elif env_device is not None:
                raw_channels = env_device.get("channels") or {}
                if isinstance(raw_channels, dict) and ch.channel_id in raw_channels:
                    full_readings[ch.channel_id] = raw_channels[ch.channel_id]

        trust_level = ""
        trust_cfg = getattr(context, "trust_config", None)
        if trust_cfg is not None:
            trust_level = getattr(trust_cfg, "initial_level", "") or ""

        degradation_status = ""
        degrad = getattr(context, "degradation", None)
        if degrad is not None:
            degradation_status = getattr(degrad, "comm_loss_policy", "") or ""

        return DeviceSnapshot(
            device_id=device_id,
            device_class=device_class,
            display_name=display_name,
            status=status,
            connected=connected,
            halt_supported=halt_supported,
            joint_positions=joint_positions,
            gripper_state=gripper_state,
            health=health_str or "ok",
            kinematics=kin_dict,
            safety_limits=safety_dict,
            full_readings=full_readings,
            trust_level=trust_level,
            degradation_status=degradation_status,
        )

    def _channel_summary(self, device_id: str, channel_id: str) -> dict[str, Any] | None:
        """Get channel reading summary from the registry's stream sources."""
        try:
            return self._registry.channel_summary(device_id, channel_id)
        except Exception:  # noqa: BLE001 – observation must not fail
            return None

    def _estimate_tokens(self, snapshot: HardwareContextSnapshot) -> int:
        """Rough token count estimate for context budget planning.

        Uses ~4 characters per token as a conservative estimate.
        """
        text = snapshot.to_prompt_text()
        return max(1, len(text) // _CHARS_PER_TOKEN)


# ===================================================================
# Module-private helpers
# ===================================================================


def _find_env_device(env_data: dict[str, Any] | None, device_id: str) -> dict[str, Any] | None:
    """Find a device dict inside a PhysicalEnvironmentSource snapshot."""
    if env_data is None:
        return None
    for dev in env_data.get("devices", ()):
        if isinstance(dev, dict) and dev.get("device_id") == device_id:
            return dev
    return None


def _compact_safety(limits: Mapping[str, Any]) -> str:
    """Render safety limits as a compact one-line string."""
    parts: list[str] = []
    if limits.get("max_velocity_rad_s"):
        parts.append(f"vel≤{limits['max_velocity_rad_s']}rad/s")
    if limits.get("max_force_n"):
        parts.append(f"force≤{limits['max_force_n']}N")
    if limits.get("emergency_decel_s"):
        parts.append(f"e-stop={limits['emergency_decel_s']}s")
    return ", ".join(parts) if parts else "default"


__all__ = [
    "DeviceSnapshot",
    "HardwareContextDelta",
    "HardwareContextSnapshot",
    "LHPGateway",
    "PCDLevel",
]
