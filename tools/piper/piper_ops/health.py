"""Fail-closed qualified actuator snapshots; unavailable never means disabled."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from leapflow.hardware.drivers._piper_feedback import CAN_GROUPS, _FAULT_FLAGS
from leapflow.hardware.registry import HardwareRegistry
from leapflow.hardware.skills.geometry import Pose

from .config import Settings
from .evidence import Refusal

JOINTS = tuple(f"joint.j{i}.position" for i in range(1, 7))
POSE = tuple(f"ee.{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz"))
CHANNELS = ("arm.status", "arm.telemetry", "arm.enabled", "gripper.position", *JOINTS, *POSE)


def number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise Refusal("missing/nonfinite numeric feedback")
    return float(value)


@dataclass(frozen=True)
class Snapshot:
    joints: tuple[float, ...]
    pose: Pose
    gripper_mm: float
    enabled: bool
    status: dict[str, Any]
    telemetry: dict[str, Any]

    @property
    def receipts(self) -> tuple[int, ...]:
        return tuple(self.telemetry["receive"][group]["sequence"] for group in CAN_GROUPS)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def read_snapshot(registry: HardwareRegistry, settings: Settings, *, ready: bool = False,
                        preflight: bool = False) -> Snapshot:
    batch = await registry.read_batch(settings.device_id, CHANNELS)
    readings = {r.channel_id: r for r in batch.readings}
    if len(batch.readings) != len(CHANNELS) or set(readings) != set(CHANNELS) or any(
        r.device_id != settings.device_id or r.quality != "ok" for r in readings.values()
    ):
        raise Refusal("missing, duplicate or unqualified controller channels")
    value = {key: row.value for key, row in readings.items()}
    status, telemetry = value["arm.status"], value["arm.telemetry"]
    if not isinstance(status, dict) or not isinstance(telemetry, dict):
        raise Refusal("controller health payload is malformed")
    for field in ("ctrl_mode", "motion_status"):
        if type(status.get(field)) is not int:
            raise Refusal("controller mode/arrival data are missing")
    allowed = (0,) if ready else (0, 1)
    if type(status.get("arm_status")) is not int or status["arm_status"] not in allowed:
        raise Refusal(f"controller fault; no reset or replay: {status}")
    motors = telemetry.get("motors")
    if telemetry.get("receive_qualified") is not True or not isinstance(motors, list):
        raise Refusal("controller receive qualification is missing")
    receive = telemetry.get("receive")
    if not isinstance(receive, dict) or any(
        not isinstance(receive.get(g), dict) or receive[g].get("valid") is not True or
        type(receive[g].get("sequence")) is not int or receive[g]["sequence"] <= 0 or
        any(isinstance(receive[g].get(k), bool) or
            not isinstance(receive[g].get(k), (int, float)) or
            not math.isfinite(receive[g][k]) for k in ("received_mono", "received_wall"))
        for g in CAN_GROUPS
    ):
        raise Refusal("receive receipts are missing or malformed")
    if len(motors) != 6 or any(not isinstance(m, dict) or type(m.get("motor")) is not int
                              for m in motors) or {
        m.get("motor") for m in motors} != set(range(1, 7)):
        raise Refusal("missing or duplicate motor records")
    flags = []
    for motor in motors:
        enabled, faults = motor.get("enabled"), motor.get("faults")
        if not isinstance(enabled, bool) or not isinstance(faults, dict) or any(
            type(faults.get(key)) is not bool for key in _FAULT_FLAGS
        ):
            raise Refusal("unknown motor enable or fault fields")
        if any(faults[key] for key in _FAULT_FLAGS):
            raise Refusal(f"motor {motor['motor']} reports a fault; no motion")
        voltage = number(motor.get("voltage_v"))
        limit = settings.preflight_temp_c if preflight else settings.max_temp_c
        temperatures = [number(motor.get(k)) for k in ("motor_temperature_c", "driver_temperature_c")]
        if not settings.voltage_min_v <= voltage <= settings.voltage_max_v or max(temperatures) >= limit:
            raise Refusal("motor thermal/electrical safety gate refused")
        flags.append(enabled)
    enabled = value["arm.enabled"]
    if type(enabled) is not bool or enabled != all(flags):
        raise Refusal("controller enable readings disagree")
    if ready and (not enabled or status.get("ctrl_mode") != 1):
        raise Refusal("all-six enable and CAN control are not confirmed")
    return Snapshot(tuple(number(value[c]) for c in JOINTS),
                    Pose(*(number(value[c]) for c in POSE)), number(value["gripper.position"]),
                    enabled, status, telemetry)
