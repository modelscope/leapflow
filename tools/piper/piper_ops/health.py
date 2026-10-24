"""Fail-closed qualified actuator snapshots; unavailable never means disabled."""
from __future__ import annotations

import asyncio
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from typing import Any
from urllib.parse import urlparse

from leapflow.hardware.drivers._piper_feedback import CAN_GROUPS, _FAULT_FLAGS
from leapflow.hardware.providers import ProviderError
from leapflow.hardware.providers.yaml_provider import YamlContextProvider
from leapflow.hardware.registry import HardwareRegistry
from leapflow.hardware.skills.geometry import Pose
from leapflow.hardware.attestation import declaration_digest

from .config import Settings
from .evidence import Refusal, StaleFeedback

JOINTS = tuple(f"joint.j{i}.position" for i in range(1, 7))
POSE = tuple(f"ee.{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz"))
CHANNELS = ("arm.status", "arm.telemetry", "arm.enabled", "gripper.position", *JOINTS, *POSE)
CAN_ERROR_STATS = ("restarts", "bus_error", "error_warning", "error_passive", "bus_off")


def _link_sample(interface: str, timeout: float) -> dict[str, Any]:
    """Kernel link statistics only; never open CAN or send a frame."""
    result = subprocess.run(["ip", "-j", "-s", "-d", "link", "show", "dev", interface],
                            capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode or len(result.stdout) > 65536:
        raise Refusal("passive kernel CAN link inspection unavailable")
    try:
        rows = json.loads(result.stdout)
    except ValueError as error:
        raise Refusal("passive CAN link JSON is malformed") from error
    if (not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict)
            or rows[0].get("ifname") != interface):
        raise Refusal("passive CAN interface identity differs")
    row = rows[0]
    info = row.get("linkinfo")
    data = info.get("info_data") if isinstance(info, dict) else None
    xstats = info.get("info_xstats") if isinstance(info, dict) else None
    berr = data.get("berr_counter") if isinstance(data, dict) else None
    # Some adapters (gs_usb) export no TEC/REC; ERROR-ACTIVE plus the kernel CAN
    # error statistics below still bound the bus, and a present counter must be 0.
    if (not isinstance(data, dict) or not isinstance(data.get("bittiming"), dict)
            or not (berr is None or isinstance(berr, dict)) or not isinstance(xstats, dict)
            or not isinstance(row.get("flags"), list)):
        raise Refusal("passive CAN link metadata is malformed")
    if (info.get("info_kind") != "can" or "UP" not in row.get("flags", [])
            or data.get("state") != "ERROR-ACTIVE"
            or number(data["bittiming"].get("bitrate")) != 1_000_000
            or (berr is not None and any(number(berr.get(key)) != 0 for key in ("rx", "tx")))):
        raise Refusal("passive CAN state, bitrate or bus-error counters are unqualified")
    stats = row.get("stats64", row.get("stats"))
    if (not isinstance(stats, dict) or any(not isinstance(stats.get(d), dict)
                                         for d in ("rx", "tx"))):
        raise Refusal("passive CAN link counters are malformed")
    try:
        counts = {f"{direction}_{key}": number(stats[direction][key])
                  for direction in ("rx", "tx") for key in ("packets", "errors", "dropped")}
        counts.update({f"can_{key}": number(xstats[key]) for key in CAN_ERROR_STATS})
    except KeyError as error:
        raise Refusal("passive CAN link counters are incomplete") from error
    if any(v < 0 or v != int(v) for v in counts.values()):
        raise Refusal("passive CAN counters are malformed")
    return {"interface": interface, "state": data["state"], "bitrate": 1_000_000,
            "counts": counts, "bus_error_counters": None if berr is None else dict(berr)}


def _unverified(context: Any) -> str:
    return declaration_digest(replace(context, provenance=replace(
        context.provenance, verified_by="", verified_at=0.0)))


def _node_can_name(settings: Settings, served: Any) -> Any:
    """The node-private CAN binding of exactly the declaration the live owner serves.

    The wire listing withholds ``transport.config``; the local profile supplies it
    only when its node-public digest equals the served declaration's.
    """
    try:
        local = [c for c in YamlContextProvider({"devices_dir": settings.node_devices}).discover()
                 if c.device_id == served.device_id]
    except (OSError, ProviderError, ValueError, TypeError):
        return None
    if len(local) != 1 or local[0].transport.kind != "piper" or _unverified(local[0]) != _unverified(served):
        return None
    return local[0].transport.config.get("can_name")


async def passive_can(session: Any) -> dict[str, Any]:
    """Read two node-local link samples and qualified decoder snapshots, with no writes.

    A remote endpoint cannot borrow this host's kernel statistics. Run this
    read-only command on the node when kernel evidence is needed.
    """
    settings, context = session.settings, session.context
    if (sys.platform != "linux" or urlparse(settings.endpoint).hostname not in ("127.0.0.1", "::1", "localhost")
            or context is None or context.transport.kind != "piper"):
        return {"qualified": False, "reason": "node-local Linux passive CAN inspection required"}
    interface = _node_can_name(settings, context)
    if not isinstance(interface, str) or re.fullmatch(r"[a-zA-Z0-9_.-]{1,15}", interface) is None:
        return {"qualified": False, "reason": "declared CAN interface is unavailable"}
    try:
        # Bus health needs qualified receipts, not an enabled arm; readiness is its own blocker.
        first = await asyncio.to_thread(_link_sample, interface, settings.request_timeout_s)
        before = await session.snapshot(ready=False)
        await asyncio.sleep(settings.poll_s)
        after = await session.snapshot(ready=False)
        last = await asyncio.to_thread(_link_sample, interface, settings.request_timeout_s)
        changes = {key: last["counts"][key] - value for key, value in first["counts"].items()}
        advancing = all(b > a for a, b in zip(before.receipts, after.receipts))
        qualified = (advancing and changes["rx_packets"] > 0
                     and all(v >= 0 for v in changes.values())
                     and all(value == 0 for key, value in changes.items()
                             if key.endswith(("errors", "dropped")) or key.startswith("can_")))
        return {"qualified": qualified, "before": first, "after": last,
                "counter_changes": changes, "decoder_groups_advanced": advancing,
                "reason": "" if qualified else "passive counters or decoder receipt continuity failed"}
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError, Refusal) as error:
        return {"qualified": False, "reason": f"passive CAN evidence unavailable ({type(error).__name__}: {error})"}


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
        r.device_id != settings.device_id for r in readings.values()
    ):
        raise Refusal("missing, duplicate or foreign controller channels")
    unqualified = sorted(key for key, row in readings.items() if row.quality != "ok")
    if unqualified:
        # Non-ok quality is the receive verdict on this sample, not a device state; name
        # every affected channel so a retry or a halt can be judged from the evidence.
        raise StaleFeedback("unqualified controller channels: " + ", ".join(unqualified))
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
    if not isinstance(motors, list):
        raise Refusal("controller telemetry payload is missing motors")
    if telemetry.get("receive_qualified") is not True:
        raise StaleFeedback("controller receive qualification is missing")
    receive = telemetry.get("receive")
    if not isinstance(receive, dict) or any(
        not isinstance(receive.get(g), dict) or receive[g].get("valid") is not True or
        type(receive[g].get("sequence")) is not int or receive[g]["sequence"] <= 0 or
        any(isinstance(receive[g].get(k), bool) or
            not isinstance(receive[g].get(k), (int, float)) or
            not math.isfinite(receive[g][k]) for k in ("received_mono", "received_wall"))
        for g in CAN_GROUPS
    ):
        raise StaleFeedback("receive receipts are missing or malformed")
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
        # Name what disagreed: a latched stop with torque still on and a real de-latch
        # are indistinguishable from the message alone, and they need different recovery.
        raise Refusal(f"all-six enable and CAN control are not confirmed: enabled={enabled} "
                      f"ctrl_mode={status.get('ctrl_mode')} motors={flags}")
    return Snapshot(tuple(number(value[c]) for c in JOINTS),
                    Pose(*(number(value[c]) for c in POSE)), number(value["gripper.position"]),
                    enabled, status, telemetry)
