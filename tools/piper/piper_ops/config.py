"""Typed operational configuration, never embedded credentials or SDK defaults."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Conservative commissioning limits, independently configurable."""

    endpoint: str = "tcp://127.0.0.1:8767"
    node_id: str = "yahboom"
    device_id: str = "robot_piper_arm"
    token_file: Path = Path.home() / ".leapflow/.lhp-token"
    issuer_key: Path = Path.home() / ".leapflow/profiles/default/hardware/lhp_permit_issuer.key"
    root: Path = Path(__file__).resolve().parents[1]
    speed_pct: float = 10.0
    waypoint_mm: float = 10.0
    position_mm: float = 2.0
    orientation_deg: float = 1.0
    joint_deg: float = 0.5
    grip_mm: float = 0.5
    poll_s: float = 0.1
    settled_samples: int = 3
    move_timeout_s: float = 20.0
    run_timeout_s: float = 300.0
    request_timeout_s: float = 10.0
    joint_margin_deg: float = 1.0
    manual_joint_limits_deg: tuple[tuple[float, float], ...] = (
        (-150.0, 150.0), (0.0, 180.0), (-170.0, 0.0),
        (-89.0, 89.0), (-89.0, 89.0), (-165.0, 165.0),
    )
    joint_step_deg: float = 10.0
    init_max_deg: float = 10.0
    deviation_deg: float = 3.0
    preflight_temp_c: float = 45.0
    max_temp_c: float = 50.0
    voltage_min_v: float = 21.6
    voltage_max_v: float = 26.0
    gripper_max_mm: float | None = None
    gripper_max_evidence: str = ""
    gripper_spec_max_mm: float | None = None
    gripper_usable_max_mm: float | None = None
    node_python: Path = Path.home() / "miniconda3/envs/leap/bin/python"
    node_leap: Path = Path.home() / "miniconda3/envs/leap/bin/leap"
    node_control_python: Path | None = None
    node_workspace: Path = Path.home() / "leap_work"
    node_host: str = "0.0.0.0"
    node_port: int = 8767

    @classmethod
    def load(cls, path: Path | None = None, **overrides: object) -> Settings:
        """Read a nonsecret JSON config and reject unknown or invalid settings."""
        data = json.loads(path.read_text()) if path else {}
        if not isinstance(data, dict):
            raise ValueError("operations configuration must be an object")
        known = {f.name for f in fields(cls)}
        unknown = (set(data) | set(overrides)) - known
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        data.update({k: v for k, v in overrides.items() if v is not None})
        for key in ("token_file", "issuer_key", "root", "node_python", "node_leap", "node_workspace",
                    "node_control_python"):
            if key in data and data[key] is not None:
                data[key] = Path(data[key]).expanduser()
        if "manual_joint_limits_deg" in data:
            intervals = data["manual_joint_limits_deg"]
            if not isinstance(intervals, (list, tuple)) or any(not isinstance(p, (list, tuple)) for p in intervals):
                raise ValueError("manual joint limits must contain intervals")
            data["manual_joint_limits_deg"] = tuple(tuple(pair) for pair in intervals)
        settings = replace(cls(), **data)
        settings.validate()
        return settings

    def validate(self) -> None:
        """Fail closed on malformed budgets, ranges or maximum-opening evidence."""
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool):
                raise ValueError(f"{field.name} cannot be boolean")
            if isinstance(value, (float, int)) and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{field.name} must be finite and positive")
        numeric = ("speed_pct", "waypoint_mm", "position_mm", "orientation_deg", "joint_deg",
                   "grip_mm", "poll_s", "settled_samples", "move_timeout_s", "run_timeout_s",
                   "request_timeout_s", "joint_margin_deg", "joint_step_deg", "init_max_deg",
                   "deviation_deg", "preflight_temp_c", "max_temp_c", "voltage_min_v", "voltage_max_v")
        for name in numeric:
            if not isinstance(getattr(self, name), (int, float)):
                raise ValueError(f"{name} must be numeric")
        for name in ("endpoint", "node_id", "device_id", "gripper_max_evidence", "node_host"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        for name in ("token_file", "issuer_key", "root", "node_python", "node_leap", "node_workspace"):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a Path; use Settings.load for JSON conversion")
        if self.node_control_python is not None and not isinstance(self.node_control_python, Path):
            raise ValueError("node_control_python must be a Path or null")
        if not isinstance(self.manual_joint_limits_deg, (tuple, list)) or len(self.manual_joint_limits_deg) != 6 or any(
            not isinstance(p, (tuple, list)) or len(p) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                              not math.isfinite(v) for v in p) or p[0] >= p[1]
            for p in self.manual_joint_limits_deg
        ):
            raise ValueError("six finite ordered manual joint intervals are required")
        for name in ("gripper_max_mm", "gripper_spec_max_mm", "gripper_usable_max_mm"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric or null")
        if type(self.node_port) is not int or not 1 <= self.node_port <= 65535:
            raise ValueError("node_port must be an integer between 1 and 65535")
        if self.position_mm > 2 or self.orientation_deg > 1 or self.grip_mm > 0.5:
            raise ValueError("arrival tolerances cannot exceed approved commissioning limits")
        if self.preflight_temp_c > self.max_temp_c:
            raise ValueError("preflight temperature cannot exceed running limit")
        if not 1 <= self.speed_pct <= 10:
            raise ValueError("commissioning speed must be between 1 and 10 percent")
        if self.waypoint_mm > 10 or self.settled_samples < 3:
            raise ValueError("waypoints must be <=10 mm; require >=3 consecutive arrivals")
        if not isinstance(self.settled_samples, int):
            raise ValueError("settled_samples must be an integer")
        if self.voltage_min_v >= self.voltage_max_v:
            raise ValueError("voltage limits are reversed")
        if self.gripper_max_mm is not None and not self.gripper_max_evidence.strip():
            raise ValueError("gripper_max_mm requires fitted-gripper evidence")
