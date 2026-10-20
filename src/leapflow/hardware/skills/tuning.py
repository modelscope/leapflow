# Copyright (c) Alibaba, Inc. and its affiliates.
"""Declared tuning tables of a workcell, each with safe defaults.

Every threshold a skill acts on lives in a declaration, never in skill code:
``limits`` (motion caps), ``freshness`` (how far the world may drift between a
plan and its execution) and ``calibration`` (acceptance thresholds of the
``calib.*`` skills). A workcell overrides any default; a key the table does not
know, or a value that is not a positive finite number, refuses the workcell at
admission rather than surfacing at the first run.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

DEFAULT_LIMITS: Mapping[str, float] = {
    "speed_pct_max": 30.0,
    "default_speed_pct": 20.0,
    "move_timeout_s": 20.0,
    "fk_tolerance_mm": 5.0,
    "fk_tolerance_deg": 2.0,
    "joint_path_step_deg": 2.0,
    "max_steps": 16.0,  # steps one motion.sequence may carry
    "max_wait_s": 10.0,  # longest single ``wait`` step
}

DEFAULT_FRESHNESS: Mapping[str, float] = {
    "start_mm": 2.0,  # the TCP may sit this far from where the plan started
    "start_deg": 1.0,
    "drift_planar_mm": 8.0,  # a referenced object may move this far before plan_stale
    "drift_height_mm": 10.0,
    "drift_yaw_deg": 10.0,
    "yaw_defined_mm": 10.0,  # footprints squarer than this have no meaningful yaw
    "hold_tolerance_mm": 5.0,  # a held width may change this much before "dropped"
}

DEFAULT_CALIBRATION: Mapping[str, float] = {
    "max_zero_error_deg": 0.5,
    "max_tool_length_mm": 400.0,
    "min_pivot_spread_deg": 20.0,
    "max_pivot_residual_mm": 1.0,
    "max_table_residual_mm": 3.0,
    "start_drift_mm": 5.0,
    "start_drift_deg": 2.0,
    "min_hand_eye_poses": 6.0,
    "min_tilt_deg": 5.0,
    "max_hand_eye_translation_rms_mm": 3.0,
    "max_hand_eye_rotation_rms_deg": 0.5,
    "min_verify_points": 5.0,
    "max_verify_p95_mm": 5.0,
}

GRIPPER_SPEC_KEYS = frozenset({"max_width_mm", "min_hold_width_mm"})
DEFAULT_MIN_HOLD_WIDTH_MM = 3.0


def tuned(raw: Any, name: str, defaults: Mapping[str, float]) -> dict[str, float]:
    """*defaults* overridden by *raw*; raises ``ValueError`` on anything unusable."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name!r} must be a mapping")
    unknown = sorted(str(key) for key in raw if key not in defaults)
    if unknown:
        raise ValueError(f"{name} does not understand {', '.join(unknown)}")
    values = dict(defaults)
    for key, value in raw.items():
        values[str(key)] = positive(value, f"{name}.{key}")
    return values


def positive(value: Any, name: str) -> float:
    """A positive finite number; raises ``ValueError`` otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def gripper_spec(raw: Any) -> dict[str, float]:
    """The declared gripper spec; ``max_width_mm`` may be left to the device envelope."""
    if not isinstance(raw, Mapping):
        raise ValueError("'gripper_spec' must be a mapping")
    unknown = sorted(str(key) for key in raw if key not in GRIPPER_SPEC_KEYS)
    if unknown:
        raise ValueError(f"gripper_spec does not understand {', '.join(unknown)}")
    spec = {"min_hold_width_mm": DEFAULT_MIN_HOLD_WIDTH_MM}
    for key, value in raw.items():
        spec[str(key)] = positive(value, f"gripper_spec.{key}")
    return spec


__all__ = [
    "DEFAULT_CALIBRATION",
    "DEFAULT_FRESHNESS",
    "DEFAULT_LIMITS",
    "DEFAULT_MIN_HOLD_WIDTH_MM",
    "GRIPPER_SPEC_KEYS",
    "gripper_spec",
    "positive",
    "tuned",
]
