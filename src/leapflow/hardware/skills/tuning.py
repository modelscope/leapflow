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
    "max_verify_rms_mm": 5.0,
    # calib.screws: three joints are the minimum that fixes the camera's height as
    # well as its heading, so fewer cannot produce a mount at all. The gap is what
    # each joint's own screw shift disagrees with the mount's prediction by; the
    # agreement is the fraction of moved points that land on the measured surface.
    #
    # Both limits come from the archived P-B capture, replayed through the skill's
    # own code path: an honest capture of that cell scores 0.48-0.51 with a worst
    # gap of 11.6-13.3 mm (always the wrist, whose 25 deg motion carries the least
    # surface). A 20 mm ceiling leaves that measured case a third of its value in
    # handroom while still refusing a mount that explains no joint at all. It is
    # not a fine discriminator -- a mis-paired sample scores *lower* on it, which
    # is why the joint-by-joint drift check, not this number, is what refuses one.
    #
    # The vertical is measured from the base joint, whose axis *is* the base's +z:
    # a pair turning about a line 2 deg off z reports a direction 2 deg off, and
    # the mount inherits it. So the pair that supplies the vertical has to be that
    # joint, within this bound; a run that names three horizontal joints is refused
    # with a message naming the base joint rather than measured from a tilted line.
    "min_screw_pairs": 3.0,
    "max_screw_shift_gap_mm": 20.0,
    "min_screw_agreement": 0.35,
    "max_vertical_axis_deg": 1.0,
}

DEFAULT_GRASP_VERIFY: Mapping[str, float] = {
    "alignment_mm": 3.0,
    "max_age_s": 1.0,
    "min_frames": 3.0,
    "min_proof_lift_mm": 5.0,
    "max_proof_lift_mm": 10.0,
    "max_relative_error_mm": 2.0,
    "max_slip_mm": 1.0,
    "max_uncertainty_mm": 1.0,
    "capture_drift_mm": 0.5,
    "capture_drift_deg": 0.5,
    "poll_s": 0.05,
    "clearance_mm": 3.0,
    "jaw_stroke_error_mm": 1.5,
    "max_sync_skew_s": 0.1,
    "pad_midpoint_mm": 2.0,
    "empty_gap_mm": 1.0,
    "contact_geometry_mm": 1.5,
    "contact_contour_px": 2.0,
    "contact_axis_deg": 2.0,
    "support_mm": 1.0,
    "effort_uncertainty_n": 0.2,
    "effort_error_n": 0.5,
    "effort_width_scope_mm": 1.0,
    "effort_pose_scope_mm": 10.0,
    "effort_pose_scope_deg": 1.0,
    "cycle_path_step_mm": 5.0,
    "cycle_max_waypoints": 128.0,
    "placement_mm": 3.0,
    "warmup_timeout_s": 5.0,
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
    "DEFAULT_GRASP_VERIFY",
    "DEFAULT_LIMITS",
    "DEFAULT_MIN_HOLD_WIDTH_MM",
    "GRIPPER_SPEC_KEYS",
    "gripper_spec",
    "positive",
    "tuned",
]
