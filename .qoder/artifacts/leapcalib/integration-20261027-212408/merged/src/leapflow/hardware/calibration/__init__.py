# Copyright (c) Alibaba, Inc. and its affiliates.
"""LeapFlow's side of the LeapCalib boundary.

Only the conversion adapter exists here. The acquisition subscription and the
admission/activation integration sketched for later phases need a live worker and
an approved candidate respectively; neither exists, and an empty module named
after one is how an undelivered capability starts looking delivered.

Nothing in this package is imported by ``leapflow.hardware`` at module load, so a
deployment that never calibrates never pays for it.
"""

from __future__ import annotations

from leapflow.hardware.calibration.adapter import (
    CALIBRATION_FAILURE_MAP,
    hardware_failure_code,
    joint_observation,
    model_identity,
    rgbd_observation,
    stream_profile,
    stream_profile_of,
)

__all__ = [
    "CALIBRATION_FAILURE_MAP",
    "hardware_failure_code",
    "joint_observation",
    "model_identity",
    "rgbd_observation",
    "stream_profile",
    "stream_profile_of",
]
