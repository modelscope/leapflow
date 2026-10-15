# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared control policy builder, extracted from tools.py (P0-D).

Both the host-side ``HardwareTools`` and the node-side
``NodeControlSessions`` need to build a ControlPolicy from user parameters.
This module keeps the logic in one place.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


def build_control_policy(
    mode: str,
    joint_ids: tuple[str, ...] | Sequence[str],
    targets: Mapping[str, float],
    waypoints: tuple[tuple[float, Mapping[str, float]], ...] | Sequence[Any],
    params: Mapping[str, Any],
) -> tuple[Any, str | None]:
    """Build a ControlPolicy from mode and user parameters.

    Returns ``(policy, None)`` on success or ``(None, error_message)`` on
    failure.  The caller is responsible for turning the error message into
    the appropriate refusal payload.
    """
    from leapflow.hardware.realtime import (
        ImpedanceController,
        PIDJointController,
        TrajectoryTracker,
    )

    gains = params.get("gains") if isinstance(params.get("gains"), Mapping) else {}
    try:
        kp = float(gains.get("kp", 10.0))
        ki = float(gains.get("ki", 0.1))
        kd = float(gains.get("kd", 1.0))
        if mode == "pid":
            policy = PIDJointController(tuple(joint_ids), kp=kp, ki=ki, kd=kd)
            policy.set_target(dict(targets))
        elif mode == "impedance":
            policy = ImpedanceController(
                tuple(joint_ids),
                stiffness=float(gains.get("stiffness", 100.0)),
                damping=float(gains.get("damping", 10.0)),
            )
            policy.set_reference(dict(targets))
        else:
            inner = PIDJointController(tuple(joint_ids), kp=kp, ki=ki, kd=kd)
            policy = TrajectoryTracker(inner, tuple(waypoints), loop_mode="hold_final")
    except (TypeError, ValueError) as exc:
        return None, str(exc)
    return policy, None


__all__ = ["build_control_policy"]
