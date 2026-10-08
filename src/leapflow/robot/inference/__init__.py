# Copyright (c) Alibaba, Inc. and its affiliates.
"""LeapRobot inference strategy framework.

Provides pluggable physical AI inference strategies for test-time scaling.
Each strategy maps observation → action through a different compute path
(local VLA, remote PolicyServer, classical controller, etc.) and declares
its compute profile so the system can adapt the budget to task complexity.
"""

from __future__ import annotations

from leapflow.robot.inference.position_control import PositionControlStrategy
from leapflow.robot.inference.registry import InferenceStrategyRegistry
from leapflow.robot.inference.strategy import (
    ComputeBudget,
    ComputeProfile,
    InferenceResult,
    InferenceStrategy,
)
from leapflow.robot.inference.vla_local import VLALocalStrategy
from leapflow.robot.inference.vla_remote import VLARemoteStrategy

__all__ = [
    "ComputeBudget",
    "ComputeProfile",
    "InferenceResult",
    "InferenceStrategy",
    "InferenceStrategyRegistry",
    "PositionControlStrategy",
    "VLALocalStrategy",
    "VLARemoteStrategy",
]
