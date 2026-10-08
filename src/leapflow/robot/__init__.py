# Copyright (c) Alibaba, Inc. and its affiliates.
"""LeapRobot: physical robot control and inference for LeapFlow.

This package provides the core abstractions for robot control, sensor
access, and policy inference.  It absorbs and adapts the inference-side
subset of the upstream open-source robot SDK, removing training dependencies and
integrating with LeapFlow's plugin, HCP, and configuration systems.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
