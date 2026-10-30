# Copyright (c) Alibaba, Inc. and its affiliates.
"""LeapCalib: typed contracts and offline tooling for target-free software calibration.

LeapCalib estimates geometry. It never acquires actuator authority, never owns a
camera, and never writes a deployment's database. The robot stack that owns those
things hands it observations and decides what to do with the result -- which is
why the dependency arrow points one way only:

    LeapFlow integration adapter --> leapcalib public contracts
    vendor/external adapters     --> leapcalib public contracts
    leapcalib core               -X-> LeapFlow, vendor SDKs, ROS, LLM engine

Nothing under ``leapcalib`` may import ``leapflow``. In particular it must not
import ``leapflow.engine.calibration``, which calibrates LLM token budgets and
shares nothing with this package but a word.

**Scope of this release.** Typed SI observations, integrity-checked replay,
robust metric correspondence/motion candidates, and measured local xyz/yaw
response fitting with bounded shadow proposals. Native mesh ICP, independently
anchored joint zeros, live acquisition ownership, and activation are not shipped.
Factory intrinsics and nominal/raw kinematics remain explicit priors.

Contracts, inspection, and geometry are stdlib-only. The optional NumPy backend
loads only during solves in ``estimation``; importing or discovering methods
does not require a numerical stack. There is no ``runtime`` service package.
"""

from __future__ import annotations

SCHEMA_NAMESPACE = "leapcalib"
"""Prefix of every schema identity this package emits."""

__all__ = ["SCHEMA_NAMESPACE"]
