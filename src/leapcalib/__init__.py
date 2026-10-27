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

**Scope of this release.** Only the offline readiness layer exists: typed
observation/parameter/estimate schemas, SI rigid-body geometry with explicit
raw-versus-corrected joint spaces, geometry-asset availability, error budgets,
and integrity-checked evidence bundles with replay. There are no solvers, no
acquisition from live hardware, and no activation path. ``estimation/`` and
``runtime/`` are deliberately absent rather than empty: an extension point with
no consumer is scaffolding, and scaffolding is how an unimplemented capability
starts looking like a delivered one.

Everything here is stdlib-only, so a hardware node can import the contracts
without a numerical stack. NumPy and friends belong to the solvers, and the
solvers do not exist yet.
"""

from __future__ import annotations

SCHEMA_NAMESPACE = "leapcalib"
"""Prefix of every schema identity this package emits."""

__all__ = ["SCHEMA_NAMESPACE"]
