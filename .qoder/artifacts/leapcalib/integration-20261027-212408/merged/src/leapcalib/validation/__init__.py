# Copyright (c) Alibaba, Inc. and its affiliates.
"""What the evidence permits: asset availability, error budgets, replayable bundles."""

from __future__ import annotations

from leapcalib.validation.assets import (
    ASSET_DH_TABLE,
    ASSET_POINT_LANDMARKS,
    ASSET_PRIMITIVE_HULL,
    ASSET_SURFACE_MESH,
    METHOD_CAD_ICP,
    METHOD_LANDMARK_PNP,
    METHOD_SCENE_MOTION,
    METHOD_TASK_LOCAL,
    GeometryAsset,
    GeometryAvailability,
    availability,
)
from leapcalib.validation.budget import (
    COMBINE_INDEPENDENT,
    COMBINE_WORST_CASE,
    TERM_RANDOM,
    TERM_SYSTEMATIC,
    BudgetVerdict,
    ErrorBudget,
    ErrorTerm,
    TaskTolerance,
    delay_error_m,
    lever_arm_error_m,
    observation_delay_budget,
)
from leapcalib.validation.bundle import (
    EvidenceBundle,
    ReplaySource,
    Retention,
    read_bundle,
    write_bundle,
)

__all__ = [
    "ASSET_DH_TABLE",
    "ASSET_POINT_LANDMARKS",
    "ASSET_PRIMITIVE_HULL",
    "ASSET_SURFACE_MESH",
    "COMBINE_INDEPENDENT",
    "COMBINE_WORST_CASE",
    "METHOD_CAD_ICP",
    "METHOD_LANDMARK_PNP",
    "METHOD_SCENE_MOTION",
    "METHOD_TASK_LOCAL",
    "TERM_RANDOM",
    "TERM_SYSTEMATIC",
    "BudgetVerdict",
    "ErrorBudget",
    "ErrorTerm",
    "EvidenceBundle",
    "GeometryAsset",
    "GeometryAvailability",
    "ReplaySource",
    "Retention",
    "TaskTolerance",
    "availability",
    "delay_error_m",
    "lever_arm_error_m",
    "observation_delay_budget",
    "read_bundle",
    "write_bundle",
]
