# Copyright (c) Alibaba, Inc. and its affiliates.
"""Native LeapRobot benchmark adapters.

All adapters exercise the LeapFlow public hardware API without external
benchmark SDKs.  Import-safe: each adapter is imported inside the factory
function so a broken module cannot prevent others from loading.
"""

from __future__ import annotations

import logging
from typing import Sequence

from benchmarks.protocol import BenchmarkAdapter

logger = logging.getLogger(__name__)

_NATIVE_ADAPTER_FACTORIES: tuple[tuple[str, str, str], ...] = (
    ("benchmarks.native.harmful_instruction", "HarmfulInstructionAdapter",
     "native_harmful_instruction"),
    ("benchmarks.native.approval_and_trust", "ApprovalAndTrustAdapter",
     "native_approval_and_trust"),
    ("benchmarks.native.safety_policy", "SafetyPolicyAdapter",
     "native_safety_policy"),
    ("benchmarks.native.degradation_recovery", "DegradationRecoveryAdapter",
     "native_degradation_recovery"),
    ("benchmarks.native.side_effect_uncertainty", "SideEffectUncertaintyAdapter",
     "native_side_effect_uncertainty"),
    ("benchmarks.native.control_jitter", "ControlJitterAdapter",
     "native_control_jitter"),
    ("benchmarks.native.fleet_isolation", "FleetIsolationAdapter",
     "native_fleet_isolation"),
    ("benchmarks.native.pcd_efficiency", "PCDEfficiencyAdapter",
     "native_pcd_efficiency"),
    ("benchmarks.native.yaml_onboarding", "YAMLOnboardingAdapter",
     "native_yaml_onboarding"),
    ("benchmarks.native.production_governance", "ProductionGovernanceAdapter",
     "native_production_governance"),
    ("benchmarks.native.realtime_local", "RealtimeLocalAdapter",
     "native_realtime_local"),
)
_EXPORTED_CLASSES = {
    class_name: module_path
    for module_path, class_name, _adapter_id in _NATIVE_ADAPTER_FACTORIES
}


def native_adapters() -> Sequence[BenchmarkAdapter]:
    """Instantiate all native benchmark adapters.

    Each adapter is imported inside the function body so that a broken
    adapter module cannot prevent others from loading.
    """
    import importlib

    adapters: list[BenchmarkAdapter] = []
    for module_path, class_name, adapter_id in _NATIVE_ADAPTER_FACTORIES:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            adapters.append(cls())
        except Exception:
            logger.warning(
                "failed to load native adapter %r from %s",
                adapter_id, module_path, exc_info=True,
            )
    return adapters


def __getattr__(name: str):
    """Lazily export adapter classes without eager module imports."""
    module_path = _EXPORTED_CLASSES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


__all__ = [
    "native_adapters",
    "HarmfulInstructionAdapter",
    "ApprovalAndTrustAdapter",
    "SafetyPolicyAdapter",
    "DegradationRecoveryAdapter",
    "SideEffectUncertaintyAdapter",
    "ControlJitterAdapter",
    "FleetIsolationAdapter",
    "PCDEfficiencyAdapter",
    "YAMLOnboardingAdapter",
    "ProductionGovernanceAdapter",
    "RealtimeLocalAdapter",
]
