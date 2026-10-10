# Copyright (c) Alibaba, Inc. and its affiliates.
"""Lazy adapter factory for external benchmark adapters.

Import-safe: importing this module never imports an external benchmark
SDK.  Adapters are instantiated on demand via ``builtin_adapters()``.
"""

from __future__ import annotations

import logging
from typing import Sequence

from benchmarks.protocol import BenchmarkAdapter

logger = logging.getLogger(__name__)


def builtin_adapters() -> Sequence[BenchmarkAdapter]:
    """Instantiate all built-in external benchmark adapters.

    Each adapter is imported inside the function body so that a broken
    adapter module cannot prevent others from loading.
    """
    adapters: list[BenchmarkAdapter] = []

    _ADAPTER_FACTORIES: tuple[tuple[str, str, str], ...] = (
        ("benchmarks.adapters.embodyguard", "EmBodyGuardAdapter", "embodyguard"),
        ("benchmarks.adapters.asimov", "AsimovAdapter", "asimov"),
        ("benchmarks.adapters.is_bench", "ISBenchAdapter", "is_bench"),
        ("benchmarks.adapters.kinder", "KinderAdapter", "kinder"),
        ("benchmarks.adapters.calvin", "CalvinAdapter", "calvin"),
        ("benchmarks.adapters.vlabench", "VLABenchAdapter", "vlabench"),
        ("benchmarks.adapters.robojailbench", "RoboJailBenchAdapter", "robojailbench"),
        ("benchmarks.adapters.attackvla", "AttackVLAAdapter", "attackvla"),
        ("benchmarks.adapters.safety_gymnasium", "SafetyGymnasiumAdapter", "safety_gymnasium"),
        ("benchmarks.adapters.maniskill", "ManiSkillAdapter", "maniskill"),
        ("benchmarks.adapters.isaac_lab", "IsaacLabAdapter", "isaac_lab"),
        ("benchmarks.adapters.external_command", "ExternalCommandAdapter", "external_command"),
        ("benchmarks.adapters.live_llm", "LiveLLMAdapter", "live_llm"),
        ("benchmarks.adapters.hardware_preflight", "HardwarePreflightAdapter", "hardware_preflight"),
    )

    for module_path, class_name, adapter_id in _ADAPTER_FACTORIES:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            adapter = cls()
            adapters.append(adapter)
        except Exception:
            logger.warning(
                "failed to load built-in adapter %r from %s",
                adapter_id, module_path, exc_info=True,
            )

    return adapters


__all__ = ["builtin_adapters"]
