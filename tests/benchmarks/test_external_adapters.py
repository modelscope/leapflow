# Copyright (c) Alibaba, Inc. and its affiliates.
"""External adapters remain import-safe and degrade to typed unavailability."""

from __future__ import annotations

import importlib
import socket
import sys
import urllib.request
from types import ModuleType

from benchmarks import adapters as adapters_package
from benchmarks.adapters import base
from benchmarks.adapters.base import SubprocessResult, trial_from_subprocess
from benchmarks.models import Scenario, TrialStatus
from benchmarks.protocol import BenchmarkAdapter


ADAPTER_MODULES = (
    "embodyguard",
    "asimov",
    "is_bench",
    "kinder",
    "calvin",
    "vlabench",
    "robojailbench",
    "attackvla",
    "safety_gymnasium",
    "maniskill",
    "isaac_lab",
    "external_command",
    "live_llm",
    "hardware_preflight",
)

EXTERNAL_IMPORT_ROOTS = {
    "attackvla",
    "calvin_env",
    "embodyguard",
    "gymnasium",
    "is_bench",
    "kinder",
    "mani_skill",
    "mujoco",
    "numpy",
    "omni",
    "robojailbench",
    "safety_gymnasium",
    "sapien",
    "torch",
    "torchvision",
    "vlabench",
}


def test_data_backed_official_command_requires_a_readable_dataset_root(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setattr(base, "_configured_command", lambda *_: "python")
    monkeypatch.setattr(base, "probe_executable", lambda _: "/usr/bin/python")
    monkeypatch.setattr(base, "configured_data_root", lambda _: "")

    missing = base.command_availability("calvin", "IGNORED", "https://official", requires_data_root=True)

    monkeypatch.setattr(base, "configured_data_root", lambda _: str(tmp_path))
    present = base.command_availability("calvin", "IGNORED", "https://official", requires_data_root=True)

    assert not missing.available
    assert missing.missing_dependencies == ("config:benchmark.data_roots.calvin",)
    assert present.available


def test_external_trial_requires_declared_official_metrics() -> None:
    scenario = Scenario(
        scenario_id="official",
        name="Official",
        adapter_id="adapter",
        parameters={"official_metrics": ["official_score"]},
    )
    missing = trial_from_subprocess(
        "adapter", "1", scenario, SubprocessResult(returncode=0, stdout='{"other": 1}'),
    )
    present = trial_from_subprocess(
        "adapter", "1", scenario, SubprocessResult(returncode=0, stdout='{"official_score": 1}'),
    )

    assert missing.status is TrialStatus.FAILED
    assert missing.error_type == "OfficialResultContractError"
    assert missing.evidence
    assert present.status is TrialStatus.PASSED
    assert present.metrics[0].name == "official_score"
    assert present.evidence


def _forbid(operation: str):
    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError(f"external adapter attempted {operation}")

    return fail


def test_all_external_adapter_modules_are_import_safe() -> None:
    before = set(sys.modules)
    loaded: list[ModuleType] = []

    for short_name in ADAPTER_MODULES:
        module_name = f"benchmarks.adapters.{short_name}"
        sys.modules.pop(module_name, None)
        loaded.append(importlib.import_module(module_name))

    newly_imported_roots = {
        name.partition(".")[0]
        for name in set(sys.modules) - before
        if name.partition(".")[0] in EXTERNAL_IMPORT_ROOTS
    }
    assert newly_imported_roots == set()
    assert len(loaded) == 14


async def test_missing_dependencies_return_typed_unavailable_without_side_effects(
    monkeypatch,
) -> None:
    monkeypatch.setattr(base, "probe_module", lambda name: False)
    monkeypatch.setattr(base, "probe_executable", lambda name: None)
    monkeypatch.setattr(base, "probe_env", lambda name: False)
    monkeypatch.setattr(base, "_gpu_available", lambda: False)
    monkeypatch.setattr(urllib.request, "urlopen", _forbid("a network download"))
    monkeypatch.setattr(urllib.request, "urlretrieve", _forbid("a network download"))
    monkeypatch.setattr(socket, "create_connection", _forbid("a network connection"))

    adapters = tuple(adapters_package.builtin_adapters())
    assert len(adapters) == 14
    assert {adapter.adapter_id for adapter in adapters} == set(ADAPTER_MODULES)
    assert all(isinstance(adapter, BenchmarkAdapter) for adapter in adapters)

    for adapter in adapters:
        if adapter.adapter_id in {"live_llm", "hardware_preflight"}:
            continue
        module = sys.modules[type(adapter).__module__]
        if hasattr(module, "run_subprocess"):
            monkeypatch.setattr(module, "run_subprocess", _forbid("a subprocess"))

        availability = await adapter.availability()
        assert not availability.available, adapter.adapter_id
        assert availability.reason

        scenarios = await adapter.list_scenarios(limit=1)
        scenario = scenarios[0] if scenarios else Scenario(
            scenario_id="dependency_probe",
            name="Dependency probe",
            adapter_id=adapter.adapter_id,
        )
        result = await adapter.run_trial(scenario, seed=17, timeout_seconds=0.1)

        assert result.status is TrialStatus.UNAVAILABLE
        assert result.adapter_id == adapter.adapter_id
        assert result.adapter_version == adapter.adapter_version
        assert result.scenario_id == scenario.scenario_id
        assert result.seed == 17
        assert result.error == availability.reason
