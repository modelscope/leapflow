# Copyright (c) Alibaba, Inc. and its affiliates.
"""Adapter registry arbitration and discovery isolation tests."""

from __future__ import annotations

import importlib.metadata
from typing import Any, Mapping, Sequence

from benchmarks.models import AvailabilityResult, Scenario, TrialResult, TrialStatus
from benchmarks.registry import AdapterRegistry, default_registry


class _Adapter:
    def __init__(self, adapter_id: str = "adapter") -> None:
        self._adapter_id = adapter_id

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def adapter_version(self) -> str:
        return "1.0.0"

    async def availability(self) -> AvailabilityResult:
        return AvailabilityResult(self.adapter_id, True)

    async def list_scenarios(
        self, *, tags: Sequence[str] = (), limit: int = 0,
    ) -> tuple[Scenario, ...]:
        return ()

    async def run_trial(
        self,
        scenario: Scenario,
        *,
        seed: int = 42,
        timeout_seconds: float = 300.0,
        parameters: Mapping[str, Any] | None = None,
    ) -> TrialResult:
        return TrialResult("trial", scenario.scenario_id, TrialStatus.PASSED)


class _SecondAdapter(_Adapter):
    pass


class _EntryPoint:
    def __init__(self, name: str, value: Any = None, error: Exception | None = None) -> None:
        self.name = name
        self._value = value
        self._error = error

    def load(self) -> Any:
        if self._error is not None:
            raise self._error
        return self._value


def test_first_registration_wins_and_conflict_is_recorded() -> None:
    first = _Adapter("same")
    second = _SecondAdapter("same")
    registry = AdapterRegistry()

    assert registry.register(first)
    assert not registry.register(second)
    assert registry.get("same") is first
    assert len(registry.conflicts) == 1
    assert registry.conflicts[0].adapter_id == "same"
    assert registry.conflicts[0].kept_type == "_Adapter"
    assert registry.conflicts[0].rejected_type == "_SecondAdapter"


def test_registry_queries_are_stable_and_sorted() -> None:
    registry = AdapterRegistry((_Adapter("zeta"), _Adapter("alpha")))

    assert registry.list_available() == ("alpha", "zeta")
    assert [adapter.adapter_id for adapter in registry.list_adapters()] == ["alpha", "zeta"]
    assert registry.get("missing") is None


def test_entry_point_failures_and_bad_adapters_are_isolated(monkeypatch) -> None:
    good = _Adapter("good")
    entries = [
        _EntryPoint("broken", error=ImportError("optional SDK missing")),
        _EntryPoint("bad-shape", value=object()),
        _EntryPoint("good", value=lambda: good),
    ]
    calls = 0

    def fake_entry_points(*, group: str):
        nonlocal calls
        calls += 1
        assert group == "leapflow.benchmarks.adapters"
        return entries

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    registry = AdapterRegistry()

    assert registry.discover_entry_points() == 1
    assert registry.get("good") is good
    assert registry.discover_entry_points() == 0
    assert calls == 1


def test_entry_point_conflict_preserves_builtin(monkeypatch) -> None:
    incumbent = _Adapter("shared")
    challenger = _SecondAdapter("shared")
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *, group: [_EntryPoint("challenger", value=challenger)],
    )
    registry = AdapterRegistry((incumbent,))

    assert registry.discover_entry_points() == 0
    assert registry.get("shared") is incumbent
    assert len(registry.conflicts) == 1


def test_default_registry_contains_all_native_and_external_adapters() -> None:
    adapter_ids = default_registry().list_available()

    assert len(adapter_ids) == 25
    assert sum(adapter_id.startswith("native_") for adapter_id in adapter_ids) == 11
    assert "external_command" in adapter_ids
    assert "live_llm" in adapter_ids
    assert "hardware_preflight" in adapter_ids
