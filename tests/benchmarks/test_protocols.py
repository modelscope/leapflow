# Copyright (c) Alibaba, Inc. and its affiliates.
"""Protocol runtime conformance tests."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from benchmarks.models import AvailabilityResult, MetricValue, Scenario, TrialResult, TrialStatus
from benchmarks.protocol import BenchmarkAdapter, BenchmarkSuite, MetricAggregator, ScenarioProvider


class _ConformingAdapter:
    """Minimal concrete adapter satisfying BenchmarkAdapter."""

    @property
    def adapter_id(self) -> str:
        return "conforming_test"

    @property
    def adapter_version(self) -> str:
        return "0.1.0"

    async def availability(self) -> AvailabilityResult:
        return AvailabilityResult(adapter_id=self.adapter_id, available=True)

    async def list_scenarios(
        self, *, tags: Sequence[str] = (), limit: int = 0,
    ) -> tuple[Scenario, ...]:
        return (Scenario(scenario_id="s1", name="S1"),)

    async def run_trial(
        self, scenario: Scenario, *, seed: int = 42,
        timeout_seconds: float = 300.0,
        parameters: Mapping[str, Any] | None = None,
    ) -> TrialResult:
        return TrialResult(trial_id="t1", scenario_id=scenario.scenario_id,
                           status=TrialStatus.PASSED)


class _ConformingSuite:
    @property
    def suite_id(self) -> str:
        return "test_suite"

    @property
    def description(self) -> str:
        return "test suite"

    @property
    def adapters(self) -> tuple[BenchmarkAdapter, ...]:
        return (_ConformingAdapter(),)


class _ConformingScenarioProvider:
    async def list_scenarios(
        self, *, tags: Sequence[str] = (), limit: int = 0,
    ) -> tuple[Scenario, ...]:
        return ()


class _ConformingAggregator:
    def aggregate(self, trials: Sequence[TrialResult]) -> tuple[MetricValue, ...]:
        return ()


class _NonConformingObject:
    pass


def test_adapter_protocol_isinstance():
    assert isinstance(_ConformingAdapter(), BenchmarkAdapter)


def test_adapter_protocol_rejects_non_conforming():
    assert not isinstance(_NonConformingObject(), BenchmarkAdapter)


def test_suite_protocol_isinstance():
    assert isinstance(_ConformingSuite(), BenchmarkSuite)


def test_scenario_provider_isinstance():
    assert isinstance(_ConformingScenarioProvider(), ScenarioProvider)


def test_aggregator_protocol_isinstance():
    assert isinstance(_ConformingAggregator(), MetricAggregator)


def test_protocols_are_runtime_checkable():
    for proto in (BenchmarkAdapter, BenchmarkSuite, ScenarioProvider, MetricAggregator):
        assert hasattr(proto, "__protocol_attrs__") or hasattr(proto, "_is_runtime_protocol")
