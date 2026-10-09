# Copyright (c) Alibaba, Inc. and its affiliates.
"""Execute every shipped native benchmark scenario against public APIs."""

from __future__ import annotations

from collections import Counter

from benchmarks.models import TrialStatus
from benchmarks.native import native_adapters
from benchmarks.protocol import BenchmarkAdapter


async def test_all_native_adapters_execute_all_scenarios() -> None:
    adapters = tuple(native_adapters())

    assert len(adapters) == 11
    assert len({adapter.adapter_id for adapter in adapters}) == 11
    assert all(isinstance(adapter, BenchmarkAdapter) for adapter in adapters)

    scenario_counts: Counter[str] = Counter()
    failures: list[str] = []
    total = 0
    for adapter in adapters:
        availability = await adapter.availability()
        assert availability.available, f"{adapter.adapter_id}: {availability.reason}"
        scenarios = await adapter.list_scenarios()
        assert scenarios, f"{adapter.adapter_id} has no scenarios"
        scenario_counts[adapter.adapter_id] = len(scenarios)
        total += len(scenarios)

        for scenario in scenarios:
            assert scenario.adapter_id == adapter.adapter_id
            result = await adapter.run_trial(scenario, seed=42, timeout_seconds=30.0)
            if result.status is not TrialStatus.PASSED:
                failures.append(
                    f"{adapter.adapter_id}/{scenario.scenario_id}: "
                    f"{result.status.value} {result.error}"
                )
            assert result.scenario_id == scenario.scenario_id
            assert result.adapter_id == adapter.adapter_id
            assert result.adapter_version == adapter.adapter_version
            assert result.seed == 42
            assert result.metrics, f"{adapter.adapter_id}/{scenario.scenario_id} lacks metrics"
            assert result.evidence, f"{adapter.adapter_id}/{scenario.scenario_id} lacks evidence"

    assert total == 36, scenario_counts
    assert failures == []


async def test_native_scenario_filtering_and_limits_are_deterministic() -> None:
    for adapter in native_adapters():
        all_scenarios = await adapter.list_scenarios()
        first = await adapter.list_scenarios(limit=1)
        native = await adapter.list_scenarios(tags=("native",))
        no_match = await adapter.list_scenarios(tags=("does-not-exist",))

        assert first == all_scenarios[:1]
        assert native == all_scenarios
        assert no_match == ()
        assert len({scenario.scenario_id for scenario in all_scenarios}) == len(all_scenarios)
