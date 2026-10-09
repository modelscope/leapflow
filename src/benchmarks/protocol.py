# Copyright (c) Alibaba, Inc. and its affiliates.
"""Runtime-checkable protocols for benchmark adapters and suites.

Every extension point is a ``typing.Protocol`` with ``runtime_checkable``,
matching LeapFlow's core convention.  Concrete adapter implementations live
outside this package (in adapter modules or third-party packages), and must
not import private LeapFlow internals.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from benchmarks.models import (
    AvailabilityResult,
    MetricValue,
    Scenario,
    TrialResult,
)


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """Southbound contract for one benchmark suite.

    An adapter wraps an external benchmark (SWE-bench, GAIA, SafetyBench,
    a LeapRobot native suite, etc.) behind a uniform async interface.

    Implementing modules must not import anything from ``leapflow`` beyond
    public protocols; they may depend on their benchmark's own SDK.
    """

    @property
    def adapter_id(self) -> str:
        """Globally unique identifier for this adapter (e.g. ``swe_bench``)."""
        ...

    @property
    def adapter_version(self) -> str:
        """SemVer string for the adapter implementation itself."""
        ...

    async def availability(self) -> AvailabilityResult:
        """Check whether this adapter can execute right now.

        Must be side-effect free: inspect installed packages, check
        executables on PATH, validate credentials, but never install
        anything or download data.
        """
        ...

    async def list_scenarios(
        self, *, tags: Sequence[str] = (), limit: int = 0,
    ) -> tuple[Scenario, ...]:
        """Return the scenarios this adapter can run.

        When *tags* is non-empty, only scenarios matching at least one tag
        are returned.  When *limit* is positive, at most that many are
        returned (for large suites like SWE-bench).
        """
        ...

    async def run_trial(
        self,
        scenario: Scenario,
        *,
        seed: int = 42,
        timeout_seconds: float = 300.0,
        parameters: Mapping[str, Any] | None = None,
    ) -> TrialResult:
        """Execute one trial of one scenario and return the structured result.

        Must never raise an unstructured exception to the caller: all
        failures (timeout, crash, assertion) must be captured and returned
        as a ``TrialResult`` with the appropriate ``TrialStatus``.
        """
        ...


@runtime_checkable
class BenchmarkSuite(Protocol):
    """A collection of adapters forming one logical benchmark suite."""

    @property
    def suite_id(self) -> str:
        """Unique identifier for the suite."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description."""
        ...

    @property
    def adapters(self) -> tuple[BenchmarkAdapter, ...]:
        """The ordered set of adapters in this suite."""
        ...


@runtime_checkable
class ScenarioProvider(Protocol):
    """Provides scenarios to the runner, abstracting over source and filtering."""

    async def list_scenarios(
        self, *, tags: Sequence[str] = (), limit: int = 0,
    ) -> tuple[Scenario, ...]:
        """Return scenarios matching the given filter criteria."""
        ...


@runtime_checkable
class MetricAggregator(Protocol):
    """Computes aggregate metrics from a collection of trial results."""

    def aggregate(self, trials: Sequence[TrialResult]) -> tuple[MetricValue, ...]:
        """Compute aggregate metrics from trial results.

        Must be pure and deterministic: same input always yields same output.
        Must handle empty input gracefully (return empty tuple).
        """
        ...


__all__ = [
    "BenchmarkAdapter",
    "BenchmarkSuite",
    "MetricAggregator",
    "ScenarioProvider",
]
