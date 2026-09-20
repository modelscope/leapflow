# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for bounded runtime latency summaries."""
from __future__ import annotations

from leapflow.performance import RollingLatency


def test_rolling_latency_reports_interpolated_percentiles() -> None:
    latency = RollingLatency(capacity=5)
    for value in (1, 2, 3, 4, 5):
        latency.observe(value)

    snapshot = latency.snapshot()

    assert snapshot.count == 5
    assert snapshot.minimum_ms == 1
    assert snapshot.mean_ms == 3
    assert snapshot.p50_ms == 3
    assert snapshot.p95_ms == 4.8
    assert snapshot.p99_ms == 4.96
    assert snapshot.maximum_ms == 5


def test_rolling_latency_is_bounded() -> None:
    latency = RollingLatency(capacity=3)
    for value in (1, 2, 3, 4):
        latency.observe(value)

    snapshot = latency.snapshot()

    assert snapshot.count == 3
    assert snapshot.minimum_ms == 2
    assert snapshot.maximum_ms == 4
