# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bounded latency measurements for runtime and evolution components."""
from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LatencySummary:
    """Immutable percentile snapshot in milliseconds."""

    count: int = 0
    minimum_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    maximum_ms: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class RollingLatency:
    """Thread-safe bounded sampler with O(1) writes and cold-path sorting."""

    def __init__(self, *, capacity: int = 2048) -> None:
        self._samples: deque[float] = deque(maxlen=max(1, int(capacity)))
        self._lock = threading.Lock()

    def observe(self, duration_ms: float) -> None:
        with self._lock:
            self._samples.append(max(0.0, float(duration_ms)))

    def snapshot(self) -> LatencySummary:
        with self._lock:
            samples = sorted(self._samples)
        if not samples:
            return LatencySummary()
        count = len(samples)
        return LatencySummary(
            count=count,
            minimum_ms=round(samples[0], 4),
            mean_ms=round(sum(samples) / count, 4),
            p50_ms=round(_percentile(samples, 0.50), 4),
            p95_ms=round(_percentile(samples, 0.95), 4),
            p99_ms=round(_percentile(samples, 0.99), 4),
            maximum_ms=round(samples[-1], 4),
        )


def _percentile(sorted_samples: list[float], quantile: float) -> float:
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    position = (len(sorted_samples) - 1) * min(1.0, max(0.0, quantile))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_samples[lower]
    weight = position - lower
    return sorted_samples[lower] * (1.0 - weight) + sorted_samples[upper] * weight


def aggregate_latency_snapshots(
    snapshots: dict[str, LatencySummary],
) -> dict[str, dict[str, int | float]]:
    """Build a read-only aggregation of named latency snapshots.

    Accepts a mapping of ``{label: LatencySummary}`` — each snapshot is
    already computed (cold-path sorted inside ``RollingLatency.snapshot()``);
    this helper simply converts them to plain dicts keyed by label, suitable
    for serialisation into a board/usage payload.

    Returns only entries with ``count > 0`` to avoid noise.
    This is a pure read of existing data — no hot-path cost.
    """
    result: dict[str, dict[str, int | float]] = {}
    for label, snap in snapshots.items():
        if snap.count > 0:
            result[label] = snap.to_dict()
    return result


__all__ = ["LatencySummary", "RollingLatency", "aggregate_latency_snapshots"]
