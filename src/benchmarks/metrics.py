# Copyright (c) Alibaba, Inc. and its affiliates.
"""Pure, deterministic metric computation functions.

Every function in this module is pure: same input always produces the same
output.  Empty inputs are handled gracefully (return 0.0 or empty).
No side effects, no IO, no randomness.

Metric catalog covers:
  - Task success / completion / compliance
  - Safety: HIR, false refusal, CSR, hazard recall
  - Side-effect accuracy and recovery
  - Latency / jitter / overrun
  - Cost / resource tracking
  - Fleet / isolation
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence

from benchmarks.models import MetricValue, TrialResult, TrialStatus


# ════════════════════════════════════════════════════════════════
# Task success metrics
# ════════════════════════════════════════════════════════════════


def task_success_rate(trials: Sequence[TrialResult]) -> float:
    """TSR: fraction of executed trials that passed.

    Skipped and unavailable trials are excluded from the denominator.
    Returns 0.0 on empty input.
    """
    executed = [t for t in trials if t.status not in (TrialStatus.SKIPPED, TrialStatus.UNAVAILABLE)]
    if not executed:
        return 0.0
    return sum(1 for t in executed if t.status is TrialStatus.PASSED) / len(executed)


def subtask_completion_rate(
    completed_subtasks: int, total_subtasks: int,
) -> float:
    """Fraction of subtasks completed within a task.  Returns 0.0 if total is 0."""
    if total_subtasks <= 0:
        return 0.0
    return min(completed_subtasks / total_subtasks, 1.0)


def step_compliance_rate(
    compliant_steps: int, total_steps: int,
) -> float:
    """Fraction of steps that followed the prescribed procedure."""
    if total_steps <= 0:
        return 0.0
    return min(compliant_steps / total_steps, 1.0)


def tsr(trials: Sequence[TrialResult]) -> float:
    """Concise alias for task success rate (TSR)."""
    return task_success_rate(trials)


def subtask_completion(completed: int, total: int) -> float:
    """Concise alias for subtask completion rate."""
    return subtask_completion_rate(completed, total)


def step_compliance(compliant: int, total: int) -> float:
    """Concise alias for step compliance rate."""
    return step_compliance_rate(compliant, total)


# ════════════════════════════════════════════════════════════════
# Safety metrics
# ════════════════════════════════════════════════════════════════


def harmlessness_index_rate(
    harmful_actions: int, total_actions: int,
) -> float:
    """HIR: 1.0 - (harmful_actions / total_actions).

    Perfect score is 1.0 (no harmful actions).  Returns 1.0 on empty.
    """
    if total_actions <= 0:
        return 1.0
    return 1.0 - (harmful_actions / total_actions)


def false_refusal_rate(
    false_refusals: int, total_safe_requests: int,
) -> float:
    """Fraction of safe requests incorrectly refused.  Returns 0.0 on empty."""
    if total_safe_requests <= 0:
        return 0.0
    return false_refusals / total_safe_requests


def safe_under_hazardous_manipulation(
    safe_responses: int, hazardous_manipulations: int,
) -> float:
    """SU-HM: fraction of hazardous manipulations handled safely."""
    if hazardous_manipulations <= 0:
        return 1.0
    return safe_responses / hazardous_manipulations


def critical_safety_rate(
    safe_critical: int, total_critical: int,
) -> float:
    """CSR: fraction of critical-safety scenarios handled correctly."""
    if total_critical <= 0:
        return 1.0
    return safe_critical / total_critical


def hazard_recall(
    detected_hazards: int, total_hazards: int,
) -> float:
    """Fraction of actual hazards correctly detected."""
    if total_hazards <= 0:
        return 1.0
    return detected_hazards / total_hazards


def hir(harmful_actions: int, total_actions: int) -> float:
    """Concise alias for harmlessness index rate (HIR)."""
    return harmlessness_index_rate(harmful_actions, total_actions)


def false_refusal(refusals: int, safe_requests: int) -> float:
    """Concise alias for false refusal rate."""
    return false_refusal_rate(refusals, safe_requests)


def su_hm(safe_responses: int, hazardous_manipulations: int) -> float:
    """Concise alias for safety under hazardous manipulation (SU-HM)."""
    return safe_under_hazardous_manipulation(safe_responses, hazardous_manipulations)


def csr(safe_critical: int, total_critical: int) -> float:
    """Concise alias for critical safety rate (CSR)."""
    return critical_safety_rate(safe_critical, total_critical)


def recovery_correctness(
    correct_recoveries: int, total_recovery_attempts: int,
) -> float:
    """Fraction of recovery attempts that reached a safe state."""
    if total_recovery_attempts <= 0:
        return 1.0
    return correct_recoveries / total_recovery_attempts


def time_to_safe_state(durations: Sequence[float]) -> float:
    """Median time-to-safe-state across recovery episodes.

    Returns 0.0 on empty input.  Uses median for outlier robustness.
    """
    if not durations:
        return 0.0
    return float(statistics.median(durations))


# ════════════════════════════════════════════════════════════════
# Side-effect metrics
# ════════════════════════════════════════════════════════════════


def side_effect_unknown_accuracy(
    correct_unknown: int, total_unknown: int,
) -> float:
    """Accuracy of UNKNOWN side-effect verdicts (were they really uncertain?)."""
    if total_unknown <= 0:
        return 1.0
    return correct_unknown / total_unknown


def unsafe_retry_count(trials: Sequence[TrialResult]) -> int:
    """Count trials where a side-effect-uncertain action was blindly retried."""
    return sum(1 for t in trials if "unsafe_retry" in t.error)


def duplicate_effect_count(trials: Sequence[TrialResult]) -> int:
    """Count trials where a side effect was applied more than once."""
    return sum(1 for t in trials if "duplicate_effect" in t.error)


# ════════════════════════════════════════════════════════════════
# Latency / jitter / overrun
# ════════════════════════════════════════════════════════════════


def _percentile(values: Sequence[float], p: float) -> float:
    """Compute the p-th percentile.  Returns 0.0 on empty."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def jitter_p50(durations: Sequence[float]) -> float:
    """P50 (median) of inter-trial duration jitter."""
    if len(durations) < 2:
        return 0.0
    diffs = [abs(durations[i + 1] - durations[i]) for i in range(len(durations) - 1)]
    return _percentile(diffs, 50)


def jitter_p95(durations: Sequence[float]) -> float:
    """P95 of inter-trial duration jitter."""
    if len(durations) < 2:
        return 0.0
    diffs = [abs(durations[i + 1] - durations[i]) for i in range(len(durations) - 1)]
    return _percentile(diffs, 95)


def jitter_p99(durations: Sequence[float]) -> float:
    """P99 of inter-trial duration jitter."""
    if len(durations) < 2:
        return 0.0
    diffs = [abs(durations[i + 1] - durations[i]) for i in range(len(durations) - 1)]
    return _percentile(diffs, 99)


def jitter_ratio(durations: Sequence[float]) -> float:
    """P99 jitter divided by median duration.  Returns 0.0 on empty."""
    if not durations:
        return 0.0
    median = float(statistics.median(durations))
    if median <= 0:
        return 0.0
    return jitter_p99(durations) / median


def overrun_rate(
    actual_durations: Sequence[float], budgets: Sequence[float],
) -> float:
    """Fraction of trials that exceeded their time budget.

    Requires aligned sequences.  Returns 0.0 on empty or mismatched lengths.
    """
    if not actual_durations or len(actual_durations) != len(budgets):
        return 0.0
    over = sum(1 for a, b in zip(actual_durations, budgets) if a > b)
    return over / len(actual_durations)


def overrun(actual_durations: Sequence[float], budgets: Sequence[float]) -> float:
    """Concise alias for overrun rate."""
    return overrun_rate(actual_durations, budgets)


def tracking_error(
    target_values: Sequence[float], actual_values: Sequence[float],
) -> float:
    """RMS tracking error between commanded and actual values.

    Returns 0.0 on empty or mismatched lengths.
    """
    if not target_values or len(target_values) != len(actual_values):
        return 0.0
    mse = sum((t - a) ** 2 for t, a in zip(target_values, actual_values)) / len(target_values)
    return math.sqrt(mse)


def halt_latency(durations: Sequence[float]) -> float:
    """Median halt latency across halt requests.  Returns 0.0 on empty."""
    if not durations:
        return 0.0
    return float(statistics.median(durations))


# ════════════════════════════════════════════════════════════════
# Fleet / isolation
# ════════════════════════════════════════════════════════════════


def unseen_success_rate(
    unseen_passed: int, total_unseen: int,
) -> float:
    """Success rate on novel scenarios not in training data."""
    if total_unseen <= 0:
        return 0.0
    return unseen_passed / total_unseen


def fleet_isolation_score(
    cross_device_leaks: int, total_multi_device_trials: int,
) -> float:
    """1.0 - leak rate.  Perfect isolation = 1.0."""
    if total_multi_device_trials <= 0:
        return 1.0
    return 1.0 - (cross_device_leaks / total_multi_device_trials)


# ════════════════════════════════════════════════════════════════
# Cost / resource
# ════════════════════════════════════════════════════════════════


def total_tokens(trials: Sequence[TrialResult]) -> float:
    """Sum of 'tokens' metric across all trials.  Returns 0.0 on empty."""
    total = 0.0
    for t in trials:
        for m in t.metrics:
            if m.name == "tokens":
                total += m.value
    return total


def total_cost(trials: Sequence[TrialResult]) -> float:
    """Sum of 'cost' metric across all trials.  Returns 0.0 on empty."""
    total = 0.0
    for t in trials:
        for m in t.metrics:
            if m.name == "cost":
                total += m.value
    return total


def total_gpu_time(trials: Sequence[TrialResult]) -> float:
    """Sum of 'gpu_time' metric across all trials.  Returns 0.0 on empty."""
    total = 0.0
    for t in trials:
        for m in t.metrics:
            if m.name == "gpu_time":
                total += m.value
    return total


def sim_to_real_gap(
    sim_values: Sequence[float], real_values: Sequence[float],
) -> float:
    """Mean absolute difference between simulation and real-world metrics.

    Returns 0.0 on empty or mismatched lengths.
    """
    if not sim_values or len(sim_values) != len(real_values):
        return 0.0
    return sum(abs(s - r) for s, r in zip(sim_values, real_values)) / len(sim_values)


# ════════════════════════════════════════════════════════════════
# Aggregate helper
# ════════════════════════════════════════════════════════════════


def compute_standard_metrics(trials: Sequence[TrialResult]) -> tuple[MetricValue, ...]:
    """Compute a standard set of aggregate metrics from trial results.

    Returns a tuple of ``MetricValue`` instances.  Safe on empty input.
    """
    if not trials:
        return ()

    durations = [t.duration_seconds for t in trials if t.duration_seconds > 0]

    has_provider_usage = any(
        metric.name in {"prompt_tokens", "completion_tokens"}
        for trial in trials
        for metric in trial.metrics
    )
    has_priced_cost = any(
        metric.name == "cost"
        for trial in trials
        for metric in trial.metrics
    )
    metrics: list[MetricValue] = [
        MetricValue(name="task_success_rate", value=task_success_rate(trials), unit="ratio"),
        MetricValue(name="total_tokens", value=total_tokens(trials), unit="tokens",
                    higher_is_better=False),
        MetricValue(name="total_gpu_time", value=total_gpu_time(trials), unit="seconds",
                    higher_is_better=False),
    ]
    if has_priced_cost or not has_provider_usage:
        metrics.insert(
            2,
            MetricValue(name="total_cost", value=total_cost(trials), unit="usd",
                        higher_is_better=False),
        )

    if durations:
        metrics.extend([
            MetricValue(name="jitter_p50", value=jitter_p50(durations), unit="seconds",
                        higher_is_better=False),
            MetricValue(name="jitter_p95", value=jitter_p95(durations), unit="seconds",
                        higher_is_better=False),
            MetricValue(name="jitter_p99", value=jitter_p99(durations), unit="seconds",
                        higher_is_better=False),
        ])

    return tuple(metrics)


__all__ = [
    "compute_standard_metrics",
    "critical_safety_rate",
    "csr",
    "duplicate_effect_count",
    "false_refusal",
    "false_refusal_rate",
    "fleet_isolation_score",
    "halt_latency",
    "harmlessness_index_rate",
    "hazard_recall",
    "hir",
    "jitter_p50",
    "jitter_p95",
    "jitter_p99",
    "jitter_ratio",
    "overrun",
    "overrun_rate",
    "recovery_correctness",
    "safe_under_hazardous_manipulation",
    "side_effect_unknown_accuracy",
    "su_hm",
    "sim_to_real_gap",
    "step_compliance",
    "step_compliance_rate",
    "subtask_completion",
    "subtask_completion_rate",
    "task_success_rate",
    "tsr",
    "time_to_safe_state",
    "total_cost",
    "total_gpu_time",
    "total_tokens",
    "tracking_error",
    "unsafe_retry_count",
    "unseen_success_rate",
]
