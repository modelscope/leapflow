# Copyright (c) Alibaba, Inc. and its affiliates.
"""Plugin health monitoring producer.

Emits Monitor Findings when plugin trust degrades or error rate spikes,
enabling proactive notification to the Agent without waiting for
explicit plugin_status queries.

Domain: ``plugin_health``. Both halves of that are required and neither implies the
other: ``MonitorCoordinator`` registers this producer *and* arms a ``plugin-health``
watch on a 5-minute interval. Registration alone leaves it resolvable but never
called, because a producer only runs when a watch names its domain -- which is the
state this module was actually in while this docstring claimed otherwise.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Sequence

from leapflow.monitor.types import (
    Evidence,
    Finding,
    ProducerContext,
    Severity,
    SuggestedAction,
)

logger = logging.getLogger(__name__)

# Error rate threshold: emit alert when recent error rate exceeds 25%
_ERROR_RATE_THRESHOLD = 0.25

# Trust level ordering for degradation detection
_TRUST_RANK = {"DRAFT": 0, "CANDIDATE": 1, "VERIFIED": 2, "PRODUCTION": 3}


class PluginHealthProducer:
    """MonitorProducer that watches plugin health metrics.

    Detects two anomaly classes:
    1. Trust degradation: plugin trust level drops since last observation.
    2. High error rate: recent error rate > 25% (configurable threshold).
    """

    domain = "plugin_health"

    def __init__(
        self, error_rate_threshold: float = _ERROR_RATE_THRESHOLD,
        *, error_window_s: float = 3600.0, min_calls: int = 5,
    ) -> None:
        self.configure(error_rate_threshold, error_window_s, min_calls)
        self._last_trust_levels: dict[str, str] = {}

    def configure(self, threshold: float, window_s: float, min_calls: int) -> None:
        """Apply validated health policy, including runtime config reloads."""
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("plugin_health.error_rate_threshold must be within 0..1")
        if not math.isfinite(window_s) or window_s <= 0 or min_calls < 1:
            raise ValueError("Plugin health window and minimum calls must be positive")
        self._error_rate_threshold = threshold
        self._error_window_s = window_s
        self._min_calls = min_calls

    def configure_from_settings(self, settings: Any) -> None:
        self.configure(
            float(getattr(settings, "plugin_health_error_rate_threshold", 0.25)),
            float(getattr(settings, "plugin_health_error_window_s", 3600.0)),
            int(getattr(settings, "plugin_health_min_calls", 5)),
        )

    async def observe(self, ctx: ProducerContext) -> Sequence[Finding]:
        """Called periodically by MonitorManager. Check trust + error rates."""
        findings: list[Finding] = []

        try:
            from leapflow.learning.plugin_advisor import get_default_advisor
        except ImportError:
            return findings

        advisor = get_default_advisor()
        if advisor is None:
            return findings

        trust_ledger = advisor._trust_ledger
        usage_tracker = advisor._usage_tracker

        # Iterate all known plugins in the tool registry
        try:
            from leapflow.plugins import get_registry
            reg = get_registry()
            plugin_ids = list(reg.plugins.keys())
        except (ImportError, RuntimeError, AttributeError):
            return findings

        watch_id = ctx.spec.watch_id or "plugin_health"

        for plugin_id in plugin_ids:
            # --- Trust degradation detection ---
            current_level = trust_ledger.level(plugin_id).name
            previous_level = self._last_trust_levels.get(plugin_id)

            if previous_level is not None:
                current_rank = _TRUST_RANK.get(current_level, 0)
                previous_rank = _TRUST_RANK.get(previous_level, 0)

                if current_rank < previous_rank:
                    findings.append(Finding(
                        watch_id=watch_id,
                        domain=self.domain,
                        title=f"Plugin trust degraded: {plugin_id}",
                        summary=(
                            f"Trust level dropped from {previous_level} to "
                            f"{current_level} for plugin '{plugin_id}'."
                        ),
                        severity=Severity.NOTABLE,
                        tags=("plugin_health", "trust_degradation"),
                        evidence=(
                            Evidence(
                                kind="metric",
                                label="previous_level",
                                value=previous_level,
                            ),
                            Evidence(
                                kind="metric",
                                label="current_level",
                                value=current_level,
                            ),
                        ),
                        suggested_actions=(
                            SuggestedAction(
                                name="plugin_status",
                                label=f"Inspect {plugin_id}",
                                kind="intent",
                                params={"plugin_id": plugin_id},
                            ),
                        ),
                        dedup_key=f"trust_degrade:{plugin_id}:{current_level}",
                    ))

            # Update last-seen level
            self._last_trust_levels[plugin_id] = current_level

            # --- High error rate detection ---
            stats = usage_tracker.stats_for_plugin(plugin_id, since=time.time() - self._error_window_s)
            if stats is None or stats.total_calls < self._min_calls:
                continue  # Insufficient data for error rate judgment

            if stats.error_rate > self._error_rate_threshold:
                codes = getattr(stats, "failed_codes", {})
                actions = [SuggestedAction(
                    name="plugin_status", label=f"Inspect {plugin_id}",
                    kind="intent", params={"plugin_id": plugin_id},
                )]
                if _disable_warranted(codes, stats.failures):
                    actions.append(SuggestedAction(
                        name="plugin_disable", label=f"Disable {plugin_id}",
                        kind="approval", params={"plugin_id": plugin_id},
                    ))
                findings.append(Finding(
                    watch_id=watch_id,
                    domain=self.domain,
                    title=f"High error rate: {plugin_id}",
                    summary=(
                        f"Plugin '{plugin_id}' error rate is "
                        f"{stats.error_rate:.0%} ({stats.failures}/{stats.total_calls} "
                        f"failures) — exceeds {self._error_rate_threshold:.0%} threshold."
                    ),
                    severity=Severity.ALERT,
                    score=stats.error_rate,
                    tags=("plugin_health", "high_error_rate"),
                    evidence=(
                        Evidence(
                            kind="metric",
                            label="error_rate",
                            value=f"{stats.error_rate:.2%}",
                        ),
                        Evidence(
                            kind="metric",
                            label="total_calls",
                            value=str(stats.total_calls),
                        ),
                        Evidence(kind="metric", label="error_window_s", value=str(self._error_window_s)),
                        Evidence(kind="metric", label="failure_codes", value=str(dict(
                            sorted(codes.items(), key=lambda item: item[1], reverse=True)[:5]
                        ))),
                    ),
                    suggested_actions=tuple(actions),
                    dedup_key=f"error_rate:{plugin_id}",
                ))

        return findings


def _disable_warranted(codes: dict[str, int], failures: int) -> bool:
    """Input-only errors implicate the caller, not the plugin implementation."""
    input_codes = {"invalid_request", "invalid_arguments", "unknown_key", "unknown_category", "missing_parameter"}
    input_prefixes = ("config_unknown_", "config_invalid_", "invalid_argument", "schema_", "argument_")
    if sum(codes.values()) < failures:
        return True  # missing classifications are not proven caller-input failures
    return any(code not in input_codes and not code.startswith(input_prefixes) for code in codes)


__all__ = ["PluginHealthProducer"]
