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
from typing import Sequence

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

    def __init__(self, error_rate_threshold: float = _ERROR_RATE_THRESHOLD) -> None:
        self._error_rate_threshold = error_rate_threshold
        # Track previous trust levels to detect degradation
        self._last_trust_levels: dict[str, str] = {}

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
            stats = usage_tracker.stats_for_plugin(plugin_id)
            if stats is None or stats.total_calls < 5:
                continue  # Insufficient data for error rate judgment

            if stats.error_rate > self._error_rate_threshold:
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
                    ),
                    suggested_actions=(
                        SuggestedAction(
                            name="plugin_status",
                            label=f"Inspect {plugin_id}",
                            kind="intent",
                            params={"plugin_id": plugin_id},
                        ),
                        SuggestedAction(
                            name="plugin_disable",
                            label=f"Disable {plugin_id}",
                            kind="approval",
                            params={"plugin_id": plugin_id},
                        ),
                    ),
                    dedup_key=f"error_rate:{plugin_id}",
                ))

        return findings


__all__ = ["PluginHealthProducer"]
