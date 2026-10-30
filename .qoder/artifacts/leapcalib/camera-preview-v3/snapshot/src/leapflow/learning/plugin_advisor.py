# Copyright (c) Alibaba, Inc. and its affiliates.
"""Stateless scoring engine that produces plugin recommendations.

Computed on-demand (when plugin_status is queried), not proactively.
No side effects, no persistence, pure function of (stats, trust).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from leapflow.learning.plugin_stats import PluginUsageTracker
from leapflow.learning.plugin_trust import PluginTrustLedger, PluginTrustLevel


@dataclass(frozen=True)
class PluginRecommendation:
    """Actionable recommendation for a plugin based on its execution history."""

    action: str  # "promote" | "investigate" | "demote"
    reason: str
    trust_level: str  # current trust level name
    confidence: float  # 0.0-1.0


class PluginAdvisor:
    """Pure scoring engine: trust + stats → recommendation."""

    def __init__(
        self,
        trust_ledger: PluginTrustLedger,
        usage_tracker: PluginUsageTracker,
    ) -> None:
        self._trust_ledger = trust_ledger
        self._usage_tracker = usage_tracker

    def recommend(self, plugin_id: str) -> Optional[PluginRecommendation]:
        """Compute a recommendation for the given plugin.

        Returns None when data is insufficient or the plugin is stable.
        """
        stats = self._usage_tracker.stats_for_plugin(plugin_id)
        trust = self._trust_ledger.level(plugin_id)

        if stats is None or stats.total_calls < 3:
            return None  # Insufficient data

        trust_name = trust.name

        # High error rate (>30%) AND trust >= VERIFIED → recommend demotion
        if stats.error_rate > 0.3 and trust >= PluginTrustLevel.VERIFIED:
            return PluginRecommendation(
                action="demote",
                reason=(
                    f"Error rate {stats.error_rate:.0%} exceeds 30% threshold "
                    f"while at {trust_name} trust"
                ),
                trust_level=trust_name,
                confidence=min(1.0, stats.error_rate * 1.5),
            )

        # Moderate error rate (>20%) → recommend investigation
        if stats.error_rate > 0.2:
            return PluginRecommendation(
                action="investigate",
                reason=(
                    f"Error rate {stats.error_rate:.0%} exceeds 20% investigation "
                    f"threshold ({stats.failures}/{stats.total_calls} failures)"
                ),
                trust_level=trust_name,
                confidence=min(1.0, stats.error_rate * 1.2),
            )

        # Low error rate (<5%) AND trust below next promotion threshold → promote
        if stats.error_rate < 0.05:
            next_threshold = self._next_promotion_threshold(trust)
            if next_threshold is not None:
                return PluginRecommendation(
                    action="promote",
                    reason=(
                        f"Error rate {stats.error_rate:.0%} is below 5% with "
                        f"{stats.successes} successes — eligible for promotion"
                    ),
                    trust_level=trust_name,
                    confidence=1.0 - stats.error_rate,
                )

        # Stable — no recommendation needed
        return None

    def _next_promotion_threshold(
        self, current: PluginTrustLevel
    ) -> Optional[PluginTrustLevel]:
        """Return the next trust level if promotion is possible, else None."""
        if current < PluginTrustLevel.PRODUCTION:
            return PluginTrustLevel(current + 1)
        return None


# ── Module-level singleton ──

_default_advisor: Optional[PluginAdvisor] = None


def get_default_advisor() -> Optional[PluginAdvisor]:
    """Return the process-global PluginAdvisor, or None if not wired."""
    return _default_advisor


def set_default_advisor(advisor: PluginAdvisor) -> None:
    """Install the process-global PluginAdvisor."""
    global _default_advisor
    _default_advisor = advisor
