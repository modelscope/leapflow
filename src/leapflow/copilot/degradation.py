"""Graceful degradation policy for the Workflow Copilot.

Monitors system resource usage and automatically disables higher-cost
prediction layers when the system is under pressure. Ensures the Copilot
never degrades user experience or blocks foreground operations.

SRP: Only evaluates resource state → degradation level. No scheduling.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import TYPE_CHECKING, Protocol, Set, runtime_checkable

if TYPE_CHECKING:
    from leapflow.copilot.config import CopilotConfig

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# DegradationLevel — five-tier degradation enum
# ────────────────────────────────────────────────────────────────────────────


class DegradationLevel(Enum):
    """Five-level degradation tiers for the Copilot prediction pipeline.

    Lower levels shed more expensive layers to preserve responsiveness.
    """

    FULL = 0  # All layers active
    NO_L3 = 1  # LLM layer disabled
    NO_L2_L3 = 2  # Only L0+L1 retained
    L0_ONLY = 3  # Hash exact-match only
    DISABLED = 4  # Predictions fully stopped


# Layer IDs corresponding to each tier
_LAYER_SETS: dict[DegradationLevel, Set[str]] = {
    DegradationLevel.FULL: {"L0", "L1", "L2", "L3"},
    DegradationLevel.NO_L3: {"L0", "L1", "L2"},
    DegradationLevel.NO_L2_L3: {"L0", "L1"},
    DegradationLevel.L0_ONLY: {"L0"},
    DegradationLevel.DISABLED: set(),
}


# ────────────────────────────────────────────────────────────────────────────
# SystemMetricsProvider Protocol
# ────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class SystemMetricsProvider(Protocol):
    """Protocol for providing system resource metrics.

    Implementations may use psutil, /proc, or platform-specific APIs.
    """

    def cpu_percent(self) -> float:
        """Current CPU utilisation percentage [0-100]."""
        ...

    def memory_mb(self) -> float:
        """Current process memory usage in megabytes."""
        ...


# ────────────────────────────────────────────────────────────────────────────
# DefaultMetricsProvider — os-based fallback implementation
# ────────────────────────────────────────────────────────────────────────────


class DefaultMetricsProvider:
    """Lightweight system metrics provider based on the os module.

    Limited precision but zero external dependencies.  Replace with a
    psutil-backed implementation for production environments.
    """

    def cpu_percent(self) -> float:
        """Estimate CPU load from os.getloadavg (Unix) or return 0 (Windows).

        Returns average load over last 1 minute normalised to CPU count.
        """
        try:
            load_1min = os.getloadavg()[0]
            cpu_count = os.cpu_count() or 1
            return min(100.0, (load_1min / cpu_count) * 100.0)
        except (OSError, AttributeError):
            # os.getloadavg not available (e.g. Windows)
            return 0.0

    def memory_mb(self) -> float:
        """Estimate current process RSS via /proc or resource module.

        Falls back to 0 if unavailable.
        """
        try:
            import resource

            # ru_maxrss is in KB on Linux, bytes on macOS
            usage = resource.getrusage(resource.RUSAGE_SELF)
            maxrss_kb = usage.ru_maxrss
            # macOS reports bytes, Linux reports KB
            import sys

            if sys.platform == "darwin":
                return maxrss_kb / (1024 * 1024)
            return maxrss_kb / 1024
        except (ImportError, OSError):
            return 0.0


# ────────────────────────────────────────────────────────────────────────────
# DegradationPolicy — resource-aware layer shedding
# ────────────────────────────────────────────────────────────────────────────


class DegradationPolicy:
    """Resource-aware tiered degradation policy.

    Five levels of graceful degradation:
        FULL     → All layers active
        NO_L3    → Disable LLM layer (CPU > 70%)
        NO_L2_L3 → L0+L1 only (CPU > 90%)
        L0_ONLY  → Hash match only (memory > 90% budget)
        DISABLED → Stop predictions (event queue backlog > max * 0.1)

    Design constraints: degradation is automatic, observable, and recoverable.
    The system never crashes or blocks user operations under any load condition.
    Worst case is that the Copilot silently stops predicting.
    """

    def __init__(self, config: CopilotConfig) -> None:
        self._config = config
        self._current_level: DegradationLevel = DegradationLevel.FULL

    @property
    def current_level(self) -> DegradationLevel:
        """The most recently evaluated degradation level."""
        return self._current_level

    def evaluate(
        self,
        cpu_percent: float,
        memory_mb: float,
        event_queue_depth: int,
    ) -> DegradationLevel:
        """Evaluate system metrics and determine the appropriate degradation level.

        Evaluation rules (checked in severity order):
        1. event_queue_depth > max_event_queue * 0.1 → DISABLED
        2. memory_mb > budget * 0.9                  → L0_ONLY
        3. cpu_percent > 90                          → NO_L2_L3
        4. cpu_percent > 70                          → NO_L3
        5. Otherwise                                 → FULL

        Args:
            cpu_percent: CPU utilisation [0-100].
            memory_mb: Current memory usage in MB.
            event_queue_depth: Number of pending events in the queue.

        Returns:
            The computed DegradationLevel.
        """
        memory_budget = self._config.memory_budget_mb
        queue_critical = int(self._config.max_event_queue * 0.1)

        if event_queue_depth > queue_critical:
            level = DegradationLevel.DISABLED
        elif memory_mb > memory_budget * 0.9:
            level = DegradationLevel.L0_ONLY
        elif cpu_percent > 90:
            level = DegradationLevel.NO_L2_L3
        elif cpu_percent > 70:
            level = DegradationLevel.NO_L3
        else:
            level = DegradationLevel.FULL

        # Log transitions
        if level != self._current_level:
            logger.info(
                "Degradation level changed: %s → %s (cpu=%.1f%%, mem=%.1fMB, queue=%d)",
                self._current_level.name,
                level.name,
                cpu_percent,
                memory_mb,
                event_queue_depth,
            )

        self._current_level = level
        return level

    def allowed_layers(self, level: DegradationLevel) -> Set[str]:
        """Return the set of layer IDs allowed to run at a given level.

        Args:
            level: The degradation level to query.

        Returns:
            Set of layer ID strings (e.g. {"L0", "L1"}).
        """
        return _LAYER_SETS.get(level, set())

    def reset(self) -> None:
        """Manually reset to FULL level (e.g. after operator intervention)."""
        if self._current_level != DegradationLevel.FULL:
            logger.info(
                "Degradation manually reset from %s to FULL",
                self._current_level.name,
            )
        self._current_level = DegradationLevel.FULL
