"""Learning effectiveness evaluation — quantifies whether the learning loop is actually learning.

Core question: "Is the system getting better over time, or just accumulating noise?"

Metrics tracked:
- Skill promotion rate: How often do skills advance through tiers
- PatternMiner precision: Ratio of user-accepted to total suggested patterns
- Regression rate: How often do verified skills regress
- Coverage rate: Fraction of user tasks that match a known skill
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LearningMetrics:
    """Quantitative learning effectiveness metrics for a time window."""
    window_start_ts: float = 0.0
    window_end_ts: float = 0.0
    
    # Skill lifecycle
    skills_created: int = 0
    skills_promoted: int = 0  # Tier advanced
    skills_demoted: int = 0   # Tier regressed
    skills_deactivated: int = 0  # Confidence below threshold
    
    # PatternMiner
    patterns_discovered: int = 0
    patterns_accepted: int = 0  # User confirmed/used the suggestion
    patterns_rejected: int = 0  # User dismissed
    
    # Execution quality
    executions_total: int = 0
    executions_successful: int = 0
    regressions_detected: int = 0
    
    # Coverage
    tasks_matched_skill: int = 0
    tasks_total: int = 0

    @property
    def promotion_rate(self) -> float:
        """Fraction of skills that advanced tier in this window."""
        total = self.skills_created + self.skills_promoted + self.skills_demoted
        return self.skills_promoted / max(total, 1)

    @property
    def pattern_precision(self) -> float:
        """Fraction of discovered patterns accepted by user."""
        total = self.patterns_accepted + self.patterns_rejected
        return self.patterns_accepted / max(total, 1)

    @property
    def success_rate(self) -> float:
        """Execution success rate."""
        return self.executions_successful / max(self.executions_total, 1)

    @property
    def regression_rate(self) -> float:
        """Fraction of executions that triggered regression."""
        return self.regressions_detected / max(self.executions_total, 1)

    @property
    def coverage_rate(self) -> float:
        """Fraction of user tasks matched by existing skills."""
        return self.tasks_matched_skill / max(self.tasks_total, 1)

    def summary(self) -> Dict[str, Any]:
        """Return metrics as a flat dict for logging/audit."""
        return {
            "promotion_rate": round(self.promotion_rate, 3),
            "pattern_precision": round(self.pattern_precision, 3),
            "success_rate": round(self.success_rate, 3),
            "regression_rate": round(self.regression_rate, 3),
            "coverage_rate": round(self.coverage_rate, 3),
            "skills_created": self.skills_created,
            "patterns_discovered": self.patterns_discovered,
            "executions_total": self.executions_total,
        }


class LearningEffectivenessTracker:
    """Tracks learning metrics over rolling time windows.
    
    Accumulates events and periodically emits metrics summaries
    to audit log for observability.
    """

    def __init__(
        self,
        *,
        window_duration_s: float = 86400.0,  # 24h window
        emit_interval_s: float = 3600.0,     # Emit every hour
    ) -> None:
        self._window_duration = window_duration_s
        self._emit_interval = emit_interval_s
        self._current = LearningMetrics(window_start_ts=time.time())
        self._last_emit_ts = time.time()
        self._history: List[LearningMetrics] = []

    # ── Event recording ──

    def record_skill_created(self) -> None:
        self._current.skills_created += 1

    def record_skill_promoted(self) -> None:
        self._current.skills_promoted += 1

    def record_skill_demoted(self) -> None:
        self._current.skills_demoted += 1

    def record_skill_deactivated(self) -> None:
        self._current.skills_deactivated += 1

    def record_pattern_discovered(self) -> None:
        self._current.patterns_discovered += 1

    def record_pattern_accepted(self) -> None:
        self._current.patterns_accepted += 1

    def record_pattern_rejected(self) -> None:
        self._current.patterns_rejected += 1

    def record_execution(self, *, success: bool) -> None:
        self._current.executions_total += 1
        if success:
            self._current.executions_successful += 1

    def record_regression(self) -> None:
        self._current.regressions_detected += 1

    def record_task(self, *, matched_skill: bool) -> None:
        self._current.tasks_total += 1
        if matched_skill:
            self._current.tasks_matched_skill += 1

    # ── Window management ──

    def maybe_emit(self) -> Optional[Dict[str, Any]]:
        """Check if it's time to emit metrics. Returns summary or None."""
        now = time.time()
        if now - self._last_emit_ts < self._emit_interval:
            return None
        
        self._last_emit_ts = now
        summary = self._current.summary()
        logger.info("LearningEffectiveness: %s", summary)
        
        # Rotate window if exceeded
        if now - self._current.window_start_ts >= self._window_duration:
            self._current.window_end_ts = now
            self._history.append(self._current)
            if len(self._history) > 30:  # Keep last 30 windows
                self._history = self._history[-30:]
            self._current = LearningMetrics(window_start_ts=now)
        
        return summary

    @property
    def current_metrics(self) -> LearningMetrics:
        return self._current

    @property 
    def history(self) -> List[LearningMetrics]:
        return self._history
