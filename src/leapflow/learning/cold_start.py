"""Cold start strategy — handles system behavior when learning data is insufficient.

Design goal (from Active Learning Design doc): "冷启动与适配成本极低 — 首次使用即开始学习"

Strategies:
- PatternMiner: lower min_frequency threshold during cold start
- Suggestions: proactively guide user toward teach/observe modes
- Fallback: graceful degradation when models have no prior
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ColdStartPhase(Enum):
    """Current cold start phase based on accumulated data."""
    EMPTY = "empty"           # No data at all
    WARMING = "warming"       # Some data, not enough for full confidence
    READY = "ready"           # Sufficient data for normal operation


@dataclass(frozen=True, slots=True)
class ColdStartConfig:
    """Configuration for cold start behavior."""
    min_events_for_warming: int = 20    # Events needed to exit EMPTY
    min_events_for_ready: int = 200     # Events needed to exit WARMING
    min_skills_for_ready: int = 3       # Skills needed to be READY
    cold_start_frequency_divisor: int = 2  # Divide min_frequency by this during cold start
    prompt_user_after_s: float = 300.0  # Suggest teach mode after 5min of no skills
    mode: str = "passive"  # "passive" | "prompt" | "demo"


class ColdStartManager:
    """Manages cold start phase transitions and adaptive thresholds.
    
    Monitors system data accumulation and adjusts PatternMiner/suggestion
    thresholds accordingly. Exits cold start automatically when sufficient
    data is available.
    """

    def __init__(self, config: ColdStartConfig = ColdStartConfig()) -> None:
        self._config = config
        self._phase = ColdStartPhase.EMPTY
        self._start_ts = time.time()
        self._events_seen: int = 0
        self._skills_count: int = 0
        self._user_prompted: bool = False

    @property
    def phase(self) -> ColdStartPhase:
        return self._phase

    @property
    def is_cold(self) -> bool:
        """Whether system is still in cold start (EMPTY or WARMING)."""
        return self._phase != ColdStartPhase.READY

    def update_stats(self, *, events_count: int = 0, skills_count: int = 0) -> None:
        """Update accumulated data stats and potentially advance phase."""
        self._events_seen = max(self._events_seen, events_count)
        self._skills_count = max(self._skills_count, skills_count)
        self._advance_phase()

    def get_adjusted_min_frequency(self, base_min_frequency: int) -> int:
        """Return adjusted min_frequency for PatternMiner during cold start."""
        if self._phase == ColdStartPhase.EMPTY:
            # Very permissive during empty phase
            return max(2, base_min_frequency // (self._config.cold_start_frequency_divisor * 2))
        elif self._phase == ColdStartPhase.WARMING:
            return max(3, base_min_frequency // self._config.cold_start_frequency_divisor)
        return base_min_frequency

    def should_prompt_user(self) -> Optional[str]:
        """Check if we should prompt user to use teach mode.
        
        Returns suggestion message or None.
        """
        if self._config.mode != "prompt":
            return None
        if self._user_prompted:
            return None
        if self._phase != ColdStartPhase.EMPTY:
            return None
        
        elapsed = time.time() - self._start_ts
        if elapsed >= self._config.prompt_user_after_s:
            self._user_prompted = True
            return (
                "LeapFlow is ready to learn from you. "
                "Try 'leap teach' to demonstrate a workflow, "
                "or just keep working — I'll observe and learn automatically."
            )
        return None

    def get_fallback_response(self) -> Optional[str]:
        """Get fallback guidance when data is insufficient for predictions."""
        if self._phase == ColdStartPhase.EMPTY:
            return "Still learning your patterns. I'll have suggestions soon."
        elif self._phase == ColdStartPhase.WARMING:
            return None  # Warming phase can attempt predictions
        return None

    def _advance_phase(self) -> None:
        """Check if phase should advance based on current stats."""
        if self._phase == ColdStartPhase.EMPTY:
            if self._events_seen >= self._config.min_events_for_warming:
                self._phase = ColdStartPhase.WARMING
                logger.info("ColdStart: advanced to WARMING (events=%d)", self._events_seen)
        
        if self._phase == ColdStartPhase.WARMING:
            if (self._events_seen >= self._config.min_events_for_ready
                    and self._skills_count >= self._config.min_skills_for_ready):
                self._phase = ColdStartPhase.READY
                logger.info(
                    "ColdStart: advanced to READY (events=%d, skills=%d)",
                    self._events_seen, self._skills_count,
                )
