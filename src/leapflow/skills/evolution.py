"""Skill evolution policy — manages confidence/version progression and degradation.

Implements the trust gradient: DRAFT → CANDIDATE → VERIFIED → PRODUCTION
through successful executions, and regression-triggered degradation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from leapflow.domain.skill_types import SkillTier


@dataclass(frozen=True, slots=True)
class EvolutionOutcome:
    """Result of applying evolution policy to a skill execution."""

    skill_name: str
    new_confidence: float
    version_bump: bool  # Whether to increment version
    tier_before: SkillTier
    tier_after: SkillTier

    @property
    def tier_changed(self) -> bool:
        return self.tier_before != self.tier_after


@runtime_checkable
class SkillEvolutionPolicy(Protocol):
    """Protocol for skill confidence/version evolution strategies."""

    def on_execution_result(
        self,
        skill_name: str,
        *,
        success: bool,
        duration_s: float,
        current_confidence: float,
        current_version: int,
        source: str = "",
    ) -> EvolutionOutcome:
        """Compute evolution outcome based on execution result."""
        ...

    def on_regression_detected(
        self,
        skill_name: str,
        *,
        current_confidence: float,
        current_version: int,
        source: str = "",
    ) -> EvolutionOutcome:
        """Handle regression — typically degrades confidence."""
        ...

    def decay_inactive(
        self,
        skill_name: str,
        *,
        current_confidence: float,
        current_version: int,
        last_used_ts: float,
        source: str = "",
    ) -> EvolutionOutcome:
        """Apply time-based confidence decay for inactive skills."""
        ...


class EMAConfidencePolicy:
    """Exponential Moving Average confidence evolution policy.

    Confidence updates:
    - Success: confidence += alpha_up * (1.0 - confidence)
    - Failure: confidence -= alpha_down * confidence
    - Regression: confidence *= (1 - regression_penalty)
    - Inactivity: confidence *= decay_factor^(days_inactive)

    Version bumps occur on every Nth successful execution.
    """

    def __init__(
        self,
        *,
        alpha_up: float = 0.12,
        alpha_down: float = 0.20,
        regression_penalty: float = 0.30,
        inactivity_decay_per_day: float = 0.02,
        version_bump_interval: int = 3,
        min_confidence: float = 0.05,
        max_confidence: float = 0.99,
    ) -> None:
        self._alpha_up = alpha_up
        self._alpha_down = alpha_down
        self._regression_penalty = regression_penalty
        self._inactivity_decay_per_day = inactivity_decay_per_day
        self._version_bump_interval = version_bump_interval
        self._min_confidence = min_confidence
        self._max_confidence = max_confidence
        # Track consecutive successes per skill for version bump
        self._success_counts: dict[str, int] = {}

    def on_execution_result(
        self,
        skill_name: str,
        *,
        success: bool,
        duration_s: float,
        current_confidence: float,
        current_version: int,
        source: str = "",
    ) -> EvolutionOutcome:
        tier_before = SkillTier.from_metadata(current_version, current_confidence, source)

        if success:
            new_conf = current_confidence + self._alpha_up * (1.0 - current_confidence)
            self._success_counts[skill_name] = self._success_counts.get(skill_name, 0) + 1
            version_bump = (self._success_counts[skill_name] % self._version_bump_interval == 0)
        else:
            new_conf = current_confidence - self._alpha_down * current_confidence
            self._success_counts[skill_name] = 0  # Reset streak
            version_bump = False

        new_conf = max(self._min_confidence, min(self._max_confidence, new_conf))
        new_version = current_version + (1 if version_bump else 0)
        tier_after = SkillTier.from_metadata(new_version, new_conf, source)

        return EvolutionOutcome(
            skill_name=skill_name,
            new_confidence=new_conf,
            version_bump=version_bump,
            tier_before=tier_before,
            tier_after=tier_after,
        )

    def on_regression_detected(
        self,
        skill_name: str,
        *,
        current_confidence: float,
        current_version: int,
        source: str = "",
    ) -> EvolutionOutcome:
        tier_before = SkillTier.from_metadata(current_version, current_confidence, source)
        new_conf = current_confidence * (1.0 - self._regression_penalty)
        new_conf = max(self._min_confidence, new_conf)
        self._success_counts[skill_name] = 0
        tier_after = SkillTier.from_metadata(current_version, new_conf, source)

        return EvolutionOutcome(
            skill_name=skill_name,
            new_confidence=new_conf,
            version_bump=False,
            tier_before=tier_before,
            tier_after=tier_after,
        )

    def decay_inactive(
        self,
        skill_name: str,
        *,
        current_confidence: float,
        current_version: int,
        last_used_ts: float,
        source: str = "",
    ) -> EvolutionOutcome:
        tier_before = SkillTier.from_metadata(current_version, current_confidence, source)
        days_inactive = (time.time() - last_used_ts) / 86400.0
        if days_inactive <= 0:
            return EvolutionOutcome(
                skill_name=skill_name,
                new_confidence=current_confidence,
                version_bump=False,
                tier_before=tier_before,
                tier_after=tier_before,
            )
        decay_factor = (1.0 - self._inactivity_decay_per_day) ** days_inactive
        new_conf = current_confidence * decay_factor
        new_conf = max(self._min_confidence, new_conf)
        tier_after = SkillTier.from_metadata(current_version, new_conf, source)

        return EvolutionOutcome(
            skill_name=skill_name,
            new_confidence=new_conf,
            version_bump=False,
            tier_before=tier_before,
            tier_after=tier_after,
        )
