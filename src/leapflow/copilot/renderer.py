"""Suggestion rendering and display gating for the Workflow Copilot.

Implements the "rather not show than show late" principle:
if a prediction cannot be displayed within the idle window, it is discarded.

SRP: Only decides *when* and *how* to display — no prediction logic.
"""

from __future__ import annotations

import logging
import time as _time
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from leapflow.copilot.config import CopilotConfig
    from leapflow.copilot.types import HintRenderer, PredictionCandidate

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# DisplayGate — decides whether to show a candidate
# ────────────────────────────────────────────────────────────────────────────


class DisplayGate:
    """展示门控 — 决定何时、以何种方式展示建议。

    核心原则："宁可不展示，不可延迟展示。"
    如果计算未在停顿窗口内完成，丢弃本次建议。
    """

    # Destructive operations require confidence above this threshold
    _DESTRUCTIVE_THRESHOLD: float = 0.8

    def __init__(self, config: CopilotConfig) -> None:
        self._config = config

    def should_display(
        self, candidate: PredictionCandidate, idle_ms: int
    ) -> bool:
        """Determine if a candidate should be displayed now.

        Rules (evaluated in order):
        1. Confidence must meet minimum display threshold.
        2. Destructive actions require higher confidence (>0.8).
        3. Idle duration must exceed the candidate's display delay.
        4. Candidate must not have expired.

        Args:
            candidate: The prediction candidate to evaluate.
            idle_ms: Current user idle duration in milliseconds.

        Returns:
            True if the candidate should be displayed.
        """
        # Rule 1: minimum confidence
        if candidate.confidence < self._config.min_confidence_display:
            return False

        # Rule 2: destructive operations need higher bar
        if candidate.is_destructive and candidate.confidence < self._DESTRUCTIVE_THRESHOLD:
            return False

        # Rule 3: wait for sufficient idle time
        if idle_ms < candidate.display_delay_ms:
            return False

        # Rule 4: expired candidates are stale
        if candidate.expire_ts > 0.0 and _time.time() > candidate.expire_ts:
            return False

        return True

    def select_best(
        self, candidates: List[PredictionCandidate], idle_ms: int
    ) -> Optional[PredictionCandidate]:
        """Select the best displayable candidate from a list.

        Filters by `should_display`, then returns the highest-confidence
        candidate. Ties broken by lower display_delay_ms (faster layer).

        Args:
            candidates: All available prediction candidates.
            idle_ms: Current user idle duration in milliseconds.

        Returns:
            Best candidate to display, or None if nothing qualifies.
        """
        displayable = [c for c in candidates if self.should_display(c, idle_ms)]
        if not displayable:
            return None
        # Sort by confidence desc, then display_delay_ms asc as tie-breaker
        displayable.sort(key=lambda c: (-c.confidence, c.display_delay_ms))
        return displayable[0]


# ────────────────────────────────────────────────────────────────────────────
# SuggestionRenderer — orchestrates display lifecycle
# ────────────────────────────────────────────────────────────────────────────


class SuggestionRenderer:
    """建议渲染协调器 — 管理建议的展示/撤回生命周期。

    职责：
    - 通过 DisplayGate 选择最佳候选
    - 调用 HintRenderer 展示/撤回
    - 追踪当前展示中的建议
    """

    def __init__(
        self,
        config: CopilotConfig,
        renderer: HintRenderer,
        gate: Optional[DisplayGate] = None,
    ) -> None:
        self._config = config
        self._renderer = renderer
        self._gate = gate or DisplayGate(config)
        self._currently_shown: Optional[PredictionCandidate] = None

    @property
    def currently_shown(self) -> Optional[PredictionCandidate]:
        """The candidate currently displayed to the user (if any)."""
        return self._currently_shown

    async def on_idle(
        self, idle_ms: int, candidates: List[PredictionCandidate]
    ) -> Optional[PredictionCandidate]:
        """Evaluate and potentially display a suggestion during idle.

        Called by the copilot engine when idle is detected.

        Args:
            idle_ms: How long the user has been idle (milliseconds).
            candidates: Available prediction candidates.

        Returns:
            The candidate that was displayed, or None.
        """
        best = self._gate.select_best(candidates, idle_ms)
        if best is None:
            return None

        # Show the suggestion via renderer
        try:
            await self._renderer.show(best)
            self._currently_shown = best
            logger.debug(
                "Displayed suggestion: %s (confidence=%.2f, layer=%s)",
                best.action_description,
                best.confidence,
                best.source_layer,
            )
        except Exception:
            logger.exception("Error showing suggestion via renderer")
            return None

        return best

    async def dismiss(self) -> None:
        """Dismiss the currently shown suggestion.

        Called when a new event arrives or the user explicitly rejects.
        """
        if self._currently_shown is not None:
            try:
                await self._renderer.dismiss()
            except Exception:
                logger.exception("Error dismissing suggestion")
            finally:
                self._currently_shown = None


# ────────────────────────────────────────────────────────────────────────────
# LogHintRenderer — default logging implementation of HintRenderer
# ────────────────────────────────────────────────────────────────────────────


class LogHintRenderer:
    """HintRenderer 的默认 logging 实现 — 用于测试/调试。

    将建议渲染为 logger.info 输出，不产生任何 UI 副作用。
    """

    def __init__(self) -> None:
        self._visible: bool = False
        self._current: Optional[PredictionCandidate] = None

    async def show(self, candidate: PredictionCandidate) -> None:
        """Log the suggestion as an info-level message."""
        self._visible = True
        self._current = candidate
        logger.info(
            "[CopilotHint] 💡 %s (confidence=%.2f, layer=%s)",
            candidate.action_description,
            candidate.confidence,
            candidate.source_layer,
        )

    async def dismiss(self) -> None:
        """Log dismissal."""
        if self._visible:
            logger.info("[CopilotHint] dismissed")
        self._visible = False
        self._current = None

    @property
    def is_visible(self) -> bool:
        """Whether a hint is currently being shown."""
        return self._visible
