"""Feedback collection and evolution loop for the Workflow Copilot.

Captures user reactions (accept / ignore / correct / reject) to displayed
suggestions and drives online model evolution via EMA confidence updates.

SRP: Only handles feedback → learning. No prediction, no rendering.
"""

from __future__ import annotations

import logging
import time as _time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from leapflow.copilot.config import CopilotConfig
    from leapflow.copilot.types import (
        ContextState,
        FeedbackSignal,
        FeedbackType,
        PredictionCandidate,
        PredictorLayer,
    )

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# FeedbackCollector — translates user behaviour into FeedbackSignal
# ────────────────────────────────────────────────────────────────────────────


class FeedbackCollector:
    """反馈信号采集器 — 追踪已展示建议的用户反应。

    职责：将用户行为（接受/忽略/纠正/拒绝）转化为结构化 FeedbackSignal。

    Usage:
        1. Call `track_shown()` when a suggestion is displayed.
        2. Call `on_accept()` / `on_explicit_reject()` / `on_next_action()`
           depending on what the user does next.
        3. Each method returns an Optional[FeedbackSignal] ready for the
           EvolutionLoop.
    """

    def __init__(self) -> None:
        # (candidate, context_at_show, shown_timestamp)
        self._pending: Optional[Tuple[PredictionCandidate, ContextState, float]] = None

    @property
    def has_pending(self) -> bool:
        """Whether there is a tracked suggestion awaiting feedback."""
        return self._pending is not None

    def track_shown(
        self, candidate: PredictionCandidate, context: ContextState
    ) -> None:
        """Record that a suggestion was shown to the user.

        Any previously pending (un-resolved) tracking is implicitly discarded
        (treated as IGNORE by timeout elsewhere).
        """
        self._pending = (candidate, context, _time.time())

    def on_accept(self) -> Optional[FeedbackSignal]:
        """User accepted the suggestion (e.g. pressed Tab).

        Returns:
            FeedbackSignal with ACCEPT type, or None if nothing is tracked.
        """
        if self._pending is None:
            return None

        candidate, context, shown_ts = self._pending
        self._pending = None

        from leapflow.copilot.types import FeedbackSignal as FS
        from leapflow.copilot.types import FeedbackType as FT

        return FS(
            feedback_type=FT.ACCEPT,
            candidate=candidate,
            response_latency_ms=int((_time.time() - shown_ts) * 1000),
            context_at_feedback=context,
            timestamp=_time.time(),
        )

    def on_explicit_reject(self) -> Optional[FeedbackSignal]:
        """User explicitly rejected the suggestion (e.g. pressed Esc).

        Returns:
            FeedbackSignal with EXPLICIT_REJECT type, or None.
        """
        if self._pending is None:
            return None

        candidate, context, shown_ts = self._pending
        self._pending = None

        from leapflow.copilot.types import FeedbackSignal as FS
        from leapflow.copilot.types import FeedbackType as FT

        return FS(
            feedback_type=FT.EXPLICIT_REJECT,
            candidate=candidate,
            response_latency_ms=int((_time.time() - shown_ts) * 1000),
            context_at_feedback=context,
            timestamp=_time.time(),
        )

    def on_next_action(
        self, action: str, context: ContextState
    ) -> Optional[FeedbackSignal]:
        """User performed an action while a suggestion was displayed.

        Semantics:
        - If the action matches the suggestion → IGNORE (user did it manually,
          without using the accept shortcut).
        - If the action differs → CORRECT (user chose a different path).

        Args:
            action: Description of the action the user actually performed.
            context: Current context state at the time of action.

        Returns:
            FeedbackSignal (CORRECT or IGNORE), or None.
        """
        if self._pending is None:
            return None

        candidate, _ctx, shown_ts = self._pending
        self._pending = None

        from leapflow.copilot.types import FeedbackSignal as FS
        from leapflow.copilot.types import FeedbackType as FT

        if action == candidate.action_description:
            # User did the same thing but without accept shortcut → IGNORE
            return FS(
                feedback_type=FT.IGNORE,
                candidate=candidate,
                actual_action=action,
                response_latency_ms=int((_time.time() - shown_ts) * 1000),
                context_at_feedback=context,
                timestamp=_time.time(),
            )

        # User did something different → CORRECT
        return FS(
            feedback_type=FT.CORRECT,
            candidate=candidate,
            actual_action=action,
            response_latency_ms=int((_time.time() - shown_ts) * 1000),
            context_at_feedback=context,
            timestamp=_time.time(),
        )

    def timeout_pending(self) -> Optional[FeedbackSignal]:
        """Mark a pending suggestion as IGNORE due to timeout.

        Called when max_idle_ms elapsed without any user response.
        """
        if self._pending is None:
            return None

        candidate, context, shown_ts = self._pending
        self._pending = None

        from leapflow.copilot.types import FeedbackSignal as FS
        from leapflow.copilot.types import FeedbackType as FT

        return FS(
            feedback_type=FT.IGNORE,
            candidate=candidate,
            response_latency_ms=int((_time.time() - shown_ts) * 1000),
            context_at_feedback=context,
            timestamp=_time.time(),
        )


# ────────────────────────────────────────────────────────────────────────────
# EvolutionLoop — feedback-driven model weight updates
# ────────────────────────────────────────────────────────────────────────────


class EvolutionLoop:
    """反馈驱动的模型演化循环 — 将 FeedbackSignal 转化为预测模型权重更新。

    EMA 置信度更新 + 信任梯度延伸。与 Loop γ 的"执行即学习"理念一致。

    Reward mapping:
        ACCEPT          → +config.accept_boost (default +1.0)
        IGNORE          → +config.ignore_decay (default -0.1)
        CORRECT         → -0.5 (strong negative)
        EXPLICIT_REJECT → -1.0 (strongest negative)
    """

    _CORRECT_PENALTY: float = -0.5
    _REJECT_PENALTY: float = -1.0

    def __init__(
        self,
        config: CopilotConfig,
        layers: List[PredictorLayer],
    ) -> None:
        self._config = config
        self._layers = layers

        # Per context_hash EMA confidence scores
        self._confidence_scores: Dict[str, float] = {}

        # Aggregate statistics
        self._accept_count: int = 0
        self._total_count: int = 0

    async def process_feedback(self, signal: FeedbackSignal) -> None:
        """Process a feedback signal: update EMA and broadcast to layers.

        Args:
            signal: Structured feedback from FeedbackCollector.
        """
        from leapflow.copilot.types import FeedbackType as FT

        self._total_count += 1

        # Determine reward
        reward = self._reward_for(signal.feedback_type)

        # Track accepts
        if signal.feedback_type == FT.ACCEPT:
            self._accept_count += 1

        # Update EMA confidence for the context
        ctx_hash = signal.candidate.context_hash
        self._update_ema(ctx_hash, reward)

        # Broadcast to all predictor layers
        for layer in self._layers:
            try:
                await layer.on_feedback(signal)
            except Exception:
                logger.warning(
                    "Layer %s failed to process feedback",
                    layer.layer_id,
                    exc_info=True,
                )

        logger.debug(
            "Feedback processed: type=%s, ctx=%s, reward=%.2f, new_conf=%.3f",
            signal.feedback_type.value,
            ctx_hash,
            reward,
            self._confidence_scores.get(ctx_hash, 0.0),
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate evolution statistics.

        Returns:
            Dict with keys: total_predictions, accept_count, accept_rate,
            avg_confidence, num_contexts_tracked.
        """
        accept_rate = (
            self._accept_count / self._total_count if self._total_count > 0 else 0.0
        )
        scores = list(self._confidence_scores.values())
        avg_confidence = sum(scores) / len(scores) if scores else 0.0

        return {
            "total_predictions": self._total_count,
            "accept_count": self._accept_count,
            "accept_rate": round(accept_rate, 4),
            "avg_confidence": round(avg_confidence, 4),
            "num_contexts_tracked": len(self._confidence_scores),
        }

    def get_confidence(self, context_hash: str) -> float:
        """Get current EMA confidence for a specific context.

        Returns 0.5 (neutral) if context has never been seen.
        """
        return self._confidence_scores.get(context_hash, 0.5)

    # ── Internal ──────────────────────────────────────────────────────────

    def _reward_for(self, feedback_type: FeedbackType) -> float:
        """Map feedback type to numeric reward signal."""
        from leapflow.copilot.types import FeedbackType as FT

        if feedback_type == FT.ACCEPT:
            return self._config.accept_boost
        elif feedback_type == FT.IGNORE:
            return self._config.ignore_decay
        elif feedback_type == FT.CORRECT:
            return self._CORRECT_PENALTY
        elif feedback_type == FT.EXPLICIT_REJECT:
            return self._REJECT_PENALTY
        return 0.0

    def _update_ema(self, key: str, reward: float) -> float:
        """Update EMA confidence for a given context hash.

        Formula: new = alpha * reward + (1 - alpha) * old
        Clamped to [0.0, 1.0].

        Returns:
            Updated confidence value.
        """
        alpha = self._config.ema_alpha
        old = self._confidence_scores.get(key, 0.5)
        new_val = alpha * reward + (1 - alpha) * old
        # Clamp to valid range
        new_val = max(0.0, min(1.0, new_val))
        self._confidence_scores[key] = new_val
        return new_val
