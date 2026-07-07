"""AttentionTuner — bridges world model learning signals to attention filter parameters.

Provides the meta-cognitive feedback loop for the attention mechanism:
curiosity signals and prediction accuracy dynamically expand or contract
the recording scope at runtime, without modifying filter chain structure.

Three feedback channels:
  1. High curiosity → expand app scope (DomainWhitelistFilter sees the app)
  2. Persistent curiosity → promote perception depth (PerceptualFieldFilter level)
  3. Low average delta → contract perception depth (mastered domain)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from leapflow.recording.attention import RecordingContext
    from leapflow.recording.perceptual_field import PerceptualFieldFilter
    from leapflow.world_model.curiosity import CuriosityScore
    from leapflow.world_model.prediction import PredictionOutcome

logger = logging.getLogger(__name__)


class AttentionTuner:
    """Bridges world model learning signals to attention filter parameters.

    Acts as a lightweight "knob controller" that holds references to
    ``RecordingContext`` and optionally ``PerceptualFieldFilter``, adjusting
    their parameters based on curiosity signals and prediction accuracy.
    """

    def __init__(
        self,
        context: "RecordingContext",
        *,
        perceptual_filter: Optional["PerceptualFieldFilter"] = None,
        curiosity_expand_threshold: float = 0.7,
        accuracy_contract_threshold: float = 0.1,
        max_dynamic_rules: int = 20,
        persistent_curiosity_count: int = 3,
    ) -> None:
        self._context = context
        self._perceptual_filter = perceptual_filter
        self._expand_threshold = curiosity_expand_threshold
        self._contract_threshold = accuracy_contract_threshold
        self._max_dynamic_rules = max_dynamic_rules
        self._persistent_count = persistent_curiosity_count
        self._curiosity_hits: Dict[str, int] = defaultdict(int)
        self._dynamic_rule_count = 0

    def on_curiosity_signal(
        self,
        score: "CuriosityScore",
        outcome: "PredictionOutcome",
    ) -> None:
        """React to a curiosity signal by expanding attention scope/depth."""
        app_id = _extract_app(outcome)
        if not app_id:
            return

        if score.total >= self._expand_threshold:
            self._context.expand_app_scope(app_id)
            self._curiosity_hits[app_id] += 1
            logger.debug(
                "attention_tuner.expand app=%s hits=%d",
                app_id, self._curiosity_hits[app_id],
            )

            if (
                self._perceptual_filter is not None
                and self._curiosity_hits[app_id] >= self._persistent_count
                and self._dynamic_rule_count < self._max_dynamic_rules
            ):
                self._promote_perception_depth(app_id)

    def boost_curiosity_domains(self, app_ids: "set[str]") -> None:
        """Expand attention scope for apps that triggered high curiosity.

        Called at session end with the accumulated set of apps from
        ``ActiveLearningObserver.drain_high_curiosity_apps()``. Ensures
        these apps will be observed in the next session.
        """
        for app_id in app_ids:
            if app_id and app_id != "unknown":
                self._context.expand_app_scope(app_id)
                self._curiosity_hits[app_id] += 1
                logger.debug("attention_tuner.boost_domain app=%s", app_id)

    def on_session_stats(self, app_deltas: Dict[str, float]) -> None:
        """Contract perception depth for apps with low average delta (mastered)."""
        if self._perceptual_filter is None:
            return

        for app_id, avg_delta in app_deltas.items():
            if avg_delta < self._contract_threshold:
                self._demote_perception_depth(app_id)

    def _upsert_perception_rule(
        self,
        app_id: str,
        level: "Any",
        source: str,
        priority: int,
    ) -> None:
        """Insert or replace a dynamic perception rule for an app.

        Removes any prior tuner-managed rule (source in ``_TUNER_SOURCES``)
        for the same app before adding the new one, preventing accumulation.
        """
        from leapflow.domain.perception import FieldRule

        if self._perceptual_filter is None:
            return
        if self._dynamic_rule_count >= self._max_dynamic_rules:
            return

        policy = self._perceptual_filter.policy
        existing = policy.get_all_rules()
        removed = [
            r for r in existing
            if r.app_pattern == app_id and r.source in self._TUNER_SOURCES
        ]
        if removed:
            kept = [r for r in existing if r not in removed]
            policy._rules = kept
            self._dynamic_rule_count -= len(removed)

        rule = FieldRule(
            app_pattern=app_id,
            context_pattern="*",
            level=level,
            source=source,
            priority=priority,
        )
        policy.add_rule(rule)
        self._dynamic_rule_count += 1
        logger.debug("attention_tuner.upsert app=%s → %s (source=%s)", app_id, level, source)

    _TUNER_SOURCES = frozenset({"curiosity", "mastered"})

    def _promote_perception_depth(self, app_id: str) -> None:
        """Set FULL perception rule for a persistently curious app."""
        from leapflow.domain.perception import PerceptionLevel
        self._upsert_perception_rule(app_id, PerceptionLevel.FULL, "curiosity", 80)

    def _demote_perception_depth(self, app_id: str) -> None:
        """Set STRUCTURAL perception rule for a mastered app."""
        from leapflow.domain.perception import PerceptionLevel
        self._upsert_perception_rule(app_id, PerceptionLevel.STRUCTURAL, "mastered", 60)


def _extract_app(outcome: Any) -> str:
    """Extract app bundle ID from a PredictionOutcome's pre_snapshot."""
    pre = getattr(outcome, "pre_snapshot", None)
    if pre is not None:
        return getattr(pre, "app_bundle_id", "") or ""
    return ""
