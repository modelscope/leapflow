# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for signal_fusion — wait_classifier, action_agent, and quality modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import pytest

from leapflow.signal_fusion.action_agent import ActionFusionAgent, _action_types_compatible
from leapflow.signal_fusion.protocol import FusionContext
from leapflow.signal_fusion.quality import FusionQuality, QualityLevel
from leapflow.signal_fusion.types import AtomicAction, FusionMode, SilentPeriodClass
from leapflow.signal_fusion.wait_classifier import GapContext, WaitPeriodClassifier


# ═══════════════════════════════════════════════════════════════════════
# Lightweight stubs for domain types (no network, no imports from heavy modules)
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class _VisualAction:
    action: str = "click"
    target: str = "button"
    detail: str = ""
    confidence: float = 0.8
    evidence: str = ""
    frame_ref_a: str = ""
    frame_ref_b: str = ""
    timestamp: float = 0.0


@dataclass(frozen=True)
class _SystemEvent:
    event_type: str = "mouse.click"
    source: str = "com.test.app"
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    platform_hint: str = ""
    priority: int = 0


# ═══════════════════════════════════════════════════════════════════════
# 1. WaitPeriodClassifier — positive / negative / boundary
# ═══════════════════════════════════════════════════════════════════════


class TestWaitClassifierPositive:
    def test_ai_generating_detected(self) -> None:
        """Gap on an AI tool URL after a submit action → AI_GENERATING."""
        clf = WaitPeriodClassifier(ai_wait_threshold=3.0)
        ctx = GapContext(
            current_app_url="https://chat.openai.com/c/abc",
            last_action_type="submit",
        )
        result = clf.classify(5.0, ctx)
        assert result == SilentPeriodClass.AI_GENERATING

    def test_user_idle_detected(self) -> None:
        clf = WaitPeriodClassifier(idle_threshold=30.0)
        ctx = GapContext()
        result = clf.classify(45.0, ctx)
        assert result == SilentPeriodClass.USER_IDLE

    def test_loading_with_indicator(self) -> None:
        clf = WaitPeriodClassifier(loading_threshold=2.0)
        ctx = GapContext(has_loading_indicator=True)
        result = clf.classify(3.0, ctx)
        assert result == SilentPeriodClass.LOADING


class TestWaitClassifierNegative:
    def test_short_gap_is_normal_pause(self) -> None:
        """Gap below loading_threshold → NORMAL_PAUSE regardless of context."""
        clf = WaitPeriodClassifier(loading_threshold=2.0)
        ctx = GapContext(
            current_app_url="https://chat.openai.com/c/abc",
            last_action_type="submit",
        )
        result = clf.classify(1.0, ctx)
        assert result == SilentPeriodClass.NORMAL_PAUSE

    def test_ai_tool_without_submit_not_generating(self) -> None:
        """On AI URL but last action is 'scroll' (not submit) → not AI_GENERATING."""
        clf = WaitPeriodClassifier(ai_wait_threshold=3.0)
        ctx = GapContext(
            current_app_url="https://claude.ai/chat",
            last_action_type="scroll",
        )
        result = clf.classify(5.0, ctx)
        assert result != SilentPeriodClass.AI_GENERATING


class TestWaitClassifierBoundary:
    def test_exactly_at_loading_threshold(self) -> None:
        clf = WaitPeriodClassifier(loading_threshold=2.0)
        ctx = GapContext(has_loading_indicator=True)
        result = clf.classify(2.0, ctx)
        assert result == SilentPeriodClass.LOADING

    def test_just_below_loading_threshold(self) -> None:
        clf = WaitPeriodClassifier(loading_threshold=2.0)
        ctx = GapContext(has_loading_indicator=True)
        result = clf.classify(1.99, ctx)
        assert result == SilentPeriodClass.NORMAL_PAUSE

    def test_register_custom_tool_extends_detection(self) -> None:
        clf = WaitPeriodClassifier(ai_wait_threshold=3.0)
        ctx = GapContext(
            current_app_url="https://myai.example.com/chat",
            last_action_type="submit",
        )
        # Before registration → unknown
        result_before = clf.classify(5.0, ctx)
        assert result_before != SilentPeriodClass.AI_GENERATING

        clf.register("myai.example.com", "myai")
        result_after = clf.classify(5.0, ctx)
        assert result_after == SilentPeriodClass.AI_GENERATING

    def test_unknown_wait_for_medium_gap_no_context(self) -> None:
        """A gap between loading and idle thresholds with no context → UNKNOWN_WAIT."""
        clf = WaitPeriodClassifier(loading_threshold=2.0, idle_threshold=30.0)
        ctx = GapContext()  # no URL, no indicator, no frame change
        result = clf.classify(10.0, ctx)
        assert result == SilentPeriodClass.UNKNOWN_WAIT


# ═══════════════════════════════════════════════════════════════════════
# 2. ActionFusionAgent — ordering / selection / empty input
# ═══════════════════════════════════════════════════════════════════════


class TestActionFusionEmpty:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_result(self) -> None:
        agent = ActionFusionAgent()
        ctx = FusionContext()
        result = await agent.fuse(ctx)
        assert result.atomic_actions == []

    @pytest.mark.asyncio
    async def test_visual_only_no_events(self) -> None:
        agent = ActionFusionAgent()
        va = _VisualAction(action="click", target="btn", confidence=0.9)
        ctx = FusionContext(visual_actions=[va])
        result = await agent.fuse(ctx)
        assert len(result.atomic_actions) == 1
        atom = result.atomic_actions[0]
        assert atom.fusion_mode == FusionMode.VISUAL_PRIMARY
        assert atom.confidence == 0.9
        assert "visual" in atom.source_signals

    @pytest.mark.asyncio
    async def test_events_only_no_visual(self) -> None:
        agent = ActionFusionAgent(event_only_confidence=0.6)
        ev = _SystemEvent(event_type="mouse.click", timestamp=1.0, source="com.app")
        ctx = FusionContext(system_events=[ev])
        result = await agent.fuse(ctx)
        assert len(result.atomic_actions) == 1
        atom = result.atomic_actions[0]
        assert atom.fusion_mode == FusionMode.EVENT_PRIMARY
        assert atom.confidence == 0.6


class TestActionFusionOrdering:
    @pytest.mark.asyncio
    async def test_output_sorted_by_timestamp(self) -> None:
        agent = ActionFusionAgent()
        ev1 = _SystemEvent(event_type="key.press", timestamp=5.0, source="com.app")
        ev2 = _SystemEvent(event_type="mouse.click", timestamp=1.0, source="com.app")
        ctx = FusionContext(system_events=[ev1, ev2])
        result = await agent.fuse(ctx)
        timestamps = [a.timestamp for a in result.atomic_actions]
        assert timestamps == sorted(timestamps)


class TestActionFusionCorroboration:
    @pytest.mark.asyncio
    async def test_matching_pair_uses_full_mode(self) -> None:
        agent = ActionFusionAgent(time_tolerance=2.0, corroboration_boost=1.2)
        va = _VisualAction(action="click", confidence=0.8)
        ev = _SystemEvent(event_type="mouse.click", timestamp=0.5, source="com.app")
        ctx = FusionContext(visual_actions=[va], system_events=[ev])
        result = await agent.fuse(ctx)
        assert len(result.atomic_actions) == 1
        atom = result.atomic_actions[0]
        assert atom.fusion_mode == FusionMode.FULL
        assert atom.confidence == pytest.approx(min(1.0, 0.8 * 1.2))
        assert set(atom.source_signals) == {"visual", "event"}


class TestActionTypesCompatible:
    def test_visual_substring_in_event(self) -> None:
        assert _action_types_compatible("click", "mouse.click")

    def test_event_suffix_in_visual(self) -> None:
        assert _action_types_compatible("click_button", "ui.click")

    def test_no_match(self) -> None:
        assert not _action_types_compatible("drag", "mouse.click")


# ═══════════════════════════════════════════════════════════════════════
# 3. FusionQuality — scoring and malformed signal handling
# ═══════════════════════════════════════════════════════════════════════


class TestFusionQualityEmpty:
    def test_no_actions_produces_warning(self) -> None:
        q = FusionQuality.from_actions([])
        assert "no_actions_produced" in q.warnings
        assert q.avg_action_confidence == 0.0
        assert q.level == QualityLevel.LOW


class TestFusionQualityScoring:
    def test_high_confidence_actions_yield_high_quality(self) -> None:
        actions = [
            AtomicAction(action="click", target="x", detail="", timestamp=0.0,
                         confidence=0.95, source_signals=["visual", "event"],
                         fusion_mode=FusionMode.FULL),
            AtomicAction(action="type", target="y", detail="", timestamp=1.0,
                         confidence=0.90, source_signals=["visual", "event"],
                         fusion_mode=FusionMode.FULL),
        ]
        q = FusionQuality.from_actions(actions)
        assert q.level == QualityLevel.HIGH
        assert q.avg_action_confidence > 0.8
        assert q.high_confidence_ratio > 0.7

    def test_low_confidence_actions_yield_low_quality(self) -> None:
        actions = [
            AtomicAction(action="a", target="", detail="", timestamp=0.0,
                         confidence=0.2, source_signals=["visual"],
                         fusion_mode=FusionMode.VISUAL_PRIMARY),
            AtomicAction(action="b", target="", detail="", timestamp=1.0,
                         confidence=0.3, source_signals=["event"],
                         fusion_mode=FusionMode.EVENT_PRIMARY),
        ]
        q = FusionQuality.from_actions(actions)
        assert q.level == QualityLevel.LOW
        assert "majority_low_confidence" in q.warnings
        assert "overall_low_quality" in q.warnings


class TestFusionQualityChannelCoverage:
    def test_channel_unavailable_warning(self) -> None:
        actions = [
            AtomicAction(action="a", target="", detail="", timestamp=0.0,
                         confidence=0.7, source_signals=["event"],
                         fusion_mode=FusionMode.EVENT_PRIMARY),
        ]
        q = FusionQuality.from_actions(actions, visual_available=False)
        assert "visual_channel_unavailable" in q.warnings
        assert "visual" not in q.channel_coverage

    def test_both_channels_available_coverage_computed(self) -> None:
        actions = [
            AtomicAction(action="a", target="", detail="", timestamp=0.0,
                         confidence=0.85, source_signals=["visual", "event"],
                         fusion_mode=FusionMode.FULL),
        ]
        q = FusionQuality.from_actions(actions, visual_available=True, events_available=True)
        assert "visual" in q.channel_coverage
        assert "event" in q.channel_coverage
        assert q.channel_coverage["visual"] == pytest.approx(1.0)
        assert q.channel_coverage["event"] == pytest.approx(1.0)


class TestFusionQualityDominantMode:
    def test_dominant_mode_reflects_majority(self) -> None:
        actions = [
            AtomicAction(action="a", target="", detail="", timestamp=0.0,
                         confidence=0.7, source_signals=["visual"],
                         fusion_mode=FusionMode.VISUAL_PRIMARY),
            AtomicAction(action="b", target="", detail="", timestamp=1.0,
                         confidence=0.7, source_signals=["visual"],
                         fusion_mode=FusionMode.VISUAL_PRIMARY),
            AtomicAction(action="c", target="", detail="", timestamp=2.0,
                         confidence=0.9, source_signals=["visual", "event"],
                         fusion_mode=FusionMode.FULL),
        ]
        q = FusionQuality.from_actions(actions)
        assert q.dominant_fusion_mode == FusionMode.VISUAL_PRIMARY.value


class TestActionFusionRelativeTimestamp:
    """Behavior tests for relative-distance matching after the timestamp fix."""

    @pytest.mark.asyncio
    async def test_nearest_relative_event_selected(self) -> None:
        """Multiple compatible events, nonzero visual timestamp → nearest wins."""
        agent = ActionFusionAgent(time_tolerance=2.0)
        va = _VisualAction(action="click", confidence=0.8, timestamp=10.0)
        ev_far = _SystemEvent(event_type="mouse.click", timestamp=8.5, source="com.app")
        ev_near = _SystemEvent(event_type="mouse.click", timestamp=10.3, source="com.app")
        ctx = FusionContext(visual_actions=[va], system_events=[ev_far, ev_near])
        result = await agent.fuse(ctx)
        # Both events are within tolerance, but ev_near (dist=0.3) is closer than
        # ev_far (dist=1.5) relative to va.timestamp=10.0.
        full_atoms = [a for a in result.atomic_actions if a.fusion_mode == FusionMode.FULL]
        assert len(full_atoms) == 1
        assert full_atoms[0].timestamp == 10.3
        # The unmatched far event appears as EVENT_PRIMARY
        event_only = [a for a in result.atomic_actions if a.fusion_mode == FusionMode.EVENT_PRIMARY]
        assert len(event_only) == 1
        assert event_only[0].timestamp == 8.5

    @pytest.mark.asyncio
    async def test_compatible_event_outside_tolerance_not_fused(self) -> None:
        """Compatible event beyond tolerance → visual stays VISUAL_PRIMARY."""
        agent = ActionFusionAgent(time_tolerance=1.0)
        va = _VisualAction(action="click", confidence=0.9, timestamp=5.0)
        ev = _SystemEvent(event_type="mouse.click", timestamp=7.5, source="com.app")
        ctx = FusionContext(visual_actions=[va], system_events=[ev])
        result = await agent.fuse(ctx)
        assert len(result.atomic_actions) == 2
        modes = {a.fusion_mode for a in result.atomic_actions}
        assert FusionMode.FULL not in modes
        assert FusionMode.VISUAL_PRIMARY in modes
        assert FusionMode.EVENT_PRIMARY in modes

    @pytest.mark.asyncio
    async def test_multi_visual_multi_event_each_consumed_once(self) -> None:
        """Each event consumed at most once; matching is nearest and deterministic."""
        agent = ActionFusionAgent(time_tolerance=2.0, corroboration_boost=1.0)
        va1 = _VisualAction(action="click", confidence=0.8, timestamp=1.0)
        va2 = _VisualAction(action="click", confidence=0.8, timestamp=5.0)
        ev1 = _SystemEvent(event_type="mouse.click", timestamp=1.2, source="com.app")
        ev2 = _SystemEvent(event_type="mouse.click", timestamp=4.8, source="com.app")
        ctx = FusionContext(
            visual_actions=[va1, va2],
            system_events=[ev1, ev2],
        )
        result = await agent.fuse(ctx)
        full_atoms = [a for a in result.atomic_actions if a.fusion_mode == FusionMode.FULL]
        # Both visual actions should match their respective nearest event
        assert len(full_atoms) == 2
        ts_set = {a.timestamp for a in full_atoms}
        assert ts_set == {1.2, 4.8}
        # No leftover event-only atoms since both events were consumed
        event_only = [a for a in result.atomic_actions if a.fusion_mode == FusionMode.EVENT_PRIMARY]
        assert len(event_only) == 0


class TestFusionQualityLevel:
    def test_medium_quality_boundary(self) -> None:
        """avg_confidence > 0.6 but not meeting HIGH criteria → MEDIUM."""
        actions = [
            AtomicAction(action="a", target="", detail="", timestamp=0.0,
                         confidence=0.65, source_signals=["visual"],
                         fusion_mode=FusionMode.VISUAL_PRIMARY),
        ]
        q = FusionQuality.from_actions(actions)
        assert q.level == QualityLevel.MEDIUM
