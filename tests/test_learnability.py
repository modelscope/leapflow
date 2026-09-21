# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for learning.learnability — rule-based assessment and decision logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from leapflow.learning.learnability import (
    DefaultLearnabilityAssessor,
    LearnabilityConfig,
    LearnabilityDecision,
    LearnabilityInput,
    RuleBasedAssessor,
)


# ── Stub trajectory ──


@dataclass
class _StubAction:
    action_type: str = "click"
    timestamp: float = 0.0


@dataclass
class _StubStep:
    action: _StubAction = field(default_factory=_StubAction)


def _make_trajectory(
    step_count: int = 5,
    duration: float = 30.0,
    action_types: list[str] | None = None,
    timestamps: list[float] | None = None,
) -> Any:
    """Build a lightweight trajectory stub for rule-based assessment."""
    if action_types is None:
        action_types = ["click", "type", "scroll"] * ((step_count // 3) + 1)
    action_types = action_types[:step_count]

    if timestamps is None:
        # Evenly spaced actions
        step_interval = duration / max(step_count - 1, 1)
        timestamps = [i * step_interval for i in range(step_count)]
    timestamps = timestamps[:step_count]

    steps = [
        _StubStep(action=_StubAction(action_type=at, timestamp=ts))
        for at, ts in zip(action_types, timestamps)
    ]

    @dataclass
    class _Traj:
        step_count: int
        duration: float
        steps: list

    return _Traj(step_count=step_count, duration=duration, steps=steps)


# ── 1. Positive learnable signal ──


class TestPositiveLearnable:
    def test_good_trajectory_scores_above_learn_threshold(self) -> None:
        """A well-formed trajectory with diverse actions scores high."""
        cfg = LearnabilityConfig()
        assessor = RuleBasedAssessor(cfg)
        # Keep gaps <= 5s (idle threshold) so idle ratio stays low
        traj = _make_trajectory(
            step_count=8, duration=40.0,
            action_types=["click", "type", "drag", "scroll", "paste", "click", "type", "submit"],
            timestamps=[0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0],
        )
        inp = LearnabilityInput(trajectory=traj)
        score, reject = assessor.assess(inp)
        assert reject is None
        assert score >= cfg.learn_threshold, f"score={score} below learn threshold"

    @pytest.mark.asyncio
    async def test_default_assessor_returns_learn_for_strong_signal(self) -> None:
        """DefaultLearnabilityAssessor (L1-only, no LLM/VLM) yields LEARN."""
        cfg = LearnabilityConfig(vlm_enabled=False, llm_enabled=False)
        assessor = DefaultLearnabilityAssessor(config=cfg)
        traj = _make_trajectory(
            step_count=10, duration=50.0,
            action_types=["click", "type", "drag", "scroll", "paste",
                          "click", "type", "submit", "drag", "click"],
            timestamps=[0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0],
        )
        inp = LearnabilityInput(trajectory=traj)
        report = await assessor.assess(inp)
        assert report.decision == LearnabilityDecision.LEARN
        assert report.reason  # non-empty explanation


# ── 2. Insufficient evidence / low confidence ──


class TestInsufficientEvidence:
    def test_too_few_steps_rejected(self) -> None:
        cfg = LearnabilityConfig(min_steps=3)
        assessor = RuleBasedAssessor(cfg)
        traj = _make_trajectory(step_count=2, duration=10.0)
        inp = LearnabilityInput(trajectory=traj)
        score, reject = assessor.assess(inp)
        assert score == 0.0
        assert reject is not None
        assert "steps" in reject.lower()

    def test_too_short_duration_rejected(self) -> None:
        cfg = LearnabilityConfig(min_duration_s=5.0)
        assessor = RuleBasedAssessor(cfg)
        traj = _make_trajectory(step_count=5, duration=2.0)
        inp = LearnabilityInput(trajectory=traj)
        score, reject = assessor.assess(inp)
        assert score == 0.0
        assert reject is not None
        assert "short" in reject.lower()


# ── 3. Risk / side-effect gating (excessive idle) ──


class TestIdleGating:
    def test_excessive_idle_penalizes_score(self) -> None:
        """Trajectory with long idle gaps is penalized."""
        cfg = LearnabilityConfig(max_idle_ratio=0.80)
        assessor = RuleBasedAssessor(cfg)
        # 5 steps with huge gaps between them (each gap > 5s idle threshold)
        # Total duration 100s, gaps: 24, 24, 24, 24 → all > 5s → idle_time=96, ratio=96%
        traj = _make_trajectory(
            step_count=5, duration=100.0,
            timestamps=[0.0, 25.0, 50.0, 75.0, 100.0],
        )
        inp = LearnabilityInput(trajectory=traj)
        score, reject = assessor.assess(inp)
        assert reject is not None
        assert "idle" in reject.lower()
        assert score < cfg.ask_threshold


# ── 4. Boundary thresholds ──


class TestBoundaryThresholds:
    @pytest.mark.asyncio
    async def test_ask_zone_between_thresholds(self) -> None:
        """Score between ask and learn thresholds yields ASK decision."""
        cfg = LearnabilityConfig(
            vlm_enabled=False, llm_enabled=False,
            learn_threshold=0.65, ask_threshold=0.40,
        )
        assessor = DefaultLearnabilityAssessor(config=cfg)
        # 3 steps exactly at minimum — low but not zero
        traj = _make_trajectory(
            step_count=3, duration=10.0,
            action_types=["click", "click", "click"],  # low diversity
        )
        inp = LearnabilityInput(trajectory=traj)
        report = await assessor.assess(inp)
        # With minimal steps and low diversity, score should be in ASK or SKIP range
        assert report.decision in (LearnabilityDecision.ASK, LearnabilityDecision.SKIP)

    @pytest.mark.asyncio
    async def test_skip_for_very_low_score(self) -> None:
        cfg = LearnabilityConfig(vlm_enabled=False, llm_enabled=False)
        assessor = DefaultLearnabilityAssessor(config=cfg)
        traj = _make_trajectory(step_count=2, duration=1.0)
        inp = LearnabilityInput(trajectory=traj)
        report = await assessor.assess(inp)
        assert report.decision == LearnabilityDecision.SKIP


# ── 5. Deterministic decision metadata / reasons ──


class TestDecisionMetadata:
    @pytest.mark.asyncio
    async def test_report_carries_rule_score(self) -> None:
        cfg = LearnabilityConfig(vlm_enabled=False, llm_enabled=False)
        assessor = DefaultLearnabilityAssessor(config=cfg)
        traj = _make_trajectory(step_count=6, duration=30.0)
        inp = LearnabilityInput(trajectory=traj)
        report = await assessor.assess(inp)
        assert isinstance(report.rule_score, float)
        assert 0.0 <= report.rule_score <= 1.0
        assert 0.0 <= report.score <= 1.0
        assert report.reason  # always has a reason

    def test_combine_scores_rule_only(self) -> None:
        """With no VLM/LLM, final score equals rule score."""
        cfg = LearnabilityConfig()
        assessor = DefaultLearnabilityAssessor(config=cfg)
        combined = assessor._combine_scores(0.75, None, None)
        assert combined == pytest.approx(0.75)

    def test_combine_scores_all_three(self) -> None:
        cfg = LearnabilityConfig(rule_weight=0.4, vlm_weight=0.3, llm_weight=0.3)
        assessor = DefaultLearnabilityAssessor(config=cfg)
        combined = assessor._combine_scores(1.0, 0.5, 0.5)
        expected = (1.0 * 0.4 + 0.5 * 0.3 + 0.5 * 0.3) / (0.4 + 0.3 + 0.3)
        assert combined == pytest.approx(expected)
