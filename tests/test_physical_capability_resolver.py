# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the physical capability resolver's four-value adjudication.

The resolver turns physical outcome observations into the same absorb / rebind
/ acquire / escalate verdict space the software evolution engine uses.  These
tests exercise the four gap classifications, each verdict path, the evidence
threshold that refuses to adjudicate on too few samples, and the rebind
candidate lookup against a mock inference registry.
"""

from __future__ import annotations

from typing import Any

import pytest

from leapflow.domain.adaptation_verdict import ADAPTATION_ACTIONS
from leapflow.plugins.physical_capability_resolver import (
    PhysicalCapabilityGap,
    PhysicalCapabilityResolver,
    PhysicalGapType,
)


# ── Test doubles ──


class _FakeStrategy:
    """Inference strategy that declares the affordances it can serve."""

    def __init__(self, strategy_id: str, affordances: tuple[str, ...] = ()) -> None:
        self.strategy_id = strategy_id
        self.affordances = affordances


class _BareStrategy:
    """Strategy that declares no affordances (exercises the id-substring path)."""

    def __init__(self, strategy_id: str) -> None:
        self.strategy_id = strategy_id


class FakeInferenceRegistry:
    """Minimal stand-in for InferenceStrategyRegistry."""

    def __init__(self, strategies: tuple[Any, ...] = ()) -> None:
        self._strategies = {s.strategy_id: s for s in strategies}

    def list_strategies(self) -> list[dict[str, Any]]:
        return [{"strategy_id": sid} for sid in self._strategies]

    def get(self, strategy_id: str) -> Any:
        return self._strategies.get(strategy_id)


class FakeEvidenceStore:
    """Placeholder evidence store — the resolver holds but does not query it here."""


def _precision_gap(**overrides: Any) -> PhysicalCapabilityGap:
    kwargs: dict[str, Any] = {
        "gap_type": PhysicalGapType.INSUFFICIENT_PRECISION.value,
        "device_id": "robot.arm",
        "affordance": "grasp",
        "success_rate": 0.3,
        "sample_count": 10,
    }
    kwargs.update(overrides)
    return PhysicalCapabilityGap(**kwargs)


def _resolver(**overrides: Any) -> PhysicalCapabilityResolver:
    kwargs: dict[str, Any] = {
        "inference_registry": None,
        "evidence_store": FakeEvidenceStore(),
        "precision_threshold": 0.7,
        "min_samples": 5,
    }
    kwargs.update(overrides)
    return PhysicalCapabilityResolver(**kwargs)


# ── Gap detection ──


async def test_detect_insufficient_precision() -> None:
    resolver = _resolver()
    outcome = {
        "devices": {
            "robot.arm": {
                "total_operations": 10,
                "success_rate": 0.4,
                "affordance": "grasp",
                "recent_failures": ["position_deviation > tolerance"],
            }
        }
    }
    gaps = await resolver.detect_gaps(outcome)
    assert len(gaps) == 1
    assert gaps[0].gap_type == PhysicalGapType.INSUFFICIENT_PRECISION.value
    assert gaps[0].sample_count == 10
    assert gaps[0].success_rate == pytest.approx(0.4)


async def test_detect_no_gap_when_performing_well() -> None:
    resolver = _resolver()
    outcome = {"devices": {"robot.arm": {"total_operations": 10, "success_rate": 0.95}}}
    assert await resolver.detect_gaps(outcome) == ()


async def test_detect_missing_device() -> None:
    resolver = _resolver()
    outcome = {"devices": {"robot.arm": {"connected": False, "affordance": "grasp"}}}
    gaps = await resolver.detect_gaps(outcome)
    assert len(gaps) == 1
    assert gaps[0].gap_type == PhysicalGapType.MISSING_DEVICE.value


async def test_detect_hardware_degraded() -> None:
    resolver = _resolver()
    outcome = {
        "devices": {
            "robot.arm": {"health": "degraded", "total_operations": 4, "success_rate": 0.1}
        }
    }
    gaps = await resolver.detect_gaps(outcome)
    assert len(gaps) == 1
    assert gaps[0].gap_type == PhysicalGapType.HARDWARE_DEGRADED.value


async def test_detect_missing_skill_from_unsupported_affordance() -> None:
    resolver = _resolver()
    outcome = {"devices": {"robot.arm": {"unsupported_affordances": ["pour"]}}}
    gaps = await resolver.detect_gaps(outcome)
    assert len(gaps) == 1
    assert gaps[0].gap_type == PhysicalGapType.MISSING_SKILL.value
    assert gaps[0].affordance == "pour"


async def test_detect_ignores_malformed_payload() -> None:
    resolver = _resolver()
    assert await resolver.detect_gaps({}) == ()
    assert await resolver.detect_gaps({"devices": "not-a-mapping"}) == ()


# ── Four-value adjudication ──


async def test_adjudicate_absorb_on_small_deviation() -> None:
    # 0.6 is only 0.1 below the 0.7 bar — within the absorb margin.
    resolver = _resolver()
    gap = _precision_gap(success_rate=0.6, sample_count=8)
    verdict = await resolver.adjudicate(gap)
    assert verdict is not None
    assert verdict.action == "absorb"
    assert verdict.knowledge  # mandatory
    assert verdict.rationale  # mandatory rationale
    assert not verdict.writes_code


async def test_adjudicate_rebind_when_alternative_strategy_exists() -> None:
    registry = FakeInferenceRegistry(
        (_FakeStrategy("diffusion_grasp_v2", affordances=("grasp",)),)
    )
    resolver = _resolver(inference_registry=registry)
    gap = _precision_gap(success_rate=0.2, sample_count=12)
    verdict = await resolver.adjudicate(gap)
    assert verdict is not None
    assert verdict.action == "rebind"
    assert verdict.target == "diffusion_grasp_v2"
    assert verdict.knowledge and verdict.rationale


async def test_adjudicate_acquire_missing_skill_when_downloadable() -> None:
    resolver = _resolver(acquire_probe=lambda gap: True)
    gap = PhysicalCapabilityGap(
        gap_type=PhysicalGapType.MISSING_SKILL.value,
        device_id="robot.arm",
        affordance="pour",
    )
    verdict = await resolver.adjudicate(gap)
    assert verdict is not None
    assert verdict.action == "acquire"
    assert verdict.writes_code
    assert verdict.to_intent() is not None


async def test_adjudicate_escalate_missing_skill_when_not_downloadable() -> None:
    resolver = _resolver(acquire_probe=lambda gap: False)
    gap = PhysicalCapabilityGap(
        gap_type=PhysicalGapType.MISSING_SKILL.value,
        device_id="robot.arm",
        affordance="pour",
    )
    verdict = await resolver.adjudicate(gap)
    assert verdict is not None
    assert verdict.action == "escalate"


async def test_adjudicate_escalate_missing_device() -> None:
    resolver = _resolver()
    gap = PhysicalCapabilityGap(
        gap_type=PhysicalGapType.MISSING_DEVICE.value,
        device_id="robot.arm",
        affordance="grasp",
    )
    verdict = await resolver.adjudicate(gap)
    assert verdict is not None
    assert verdict.action == "escalate"
    assert "connect" in verdict.target.lower()


async def test_adjudicate_escalate_hardware_degraded() -> None:
    resolver = _resolver()
    gap = PhysicalCapabilityGap(
        gap_type=PhysicalGapType.HARDWARE_DEGRADED.value,
        device_id="robot.arm",
        affordance="grasp",
        success_rate=0.1,
        sample_count=6,
    )
    verdict = await resolver.adjudicate(gap)
    assert verdict is not None
    assert verdict.action == "escalate"
    assert "inspect" in verdict.target.lower()


async def test_adjudicate_escalate_teleop_when_nothing_cheaper_applies() -> None:
    # Large deviation, no alternative strategy available → teleoperation.
    resolver = _resolver(inference_registry=FakeInferenceRegistry())
    gap = _precision_gap(success_rate=0.2, sample_count=12)
    verdict = await resolver.adjudicate(gap)
    assert verdict is not None
    assert verdict.action == "escalate"
    assert "teleoperate" in verdict.target.lower()


async def test_all_verdict_actions_are_within_the_closed_space() -> None:
    resolver = _resolver(
        inference_registry=FakeInferenceRegistry(
            (_FakeStrategy("grasp_alt", affordances=("grasp",)),)
        )
    )
    gaps = (
        _precision_gap(success_rate=0.6, sample_count=8),  # absorb
        _precision_gap(success_rate=0.2, sample_count=8),  # rebind
        PhysicalCapabilityGap(
            gap_type=PhysicalGapType.MISSING_DEVICE.value, device_id="robot.arm"
        ),  # escalate
    )
    for gap in gaps:
        verdict = await resolver.adjudicate(gap)
        assert verdict is not None
        assert verdict.action in ADAPTATION_ACTIONS


# ── Evidence threshold ──


async def test_precision_gap_below_min_samples_is_not_adjudicated() -> None:
    resolver = _resolver(min_samples=5)
    gap = _precision_gap(success_rate=0.2, sample_count=2)
    assert await resolver.adjudicate(gap) is None


async def test_precision_gap_at_min_samples_is_adjudicated() -> None:
    resolver = _resolver(min_samples=5)
    gap = _precision_gap(success_rate=0.6, sample_count=5)
    verdict = await resolver.adjudicate(gap)
    assert verdict is not None
    assert verdict.action == "absorb"


async def test_structural_gap_not_gated_by_sample_count() -> None:
    # A missing device is a fact, not a statistic; zero samples must still escalate.
    resolver = _resolver(min_samples=5)
    gap = PhysicalCapabilityGap(
        gap_type=PhysicalGapType.MISSING_DEVICE.value,
        device_id="robot.arm",
        sample_count=0,
    )
    verdict = await resolver.adjudicate(gap)
    assert verdict is not None
    assert verdict.action == "escalate"


# ── Rebind candidate lookup ──


def test_find_rebind_candidate_matches_declared_affordance() -> None:
    registry = FakeInferenceRegistry(
        (
            _FakeStrategy("place_policy", affordances=("place",)),
            _FakeStrategy("grasp_policy", affordances=("grasp", "pick")),
        )
    )
    resolver = _resolver(inference_registry=registry)
    assert resolver._find_rebind_candidate(_precision_gap()) == "grasp_policy"


def test_find_rebind_candidate_falls_back_to_id_substring() -> None:
    registry = FakeInferenceRegistry((_BareStrategy("vla_grasp_baseline"),))
    resolver = _resolver(inference_registry=registry)
    assert resolver._find_rebind_candidate(_precision_gap()) == "vla_grasp_baseline"


def test_find_rebind_candidate_none_without_registry() -> None:
    resolver = _resolver(inference_registry=None)
    assert resolver._find_rebind_candidate(_precision_gap()) is None


def test_find_rebind_candidate_none_when_no_match() -> None:
    registry = FakeInferenceRegistry(
        (_FakeStrategy("place_policy", affordances=("place",)),)
    )
    resolver = _resolver(inference_registry=registry)
    assert resolver._find_rebind_candidate(_precision_gap(affordance="grasp")) is None


# ── Capability naming ──


def test_capability_name_is_valid_dotted_form() -> None:
    from leapflow.domain.evolution_intent import is_capability_name

    resolver = _resolver()
    gap = _precision_gap(device_id="robot.arm", affordance="grasp")
    name = resolver._capability(gap)
    assert name == "physical.robot_arm.grasp"
    assert is_capability_name(name)


def test_capability_name_without_affordance() -> None:
    from leapflow.domain.evolution_intent import is_capability_name

    resolver = _resolver()
    gap = PhysicalCapabilityGap(
        gap_type=PhysicalGapType.MISSING_DEVICE.value, device_id="robot.arm"
    )
    name = resolver._capability(gap)
    assert name == "physical.robot_arm"
    assert is_capability_name(name)
