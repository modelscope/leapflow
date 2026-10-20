# Copyright (c) Alibaba, Inc. and its affiliates.
"""The pure pieces under policy streaming: DH forward kinematics, the chunk
aggregator and low-pass, the safety shield and contact limits, and the
streaming settings.

Each is clock- and device-free, so the properties streaming relies on -- time
indexing that skips inference latency, continuous cross-fades, the temporal
ensemble, refusal of malformed chunks, envelope/jump/acceleration/slew/pose
filtering -- are pinned exactly here.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import yaml

from leapflow.hardware.skills.adapt import ChunkAggregator, LowPass
from leapflow.hardware.skills.geometry import Pose
from leapflow.hardware.skills.kinematics import DhChain
from leapflow.hardware.skills.policy_binding import PolicySettings
from leapflow.hardware.skills.shield import (
    ContactLimit,
    SafetyShield,
    ShieldChannel,
    contact_limits,
    contact_violation,
)
from tests._harness.skill_stubs import PIPER_YAML

# ── Kinematics ───────────────────────────────────────────────────────────────

_PLANAR = {"links": [{"a": 0, "alpha": 0, "d": 0}, {"a": 100, "alpha": 0, "d": 0}]}


def test_modified_dh_planar_arm_reaches_where_geometry_says() -> None:
    chain = DhChain.from_mapping(_PLANAR)
    # Modified DH: link 2's ``a`` lies along joint 1's x axis, rotated by q1.
    flange = chain.forward([90.0, 0.0])
    assert (flange.x, flange.y, flange.z) == pytest.approx((0.0, 100.0, 0.0), abs=1e-9)
    assert flange.rz == pytest.approx(90.0)


def test_standard_dh_reads_each_link_in_the_other_order() -> None:
    chain = DhChain.from_mapping({**_PLANAR, "convention": "standard"})
    # Standard DH: link 2's ``a`` follows joint 2's rotation.
    flange = chain.forward([0.0, 90.0])
    assert (flange.x, flange.y) == pytest.approx((0.0, 100.0), abs=1e-9)


def test_agreement_measures_distance_and_rotation_to_a_reported_flange() -> None:
    chain = DhChain.from_mapping(_PLANAR)
    predicted = chain.forward([30.0, 20.0])
    assert chain.agreement([30.0, 20.0], predicted) == pytest.approx((0.0, 0.0), abs=1e-9)
    moved = Pose(predicted.x + 3.0, predicted.y + 4.0, predicted.z, *predicted.as_tuple()[3:])
    mm, deg = chain.agreement([30.0, 20.0], moved)
    assert mm == pytest.approx(5.0) and deg == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "data",
    [
        {"links": []},
        {"links": "a"},
        {"links": [{"a": 0, "alpha": 0}]},
        {"links": [{"a": 0, "alpha": True, "d": 0}]},
        {"links": [{"a": math.inf, "alpha": 0, "d": 0}]},
        {"links": [{"a": 0, "alpha": 0, "d": 0}], "convention": "craig"},
    ],
)
def test_bad_dh_tables_are_refused(data: dict) -> None:
    with pytest.raises(ValueError):
        DhChain.from_mapping(data)


def test_forward_refuses_a_joint_vector_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="2 joints"):
        DhChain.from_mapping(_PLANAR).forward([0.0])


def test_the_shipped_piper_table_is_a_six_link_chain() -> None:
    declaration = yaml.safe_load(PIPER_YAML.read_text(encoding="utf-8"))
    chain = DhChain.from_mapping(declaration["kinematics"]["dh"])
    assert chain.dof == 6
    flange = chain.forward([0.0, 80.0, -55.0, 0.0, 70.0, 0.0])
    assert 150.0 < flange.x < 450.0 and abs(flange.y) < 1e-6


# ── Chunk aggregation ────────────────────────────────────────────────────────


def test_sample_is_indexed_by_time_and_skips_inference_latency() -> None:
    scheduler = ChunkAggregator(1)
    scheduler.offer([[0.0], [10.0], [20.0], [30.0]], observed_at=0.0, step_s=0.1, now=0.15)
    assert scheduler.sample(0.15) == pytest.approx((15.0,))  # 0.15 s already passed
    assert scheduler.sample(0.3) == pytest.approx((30.0,))
    assert scheduler.remaining_s(0.3) == pytest.approx(0.1)
    assert scheduler.sample(0.4) is None  # exhausted: one step past the last action


def test_a_fresh_chunk_cross_fades_from_the_one_it_replaces() -> None:
    scheduler = ChunkAggregator(1, blend_s=0.1)
    scheduler.offer([[0.0]] * 10, observed_at=0.0, step_s=0.1, now=0.0)
    scheduler.offer([[10.0]] * 10, observed_at=0.2, step_s=0.1, now=0.2)
    assert scheduler.sample(0.2) == pytest.approx((0.0,))
    assert scheduler.sample(0.25) == pytest.approx((5.0,))
    assert scheduler.sample(0.3) == pytest.approx((10.0,))
    assert scheduler.chunks == 2


def test_a_chunk_arriving_after_starvation_starts_without_a_blend() -> None:
    scheduler = ChunkAggregator(1, blend_s=0.5)
    scheduler.offer([[0.0]], observed_at=0.0, step_s=0.1, now=0.0)
    scheduler.offer([[10.0]] * 3, observed_at=0.5, step_s=0.1, now=0.5)
    assert scheduler.sample(0.5) == pytest.approx((10.0,))


def test_cyclic_channels_interpolate_the_short_way_round() -> None:
    scheduler = ChunkAggregator(1, cyclic=[True])
    scheduler.offer([[170.0], [-170.0]], observed_at=0.0, step_s=0.1, now=0.0)
    assert scheduler.sample(0.05) == pytest.approx((180.0,))


@pytest.mark.parametrize(
    ("rows", "step_s", "match"),
    [
        ([], 0.1, "empty"),
        ([[1.0, 2.0]], 0.1, "expected 1"),
        ([[math.nan]], 0.1, "non-finite"),
        ([[1.0]], 0.0, "positive"),
    ],
)
def test_malformed_chunks_never_become_commands(rows: list, step_s: float, match: str) -> None:
    scheduler = ChunkAggregator(1)
    with pytest.raises(ValueError, match=match):
        scheduler.offer(rows, observed_at=0.0, step_s=step_s, now=0.0)
    assert scheduler.chunks == 0 and scheduler.sample(0.0) is None


def test_latest_replaces_the_executing_chunk_at_once() -> None:
    aggregator = ChunkAggregator(1, mode="latest")
    aggregator.offer([[0.0]] * 10, observed_at=0.0, step_s=0.1, now=0.0)
    aggregator.offer([[10.0]] * 10, observed_at=0.2, step_s=0.1, now=0.2)
    assert aggregator.sample(0.2) == pytest.approx((10.0,))


def test_hold_interpolation_keeps_the_row_whose_slot_is_running() -> None:
    aggregator = ChunkAggregator(1, interpolation="hold")
    aggregator.offer([[0.0], [10.0]], observed_at=0.0, step_s=0.1, now=0.0)
    assert aggregator.sample(0.09) == pytest.approx((0.0,))
    assert aggregator.sample(0.1) == pytest.approx((10.0,))


def test_the_temporal_ensemble_weights_every_live_chunk_oldest_first() -> None:
    decay = math.log(2.0)  # the second-oldest vote weighs half the oldest
    aggregator = ChunkAggregator(1, mode="temporal_ensemble", decay=decay)
    aggregator.offer([[0.0]] * 10, observed_at=0.0, step_s=0.1, now=0.0)
    aggregator.offer([[30.0]] * 10, observed_at=0.2, step_s=0.1, now=0.2)
    assert aggregator.sample(0.1) == pytest.approx((0.0,))  # the new chunk starts at 0.2
    assert aggregator.sample(0.25) == pytest.approx((10.0,))  # (1 * 0 + 0.5 * 30) / 1.5
    assert aggregator.sample(1.05) == pytest.approx((30.0,))  # the old chunk has ended


def test_pending_lists_the_rows_whose_slot_has_not_started() -> None:
    aggregator = ChunkAggregator(1)
    assert aggregator.pending(0.0) == ()
    aggregator.offer([[0.0], [1.0], [2.0], [3.0]], observed_at=0.0, step_s=0.1, now=0.0)
    assert aggregator.pending(0.15) == ((2.0,), (3.0,))
    assert aggregator.pending(0.2) == ((2.0,), (3.0,))
    assert aggregator.pending(0.5) == ()


def test_the_low_pass_converges_with_a_tick_rate_independent_time_constant() -> None:
    fast, slow = LowPass(1.0, [False]), LowPass(1.0, [False])
    fast.reset((0.0,))
    slow.reset((0.0,))
    for _ in range(100):
        quick = fast((1.0,), 0.01)
    for _ in range(10):
        out = slow((1.0,), 0.1)
    assert quick == pytest.approx(out, abs=1e-9)
    assert out[0] == pytest.approx(1.0 - math.exp(-2.0 * math.pi), abs=1e-9)
    assert LowPass(0.0, [False])((5.0,), 0.1) == (5.0,)  # off passes through


# ── Safety shield ────────────────────────────────────────────────────────────

_Z = ShieldChannel("ee.z", "arm", low=0.0, high=500.0, max_rate=100.0, tolerance=2.0)
_YAW = ShieldChannel("ee.rz", "arm", low=-180.0, high=180.0, max_rate=90.0, cyclic=True)


def test_values_within_tolerance_of_a_bound_are_clamped_onto_it() -> None:
    verdict = SafetyShield([_Z], rate_margin=1.0).filter([501.5], [500.0], 1.0)
    # Within the declared tolerance the clamp is not an intervention.
    assert verdict.ok and verdict.values == (500.0,) and verdict.clamped == 0


def test_values_beyond_tolerance_are_violations_without_a_command() -> None:
    verdict = SafetyShield([_Z]).filter([503.0], [500.0], 1.0)
    assert not verdict.ok and "above its limit" in verdict.violation


def test_slew_is_limited_to_the_declared_rate_with_margin() -> None:
    verdict = SafetyShield([_Z], rate_margin=0.5).filter([100.0], [0.0], 0.2)
    assert verdict.ok and verdict.values == pytest.approx((10.0,)) and verdict.clamped == 1


def test_cyclic_channels_wrap_and_slew_the_short_way() -> None:
    shield = SafetyShield([_YAW], rate_margin=1.0)
    wrapped = shield.filter([190.0], [-175.0], 1.0)
    assert wrapped.values == pytest.approx((-170.0,)) and wrapped.clamped == 0
    limited = shield.filter([-170.0], [170.0], 0.1)
    assert limited.values == pytest.approx((179.0,))


def test_the_pose_check_turns_a_bad_tool_pose_into_a_violation() -> None:
    shield = SafetyShield([_Z], pose_check=lambda action: "too low" if action[0] < 50 else "")
    assert shield.filter([60.0], [60.0], 0.1).ok
    assert shield.filter([40.0], [40.0], 0.1).violation == "too low"


def test_non_finite_targets_are_violations() -> None:
    assert not SafetyShield([_Z]).filter([math.inf], [0.0], 0.1).ok


def test_a_step_larger_than_max_jump_is_a_violation_not_a_slew() -> None:
    channel = ShieldChannel("ee.z", "arm", low=0.0, high=500.0, max_rate=100.0, max_jump=20.0)
    shield = SafetyShield([channel], rate_margin=1.0)
    assert shield.filter([115.0], [100.0], 0.01).ok
    verdict = shield.filter([125.0], [100.0], 0.01)
    assert not verdict.ok and "jumps 25.00" in verdict.violation


def test_acceleration_is_limited_from_the_previous_rate() -> None:
    channel = ShieldChannel("ee.z", "arm", max_rate=100.0, max_accel=200.0)
    shield = SafetyShield([channel], rate_margin=1.0)
    first = shield.filter([10.0], [0.0], 0.1, [0.0])
    # From rest, 0.1 s at 200 mm/s^2 reaches 20 mm/s: 2 mm, not the 10 asked.
    assert first.values == pytest.approx((2.0,)) and first.rates == pytest.approx((20.0,))
    second = shield.filter([10.0], first.values, 0.1, first.rates)
    assert second.values == pytest.approx((6.0,)) and second.rates == pytest.approx((40.0,))
    assert second.clamped == 1


def test_braking_refuses_when_holding_would_break_the_acceleration_limit() -> None:
    channel = ShieldChannel("ee.z", "arm", max_rate=100.0, max_accel=100.0)
    shield = SafetyShield([channel], rate_margin=1.0)
    # Moving up at 50 mm/s, asked to go back down: the acceleration bound permits
    # only 40..60 mm/s, while approaching without overshoot permits -50..0.
    verdict = shield.filter([0.0], [5.0], 0.1, [50.0])
    assert not verdict.ok and "no command" in verdict.violation
    assert verdict.values == () and verdict.rates == ()


def test_jump_limits_come_from_the_device_and_can_be_tightened() -> None:
    envelope = SimpleNamespace(
        min_value=-150.0,
        max_value=150.0,
        max_rate=36.0,
        max_accel=None,
        max_jump=18.0,
        tolerance=0.5,
    )
    channel = SimpleNamespace(
        channel_id="joint.j1.position", envelope=envelope, quantity="", unit="deg"
    )
    assert ShieldChannel.from_channel("arm", channel).max_jump == pytest.approx(18.0)
    tightened = ShieldChannel.from_channel("arm", channel, max_jump=9.0)
    assert tightened.max_jump == pytest.approx(9.0)


# ── Contact ──────────────────────────────────────────────────────────────────


def _channel(channel_id: str, quantity: str, *, writable: bool = False, high: float = 5.0):
    envelope = SimpleNamespace(declared=True, min_value=-high, max_value=high)
    return SimpleNamespace(
        channel_id=channel_id,
        quantity=quantity,
        envelope=envelope,
        is_readable=True,
        is_writable=writable,
    )


def test_only_read_only_force_channels_with_bounds_arm_contact_limits() -> None:
    arm = SimpleNamespace(
        channels=(
            _channel("joint.j1.effort", "torque"),
            _channel("gripper.effort", "effort", writable=True),  # a command, not a reading
            _channel("joint.j1.position", "angular_position"),
        )
    )
    (limit,) = contact_limits([("arm", arm), ("arm", arm)])
    assert limit == ContactLimit("joint.j1.effort", "arm", -5.0, 5.0)


def test_a_reading_outside_its_contact_limit_stops_the_stream() -> None:
    limits = (ContactLimit("joint.j1.effort", "arm", -5.0, 5.0),)
    assert contact_violation(limits, {"arm": {"joint.j1.effort": 4.0}}) == ""
    assert "above its contact limit" in contact_violation(limits, {"arm": {"joint.j1.effort": 6.0}})
    assert "no usable reading" in contact_violation(limits, {"arm": {}})


# ── Settings ─────────────────────────────────────────────────────────────────


def test_policy_settings_default_and_override() -> None:
    assert PolicySettings.from_mapping({}) == PolicySettings()
    assert PolicySettings.from_mapping({"tick_hz": 100}).tick_hz == 100.0


@pytest.mark.parametrize(
    "data",
    [
        {"tick": 50},
        {"tick_hz": 0},
        {"max_jump_s": -1},
        {"ensemble_decay": -0.1},
        {"prefetch_ratio": 1.5},
        {"jpeg_quality": 101},
    ],
)
def test_policy_settings_refuse_unusable_values(data: dict) -> None:
    with pytest.raises(ValueError):
        PolicySettings.from_mapping(data)
