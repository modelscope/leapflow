# Copyright (c) Alibaba, Inc. and its affiliates.
"""The declarative action adapter: every step reversible, nothing model-specific.

Round trips start from canonical robot rows (joints deg, flange pose mm with
fixed-axis X-Y-Z deg, gripper mm): ``invert`` takes them into the model's
native space and ``adapt`` must bring them back -- and the reverse -- to
within 1e-6, for each declared representation, unit, frame and reference.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import pytest

from leapflow.hardware.skills.adapt import (
    ActionStep,
    AdaptError,
    AdaptViolation,
    BoundGroup,
    StepContext,
    build_action_pipeline,
    build_plugins,
    build_state_pipeline,
)
from leapflow.hardware.skills.adapt.pipeline import ActionPipeline
from leapflow.hardware.skills.adapt.rotations import (
    ROTATION_NONE,
    from_matrix,
    rotation_width,
    to_matrix,
)
from leapflow.hardware.skills.adapt.spec import Normalization, VectorSpace, stats_checksum
from leapflow.hardware.skills.adapt.steps import Chunk
from leapflow.hardware.skills.geometry import Pose, Transform, rpy_to_matrix, wrap_deg
from tests._harness.policy_cell import EE, GRIPPER, JOINT_START, piper_chain
from tests._harness.skill_stubs import JOINTS

TOL = 1e-6
PIPER_LIMITS = ((-150, 150), (0, 180), (-170, 0), (-100, 100), (-70, 70), (-120, 120))
T_BASE_CAMERA = Transform.from_pose(Pose(400.0, -50.0, 600.0, -170.0, 5.0, 90.0))
POSES = (
    (300.0, 20.0, 200.0, 170.0, 10.0, -30.0),
    (305.0, 18.0, 195.0, 165.0, 12.0, -25.0),
    (310.0, 15.0, 190.0, 160.0, 15.0, -20.0),
)
JOINT_ROWS = (
    (1.0, 80.0, -55.0, 2.0, 65.0, 3.0),
    (2.0, 81.0, -56.0, 3.0, 64.0, 4.0),
    (3.0, 82.0, -57.0, 4.0, 63.0, 5.0),
)
GRIP_ROWS = ((40.0,), (42.0,), (45.0,))
POSE_ANCHOR = (295.0, 22.0, 205.0, 172.0, 8.0, -32.0)
GRIP_ANCHOR = (38.0,)
CHANNELS = {"joints": JOINTS, "ee_pose": EE, "gripper": (GRIPPER,)}


def _frames(name: str) -> Transform:
    if name == "camera":
        return T_BASE_CAMERA
    raise KeyError(name)


def _ctx(**kw: Any) -> StepContext:
    values: dict[str, Any] = {
        "anchor": {"arm": POSE_ANCHOR, "gripper": GRIP_ANCHOR},
        "frames": _frames,
        "gripper_mm": {"gripper": (0.0, 70.0)},
    }
    return StepContext(**{**values, **kw})


def _space(raw: Mapping[str, Any]) -> VectorSpace:
    return VectorSpace.from_mapping(raw, "action_space")


def _bound(space: VectorSpace, *, ik: bool = False) -> list[BoundGroup]:
    bound = []
    for group in space.groups:
        drives_joints = ik and group.kind == "ee_pose"
        channels = JOINTS if drives_joints else CHANNELS[group.kind]
        bound.append(BoundGroup(group, tuple(channels), ik=drives_joints))
    return bound


def _pipeline(raw: Mapping[str, Any]) -> ActionPipeline:
    space = _space(raw)
    return build_action_pipeline(space, _bound(space))


def _rows(pipeline: ActionPipeline, by_group: Mapping[str, Sequence[Sequence[float]]]) -> list:
    count = len(next(iter(by_group.values())))
    return [
        tuple(v for name, _ in pipeline.layout for v in by_group[name][i]) for i in range(count)
    ]


def _assert_rows(actual: Sequence, expected: Sequence, *, angles: Sequence[int] = ()) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert len(got) == len(want)
        for dim, (a, b) in enumerate(zip(got, want)):
            error = abs(wrap_deg(a - b)) if dim in angles else abs(a - b)
            assert error < TOL, (dim, got, want)


def _ee(rotation: str, **kw: Any) -> dict[str, Any]:
    width = 3 + rotation_width(rotation)
    return {
        "name": "arm",
        "kind": "ee_pose",
        "dims": list(range(width)),
        "rotation": rotation,
        **kw,
    }


def _joints(**kw: Any) -> dict[str, Any]:
    return {"name": "arm", "kind": "joints", "dims": [5, 4, 3, 2, 1, 0], **kw}


def _gripper(dim: int, **kw: Any) -> dict[str, Any]:
    return {"name": "gripper", "kind": "gripper", "dims": [dim], **kw}


def _stats(mode: str, low: Sequence[float], high: Sequence[float]) -> dict[str, Any]:
    a, b = ("mean", "std") if mode == "mean_std" else ("min", "max")
    return {"mode": mode, a: list(low), b: list(high), "checksum": stats_checksum(mode, low, high)}


# ── Round trips ──────────────────────────────────────────────────────────────

_JOINT_CASES = {
    "rad_mean_std_normalized_gripper": {
        "width": 7,
        "normalization": _stats("mean_std", [0.1] * 7, [0.5, 0.4, 0.3, 0.2, 0.6, 0.7, 0.25]),
        "groups": [_joints(unit="rad"), _gripper(6, unit="normalized")],
    },
    "deg_min_max_scaled_gripper_in_m": {
        "width": 7,
        "normalization": _stats("min_max", [-200.0] * 6 + [0.0], [200.0] * 6 + [0.1]),
        "groups": [
            _joints(scale=-1.0, offset=5.0),
            _gripper(6, unit="m", scale=2.0, offset=0.01),
        ],
    },
    "normalized_inverted_gripper_delta": {
        "width": 7,
        "names": ["a", "b", "c", "d", "e", "f", "grip"],
        "groups": [
            _joints(unit="rad", reference="delta"),
            _gripper(
                "grip", unit="normalized", open_value=0.0, closed_value=1.0, reference="delta"
            ),
        ],
    },
}


@pytest.mark.parametrize("raw", list(_JOINT_CASES.values()), ids=list(_JOINT_CASES))
def test_joint_spaces_round_trip(raw: Mapping[str, Any]) -> None:
    pipeline = _pipeline(raw)
    robot = _rows(pipeline, {"arm": JOINT_ROWS, "gripper": GRIP_ROWS})
    ctx = _ctx(anchor={"arm": JOINT_START, "gripper": GRIP_ANCHOR})
    native = pipeline.invert(robot, ctx)
    assert all(len(row) == pipeline.native_width == 7 for row in native)
    _assert_rows(pipeline.adapt(native, ctx), robot)
    _assert_rows(pipeline.invert(pipeline.adapt(native, ctx), ctx), native)


_POSE_CASES = {
    "quat_xyzw_relative_base_min_max": {
        "width": 8,
        "normalization": _stats("min_max", [-100.0] * 7 + [0.0], [100.0] * 7 + [1.0]),
        "groups": [_ee("quat_xyzw", reference="relative"), _gripper(7, unit="normalized")],
    },
    "quat_wxyz_absolute_m": {
        "width": 7,
        "groups": [_ee("quat_wxyz", unit="m")],
    },
    "rot6d_absolute_camera": {
        "width": 9,
        "groups": [_ee("rot6d", frame="camera")],
    },
    "euler_zyx_absolute": {"width": 6, "groups": [_ee("euler_zyx")]},
    "euler_XYZ_rad_delta_tool_m": {
        "width": 6,
        "groups": [_ee("euler_XYZ", angle_unit="rad", reference="delta", frame="tool", unit="m")],
    },
    "axis_angle_delta_camera": {
        "width": 7,
        "groups": [_ee("axis_angle", reference="delta", frame="camera"), _gripper(6)],
    },
    "rot6d_rows_relative_tool": {
        "width": 9,
        "groups": [_ee("rot6d_rows", reference="relative", frame="tool")],
    },
    "arm_euler_relative_base": {"width": 6, "groups": [_ee("euler_xyz", reference="relative")]},
}


@pytest.mark.parametrize("raw", list(_POSE_CASES.values()), ids=list(_POSE_CASES))
def test_pose_spaces_round_trip(raw: Mapping[str, Any]) -> None:
    pipeline = _pipeline(raw)
    robot = _rows(pipeline, {"arm": POSES, "gripper": GRIP_ROWS})
    ctx = _ctx()
    native = pipeline.invert(robot, ctx)
    assert all(len(row) == raw["width"] for row in native)
    _assert_rows(pipeline.adapt(native, ctx), robot, angles=(3, 4, 5))
    _assert_rows(pipeline.invert(pipeline.adapt(native, ctx), ctx), native)


def test_steps_follow_the_declaration_only() -> None:
    minimal = _pipeline({"width": 7, "groups": [_joints(), _gripper(6)]})
    assert minimal.describe() == ["map", "validate"]
    full = _pipeline(_POSE_CASES["quat_xyzw_relative_base_min_max"])
    assert full.describe() == ["unnormalize", "map", "gripper", "rotation", "reference", "validate"]
    framed = _pipeline(_POSE_CASES["axis_angle_delta_camera"])
    assert framed.describe() == ["map", "rotation", "frame", "reference", "validate"]


def test_a_delta_chunk_chains_from_the_previous_row() -> None:
    pipeline = _pipeline({"width": 7, "groups": [_joints(reference="delta"), _gripper(6)]})
    step = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # dims are reversed: the last value drives j1
    rows = [(*step, 40.0)] * 3
    out = pipeline.adapt(rows, _ctx(anchor={"arm": JOINT_START}))
    assert [row[5] for row in out] == [1.0, 2.0, 3.0]  # j6 is native dim 0
    assert all(row[6] == 40.0 for row in out)


def test_a_position_only_pose_holds_the_observed_orientation() -> None:
    pipeline = _pipeline({"width": 3, "groups": [_ee(ROTATION_NONE)]})
    (out,) = pipeline.adapt([(300.0, 20.0, 200.0)], _ctx())
    assert out == (300.0, 20.0, 200.0, *POSE_ANCHOR[3:])
    assert pipeline.invert([out], _ctx()) == ((300.0, 20.0, 200.0),)


# ── Rotations ────────────────────────────────────────────────────────────────

_ROTATIONS = (
    "axis_angle", "quat_xyzw", "quat_wxyz", "rot6d", "rot6d_rows",
    "euler_xyz", "euler_zyx", "euler_yzx", "euler_XYZ", "euler_ZYX", "euler_YXZ",
)  # fmt: skip


def _assert_matrix(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> None:
    assert all(abs(x - y) < 1e-9 for ra, rb in zip(a, b) for x, y in zip(ra, rb)), (a, b)


@pytest.mark.parametrize("name", _ROTATIONS)
@pytest.mark.parametrize("unit", ["deg", "rad"])
def test_every_rotation_encodes_and_decodes(name: str, unit: str) -> None:
    r = rpy_to_matrix(30.0, -20.0, 50.0)
    encoded = from_matrix(name, r, unit)
    assert len(encoded) == rotation_width(name)
    _assert_matrix(to_matrix(name, encoded, unit), r)


def test_the_arm_convention_is_fixed_axis_xyz() -> None:
    _assert_matrix(to_matrix("euler_xyz", (30.0, -20.0, 50.0)), rpy_to_matrix(30.0, -20.0, 50.0))
    # Extrinsic z-y-x equals intrinsic X-Y-Z with the angles reversed.
    _assert_matrix(
        to_matrix("euler_zyx", (10.0, 20.0, 30.0)), to_matrix("euler_XYZ", (30.0, 20.0, 10.0))
    )


def test_quaternions_are_written_with_a_non_negative_w() -> None:
    r = rpy_to_matrix(179.0, 10.0, -170.0)
    x, y, z, w = from_matrix("quat_xyzw", r)
    assert w >= 0.0 and abs(x * x + y * y + z * z + w * w - 1.0) < 1e-9
    _assert_matrix(to_matrix("quat_xyzw", (-x, -y, -z, -w)), r)  # either sign reads alike


def test_a_6d_rotation_need_not_be_orthonormal() -> None:
    r = rpy_to_matrix(30.0, -20.0, 50.0)
    columns = from_matrix("rot6d", r)
    skewed = (*(2.0 * v for v in columns[:3]), *(c + 0.3 * a for c, a in zip(columns[3:], columns)))
    _assert_matrix(to_matrix("rot6d", skewed), r)


@pytest.mark.parametrize(
    ("name", "values", "match"),
    [
        ("euler_xyy", (0.0, 0.0, 0.0), "unknown rotation"),
        ("euler_xYz", (0.0, 0.0, 0.0), "mixes"),
        ("quat_xyzw", (0.0, 0.0, 0.0, 0.0), "non-zero length"),
        ("rot6d", (1.0, 0.0, 0.0, 2.0, 0.0, 0.0), "independent"),
        ("axis_angle", (0.0, 0.0), "takes 3 values"),
    ],
)
def test_a_degenerate_rotation_is_refused(name: str, values: tuple, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        to_matrix(name, values)


def test_a_degenerate_model_rotation_is_an_adapt_error() -> None:
    pipeline = _pipeline({"width": 7, "groups": [_ee("quat_xyzw")]})
    with pytest.raises(AdaptError, match="rotation is unusable"):
        pipeline.adapt([(300.0, 0.0, 200.0, 0.0, 0.0, 0.0, 0.0)], _ctx())


# ── Normalisation ────────────────────────────────────────────────────────────


def test_stats_are_pinned_by_their_checksum() -> None:
    good = _stats("mean_std", [0.0, 1.0], [1.0, 2.0])
    assert Normalization.from_mapping(good, 2, "n").checksum == good["checksum"]
    tampered = {**good, "std": [1.0, 2.5]}
    with pytest.raises(ValueError, match="checksum"):
        Normalization.from_mapping(tampered, 2, "n")


def test_a_normalized_space_needs_stats() -> None:
    with pytest.raises(ValueError, match="declares no normalization stats"):
        _space({"width": 7, "normalized": True, "groups": [_joints(), _gripper(6)]})


def test_the_state_pipeline_feeds_the_model_its_own_normalisation() -> None:
    space = _space(
        {
            "width": 7,
            "normalization": _stats("mean_std", [0.0] * 6 + [35.0], [1.0] * 6 + [35.0]),
            "groups": [_joints(unit="rad"), _gripper(6)],
        }
    )
    state = build_state_pipeline(space, _bound(space))
    (row,) = state.invert([(*JOINT_START, 70.0)], StepContext())
    assert abs(row[6] - 1.0) < 1e-6
    assert abs(row[1] - math.radians(65.0)) < 1e-6  # dims are reversed: j5 is native dim 1


def test_observation_state_must_be_absolute() -> None:
    space = _space({"width": 6, "groups": [_joints(reference="relative")]})
    with pytest.raises(ValueError, match="must be absolute"):
        build_state_pipeline(space, _bound(space))


# ── Binding ──────────────────────────────────────────────────────────────────


def test_the_binding_must_cover_the_space_exactly() -> None:
    space = _space({"width": 7, "groups": [_joints(), _gripper(6)]})
    arm, gripper = _bound(space)
    with pytest.raises(ValueError, match="binds groups"):
        build_action_pipeline(space, [arm])
    with pytest.raises(ValueError, match="binds 5 channels"):
        build_action_pipeline(space, [BoundGroup(arm.group, JOINTS[:5]), gripper])


def test_a_pose_driving_joints_needs_the_dh_table() -> None:
    space = _space({"width": 6, "groups": [_ee("euler_xyz")]})
    with pytest.raises(ValueError, match="kinematics.dh"):
        build_action_pipeline(space, _bound(space, ik=True))


# ── IK ───────────────────────────────────────────────────────────────────────


def _ik_pipeline() -> ActionPipeline:
    space = _space({"width": 6, "groups": [_ee("euler_xyz")]})
    return build_action_pipeline(
        space, _bound(space, ik=True), chain=piper_chain(), joint_limits=PIPER_LIMITS
    )


def test_ik_solves_a_pose_chunk_for_joints() -> None:
    chain, pipeline = piper_chain(), _ik_pipeline()
    assert pipeline.describe() == ["map", "ik", "validate"] and pipeline.layout == (("arm", 6),)
    start = chain.forward(JOINT_START)
    targets = [start.translated(dz=-5.0 * (i + 1)) for i in range(3)]
    ctx = StepContext(current={"arm": JOINT_START})
    joints = pipeline.adapt([t.as_tuple() for t in targets], ctx)
    for target, solution in zip(targets, joints):
        reached = chain.forward(solution)
        assert reached.distance_mm(target) < 0.1 and reached.rotation_to_deg(target) < 0.1
        assert all(lo <= q <= hi for q, (lo, hi) in zip(solution, PIPER_LIMITS))
    # The inverse is forward kinematics: recorded joints become the model's poses.
    poses = pipeline.invert(joints, ctx)
    _assert_rows(poses, [chain.forward(q).as_tuple() for q in joints], angles=(3, 4, 5))


def test_ik_refuses_an_unreachable_pose() -> None:
    target = piper_chain().forward(JOINT_START).translated(dz=500.0)
    with pytest.raises(AdaptViolation, match="action 0"):
        _ik_pipeline().adapt([target.as_tuple()], StepContext(current={"arm": JOINT_START}))


def test_ik_needs_the_observed_joints() -> None:
    with pytest.raises(AdaptError, match="seed"):
        _ik_pipeline().adapt([POSES[0]], StepContext())


# ── Gripper ──────────────────────────────────────────────────────────────────


def test_a_binary_gripper_keeps_its_decision_inside_the_band() -> None:
    pipeline = _pipeline(
        {"width": 1, "groups": [_gripper(0, unit="normalized", binary=True, hysteresis=0.1)]}
    )
    memory: dict[str, Any] = {}
    first = _ctx(anchor={"gripper": (10.0,)}, memory=memory)  # observed closed
    assert [r[0] for r in pipeline.adapt([(0.55,), (0.7,), (0.55,)], first)] == [0.0, 70.0, 70.0]
    second = _ctx(anchor={"gripper": (0.0,)}, memory=memory)  # a later chunk, same run
    assert [r[0] for r in pipeline.adapt([(0.45,), (0.3,), (0.45,)], second)] == [70.0, 0.0, 0.0]


def test_a_binary_gripper_inverts_to_its_decision() -> None:
    pipeline = _pipeline({"width": 1, "groups": [_gripper(0, unit="normalized", binary=True)]})
    rows = pipeline.invert([(10.0,), (34.0,), (36.0,), (70.0,)], _ctx())
    assert [row[0] for row in rows] == [0.0, 0.0, 1.0, 1.0]  # what a binary model emits


def test_a_normalized_gripper_needs_its_declared_widths() -> None:
    pipeline = _pipeline({"width": 1, "groups": [_gripper(0, unit="normalized")]})
    with pytest.raises(AdaptError, match="closed/open width"):
        pipeline.adapt([(0.5,)], _ctx(gripper_mm={}))


def test_a_binary_gripper_must_be_a_normalized_absolute_command() -> None:
    with pytest.raises(ValueError, match="normalized units"):
        _space({"width": 1, "groups": [_gripper(0, binary=True)]})
    with pytest.raises(ValueError, match="must be absolute"):
        _space(
            {
                "width": 1,
                "groups": [_gripper(0, unit="normalized", binary=True, reference="delta")],
            }
        )


# ── Validation ───────────────────────────────────────────────────────────────


def _joint_pipeline() -> ActionPipeline:
    return _pipeline({"width": 6, "groups": [{**_joints(), "dims": list(range(6))}]})


def test_a_first_action_that_jumps_is_a_violation() -> None:
    ctx = StepContext(current={"arm": JOINT_START}, max_jump={"arm": (5.0,) * 6})
    near = (JOINT_START[0] + 4.0, *JOINT_START[1:])
    assert _joint_pipeline().adapt([near], ctx) == (near,)
    far = (JOINT_START[0] + 10.0, *JOINT_START[1:])
    with pytest.raises(AdaptViolation, match="moves dim 0 by 10.00"):
        _joint_pipeline().adapt([far], ctx)


def test_the_jump_check_wraps_cyclic_channels() -> None:
    now = (179.0, *JOINT_START[1:])
    ctx = StepContext(
        current={"arm": now},
        max_jump={"arm": (5.0,) * 6},
        cyclic={"arm": (True,) + (False,) * 5},
    )
    assert _joint_pipeline().adapt([(-179.0, *JOINT_START[1:])], ctx)


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([], "empty chunk"),
        ([(math.nan, *JOINT_START[1:])], "non-finite"),
        ([JOINT_START[:5]], "the space 6"),
    ],
)
def test_an_unusable_chunk_is_an_adapt_error(rows: list, match: str) -> None:
    with pytest.raises(AdaptError, match=match) as raised:
        _joint_pipeline().adapt(rows, StepContext())
    assert not isinstance(raised.value, AdaptViolation)


# ── Plugins ──────────────────────────────────────────────────────────────────


class _Nudge:
    """A plugin step: shifts the gripper by ``by`` mm (and back)."""

    name = "nudge"

    def __init__(self, by: float) -> None:
        self._by = by

    def _shift(self, chunk: Chunk, by: float) -> Chunk:
        return tuple({**row, "gripper": (row["gripper"][0] + by,)} for row in chunk)

    def apply(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        return self._shift(chunk, self._by)

    def invert(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        return self._shift(chunk, -self._by)


def _nudge(params: Mapping[str, Any], groups: Sequence[Any]) -> ActionStep:
    return _Nudge(float(params["by"]))


def test_plugin_steps_run_before_validation_and_invert() -> None:
    space = _space({"width": 7, "groups": [_joints(), _gripper(6)]})
    plugins = build_plugins([("nudge", {"by": 2.5})], space.groups, {"nudge": _nudge})
    pipeline = build_action_pipeline(space, _bound(space), plugins=plugins)
    assert pipeline.describe() == ["map", "nudge", "validate"]
    native = [(*JOINT_START, 40.0)]
    (out,) = pipeline.adapt(native, StepContext())
    assert out[6] == 42.5
    _assert_rows(pipeline.invert([out], StepContext()), native)


def test_a_plugin_must_be_installed_and_be_a_step() -> None:
    space = _space({"width": 7, "groups": [_joints(), _gripper(6)]})
    assert build_plugins([], space.groups, {}) == []
    with pytest.raises(ValueError, match="is not installed"):
        build_plugins([("ghost", {})], space.groups, {})
    with pytest.raises(ValueError, match="does not implement ActionStep"):
        build_plugins([("bad", {})], space.groups, {"bad": lambda params, groups: object()})
