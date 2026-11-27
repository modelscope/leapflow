# Copyright (c) Alibaba, Inc. and its affiliates.
"""Camera calibration skills: measure where the camera sits, then prove it.

Perception maps camera points into the arm base frame through the ``camera``
frame, so a camera is usable only once that frame is measured:

1. ``calib.handeye`` -- the arm shows the workcell's calibration board to the
   camera from a ring of tilted, yawed poses and solves ``A_i X = Z B_i``. A
   fixed camera (frame parent ``base``, eye-to-hand) holds the board on the
   flange; a wrist camera (parent ``flange``, eye-in-hand) looks at a board
   lying still in the cell. The record carries the live intrinsics (perception
   refuses a stream that no longer matches them) and the board's mount, which
   its held-out grid and later ``calib.verify`` checks use as ground truth.
   The fit and depth check must both pass before a new record is installed.
2. ``calib.screws`` -- the same mount with no board at all: the arm moves one
   joint at a time, and the surfaces that joint carries must land where the
   firmware's own rotation says they went. See ``screw`` for why that is enough.
3. ``calib.ball`` -- the same mount again with nothing in the cell cooperating:
   the gripper carries a ball whose radius the jaws measured, and the camera's
   fit of that ball at a ring of poses fixes the mount *and* the ball's place in
   the gripper together. See ``ball`` for why a sphere is the cheapest target
   that can measure a mount to a millimetre.
4. ``calib.verify`` -- the arm moves over a small grid; at every stop the arm
   predicts where the board centre is, and the camera measures it through the
   depth image and the hand-eye mount -- the same chain ``perceive.locate``
   uses. The 95th-percentile disagreement is the accuracy perception has.

A fit or check out of tolerance is ``check_failed`` with side effect COMMITTED:
the arm moved to measure it.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from leapflow.hardware.skills.ball import fit_ball, residual_mm
from leapflow.hardware.skills.frame_status import Accuracy, FrameStatus
from leapflow.hardware.skills.geometry import (
    Mat3,
    Pose,
    Transform,
    Vec3,
    add,
    mat_mul,
    mat_vec,
    matrix_to_rpy,
    norm,
    rpy_to_matrix,
    solve_hand_eye,
    solve_tool_hand_eye,
    sub,
    tool_cone_deg,
)
from leapflow.hardware.skills.ports import (
    FrameView,
    SkillContext,
    WorkcellView,
)
from leapflow.hardware.skills.protocol import (
    CONSENT_EACH_RUN,
    EFFECT_ACTUATE,
    FAILURE_ARM_NOT_READY,
    FAILURE_CHECK_FAILED,
    FAILURE_INVALID_ARGS,
    FAILURE_OUT_OF_WORKSPACE,
    FAILURE_PLAN_STALE,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_TARGET_NOT_FOUND,
    FAILURE_UNREACHABLE,
    METHOD_CARRIED_BALL,
    METHOD_MARKERLESS_SCREW,
    REGION_BOOTSTRAP_UNTIL_MEASURED,
    REQUIRE_CALIBRATION,
    REQUIRE_RECORD,
    REQUIRE_ROLE,
    SIDE_EFFECT_COMMITTED,
    ArmState,
    SkillError,
    SkillOutcome,
    SkillPlan,
    SkillSpec,
    TargetObservation,
)
from leapflow.hardware.skills.screw import (
    JointEvidence,
    ScrewConfig,
    screw_from_poses,
    solve_mount,
    solve_vertical,
)

CAMERA_FRAME = "camera"
VERIFY_PROCEDURE = "camera.verify"
EYE_TO_HAND = "eye_to_hand"
EYE_IN_HAND = "eye_in_hand"
METHOD_SCREW = METHOD_MARKERLESS_SCREW

# A capture whose flange drifted more than this between the two reads inside it
# is not a still image of the arm.
CAPTURE_DRIFT_MM = 2.0
CAPTURE_DRIFT_DEG = 1.0
# Joints the arm was not told to move, and did not move, differ by less than this.
JOINT_MATCH_DEG = 0.5

Sample = tuple[Pose, TargetObservation]


def _number(minimum: float, maximum: float, default: float) -> dict[str, Any]:
    return {"type": "number", "minimum": minimum, "maximum": maximum, "default": default}


def _fmt(values: Sequence[float]) -> str:
    return "(" + ", ".join(f"{v:.1f}" for v in values) + ")"


def _check_failed(message: str, **result: Any) -> SkillError:
    return SkillError(
        FAILURE_CHECK_FAILED, message, side_effect_state=SIDE_EFFECT_COMMITTED, result=result
    )


def _procedure(ctx: SkillContext, frame: str) -> str:
    procedure = ctx.frames.procedure(frame)
    if not procedure:
        raise SkillError(
            FAILURE_SKILL_UNAVAILABLE,
            f"workcell {ctx.workcell.workcell_id!r} declares no {frame} calibration procedure",
        )
    return procedure


def _camera(ctx: SkillContext, args: Mapping[str, Any], *, calibrated: bool = False) -> str:
    role = str(args.get("camera", CAMERA_FRAME))
    if role.split(".", 1)[0] != CAMERA_FRAME or not ctx.workcell.role(role):
        raise SkillError(FAILURE_SKILL_UNAVAILABLE, f"no camera role {role!r} in this workcell")
    _procedure(ctx, role)
    if calibrated and ctx.frames.transform(role) is None:
        raise SkillError(FAILURE_SKILL_UNAVAILABLE, f"{role} is not calibrated; run calib.handeye camera={role}")
    return role


def _vision(ctx: SkillContext, role: str) -> Any:
    return ctx.vision if role == CAMERA_FRAME else ctx.camera_vision(role)


def _mounting(ctx: SkillContext, role: str = CAMERA_FRAME) -> str:
    parent = ctx.frames.parent(role)
    if parent == "base":
        return EYE_TO_HAND
    if parent == "flange":
        return EYE_IN_HAND
    raise SkillError(
        FAILURE_SKILL_UNAVAILABLE,
        "the workcell's camera frame must have parent: base (a fixed camera) or "
        "parent: flange (a wrist camera)",
    )


async def _ready_state(ctx: SkillContext) -> ArmState:
    state = await ctx.motion.state()
    if state.enabled is False:
        raise SkillError(FAILURE_ARM_NOT_READY, "the arm is not enabled; ask the user to enable it")
    return state


async def _check_start(ctx: SkillContext, start: Pose) -> None:
    current = (await ctx.motion.state()).tcp
    moved_mm, moved_deg = current.distance_mm(start), current.rotation_to_deg(start)
    accept = ctx.workcell.calibration
    if moved_mm > accept["start_drift_mm"] or moved_deg > accept["start_drift_deg"]:
        raise SkillError(
            FAILURE_PLAN_STALE,
            f"the tool moved {moved_mm:.1f} mm / {moved_deg:.1f} deg since planning; plan again",
        )


def _reachable(ctx: SkillContext, poses: Sequence[Pose]) -> tuple[list[Pose], str]:
    """The poses the arm may visit, and why the first refused one was refused."""
    kept: list[Pose] = []
    reason = ""
    for pose in poses:
        try:
            ctx.motion.check_target(pose)
        except SkillError as error:
            reason = reason or error.message
            continue
        kept.append(pose)
    return kept, reason


def _with_return(ctx: SkillContext, targets: list[Pose], start: Pose) -> tuple[Pose, ...]:
    try:
        ctx.motion.check_target(start)
    except SkillError:
        return tuple(targets)  # the start lies outside the calibration region; stay put
    return (*targets, start)


async def _observe(ctx: SkillContext, role: str) -> TargetObservation | None:
    try:
        return await _vision(ctx, role).observe_target()
    except SkillError as error:
        if error.failure_code == FAILURE_TARGET_NOT_FOUND:
            return None
        raise


async def _sweep(ctx: SkillContext, plan: SkillPlan) -> tuple[list[tuple[int, Sample]], int]:
    """Visit every target; observe the board at the first ``basis.samples`` of them."""
    count, settle = int(plan.basis["samples"]), float(plan.basis["settle_s"])
    seen: list[tuple[int, Sample]] = []
    skipped = 0
    for index, target in enumerate(plan.targets):
        await ctx.motion.move_tcp(target, speed_pct=plan.limits.get("speed_pct"))
        if index >= count:
            continue
        if settle > 0:
            await asyncio.sleep(settle)
        before = (await ctx.motion.state()).flange
        observation = await _observe(ctx, str(plan.basis.get("camera_role", CAMERA_FRAME)))
        after = (await ctx.motion.state()).flange
        accept = ctx.workcell.calibration
        if (before.distance_mm(after) > accept["start_drift_mm"]
                or before.rotation_to_deg(after) > accept["start_drift_deg"]):
            raise _check_failed("the flange moved during camera capture", problems=["capture drift"])
        if observation is None:
            skipped += 1
            continue
        seen.append((index, (after, observation)))
    return seen, skipped


def _target_scale(ctx: SkillContext, seen: TargetObservation) -> float | None:
    """Screen gross depth/PnP disagreement without claiming a metrological result."""
    scale = seen.depth_scale
    if scale is not None and abs(scale - 1.0) > ctx.workcell.calibration["max_target_scale_error"]:
        target = ctx.workcell.calibration_target
        square, marker = float(target["square_mm"]), float(target["marker_mm"])
        raise SkillError(
            FAILURE_CHECK_FAILED,
            f"depth and the declared board disagree by a scale of {scale:.2f}x "
            f"(square_mm={square:g}, marker_mm={marker:g}); measure the physical board "
            f"with a ruler. If printing caused the mismatch, its estimated dimensions "
            f"are {square * scale:.1f}/{marker * scale:.1f} mm. Reprint at actual size or "
            "correct the declaration and reload; also check depth alignment and intrinsics",
        )
    return scale


def _planning_scales(ctx: SkillContext) -> tuple[float, ...]:
    accept = ctx.workcell.calibration
    count, floor = int(accept["ring_attempts"]), float(accept["min_ring_scale"])
    return tuple(1.0 - (1.0 - floor) * i / max(1, count - 1) for i in range(count))


def _ring_targets(
    ctx: SkillContext, start: Pose, pivot: Vec3, args: Mapping[str, Any],
) -> tuple[list[Pose], dict[str, Any], float, int]:
    accept = ctx.workcell.calibration
    tilt = min(float(args["tilt_deg"]), ctx.workcell.tool_cone_deg - tool_cone_deg(start))
    if tilt < accept["min_tilt_deg"]:
        raise SkillError(
            FAILURE_UNREACHABLE,
            f"the tool may tilt only {max(tilt, 0.0):.1f} deg more here (tool_cone_deg "
            f"{ctx.workcell.tool_cone_deg:.0f}); start with the tool closer to straight down",
        )
    count = int(args["poses"])
    required = max(int(accept["min_hand_eye_poses"]), count // 2)
    for scale in _planning_scales(ctx):
        effective = {**args, **{key: float(args[key]) * scale for key in ("yaw_deg", "spread_mm", "lift_mm")},
                     "tilt_deg": max(float(accept["min_tilt_deg"]), tilt * scale)}
        targets, reason = _reachable(ctx, _ring(start, pivot, count, tilt=effective["tilt_deg"], args=effective))
        if len(targets) >= required:
            return targets, effective, scale, required
    raise SkillError(
        FAILURE_OUT_OF_WORKSPACE,
        f"only {len(targets)} of {count} calibration poses are reachable (need {required}): "
        f"{reason}. Start nearer the middle of the bootstrap box, or lower spread_mm / tilt_deg",
    )


# ── calib.handeye ────────────────────────────────────────────────────────────


class HandEye:
    """Measure the camera mount from board observations over a ring of arm poses."""

    spec = SkillSpec(
        skill_id="calib.handeye",
        version="1.3.0",
        card=(
            "Calibrate and verify where the camera sits (hand-eye). The workcell's calibration board "
            "must be in view: held in the gripper or fixed to the flange for a fixed camera, "
            "lying still in the cell for a wrist camera. The arm visits `poses` poses "
            "around the current one -- tilted up to tilt_deg, yawed up to yaw_deg, shifted "
            "up to spread_mm -- slowly, observing the board at each, then returns. Start "
            "with the board centred in the image. A separate grid measures depth accuracy "
            "before the calibration is installed; keep the board in the same mount."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "poses": {"type": "integer", "minimum": 8, "maximum": 24, "default": 12},
                "tilt_deg": _number(5, 30, 15.0),
                "yaw_deg": _number(0, 60, 30.0),
                "spread_mm": _number(0, 80, 30.0),
                "verify_spacing_mm": _number(10, 80, 40.0),
                "lift_mm": _number(0, 50, 15.0),
                "camera": {"type": "string", "pattern": "^camera(\\.[a-z0-9_]+)*$", "default": CAMERA_FRAME},
                "settle_s": _number(0, 5, 0.5),
                "speed_pct": _number(1, 100, 10.0),
            },
        },
        effect=EFFECT_ACTUATE,
        requires=(f"{REQUIRE_ROLE}arm",),
        consent=CONSENT_EACH_RUN,
        region=REGION_BOOTSTRAP_UNTIL_MEASURED,
        provides=(f"{REQUIRE_CALIBRATION}{{camera}}", f"{REQUIRE_RECORD}{{camera}}.verify"),
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        role = _camera(ctx, args)
        procedure = _procedure(ctx, role)
        mounting = _mounting(ctx, role)
        state = await _ready_state(ctx)
        seen = await _vision(ctx, role).observe_target()
        target_scale = _target_scale(ctx, seen)
        start = state.tcp
        if mounting == EYE_TO_HAND:
            pivot = start.position
        else:
            reach = norm(seen.center_in_camera)
            pivot = add(state.flange.position, [v * reach for v in state.flange.tool_axis])
        targets, effective, ring_scale, required = _ring_targets(ctx, start, pivot, args)
        calibration = ctx.workcell.calibration
        training = [_flange_at(state, pose).to_dict() for pose in targets]
        grid, spacing = _heldout_grid(ctx, state, float(args["verify_spacing_mm"]), training)
        return SkillPlan(
            skill_id=self.spec.skill_id,
            args=args,
            summary=(
                f"Hand-eye calibrate the {mounting.replace('_', '-')} camera: visit "
                f"{len(targets)} poses (tilt up to {effective['tilt_deg']:.0f} deg, yaw up to "
                f"{effective['yaw_deg']:.0f} deg, ring scale {ring_scale:.0%}) around {_fmt(pivot)} mm at "
                f"{float(args['speed_pct']):.0f}% speed, observing the board at each, then "
                f"verify at {len(grid)} held-out points ({spacing:g} mm spacing), then return -- "
                "keep the area clear and the board still"
                + ("; board scale unchecked (no usable depth)" if target_scale is None else "")
            ),
            targets=_with_return(ctx, targets + grid, start),
            limits={"speed_pct": float(args["speed_pct"])},
            basis={
                "procedure": procedure,
                "camera_role": role,
                "camera_device": ctx.workcell.role(role),
                "mounting": mounting,
                "start": start.to_dict(),
                "settle_s": float(args["settle_s"]),
                "samples": len(targets) + len(grid),
                "training_samples": len(targets),
                "ring_scale": ring_scale,
                "effective_args": effective,
                "verify_spacing_mm": spacing,
                "target_scale": target_scale,
                "required": required,
                "max_p95_mm": calibration["max_verify_p95_mm"],
                "max_rms_mm": calibration["max_verify_rms_mm"],
                "max_translation_rms_mm": calibration["max_hand_eye_translation_rms_mm"],
                "max_rotation_rms_deg": calibration["max_hand_eye_rotation_rms_deg"],
                "seen": seen.to_dict(),
            },
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        await _check_start(ctx, Pose.from_mapping(plan.basis["start"]))
        indexed, _ = await _sweep(ctx, plan)
        training_count = int(plan.basis["training_samples"])
        seen = [sample for index, sample in indexed if index < training_count]
        heldout = [sample for index, sample in indexed if index >= training_count]
        skipped = training_count - len(seen)
        required = int(plan.basis["required"])
        if len(seen) < required:
            raise _check_failed(
                f"the board was seen in only {len(seen)} of {training_count} training poses "
                f"(need {required}); keep it in view and lit evenly, or reduce tilt_deg",
                problems=["board not seen often enough"],
                skipped=skipped,
            )
        intrinsics = seen[0][1].intrinsics
        if any(observation.intrinsics != intrinsics for _, observation in seen + heldout):
            raise _check_failed(
                "the camera stream changed during calibration; run it again",
                problems=["intrinsics changed"],
            )
        mounting = plan.basis["mounting"]
        mount, target_mount, fit = _solve(mounting, seen)
        limit_mm = float(plan.basis["max_translation_rms_mm"])
        limit_deg = float(plan.basis["max_rotation_rms_deg"])
        if fit.translation_rms_mm > limit_mm or fit.rotation_rms_deg > limit_deg:
            raise _check_failed(
                f"the observations disagree by {fit.translation_rms_mm:.2f} mm / "
                f"{fit.rotation_rms_deg:.3f} deg rms (limits {limit_mm} mm / {limit_deg} deg); "
                "check the board is rigid and did not slip, and its dimensions match "
                f"calibration_target. {_fit_advice(plan)}",
                problems=["fit residual above limit"],
                fit=fit.to_dict(),
            )
        training = [flange.to_dict() for flange, _ in seen]
        if any(_overlaps_training(flange, training) for flange, _ in heldout):
            raise _check_failed("verification repeated a training pose", problems=["not held-out"])
        stats = _verification_stats(
            ctx, plan, mount, target_mount, heldout,
            int(plan.basis["samples"]) - training_count - len(heldout),
        )
        reprojection = max(observation.reprojection_px for _, observation in seen)
        recorded_at = ctx.record_calibration(
            plan.basis["procedure"],
            matrix=mount.to_matrix(),
            parameters={
                "mounting": mounting,
                "camera_role": plan.basis.get("camera_role", CAMERA_FRAME),
                "camera_device": plan.basis.get("camera_device", ctx.workcell.role(CAMERA_FRAME)),
                "geometry_revision": 2,
                "units": "mm",
                "orientation_convention": "Rz@Ry@Rx;degrees",
                "training_poses": [flange.to_dict() for flange, _ in seen],
                "intrinsics": dict(intrinsics),
                "target_mount": target_mount.to_matrix(),
                "fit": fit.to_dict(),
                "skipped": skipped,
                "reprojection_px_max": round(reprojection, 3),
            },
        )
        verified_at = _record_verification(
            ctx, str(plan.basis["camera_role"]), recorded_at, plan, stats, heldout
        )
        camera = mount.to_pose()
        return SkillOutcome.succeeded(
            {
                "camera": camera.to_dict(),
                "mounting": mounting,
                "fit": fit.to_dict(),
                "samples": len(seen),
                "skipped": skipped,
                "recorded_at": recorded_at,
                "verification": {**stats, "recorded_at": verified_at},
            },
            message=(
                f"The camera sits at {_fmt(camera.position)} mm in the "
                f"{'base' if mounting == EYE_TO_HAND else 'flange'} frame "
                f"({fit.translation_rms_mm:.2f} mm / {fit.rotation_rms_deg:.3f} deg rms over "
                f"{len(seen)} poses); held-out depth accuracy is {stats['p95_mm']:.1f} mm "
                "at p95. calib.verify can recheck this version later."
            ),
        )


def _fit_advice(plan: SkillPlan) -> str:
    schema = HandEye.spec.args_schema["properties"]
    effective = plan.basis["effective_args"]
    poses = min(schema["poses"]["maximum"], int(plan.args["poses"]) + schema["poses"]["minimum"] // 2)
    settle = min(schema["settle_s"]["maximum"], max(schema["settle_s"]["default"], float(plan.args["settle_s"])) * 2)
    return (f"After checking the setup, consider poses={poses}, settle_s={settle:g}, "
            f"tilt_deg={effective['tilt_deg']:g}, yaw_deg={effective['yaw_deg']:g}; "
            "no automatic retry is attempted")


def _rotation_about(axis_deg: float, tilt_deg: float) -> Mat3:
    """A tilt of *tilt_deg* about the horizontal axis at azimuth *axis_deg*."""
    return mat_mul(
        mat_mul(rpy_to_matrix(0.0, 0.0, axis_deg), rpy_to_matrix(tilt_deg, 0.0, 0.0)),
        rpy_to_matrix(0.0, 0.0, -axis_deg),
    )


def _ring(
    start: Pose, pivot: Vec3, count: int, *, tilt: float, args: Mapping[str, Any]
) -> list[Pose]:
    """Poses turned about *pivot*: tilts about every azimuth, alternating full and
    half tilt, with a yaw and a lift that vary independently of the tilt so the
    motions span three rotation axes and keep the board near the image centre."""
    spread, lift, yaw = float(args["spread_mm"]), float(args["lift_mm"]), float(args["yaw_deg"])
    poses: list[Pose] = []
    for i in range(count):
        azimuth = 360.0 * i / count
        phi = math.radians(azimuth)
        turn = mat_mul(
            _rotation_about(azimuth, tilt if i % 2 == 0 else tilt / 2.0),
            rpy_to_matrix(0.0, 0.0, yaw * math.cos(3.0 * phi)),
        )
        lever = mat_vec(turn, sub(start.position, pivot))
        offset = (spread * math.cos(phi), spread * math.sin(phi), lift if i % 2 else 0.0)
        position = add(add(pivot, lever), offset)
        poses.append(Pose(*position, *matrix_to_rpy(mat_mul(turn, start.rotation))))
    return poses


def _solve(mounting: str, seen: Sequence[Sample]) -> tuple[Transform, Transform, Any]:
    """``(camera mount, board mount, fit)`` for either mounting."""
    arm = [Transform.from_pose(flange) for flange, _ in seen]
    camera = [observation.transform for _, observation in seen]
    if mounting == EYE_IN_HAND:
        camera = [transform.inverse() for transform in camera]
    try:
        fit = solve_hand_eye(arm, camera)
    except ValueError as exc:
        raise _check_failed(
            f"the camera mount is not observable from these poses ({exc})",
            problems=["degenerate poses"],
        ) from exc
    if mounting == EYE_TO_HAND:
        return fit.z, fit.x, fit  # T_base_camera, T_flange_board
    return fit.x, fit.z, fit  # T_flange_camera, T_base_board


# ── calib.screws ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Still:
    """One still: where the arm was, and the surface the camera measured."""

    flange: Pose
    joints: Mapping[str, float]
    cloud: Any
    intrinsics: Mapping[str, Any]


async def _still(
    ctx: SkillContext,
    vision: Any,
    purpose: str,
    *,
    near_mm: float,
    far_mm: float,
    frames: int,
) -> _Still:
    """Capture one still -- *frames* clouds -- and refuse a frame the arm moved in.

    One frame carries a millimetre of depth noise, and the mount's weak direction
    is the camera's height, which only one joint constrains; stacking the frames
    of a still the arm is holding costs a few milliseconds and averages that
    noise out before any grid sees it. The depth window goes to the camera so the
    stride spends its budget on the arm rather than on the room behind it.
    """
    accept = ctx.workcell.calibration
    before = await ctx.motion.state()
    points: list[Any] = []
    intrinsics: Mapping[str, Any] = {}
    for _ in range(max(1, int(frames))):
        observation = await vision.observe_cloud(near_mm=near_mm, far_mm=far_mm)
        intrinsics = observation.intrinsics
        points.extend(observation.points)
    after = await ctx.motion.state()
    if (before.flange.distance_mm(after.flange) > accept["start_drift_mm"]
            or before.flange.rotation_to_deg(after.flange) > accept["start_drift_deg"]):
        raise _check_failed(f"the flange moved during the {purpose} capture", problems=["capture drift"])
    return _Still(
        flange=after.flange,
        joints=after.joints,
        cloud=tuple(points),
        intrinsics=intrinsics,
    )


def _other_joints_moved(
    first: Mapping[str, float], second: Mapping[str, float], channel: str
) -> list[str]:
    """Joints that differ between two stills other than the one being swung.

    A screw fit assumes the pair differs in exactly one joint. Anything else that
    moved -- a halt that let a joint sag, a hand on the arm -- would be folded
    into that joint's axis and would land in the mount as a plausible error.
    """
    return [
        name
        for name, value in first.items()
        if name != channel and abs(float(second.get(name, value)) - float(value)) > JOINT_MATCH_DEG
    ]


def _joint_channels(joints: Mapping[str, float]) -> dict[str, str]:
    """Label (``j1``) to the channel id this arm reports that joint under."""
    return {channel.split(".")[-2]: channel for channel in joints}


def _named_joint(joints: Mapping[str, float], label: str) -> str:
    """The channel id *label* names (label or id), or a refusal listing the arm's joints."""
    if label in joints:
        return label
    available = _joint_channels(joints)
    if label not in available:
        raise SkillError(
            FAILURE_INVALID_ARGS,
            f"{label!r} is not a joint of this arm; it reports {', '.join(sorted(available))}",
        )
    return available[label]


def _stand(
    ctx: SkillContext, state: ArmState, requested: Mapping[str, Any]
) -> tuple[dict[str, float], Pose, dict[str, float]]:
    """The joints the swings start from, the tool pose once there, and the transit.

    A measurement posture is part of the measurement. The arm may be resting
    wherever the previous job, a halt, or a hand left it, and a swing from there
    can leave the cell or add tilt the cone forbids -- which leaves the mount
    unmeasurable through no fault of the camera. Naming a stand posture moves
    that decision into the plan a human approves, and the transit is checked by
    the very joint-move policy that will execute it.
    """
    targets: dict[str, float] = {}
    for label, value in requested.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SkillError(
                FAILURE_INVALID_ARGS,
                f"stand joint {label!r} must be a number of degrees",
            )
        targets[_named_joint(state.joints, str(label))] = float(value)
    if not targets:
        return dict(state.joints), state.tcp, {}
    try:
        tcp = ctx.motion.check_joints(targets, state.joints)
    except SkillError as error:
        raise SkillError(
            error.failure_code, f"the arm cannot stand where the plan asks: {error.message}"
        ) from error
    return {**state.joints, **targets}, tcp, {
        channel: round(value, 3) for channel, value in targets.items()
    }


def _excursions(
    ctx: SkillContext, joints: Mapping[str, float], names: Sequence[str], excursion_deg: float
) -> tuple[list[dict[str, Any]], str]:
    """One +-excursion pair per named joint that the workspace check allows from *joints*."""
    moves: list[dict[str, Any]] = []
    reason = ""
    for label in names:
        channel = _named_joint(joints, label)
        start = float(joints[channel])
        try:
            plus = ctx.motion.check_joints({channel: start + excursion_deg}, joints)
            minus = ctx.motion.check_joints({channel: start - excursion_deg}, joints)
        except SkillError as error:
            reason = reason or error.message
            continue
        moves.append(
            {
                "joint": channel.split(".")[-2],
                "channel": channel,
                "start_deg": round(start, 3),
                "plus_deg": round(start + excursion_deg, 3),
                "minus_deg": round(start - excursion_deg, 3),
                "plus_tcp": plus.to_dict(),
                "minus_tcp": minus.to_dict(),
            }
        )
    return moves, reason


class Screws:
    """Measure a fixed camera's mount from single-joint motions, with no board."""

    spec = SkillSpec(
        skill_id="calib.screws",
        version="1.1.0",
        card=(
            "Calibrate where a fixed camera sits without a calibration board. The arm "
            "swings `joints` one at a time, `excursion_deg` each way from `stand` -- a "
            "joint posture it transits to first when one is given, checked like any "
            "joint move -- or from where it is now. The camera watches the surfaces "
            "that joint carries: at the right mount the firmware's own rotation lands "
            "them on the surface it saw, and the residual is reported per joint in "
            "millimetres. The base joint's swing is what measures the vertical, so "
            "neither a table nor a board has to be in view; the arm does have to be "
            "unobstructed. Needs at least three joints with different axes: the base "
            "joint, plus one horizontal."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "joints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 4,
                    "default": ["j1", "j4", "j5"],
                },
                "stand": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": (
                        "Joint posture to swing from, as {j1: deg, ...} (labels or channel "
                        "ids); the arm transits there before the first still and stays "
                        "there between swings"
                    ),
                },
                "excursion_deg": _number(5, 30, 15.0),
                "near_mm": _number(100, 600, 200.0),
                "far_mm": _number(300, 2000, 750.0),
                "camera": {"type": "string", "pattern": "^camera(\\.[a-z0-9_]+)*$", "default": CAMERA_FRAME},
                "frames": _number(1, 6, 3),
                "settle_s": _number(0, 5, 0.5),
                "speed_pct": _number(1, 100, 10.0),
            },
        },
        effect=EFFECT_ACTUATE,
        requires=(f"{REQUIRE_ROLE}arm",),
        consent=CONSENT_EACH_RUN,
        region=REGION_BOOTSTRAP_UNTIL_MEASURED,
        provides=(f"{REQUIRE_CALIBRATION}{{camera}}",),
    )

    def provides_in(self, workcell: WorkcellView, frames: FrameView) -> tuple[str, ...]:
        """The fixed cameras of *workcell*: only a camera on the base can be measured here."""
        return tuple(
            f"{REQUIRE_CALIBRATION}{role}"
            for role in workcell.roles
            if role.split(".", 1)[0] == CAMERA_FRAME and frames.parent(role) == "base"
        )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        role = _camera(ctx, args)
        if _mounting(ctx, role) != EYE_TO_HAND:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                "calib.screws measures a camera fixed to the cell; a wrist camera moves "
                "with the arm it would be watching",
            )
        procedure = _procedure(ctx, role)
        state = await _ready_state(ctx)
        posture, stand_tcp, transit = _stand(ctx, state, args.get("stand") or {})
        excursion = float(args["excursion_deg"])
        names = [str(name) for name in args["joints"]]
        moves, reason = _excursions(ctx, posture, names, excursion)
        needed = int(ctx.workcell.calibration["min_screw_pairs"])
        if len(moves) < needed:
            where = "the stand posture" if transit else "where the arm is now"
            raise SkillError(
                FAILURE_OUT_OF_WORKSPACE,
                f"only {len(moves)} of {len(names)} joints can swing {excursion:.0f} deg from "
                f"{where} (need {needed}): {reason}. Name a stand posture to measure from "
                "(stand: {j2: 74, j3: -42, ...}), move nearer the middle of the workspace, "
                "or lower excursion_deg",
            )
        start = state.tcp
        targets = [
            Pose.from_mapping(move[key])
            for move in moves
            for key in ("plus_tcp", "minus_tcp")
        ]
        calibration = ctx.workcell.calibration
        frames = int(args["frames"])
        stills = 2 * len(moves) + 1
        moving_to = (
            " after moving to the stand posture ("
            + ", ".join(
                f"{label} {float(value):g}" for label, value in sorted(args["stand"].items())
            )
            + ")"
            if transit
            else ""
        )
        return SkillPlan(
            skill_id=self.spec.skill_id,
            args=args,
            summary=(
                f"Screw-calibrate the fixed camera without a board: swing "
                f"{', '.join(move['joint'] for move in moves)} one at a time by "
                f"{excursion:.0f} deg each way{moving_to} from {_fmt(stand_tcp.position)} mm, "
                f"capturing {stills} stills of {frames} depth frame(s) each and returning "
                f"between swings -- keep the arm's sweep clear and unobstructed in the "
                f"camera's view"
            ),
            targets=_with_return(ctx, targets, stand_tcp),
            limits={"speed_pct": float(args["speed_pct"])},
            basis={
                "procedure": procedure,
                "camera_role": role,
                "camera_device": ctx.workcell.role(role),
                "mounting": EYE_TO_HAND,
                "start": start.to_dict(),
                "start_joints": {name: round(float(v), 3) for name, v in state.joints.items()},
                "stand": transit,
                "stand_tcp": stand_tcp.to_dict(),
                "settle_s": float(args["settle_s"]),
                "near_mm": float(args["near_mm"]),
                "far_mm": float(args["far_mm"]),
                "frames": int(args["frames"]),
                "moves": moves,
                "min_pairs": needed,
                "max_shift_gap_mm": calibration["max_screw_shift_gap_mm"],
                "min_agreement": calibration["min_screw_agreement"],
                "max_vertical_axis_deg": calibration["max_vertical_axis_deg"],
            },
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        await _check_start(ctx, Pose.from_mapping(plan.basis["start"]))
        role = str(plan.basis.get("camera_role", CAMERA_FRAME))
        vision = _vision(ctx, role)
        settle = float(plan.basis["settle_s"])
        speed = plan.limits.get("speed_pct")
        near, far = float(plan.basis["near_mm"]), float(plan.basis["far_mm"])
        frames = int(plan.basis["frames"])
        moves = list(plan.basis["moves"])
        await self._transit(ctx, plan, settle=settle, speed=speed)
        shots: dict[Any, _Still] = {
            "reference": await _still(
                ctx, vision, "reference", near_mm=near, far_mm=far, frames=frames
            )
        }
        for move in moves:
            for key in ("plus", "minus"):
                await ctx.motion.move_joints(
                    {move["channel"]: float(move[f"{key}_deg"])}, speed_pct=speed
                )
                if settle > 0:
                    await asyncio.sleep(settle)
                shots[(move["joint"], key)] = await _still(
                    ctx, vision, f"{move['joint']} {key}",
                    near_mm=near, far_mm=far, frames=frames,
                )
                await ctx.motion.move_joints(
                    {move["channel"]: float(move["start_deg"])}, speed_pct=speed
                )
        # The solve is seconds of numpy over the arm clouds, and it used to run on the
        # event loop. On the cell it blocked for ~25 s at once, which starved the node's
        # RPC plane and its lease watchdog together: the host read the run as unreadable
        # and the deadman halted the arm, losing a measurement that had already been
        # captured. Nothing here needs the loop, so it waits in a worker thread.
        fit, evidence = await asyncio.to_thread(self._measure, plan, shots, moves)
        return self._record(ctx, plan, role, fit, evidence)

    async def _transit(
        self, ctx: SkillContext, plan: SkillPlan, *, settle: float, speed: float | None
    ) -> None:
        """Move to the planned stand posture, or refuse to measure from the wrong one.

        The swings are absolute joint targets computed from the stand posture, so
        an arm that stopped short of it would be swung from a place the plan never
        checked. The comparison uses the workcell's own drift tolerances -- the same
        ones every start check in this file uses.
        """
        transit = dict(plan.basis.get("stand") or {})
        if not transit:
            return
        await ctx.motion.move_joints(transit, speed_pct=speed)
        expected = Pose.from_mapping(plan.basis["stand_tcp"])
        here = (await ctx.motion.state()).tcp
        moved_mm, moved_deg = here.distance_mm(expected), here.rotation_to_deg(expected)
        accept = ctx.workcell.calibration
        if moved_mm > accept["start_drift_mm"] or moved_deg > accept["start_drift_deg"]:
            raise _check_failed(
                f"the arm stopped {moved_mm:.1f} mm / {moved_deg:.1f} deg from the stand "
                "posture the plan measured from; clear the way and plan again",
                problems=["stand posture not reached"],
            )
        if settle > 0:
            await asyncio.sleep(settle)

    def _measure(
        self,
        plan: SkillPlan,
        shots: Mapping[Any, _Still],
        moves: Sequence[Mapping[str, Any]],
    ) -> tuple[Any, dict[str, Any]]:
        """Solve the mount from the stills, or explain why they cannot determine it."""
        intrinsics = shots["reference"].intrinsics
        if any(still.intrinsics != intrinsics for still in shots.values()):
            raise _check_failed(
                "the camera stream changed during calibration; run it again",
                problems=["intrinsics changed"],
            )
        evidence: list[JointEvidence] = []
        for move in moves:
            first = shots[(move["joint"], "plus")]
            second = shots[(move["joint"], "minus")]
            stray = _other_joints_moved(first.joints, second.joints, move["channel"])
            if stray:
                raise _check_failed(
                    f"{move['joint']} was the joint being swung, but "
                    f"{', '.join(name.split('.')[-2] for name in stray)} moved too; "
                    "hold the arm clear and run it again",
                    problems=["more than one joint moved between a pair"],
                )
            try:
                screw = screw_from_poses(first.flange, second.flange)
                evidence.append(
                    JointEvidence(move["joint"], screw, first.cloud, second.cloud, ScrewConfig())
                )
            except ValueError as exc:
                raise _check_failed(
                    f"{move['joint']}: {exc}",
                    problems=["that joint's pair carries no usable motion"],
                ) from exc
        try:
            up = solve_vertical(
                evidence, max_axis_deg=float(plan.basis["max_vertical_axis_deg"])
            )
        except ValueError as exc:
            raise _check_failed(
                str(exc), problems=["no joint measures the vertical"]
            ) from exc
        try:
            fit = solve_mount(evidence, up=up)
        except ValueError as exc:
            raise _check_failed(str(exc), problems=["degenerate joints"]) from exc
        limit_gap = float(plan.basis["max_shift_gap_mm"])
        min_agreement = float(plan.basis["min_agreement"])
        if fit.max_shift_gap_mm > limit_gap or fit.score < min_agreement:
            raise _check_failed(
                f"no single mount explains the joints: the worst joint is "
                f"{fit.max_shift_gap_mm:.1f} mm off (limit {limit_gap}) and the mean "
                f"agreement is {fit.score:.3f} (minimum {min_agreement}); check nothing "
                "else in the cell moved between captures and the arm stays fully in view",
                problems=["joints disagree with one mount"],
                fit=fit.to_dict(),
            )
        summary = {
            "method": METHOD_SCREW,
            "intrinsics": dict(intrinsics),
            "vertical": [round(value, 6) for value in up],
            **fit.to_dict(),
        }
        return fit, summary

    def _record(
        self,
        ctx: SkillContext,
        plan: SkillPlan,
        role: str,
        fit: Any,
        evidence: dict[str, Any],
    ) -> SkillOutcome:
        camera = fit.camera.to_pose()
        recorded_at = ctx.record_calibration(
            str(plan.basis["procedure"]),
            matrix=fit.camera.to_matrix(),
            parameters={
                "mounting": EYE_TO_HAND,
                "camera_role": role,
                "camera_device": plan.basis.get("camera_device", ctx.workcell.role(role)),
                "method": METHOD_SCREW,
                "geometry_revision": 2,
                "units": "mm",
                "orientation_convention": "Rz@Ry@Rx;degrees",
                "intrinsics": evidence["intrinsics"],
                "screw_evidence": {key: value for key, value in evidence.items() if key != "intrinsics"},
                "captures": len(plan.basis["moves"]) * 2 + 1,
                "near_mm": plan.basis["near_mm"],
                "far_mm": plan.basis["far_mm"],
            },
        )
        worst = max(fit.joints, key=lambda joint: joint.shift_gap_mm)
        return SkillOutcome.succeeded(
            {
                "camera": camera.to_dict(),
                "mounting": EYE_TO_HAND,
                "method": METHOD_SCREW,
                "vertical": evidence["vertical"],
                "score": evidence["score"],
                "max_shift_gap_mm": evidence["max_shift_gap_mm"],
                "pairs": evidence["pairs"],
                "captures": len(plan.basis["moves"]) * 2 + 1,
                "frames": int(plan.basis["frames"]),
                "recorded_at": recorded_at,
            },
            message=(
                f"The camera sits at {_fmt(camera.position)} mm in the base frame, measured "
                f"without a board from {len(fit.joints)} joints; the worst of them "
                f"({worst.label}) leaves {worst.shift_gap_mm:.1f} mm unexplained. "
                "Accuracy is not yet verified; calib.attest can check this frame against "
                "an independently verified reference."
            ),
        )


# ── calib.ball ───────────────────────────────────────────────────────────────


class CarriedBall:
    """Measure a fixed camera's mount from a ball the arm carries."""

    spec = SkillSpec(
        skill_id="calib.ball",
        version="1.0.0",
        card=(
            "Calibrate where a fixed camera sits by carrying a ball of known radius -- no "
            "board, no table in view, and no TCP measurement. The gripper must already "
            "hold the ball, and `radius_mm` is the radius the jaws measured, not a nominal "
            "one. The arm visits `poses` poses around the current one -- tilted up to "
            "tilt_deg, yawed up to yaw_deg, shifted up to spread_mm, lifted up to "
            "lift_mm -- and at every stop the camera fits the ball's surface to a sphere. "
            "The mount is then the motion that carries every measured centre onto the "
            "centre the arm is holding, with the ball's own place in the gripper solved as "
            "the ninth unknown. The `heldout` poses are measured but kept out of the "
            "solve: what the fitted mount misses them by is the accuracy the record "
            "carries. The ball has to stay in the camera's view throughout, so sweep "
            "where it is already visible."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "poses": {"type": "integer", "minimum": 9, "maximum": 24, "default": 14},
                "heldout": {"type": "integer", "minimum": 3, "maximum": 8, "default": 4},
                "radius_mm": _number(10, 80, 30.0),
                "tilt_deg": _number(0, 30, 15.0),
                "yaw_deg": _number(0, 60, 25.0),
                "spread_mm": _number(0, 80, 50.0),
                "lift_mm": _number(0, 50, 10.0),
                "camera": {"type": "string", "pattern": "^camera(\\.[a-z0-9_]+)*$", "default": CAMERA_FRAME},
                "frames": _number(1, 6, 3),
                "settle_s": _number(0, 5, 0.5),
                "speed_pct": _number(1, 100, 10.0),
            },
        },
        effect=EFFECT_ACTUATE,
        requires=(f"{REQUIRE_ROLE}arm",),
        consent=CONSENT_EACH_RUN,
        region=REGION_BOOTSTRAP_UNTIL_MEASURED,
        provides=(f"{REQUIRE_CALIBRATION}{{camera}}",),
        # The sweep starts from the mount that is already measured: it tightens a
        # measurement, it does not create the first one. Advertising it for a
        # missing or unproven frame would point a caller at a skill that refuses on
        # its first line, so only the states that carry a measured mount it could
        # improve are offered here. A merely inaccurate or drifted frame still has
        # that mount in its latest record, which is what seeds the search.
        provides_when=(
            FrameStatus.INSUFFICIENT_ACCURACY.value,
            FrameStatus.DRIFT_DETECTED.value,
        ),
    )

    def provides_in(self, workcell: WorkcellView, frames: FrameView) -> tuple[str, ...]:
        """The fixed cameras of *workcell*: only a camera on the base can be measured here."""
        return tuple(
            f"{REQUIRE_CALIBRATION}{role}"
            for role in workcell.roles
            if role.split(".", 1)[0] == CAMERA_FRAME and frames.parent(role) == "base"
        )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        role = _camera(ctx, args)
        if _mounting(ctx, role) != EYE_TO_HAND:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                "calib.ball measures a camera fixed to the cell; a wrist camera rides the "
                "arm it would be watching, so carrying a ball past it measures nothing",
            )
        procedure = _procedure(ctx, role)
        seed = _ball_seed(ctx.frames, role)
        if seed is None:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                f"{role} has no measured mount to start the search from; measure it "
                "once with calib.handeye or calib.screws, and use calib.ball to tighten it",
            )
        state = await _ready_state(ctx)
        accept = ctx.workcell.calibration
        total, held_out = int(args["poses"]), int(args["heldout"])
        need_train = int(accept["min_ball_poses"])
        need_held = int(accept["min_ball_held_out"])
        if total - held_out < need_train or held_out < need_held:
            raise SkillError(
                FAILURE_INVALID_ARGS,
                f"{total} poses with {held_out} held out leaves {total - held_out} to fit "
                f"the mount (need {need_train}) and {held_out} to check it (need {need_held}); "
                "raise poses or lower heldout",
            )
        # The ring turns the tool about the *flange*, not about the TCP. A carried
        # ball hangs between 30 and 60 mm past the jaws, so turning about the TCP
        # moves it a few millimetres while the tool turns -- and the mount is read
        # off exactly that travel, against a cloud noisy to several millimetres. On
        # the 24-pose sweep of 2026-11-26 the TCP pivot left 3 sigma at 171 mm for
        # a ball that moved 85 mm; turning about the flange puts the ball on its
        # full 156 mm lever (175 mm of travel at 30 deg) and the same noise then
        # supports a 3 mm bound. Reachability is unchanged in kind, only in range.
        ring = _ring(state.tcp, state.flange.position, total,
                     tilt=float(args["tilt_deg"]), args=args)
        kept: list[Pose] = []
        reason = ""
        for pose in ring:
            try:
                ctx.motion.check_target(pose)
            except SkillError as error:
                reason = reason or error.message
                continue
            kept.append(pose)
        marks = _spread_marks(len(kept), held_out)
        fitted_poses = sum(1 for mark in marks if not mark)
        if fitted_poses < need_train or len(kept) - fitted_poses < need_held:
            raise SkillError(
                FAILURE_OUT_OF_WORKSPACE,
                f"only {len(kept)} of {total} poses are reachable, leaving {fitted_poses} to "
                f"fit and {len(kept) - fitted_poses} to check (need {need_train} and "
                f"{need_held}): {reason}. Start nearer the middle of the workspace, or "
                "lower spread_mm / tilt_deg",
            )
        radius = float(args["radius_mm"])
        start = state.tcp
        marks_label = sum(1 for mark in marks if mark)
        return SkillPlan(
            skill_id=self.spec.skill_id,
            args=args,
            summary=(
                f"Measure the fixed camera with a ball of radius {radius:.1f} mm the gripper "
                f"is holding: visit {len(kept)} poses around {_fmt(start.position)} mm "
                f"(tilt up to {float(args['tilt_deg']):.0f} deg, spread up to "
                f"{float(args['spread_mm']):.0f} mm), sphere-fit it at each, and keep "
                f"{marks_label} of them out of the fit as the accuracy check; the ball must "
                "stay in the camera's view the whole way"
            ),
            targets=_with_return(ctx, kept, start),
            limits={"speed_pct": float(args["speed_pct"])},
            basis={
                "procedure": procedure,
                "camera_role": role,
                "camera_device": ctx.workcell.role(role),
                "mounting": EYE_TO_HAND,
                "seed_unproven": ctx.frames.transform(role) is None,
                "seed_translation_mm": [round(float(value), 3) for value in seed.translation],
                "start": start.to_dict(),
                "radius_mm": radius,
                "poses": [
                    {"tcp": pose.to_dict(), "heldout": bool(mark)}
                    for pose, mark in zip(kept, marks, strict=True)
                ],
                "collect": len(kept),
                "min_train": need_train,
                "min_held_out": need_held,
                "settle_s": float(args["settle_s"]),
                "frames": int(args["frames"]),
                "depth_margin_mm": accept["ball_depth_margin_mm"],
                "window_extra_mm": accept["ball_window_extra_mm"],
                "max_fit_rms_mm": accept["max_ball_fit_rms_mm"],
                "max_radius_error_mm": accept["max_ball_radius_error_mm"],
                "max_residual_mm": accept["max_ball_residual_mm"],
            },
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        await _check_start(ctx, Pose.from_mapping(plan.basis["start"]))
        role = str(plan.basis["camera_role"])
        vision = _vision(ctx, role)
        seed = _ball_seed(ctx.frames, role)
        if seed is None:
            raise SkillError(FAILURE_PLAN_STALE, f"{role} was uninstalled inside the run; plan again")
        radius = float(plan.basis["radius_mm"])
        settle, speed = float(plan.basis["settle_s"]), plan.limits.get("speed_pct")
        frames = int(plan.basis["frames"])
        margin, extra = float(plan.basis["depth_margin_mm"]), float(plan.basis["window_extra_mm"])
        max_rms = float(plan.basis["max_fit_rms_mm"])
        max_radius = float(plan.basis["max_radius_error_mm"])
        samples: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        hint: Any = None
        for entry in plan.basis["poses"]:
            pose = Pose.from_mapping(entry["tcp"])
            await ctx.motion.move_tcp(pose, speed_pct=speed)
            if settle > 0:
                await asyncio.sleep(settle)
            # Each still is predicted from the recorded mount, which is wrong by the
            # amount this run exists to remove -- tens of millimetres -- while the
            # previous still's centre is wrong by wherever the arm went since, which
            # on a ring about the flange is most of the lever. The mount is the
            # better guess, and the previous centre is kept as the second try.
            predicted = _as_seen(seed, pose.position)
            reach = norm(predicted)
            still = await _still(
                ctx,
                vision,
                "ball",
                near_mm=max(50.0, reach - radius - margin),
                far_mm=reach + radius + margin,
                frames=frames,
            )
            reason = ""
            guesses = [predicted] + ([hint] if hint is not None and hint != predicted else [])
            ball = None
            for guess in guesses:
                try:
                    ball = fit_ball(
                        still.cloud, radius_mm=radius, hint=guess, window_mm=radius + extra
                    )
                    break
                except ValueError as error:
                    reason = str(error)
            if ball is None:
                # A mount the workcell calls inaccurate may sit far enough out that
                # no ordinary window sees the ball at all. Searching the still's
                # whole depth shell costs one pass and admits nothing: the radius,
                # the fit and the radius agreement still decide what may be used.
                try:
                    ball = fit_ball(
                        still.cloud, radius_mm=radius, hint=predicted,
                        window_mm=radius + margin,
                    )
                    reason = ""
                except ValueError as wider:
                    reason = str(wider)
            if ball is not None:
                if ball.rms_mm > max_rms:
                    reason = (
                        f"its surface fits a sphere to {ball.rms_mm:.1f} mm (limit {max_rms:g})"
                    )
                    ball = None
                elif abs(ball.fitted_radius_mm - radius) > max_radius:
                    reason = (
                        f"the ball measures {ball.fitted_radius_mm:.1f} mm where the jaws "
                        f"said {radius:.1f} mm (limit {max_radius:g}); it is deforming or slipping"
                    )
                    ball = None
            if ball is None:
                skipped.append({"pose": pose.to_dict(), "reason": reason})
                continue
            hint = ball.centre
            samples.append(
                {
                    "tcp": pose.to_dict(),
                    "flange": still.flange.to_dict(),
                    "centre": list(ball.centre),
                    "ball": ball.to_dict(),
                    "intrinsics": dict(still.intrinsics),
                    "heldout": bool(entry["heldout"]),
                }
            )
        # The joint solve is seconds of numpy over every pose, and running it on the
        # event loop starves the node's RPC plane and the lease watchdog together.
        solved = await asyncio.to_thread(self._solve, plan, samples, skipped)
        return self._record(ctx, plan, role, solved)

    def _solve(
        self, plan: SkillPlan, samples: Sequence[Mapping[str, Any]], skipped: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """The mount and the ball's place in the gripper, with what the held-out poses say."""
        train = [sample for sample in samples if not sample["heldout"]]
        held = [sample for sample in samples if sample["heldout"]]
        basis = plan.basis
        if len(train) < int(basis["min_train"]) or len(held) < int(basis["min_held_out"]):
            raise _check_failed(
                f"{len(train)} poses could be fitted and {len(held)} held out (need "
                f"{int(basis['min_train'])} and {int(basis['min_held_out'])}; {len(skipped)} "
                "skipped); keep the ball in the camera's view and run it again",
                problems=["too few usable stills"],
                skipped=list(skipped),
            )
        intrinsics = samples[0]["intrinsics"]
        if any(sample["intrinsics"] != intrinsics for sample in samples):
            raise _check_failed(
                "the camera stream changed during the sweep; run it again",
                problems=["intrinsics changed"],
            )
        try:
            fit = solve_tool_hand_eye(
                [Pose.from_mapping(sample["flange"]) for sample in samples],
                [tuple(sample["centre"]) for sample in samples],
            )
            training = solve_tool_hand_eye(
                [Pose.from_mapping(sample["flange"]) for sample in train],
                [tuple(sample["centre"]) for sample in train],
            )
        except ValueError as error:
            raise _check_failed(
                f"the carried ball does not determine the mount ({error}); sweep with more "
                "tilt and spread",
                problems=["degenerate poses"],
            ) from error
        errors = [
            residual_mm(training, Pose.from_mapping(sample["flange"]), tuple(sample["centre"]))
            for sample in held
        ]
        stats = _stats(errors, prefix="held_out_")
        sigma_mm = max(fit.translation_sigma_mm, training.translation_sigma_mm)
        sigma_deg = max(fit.rotation_sigma_deg, training.rotation_sigma_deg)
        achieved = Accuracy(
            max(float(stats["held_out_p95_mm"]), 3.0 * sigma_mm),
            max(3.0 * sigma_deg, 1e-3),
        )
        limit = float(basis["max_residual_mm"])
        if not achieved.translation_mm <= limit:
            # The bound is the larger of the held-out p95 and three sigma, and the two
            # say different things: a small p95 with a large sigma is a well-measured
            # sweep whose geometry hardly constrains the mount, while a large p95 with
            # a small sigma is a bad still. Both are reported with every centre and
            # every skip, because the verdict alone leaves the operator guessing which
            # of the two happened -- and the arm cannot be re-swept cheaply.
            raise _check_failed(
                f"the held-out poses miss by {achieved.translation_mm:.1f} mm (ceiling {limit:g}): "
                f"p95 {float(stats['held_out_p95_mm']):.1f} mm over {len(held)} held-out "
                f"poses and 3 sigma {3.0 * sigma_mm:.1f} mm from {len(train)} fitted ones; "
                "the sweep did not measure this camera -- check the ball is rigidly held, "
                "that nothing touched the camera, and that the arm is the same arm",
                problems=["held-out residual above the ceiling"],
                **stats,
                sigma_mm=round(sigma_mm, 3),
                residual_max_mm=round(fit.residual_max_mm, 3),
                condition_number=round(fit.condition_number, 3),
                outliers=int(fit.outliers),
                samples=[
                    {"tcp": sample["tcp"], "centre": sample["centre"],
                     "heldout": bool(sample["heldout"]),
                     "radius_mm": sample["ball"]["fitted_radius_mm"],
                     "rms_mm": sample["ball"]["rms_mm"]}
                    for sample in samples
                ],
                skipped=list(skipped),
            )
        radii = sorted(float(sample["ball"]["fitted_radius_mm"]) for sample in samples)
        evidence = {
            "method": METHOD_CARRIED_BALL,
            "radius_mm": float(basis["radius_mm"]),
            "samples": len(samples),
            "held_out": len(held),
            "skipped": len(skipped),
            **stats,
            "residual_rms_mm": round(fit.residual_rms_mm, 3),
            "residual_max_mm": round(fit.residual_max_mm, 3),
            "leave_one_out_max_mm": round(fit.held_out_max_mm, 3),
            "condition_number": round(fit.condition_number, 3),
            "translation_sigma_mm": round(fit.translation_sigma_mm, 3),
            "rotation_sigma_deg": round(fit.rotation_sigma_deg, 4),
            "ball_sigma_mm": round(fit.tool_sigma_mm, 3),
            "outliers": int(fit.outliers),
            "fitted_radius_mm": {
                "min": round(radii[0], 3),
                "max": round(radii[-1], 3),
            },
            "poses": [
                {
                    "tcp": sample["tcp"],
                    "flange": sample["flange"],
                    "centre": sample["centre"],
                    "heldout": bool(sample["heldout"]),
                    "ball": sample["ball"],
                }
                for sample in samples
            ],
            "skipped_poses": list(skipped),
        }
        return {
            "matrix": fit.mount.inverse().to_matrix(),
            "camera": fit.mount.inverse().to_pose(),
            "ball_in_flange_mm": [round(float(value), 3) for value in fit.tool],
            "achieved": achieved,
            "intrinsics": dict(intrinsics),
            "evidence": evidence,
            "training_flanges": [sample["flange"] for sample in train],
        }

    def _record(
        self, ctx: SkillContext, plan: SkillPlan, role: str, solved: Mapping[str, Any]
    ) -> SkillOutcome:
        achieved: Accuracy = solved["achieved"]
        camera: Pose = solved["camera"]
        evidence = dict(solved["evidence"])
        recorded_at = ctx.record_calibration(
            str(plan.basis["procedure"]),
            matrix=solved["matrix"],
            parameters={
                "mounting": EYE_TO_HAND,
                "camera_role": role,
                "camera_device": plan.basis["camera_device"],
                "method": METHOD_CARRIED_BALL,
                "seed_unproven": bool(plan.basis.get("seed_unproven", False)),
                "units": "mm",
                "orientation_convention": "Rz@Ry@Rx;degrees",
                "intrinsics": solved["intrinsics"],
                "ball_in_flange_mm": solved["ball_in_flange_mm"],
                "training_poses": list(solved["training_flanges"]),
                "captures": int(evidence["samples"]) * int(plan.basis["frames"]),
                "carried_ball_evidence": evidence,
                "achieved_accuracy": achieved.to_dict(),
            },
        )
        return SkillOutcome.succeeded(
            {
                "camera": camera.to_dict(),
                "mounting": EYE_TO_HAND,
                "method": METHOD_CARRIED_BALL,
                "radius_mm": evidence["radius_mm"],
                "ball_in_flange_mm": solved["ball_in_flange_mm"],
                "samples": evidence["samples"],
                "held_out": evidence["held_out"],
                "skipped": evidence["skipped"],
                "held_out_p95_mm": evidence["held_out_p95_mm"],
                "held_out_max_mm": evidence["held_out_max_mm"],
                "residual_rms_mm": evidence["residual_rms_mm"],
                "rotation_sigma_deg": evidence["rotation_sigma_deg"],
                "achieved": achieved.to_dict(),
                "recorded_at": recorded_at,
            },
            message=(
                f"The camera sits at {_fmt(camera.position)} mm in the base frame, measured "
                f"from a ball of radius {float(evidence['radius_mm']):.1f} mm the arm "
                f"carried; the {evidence['held_out']} held-out poses miss it by "
                f"{float(evidence['held_out_p95_mm']):.1f} mm at p95 (worst "
                f"{float(evidence['held_out_max_mm']):.1f} mm), and the bound the record "
                f"carries is {achieved.describe()}"
            ),
        )


def _ball_seed(frames: FrameView, role: str) -> Transform | None:
    """Where the carried-ball sweep starts searching for *role*.

    The installed mount when it holds, else the geometry the last record carries.
    ``calib.ball`` exists to tighten a mount the workcell calls inaccurate or
    drifted, and such a frame publishes no transform -- so falling back to the
    record's geometry is the difference between the skill working on the state it
    advertises and refusing on it. The seed is only a search hint: every measured
    centre is fitted independently inside a window far wider than the worst seed
    error, so where the seed lands never enters the solved mount.
    """
    return frames.transform(role) or frames.measured(role)


def _as_seen(mount: Transform, position: Sequence[float]) -> Vec3:
    """Where a point the arm reports in the base frame appears in the camera frame.

    ``frames.transform`` publishes ``T_base_camera`` -- camera points into the base,
    the direction ``calib.verify`` reads it -- so a TCP the arm reports is carried
    into the camera by the inverse. Applying the mount forwards puts the guess
    hundreds of millimetres away, outside any window, and no still is ever found.
    """
    return mount.inverse().apply(position)


def _spread_marks(count: int, marked: int) -> list[bool]:
    """*count* flags with *marked* of them spread through it, never two adjacent.

    The held-out poses have to be *drawn from* the sweep rather than appended to
    it: a ring whose last four poses are the check would leave the fit untested
    exactly where the ring ends.
    """
    flags = [False] * max(0, count)
    if marked <= 0 or count <= 0:
        return flags
    for step in range(min(marked, count)):
        flags[int((step + 0.5) * count / marked) % count] = True
    return flags


def _flange_at(state: ArmState, pose: Pose) -> Pose:
    tcp_to_flange = Transform.from_pose(state.tcp).inverse().compose(Transform.from_pose(state.flange))
    return Transform.from_pose(pose).compose(tcp_to_flange).to_pose()


def _heldout_grid(
    ctx: SkillContext, state: ArmState, spacing: float,
    training: Sequence[Mapping[str, Any]], count: int = 9,
) -> tuple[list[Pose], float]:
    needed = int(ctx.workcell.calibration["min_verify_points"])
    minimum = VerifyCamera.spec.args_schema["properties"]["spacing_mm"]["minimum"]
    for scale in _planning_scales(ctx):
        actual = max(minimum, spacing * scale)
        grid = [
            state.tcp.translated(dx * actual * (1 if row % 2 == 0 else -1), dy * actual)
            for row, dy in enumerate((-1, 0, 1)) for dx in (-1, 0, 1)
        ]
        heldout = [pose for pose in grid[:count] if not _overlaps_training(_flange_at(state, pose), training)]
        targets, reason = _reachable(ctx, heldout)
        if len(targets) >= needed:
            return targets, actual
    raise SkillError(
        FAILURE_OUT_OF_WORKSPACE,
        f"only {len(targets)} of {len(grid)} held-out grid points are reachable (need "
        f"{needed}): {reason}. Start nearer the middle, or lower spacing_mm",
    )


def _verification_stats(
    ctx: SkillContext, plan: SkillPlan, mount: Transform, target_mount: Transform,
    seen: Sequence[Sample], skipped: int,
) -> dict[str, Any]:
    depth, pnp = _errors(plan.basis["mounting"], mount, target_mount, seen)
    stats = {
        **_stats(depth, prefix=""), **_stats(pnp, prefix="pnp_"),
        "points": len(depth), "skipped": skipped, "no_depth": len(seen) - len(depth),
        "errors_mm": depth,
    }
    needed = int(ctx.workcell.calibration["min_verify_points"])
    if len(depth) < needed:
        raise _check_failed(
            f"the board centre was measured at only {len(depth)} points (need {needed}); "
            "keep the board in view and check the depth stream",
            problems=["too few measurements"], **stats,
        )
    limit, rms_limit = float(plan.basis["max_p95_mm"]), float(plan.basis["max_rms_mm"])
    if stats["p95_mm"] > limit or stats["rms_mm"] > rms_limit:
        raise _check_failed(
            f"the camera is off by {stats['p95_mm']:.1f} mm at p95 / "
            f"{stats['rms_mm']:.1f} mm RMS (limits {limit} / {rms_limit} mm, "
            f"PnP-only {stats['pnp_p95_mm']:.1f} mm); check board dimensions and mounting "
            "before running calib.handeye again",
            problems=["accuracy above limit"], **stats,
        )
    return stats


def _record_verification(
    ctx: SkillContext, role: str, version: float, plan: SkillPlan,
    stats: Mapping[str, Any], seen: Sequence[Sample],
) -> float:
    return ctx.record_calibration(
        f"{role}.verify",
        parameters={
            **stats, "camera_role": role, "max_p95_mm": plan.basis["max_p95_mm"],
            "max_rms_mm": plan.basis["max_rms_mm"], "depends_on": {role: version},
            "heldout_poses": [flange.to_dict() for flange, _ in seen],
            "training_disjoint": True,
        },
    )


# ── calib.verify ─────────────────────────────────────────────────────────────


class VerifyCamera:
    """Measure perception accuracy against the arm over a grid of positions."""

    spec = SkillSpec(
        skill_id="calib.verify",
        version="1.2.0",
        card=(
            "Recheck an already verified camera calibration, with the calibration board "
            "mounted exactly as it was then. The arm moves over a 3 x 3 grid spacing_mm "
            "apart around the current pose; at each stop the arm predicts the board "
            "centre and the camera measures it through depth -- the chain perceive.locate "
            "uses. The workcell's max_verify_p95_mm is the acceptance limit."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "spacing_mm": _number(10, 80, 40.0),
                "poses": {"type": "integer", "minimum": 4, "maximum": 9, "default": 9},
                "camera": {"type": "string", "pattern": "^camera(\\.[a-z0-9_]+)*$", "default": CAMERA_FRAME},
                "settle_s": _number(0, 5, 0.5),
                "speed_pct": _number(1, 100, 10.0),
            },
        },
        effect=EFFECT_ACTUATE,
        requires=(f"{REQUIRE_ROLE}arm", f"{REQUIRE_CALIBRATION}{{camera}}"),
        consent=CONSENT_EACH_RUN,
        region=REGION_BOOTSTRAP_UNTIL_MEASURED,
        provides=(f"{REQUIRE_RECORD}{{camera}}.verify",),
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        role = _camera(ctx, args, calibrated=True)
        mounting = _mounting(ctx, role)
        if ctx.frames.parameters(role).get("target_mount") is None:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                "the camera mount was not measured with calib.handeye, so the board has no "
                "known position; run calib.handeye first",
            )
        state = await _ready_state(ctx)
        seen = await _vision(ctx, role).observe_target()
        target_scale = _target_scale(ctx, seen)
        start, spacing = state.tcp, float(args["spacing_mm"])
        training = ctx.frames.parameters(role).get("training_poses", [])
        targets, spacing = _heldout_grid(ctx, state, spacing, training, int(args["poses"]))
        return SkillPlan(
            skill_id=self.spec.skill_id,
            args=args,
            summary=(
                f"Verify the camera: visit {len(targets)} points {spacing:.0f} mm apart around "
                f"{_fmt(start.position)} mm at {float(args['speed_pct']):.0f}% speed, comparing "
                "the board centre the arm predicts with the one the camera measures"
                + ("; board scale unchecked (no usable depth)" if target_scale is None else "")
            ),
            targets=_with_return(ctx, targets, start),
            limits={"speed_pct": float(args["speed_pct"])},
            basis={
                "camera_role": role,
                "mounting": mounting,
                "start": start.to_dict(),
                "settle_s": float(args["settle_s"]),
                "samples": len(targets),
                "max_p95_mm": ctx.workcell.calibration["max_verify_p95_mm"],
                "max_rms_mm": ctx.workcell.calibration["max_verify_rms_mm"],
                "seen": seen.to_dict(),
                "depends_on": {role: ctx.frames.versions[role]},
                "target_scale": target_scale,
                "spacing_mm": spacing,
            },
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        await _check_start(ctx, Pose.from_mapping(plan.basis["start"]))
        role = str(plan.basis.get("camera_role", CAMERA_FRAME))
        mount = ctx.frames.transform(role)
        target_mount = ctx.frames.parameters(role).get("target_mount")
        if mount is None or target_mount is None:
            raise SkillError(FAILURE_PLAN_STALE, "the camera calibration changed; plan again")
        indexed, skipped = await _sweep(ctx, plan)
        seen = [sample for _, sample in indexed]
        training = ctx.frames.parameters(role).get("training_poses", [])
        if any(_overlaps_training(flange, training) for flange, _ in seen):
            raise _check_failed("verification repeated a training pose", problems=["not held-out"])
        stats = _verification_stats(ctx, plan, mount, Transform.from_matrix(target_mount), seen, skipped)
        recorded_at = _record_verification(
            ctx, role, float(plan.basis["depends_on"][role]), plan, stats, seen
        )
        return SkillOutcome.succeeded(
            {**stats, "recorded_at": recorded_at},
            message=(
                f"The camera measures within {stats['p95_mm']:.1f} mm at p95 "
                f"(max {stats['max_mm']:.1f} mm over {stats['points']} points)"
            ),
        )


def _overlaps_training(flange: Pose, training: Sequence[Mapping[str, Any]]) -> bool:
    for value in training:
        pose = Pose.from_mapping(value)
        # Stored poses use Pose.to_dict precision; an identical capture must
        # still match after that serialization, not fall through its rounding gap.
        if flange.to_dict() == pose.to_dict() or (
            flange.distance_mm(pose) < 1e-6 and flange.rotation_to_deg(pose) < 1e-6
        ):
            return True
    return False


def _errors(
    mounting: str, mount: Transform, target_mount: Transform, seen: Sequence[Sample]
) -> tuple[list[float], list[float]]:
    """Per stop: arm-predicted board centre vs the depth-chain and PnP-chain measurement."""
    depth: list[float] = []
    pnp: list[float] = []
    for flange, observation in seen:
        arm = Transform.from_pose(flange)
        board = arm.compose(target_mount) if mounting == EYE_TO_HAND else target_mount
        camera = mount if mounting == EYE_TO_HAND else arm.compose(mount)
        predicted = board.apply(observation.center)
        pnp.append(norm(sub(camera.apply(observation.center_in_camera), predicted)))
        if observation.depth_point is not None:
            depth.append(norm(sub(camera.apply(observation.depth_point), predicted)))
    return depth, pnp


def _stats(errors: Sequence[float], *, prefix: str) -> dict[str, float | None]:
    if not errors:
        return {f"{prefix}{key}": None for key in ("mean_mm", "rms_mm", "p95_mm", "max_mm")}
    ordered = sorted(errors)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        f"{prefix}mean_mm": round(sum(ordered) / len(ordered), 3),
        f"{prefix}rms_mm": round(math.sqrt(sum(v * v for v in ordered) / len(ordered)), 3),
        f"{prefix}p95_mm": round(p95, 3),
        f"{prefix}max_mm": round(ordered[-1], 3),
    }


def skills() -> tuple[HandEye, Screws, CarriedBall, VerifyCamera]:
    """Entry point for ``leapflow.hardware.skills``."""
    return (HandEye(), Screws(), CarriedBall(), VerifyCamera())


__all__ = [
    "EYE_IN_HAND",
    "EYE_TO_HAND",
    "METHOD_SCREW",
    "VERIFY_PROCEDURE",
    "CarriedBall",
    "HandEye",
    "Screws",
    "VerifyCamera",
    "skills",
]
