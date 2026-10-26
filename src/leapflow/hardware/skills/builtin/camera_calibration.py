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
   ``calib.verify`` uses as ground truth.
2. ``calib.screws`` -- the same mount with no board at all: the arm moves one
   joint at a time, and the surfaces that joint carries must land where the
   firmware's own rotation says they went. See ``screw`` for why that is enough.
3. ``calib.verify`` -- the arm moves over a small grid; at every stop the arm
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
    sub,
    tool_cone_deg,
)
from leapflow.hardware.skills.ports import (
    SkillContext,
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
    METHOD_MARKERLESS_SCREW,
    REGION_BOOTSTRAP_UNTIL_MEASURED,
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


async def _sweep(ctx: SkillContext, plan: SkillPlan) -> tuple[list[Sample], int]:
    """Visit every target; observe the board at the first ``basis.samples`` of them."""
    count, settle = int(plan.basis["samples"]), float(plan.basis["settle_s"])
    seen: list[Sample] = []
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
        seen.append((after, observation))
    return seen, skipped


# ── calib.handeye ────────────────────────────────────────────────────────────


class HandEye:
    """Measure the camera mount from board observations over a ring of arm poses."""

    spec = SkillSpec(
        skill_id="calib.handeye",
        version="1.2.0",
        card=(
            "Calibrate where the camera sits (hand-eye). The workcell's calibration board "
            "must be in view: held in the gripper or fixed to the flange for a fixed camera, "
            "lying still in the cell for a wrist camera. The arm visits `poses` poses "
            "around the current one -- tilted up to tilt_deg, yawed up to yaw_deg, shifted "
            "up to spread_mm -- slowly, observing the board at each, then returns. Start "
            "with the board centred in the image. Run calib.verify afterwards."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "poses": {"type": "integer", "minimum": 8, "maximum": 24, "default": 12},
                "tilt_deg": _number(5, 30, 15.0),
                "yaw_deg": _number(0, 60, 30.0),
                "spread_mm": _number(0, 80, 30.0),
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
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        role = _camera(ctx, args)
        procedure = _procedure(ctx, role)
        mounting = _mounting(ctx, role)
        state = await _ready_state(ctx)
        seen = await _vision(ctx, role).observe_target()
        start = state.tcp
        if mounting == EYE_TO_HAND:
            pivot = start.position
        else:
            reach = norm(seen.center_in_camera)
            pivot = add(state.flange.position, [v * reach for v in state.flange.tool_axis])
        tilt = min(float(args["tilt_deg"]), ctx.workcell.tool_cone_deg - tool_cone_deg(start))
        if tilt < ctx.workcell.calibration["min_tilt_deg"]:
            raise SkillError(
                FAILURE_UNREACHABLE,
                f"the tool may tilt only {max(tilt, 0.0):.1f} deg more here (tool_cone_deg "
                f"{ctx.workcell.tool_cone_deg:.0f}); start with the tool closer to straight down",
            )
        count = int(args["poses"])
        ring = _ring(start, pivot, count, tilt=tilt, args=args)
        targets, reason = _reachable(ctx, ring)
        calibration = ctx.workcell.calibration
        required = max(int(calibration["min_hand_eye_poses"]), count // 2)
        if len(targets) < required:
            raise SkillError(
                FAILURE_OUT_OF_WORKSPACE,
                f"only {len(targets)} of {count} calibration poses are reachable (need "
                f"{required}): {reason}. Start nearer the middle of the bootstrap box, or "
                "lower spread_mm / tilt_deg",
            )
        return SkillPlan(
            skill_id=self.spec.skill_id,
            args=args,
            summary=(
                f"Hand-eye calibrate the {mounting.replace('_', '-')} camera: visit "
                f"{len(targets)} poses (tilt up to {tilt:.0f} deg, yaw up to "
                f"{float(args['yaw_deg']):.0f} deg) around {_fmt(pivot)} mm at "
                f"{float(args['speed_pct']):.0f}% speed, observing the board at each, then "
                "return -- keep the area clear and the board still"
            ),
            targets=_with_return(ctx, targets, start),
            limits={"speed_pct": float(args["speed_pct"])},
            basis={
                "procedure": procedure,
                "camera_role": role,
                "camera_device": ctx.workcell.role(role),
                "mounting": mounting,
                "start": start.to_dict(),
                "settle_s": float(args["settle_s"]),
                "samples": len(targets),
                "required": required,
                "max_translation_rms_mm": calibration["max_hand_eye_translation_rms_mm"],
                "max_rotation_rms_deg": calibration["max_hand_eye_rotation_rms_deg"],
                "seen": seen.to_dict(),
            },
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        await _check_start(ctx, Pose.from_mapping(plan.basis["start"]))
        seen, skipped = await _sweep(ctx, plan)
        required = int(plan.basis["required"])
        if len(seen) < required:
            raise _check_failed(
                f"the board was seen in only {len(seen)} of {plan.basis['samples']} poses "
                f"(need {required}); keep it in view and lit evenly, or reduce tilt_deg",
                problems=["board not seen often enough"],
                skipped=skipped,
            )
        intrinsics = seen[0][1].intrinsics
        if any(observation.intrinsics != intrinsics for _, observation in seen):
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
                "calibration_target",
                problems=["fit residual above limit"],
                fit=fit.to_dict(),
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
        camera = mount.to_pose()
        return SkillOutcome.succeeded(
            {
                "camera": camera.to_dict(),
                "mounting": mounting,
                "fit": fit.to_dict(),
                "samples": len(seen),
                "skipped": skipped,
                "recorded_at": recorded_at,
            },
            message=(
                f"The camera sits at {_fmt(camera.position)} mm in the "
                f"{'base' if mounting == EYE_TO_HAND else 'flange'} frame "
                f"({fit.translation_rms_mm:.2f} mm / {fit.rotation_rms_deg:.3f} deg rms over "
                f"{len(seen)} poses). Run calib.verify to measure perception accuracy."
            ),
        )


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
        fit, evidence = self._measure(plan, shots, moves)
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
                f"({worst.label}) leaves {worst.shift_gap_mm:.1f} mm unexplained"
            ),
        )


# ── calib.verify ─────────────────────────────────────────────────────────────


class VerifyCamera:
    """Measure perception accuracy against the arm over a grid of positions."""

    spec = SkillSpec(
        skill_id="calib.verify",
        version="1.2.0",
        card=(
            "Measure camera accuracy after calib.handeye, with the calibration board "
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
        requires=(f"{REQUIRE_ROLE}arm",),
        consent=CONSENT_EACH_RUN,
        region=REGION_BOOTSTRAP_UNTIL_MEASURED,
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
        start, spacing = state.tcp, float(args["spacing_mm"])
        grid = [
            start.translated(dx * spacing * (1 if row % 2 == 0 else -1), dy * spacing)
            for row, dy in enumerate((-1, 0, 1))
            for dx in (-1, 0, 1)
        ]
        training = ctx.frames.parameters(role).get("training_poses", [])
        tcp_to_flange = Transform.from_pose(state.tcp).inverse().compose(
            Transform.from_pose(state.flange)
        )
        heldout = []
        for pose in grid[:int(args.get("poses", 9))]:
            flange = Transform.from_pose(pose).compose(tcp_to_flange).to_pose()
            if not _overlaps_training(flange, training):
                heldout.append(pose)
        targets, reason = _reachable(ctx, heldout)
        needed = int(ctx.workcell.calibration["min_verify_points"])
        if len(targets) < needed:
            raise SkillError(
                FAILURE_OUT_OF_WORKSPACE,
                f"only {len(targets)} of {len(grid)} grid points are reachable (need "
                f"{needed}): {reason}. Start nearer the middle, or lower spacing_mm",
            )
        return SkillPlan(
            skill_id=self.spec.skill_id,
            args=args,
            summary=(
                f"Verify the camera: visit {len(targets)} points {spacing:.0f} mm apart around "
                f"{_fmt(start.position)} mm at {float(args['speed_pct']):.0f}% speed, comparing "
                "the board centre the arm predicts with the one the camera measures"
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
            },
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        await _check_start(ctx, Pose.from_mapping(plan.basis["start"]))
        role = str(plan.basis.get("camera_role", CAMERA_FRAME))
        mount = ctx.frames.transform(role)
        target_mount = ctx.frames.parameters(role).get("target_mount")
        if mount is None or target_mount is None:
            raise SkillError(FAILURE_PLAN_STALE, "the camera calibration changed; plan again")
        seen, skipped = await _sweep(ctx, plan)
        training = ctx.frames.parameters(role).get("training_poses", [])
        if any(_overlaps_training(flange, training) for flange, _ in seen):
            raise _check_failed("verification repeated a training pose", problems=["not held-out"])
        depth, pnp = _errors(
            plan.basis["mounting"], mount, Transform.from_matrix(target_mount), seen
        )
        stats = {
            **_stats(depth, prefix=""),
            **_stats(pnp, prefix="pnp_"),
            "points": len(depth),
            "skipped": skipped,
            "no_depth": len(seen) - len(depth),
        }
        needed = int(ctx.workcell.calibration["min_verify_points"])
        if len(depth) < needed:
            raise _check_failed(
                f"the board centre was measured at only {len(depth)} points (need "
                f"{needed}); keep the board in view and check the depth stream",
                problems=["too few measurements"],
                **stats,
            )
        limit = float(plan.basis["max_p95_mm"])
        rms_limit = float(plan.basis["max_rms_mm"])
        if stats["p95_mm"] > limit or stats["rms_mm"] > rms_limit:
            raise _check_failed(
                f"the camera is off by {stats['p95_mm']:.1f} mm at p95 / "
                f"{stats['rms_mm']:.1f} mm RMS (limits {limit} / {rms_limit} mm, "
                f"PnP-only {stats['pnp_p95_mm']:.1f} mm); run calib.handeye again",
                problems=["accuracy above limit"],
                **stats,
            )
        recorded_at = ctx.record_calibration(
            f"{role}.verify",
            parameters={**stats, "camera_role": role, "max_p95_mm": limit,
                        "max_rms_mm": plan.basis["max_rms_mm"], "depends_on": plan.basis["depends_on"],
                        "heldout_poses": [flange.to_dict() for flange, _ in seen],
                        "errors_mm": depth, "training_disjoint": True},
        )
        return SkillOutcome.succeeded(
            {**stats, "recorded_at": recorded_at},
            message=(
                f"The camera measures within {stats['p95_mm']:.1f} mm at p95 "
                f"(max {stats['max_mm']:.1f} mm over {len(depth)} points)"
            ),
        )


def _overlaps_training(flange: Pose, training: Sequence[Mapping[str, Any]]) -> bool:
    for value in training:
        pose = Pose.from_mapping(value)
        if flange.distance_mm(pose) < 1e-6 and flange.rotation_to_deg(pose) < 1e-6:
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


def skills() -> tuple[HandEye, Screws, VerifyCamera]:
    """Entry point for ``leapflow.hardware.skills``."""
    return (HandEye(), Screws(), VerifyCamera())


__all__ = [
    "EYE_IN_HAND",
    "EYE_TO_HAND",
    "METHOD_SCREW",
    "VERIFY_PROCEDURE",
    "HandEye",
    "Screws",
    "VerifyCamera",
    "skills",
]
