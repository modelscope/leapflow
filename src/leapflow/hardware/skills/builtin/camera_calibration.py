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
2. ``calib.verify`` -- the arm moves over a small grid; at every stop the arm
   predicts where the board centre is, and the camera measures it through the
   depth image and the hand-eye mount -- the same chain ``perceive.locate``
   uses. The 95th-percentile disagreement is the accuracy perception has.

A fit or check out of tolerance is ``check_failed`` with side effect COMMITTED:
the arm moved to measure it.
"""

from __future__ import annotations

import asyncio
import math
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
    FAILURE_OUT_OF_WORKSPACE,
    FAILURE_PLAN_STALE,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_TARGET_NOT_FOUND,
    FAILURE_UNREACHABLE,
    REQUIRE_CALIBRATION,
    REQUIRE_ROLE,
    SIDE_EFFECT_COMMITTED,
    ArmState,
    SkillError,
    SkillOutcome,
    SkillPlan,
    SkillSpec,
    TargetObservation,
)

CAMERA_FRAME = "camera"
VERIFY_PROCEDURE = "camera.verify"
EYE_TO_HAND = "eye_to_hand"
EYE_IN_HAND = "eye_in_hand"

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


def _mounting(ctx: SkillContext) -> str:
    parent = ctx.frames.parent(CAMERA_FRAME)
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


async def _observe(ctx: SkillContext) -> TargetObservation | None:
    try:
        return await ctx.vision.observe_target()
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
        observation = await _observe(ctx)
        if observation is None:
            skipped += 1
            continue
        seen.append(((await ctx.motion.state()).flange, observation))
    return seen, skipped


# ── calib.handeye ────────────────────────────────────────────────────────────


class HandEye:
    """Measure the camera mount from board observations over a ring of arm poses."""

    spec = SkillSpec(
        skill_id="calib.handeye",
        version="1.1.0",
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
                "settle_s": _number(0, 5, 0.5),
                "speed_pct": _number(1, 100, 10.0),
            },
        },
        effect=EFFECT_ACTUATE,
        requires=(f"{REQUIRE_ROLE}arm", f"{REQUIRE_ROLE}camera"),
        consent=CONSENT_EACH_RUN,
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        procedure = _procedure(ctx, CAMERA_FRAME)
        mounting = _mounting(ctx)
        state = await _ready_state(ctx)
        seen = await ctx.vision.observe_target()
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


# ── calib.verify ─────────────────────────────────────────────────────────────


class VerifyCamera:
    """Measure perception accuracy against the arm over a grid of positions."""

    spec = SkillSpec(
        skill_id="calib.verify",
        version="1.1.0",
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
                "settle_s": _number(0, 5, 0.5),
                "speed_pct": _number(1, 100, 10.0),
            },
        },
        effect=EFFECT_ACTUATE,
        requires=(f"{REQUIRE_ROLE}arm", f"{REQUIRE_ROLE}camera", f"{REQUIRE_CALIBRATION}camera"),
        consent=CONSENT_EACH_RUN,
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        mounting = _mounting(ctx)
        if ctx.frames.parameters(CAMERA_FRAME).get("target_mount") is None:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                "the camera mount was not measured with calib.handeye, so the board has no "
                "known position; run calib.handeye first",
            )
        state = await _ready_state(ctx)
        seen = await ctx.vision.observe_target()
        start, spacing = state.tcp, float(args["spacing_mm"])
        grid = [
            start.translated(dx * spacing * (1 if row % 2 == 0 else -1), dy * spacing)
            for row, dy in enumerate((-1, 0, 1))
            for dx in (-1, 0, 1)
        ]
        targets, reason = _reachable(ctx, grid)
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
                "mounting": mounting,
                "start": start.to_dict(),
                "settle_s": float(args["settle_s"]),
                "samples": len(targets),
                "max_p95_mm": ctx.workcell.calibration["max_verify_p95_mm"],
                "seen": seen.to_dict(),
                "depends_on": {CAMERA_FRAME: ctx.frames.versions[CAMERA_FRAME]},
            },
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        await _check_start(ctx, Pose.from_mapping(plan.basis["start"]))
        mount = ctx.frames.transform(CAMERA_FRAME)
        target_mount = ctx.frames.parameters(CAMERA_FRAME).get("target_mount")
        if mount is None or target_mount is None:
            raise SkillError(FAILURE_PLAN_STALE, "the camera calibration changed; plan again")
        seen, skipped = await _sweep(ctx, plan)
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
        if stats["p95_mm"] > limit:
            raise _check_failed(
                f"the camera is off by {stats['p95_mm']:.1f} mm at p95 (limit {limit} mm, "
                f"PnP-only {stats['pnp_p95_mm']:.1f} mm); run calib.handeye again",
                problems=["accuracy above limit"],
                **stats,
            )
        recorded_at = ctx.record_calibration(
            VERIFY_PROCEDURE,
            parameters={**stats, "max_p95_mm": limit, "depends_on": plan.basis["depends_on"]},
        )
        return SkillOutcome.succeeded(
            {**stats, "recorded_at": recorded_at},
            message=(
                f"The camera measures within {stats['p95_mm']:.1f} mm at p95 "
                f"(max {stats['max_mm']:.1f} mm over {len(depth)} points)"
            ),
        )


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
        return {f"{prefix}{key}": None for key in ("mean_mm", "p95_mm", "max_mm")}
    ordered = sorted(errors)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        f"{prefix}mean_mm": round(sum(ordered) / len(ordered), 3),
        f"{prefix}p95_mm": round(p95, 3),
        f"{prefix}max_mm": round(ordered[-1], 3),
    }


def skills() -> tuple[HandEye, VerifyCamera]:
    """Entry point for ``leapflow.hardware.skills``."""
    return (HandEye(), VerifyCamera())


__all__ = [
    "EYE_IN_HAND",
    "EYE_TO_HAND",
    "VERIFY_PROCEDURE",
    "HandEye",
    "VerifyCamera",
    "skills",
]
