# Copyright (c) Alibaba, Inc. and its affiliates.
"""Calibration skills: bring the arm up, measure the TCP, measure the workspace.

A fresh workcell has no geometry, so no Cartesian skill is available. These
skills create it, in order, each one a governed and individually approved run:

1. ``calib.bringup`` -- the arm answers, is enabled, (optionally) reads zero at
   its mechanical zero, and the gripper travels its full stroke.
2. ``calib.tcp`` -- where the tool centre point sits on the flange: a declared
   offset (from the gripper drawing) or a pivot fit over flange poses read with
   ``arm.state`` while the tool tip rests on one fixed point.
3. ``calib.workspace`` -- the box the TCP may move in. Its floor comes from a
   direct height, the table height plus clearance, or a plane fitted to table
   points touched with the TCP; the box is then verified by visiting its corners
   inside the workcell's human-declared bootstrap box.

Every result is a versioned calibration record. The workspace names the TCP
version it was measured with, so re-measuring the TCP makes it stale until it is
measured again. A check that measures out of tolerance is ``check_failed`` --
with side effect NONE at planning time, COMMITTED when the measurement moved.
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

from leapflow.hardware.skills.geometry import (
    Pose,
    WorkspaceBox,
    angle_between_deg,
    fit_plane,
    norm,
    plane_height,
    solve_pivot,
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
    FAILURE_PLAN_STALE,
    FAILURE_SKILL_UNAVAILABLE,
    REQUIRE_CALIBRATION,
    REQUIRE_ROLE,
    SIDE_EFFECT_COMMITTED,
    ArmState,
    SkillError,
    SkillOutcome,
    SkillPlan,
    SkillSpec,
)

BRINGUP_PROCEDURE = "arm.bringup"

_NUMBER = {"type": "number"}


def _numbers(count: int, description: str, *, min_items: int | None = None) -> dict[str, Any]:
    return {
        "type": "array",
        "items": _NUMBER,
        "minItems": count if min_items is None else min_items,
        "maxItems": count,
        "description": description,
    }


def _fmt(values: Sequence[float]) -> str:
    return "(" + ", ".join(f"{v:.1f}" for v in values) + ")"


def _check_failed(message: str, **result: Any) -> SkillError:
    return SkillError(FAILURE_CHECK_FAILED, message, result=result)


def _procedure(ctx: SkillContext, frame: str) -> str:
    procedure = ctx.frames.procedure(frame)
    if not procedure:
        raise SkillError(
            FAILURE_SKILL_UNAVAILABLE,
            f"workcell {ctx.workcell.workcell_id!r} declares no {frame} calibration procedure",
        )
    return procedure


async def _ready_state(ctx: SkillContext) -> ArmState:
    state = await ctx.motion.state()
    if state.enabled is False:
        raise SkillError(FAILURE_ARM_NOT_READY, "the arm is not enabled; ask the user to enable it")
    return state


# ── calib.bringup ────────────────────────────────────────────────────────────


class Bringup:
    """Check the arm answers, is enabled, reads zero, and the gripper travels."""

    spec = SkillSpec(
        skill_id="calib.bringup",
        version="1.0.0",
        card=(
            "First-run check of the arm: it reports every joint and is enabled; with "
            "expect_zero=true (the user has put the arm at its mechanical zero) every "
            "joint must read zero within zero_tol_deg; with gripper_travel=true the "
            "gripper opens fully, closes fully, and returns to its width. The arm itself "
            "does not move. Run it before calib.tcp on new or re-assembled hardware."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "expect_zero": {"type": "boolean", "default": False},
                "zero_tol_deg": {
                    "type": "number",
                    "minimum": 0.05,
                    "maximum": 5,
                    "default": 0.5,
                },
                "gripper_travel": {"type": "boolean", "default": True},
            },
        },
        effect=EFFECT_ACTUATE,
        requires=(f"{REQUIRE_ROLE}arm",),
        consent=CONSENT_EACH_RUN,
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        state = await _ready_state(ctx)
        if not state.joints:
            raise _check_failed("the arm reported no joint positions", problems=["no joints"])
        if args["expect_zero"]:
            tol = float(args["zero_tol_deg"])
            off = [f"{j} reads {v:.2f} deg" for j, v in state.joints.items() if abs(v) > tol]
            if off:
                raise _check_failed(
                    f"the arm is not at zero within {tol} deg ({'; '.join(off)}); re-zero it "
                    "with the vendor zero calibration, or check it is at its mechanical zero",
                    problems=off,
                )
        travel = bool(args["gripper_travel"]) and bool(ctx.workcell.role("gripper"))
        max_width = ctx.workcell.gripper_spec.get("max_width_mm")
        if travel and max_width is None:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE, "gripper travel needs gripper_spec.max_width_mm"
            )
        steps = [f"read {len(state.joints)} joints"]
        if args["expect_zero"]:
            steps.append(f"check they read zero within {args['zero_tol_deg']} deg")
        if travel:
            steps.append(
                f"open the gripper to {float(max_width):.0f} mm, close it fully, and restore "
                "its width -- keep the jaws clear"
            )
        return SkillPlan(
            skill_id=self.spec.skill_id,
            args=args,
            summary="Bring-up check: " + "; ".join(steps),
            basis={
                "joints": {j: round(v, 3) for j, v in state.joints.items()},
                "gripper_mm": state.gripper_mm,
                "travel": travel,
                "max_width_mm": None if max_width is None else float(max_width),
            },
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        await _ready_state(ctx)
        measured: dict[str, Any] = {"joints": plan.basis["joints"]}
        problems: list[str] = []
        if plan.basis["travel"]:
            problems = await self._travel(ctx, plan.basis, measured)
        if problems:
            return SkillOutcome.failed(
                FAILURE_CHECK_FAILED,
                "; ".join(problems),
                side_effect_state=SIDE_EFFECT_COMMITTED,
                result={"problems": problems, "measured": measured},
            )
        recorded_at = ctx.record_calibration(
            BRINGUP_PROCEDURE,
            role="arm",
            parameters={**measured, "zero_checked": bool(plan.args["expect_zero"])},
        )
        return SkillOutcome.succeeded(
            {"measured": measured, "recorded_at": recorded_at}, message="The arm passed bring-up"
        )

    @staticmethod
    async def _travel(
        ctx: SkillContext, basis: Mapping[str, Any], measured: dict[str, Any]
    ) -> list[str]:
        max_width = float(basis["max_width_mm"])
        problems: list[str] = []
        opened = await ctx.motion.gripper(max_width)
        measured["gripper_open_mm"] = round(opened.width_mm, 1)
        if opened.blocked:
            problems.append(f"the gripper opened to only {opened.width_mm:.1f} of {max_width} mm")
        closed = await ctx.motion.gripper(0.0)
        measured["gripper_closed_mm"] = round(closed.width_mm, 1)
        if closed.blocked:
            problems.append(
                f"the gripper stopped at {closed.width_mm:.1f} mm while closing; clear the jaws"
            )
        original = basis.get("gripper_mm")
        if original is not None:
            await ctx.motion.gripper(min(max(float(original), 0.0), max_width))
        return problems


# ── calib.tcp ────────────────────────────────────────────────────────────────


class ToolCentrePoint:
    """Record where the tool centre point sits in the flange frame."""

    spec = SkillSpec(
        skill_id="calib.tcp",
        version="1.0.0",
        card=(
            "Calibrate the tool centre point (TCP) on the flange. method=declared takes "
            "offset=[x, y, z] or [x, y, z, rx, ry, rz] (mm, deg; e.g. the gripper length "
            "from its drawing). method=pivot takes 4-12 flange poses [x, y, z, rx, ry, rz] "
            "read with arm.state while the tool tip rests on one fixed point in different "
            "orientations (at least 20 deg apart) and fits the tip. Nothing moves. Plans "
            "and a workspace measured with the previous TCP become stale."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ["declared", "pivot"], "default": "declared"},
                "offset": _numbers(6, "TCP in the flange frame", min_items=3),
                "samples": {
                    "type": "array",
                    "items": _numbers(6, "flange pose"),
                    "minItems": 4,
                    "maxItems": 12,
                },
                "max_residual_mm": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 5,
                    "default": 1.0,
                },
            },
        },
        effect=EFFECT_ACTUATE,
        requires=(f"{REQUIRE_ROLE}arm",),
        consent=CONSENT_EACH_RUN,
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        procedure = _procedure(ctx, "tcp")
        if ctx.frames.parent("tcp") != "flange":
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE, "the workcell's tcp frame must have parent: flange"
            )
        accept = ctx.workcell.calibration
        if args["method"] == "pivot":
            tcp, fit = _pivot_fit(args, accept)
        else:
            tcp, fit = _declared(args, accept)
        previous = ctx.frames.transform("tcp")
        was = "not calibrated" if previous is None else f"was {_fmt(previous.to_pose().position)}"
        summary = (
            f"Set the TCP to {_fmt(tcp.position)} mm from the flange ({was}; "
            f"{args['method']}{_fit_note(fit)}). Plans and the workspace measured with the "
            "previous TCP become stale."
        )
        return SkillPlan(
            skill_id=self.spec.skill_id,
            args=args,
            summary=summary,
            basis={
                "procedure": procedure,
                "tcp": tcp.to_dict(),
                "previous": None if previous is None else previous.to_pose().to_dict(),
                "fit": fit,
            },
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        recorded_at = ctx.record_calibration(
            plan.basis["procedure"],
            pose=plan.basis["tcp"],
            parameters={"method": plan.args["method"], **plan.basis["fit"]},
        )
        return SkillOutcome.succeeded(
            {"tcp": plan.basis["tcp"], "fit": plan.basis["fit"], "recorded_at": recorded_at},
            message=plan.summary,
        )


def _declared(args: Mapping[str, Any], accept: Mapping[str, float]) -> tuple[Pose, dict[str, Any]]:
    offset = args.get("offset")
    if offset is None:
        raise SkillError(
            FAILURE_INVALID_ARGS, "method=declared needs offset=[x, y, z(, rx, ry, rz)]"
        )
    if len(offset) not in (3, 6):
        raise SkillError(FAILURE_INVALID_ARGS, "offset takes 3 or 6 numbers")
    tcp = Pose(*offset)
    longest = float(accept["max_tool_length_mm"])
    if norm(tcp.position) > longest:
        raise SkillError(
            FAILURE_INVALID_ARGS,
            f"a {norm(tcp.position):.0f} mm TCP offset is implausible "
            f"(at most {longest:.0f} mm); offsets are in millimetres",
        )
    return tcp, {}


def _pivot_fit(args: Mapping[str, Any], accept: Mapping[str, float]) -> tuple[Pose, dict[str, Any]]:
    samples = args.get("samples")
    if not samples:
        raise SkillError(FAILURE_INVALID_ARGS, "method=pivot needs 4-12 samples")
    poses = [Pose(*sample) for sample in samples]
    spread = max(a.rotation_to_deg(b) for a, b in itertools.combinations(poses, 2))
    min_spread = float(accept["min_pivot_spread_deg"])
    if spread < min_spread:
        raise _check_failed(
            f"the sampled orientations span only {spread:.1f} deg; tilt the tool at least "
            f"{min_spread:.0f} deg between samples, in more than one direction",
            problems=["orientation spread too small"],
        )
    try:
        tip, pivot, residual = solve_pivot(poses)
    except ValueError as exc:
        raise _check_failed(
            f"the pivot is not observable from these samples ({exc}); tilt about different axes",
            problems=["degenerate samples"],
        ) from exc
    limit = float(args["max_residual_mm"])
    if residual > limit:
        raise _check_failed(
            f"the samples disagree by {residual:.2f} mm rms (limit {limit} mm); the tip "
            "likely slipped -- collect them again",
            problems=["residual above limit"],
            residual_mm=round(residual, 3),
        )
    longest = float(accept["max_tool_length_mm"])
    if norm(tip) > longest:
        raise _check_failed(
            f"the fitted tip is {norm(tip):.0f} mm from the flange, beyond "
            f"{longest:.0f} mm; the samples are not of one fixed point",
            problems=["implausible tool length"],
        )
    fit = {
        "samples": len(poses),
        "residual_mm": round(residual, 3),
        "spread_deg": round(spread, 1),
        "pivot": [round(v, 3) for v in pivot],
    }
    return Pose(*tip), fit


def _fit_note(fit: Mapping[str, Any]) -> str:
    if not fit:
        return ""
    return (
        f" over {fit['samples']} poses, {fit['residual_mm']:.2f} mm rms, "
        f"{fit['spread_deg']:.0f} deg spread"
    )


# ── calib.workspace ──────────────────────────────────────────────────────────


class Workspace:
    """Measure and verify the box the TCP may move in."""

    spec = SkillSpec(
        skill_id="calib.workspace",
        version="1.0.0",
        card=(
            "Calibrate the workspace: the box (arm base frame, mm) the TCP may move in. "
            "Give x=[min, max], y=[min, max], z_max, and exactly one floor: z_min; "
            "table_z (the table height; the floor is clearance_mm above it); or "
            "table_points, 3-12 [x, y, z] TCP positions touched on the table (read with "
            "arm.state), fitted as a plane. With verify=true the arm visits the box's 8 "
            "corners slowly, inside the workcell's bootstrap box. Requires calib.tcp."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "x": _numbers(2, "[min, max] mm"),
                "y": _numbers(2, "[min, max] mm"),
                "z_max": _NUMBER,
                "z_min": _NUMBER,
                "table_z": _NUMBER,
                "table_points": {
                    "type": "array",
                    "items": _numbers(3, "TCP position on the table"),
                    "minItems": 3,
                    "maxItems": 12,
                },
                "clearance_mm": {"type": "number", "minimum": 5, "maximum": 200, "default": 30.0},
                "verify": {"type": "boolean", "default": True},
                "speed_pct": {"type": "number", "minimum": 1, "maximum": 100, "default": 10.0},
            },
            "required": ["x", "y", "z_max"],
        },
        effect=EFFECT_ACTUATE,
        requires=(f"{REQUIRE_ROLE}arm", f"{REQUIRE_CALIBRATION}tcp"),
        consent=CONSENT_EACH_RUN,
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        procedure = _procedure(ctx, "workspace")
        floor_z, floor = _floor(args, ctx.workcell.calibration)
        try:
            box = WorkspaceBox(*args["x"], *args["y"], floor_z, float(args["z_max"]))
        except ValueError as exc:
            raise SkillError(
                FAILURE_INVALID_ARGS, f"the workspace box is empty or inverted: {exc}"
            ) from exc
        start = (await ctx.motion.state()).tcp
        targets = _tour(ctx, box, start) if args["verify"] else ()
        how = (
            f"verify it by visiting its 8 corners at {args['speed_pct']:.0f}% speed -- keep "
            "the area clear"
            if args["verify"]
            else "record it without moving (corners checked by hand)"
        )
        return SkillPlan(
            skill_id=self.spec.skill_id,
            args=args,
            summary=(
                f"Set the workspace to x{_fmt(args['x'])} y{_fmt(args['y'])} "
                f"z({floor_z:.1f}, {float(args['z_max']):.1f}) mm ({floor['note']}); {how}"
            ),
            targets=targets,
            limits={"speed_pct": float(args["speed_pct"])},
            basis={
                "procedure": procedure,
                "start": start.to_dict(),
                "box": box.to_dict(),
                "floor": floor,
                "depends_on": {"tcp": ctx.frames.versions["tcp"]},
            },
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        if plan.targets:
            await _check_start(ctx, Pose.from_mapping(plan.basis["start"]))
        for target in plan.targets:
            await ctx.motion.move_tcp(target, speed_pct=plan.limits.get("speed_pct"))
        recorded_at = ctx.record_calibration(
            plan.basis["procedure"],
            parameters={
                **plan.basis["box"],
                "floor": plan.basis["floor"],
                "verified": bool(plan.args["verify"]),
                "depends_on": plan.basis["depends_on"],
            },
        )
        return SkillOutcome.succeeded(
            {
                "workspace": plan.basis["box"],
                "verified": bool(plan.args["verify"]),
                "poses_visited": len(plan.targets),
                "recorded_at": recorded_at,
            },
            message=plan.summary,
        )


def _floor(args: Mapping[str, Any], accept: Mapping[str, float]) -> tuple[float, dict[str, Any]]:
    given = [name for name in ("z_min", "table_z", "table_points") if name in args]
    if len(given) != 1:
        raise SkillError(
            FAILURE_INVALID_ARGS, "give exactly one floor: z_min, table_z, or table_points"
        )
    clearance = float(args["clearance_mm"])
    if given[0] == "z_min":
        return float(args["z_min"]), {"method": "z_min", "note": "floor given directly"}
    if given[0] == "table_z":
        floor_z = float(args["table_z"]) + clearance
        return floor_z, {
            "method": "table_z",
            "table_z": float(args["table_z"]),
            "clearance_mm": clearance,
            "note": f"table at {float(args['table_z']):.1f} mm + {clearance:.0f} mm clearance",
        }
    return _table_floor(args, clearance, float(accept["max_table_residual_mm"]))


def _table_floor(
    args: Mapping[str, Any], clearance: float, max_residual: float
) -> tuple[float, dict[str, Any]]:
    try:
        normal, offset, residual = fit_plane(args["table_points"])
    except ValueError as exc:
        raise _check_failed(
            f"the table points do not define a plane ({exc}); touch points spread over "
            "the table, not along one line",
            problems=["degenerate table points"],
        ) from exc
    if residual > max_residual:
        raise _check_failed(
            f"the table points disagree by {residual:.1f} mm rms (limit "
            f"{max_residual:g} mm); touch the table again, gently",
            problems=["table residual above limit"],
            residual_mm=round(residual, 2),
        )
    (x0, x1), (y0, y1) = args["x"], args["y"]
    highest = max(plane_height(normal, offset, x, y) for x in (x0, x1) for y in (y0, y1))
    tilt = angle_between_deg(normal, (0.0, 0.0, 1.0))
    return highest + clearance, {
        "method": "table_points",
        "normal": [round(v, 6) for v in normal],
        "offset_mm": round(offset, 3),
        "residual_mm": round(residual, 3),
        "tilt_deg": round(tilt, 2),
        "clearance_mm": clearance,
        "note": (
            f"table plane tilted {tilt:.1f} deg, highest corner {highest:.1f} mm, "
            f"+ {clearance:.0f} mm clearance"
        ),
    }


def _tour(ctx: SkillContext, box: WorkspaceBox, start: Pose) -> tuple[Pose, ...]:
    """Top ring, bottom ring backwards, lift back up, and home to the start if allowed."""
    ring = [
        (box.x_min, box.y_min),
        (box.x_max, box.y_min),
        (box.x_max, box.y_max),
        (box.x_min, box.y_max),
    ]
    points = [(x, y, box.z_max) for x, y in ring]
    points += [(x, y, box.z_min) for x, y in reversed(ring)]
    points.append((ring[0][0], ring[0][1], box.z_max))
    targets = [Pose(x, y, z, start.rx, start.ry, start.rz) for x, y, z in points]
    for target in targets:
        try:
            ctx.motion.check_target(target)
        except SkillError as error:
            raise SkillError(
                error.failure_code,
                f"corner {_fmt(target.position)} cannot be verified: {error.message}. Widen "
                "the workcell bootstrap_box to cover the new workspace, or pass verify=false "
                "after checking the corners by hand",
            ) from error
    try:
        ctx.motion.check_target(start)
        targets.append(start)
    except SkillError:
        pass  # the start lies outside the verification region; stay at the last corner
    return tuple(targets)


async def _check_start(ctx: SkillContext, start: Pose) -> None:
    current = (await ctx.motion.state()).tcp
    moved_mm, moved_deg = current.distance_mm(start), current.rotation_to_deg(start)
    accept = ctx.workcell.calibration
    if moved_mm > accept["start_drift_mm"] or moved_deg > accept["start_drift_deg"]:
        raise SkillError(
            FAILURE_PLAN_STALE,
            f"the tool moved {moved_mm:.1f} mm / {moved_deg:.1f} deg since planning; plan again",
        )


def skills() -> tuple[Bringup, ToolCentrePoint, Workspace]:
    """Entry point for ``leapflow.hardware.skills``."""
    return (Bringup(), ToolCentrePoint(), Workspace())


__all__ = [
    "BRINGUP_PROCEDURE",
    "Bringup",
    "ToolCentrePoint",
    "Workspace",
    "skills",
]
