# Copyright (c) Alibaba, Inc. and its affiliates.
"""``motion.sequence``: a motion program S2 writes, approved once, run step by step.

S2 composes manipulation itself -- approach, descend, close, lift, carry,
release -- from five ops (``move``, ``move_linear``, ``move_joints``,
``gripper``, ``wait``). Targets are absolute poses, offsets from the previous
waypoint, or references to the handles perception and the grasp planner
returned (``object:o3.top``, ``grasp:g2.pregrasp``). The node owns what a
model must never be trusted with:

- **Plan.** Every reference is resolved and every TCP target checked against
  the workspace, the tool cone and the arm's envelopes before approval. The
  workspace box is convex, so a straight segment between two checked targets
  stays inside it. The approved targets are exactly the targets that run.
- **Freshness.** ``execute`` refuses (``plan_stale``, nothing moved) when the
  tool left its planned start, the held object changed, or a referenced object
  drifted beyond the workcell's declared ``freshness`` tolerances.
- **Guards.** A step may declare what must hold after it (``held``,
  ``released``, ``reached``, ``no_contact``). A failed guard ends the program:
  ``on_fail: stop`` halts the arm, ``hold`` leaves it where it is. Either way
  the outcome names the step and what was observed.
- **Holding.** Closing on a referenced grasp or object records it as held;
  opening releases it; a held width that drifts beyond the declared tolerance
  forgets it -- so the next plan starts from what the jaws report.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

from leapflow.hardware.skills.geometry import Pose, mat_vec, wrap_axis_deg
from leapflow.hardware.skills.ports import PATH_DEFAULT, PATH_LINEAR, SkillContext
from leapflow.hardware.skills.protocol import (
    EFFECT_ACTUATE,
    FAILURE_CALIBRATION_MISSING,
    FAILURE_GUARD_FAILED,
    FAILURE_INVALID_ARGS,
    FAILURE_PLAN_STALE,
    FAILURE_SKILL_UNAVAILABLE,
    REQUIRE_CALIBRATION,
    REQUIRE_ROLE,
    SIDE_EFFECT_PARTIAL,
    STATUS_FAILED,
    STATUS_HALTED,
    ArmState,
    GraspCandidate,
    GripperResult,
    HeldObject,
    LocatedObject,
    SkillError,
    SkillOutcome,
    SkillPlan,
    SkillSpec,
)

OP_MOVE = "move"
OP_MOVE_LINEAR = "move_linear"
OP_MOVE_JOINTS = "move_joints"
OP_GRIPPER = "gripper"
OP_WAIT = "wait"
OPS = (OP_MOVE, OP_MOVE_LINEAR, OP_MOVE_JOINTS, OP_GRIPPER, OP_WAIT)

EXPECT_HELD = "held"
EXPECT_RELEASED = "released"
EXPECT_REACHED = "reached"
EXPECT_NO_CONTACT = "no_contact"
ON_FAIL_STOP = "stop"
ON_FAIL_HOLD = "hold"

_POSE_OPS = frozenset({OP_MOVE, OP_MOVE_LINEAR})
_COMMON = frozenset({"op", "expect", "on_fail"})
_FIELDS: Mapping[str, frozenset[str]] = {
    OP_MOVE: frozenset({"pose", "relative", "ref", "offset", "speed_pct"}),
    OP_MOVE_LINEAR: frozenset({"pose", "relative", "ref", "offset", "speed_pct"}),
    OP_MOVE_JOINTS: frozenset({"joints", "speed_pct"}),
    OP_GRIPPER: frozenset({"width_mm", "grip", "ref", "effort"}),
    OP_WAIT: frozenset({"seconds"}),
}
_EXPECTS: Mapping[str, frozenset[str]] = {
    OP_MOVE: frozenset({EXPECT_HELD, EXPECT_REACHED, EXPECT_NO_CONTACT}),
    OP_MOVE_LINEAR: frozenset({EXPECT_HELD, EXPECT_REACHED, EXPECT_NO_CONTACT}),
    OP_MOVE_JOINTS: frozenset({EXPECT_HELD, EXPECT_REACHED, EXPECT_NO_CONTACT}),
    OP_GRIPPER: frozenset({EXPECT_HELD, EXPECT_RELEASED, EXPECT_REACHED}),
    OP_WAIT: frozenset({EXPECT_HELD, EXPECT_REACHED}),
}
_OBJECT_POINTS = ("", "top", "bottom")
_GRASP_POINTS = ("pregrasp", "grasp", "retreat")

_SPEED_ARG = {
    "type": "number",
    "minimum": 1,
    "maximum": 100,
    "description": "Speed, percent of the arm maximum; capped by the run's limit.",
}
_ANGLE = {"type": "number", "minimum": -180, "maximum": 180}
_OFFSET = {"type": "number", "minimum": -1000, "maximum": 1000, "default": 0.0}
_STEP_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": list(OPS)},
        "pose": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "rx": _ANGLE,
                "ry": _ANGLE,
                "rz": _ANGLE,
                "frame": {"type": "string", "enum": ["base", "table"], "default": "base"},
            },
            "required": ["x", "y", "z"],
        },
        "relative": {
            "type": "object",
            "properties": {
                "dx": _OFFSET,
                "dy": _OFFSET,
                "dz": _OFFSET,
                "frame": {"type": "string", "enum": ["base", "tool"], "default": "base"},
            },
        },
        "ref": {"type": "string", "minLength": 3, "maxLength": 40},
        "offset": {
            "type": "object",
            "properties": {"dx": _OFFSET, "dy": _OFFSET, "dz": _OFFSET},
        },
        "joints": {"type": "object", "additionalProperties": True},
        "width_mm": {"type": "number", "minimum": 0},
        "grip": {"type": "string", "enum": ["open", "close"]},
        "effort": {"type": "number", "minimum": 0},
        "seconds": {"type": "number", "minimum": 0},
        "speed_pct": _SPEED_ARG,
        "expect": {
            "type": "string",
            "enum": [EXPECT_HELD, EXPECT_RELEASED, EXPECT_REACHED, EXPECT_NO_CONTACT],
        },
        "on_fail": {"type": "string", "enum": [ON_FAIL_STOP, ON_FAIL_HOLD], "default": "stop"},
    },
    "required": ["op"],
}

_CARD = (
    "Run a motion program on the arm and gripper, approved once as a whole. Each step "
    "has an op: move (the arm's default path) or move_linear (straight line) to a TCP "
    "target; move_joints {joint: deg}; gripper (width_mm, or grip open|close; "
    "ref grasp:gN opens to that grasp's open_mm and records the object as held when "
    "the jaws close on it); wait (seconds). A move target is one of: pose {x, y, z, "
    "rx?, ry?, rz?} in mm/deg (frame base, or table: z above the table surface); "
    "relative {dx, dy, dz, frame base|tool} from the previous target; or ref "
    "object:oN[.top|.bottom] / grasp:gN.pregrasp|grasp|retreat plus an optional "
    "offset {dx, dy, dz} in the base frame. Omitted angles keep the previous target's "
    "orientation. A step may declare expect: held (the jaws hold something), "
    "released (the jaws opened freely), reached (the TCP is at the last target), or "
    "no_contact (the move arrived and the jaws were not pushed); on_fail stop halts "
    "the arm, hold leaves it in place. Plan uses the arm's declared FK to sample every "
    "joint path and refuses any tool target outside the workspace; execute repeats that check "
    "from fresh joint state and refuses when the tool or a referenced object moved."
)


@dataclass(frozen=True)
class _Ref:
    kind: str  # "object" or "grasp"
    handle: str
    point: str


def _parse_ref(text: str, *, index: int) -> _Ref:
    kind, _, rest = text.partition(":")
    handle, _, point = rest.partition(".")
    points = _OBJECT_POINTS if kind == "object" else _GRASP_POINTS if kind == "grasp" else ()
    if not handle or (point not in points and not (kind == "grasp" and point == "")):
        raise _invalid(
            index,
            f"ref {text!r} must be object:<id>[.top|.bottom] or "
            "grasp:<id>[.pregrasp|.grasp|.retreat]",
        )
    return _Ref(kind, handle, point)


def _invalid(index: int, message: str) -> SkillError:
    return SkillError(FAILURE_INVALID_ARGS, f"step {index}: {message}")


def _fmt(pose: Pose) -> str:
    return f"({pose.x:.0f}, {pose.y:.0f}, {pose.z:.0f}) mm"


# ── Plan ─────────────────────────────────────────────────────────────────────


class _Planner:
    """Resolves steps in order, tracking predicted TCP and joint targets."""

    def __init__(self, ctx: SkillContext, state: ArmState) -> None:
        self._ctx = ctx
        self._start = state
        self._previous: Pose | None = state.tcp
        self._joints = frozenset(state.joints)
        self._joint_state: dict[str, float] | None = dict(state.joints)
        self._kinematics_checked = False
        self.objects: dict[str, LocatedObject] = {}
        self.grasps: dict[str, GraspCandidate] = {}
        self.targets: list[Pose] = []

    def resolve(self, index: int, step: Mapping[str, Any]) -> dict[str, Any]:
        op = step["op"]
        extra = sorted(set(step) - _COMMON - _FIELDS[op])
        if extra:
            raise _invalid(index, f"{', '.join(extra)} do(es) not apply to op {op}")
        expect = step.get("expect", "")
        if expect and expect not in _EXPECTS[op]:
            raise _invalid(index, f"expect {expect} does not apply to op {op}")
        resolved: dict[str, Any] = {"op": op, "expect": expect, "on_fail": step["on_fail"]}
        if "speed_pct" in step:
            resolved["speed_pct"] = float(step["speed_pct"])
        if op in _POSE_OPS:
            resolved.update(self._move(index, step))
        elif op == OP_MOVE_JOINTS:
            resolved.update(self._move_joints(index, step))
        elif op == OP_GRIPPER:
            resolved.update(self._gripper(index, step))
        else:
            resolved.update(self._wait(index, step))
        if expect == EXPECT_REACHED and self._previous is None:
            raise _invalid(index, "expect reached needs a known TCP target")
        return resolved

    def _move(self, index: int, step: Mapping[str, Any]) -> dict[str, Any]:
        given = [name for name in ("pose", "relative", "ref") if name in step]
        if len(given) != 1:
            raise _invalid(index, "a move needs exactly one of pose, relative, ref")
        if "offset" in step and "ref" not in step:
            raise _invalid(index, "offset only applies to a ref target")
        path = PATH_DEFAULT
        if step["op"] == OP_MOVE_LINEAR:
            if PATH_LINEAR not in self._ctx.motion.paths:
                raise SkillError(
                    FAILURE_SKILL_UNAVAILABLE,
                    f"step {index}: this arm declares no linear path (arm.pose_path)",
                )
            path = PATH_LINEAR
        entry: dict[str, Any] = {"path": path}
        if "pose" in step:
            target = self._absolute(index, step["pose"])
            text = f"to {_fmt(target)}"
        elif "relative" in step:
            target = self._relative(index, step["relative"])
            rel = step["relative"]
            text = (
                f"by ({rel['dx']:+.0f}, {rel['dy']:+.0f}, {rel['dz']:+.0f}) mm "
                f"({rel['frame']} frame) to {_fmt(target)}"
            )
        else:
            target, entry["object"], entry["grasp"] = self._referenced(index, step)
            text = f"to {step['ref']} at {_fmt(target)}"
        self._ctx.motion.check_target(target)
        self._previous = target
        self._joint_state = None
        self.targets.append(target)
        verb = "move linear" if path == PATH_LINEAR else "move"
        return {**entry, "pose": target.to_dict(), "text": f"{verb} {text}"}

    def _orientation(self, index: int, data: Mapping[str, Any]) -> tuple[float, float, float]:
        previous = self._previous
        if previous is None and not all(axis in data for axis in ("rx", "ry", "rz")):
            raise _invalid(index, "after move_joints a target must give rx, ry and rz")
        return tuple(  # type: ignore[return-value]
            float(data[axis]) if axis in data else float(getattr(previous, axis))
            for axis in ("rx", "ry", "rz")
        )

    def _absolute(self, index: int, pose: Mapping[str, Any]) -> Pose:
        x, y, z = float(pose["x"]), float(pose["y"]), float(pose["z"])
        if pose["frame"] == "table":
            table = self._ctx.frames.table
            if table is None:
                raise SkillError(
                    FAILURE_CALIBRATION_MISSING,
                    f"step {index}: a table-frame pose needs the table calibrated "
                    "(calib.workspace with table_z or table_points)",
                )
            z += table.height(x, y)
        return Pose(x, y, z, *self._orientation(index, pose))

    def _relative(self, index: int, rel: Mapping[str, Any]) -> Pose:
        previous = self._previous
        if previous is None:
            raise _invalid(index, "a relative move cannot follow move_joints")
        offset = (float(rel["dx"]), float(rel["dy"]), float(rel["dz"]))
        delta = mat_vec(previous.rotation, offset) if rel["frame"] == "tool" else offset
        return previous.translated(*delta)

    def _referenced(self, index: int, step: Mapping[str, Any]) -> tuple[Pose, str, str]:
        ref = _parse_ref(step["ref"], index=index)
        offset = step.get("offset") or {"dx": 0.0, "dy": 0.0, "dz": 0.0}
        if ref.kind == "grasp":
            if not ref.point:
                raise _invalid(index, "a move to a grasp names .pregrasp, .grasp or .retreat")
            grasp = self.grasp(ref.handle)
            base = grasp.waypoint(ref.point)
            target = base.translated(offset["dx"], offset["dy"], offset["dz"])
            return target, grasp.object_id, grasp.grasp_id
        obj = self.object(ref.handle)
        z = {"": obj.center[2], "top": obj.top_z, "bottom": obj.bottom_z}[ref.point]
        x, y = obj.center[0] + offset["dx"], obj.center[1] + offset["dy"]
        orientation = self._orientation(index, {})
        return Pose(x, y, z + offset["dz"], *orientation), obj.object_id, ""

    def _move_joints(self, index: int, step: Mapping[str, Any]) -> dict[str, Any]:
        raw = step.get("joints") or {}
        if not raw:
            raise _invalid(index, "move_joints needs joints {name: deg}")
        unknown = sorted(str(name) for name in raw if name not in self._joints)
        if unknown:
            raise _invalid(
                index, f"{', '.join(unknown)} are not joints of this arm ({sorted(self._joints)})"
            )
        joints: dict[str, float] = {}
        for name, value in raw.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _invalid(index, f"joint {name} must be a number")
            if not math.isfinite(float(value)):
                raise _invalid(index, f"joint {name} must be finite")
            joints[str(name)] = float(value)
        if not self._kinematics_checked:
            self._ctx.motion.check_kinematics(self._start)
            self._kinematics_checked = True
        target = self._ctx.motion.check_joints(joints, self._joint_state)
        if self._joint_state is None:
            self._joint_state = dict(joints)
        else:
            self._joint_state.update(joints)
        self._previous = target
        self.targets.append(target)
        return {
            "joints": joints,
            "pose": target.to_dict(),
            "text": f"move {len(joints)} joint(s) to TCP {_fmt(target)}",
        }

    def _gripper(self, index: int, step: Mapping[str, Any]) -> dict[str, Any]:
        if not self._ctx.workcell.role("gripper"):
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, "this workcell binds no gripper")
        if ("width_mm" in step) == ("grip" in step):
            raise _invalid(index, "a gripper step needs exactly one of width_mm, grip")
        widest = float(self._ctx.workcell.gripper_spec["max_width_mm"])
        obj_id = grasp_id = ""
        grasp: GraspCandidate | None = None
        if "ref" in step:
            ref = _parse_ref(step["ref"], index=index)
            if ref.point:
                raise _invalid(index, "a gripper ref names a handle only (grasp:g1, object:o2)")
            if ref.kind == "grasp":
                grasp = self.grasp(ref.handle)
                obj_id, grasp_id = grasp.object_id, grasp.grasp_id
            else:
                obj_id = self.object(ref.handle).object_id
        if "width_mm" in step:
            width = float(step["width_mm"])
            text = f"set the gripper to {width:.0f} mm"
        elif step["grip"] == "open":
            width = grasp.open_mm if grasp is not None else widest
            text = f"open the gripper to {width:.0f} mm"
        else:
            width, text = 0.0, "close the gripper"
        if width > widest:
            raise _invalid(index, f"width {width:.0f} mm exceeds the gripper's {widest:.0f} mm")
        if obj_id:
            text += f" on {grasp_id or obj_id}"
        entry: dict[str, Any] = {"width_mm": width, "object": obj_id, "grasp": grasp_id}
        if "effort" in step:
            entry["effort"] = float(step["effort"])
        return {**entry, "text": text}

    def _wait(self, index: int, step: Mapping[str, Any]) -> dict[str, Any]:
        if "seconds" not in step:
            raise _invalid(index, "wait needs seconds")
        seconds = float(step["seconds"])
        longest = float(self._ctx.limits["max_wait_s"])
        if seconds > longest:
            raise _invalid(index, f"wait {seconds:g} s exceeds the workcell's {longest:g} s")
        return {"seconds": seconds, "text": f"wait {seconds:g} s"}

    def object(self, object_id: str) -> LocatedObject:
        if object_id not in self.objects:
            self.objects[object_id] = self._ctx.scene.object(object_id)
        return self.objects[object_id]

    def grasp(self, grasp_id: str) -> GraspCandidate:
        if grasp_id not in self.grasps:
            grasp = self._ctx.scene.grasp(grasp_id)
            self.grasps[grasp_id] = grasp
            self.object(grasp.object_id)
        return self.grasps[grasp_id]


def _current_hold(ctx: SkillContext, state: ArmState) -> HeldObject | None:
    """The held object, forgetting it when the jaws no longer report its width."""
    held = ctx.scene.held
    if held is None or state.gripper_mm is None:
        return held
    if abs(state.gripper_mm - held.width_mm) <= ctx.workcell.freshness["hold_tolerance_mm"]:
        return held
    ctx.scene.release()
    ctx.note(
        "hold_forgotten",
        {
            "object_id": held.obj.object_id,
            "held_mm": round(held.width_mm, 1),
            "gripper_mm": round(state.gripper_mm, 1),
        },
    )
    return None


# ── Execute ──────────────────────────────────────────────────────────────────


def _check_drift(fresh: Mapping[str, float], planned: LocatedObject, seen: LocatedObject) -> None:
    planar = math.hypot(seen.center[0] - planned.center[0], seen.center[1] - planned.center[1])
    height = abs(seen.top_z - planned.top_z)
    problems = []
    if planar > fresh["drift_planar_mm"]:
        problems.append(f"moved {planar:.0f} mm")
    if height > fresh["drift_height_mm"]:
        problems.append(f"its top changed {height:.0f} mm")
    if min(planned.length - planned.width, seen.length - seen.width) > fresh["yaw_defined_mm"]:
        turned = abs(wrap_axis_deg(seen.yaw_deg - planned.yaw_deg))
        if turned > fresh["drift_yaw_deg"]:
            problems.append(f"turned {turned:.0f} deg")
    if problems:
        raise SkillError(
            FAILURE_PLAN_STALE,
            f"{planned.object_id} ({planned.label}) {', '.join(problems)} since planning; "
            "locate and plan again",
        )


class _Run:
    """Executes approved steps, tracking the jaws, the hold and the last TCP target."""

    def __init__(
        self,
        ctx: SkillContext,
        plan: SkillPlan,
        objects: Mapping[str, LocatedObject],
        state: ArmState,
    ) -> None:
        self._ctx = ctx
        self._plan = plan
        self._objects = objects
        self._grasps = {
            gid: GraspCandidate.from_mapping(data) for gid, data in plan.basis["grasps"].items()
        }
        self._tolerance = float(ctx.workcell.freshness["hold_tolerance_mm"])
        self._min_hold = float(ctx.workcell.gripper_spec["min_hold_width_mm"])
        self.state = state
        self.jaw = state.gripper_mm
        held = ctx.scene.held
        self._grip = None if held is None else held.width_mm  # width the jaws closed at
        self._target: Pose | None = state.tcp
        self._gripped: GripperResult | None = None
        self.log: list[dict[str, Any]] = []

    async def step(self, index: int, step: Mapping[str, Any]) -> None:
        op = step["op"]
        before = self.jaw
        self._gripped = None
        if op in _POSE_OPS:
            target = Pose.from_mapping(step["pose"])
            moved = await self._ctx.motion.move_tcp(
                target, speed_pct=self._speed(step), path=step["path"]
            )
            self._observed(moved.state)
            self._target = target
        elif op == OP_MOVE_JOINTS:
            moved = await self._ctx.motion.move_joints(step["joints"], speed_pct=self._speed(step))
            self._observed(moved.state)
            self._target = moved.state.tcp
        elif op == OP_GRIPPER:
            await self._gripper(step)
        else:
            await self._ctx.motion.pause(float(step["seconds"]))
            self._observed(await self._ctx.motion.state())
        if op == OP_GRIPPER and step["expect"] == EXPECT_REACHED:
            self._observed(await self._ctx.motion.state())
        self.log.append(
            {
                "step": index,
                "op": op,
                "tcp": self.state.tcp.to_dict(1),
                "gripper_mm": None if self.jaw is None else round(self.jaw, 1),
            }
        )
        self._guard(index, step, before)

    def _speed(self, step: Mapping[str, Any]) -> float | None:
        run = self._plan.limits.get("speed_pct")
        own = step.get("speed_pct")
        if own is None:
            return run
        return float(own) if run is None else min(float(own), float(run))

    def _observed(self, state: ArmState) -> None:
        """Take a fresh state; a held width that drifted means the object is gone."""
        self.state = state
        if state.gripper_mm is not None:
            self.jaw = state.gripper_mm
        if self._grip is None or self.jaw is None:
            return
        if abs(self.jaw - self._grip) > self._tolerance:
            self._grip = None
            _current_hold(self._ctx, state)

    async def _gripper(self, step: Mapping[str, Any]) -> None:
        effort = step.get("effort")
        result = await self._ctx.motion.gripper(float(step["width_mm"]), effort=effort)
        self._gripped = result
        self.jaw = result.width_mm
        scene = self._ctx.scene
        if result.blocked and result.width_mm >= self._min_hold:
            self._grip = result.width_mm
            obj = self._objects.get(step.get("object") or "")
            if obj is not None:
                grasp = self._grasps.get(step.get("grasp") or "")
                scene.hold(HeldObject(obj, result.width_mm, time.time(), grasp))
        elif not result.blocked:
            self._grip = None
            if scene.held is not None:
                scene.release()

    def _guard(self, index: int, step: Mapping[str, Any], before: float | None) -> None:
        expect = step["expect"]
        if not expect:
            return
        failure = self._violation(expect, before)
        if failure is None:
            return
        stop = step["on_fail"] == ON_FAIL_STOP
        raise SkillError(
            FAILURE_GUARD_FAILED,
            f"step {index} ({step['text']}): expected {expect}, but {failure}; the arm "
            + ("was halted" if stop else "holds its position"),
            side_effect_state=SIDE_EFFECT_PARTIAL,
            status=STATUS_HALTED if stop else STATUS_FAILED,
            result={
                "guard": expect,
                "tcp": self.state.tcp.to_dict(1),
                "gripper_mm": None if self.jaw is None else round(self.jaw, 1),
            },
        )

    def _violation(self, expect: str, before: float | None) -> str | None:
        jaw = self.jaw
        if expect == EXPECT_HELD:
            if jaw is None:
                return "the gripper width is unreadable"
            if self._grip is None or jaw < self._min_hold:
                return f"the jaws are at {jaw:.1f} mm with nothing between them"
            return None
        if expect == EXPECT_RELEASED:
            gripped = self._gripped
            if gripped is not None and gripped.blocked:
                return f"the jaws stopped at {gripped.width_mm:.1f} mm of {gripped.target_mm:.1f}"
            return None
        target = self._target
        if target is None or not self._ctx.motion.reached(target, self.state):
            where = "unknown" if target is None else f"{self.state.tcp.distance_mm(target):.1f} mm"
            return f"the tool is off its target ({where})"
        if expect == EXPECT_NO_CONTACT and before is not None and jaw is not None:
            if abs(jaw - before) > self._tolerance:
                return f"the jaws were pushed from {before:.1f} to {jaw:.1f} mm"
        return None


# ── Skill ────────────────────────────────────────────────────────────────────


class MotionSequence:
    """A model-written motion program: resolved and checked at plan, guarded at execute."""

    spec = SkillSpec(
        skill_id="motion.sequence",
        version="1.0.0",
        card=_CARD,
        args_schema={
            "type": "object",
            "properties": {
                "steps": {"type": "array", "items": _STEP_SCHEMA, "minItems": 1, "maxItems": 64},
                "speed_pct": _SPEED_ARG,
            },
            "required": ["steps"],
        },
        effect=EFFECT_ACTUATE,
        requires=(
            f"{REQUIRE_ROLE}arm",
            f"{REQUIRE_CALIBRATION}tcp",
            f"{REQUIRE_CALIBRATION}workspace",
        ),
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        steps = args["steps"]
        most = int(ctx.limits["max_steps"])
        if len(steps) > most:
            raise SkillError(
                FAILURE_INVALID_ARGS,
                f"{len(steps)} steps exceed this workcell's limit of {most}; split the program",
            )
        state = await ctx.motion.state()
        held = _current_hold(ctx, state)
        planner = _Planner(ctx, state)
        resolved = [planner.resolve(i, step) for i, step in enumerate(steps, 1)]
        listed = "; ".join(f"{i}) {step['text']}" for i, step in enumerate(resolved, 1))
        speed = args.get("speed_pct")
        return SkillPlan(
            skill_id=self.spec.skill_id,
            args=args,
            summary=f"Run {len(resolved)} step(s) from {_fmt(state.tcp)}: {listed}",
            targets=tuple(planner.targets),
            limits={} if speed is None else {"speed_pct": float(speed)},
            basis={
                "start": state.tcp.to_dict(),
                "held": "" if held is None else held.obj.object_id,
                "objects": {oid: obj.to_dict() for oid, obj in planner.objects.items()},
                "grasps": {gid: grasp.to_dict() for gid, grasp in planner.grasps.items()},
                "steps": resolved,
            },
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        fresh = ctx.workcell.freshness
        state = await ctx.motion.state()
        start = Pose.from_mapping(plan.basis["start"])
        moved_mm, moved_deg = state.tcp.distance_mm(start), state.tcp.rotation_to_deg(start)
        if moved_mm > fresh["start_mm"] or moved_deg > fresh["start_deg"]:
            raise SkillError(
                FAILURE_PLAN_STALE,
                f"the tool moved {moved_mm:.1f} mm / {moved_deg:.1f} deg since planning; "
                "plan again",
            )
        held = _current_hold(ctx, state)
        if ("" if held is None else held.obj.object_id) != plan.basis["held"]:
            raise SkillError(FAILURE_PLAN_STALE, "what the gripper holds changed since planning")
        objects: dict[str, LocatedObject] = {}
        for oid, data in plan.basis["objects"].items():
            planned = LocatedObject.from_mapping(data)
            seen = await ctx.scene.relocate(planned)
            _check_drift(fresh, planned, seen)
            objects[oid] = seen
        run = _Run(ctx, plan, objects, state)
        for index, step in enumerate(plan.basis["steps"], 1):
            try:
                await run.step(index, step)
            except SkillError as error:
                error.result = {**error.result, "step": index, "completed": index - 1}
                raise
        held = ctx.scene.held
        return SkillOutcome.succeeded(
            {
                "steps": run.log,
                "tcp": run.state.tcp.to_dict(),
                "gripper_mm": None if run.jaw is None else round(run.jaw, 1),
                "held": None if held is None else held.to_dict(),
            },
            message=f"Ran {len(run.log)} step(s); the tool is at {_fmt(run.state.tcp)}",
        )


def skills() -> tuple[MotionSequence]:
    """Entry point for ``leapflow.hardware.skills``."""
    return (MotionSequence(),)


__all__ = [
    "EXPECT_HELD",
    "EXPECT_NO_CONTACT",
    "EXPECT_REACHED",
    "EXPECT_RELEASED",
    "MotionSequence",
    "ON_FAIL_HOLD",
    "ON_FAIL_STOP",
    "OPS",
    "skills",
]
