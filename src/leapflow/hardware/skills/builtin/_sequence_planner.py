# Copyright (c) Alibaba, Inc. and its affiliates.
"""Pure sequence resolution over one detached owner-captured snapshot."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

from leapflow.hardware.skills.builtin._sequence_guards import ReleaseRule, current_hold
from leapflow.hardware.skills.builtin._sequence_schema import (
    COMMON_FIELDS, EXPECT_REACHED, EXPECT_RELEASED, EXPECTS, FIELDS, GRASP_POINTS,
    OBJECT_POINTS, ON_FAIL_STOP, OP_GRIPPER, OP_MOVE_JOINTS, OP_MOVE_LINEAR, POSE_OPS,
)
from leapflow.hardware.skills.geometry import Plane, Pose, mat_vec, tool_cone_deg
from leapflow.hardware.skills.ports import (
    PATH_DEFAULT, PATH_LINEAR, PATH_POINT, SkillContext, MotionGeometry,
    Evaluation, FrozenJoints, PlanningCheckpoint, PreparedPlanning, encoded,
)
from leapflow.hardware.skills.protocol import (
    FAILURE_CALIBRATION_MISSING, FAILURE_INVALID_ARGS, FAILURE_OUT_OF_WORKSPACE,
    FAILURE_SKILL_UNAVAILABLE, FAILURE_TARGET_NOT_FOUND, FAILURE_UNREACHABLE,
    ArmState, GraspCandidate, HeldObject, LocatedObject, SkillError, SkillOutcome, SkillPlan,
)


@dataclass(frozen=True)
class _Ref:
    kind: str  # "object" or "grasp"
    handle: str
    point: str


def _parse_ref(text: str, *, index: int) -> _Ref:
    kind, _, rest = text.partition(":")
    handle, _, point = rest.partition(".")
    points = OBJECT_POINTS if kind == "object" else GRASP_POINTS if kind == "grasp" else ()
    if not handle or (point not in points and not (kind == "grasp" and point == "")):
        raise _invalid(index, f"ref {text!r} must be object:<id>[.top|.bottom] or grasp:<id>[.pregrasp|.grasp|.retreat]")
    return _Ref(kind, handle, point)


def _invalid(index: int, message: str) -> SkillError:
    return SkillError(FAILURE_INVALID_ARGS, f"step {index}: {message}")


def _joint_channel(name: str, joints: frozenset[str]) -> str:
    """Resolve an unambiguous short name without guessing among several channels."""
    if name in joints:
        return name
    matches = [joint for joint in joints if name in joint.split(".")]
    return matches[0] if len(matches) == 1 else name


def _fmt(pose: Pose) -> str:
    return f"({pose.x:.0f}, {pose.y:.0f}, {pose.z:.0f}) mm"


def _listed(step: Mapping[str, Any]) -> str:
    expect = step["expect"]
    if not expect:
        return str(step["text"])
    reaction = "halt in place" if step["on_fail"] == ON_FAIL_STOP else "hold"
    return f"{step['text']} [expect {expect}, else {reaction}]"


def _envelope(start: Pose, targets: list[Pose]) -> dict[str, float]:
    poses = [start, *targets]
    return {
        "z_min_mm": round(min(pose.z for pose in poses), 1),
        "z_max_mm": round(max(pose.z for pose in poses), 1),
        "tilt_max_deg": round(max(tool_cone_deg(pose) for pose in targets), 1) if targets else 0.0,
    }


def _envelope_text(envelope: Mapping[str, float], speed: float | None) -> str:
    text = (f"Tool z spans {envelope['z_min_mm']:.0f}-{envelope['z_max_mm']:.0f} mm; "
            f"it tilts at most {envelope['tilt_max_deg']:.0f} deg from straight down")
    return text + ("." if speed is None else f"; speed capped at {speed:.0f}%.")


@dataclass(frozen=True)
class PlanningCell:
    has_gripper: bool
    gripper_json: str
    freshness_json: str

    def role(self, name: str) -> str:
        return "gripper" if name == "gripper" and self.has_gripper else ""

    @property
    def gripper_spec(self) -> Mapping[str, float]:
        return json.loads(self.gripper_json)

    @property
    def freshness(self) -> Mapping[str, float]:
        return json.loads(self.freshness_json)


@dataclass(frozen=True)
class PlanningFrames:
    table: Plane | None


@dataclass(frozen=True)
class PlanningScene:
    objects_json: str
    grasps_json: str

    def object(self, object_id: str) -> LocatedObject:
        value = json.loads(self.objects_json).get(object_id)
        if value is None:
            raise SkillError(FAILURE_TARGET_NOT_FOUND, f"object {object_id!r} is not in this snapshot")
        return LocatedObject(**{**value, "center": tuple(value["center"])})

    def grasp(self, grasp_id: str) -> GraspCandidate:
        value = json.loads(self.grasps_json).get(grasp_id)
        if value is None:
            raise SkillError(FAILURE_TARGET_NOT_FOUND, f"grasp {grasp_id!r} is not in this snapshot")
        return GraspCandidate.from_mapping(value)


@dataclass(frozen=True)
class PlanningView:
    motion: MotionGeometry
    workcell: PlanningCell
    frames: PlanningFrames
    scene: PlanningScene
    limits_json: str

    @property
    def limits(self) -> Mapping[str, float]:
        return json.loads(self.limits_json)


class _Planner:
    """Resolves steps in order, tracking predicted TCP and joint targets."""

    def __init__(self, ctx: PlanningView, state: ArmState, held: HeldObject | None) -> None:
        self._ctx = ctx
        self._start = state
        self._previous: Pose | None = state.tcp
        # The width the jaws are predicted to hold at, when it is known: what they
        # hold now, or the width of a grasp the program closes on.
        self._held_mm: float | None = None if held is None else held.width_mm
        self._joints = frozenset(state.joints)
        self._joint_state: dict[str, float] | None = dict(state.joints)
        self._kinematics_checked = False
        self.reach = ctx.motion.reach_tracker(state)
        self.objects: dict[str, LocatedObject] = {}
        self.grasps: dict[str, GraspCandidate] = {}
        self.targets: list[Pose] = []
        self.problems: list[dict[str, Any]] = []

    def resolve(self, index: int, step: Mapping[str, Any]) -> dict[str, Any]:
        self._ctx.motion.check()
        op = step["op"]
        extra = sorted(set(step) - COMMON_FIELDS - FIELDS[op])
        if extra:
            raise _invalid(index, f"{', '.join(extra)} do(es) not apply to op {op}")
        expect = step.get("expect", "")
        if expect and expect not in EXPECTS[op]:
            raise _invalid(index, f"expect {expect} does not apply to op {op}")
        resolved: dict[str, Any] = {"op": op, "expect": expect, "on_fail": step["on_fail"]}
        if "speed_pct" in step:
            resolved["speed_pct"] = float(step["speed_pct"])
        if op in POSE_OPS:
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
        path = self._path(index, step)
        entry: dict[str, Any] = {"path": path}
        if "pose" in step:
            target = self._absolute(index, step["pose"])
            text = f"to {_fmt(target)}"
        elif "relative" in step:
            target = self._relative(index, step["relative"])
            rel = step["relative"]
            text = (f"by ({rel['dx']:+.0f}, {rel['dy']:+.0f}, {rel['dz']:+.0f}) mm "
                    f"({rel['frame']} frame) to {_fmt(target)}")
        else:
            target, entry["object"], entry["grasp"] = self._referenced(index, step)
            text = f"to {step['ref']} at {_fmt(target)}"
        try:
            self._ctx.motion.check_target(target)
            self.reach.move(target, linear=self._linear(path), label=f"step {index}")
        except SkillError as error:
            if error.failure_code not in {FAILURE_UNREACHABLE, FAILURE_OUT_OF_WORKSPACE}:
                raise
            self.collect_problem(index, error)
        self._previous = target
        self._joint_state = None
        self.targets.append(target)
        verb = "move" if path == PATH_DEFAULT else f"move {path}"
        return {**entry, "pose": target.to_dict(), "text": f"{verb} {text}"}

    def collect_problem(self, index: int, error: SkillError) -> None:
        message = error.message
        if not message.startswith(f"step {index}:"):
            message = f"step {index}: {message}"
        self.problems.append({**error.result, "step": index,
                             "failure_code": error.failure_code, "error": message})

    def reject_joint_step(self, index: int, error: SkillError) -> None:
        """An invalid joint endpoint cannot establish a TCP for relative steps."""
        self.collect_problem(index, error)
        self._previous, self._joint_state = None, None

    def _path(self, index: int, step: Mapping[str, Any]) -> str:
        declared = self._ctx.motion.paths
        if step["op"] == OP_MOVE_LINEAR:
            if PATH_LINEAR not in declared:
                raise SkillError(FAILURE_SKILL_UNAVAILABLE,
                    f"step {index}: this arm declares no linear path (arm.pose_path)")
            return PATH_LINEAR
        named = step.get("path")
        if named is None:
            return PATH_POINT if PATH_POINT in declared else PATH_DEFAULT
        if named not in declared:
            available = ", ".join(declared) or "none"
            raise _invalid(index, f"path {named!r} is not declared by this arm (declared: {available})")
        return str(named)

    def _linear(self, path: str) -> bool:
        if path == PATH_DEFAULT:
            path = next(iter(self._ctx.motion.paths), PATH_DEFAULT)
        return path == PATH_LINEAR

    def _orientation(self, index: int, data: Mapping[str, Any]) -> tuple[float, float, float]:
        previous = self._previous
        if previous is None and not all(axis in data for axis in ("rx", "ry", "rz")):
            raise _invalid(index, "after move_joints a target must give rx, ry and rz")
        return tuple(  # type: ignore[return-value]
            float(data[axis]) if axis in data else float(getattr(previous, axis))
            for axis in ("rx", "ry", "rz")
        )

    def _absolute(self, index: int, pose: Mapping[str, Any]) -> Pose:
        if not any(axis in pose for axis in ('x', 'y', 'z', 'rx', 'ry', 'rz')):
            raise _invalid(index, 'pose needs at least one coordinate or angle; use relative for offsets')
        previous = self._previous
        if previous is None and not all(axis in pose for axis in ('x', 'y', 'z')):
            raise _invalid(index, 'a partial pose needs a known previous TCP target')
        x, y, z = (float(pose[axis]) if axis in pose else float(getattr(previous, axis))
                   for axis in ('x', 'y', 'z'))
        if pose["frame"] == "table":
            table = self._ctx.frames.table
            if table is None:
                raise SkillError(FAILURE_CALIBRATION_MISSING,
                    f"step {index}: a table-frame pose needs the table calibrated "
                    "(calib.workspace with table_z or table_points)")
            # Omitted z preserves absolute height, not an implicit table offset.
            if 'z' in pose:
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
        named = {_joint_channel(str(name), self._joints): value for name, value in raw.items()}
        unknown = sorted(name for name in named if name not in self._joints)
        if unknown:
            raise _invalid(index, f"{', '.join(unknown)} are not joints of this arm ({sorted(self._joints)})")
        if len(named) != len(raw):
            raise _invalid(index, "a joint is named twice (by its short and its full name)")
        joints: dict[str, float] = {}
        for name, value in named.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _invalid(index, f"joint {name} must be a number")
            if not math.isfinite(float(value)):
                raise _invalid(index, f"joint {name} must be finite")
            joints[str(name)] = float(value)
        if not self._kinematics_checked:
            self._ctx.motion.check_kinematics(self._start)
            self._kinematics_checked = True
        target = self._ctx.motion.check_joints(joints, self._joint_state, observe=self.reach.observe_joints)
        if self._joint_state is None:
            self._joint_state = dict(joints)
        else:
            self._joint_state.update(joints)
        self.reach.joints_moved(joints, target)
        self._previous = target
        self.targets.append(target)
        return {"joints": joints, "pose": target.to_dict(),
                "text": f"move {len(joints)} joint(s) to TCP {_fmt(target)}"}

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
        self._predict_hold(index, step, width, widest, grasp)
        if obj_id:
            text += f" on {grasp_id or obj_id}"
        entry: dict[str, Any] = {"width_mm": width, "object": obj_id, "grasp": grasp_id}
        if "effort" in step:
            entry["effort"] = float(step["effort"])
        return {**entry, "text": text}

    def _predict_hold(self, index: int, step: Mapping[str, Any], width: float,
                      widest: float, grasp: GraspCandidate | None) -> None:
        """Refuse a release no width can confirm; follow what the jaws will hold."""
        held = self._held_mm
        if held is None:
            # Closing on a planned grasp holds at its width; any other close holds
            # something of unknown width, which only the run can judge.
            if grasp is not None and width < grasp.width_mm:
                self._held_mm = grasp.width_mm
            return
        rule = ReleaseRule.of(self._ctx)
        if step.get("expect") == EXPECT_RELEASED:
            rule.check(index, held, width, widest)
        if rule.threshold(held, width) is not None:
            self._held_mm = None

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


@dataclass(frozen=True)
class SequenceJob:
    skill_id: str
    view: PlanningView
    state: ArmState
    held: HeldObject | None
    args_json: str

    def evaluate(self, checkpoint: PlanningCheckpoint) -> Evaluation:
        ctx = replace(self.view, motion=self.view.motion.bind(checkpoint))
        args = json.loads(self.args_json)
        most = int(ctx.limits["max_steps"])
        if len(args["steps"]) > most:
            raise SkillError(FAILURE_INVALID_ARGS,
                f"{len(args['steps'])} steps exceed this workcell's limit of {most}; split the program")
        planner = _Planner(ctx, self.state, self.held)
        try:
            resolved = self._resolve(planner, args["steps"])
        except SkillError as error:
            return Evaluation(SkillOutcome.from_error(error), evidence={"ik_calls": planner.reach.ik_calls})
        speed = args.get("speed_pct")
        envelope = _envelope(self.state.tcp, planner.targets)
        reach = planner.reach.to_dict()
        checked = ("Reach checked with the arm's kinematics." if reach["verified"]
                   else f"Reach not verified: {reach['detail'].rstrip('.')}.")
        poses = [self.state.tcp, *planner.targets]
        metrics = {**planner.reach.metrics(),
            "tcp_distance_mm": sum(a.distance_mm(b) for a, b in zip(poses, poses[1:])),
            "rotation_deg": sum(a.rotation_to_deg(b) for a, b in zip(poses, poses[1:])),
            "wait_s": sum(float(s.get("seconds", 0)) for s in resolved)}
        listed = "; ".join(f"{i}) {_listed(step)}" for i, step in enumerate(resolved, 1))
        plan = SkillPlan(skill_id=self.skill_id, args=args,
            summary=f"Run {len(resolved)} step(s) from {_fmt(self.state.tcp)}: {listed}. {_envelope_text(envelope, speed)} {checked}",
            targets=tuple(planner.targets), limits={} if speed is None else {"speed_pct": float(speed)},
            basis={"start": self.state.tcp.to_dict(), "start_joints": dict(self.state.joints),
                "start_joint_tolerances": {n: float(e.tolerance) if e.tolerance is not None else 0.0
                                                           for n, e in ctx.motion.joint_envelopes},
                "held": "" if self.held is None else self.held.obj.object_id,
                "objects": {oid: obj.to_dict() for oid, obj in planner.objects.items()},
                "grasps": {gid: grasp.to_dict() for gid, grasp in planner.grasps.items()},
                "steps": resolved, "envelope": envelope, "reach": reach, "metrics": metrics})
        return Evaluation(plan, metrics, reach["verified"], {"ik_calls": planner.reach.ik_calls})

    def _resolve(self, planner: _Planner, steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        resolved = []
        for index, step in enumerate(steps, 1):
            try:
                resolved.append(planner.resolve(index, step))
            except SkillError as error:
                if error.failure_code in {FAILURE_UNREACHABLE, FAILURE_OUT_OF_WORKSPACE}:
                    planner.reject_joint_step(index, error)
                    continue
                error.result = {**error.result, "step": index}
                if planner.problems:
                    error.result = {**error.result, "problems": planner.problems}
                raise
        if planner.problems:
            first = planner.problems[0]
            raise SkillError(first["failure_code"], "; ".join(p["error"] for p in planner.problems),
                result={**{k: v for k, v in first.items() if k not in {"failure_code", "error"}},
                        "problems": planner.problems})
        return resolved


async def prepare(ctx: SkillContext, skill_id: str,
                  args: Sequence[Mapping[str, Any]]) -> PreparedPlanning:
    state = await ctx.motion.state()
    state = replace(state, joints=FrozenJoints.capture(state.joints))
    held = current_hold(ctx, state)
    motion = ctx.motion.geometry_snapshot()
    objects: dict[str, LocatedObject] = {}
    grasps: dict[str, GraspCandidate] = {}
    errors: dict[int, SkillOutcome] = {}
    for candidate, arguments in enumerate(args):
        try:
            for index, step in enumerate(arguments["steps"], 1):
                if "ref" not in step:
                    continue
                ref = _parse_ref(step["ref"], index=index)
                oid = ref.handle
                if ref.kind == "grasp":
                    grasp = grasps.get(ref.handle) or ctx.scene.grasp(ref.handle)
                    grasps[ref.handle] = grasp
                    oid = grasp.object_id
                if oid not in objects:
                    objects[oid] = ctx.scene.object(oid)
        except SkillError as error:
            errors[candidate] = SkillOutcome.from_error(error)
    view = PlanningView(motion, PlanningCell(bool(ctx.workcell.role("gripper")),
        encoded(dict(ctx.workcell.gripper_spec)), encoded(dict(ctx.workcell.freshness))),
        PlanningFrames(ctx.frames.table), PlanningScene(
            encoded({n: asdict(o) for n, o in objects.items()}),
            encoded({n: asdict(g) for n, g in grasps.items()})), encoded(dict(ctx.limits)))
    jobs = tuple(errors[i] if i in errors else SequenceJob(skill_id, view, state, held, encoded(a))
                 for i, a in enumerate(args))
    return PreparedPlanning(jobs, tuple(objects.values()), tuple(grasps.values()), held, state_reads=1)
