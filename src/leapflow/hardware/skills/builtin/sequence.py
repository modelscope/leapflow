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
- **Reach.** The program is followed with the arm's declared kinematics from
  the joints it reports: a target the arm cannot reach inside its joint
  limits -- along the whole line, for a linear move -- is refused before
  approval with how far the arm gets, instead of halting the program midway.
- **Freshness.** ``execute`` refuses (``plan_stale``, nothing moved) when the
  tool left its planned start, the held object changed, or a referenced object
  drifted beyond the workcell's declared ``freshness`` tolerances.
- **Guards.** A step may declare what must hold after it (``held``,
  ``released``, ``reached``, ``no_contact``). A failed guard ends the program
  with the arm held where it is: ``on_fail: hold`` (the default) fails the
  run, ``stop`` also halts it, which latches the arm in place until it is
  re-enabled. Neither releases torque. A guard is checked for satisfiability
  at plan: a ``released`` the jaws can never confirm is refused before
  approval (``_sequence_guards.ReleaseRule``). The outcome names the step,
  what it targeted and what was observed.
- **Holding.** Closing on a referenced grasp or object records it as held;
  opening releases it; a held width that drifts beyond the declared tolerance
  forgets it -- so the next plan starts from what the jaws report.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any, Mapping, Sequence

from leapflow.hardware.skills.builtin._sequence_planner import prepare as prepare_sequence
from leapflow.hardware.skills.builtin._observation_gate import require_observation
from leapflow.hardware.skills.ports import PreparedPlanning

from leapflow.hardware.skills.builtin._sequence_guards import (
    ReleaseRule,
    check_drift,
    current_hold,
)
from leapflow.hardware.skills.builtin._sequence_schema import (
    CARD,
    EXPECT_HELD,
    EXPECT_NO_CONTACT,
    EXPECT_REACHED,
    EXPECT_RELEASED,
    ON_FAIL_HOLD,
    ON_FAIL_STOP,
    OP_GRIPPER,
    OP_MOVE_JOINTS,
    OP_WAIT,
    OPS,
    POSE_OPS,
    SPEED_ARG,
    STEP_SCHEMA,
)
from leapflow.hardware.skills.geometry import Pose
from leapflow.hardware.skills.ports import SkillContext
from leapflow.hardware.skills.protocol import (
    EFFECT_ACTUATE,
    FAILURE_GUARD_FAILED,
    FAILURE_PLAN_STALE,
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


def _fmt(pose: Pose) -> str:
    return f"({pose.x:.0f}, {pose.y:.0f}, {pose.z:.0f}) mm"


# Two resolved targets whose attitudes agree this closely are the same attitude,
# including whatever rounding the plan's JSON round trip applied.
_SAME_ATTITUDE_DEG = 0.05


# ── Execute ──────────────────────────────────────────────────────────────────


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
        self._released: bool | None = None
        self._release_from: float | None = None
        self._release_at: float | None = None
        self.log: list[dict[str, Any]] = []

    async def step(self, index: int, step: Mapping[str, Any]) -> None:
        op = step["op"]
        before = self.jaw
        self._gripped = None
        self._released = None
        self._release_from = None
        self._release_at = None
        if op in POSE_OPS:
            target = Pose.from_mapping(step["pose"])
            moved = await self._ctx.motion.move_tcp(
                target, speed_pct=self._speed(step), path=step["path"],
                carried_from=self._carried_from(target),
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
        self.log.append(self._entry(index, op))
        self._guard(index, step, before)

    def _carried_from(self, target: Pose) -> Pose | None:
        """The previous target when this step keeps its attitude instead of stating one.

        Planning refuses a *commanded* attitude outside the cone but lets a step that
        inherits the previous attitude leave a posture the cone would forbid, so the
        same distinction has to hold here or the approved lift would be refused at
        execution.  The comparison is between resolved targets, not the measured arm.
        """
        previous = self._target
        if previous is None or target.rotation_to_deg(previous) > _SAME_ATTITUDE_DEG:
            return None
        return previous

    def _entry(self, index: int, op: str) -> dict[str, Any]:
        """One step as it ran: what it aimed at, where it ended, and how far apart."""
        entry: dict[str, Any] = {
            "step": index,
            "op": op,
            "tcp": self.state.tcp.to_dict(1),
            "gripper_mm": None if self.jaw is None else round(self.jaw, 1),
        }
        gripped = self._gripped
        if gripped is not None:
            entry["target_mm"] = round(gripped.target_mm, 1)
            entry["error_mm"] = round(gripped.width_mm - gripped.target_mm, 2)
            entry["blocked"] = gripped.blocked
            if self._release_at is not None:
                entry["release_at_mm"] = round(self._release_at, 1)
        elif op != OP_WAIT and self._target is not None:
            entry["target"] = self._target.to_dict(1)
            entry["error_mm"] = round(self.state.tcp.distance_mm(self._target), 2)
        return entry

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
            current_hold(self._ctx, state)

    async def _gripper(self, step: Mapping[str, Any]) -> None:
        effort = step.get("effort")
        before = self.jaw
        held_width = self._grip
        result = await self._ctx.motion.gripper(float(step["width_mm"]), effort=effort)
        self._gripped = result
        self.jaw = result.width_mm
        self._release_from = held_width
        self._release_at = (
            None
            if held_width is None
            else ReleaseRule.of(self._ctx).threshold(held_width, result.target_mm)
        )
        released = self._release_at is not None and result.width_mm >= self._release_at
        self._released = released or (held_width is None and not result.blocked)
        scene = self._ctx.scene
        if released:
            self._grip = None
            if scene.held is not None:
                scene.release()
            return
        closing = before is None or result.target_mm < before - self._tolerance
        grasped = held_width is None and closing and result.blocked
        if grasped and result.width_mm >= self._min_hold:
            self._grip = result.width_mm
            obj = self._objects.get(step.get("object") or "")
            if obj is not None:
                grasp = self._grasps.get(step.get("grasp") or "")
                scene.hold(HeldObject(obj, result.width_mm, time.time(), grasp))

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
            + ("was halted in place" if stop else "holds its position"),
            side_effect_state=SIDE_EFFECT_PARTIAL,
            status=STATUS_HALTED if stop else STATUS_FAILED,
            result={
                "guard": expect,
                "tcp": self.state.tcp.to_dict(1),
                "gripper_mm": None if self.jaw is None else round(self.jaw, 1),
                "observed": self.log[-1],
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
            if self._released:
                return None
            if self._release_from is not None and jaw is not None:
                needed = self._release_at
                if needed is None:
                    target = "its target" if gripped is None else f"{gripped.target_mm:.1f} mm"
                    return (
                        f"opening to {target} cannot release the {self._release_from:.1f} mm "
                        "hold: the jaws would not clear it by more than their tolerance"
                    )
                return (
                    f"the jaws opened only to {jaw:.1f} mm; "
                    f"release requires at least {needed:.1f}"
                )
            if gripped is not None and gripped.blocked:
                return f"the jaws stopped at {gripped.width_mm:.1f} mm of {gripped.target_mm:.1f}"
            return "release was not confirmed"
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
        version="1.5.0",
        card=CARD,
        args_schema={
            "type": "object",
            "properties": {
                "steps": {"type": "array", "items": STEP_SCHEMA, "minItems": 1, "maxItems": 64},
                "speed_pct": SPEED_ARG,
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

    async def prepare(self, ctx: SkillContext, args: Sequence[Mapping[str, Any]]) -> PreparedPlanning:
        return await prepare_sequence(ctx, self.spec.skill_id, args)

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan | SkillOutcome:
        prepared = await self.prepare(ctx, [args])
        job = prepared.jobs[0]
        if isinstance(job, SkillOutcome):
            return job
        result = await ctx.calculations.evaluate(
            [job], deadline=time.monotonic() + ctx.planning_settings.timeout_s)
        prepared.check_scene(ctx)
        return result[0].value

    def alternatives(self, ctx: SkillContext, args: Mapping[str, Any], *, limit: int) -> Sequence[Mapping[str, Any]]:
        """Substitute one grasp handle consistently; never invent geometry or yaw."""
        handles = {step['ref'].partition(':')[2].partition('.')[0]
                   for step in args['steps'] if step.get('ref', '').startswith('grasp:')}
        # Mixed-grasp programs may encode distinct tasks; do not guess equivalence.
        if len(handles) != 1 or limit <= 0:
            return ()
        handle = next(iter(handles))
        original = ctx.scene.grasp(handle)
        entries = []
        for sibling in ctx.scene.grasps(original.object_id):
            if sibling.grasp_id == handle:
                continue
            steps = []
            for step in args['steps']:
                entry = dict(step)
                ref = entry.get('ref', '')
                if ref.partition(':')[0] == 'grasp' and ref.partition(':')[2].partition('.')[0] == handle:
                    _, dot, point = ref.partition('.')
                    entry['ref'] = f'grasp:{sibling.grasp_id}' + (dot + point if dot else '')
                steps.append(entry)
            entries.append({'id': f'{handle}-to-{sibling.grasp_id}', 'args': {**args, 'steps': steps}})
            if len(entries) >= limit:
                break
        return entries

    async def confirm(self, ctx: SkillContext, plan: SkillPlan) -> SkillPlan:
        """Dry-run/live previews observe without changing the approved snapshot."""
        checks = []
        for oid, data in plan.basis['objects'].items():
            obj = LocatedObject.from_mapping(data)
            seen = await ctx.scene.relocate(obj, commit=False)
            check_drift(ctx.workcell.freshness, obj, seen)
            checks.append({'object_id': oid, 'camera': seen.camera, 'state': 'confirmed',
                           'table_offset_mm': seen.table_offset_mm})
        return replace(plan, basis={**plan.basis, 'object_recheck': {'read_only': True, 'checks': checks}})

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        fresh = ctx.workcell.freshness
        objects: dict[str, LocatedObject] = {}
        for oid, data in plan.basis["objects"].items():
            planned = LocatedObject.from_mapping(data)
            seen = await ctx.scene.relocate(planned)
            check_drift(fresh, planned, seen)
            objects[oid] = seen
        for data in plan.basis['grasps'].values():
            grasp = GraspCandidate.from_mapping(data)
            require_observation(ctx, objects[grasp.object_id], grasp)
        # Read the start after observation I/O, immediately before exact execution.
        state = await ctx.motion.state()
        start = Pose.from_mapping(plan.basis["start"])
        moved_mm, moved_deg = state.tcp.distance_mm(start), state.tcp.rotation_to_deg(start)
        if moved_mm > fresh["start_mm"] or moved_deg > fresh["start_deg"]:
            raise SkillError(
                FAILURE_PLAN_STALE,
                f"the tool moved {moved_mm:.1f} mm / {moved_deg:.1f} deg since planning; "
                "plan again",
            )
        for name, tolerance in plan.basis.get("start_joint_tolerances", {}).items():
            expected = plan.basis.get("start_joints", {}).get(name)
            actual = state.joints.get(name)
            if (expected is None or actual is None or not math.isfinite(actual)
                    or abs(actual - expected) > tolerance):
                raise SkillError(FAILURE_PLAN_STALE,
                                 f"starting joint {name} changed or is missing; plan again")
        held = current_hold(ctx, state)
        if ("" if held is None else held.obj.object_id) != plan.basis["held"]:
            raise SkillError(FAILURE_PLAN_STALE, "what the gripper holds changed since planning")
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
