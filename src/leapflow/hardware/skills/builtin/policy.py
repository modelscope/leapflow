# Copyright (c) Alibaba, Inc. and its affiliates.
"""Closed-loop policy execution: stream a served visuomotor policy for a while.

``policy.run`` is the System 1.5 bridge for work no discrete skill describes --
a task a policy (SmolVLA, EdgeVLA, ACT, ...) was trained on. Planning runs one
observation and inference without moving, so the approver sees where the first
chunk would take the tool and whether the safety shield would let it; because
the rest of the run is closed-loop and nobody can review it pose by pose, every
run needs its own consent.

Execution streams the policy through its declared action adapter and the
shield: every tick is bounded, jump-, acceleration- and slew-limited and
pose-checked like a discrete move, then passes the node write gate. The run
ends early when the model reports the instruction done; ``duration_s`` is a
safety cap. A policy failure (model down, starvation, shield refusal) leaves
the arm at its last approved target; a stop or a device fault halts it as for
every skill. The approved plan pins the model identity and the adapter digest,
so a model or embodiment changed since approval is ``plan_stale``.
"""

from __future__ import annotations

from typing import Any, Mapping

from leapflow.hardware.skills.ports import SkillContext
from leapflow.hardware.skills.protocol import (
    CONSENT_EACH_RUN,
    EFFECT_ACTUATE,
    FAILURE_INVALID_ARGS,
    FAILURE_SHIELD_VIOLATION,
    REQUIRE_CALIBRATION,
    REQUIRE_ROLE,
    PolicyReport,
    SkillError,
    SkillOutcome,
    SkillPlan,
    SkillSpec,
)


def _fmt_end(report: PolicyReport) -> str:
    end = report.end
    if end is None:
        return ""
    return f"; the first chunk ends near ({end.x:.0f}, {end.y:.0f}, {end.z:.0f}) mm"


class RunPolicy:
    """Stream a served action-chunk policy for a bounded time."""

    spec = SkillSpec(
        skill_id="policy.run",
        version="2.0.0",
        card=(
            "Run a learned visuomotor policy closed-loop for at most duration_s seconds. "
            "'policy' is a model the workcell binds under 'models' that serves action "
            "chunks; 'task' is the instruction for it. A model whose instruction_policy is "
            "'exact' accepts only the tasks it declares; 'open' takes free language. "
            "Planning previews the first chunk without moving; every action is checked by "
            "the safety shield and the run stops at the first refusal, leaving the arm "
            "where it is. The run ends when the model reports done or duration_s elapses "
            "(result 'ended'); 'progress' is the model's own estimate when it reports one, "
            "not a verification. The run's duration cap bounds duration_s."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "policy": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                    "description": "The model name bound under the workcell's 'models'.",
                },
                "task": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "description": "The instruction the policy should perform.",
                },
                "duration_s": {
                    "type": "number",
                    "minimum": 0.5,
                    "default": 30.0,
                    "description": "Upper bound on streaming after the first chunk arrives.",
                },
                "speed_pct": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Speed, percent of the arm maximum; capped by the workcell.",
                },
            },
            "required": ["policy", "task"],
        },
        effect=EFFECT_ACTUATE,
        requires=(
            f"{REQUIRE_ROLE}arm",
            f"{REQUIRE_CALIBRATION}tcp",
            f"{REQUIRE_CALIBRATION}workspace",
        ),
        consent=CONSENT_EACH_RUN,
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        duration = float(args["duration_s"])
        startup = float(ctx.policies.startup_s)
        cap = float(ctx.limits["max_duration_s"])
        if duration + startup > cap:
            raise SkillError(
                FAILURE_INVALID_ARGS,
                f"duration_s {duration:g} plus {startup:g} s of start-up exceeds the run cap of "
                f"{cap:g} s; ask for at most {max(0.0, cap - startup):g} s",
            )
        report = await ctx.policies.preview(args["policy"], args["task"])
        if report.violation:
            raise SkillError(
                FAILURE_SHIELD_VIOLATION,
                f"the safety shield refuses the policy's first chunk: {report.violation}",
                result={"policy": report.to_dict()},
            )
        card = report.card
        limits: dict[str, float] = {"max_duration_s": duration + startup}
        if args.get("speed_pct") is not None:
            limits["speed_pct"] = float(args["speed_pct"])
        warnings = "".join(f" Warning: {warning}." for warning in card.warnings)
        done = " The model already reports the task done." if report.ended else ""
        return SkillPlan(
            skill_id=self.spec.skill_id,
            args=args,
            summary=(
                f"Run policy {card.name!r} ({card.model.get('name')} "
                f"{card.model.get('version')}, {card.space} space, {card.fps:g} Hz chunks of "
                f"{report.horizon}) on {args['task']!r} for up to {duration:.0f} s"
                f"{_fmt_end(report)}; adapter {', '.join(card.adapter) or 'none'}."
                f"{done}{warnings}"
            ),
            targets=() if report.end is None else (report.end,),
            limits=limits,
            basis={
                "model": dict(card.model),
                "space": card.space,
                "channels": list(card.channels),
                "adapter_digest": card.adapter_digest,
            },
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        name, task = plan.args["policy"], plan.args["task"]
        report = await ctx.policies.run(
            name,
            task,
            approved=plan.basis,
            duration_s=float(plan.args["duration_s"]),
            speed_pct=plan.limits.get("speed_pct"),
        )
        return SkillOutcome.succeeded(
            {"policy": report.to_dict()},
            message=(
                f"Ran {name!r} on {task!r} for {report.elapsed_s:.1f} s: {report.ticks} "
                f"actions from {report.chunks} chunk(s), {report.interventions} clamped, "
                f"ended by {report.ended or 'stop'}"
            ),
        )


def skills() -> tuple[RunPolicy, ...]:
    """Entry point for ``leapflow.hardware_skills``."""
    return (RunPolicy(),)


__all__ = ["RunPolicy", "skills"]
