# Copyright (c) Alibaba, Inc. and its affiliates.
"""Owner-loop snapshot orchestration; only detached plugin jobs leave this boundary."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Mapping, Sequence, TYPE_CHECKING, TypeVar

from leapflow.hardware.skills.admission import ensure_model_use
from leapflow.hardware.skills.args import validate_args
from leapflow.hardware.skills.planning import (
    Deadline, Evaluation, PlanConfirmationSkill, PlanningAlternativesSkill,
    PreparedPlanning, PreparedPlanningSkill, Selection,
    candidates, encoded, no_selection,
)
from leapflow.hardware.skills.protocol import (
    FAILURE_INTERNAL_DEFECT, FAILURE_OUT_OF_WORKSPACE, FAILURE_PLAN_STALE, FAILURE_UNREACHABLE,
    FAILURE_SKILL_UNAVAILABLE, FAILURE_TIMEOUT, REQUIRE_SHARE, SHARE_FRAMES,
    SkillError, SkillOutcome, SkillPlan,
)
from leapflow.hardware.skills.runtime_readiness import plan_refusal

if TYPE_CHECKING:
    from leapflow.hardware.skills.runtime import SkillRuntime
    from leapflow.hardware.skills.context import RunContext


_T = TypeVar("_T")


async def run_owned(runtime: SkillRuntime, operation: Awaitable[_T]) -> _T:
    """Track preparation as well as worker time so disposal drains both."""
    task = asyncio.create_task(operation)
    runtime._planning_tasks.add(task)
    try:
        return await task
    finally:
        runtime._planning_tasks.discard(task)


@dataclass(frozen=True)
class BatchPlanningResult:
    selected: SkillPlan | SkillOutcome
    report: Mapping[str, Any]


async def evaluate(runtime: SkillRuntime, skill: PreparedPlanningSkill, ctx: RunContext,
                   args: Sequence[Mapping[str, Any]], *, deadline: float) -> tuple[tuple[Evaluation, ...], PreparedPlanning]:
    """Single and batch requests use the same preparation and pure evaluator."""
    prepared = await skill.prepare(ctx, args)
    if not isinstance(prepared, PreparedPlanning) or len(prepared.jobs) != len(args):
        raise SkillError(FAILURE_INTERNAL_DEFECT, "invalid prepared planning response")
    Deadline(deadline).check()
    positions = [i for i, job in enumerate(prepared.jobs) if not isinstance(job, SkillOutcome)]
    computed = await runtime._calculations.evaluate(
        [prepared.jobs[i] for i in positions], deadline=deadline)
    if len(computed) != len(positions):
        raise SkillError(FAILURE_INTERNAL_DEFECT, "planning worker response count differs")
    results = [Evaluation(job) if isinstance(job, SkillOutcome) else None for job in prepared.jobs]
    for index, result in zip(positions, computed):
        results[index] = result
    prepared.check_scene(ctx)
    Deadline(deadline).check()
    for result in results:
        if not isinstance(result, Evaluation):
            raise SkillError(FAILURE_INTERNAL_DEFECT, "invalid calculation result")
        if isinstance(result.value, SkillOutcome) and result.value.failure_code in {
                FAILURE_INTERNAL_DEFECT, FAILURE_TIMEOUT, "stopped"}:
            raise SkillError(result.value.failure_code, result.value.message)
    ctx.note("planning_snapshot", {"state_reads": prepared.state_reads})
    return tuple(results), prepared


async def plan_single(runtime: SkillRuntime, skill: PreparedPlanningSkill, ctx: RunContext,
                      arguments: Mapping[str, Any], *, deadline: float) -> tuple[Any, PreparedPlanning]:
    """Try the request first; bounded plugin variants require no additional LLM call."""
    evaluated, prepared = await evaluate(runtime, skill, ctx, [arguments], deadline=deadline)
    original = evaluated[0].value
    settings = runtime._settings.planning
    if (not settings.auto_variants or not isinstance(skill, PlanningAlternativesSkill)
            or not isinstance(original, SkillOutcome)
            or original.failure_code not in {FAILURE_UNREACHABLE, FAILURE_OUT_OF_WORKSPACE}):
        return original, prepared
    offered = skill.alternatives(ctx, arguments, limit=settings.max_candidates - 1)
    if not offered:
        return original, prepared
    raw = [{'id': 'requested', 'args': arguments}, *offered]
    settings.check_size(raw)
    entries = candidates(raw, settings)
    values = [validate_args(skill.spec.args_schema, a) for _, a in entries]
    started = time.monotonic()
    evaluated, prepared = await evaluate(runtime, skill, ctx, values, deadline=deadline)
    selected, report = _report(entries, evaluated, range(len(entries)), Selection(), started, prepared.state_reads)
    report['automatic'] = True
    report['trigger'] = original.failure_code
    if selected is None:
        return replace(original, result={**original.result, 'batch': report}), prepared
    plan = evaluated[selected].value
    return replace(plan, summary=f'{plan.summary} Automatic alternative: {entries[selected][0]} (requested path infeasible).',
                   basis={**plan.basis, 'batch': report,
                          'substituted': {'selected_id': entries[selected][0], 'reason': original.failure_code}}), prepared


async def confirm(skill: Any, ctx: RunContext, plan: SkillPlan, *, deadline: float) -> SkillPlan:
    if isinstance(skill, PlanConfirmationSkill):
        plan = await skill.confirm(ctx, plan)
    Deadline(deadline).check()
    return plan


async def check_basis(runtime: SkillRuntime, spec: Any, frames: Any,
                      model_basis: Mapping[str, Any]) -> None:
    if model_basis != await runtime._generation_basis(spec):
        raise SkillError(FAILURE_PLAN_STALE, "physical model generation changed during planning")
    check_owner_basis(runtime, spec, frames)


def check_owner_basis(runtime: SkillRuntime, spec: Any, frames: Any) -> None:
    """Final synchronous owner checks, with no scheduling gap before publication."""
    if runtime._closing or runtime._workcells.get(spec.workcell_id) is not spec:
        raise SkillError(FAILURE_PLAN_STALE, "workcell changed or node shut down during planning")
    if dict(runtime._frames(spec).versions) != dict(frames.versions):
        raise SkillError(FAILURE_PLAN_STALE, "calibration changed during planning")


def _arguments(runtime: SkillRuntime, spec: Any, skill: Any, frames: Any,
               values: Sequence[Mapping[str, Any]], share_frames: bool) -> tuple[list[Any], list[int]]:
    """Candidate-local refusals do not conceal the other alternatives' diagnostics."""
    unique: list[Any] = []
    indices: list[int] = []
    keys: dict[str, int] = {}
    offered = [runtime._skills[sid] for sid in runtime._offered(spec)]
    for raw in values:
        try:
            args = validate_args(skill.spec.args_schema, raw)
            reasons, readiness = plan_refusal(offered, spec, frames, skill.spec, args)
            if f"{REQUIRE_SHARE}{SHARE_FRAMES}" in skill.spec.requires and not share_frames:
                reasons.append("the host has not consented to receive camera frames")
            if reasons:
                raise SkillError(FAILURE_SKILL_UNAVAILABLE, "; ".join(reasons), result=readiness)
            key = encoded(args)
            if key in keys:
                indices.append(keys[key])
                continue
            keys[key] = len(unique)
            indices.append(len(unique))
            unique.append(args)
        except SkillError as error:
            indices.append(len(unique))
            unique.append(SkillOutcome.from_error(error))
    return unique, indices


def _report(entries: Sequence[Any], results: Sequence[Evaluation], indices: Sequence[int],
            selection: Selection, started: float, state_reads: int) -> tuple[int | None, dict[str, Any]]:
    mapped = [results[i] for i in indices]
    selected, rationale = selection.choose(mapped)
    rows = []
    for (name, _), index in zip(entries, indices):
        result = results[index]
        problem = result.value if isinstance(result.value, SkillOutcome) else None
        rows.append({"id": name,
                     "status": "refused" if problem else "feasible" if result.verified else "unverified",
                     "verified": result.verified, "metrics": dict(result.metrics),
                     "problems": [] if problem is None else [problem.to_dict()],
                     "evidence": dict(result.evidence)})
    return selected, {
        "candidates": rows, "selected_id": None if selected is None else entries[selected][0],
        "selection": rationale,
        "evidence": {"planning_s": time.monotonic() - started, "candidate_count": len(entries),
                     "unique_programs": len(results), "duplicate_reuse": len(entries) - len(results),
                     "ik_calls": sum(int(r.evidence.get("ik_calls", 0)) for r in results),
                     "state_reads": state_reads,
                     "llm_calls": 0},
    }


async def plan_batch(runtime: SkillRuntime, workcell: str, skill_id: str, raw: Any,
                     selection: Any = None, *, share_frames: bool = False) -> BatchPlanningResult:
    settings = runtime._settings.planning
    started = time.monotonic()
    deadline = started + settings.timeout_s
    report: dict[str, Any] = {"selected_id": None, "candidates": []}
    entries: Sequence[Any] = ()
    try:
        settings.check_size({"candidates": raw, "selection": selection})
        entries = candidates(raw, settings)
        policy = Selection.from_mapping(selection)
        spec, skill = runtime._resolve(workcell, skill_id)
        if not isinstance(skill, PreparedPlanningSkill) or not skill.spec.runs_leased:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, "this skill does not support prepared batch planning")
        async def compute() -> tuple[Any, ...]:
            frames = runtime._frames(spec)
            unique, indices = _arguments(runtime, spec, skill, frames, [a for _, a in entries], share_frames)
            model_basis = await runtime._generation_basis(spec)
            if skill.spec.uses_motion:
                ensure_model_use(runtime._registry, spec, model_basis)
            ctx = runtime._context(spec, frames, runtime._effective_limits(spec, {}), asyncio.Event(),
                                   skill.spec.region, writes=False)
            positions = [i for i, value in enumerate(unique) if not isinstance(value, SkillOutcome)]
            results = [Evaluation(v) if isinstance(v, SkillOutcome) else None for v in unique]
            prepared = None
            if positions:
                computed, prepared = await evaluate(runtime, skill, ctx, [unique[i] for i in positions], deadline=deadline)
                for index, result in zip(positions, computed):
                    results[index] = result
            state_reads = int(ctx.evidence.get("planning_snapshot", {}).get("state_reads", 0))
            return frames, results, indices, model_basis, state_reads, ctx, prepared

        frames, results, indices, model_basis, state_reads, ctx, prepared = await asyncio.wait_for(
            compute(), timeout=max(0, deadline - time.monotonic()))
        await asyncio.wait_for(check_basis(runtime, spec, frames, model_basis),
                               timeout=max(0, deadline - time.monotonic()))
        check_owner_basis(runtime, spec, frames)
        if prepared is not None:
            prepared.check_scene(ctx)
        Deadline(deadline).check()
        selected, report = _report(entries, results, indices, policy, started, state_reads)
        if selected is None:
            settings.check_size({"outcome": no_selection().to_dict(), "batch": report})
            return BatchPlanningResult(no_selection(), report)
        value = results[indices[selected]].value
        planned = replace(value, basis={**value.basis, "batch": report})
        planned = await asyncio.wait_for(confirm(skill, ctx, planned, deadline=deadline),
                                         timeout=max(0, deadline - time.monotonic()))
        await asyncio.wait_for(check_basis(runtime, spec, frames, model_basis),
                               timeout=max(0, deadline - time.monotonic()))
        if prepared is not None:
            prepared.check_scene(ctx)
        sealed = runtime._seal(spec, skill, planned, frames, model_basis=model_basis,
                                       result_budget=settings, deadline=deadline)
        return BatchPlanningResult(sealed, report)
    except asyncio.TimeoutError:
        error = SkillError(FAILURE_TIMEOUT, "planning deadline exceeded; nothing executed")
    except SkillError as caught:
        error = caught
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("batch planning raised", exc_info=True)
        error = SkillError(FAILURE_INTERNAL_DEFECT, f"{type(exc).__name__}: {exc}")
    report["selected_id"] = None
    report["candidates"] = [{"id": name, "status": "timeout" if error.failure_code == FAILURE_TIMEOUT else "refused",
                             "verified": False, "metrics": {},
                             "problems": [{"failure_code": error.failure_code, "error": error.message}]}
                            for name, _ in entries]
    report["evidence"] = {"planning_s": time.monotonic() - started, "candidate_count": len(entries), "llm_calls": 0}
    report["failure_code"] = error.failure_code
    report["error"] = error.message
    outcome = SkillOutcome.from_error(error)
    try:
        settings.check_size({"outcome": outcome.to_dict(), "batch": report})
    except SkillError:
        outcome = SkillOutcome.failed("invalid_args", "planning diagnostics exceed result budget; nothing executed")
        report = {"selected_id": None,
                  "candidates": [{"id": name, "status": "refused"} for name, _ in entries],
                  "failure_code": "invalid_args", "diagnostics_omitted": "result budget exceeded",
                  "evidence": {"candidate_count": len(entries), "llm_calls": 0}}
        # IDs can themselves exhaust a tiny budget. Refuse admission explicitly,
        # never truncate identity diagnostics or cache an executable selection.
        try:
            settings.check_size({"outcome": outcome.to_dict(), "batch": report})
        except SkillError:
            report = {"selected_id": None, "candidates": [], "failure_code": "invalid_args",
                      "diagnostics_omitted": "candidate identities exceed result budget",
                      "evidence": {"candidate_count": len(entries), "llm_calls": 0}}
    return BatchPlanningResult(outcome, report)

