# Copyright (c) Alibaba, Inc. and its affiliates.
"""The node skill runtime: discover, plan, approve-by-digest, execute, stop.

One ``SkillRuntime`` per hardware node. It owns the skill catalog, the admitted
workcells, a short-lived plan cache, and the active runs, and it is the only
component that builds a ``SkillContext``. The LHP server routes ``skill.*``
methods here; the host never addresses a device directly while a skill holds it.

Lifecycle of one actuating run:

1. ``plan(workcell, skill, args)`` validates arguments, checks requirements
   against the live calibration snapshot, and asks the skill for a plan. The
   runtime fills identity, the effective limits, and the calibration versions
   the plan was computed against, then digests and caches it.
2. The host shows the plan summary to a human and mints a permit bound to
   ``permit_digest(plan)``.
3. ``execute(plan_digest)`` refuses a plan it no longer holds, a device another
   run holds, or a calibration that moved since planning (``plan_stale``) --
   all with side effect NONE -- and otherwise starts the run as a task.
4. ``stop(run_id)`` sets the run's cancel event and halts its devices at once
   whenever a dispatch has begun (even one still inside the transport); the
   skill unwinds at its next motion checkpoint.
5. A run's verdict merges the skill's own word with what physically happened
   and never downgrades it: a landed batch or a persisted calibration is at
   least COMMITTED (PARTIAL on failure), an unresolved dispatch is UNKNOWN. The
   devices stay held until every in-flight dispatch has been reconciled; one
   that lands after the halt is halted again.
6. Every executed run is recorded as an episode (``episodes.py``) when the node
   has an episode directory; the outcome's evidence names it, and its manifest
   carries the host's ``intent`` (S2's sub-goal). ``label(run_id, ...)`` later
   records whether S2 verified that goal.

A READ-effect skill folds both phases into ``plan`` and returns its outcome,
unless it is ``leased`` (a recording): then it runs like an actuating skill,
holding the workcell's devices under the host's lease, but cannot command.

Per workcell the runtime also keeps what outlives one run: the scene memory
(what perception last saw, the grasps proposed, and what the gripper holds).
The declared grasp planner is validated at admission and built per context,
bound to that context's sensing.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import time
import uuid
from dataclasses import replace
from typing import Any, Callable, Iterable, Mapping

from leapflow.hardware.node_gate import NodeWriteGate
from leapflow.hardware.skills.args import validate_args
from leapflow.hardware.skills.context import RunContext
from leapflow.hardware.skills.digest import plan_digest, run_permit_digest
from leapflow.hardware.skills.episodes import (
    NULL_TAP,
    EpisodeRecorder,
    EpisodeTap,
    label_episode,
)
from leapflow.hardware.skills.frames import FrameGraph, unmet_requirements
from leapflow.hardware.skills.grasp import (
    AdmissionSensing,
    GraspSensing,
    PlannerFactory,
    discover_grasp_planners,
)
from leapflow.hardware.skills.grasp_model import DEFAULT_GRASP_MODEL, MODEL_PLANNER
from leapflow.hardware.skills.motion import POLL_S, GovernedMotion, Sleep
from leapflow.hardware.skills.perception import PerceptionSettings
from leapflow.hardware.skills.scene_memory import SceneMemory
from leapflow.hardware.skills.ports import GraspPlanner, NodeSkill
from leapflow.hardware.skills.protocol import (
    CONSENT_EACH_RUN,
    FAILURE_DEVICE_BUSY,
    FAILURE_INTERNAL_DEFECT,
    FAILURE_INVALID_ARGS,
    FAILURE_PLAN_STALE,
    FAILURE_PLAN_UNKNOWN,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_STOPPED,
    FAILURE_TIMEOUT,
    FAILURE_UNKNOWN_SKILL,
    FAILURE_UNKNOWN_WORKCELL,
    REQUIRE_SHARE,
    SHARE_FRAMES,
    STATUS_HALTED,
    SkillError,
    SkillOutcome,
    SkillPlan,
)
from leapflow.hardware.skills.runs import SkillRun, merge_outcome, needs_halt
from leapflow.hardware.skills.runtime_settings import DEFAULT_PLAN_TTL_S, SkillRuntimeSettings
from leapflow.hardware.skills.policy_binding import policy_problems
from leapflow.hardware.skills.vision import CharucoTarget
from leapflow.hardware.skills.workcell import (
    ROLE_GRIPPER,
    WorkcellLoadReport,
    WorkcellSpec,
    load_workcells,
)

logger = logging.getLogger(__name__)

SKILL_ENTRY_POINT_GROUP = "leapflow.hardware.skills"
FINISHED_RUNS_KEPT = 32


def discover_skills() -> tuple[NodeSkill, ...]:
    """Skills advertised under the ``leapflow.hardware.skills`` entry-point group.

    Each entry point is a zero-argument callable returning an iterable of skills.
    A package that fails to load is skipped with a warning, never fatal to the node.
    """
    from importlib.metadata import entry_points

    found: list[NodeSkill] = []
    for entry in entry_points(group=SKILL_ENTRY_POINT_GROUP):
        try:
            found.extend(entry.load()())
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not stop the node
            logger.warning("skill entry point %r failed to load: %s", entry.name, exc)
    return tuple(found)


class SkillRuntime:
    """Skill catalog, plan cache, and run registry for one node."""

    def __init__(
        self,
        registry: Any,
        *,
        settings: SkillRuntimeSettings | None = None,
        skills: Iterable[NodeSkill] | None = None,
        workcells: Mapping[str, WorkcellSpec] | None = None,
        gate: NodeWriteGate | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
        poll_s: float = POLL_S,
        planners: Mapping[str, PlannerFactory] | None = None,
    ) -> None:
        self._registry = registry
        self._settings = settings or SkillRuntimeSettings()
        self._gate = gate or NodeWriteGate(registry)
        self._clock = clock
        self._sleep = sleep
        self._poll_s = poll_s
        self._skills: dict[str, NodeSkill] = {}
        for skill in discover_skills() if skills is None else skills:
            self._register(skill)
        self._planners = discover_grasp_planners() if planners is None else dict(planners)
        self._scenes: dict[str, SceneMemory] = {}  # workcell -> memory across runs
        self._report = WorkcellLoadReport()
        if workcells is None:
            self.reload_workcells()
        else:
            self._workcells = dict(workcells)
        self._plans: dict[str, tuple[SkillPlan, float]] = {}
        self._runs: dict[str, SkillRun] = {}
        self._held: dict[str, str] = {}  # device_id -> run_id

    # ── Catalog ──

    def _register(self, skill: NodeSkill) -> None:
        if not isinstance(skill, NodeSkill):
            logger.warning("ignoring %r: it does not implement NodeSkill", skill)
            return
        skill_id = skill.spec.skill_id
        if not self._settings.admits(skill_id):
            return
        if skill_id in self._skills:
            logger.warning("duplicate skill %r ignored", skill_id)
            return
        self._skills[skill_id] = skill

    def reload_workcells(self) -> WorkcellLoadReport:
        directory = self._settings.workcells_dir
        if directory is None:
            self._workcells, self._report = {}, WorkcellLoadReport()
        else:
            self._workcells, self._report = load_workcells(
                directory,
                self._registry,
                checks=(
                    self._grasp_problems,
                    self._perception_problems,
                    self._vision_problems,
                    self._policy_problems,
                ),
            )
        return self._report

    def _grasp_problems(self, spec: WorkcellSpec) -> list[str]:
        if not spec.role(ROLE_GRIPPER):
            return []  # nothing to grasp with; grasp.propose reports itself unavailable
        try:
            self._build_planner(spec, AdmissionSensing())
        except ValueError as exc:
            return [f"grasp: {exc}"]
        except Exception as exc:  # noqa: BLE001 - a broken planner plugin refuses one workcell
            return [f"grasp: planner {spec.grasp_planner!r} failed: {type(exc).__name__}: {exc}"]
        if spec.grasp_planner == MODEL_PLANNER:
            binding = str(spec.grasp.get("model") or DEFAULT_GRASP_MODEL)
            if not spec.model(binding):
                return [f"grasp: planner 'model' needs a models.{binding} binding"]
        return []

    @staticmethod
    def _perception_problems(spec: WorkcellSpec) -> list[str]:
        try:
            PerceptionSettings.from_mapping(spec.perception)
        except ValueError as exc:
            return [f"perception: {exc}"]
        return []

    def _policy_problems(self, spec: WorkcellSpec) -> list[str]:
        return policy_problems(spec, self._registry)

    @staticmethod
    def _vision_problems(spec: WorkcellSpec) -> list[str]:
        if not spec.calibration_target:
            return []
        try:
            CharucoTarget.from_mapping(spec.calibration_target)
        except ValueError as exc:
            return [f"calibration_target: {exc}"]
        return []

    @property
    def load_report(self) -> WorkcellLoadReport:
        return self._report

    @property
    def skill_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._skills))

    def list(self, workcell: str = "") -> dict[str, Any]:
        """Every workcell with its skills, each marked available or why not."""
        cells = []
        for spec in self._workcells.values():
            if workcell and spec.workcell_id != workcell:
                continue
            frames = self._frames(spec)
            skills = []
            for skill_id in self._offered(spec):
                skill_spec = self._skills[skill_id].spec
                reasons = unmet_requirements(skill_spec.requires, spec, frames)
                skills.append(
                    {**skill_spec.to_dict(), "available": not reasons, "reason": "; ".join(reasons)}
                )
            cells.append(
                {
                    **spec.to_dict(),
                    "calibration": dict(frames.versions),
                    "calibration_stale": dict(frames.stale),
                    "skills": skills,
                }
            )
        return {"workcells": cells, "rejected": self._report.to_dict()["rejected"]}

    def _offered(self, spec: WorkcellSpec) -> tuple[str, ...]:
        if not spec.skills:
            return self.skill_ids
        return tuple(
            sid for sid in self.skill_ids if any(fnmatch.fnmatchcase(sid, p) for p in spec.skills)
        )

    # ── Plan ──

    async def plan(
        self,
        workcell: str,
        skill_id: str,
        args: Mapping[str, Any] | None = None,
        *,
        share_frames: bool = False,
    ) -> SkillPlan | SkillOutcome:
        """Plan an actuating skill (cached by digest), or run a READ skill outright.

        *share_frames* is the requesting host's consent to receive camera frames;
        a skill requiring ``share:frames`` is refused without it, whatever the
        workcell declares.
        """
        try:
            spec, skill = self._resolve(workcell, skill_id)
            arguments = validate_args(skill.spec.args_schema, args)
            frames = self._frames(spec)
            reasons = unmet_requirements(skill.spec.requires, spec, frames)
            if f"{REQUIRE_SHARE}{SHARE_FRAMES}" in skill.spec.requires and not share_frames:
                reasons.append(
                    "the host has not consented to receive camera frames (its operator sets "
                    "hardware.share_frames)"
                )
            if reasons:
                raise SkillError(FAILURE_SKILL_UNAVAILABLE, "; ".join(reasons))
            limits = self._effective_limits(spec, {})
            if not skill.spec.runs_leased:
                return await self._run_read(spec, skill, frames, limits, arguments)
            ctx = self._context(
                spec,
                frames,
                limits,
                asyncio.Event(),
                skill_id,
                writes=False,
                recording=lambda: self._settings.episodes.enabled,
            )
            planned = await skill.plan(ctx, arguments)
        except SkillError as error:
            return SkillOutcome.from_error(error)
        except Exception as exc:  # noqa: BLE001 - a skill defect must still answer typed
            logger.error("planning %s raised", skill_id, exc_info=True)
            return SkillOutcome.failed(FAILURE_INTERNAL_DEFECT, f"{type(exc).__name__}: {exc}")
        if isinstance(planned, SkillOutcome):
            return planned
        try:
            return self._seal(spec, skill, planned, frames)
        except SkillError as error:
            return SkillOutcome.from_error(error)

    def _seal(
        self, spec: WorkcellSpec, skill: NodeSkill, planned: SkillPlan, frames: FrameGraph
    ) -> SkillPlan:
        sealed = replace(
            planned,
            skill_id=skill.spec.skill_id,
            workcell=spec.workcell_id,
            version=skill.spec.version,
            limits=self._effective_limits(spec, planned.limits),
            basis={**dict(planned.basis), "calibration": dict(frames.versions)},
            digest="",
        )
        sealed = replace(sealed, digest=plan_digest(sealed))
        self._evict_expired()
        self._plans[sealed.digest] = (sealed, self._clock())
        return sealed

    def permit_digest(self, plan: SkillPlan) -> str:
        """The action digest a run-scoped permit for *plan* must bind."""
        return run_permit_digest(
            workcell=plan.workcell,
            skill_id=plan.skill_id,
            version=plan.version,
            plan_digest=plan.digest,
            limits=plan.limits,
        )

    def cached_plan(self, digest: str) -> SkillPlan | None:
        self._evict_expired()
        entry = self._plans.get(digest)
        return entry[0] if entry is not None else None

    def _effective_limits(
        self, spec: WorkcellSpec, requested: Mapping[str, float]
    ) -> dict[str, float]:
        """Requested limits bounded by the workcell and node caps -- never raised by a floor."""
        cap = min(float(spec.limits["speed_pct_max"]), self._settings.speed_pct_max)
        speed = min(cap, float(requested.get("speed_pct", spec.limits["default_speed_pct"])))
        duration = min(
            self._settings.max_duration_s,
            float(requested.get("max_duration_s", self._settings.max_duration_s)),
        )
        if not speed > 0.0 or not duration > 0.0:
            raise SkillError(
                FAILURE_INVALID_ARGS,
                f"speed {speed:g} % and duration {duration:g} s must both be positive",
            )
        return {
            "speed_pct": speed,
            "speed_pct_max": cap,
            "move_timeout_s": float(spec.limits["move_timeout_s"]),
            "fk_tolerance_mm": float(spec.limits["fk_tolerance_mm"]),
            "fk_tolerance_deg": float(spec.limits["fk_tolerance_deg"]),
            "joint_path_step_deg": float(spec.limits["joint_path_step_deg"]),
            "max_duration_s": duration,
            "max_steps": float(spec.limits["max_steps"]),
            "max_wait_s": float(spec.limits["max_wait_s"]),
        }

    async def _run_read(
        self,
        spec: WorkcellSpec,
        skill: NodeSkill,
        frames: FrameGraph,
        limits: Mapping[str, float],
        arguments: Mapping[str, Any],
    ) -> SkillOutcome:
        ctx = self._context(
            spec, frames, limits, asyncio.Event(), skill.spec.skill_id, writes=False
        )
        planned = await skill.plan(ctx, arguments)
        if isinstance(planned, SkillOutcome):
            return planned
        outcome = await skill.execute(ctx, planned)
        return replace(outcome, evidence={**dict(outcome.evidence), **ctx.evidence})

    # ── Execute ──

    async def execute(self, digest: str, *, intent: str = "") -> SkillRun | SkillOutcome:
        """Start the cached plan *digest*; a refusal is a NONE outcome, not a run.

        *intent* is the host's sub-goal for the run (its episode's language
        annotation); it is not part of the approved plan and changes nothing it does.
        """
        plan = self.cached_plan(digest)
        if plan is None:
            return SkillOutcome.failed(
                FAILURE_PLAN_UNKNOWN, "no such plan on this node (expired or already executed)"
            )
        spec = self._workcells.get(plan.workcell)
        skill = self._skills.get(plan.skill_id)
        if spec is None or skill is None or skill.spec.version != plan.version:
            return SkillOutcome.failed(FAILURE_PLAN_STALE, "the skill or workcell changed")
        busy = [device for device in spec.device_ids if device in self._held]
        if busy:
            return SkillOutcome.failed(
                FAILURE_DEVICE_BUSY, f"{', '.join(busy)} held by run {self._held[busy[0]]}"
            )
        frames = self._frames(spec)
        if dict(frames.versions) != dict(plan.basis.get("calibration") or {}):
            return SkillOutcome.failed(
                FAILURE_PLAN_STALE, "calibration changed since planning; plan again"
            )
        del self._plans[digest]
        run = SkillRun(
            run_id=uuid.uuid4().hex,
            plan=plan,
            devices=spec.device_ids,
            actuators=spec.actuator_ids,
            intent=intent,
        )
        recorder = self._recorder(spec, run)
        tap: EpisodeTap = recorder or NULL_TAP
        ctx = self._context(
            spec,
            frames,
            plan.limits,
            run.cancel,
            plan.skill_id,
            writes=skill.spec.is_actuating,
            tap=tap,
            recording=lambda: tap.recording,
        )
        run.motion = ctx.motion
        for device in run.devices:
            self._held[device] = run.run_id
        self._runs[run.run_id] = run
        run.task = asyncio.get_running_loop().create_task(
            self._drive(run, skill, ctx, recorder),
            name=f"skill-{plan.skill_id}-{run.run_id[:8]}",
        )
        logger.info("skill run %s started: %s on %s", run.run_id[:8], plan.skill_id, plan.workcell)
        return run

    def _recorder(self, spec: WorkcellSpec, run: SkillRun) -> EpisodeRecorder | None:
        settings = self._settings.episodes
        if not settings.enabled:
            return None
        return EpisodeRecorder(
            settings=settings,
            registry=self._registry,
            spec=spec,
            plan=run.plan,
            run_id=run.run_id,
            intent=run.intent,
            clock=self._clock,
            sleep=self._sleep,
        )

    def label(self, run_id: str, *, success: bool, note: str = "") -> dict[str, Any]:
        """Record whether *run_id*'s goal was verified reached; ``SkillError`` when it
        cannot be (recording off, a run still going, or no such episode)."""
        run = self._runs.get(run_id)
        if run is not None and run.running:
            raise SkillError(FAILURE_INVALID_ARGS, f"run {run_id!r} is still running")
        try:
            return label_episode(
                self._settings.episodes.directory if self._settings.episodes.enabled else None,
                run_id,
                success=success,
                note=note,
                source="host",
            )
        except ValueError as exc:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, str(exc)) from exc

    async def _drive(
        self, run: SkillRun, skill: NodeSkill, ctx: RunContext, recorder: EpisodeRecorder | None
    ) -> None:
        motion = ctx.motion
        try:
            if recorder is not None:
                await recorder.start()
            outcome = await asyncio.wait_for(
                skill.execute(ctx, run.plan), timeout=run.plan.limits["max_duration_s"]
            )
        except SkillError as error:
            outcome = SkillOutcome.from_error(error)
        except asyncio.TimeoutError:
            outcome = SkillOutcome.failed(
                FAILURE_TIMEOUT, "the run exceeded its maximum duration", status=STATUS_HALTED
            )
        except asyncio.CancelledError:
            outcome = self._stopped(run)
        except Exception as exc:  # noqa: BLE001 - a skill defect must still end with a verdict
            logger.error("skill %s raised", run.plan.skill_id, exc_info=True)
            outcome = SkillOutcome.failed(
                FAILURE_INTERNAL_DEFECT, f"{type(exc).__name__}: {exc}", status=STATUS_HALTED
            )
        if run.cancel.is_set():
            if outcome.ok:
                outcome = self._stopped(run)
            elif outcome.failure_code == FAILURE_STOPPED:
                outcome = replace(
                    outcome,
                    failure_code=run.stop_code,
                    message=run.stop_reason or outcome.message,
                )
        await motion.settle(self._settings.dispatch_settle_s)
        await self._halt_late_landing(run)
        outcome = merge_outcome(
            outcome, motion=motion, committed=ctx.committed, evidence=ctx.evidence
        )
        if needs_halt(outcome, motion.sent_any):
            await self._halt_run(run)
        if recorder is not None:
            episode = await recorder.finish(outcome)
            outcome = replace(outcome, evidence={**dict(outcome.evidence), "episode": episode})
        self._finish(run, outcome)

    def _stopped(self, run: SkillRun) -> SkillOutcome:
        return SkillOutcome.failed(
            run.stop_code, run.stop_reason or "the run was stopped", status=STATUS_HALTED
        )

    def _finish(self, run: SkillRun, outcome: SkillOutcome) -> None:
        run.outcome = outcome
        run.finished_at = time.time()
        if run.motion is None:
            self._release(run)
        else:
            # A dispatch still inside a transport keeps the devices held: the next
            # run must not race a write whose verdict nobody has seen yet.
            run.motion.when_settled(lambda: self._settled(run))
        logger.info(
            "skill run %s ended: %s %s (%s)",
            run.run_id[:8],
            outcome.status,
            outcome.failure_code or "",
            outcome.side_effect_state,
        )
        finished = [r for r in self._runs.values() if not r.running]
        for stale in finished[:-FINISHED_RUNS_KEPT]:
            self._runs.pop(stale.run_id, None)

    def _release(self, run: SkillRun) -> None:
        for device in run.devices:
            if self._held.get(device) == run.run_id:
                del self._held[device]

    def _settled(self, run: SkillRun) -> None:
        """The run's last dispatch resolved: halt a late landing, then release."""
        if not run.landed_after_halt:
            self._release(run)
            return

        async def _halt_then_release() -> None:
            try:
                await self._halt_late_landing(run)
            finally:
                self._release(run)

        asyncio.get_running_loop().create_task(
            _halt_then_release(), name=f"skill-late-halt-{run.run_id[:8]}"
        )

    async def _halt_late_landing(self, run: SkillRun) -> None:
        """A write that landed after the run's halt is halted again."""
        if run.landed_after_halt:
            logger.warning("skill run %s: a write landed after its halt", run.run_id[:8])
            run.halted = False
            await self._halt_run(run)

    # ── Control ──

    def status(self, run_id: str = "") -> tuple[SkillRun, ...]:
        if run_id:
            run = self._runs.get(run_id)
            return () if run is None else (run,)
        return tuple(self._runs.values())

    async def wait(self, run_id: str, timeout_s: float) -> SkillRun | None:
        """Wait up to *timeout_s* for a run to end; returns it either way."""
        run = self._runs.get(run_id)
        if run is not None and run.task is not None and run.running:
            await asyncio.wait({run.task}, timeout=max(0.0, timeout_s))
        return run

    async def stop(
        self,
        run_id: str,
        reason: str = "stopped by request",
        *,
        failure_code: str = FAILURE_STOPPED,
    ) -> SkillRun | None:
        """Halt a run's devices now, then let the skill unwind (cancelled after a grace).

        *failure_code* names why, e.g. ``lease_lost`` when the host went silent.
        """
        run = self._runs.get(run_id)
        if run is None or not run.running:
            return run
        run.stop_reason = reason
        run.stop_code = failure_code
        run.cancel.set()
        # Halt at once when anything may have moved -- including a batch still
        # inside the transport; a run that never dispatched stays energized (a
        # halt latches the arm's emergency stop).
        if run.motion is None or run.motion.sent_any:
            await self._halt_run(run)
        grace = self._settings.stop_grace_s
        if run.task is not None:
            _, pending = await asyncio.wait({run.task}, timeout=grace)
            if pending:
                run.task.cancel()
                await asyncio.wait({run.task}, timeout=grace)
        return run

    async def stop_all(self, reason: str) -> None:
        for run in [r for r in self._runs.values() if r.running]:
            await self.stop(run.run_id, reason)

    def holds_device(self, device_id: str) -> bool:
        return device_id in self._held

    def plan_devices(self, plan: SkillPlan) -> tuple[str, ...]:
        """Devices a run of *plan* would hold (empty when its workcell is gone)."""
        spec = self._workcells.get(plan.workcell)
        return spec.device_ids if spec is not None else ()

    def consent(self, plan: SkillPlan) -> str:
        """How far one approval of *plan*'s skill reaches (``SkillSpec.consent``)."""
        skill = self._skills.get(plan.skill_id)
        return skill.spec.consent if skill is not None else CONSENT_EACH_RUN

    async def close(self) -> None:
        await self.stop_all("node shutting down")

    # ── Internals ──

    def _resolve(self, workcell: str, skill_id: str) -> tuple[WorkcellSpec, NodeSkill]:
        spec = self._workcells.get(workcell)
        if spec is None:
            known = ", ".join(sorted(self._workcells)) or "none"
            raise SkillError(
                FAILURE_UNKNOWN_WORKCELL, f"unknown workcell {workcell!r} (known: {known})"
            )
        skill = self._skills.get(skill_id)
        if skill is None or skill_id not in self._offered(spec):
            raise SkillError(
                FAILURE_UNKNOWN_SKILL, f"workcell {workcell!r} offers no skill {skill_id!r}"
            )
        return spec, skill

    def _frames(self, spec: WorkcellSpec) -> FrameGraph:
        return FrameGraph.snapshot(spec, getattr(self._registry, "calibration_store", None))

    def _motion(
        self,
        spec: WorkcellSpec,
        frames: FrameGraph,
        limits: Mapping[str, float],
        cancel: asyncio.Event,
        skill_id: str,
        *,
        writes: bool,
        tap: EpisodeTap = NULL_TAP,
    ) -> GovernedMotion:
        return GovernedMotion(
            registry=self._registry,
            gate=self._gate,
            spec=spec,
            frames=frames,
            limits=limits,
            cancel=cancel,
            deadline=self._clock() + float(limits["max_duration_s"]),
            allow_bootstrap=skill_id.startswith("calib."),
            allow_writes=writes,
            clock=self._clock,
            sleep=self._sleep,
            poll_s=self._poll_s,
            tap=tap,
        )

    def _context(
        self,
        spec: WorkcellSpec,
        frames: FrameGraph,
        limits: Mapping[str, float],
        cancel: asyncio.Event,
        skill_id: str,
        *,
        writes: bool,
        tap: EpisodeTap = NULL_TAP,
        recording: Callable[[], bool] = lambda: False,
    ) -> RunContext:
        motion = self._motion(spec, frames, limits, cancel, skill_id, writes=writes, tap=tap)
        return RunContext(
            registry=self._registry,
            spec=spec,
            frames=frames,
            motion=motion,
            limits=limits,
            cancel=cancel,
            memory=self._scenes.setdefault(spec.workcell_id, SceneMemory(self._clock)),
            planner=lambda sensing: self._planner(spec, sensing),
            recording=recording,
        )

    def _planner(self, spec: WorkcellSpec, sensing: GraspSensing) -> GraspPlanner:
        try:
            return self._build_planner(spec, sensing)
        except ValueError as exc:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, f"grasp: {exc}") from exc

    def _build_planner(self, spec: WorkcellSpec, sensing: GraspSensing) -> GraspPlanner:
        """The workcell's planner; ``ValueError`` when it is unknown or misconfigured."""
        name = spec.grasp_planner
        factory = self._planners.get(name)
        if factory is None:
            known = ", ".join(sorted(self._planners)) or "none"
            raise ValueError(f"unknown planner {name!r} (known: {known})")
        planner = factory(spec.gripper_spec, spec.grasp, sensing)
        if not isinstance(planner, GraspPlanner):
            raise ValueError(f"planner {name!r} does not implement GraspPlanner")
        return planner

    def _evict_expired(self) -> None:
        horizon = self._clock() - self._settings.plan_ttl_s
        for digest in [d for d, (_, at) in self._plans.items() if at < horizon]:
            del self._plans[digest]

    async def _halt_run(self, run: SkillRun) -> None:
        """Halt a run's devices exactly once; preempts I/O (halt never takes the lock).

        Through the registry, not the transport: a stop that ends the link would
        otherwise leave a closed instance cached and the device would answer "is not
        open" until the node restarted.
        """
        if run.halted:
            return
        run.halted = True
        if run.motion is not None:
            run.halt_mark = run.motion.landed
        for device_id in run.actuators:
            try:
                await asyncio.wait_for(
                    self._registry.halt(device_id), timeout=self._settings.halt_timeout_s
                )
            except (asyncio.TimeoutError, OSError, RuntimeError) as exc:
                logger.error("halting %s failed: %s", device_id, exc, exc_info=True)


__all__ = [
    "DEFAULT_PLAN_TTL_S",
    "SKILL_ENTRY_POINT_GROUP",
    "SkillRun",
    "SkillRuntime",
    "SkillRuntimeSettings",
    "discover_skills",
]
