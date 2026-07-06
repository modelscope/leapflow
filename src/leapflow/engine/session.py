"""Session controller — orchestrates LEARN → DISTILL → EXECUTE lifecycle.

Manages the SessionMode state machine and coordinates between the imitation
learning pipeline, skill registry, and user-facing I/O.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from leapflow.engine.confirmation import ConfirmationHandler, ConfirmLevel, IOProvider
from leapflow.analysis.pipeline import ImitationPipeline
from leapflow.storage.session_store import LearningSessionStore
from leapflow.domain.trajectory import Episode, Trajectory
from leapflow.learning.distiller import DistillationCandidate
from leapflow.skills.evolution import SkillEvolutionPolicy
from leapflow.skills.registry import Skill, SkillRegistry, TriggerMatch

logger = logging.getLogger(__name__)

LearnCompleteCallback = Callable[["LearnResult"], None]
ProgressCallback = Optional[Callable[[str, int, int], None]]
StepProgressCallback = Optional[Callable[[int, int, str], None]]


class SessionMode(Enum):
    IDLE = "idle"
    LEARNING = "learning"
    EVOLVING = "evolving"
    EXECUTING = "executing"


@dataclass
class LearningSession:
    session_id: str
    goal: str
    trajectory_id: str
    start_time: float
    end_time: Optional[float] = None
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time


@dataclass
class LearnResult:
    trajectory_id: str
    step_count: int
    duration: float
    candidates: List[DistillationCandidate] = field(default_factory=list)
    new_skills: List[str] = field(default_factory=list)
    suggestions: int = 0
    storage_path: str = ""
    audit_log_path: str = ""
    event_stats: Dict[str, int] = field(default_factory=dict)
    activated_skill_names: Set[str] = field(default_factory=set)
    learnability_report: Optional[Any] = None  # LearnabilityReport


@dataclass
class ExecutionResult:
    ok: bool
    skill_name: str
    output: Any = None
    error: Optional[str] = None
    duration_s: float = 0.0
    steps_executed: int = 0
    steps_total: int = 0


class SessionError(Exception):
    pass


class SessionController:
    """Orchestrates the LEARN → DISTILL → EXECUTE lifecycle.

    Thread-safe via asyncio.Lock. Distillation runs in the background so
    ``exit_learning()`` returns immediately.
    """

    def __init__(
        self,
        pipeline: ImitationPipeline,
        registry: SkillRegistry,
        *,
        idle_timeout: float = 300.0,
        auto_learn: bool = True,
        confirmation: Optional[ConfirmationHandler] = None,
        audit: Optional[Any] = None,
        storage_path: str = "",
        audit_log_path: str = "",
        active_learning_observer: Optional[Any] = None,
        session_store: Optional[LearningSessionStore] = None,
        learnability_assessor: Optional[Any] = None,
        evolution_policy: Optional[SkillEvolutionPolicy] = None,
        skill_store: Optional[Any] = None,
    ) -> None:
        self._pipeline = pipeline
        self._registry = registry
        self._idle_timeout = idle_timeout
        self._auto_learn = auto_learn
        self._confirmation = confirmation or ConfirmationHandler()
        self._audit = audit
        self._storage_path = storage_path
        self._audit_log_path = audit_log_path
        self._observer = active_learning_observer
        self._session_store = session_store
        self._learnability_assessor = learnability_assessor
        self._evolution_policy = evolution_policy
        self._skill_store = skill_store

        self._mode = SessionMode.IDLE
        self._session: Optional[LearningSession] = None
        self._idle_task: Optional[asyncio.TimerHandle] = None
        self._lock = asyncio.Lock()
        self._on_learn_complete: Optional[LearnCompleteCallback] = None
        self._learn_task: Optional[asyncio.Task] = None
        self._last_learn_result: Optional[LearnResult] = None
        self._pending_learn: Optional[tuple[Trajectory, LearningSession]] = None
        self._pending_learn_deferred: Optional[tuple[Trajectory, LearningSession]] = None

    @property
    def idle_timeout(self) -> float:
        return self._idle_timeout

    @idle_timeout.setter
    def idle_timeout(self, value: float) -> None:
        self._idle_timeout = value

    @property
    def mode(self) -> SessionMode:
        return self._mode

    @property
    def current_session(self) -> Optional[LearningSession]:
        return self._session

    def set_on_learn_complete(self, callback: Optional[LearnCompleteCallback]) -> None:
        self._on_learn_complete = callback

    def set_on_learn_progress(self, callback: ProgressCallback) -> None:
        """Forward progress events from the imitation pipeline.

        Callback signature: ``callback(stage, current, total)``.
        """
        self._pipeline.progress_callback = callback

    def set_on_execute_step(self, callback: StepProgressCallback) -> None:
        """Register a per-step callback for step-through skill execution.

        Callback signature: ``callback(step_idx, total_steps, step_description)``.
        """
        self._confirmation.set_on_step(callback)

    async def await_learning(self) -> Optional[LearnResult]:
        """Wait for the most recent background learning task to complete.

        Returns the cached :class:`LearnResult` from the latest learning run,
        or ``None`` if no learning has been performed yet.
        """
        self._start_pending_learn()
        task = self._learn_task
        if task is not None and not task.done():
            try:
                await task
            except Exception as e:
                logger.debug("session.await_learning_error error=%s", e)
        return self._last_learn_result

    def _start_pending_learn(self) -> None:
        """Start deferred learning after the CLI is ready to show progress."""
        pending = self._pending_learn
        if pending is None:
            return
        if self._learn_task is not None and not self._learn_task.done():
            return
        trajectory, session = pending
        self._pending_learn = None
        self._learn_task = asyncio.create_task(
            self._learn_background(trajectory, session)
        )

    def _audit_log(self, event: str, **data: Any) -> None:
        if self._audit is not None:
            sid = self._session.session_id if self._session else ""
            self._audit.log(event, session_id=sid, **data)

    # ── LEARN (recording) ──

    async def enter_learning(self, goal: str = "") -> LearningSession:
        async with self._lock:
            if self._mode not in (SessionMode.IDLE, SessionMode.EVOLVING):
                raise SessionError(
                    f"Cannot enter learning from {self._mode.value} mode"
                )

            tid = await self._pipeline.start_recording(goal=goal)
            session = LearningSession(
                session_id=uuid.uuid4().hex[:16],
                goal=goal,
                trajectory_id=tid,
                start_time=time.time(),
            )
            self._session = session
            self._mode = SessionMode.LEARNING
            self._start_idle_watchdog()

            if self._session_store:
                self._session_store.save(
                    session.session_id, tid,
                    goal=goal, start_time=session.start_time,
                )

            logger.info(
                "session.learn_start id=%s goal=%s trajectory=%s",
                session.session_id, goal, tid,
            )
            self._audit_log(
                "mode.learning.start", goal=goal, trajectory=tid,
            )
            return session

    async def exit_learning(self) -> LearnResult:
        async with self._lock:
            if self._mode != SessionMode.LEARNING:
                raise SessionError(
                    f"Cannot exit learning from {self._mode.value} mode"
                )

            self._stop_idle_watchdog()
            traj = await self._pipeline.stop_recording()
            if traj is None:
                self._mode = SessionMode.IDLE
                self._session = None
                raise SessionError("No trajectory recorded")

            session = self._session
            assert session is not None
            session.end_time = time.time()

            if self._session_store:
                self._session_store.mark_completed(
                    session.session_id, end_time=session.end_time,
                )

            logger.info(
                "session.learn_stop id=%s steps=%d duration=%.1fs",
                session.session_id, traj.step_count, traj.duration,
            )
            self._audit_log(
                "mode.learning.stop",
                steps=traj.step_count, duration=traj.duration,
            )

            result = LearnResult(
                trajectory_id=traj.trajectory_id,
                step_count=traj.step_count,
                duration=traj.duration,
                storage_path=self._storage_path,
                audit_log_path=self._audit_log_path,
                event_stats=_event_stats_from_trajectory(traj),
            )

            if self._auto_learn and traj.step_count > 0:
                # Learnability assessment before committing to distillation
                if self._learnability_assessor:
                    from leapflow.learning.learnability import LearnabilityInput, LearnabilityDecision
                    assessment_input = LearnabilityInput(
                        trajectory=traj,
                        goal=session.goal if hasattr(session, 'goal') else "",
                        video_actions=getattr(self._pipeline, '_extracted_video_actions', None) or [],
                        has_video=getattr(self._pipeline, '_video_available', False),
                    )
                    report = await self._learnability_assessor.assess(assessment_input)
                    result.learnability_report = report
                    logger.info(
                        "learnability_assessment decision=%s score=%.2f reason=%s",
                        report.decision.value, report.score, report.reason,
                    )

                    if report.decision == LearnabilityDecision.SKIP:
                        self._pending_learn = None
                        self._learn_task = None
                        self._last_learn_result = result
                        self._session = None
                    elif report.decision == LearnabilityDecision.LEARN:
                        self._pending_learn = (traj, session)
                        self._learn_task = None
                    else:  # ASK — defer to CLI for user confirmation
                        self._pending_learn_deferred = (traj, session)
                        self._pending_learn = None
                        self._learn_task = None
                else:
                    self._pending_learn = (traj, session)
                    self._learn_task = None
            else:
                self._learn_task = None
                self._pending_learn = None
                self._last_learn_result = result
                self._session = None

            self._mode = SessionMode.IDLE
            return result

    def confirm_learning(self) -> None:
        """User confirmed learning after ASK decision."""
        if self._pending_learn_deferred:
            self._pending_learn = self._pending_learn_deferred
            self._pending_learn_deferred = None

    def reject_learning(self) -> None:
        """User rejected learning after ASK decision."""
        if self._pending_learn_deferred:
            traj, session = self._pending_learn_deferred
            # Store result with the report so callers can inspect it
            if self._last_learn_result is None:
                self._last_learn_result = LearnResult(
                    trajectory_id=traj.trajectory_id,
                    step_count=traj.step_count,
                    duration=traj.duration,
                    storage_path=self._storage_path,
                    audit_log_path=self._audit_log_path,
                    event_stats=_event_stats_from_trajectory(traj),
                )
            self._pending_learn_deferred = None
        self._session = None

    def pause_learning(self) -> None:
        if self._mode != SessionMode.LEARNING:
            return
        self._pipeline.pause_recording()
        self._stop_idle_watchdog()
        logger.info("session.learn_pause")
        self._audit_log("mode.learning.pause")

    def resume_learning(self) -> None:
        if self._mode != SessionMode.LEARNING:
            return
        self._pipeline.unpause_recording()
        self._start_idle_watchdog()
        logger.info("session.learn_resume")
        self._audit_log("mode.learning.resume")

    def annotate(self, text: str) -> None:
        if self._session is None:
            return
        self._session.annotations.append({
            "text": text,
            "timestamp": time.time(),
        })
        self._reset_idle_watchdog()
        logger.debug("session.annotate text=%.50s", text)

    def mark_skip(self, n: int = 1) -> int:
        """Mark last *n* recorded steps as user-skipped noise."""
        if self._mode != SessionMode.LEARNING:
            return 0
        count = self._pipeline.recorder.mark_skip(n)
        self._reset_idle_watchdog()
        return count

    async def abandon_learning(self) -> None:
        """Abandon learning without triggering distillation. Session remains resumable."""
        async with self._lock:
            if self._mode != SessionMode.LEARNING:
                return
            self._stop_idle_watchdog()
            traj = await self._pipeline.stop_recording()
            session = self._session
            if session and self._session_store:
                if traj and traj.step_count > 0:
                    self._session_store.save(
                        session.session_id, session.trajectory_id,
                        goal=session.goal, start_time=session.start_time,
                        annotations=session.annotations,
                        metadata=session.metadata,
                    )
                else:
                    self._session_store.mark_abandoned(session.session_id)
            self._mode = SessionMode.IDLE
            self._session = None
            logger.info("session.learn_abandoned")
            self._audit_log("mode.learning.abandoned")

    async def discard_learning(self) -> None:
        """Discard learning entirely — no distillation, no save for resume."""
        async with self._lock:
            if self._mode != SessionMode.LEARNING:
                return
            self._stop_idle_watchdog()
            await self._pipeline.stop_recording(discard=True)
            session = self._session
            if session and self._session_store:
                self._session_store.mark_abandoned(session.session_id)
            self._mode = SessionMode.IDLE
            self._session = None
            logger.info("session.learn_discarded")
            self._audit_log("mode.learning.discarded")

    async def resume_session(self, session_id: str) -> LearningSession:
        """Resume a previously saved learning session.

        Accepts either a session_id or trajectory_id.
        """
        async with self._lock:
            if self._mode not in (SessionMode.IDLE, SessionMode.EVOLVING):
                raise SessionError(
                    f"Cannot resume learning from {self._mode.value} mode"
                )
            if self._session_store is None:
                raise SessionError("Session store not available")

            record = self._session_store.load(session_id)
            if record is None:
                record = self._session_store.find_by_trajectory(session_id)
            if record is None:
                raise SessionError(f"Session not found: {session_id}")
            if record["status"] == "completed":
                raise SessionError(
                    f"Session {record['session_id']} is already completed"
                )

            tid = record["trajectory_id"]
            self._pipeline.resume_recording(tid)

            session = LearningSession(
                session_id=record["session_id"],
                goal=record["goal"],
                trajectory_id=tid,
                start_time=record["start_time"],
                annotations=record.get("annotations") or [],
                metadata=record.get("metadata") or {},
            )
            self._session = session
            self._mode = SessionMode.LEARNING
            self._start_idle_watchdog()

            logger.info(
                "session.learn_resume_session id=%s trajectory=%s",
                session.session_id, tid,
            )
            self._audit_log(
                "mode.learning.resume_session",
                session_id=session.session_id, trajectory=tid,
            )
            return session

    # ── LEARN ──

    async def _learn_background(
        self,
        trajectory: Trajectory,
        session: LearningSession,
    ) -> None:
        """Background distillation — never blocks the caller."""
        async with self._lock:
            self._mode = SessionMode.EVOLVING
        self._audit_log("mode.learning.start", trajectory=trajectory.trajectory_id)
        try:
            logger.info(
                "learn_background: starting analysis for %s (%d steps)",
                trajectory.trajectory_id, trajectory.step_count,
            )
            candidates = await self._pipeline.distill(
                trajectory.trajectory_id, goal=session.goal,
            )
            logger.info(
                "learn_background: analysis complete, %d candidates from distill",
                len(candidates) if candidates else 0,
            )
            if not candidates:
                # Fallback: synthesize a candidate from goal or trajectory structure
                # even when no goal was explicitly provided
                candidates = self._create_goal_based_candidate(
                    trajectory, session,
                )
                if candidates and self._observer is not None:
                    episodes = self._pipeline.get_episodes(trajectory.trajectory_id)
                    if episodes:
                        self._observer.on_candidates_ready(candidates, episodes)
                    else:
                        synth = Episode(
                            trajectory_id=trajectory.trajectory_id,
                            start_idx=0,
                            end_idx=trajectory.step_count,
                            app_sequence=list(trajectory.app_sequence),
                        )
                        self._observer.on_candidates_ready(candidates, [synth])

            activated_keys: Set[str] = set()
            if self._observer is not None:
                logger.info("learn_background: starting skill activation")
                if self._pipeline.progress_callback:
                    self._pipeline.progress_callback("activate", 0, 1)
                try:
                    activated_keys = await self._observer.await_activations()
                except Exception as e:
                    logger.debug("session.await_activations_error error=%s", e)
                if self._pipeline.progress_callback:
                    self._pipeline.progress_callback("activate", 1, 1)
            new_skills = [
                c.title for c in candidates if c.title in activated_keys
            ]
            recorder = getattr(self._pipeline, "recorder", None)
            event_stats = (
                dict(recorder.event_stats) if recorder is not None else {}
            )
            if not event_stats:
                # Recorder trajectory is cleared after stop(); fall back to the
                # passed-in trajectory so callers still see the breakdown.
                event_stats = _event_stats_from_trajectory(trajectory)
            result = LearnResult(
                trajectory_id=trajectory.trajectory_id,
                step_count=trajectory.step_count,
                duration=trajectory.duration,
                candidates=candidates,
                new_skills=new_skills,
                suggestions=len(candidates) - len(new_skills),
                storage_path=self._storage_path,
                audit_log_path=self._audit_log_path,
                event_stats=event_stats,
                activated_skill_names=activated_keys,
            )
            self._last_learn_result = result
            logger.info(
                "session.learn_done trajectory=%s candidates=%d new=%d",
                trajectory.trajectory_id, len(candidates), len(new_skills),
            )
            self._audit_log(
                "mode.learning.done",
                candidates=len(candidates), new_skills=len(new_skills),
            )
            if self._on_learn_complete:
                self._on_learn_complete(result)
        except Exception as e:
            logger.warning("session.learn_failed error=%s", e)
            self._audit_log("mode.learning.failed", error=str(e))
        finally:
            async with self._lock:
                if self._mode == SessionMode.EVOLVING:
                    self._mode = SessionMode.IDLE
            self._session = None

    def _create_goal_based_candidate(
        self,
        trajectory: Trajectory,
        session: LearningSession,
    ) -> List[DistillationCandidate]:
        """Create a candidate from session goal or trajectory structure.

        When no explicit goal is provided, derives one from the trajectory's
        app sequence and action distribution.
        """
        goal = session.goal
        apps = trajectory.app_sequence

        # Derive goal from trajectory if not explicitly provided
        if not goal:
            goal = self._infer_goal_from_trajectory(trajectory)
            if not goal:
                return []

        steps = []
        if apps:
            steps.append(f"Open {apps[0]}")

        action_steps = _summarize_trajectory_actions(trajectory)
        if action_steps:
            steps.extend(action_steps)

        steps.append(f"Complete: {goal}")

        triggers = [goal.lower()]
        words = [w for w in goal.lower().split() if len(w) > 2]
        if len(words) >= 2:
            triggers.append(" ".join(words))

        candidate = DistillationCandidate(
            title=goal,
            trigger_phrases=triggers,
            steps=steps,
            source_trajectory_id=trajectory.trajectory_id,
            confidence=0.3 if not session.goal else 0.4,
        )
        return [candidate]

    def _infer_goal_from_trajectory(self, trajectory: Trajectory) -> str:
        """Best-effort goal inference from trajectory structure when no goal was given."""
        apps = trajectory.app_sequence
        stats = _event_stats_from_trajectory(trajectory)

        if not stats:
            return ""

        # Find the dominant action type
        sorted_actions = sorted(stats.items(), key=lambda x: x[1], reverse=True)
        primary_action = sorted_actions[0][0] if sorted_actions else ""

        # Build a descriptive title from apps + primary action
        action_labels = {
            "file.create": "Create files",
            "file.modify": "Edit files",
            "file.delete": "Clean up files",
            "file.rename": "Rename files",
            "clipboard.copy": "Copy data",
            "ui.click": "UI interaction",
            "ui.type": "Text input",
            "ui.shortcut": "Keyboard workflow",
            "app.switch": "Multi-app workflow",
        }

        label = action_labels.get(primary_action, "Workflow")
        if apps:
            return f"{label} in {apps[0]}"
        return label

    # ── EXECUTE ──

    async def execute_skill(
        self,
        skill_name: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        io: Optional[IOProvider] = None,
        confirm_override: Optional[ConfirmLevel] = None,
    ) -> ExecutionResult:
        async with self._lock:
            if self._mode not in (SessionMode.IDLE, SessionMode.EVOLVING):
                raise SessionError(
                    f"Cannot execute from {self._mode.value} mode"
                )

            skill = self._registry.get(skill_name)
            if skill is None:
                return ExecutionResult(
                    ok=False,
                    skill_name=skill_name,
                    error=f"Skill '{skill_name}' not found",
                )

            self._mode = SessionMode.EXECUTING

        level = self._confirmation.determine_level(
            skill, override=confirm_override
        )

        if io is not None and level != ConfirmLevel.AUTO:
            decision = await self._confirmation.request_confirmation(
                skill, params or {}, level, io
            )
            if decision == "no":
                async with self._lock:
                    self._mode = SessionMode.IDLE
                return ExecutionResult(
                    ok=False,
                    skill_name=skill_name,
                    error="User declined execution",
                )
            if decision == "step" and io is not None:
                steps = skill.instructions if skill.instructions else [skill.description]
                step_results = await self._confirmation.step_through(
                    skill, steps, io,
                    executor=self._build_step_executor(skill, params),
                )
                async with self._lock:
                    self._mode = SessionMode.IDLE
                ok = all(r.status == "completed" for r in step_results)
                self._audit_log(
                    "skill.execute.result",
                    skill=skill_name, ok=ok, mode="step",
                )
                return ExecutionResult(
                    ok=ok,
                    skill_name=skill_name,
                    output=step_results,
                    steps_executed=sum(1 for r in step_results if r.status == "completed"),
                    steps_total=len(step_results),
                )

        self._audit_log("skill.execute", skill=skill_name, level=level.value)
        sys.stderr.write(f"\033[2m→ Running skill '{skill_name}' (confirm={level.value})\033[0m\n")
        sys.stderr.flush()

        invoke_kwargs = dict(params or {})
        if level != ConfirmLevel.AUTO and io is not None:
            from leapflow.skills.action_policy import PolicyEngine, default_rules

            invoke_kwargs["_policy"] = PolicyEngine(default_rules())
            invoke_kwargs["_io"] = io

        t0 = time.perf_counter()
        try:
            result = await self._registry.invoke(
                skill_name, **invoke_kwargs
            )
            elapsed = time.perf_counter() - t0
            exec_result = ExecutionResult(
                ok=result.ok,
                skill_name=skill_name,
                output=result.output,
                error=result.error,
                duration_s=elapsed,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            exec_result = ExecutionResult(
                ok=False,
                skill_name=skill_name,
                error=str(e),
                duration_s=elapsed,
            )
        finally:
            async with self._lock:
                self._mode = SessionMode.IDLE

        self._audit_log(
            "skill.execute.result",
            skill=skill_name, ok=exec_result.ok, duration_s=exec_result.duration_s,
        )

        if not exec_result.ok:
            self._record_skill_failure(skill, exec_result)

        self._apply_evolution_policy(skill_name, exec_result)

        status = "✓" if exec_result.ok else "✗"
        sys.stderr.write(
            f"\033[2m→ {status} Skill '{skill_name}' finished in {elapsed:.1f}s\033[0m\n"
        )
        sys.stderr.flush()
        return exec_result

    def _record_skill_failure(
        self, skill: Skill, result: ExecutionResult,
    ) -> None:
        """Record a runtime skill failure as a negative experience for future avoidance."""
        try:
            pl = self._registry.prediction_loop
            if pl is None or not pl.enabled:
                return
            pl.record_failure(
                action_desc=f"skill_failure:{result.skill_name}",
                error=result.error or "unknown error",
            )
        except Exception:
            logger.debug("record_skill_failure failed", exc_info=True)

    def _apply_evolution_policy(
        self, skill_name: str, result: ExecutionResult,
    ) -> None:
        """Apply evolution policy to update skill confidence/version after execution."""
        if self._evolution_policy is None or self._skill_store is None:
            return
        try:
            stored = self._skill_store.load_parameterized_skill(skill_name)
            if stored is None:
                return
            outcome = self._evolution_policy.on_execution_result(
                skill_name,
                success=result.ok,
                duration_s=result.duration_s,
                current_confidence=stored.get("confidence", 0.3),
                current_version=stored.get("version", 1),
                source=stored.get("source", ""),
            )
            self._skill_store.update_skill_confidence(
                skill_name, outcome.new_confidence
            )
            if outcome.version_bump:
                self._skill_store.update_parameterized_version(
                    skill_name, stored.get("code", "") or ""
                )
            if outcome.tier_changed:
                logger.info(
                    "Skill '%s' tier changed: %s -> %s (confidence=%.3f)",
                    skill_name, outcome.tier_before.name, outcome.tier_after.name,
                    outcome.new_confidence,
                )
        except Exception:
            logger.warning("skill_evolution update failed for '%s'", skill_name, exc_info=True)

    async def execute_skill_sequence(
        self,
        skill_names: List[str],
        params_list: Optional[List[Optional[Dict[str, Any]]]] = None,
        *,
        io: Optional[IOProvider] = None,
    ) -> List[ExecutionResult]:
        """Execute multiple skills in sequence (skill composition)."""
        params_list = params_list or [None] * len(skill_names)
        results: List[ExecutionResult] = []
        for name, params in zip(skill_names, params_list):
            result = await self.execute_skill(name, params, io=io)
            results.append(result)
            if not result.ok:
                break
        return results

    def _build_step_executor(self, skill: Any, params: Optional[Dict[str, Any]]):
        """Build a per-instruction step executor for step-through mode."""
        async def _executor(step_idx: int, step_desc: str) -> Dict[str, Any]:
            invoke_params = dict(params or {})
            if skill.instructions:
                invoke_params["instruction_idx"] = step_idx
            result = await self._registry.invoke(skill.name, **invoke_params)
            return {"ok": result.ok, "output": result.output, "error": result.error}
        return _executor

    def find_skill(self, phrase: str, threshold: float = 0.5) -> Optional[str]:
        """Return the name of the best-matching skill for ``phrase`` or None.

        Thin wrapper over :meth:`find_skill_match`.
        """
        m = self.find_skill_match(phrase, threshold=threshold)
        return m.skill.name if m else None

    def find_skill_match(
        self, phrase: str, threshold: float = 0.3,
    ) -> Optional[TriggerMatch]:
        """Return the best-scoring :class:`TriggerMatch` for ``phrase`` or None."""
        matches = self._registry.find_matches(phrase, threshold=threshold)
        return matches[0] if matches else None

    # ── Idle watchdog ──

    def _start_idle_watchdog(self) -> None:
        self._stop_idle_watchdog()
        try:
            loop = asyncio.get_running_loop()
            self._idle_task = loop.call_later(
                self._idle_timeout, self._on_idle_timeout
            )
        except RuntimeError:
            pass

    def _stop_idle_watchdog(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            self._idle_task = None

    def _on_idle_timeout(self) -> None:
        if self._mode == SessionMode.LEARNING:
            logger.info("session.idle_timeout after %.0fs", self._idle_timeout)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.exit_learning())
            except RuntimeError:
                pass

    def _reset_idle_watchdog(self) -> None:
        if self._mode == SessionMode.LEARNING:
            self._start_idle_watchdog()


def _event_stats_from_trajectory(trajectory: Trajectory) -> Dict[str, int]:
    """Compute action-type counts from a trajectory's recorded steps."""
    counts: Dict[str, int] = {}
    for step in trajectory.steps:
        key = step.action.action_type.value
        counts[key] = counts.get(key, 0) + 1
    return counts


_ACTION_STEP_TEMPLATES: Dict[str, str] = {
    "file.create": "Create new files in target location",
    "file.modify": "Edit file contents",
    "file.delete": "Remove unwanted files",
    "file.rename": "Rename or move files to organize",
    "clipboard.copy": "Copy data via clipboard",
    "ui.click": "Interact with UI controls",
    "ui.type": "Enter text input",
    "ui.shortcut": "Use keyboard shortcuts",
    "ui.scroll": "Navigate content by scrolling",
    "ui.drag": "Drag elements to reposition",
}

_NOISE_ACTIONS = frozenset({"app.switch", "unknown", "ui.scroll"})


def _summarize_trajectory_actions(trajectory: Trajectory) -> List[str]:
    """Extract up to 3 workflow-descriptive steps from trajectory action distribution."""
    stats = _event_stats_from_trajectory(trajectory)
    if not stats:
        return []

    meaningful = {k: v for k, v in stats.items() if k not in _NOISE_ACTIONS}
    if not meaningful:
        return []

    sorted_actions = sorted(meaningful.items(), key=lambda x: x[1], reverse=True)
    steps: List[str] = []
    for action_type, _ in sorted_actions[:3]:
        label = _ACTION_STEP_TEMPLATES.get(action_type)
        if label:
            steps.append(label)
    return steps
