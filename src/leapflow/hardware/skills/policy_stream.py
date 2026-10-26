# Copyright (c) Alibaba, Inc. and its affiliates.
"""Streaming a served action-chunk policy through its adapter and the safety shield.

``WorkcellPolicies`` is the node's ``PolicyPort``. A policy is a model-service
device bound under the workcell's ``models``; what its vectors mean is declared
by the server in ``model_info`` and embodied by the workcell's ``policies``
(see ``policy_binding``). The binding is re-checked against the live
declarations on every use, a run streams only the binding that matches the
approved model and adapter digest, and the model identity is checked on every
response.

One run:

1. observe -- the arm state, the declared contact channels, and one frame per
   camera fitted to the size the model was trained at -- and encode the state
   into the model's observation space;
2. infer concurrently with the stream: the next chunk is requested once less
   than ``prefetch_ratio`` of the current one remains, and a model declaring
   ``action_space.pending`` is sent the rows not yet executed (real-time
   chunking), in its own space, with the expected latency in steps;
3. the action adapter turns the native chunk into channel targets anchored at
   the observation it answers (normalisation, units, rotation, frame,
   reference, gripper, IK, plugins, then validation and the first-action jump);
4. ``ChunkAggregator`` merges chunks into one time-indexed stream and an
   optional low-pass smooths it;
5. every tick ``SafetyShield`` bounds, jump-checks, acceleration- and
   slew-limits and pose-checks the sample, and ``GovernedMotion.command`` sends
   it through the node write gate.

The run ends when the model reports ``done`` and its last chunk is spent, when
``duration_s`` (a safety cap, not the success criterion) elapses, or on a
violation, a model failure, starvation (holding longer than ``starve_s``,
``warmup_s`` before the first chunk) or a stop. Ending leaves the arm at its
last shield-approved target, so a policy failure needs no halt; a stop, a
timeout, shield refusal or device fault is halted by the runtime as for every skill.

A recorded run's episode carries the full ``model_info``, the adapter
declaration and its digest, and every chunk both as the model returned it and
as adapted, next to what the shield let through.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Mapping

from leapflow.hardware.model_contract import (
    TASK_ACTION_CHUNK,
    ActionChunkResult,
    ModelService,
    act,
)
from leapflow.hardware.rgbd import FrameSource
from leapflow.hardware.skills.adapt import Action, AdaptError, AdaptViolation, ChunkAggregator
from leapflow.hardware.skills.adapt.aggregate import LowPass
from leapflow.hardware.skills.adapt.observe import encode_image
from leapflow.hardware.skills.frames import FrameGraph
from leapflow.hardware.skills.geometry import Pose
from leapflow.hardware.skills.motion import GovernedMotion
from leapflow.hardware.skills.policy_binding import (
    PolicyBinding,
    PolicySettings,
    PosePredictor,
    bind_policy,
    pose_frames,
)
from leapflow.hardware.skills.protocol import (
    FAILURE_ARM_NOT_READY,
    FAILURE_CALIBRATION_MISSING,
    FAILURE_CALIBRATION_STALE,
    FAILURE_DEVICE_FAULT,
    FAILURE_INVALID_ARGS,
    FAILURE_MODEL_UNAVAILABLE,
    FAILURE_PLAN_STALE,
    FAILURE_SHIELD_VIOLATION,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_TIMEOUT,
    ArmState,
    PolicyCard,
    STATUS_HALTED,
    PolicyReport,
    SkillError,
)
from leapflow.hardware.skills.shield import SafetyShield, contact_violation
from leapflow.hardware.skills.workcell import ROLE_ARM, SPEED_CHANNEL, WorkcellSpec
from leapflow.hardware.transport import TransportError

DEADLINE_MARGIN_S = 0.5  # end the stream before the run deadline turns into a halt
ENDED_DURATION = "duration"
ENDED_DONE = "done"
_VERSION_TOLERANCE = 1e-6
_LATENCY_GAIN = 0.5  # weight of the newest round trip in the latency estimate


def _trusted_contact(reading: Any, device_id: str, channel_id: str) -> float:
    """Return one trustworthy finite contact sample, or a typed device fault."""
    label = f"{device_id}.{channel_id}"
    if reading is None:
        raise SkillError(FAILURE_DEVICE_FAULT, f"contact channel {label} returned no reading")
    if not bool(getattr(reading, "is_trustworthy", False)):
        quality = getattr(reading, "quality", "unknown")
        raise SkillError(
            FAILURE_DEVICE_FAULT,
            f"contact channel {label} reported untrustworthy quality {quality!r}",
        )
    value = getattr(reading, "value", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SkillError(
            FAILURE_DEVICE_FAULT, f"contact channel {label} returned no numeric reading"
        )
    sample = float(value)
    if not math.isfinite(sample):
        raise SkillError(
            FAILURE_DEVICE_FAULT, f"contact channel {label} returned a non-finite reading"
        )
    return sample


@dataclass(frozen=True)
class Observation:
    state: ArmState
    observed_at: float  # run clock, when the state was read
    images: Mapping[str, bytes]


@dataclass(frozen=True)
class Adapted:
    """One answered chunk: the model's result, its robot-side rows, what it answered."""

    result: ActionChunkResult
    rows: tuple[Action, ...]
    observed_at: float
    pending: tuple[Action, ...] = ()  # the native rows sent back for real-time chunking


Pending = Callable[[float], tuple[Action, ...]]


class PolicySession:
    """One run's inference: observe, encode, ask, check the identity, adapt.

    Owns what outlives a chunk: the adapter's memory (a binary gripper's last
    decision) and the round-trip latency estimate real-time chunking reports.
    """

    def __init__(
        self,
        binding: PolicyBinding,
        task: str,
        frames: FrameGraph,
        observe: Callable[[], Awaitable[Observation]],
        clock: Callable[[], float],
    ) -> None:
        self._binding = binding
        self._task = task
        self._frames = frames
        self._observe = observe
        self._clock = clock
        self._memory: dict[str, Any] = {}
        self._latency_s: float | None = None

    async def chunk(
        self, observation: Observation | None = None, pending: Pending | None = None
    ) -> Adapted:
        """The next adapted chunk for *observation* (a fresh one when None)."""
        binding = self._binding
        name = binding.card.name
        if observation is None:
            observation = await self._observe()
        ctx = binding.context(observation.state, self._frames, self._memory)
        sent: tuple[Action, ...] | None = None
        try:
            state = binding.observed_state(observation.state, ctx)
            if pending is not None and binding.spaces.action.pending:
                rows = pending(self._clock())
                sent = binding.actions.invert(rows, ctx) if rows else ()
        except AdaptError as exc:
            raise SkillError(
                FAILURE_MODEL_UNAVAILABLE, f"policy {name!r} cannot encode its observation: {exc}"
            ) from exc
        try:
            result = await act(
                binding.service,
                observation.images,
                state,
                self._task,
                pending=sent,
                latency_steps=self._latency_steps(),
            )
        except TransportError as exc:
            raise SkillError(FAILURE_MODEL_UNAVAILABLE, f"policy {name!r} failed: {exc}") from exc
        self._learn(self._clock() - observation.observed_at)
        changed = binding.info.identity.difference(result.model) if binding.info else ""
        if changed:
            raise SkillError(
                FAILURE_MODEL_UNAVAILABLE,
                f"policy {name!r} answered as a different model ({changed}); re-plan against "
                "the model now serving",
            )
        try:
            rows = await asyncio.to_thread(binding.actions.adapt, result.actions, ctx)
        except AdaptViolation as exc:
            raise SkillError(
                FAILURE_SHIELD_VIOLATION, f"the adapter refused policy {name!r}'s chunk: {exc}"
            ) from exc
        except AdaptError as exc:
            raise SkillError(
                FAILURE_MODEL_UNAVAILABLE, f"policy {name!r} returned an unusable chunk: {exc}"
            ) from exc
        return Adapted(result, rows, observation.observed_at, sent or ())

    def _latency_steps(self) -> int:
        if self._latency_s is None:
            return 0
        return max(0, math.ceil(self._latency_s / self._binding.step_s - 1e-9))

    def _learn(self, latency_s: float) -> None:
        latency_s = max(0.0, latency_s)
        if self._latency_s is None:
            self._latency_s = latency_s
        else:
            self._latency_s += _LATENCY_GAIN * (latency_s - self._latency_s)


class WorkcellPolicies:
    """Policy streaming for one skill context. Satisfies ``PolicyPort``."""

    def __init__(
        self,
        *,
        registry: Any,
        spec: WorkcellSpec,
        frames: FrameGraph,
        motion: GovernedMotion,
        settings: PolicySettings,
        note: Callable[[str, Any], None],
    ) -> None:
        self._registry = registry
        self._spec = spec
        self._frames = frames
        self._motion = motion
        self._settings = settings
        self._note = note

    @property
    def startup_s(self) -> float:
        return self._settings.startup_s

    async def preview(self, name: str, task: str) -> PolicyReport:
        binding = await self._bind(name, task)
        observation = await self._observe(binding, first=True)
        predictor = PosePredictor(binding, observation.state, self._motion)
        try:
            adapted = await self._session(binding, task).chunk(observation)
        except SkillError as error:
            if error.failure_code != FAILURE_SHIELD_VIOLATION:
                raise
            report = PolicyReport(
                card=binding.card, chunks=1, end=observation.state.tcp, violation=error.message
            )
            self._note("policy", report.to_dict())
            return report
        shield = self._shield(binding, predictor)
        previous = binding.values(observation.state)
        rates: Action = (0.0,) * len(previous)
        interventions, violation = 0, ""
        for row in adapted.rows:
            verdict = shield.filter(row, previous, binding.step_s, rates)
            interventions += verdict.clamped
            if not verdict.ok:
                violation = verdict.violation
                break
            previous, rates = verdict.values, verdict.rates
        result = adapted.result
        report = PolicyReport(
            card=binding.card,
            chunks=1,
            horizon=result.horizon,
            interventions=interventions,
            infer_ms=result.infer_ms,
            end=predictor.tcp(previous),
            violation=violation,
            ended=ENDED_DONE if result.done else "",
            progress=result.progress,
        )
        self._note("policy", report.to_dict())
        return report

    async def run(
        self,
        name: str,
        task: str,
        *,
        approved: Mapping[str, Any],
        duration_s: float,
        speed_pct: float | None = None,
    ) -> PolicyReport:
        if not (math.isfinite(duration_s) and duration_s > 0):
            raise SkillError(FAILURE_INVALID_ARGS, "duration_s must be a positive number")
        binding = await self._bind(name, task)
        _check_approved(binding.card, approved)
        self._motion.tap.annotate("policy", _episode_binding(binding, task, self._frames))
        first = await self._observe(binding, first=True)
        arm_id = self._spec.role(ROLE_ARM)
        has_speed = self._registry.context(arm_id).channel(SPEED_CHANNEL) is not None
        stream = PolicyStream(
            binding=binding,
            session=self._session(binding, task),
            first=first,
            motion=self._motion,
            settings=self._settings,
            shield=self._shield(binding, PosePredictor(binding, first.state, self._motion)),
            arm_id=arm_id,
            speed=self._motion.speed_for(speed_pct) if has_speed else None,
        )
        try:
            await stream.drive(float(duration_s))
        except SkillError as error:
            report = stream.report()
            error.result.setdefault("policy", report.to_dict())
            self._note("policy", report.to_dict())
            raise
        report = stream.report(end=(await self._motion.state()).tcp)
        self._note("policy", report.to_dict())
        return report

    # ── Internals ──

    def _session(self, binding: PolicyBinding, task: str) -> PolicySession:
        return PolicySession(
            binding, task, self._frames, lambda: self._observe(binding), self._motion.now
        )

    async def _bind(self, name: str, task: str = "") -> PolicyBinding:
        workcell = self._spec.workcell_id
        device_id = self._spec.model(name)
        if not device_id:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                f"workcell {workcell!r} binds no {name!r} model (declare it under 'models')",
            )
        service = await self._transport(
            device_id, ModelService, "a model service", FAILURE_MODEL_UNAVAILABLE
        )
        try:
            # Re-verified on every bind, so a model swapped since planning shows up
            # as a different card identity, not as a stale cache.
            info = await service.refresh()
        except TransportError as exc:
            raise SkillError(FAILURE_MODEL_UNAVAILABLE, f"policy {name!r}: {exc}") from exc
        if info.task != TASK_ACTION_CHUNK:
            raise SkillError(
                FAILURE_MODEL_UNAVAILABLE,
                f"model {info.name!r} serves {info.task!r}, not {TASK_ACTION_CHUNK!r}",
            )
        try:
            binding = bind_policy(name, device_id, info, self._spec, self._registry, self._settings)
        except ValueError as exc:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE, f"policy {name!r} cannot drive {workcell!r}: {exc}"
            ) from exc
        self._check_calibration(binding)
        if task:
            _check_task(binding, task)
        return replace(binding, service=service)

    def _check_calibration(self, binding: PolicyBinding) -> None:
        current = self._frames.versions
        name = binding.card.name
        for frame, version in binding.spaces.calibration.items():
            now = current.get(frame)
            if now is None or abs(now - version) > _VERSION_TOLERANCE:
                raise SkillError(
                    FAILURE_CALIBRATION_STALE,
                    f"policy {name!r} was trained with {frame} calibration "
                    f"{version}, the workcell now has {now if now is not None else 'none'}; "
                    "restore the setup or retrain the policy",
                )
        for frame in pose_frames(binding):
            if frame not in current:
                raise SkillError(
                    FAILURE_CALIBRATION_MISSING,
                    f"policy {name!r} expresses poses in the {frame!r} frame, which is not "
                    "calibrated",
                )

    def _check_kinematics(self, binding: PolicyBinding, state: ArmState) -> None:
        if binding.chain is None:
            return
        joints = binding.values(state, binding.joint_order)
        mm, deg = binding.chain.agreement(joints, state.flange)
        mm_limit = float(self._spec.limits["fk_tolerance_mm"])
        deg_limit = float(self._spec.limits["fk_tolerance_deg"])
        if mm > mm_limit or deg > deg_limit:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                f"the arm's kinematics.dh puts the flange {mm:.1f} mm / {deg:.1f} deg from where "
                f"the firmware reports it (tolerance {mm_limit:g} mm / {deg_limit:g} deg); "
                "fix the table before streaming joints",
            )

    def _shield(self, binding: PolicyBinding, predictor: PosePredictor) -> SafetyShield:
        return SafetyShield(
            binding.channels, rate_margin=self._settings.rate_margin, pose_check=predictor.check
        )

    async def _observe(self, binding: PolicyBinding, *, first: bool = False) -> Observation:
        state = await self._motion.state()
        observed_at = self._motion.now()
        if state.enabled is False:
            raise SkillError(
                FAILURE_ARM_NOT_READY, "the arm is not enabled; energize it (arm.enable=1) first"
            )
        if first:
            self._check_kinematics(binding, state)
        await self._check_contact(binding)
        quality = int(self._settings.jpeg_quality)
        fitted = binding.spaces.observation.images
        images: dict[str, bytes] = {}
        for name, device_id in binding.cameras.items():
            try:
                # Resolved under the device lock so a preview release cannot close the
                # camera between the lookup and the capture.
                async with self._registry.device_io(device_id):
                    source = await self._transport(
                        device_id, FrameSource, "an RGB-D camera", FAILURE_SKILL_UNAVAILABLE
                    )
                    frame = await source.capture_rgbd()
            except TransportError as exc:
                raise SkillError(
                    FAILURE_DEVICE_FAULT, f"camera {device_id!r} capture failed: {exc}"
                ) from exc
            images[name] = await asyncio.to_thread(
                encode_image, frame, fitted.get(name), quality=quality
            )
        return Observation(state, observed_at, images)

    async def _check_contact(self, binding: PolicyBinding) -> None:
        """Stop on an untrustworthy or out-of-envelope contact reading."""
        if not binding.contacts:
            return
        wanted: dict[str, list[str]] = {}
        for limit in binding.contacts:
            wanted.setdefault(limit.device_id, []).append(limit.channel_id)
        readings: dict[str, dict[str, Any]] = {}
        for device_id, channel_ids in wanted.items():
            try:
                batch = await self._registry.read_batch(device_id, tuple(channel_ids))
            except TransportError as exc:
                raise SkillError(
                    FAILURE_DEVICE_FAULT, f"{device_id!r} contact channels unreadable: {exc}"
                ) from exc
            sampled = {reading.channel_id: reading for reading in batch.readings}
            readings[device_id] = {
                channel_id: _trusted_contact(sampled.get(channel_id), device_id, channel_id)
                for channel_id in channel_ids
            }
        problem = contact_violation(binding.contacts, readings)
        if problem:
            raise SkillError(
                FAILURE_SHIELD_VIOLATION,
                f"contact limit: {problem}",
                status=STATUS_HALTED,
            )

    async def _transport(self, device_id: str, capability: type, what: str, code: str) -> Any:
        try:
            transport = await self._registry.transport(device_id)
        except TransportError as exc:
            raise SkillError(code, f"{device_id!r} is unavailable: {exc}") from exc
        if not isinstance(transport, capability):
            raise SkillError(code, f"{device_id!r} is not {what}")
        return transport


def _check_approved(card: PolicyCard, approved: Mapping[str, Any]) -> None:
    """The binding about to stream is the model and embodiment the plan pinned."""
    name = card.name
    if dict(card.model) != dict(approved.get("model") or {}):
        raise SkillError(
            FAILURE_PLAN_STALE,
            f"policy {name!r} now serves {card.model.get('name')} "
            f"{card.model.get('version')}, not the approved model; plan again",
        )
    if card.adapter_digest != approved.get("adapter_digest"):
        raise SkillError(
            FAILURE_PLAN_STALE,
            f"policy {name!r}'s embodiment (policies.{name}, its spaces or the bound "
            "channels) changed since approval; plan again",
        )


def _episode_binding(binding: PolicyBinding, task: str, frames: FrameGraph) -> dict[str, Any]:
    """What a converter needs to turn a recorded run into training rows."""
    return {
        "name": binding.card.name,
        "task": task,
        "device_id": binding.device_id,
        "model_info": None if binding.info is None else binding.info.to_dict(),
        "space": binding.card.space,
        "action_channels": list(binding.card.channels),
        "state_channels": [c for bound in binding.state_groups for c in bound.channels],
        "cameras": dict(binding.cameras),
        "fps": binding.card.fps,
        "adapter": binding.describe(),
        "encoding": binding.encoding(frames),
    }


def _check_task(binding: PolicyBinding, task: str) -> None:
    card = binding.card
    if not task.strip():
        raise SkillError(FAILURE_INVALID_ARGS, "name the task the policy should perform")
    if not binding.spaces.accepts(task):
        raise SkillError(
            FAILURE_INVALID_ARGS,
            f"policy {card.name!r} accepts only its own instructions: {'; '.join(card.tasks)}",
        )


# ── Stream ───────────────────────────────────────────────────────────────────


class PolicyStream:
    """One run's streaming state: aggregator, filter, shield, last command, and counters."""

    def __init__(
        self,
        *,
        binding: PolicyBinding,
        session: PolicySession,
        first: Observation,
        motion: GovernedMotion,
        settings: PolicySettings,
        shield: SafetyShield,
        arm_id: str,
        speed: float | None,
    ) -> None:
        self._binding = binding
        self._session = session
        self._first = first
        self._motion = motion
        self._settings = settings
        self._shield = shield
        self._arm_id = arm_id
        self._speed = speed  # rides once, in the first arm batch
        decl = binding.decl
        self._aggregator = ChunkAggregator(
            binding.actions.width,
            cyclic=binding.cyclic,
            mode=decl.aggregate,
            blend_s=settings.blend_s,
            decay=settings.ensemble_decay,
            interpolation=decl.interpolation,
        )
        start = binding.values(first.state)
        self._smooth = LowPass(decl.smooth_hz, binding.cyclic)
        self._smooth.reset(start)
        self._previous: Action = start
        self._rates: Action = (0.0,) * len(start)
        self._last_sent: float | None = None
        self._started = motion.now()
        self._infer_ms = 0.0
        self._ticks = 0
        self._interventions = 0
        self._horizon = 0
        self._violation = ""
        self._done = False
        self._progress: float | None = None
        self._ended = ""

    async def drive(self, duration_s: float) -> None:
        """Stream until done, *duration_s* after the first chunk, or raise the run's verdict."""
        motion, period = self._motion, 1.0 / self._settings.tick_hz
        hard_stop = motion.deadline - DEADLINE_MARGIN_S
        end = hard_stop  # until the first chunk lands
        inflight: asyncio.Future[Adapted] | None
        inflight = asyncio.ensure_future(self._session.chunk(self._first))
        fed = next_tick = self._started
        try:
            while (now := motion.now()) < end:
                if inflight is not None and inflight.done():
                    if self._aggregator.chunks == 0:
                        end = min(now + duration_s, hard_stop)
                    self._adopt(inflight.result(), now)
                    inflight = None
                if inflight is None and self._wants_chunk(now):
                    inflight = asyncio.ensure_future(
                        self._session.chunk(None, self._aggregator.pending)
                    )
                target = self._aggregator.sample(now)
                if target is None:
                    if self._done:
                        self._ended = ENDED_DONE
                        break
                    self._check_starved(now - fed)
                else:
                    await self._send(target, now)
                    fed = now
                next_tick += period
                current = motion.now()
                if next_tick <= current:
                    next_tick = current + period  # behind: skip ticks, never burst
                await motion.pause(next_tick - current)
        finally:
            await _discard(inflight)
        self._ended = self._ended or ENDED_DURATION

    def report(self, *, end: Pose | None = None) -> PolicyReport:
        chunks = self._aggregator.chunks
        return PolicyReport(
            card=self._binding.card,
            ticks=self._ticks,
            chunks=chunks,
            horizon=self._horizon,
            interventions=self._interventions,
            elapsed_s=self._motion.now() - self._started,
            infer_ms=self._infer_ms / chunks if chunks else 0.0,
            end=end,
            violation=self._violation,
            ended=self._ended,
            progress=self._progress,
        )

    def _adopt(self, adapted: Adapted, now: float) -> None:
        result = adapted.result
        try:
            self._aggregator.offer(
                adapted.rows,
                observed_at=adapted.observed_at,
                step_s=self._binding.step_s,
                now=now,
            )
        except ValueError as exc:
            raise SkillError(
                FAILURE_MODEL_UNAVAILABLE,
                f"policy {self._binding.card.name!r} returned an unusable chunk: {exc}",
            ) from exc
        self._horizon = result.horizon
        self._infer_ms += result.infer_ms
        self._done = self._done or result.done
        if result.progress is not None:
            self._progress = result.progress
        self._motion.tap.event(
            "chunk",
            index=self._aggregator.chunks - 1,
            latency_s=round(now - adapted.observed_at, 4),
            infer_ms=result.infer_ms,
            pending=len(adapted.pending),
            done=result.done,
            progress=result.progress,
            native=[list(row) for row in result.actions],
            actions=[list(row) for row in adapted.rows],
        )

    def _wants_chunk(self, now: float) -> bool:
        aggregator = self._aggregator
        if aggregator.chunks == 0 or self._done:
            return False  # the first inference is in flight, or the model has finished
        prefetch = self._settings.prefetch_ratio * aggregator.duration_s()
        return aggregator.remaining_s(now) < prefetch

    def _check_starved(self, waited_s: float) -> None:
        settings = self._settings
        limit = settings.starve_s if self._aggregator.chunks else settings.warmup_s
        if waited_s > limit:
            raise SkillError(
                FAILURE_TIMEOUT,
                f"policy {self._binding.card.name!r} supplied no action for {waited_s:.1f} s "
                f"(limit {limit:g} s); inference is too slow or stalled",
            )

    async def _send(self, target: Action, now: float) -> None:
        period = 1.0 / self._settings.tick_hz
        dt = period if self._last_sent is None else now - self._last_sent
        smoothed = self._smooth(target, dt)
        verdict = self._shield.filter(smoothed, self._previous, dt, self._rates)
        self._interventions += verdict.clamped
        if not verdict.ok:
            self._violation = verdict.violation
            raise SkillError(
                FAILURE_SHIELD_VIOLATION,
                f"the safety shield refused the policy's next action: {verdict.violation}",
                status=STATUS_HALTED,
            )
        for device_id, commands in self._batches(verdict.values):
            await self._motion.command(device_id, commands)
        self._previous, self._rates, self._last_sent = verdict.values, verdict.rates, now
        self._ticks += 1

    def _batches(self, values: Action) -> list[tuple[str, list[tuple[str, Any]]]]:
        batches: dict[str, list[tuple[str, Any]]] = {}
        for channel, value in zip(self._binding.channels, values):
            batches.setdefault(channel.device_id, []).append((channel.channel_id, value))
        if self._speed is not None and self._arm_id in batches:
            batches[self._arm_id].insert(0, (SPEED_CHANNEL, self._speed))
            self._speed = None
        return list(batches.items())


async def _discard(task: asyncio.Future[Any] | None) -> None:
    """Cancel and reap an inference the run no longer needs; its result is moot."""
    if task is None:
        return
    if not task.done():
        task.cancel()
    await asyncio.wait({task})
    if not task.cancelled():
        task.exception()  # retrieved, so it is not reported as never awaited


__all__ = [
    "DEADLINE_MARGIN_S",
    "ENDED_DONE",
    "ENDED_DURATION",
    "Adapted",
    "Observation",
    "PolicySession",
    "PolicyStream",
    "WorkcellPolicies",
]
