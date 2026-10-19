# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression pins for how a run's end is judged when the I/O misbehaves.

Each test drives a minimal test skill through the production runtime, gate and
motion facade against a simulated arm whose transport blocks, raises, or
refuses -- the failure branches a review found unguarded:

- a stop while the first batch is still inside the transport must halt and
  report UNKNOWN, and the devices stay held until that write is reconciled
  (a late landing is halted again);
- a dispatch that raises is UNKNOWN; one that could not be attempted is NONE;
- the commanded speed is quantized *down* and never exceeds the effective cap,
  and a limit below what the arm accepts is refused rather than raised;
- a persisted calibration is a COMMITTED side effect, and the merge never
  downgrades what the skill itself stated.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from leapflow.hardware.skills.ports import (
    SkillContext,
)
from leapflow.hardware.skills.protocol import (
    FAILURE_CHECK_FAILED,
    FAILURE_DEVICE_BUSY,
    FAILURE_DEVICE_FAULT,
    FAILURE_INVALID_ARGS,
    FAILURE_STOPPED,
    FAILURE_WRITE_REFUSED,
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_PARTIAL,
    SIDE_EFFECT_UNKNOWN,
    STATUS_HALTED,
    SkillError,
    SkillOutcome,
    SkillPlan,
    SkillSpec,
)
from leapflow.hardware.skills.runtime import SkillRun, SkillRuntime, SkillRuntimeSettings
from leapflow.hardware.skills.workcell import (
    SPEED_CHANNEL,
)
from tests._harness.skill_stubs import (
    ARM_ID,
    WORKCELL_ID,
    SimArmTransport,
    SkillNodeRegistry,
    write_workcell,
)

pytestmark = pytest.mark.asyncio

_SCHEMA: Mapping[str, Any] = {"type": "object"}


class _MoveDown:
    """Lowers the TCP 10 mm once, under whatever limits the test asks for."""

    def __init__(self, **limits: float) -> None:
        self.spec = SkillSpec(
            skill_id="test.move",
            version="1",
            card="Lowers the TCP once.",
            args_schema=_SCHEMA,
            requires=("role:arm", "calib:tcp", "calib:workspace"),
        )
        self._limits = dict(limits)

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        tcp = (await ctx.motion.state()).tcp
        target = replace(tcp, z=tcp.z - 10.0)
        ctx.motion.check_target(target)
        return SkillPlan(
            skill_id=self.spec.skill_id,
            args=args,
            summary="lower 10 mm",
            targets=(target,),
            limits=self._limits,
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        await ctx.motion.move_tcp(plan.targets[0])
        return SkillOutcome.succeeded()


class _Idle:
    """An actuating skill that never touches I/O (for busy-device checks)."""

    spec = SkillSpec(skill_id="test.idle", version="1", card="Does nothing.", args_schema=_SCHEMA)

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        return SkillPlan(skill_id="test.idle", args=args, summary="idle")

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        return SkillOutcome.succeeded(side_effect_state=SIDE_EFFECT_NONE)


class _BlockingArm(SimArmTransport):
    """Holds every batch inside the transport until the test releases it."""

    def __init__(self, **sim: Any) -> None:
        super().__init__(**sim)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def write_batch(self, commands: tuple[tuple[str, Any], ...]) -> Any:
        self.entered.set()
        await self.release.wait()
        return await super().write_batch(commands)


class _RaisingArm(SimArmTransport):
    """A driver bug: the batch ends in an exception that says nothing about the bus."""

    async def write_batch(self, commands: tuple[tuple[str, Any], ...]) -> Any:
        raise RuntimeError("driver state machine wedged")


def _registry(tmp_path: Path, transport: SimArmTransport | None = None) -> SkillNodeRegistry:
    registry = SkillNodeRegistry(
        transport or SimArmTransport(), store_path=tmp_path / "instrument.duckdb"
    )
    registry.calibrate_cell()
    return registry


def _runtime(
    tmp_path: Path,
    registry: SkillNodeRegistry,
    skills: list[Any],
    *,
    workcell: Mapping[str, Any] | None = None,
    **settings: Any,
) -> SkillRuntime:
    directory = tmp_path / "workcells"
    write_workcell(directory, workcell)
    return SkillRuntime(
        registry,
        settings=SkillRuntimeSettings(workcells_dir=directory, **settings),
        skills=skills,
        poll_s=0.005,
    )


async def _plan(runtime: SkillRuntime, skill_id: str) -> SkillPlan:
    plan = await runtime.plan(WORKCELL_ID, skill_id, {})
    assert isinstance(plan, SkillPlan), plan
    return plan


async def _start(runtime: SkillRuntime, skill_id: str) -> SkillRun:
    run = await runtime.execute((await _plan(runtime, skill_id)).digest)
    assert isinstance(run, SkillRun), run
    return run


async def _outcome(runtime: SkillRuntime, skill_id: str) -> SkillOutcome:
    run = await _start(runtime, skill_id)
    finished = await runtime.wait(run.run_id, timeout_s=5.0)
    assert finished is not None and finished.outcome is not None
    return finished.outcome


async def _until(predicate: Any, timeout_s: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "condition never held"
        await asyncio.sleep(0.005)


def _speeds(sim: SimArmTransport) -> list[float]:
    return [value for batch in sim.batches for cid, value in batch if cid == SPEED_CHANNEL]


# ── Dispatch verdicts ────────────────────────────────────────────────────────


async def test_a_stop_during_the_first_dispatch_halts_and_reports_unknown(
    tmp_path: Path,
) -> None:
    arm = _BlockingArm()
    registry = _registry(tmp_path, arm)
    runtime = _runtime(
        tmp_path, registry, [_MoveDown(), _Idle()], stop_grace_s=0.05, dispatch_settle_s=0.05
    )
    idle = await _plan(runtime, "test.idle")
    run = await _start(runtime, "test.move")
    await asyncio.wait_for(arm.entered.wait(), timeout=2.0)

    await runtime.stop(run.run_id, "operator pressed stop")
    await runtime.wait(run.run_id, timeout_s=2.0)
    assert run.outcome is not None
    assert (run.outcome.failure_code, run.outcome.status, run.outcome.side_effect_state) == (
        FAILURE_STOPPED,
        STATUS_HALTED,
        SIDE_EFFECT_UNKNOWN,
    )
    assert arm.halts == 1, "a write still inside the transport may move the arm"

    # The verdict is out, but the write is not: the devices stay held.
    assert runtime.holds_device(ARM_ID)
    busy = await runtime.execute(idle.digest)
    assert isinstance(busy, SkillOutcome) and busy.failure_code == FAILURE_DEVICE_BUSY

    arm.release.set()
    await _until(lambda: not runtime.holds_device(ARM_ID))
    assert arm.batches, "the late batch did land"
    assert arm.halts == 2, "a write that lands after the halt is halted again"


async def test_a_dispatch_that_raises_is_unknown_and_halts(tmp_path: Path) -> None:
    arm = _RaisingArm()
    runtime = _runtime(tmp_path, _registry(tmp_path, arm), [_MoveDown()])
    outcome = await _outcome(runtime, "test.move")
    assert (outcome.failure_code, outcome.status, outcome.side_effect_state) == (
        FAILURE_DEVICE_FAULT,
        STATUS_HALTED,
        SIDE_EFFECT_UNKNOWN,
    )
    assert "driver state machine wedged" in outcome.message
    assert arm.halts == 1
    assert not runtime.holds_device(ARM_ID)


async def test_a_dispatch_that_could_not_be_attempted_is_none(tmp_path: Path) -> None:
    arm = SimArmTransport(refuse_writes=True)
    runtime = _runtime(tmp_path, _registry(tmp_path, arm), [_MoveDown()])
    outcome = await _outcome(runtime, "test.move")
    assert (outcome.failure_code, outcome.side_effect_state) == (
        FAILURE_WRITE_REFUSED,
        SIDE_EFFECT_NONE,
    )
    assert arm.halts == 0, "nothing left the node; a halt would latch the e-stop"


# ── Limits ───────────────────────────────────────────────────────────────────


async def test_the_commanded_speed_is_quantized_down(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    runtime = _runtime(tmp_path, registry, [_MoveDown(speed_pct=12.7)])
    assert (await _outcome(runtime, "test.move")).ok
    assert _speeds(registry.sim) == [12.0]


@pytest.mark.parametrize(("node_cap", "sent"), [(30.0, 30.0), (25.5, 25.0)])
async def test_the_commanded_speed_never_exceeds_the_effective_cap(
    tmp_path: Path, node_cap: float, sent: float
) -> None:
    registry = _registry(tmp_path)
    runtime = _runtime(tmp_path, registry, [_MoveDown(speed_pct=80.0)], speed_pct_max=node_cap)
    plan = await _plan(runtime, "test.move")
    assert plan.limits["speed_pct"] <= plan.limits["speed_pct_max"] <= node_cap
    assert (await _outcome(runtime, "test.move")).ok
    assert _speeds(registry.sim) == [sent]


async def test_a_speed_below_the_arm_minimum_is_refused_not_raised(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    runtime = _runtime(tmp_path, registry, [_MoveDown(speed_pct=0.5)])
    plan = await _plan(runtime, "test.move")
    assert plan.limits["speed_pct"] == 0.5, "no floor lifts a requested limit"
    outcome = await _outcome(runtime, "test.move")
    assert (outcome.failure_code, outcome.side_effect_state) == (
        FAILURE_INVALID_ARGS,
        SIDE_EFFECT_NONE,
    )
    assert registry.sim.batches == []


async def test_a_non_positive_duration_is_refused_at_planning(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, _registry(tmp_path), [_MoveDown(max_duration_s=0.0)])
    refused = await runtime.plan(WORKCELL_ID, "test.move", {})
    assert isinstance(refused, SkillOutcome)
    assert (refused.failure_code, refused.side_effect_state) == (
        FAILURE_INVALID_ARGS,
        SIDE_EFFECT_NONE,
    )


# ── Side-effect ledger ───────────────────────────────────────────────────────


class _Records:
    """Persists a calibration, then ends however the test says."""

    spec = SkillSpec(
        skill_id="calib.test", version="1", card="Writes the safe box.", args_schema=_SCHEMA
    )

    def __init__(self, end: str) -> None:
        self._end = end

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        return SkillPlan(skill_id="calib.test", args=args, summary="calibrate")

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        ctx.record_calibration(
            "workspace.safe_box",
            parameters={"x": [120, 420], "y": [-150, 150], "z": [90, 400]},
        )
        if self._end == "raise":
            raise SkillError(FAILURE_CHECK_FAILED, "residual too large")
        if self._end == "stated":
            return SkillOutcome.failed(
                FAILURE_CHECK_FAILED,
                "saved, but the check after it failed",
                side_effect_state=SIDE_EFFECT_COMMITTED,
            )
        return SkillOutcome.succeeded(side_effect_state=SIDE_EFFECT_NONE)


@pytest.mark.parametrize(
    ("end", "ok", "state"),
    [
        ("succeed", True, SIDE_EFFECT_COMMITTED),  # the skill said NONE; a record persisted
        ("raise", False, SIDE_EFFECT_PARTIAL),  # failed after persisting
        ("stated", False, SIDE_EFFECT_COMMITTED),  # the skill's own word is kept
    ],
)
async def test_a_persisted_calibration_is_a_side_effect_never_downgraded(
    tmp_path: Path, end: str, ok: bool, state: str
) -> None:
    registry = _registry(tmp_path)
    runtime = _runtime(tmp_path, registry, [_Records(end)])
    outcome = await _outcome(runtime, "calib.test")
    assert (outcome.ok, outcome.side_effect_state) == (ok, state)
    assert "workspace.safe_box" in outcome.evidence["calibrations"]
    assert registry.sim.halts == 0, "nothing moved; persisting is not motion"
