# Copyright (c) Alibaba, Inc. and its affiliates.
"""A simulated Piper cell serving one ``policy.run`` policy, and the run helpers.

The arm, camera and policy server are simulated; the runtime, write gate,
binding, adapter, aggregator, shield and skill are the production ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from leapflow.hardware.rgbd import CameraIntrinsics
from leapflow.hardware.skills.builtin.policy import skills as policy_skills
from leapflow.hardware.skills.episodes import EpisodeSettings
from leapflow.hardware.skills.kinematics import DhChain
from leapflow.hardware.skills.protocol import SIDE_EFFECT_NONE, SkillOutcome, SkillPlan
from leapflow.hardware.skills.runtime import SkillRun, SkillRuntime, SkillRuntimeSettings
from tests._harness.skill_stubs import (
    CAMERA_ID,
    PIPER_YAML,
    POLICY_ID,
    WORKCELL_ID,
    WORKCELL_YAML,
    SimArmTransport,
    SimPolicy,
    SimRgbdCamera,
    SkillNodeRegistry,
    embodiment,
    write_workcell,
)

EE = ("ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz")
GRIPPER = "gripper.position"
TASK = "pick up the cube"
TINY_CAMERA = CameraIntrinsics(width=16, height=12, fx=20.0, fy=20.0, ppx=8.0, ppy=6.0)
# Straight down over the table with the Piper DH table (tool cone ~5 deg).
JOINT_START = (0.0, 80.0, -55.0, 0.0, 65.0, 0.0)
_DEFAULT = object()


def piper_chain() -> DhChain:
    declaration = yaml.safe_load(PIPER_YAML.read_text(encoding="utf-8"))
    return DhChain.from_mapping(declaration["kinematics"]["dh"])


def policy_cell(
    tmp_path: Path,
    policy: SimPolicy,
    *,
    arm: SimArmTransport | None = None,
    settings: Mapping[str, float] | None = None,
    decl: Mapping[str, Any] | None | object = _DEFAULT,
    episodes: EpisodeSettings | None = None,
) -> SkillRuntime:
    """A runtime over the cell; *decl* is ``policies.cube`` (default: bind every group)."""
    registry = SkillNodeRegistry(
        arm or SimArmTransport(travel_reads=1),
        store_path=tmp_path / "instrument.duckdb",
        devices={CAMERA_ID: SimRgbdCamera(intrinsics=TINY_CAMERA), POLICY_ID: policy},
    )
    registry.calibrate_cell()
    data: dict[str, Any] = {
        **WORKCELL_YAML,
        "roles": {**WORKCELL_YAML["roles"], "camera": CAMERA_ID},
        "models": {"cube": POLICY_ID},
        "policy": dict(settings or {}),
    }
    binding = embodiment(policy.space, cameras={"top": "camera"}) if decl is _DEFAULT else decl
    if binding is not None:
        data["policies"] = {"cube": binding}
    directory = tmp_path / "workcells"
    write_workcell(directory, data)
    return SkillRuntime(
        registry,
        settings=SkillRuntimeSettings(
            workcells_dir=directory, episodes=episodes or EpisodeSettings()
        ),
        skills=list(policy_skills()),
        poll_s=0.005,
    )


def sim(runtime: SkillRuntime) -> SimArmTransport:
    return runtime._registry.sim  # type: ignore[attr-defined] - the test owns the sim


def run_args(**overrides: Any) -> dict[str, Any]:
    return {"policy": "cube", "task": TASK, "duration_s": 0.6, **overrides}


async def plan_run(runtime: SkillRuntime, **overrides: Any) -> SkillPlan:
    plan = await runtime.plan(WORKCELL_ID, "policy.run", run_args(**overrides))
    assert isinstance(plan, SkillPlan), plan
    return plan


async def refuse_run(runtime: SkillRuntime, **overrides: Any) -> SkillOutcome:
    outcome = await runtime.plan(WORKCELL_ID, "policy.run", run_args(**overrides))
    assert isinstance(outcome, SkillOutcome) and not outcome.ok, outcome
    assert outcome.side_effect_state == SIDE_EFFECT_NONE
    assert not sim(runtime).batches
    return outcome


async def execute_run(runtime: SkillRuntime, plan: SkillPlan) -> SkillOutcome:
    run = await runtime.execute(plan.digest)
    assert isinstance(run, SkillRun), run
    finished = await runtime.wait(run.run_id, timeout_s=10.0)
    assert finished is not None and finished.outcome is not None
    return finished.outcome


def commanded(arm: SimArmTransport, channel_id: str) -> list[float]:
    return [value for batch in arm.batches for cid, value in batch if cid == channel_id]


__all__ = [
    "EE",
    "GRIPPER",
    "JOINT_START",
    "TASK",
    "TINY_CAMERA",
    "commanded",
    "execute_run",
    "piper_chain",
    "plan_run",
    "policy_cell",
    "refuse_run",
    "run_args",
    "sim",
]
