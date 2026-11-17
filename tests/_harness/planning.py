# Copyright (c) Alibaba, Inc. and its affiliates.
"""Calibrated simulated placement using the real registry, transport, and skill discovery."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import AsyncIterator

from leapflow.hardware.calibration_store import CalibrationRecord
from leapflow.hardware.context import HardwareContext
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.skills.kinematics import DhChain
from leapflow.hardware.skills.planning import PlanningSettings
from leapflow.hardware.skills.runtime import SkillRuntime, SkillRuntimeSettings
from tests._harness.skill_stubs import (
    ARM_ID, JOINTS, SAFE_BOX, TOOL_LENGTH_MM, WORKCELL_ID, piper_context, write_workcell,
)

SAFE_JOINTS = (0.0, 80.0, -55.0, 69.99, 10.0, 0.0)


def simulated_arm() -> HardwareContext:
    """Declaration-driven feedback; initial flange and joints agree with published FK."""
    arm = piper_context()
    pose = DhChain.from_mapping(arm.kinematics.dh).forward(SAFE_JOINTS)
    values = {**dict(zip(JOINTS, SAFE_JOINTS)),
              **dict(zip(("ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz"), pose.as_tuple())),
              "gripper.position": 60.0, "arm.enabled": True, "arm.motion": True}
    return replace(
        arm, channels=tuple(replace(channel, sample_rate_hz=0) for channel in arm.channels),
        transport=replace(arm.transport, kind="simulated", config={"values": values}),
        provenance=replace(arm.provenance, verified_by="offline-planning-fixture"),
    )


@asynccontextmanager
async def placement(root: Path, *, workers: int = 2) -> AsyncIterator[tuple[HardwareRegistry, SkillRuntime]]:
    """Own real persistence and spawned calculations; never open physical drivers."""
    root.mkdir(parents=True, exist_ok=True)
    registry = HardwareRegistry(HardwareSettings(
        enabled=True, instrument_db_path=str(root / "instrument.duckdb"),
        stream_enabled=False, require_describe_before_write=False,
    ), providers=())
    runtime = None
    try:
        registry.register_external_contexts("offline_simulation", (simulated_arm(),))
        report = registry.load()
        assert report.admitted == (ARM_ID,), report.to_dict()
        store = registry.calibration_store
        assert store is not None
        for procedure, parameters, pose in (
            ("tcp.offset", {}, {"x": 0, "y": 0, "z": TOOL_LENGTH_MM}),
            ("workspace.safe_box", dict(SAFE_BOX), None),
        ):
            assert store.record(CalibrationRecord(
                device_id=WORKCELL_ID, procedure_id=procedure, recorded_at=time.time(),
                parameters=parameters, pose=pose,
            ))
        write_workcell(root / "workcells")
        await registry.transport(ARM_ID)
        runtime = SkillRuntime(registry, settings=SkillRuntimeSettings(
            workcells_dir=root / "workcells", planning=PlanningSettings(workers=workers),
        ), poll_s=0.005)
        yield registry, runtime
    finally:
        if runtime is not None:
            await runtime.close()
        await registry.close_all()
