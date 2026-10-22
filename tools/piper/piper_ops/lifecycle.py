"""Arm lifecycle is distinct from node service lifecycle; halt never drops torque."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from .evidence import Refusal
from .motion import check_start, tolerance, wait_arrived
from .session import Session, unverified_digest
from .health import JOINTS, POSE


async def preview(session: Session, operation: str, *, torque_release: bool = False) -> dict[str, Any]:
    assert session.context is not None
    if operation == "stop":
        return {"version": 1, "operation": "stop", "settings": asdict(session.settings),
                "declaration_digest": unverified_digest(session.context), "steps": [],
                "warning": "Damped halt; does not release torque; feedback may be unqualified"}
    start = await session.snapshot(preflight=True)
    if operation == "disable" and not torque_release:
        raise Refusal("disable releases torque and may drop the arm; explicit --torque-release-confirmed required")
    if operation not in ("enable", "resume", "disable", "stop"):
        raise Refusal("unknown arm lifecycle operation")
    commands = (("arm.enable", 0 if operation == "disable" else 1),)
    return {"version": 1, "operation": operation, "declaration_digest": unverified_digest(session.context),
            "settings": asdict(session.settings),
            "start": {"joints": start.joints, "pose": asdict(start.pose), "gripper_mm": start.gripper_mm},
            "arrival": {"joint": tolerance(session.context, JOINTS, session.settings.joint_deg),
                        "position": tolerance(session.context, POSE[:3], session.settings.position_mm),
                        "orientation": tolerance(session.context, POSE[3:], session.settings.orientation_deg),
                        "gripper": tolerance(session.context, ("gripper.position",), session.settings.grip_mm)},
            "steps": [] if operation == "stop" else [{"label": operation, "commands": commands, "configure": True}],
            "warning": "Torque release can drop the arm" if operation == "disable" else "Enable/resume can energize the arm"}


async def execute(session: Session, plan: dict[str, Any], *, torque_release: bool = False) -> dict[str, Any]:
    operation = plan["operation"]
    if operation == "stop":
        # Never make a protective halt depend on healthy motion telemetry.
        await session.stop()
        session.success = True
        return {"ok": True, "operation": "stop", "halt_delivered": True,
                "feedback_verified": False}
    if operation not in ("enable", "resume", "disable"):
        raise Refusal("unknown arm lifecycle operation")
    if operation == "disable" and not torque_release:
        raise Refusal("explicit torque-release confirmation required")
    before = await session.snapshot(preflight=True)
    check_start(plan, before)
    session.bind_steps(plan["steps"])
    row = plan["steps"][0]
    await session.command(row["label"], tuple((c, v) for c, v in row["commands"]), configure=True)
    if operation in ("enable", "resume"):
        final = await wait_arrived(session, row, before, plan["arrival"])
    else:
        async def disabled() -> Any:
            while True:
                state = await session.snapshot()
                if all(m["enabled"] is False for m in state.telemetry["motors"]):
                    return state
                await asyncio.sleep(session.settings.poll_s)
        final = await asyncio.wait_for(disabled(), session.settings.move_timeout_s)
    session.evidence.record("lifecycle_verified", snapshot=final.to_dict())
    session.success = True
    return {"ok": True, "operation": operation, "snapshot": final.to_dict()}
