"""Pure declared-model previews and bounded, monitored governed execution."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from dataclasses import asdict
from typing import Any

from leapflow.engine.recovery.failure_envelope import SideEffectState
from leapflow.hardware.context import HardwareContext
from leapflow.hardware.skills.adapt.ik import solve_ik
from leapflow.hardware.skills.adapt.spec import IkSettings
from leapflow.hardware.skills.geometry import Pose, Transform
from leapflow.hardware.skills.kinematics import DhChain

from .config import Settings
from .evidence import Refusal, json_default
from .health import JOINTS, POSE, Snapshot, number
from .session import Session, unverified_digest


def digest_plan(plan: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"),
                                    default=json_default, allow_nan=False).encode()).hexdigest()


def bounds(context: HardwareContext, channel_id: str) -> tuple[float, float]:
    channel = context.channel(channel_id)
    if channel is None or not channel.envelope.declared:
        raise Refusal(f"missing declared envelope: {channel_id}")
    return number(channel.envelope.min_value), number(channel.envelope.max_value)


def limits(context: HardwareContext, settings: Settings) -> tuple[tuple[float, float], ...]:
    kin = context.kinematics
    if kin is None or kin.dof != 6 or len(kin.joint_limits) != 6:
        raise Refusal("six-axis declared kinematics required")
    result = []
    for channel, manual, radians in zip(JOINTS, settings.manual_joint_limits_deg, kin.joint_limits):
        declared = bounds(context, channel)
        low = max(declared[0], manual[0], math.degrees(number(radians[0]))) + settings.joint_margin_deg
        high = min(declared[1], manual[1], math.degrees(number(radians[1]))) - settings.joint_margin_deg
        if low >= high:
            raise Refusal("manual/SDK/declaration intersection is empty")
        result.append((low, high))
    return tuple(result)


def chain_for(context: HardwareContext, snapshot: Snapshot, settings: Settings) -> DhChain:
    if context.kinematics is None or not context.kinematics.dh:
        raise Refusal("declared DH table is missing")
    chain = DhChain.from_mapping(context.kinematics.dh)
    mm, deg = chain.agreement(snapshot.joints, snapshot.pose)
    if mm > settings.position_mm or deg > settings.orientation_deg:
        raise Refusal("live flange disagrees with declared FK; no predicted motion")
    return chain


def opening_max(context: HardwareContext, settings: Settings) -> float:
    """Require the fitted specification, usable-width config and evidence to agree."""
    declared_low, declared_high = bounds(context, "gripper.position")
    fitted, usable, chosen = settings.gripper_spec_max_mm, settings.gripper_usable_max_mm, settings.gripper_max_mm
    if any(v is None for v in (fitted, usable, chosen)) or not settings.gripper_max_evidence.strip():
        raise Refusal("maximum opening unresolved: supply fitted spec, validated usable width, "
                      "selected maximum and evidence; nominal 100 mm is not a fitted maximum")
    fitted, usable, chosen = number(fitted), number(usable), number(chosen)
    if not declared_low < usable <= min(fitted, declared_high) or chosen != usable:
        raise Refusal("fitted-gripper/declaration/usable-maximum contradiction")
    return chosen


def tolerance(context: HardwareContext, channels: tuple[str, ...], fallback: float) -> float:
    values = [fallback]
    for cid in channels:
        channel = context.channel(cid)
        if channel is None or channel.envelope.tolerance is None:
            raise Refusal(f"missing declared arrival tolerance: {cid}")
        value = number(channel.envelope.tolerance)
        if value <= 0:
            raise Refusal("arrival tolerance must be positive")
        values.append(value)
    return min(values)


def initialization(context: HardwareContext, start: Snapshot, settings: Settings) -> tuple[float, ...]:
    safe = limits(context, settings)
    candidate = [min(max(q, low), high) for q, (low, high) in zip(start.joints, safe)]
    candidate[1], candidate[2] = 5.0, -5.0
    if any(not low <= q <= high for q, (low, high) in zip(candidate, safe)):
        raise Refusal("nearby initialization is outside the conservative intervals")
    if max(abs(q - p) for q, p in zip(candidate, start.joints)) > settings.init_max_deg:
        raise Refusal("initialization exceeds the approved nearby-posture budget")
    return tuple(candidate)


def step(label: str, commands: tuple[tuple[str, Any], ...], *, configure: bool = False,
         joints: tuple[float, ...] | None = None, pose: Pose | None = None,
         grip: float | None = None) -> dict[str, Any]:
    return {"label": label, "commands": commands, "configure": configure,
            "joints": joints, "pose": asdict(pose) if pose else None, "grip": grip}


def _pose_check(context: HardwareContext, pose: Pose) -> None:
    for cid, value in zip(POSE, pose.as_tuple()):
        low, high = bounds(context, cid)
        if not low <= value <= high:
            raise Refusal(f"waypoint outside declared workspace: {cid}")


def _segment(context: HardwareContext, chain: DhChain, origin: Pose, destination: Pose,
             seed: tuple[float, ...], settings: Settings, label: str) -> tuple[list[dict[str, Any]], tuple[float, ...]]:
    delta = tuple(b - a for a, b in zip(origin.position, destination.position))
    count = max(1, math.ceil(origin.distance_mm(destination) / settings.waypoint_mm))
    safe = limits(context, settings)
    ik = IkSettings(tolerance_mm=min(0.1, settings.position_mm),
                    tolerance_deg=min(0.1, settings.orientation_deg), max_step_deg=settings.joint_step_deg)
    rows = []
    previous = origin
    for index in range(1, count + 1):
        target = origin.translated(*(d * index / count for d in delta))
        _pose_check(context, target)
        # Solve midpoint and endpoint in order: never seed a return from an unrelated posture.
        midpoint = previous.translated(*(d / count / 2 for d in delta))
        try:
            mid_q = solve_ik(chain, Transform.from_pose(midpoint), seed, safe, ik)
            q = solve_ik(chain, Transform.from_pose(target), mid_q, safe, ik)
        except ValueError as exc:
            raise Refusal(f"{label} waypoint {index} has no conservative continuous IK solution") from exc
        max_jump = max(abs(a - b) for a, b in zip(q, seed))
        if max_jump > settings.joint_step_deg:
            raise Refusal("IK branch jump exceeds configured waypoint joint budget")
        for cid, a, b in zip(JOINTS, seed, q):
            channel = context.channel(cid)
            if channel is None or channel.envelope.max_jump is None or abs(a - b) > channel.envelope.max_jump:
                raise Refusal("waypoint exceeds a declared joint jump")
        rows.append(step(f"{label}.{index:03d}", tuple(zip(POSE, target.as_tuple())), joints=q, pose=target))
        seed, previous = q, target
    return rows, seed


def preview(context: HardwareContext, start: Snapshot, settings: Settings, *, operation: str,
            dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
            width: float | None = None) -> dict[str, Any]:
    """A complete immutable-by-hash request, including return and safety configuration."""
    settings.validate()
    dx, dy, dz = number(dx), number(dy), number(dz)
    safe = limits(context, settings)
    chain = chain_for(context, start, settings)
    rows = [step("speed", (("arm.speed_pct", settings.speed_pct),), configure=True)]
    origin_q, origin_pose = start.joints, start.pose
    swept = []
    if operation in ("initialize", "verify"):
        origin_q = initialization(context, start, settings)
        origin_pose = chain.forward(origin_q)
        _pose_check(context, origin_pose)
        for i in range(21):
            q = tuple(a + (b - a) * i / 20 for a, b in zip(start.joints, origin_q))
            swept.append({"joints": q, "pose": asdict(chain.forward(q))})
        rows.append(step("enable", (("arm.enable", 1),), configure=True))
        rows.append(step("initialize", tuple(zip(JOINTS, origin_q)), joints=origin_q, pose=origin_pose,
                         grip=start.gripper_mm))
    else:
        if not start.enabled or start.status.get("ctrl_mode") != 1 or start.status.get("arm_status") != 0:
            raise Refusal("movement requires healthy enabled CAN state; preview arm initialize first")
        if any(not low <= q <= high for q, (low, high) in zip(origin_q, safe)):
            raise Refusal("starting posture is outside conservative planning limits")
    maximum = None
    if operation in ("verify", "gripper"):
        maximum = opening_max(context, settings)
        if not 0 <= start.gripper_mm <= maximum:
            raise Refusal("starting opening conflicts with fitted usable range")
    seed = origin_q
    if operation == "verify":
        if dx != 0 or dy <= 0 or dz <= 0:
            raise Refusal("verification is base +Z then +Y with positive distances and no X")
        rows.append(step("linear_path", (("arm.pose_path", "linear"),), configure=True))
        lifted = origin_pose.translated(dz=dz)
        left = lifted.translated(dy=dy)
        path, seed = _segment(context, chain, origin_pose, lifted, seed, settings, "lift")
        rows.extend(path)
        path, seed = _segment(context, chain, lifted, left, seed, settings, "left")
        rows.extend(path)
        rows.extend([step("gripper_max", (("gripper.position", maximum),), joints=seed, pose=left, grip=maximum),
                     step("gripper_restore", (("gripper.position", start.gripper_mm),),
                          joints=seed, pose=left, grip=start.gripper_mm)])
        path, seed = _segment(context, chain, left, lifted, seed, settings, "return_y")
        rows.extend(path)
        path, seed = _segment(context, chain, lifted, origin_pose, seed, settings, "return_z")
        rows.extend(path)
        if max(abs(a - b) for a, b in zip(seed, origin_q)) > tolerance(context, JOINTS, settings.joint_deg):
            raise Refusal("return IK does not recover the origin joint branch")
    elif operation == "move":
        if dx == dy == dz == 0:
            raise Refusal("move needs a nonzero displacement")
        rows.append(step("linear_path", (("arm.pose_path", "linear"),), configure=True))
        path, seed = _segment(context, chain, origin_pose, origin_pose.translated(dx, dy, dz),
                              seed, settings, "move")
        rows.extend(path)
    elif operation == "gripper":
        target = maximum if width is None else number(width)
        if not 0 <= target <= maximum:
            raise Refusal("requested opening exceeds fitted maximum")
        rows.append(step("gripper", (("gripper.position", target),), joints=seed, pose=origin_pose, grip=target))
    elif operation != "initialize":
        raise Refusal("unknown motion operation")
    # Bind effective arrival tolerances to the same preview as every numeric target.
    arrival = {"joint": tolerance(context, JOINTS, settings.joint_deg),
               "position": tolerance(context, POSE[:3], settings.position_mm),
               "orientation": tolerance(context, POSE[3:], settings.orientation_deg),
               "gripper": tolerance(context, ("gripper.position",), settings.grip_mm)}
    return {"version": 1, "operation": operation, "declaration_digest": unverified_digest(context),
            "settings": asdict(settings), "start": {"joints": start.joints, "pose": asdict(start.pose),
                                                    "gripper_mm": start.gripper_mm},
            "origin_joints": origin_q, "origin_pose": asdict(origin_pose), "gripper_max_mm": maximum,
            "delta_base_mm": {"x": dx, "y": dy, "z": dz}, "finish": "enabled",
            "arrival": arrival, "initialization_sweep": swept, "steps": rows,
            "clearance": "Operator must confirm unobstructed swept volume, empty jaws and accessible physical emergency stop"}


def check_start(plan: dict[str, Any], current: Snapshot) -> None:
    arrival, start = plan["arrival"], plan["start"]
    pose = Pose.from_mapping(start["pose"])
    if (max(abs(a - b) for a, b in zip(current.joints, start["joints"])) > arrival["joint"] or
            pose.distance_mm(current.pose) > arrival["position"] or
            pose.rotation_to_deg(current.pose) > arrival["orientation"] or
            abs(current.gripper_mm - start["gripper_mm"]) > arrival["gripper"]):
        raise Refusal("starting state drifted since preview; obtain a new preview and approval")


async def wait_arrived(session: Session, row: dict[str, Any], before: Snapshot,
                       arrival: dict[str, float]) -> Snapshot:
    settings = session.settings
    deadline, stable = time.monotonic() + settings.move_timeout_s, 0
    last = before.receipts
    target = Pose.from_mapping(row["pose"]) if row.get("pose") else None
    q = row.get("joints")
    while time.monotonic() < deadline:
        current = await session.snapshot(ready=True)
        fresh = all(a > b for a, b in zip(current.receipts, last))
        if not fresh:
            stable = 0
            await asyncio.sleep(settings.poll_s)
            continue
        last = current.receipts
        if q is not None and any(not min(a, b) - settings.deviation_deg <= c <= max(a, b) + settings.deviation_deg
                                 for a, b, c in zip(before.joints, q, current.joints)):
            raise Refusal("observed joint deviation exceeds planned segment", state=SideEffectState.UNKNOWN)
        # The initialization intentionally changes orientation; translations and jaw moves must not.
        if target is not None and row["label"] != "initialize":
            if target.rotation_to_deg(current.pose) > arrival["orientation"]:
                raise Refusal("orientation changed during translation/gripper operation", state=SideEffectState.UNKNOWN)
            a, b, p = before.pose.position, target.position, current.pose.position
            d = tuple(y - x for x, y in zip(a, b))
            length2 = sum(x * x for x in d)
            fraction = min(1.0, max(0.0, sum((z-x)*y for x, y, z in zip(a, d, p)) / length2)) if length2 else 0
            miss = math.dist(p, tuple(x + fraction*y for x, y in zip(a, d)))
            if miss > arrival["position"]:
                raise Refusal("flange departed the preflighted line", state=SideEffectState.UNKNOWN)
        arrived = current.status.get("motion_status") == 0
        if q is not None:
            arrived &= max(abs(a - b) for a, b in zip(q, current.joints)) <= arrival["joint"]
        if target is not None:
            arrived &= target.distance_mm(current.pose) <= arrival["position"]
            arrived &= target.rotation_to_deg(current.pose) <= arrival["orientation"]
        if row.get("grip") is not None:
            arrived &= abs(current.gripper_mm - row["grip"]) <= arrival["gripper"]
        stable = stable + 1 if arrived else 0
        if stable >= settings.settled_samples:
            session.evidence.record(row["label"] + ".settled", snapshot=current.to_dict(), fresh_samples=stable)
            return current
        await asyncio.sleep(settings.poll_s)
    raise Refusal("arrival deadline expired; halt and inspect, do not replay", state=SideEffectState.UNKNOWN)


async def execute_plan(session: Session, plan: dict[str, Any]) -> Snapshot:
    check_start(plan, await session.snapshot(preflight=True))
    if not session.context or plan["declaration_digest"] != unverified_digest(session.context):
        raise Refusal("preview declaration changed")
    if digest_plan(plan["settings"]) != digest_plan(asdict(session.settings)):
        raise Refusal("effective settings differ from approved preview")
    session.bind_steps(plan["steps"])
    session.evidence.record("approved_plan", plan=plan, approval_note=session.approval_note)
    before = await session.snapshot(preflight=True)
    origin = None
    for row in plan["steps"]:
        await session.command(row["label"], tuple((c, v) for c, v in row["commands"]), configure=row["configure"])
        if row["label"] == "enable":
            before = await wait_arrived(session, row, before, plan["arrival"])
        elif row.get("joints") is not None:
            before = await wait_arrived(session, row, before, plan["arrival"])
            if row["label"] == "initialize":
                origin = before
                session.evidence.record("return_origin", snapshot=origin.to_dict())
    final = await session.snapshot(ready=True)
    if plan["operation"] == "verify":
        if origin is None:
            raise Refusal("settled return origin not recorded", state=SideEffectState.UNKNOWN)
        check_start({"arrival": plan["arrival"], "start": {"joints": origin.joints,
                    "pose": asdict(origin.pose), "gripper_mm": origin.gripper_mm}}, final)
    session.evidence.record("final_enabled", snapshot=final.to_dict())
    session.success = True
    return final
