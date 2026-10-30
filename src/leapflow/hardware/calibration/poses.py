# Copyright (c) Alibaba, Inc. and its affiliates.
"""Governed bounded nominal-only acquisition; no autonomous return or corrected model.

The original campaign anchor is an indexed admitted stationary capture, shared
across separately approved runs. Geometry/rate declarations are operator inputs,
not new qualification certificates. All writes remain on GovernedMotion and the
ordinary node gate; raw camera captures remain on the existing registry owner.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Mapping

from leapcalib.contracts.identity import DeviceIdentity, TopologyIdentity, digest_of
from leapcalib.geometry.se3 import Transform
from leapcalib.validation.bundle import EvidenceBundle, Retention
from leapflow.hardware.calibration.acquisition import SettledObservation
from leapflow.hardware.calibration.physical_model import PhysicalModel
from leapflow.hardware.calibration.pose_policy import PoseCollectionPolicy
from leapflow.hardware.calibration.trial import _capture_rows
from leapflow.hardware.skills.kinematics import DhChain
from leapflow.hardware.skills.protocol import (
    FAILURE_CHECK_FAILED,
    FAILURE_INVALID_ARGS,
    FAILURE_PLAN_STALE,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_STOPPED,
    FAILURE_TIMEOUT,
    SkillError,
    SkillPlan,
)
from leapflow.hardware.skills.workcell import SPEED_CHANNEL, joint_channels

SCHEMA = "leapflow.pose_collection.v1"


class WorkcellPoseCollection:
    def __init__(self, collection: Any) -> None:
        self._collection = collection

    def _policy_model(self) -> tuple[PoseCollectionPolicy, PhysicalModel, tuple[str, ...]]:
        collection, ctx = self._collection, self._collection._ctx
        if collection._profile().camera_method != "wrist":
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE, "this milestone supports one wrist camera only"
            )
        arm = collection._registry.context(ctx.workcell.role("arm"))
        if arm is None or arm.kinematics is None:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, "declared nominal raw DH model required")
        try:
            policy = PoseCollectionPolicy.from_mapping(ctx.workcell.calibration_motion)
            chain = DhChain.from_mapping(arm.kinematics.dh)
            tool = ctx.frames.transform("tcp")
            si_tool = (
                Transform()
                if tool is None
                else Transform(tool.rotation, tuple(v / 1000 for v in tool.translation))
            )
            model = PhysicalModel(
                chain,
                (0.0,) * chain.dof,
                si_tool,
                digest_of(chain.to_dict()),
                digest_of(ctx.workcell.to_dict()),
                "declared_nominal_acquisition_not_qualified",
            )
            policy.validate_model(
                model,
                collection._profile().settled,
                fk_mm=ctx.limits["fk_tolerance_mm"],
                fk_deg=ctx.limits["fk_tolerance_deg"],
            )
            if (
                arm.safety
                and arm.safety.max_velocity_rad_s > 0
                and any(r > arm.safety.max_velocity_rad_s for r in policy.joint_rate_bounds_rad_s)
            ):
                raise ValueError("declared acquisition rates exceed tighter live device safety")
        except (KeyError, TypeError, ValueError) as exc:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                f"declare verified calibration_motion inputs before acquisition: {exc}",
            ) from exc
        return policy, model, joint_channels(arm)

    async def _reference(self, policy: PoseCollectionPolicy) -> tuple[SettledObservation, str]:
        collection, registry = self._collection, self._collection._registry
        store, keys = registry.calibration_evidence, registry.approval_keys
        if store is None or keys is None:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                "indexed campaign reference and historical owner public keys required",
            )
        content = await store.read_capture(
            policy.reference_acquisition, workcell_id=collection._ctx.workcell.workcell_id
        )
        capture = SettledObservation.from_mapping(content["observation"])
        if content["binding"]["node_session_id"] != collection._node_session or content[
            "profile"
        ] != vars(collection._profile().settled):
            raise ValueError(
                "campaign reference node clock or settled uncertainty declaration changed"
            )
        model = collection._feedback(collection._profile()).model
        rows = _capture_rows(capture, model)
        topology = TopologyIdentity(
            collection._ctx.workcell.workcell_id,
            DeviceIdentity(capture.feedback[0].device_id),
            model,
            (capture.roi.profile,),
            (policy.mount,),
        )
        bundle = EvidenceBundle(
            "pose-campaign-reference:" + policy.reference_acquisition,
            topology,
            rows,
            time.time(),
            Retention(collection._profile().ttl_s),
            metadata={"node_session_id": collection._node_session},
        )
        await store.source_manifest(
            bundle,
            dict.fromkeys((r.observation_id for r in rows), policy.reference_acquisition),
            trusted_keys=keys[0],
            node_id=keys[1],
        )
        return capture, digest_of(content)

    async def _state(self, policy: PoseCollectionPolicy) -> Any:
        collection, ctx = self._collection, self._collection._ctx
        if collection._blocked():
            raise SkillError(
                FAILURE_STOPPED,
                "pose acquisition blocked; inspect motion/dispatch before another run",
            )
        # Health/coherence/sample age are checked by the actual raw source.
        await collection._feedback(collection._profile()).sample()
        state = await ctx.motion.state()
        ctx.motion.check_kinematics(state)
        batch = await collection._registry.read_batch(ctx.workcell.role("arm"), (SPEED_CHANNEL,))
        if batch.device_id != ctx.workcell.role("arm") or len(batch.readings) != 1:
            raise ValueError("same-owner live speed feedback required")
        reading = batch.readings[0]
        age = time.monotonic() - reading.monotonic_at
        if (
            reading.device_id != batch.device_id
            or reading.channel_id != SPEED_CHANNEL
            or reading.unit != "percent"
            or not reading.is_trustworthy
            or not 0 <= age <= collection._profile().settled.max_feedback_age_s
        ):
            raise ValueError("fresh trustworthy percent speed feedback required")
        if (
            isinstance(reading.value, bool)
            or not isinstance(reading.value, (int, float))
            or not math.isfinite(reading.value)
            or not 0 < reading.value <= policy.speed_pct_max
        ):
            raise ValueError(
                "live speed exceeds campaign cap; request a separately approved speed setup"
            )
        return state

    def _arguments(
        self, args: Mapping[str, Any], policy: PoseCollectionPolicy, joints: tuple[str, ...]
    ) -> tuple[list[dict[str, float]], float, float]:
        poses = args.get("poses")
        if not isinstance(poses, (list, tuple)) or not 2 <= len(poses) <= policy.max_moves:
            raise SkillError(FAILURE_INVALID_ARGS, "two to eight bounded poses required")
        commands = []
        for pose in poses:
            if not isinstance(pose, Mapping) or set(pose) != set(joints):
                raise SkillError(
                    FAILURE_INVALID_ARGS,
                    "each approved pose must name every raw joint exactly once",
                )
            if any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in pose.values()
            ):
                raise SkillError(FAILURE_INVALID_ARGS, "finite raw degree joint targets required")
            commands.append({j: float(pose[j]) for j in joints})
        speed = args.get("speed_pct", 10)
        if (
            isinstance(speed, bool)
            or not isinstance(speed, (int, float))
            or not math.isfinite(speed)
            or not 0
            < speed
            <= min(policy.speed_pct_max, self._collection._ctx.limits["speed_pct_max"])
        ):
            raise SkillError(
                FAILURE_INVALID_ARGS, "requested acquisition speed exceeds declared ten-percent cap"
            )
        settled = self._collection._profile().settled
        dwell = 0.3 + 2 * (
            settled.receipt_uncertainty_s
            + settled.exposure_uncertainty_s
            + settled.feedback_uncertainty_s
        )
        duration = (
            len(commands)
            * (self._collection._ctx.limits["move_timeout_s"] + 2 * settled.timeout_s + dwell)
            + 10
        )
        if duration > min(policy.max_duration_s, self._collection._ctx.limits["max_duration_s"]):
            raise SkillError(
                FAILURE_INVALID_ARGS,
                "moves, separated captures and validation cannot fit frozen acquisition deadline",
            )
        return commands, dwell, duration

    async def _sweep(
        self,
        policy: PoseCollectionPolicy,
        model: PhysicalModel,
        reference: tuple[float, ...],
        before: Mapping[str, float],
        target: Mapping[str, float],
        joints: tuple[str, ...],
    ) -> tuple[float, float]:
        # Bounded pure geometry is drained on cancellation, preventing orphaned
        # compute and overlap with another requesting run after lease release.
        task = asyncio.create_task(
            asyncio.to_thread(
                policy.sweep,
                model,
                reference,
                tuple(before[j] for j in joints),
                tuple(target[j] for j in joints),
            )
        )
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
            except ValueError:
                break
        result = task.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        collection = self._collection
        try:
            policy, model, joints = self._policy_model()
            commands, dwell, duration = self._arguments(args, policy, joints)
            capture, reference_digest = await self._reference(policy)
            reference = tuple(map(math.degrees, capture.feedback[-2].joints_rad))
            state = await self._state(policy)
            current = dict(state.joints)
            targets, travel, joint_l1 = [], 0.0, 0.0
            for command in commands:
                target = collection._ctx.motion.check_joints(command, current)
                distance, angle = await self._sweep(
                    policy, model, reference, current, command, joints
                )
                travel, joint_l1 = travel + distance, joint_l1 + angle
                targets.append(dict(vars(target)))
                current = command
            if travel > policy.max_travel_mm or joint_l1 > policy.max_joint_l1_deg:
                raise ValueError("planned acquisition exceeds cumulative travel or joint L1 budget")
            basis = dict(
                await collection.collection_basis(
                    {"region": args["region"], "frames": 2 * len(commands)}
                )
            )
            if (
                basis["mount_digest"] != policy.mount.digest
                or basis["camera_session_id"] != capture.roi.session_id
                or basis["metric_profile_digest"] != capture.roi.profile.digest
            ):
                raise ValueError(
                    "camera owner, mount or metric profile differs from immutable campaign reference"
                )
            basis["commands"] = commands
            return {
                "collection": basis,
                "poses": {
                    "schema": SCHEMA,
                    "policy": policy.to_dict(),
                    "nominal_model": model.to_dict(),
                    "reference_digest": reference_digest,
                    "reference_deg": list(reference),
                    "start_joints": dict(state.joints),
                    "targets": targets,
                    "dwell_s": dwell,
                    "duration_s": duration,
                    "travel_bound_mm": travel,
                    "joint_l1_deg": joint_l1,
                    "auto_return": False,
                    "activation": "none",
                },
            }
        except ValueError as exc:
            raise SkillError(
                FAILURE_CHECK_FAILED, f"pose collection planning refused: {exc}"
            ) from exc
        except TimeoutError as exc:
            raise SkillError(FAILURE_TIMEOUT, "pose collection planning deadline expired") from exc

    async def execute(self, plan: SkillPlan) -> Mapping[str, Any]:
        try:
            return await self._execute(plan)
        except ValueError as exc:
            raise SkillError(
                FAILURE_CHECK_FAILED, f"pose acquisition stopped; no automatic return/replay: {exc}"
            ) from exc

    async def _execute(self, plan: SkillPlan) -> Mapping[str, Any]:
        collection, ctx = self._collection, self._collection._ctx
        policy, model, joints = self._policy_model()
        commands, dwell, duration = self._arguments(plan.args, policy, joints)
        sealed = plan.basis.get("poses", {})
        if (
            sealed.get("schema") != SCHEMA
            or sealed.get("policy") != policy.to_dict()
            or sealed.get("nominal_model") != model.to_dict()
            or sealed.get("dwell_s") != dwell
            or sealed.get("duration_s") != duration
        ):
            raise SkillError(
                FAILURE_PLAN_STALE,
                "campaign policy, nominal model or duration changed since approval",
            )
        capture, reference_digest = await self._reference(policy)
        reference = tuple(map(math.degrees, capture.feedback[-2].joints_rad))
        if (
            reference_digest != sealed["reference_digest"]
            or list(reference) != sealed["reference_deg"]
        ):
            raise SkillError(FAILURE_PLAN_STALE, "original campaign reference changed; no motion")
        state = await self._state(policy)
        if set(state.joints) != set(joints) or any(
            abs(state.joints[j] - sealed["start_joints"][j])
            > math.degrees(collection._profile().settled.max_joint_drift_rad)
            for j in joints
        ):
            raise SkillError(
                FAILURE_PLAN_STALE, "raw arm posture changed since approval; plan again"
            )
        recorded, _, _ = await collection._collection_source(
            plan, commands=commands, frames=2 * len(commands)
        )
        captures, travel, joint_l1 = [], 0.0, 0.0
        for command in commands:
            state = await self._state(policy)
            distance, angle = await self._sweep(
                policy, model, reference, state.joints, command, joints
            )
            if (
                travel + distance > policy.max_travel_mm
                or joint_l1 + angle > policy.max_joint_l1_deg
            ):
                raise ValueError(
                    "measured starting state exhausts cumulative run travel/joint budget"
                )
            fresh = await self._state(policy)
            if any(
                abs(fresh.joints[j] - state.joints[j])
                > math.degrees(collection._profile().settled.max_joint_drift_rad)
                for j in joints
            ):
                raise ValueError("raw state changed during swept-path validation; no dispatch")
            if collection._blocked():
                raise SkillError(
                    FAILURE_STOPPED, "acquisition stopped before dispatch; no return motion"
                )
            await ctx.motion.move_joints(command, speed_pct=plan.args.get("speed_pct", 10))
            travel, joint_l1 = travel + distance, joint_l1 + angle
            await self._state(policy)
            per_pose = []
            for index in range(2):
                if index:
                    await ctx.motion.pause(dwell)
                observation = await recorded.capture()
                measured = tuple(map(math.degrees, observation.feedback[-2].joints_rad))
                await self._sweep(
                    policy,
                    model,
                    reference,
                    dict(zip(joints, measured)),
                    dict(zip(joints, measured)),
                    joints,
                )
                if any(
                    abs(measured[i] - command[j])
                    > math.degrees(collection._profile().settled.max_joint_drift_rad)
                    for i, j in enumerate(joints)
                ):
                    raise ValueError("settled capture is not at the approved raw joint endpoint")
                per_pose.append(recorded.last_record)
            captures.append(per_pose)
        return {
            "schema": SCHEMA,
            "captures_by_pose": captures,
            "factor_pairs": [
                [captures[i][-1]["evidence_digest"], captures[i + 1][0]["evidence_digest"]]
                for i in range(len(captures) - 1)
            ],
            "reference_digest": reference_digest,
            "commands": commands,
            "travel_bound_mm": travel,
            "joint_l1_deg": joint_l1,
            "auto_return": False,
            "activation": "none",
            "raw_exported": False,
            "physical_admission": "not_established",
            "hardware_accuracy_certified": False,
        }
