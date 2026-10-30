# Copyright (c) Alibaba, Inc. and its affiliates.
"""Plan-bound indexed camera trial behind the existing workcell owner."""

from __future__ import annotations

from typing import Any, Mapping

from leapcalib.contracts.failures import CalibrationFailure
from leapcalib.contracts.identity import digest_of
from leapflow.hardware.calibration.trial import SCHEMA, TrialPolicy, run_motion_trial
from leapflow.hardware.skills.protocol import (
    FAILURE_CHECK_FAILED,
    FAILURE_INVALID_ARGS,
    FAILURE_PLAN_STALE,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_TIMEOUT,
    SkillError,
    SkillPlan,
)


class WorkcellTrial:
    def __init__(self, collection: Any) -> None:
        self._collection = collection

    def _resources(self) -> tuple[Any, Any, Any]:
        registry = self._collection._registry
        worker = self._collection._worker()
        store, keys = registry.calibration_evidence, registry.approval_keys
        if store is None or keys is None:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                "indexed acquisition store and owner historical public keys required",
            )
        return store, worker, keys

    def _basis(self, args: Mapping[str, Any]) -> tuple[dict[str, Any], TrialPolicy]:
        collection, ctx = self._collection, self._collection._ctx
        _, worker, keys = self._resources()
        profile = collection._profile()
        try:
            policy = TrialPolicy.from_mapping(args, worker.budgets)
        except (KeyError, TypeError, ValueError) as exc:
            raise SkillError(FAILURE_INVALID_ARGS, f"camera trial policy refused: {exc}") from exc
        if policy.mount.parent_frame != ("flange" if profile.camera_method == "wrist" else "base"):
            raise SkillError(
                FAILURE_INVALID_ARGS, "trial mount parent disagrees with declared camera method"
            )
        duration = (
            worker.budgets.startup_s
            + len(policy.pairs) * worker.budgets.update_s
            + worker.budgets.solve_s
            + 10
        )
        if duration > min(120, ctx.limits["max_duration_s"]):
            raise SkillError(
                FAILURE_INVALID_ARGS,
                "trial tracking, solve and validation cannot fit approved run deadline",
            )
        arm = collection._feedback(profile)
        return {
            "schema": SCHEMA,
            "policy": policy.to_dict(),
            "profile": profile.to_dict(),
            "model_digest": arm.model.digest,
            "workcell_digest": digest_of(ctx.workcell.to_dict()),
            "node_session_id": collection._node_session,
            "worker_budgets": vars(worker.budgets),
            "node_id": keys[1],
            "key_ids": sorted(keys[0]),
            "commands": [],
            "activation": "none",
        }, policy

    async def basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        basis, policy = self._basis(args)
        store, _, _ = self._resources()
        # Reopen unexpired raw bytes at planning as well as execution. Durable
        # summaries alone cannot establish that the trial is currently runnable.
        try:
            for pair in policy.pairs:
                for identity in pair:
                    content = await store.read_capture(
                        identity, workcell_id=self._collection._ctx.workcell.workcell_id
                    )
                    if content["binding"]["node_session_id"] != basis["node_session_id"]:
                        raise ValueError(
                            "acquisition belongs to another node boot; receipt clocks cannot be mixed"
                        )
                    capture = content["observation"]
                    if capture["roi"]["mount_digest"] != policy.mount.digest:
                        raise ValueError("trial mount identity differs from recorded camera mount")
        except ValueError as exc:
            raise SkillError(FAILURE_CHECK_FAILED, f"trial acquisition unavailable: {exc}") from exc
        return basis

    async def execute(self, plan: SkillPlan) -> Mapping[str, Any]:
        basis, policy = self._basis(plan.args)
        if plan.basis.get("trial") != basis:
            raise SkillError(
                FAILURE_PLAN_STALE,
                "trial owner, model, split or compute policy changed since approval",
            )
        store, worker, keys = self._resources()
        collection = self._collection
        try:
            return await run_motion_trial(
                store,
                worker,
                policy=policy,
                profile=collection._profile(),
                model=collection._feedback(collection._profile()).model,
                workcell_id=plan.workcell,
                run_id=collection._ctx.run_id,
                node_session_id=collection._node_session,
                approval_keys=keys,
                blocked=collection._blocked,
            )
        except (CalibrationFailure, ValueError) as exc:
            raise SkillError(
                FAILURE_CHECK_FAILED, f"camera shadow trial refused; no activation: {exc}"
            ) from exc
        except TimeoutError as exc:
            raise SkillError(
                FAILURE_TIMEOUT, "camera trial compute deadline expired; no motion or replay"
            ) from exc
