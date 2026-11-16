# Copyright (c) Alibaba, Inc. and its affiliates.
"""Governed between-run rollback through the existing owner reservation and run hold.

A retained same-owner projection is required to restore geometry. Missing history,
changed dependencies, and restarts produce an atomic stop-use result instead. This
module never commands motion, manufactures qualification, or consumes a permit.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Mapping

from leapcalib.contracts.identity import digest_of
from leapflow.hardware.calibration.acquisition import NOMINAL_FLANGE, RegistryFeedbackSource
from leapflow.hardware.calibration.adapter import model_identity
from leapflow.hardware.calibration.command_safety import (
    CommandSafetySnapshot,
    DualModelCommandSafety,
)
from leapflow.hardware.calibration.generations import ModelGeneration
from leapflow.hardware.calibration.evidence import _settle_tasks
from leapflow.hardware.transport import TransportError
from leapflow.hardware.skills.protocol import (
    FAILURE_CHECK_FAILED,
    FAILURE_PLAN_STALE,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_STOPPED,
    SkillError,
    SkillPlan,
)

SCHEMA = "leapflow.calibration_rollback.v1"


async def _read(function: Callable[..., Any], *args: Any) -> Any:
    """Finish shared-holder reads before cancellation can release owner resources."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    await _settle_tasks((task,))
    return task.result()


class WorkcellRollback:
    """Concrete non-motion owner facade, with exact dependency and plan identity."""

    def __init__(self, context: Any, registry: Any) -> None:
        self._ctx, self._registry = context, registry

    def _store(self) -> Any:
        store = self._registry.calibration_generations
        if store is None or self._registry.node_sessions is None:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                "rollback requires the existing instrument DB and node session owner",
            )
        return store

    def _dependency(self, snapshot: CommandSafetySnapshot) -> str:
        context = self._registry.context(snapshot.device_id)
        current = digest_of(context.to_dict()) if context is not None else "missing"
        if current == snapshot.declaration_digest:
            return snapshot.corrected.dependency_digest
        # A changed declaration can only stop use; never restore matching old JSON.
        return digest_of({"prior": snapshot.corrected.dependency_digest, "declaration": current})

    async def basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        store = self._store()
        active = await _read(store.snapshot, self._ctx.workcell.workcell_id)
        if active is None:
            raise SkillError(
                FAILURE_CHECK_FAILED, "no active correction to roll back; inspect calibration state"
            )
        expected = args.get("expected_generation")
        if type(expected) is not int or expected != active.generation:
            raise SkillError(FAILURE_PLAN_STALE, "rollback must name the exact active generation")
        try:
            snapshot = CommandSafetySnapshot.from_mapping(active.model)
        except (TypeError, ValueError, KeyError) as exc:
            raise SkillError(
                FAILURE_CHECK_FAILED,
                "active model has incompatible command semantics; owner reconciliation required",
            ) from exc
        if snapshot.device_id not in self._ctx.workcell.actuator_ids:
            raise SkillError(
                FAILURE_CHECK_FAILED, "rollback generation belongs to another workcell device"
            )
        return {
            "schema": SCHEMA,
            "active": active.basis(),
            "workcell_digest": digest_of(self._ctx.workcell.to_dict()),
            "dependency_digest": self._dependency(snapshot),
            "device_id": snapshot.device_id,
            "previous_generation": active.previous_generation,
            "commands": [],
            "restore_requires": "revalidated_same_owner_projection",
            "unavailable_prior": "atomic_stop_use",
        }

    async def _still(self, snapshot: CommandSafetySnapshot) -> None:
        profile = snapshot.feedback_profile
        source = RegistryFeedbackSource(
            self._registry,
            device_id=snapshot.device_id,
            model=model_identity(snapshot.nominal.nominal, model_id=snapshot.device_id),
            joint_ids=snapshot.joint_ids,
            pose_ids=snapshot.pose_ids,
            health_ids=snapshot.health_ids,
            allowed_modes=snapshot.allowed_modes,
            profile=profile,
            node_session_id=snapshot.node_session_id,
            pose_semantics=NOMINAL_FLANGE,
        )
        samples = []
        async with asyncio.timeout(profile.timeout_s):
            while (
                len(samples) < profile.min_samples
                or samples[-1].first_receipt_s - samples[0].last_receipt_s < profile.span_s
            ):
                if self._ctx.cancelled:
                    raise SkillError(
                        FAILURE_STOPPED, "rollback stopped before its transaction; no motion"
                    )
                current = await source.sample()
                if samples:
                    first = samples[0]
                    if (
                        current.first_receipt_s <= samples[-1].last_receipt_s
                        or current.source_id == samples[-1].source_id
                    ):
                        raise SkillError(
                            FAILURE_CHECK_FAILED, "rollback requires increasing fresh raw feedback"
                        )
                    if current.ctrl_mode != first.ctrl_mode:
                        raise SkillError(
                            FAILURE_CHECK_FAILED, "feedback mode changed before rollback"
                        )
                    if (
                        max(abs(a - b) for a, b in zip(first.joints_rad, current.joints_rad))
                        + 2 * profile.joint_uncertainty_rad
                        > profile.max_joint_drift_rad
                    ):
                        raise SkillError(
                            FAILURE_CHECK_FAILED, "actuator motion blocks between-run rollback"
                        )
                    if (
                        first.tcp.distance_mm(current.tcp) + 2 * profile.tcp_uncertainty_mm
                        > profile.max_tcp_drift_mm
                        or first.tcp.rotation_to_deg(current.tcp)
                        + 2 * profile.rotation_uncertainty_deg
                        > profile.max_tcp_rotation_deg
                    ):
                        raise SkillError(
                            FAILURE_CHECK_FAILED, "measured flange motion blocks rollback"
                        )
                samples.append(current)
                if len(samples) >= 32:
                    raise SkillError(
                        FAILURE_CHECK_FAILED, "rollback stillness sample budget exhausted"
                    )
                await asyncio.sleep(profile.poll_s)

    async def _prepare(
        self, plan: SkillPlan, sealed: Mapping[str, Any], safety: DualModelCommandSafety
    ) -> tuple[ModelGeneration, CommandSafetySnapshot, CommandSafetySnapshot | None]:
        store = self._store()
        if sealed != await self.basis(plan.args):
            raise SkillError(
                FAILURE_PLAN_STALE, "rollback model or dependencies changed since approval"
            )
        active = await _read(store.snapshot, plan.workcell)
        if active is None or active.basis() != sealed["active"]:
            raise SkillError(FAILURE_PLAN_STALE, "active generation changed before rollback")
        current = CommandSafetySnapshot.from_mapping(active.model)
        try:
            await self._still(current)
        except TimeoutError as exc:
            raise SkillError(
                FAILURE_CHECK_FAILED,
                "rollback stillness deadline expired; inspect feedback before another request",
            ) from exc
        except (TransportError, OSError, ValueError, TypeError) as exc:
            raise SkillError(
                FAILURE_CHECK_FAILED,
                "rollback raw feedback unavailable; inspect device readiness before another request",
            ) from exc
        prior = None
        if active.previous_generation is not None:
            prior_generation = await _read(
                store.generation, plan.workcell, active.previous_generation
            )
            prior = safety.rollback_projection(prior_generation)
        # Check the retained projection against fresh feedback and both stationary
        # swept volumes, rather than interpreting persisted certificate strings.
        if prior is not None and self._dependency(prior) == sealed["dependency_digest"]:
            try:
                await safety.validate_rollback_projection(prior)
            except (SkillError, TransportError, OSError, ValueError, TypeError, TimeoutError):
                prior = None
        else:
            prior = None
        if self._ctx.cancelled:
            raise SkillError(FAILURE_STOPPED, "rollback cancelled before persistence; no motion")
        if self._dependency(current) != sealed["dependency_digest"]:
            raise SkillError(
                FAILURE_PLAN_STALE, "rollback dependencies changed during reserved preflight"
            )
        return active, current, prior

    async def execute(self, plan: SkillPlan) -> Mapping[str, Any]:
        store = self._store()
        sealed = plan.basis.get("rollback")
        if not isinstance(sealed, Mapping):
            raise SkillError(FAILURE_PLAN_STALE, "missing sealed rollback policy")
        safety = self._registry.command_safety
        if safety is None:
            safety = DualModelCommandSafety(self._registry)
            self._registry.bind_command_safety(safety)
        if not isinstance(safety, DualModelCommandSafety):
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE, "rollback requires the owner physical safety projection"
            )
        state: (
            tuple[ModelGeneration, CommandSafetySnapshot, CommandSafetySnapshot | None] | None
        ) = None

        async def preflight() -> None:
            nonlocal state
            state = await self._prepare(plan, sealed, safety)

        def dispatch() -> ModelGeneration | None:
            if state is None:
                raise ValueError("rollback preflight did not complete")
            active, current, prior = state
            compatible = digest_of(prior.to_dict()) if prior is not None else ""
            causal = {
                "device_id": current.device_id,
                "plan_digest": plan.digest,
                "run_id": self._ctx.run_id,
                "restore_model_digest": compatible,
                "reason": "revalidated_prior"
                if prior is not None
                else "prior_not_requalified_stop_use",
            }
            return store.rollback(
                workcell_id=plan.workcell,
                request_id=self._ctx.run_id,
                expected_generation=active.generation,
                dependency_digest=sealed["dependency_digest"],
                causal=causal,
                compatible_model_digest=compatible,
            )

        def publish(generation: ModelGeneration | None) -> None:
            if state is None:
                raise ValueError("rollback publication lacks reserved preflight")
            _, current, prior = state
            if generation is None:
                safety.invalidate_reserved(current.device_id)
            elif prior is not None:
                safety.publish_reserved(prior, generation)
            else:
                raise ValueError(
                    "rollback restored a generation without a retained safety projection"
                )

        result = await self._ctx.model_transaction(dispatch, publish=publish, preflight=preflight)
        return {
            "schema": SCHEMA,
            "state": "rolled_back" if result is not None else "invalidated",
            "generation": result.generation if result is not None else None,
            "previous_generation": sealed["active"]["generation"],
            "commands": [],
            "precision_use": "revalidated_prior"
            if result is not None
            else "stopped_requalification_required",
        }
