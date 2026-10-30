# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared raw-joint safety projection; publishing is not physical qualification.

Only an owner-held, committed generation may be projected here. Admission must
separately recompute its certificate before committing it. This adapter never
issues permits, changes the database, opens devices, or commands recovery motion.
Native Cartesian and continuous consumers are incompatible until independently
bounded under both models; their existing nominal behavior is unchanged otherwise.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from leapcalib.contracts.identity import digest_of
from leapcalib.contracts.raw_task import identities
from leapcalib.contracts.motion import finite_vector
from leapflow.hardware.calibration.acquisition import NOMINAL_FLANGE, RegistryFeedbackSource, SettledProfile
from leapflow.hardware.calibration.adapter import model_identity
from leapflow.hardware.calibration.generations import ModelGeneration
from leapflow.hardware.calibration.physical_model import PhysicalModel, SafetyGeometry, check_dual_sweep, dual_model_reach_mm
from leapflow.hardware.context import HardwareContext
from leapflow.hardware.node_gate import FAILURE_SAFETY_DENIED, refusal
from leapflow.hardware.skills.protocol import SkillError
from leapflow.hardware.transport import BatchTransport, TransportError, WriteOutcome


@dataclass(frozen=True)
class CommandSafetySnapshot:
    device_id: str
    declaration_digest: str
    node_session_id: str
    generation: int
    nominal: PhysicalModel
    corrected: PhysicalModel
    geometry: SafetyGeometry
    joint_ids: tuple[str, ...]
    pose_ids: tuple[str, ...]
    health_ids: tuple[str, ...]
    allowed_modes: tuple[str, ...]
    feedback_profile: SettledProfile
    nominal_fk_mm: float
    nominal_fk_deg: float
    speed_channel: str
    speed_pct_max: float
    joint_rate_bounds_rad_s: tuple[float, ...]
    endpoint_rate_bound_mm_s: float
    rate_provenance_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.nominal, PhysicalModel) or not isinstance(self.corrected, PhysicalModel) or not isinstance(self.geometry, SafetyGeometry) or not isinstance(self.feedback_profile, SettledProfile):
            raise ValueError("typed immutable model, geometry and feedback policy required")
        identities((self.device_id, self.declaration_digest, self.node_session_id, self.rate_provenance_digest))
        for name, count in (("joint_ids", self.nominal.dof), ("pose_ids", 6), ("health_ids", 4)):
            object.__setattr__(self, name, identities(getattr(self, name), minimum=count, maximum=count))
        object.__setattr__(self, "allowed_modes", identities(self.allowed_modes, maximum=16))
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("committed positive model generation required")
        if self.nominal.nominal_digest != self.corrected.nominal_digest or self.nominal.dof != self.corrected.dof or any(self.nominal.zero_offsets_rad):
            raise ValueError("nominal safety model must retain raw firmware zeros and matching DH")
        if len(self.geometry.link_radii_mm) != self.nominal.dof + 1:
            raise ValueError("safety bounds must cover every link and tool")
        channels = (*self.joint_ids, *self.pose_ids, *self.health_ids)
        if len(set(channels)) != len(channels) or not isinstance(self.speed_channel, str) or not self.speed_channel or len(self.speed_channel) > 256 or self.speed_channel in channels:
            raise ValueError("distinct explicitly bound raw/pose/health/speed channels required")
        object.__setattr__(self, "joint_rate_bounds_rad_s", finite_vector(self.joint_rate_bounds_rad_s, self.nominal.dof))
        for value in (self.nominal_fk_mm, self.nominal_fk_deg, self.speed_pct_max, self.endpoint_rate_bound_mm_s, *self.joint_rate_bounds_rad_s):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("finite positive declaration-owned tolerances and speed cap required")
        if self.speed_pct_max > 10:
            raise ValueError("corrected regression speed cannot exceed ten percent")
        if self.geometry.minimum_clearance_mm < 30 or self.geometry.boundary_margin_mm < 20:
            raise ValueError("physical study clearance/boundary floors cannot be waived")
        reach = dual_model_reach_mm(self.nominal, self.corrected, self.geometry)
        profile = self.feedback_profile
        sensing_floor = profile.tcp_uncertainty_mm + reach * (
            self.nominal.dof * profile.joint_uncertainty_rad + math.radians(profile.rotation_uncertainty_deg)
        )
        delay = profile.max_feedback_age_s + profile.feedback_uncertainty_s + profile.max_coherence_s
        timing_floor = delay * (self.endpoint_rate_bound_mm_s + reach * sum(self.joint_rate_bounds_rad_s))
        if self.geometry.sensing_mm < sensing_floor or self.geometry.timing_mm < timing_floor:
            raise ValueError("swept geometry cannot understate sensing/angular/timing uncertainty")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "leapflow.command_safety.v1", **vars(self),
                "nominal": self.nominal.to_dict(), "corrected": self.corrected.to_dict(),
                "geometry": self.geometry.to_dict(), "feedback_profile": vars(self.feedback_profile),
                **{name: list(getattr(self, name)) for name in ("joint_ids", "pose_ids", "health_ids", "allowed_modes", "joint_rate_bounds_rad_s")},
                "consumer_scope": ["raw_joint_discrete"], "coordinates": "raw_deg"}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> CommandSafetySnapshot:
        if not isinstance(data, Mapping) or set(data) != {"schema", "consumer_scope", "coordinates", *cls.__dataclass_fields__} or data["schema"] != "leapflow.command_safety.v1" or data["consumer_scope"] != ["raw_joint_discrete"] or data["coordinates"] != "raw_deg":
            raise ValueError("incompatible physical command snapshot scope or coordinates")
        fields = {k: data[k] for k in cls.__dataclass_fields__}
        return cls(**{**fields, "nominal": PhysicalModel.from_mapping(data["nominal"]),
                      "corrected": PhysicalModel.from_mapping(data["corrected"]),
                      "geometry": SafetyGeometry.from_mapping(data["geometry"]),
                      "feedback_profile": SettledProfile(**data["feedback_profile"])})


class _LockedReader:
    """Borrow the gate-held transport without reacquiring its registry I/O lock."""

    def __init__(self, registry: Any, transport: Any, device_id: str) -> None:
        self._registry, self._transport, self._device = registry, transport, device_id

    def context(self, device_id: str) -> HardwareContext | None:
        return self._registry.context(device_id)

    async def read_batch(self, device_id: str, channel_ids: Sequence[str]) -> Any:
        if device_id != self._device or not isinstance(self._transport, BatchTransport):
            raise ValueError("corrected safety requires coherent same-owner batch feedback")
        return await self._transport.read_batch(tuple(channel_ids))


class DualModelCommandSafety:
    """Registry-owned snapshots, atomically published while owner resources are reserved."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._snapshots: dict[str, CommandSafetySnapshot] = {}
        self._history: dict[str, CommandSafetySnapshot] = {}
        self._invalidated: set[str] = set()

    def publish_reserved(self, snapshot: CommandSafetySnapshot, generation: ModelGeneration) -> None:
        """Project a committed value; this does not assess evidence or grant authority."""
        if not isinstance(snapshot, CommandSafetySnapshot) or not isinstance(generation, ModelGeneration):
            raise ValueError("typed committed command generation required")
        sessions = getattr(self._registry, "node_sessions", None)
        if sessions is None or not sessions.reserved_device(snapshot.device_id):
            raise ValueError("physical projection requires the owner's between-run reservation")
        if generation.generation != snapshot.generation or generation.model != snapshot.to_dict() or generation.model_digest != digest_of(snapshot.to_dict()) or generation.dependency_digest != snapshot.corrected.dependency_digest:
            raise ValueError("committed model differs from the reserved immutable safety projection")
        context = self._registry.context(snapshot.device_id)
        if context is None or digest_of(context.to_dict()) != snapshot.declaration_digest:
            raise ValueError("physical projection declaration changed")
        speed = context.channel(snapshot.speed_channel)
        if speed is None or not speed.is_readable or not speed.is_writable or speed.unit != "percent" or not speed.envelope.contains(snapshot.speed_pct_max):
            raise ValueError("physical projection requires admitted percent speed feedback and cap")
        if context.safety is not None and context.safety.max_velocity_rad_s > 0 and any(rate > context.safety.max_velocity_rad_s for rate in snapshot.joint_rate_bounds_rad_s):
            raise ValueError("physical rate bound exceeds tighter live safety declaration")
        status = getattr(self._registry, "control_loop_status", lambda _: None)(snapshot.device_id)
        if status and status.get("running"):
            raise ValueError("active streaming consumer is incompatible with corrected activation")
        clear_recovery = getattr(self._registry, "clear_command_recovery_reserved", None)
        if callable(clear_recovery):
            clear_recovery(snapshot.device_id, generation)
        self._snapshots[snapshot.device_id] = snapshot
        self._history[generation.model_digest] = snapshot
        while len(self._history) > 32:
            self._history.pop(next(iter(self._history)))
        self._invalidated.discard(snapshot.device_id)

    def rollback_projection(self, generation: ModelGeneration) -> CommandSafetySnapshot | None:
        """Reuse only a previously published same-owner snapshot, never trust model JSON.

        Restart or history eviction requires requalification; absence is a stop-use
        result. Every ordinary write still independently checks fresh feedback and
        both swept models after a compatible snapshot has been republished.
        """
        snapshot = self._history.get(generation.model_digest)
        if snapshot is None or snapshot.generation != generation.generation:
            return None
        context = self._registry.context(snapshot.device_id)
        if context is None or digest_of(context.to_dict()) != snapshot.declaration_digest:
            return None
        if generation.model != snapshot.to_dict():
            return None
        return snapshot

    async def validate_rollback_projection(self, snapshot: CommandSafetySnapshot) -> None:
        """Recheck a retained same-owner value; validation itself grants no authority."""
        if not isinstance(snapshot, CommandSafetySnapshot) or self._history.get(digest_of(snapshot.to_dict())) is not snapshot:
            raise ValueError("rollback needs a previously published same-owner projection")
        context = self._registry.context(snapshot.device_id)
        if context is None:
            raise ValueError("rollback device declaration unavailable")
        transport = await self._registry.transport(snapshot.device_id)
        async with asyncio.timeout(snapshot.feedback_profile.timeout_s):
            await self._check(snapshot, context, transport,
                              ((snapshot.speed_channel, snapshot.speed_pct_max),), retained=True)

    def invalidate_reserved(self, device_id: str) -> None:
        """Stop use of correction without silently restoring unqualified nominal consumers."""
        sessions = getattr(self._registry, "node_sessions", None)
        if sessions is None or not sessions.reserved_device(device_id):
            raise ValueError("physical invalidation requires an owner reservation")
        self._snapshots.pop(device_id, None)
        self._invalidated.add(device_id)

    def control_refusal(self, device_id: str) -> str | None:
        if device_id in self._invalidated:
            return "physical correction invalidated; requalify dependencies before precision/control use"
        snapshot = self._snapshots.get(device_id)
        if snapshot is not None:
            return f"corrected generation {snapshot.generation} excludes native Cartesian/streaming control; no verified dual-model swept-path bound"
        return None

    async def check_write(self, context: HardwareContext, transport: Any, commands: Sequence[tuple[str, Any]]) -> WriteOutcome | None:
        if context.device_id in self._invalidated:
            return refusal("physical correction invalidated; inspect dependencies before another command", FAILURE_SAFETY_DENIED)
        snapshot = self._snapshots.get(context.device_id)
        if snapshot is None:
            return None
        try:
            async with asyncio.timeout(snapshot.feedback_profile.timeout_s):
                await self._check(snapshot, context, transport, commands)
        except (SkillError, TransportError, OSError, TimeoutError, ValueError, TypeError) as error:
            return refusal(f"corrected raw-joint safety refused before dispatch: {error}", FAILURE_SAFETY_DENIED)
        return None

    async def _check(self, snapshot: CommandSafetySnapshot, context: HardwareContext, transport: Any, commands: Sequence[tuple[str, Any]], *, retained: bool = False) -> None:
        if digest_of(context.to_dict()) != snapshot.declaration_digest:
            raise ValueError("physical command declaration no longer matches frozen generation")
        if not isinstance(transport, BatchTransport):
            raise ValueError("corrected safety requires coherent batch feedback")
        values, speed_receipt = await self._commands(snapshot, context, transport, commands, require_live_speed=retained)
        source = RegistryFeedbackSource(_LockedReader(self._registry, transport, context.device_id),
            device_id=context.device_id, model=model_identity(snapshot.nominal.nominal, model_id=context.device_id),
            joint_ids=snapshot.joint_ids, pose_ids=snapshot.pose_ids, health_ids=snapshot.health_ids,
            allowed_modes=snapshot.allowed_modes, profile=snapshot.feedback_profile,
            node_session_id=snapshot.node_session_id, pose_semantics=NOMINAL_FLANGE)
        measured = await source.sample()
        current = tuple(map(math.degrees, measured.joints_rad))
        nominal_flange = snapshot.nominal.nominal.forward(current)
        if nominal_flange.distance_mm(measured.tcp) > snapshot.nominal_fk_mm or nominal_flange.rotation_to_deg(measured.tcp) > snapshot.nominal_fk_deg:
            raise ValueError("nominal DH versus nominal firmware flange check failed")
        target = tuple(values.get(cid, q) for cid, q in zip(snapshot.joint_ids, current))
        await self._sweep(snapshot, current, target)
        # Holding the I/O lock excludes intervening writes, but slow geometry must
        # not turn fresh feedback into a stale dispatch or conceal a remount.
        first_receipt = min(measured.first_receipt_s, speed_receipt or measured.first_receipt_s)
        if time.monotonic() - first_receipt > snapshot.feedback_profile.max_feedback_age_s:
            raise ValueError("raw feedback expired during dual-model swept-path validation")
        bound = (self._history.get(digest_of(snapshot.to_dict())) if retained
                 else self._snapshots.get(context.device_id))
        if self._registry.context(context.device_id) is not context or bound is not snapshot:
            raise ValueError("physical dependencies changed during final safety validation")

    @staticmethod
    async def _commands(snapshot: CommandSafetySnapshot, context: HardwareContext, transport: BatchTransport, commands: Sequence[tuple[str, Any]], *, require_live_speed: bool = False) -> tuple[dict[str, Any], float | None]:
        if not commands or len(commands) > len(snapshot.joint_ids) + 1 or len({cid for cid, _ in commands}) != len(commands):
            raise ValueError("bounded distinct raw-joint command vector required")
        values = dict(commands)
        if any(cid not in (*snapshot.joint_ids, snapshot.speed_channel) for cid in values):
            raise ValueError("consumer is outside the admitted raw-joint scope; nominal firmware Cartesian IK is incompatible")
        finite_vector(tuple(values.values()), len(values))
        def check_speed(speed: Any) -> None:
            if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or not 0 < speed <= snapshot.speed_pct_max:
                raise ValueError("speed exceeds the frozen physical regression cap")
        if snapshot.speed_channel in values:
            check_speed(values[snapshot.speed_channel])
            if commands[0][0] != snapshot.speed_channel:
                raise ValueError("bounded speed must be applied before raw-joint targets")
        receipt = None
        if require_live_speed or any(cid in snapshot.joint_ids for cid in values):
            # BatchTransport may reorder channels. BOTH the prior and proposed
            # speed must be bounded; a combined batch cannot fix an unsafe prior
            # speed with a promise to apply its speed field first.
            batch = await transport.read_batch((snapshot.speed_channel,))
            if batch.device_id != context.device_id or len(batch.readings) != 1:
                raise ValueError("complete same-owner live speed feedback required")
            reading = batch.readings[0]
            now = time.monotonic()
            if reading.device_id != context.device_id or reading.channel_id != snapshot.speed_channel or reading.unit != "percent" or not reading.is_trustworthy or type(reading.sequence) is not int or reading.sequence < 0 or isinstance(reading.monotonic_at, bool) or not isinstance(reading.monotonic_at, (int, float)) or not math.isfinite(reading.monotonic_at) or not 0 <= now - reading.monotonic_at <= snapshot.feedback_profile.max_feedback_age_s:
                raise ValueError("fresh trustworthy percent speed feedback required")
            check_speed(reading.value)
            receipt = reading.monotonic_at
        return values, receipt

    @staticmethod
    async def _sweep(snapshot: CommandSafetySnapshot, current: tuple[float, ...], target: tuple[float, ...]) -> None:
        sweep = asyncio.create_task(asyncio.to_thread(check_dual_sweep, snapshot.nominal, snapshot.corrected, current, target, snapshot.geometry))
        try:
            await asyncio.shield(sweep)
        except asyncio.CancelledError:
            # Bounded pure geometry cannot be killed; keep dispatch ownership until
            # its thread settles so repeated cancellation cannot pile up work.
            while not sweep.done():
                try:
                    await asyncio.shield(sweep)
                except asyncio.CancelledError:
                    continue
                except ValueError:
                    break
            if not sweep.cancelled():
                sweep.exception()
            raise

