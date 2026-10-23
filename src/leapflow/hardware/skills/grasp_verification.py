# Copyright (c) Alibaba, Inc. and its affiliates.
"""One run's qualified grasp observation, retained across contact and proof lift."""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

from leapflow.hardware.grasp_evidence import GraspLimits, GraspState, TemporalGraspObserver
from leapflow.hardware.grasp_measurement import GraspQualification, digest, number
from leapflow.hardware.providers.yaml_provider import declaration_digest
from leapflow.hardware.skills.protocol import (
    FAILURE_GUARD_FAILED,
    FAILURE_SKILL_UNAVAILABLE,
    SkillError,
)

PROCEDURE = "grasp.qualification"


class WorkcellGraspVerification:
    """Backend behind the skill port; neither creates a transport nor writes one.

    The physical package is stored through the governed calibration skill. Its
    camera/TCP/table geometry must equal the active FrameGraph snapshot, and its
    dependency versions are checked again on every sample. No cached observation
    can be re-stamped with a newly installed package or camera session.
    """

    def __init__(self, ctx: Any, registry: Any) -> None:
        self.ctx, self.registry = ctx, registry
        self._observer: TemporalGraspObserver | None = None
        self._sample: Any = None
        self._qualification_digest = ""
        self._session = ""
        self._sources: tuple[float, ...] | None = None
        self._feedback_at = float("-inf")
        self._recorded_at: float | None = None
        self._stage_at = time.monotonic()

    def admit(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
        qualification = self._admit(data)
        return {
            "qualification_digest": qualification.digest,
            "parameters": dict(qualification.data),
        }

    def qualification(self) -> Mapping[str, Any]:
        qualification = self._qualification()
        store = self.registry.calibration_store
        overview = store.latest(self.ctx.workcell.workcell_id, "camera.verify")
        if (
            overview is None
            or overview.parameters.get("camera_role") != "camera"
            or overview.parameters.get("training_disjoint") is not True
            or overview.parameters.get("depends_on")
            != {"camera": self.ctx.frames.versions.get("camera")}
        ):
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE, "fresh held-out overview-camera verification required"
            )
        errors = overview.parameters.get("errors_mm", [])
        if (
            not isinstance(errors, (list, tuple))
            or len(errors) < self.ctx.workcell.calibration["min_verify_points"]
        ):
            raise ValueError("overview verification lacks held-out measurements")
        errors = sorted(number(e) for e in errors)
        p95 = errors[max(0, math.ceil(len(errors) * 0.95) - 1)]
        if min(errors) < 0 or p95 > self.ctx.workcell.calibration["max_verify_p95_mm"]:
            raise ValueError("overview held-out physical errors exceed acceptance")
        return {
            "qualification_digest": qualification.digest,
            "planar_p95_mm": p95,
            "overview_verified_at": overview.recorded_at,
            "command_force_n": qualification.effort.data["command_force_n"],
        }

    def _admit(self, data: Mapping[str, Any]) -> GraspQualification:
        spec, frames = self.ctx.workcell, self.ctx.frames
        qualification = GraspQualification.admit(
            data,
            device_id=spec.workcell_id,
            recorded_at=time.time(),
            limits=spec.grasp_verify,
        )
        if (
            qualification.data["camera_device"] != spec.role("camera.wrist")
            or frames.parent("camera.wrist") != "flange"
        ):
            raise ValueError("qualification needs the exact flange-mounted wrist camera")
        camera = self.registry.context(spec.role("camera.wrist"))
        config = camera.transport.config.get("geometry_measurement") if camera is not None else None
        if (
            not isinstance(config, Mapping)
            or digest(config) != qualification.data["geometry_configuration_digest"]
        ):
            raise ValueError(
                "camera measurement selectors/configuration differ from physical qualification"
            )
        measurement = camera.transport.config.get("measurement")
        if (
            not isinstance(measurement, Mapping)
            or digest(measurement) != qualification.data["depth_configuration_digest"]
        ):
            raise ValueError("camera depth ROI/configuration differs from physical qualification")
        arm = self.registry.context(spec.role("arm"))
        if arm is None or qualification.data["arm_declaration_digest"] != declaration_digest(arm):
            raise ValueError("arm declaration differs from qualified SDK/firmware geometry")
        for frame, component in (("camera.wrist", "camera_to_base"), ("tcp", "flange_tcp")):
            transform = frames.transform(frame)
            if transform is None:
                raise ValueError(f"active {frame} calibration required")
            qualification.package.require_geometry(
                component, frames.parameters(frame), transform.to_matrix()
            )
        if frames.table is None:
            raise ValueError("active qualified support plane required")
        qualification.package.require_geometry("table_plane", frames.parameters("workspace"))
        if dict(qualification.data.get("depends_on", {})) != dict(frames.versions):
            raise ValueError("qualification does not bind all active calibration versions")
        return qualification

    def _qualification(self) -> GraspQualification:
        store = self.registry.calibration_store
        record = (
            store.latest(self.ctx.workcell.workcell_id, PROCEDURE) if store is not None else None
        )
        if record is None:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE, "independent grasp qualification is not installed"
            )
        qualification = self._admit(record.parameters)
        if self._recorded_at is not None and self._recorded_at != record.recorded_at:
            raise ValueError("grasp qualification changed during the run")
        self._recorded_at = record.recorded_at
        for frame, version in self.ctx.frames.versions.items():
            live = store.latest(self.ctx.workcell.workcell_id, self.ctx.frames.procedure(frame))
            if live is None or live.recorded_at != version:
                raise ValueError("active calibration changed during grasp observation")
        return qualification

    async def _read(self, role: str, channels: tuple[str, ...]) -> Mapping[str, Any]:
        device = self.ctx.workcell.role(role)
        batch = await self.registry.read_batch(device, channels)
        rows = {r.channel_id: r for r in batch.readings}
        if len(rows) != len(batch.readings) or set(rows) != set(channels):
            raise ValueError("incomplete or duplicate qualified reading batch")
        now = time.monotonic()
        for row in rows.values():
            if (
                row.device_id != device
                or not row.is_trustworthy
                or not 0
                <= now - number(row.monotonic_at)
                <= self.ctx.workcell.grasp_verify["max_age_s"]
            ):
                raise ValueError("stale, unqualified or misbound measurement reading")
        return rows

    async def _capture(self, channel: str) -> Any:
        """Bounded camera-only warmup; never admit or restamp an immature window."""
        limits = self.ctx.workcell.grasp_verify
        device = self.ctx.workcell.role("camera.wrist")
        deadline = time.monotonic() + limits["warmup_timeout_s"]
        while True:
            batch = await self.registry.read_batch(device, (channel,))
            if batch.device_id != device or len(batch.readings) != 1:
                raise ValueError("incomplete camera warmup reading")
            row = batch.readings[0]
            if (
                row.device_id != device
                or row.channel_id != channel
                or not isinstance(row.value, Mapping)
            ):
                raise ValueError("misbound camera warmup source")
            if not 0 <= time.monotonic() - number(row.monotonic_at) <= limits["max_age_s"]:
                raise ValueError("stale camera warmup source")
            depths = row.value.get("pad_depth", []) if channel.endswith("geometry") else [row.value]
            fresh = bool(depths)
            for depth in depths:
                times, ids = depth.get("observation_times", []), depth.get("frame_ids", [])
                if not times:
                    fresh = False
                    continue
                if (
                    not isinstance(times, (tuple, list))
                    or not isinstance(ids, (tuple, list))
                    or not 1 <= len(times) == len(ids) <= 50
                    or any(not isinstance(i, str) or not i for i in ids)
                    or len(set(ids)) != len(ids)
                ):
                    raise ValueError("malformed depth capture window")
                stamps = [number(t) for t in times]
                if (
                    any(b <= a for a, b in zip(stamps, stamps[1:]))
                    or stamps[-1] > row.monotonic_at
                    or time.monotonic() - stamps[0] > limits["max_age_s"]
                ):
                    raise ValueError("depth capture clocks are stale, future or unordered")
                fresh &= stamps[0] >= self._stage_at
            if fresh and row.is_trustworthy:
                return row
            if time.monotonic() >= deadline:
                raise ValueError("stationary camera measurement window did not qualify")
            await self.ctx.motion.pause(limits["poll_s"])

    async def inspect(self) -> Mapping[str, Any]:
        self._stage_at = time.monotonic()
        scope = self.qualification()
        qualification = self._qualification()
        before = await self.ctx.motion.state()
        quality = await self._capture("camera.rgbd.quality")
        sample, alignment = await self._observe(qualification)
        depth = quality.value
        if (
            depth.get("qualified") is not True
            or depth.get("calibration_digest") != qualification.package.digest
            or depth.get("mount_digest") != qualification.package.certificates[0].mount_digest
            or depth.get("profile_digest") != qualification.data["profile_digest"]
            or depth.get("configuration_digest") != qualification.data["depth_configuration_digest"]
            or any(
                not frame.startswith(sample.identity.session_id + ":")
                for frame in depth["frame_ids"]
            )
        ):
            raise SkillError(
                FAILURE_GUARD_FAILED, "report depth scope differs from physical qualification"
            )
        after = await self.ctx.motion.state()
        limits = self.ctx.workcell.grasp_verify
        if (
            before.flange.distance_mm(after.flange) > limits["capture_drift_mm"]
            or before.flange.rotation_to_deg(after.flange) > limits["capture_drift_deg"]
        ):
            raise ValueError("report capture moved during the stationary depth window")
        return {
            "qualification": scope,
            "calibration_versions": dict(self.ctx.frames.versions),
            "depth_quality": dict(quality.value),
            "measured_geometry": {
                "identity": sample.identity.__dict__,
                "alignment_tool_mm": alignment,
                "independent_jaws": sample.independent_jaws,
                "object_on_support": sample.object_on_support,
            },
        }

    async def _observe(self, qualification: GraspQualification) -> tuple[Any, float | None]:
        limits = self.ctx.workcell.grasp_verify
        before = await self.ctx.motion.state()
        geometry_row = await self._capture("camera.rgbd.geometry")
        feedback = await self._read("gripper", ("gripper.position", "gripper.effort"))
        after = await self.ctx.motion.state()
        if (
            before.arrived is not True
            or after.arrived is not True
            or before.flange.distance_mm(after.flange) > limits["capture_drift_mm"]
            or before.flange.rotation_to_deg(after.flange) > limits["capture_drift_deg"]
        ):
            raise ValueError("effort and wrist measurements require stationary capture")
        geometry = geometry_row.value
        position, effort = feedback["gripper.position"], feedback["gripper.effort"]
        if position.unit != "mm" or effort.unit != "N":
            raise ValueError("qualified gripper feedback must use mm and N")
        stamps = [geometry_row.monotonic_at, position.monotonic_at, effort.monotonic_at]
        if max(stamps) - min(stamps) > limits["max_sync_skew_s"]:
            raise ValueError("camera/gripper source receive skew exceeds acceptance")
        if min(position.monotonic_at, effort.monotonic_at) <= self._feedback_at:
            raise ValueError("gripper receive evidence did not advance")
        source = geometry["source"]
        clocks = tuple(number(source[k]["timestamp_ms"]) for k in ("color_source", "depth_source"))
        numbers = tuple(source[k]["frame_number"] for k in ("color_source", "depth_source"))
        domains = tuple(source[k].get("timestamp_domain") for k in ("color_source", "depth_source"))
        if (
            any(type(n) is not int or n <= 0 for n in numbers)
            or not domains[0]
            or domains[0] != domains[1]
            or abs(clocks[0] - clocks[1]) > limits["max_sync_skew_s"] * 1000
        ):
            raise ValueError("SDK frame identities or clock domains are unqualified")
        newest = f"{geometry['session_id']}:{numbers[0]}:{numbers[1]}"
        if any(
            not depth.get("frame_ids") or depth["frame_ids"][-1] != newest
            for depth in (*geometry["pad_depth"], geometry["depth"])
        ):
            raise ValueError("depth windows do not identify the current SDK bundle")
        progression = (*clocks, *numbers)
        if self._sources is not None and any(a <= b for a, b in zip(progression, self._sources)):
            raise ValueError("SDK source clocks and frame numbers must advance")
        if self._session and self._session != geometry["session_id"]:
            raise ValueError("camera session changed during grasp verification")
        if (
            abs(number(geometry["observed_mono"]) - geometry_row.monotonic_at)
            > limits["max_sync_skew_s"]
        ):
            raise ValueError("geometry observation and source envelope disagree")
        self._sources, self._feedback_at = (
            progression,
            min(position.monotonic_at, effort.monotonic_at),
        )
        self._session = geometry["session_id"]
        return qualification.measure(
            geometry,
            width_mm=position.value,
            effort_n=effort.value,
            flange=after.flange,
            tcp=after.tcp,
            mount=self.ctx.frames.transform("camera.wrist"),
            table=self.ctx.frames.table,
            synchronized=True,
            limits=limits,
        )

    async def verify(
        self, stage: str, *, samples: int, grasp: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        limits = self.ctx.workcell.grasp_verify
        if (
            stage not in ("pre_grasp", "hold", "empty")
            or type(samples) is not int
            or not 3 <= samples <= 10
        ):
            raise SkillError(
                FAILURE_GUARD_FAILED, "invalid grasp verification stage or sample count"
            )
        observations = []
        self._stage_at = time.monotonic()
        try:
            qualification = self._qualification()
            if self._qualification_digest and qualification.digest != self._qualification_digest:
                raise ValueError("physical evidence scope changed")
            self._qualification_digest = qualification.digest
            grasp_limits = GraspLimits(
                **{
                    k: int(limits[k]) if k == "min_frames" else limits[k]
                    for k in GraspLimits.__dataclass_fields__
                }
            )
            if self._observer is None:
                self._observer = TemporalGraspObserver(
                    lambda: self._sample,
                    limits=grasp_limits,
                    current=lambda: self._sample.identity,
                    clock=time.monotonic,
                )
            result: dict[str, Any] = {}
            for index in range(samples):
                # Re-admit the physical store/version bindings on every sample.
                qualification = self._qualification()
                self._sample, alignment = await self._observe(qualification)
                evidence = self._observer.observe_grasp()
                state = evidence.state(
                    self._sample.identity, now=time.monotonic(), limits=grasp_limits
                )
                aligned = (
                    alignment is not None
                    and abs(alignment) + self._sample.uncertainty_mm <= limits["alignment_mm"]
                )
                pre = (
                    aligned
                    and self._sample.visible
                    and evidence.synchronized
                    and self._sample.independent_jaws
                    and self._sample.width_consistent
                )
                if grasp is not None and stage == "pre_grasp":
                    pre &= (
                        self._sample.width_mm
                        >= number(grasp["open_mm"]) - limits["jaw_stroke_error_mm"]
                    )
                result = {
                    "stage": stage,
                    "state": state.value,
                    "alignment_tool_mm": alignment,
                    "passed": pre
                    if stage == "pre_grasp"
                    else state in (GraspState.CONTACT_CANDIDATE, GraspState.HELD_VERIFIED)
                    if stage == "hold"
                    else state == GraspState.EMPTY,
                    "qualification_digest": qualification.digest,
                    "source_identity": dict(self._sample.identity.__dict__),
                    "evidence": evidence.to_dict(),
                }
                observations.append(dict(result))
                if not result["passed"]:
                    break
                if index + 1 < samples:
                    await self.ctx.motion.pause(limits["poll_s"])
            result["observations"] = observations
            result["passed"] = bool(
                result.get("passed")
                and len(observations) >= max(samples, int(limits["min_frames"]))
            )
            self.ctx.note("grasp_verify", result)
            return result
        except Exception as error:
            self._observer = None
            self.ctx.note(
                "grasp_verify",
                {
                    "stage": stage,
                    "state": "unknown",
                    "passed": False,
                    "reason": str(error),
                    "observations": observations,
                },
            )
            if isinstance(error, SkillError):
                raise
            raise SkillError(
                FAILURE_GUARD_FAILED,
                f"grasp verification refused: {error}",
                result={"stage": stage, "state": "unknown", "passed": False},
            ) from error
