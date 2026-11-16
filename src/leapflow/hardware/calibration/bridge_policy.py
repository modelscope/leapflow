# Copyright (c) Alibaba, Inc. and its affiliates.
"""Declared bridge scope and exact, numerically verified wrist dependency."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from leapcalib.contracts.identity import digest_of
from leapcalib.estimation.bridge import BridgeBudget
from leapflow.hardware.calibration.acquisition import CollectionProfile
from leapflow.hardware.skills.vision import CharucoTarget


@dataclass(frozen=True)
class BridgePolicy:
    fixed_role: str
    wrist_role: str
    target_id: str
    stationary_epoch: str
    wrist_record_digest: str
    wrist_verification_digest: str
    freshness: Mapping[str, str]
    budget: BridgeBudget
    min_pose_gap_px: float
    tolerance_ms: float
    attempts: int
    training: int
    holdout: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> BridgePolicy:
        if not isinstance(data, Mapping) or set(data) != set(cls.__dataclass_fields__):
            raise ValueError("complete exact calibration_bridge declaration required")
        if not isinstance(data["budget"], Mapping):
            raise ValueError("explicit bridge budget required")
        return cls(**{**data, "budget": BridgeBudget(**data["budget"])})

    def __post_init__(self) -> None:
        if self.fixed_role == self.wrist_role or any(not isinstance(v, str) or v.split(".", 1)[0] != "camera" for v in (self.fixed_role, self.wrist_role)):
            raise ValueError("distinct declared camera roles required")
        for value in (self.target_id, self.stationary_epoch):
            if not isinstance(value, str) or not 0 < len(value) <= 256:
                raise ValueError("physical sheet identity and stationary mount epoch required")
        for value in (self.wrist_record_digest, self.wrist_verification_digest):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("exact wrist calibration/verification record digests required")
        if not isinstance(self.freshness, Mapping) or set(self.freshness) != {"node_session_id", "mount_digest", "source_profile_digest", "confirmation_id"} or any(not isinstance(v, str) or not 0 < len(v) <= 256 for v in self.freshness.values()):
            raise ValueError("operator-confirmed wrist-relative-to-flange freshness and current owner/profile required")
        from types import MappingProxyType
        object.__setattr__(self, "freshness", MappingProxyType(dict(self.freshness)))
        for name in ("min_pose_gap_px", "tolerance_ms"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("positive declared detection/time bounds required")
        if not isinstance(self.budget, BridgeBudget) or type(self.attempts) is not int or not 1 <= self.attempts <= 3:
            raise ValueError("bounded capture attempts and explicit budget required")
        if any(type(v) is not int or not minimum <= v <= 64 for v, minimum in ((self.training, self.budget.min_training), (self.holdout, self.budget.min_holdout))):
            raise ValueError("preallocated training/holdout counts below declared support")

    def to_dict(self) -> dict[str, Any]:
        return {**vars(self), "freshness": dict(self.freshness), "budget": self.budget.to_dict()}


def validate_declaration(data: Mapping[str, Any], *, roles: Mapping[str, str], frames: Mapping[str, Any], acquisition: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    if not data:
        return {}
    policy = BridgePolicy.from_mapping(data)
    collection = CollectionProfile.from_mapping(acquisition)
    CharucoTarget.from_mapping(target)
    for role, parent in ((policy.fixed_role, "base"), (policy.wrist_role, "flange")):
        if not roles.get(role) or role not in frames or frames[role].parent != parent:
            raise ValueError("bridge roles must match base-fixed/flange-wrist topology")
    if roles[policy.fixed_role] == roles[policy.wrist_role] or policy.training + policy.holdout > collection.max_frames:
        raise ValueError("paired owners/count exceed acquisition declaration")
    return policy.to_dict()


def wrist_dependency(registry: Any, spec: Any, policy: BridgePolicy, *, node_session_id: str) -> dict[str, Any]:
    """Resolve actual versioned records; approval attests freshness, not fit quality."""
    from leapflow.hardware.skills.geometry import Transform
    from leapflow.hardware.calibration.measurements import si_transform

    store = registry.calibration_store
    record = store.latest(spec.workcell_id, spec.frames[policy.wrist_role].procedure) if store else None
    verification = store.latest(spec.workcell_id, f"{policy.wrist_role}.verify") if store else None
    if record is None or verification is None or digest_of(vars(record)) != policy.wrist_record_digest or digest_of(vars(verification)) != policy.wrist_verification_digest:
        raise ValueError("wrist calibration or verification changed/missing; refresh before bridge acquisition")
    parameters, checked = record.parameters, verification.parameters
    if parameters.get("mounting") != "eye_in_hand" or parameters.get("camera_role") != policy.wrist_role or parameters.get("camera_device") != spec.role(policy.wrist_role) or parameters.get("target_mount") is None or parameters.get("units") != "mm" or parameters.get("orientation_convention") != "Rz@Ry@Rx;degrees":
        raise ValueError("wrist record is not a measured eye-in-hand flange mount")
    version = checked.get("depends_on", {}).get(policy.wrist_role)
    if version != record.recorded_at or checked.get("training_disjoint") is not True or checked.get("camera_role") != policy.wrist_role:
        raise ValueError("wrist verification does not validate this exact mount on withheld poses")
    for name, limit in (("p95_mm", "max_p95_mm"), ("rms_mm", "max_rms_mm")):
        value, cap = checked.get(name), checked.get(limit)
        declared_cap = spec.calibration[f"max_verify_{name}"]
        if type(value) not in (int, float) or type(cap) not in (int, float) or not math.isfinite(value) or not math.isfinite(cap) or not 0 <= value <= cap <= declared_cap or cap <= 0:
            raise ValueError("wrist record has not passed its declared withheld verification")
    if type(checked.get("points")) is not int or checked["points"] < spec.calibration["min_verify_points"]:
        raise ValueError("wrist verification has insufficient withheld geometry")
    if not isinstance(parameters.get("intrinsics"), Mapping) or not parameters["intrinsics"]:
        raise ValueError("wrist calibration intrinsics unavailable")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (record.recorded_at, verification.recorded_at)) or verification.recorded_at < record.recorded_at:
        raise ValueError("wrist calibration/verification ordering invalid")
    fit = parameters.get("fit", {})
    translation, rotation = fit.get("translation_rms_mm"), fit.get("rotation_rms_deg")
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= cap for v, cap in
           ((translation, spec.calibration["max_hand_eye_translation_rms_mm"]),
            (rotation, spec.calibration["max_hand_eye_rotation_rms_deg"]))):
        raise ValueError("wrist fit uncertainty floor unavailable or above declared validation limits")
    floor = policy.budget.terms["wrist_mount"]
    if floor[0] < max(translation, checked["rms_mm"]) / 1000 or floor[1] < math.radians(rotation):
        raise ValueError("bridge budget understates measured wrist dependency residual floors")
    if policy.freshness["node_session_id"] != node_session_id:
        raise ValueError("owner restarted; reconfirm mounting freshness for this acquisition session")
    transform = si_transform(Transform.from_matrix(record.matrix), parent="flange", child="wrist")
    lineage = [parameters.get("source_ids", []), checked.get("source_ids", [])]
    if any(not isinstance(ids, (list, tuple)) for ids in lineage):
        raise ValueError("invalid wrist refinement source lineage")
    sources = [source for ids in lineage for source in ids]
    if len(sources) > 8192 or any(not isinstance(v, str) or not 0 < len(v) <= 256 for v in sources) or len(set(sources)) != len(sources):
        raise ValueError("invalid wrist refinement source lineage")
    return {"record_digest": policy.wrist_record_digest, "verification_digest": policy.wrist_verification_digest,
            "transform": transform.to_dict(), "recorded_at": record.recorded_at,
            "freshness": dict(policy.freshness), "source_ids": sources,
            "translation_floor_m": max(translation, checked["rms_mm"], checked["p95_mm"]) / 1000,
            "rotation_floor_rad": math.radians(rotation), "intrinsics_digest": digest_of(parameters["intrinsics"]),
            "source_disjointness_basis": "historical_wrist_refinement_windows_unverified; conditional_prior_only",
            "freshness_scope": "operator_confirmed_camera_relative_to_flange; conditional_prior",
            "independent_absolute_accuracy": False}
