# Copyright (c) Alibaba, Inc. and its affiliates.
"""Immutable raw-coordinate physical model and uncertainty-inflated dual-model sweep."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from leapcalib.contracts.identity import digest_of
from leapcalib.contracts.motion import finite_vector
from leapcalib.geometry.se3 import Transform as SiTransform
from leapflow.hardware.skills.geometry import Pose, Transform, WorkspaceBox
from leapflow.hardware.skills.kinematics import DhChain


def _hardware(transform: SiTransform) -> Transform:
    return Transform(transform.rotation, tuple(v * 1000 for v in transform.translation_m))


@dataclass(frozen=True)
class PhysicalModel:
    nominal: DhChain
    zero_offsets_rad: tuple[float, ...]
    tool_transform: SiTransform
    nominal_digest: str
    dependency_digest: str
    certificate_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.nominal, DhChain) or not isinstance(self.tool_transform, SiTransform):
            raise ValueError("physical model needs frozen nominal DH and validated SI transform")
        # Reparse even directly constructed chains: frozen dataclasses can contain
        # caller-owned mutable link containers or unvalidated numeric values.
        object.__setattr__(self, "nominal", DhChain.from_mapping(self.nominal.to_dict()))
        object.__setattr__(self, "zero_offsets_rad", finite_vector(self.zero_offsets_rad, self.nominal.dof))
        if self.nominal_digest != digest_of(self.nominal.to_dict()):
            raise ValueError("physical model nominal digest differs from immutable chain")
        if not self.dependency_digest or not self.certificate_digest:
            raise ValueError("physical model requires dependency and certificate bindings")

    @property
    def dof(self) -> int:
        return self.nominal.dof

    def raw_to_physical_deg(self, raw_deg: Sequence[float]) -> tuple[float, ...]:
        values = finite_vector(raw_deg, self.dof)
        return tuple(raw + math.degrees(offset) for raw, offset in zip(values, self.zero_offsets_rad))

    def flange_transform(self, raw_deg: Sequence[float]) -> Transform:
        return self.nominal.transform(self.raw_to_physical_deg(raw_deg))

    def transform(self, joints_deg: Sequence[float]) -> Transform:
        """Corrected tool FK as a function of RAW degrees; IK returns RAW degrees."""
        return self.flange_transform(joints_deg).compose(_hardware(self.tool_transform))

    def forward(self, raw_deg: Sequence[float]) -> Pose:
        return self.transform(raw_deg).to_pose()

    def link_transforms(self, raw_deg: Sequence[float]) -> tuple[Transform, ...]:
        transforms = [Transform()]
        for link, q in zip(self.nominal.links, self.raw_to_physical_deg(raw_deg)):
            transforms.append(transforms[-1].compose(link.transform(q, self.nominal.convention)))
        transforms.append(transforms[-1].compose(_hardware(self.tool_transform)))
        return tuple(transforms)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "leapflow.physical_model.v1", "nominal": self.nominal.to_dict(),
                "zero_offsets_rad": list(self.zero_offsets_rad), "tool_transform": self.tool_transform.to_dict(),
                "nominal_digest": self.nominal_digest, "dependency_digest": self.dependency_digest,
                "certificate_digest": self.certificate_digest, "coordinates": "raw_deg", "composition": "raw_plus_zero_once;right_tool_SI"}

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PhysicalModel:
        expected = {"schema", "nominal", "zero_offsets_rad", "tool_transform", "nominal_digest", "dependency_digest", "certificate_digest", "coordinates", "composition"}
        if set(data) != expected or data.get("schema") != "leapflow.physical_model.v1" or data.get("coordinates") != "raw_deg" or data.get("composition") != "raw_plus_zero_once;right_tool_SI":
            raise ValueError("incompatible physical model coordinates/composition")
        return cls(DhChain.from_mapping(data["nominal"]), tuple(data["zero_offsets_rad"]), SiTransform.from_mapping(data["tool_transform"]), data["nominal_digest"], data["dependency_digest"], data["certificate_digest"])


@dataclass(frozen=True)
class SafetyGeometry:
    """Independently verified outer capsule bounds, never fabricated native meshes.

    The box is a declared whole-link safe volume, NOT an existing TCP workspace.
    A capsule per link segment and one tool capsule conservatively encloses material.
    """
    link_radii_mm: tuple[float, ...]
    whole_link_workspace: WorkspaceBox
    keepouts: tuple[WorkspaceBox, ...]
    provenance_digest: str
    uncertainty_mm: float
    sensing_mm: float
    timing_mm: float
    arrival_mm: float
    minimum_clearance_mm: float = 30.0
    boundary_margin_mm: float = 20.0
    path_step_deg: float = 0.5
    max_path_samples: int = 4096

    def __post_init__(self) -> None:
        object.__setattr__(self, "link_radii_mm", finite_vector(self.link_radii_mm, len(self.link_radii_mm)))
        object.__setattr__(self, "keepouts", tuple(self.keepouts))
        if not self.link_radii_mm or any(v <= 0 for v in self.link_radii_mm) or not self.provenance_digest:
            raise ValueError("complete independent link/tool bounds and provenance are required")
        if not isinstance(self.whole_link_workspace, WorkspaceBox) or any(not isinstance(box, WorkspaceBox) for box in self.keepouts):
            raise ValueError("safety geometry needs typed whole-link safe volume and keepouts")
        for name in ("uncertainty_mm", "sensing_mm", "timing_mm", "arrival_mm", "minimum_clearance_mm", "boundary_margin_mm", "path_step_deg"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("safety uncertainties and bounds must be finite and nonnegative")
        if self.path_step_deg <= 0 or type(self.max_path_samples) is not int or not 2 <= self.max_path_samples <= 4096:
            raise ValueError("swept-path sampling budgets must be positive and bounded")

    @property
    def uncertainty_total_mm(self) -> float:
        return self.uncertainty_mm + self.sensing_mm + self.timing_mm + self.arrival_mm

    def to_dict(self) -> dict[str, Any]:
        return {**vars(self), "link_radii_mm": list(self.link_radii_mm), "whole_link_workspace": self.whole_link_workspace.to_dict(), "keepouts": [b.to_dict() for b in self.keepouts]}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SafetyGeometry:
        return cls(**{**data, "whole_link_workspace": WorkspaceBox.from_mapping(data["whole_link_workspace"]), "keepouts": tuple(WorkspaceBox.from_mapping(b) for b in data["keepouts"])})


def _box_distance(first: Sequence[tuple[float, float]], second: WorkspaceBox) -> float:
    bounds = ((second.x_min, second.x_max), (second.y_min, second.y_max), (second.z_min, second.z_max))
    return math.sqrt(sum(max(blo - ahi, alo - bhi, 0)**2 for (alo, ahi), (blo, bhi) in zip(first, bounds)))


def dual_model_reach_mm(nominal: PhysicalModel, corrected: PhysicalModel, geometry: SafetyGeometry) -> float:
    """Conservative reach of any material point for angular uncertainty inflation."""
    return (sum(abs(link.a) + abs(link.d) for link in nominal.nominal.links)
            + max(math.sqrt(sum(v*v for v in model.tool_transform.translation_m)) * 1000 for model in (nominal, corrected))
            + max(geometry.link_radii_mm))


def check_dual_sweep(nominal: PhysicalModel, corrected: PhysicalModel, current_raw_deg: Sequence[float], target_raw_deg: Sequence[float], geometry: SafetyGeometry) -> dict[str, Any]:
    """Conservative capsule-AABB checks at raw interpolation with analytic sweep inflation."""
    if nominal.nominal_digest != corrected.nominal_digest or nominal.dof != corrected.dof:
        raise ValueError("dual safety models have different nominal identities")
    current = finite_vector(current_raw_deg, nominal.dof)
    target = finite_vector(target_raw_deg, nominal.dof)
    if len(geometry.link_radii_mm) != nominal.dof + 1:
        raise ValueError("independently bound geometry must cover every link and tool")
    deltas = tuple(b - a for a, b in zip(current, target))
    steps = max(1, math.ceil(max(abs(v) for v in deltas) / geometry.path_step_deg))
    if steps + 1 > geometry.max_path_samples:
        raise ValueError("dual-model sweep exceeds bounded sampling work")
    reach = dual_model_reach_mm(nominal, corrected, geometry)
    # Any point can move at most reach * L1 joint rotation between samples.
    discretization = reach * sum(math.radians(abs(v)) for v in deltas) / (2 * steps)
    inflation = geometry.uncertainty_total_mm + discretization
    minimum = float("inf")
    for model in (nominal, corrected):
        for step in range(steps + 1):
            q = tuple(a + delta * step / steps for a, delta in zip(current, deltas))
            transforms = model.link_transforms(q)
            for index, radius in enumerate(geometry.link_radii_mm):
                a, b = transforms[index].translation, transforms[index + 1].translation
                margin = radius + inflation
                if any(not geometry.whole_link_workspace.contains(p, margin=margin + geometry.boundary_margin_mm) for p in (a, b)):
                    raise ValueError("uncertainty-inflated link/tool sweep leaves verified whole-link boundary")
                box = tuple((min(x, y) - margin, max(x, y) + margin) for x, y in zip(a, b))
                for keepout in geometry.keepouts:
                    clearance = _box_distance(box, keepout)
                    minimum = min(minimum, clearance)
                    if clearance < geometry.minimum_clearance_mm:
                        raise ValueError("dual-model link/tool sweep has insufficient residual non-contact clearance")
    return {"nominal_model_digest": nominal.digest, "corrected_model_digest": corrected.digest,
            "geometry_digest": digest_of(geometry.to_dict()), "samples_per_model": steps + 1,
            "discretization_mm": discretization, "uncertainty_mm": inflation,
            "minimum_keepout_clearance_mm": minimum if math.isfinite(minimum) else None}
