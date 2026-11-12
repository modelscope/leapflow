# Copyright (c) Alibaba, Inc. and its affiliates.
"""Declared nominal-only acquisition envelope and conservative swept-path checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Sequence

from leapcalib.contracts.identity import MountIdentity
from leapcalib.contracts.motion import finite_vector
from leapflow.hardware.calibration.physical_model import (
    PhysicalModel,
    SafetyGeometry,
    check_dual_sweep,
    dual_model_reach_mm,
)
from leapflow.hardware.calibration.acquisition import SettledProfile

# Absolute sanity ceilings on a declared campaign envelope.  These are not the
# campaign authority -- the approved preview carries the real travel and joint
# bounds -- and they exist only so that a typo cannot authorize an unbounded
# move.  A campaign that needs a wider envelope than the defaults declares it
# explicitly instead of being vetoed by a constant nobody can see.
ENVELOPE_CEILINGS: Mapping[str, float] = {
    "campaign_translation_mm": 500.0,
    "campaign_yaw_deg": 60.0,
    "campaign_joint_deg": 30.0,
    "max_travel_mm": 2000.0,
    "max_joint_l1_deg": 180.0,
    "max_duration_s": 1800.0,
}
MAX_ACQUISITION_MOVES = 16
MAX_ACQUISITION_SPEED_PCT = 10.0


class SweepCheck(NamedTuple):
    """Bounded swept-path geometry plus every envelope miss it observed."""

    travel_mm: float
    joint_l1_deg: float
    advisories: tuple[str, ...]


@dataclass(frozen=True)
class PoseCollectionPolicy:
    reference_acquisition: str
    mount: MountIdentity
    geometry: SafetyGeometry
    joint_rate_bounds_rad_s: tuple[float, ...]
    endpoint_rate_bound_mm_s: float
    rate_provenance_digest: str
    speed_pct_max: float = 10
    campaign_translation_mm: float = 10
    campaign_yaw_deg: float = 3
    campaign_joint_deg: float = 5
    max_moves: int = 8
    max_travel_mm: float = 40
    max_joint_l1_deg: float = 12
    max_duration_s: float = 120

    def __post_init__(self) -> None:
        for identity in (self.reference_acquisition, self.rate_provenance_digest):
            if (
                not isinstance(identity, str)
                or len(identity) != 64
                or any(c not in "0123456789abcdef" for c in identity)
            ):
                raise ValueError(
                    "exact indexed campaign reference and independently measured rate provenance digests required"
                )
        if not isinstance(self.mount, MountIdentity) or self.mount.parent_frame != "flange":
            raise ValueError("camera-only pose acquisition requires an explicit wrist mount")
        if (
            not isinstance(self.geometry, SafetyGeometry)
            or self.geometry.minimum_clearance_mm < 30
            or self.geometry.boundary_margin_mm < 20
        ):
            raise ValueError(
                "verified whole-link clearance >=30mm and boundary margin >=20mm required"
            )
        if not 1 <= len(self.joint_rate_bounds_rad_s) <= 16:
            raise ValueError("bounded complete joint rate vector required")
        object.__setattr__(
            self,
            "joint_rate_bounds_rad_s",
            finite_vector(self.joint_rate_bounds_rad_s, len(self.joint_rate_bounds_rad_s)),
        )
        if any(v <= 0 for v in self.joint_rate_bounds_rad_s):
            raise ValueError("positive independently measured joint rates required")
        # The speed authority of a calibration campaign stays at ten percent: it is
        # the one envelope term a slow arm cannot compensate for.  Everything else
        # is a declared campaign size, bounded only by a sanity ceiling.
        if (
            isinstance(self.speed_pct_max, bool)
            or not isinstance(self.speed_pct_max, (int, float))
            or not math.isfinite(self.speed_pct_max)
            or not 0 < self.speed_pct_max <= MAX_ACQUISITION_SPEED_PCT
        ):
            raise ValueError("acquisition speed above the ten-percent campaign cap")
        for name, ceiling in ENVELOPE_CEILINGS.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= ceiling
            ):
                raise ValueError(f"{name} exceeds the declared-envelope sanity ceiling")
        if type(self.max_moves) is not int or not 1 <= self.max_moves <= MAX_ACQUISITION_MOVES:
            raise ValueError("at most sixteen acquisition moves per run")
        if (
            isinstance(self.endpoint_rate_bound_mm_s, bool)
            or not isinstance(self.endpoint_rate_bound_mm_s, (int, float))
            or not math.isfinite(self.endpoint_rate_bound_mm_s)
            or self.endpoint_rate_bound_mm_s <= 0
        ):
            raise ValueError("positive measured endpoint rate bound required")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PoseCollectionPolicy:
        if not isinstance(data, Mapping) or set(data) - set(cls.__dataclass_fields__):
            raise ValueError("exact calibration_motion policy fields required")
        return cls(
            **{
                **data,
                "mount": MountIdentity(**data["mount"]),
                "geometry": SafetyGeometry.from_mapping(data["geometry"]),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **vars(self),
            "mount": self.mount.to_dict(),
            "geometry": self.geometry.to_dict(),
            "joint_rate_bounds_rad_s": list(self.joint_rate_bounds_rad_s),
        }

    def validate_model(
        self, model: PhysicalModel, settled: SettledProfile, *, fk_mm: float, fk_deg: float
    ) -> tuple[str, ...]:
        """Reject understated whole-link safety inflation before approving motion."""
        if (
            len(self.joint_rate_bounds_rad_s) != model.dof
            or len(self.geometry.link_radii_mm) != model.dof + 1
        ):
            raise ValueError(
                "complete independently bounded arm, camera and tool material required"
            )
        reach = dual_model_reach_mm(model, model, self.geometry)
        sensing = settled.tcp_uncertainty_mm + reach * (
            model.dof * settled.joint_uncertainty_rad
            + math.radians(settled.rotation_uncertainty_deg)
        )
        delay = (
            settled.max_feedback_age_s + settled.feedback_uncertainty_s + settled.max_coherence_s
        )
        timing = delay * (self.endpoint_rate_bound_mm_s + reach * sum(self.joint_rate_bounds_rad_s))
        declared = {
            "sensing": (self.geometry.sensing_mm, sensing),
            "timing": (self.geometry.timing_mm, timing),
            "uncertainty": (self.geometry.uncertainty_mm, fk_mm + reach * math.radians(fk_deg)),
            "arrival": (self.geometry.arrival_mm, settled.max_tcp_drift_mm),
        }
        understated = tuple(
            f"declared {name} bound {given:.1f} mm understates the derived {needed:.1f} mm"
            for name, (given, needed) in declared.items()
            if given < needed
        )
        if understated:
            raise ValueError("whole-link uncertainty understates derived safety bounds: "
                             + "; ".join(understated))
        return ()

    def sweep(
        self,
        model: PhysicalModel,
        reference_deg: Sequence[float],
        before_deg: Sequence[float],
        after_deg: Sequence[float],
    ) -> SweepCheck:
        """Bound campaign position and run travel across the entire nominal interpolation.

        Clearance against the declared safe volume stays a refusal: it is the one
        check that says the arm may hit something.  Campaign-size misses are
        returned as advisories so a single oversized move is reported at the end
        instead of discarding every capture already taken.
        """
        reference = finite_vector(reference_deg, model.dof)
        before = finite_vector(before_deg, model.dof)
        after = finite_vector(after_deg, model.dof)
        advisories: list[str] = []
        outside = [
            abs(q - r) for q, r in zip(after, reference) if abs(q - r) > self.campaign_joint_deg
        ]
        if outside:
            advisories.append(
                f"target leaves the declared {self.campaign_joint_deg:g} deg campaign joint envelope by {max(outside):.1f} deg"
            )
        checked = check_dual_sweep(model, model, before, after, self.geometry)
        steps = checked["samples_per_model"] - 1
        anchor, previous = model.forward(reference), model.forward(before)
        reach = dual_model_reach_mm(model, model, self.geometry)
        joint_l1_rad = sum(math.radians(abs(b - a)) for a, b in zip(before, after))
        between_mm = reach * joint_l1_rad / (2 * steps)
        travel = 0.0
        worst_translation = worst_rotation = 0.0
        for index in range(steps + 1):
            q = tuple(a + (b - a) * index / steps for a, b in zip(before, after))
            pose = model.forward(q)
            yaw = abs((pose.rz - anchor.rz + 180) % 360 - 180)
            # Translation inflation also covers points between samples; enforce
            # the full angular displacement as a conservative yaw bound.
            angle_bound = anchor.rotation_to_deg(pose) + math.degrees(joint_l1_rad / (2 * steps))
            worst_translation = max(
                worst_translation, pose.distance_mm(anchor) + between_mm - self.campaign_translation_mm
            )
            worst_rotation = max(worst_rotation, max(yaw, angle_bound) - self.campaign_yaw_deg)
            travel += pose.distance_mm(previous)
            previous = pose
        travel += reach * joint_l1_rad * joint_l1_rad / (2 * steps)
        if worst_translation > 0:
            advisories.append(
                f"path leaves the declared {self.campaign_translation_mm:g} mm campaign translation envelope by {worst_translation:.1f} mm"
            )
        if worst_rotation > 0:
            advisories.append(
                f"path leaves the declared {self.campaign_yaw_deg:g} deg campaign orientation envelope by {worst_rotation:.1f} deg"
            )
        return SweepCheck(travel, math.degrees(joint_l1_rad), tuple(advisories))
