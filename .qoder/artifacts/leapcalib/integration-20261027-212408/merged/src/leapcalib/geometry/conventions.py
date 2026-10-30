# Copyright (c) Alibaba, Inc. and its affiliates.
"""Raw firmware coordinates and corrected physical coordinates, kept apart by type.

Software calibration creates two models of the same arm:

**Nominal command model** -- raw joint feedback in firmware conventions. This is
what the CAN bus reports, what joint channels accept, and what LeapFlow's
``DhChain.agreement`` check compares against the firmware's own flange pose. A
corrected model will *intentionally* disagree with that check, which is exactly
why the check must keep comparing nominal against nominal rather than being
loosened to accommodate a correction.

**Corrected observation model** -- the same arm with accepted software offsets
applied, which is where a residual, a clearance check, or a plan should reason.

The offset between them is one addition, and every expensive mistake here is an
addition done the wrong number of times: twice in feedback, zero times in a
command, once in the adapter and again in an action converter. Comments do not
prevent that; the type system can. A :class:`JointVector` knows which space it
is in, so :func:`to_physical` refuses a vector that is already physical and
:func:`to_raw` refuses one that is already raw. Applying a correction twice stops
being a plausible bug and becomes an exception.

Nothing here commands anything. Converting a target into raw coordinates
produces numbers; whether those numbers may be sent is decided by the governed
motion path that owns the lease, the approval and the deadman.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from leapcalib.contracts.failures import (
    FAILURE_UNSUPPORTED_PARAMETER,
    CalibrationFailure,
)
from leapcalib.contracts.identity import ModelIdentity, digest_of
from leapcalib.contracts.units import canonical_number

SPACE_RAW = "raw"
"""Firmware coordinates: what the encoders report and what a joint channel accepts."""
SPACE_PHYSICAL = "physical"
"""Corrected coordinates: raw plus accepted software offsets."""
JOINT_SPACES = frozenset({SPACE_RAW, SPACE_PHYSICAL})


@dataclass(frozen=True)
class JointVector:
    """A joint-angle vector in radians that knows which model it belongs to."""

    values_rad: tuple[float, ...]
    space: str
    model_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "values_rad", tuple(canonical_number(v) for v in self.values_rad))
        if not self.values_rad:
            raise ValueError("a joint vector needs at least one joint")
        if self.space not in JOINT_SPACES:
            raise ValueError(f"joint space must be one of {sorted(JOINT_SPACES)}")

    @property
    def dof(self) -> int:
        return len(self.values_rad)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values_rad": list(self.values_rad),
            "space": self.space,
            "model_digest": self.model_digest,
        }


@dataclass(frozen=True)
class JointOffsets:
    """Per-joint software zero offsets, with the joints they are actually supported on.

    ``supported_joints`` is the subset the evidence established. An offset on an
    unsupported joint is kept at zero so that "we did not establish this" and
    "this is zero" are not the same statement made by accident -- the constructor
    refuses a nonzero offset on an unsupported joint.
    """

    values_rad: tuple[float, ...]
    model: ModelIdentity
    supported_joints: tuple[int, ...] = ()
    method_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "values_rad", tuple(canonical_number(v) for v in self.values_rad))
        if len(self.values_rad) != self.model.dof:
            raise ValueError(
                f"model {self.model.model_id} has {self.model.dof} joints, "
                f"got {len(self.values_rad)} offsets"
            )
        indices = tuple(sorted(set(self.supported_joints)))
        if len(indices) != len(tuple(self.supported_joints)):
            raise ValueError("supported joints must be unique")
        if any(i < 0 or i >= self.model.dof for i in indices):
            raise ValueError("supported joints must index the model")
        object.__setattr__(self, "supported_joints", indices)
        for index, offset in enumerate(self.values_rad):
            if offset != 0.0 and index not in indices:
                raise ValueError(f"joint {index} carries a nonzero offset but is not supported")

    @classmethod
    def zero(cls, model: ModelIdentity) -> "JointOffsets":
        """The no-correction offsets: the nominal model, stated explicitly."""
        return cls(values_rad=tuple([0.0] * model.dof), model=model)

    @property
    def is_zero(self) -> bool:
        return not any(self.values_rad)

    @property
    def digest(self) -> str:
        return digest_of(
            {
                "values_rad": list(self.values_rad),
                "model": self.model.to_dict(),
                "supported_joints": list(self.supported_joints),
                "method_id": self.method_id,
            }
        )

    def require_activatable(self, *, command_path_qualified: bool) -> None:
        """Refuse to be applied globally unless both preconditions actually hold.

        Two independent things must be true before corrected coordinates may
        drive real motion: every joint carrying a correction is supported by
        evidence, and the dual-model command and safety path has itself been
        qualified end to end.

        LeapCalib cannot verify the second -- it owns no command path -- so the
        caller asserts it and owns that assertion. P0 has no qualified command
        path at all, which is why this exists before anything can call it with a
        truthful ``True``.
        """
        unsupported = [
            index
            for index, offset in enumerate(self.values_rad)
            if offset != 0.0 and index not in self.supported_joints
        ]
        if unsupported:
            raise CalibrationFailure(
                FAILURE_UNSUPPORTED_PARAMETER,
                f"joints {unsupported} carry unsupported corrections",
            )
        if not command_path_qualified:
            raise CalibrationFailure(
                FAILURE_UNSUPPORTED_PARAMETER,
                "the raw/corrected command and safety path is not qualified",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "values_rad": list(self.values_rad),
            "model": self.model.to_dict(),
            "supported_joints": list(self.supported_joints),
            "method_id": self.method_id,
        }


def to_physical(raw: JointVector, offsets: JointOffsets) -> JointVector:
    """``q_physical = q_raw + delta_q``.

    Refuses a vector that is already in physical space, which is the double-apply
    bug this module exists to make impossible.
    """
    _require_space(raw, SPACE_RAW, "converting to physical coordinates")
    _require_model(raw, offsets)
    return JointVector(
        values_rad=tuple(q + d for q, d in zip(raw.values_rad, offsets.values_rad)),
        space=SPACE_PHYSICAL,
        model_digest=offsets.model.digest,
    )


def to_raw(physical: JointVector, offsets: JointOffsets) -> JointVector:
    """``q_raw = q_physical - delta_q``, the exact inverse of :func:`to_physical`.

    This is the command adapter for a pure zero-offset correction. General
    geometry corrections need inverse kinematics against the corrected model,
    which is a later stage: handing a corrected Cartesian target to nominal
    firmware IK would discard the correction while appearing to honour it.
    """
    _require_space(physical, SPACE_PHYSICAL, "converting to raw coordinates")
    _require_model(physical, offsets)
    return JointVector(
        values_rad=tuple(q - d for q, d in zip(physical.values_rad, offsets.values_rad)),
        space=SPACE_RAW,
        model_digest=offsets.model.digest,
    )


def _require_space(vector: JointVector, expected: str, doing: str) -> None:
    if vector.space != expected:
        raise CalibrationFailure(
            FAILURE_UNSUPPORTED_PARAMETER,
            f"{doing}: expected a {expected} vector, got {vector.space}",
        )


def _require_model(vector: JointVector, offsets: JointOffsets) -> None:
    if vector.dof != offsets.model.dof:
        raise ValueError(f"offsets are for {offsets.model.dof} joints, vector has {vector.dof}")
    if vector.model_digest and vector.model_digest != offsets.model.digest:
        raise CalibrationFailure(
            FAILURE_UNSUPPORTED_PARAMETER,
            "joint vector and offsets were formed against different kinematic models",
        )


@dataclass(frozen=True)
class JointLimits:
    """Per-joint ``(low, high)`` bounds in radians, in one named space.

    Limits live in raw space, because that is where the firmware enforces them.
    Checking a corrected target therefore means converting back to raw first --
    not widening the raw bounds to fit the corrected number.
    """

    bounds_rad: tuple[tuple[float, float], ...]
    space: str = SPACE_RAW
    source: str = "declared"

    def __post_init__(self) -> None:
        bounds = []
        for low, high in self.bounds_rad:
            lo, hi = canonical_number(low), canonical_number(high)
            if hi < lo:
                raise ValueError("a joint bound must be increasing")
            bounds.append((lo, hi))
        object.__setattr__(self, "bounds_rad", tuple(bounds))
        if not self.bounds_rad:
            raise ValueError("joint limits need at least one joint")
        if self.space not in JOINT_SPACES:
            raise ValueError(f"joint space must be one of {sorted(JOINT_SPACES)}")

    def intersect(self, other: "JointLimits") -> "JointLimits":
        """The narrower of two limit sets, joint by joint.

        A declaration, a vendor manual and the live firmware readback each have
        their own bounds, and the firmware's are authoritative but not always the
        narrowest. The intersection is the only set that satisfies all of them;
        an empty intersection is a contradiction to surface, not to clamp away.
        """
        if self.space != other.space:
            raise ValueError("cannot intersect limits from different joint spaces")
        if len(self.bounds_rad) != len(other.bounds_rad):
            raise ValueError("cannot intersect limits with different joint counts")
        merged = []
        for index, ((alo, ahi), (blo, bhi)) in enumerate(zip(self.bounds_rad, other.bounds_rad)):
            low, high = max(alo, blo), min(ahi, bhi)
            if high < low:
                raise ValueError(f"joint {index} limits do not overlap")
            merged.append((low, high))
        return JointLimits(
            bounds_rad=tuple(merged),
            space=self.space,
            source=f"{self.source}+{other.source}",
        )

    def violation(self, vector: JointVector, *, margin_rad: float = 0.0) -> str:
        """Why *vector* is outside these limits (``""`` when it is inside).

        *margin_rad* is where geometric uncertainty belongs: a target that only
        fits the envelope if the model is exactly right does not fit it.
        """
        if vector.space != self.space:
            raise CalibrationFailure(
                FAILURE_UNSUPPORTED_PARAMETER,
                f"limits are in {self.space} space, vector is in {vector.space}",
            )
        if vector.dof != len(self.bounds_rad):
            raise ValueError("limit and vector joint counts differ")
        if not math.isfinite(margin_rad) or margin_rad < 0:
            raise ValueError("an uncertainty margin must be finite and nonnegative")
        for index, (value, (low, high)) in enumerate(zip(vector.values_rad, self.bounds_rad)):
            if value < low + margin_rad or value > high - margin_rad:
                return (
                    f"joint {index} at {value:.6f} rad is outside "
                    f"[{low:.6f}, {high:.6f}] with margin {margin_rad:.6f}"
                )
        return ""

    @classmethod
    def from_degrees(
        cls, bounds_deg: Sequence[Sequence[float]], *, source: str = "declared"
    ) -> "JointLimits":
        """Build from a declaration in degrees, which is how LeapFlow states them."""
        return cls(
            bounds_rad=tuple(
                (math.radians(float(low)), math.radians(float(high))) for low, high in bounds_deg
            ),
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bounds_rad": [list(b) for b in self.bounds_rad],
            "space": self.space,
            "source": self.source,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "JointLimits":
        return cls(
            bounds_rad=tuple((float(b[0]), float(b[1])) for b in data["bounds_rad"]),
            space=str(data.get("space") or SPACE_RAW),
            source=str(data.get("source") or "declared"),
        )


__all__ = [
    "JOINT_SPACES",
    "SPACE_PHYSICAL",
    "SPACE_RAW",
    "JointLimits",
    "JointOffsets",
    "JointVector",
    "to_physical",
    "to_raw",
]
