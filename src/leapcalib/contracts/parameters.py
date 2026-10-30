# Copyright (c) Alibaba, Inc. and its affiliates.
"""Parameter blocks: what may be estimated, in what units, composed on which side.

A parameter vector is not self-describing. The same six numbers are a camera pose
in a base frame or a base pose in a camera frame; the same three-element offset
is a correction applied before a nominal transform or after it. Get the direction
or the side wrong and the solve still converges -- to a mirror of the truth, with
a small residual and a wrong sign on the correction that reaches a safety check.

So a block declares its dimension, the unit of each component, the frames it
relates, which side it composes on, and what must already be fixed before it can
be estimated (``prerequisites``). The staged solving this package is built around
depends on those prerequisites being data, not comments: joint zeros are
estimated before a wrist mount precisely so a weak camera transform cannot absorb
joint-zero error, and a block that lists a prerequisite cannot be solved while
that prerequisite is still free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from leapcalib.contracts.identity import digest_of
from leapcalib.contracts.units import SI_UNITS, UNIT_METRE, UNIT_RADIAN, UNIT_SECOND

KIND_CAMERA_EXTRINSIC = "camera_extrinsic"
"""A camera pose relative to a link: 6-DoF, the fixed-camera and wrist-camera mounts."""
KIND_JOINT_ZERO = "joint_zero"
"""Per-joint angular offsets between firmware zero and physical zero."""
KIND_TOOL_TRANSFORM = "tool_transform"
"""A tool reference pose relative to the flange."""
KIND_DEPTH_SCALE = "depth_scale"
"""A multiplicative correction on a stream's metres-per-depth-unit."""
KIND_TIME_OFFSET = "time_offset"
"""A constant shift between a camera clock and the robot clock."""
KIND_TASK_RESPONSE = "task_response"
"""A task-local feature-to-joint response; valid in a declared domain, never global geometry."""
PARAMETER_KINDS = frozenset(
    {
        KIND_CAMERA_EXTRINSIC,
        KIND_JOINT_ZERO,
        KIND_TOOL_TRANSFORM,
        KIND_DEPTH_SCALE,
        KIND_TIME_OFFSET,
        KIND_TASK_RESPONSE,
    }
)

COMPOSE_LEFT = "left"
"""The correction pre-multiplies the nominal transform: ``T = dT @ T_nominal``."""
COMPOSE_RIGHT = "right"
"""The correction post-multiplies it: ``T = T_nominal @ dT``."""
COMPOSE_ADDITIVE = "additive"
"""The correction adds in the parameter's own space, as a joint offset does."""
COMPOSITION_SIDES = frozenset({COMPOSE_LEFT, COMPOSE_RIGHT, COMPOSE_ADDITIVE})

SUPPORT_SUPPORTED = "supported"
"""Every component is established by the observations."""
SUPPORT_PARTIAL = "partial"
"""Some components are established; the rest keep their priors."""
SUPPORT_RETAINED = "retained"
"""Nothing is established; the declared prior is kept unchanged and said to be kept."""
SUPPORT_INSUFFICIENT = "insufficient"
"""The evidence did not even permit the attempt, so the prior stands untested."""
SUPPORT_STATES = frozenset(
    {SUPPORT_SUPPORTED, SUPPORT_PARTIAL, SUPPORT_RETAINED, SUPPORT_INSUFFICIENT}
)

PRIOR_ONLY_STATES = frozenset({SUPPORT_RETAINED, SUPPORT_INSUFFICIENT})
"""States in which the estimate must still *be* the prior, bit for bit."""

# Component units per kind, so a block cannot declare a dimension its kind
# contradicts. Camera and tool transforms are (x, y, z, rx, ry, rz) in metres
# and radians; a joint-zero block is one radian per joint, so its dimension is
# the model's dof and is supplied by the caller.
_KIND_UNITS: dict[str, tuple[str, ...]] = {
    KIND_CAMERA_EXTRINSIC: (
        UNIT_METRE,
        UNIT_METRE,
        UNIT_METRE,
        UNIT_RADIAN,
        UNIT_RADIAN,
        UNIT_RADIAN,
    ),
    KIND_TOOL_TRANSFORM: (
        UNIT_METRE,
        UNIT_METRE,
        UNIT_METRE,
        UNIT_RADIAN,
        UNIT_RADIAN,
        UNIT_RADIAN,
    ),
}


@dataclass(frozen=True)
class ParameterBlock:
    """One estimable group of parameters, fully described rather than conventional."""

    block_id: str
    kind: str
    dimension: int
    units: tuple[str, ...]
    parent_frame: str = ""
    child_frame: str = ""
    composition: str = COMPOSE_RIGHT
    prerequisites: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.block_id.strip():
            raise ValueError("a parameter block needs a nonempty id")
        if self.kind not in PARAMETER_KINDS:
            raise ValueError(f"parameter kind must be one of {sorted(PARAMETER_KINDS)}")
        if type(self.dimension) is not int or self.dimension < 1:
            raise ValueError("a parameter block needs a positive integer dimension")
        object.__setattr__(self, "units", tuple(self.units))
        if len(self.units) != self.dimension:
            raise ValueError("every parameter component declares its unit")
        unknown = set(self.units) - SI_UNITS - {""}
        if unknown:
            raise ValueError(f"parameter units must be SI, got {sorted(unknown)}")
        expected = _KIND_UNITS.get(self.kind)
        if expected is not None and self.units != expected:
            raise ValueError(f"a {self.kind} block is {expected}")
        if self.composition not in COMPOSITION_SIDES:
            raise ValueError(f"composition must be one of {sorted(COMPOSITION_SIDES)}")
        if self.kind == KIND_JOINT_ZERO and set(self.units) != {UNIT_RADIAN}:
            raise ValueError("joint-zero offsets are radians")
        if self.kind == KIND_TIME_OFFSET and set(self.units) != {UNIT_SECOND}:
            raise ValueError("a time offset is seconds")
        if self.kind in (KIND_CAMERA_EXTRINSIC, KIND_TOOL_TRANSFORM) and not (
            self.parent_frame.strip() and self.child_frame.strip()
        ):
            raise ValueError(f"a {self.kind} block must name both frames it relates")
        object.__setattr__(self, "prerequisites", tuple(self.prerequisites))
        if self.block_id in self.prerequisites:
            raise ValueError("a block cannot be its own prerequisite")

    @property
    def transform_name(self) -> str:
        """``T_parent_child``, so the direction is readable wherever it is printed."""
        return f"T_{self.parent_frame}_{self.child_frame}" if self.parent_frame else self.block_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "kind": self.kind,
            "dimension": self.dimension,
            "units": list(self.units),
            "parent_frame": self.parent_frame,
            "child_frame": self.child_frame,
            "composition": self.composition,
            "prerequisites": list(self.prerequisites),
            "notes": self.notes,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ParameterBlock":
        if not isinstance(data, Mapping) or set(data) - set(cls.__dataclass_fields__):
            raise ValueError("unknown parameter block fields")
        dimension = data.get("dimension")
        if type(dimension) is not int or not 1 <= dimension <= 1024:
            raise ValueError("bounded integer parameter dimension required before decoding")
        for name in ("units", "prerequisites"):
            values = data.get(name, ())
            if not isinstance(values, (list, tuple)) or len(values) > 1024:
                raise ValueError("bounded typed parameter array required")
            if any(not isinstance(value, str) or len(value) > 4096 for value in values):
                raise ValueError("parameter array values must be typed strings")
        for name in ("block_id", "kind", "parent_frame", "child_frame", "composition", "notes"):
            value = data.get(name, "")
            if not isinstance(value, str) or len(value) > 4096:
                raise ValueError("parameter identity and convention must be typed strings")
        return cls(
            block_id=data["block_id"],
            kind=data["kind"],
            dimension=dimension,
            units=tuple(data["units"]),
            parent_frame=data.get("parent_frame", ""),
            child_frame=data.get("child_frame", ""),
            composition=data.get("composition", COMPOSE_RIGHT),
            prerequisites=tuple(data.get("prerequisites", ())),
            notes=data.get("notes", ""),
        )

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())


def camera_extrinsic_block(
    block_id: str, *, parent_frame: str, child_frame: str, prerequisites: tuple[str, ...] = ()
) -> ParameterBlock:
    """A 6-DoF camera mount block, composed on the right of the nominal mount."""
    return ParameterBlock(
        block_id=block_id,
        kind=KIND_CAMERA_EXTRINSIC,
        dimension=6,
        units=_KIND_UNITS[KIND_CAMERA_EXTRINSIC],
        parent_frame=parent_frame,
        child_frame=child_frame,
        composition=COMPOSE_RIGHT,
        prerequisites=prerequisites,
    )


def joint_zero_block(
    block_id: str, *, dof: int, prerequisites: tuple[str, ...] = ()
) -> ParameterBlock:
    """A per-joint additive offset block in radians, one component per joint."""
    return ParameterBlock(
        block_id=block_id,
        kind=KIND_JOINT_ZERO,
        dimension=dof,
        units=tuple([UNIT_RADIAN] * dof),
        composition=COMPOSE_ADDITIVE,
        prerequisites=prerequisites,
    )


def ordered_for_solve(blocks: tuple[ParameterBlock, ...]) -> tuple[ParameterBlock, ...]:
    """*blocks* in an order that satisfies every prerequisite, or a refusal.

    Staging is the method's substance, so an impossible stage order is an error
    rather than a warning: a cycle, or a prerequisite that is not in the set at
    all, means the caller has not decided what is held fixed.
    """
    remaining = list(blocks)
    known = {b.block_id for b in blocks}
    for block in blocks:
        missing = set(block.prerequisites) - known
        if missing:
            raise ValueError(f"{block.block_id} needs absent prerequisites {sorted(missing)}")
    ordered: list[ParameterBlock] = []
    placed: set[str] = set()
    while remaining:
        ready = [b for b in remaining if set(b.prerequisites) <= placed]
        if not ready:
            stuck = sorted(b.block_id for b in remaining)
            raise ValueError(f"circular parameter prerequisites among {stuck}")
        ready.sort(key=lambda b: b.block_id)
        ordered.extend(ready)
        placed.update(b.block_id for b in ready)
        remaining = [b for b in remaining if b.block_id not in placed]
    return tuple(ordered)


__all__ = [
    "COMPOSE_ADDITIVE",
    "COMPOSE_LEFT",
    "COMPOSE_RIGHT",
    "COMPOSITION_SIDES",
    "KIND_CAMERA_EXTRINSIC",
    "KIND_DEPTH_SCALE",
    "KIND_JOINT_ZERO",
    "KIND_TASK_RESPONSE",
    "KIND_TIME_OFFSET",
    "KIND_TOOL_TRANSFORM",
    "PARAMETER_KINDS",
    "PRIOR_ONLY_STATES",
    "SUPPORT_INSUFFICIENT",
    "SUPPORT_PARTIAL",
    "SUPPORT_RETAINED",
    "SUPPORT_STATES",
    "SUPPORT_SUPPORTED",
    "ParameterBlock",
    "camera_extrinsic_block",
    "joint_zero_block",
    "ordered_for_solve",
]
