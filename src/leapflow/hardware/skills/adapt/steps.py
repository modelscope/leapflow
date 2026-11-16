# Copyright (c) Alibaba, Inc. and its affiliates.
"""The steps that turn a policy's native vectors into channel targets, and back.

A chunk is a tuple of rows; a row maps a group name to its values (the flat,
not-yet-split vector lives under ``VECTOR``). Each step rewrites some groups
from one representation to the next, so a pipeline reads top to bottom::

    unnormalize -> map -> units -> gripper -> rotation -> frame -> reference
                -> ik -> <plugin steps> -> validate

After ``reference`` every group is *canonical*: joints in degrees, ``ee_pose``
as the arm's flange pose ``(x, y, z, rx, ry, rz)`` in base (mm, fixed-axis
X-Y-Z degrees), the gripper in millimetres. ``invert`` walks the same steps in
reverse, which is what an observation's state, a pending (RTC) chunk and a
recorded episode's commands use to reach the model's space -- one declaration,
both directions.

Conventions a model's ``relative``/``delta`` values follow:

- joints and gripper add: ``q = anchor + d``.
- an ``ee_pose`` in base (or a workcell frame, rotated into base first) adds the
  position and left-multiplies the rotation: ``p = pa + dp``, ``R = Rd @ Ra``.
- an ``ee_pose`` in ``tool`` composes in the flange's own axes: ``T = Ta @ Td``.
- ``delta`` chains from the previous row; its first row from the anchor.

Steps raise ``AdaptError`` for an answer that cannot become a command and
``AdaptViolation`` for one that could but must not (a first action that jumps).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from leapflow.hardware.skills.adapt.ik import solve_ik
from leapflow.hardware.skills.adapt.rotations import ARM_EULER, ROTATION_NONE, to_matrix
from leapflow.hardware.skills.adapt.rotations import from_matrix as encode_rotation
from leapflow.hardware.skills.adapt.spec import (
    FRAME_BASE,
    FRAME_TOOL,
    KIND_EE_POSE,
    KIND_GRIPPER,
    KIND_JOINTS,
    REFERENCE_ABSOLUTE,
    REFERENCE_DELTA,
    UNIT_NORMALIZED,
    Group,
    IkSettings,
    Normalization,
)
from leapflow.hardware.skills.geometry import (
    Pose,
    Transform,
    mat_mul,
    mat_vec,
    matrix_to_rpy,
    rpy_to_matrix,
    transpose,
    wrap_deg,
)
from leapflow.hardware.skills.kinematics import DhChain

VECTOR = "*"
Values = tuple[float, ...]
Row = Mapping[str, Values]
Chunk = tuple[Row, ...]


class AdaptError(ValueError):
    """The model's answer cannot be turned into a command."""


class AdaptViolation(AdaptError):
    """The answer could be commanded but must not be (a safety refusal)."""


@dataclass(frozen=True)
class StepContext:
    """What one chunk is adapted against: the state at its observation.

    ``anchor`` holds each group's canonical value at the observation (what
    ``relative`` and ``delta`` start from); ``current`` the bound channels'
    values per group, which seed IK and judge the first action's jump.
    ``memory`` belongs to the pipeline and outlives the chunk (a binary
    gripper's last decision).
    """

    anchor: Mapping[str, Values] = field(default_factory=dict)
    current: Mapping[str, Values] = field(default_factory=dict)
    frames: Callable[[str], Transform] | None = None
    gripper_mm: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    max_jump: Mapping[str, tuple[float | None, ...]] = field(default_factory=dict)
    cyclic: Mapping[str, tuple[bool, ...]] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)

    def anchor_of(self, group: str) -> Values:
        value = self.anchor.get(group)
        if value is None:
            raise AdaptError(f"no observed state anchors the {group!r} group")
        return value

    def frame(self, name: str) -> Transform:
        if self.frames is None:
            raise AdaptError(f"frame {name!r} cannot be resolved here")
        try:
            return self.frames(name)
        except (KeyError, ValueError) as exc:
            raise AdaptError(f"frame {name!r} is not known to the workcell: {exc}") from exc

    def gripper_span(self, group: str) -> tuple[float, float]:
        span = self.gripper_mm.get(group)
        if span is None or span[0] == span[1]:
            raise AdaptError(f"the {group!r} gripper has no declared closed/open width")
        return span


@runtime_checkable
class ActionStep(Protocol):
    """One reversible stage of an adapter pipeline."""

    @property
    def name(self) -> str: ...

    def apply(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        """The model-to-robot direction."""
        ...

    def invert(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        """The robot-to-model direction."""
        ...


StepFactory = Callable[[Mapping[str, Any], Sequence[Group]], ActionStep]
"""Builds a plugin step from its ``policies.<name>.steps`` params and the action groups."""


def _rewrite(chunk: Chunk, groups: Sequence[Group], fn: Callable[[Group, Values], Values]) -> Chunk:
    """Apply *fn* to each named group of every row, keeping the others."""
    out: list[Row] = []
    for row in chunk:
        new = dict(row)
        for group in groups:
            if group.name in new:
                new[group.name] = fn(group, new[group.name])
        out.append(new)
    return tuple(out)


def _as_transform(values: Sequence[float]) -> Transform:
    return Transform.from_pose(Pose(*values[:6]))


def _as_values(transform: Transform) -> Values:
    return transform.to_pose().as_tuple()


# ── Vector stages ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UnnormalizeStep:
    """The model's normalised vector to physical units (and back)."""

    normalization: Normalization
    name: str = "unnormalize"

    def apply(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        return tuple({**r, VECTOR: self.normalization.to_physical(r[VECTOR])} for r in chunk)

    def invert(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        return tuple({**r, VECTOR: self.normalization.to_model(r[VECTOR])} for r in chunk)


@dataclass(frozen=True)
class MapStep:
    """The flat vector split into its groups; the inverse leaves undeclared dims at 0."""

    width: int
    groups: tuple[Group, ...]
    name: str = "map"

    def apply(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        out: list[Row] = []
        for row in chunk:
            vector = row[VECTOR]
            if len(vector) != self.width:
                raise AdaptError(f"an action has {len(vector)} values, the space {self.width}")
            out.append({g.name: tuple(float(vector[d]) for d in g.dims) for g in self.groups})
        return tuple(out)

    def invert(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        out: list[Row] = []
        for row in chunk:
            vector = [0.0] * self.width
            for group in self.groups:
                for dim, value in zip(group.dims, row[group.name]):
                    vector[dim] = float(value)
            out.append({VECTOR: tuple(vector)})
        return tuple(out)


# ── Group stages ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UnitsStep:
    """Joint and gripper ``scale``/``offset``, then rad -> deg and m -> mm.

    An ``ee_pose`` group converts its position only; its rotation carries its
    own ``angle_unit`` into ``RotationStep``.
    """

    groups: tuple[Group, ...]
    name: str = "units"

    def apply(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        return _rewrite(chunk, self.groups, _to_canonical_units)

    def invert(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        return _rewrite(chunk, self.groups, _from_canonical_units)


def _to_canonical_units(group: Group, values: Values) -> Values:
    if group.kind == KIND_EE_POSE:
        factor = 1000.0 if group.unit == "m" else 1.0
        return (*(v * factor for v in values[:3]), *values[3:])
    scaled = tuple(v * group.scale + group.offset for v in values)
    if group.unit == "rad":
        return tuple(math.degrees(v) for v in scaled)
    if group.unit == "m":
        return tuple(v * 1000.0 for v in scaled)
    return scaled


def _from_canonical_units(group: Group, values: Values) -> Values:
    if group.kind == KIND_EE_POSE:
        factor = 1000.0 if group.unit == "m" else 1.0
        return (*(v / factor for v in values[:3]), *values[3:])
    if group.unit == "rad":
        values = tuple(math.radians(v) for v in values)
    elif group.unit == "m":
        values = tuple(v / 1000.0 for v in values)
    return tuple((v - group.offset) / group.scale for v in values)


@dataclass(frozen=True)
class GripperStep:
    """A normalised gripper value to millimetres between the declared closed/open widths.

    ``absolute`` maps ``closed_value``..``open_value`` onto closed..open mm;
    ``relative``/``delta`` scale by the span only. ``binary`` reads the fraction
    of the way open as a command: past ``threshold + hysteresis`` opens, below
    ``threshold - hysteresis`` closes, in between keeps the last decision; its
    inverse is that decision (``closed_value`` or ``open_value``), what a
    binary model is trained to emit.
    """

    groups: tuple[Group, ...]
    name: str = "gripper"

    def apply(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        out = chunk
        for group in self.groups:
            out = self._apply_group(out, group, ctx)
        return out

    def _apply_group(self, chunk: Chunk, group: Group, ctx: StepContext) -> Chunk:
        closed_mm, open_mm = ctx.gripper_span(group.name)
        span = group.open_value - group.closed_value
        key = f"gripper.{group.name}.open"
        rows: list[Row] = []
        for row in chunk:
            (value,) = row[group.name]
            if group.binary:
                if key not in ctx.memory:
                    (now_mm,) = ctx.anchor_of(group.name)
                    ctx.memory[key] = now_mm > (closed_mm + open_mm) / 2.0
                fraction = (value - group.closed_value) / span
                if fraction > group.threshold + group.hysteresis:
                    ctx.memory[key] = True
                elif fraction < group.threshold - group.hysteresis:
                    ctx.memory[key] = False
                mm = open_mm if ctx.memory[key] else closed_mm
            elif group.reference == REFERENCE_ABSOLUTE:
                mm = closed_mm + (value - group.closed_value) / span * (open_mm - closed_mm)
            else:
                mm = value / span * (open_mm - closed_mm)
            rows.append({**row, group.name: (mm,)})
        return tuple(rows)

    def invert(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        def back(group: Group, values: Values) -> Values:
            closed_mm, open_mm = ctx.gripper_span(group.name)
            span = group.open_value - group.closed_value
            (mm,) = values
            if group.binary:
                fraction = (mm - closed_mm) / (open_mm - closed_mm)
                return (group.open_value if fraction > group.threshold else group.closed_value,)
            if group.reference != REFERENCE_ABSOLUTE:
                return (mm / (open_mm - closed_mm) * span,)
            return (group.closed_value + (mm - closed_mm) / (open_mm - closed_mm) * span,)

        return _rewrite(chunk, self.groups, back)


@dataclass(frozen=True)
class RotationStep:
    """An ``ee_pose`` group's rotation to the arm's fixed-axis X-Y-Z degrees.

    A position-only (``none``) group holds the observed orientation when
    absolute and turns nothing when relative.
    """

    groups: tuple[Group, ...]
    name: str = "rotation"

    def apply(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        def forward(group: Group, values: Values) -> Values:
            position, encoded = values[:3], values[3:]
            if group.rotation == ROTATION_NONE:
                if group.reference == REFERENCE_ABSOLUTE:
                    return (*position, *ctx.anchor_of(group.name)[3:6])
                return (*position, 0.0, 0.0, 0.0)
            if group.rotation == ARM_EULER and group.angle_unit == "deg":
                return values
            try:
                matrix = to_matrix(group.rotation, encoded, group.angle_unit)
            except ValueError as exc:
                raise AdaptError(f"the {group.name!r} rotation is unusable: {exc}") from exc
            return (*position, *matrix_to_rpy(matrix))

        return _rewrite(chunk, self.groups, forward)

    def invert(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        def back(group: Group, values: Values) -> Values:
            if group.rotation == ROTATION_NONE:
                return values[:3]
            if group.rotation == ARM_EULER and group.angle_unit == "deg":
                return values
            encoded = encode_rotation(group.rotation, rpy_to_matrix(*values[3:6]), group.angle_unit)
            return (*values[:3], *encoded)

        return _rewrite(chunk, self.groups, back)


@dataclass(frozen=True)
class FrameStep:
    """An ``ee_pose`` declared in a workcell frame (a camera) brought into base.

    Absolute poses are composed with ``T_base_frame``; relative and delta ones
    have their translation and rotation re-expressed in base axes.
    """

    groups: tuple[Group, ...]
    name: str = "frame"

    def apply(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        return self._run(chunk, ctx, inverse=False)

    def invert(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        return self._run(chunk, ctx, inverse=True)

    def _run(self, chunk: Chunk, ctx: StepContext, *, inverse: bool) -> Chunk:
        def convert(group: Group, values: Values) -> Values:
            t_base_frame = ctx.frame(group.frame)
            if group.reference == REFERENCE_ABSOLUTE:
                t = t_base_frame.inverse() if inverse else t_base_frame
                return _as_values(t.compose(_as_transform(values)))
            r = transpose(t_base_frame.rotation) if inverse else t_base_frame.rotation
            delta = _as_transform(values)
            turned = mat_mul(mat_mul(r, delta.rotation), transpose(r))
            return _as_values(Transform(turned, mat_vec(r, delta.translation)))

        return _rewrite(chunk, self.groups, convert)


@dataclass(frozen=True)
class ReferenceStep:
    """``relative`` and ``delta`` values made absolute against the observation."""

    groups: tuple[Group, ...]
    name: str = "reference"

    def apply(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        out = chunk
        for group in self.groups:
            out = self._chain(out, group, ctx.anchor_of(group.name), inverse=False)
        return out

    def invert(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        out = chunk
        for group in self.groups:
            out = self._chain(out, group, ctx.anchor_of(group.name), inverse=True)
        return out

    @staticmethod
    def _chain(chunk: Chunk, group: Group, anchor: Values, *, inverse: bool) -> Chunk:
        base = anchor
        rows: list[Row] = []
        for row in chunk:
            value = row[group.name]
            if inverse:
                result = _difference(group, base, value)
                step_base = value
            else:
                result = _combine(group, base, value)
                step_base = result
            if group.reference == REFERENCE_DELTA:
                base = step_base
            rows.append({**row, group.name: result})
        return tuple(rows)


def _combine(group: Group, base: Values, delta: Values) -> Values:
    if group.kind != KIND_EE_POSE:
        return tuple(b + d for b, d in zip(base, delta))
    tb, td = _as_transform(base), _as_transform(delta)
    if group.frame == FRAME_TOOL:
        return _as_values(tb.compose(td))
    position = tuple(b + d for b, d in zip(tb.translation, td.translation))
    turned = mat_mul(td.rotation, tb.rotation)
    return _as_values(Transform(turned, position))  # type: ignore[arg-type]


def _difference(group: Group, base: Values, value: Values) -> Values:
    if group.kind != KIND_EE_POSE:
        return tuple(v - b for b, v in zip(base, value))
    tb, tv = _as_transform(base), _as_transform(value)
    if group.frame == FRAME_TOOL:
        return _as_values(tb.inverse().compose(tv))
    position = tuple(v - b for b, v in zip(tb.translation, tv.translation))
    turned = mat_mul(tv.rotation, transpose(tb.rotation))
    return _as_values(Transform(turned, position))  # type: ignore[arg-type]


@dataclass(frozen=True)
class IkStep:
    """An ``ee_pose`` group bound to joint channels, solved row by row.

    Each row is seeded with the previous row's solution (the first with the
    observed joints), so a smooth pose trajectory stays on one IK branch. A
    row that does not converge refuses the chunk. The inverse is forward
    kinematics.
    """

    group: Group
    chain: DhChain
    limits: tuple[tuple[float | None, float | None], ...]
    settings: IkSettings
    name: str = "ik"

    def apply(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        seed = ctx.current.get(self.group.name)
        if seed is None or len(seed) != self.chain.dof:
            raise AdaptError(f"IK for {self.group.name!r} needs the observed joints as a seed")
        rows: list[Row] = []
        for index, row in enumerate(chunk):
            target = _as_transform(row[self.group.name])
            try:
                seed = solve_ik(self.chain, target, seed, self.limits, self.settings)
            except ValueError as exc:
                raise AdaptViolation(f"action {index}: {exc}") from exc
            rows.append({**row, self.group.name: seed})
        return tuple(rows)

    def invert(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        def back(group: Group, values: Values) -> Values:
            return self.chain.forward(values).as_tuple()

        return _rewrite(chunk, (self.group,), back)


@dataclass(frozen=True)
class ValidateStep:
    """Every group finite and as wide as its channels; the first action may not jump.

    ``max_jump`` (per channel, from the embodiment, never the model) bounds how
    far the first action of a chunk may sit from the observed state: a policy
    answering a different scene, or a wrongly declared space, shows up here as
    a leap the slew limiter would otherwise silently drag the arm through.
    """

    widths: Mapping[str, int]
    name: str = "validate"

    def apply(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        if not chunk:
            raise AdaptError("the policy returned an empty chunk")
        for index, row in enumerate(chunk):
            for group, width in self.widths.items():
                values = row.get(group)
                if values is None or len(values) != width:
                    raise AdaptError(f"action {index} has no {width}-wide {group!r} group")
                if not all(math.isfinite(v) for v in values):
                    raise AdaptError(f"action {index} {group!r} holds a non-finite value")
        first = chunk[0]
        for group in self.widths:
            self._check_jump(group, first[group], ctx)
        return chunk

    @staticmethod
    def _check_jump(group: str, values: Values, ctx: StepContext) -> None:
        current = ctx.current.get(group)
        limits = ctx.max_jump.get(group)
        if current is None or limits is None:
            return
        cyclic = ctx.cyclic.get(group) or (False,) * len(values)
        for dim, (value, now, limit, turn) in enumerate(zip(values, current, limits, cyclic)):
            if limit is None:
                continue
            jump = abs(wrap_deg(value - now) if turn else value - now)
            if jump > limit:
                raise AdaptViolation(
                    f"the first {group!r} action moves dim {dim} by {jump:.2f}, more than "
                    f"the {limit:g} a chunk may start away from the observed state"
                )

    def invert(self, chunk: Chunk, ctx: StepContext) -> Chunk:
        return chunk


def unit_groups(groups: Sequence[Group]) -> tuple[Group, ...]:
    """The groups ``UnitsStep`` changes (a normalised gripper is left to ``GripperStep``)."""
    return tuple(
        g
        for g in groups
        if (g.kind == KIND_EE_POSE and g.unit == "m")
        or (g.kind == KIND_JOINTS and (g.unit == "rad" or g.scale != 1.0 or g.offset != 0.0))
        or (g.kind == KIND_GRIPPER and (g.unit == "m" or g.scale != 1.0 or g.offset != 0.0))
    )


def gripper_groups(groups: Sequence[Group]) -> tuple[Group, ...]:
    return tuple(g for g in groups if g.kind == KIND_GRIPPER and g.unit == UNIT_NORMALIZED)


def rotation_groups(groups: Sequence[Group]) -> tuple[Group, ...]:
    return tuple(
        g
        for g in groups
        if g.kind == KIND_EE_POSE and not (g.rotation == ARM_EULER and g.angle_unit == "deg")
    )


def frame_groups(groups: Sequence[Group]) -> tuple[Group, ...]:
    return tuple(
        g for g in groups if g.kind == KIND_EE_POSE and g.frame not in (FRAME_BASE, FRAME_TOOL)
    )


def reference_groups(groups: Sequence[Group]) -> tuple[Group, ...]:
    return tuple(g for g in groups if g.reference != REFERENCE_ABSOLUTE)


__all__ = [
    "VECTOR",
    "ActionStep",
    "AdaptError",
    "AdaptViolation",
    "Chunk",
    "FrameStep",
    "GripperStep",
    "IkStep",
    "MapStep",
    "ReferenceStep",
    "RotationStep",
    "Row",
    "StepContext",
    "StepFactory",
    "UnitsStep",
    "UnnormalizeStep",
    "ValidateStep",
    "Values",
    "frame_groups",
    "gripper_groups",
    "reference_groups",
    "rotation_groups",
    "unit_groups",
]
