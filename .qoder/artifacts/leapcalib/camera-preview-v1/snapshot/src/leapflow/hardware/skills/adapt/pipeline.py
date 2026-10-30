# Copyright (c) Alibaba, Inc. and its affiliates.
"""Adapter pipelines built from declarations: no model is special-cased here.

``build_action_pipeline`` reads the model's action space and the workcell's
binding and lays out only the steps the declaration needs (an absolute joint
policy in degrees is ``map -> validate``). ``build_state_pipeline`` builds the
mirror image for the observation's state, which runs in the inverse
direction: canonical robot state in, the model's (normalised) vector out.

Plugin steps come from the entry-point group ``leapflow.hardware.action_steps``
(a ``StepFactory`` per name) and run on canonical channel values, after IK and
before validation, so nothing a plugin does escapes the final checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Callable, Mapping, Sequence

from leapflow.hardware.skills.adapt.spec import (
    KIND_EE_POSE,
    REFERENCE_ABSOLUTE,
    Group,
    IkSettings,
    VectorSpace,
)
from leapflow.hardware.skills.adapt.steps import (
    VECTOR,
    ActionStep,
    AdaptError,
    Chunk,
    FrameStep,
    GripperStep,
    IkStep,
    MapStep,
    ReferenceStep,
    RotationStep,
    StepContext,
    StepFactory,
    UnitsStep,
    UnnormalizeStep,
    ValidateStep,
    frame_groups,
    gripper_groups,
    reference_groups,
    rotation_groups,
    unit_groups,
)
from leapflow.hardware.skills.kinematics import DhChain

logger = logging.getLogger(__name__)

ACTION_STEP_GROUP = "leapflow.hardware.action_steps"
CANONICAL_POSE_WIDTH = 6


@dataclass(frozen=True)
class BoundGroup:
    """A space group and the channels it drives (or is read from), in value order.

    ``ik`` marks an ``ee_pose`` group bound to joint channels: its canonical pose
    is solved for joints on the way out and computed by FK on the way in.
    """

    group: Group
    channels: tuple[str, ...]
    ik: bool = False

    @property
    def canonical_width(self) -> int:
        return CANONICAL_POSE_WIDTH if self.group.kind == KIND_EE_POSE else self.group.width

    @property
    def channel_width(self) -> int:
        return len(self.channels)


class ActionPipeline:
    """A fixed sequence of steps between a model's vectors and channel values."""

    def __init__(
        self, steps: Sequence[ActionStep], layout: Sequence[tuple[str, int]], *, width: int
    ) -> None:
        self._steps = tuple(steps)
        self._layout = tuple(layout)
        self._native_width = width

    @property
    def steps(self) -> tuple[ActionStep, ...]:
        return self._steps

    @property
    def native_width(self) -> int:
        """Values per model-side row."""
        return self._native_width

    @property
    def layout(self) -> tuple[tuple[str, int], ...]:
        """``(group, values)`` of a robot-side row, in order."""
        return self._layout

    @property
    def width(self) -> int:
        """Values per robot-side row (the concatenated layout)."""
        return sum(width for _, width in self._layout)

    def describe(self) -> list[str]:
        return [step.name for step in self._steps]

    def adapt(
        self, rows: Sequence[Sequence[float]], ctx: StepContext
    ) -> tuple[tuple[float, ...], ...]:
        """Native model rows to robot-side rows; raises ``AdaptError``."""
        chunk: Chunk = tuple({VECTOR: tuple(float(v) for v in row)} for row in rows)
        for step in self._steps:
            chunk = _run(step, step.apply, chunk, ctx)
        return tuple(tuple(v for name, _ in self._layout for v in row[name]) for row in chunk)

    def invert(
        self, rows: Sequence[Sequence[float]], ctx: StepContext
    ) -> tuple[tuple[float, ...], ...]:
        """Robot-side rows back to native model rows; raises ``AdaptError``."""
        chunk: Chunk = tuple(self._split(row) for row in rows)
        for step in reversed(self._steps):
            chunk = _run(step, step.invert, chunk, ctx)
        return tuple(tuple(row[VECTOR]) for row in chunk)

    def _split(self, row: Sequence[float]) -> dict[str, tuple[float, ...]]:
        if len(row) != self.width:
            raise AdaptError(f"a robot-side row has {len(row)} values, the layout {self.width}")
        out: dict[str, tuple[float, ...]] = {}
        at = 0
        for name, width in self._layout:
            out[name] = tuple(float(v) for v in row[at : at + width])
            at += width
        return out


def _run(
    step: ActionStep, method: Callable[[Chunk, StepContext], Chunk], chunk: Chunk, ctx: StepContext
) -> Chunk:
    try:
        return method(chunk, ctx)
    except AdaptError:
        raise
    except (ValueError, ArithmeticError, KeyError, IndexError) as exc:
        raise AdaptError(f"adapter step {step.name!r} failed: {exc}") from exc


def _vector_steps(space: VectorSpace, groups: Sequence[Group]) -> list[ActionStep]:
    steps: list[ActionStep] = []
    if space.normalized:
        steps.append(UnnormalizeStep(space.normalization))
    steps.append(MapStep(space.width, tuple(groups)))
    for maker, chosen in (
        (UnitsStep, unit_groups(groups)),
        (GripperStep, gripper_groups(groups)),
        (RotationStep, rotation_groups(groups)),
        (FrameStep, frame_groups(groups)),
    ):
        if chosen:
            steps.append(maker(chosen))
    return steps


def _check_bound(space: VectorSpace, bound: Sequence[BoundGroup], where: str) -> None:
    named = {b.group.name for b in bound}
    declared = {g.name for g in space.groups}
    if named != declared:
        raise ValueError(
            f"{where} binds groups {sorted(named)} but the model declares {sorted(declared)}"
        )
    for item in bound:
        expected = item.channel_width if item.ik else item.canonical_width
        if item.channel_width != expected or (item.ik and item.group.kind != KIND_EE_POSE):
            raise ValueError(
                f"{where}.{item.group.name} binds {item.channel_width} channels; a "
                f"{item.group.kind} group of this space needs {item.canonical_width}"
            )


def build_action_pipeline(
    space: VectorSpace,
    bound: Sequence[BoundGroup],
    *,
    chain: DhChain | None = None,
    joint_limits: Sequence[tuple[float | None, float | None]] = (),
    ik: IkSettings | None = None,
    plugins: Sequence[ActionStep] = (),
) -> ActionPipeline:
    """The model-to-robot pipeline; raises ``ValueError`` for an unusable declaration."""
    _check_bound(space, bound, "policies.actions")
    order = {g.name: i for i, g in enumerate(space.groups)}
    bound = sorted(bound, key=lambda b: order[b.group.name])
    groups = [b.group for b in bound]
    steps = _vector_steps(space, groups)
    moving = reference_groups(groups)
    if moving:
        steps.append(ReferenceStep(moving))
    for item in bound:
        if not item.ik:
            continue
        if chain is None:
            raise ValueError(
                f"the {item.group.name!r} pose group drives joints, which needs the arm's "
                "kinematics.dh table"
            )
        if item.channel_width != chain.dof:
            raise ValueError(
                f"{item.group.name!r} binds {item.channel_width} joints, the DH table {chain.dof}"
            )
        steps.append(IkStep(item.group, chain, tuple(joint_limits), ik or IkSettings()))
    steps.extend(plugins)
    steps.append(ValidateStep({b.group.name: b.channel_width for b in bound}))
    layout = [(b.group.name, b.channel_width) for b in bound]
    return ActionPipeline(steps, layout, width=space.width)


def build_state_pipeline(space: VectorSpace, bound: Sequence[BoundGroup]) -> ActionPipeline:
    """The robot-to-model pipeline for observation state (run with ``invert``).

    Its robot side is canonical: an ``ee_pose`` state group takes the flange pose
    (the binder computes it by FK when the group reads joints).
    """
    _check_bound(space, bound, "policies.state")
    if any(g.reference != REFERENCE_ABSOLUTE for g in space.groups):
        raise ValueError("observation state groups must be absolute")
    order = {g.name: i for i, g in enumerate(space.groups)}
    bound = sorted(bound, key=lambda b: order[b.group.name])
    groups = [b.group for b in bound]
    layout = [(b.group.name, b.canonical_width) for b in bound]
    return ActionPipeline(_vector_steps(space, groups), layout, width=space.width)


def discover_action_steps() -> dict[str, StepFactory]:
    """Plugin step factories by name; a broken plugin is logged and skipped."""
    found: dict[str, StepFactory] = {}
    for entry in entry_points(group=ACTION_STEP_GROUP):
        try:
            factory = entry.load()
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not break binding
            logger.warning("action step plugin %s failed to load: %s", entry.name, exc)
            continue
        if callable(factory):
            found[entry.name] = factory
    return found


def build_plugins(
    declared: Sequence[tuple[str, Mapping[str, object]]],
    groups: Sequence[Group],
    factories: Mapping[str, StepFactory] | None = None,
) -> list[ActionStep]:
    """Instantiate the declared plugin steps; raises ``ValueError`` for an unknown one."""
    if not declared:
        return []
    available = discover_action_steps() if factories is None else dict(factories)
    steps: list[ActionStep] = []
    for name, params in declared:
        factory = available.get(name)
        if factory is None:
            raise ValueError(f"action step {name!r} is not installed ({ACTION_STEP_GROUP})")
        step = factory(params, tuple(groups))
        if not isinstance(step, ActionStep):
            raise ValueError(f"action step {name!r} does not implement ActionStep")
        steps.append(step)
    return steps


__all__ = [
    "ACTION_STEP_GROUP",
    "ActionPipeline",
    "BoundGroup",
    "build_action_pipeline",
    "build_plugins",
    "build_state_pipeline",
    "discover_action_steps",
]
