# Copyright (c) Alibaba, Inc. and its affiliates.
"""Recorded channel values re-encoded in a policy's native space (training data).

A policy run's episode pins the binding it ran under -- the model's spaces,
the channels each group was bound to, the gripper spans, the calibrated frames
its pose groups are declared in and the DH table, joint limits and tuning IK
solved with (see ``PolicyBinding.encoding``). ``EpisodeEncoder`` rebuilds the
very same adapter pipelines from that record and runs them in the inverse
direction, so a dataset's state and action vectors are what the model consumes
and emits on this embodiment: one declaration serves inference and training,
and nothing is re-implemented in a converter.

The same record, written by hand, encodes demonstrations recorded without a
policy (``episode.record``) for a model that has not been served yet::

    {"spaces": {"action_space": {...}, "observation_space": {...}},
     "actions": [{"group": "arm", "channels": ["joint.j1.position", ...]}],
     "state": [...], "gripper_mm": {"gripper": [0, 70]}, "frames": {}, "dh": null}

A sample maps channel ids to values (``joint.<j>.position``, ``ee.x`` ..
``ee.rz`` for the flange, ``gripper.position``). Each action is encoded
against the sample observed with it, so ``relative`` and ``delta`` groups
become the target minus that observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from leapflow.hardware.skills.adapt.pipeline import (
    ActionPipeline,
    BoundGroup,
    build_action_pipeline,
    build_plugins,
    build_state_pipeline,
)
from leapflow.hardware.skills.adapt.spec import Group, IkSettings, ModelSpaces, VectorSpace
from leapflow.hardware.skills.adapt.steps import AdaptError, StepContext, StepFactory
from leapflow.hardware.skills.geometry import Pose, Transform
from leapflow.hardware.skills.kinematics import DhChain

FLANGE_POSE = ("ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz")
Sample = Mapping[str, float]


@dataclass(frozen=True)
class RecordedFrame:
    """A calibrated frame as the run saw it: ``T_parent_frame`` and whether it rides the flange."""

    transform: Transform
    on_flange: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"matrix": self.transform.to_matrix(), "on_flange": self.on_flange}


@dataclass(frozen=True)
class EpisodeEncoder:
    """A recorded policy binding rebuilt offline: channel samples to native vectors."""

    spaces: ModelSpaces
    actions: ActionPipeline
    state: ActionPipeline | None
    action_groups: tuple[BoundGroup, ...]
    state_groups: tuple[BoundGroup, ...]
    chain: DhChain | None = None
    gripper_mm: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    frames: Mapping[str, RecordedFrame] = field(default_factory=dict)

    @property
    def action_names(self) -> tuple[str, ...]:
        return _dim_names(self.spaces.action.vector)

    @property
    def state_names(self) -> tuple[str, ...]:
        state = self.spaces.observation.state
        return () if state is None else _dim_names(state)

    @property
    def channels(self) -> tuple[str, ...]:
        """Every channel a sample must hold for both vectors to be encoded."""
        wanted = [c for b in (*self.action_groups, *self.state_groups) for c in b.channels]
        if any(frame.on_flange for frame in self.frames.values()):
            wanted.extend(FLANGE_POSE)
        return tuple(dict.fromkeys(wanted))

    def encode_state(self, sample: Sample) -> tuple[float, ...]:
        """The observation's state vector (``()`` when the model observes none)."""
        if self.state is None:
            return ()
        row = [v for bound in self.state_groups for v in self._canonical(bound, sample)]
        return self.state.invert([row], self._context(sample))[0]

    def encode_action(self, target: Sample, sample: Sample) -> tuple[float, ...]:
        """The native action that, observed at *sample*, commands *target*."""
        row = [_value(target, c) for bound in self.action_groups for c in bound.channels]
        return self.actions.invert([row], self._context(sample))[0]

    def decode_action(self, native: Sequence[float], sample: Sample) -> dict[str, float]:
        """The channel targets *native*, observed at *sample*, commands (``encode_action``'s
        inverse, to audit a dataset)."""
        (row,) = self.actions.adapt([native], self._context(sample))
        channels = [c for bound in self.action_groups for c in bound.channels]
        return dict(zip(channels, row))

    def _context(self, sample: Sample) -> StepContext:
        anchor = {b.group.name: self._canonical(b, sample) for b in self.action_groups}
        current = {
            b.group.name: tuple(_value(sample, c) for c in b.channels) for b in self.action_groups
        }
        return StepContext(
            anchor=anchor,
            current=current,
            frames=lambda name: self._frame(name, sample),
            gripper_mm=dict(self.gripper_mm),
        )

    def _canonical(self, bound: BoundGroup, sample: Sample) -> tuple[float, ...]:
        values = tuple(_value(sample, c) for c in bound.channels)
        if not bound.ik:
            return values
        if self.chain is None:
            raise AdaptError(
                f"the {bound.group.name!r} group reads joints but no DH table is pinned"
            )
        return self.chain.forward(values).as_tuple()

    def _frame(self, name: str, sample: Sample) -> Transform:
        frame = self.frames.get(name)
        if frame is None:
            raise KeyError(f"{name} was not recorded with the episode")
        if not frame.on_flange:
            return frame.transform
        flange = Pose(*(_value(sample, c) for c in FLANGE_POSE))
        return Transform.from_pose(flange).compose(frame.transform)

    @classmethod
    def from_record(
        cls, record: Mapping[str, Any], *, factories: Mapping[str, StepFactory] | None = None
    ) -> "EpisodeEncoder":
        """Rebuild the encoder; raises ``ValueError`` for an unusable record."""
        spaces = ModelSpaces.from_details(_mapping(record.get("spaces"), "spaces"))
        dh = record.get("dh")
        chain = DhChain.from_mapping(dh) if dh else None
        action_space = spaces.action.vector
        action_groups = _bound(action_space, record.get("actions"), "actions")
        state_space = spaces.observation.state
        state_groups: tuple[BoundGroup, ...] = ()
        if state_space is not None:
            state_groups = _bound(state_space, record.get("state"), "state")
        plugins = build_plugins(
            _plugin_decls(record.get("steps")), [b.group for b in action_groups], factories
        )
        actions = build_action_pipeline(
            action_space,
            action_groups,
            chain=chain,
            joint_limits=_limits(record.get("joint_limits")),
            ik=IkSettings.from_mapping(record.get("ik")),
            plugins=plugins,
        )
        state = build_state_pipeline(state_space, state_groups) if state_space else None
        return cls(
            spaces=spaces,
            actions=actions,
            state=state,
            action_groups=_in_layout(action_groups, [name for name, _ in actions.layout]),
            state_groups=_in_layout(state_groups, [g.name for g in _groups(state_space)]),
            chain=chain,
            gripper_mm=_spans(record.get("gripper_mm")),
            frames=_frames(record.get("frames")),
        )


def _value(sample: Sample, channel: str) -> float:
    value = sample.get(channel)
    if value is None:
        raise AdaptError(f"the sample holds no {channel!r}")
    return float(value)


def _dim_names(space: VectorSpace) -> tuple[str, ...]:
    if space.names:
        return space.names
    names = [f"dim{i}" for i in range(space.width)]
    for group in space.groups:
        for index, dim in enumerate(group.dims):
            names[dim] = f"{group.name}.{index}"
    return tuple(names)


def _mapping(raw: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"the encoding record's {where} must be a mapping")
    return raw


def _bound(space: VectorSpace, raw: Any, where: str) -> tuple[BoundGroup, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"the encoding record's {where} must list the bound groups")
    bound: list[BoundGroup] = []
    for item in raw:
        entry = _mapping(item, where)
        group: Group | None = space.group(str(entry.get("group")))
        if group is None:
            raise ValueError(f"{where} binds {entry.get('group')!r}, which the space does not hold")
        channels = tuple(str(c) for c in entry.get("channels") or ())
        bound.append(BoundGroup(group, channels, ik=bool(entry.get("ik"))))
    return tuple(bound)


def _in_layout(bound: Sequence[BoundGroup], names: Sequence[str]) -> tuple[BoundGroup, ...]:
    rank = {name: index for index, name in enumerate(names)}
    return tuple(sorted(bound, key=lambda item: rank[item.group.name]))


def _groups(space: VectorSpace | None) -> tuple[Group, ...]:
    return () if space is None else space.groups


def _plugin_decls(raw: Any) -> list[tuple[str, Mapping[str, object]]]:
    decls: list[tuple[str, Mapping[str, object]]] = []
    for item in raw or ():
        entry = _mapping(item, "steps")
        decls.append((str(entry["name"]), {k: v for k, v in entry.items() if k != "name"}))
    return decls


def _spans(raw: Any) -> dict[str, tuple[float, float]]:
    spans: dict[str, tuple[float, float]] = {}
    for group, span in (raw or {}).items():
        closed, opened = span
        spans[str(group)] = (float(closed), float(opened))
    return spans


def _limits(raw: Any) -> tuple[tuple[float | None, float | None], ...]:
    def bound(value: Any) -> float | None:
        return None if value is None else float(value)

    return tuple((bound(low), bound(high)) for low, high in raw or ())


def _frames(raw: Any) -> dict[str, RecordedFrame]:
    frames: dict[str, RecordedFrame] = {}
    for name, entry in (raw or {}).items():
        data = _mapping(entry, f"frames.{name}")
        frames[str(name)] = RecordedFrame(
            Transform.from_matrix(data["matrix"]), on_flange=bool(data.get("on_flange"))
        )
    return frames


__all__ = ["FLANGE_POSE", "EpisodeEncoder", "RecordedFrame"]
