# Copyright (c) Alibaba, Inc. and its affiliates.
"""Binding a served policy to a workcell: two declarations in, adapter and limits out.

The model declares its native action and observation spaces in ``model_info``
(``adapt.spec.ModelSpaces``); the workcell declares under ``policies.<name>``
which channels and cameras embody them (``adapt.spec.PolicyDecl``). Neither
knows the other's half. ``bind_policy`` checks both against each other and
against the live device declarations and returns a ``PolicyBinding``: the
adapter pipelines, the shield channels in the adapter's output order, the
camera map, and what a chunk is anchored against at its observation.

Robot-side values are the channels' own -- joints in deg, the flange pose in
mm and fixed-axis X-Y-Z deg, the gripper opening in mm. A channel declaring
another unit is refused rather than converted: the adapter would be guessing.

Group binding:

- ``joints``: joint position channels of the arm;
- ``ee_pose``: exactly the arm's ``ee.*`` pose channels, or all of its joints
  in declaration order (the pose is solved by IK, which needs ``kinematics.dh``);
- ``gripper``: the gripper's ``gripper.position``.

One stream drives one motion space, so joint and pose channels together are
refused. Every limit the shield applies comes from the device declarations
and ``PolicySettings``; nothing the model declares can widen one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Mapping, Sequence

from leapflow.hardware.model_contract import ModelInfo
from leapflow.hardware.skills.adapt import (
    Action,
    ActionPipeline,
    BoundGroup,
    ModelSpaces,
    PolicyDecl,
    RecordedFrame,
    StepContext,
    build_action_pipeline,
    build_plugins,
    build_state_pipeline,
    discover_action_steps,
)
from leapflow.hardware.skills.adapt.spec import (
    FRAME_BASE,
    FRAME_TOOL,
    KIND_EE_POSE,
    KIND_GRIPPER,
    KIND_JOINTS,
    Group,
    VectorSpace,
    digest,
)
from leapflow.hardware.skills.adapt.steps import IkStep
from leapflow.hardware.skills.frames import FrameGraph
from leapflow.hardware.skills.geometry import Pose, Transform
from leapflow.hardware.skills.kinematics import DhChain
from leapflow.hardware.skills.motion import GovernedMotion
from leapflow.hardware.skills.protocol import (
    FAILURE_DEVICE_FAULT,
    ArmState,
    PolicyCard,
    SkillError,
)
from leapflow.hardware.skills.shield import ContactLimit, ShieldChannel, contact_limits
from leapflow.hardware.skills.workcell import (
    GRIPPER_POSITION,
    POSE_CHANNELS,
    ROLE_ARM,
    ROLE_GRIPPER,
    WorkcellSpec,
    joint_channels,
)

SPACE_JOINT = "joint"
SPACE_EE = "ee"
SPACE_GRIPPER = "gripper"
FLANGE_PARENT = "flange"
_DEGREES = frozenset({"deg", "degree", "degrees"})
_MILLIMETRES = frozenset({"mm", "millimeter", "millimetre", "millimeters", "millimetres"})


@dataclass(frozen=True)
class PolicySettings:
    """Streaming tuning; a workcell overrides any of it under ``policy``.

    ``max_jump_s`` bounds discontinuities: an arm channel's command may not step,
    and a chunk's first action may not start, further from the last command (or
    the observed state) than that many seconds at the channel's ``max_rate``.
    The tick rate is bounded by the action channels' declared ``sample_rate_hz``.
    """

    tick_hz: float = 50.0
    blend_s: float = 0.1
    prefetch_ratio: float = 0.5
    starve_s: float = 1.5
    warmup_s: float = 5.0
    jpeg_quality: float = 90.0
    rate_margin: float = 0.9
    startup_s: float = 15.0  # run-deadline room for binding, first observation and inference
    max_jump_s: float = 1.0
    ensemble_decay: float = 0.01  # temporal_ensemble weight decay per older chunk

    @classmethod
    def from_mapping(cls, data: Mapping[str, float]) -> "PolicySettings":
        """Defaults overridden by *data*; raises ``ValueError`` on anything unusable."""
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"policy does not understand {', '.join(unknown)}")
        settings = replace(cls(), **{name: float(value) for name, value in data.items()})
        values = {name: getattr(settings, name) for name in known}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("policy values must be finite")
        if not all(value > 0 for name, value in values.items() if name != "ensemble_decay"):
            raise ValueError("policy values must be positive")
        if settings.ensemble_decay < 0:
            raise ValueError("policy.ensemble_decay must be non-negative")
        if settings.prefetch_ratio > 1 or settings.rate_margin > 1:
            raise ValueError("policy.prefetch_ratio and policy.rate_margin must be in (0, 1]")
        if settings.jpeg_quality > 100:
            raise ValueError("policy.jpeg_quality must be in (0, 100]")
        return settings


def state_value(state: ArmState, channel_id: str) -> float:
    """One channel's value in an ``ArmState``; ``device_fault`` when it is missing."""
    if channel_id in POSE_CHANNELS:
        return state.flange.as_tuple()[POSE_CHANNELS.index(channel_id)]
    value = state.gripper_mm if channel_id == GRIPPER_POSITION else state.joints.get(channel_id)
    if value is None:
        raise SkillError(
            FAILURE_DEVICE_FAULT, f"the arm did not report a trustworthy {channel_id!r}"
        )
    return float(value)


# ── Binding ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PolicyBinding:
    """What one served policy's vectors mean on this workcell.

    ``channels`` follow the action adapter's output layout, so a robot-side row
    and the shield agree index by index.
    """

    card: PolicyCard
    device_id: str
    spaces: ModelSpaces
    decl: PolicyDecl
    actions: ActionPipeline
    state: ActionPipeline | None
    channels: tuple[ShieldChannel, ...]
    action_groups: tuple[BoundGroup, ...]
    state_groups: tuple[BoundGroup, ...]
    cameras: Mapping[str, str]  # model camera name -> device id
    joint_order: tuple[str, ...]
    chain: DhChain | None = None
    gripper_mm: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    contacts: tuple[ContactLimit, ...] = ()
    info: ModelInfo | None = field(default=None, compare=False, repr=False)
    service: Any = field(default=None, compare=False, repr=False)

    @property
    def step_s(self) -> float:
        return 1.0 / self.spaces.action.fps

    @property
    def channel_ids(self) -> tuple[str, ...]:
        return tuple(channel.channel_id for channel in self.channels)

    @property
    def cyclic(self) -> tuple[bool, ...]:
        return tuple(channel.cyclic for channel in self.channels)

    def values(self, state: ArmState, channel_ids: Sequence[str] | None = None) -> Action:
        """*channel_ids* (default: the action channels) read from one arm snapshot."""
        ids = self.channel_ids if channel_ids is None else channel_ids
        return tuple(state_value(state, channel_id) for channel_id in ids)

    def context(self, state: ArmState, frames: FrameGraph, memory: dict[str, Any]) -> StepContext:
        """What a chunk answering an observation of *state* is adapted against."""
        limits = {channel.channel_id: channel for channel in self.channels}
        anchor: dict[str, Action] = {}
        current: dict[str, Action] = {}
        jump: dict[str, tuple[float | None, ...]] = {}
        cyclic: dict[str, tuple[bool, ...]] = {}
        for bound in self.action_groups:
            name = bound.group.name
            values = self.values(state, bound.channels)
            current[name] = values
            anchor[name] = self._canonical(bound, values)
            jump[name] = tuple(limits[c].max_jump for c in bound.channels)
            cyclic[name] = tuple(limits[c].cyclic for c in bound.channels)
        return StepContext(
            anchor=anchor,
            current=current,
            frames=frame_resolver(frames, state),
            gripper_mm=dict(self.gripper_mm),
            max_jump=jump,
            cyclic=cyclic,
            memory=memory,
        )

    def observed_state(self, state: ArmState, ctx: StepContext) -> Action:
        """The observation's state vector in the model's own space (``()`` when it takes none)."""
        if self.state is None:
            return ()
        row = [
            value
            for bound in self.state_groups
            for value in self._canonical(bound, self.values(state, bound.channels))
        ]
        return self.state.invert([row], ctx)[0]

    def describe(self) -> dict[str, Any]:
        """The adapter declaration an episode pins (``digest`` is the card's)."""
        return {**_summary(self), "digest": self.card.adapter_digest}

    def encoding(self, frames: FrameGraph) -> dict[str, Any]:
        """What ``EpisodeEncoder.from_record`` rebuilds this binding's pipelines from."""
        recorded = {}
        for name in pose_frames(self):
            transform = frames.transform(name)
            if transform is not None:
                on_flange = frames.parent(name) == FLANGE_PARENT
                recorded[name] = RecordedFrame(transform, on_flange=on_flange).to_dict()
        return {
            "spaces": self.spaces.to_dict(),
            "actions": _bound_record(self.action_groups),
            "state": _bound_record(self.state_groups),
            "steps": self.decl.to_dict()["steps"],
            "ik": self.decl.to_dict()["ik"],
            "dh": None if self.chain is None else self.chain.to_dict(),
            "joint_limits": _ik_limits(self.actions),
            "gripper_mm": {group: list(span) for group, span in self.gripper_mm.items()},
            "frames": recorded,
        }

    def _canonical(self, bound: BoundGroup, values: Action) -> Action:
        if bound.ik and self.chain is not None:
            return self.chain.forward(values).as_tuple()
        return values


def frame_resolver(frames: FrameGraph, state: ArmState) -> Callable[[str], Transform]:
    """``T_base_frame`` for a calibrated workcell frame at the observed flange pose."""

    def resolve(name: str) -> Transform:
        transform = frames.transform(name)
        if transform is None:
            raise KeyError(f"{name} is not calibrated")
        if frames.parent(name) == FLANGE_PARENT:
            return Transform.from_pose(state.flange).compose(transform)
        return transform

    return resolve


@dataclass(frozen=True)
class _Embodiment:
    """The arm and gripper a policy may drive, as the live declarations describe them."""

    arm_id: str
    arm: Any
    gripper_id: str
    gripper: Any
    joints: tuple[str, ...]

    @classmethod
    def of(cls, spec: WorkcellSpec, registry: Any) -> "_Embodiment":
        arm_id, gripper_id = spec.role(ROLE_ARM), spec.role(ROLE_GRIPPER)
        arm = registry.context(arm_id) if arm_id else None
        if arm is None:
            raise ValueError("a policy needs the workcell's arm")
        gripper = registry.context(gripper_id) if gripper_id else None
        return cls(arm_id, arm, gripper_id, gripper, joint_channels(arm))

    @property
    def devices(self) -> tuple[tuple[str, Any], ...]:
        if self.gripper is None or self.gripper_id == self.arm_id:
            return ((self.arm_id, self.arm),)
        return ((self.arm_id, self.arm), (self.gripper_id, self.gripper))

    def channel(self, channel_id: str, *, write: bool, where: str) -> tuple[str, Any]:
        """``(device_id, channel)`` for *channel_id*; raises ``ValueError`` when unusable."""
        on_gripper = channel_id == GRIPPER_POSITION
        device_id, context = (
            (self.gripper_id, self.gripper) if on_gripper else (self.arm_id, self.arm)
        )
        if on_gripper and context is None:
            raise ValueError(f"{where} needs the workcell to bind a gripper")
        channel = context.channel(channel_id)
        if channel is None:
            raise ValueError(f"{where}: {device_id!r} declares no {channel_id!r}")
        if write and not channel.is_writable:
            raise ValueError(f"{where}: {device_id}.{channel_id} is not writable")
        if not write and not channel.is_readable:
            raise ValueError(f"{where}: {device_id}.{channel_id} is not readable")
        _check_unit(channel, where)
        return device_id, channel

    def bind(self, group: Group, channels: Sequence[str], where: str) -> BoundGroup:
        names = tuple(channels)
        if group.kind == KIND_JOINTS:
            stray = [c for c in names if c not in self.joints]
            if stray:
                raise ValueError(f"{where} binds {stray}, which are not joints of the arm")
            return BoundGroup(group, names)
        if group.kind == KIND_EE_POSE:
            if names == POSE_CHANNELS:
                return BoundGroup(group, names)
            if names == self.joints:
                return BoundGroup(group, names, ik=True)
            raise ValueError(
                f"{where} must bind {list(POSE_CHANNELS)} or every arm joint in order "
                f"{list(self.joints)} (solved by IK)"
            )
        if names != (GRIPPER_POSITION,):
            raise ValueError(f"{where} must bind [{GRIPPER_POSITION}]")
        return BoundGroup(group, names)

    def bind_space(
        self, space: VectorSpace, channels: Mapping[str, Sequence[str]], where: str, *, write: bool
    ) -> tuple[BoundGroup, ...]:
        stray = sorted(set(channels) - {g.name for g in space.groups})
        if stray:
            raise ValueError(f"{where} binds groups {stray} the model does not declare")
        bound: list[BoundGroup] = []
        for group in space.groups:
            names = channels.get(group.name)
            if not names:
                raise ValueError(f"{where}.{group.name} binds no channels")
            item = self.bind(group, names, f"{where}.{group.name}")
            for channel_id in item.channels:
                self.channel(channel_id, write=write, where=f"{where}.{group.name}")
            bound.append(item)
        return tuple(bound)

    def chain(self) -> DhChain:
        dh = getattr(getattr(self.arm, "kinematics", None), "dh", None) or {}
        if not dh:
            raise ValueError("driving joints needs the arm's kinematics.dh table to judge targets")
        chain = DhChain.from_mapping(dh)
        if chain.dof != len(self.joints):
            raise ValueError(f"kinematics.dh has {chain.dof} links for {len(self.joints)} joints")
        return chain

    def joint_limits(self) -> tuple[tuple[float | None, float | None], ...]:
        return tuple(
            (channel.envelope.min_value, channel.envelope.max_value)
            for channel in (self.arm.channel(cid) for cid in self.joints)
        )

    def gripper_span(self, spec: WorkcellSpec) -> tuple[float, float]:
        channel = self.gripper.channel(GRIPPER_POSITION) if self.gripper is not None else None
        if channel is None:
            return (0.0, 0.0)
        closed = channel.envelope.min_value or 0.0
        opened = spec.gripper_spec.get("max_width_mm") or channel.envelope.max_value or 0.0
        return (float(closed), float(opened))


def _check_unit(channel: Any, where: str) -> None:
    unit = str(getattr(channel, "unit", "") or "").strip().lower()
    if not unit:
        return
    cid = channel.channel_id
    angular = cid.startswith("joint.") or cid in POSE_CHANNELS[3:]
    allowed = _DEGREES if angular else _MILLIMETRES
    if unit not in allowed:
        expected = "deg" if angular else "mm"
        raise ValueError(f"{where}: {cid} is declared in {unit!r}; the adapter drives {expected}")


def _motion_space(bound: Sequence[BoundGroup], joints: Sequence[str]) -> str:
    channels = {c for item in bound for c in item.channels}
    spaces = set()
    if channels & set(joints):
        spaces.add(SPACE_JOINT)
    if channels & set(POSE_CHANNELS):
        spaces.add(SPACE_EE)
    if len(spaces) > 1:
        raise ValueError("the policy's actions mix joint and end-effector targets")
    return spaces.pop() if spaces else SPACE_GRIPPER


def _tick_problem(channels: Sequence[Any], settings: PolicySettings) -> str:
    rates = [(c.sample_rate_hz, c.channel_id) for c in channels if c.sample_rate_hz > 0]
    if not rates:
        return ""
    rate, channel_id = min(rates)
    if settings.tick_hz > rate:
        return f"policy.tick_hz {settings.tick_hz:g} exceeds the {rate:g} Hz {channel_id} accepts"
    return ""


def _cameras(
    spaces: ModelSpaces, decl: PolicyDecl, spec: WorkcellSpec, registry: Any
) -> dict[str, str]:
    declared = tuple(spaces.observation.images) or tuple(decl.cameras)
    if not declared:
        raise ValueError("no camera: declare observation_space.images or policies.<name>.cameras")
    stray = sorted(set(decl.cameras) - set(declared))
    if stray:
        raise ValueError(f"cameras maps {stray}, which the model does not observe")
    return {name: _camera_device(name, decl, spec, registry) for name in declared}


def _camera_device(name: str, decl: PolicyDecl, spec: WorkcellSpec, registry: Any) -> str:
    target = decl.cameras.get(name, name)
    device_id = spec.role(target) or target
    if registry.context(device_id) is None:
        raise ValueError(f"camera {name!r} names {target!r}, not a role or declared device")
    return device_id


def _warnings(spaces: ModelSpaces) -> tuple[str, ...]:
    warnings = []
    vectors = [("action", spaces.action.vector), ("state", spaces.observation.state)]
    for label, vector in vectors:
        if vector is not None and vector.normalized and not vector.normalization.checksum:
            warnings.append(
                f"the {label} normalisation stats carry no checksum; a server/node mismatch "
                "goes unnoticed"
            )
    if not spaces.calibration:
        warnings.append("the policy pins no calibration; a moved camera goes unnoticed")
    return tuple(warnings)


def _summary(binding: PolicyBinding) -> dict[str, Any]:
    return {
        "steps": binding.actions.describe(),
        "state_steps": [] if binding.state is None else binding.state.describe(),
        "channels": list(binding.channel_ids),
        "cameras": dict(binding.cameras),
        "decl": binding.decl.to_dict(),
        "spaces": binding.spaces.to_dict(),
    }


def bind_policy(
    name: str,
    device_id: str,
    info: ModelInfo,
    spec: WorkcellSpec,
    registry: Any,
    settings: PolicySettings,
) -> PolicyBinding:
    """Bind *info*'s declared spaces through ``policies.<name>``; raises ``ValueError``."""
    raw = spec.policies.get(name)
    if raw is None:
        raise ValueError(f"the workcell declares no policies.{name} embodiment")
    where = f"policies.{name}"
    decl = PolicyDecl.from_mapping(raw, where)
    spaces = ModelSpaces.from_details(info.details, decl.overrides)
    body = _Embodiment.of(spec, registry)
    action_groups = body.bind_space(
        spaces.action.vector, decl.actions, f"{where}.actions", write=True
    )
    state_space = spaces.observation.state
    state_groups: tuple[BoundGroup, ...] = ()
    if state_space is not None:
        wanted = {g.name: decl.state_channels(g.name) for g in state_space.groups}
        state_groups = body.bind_space(state_space, wanted, f"{where}.state", write=False)
    elif decl.state:
        raise ValueError(f"{where}.state binds groups but the model observes no state")
    space = _motion_space(action_groups, body.joints)
    needs_chain = space == SPACE_JOINT or any(b.ik for b in (*action_groups, *state_groups))
    chain = body.chain() if needs_chain else None
    plugins = build_plugins(
        [(step.name, step.params) for step in decl.steps], [b.group for b in action_groups]
    )
    actions = build_action_pipeline(
        spaces.action.vector,
        action_groups,
        chain=chain,
        joint_limits=body.joint_limits(),
        ik=decl.ik,
        plugins=plugins,
    )
    state = build_state_pipeline(state_space, state_groups) if state_space is not None else None
    # Both pipelines sort their groups into the space's order; follow them.
    action_groups = _ordered(action_groups, [group for group, _ in actions.layout])
    state_groups = _ordered(state_groups, [g.name for g in _space_groups(state_space)])
    channels = _shield_channels(body, action_groups, settings, f"{where}.actions")
    card = PolicyCard(
        name=name,
        model=info.identity.to_dict(),
        space=space,
        channels=tuple(c.channel_id for c in channels),
        fps=spaces.action.fps,
        tasks=spaces.tasks,
        warnings=_warnings(spaces),
        instruction_policy=spaces.instruction_policy,
        adapter=tuple(actions.describe()),
    )
    binding = PolicyBinding(
        card=card,
        device_id=device_id,
        spaces=spaces,
        decl=decl,
        actions=actions,
        state=state,
        channels=channels,
        action_groups=action_groups,
        state_groups=state_groups,
        cameras=_cameras(spaces, decl, spec, registry),
        joint_order=body.joints,
        chain=chain,
        gripper_mm=_gripper_spans(body, spec, spaces),
        contacts=contact_limits(body.devices),
        info=info,
    )
    return replace(binding, card=replace(card, adapter_digest=digest(_summary(binding))))


def _space_groups(space: VectorSpace | None) -> tuple[Group, ...]:
    return () if space is None else space.groups


def _bound_record(bound: Sequence[BoundGroup]) -> list[dict[str, Any]]:
    return [{"group": b.group.name, "channels": list(b.channels), "ik": b.ik} for b in bound]


def _ik_limits(actions: ActionPipeline) -> list[list[float | None]]:
    """The joint limits the pipeline's IK solves within (``[]`` without IK)."""
    for step in actions.steps:
        if isinstance(step, IkStep):
            return [list(limit) for limit in step.limits]
    return []


def _ordered(bound: Sequence[BoundGroup], names: Sequence[str]) -> tuple[BoundGroup, ...]:
    rank = {name: index for index, name in enumerate(names)}
    return tuple(sorted(bound, key=lambda item: rank[item.group.name]))


def _shield_channels(
    body: _Embodiment, bound: Sequence[BoundGroup], settings: PolicySettings, where: str
) -> tuple[ShieldChannel, ...]:
    """The action channels' limits in layout order; a gripper's open/close may jump."""
    declared = [
        body.channel(channel_id, write=True, where=where)
        for item in bound
        for channel_id in item.channels
    ]
    problem = _tick_problem([channel for _, channel in declared], settings)
    if problem:
        raise ValueError(problem)
    return tuple(
        ShieldChannel.from_channel(
            owner,
            channel,
            jump_s=None if channel.channel_id == GRIPPER_POSITION else settings.max_jump_s,
        )
        for owner, channel in declared
    )


def _gripper_spans(
    body: _Embodiment, spec: WorkcellSpec, spaces: ModelSpaces
) -> dict[str, tuple[float, float]]:
    """Closed/open width (mm) of every gripper group the model declares, action or state."""
    if body.gripper is None:
        return {}
    span = body.gripper_span(spec)
    groups = (*spaces.action.vector.groups, *_space_groups(spaces.observation.state))
    return {g.name: span for g in groups if g.kind == KIND_GRIPPER}


def pose_frames(binding: PolicyBinding) -> tuple[str, ...]:
    """Workcell frames the binding's pose groups are declared in (need calibration)."""
    frames = {
        bound.group.frame
        for bound in (*binding.action_groups, *binding.state_groups)
        if bound.group.kind == KIND_EE_POSE and bound.group.frame not in (FRAME_BASE, FRAME_TOOL)
    }
    return tuple(sorted(frames))


def policy_problems(spec: WorkcellSpec, registry: Any) -> list[str]:
    """Why the workcell's policy tuning or ``policies`` cannot be admitted (offline checks).

    The model half (spaces, widths, units) is only known once the server is up,
    so it is checked when a policy is bound; what the workcell alone decides --
    channels, cameras, plugin steps, tick rate -- is refused here.
    """
    try:
        settings = PolicySettings.from_mapping(spec.policy)
    except ValueError as exc:
        return [f"policy: {exc}"]
    problems: list[str] = []
    installed: set[str] | None = None
    for name, raw in spec.policies.items():
        where = f"policies.{name}"
        if not spec.model(name):
            problems.append(f"{where}: no models.{name} binds the policy it embodies")
            continue
        try:
            decl = PolicyDecl.from_mapping(raw, where)
            body = _Embodiment.of(spec, registry)
            declared = [
                body.channel(cid, write=True, where=f"{where}.actions.{group}")[1]
                for group, ids in decl.actions.items()
                for cid in ids
            ]
            for group, ids in decl.state.items():
                for cid in ids:
                    body.channel(cid, write=False, where=f"{where}.state.{group}")
            for camera in decl.cameras:
                _camera_device(camera, decl, spec, registry)
            problem = _tick_problem(declared, settings)
            if problem:
                raise ValueError(problem)
        except ValueError as exc:
            problems.append(f"{where}: {exc}")
            continue
        if decl.steps:
            installed = set(discover_action_steps()) if installed is None else installed
            missing = sorted({s.name for s in decl.steps} - installed)
            if missing:
                problems.append(f"{where}.steps: {missing} are not installed action steps")
    return problems


# ── Prediction ───────────────────────────────────────────────────────────────


class PosePredictor:
    """Where a robot-side row puts the TCP; axes it does not name hold their observed value."""

    def __init__(self, binding: PolicyBinding, state: ArmState, motion: GovernedMotion) -> None:
        self._motion = motion
        self._chain = binding.chain
        self._tcp = state.tcp
        self._base: Action = ()
        space = binding.card.space
        order: Sequence[str] = ()
        if space == SPACE_EE:
            order, self._base = POSE_CHANNELS, state.flange.as_tuple()
        elif space == SPACE_JOINT:
            order, self._base = binding.joint_order, binding.values(state, binding.joint_order)
        self._slots = tuple(
            (index, order.index(channel_id))
            for index, channel_id in enumerate(binding.channel_ids)
            if channel_id in order
        )

    def tcp(self, action: Sequence[float]) -> Pose:
        if not self._slots:
            return self._tcp
        values = list(self._base)
        for source, target in self._slots:
            values[target] = float(action[source])
        flange = self._chain.forward(values) if self._chain is not None else Pose(*values)
        return self._motion.flange_to_tcp(flange)

    def check(self, action: Action) -> str:
        """The shield's pose check: the discrete-move region rules, as a reason or ``""``."""
        if not self._slots:
            return ""
        try:
            self._motion.check_target(self.tcp(action))
        except SkillError as error:
            return error.message
        return ""


__all__ = [
    "SPACE_EE",
    "SPACE_GRIPPER",
    "SPACE_JOINT",
    "PolicyBinding",
    "PolicySettings",
    "PosePredictor",
    "bind_policy",
    "frame_resolver",
    "policy_problems",
    "pose_frames",
    "state_value",
]
