# Copyright (c) Alibaba, Inc. and its affiliates.
"""Declared action and observation spaces: what a policy's vectors mean.

Two declarations meet here, and neither knows the other's half.

The **model** declares its native semantics in ``model_info`` -- the space it
was trained in, independent of any robot it may drive::

    action_space:
      fps: 30                          # rate the chunk's rows are spaced at
      pending: true                    # accepts the not-yet-executed rows (RTC)
      width: 7
      names: [j1, j2, j3, j4, j5, j6, grip]        # optional; dims may use them
      normalization: {mode: mean_std, mean: [..], std: [..], checksum: "sha256:.."}
      normalized: true                 # false: the server already de-normalised
      groups:
        - {name: arm, kind: joints, dims: [0, 1, 2, 3, 4, 5], unit: rad}
        - {name: gripper, kind: gripper, dims: [grip], unit: normalized}
    observation_space:
      state: {width: 7, groups: [...], normalization: {...}}   # absolute groups
      images: {top: {width: 224, height: 224, fit: pad}}       # stretch | crop | pad
    instruction_policy: open           # exact: only the listed tasks are accepted
    tasks: [pick up the cube]
    calibration: {camera: 1.7e9}       # frame versions it was trained against

Group kinds and their fields:

- ``joints``: ``unit`` deg | rad; optional ``scale``/``offset`` applied first.
- ``ee_pose``: the *flange* pose (what the arm's ``ee.*`` channels report).
  ``unit`` mm | m for the position, ``rotation`` (see ``rotations``) with its
  ``angle_unit``, and ``frame``: ``base``, ``tool`` (the flange's own axes; for
  ``relative``/``delta`` only) or any workcell frame (``camera``).
- ``gripper``: one value; ``unit`` mm | m | normalized, where ``open_value`` /
  ``closed_value`` name the normalised ends and ``binary`` (with ``threshold``
  and ``hysteresis``) reads it as a command to open or close.

Every group has a ``reference``: ``absolute``, ``relative`` (to the state at
the observation the chunk answers) or ``delta`` (to the previous row; the first
row to the observation).

Normalisation stats are pinned by ``checksum``: ``"sha256:"`` plus the hex
SHA-256 of ``json.dumps({"mode": m, <a>: [...], <b>: [...]}, sort_keys=True,
separators=(",", ":"))`` with every value a float -- ``mean``/``std`` for
``mean_std`` (``x = z * (std + 1e-8) + mean``), ``min``/``max`` for ``min_max``
(``x = (z + 1) / 2 * (max - min) + min``). A mismatch refuses the binding.

The **workcell** declares the embodiment binding under ``policies.<name>``
(``PolicyDecl``)::

    policies:
      cube:
        actions: {arm: [joint.j1.position, ...], gripper: [gripper.position]}
        state: {arm: [...]}           # default: the group's action channels
        cameras: {top: camera}        # model camera -> workcell role or device id
        overrides: {action: {gripper: {binary: true, hysteresis: 0.1}}}
        aggregate: crossfade          # temporal_ensemble | latest
        interpolation: linear         # hold
        smooth_hz: 0                  # first-order low-pass cutoff; 0 is off
        ik: {damping: 1.0}            # an ee_pose group bound to joint channels
        steps: [my_step]              # plugin action steps, before validation
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, fields, replace
from typing import Any, Mapping, Sequence

from leapflow.hardware.skills.adapt.rotations import ARM_EULER, ROTATION_NONE, rotation_width

KIND_JOINTS = "joints"
KIND_EE_POSE = "ee_pose"
KIND_GRIPPER = "gripper"
REFERENCE_ABSOLUTE = "absolute"
REFERENCE_RELATIVE = "relative"
REFERENCE_DELTA = "delta"
REFERENCES = (REFERENCE_ABSOLUTE, REFERENCE_RELATIVE, REFERENCE_DELTA)
FRAME_BASE = "base"
FRAME_TOOL = "tool"
UNIT_NORMALIZED = "normalized"
UNITS: Mapping[str, tuple[str, ...]] = {
    KIND_JOINTS: ("deg", "rad"),
    KIND_EE_POSE: ("mm", "m"),
    KIND_GRIPPER: ("mm", "m", UNIT_NORMALIZED),
}
CANONICAL_UNIT = {KIND_JOINTS: "deg", KIND_EE_POSE: "mm", KIND_GRIPPER: "mm"}
ANGLE_UNITS = ("deg", "rad")

NORM_IDENTITY = "identity"
NORM_MEAN_STD = "mean_std"
NORM_MIN_MAX = "min_max"
NORM_EPS = 1e-8
_NORM_KEYS = {NORM_MEAN_STD: ("mean", "std"), NORM_MIN_MAX: ("min", "max")}

INSTRUCTION_EXACT = "exact"
INSTRUCTION_OPEN = "open"
AGGREGATE_CROSSFADE = "crossfade"
AGGREGATE_ENSEMBLE = "temporal_ensemble"
AGGREGATE_LATEST = "latest"
INTERPOLATION_LINEAR = "linear"
INTERPOLATION_HOLD = "hold"
FIT_STRETCH = "stretch"
FIT_CROP = "crop"
FIT_PAD = "pad"

_FRAME_NAME = re.compile(r"[a-z0-9_]+")
_GROUP_KEYS = frozenset(
    {
        "name", "kind", "dims", "unit", "rotation", "angle_unit", "reference", "frame",
        "scale", "offset", "open_value", "closed_value", "binary", "threshold", "hysteresis",
    }
)  # fmt: skip
_OVERRIDABLE = _GROUP_KEYS - {"name", "kind", "dims"}


# ── Scalars ──────────────────────────────────────────────────────────────────


def _number(raw: Any, where: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{where} must be a number")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{where} must be finite")
    return value


def _positive(raw: Any, where: str) -> float:
    value = _number(raw, where)
    if value <= 0:
        raise ValueError(f"{where} must be positive")
    return value


def _choice(raw: Any, allowed: Sequence[str], where: str) -> str:
    value = str(raw).strip() if raw is not None else ""
    if value not in allowed:
        raise ValueError(f"{where} must be one of {', '.join(allowed)}, not {raw!r}")
    return value


def _mapping(raw: Any, where: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be a mapping")
    return raw


def _strings(raw: Any, where: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise ValueError(f"{where} must be a list of strings")
    if not all(isinstance(item, str) and item.strip() for item in raw):
        raise ValueError(f"{where} must hold non-empty strings")
    return tuple(item.strip() for item in raw)


def _floats(raw: Any, width: int, where: str) -> tuple[float, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != width:
        raise ValueError(f"{where} must list {width} numbers")
    return tuple(_number(value, where) for value in raw)


def stats_checksum(mode: str, low: Sequence[float], high: Sequence[float]) -> str:
    """The canonical checksum of normalisation stats (see the module docstring)."""
    a, b = _NORM_KEYS[mode]
    payload = {"mode": mode, a: [float(v) for v in low], b: [float(v) for v in high]}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Normalisation ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Normalization:
    """Per-dimension stats: ``low``/``high`` are mean/std or min/max."""

    mode: str = NORM_IDENTITY
    low: tuple[float, ...] = ()
    high: tuple[float, ...] = ()
    checksum: str = ""

    def to_physical(self, values: Sequence[float]) -> tuple[float, ...]:
        if self.mode == NORM_MEAN_STD:
            return tuple(z * (s + NORM_EPS) + m for z, m, s in zip(values, self.low, self.high))
        if self.mode == NORM_MIN_MAX:
            return tuple(
                (z + 1.0) / 2.0 * max(hi - lo, NORM_EPS) + lo
                for z, lo, hi in zip(values, self.low, self.high)
            )
        return tuple(float(v) for v in values)

    def to_model(self, values: Sequence[float]) -> tuple[float, ...]:
        if self.mode == NORM_MEAN_STD:
            return tuple((x - m) / (s + NORM_EPS) for x, m, s in zip(values, self.low, self.high))
        if self.mode == NORM_MIN_MAX:
            return tuple(
                (x - lo) / max(hi - lo, NORM_EPS) * 2.0 - 1.0
                for x, lo, hi in zip(values, self.low, self.high)
            )
        return tuple(float(v) for v in values)

    def to_dict(self) -> dict[str, Any]:
        if self.mode == NORM_IDENTITY:
            return {"mode": self.mode}
        a, b = _NORM_KEYS[self.mode]
        return {"mode": self.mode, a: list(self.low), b: list(self.high), "checksum": self.checksum}

    @classmethod
    def from_mapping(cls, raw: Any, width: int, where: str) -> "Normalization":
        data = _mapping(raw, where)
        if not data:
            return cls()
        mode = _choice(data.get("mode"), (NORM_IDENTITY, NORM_MEAN_STD, NORM_MIN_MAX), where)
        if mode == NORM_IDENTITY:
            return cls()
        a, b = _NORM_KEYS[mode]
        low = _floats(data.get(a), width, f"{where}.{a}")
        high = _floats(data.get(b), width, f"{where}.{b}")
        if mode == NORM_MEAN_STD and any(s < 0 for s in high):
            raise ValueError(f"{where}.std must be non-negative")
        if mode == NORM_MIN_MAX and any(hi < lo for lo, hi in zip(low, high)):
            raise ValueError(f"{where}.max must not be below {where}.min")
        actual = stats_checksum(mode, low, high)
        declared = str(data.get("checksum") or "").strip()
        if declared and declared != actual:
            raise ValueError(
                f"{where} stats do not match their checksum ({declared} != {actual}); the "
                "server and the node disagree on the training statistics"
            )
        return cls(mode, low, high, declared)


# ── Groups ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Group:
    """One semantic slice of a vector: which dims, what they mean."""

    name: str
    kind: str
    dims: tuple[int, ...]
    unit: str
    rotation: str = ""
    angle_unit: str = "deg"
    reference: str = REFERENCE_ABSOLUTE
    frame: str = FRAME_BASE
    scale: float = 1.0
    offset: float = 0.0
    open_value: float = 1.0
    closed_value: float = 0.0
    binary: bool = False
    threshold: float = 0.5
    hysteresis: float = 0.0

    @property
    def width(self) -> int:
        return len(self.dims)

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)} | {
            "dims": list(self.dims)
        }

    @classmethod
    def from_mapping(cls, data: Any, *, width: int, names: Sequence[str], where: str) -> "Group":
        data = _mapping(data, where)
        unknown = sorted(set(data) - _GROUP_KEYS)
        if unknown:
            raise ValueError(f"{where} does not understand {', '.join(unknown)}")
        name = str(data.get("name") or "").strip()
        if not _FRAME_NAME.fullmatch(name):
            raise ValueError(f"{where} needs a [a-z0-9_] name")
        where = f"{where} {name!r}"
        kind = _choice(data.get("kind"), tuple(UNITS), f"{where}.kind")
        group = cls(
            name=name,
            kind=kind,
            dims=_dims(data.get("dims"), width, names, where),
            unit=_choice(data.get("unit", CANONICAL_UNIT[kind]), UNITS[kind], f"{where}.unit"),
            rotation=str(data.get("rotation", ARM_EULER if kind == KIND_EE_POSE else "")),
            angle_unit=_choice(data.get("angle_unit", "deg"), ANGLE_UNITS, f"{where}.angle_unit"),
            reference=_choice(
                data.get("reference", REFERENCE_ABSOLUTE), REFERENCES, f"{where}.reference"
            ),
            frame=str(data.get("frame", FRAME_BASE)),
            scale=_number(data.get("scale", 1.0), f"{where}.scale"),
            offset=_number(data.get("offset", 0.0), f"{where}.offset"),
            open_value=_number(data.get("open_value", 1.0), f"{where}.open_value"),
            closed_value=_number(data.get("closed_value", 0.0), f"{where}.closed_value"),
            binary=bool(data.get("binary", False)),
            threshold=_number(data.get("threshold", 0.5), f"{where}.threshold"),
            hysteresis=_number(data.get("hysteresis", 0.0), f"{where}.hysteresis"),
        )
        _check_group(group, where)
        return group


def _dims(raw: Any, width: int, names: Sequence[str], where: str) -> tuple[int, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise ValueError(f"{where}.dims must list indices or names")
    dims: list[int] = []
    for item in raw:
        if isinstance(item, str):
            if item not in names:
                raise ValueError(f"{where}.dims names {item!r}, which the space does not")
            dims.append(names.index(item))
        elif isinstance(item, int) and not isinstance(item, bool) and 0 <= item < width:
            dims.append(item)
        else:
            raise ValueError(f"{where}.dims entry {item!r} is not an index below {width}")
    if len(set(dims)) != len(dims):
        raise ValueError(f"{where}.dims names a dimension twice")
    return tuple(dims)


def _check_group(group: Group, where: str) -> None:
    if group.scale == 0:
        raise ValueError(f"{where}.scale must not be zero")
    if group.kind == KIND_EE_POSE:
        _check_pose_group(group, where)
        return
    if group.rotation or group.frame != FRAME_BASE:
        raise ValueError(f"{where}: rotation and frame apply to ee_pose groups only")
    if group.kind == KIND_GRIPPER:
        if group.width != 1:
            raise ValueError(f"{where}: a gripper group is one dimension")
        if group.unit == UNIT_NORMALIZED and group.open_value == group.closed_value:
            raise ValueError(f"{where}: open_value and closed_value must differ")
        if group.binary and group.unit != UNIT_NORMALIZED:
            raise ValueError(f"{where}: a binary gripper must be in normalized units")
        if group.binary and group.reference != REFERENCE_ABSOLUTE:
            raise ValueError(f"{where}: a binary gripper command must be absolute")
        if group.hysteresis < 0:
            raise ValueError(f"{where}.hysteresis must be non-negative")


def _check_pose_group(group: Group, where: str) -> None:
    expected = 3 + rotation_width(group.rotation)
    if group.width != expected:
        raise ValueError(f"{where}: {group.rotation} pose takes {expected} dims, not {group.width}")
    if group.scale != 1.0 or group.offset != 0.0:
        raise ValueError(f"{where}: scale/offset apply to joints and gripper groups only")
    if group.binary:
        raise ValueError(f"{where}: binary applies to gripper groups only")
    if not _FRAME_NAME.fullmatch(group.frame):
        raise ValueError(f"{where}.frame must be base, tool or a workcell frame name")
    absolute = group.reference == REFERENCE_ABSOLUTE
    if absolute and group.frame == FRAME_TOOL:
        raise ValueError(f"{where}: an absolute pose cannot be expressed in the moving tool frame")
    if absolute and group.rotation == ROTATION_NONE and group.frame != FRAME_BASE:
        raise ValueError(f"{where}: a position-only absolute pose must be in the base frame")


# ── Spaces ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VectorSpace:
    """A flat vector split into semantic groups, optionally normalised."""

    width: int
    groups: tuple[Group, ...]
    normalization: Normalization = field(default_factory=Normalization)
    normalized: bool = False
    names: tuple[str, ...] = ()

    def group(self, name: str) -> Group | None:
        return next((g for g in self.groups if g.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "names": list(self.names),
            "normalization": self.normalization.to_dict(),
            "normalized": self.normalized,
            "groups": [group.to_dict() for group in self.groups],
        }

    @classmethod
    def from_mapping(
        cls, raw: Any, where: str, overrides: Mapping[str, Mapping[str, Any]] | None = None
    ) -> "VectorSpace":
        data = _mapping(raw, where)
        width = int(_positive(data.get("width"), f"{where}.width"))
        names = _strings(data.get("names"), f"{where}.names")
        if names and len(names) != width:
            raise ValueError(f"{where}.names must name all {width} dims")
        raw_groups = data.get("groups")
        if isinstance(raw_groups, (str, bytes)) or not isinstance(raw_groups, Sequence):
            raise ValueError(f"{where}.groups must be a list")
        changes = dict(overrides or {})
        groups: list[Group] = []
        for index, item in enumerate(raw_groups):
            at = f"{where}.groups[{index}]"
            item = dict(_mapping(item, at))
            label = f"overrides.{item.get('name')}"
            change = _mapping(changes.pop(str(item.get("name")), None), label)
            bad = sorted(set(change) - _OVERRIDABLE)
            if bad:
                raise ValueError(f"{label} may not change {', '.join(bad)}")
            groups.append(
                Group.from_mapping({**item, **change}, width=width, names=names, where=at)
            )
        if changes:
            raise ValueError(f"overrides name groups {sorted(changes)} {where} does not declare")
        _check_layout(groups, where)
        normalization = Normalization.from_mapping(
            data.get("normalization"), width, f"{where}.normalization"
        )
        normalized = data.get("normalized", normalization.mode != NORM_IDENTITY)
        if normalized and normalization.mode == NORM_IDENTITY:
            raise ValueError(f"{where} is normalized but declares no normalization stats")
        return cls(width, tuple(groups), normalization, bool(normalized), names)


def _check_layout(groups: Sequence[Group], where: str) -> None:
    if not groups:
        raise ValueError(f"{where} declares no groups")
    seen: set[int] = set()
    for group in groups:
        if seen & set(group.dims):
            raise ValueError(f"{where}: group {group.name!r} reuses a dimension")
        seen |= set(group.dims)
    if len({g.name for g in groups}) != len(groups):
        raise ValueError(f"{where} names a group twice")


@dataclass(frozen=True)
class ActionSpace:
    vector: VectorSpace
    fps: float
    pending: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {**self.vector.to_dict(), "fps": self.fps, "pending": self.pending}

    @classmethod
    def from_mapping(
        cls, raw: Any, overrides: Mapping[str, Mapping[str, Any]] | None = None
    ) -> "ActionSpace":
        data = _mapping(raw, "action_space")
        if not data:
            raise ValueError("model_info declares no action_space")
        vector = VectorSpace.from_mapping(data, "action_space", overrides)
        return cls(
            vector, _positive(data.get("fps"), "action_space.fps"), bool(data.get("pending"))
        )


@dataclass(frozen=True)
class ImageSpec:
    """The size a model expects one camera at, and how the frame is fitted to it."""

    width: int
    height: int
    fit: str = FIT_STRETCH

    def to_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height, "fit": self.fit}

    @classmethod
    def from_mapping(cls, raw: Any, where: str) -> "ImageSpec":
        data = _mapping(raw, where)
        width = int(_positive(data.get("width"), f"{where}.width"))
        height = int(_positive(data.get("height"), f"{where}.height"))
        fit = _choice(
            data.get("fit", FIT_STRETCH), (FIT_STRETCH, FIT_CROP, FIT_PAD), f"{where}.fit"
        )
        return cls(width, height, fit)


@dataclass(frozen=True)
class ObservationSpace:
    state: VectorSpace | None = None
    images: Mapping[str, ImageSpec] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": None if self.state is None else self.state.to_dict(),
            "images": {name: spec.to_dict() for name, spec in self.images.items()},
        }

    @classmethod
    def from_mapping(
        cls, raw: Any, overrides: Mapping[str, Mapping[str, Any]] | None = None
    ) -> "ObservationSpace":
        data = _mapping(raw, "observation_space")
        state = None
        if data.get("state") is not None:
            state = VectorSpace.from_mapping(data["state"], "observation_space.state", overrides)
            moving = [g.name for g in state.groups if g.reference != REFERENCE_ABSOLUTE]
            if moving:
                raise ValueError(f"observation_space.state groups {moving} must be absolute")
        elif overrides:
            raise ValueError("overrides.state names groups but the model declares no state")
        images = {
            str(name): ImageSpec.from_mapping(spec, f"observation_space.images.{name}")
            for name, spec in _mapping(data.get("images"), "observation_space.images").items()
        }
        return cls(state, images)


@dataclass(frozen=True)
class ModelSpaces:
    """Everything a served policy declares about what its vectors and words mean."""

    action: ActionSpace
    observation: ObservationSpace
    instruction_policy: str = INSTRUCTION_OPEN
    tasks: tuple[str, ...] = ()
    calibration: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_space": self.action.to_dict(),
            "observation_space": self.observation.to_dict(),
            "instruction_policy": self.instruction_policy,
            "tasks": list(self.tasks),
            "calibration": dict(self.calibration),
        }

    def accepts(self, task: str) -> bool:
        """Whether the model takes *task* as an instruction under its own policy."""
        if self.instruction_policy == INSTRUCTION_OPEN:
            return True
        wanted = task.strip().casefold()
        return wanted in {known.casefold() for known in self.tasks}

    @classmethod
    def from_details(
        cls, details: Mapping[str, Any], overrides: Mapping[str, Any] | None = None
    ) -> "ModelSpaces":
        """Parse ``model_info``'s declarations; raises ``ValueError`` when unusable."""
        changes = _mapping(overrides, "overrides")
        bad = sorted(set(changes) - {"action", "state"})
        if bad:
            raise ValueError(f"overrides keys must be action and state, not {', '.join(bad)}")
        tasks = _strings(details.get("tasks"), "tasks")
        policy = _choice(
            details.get("instruction_policy", INSTRUCTION_OPEN),
            (INSTRUCTION_EXACT, INSTRUCTION_OPEN),
            "instruction_policy",
        )
        if policy == INSTRUCTION_EXACT and not tasks:
            raise ValueError("instruction_policy exact needs the tasks it accepts")
        return cls(
            action=ActionSpace.from_mapping(details.get("action_space"), changes.get("action")),
            observation=ObservationSpace.from_mapping(
                details.get("observation_space"), changes.get("state")
            ),
            instruction_policy=policy,
            tasks=tasks,
            calibration=_versions(details.get("calibration")),
        )


def _versions(raw: Any) -> dict[str, float]:
    versions: dict[str, float] = {}
    for frame, version in _mapping(raw, "calibration").items():
        versions[str(frame)] = _number(version, f"calibration.{frame}")
    return versions


def digest(data: Mapping[str, Any]) -> str:
    """A stable short hash of a declaration (what an episode pins an adapter by)."""
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── Workcell binding ─────────────────────────────────────────────────────────

_DECL_KEYS = frozenset(
    {
        "actions", "state", "cameras", "overrides", "aggregate", "interpolation",
        "smooth_hz", "ik", "steps",
    }
)  # fmt: skip


@dataclass(frozen=True)
class IkSettings:
    """Damped-least-squares IK tuning; lengths mm, angles deg."""

    damping: float = 1.0  # lambda, in the error's mm per joint degree
    max_iterations: int = 50
    tolerance_mm: float = 0.1
    tolerance_deg: float = 0.1
    max_step_deg: float = 10.0  # per joint per iteration
    rotation_weight_mm: float = 200.0  # mm of error one radian of orientation error is worth

    @classmethod
    def from_mapping(cls, raw: Any) -> "IkSettings":
        data = _mapping(raw, "ik")
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"ik does not understand {', '.join(unknown)}")
        values: dict[str, Any] = {k: _positive(v, f"ik.{k}") for k, v in data.items()}
        if "max_iterations" in values:
            values["max_iterations"] = max(1, int(values["max_iterations"]))
        return replace(cls(), **values)


@dataclass(frozen=True)
class StepDecl:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecl:
    """How one workcell embodies a policy: channels, cameras, overrides, aggregation."""

    actions: Mapping[str, tuple[str, ...]]
    state: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    cameras: Mapping[str, str] = field(default_factory=dict)
    overrides: Mapping[str, Mapping[str, Mapping[str, Any]]] = field(default_factory=dict)
    aggregate: str = AGGREGATE_CROSSFADE
    interpolation: str = INTERPOLATION_LINEAR
    smooth_hz: float = 0.0
    ik: IkSettings = field(default_factory=IkSettings)
    steps: tuple[StepDecl, ...] = ()

    def state_channels(self, group: str) -> tuple[str, ...]:
        return self.state.get(group) or self.actions.get(group) or ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": {k: list(v) for k, v in self.actions.items()},
            "state": {k: list(v) for k, v in self.state.items()},
            "cameras": dict(self.cameras),
            "overrides": {k: {g: dict(c) for g, c in v.items()} for k, v in self.overrides.items()},
            "aggregate": self.aggregate,
            "interpolation": self.interpolation,
            "smooth_hz": self.smooth_hz,
            "ik": {item.name: getattr(self.ik, item.name) for item in fields(self.ik)},
            "steps": [{"name": s.name, **dict(s.params)} for s in self.steps],
        }

    @classmethod
    def from_mapping(cls, raw: Any, where: str = "policies") -> "PolicyDecl":
        data = _mapping(raw, where)
        unknown = sorted(set(data) - _DECL_KEYS)
        if unknown:
            raise ValueError(f"{where} does not understand {', '.join(unknown)}")
        actions = _channel_map(data.get("actions"), f"{where}.actions")
        if not actions:
            raise ValueError(f"{where}.actions must bind at least one group")
        smooth = _number(data.get("smooth_hz", 0.0), f"{where}.smooth_hz")
        if smooth < 0:
            raise ValueError(f"{where}.smooth_hz must be non-negative")
        cameras = _mapping(data.get("cameras"), f"{where}.cameras")
        return cls(
            actions=actions,
            state=_channel_map(data.get("state"), f"{where}.state"),
            cameras={str(k): str(v).strip() for k, v in cameras.items()},
            overrides=_overrides(data.get("overrides"), f"{where}.overrides"),
            aggregate=_choice(
                data.get("aggregate", AGGREGATE_CROSSFADE),
                (AGGREGATE_CROSSFADE, AGGREGATE_ENSEMBLE, AGGREGATE_LATEST),
                f"{where}.aggregate",
            ),
            interpolation=_choice(
                data.get("interpolation", INTERPOLATION_LINEAR),
                (INTERPOLATION_LINEAR, INTERPOLATION_HOLD),
                f"{where}.interpolation",
            ),
            smooth_hz=smooth,
            ik=IkSettings.from_mapping(data.get("ik")),
            steps=_step_decls(data.get("steps"), f"{where}.steps"),
        )


def _channel_map(raw: Any, where: str) -> dict[str, tuple[str, ...]]:
    bound: dict[str, tuple[str, ...]] = {}
    for group, channels in _mapping(raw, where).items():
        names = _strings(channels, f"{where}.{group}")
        if not names:
            raise ValueError(f"{where}.{group} binds no channels")
        bound[str(group)] = names
    every = [c for names in bound.values() for c in names]
    if len(set(every)) != len(every):
        raise ValueError(f"{where} binds a channel twice")
    return bound


def _overrides(raw: Any, where: str) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for space, groups in _mapping(raw, where).items():
        result[str(space)] = {
            str(group): dict(_mapping(change, f"{where}.{space}.{group}"))
            for group, change in _mapping(groups, f"{where}.{space}").items()
        }
    return result


def _step_decls(raw: Any, where: str) -> tuple[StepDecl, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{where} must be a list of step names or {{name, ...}} mappings")
    steps: list[StepDecl] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            steps.append(StepDecl(item.strip()))
        elif isinstance(item, Mapping) and str(item.get("name") or "").strip():
            params = {str(k): v for k, v in item.items() if k != "name"}
            steps.append(StepDecl(str(item["name"]).strip(), params))
        else:
            raise ValueError(f"{where} entry {item!r} names no step")
    return tuple(steps)


__all__ = [
    "AGGREGATE_CROSSFADE",
    "AGGREGATE_ENSEMBLE",
    "AGGREGATE_LATEST",
    "CANONICAL_UNIT",
    "FIT_CROP",
    "FIT_PAD",
    "FIT_STRETCH",
    "FRAME_BASE",
    "FRAME_TOOL",
    "INSTRUCTION_EXACT",
    "INSTRUCTION_OPEN",
    "INTERPOLATION_HOLD",
    "INTERPOLATION_LINEAR",
    "KIND_EE_POSE",
    "KIND_GRIPPER",
    "KIND_JOINTS",
    "NORM_IDENTITY",
    "NORM_MEAN_STD",
    "NORM_MIN_MAX",
    "REFERENCE_ABSOLUTE",
    "REFERENCE_DELTA",
    "REFERENCE_RELATIVE",
    "UNIT_NORMALIZED",
    "ActionSpace",
    "Group",
    "IkSettings",
    "ImageSpec",
    "ModelSpaces",
    "Normalization",
    "ObservationSpace",
    "PolicyDecl",
    "StepDecl",
    "VectorSpace",
    "digest",
    "stats_checksum",
]
