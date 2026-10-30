# Copyright (c) Alibaba, Inc. and its affiliates.
"""Workcells: which devices cooperate, in which frames, inside which box.

A device declaration (hc.v1) describes one device. A skill needs more: the arm
*and* the gripper *and* the camera that watches them, the tool centre point on
the flange, where the camera sits relative to the arm base, and the box the
tool may move in. That is a workcell -- a durable, human-authored declaration
under ``<profile>/hardware/workcells/*.yaml``, admitted fail-closed against the
live registry:

.. code-block:: yaml

    workcell: piper_cell
    roles:                                # role name -> device id (open: any [a-z0-9_.] name)
      arm: robot_piper_arm
      gripper: robot_piper_arm
      camera: realsense_d435if
      camera.wrist: wrist_cam             # a qualified role of the camera kind
    frames:
      tcp:    {parent: flange, procedure: tcp.offset}
      camera: {parent: base,   procedure: handeye.eye_to_hand}
    workspace:
      procedure: workspace.safe_box
      tool_cone_deg: 30
      bootstrap_box: {x: [150, 400], y: [-200, 200], z: [120, 400]}
    home: {joint.j1.position: 0.0}        # default: arm kinematics.home_joint_targets
    gripper_spec: {max_width_mm: 70, min_hold_width_mm: 3}   # max defaults to the envelope
    limits: {speed_pct_max: 30, default_speed_pct: 20, move_timeout_s: 20, max_steps: 16}
    freshness: {drift_planar_mm: 8}       # plan->execute drift tolerances (see tuning)
    calibration: {max_tool_length_mm: 400} # calib.* acceptance thresholds (see tuning)
    models: {locator: grounding_sam}      # model name -> model-service device id
    grasp: {planner: top_down}            # planner name plus its own settings
    perception: {min_points: 40}          # optional perception tuning
    policy: {tick_hz: 50}                 # optional policy streaming tuning
    policies:                             # how each served policy embodies here (adapt.spec)
      cube: {actions: {arm: [joint.j1.position, ...], gripper: [gripper.position]}}
    calibration_target:                   # the ChArUco board calib.handeye observes
      {type: charuco, squares_x: 5, squares_y: 4, square_mm: 30, marker_mm: 22}

Geometry is never declared inline: frames and the workspace resolve from
versioned ``CalibrationStore`` records (device id = workcell id), so a skill
plan can name the exact calibration versions it was computed against and a
re-calibration invalidates every plan made before it. A record may also name
the versions it was measured against (``parameters.depends_on``, e.g. a
workspace measured with one TCP); when one of those is superseded the record
is stale and resolves as uncalibrated. ``bootstrap_box`` is the one exception
to "never inline" -- a conservative box only ``calib.*`` skills may move in, so
the workspace can be calibrated at all.

Models are devices too: ``models`` names the model-service devices a skill may
query (``model:<name>`` in its requirements), so swapping a detector or a
policy is a declaration change, never a skill change.

Roles are open: a role's *kind* is its first dotted segment (``camera.wrist``
is a camera), the kind decides which admission rules apply, and a skill asks
for the exact role it uses. Every bound device is leased with the workcell, and
every bound device that declares a writable channel is halted with it. A field
this module does not know is refused, never ignored.

Admission also requires every declared value skills act on: the arm's pose and
joint channels and the gripper's position channel must declare a positive
``tolerance`` (and the gripper a positive ``settling_time_s``), since those are
what "arrived", "stable", and "blocked" are judged by.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from leapflow.hardware.skills.geometry import WorkspaceBox
from leapflow.hardware.skills.tuning import (
    DEFAULT_CALIBRATION,
    DEFAULT_FRESHNESS,
    DEFAULT_GRASP_VERIFY,
    DEFAULT_LIMITS,
    gripper_spec,
    tuned,
)

logger = logging.getLogger(__name__)

ROLE_ARM = "arm"
ROLE_GRIPPER = "gripper"
ROLE_CAMERA = "camera"
ACTUATOR_KINDS = frozenset({ROLE_ARM, ROLE_GRIPPER})

WORKCELL_FIELDS = frozenset(
    {
        "workcell",
        "roles",
        "frames",
        "workspace",
        "home",
        "gripper_spec",
        "limits",
        "freshness",
        "calibration",
        "models",
        "grasp",
        "grasp_verify",
        "perception",
        "policy",
        "policies",
        "calibration_target",
        "skills",
        "share_frames",
    }
)
WORKSPACE_FIELDS = frozenset({"procedure", "tool_cone_deg", "bootstrap_box"})

FRAME_PARENTS = frozenset({"flange", "base"})
WORKSPACE_FRAME = "workspace"
DEPENDS_ON = "depends_on"
DEFAULT_GRASP_PLANNER = "top_down"
_IDENTIFIER = re.compile(r"[a-z0-9_]+")
_ROLE_NAME = re.compile(r"[a-z0-9_]+(\.[a-z0-9_]+)*")

POSE_CHANNELS = ("ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz")
ARRIVED_CHANNEL = "arm.motion"
ENABLED_CHANNEL = "arm.enabled"
CTRL_MODE_CHANNEL = "arm.ctrl_mode"  # optional: standby / can / teach, as the arm reports it
ARM_STATUS_CHANNEL = "arm.status"  # optional: the vendor's verdict on the last target
SPEED_CHANNEL = "arm.speed_pct"
POSE_PATH_CHANNEL = "arm.pose_path"  # optional: the path shape of later pose targets
GRIPPER_POSITION = "gripper.position"
GRIPPER_EFFORT = "gripper.effort"


class WorkcellError(ValueError):
    """A workcell declaration that cannot be admitted."""


def role_kind(role: str) -> str:
    """A role's kind: its first dotted segment (``camera.wrist`` -> ``camera``)."""
    return role.split(".", 1)[0]


@dataclass(frozen=True)
class FrameDecl:
    """A named frame, the frame it is expressed in, and the procedure that measures it."""

    name: str
    parent: str
    procedure: str


@dataclass(frozen=True)
class WorkcellSpec:
    """An admitted workcell. Satisfies ``WorkcellView``."""

    workcell_id: str
    roles: Mapping[str, str]
    frames: Mapping[str, FrameDecl] = field(default_factory=dict)
    workspace_procedure: str = ""
    tool_cone_deg: float = 30.0
    bootstrap_box: WorkspaceBox | None = None
    home: Mapping[str, float] = field(default_factory=dict)
    gripper_spec: Mapping[str, float] = field(default_factory=dict)
    limits: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_LIMITS))
    freshness: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_FRESHNESS))
    calibration: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_CALIBRATION))
    models: Mapping[str, str] = field(default_factory=dict)
    grasp: Mapping[str, Any] = field(default_factory=dict)
    grasp_verify: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_GRASP_VERIFY))
    perception: Mapping[str, float] = field(default_factory=dict)
    policy: Mapping[str, float] = field(default_factory=dict)
    policies: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    calibration_target: Mapping[str, Any] = field(default_factory=dict)
    skills: tuple[str, ...] = ()
    # Whether camera frames may leave the node for the model to look at
    # (``perceive.snapshot``); off unless the operator declares it.
    share_frames: bool = False
    source: str = ""
    # Devices a halt must reach; admission fills it from the live declarations
    # (every bound device with a writable channel). Empty means "by role kind".
    actuators: tuple[str, ...] = ()

    def role(self, name: str) -> str:
        return str(self.roles.get(name) or "")

    def model(self, name: str) -> str:
        """Device id of the model bound as *name*, or ``""``."""
        return str(self.models.get(name) or "")

    @property
    def grasp_planner(self) -> str:
        return str(self.grasp.get("planner") or DEFAULT_GRASP_PLANNER)

    @property
    def device_ids(self) -> tuple[str, ...]:
        """Every distinct device the workcell binds, in declaration order."""
        return self._distinct(self.roles)

    @property
    def actuator_ids(self) -> tuple[str, ...]:
        """The devices a halt must reach."""
        if self.actuators:
            return self.actuators
        return self._distinct(role for role in self.roles if role_kind(role) in ACTUATOR_KINDS)

    def _distinct(self, roles: Iterable[str]) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for role in roles:
            if self.roles.get(role):
                seen.setdefault(self.roles[role], None)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workcell": self.workcell_id,
            "roles": dict(self.roles),
            "frames": {
                name: {"parent": decl.parent, "procedure": decl.procedure}
                for name, decl in self.frames.items()
            },
            "workspace_procedure": self.workspace_procedure,
            "tool_cone_deg": self.tool_cone_deg,
            "home": dict(self.home),
            "gripper_spec": dict(self.gripper_spec),
            "limits": dict(self.limits),
            "freshness": dict(self.freshness),
            "calibration": dict(self.calibration),
            "models": dict(self.models),
            "grasp": {**dict(self.grasp), "planner": self.grasp_planner},
            "grasp_verify": dict(self.grasp_verify),
            "perception": dict(self.perception),
            "policy": dict(self.policy),
            "policies": {name: dict(decl) for name, decl in self.policies.items()},
            "calibration_target": dict(self.calibration_target),
            "share_frames": self.share_frames,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, source: str = "") -> "WorkcellSpec":
        """Parse a declaration; raises ``WorkcellError`` on any malformed field."""
        if not isinstance(data, Mapping):
            raise WorkcellError("a workcell declaration must be a mapping")
        workcell_id = str(data.get("workcell") or "").strip()
        if not _IDENTIFIER.fullmatch(workcell_id):
            raise WorkcellError("'workcell' must be a non-empty [a-z0-9_] identifier")
        unknown = sorted(str(key) for key in data if key not in WORKCELL_FIELDS)
        if unknown:
            raise WorkcellError(
                f"workcell {workcell_id!r} has unknown fields {unknown} (devices bind under "
                "'roles')"
            )
        roles = _parse_roles(data.get("roles"), workcell_id)
        workspace = data.get("workspace") or {}
        if not isinstance(workspace, Mapping):
            raise WorkcellError("'workspace' must be a mapping")
        unknown = sorted(str(key) for key in workspace if key not in WORKSPACE_FIELDS)
        if unknown:
            raise WorkcellError(f"workspace has unknown fields {unknown}")
        try:
            bootstrap = workspace.get("bootstrap_box")
            frames = _parse_frames(data.get("frames") or {}, roles=roles)
            procedures = [decl.procedure for decl in frames.values()]
            if workspace.get("procedure"):
                procedures.append(str(workspace["procedure"]))
            if len(set(procedures)) != len(procedures):
                raise WorkcellError("calibration procedures must be unique within a workcell")
            return cls(
                workcell_id=workcell_id,
                roles=roles,
                frames=frames,
                workspace_procedure=str(workspace.get("procedure") or ""),
                tool_cone_deg=_positive(workspace.get("tool_cone_deg", 30.0), "tool_cone_deg"),
                bootstrap_box=WorkspaceBox.from_mapping(bootstrap) if bootstrap else None,
                home=_float_map(data.get("home") or {}, "home"),
                gripper_spec=gripper_spec(data.get("gripper_spec") or {}),
                limits=_parse_limits(data.get("limits") or {}),
                freshness=tuned(data.get("freshness") or {}, "freshness", DEFAULT_FRESHNESS),
                calibration=tuned(
                    data.get("calibration") or {}, "calibration", DEFAULT_CALIBRATION
                ),
                models=_parse_models(data.get("models") or {}),
                grasp=_parse_grasp(data.get("grasp") or {}),
                grasp_verify=_parse_grasp_verify(data.get("grasp_verify") or {}),
                perception=_float_map(data.get("perception") or {}, "perception"),
                policy=_float_map(data.get("policy") or {}, "policy"),
                policies=_parse_policies(data.get("policies") or {}),
                calibration_target=_mapping(
                    data.get("calibration_target") or {}, "calibration_target"
                ),
                skills=tuple(str(item) for item in data.get("skills") or ()),
                share_frames=_flag(data.get("share_frames", False), "share_frames"),
                source=source,
            )
        except WorkcellError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkcellError(f"workcell {workcell_id!r}: {exc}") from exc


# ── Admission ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WorkcellNote:
    """Why one declaration was rejected."""

    source: str
    workcell: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "workcell": self.workcell, "detail": self.detail}


@dataclass(frozen=True)
class WorkcellLoadReport:
    admitted: tuple[str, ...] = ()
    rejected: tuple[WorkcellNote, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": list(self.admitted),
            "rejected": [note.to_dict() for note in self.rejected],
        }


AdmissionCheck = Callable[["WorkcellSpec"], Sequence[str]]
"""An extra admission rule a caller owns (e.g. "the grasp planner is installed")."""


def admission_problems(
    spec: WorkcellSpec, registry: Any, *, checks: Iterable[AdmissionCheck] = ()
) -> list[str]:
    """Check *spec* against the live device declarations; empty means admissible.

    *checks* add the caller's own rules, so the runtime can refuse a workcell
    naming a planner or tuning it cannot honour without this module knowing them.
    """
    problems: list[str] = []
    contexts: dict[str, Any] = {}
    for role, device_id in spec.roles.items():
        context = registry.context(device_id)
        if context is None:
            problems.append(f"{role} device {device_id!r} is not declared on this node")
        else:
            contexts[role] = context
    for role, context in contexts.items():
        kind = role_kind(role)
        if kind == ROLE_ARM:
            problems.extend(_arm_problems(context))
        elif kind == ROLE_GRIPPER:
            problems.extend(_gripper_problems(role, context, spec))
    arm = contexts.get(ROLE_ARM)
    if arm is not None:
        problems.extend(_home_problems(_resolved_home(spec, arm), arm))
        problems.extend(_speed_problems(spec, arm))
    for name, device_id in spec.models.items():
        if registry.context(device_id) is None:
            problems.append(f"model {name!r} device {device_id!r} is not declared on this node")
    for check in checks:
        problems.extend(check(spec))
    return problems


def load_workcells(
    directory: Path, registry: Any, *, checks: Iterable[AdmissionCheck] = ()
) -> tuple[dict[str, WorkcellSpec], WorkcellLoadReport]:
    """Load and admit every ``*.yaml`` in *directory*; rejections are reported, not raised."""
    import yaml

    rules = tuple(checks)
    admitted: dict[str, WorkcellSpec] = {}
    rejected: list[WorkcellNote] = []
    paths = sorted(directory.glob("*.yaml")) if directory.is_dir() else []
    for path in paths:
        try:
            spec = WorkcellSpec.from_mapping(
                yaml.safe_load(path.read_text(encoding="utf-8")) or {}, source=str(path)
            )
        except (OSError, yaml.YAMLError, WorkcellError) as exc:
            rejected.append(WorkcellNote(str(path), "", str(exc)))
            continue
        if spec.workcell_id in admitted:
            rejected.append(WorkcellNote(str(path), spec.workcell_id, "duplicate workcell id"))
            continue
        spec = resolve_declared(spec, registry)
        problems = admission_problems(spec, registry, checks=rules)
        if problems:
            rejected.append(WorkcellNote(str(path), spec.workcell_id, "; ".join(problems)))
            continue
        admitted[spec.workcell_id] = replace(spec, actuators=_actuators(spec, registry))
    for note in rejected:
        logger.warning("workcell rejected (%s): %s", note.source, note.detail)
    return admitted, WorkcellLoadReport(tuple(admitted), tuple(rejected))


def resolve_declared(spec: WorkcellSpec, registry: Any) -> WorkcellSpec:
    """Fill what the device declarations imply: the arm's home, the gripper's width.

    An undeclared ``gripper_spec.max_width_mm`` is the gripper position
    envelope's ``max_value``; a declared one is checked against it at admission.
    """
    arm = registry.context(spec.role(ROLE_ARM))
    resolved = spec if arm is None else _with_home(spec, _resolved_home(spec, arm))
    gripper = registry.context(spec.role(ROLE_GRIPPER)) if spec.role(ROLE_GRIPPER) else None
    channel = gripper.channel(GRIPPER_POSITION) if gripper is not None else None
    widest = None if channel is None else channel.envelope.max_value
    if widest is not None and "max_width_mm" not in resolved.gripper_spec:
        resolved = replace(
            resolved, gripper_spec={**resolved.gripper_spec, "max_width_mm": float(widest)}
        )
    return resolved


def _arm_problems(arm: Any) -> list[str]:
    problems: list[str] = []
    missing = [cid for cid in POSE_CHANNELS if not _writable(arm, cid)]
    if missing:
        problems.append(f"arm {arm.device_id!r} lacks writable pose channels {missing}")
    arrived = arm.channel(ARRIVED_CHANNEL)
    if arrived is None or not arrived.is_readable:
        problems.append(f"arm {arm.device_id!r} lacks a readable {ARRIVED_CHANNEL!r}")
    joints = joint_channels(arm)
    if not joints:
        problems.append(f"arm {arm.device_id!r} declares no joint.*.position channels")
    untoleranced = [
        cid
        for cid in (*POSE_CHANNELS, *joints)
        if arm.channel(cid) is not None and not _declared_positive(arm, cid, "tolerance")
    ]
    if untoleranced:
        problems.append(
            f"arm {arm.device_id!r} declares no positive tolerance on {untoleranced} "
            "(arrival is judged by it)"
        )
    path = arm.channel(POSE_PATH_CHANNEL)
    if path is not None and not (
        path.is_writable
        and path.envelope.allowed_values
        and all(isinstance(v, str) and v for v in path.envelope.allowed_values)
    ):
        problems.append(
            f"arm {arm.device_id!r} {POSE_PATH_CHANNEL} must be writable and enumerate "
            "its path names in allowed_values"
        )
    return problems


def _gripper_problems(role: str, gripper: Any, spec: WorkcellSpec) -> list[str]:
    if not _writable(gripper, GRIPPER_POSITION):
        return [f"{role} {gripper.device_id!r} has no writable {GRIPPER_POSITION!r}"]
    problems = [
        f"{role} {gripper.device_id!r} {GRIPPER_POSITION} declares no positive {name} "
        "(gripper stability is judged by it)"
        for name in ("tolerance", "settling_time_s")
        if not _declared_positive(gripper, GRIPPER_POSITION, name)
    ]
    widest = gripper.channel(GRIPPER_POSITION).envelope.max_value
    declared = spec.gripper_spec.get("max_width_mm")
    if declared is None:
        problems.append(
            f"gripper_spec.max_width_mm is not declared and {GRIPPER_POSITION} has no max_value"
        )
    elif widest is not None and declared > widest:
        problems.append(
            f"gripper_spec.max_width_mm {declared:g} exceeds the {GRIPPER_POSITION} envelope "
            f"max {widest:g}"
        )
    if declared is not None and spec.gripper_spec["min_hold_width_mm"] >= declared:
        problems.append("gripper_spec.min_hold_width_mm must be below max_width_mm")
    return problems


def _declared_positive(context: Any, channel_id: str, name: str) -> bool:
    value = getattr(context.channel(channel_id).envelope, name, None)
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _speed_problems(spec: WorkcellSpec, arm: Any) -> list[str]:
    """A speed cap the arm cannot command would refuse every move at run time."""
    channel = arm.channel(SPEED_CHANNEL)
    floor = None if channel is None else channel.envelope.min_value
    if floor is None or spec.limits["speed_pct_max"] >= floor:
        return []
    return [
        f"limits.speed_pct_max {spec.limits['speed_pct_max']:g} is below the arm's lowest "
        f"commandable speed {floor:g}"
    ]


def _actuators(spec: WorkcellSpec, registry: Any) -> tuple[str, ...]:
    """Every bound device that can be commanded, plus every arm/gripper-kind role."""
    found: dict[str, None] = {device: None for device in spec.actuator_ids}
    for device_id in spec.device_ids:
        context = registry.context(device_id)
        channels = getattr(context, "channels", ()) if context is not None else ()
        if any(channel.is_writable for channel in channels):
            found.setdefault(device_id, None)
    return tuple(found)


def _home_problems(home: Mapping[str, float], arm: Any) -> list[str]:
    problems: list[str] = []
    for channel_id, value in home.items():
        channel = arm.channel(channel_id)
        if channel is None or not channel.is_writable:
            problems.append(f"home names {channel_id!r}, not a writable channel on the arm")
        elif not channel.envelope.contains(value):
            problems.append(f"home {channel_id}={value} is outside its envelope")
    return problems


def joint_channels(arm: Any) -> tuple[str, ...]:
    """The arm's joint position channels, in declaration order."""
    return tuple(
        channel.channel_id
        for channel in arm.channels
        if channel.channel_id.startswith("joint.") and channel.channel_id.endswith(".position")
    )


def _resolved_home(spec: WorkcellSpec, arm: Any) -> Mapping[str, float]:
    if spec.home:
        return spec.home
    kinematics = getattr(arm, "kinematics", None)
    return dict(getattr(kinematics, "home_joint_targets", None) or {})


def _with_home(spec: WorkcellSpec, home: Mapping[str, float]) -> WorkcellSpec:
    return spec if home == spec.home else replace(spec, home=dict(home))


def _writable(context: Any, channel_id: str) -> bool:
    channel = context.channel(channel_id)
    return channel is not None and channel.is_writable


# ── Parsing helpers ──────────────────────────────────────────────────────────


def _parse_frames(raw: Any, *, roles: Mapping[str, str]) -> dict[str, FrameDecl]:
    if not isinstance(raw, Mapping):
        raise WorkcellError("'frames' must be a mapping")
    frames: dict[str, FrameDecl] = {}
    for name, decl in raw.items():
        name = str(name)
        if not _ROLE_NAME.fullmatch(name):
            raise WorkcellError(f"invalid frame name {name!r}")
        if role_kind(name) == ROLE_CAMERA and name not in roles:
            raise WorkcellError(f"camera frame {name!r} must bind a camera role")
        if name == WORKSPACE_FRAME or name in FRAME_PARENTS:
            raise WorkcellError(f"frame name {name!r} is reserved")
        if not isinstance(decl, Mapping):
            raise WorkcellError(f"frame {name!r} must be a mapping")
        if set(decl) - {"parent", "procedure"}:
            raise WorkcellError(f"frame {name!r} has unknown fields")
        parent = str(decl.get("parent") or "")
        if parent not in FRAME_PARENTS:
            raise WorkcellError(f"frame {name!r} parent must be one of {sorted(FRAME_PARENTS)}")
        procedure = str(decl.get("procedure") or "")
        if not procedure:
            raise WorkcellError(f"frame {name!r} needs a calibration 'procedure'")
        frames[name] = FrameDecl(name, parent, procedure)
    return frames


def _parse_roles(raw: Any, workcell_id: str) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise WorkcellError(f"workcell {workcell_id!r} needs 'roles': role name -> device id")
    roles: dict[str, str] = {}
    for name, device_id in raw.items():
        if not isinstance(name, str) or not _ROLE_NAME.fullmatch(name):
            raise WorkcellError(f"role name {name!r} must be [a-z0-9_] segments joined by '.'")
        if not isinstance(device_id, str) or not device_id.strip():
            raise WorkcellError(f"role {name!r} must name a device id")
        roles[name] = device_id.strip()
    if ROLE_ARM not in roles:
        raise WorkcellError(f"workcell {workcell_id!r} must bind an 'arm' role")
    return roles


def _parse_models(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise WorkcellError("'models' must be a mapping of model name to device id")
    models: dict[str, str] = {}
    for name, device_id in raw.items():
        if not _IDENTIFIER.fullmatch(str(name)):
            raise WorkcellError(f"model name {name!r} must be a [a-z0-9_] identifier")
        if not isinstance(device_id, str) or not device_id.strip():
            raise WorkcellError(f"model {name!r} must name a device id")
        models[str(name)] = device_id.strip()
    return models


def _parse_grasp(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise WorkcellError("'grasp' must be a mapping")
    planner = raw.get("planner", DEFAULT_GRASP_PLANNER)
    if not isinstance(planner, str) or not _IDENTIFIER.fullmatch(planner):
        raise WorkcellError("grasp.planner must be a [a-z0-9_] identifier")
    return {**{str(k): v for k, v in raw.items()}, "planner": planner}


def _parse_policies(raw: Any) -> dict[str, dict[str, Any]]:
    """Policy name -> its embodiment binding (validated against the model at admission)."""
    if not isinstance(raw, Mapping):
        raise WorkcellError("'policies' must be a mapping of policy name to its binding")
    policies: dict[str, dict[str, Any]] = {}
    for name, decl in raw.items():
        if not _IDENTIFIER.fullmatch(str(name)):
            raise WorkcellError(f"policy name {name!r} must be a [a-z0-9_] identifier")
        policies[str(name)] = _mapping(decl, f"policies.{name}")
    return policies


def _mapping(raw: Any, name: str) -> dict[str, Any]:
    """A free-form section whose content its consumer validates at admission."""
    if not isinstance(raw, Mapping):
        raise WorkcellError(f"{name!r} must be a mapping")
    return {str(key): value for key, value in raw.items()}


def _parse_grasp_verify(raw: Any) -> dict[str, float]:
    values = tuned(raw, "grasp_verify", DEFAULT_GRASP_VERIFY)
    count = values["min_frames"]
    if count != int(count) or not 3 <= count <= 10:
        raise WorkcellError("grasp_verify.min_frames must be a whole number in [3, 10]")
    if values["min_proof_lift_mm"] > values["max_proof_lift_mm"]:
        raise WorkcellError("grasp_verify proof-lift bounds are reversed")
    if values["poll_s"] * count >= values["max_age_s"]:
        raise WorkcellError("grasp_verify sampling window must fit within max_age_s")
    budget = values["cycle_max_waypoints"]
    if budget != int(budget) or not 7 <= budget <= 4096:
        raise WorkcellError("grasp_verify.cycle_max_waypoints must be a whole number in [7, 4096]")
    return values


def _parse_limits(raw: Any) -> dict[str, float]:
    limits = tuned(raw, "limits", DEFAULT_LIMITS)
    for key in ("speed_pct_max", "default_speed_pct"):
        if not 0.0 < limits[key] <= 100.0:
            raise WorkcellError(f"limits.{key} must be in (0, 100]")
    if limits["max_steps"] != int(limits["max_steps"]):
        raise WorkcellError("limits.max_steps must be a whole number")
    return limits


def _float_map(raw: Any, name: str) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise WorkcellError(f"{name!r} must be a mapping")
    result: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WorkcellError(f"{name}.{key} must be a number")
        if not math.isfinite(float(value)):
            raise WorkcellError(f"{name}.{key} must be finite")
        result[str(key)] = float(value)
    return result


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise WorkcellError(f"{name} must be a positive number")
    return float(value)


def _flag(value: Any, name: str) -> bool:
    """A YAML boolean; anything else (``"yes"``, ``1``) is refused rather than guessed."""
    if not isinstance(value, bool):
        raise WorkcellError(f"{name} must be true or false")
    return value


__all__ = [
    "ACTUATOR_KINDS",
    "ARRIVED_CHANNEL",
    "DEFAULT_GRASP_PLANNER",
    "DEPENDS_ON",
    "ENABLED_CHANNEL",
    "GRIPPER_EFFORT",
    "GRIPPER_POSITION",
    "POSE_CHANNELS",
    "POSE_PATH_CHANNEL",
    "ROLE_ARM",
    "ROLE_CAMERA",
    "ROLE_GRIPPER",
    "SPEED_CHANNEL",
    "WORKCELL_FIELDS",
    "WORKSPACE_FRAME",
    "AdmissionCheck",
    "FrameDecl",
    "WorkcellError",
    "WorkcellLoadReport",
    "WorkcellNote",
    "WorkcellSpec",
    "admission_problems",
    "joint_channels",
    "load_workcells",
    "resolve_declared",
    "role_kind",
]
