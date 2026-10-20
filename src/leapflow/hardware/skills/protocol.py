# Copyright (c) Alibaba, Inc. and its affiliates.
"""The node-skill contract: specs, plans, outcomes, and the values ports carry.

A node skill is the S1.5 layer: a parameterised behaviour that runs entirely on
the hardware node, close to the devices, under the node's write gate. The host
(S2, the LLM) chooses a skill and its arguments; the node plans, the host approves
the plan, and the node executes exactly that plan.

Dependency rule (enforced by a contract test): skill implementations import only
this module, ``ports``, ``geometry``, and the standard library. They reach
devices solely through ``ports.SkillContext`` -- never a driver, a vendor SDK, a
registry, or a model library -- so swapping an arm, a camera, or a model never
touches a skill.

Two-phase execution:

- ``plan(ctx, args)`` is read-only and repeatable. It returns a ``SkillPlan``
  whose ``digest`` the approval and the execution permit bind, or a terminal
  ``SkillOutcome`` (side effect NONE) when no plan exists.
- ``execute(ctx, plan)`` runs only an approved plan and re-validates its
  ``basis`` first; a basis that no longer holds is ``plan_stale`` (NONE).

READ-effect skills fold both phases into one call and need no approval. The
digests that bind an approval to one plan live in ``digest``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from leapflow.hardware.context import HardwareEffect
from leapflow.hardware.skills.geometry import Pose, Transform, Vec3
from leapflow.hardware.transport import (
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_PARTIAL,
    SIDE_EFFECT_UNKNOWN,
)

# ── Vocabulary ───────────────────────────────────────────────────────────────

EFFECT_READ = HardwareEffect.READ.value
EFFECT_ACTUATE = HardwareEffect.ACTUATE.value
SKILL_EFFECTS = frozenset({EFFECT_READ, EFFECT_ACTUATE})

STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_HALTED = "halted"
STATUS_ESCALATED = "escalated"

# Planning-time verdicts: nothing moved (side effect NONE).
FAILURE_INVALID_ARGS = "invalid_args"
FAILURE_UNKNOWN_SKILL = "unknown_skill"
FAILURE_UNKNOWN_WORKCELL = "unknown_workcell"
FAILURE_SKILL_UNAVAILABLE = "skill_unavailable"
FAILURE_CALIBRATION_MISSING = "calibration_missing"
FAILURE_CALIBRATION_STALE = "calibration_stale"
FAILURE_TARGET_NOT_FOUND = "target_not_found"
FAILURE_UNREACHABLE = "unreachable"
FAILURE_OUT_OF_WORKSPACE = "out_of_workspace"
FAILURE_PLAN_STALE = "plan_stale"
FAILURE_PLAN_UNKNOWN = "plan_unknown"
FAILURE_ARM_NOT_READY = "arm_not_ready"
FAILURE_DEVICE_BUSY = "device_busy"
FAILURE_WRITE_REFUSED = "write_refused"
# Execution-time verdicts: something may have moved.
# A declared step expectation (``held``, ``released``, ``reached``...) did not hold.
FAILURE_GUARD_FAILED = "guard_failed"
FAILURE_SHIELD_VIOLATION = "shield_violation"
FAILURE_TIMEOUT = "timeout"
FAILURE_MODEL_UNAVAILABLE = "model_unavailable"
FAILURE_STOPPED = "stopped"
FAILURE_LEASE_LOST = "lease_lost"
FAILURE_ESTOPPED = "estopped"
FAILURE_DEVICE_FAULT = "device_fault"
FAILURE_INTERNAL_DEFECT = "internal_defect"
# Either phase: a calibration or bring-up check measured out of tolerance. The
# side effect state tells whether the measurement moved the arm.
FAILURE_CHECK_FAILED = "check_failed"

REQUIRE_ROLE = "role:"
REQUIRE_CALIBRATION = "calib:"
REQUIRE_MODEL = "model:"
# Something the workcell discloses beyond geometry: ``share:frames`` needs the
# workcell's ``share_frames`` and, per request, the host's consent.
REQUIRE_SHARE = "share:"
SHARE_FRAMES = "frames"

# Who grounded a ``LocatedObject``: the workcell's locator model, or S2 itself
# pointing into a shared frame (``perceive.lift``).
SOURCE_LOCATOR = "locator"
SOURCE_LIFT = "lift"

# How ``perceive.lift`` reads S2's coordinates: pixels of the shared snapshot,
# fractions of its sides, or thousandths of them (as many VLMs point).
COORDS_PIXEL = "pixel"
COORDS_UNIT = "unit"
COORDS_PER_MILLE = "per_mille"
COORDS = (COORDS_PIXEL, COORDS_UNIT, COORDS_PER_MILLE)

# How far one human approval of an actuating skill reaches. ``session`` lets a
# consent cover later plans of the same skill on the same workcell in this
# session; ``each_run`` (calibration, anything that sweeps the arm through poses
# nobody reviewed one by one) binds consent to a single plan digest.
CONSENT_SESSION = "session"
CONSENT_EACH_RUN = "each_run"
CONSENT_SCOPES = frozenset({CONSENT_SESSION, CONSENT_EACH_RUN})

SIDE_EFFECT_STATES = (
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_PARTIAL,
    SIDE_EFFECT_UNKNOWN,
)


class SkillError(Exception):
    """A typed, terminal skill verdict raised from inside ``plan``/``execute``.

    Deliberately not a frozen dataclass: Python assigns ``__traceback__`` on every
    re-raise. The runtime converts it into a ``SkillOutcome``.
    """

    def __init__(
        self,
        failure_code: str,
        message: str,
        *,
        side_effect_state: str = SIDE_EFFECT_NONE,
        status: str = STATUS_FAILED,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code
        self.message = message
        self.side_effect_state = side_effect_state
        self.status = status
        self.result = dict(result or {})


def _plain(value: Any) -> dict[str, Any]:
    """A flat dataclass as JSON-ready fields: tuples become lists, mappings dicts."""
    out: dict[str, Any] = {}
    for item in fields(value):
        raw = getattr(value, item.name)
        out[item.name] = (
            list(raw) if isinstance(raw, tuple) else dict(raw) if isinstance(raw, Mapping) else raw
        )
    return out


# ── Value types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SkillSpec:
    """What a skill is, as the LLM and the approval layer see it.

    ``requires`` tokens: ``role:<name>`` (``arm``, ``camera.wrist``...); ``calib:<frame>`` where
    ``<frame>`` is a workcell frame name (``tcp``, ``camera``) or ``workspace``;
    and ``model:<name>`` for a model the workcell binds under ``models``.

    ``leased`` marks a READ skill that still runs as a leased run holding the
    workcell's devices (a recording) instead of answering from ``plan``: it is
    planned and approved like an actuating skill, and its run can command nothing.
    """

    skill_id: str
    version: str
    card: str
    args_schema: Mapping[str, Any]
    effect: str = EFFECT_ACTUATE
    requires: tuple[str, ...] = ()
    consent: str = CONSENT_SESSION
    leased: bool = False

    def __post_init__(self) -> None:
        if self.effect not in SKILL_EFFECTS:
            raise ValueError(
                f"skill {self.skill_id!r} effect must be one of {sorted(SKILL_EFFECTS)}"
            )
        if self.consent not in CONSENT_SCOPES:
            raise ValueError(
                f"skill {self.skill_id!r} consent must be one of {sorted(CONSENT_SCOPES)}"
            )
        if not self.card.strip():
            raise ValueError(f"skill {self.skill_id!r} needs a card")

    @property
    def is_actuating(self) -> bool:
        return self.effect == EFFECT_ACTUATE

    @property
    def runs_leased(self) -> bool:
        """Whether executing the skill starts a leased run (versus answering from plan)."""
        return self.is_actuating or self.leased

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class SkillPlan:
    """One concrete, approvable intention. The runtime fills identity and digest."""

    skill_id: str
    args: Mapping[str, Any]
    summary: str
    targets: tuple[Pose, ...] = ()
    limits: Mapping[str, float] = field(default_factory=dict)
    basis: Mapping[str, Any] = field(default_factory=dict)
    workcell: str = ""
    version: str = ""
    digest: str = ""
    created_at: float = field(default_factory=time.time)

    def digest_payload(self) -> dict[str, Any]:
        return {
            "workcell": self.workcell,
            "skill_id": self.skill_id,
            "version": self.version,
            "args": dict(self.args),
            "summary": self.summary,
            "targets": [target.to_dict() for target in self.targets],
            "limits": dict(self.limits),
            "basis": dict(self.basis),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.digest_payload(), "digest": self.digest, "created_at": self.created_at}


@dataclass(frozen=True)
class SkillOutcome:
    """The terminal, typed result of a skill run (or of a plan that could not form)."""

    status: str
    failure_code: str | None = None
    side_effect_state: str = SIDE_EFFECT_NONE
    result: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == STATUS_SUCCEEDED

    @classmethod
    def succeeded(
        cls,
        result: Mapping[str, Any] | None = None,
        *,
        side_effect_state: str = SIDE_EFFECT_COMMITTED,
        message: str = "",
    ) -> "SkillOutcome":
        return cls(
            status=STATUS_SUCCEEDED,
            side_effect_state=side_effect_state,
            result=dict(result or {}),
            message=message,
        )

    @classmethod
    def failed(
        cls,
        failure_code: str,
        message: str,
        *,
        side_effect_state: str = SIDE_EFFECT_NONE,
        status: str = STATUS_FAILED,
        result: Mapping[str, Any] | None = None,
    ) -> "SkillOutcome":
        return cls(
            status=status,
            failure_code=failure_code,
            side_effect_state=side_effect_state,
            result=dict(result or {}),
            message=message,
        )

    @classmethod
    def from_error(cls, error: SkillError) -> "SkillOutcome":
        return cls.failed(
            error.failure_code,
            error.message,
            side_effect_state=error.side_effect_state,
            status=error.status,
            result=error.result,
        )

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class ArmState:
    """One coherent snapshot of the arm, read in a single batch."""

    flange: Pose
    tcp: Pose
    joints: Mapping[str, float] = field(default_factory=dict)
    gripper_mm: float | None = None
    enabled: bool | None = None
    arrived: bool | None = None
    observed_at: float = 0.0
    ctrl_mode: str | None = None
    # The vendor's verdict on the last commanded target (0 = normal). A refusing
    # code means its inverse kinematics rejected the pose before moving: the arm
    # is not late, it is never going to arrive.
    arm_status: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tcp": self.tcp.to_dict(),
            "flange": self.flange.to_dict(),
            "joints": {k: round(v, 3) for k, v in self.joints.items()},
            "gripper_mm": self.gripper_mm,
            "enabled": self.enabled,
            "arrived": self.arrived,
            "ctrl_mode": self.ctrl_mode,
            "arm_status": self.arm_status,
        }


@dataclass(frozen=True)
class MotionResult:
    """Where a governed move ended and how long it took."""

    state: ArmState
    elapsed_s: float
    commanded: bool = True


@dataclass(frozen=True)
class GripperResult:
    """Where the gripper settled; ``blocked`` means it stopped short of the target."""

    width_mm: float
    target_mm: float
    blocked: bool
    elapsed_s: float


@dataclass(frozen=True)
class LocatedObject:
    """One object a perception pass found, in the arm base frame (mm, deg).

    ``center`` is the middle of the object's oriented box: the footprint centre
    seen from above, half way between the support surface and the object's top.
    ``yaw_deg`` is the footprint's major axis, wrapped to [-90, 90).
    ``geometry`` says what the box was lifted from: a ``mask`` or a ``box`` of
    pixels, or a single ``point`` -- a surface point with no extent, which
    ``center`` then is. ``source`` says who grounded it: the workcell's locator
    model (``locator``), or S2 pointing into a shared frame (``lift``) -- which
    is how a relocation knows whether the label can find it again.
    """

    object_id: str
    label: str
    score: float
    center: Vec3
    length: float
    width: float
    height: float
    yaw_deg: float
    observed_at: float
    points: int = 0
    geometry: str = "mask"
    source: str = SOURCE_LOCATOR

    @property
    def top_z(self) -> float:
        return self.center[2] + self.height / 2.0

    @property
    def bottom_z(self) -> float:
        return self.center[2] - self.height / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "score": round(self.score, 3),
            "center": [round(v, 1) for v in self.center],
            "size": {
                "length": round(self.length, 1),
                "width": round(self.width, 1),
                "height": round(self.height, 1),
            },
            "yaw_deg": round(self.yaw_deg, 1),
            "observed_at": self.observed_at,
            "points": self.points,
            "geometry": self.geometry,
            "source": self.source,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LocatedObject":
        """Parse ``to_dict`` output (e.g. from a plan basis); raises on a bad value."""
        size = data["size"]
        center = tuple(float(v) for v in data["center"])
        if len(center) != 3:
            raise ValueError("an object centre needs three coordinates")
        return cls(
            object_id=str(data["object_id"]),
            label=str(data["label"]),
            score=float(data["score"]),
            center=center,  # type: ignore[arg-type]
            length=float(size["length"]),
            width=float(size["width"]),
            height=float(size["height"]),
            yaw_deg=float(data["yaw_deg"]),
            observed_at=float(data.get("observed_at") or 0.0),
            points=int(data.get("points") or 0),
            geometry=str(data.get("geometry") or "mask"),
            source=str(data.get("source") or SOURCE_LOCATOR),
        )


@dataclass(frozen=True)
class PixelTarget:
    """Where S2 points in a shared frame: a ``point`` ``(u, v)`` or a ``box``
    ``(x0, y0, x1, y1)``, in the coordinates the ``lift`` call names."""

    label: str
    point: tuple[float, float] | None = None
    box: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if (self.point is None) == (self.box is None):
            raise ValueError("a target is exactly one of a point or a box")


@dataclass(frozen=True)
class GraspCandidate:
    """A TCP pose to close the gripper at, with the straight approach and retreat.

    ``pregrasp`` lies *approach_mm* and ``retreat`` *retreat_mm* back along the
    tool axis from the grasp. ``grasp_id``/``object_id`` are filled when the
    scene remembers the candidate; a planner leaves them empty.
    """

    grasp: Pose
    width_mm: float
    open_mm: float
    approach_mm: float
    score: float
    planner: str
    retreat_mm: float = 0.0
    grasp_id: str = ""
    object_id: str = ""

    @property
    def pregrasp(self) -> Pose:
        return self._back(self.approach_mm)

    @property
    def retreat(self) -> Pose:
        return self._back(self.retreat_mm or self.approach_mm)

    def waypoint(self, name: str) -> Pose:
        """``pregrasp``, ``grasp`` or ``retreat``; raises ``KeyError`` on anything else."""
        if name == "grasp":
            return self.grasp
        if name == "pregrasp":
            return self.pregrasp
        if name == "retreat":
            return self.retreat
        raise KeyError(name)

    def _back(self, distance: float) -> Pose:
        ax, ay, az = self.grasp.tool_axis
        return self.grasp.translated(-ax * distance, -ay * distance, -az * distance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grasp_id": self.grasp_id,
            "object_id": self.object_id,
            "grasp": self.grasp.to_dict(),
            "pregrasp": self.pregrasp.to_dict(),
            "retreat": self.retreat.to_dict(),
            "width_mm": round(self.width_mm, 1),
            "open_mm": round(self.open_mm, 1),
            "approach_mm": round(self.approach_mm, 1),
            "retreat_mm": round(self.retreat_mm or self.approach_mm, 1),
            "score": round(self.score, 3),
            "planner": self.planner,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GraspCandidate":
        """Parse ``to_dict`` output (e.g. from a plan basis); raises on a bad value."""
        return cls(
            grasp=Pose.from_mapping(data["grasp"]),
            width_mm=float(data["width_mm"]),
            open_mm=float(data["open_mm"]),
            approach_mm=float(data["approach_mm"]),
            score=float(data["score"]),
            planner=str(data["planner"]),
            retreat_mm=float(data.get("retreat_mm") or 0.0),
            grasp_id=str(data.get("grasp_id") or ""),
            object_id=str(data.get("object_id") or ""),
        )


@dataclass(frozen=True)
class HeldObject:
    """What the gripper holds, and (when known) the grasp that took it."""

    obj: LocatedObject
    width_mm: float
    since: float
    grasp: GraspCandidate | None = None

    @property
    def hold_offset_mm(self) -> float | None:
        """Height of the grasp TCP above the object's bottom, when the grasp is known."""
        return None if self.grasp is None else self.grasp.grasp.z - self.obj.bottom_z

    def to_dict(self) -> dict[str, Any]:
        offset = self.hold_offset_mm
        return {
            "object": self.obj.to_dict(),
            "grasp_id": "" if self.grasp is None else self.grasp.grasp_id,
            "width_mm": round(self.width_mm, 1),
            "hold_offset_mm": None if offset is None else round(offset, 1),
            "since": self.since,
        }


@dataclass(frozen=True)
class TargetObservation:
    """The workcell's calibration target, measured in one fresh camera frame.

    ``transform`` is ``T_camera_target`` (mm). ``center`` is the target's middle
    in its own frame; ``depth_point`` is where the depth image puts the pixel the
    centre is seen at (camera frame, mm) -- the same deprojection perception uses
    -- or None when that pixel has no depth. ``intrinsics`` is the live stream's
    ``CameraIntrinsics.to_dict()``, recorded with any calibration measured from it.
    """

    transform: Transform
    center: Vec3
    corners: int
    reprojection_px: float
    intrinsics: Mapping[str, Any]
    observed_at: float
    depth_point: Vec3 | None = None

    @property
    def center_in_camera(self) -> Vec3:
        return self.transform.apply(self.center)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_in_camera": self.transform.to_pose().to_dict(),
            "corners": self.corners,
            "reprojection_px": round(self.reprojection_px, 3),
            "depth_point": None
            if self.depth_point is None
            else [round(v, 1) for v in self.depth_point],
        }


@dataclass(frozen=True)
class PolicyCard:
    """A served policy bound to this workcell: identity, motion space, channel per action.

    ``space`` is what the commanded channels move (``joint``, ``ee`` or
    ``gripper``); ``adapter`` the declared steps that turn the model's native
    vectors into those channels, pinned by ``adapter_digest``.
    """

    name: str
    model: Mapping[str, str]  # served name, version, checksum
    space: str  # "joint", "ee" or "gripper"
    channels: tuple[str, ...]
    fps: float
    tasks: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    instruction_policy: str = "open"
    adapter: tuple[str, ...] = ()
    adapter_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class PolicyReport:
    """A preview (``ticks == 0``, ``end`` predicted) or a streamed run (``end`` observed).

    ``ended`` says why a run stopped streaming: ``duration`` (the cap elapsed)
    or ``done`` (the model said the instruction is complete); ``progress`` is
    the model's last estimate in [0, 1] when it reports one.
    """

    card: PolicyCard
    ticks: int = 0
    chunks: int = 0
    horizon: int = 0
    interventions: int = 0
    elapsed_s: float = 0.0
    infer_ms: float = 0.0
    end: Pose | None = None
    violation: str = ""
    ended: str = ""
    progress: float | None = None

    def to_dict(self) -> dict[str, Any]:
        counts = ("ticks", "chunks", "horizon", "interventions", "violation", "ended")
        return {
            **{name: getattr(self, name) for name in counts},
            "policy": self.card.to_dict(),
            "elapsed_s": round(self.elapsed_s, 3),
            "infer_ms": round(self.infer_ms, 1),
            "progress": None if self.progress is None else round(self.progress, 3),
            "end": None if self.end is None else self.end.to_dict(1),
        }


__all__ = [
    "CONSENT_EACH_RUN",
    "CONSENT_SCOPES",
    "CONSENT_SESSION",
    "COORDS",
    "COORDS_PER_MILLE",
    "COORDS_PIXEL",
    "COORDS_UNIT",
    "EFFECT_ACTUATE",
    "EFFECT_READ",
    "FAILURE_ARM_NOT_READY",
    "FAILURE_CALIBRATION_MISSING",
    "FAILURE_CALIBRATION_STALE",
    "FAILURE_CHECK_FAILED",
    "FAILURE_DEVICE_BUSY",
    "FAILURE_DEVICE_FAULT",
    "FAILURE_ESTOPPED",
    "FAILURE_GUARD_FAILED",
    "FAILURE_INTERNAL_DEFECT",
    "FAILURE_INVALID_ARGS",
    "FAILURE_LEASE_LOST",
    "FAILURE_MODEL_UNAVAILABLE",
    "FAILURE_OUT_OF_WORKSPACE",
    "FAILURE_PLAN_STALE",
    "FAILURE_PLAN_UNKNOWN",
    "FAILURE_SHIELD_VIOLATION",
    "FAILURE_SKILL_UNAVAILABLE",
    "FAILURE_STOPPED",
    "FAILURE_TARGET_NOT_FOUND",
    "FAILURE_TIMEOUT",
    "FAILURE_UNKNOWN_SKILL",
    "FAILURE_UNKNOWN_WORKCELL",
    "FAILURE_UNREACHABLE",
    "FAILURE_WRITE_REFUSED",
    "REQUIRE_CALIBRATION",
    "REQUIRE_MODEL",
    "REQUIRE_ROLE",
    "REQUIRE_SHARE",
    "SHARE_FRAMES",
    "SIDE_EFFECT_COMMITTED",
    "SIDE_EFFECT_NONE",
    "SIDE_EFFECT_PARTIAL",
    "SIDE_EFFECT_STATES",
    "SIDE_EFFECT_UNKNOWN",
    "SKILL_EFFECTS",
    "SOURCE_LIFT",
    "SOURCE_LOCATOR",
    "STATUS_ESCALATED",
    "STATUS_FAILED",
    "STATUS_HALTED",
    "STATUS_SUCCEEDED",
    "ArmState",
    "GraspCandidate",
    "GripperResult",
    "HeldObject",
    "LocatedObject",
    "MotionResult",
    "PixelTarget",
    "PolicyCard",
    "PolicyReport",
    "SkillError",
    "SkillOutcome",
    "SkillPlan",
    "SkillSpec",
    "TargetObservation",
]
