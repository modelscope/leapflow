# Copyright (c) Alibaba, Inc. and its affiliates.
"""The facade protocols a node skill reaches the workcell through.

``protocol`` holds the contract's value types (specs, plans, outcomes, the
geometry a perception pass reports); this module holds the ports those values
flow through. A skill imports both, ``geometry``, and the standard library --
nothing else -- so swapping an arm, a camera, a grasp planner, or a model never
touches a skill.

Every port reports failure as a ``SkillError``. A port never chooses on the
model's behalf: perception returns every candidate it saw with its score, a
grasp planner returns every candidate with its score, and the skill (or S2
through the skill's arguments) picks by id.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

# Pure prepared-planning contracts are part of the skill-facing boundary.
from leapflow.hardware.skills.motion_geometry import MotionGeometry as MotionGeometry
from leapflow.hardware.skills.planning import (
    CalculationPort as CalculationPort,
    Deadline as Deadline,
    Evaluation as Evaluation,
    FrozenJoints as FrozenJoints,
    PlanningCheckpoint as PlanningCheckpoint,
    PlanningSettings as PlanningSettings,
    PreparedPlanning as PreparedPlanning,
    encoded as encoded,
)

from leapflow.hardware.skills.frame_status import Accuracy, FrameDiagnosis
from leapflow.hardware.skills.geometry import Plane, Pose, Transform, WorkspaceBox
from leapflow.hardware.skills.protocol import (
    ArmState,
    CloudObservation,
    GraspCandidate,
    GripperResult,
    HeldObject,
    LocatedObject,
    MotionResult,
    PixelTarget,
    PolicyReport,
    SkillOutcome,
    SkillPlan,
    SkillSpec,
    TargetObservation,
)

# Pose path modes a ``move_tcp`` may request. ``""`` leaves the arm's default.
PATH_DEFAULT = ""
PATH_LINEAR = "linear"
PATH_POINT = "point"


@runtime_checkable
class WorkcellView(Protocol):
    """The declared workcell a skill runs in, with every tuning it declares."""

    @property
    def workcell_id(self) -> str: ...

    @property
    def roles(self) -> Mapping[str, str]:
        """Every bound role and the device id it names."""
        ...

    def role(self, name: str) -> str:
        """Device id bound to a role (``arm``, ``gripper``, ``camera.wrist``...), or ``""``."""
        ...

    @property
    def gripper_spec(self) -> Mapping[str, float]:
        """``max_width_mm`` and ``min_hold_width_mm``, resolved at admission."""
        ...

    @property
    def grasp_verify(self) -> Mapping[str, float]: ...

    @property
    def home(self) -> Mapping[str, float]: ...

    @property
    def tool_cone_deg(self) -> float: ...

    @property
    def freshness(self) -> Mapping[str, float]:
        """How far the world may drift between plan and execute (mm, deg)."""
        ...

    @property
    def observation(self) -> Mapping[str, Any]:
        """Declared observation policy, camera ranges and optional yielding posture."""
        ...

    @property
    def calibration(self) -> Mapping[str, float]:
        """Acceptance thresholds of the calibration skills."""
        ...


@runtime_checkable
class FrameView(Protocol):
    """Calibrated frames and the workspace, resolved from versioned calibrations."""

    def transform(self, name: str) -> Transform | None:
        """``T_parent_frame`` for a declared frame, or None when uncalibrated."""
        ...

    def parent(self, name: str) -> str:
        """The frame *name* is expressed in: ``flange`` or ``base`` (``""`` if unknown)."""
        ...

    def procedure(self, name: str) -> str:
        """The calibration procedure that measures a frame or ``workspace`` (``""`` if none)."""
        ...

    @property
    def workspace(self) -> WorkspaceBox | None: ...

    @property
    def table(self) -> Plane | None:
        """The support surface measured with the workspace, or None when not measured."""
        ...

    @property
    def versions(self) -> Mapping[str, float]: ...

    def parameters(self, name: str) -> Mapping[str, Any]:
        """The parameters recorded with a resolved frame; empty when unresolved."""
        ...

    def diagnosis(self, name: str) -> FrameDiagnosis:
        """Why *name* does or does not hold, with its achieved and required accuracy."""
        ...

    def required_accuracy(self, name: str) -> Accuracy | None:
        """The accuracy the workcell requires of a frame; None when it sets none."""
        ...


@runtime_checkable
class ReachCheck(Protocol):
    """Follows a program's pose targets with the arm's kinematics at plan time.

    ``move`` raises ``SkillError`` (``unreachable``) naming how far along the
    move the arm gets. An arm whose kinematics cannot predict reach yields a
    check that is not ``verified`` and passes every move.
    """

    @property
    def verified(self) -> bool: ...

    def to_dict(self) -> dict[str, Any]: ...

    def move(self, target: Pose, *, linear: bool, label: str) -> None: ...

    def joints_moved(self, joints: Mapping[str, float], tcp: Pose) -> None: ...


@runtime_checkable
class MotionFacade(Protocol):
    """Governed discrete motion. Every call goes through the node write gate."""

    def geometry_snapshot(self) -> MotionGeometry:
        """Detached frozen motion values suitable for pure planning jobs."""
        ...

    async def state(self) -> ArmState: ...

    def check_target(self, pose: Pose) -> None:
        """Raise ``SkillError`` (NONE) when *pose* is outside the allowed region."""
        ...

    def check_kinematics(self, state: ArmState) -> None:
        """Refuse when declared FK disagrees with the arm's reported flange."""
        ...

    def check_joints(
        self, targets: Mapping[str, float], current: Mapping[str, float] | None
    ) -> Pose:
        """Check a joint endpoint and path with FK; return its predicted TCP."""
        ...

    def check_cartesian_path(
        self, state: ArmState, targets: Sequence[Pose]
    ) -> tuple[Mapping[str, Any], ...]:
        """Bounded IK waypoints whose actual joint-space paths pass workspace checks."""
        ...

    def reach_envelope(self) -> Mapping[str, Any]:
        """Cached posture-specific guidance; never authority to execute a move."""
        ...

    def reach_tracker(self, state: ArmState) -> ReachCheck:
        """Plan-time IK that follows pose targets from *state* inside the joint limits."""
        ...

    @property
    def paths(self) -> tuple[str, ...]:
        """The pose path modes this arm declares (``linear``, ``point``); may be empty."""
        ...

    def reached(self, pose: Pose, state: ArmState) -> bool:
        """Whether *state*'s TCP is at *pose* within the arm's declared tolerance."""
        ...

    async def move_tcp(
        self, pose: Pose, *, speed_pct: float | None = None, path: str = PATH_DEFAULT
    ) -> MotionResult: ...

    async def move_joints(
        self, targets: Mapping[str, float], *, speed_pct: float | None = None,
        validate: Callable[[ArmState], None] | None = None,
    ) -> MotionResult:
        """Revalidate an optional pure path guard against the final pre-write state."""
        ...

    async def gripper(self, width_mm: float, *, effort: float | None = None) -> GripperResult:
        """Drive the gripper and wait until its width is stable (the declared settling)."""
        ...

    @property
    def gripper_tolerance_mm(self) -> float:
        """The gripper's declared position tolerance: widths closer than this are one width."""
        ...

    async def pause(self, seconds: float) -> None:
        """Hold still for *seconds*; raises at once when the run is stopped or out of time."""
        ...


@runtime_checkable
class ScenePort(Protocol):
    """Node-local perception of the workcell: what lies on the table, and what is held.

    Objects and grasps are handles: ``locate`` names every object it found
    ``oN`` and ``remember_grasps`` names candidates ``gN``; later calls refer to
    them only by those ids. A frame shared with S2 is ``fN`` (a mosaic of several
    frames ``vN``), and what S2 points at in it becomes an object through ``lift``.
    Nothing here matches labels or breaks ties.
    """

    async def locate(
        self, queries: Sequence[str], *, min_score: float | None = None, camera: str = "camera"
    ) -> tuple[LocatedObject, ...]:
        """Every detection of *queries* in a fresh frame of *camera*, each under a new id."""
        ...

    async def snapshot(
        self,
        camera: str = "camera",
        *,
        with_cameras: Sequence[str] = (),
        planes: Sequence[str] = ("color",),
        detail: str = "auto",
        overlay: bool = True,
    ) -> Mapping[str, Any]:
        """Fresh frames of *camera* and *with_cameras*, each showing *planes*, shared as
        one ``image`` named ``frame`` (``fN`` for one tile, ``vN`` for a mosaic);
        ``liftable`` says whether ``lift`` can place what is pointed at in it, and when
        it cannot, ``unliftable`` says why and which camera could instead. Also reports
        frame statistics, changes since each camera's previous snapshot, how each
        remembered object appears in every view, and a ``situation`` summary."""
        ...

    async def zoom(
        self, frame_id: str, box: Sequence[float], *, coords: str, planes: Sequence[str] = ()
    ) -> Mapping[str, Any]:
        """Window *box* of a remembered frame (or mosaic tile) re-shown at native
        resolution under a new ``fN`` that keeps the original's age."""
        ...

    async def lift(
        self,
        frame_id: str,
        targets: Sequence[PixelTarget],
        *,
        coords: str,
        capture_id: str | None = None,
    ) -> tuple[tuple[LocatedObject, ...], tuple[Mapping[str, str], ...]]:
        """The base-frame objects under *targets* in a remembered frame, each under a new
        id, and the targets that lifted to nothing (``label``, ``reason``)."""
        ...

    def observation(self, obj: LocatedObject, grasp: GraspCandidate | None = None) -> Mapping[str, Any]:
        """Evidence for the exact geometry; optionally evaluated against a grasp's slack."""
        ...

    def object(self, object_id: str) -> LocatedObject:
        """A remembered object; raises ``target_not_found`` when unknown or aged out."""
        ...

    async def relocate(self, obj: LocatedObject, *, commit: bool = True) -> LocatedObject:
        """Measure *obj* again only where its approved geometry projects.

        This never grounds by label or selects another candidate; it raises when
        the approved candidate can no longer be measured in a fresh frame.
        With commit=False no object, grasp, or observation memory is changed.
        """
        ...

    def remember_grasps(
        self, obj: LocatedObject, candidates: Sequence[GraspCandidate], *,
        validate: Callable[[tuple[GraspCandidate, ...]], None] | None = None,
    ) -> tuple[GraspCandidate, ...]:
        """Name candidates atomically after synchronous validation of the exact named values."""
        ...

    def grasp(self, grasp_id: str) -> GraspCandidate:
        """A remembered grasp; raises ``target_not_found`` when unknown or aged out."""
        ...

    def grasps(self, object_id: str) -> tuple[GraspCandidate, ...]:
        """Live task-equivalent proposals for one object, in publication order."""
        ...

    @property
    def held(self) -> HeldObject | None: ...

    def hold(self, held: HeldObject) -> None: ...

    def release(self) -> HeldObject | None: ...


@runtime_checkable
class SurveyPort(Protocol):
    """Finite owner-side observation under the ordinary skill lifecycle."""

    async def plan(self, args: Mapping[str, Any]) -> SkillPlan: ...

    async def execute(self, plan: SkillPlan) -> SkillOutcome: ...


@runtime_checkable
class PairedObservationPort(Protocol):
    """Read-only paired acquisition with lossless owner-local evidence."""
    async def capture(self) -> Any: ...


@runtime_checkable
class BridgeCalibrationPort(Protocol):
    """Approved stationary acquisition and conditional shadow evaluation; no motion."""
    async def basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]: ...
    async def execute(self, plan: SkillPlan) -> Mapping[str, Any]: ...


@runtime_checkable
class FrameLifecyclePort(Protocol):
    """Install a measured candidate, or vouch for a frame by an independent check.

    Neither moves an actuator: promotion persists an estimate some calibration
    already measured, and attestation compares two cameras' simultaneous view
    of one stationary target. Each ``*_basis`` is read-only and computes what the
    approved run will do; the run re-checks that basis before it writes.
    """

    async def promotion_basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def promote(self, plan: SkillPlan) -> Mapping[str, Any]: ...

    async def attestation_basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def attest(self, plan: SkillPlan) -> Mapping[str, Any]: ...


@runtime_checkable
class VisionPort(Protocol):
    """Node-local camera measurements a calibration may take.

    Two of them: the declared target's pose, and the raw measured surface. Both
    stay on the node. Not seeing the target is ``target_not_found`` (NONE); a
    missing target declaration or vision library is ``skill_unavailable``.
    """

    async def observe_target(self) -> TargetObservation: ...

    async def observe_cloud(
        self, *, near_mm: float | None = None, far_mm: float | None = None
    ) -> CloudObservation:
        """A fresh frame's valid depth points, in the camera frame.

        The depth window is applied before the cloud is thinned, so a caller that
        names the region it cares about gets its whole point budget there.
        """
        ...


@runtime_checkable
class MetricReadinessPort(Protocol):
    """Target-free bounded capture; raw samples stay in the owning node's cache."""

    async def metric_readiness(
        self,
        *,
        frames: int,
        points: int,
        role: str,
        stamp_uncertainty_s: float,
        ttl_s: float,
        quota_bytes: int,
    ) -> Mapping[str, Any]: ...


@runtime_checkable
class GraspPlanner(Protocol):
    """Turns a located object into gripper poses; one per workcell, chosen by config."""

    @property
    def name(self) -> str: ...

    async def candidates(
        self, obj: LocatedObject, *, table: Plane, current: Pose
    ) -> tuple[GraspCandidate, ...]:
        """Grasps for *obj* with scores; empty when this gripper cannot take it."""
        ...


@runtime_checkable
class PolicyPort(Protocol):
    """Node-local streaming of a served policy (``model:<name>``) under the safety shield.

    Every streamed action passes envelope, slew and workspace checks and the node
    write gate. Unavailable or unbound policies raise ``skill_unavailable``.
    """

    @property
    def startup_s(self) -> float:
        """Run-deadline room a run needs before its first action (bind, observe, infer)."""
        ...

    async def preview(self, name: str, task: str) -> PolicyReport:
        """One observation and inference, filtered but never commanded."""
        ...

    async def run(
        self,
        name: str,
        task: str,
        *,
        approved: Mapping[str, Any],
        duration_s: float,
        speed_pct: float | None = None,
    ) -> PolicyReport:
        """Stream the policy for at most *duration_s*; failures raise ``SkillError``.

        *approved* is the plan basis (``model``, ``adapter_digest``). The binding
        that streams is the one checked against it: a different model or
        embodiment is ``plan_stale`` before anything moves.
        """
        ...


@runtime_checkable
class GraspVerificationPort(Protocol):
    """A run's continuous qualified observer; read-only and no transport ownership."""

    def admit(self, data: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def qualification(self) -> Mapping[str, Any]: ...

    async def inspect(self) -> Mapping[str, Any]: ...

    async def verify(
        self, stage: str, *, samples: int, grasp: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]: ...


@runtime_checkable
class CalibrationPort(Protocol):
    """Private admitted acquisition and redacted diagnostics through the hardware owner."""

    async def collection_basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def collect(self, plan: SkillPlan) -> Mapping[str, Any]: ...

    async def inspect(self, *, evidence_digest: str = "") -> Mapping[str, Any]: ...


@runtime_checkable
class CalibrationReferencePort(Protocol):
    """Stationary task-local registration, separately optional from collection."""

    async def reference_basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def align_reference(self, plan: SkillPlan) -> Mapping[str, Any]: ...


@runtime_checkable
class CalibrationTrialPort(Protocol):
    """Indexed camera-only candidate assessment; no activation or actuator access."""

    async def trial_basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def assess_motion(self, plan: SkillPlan) -> Mapping[str, Any]: ...


@runtime_checkable
class CalibrationPosePort(Protocol):
    """Bounded nominal-only acquisition through the ordinary governed motion facade."""

    async def poses_basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def collect_poses(self, plan: SkillPlan) -> Mapping[str, Any]: ...


@runtime_checkable
class CalibrationRollbackPort(Protocol):
    """Optional governed between-run rollback; persistence is effectful, never motion."""

    async def rollback_basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def rollback(self, plan: SkillPlan) -> Mapping[str, Any]: ...


@runtime_checkable
class SkillContext(Protocol):
    """Everything a skill may touch. Nothing else is reachable from a skill."""

    @property
    def workcell(self) -> WorkcellView: ...

    @property
    def frames(self) -> FrameView: ...

    @property
    def motion(self) -> MotionFacade: ...

    @property
    def calculations(self) -> CalculationPort: ...

    @property
    def planning_settings(self) -> PlanningSettings: ...

    @property
    def planning_deadline(self) -> float:
        """End-to-end owner deadline, including admission and preparation."""
        ...

    async def planning_basis(self) -> Mapping[str, Any]:
        """Current owner identity, physical generation, and calibration; refuses shutdown."""
        ...

    @property
    def scene(self) -> ScenePort:
        """Perception; raises ``SkillError`` when the workcell has no camera or locator."""
        ...

    @property
    def survey(self) -> SurveyPort: ...

    @property
    def grasp_planner(self) -> GraspPlanner:
        """The workcell's grasp planner; raises ``SkillError`` when it is not installed."""
        ...

    @property
    def vision(self) -> VisionPort:
        """Calibration-target measurement; raises ``SkillError`` when there is no camera."""
        ...

    def camera_vision(self, role: str = "camera") -> VisionPort:
        """Target observation through the named camera role and its mount."""
        ...

    @property
    def grasp_verification(self) -> GraspVerificationPort: ...

    @property
    def calibration(self) -> CalibrationPort: ...

    @property
    def bridge(self) -> BridgeCalibrationPort: ...

    @property
    def frame_lifecycle(self) -> FrameLifecyclePort: ...

    @property
    def policies(self) -> PolicyPort: ...

    @property
    def limits(self) -> Mapping[str, float]: ...

    @property
    def cancelled(self) -> bool: ...

    @property
    def recording(self) -> bool:
        """Whether runs here are recorded as episodes (a run's own recording, once started)."""
        ...

    async def read(self, role: str, channel_ids: Sequence[str]) -> Mapping[str, Any]: ...

    def record_calibration(
        self,
        procedure_id: str,
        *,
        role: str = "",
        parameters: Mapping[str, Any] | None = None,
        matrix: Sequence[Sequence[float]] | None = None,
        pose: Mapping[str, Any] | None = None,
        notes: str = "",
    ) -> float:
        """Persist a calibration version; returns its ``recorded_at``."""
        ...

    @property
    def evidence(self) -> Mapping[str, Any]:
        """Current owner-loop evidence, used to bound complete planning responses."""
        ...

    def note(self, key: str, value: Any) -> None:
        """Attach evidence to the run (timings, verdicts, measured residuals)."""
        ...


@runtime_checkable
class NodeSkill(Protocol):
    """A node-local skill. See ``protocol`` for the two-phase contract."""

    @property
    def spec(self) -> SkillSpec: ...

    async def plan(
        self, ctx: SkillContext, args: Mapping[str, Any]
    ) -> SkillPlan | SkillOutcome: ...

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome: ...


@runtime_checkable
class ProvidingSkill(Protocol):
    """A skill whose provisions depend on the workcell it is listed in.

    ``SkillSpec.provides`` is enough when a provision follows from the skill's
    own arguments; a skill whose output is fixed by a workcell declaration (which
    camera a bridge estimates, which frames a fixed-camera method can measure)
    answers per workcell instead. Its answer replaces ``provides`` for that
    workcell; a concrete token that instantiates one of the ``provides``
    templates (``calib:camera`` for ``calib:{camera}``) carries the argument
    values that template implies, so a skill can narrow its own templates.
    """

    def provides_in(self, workcell: WorkcellView, frames: FrameView) -> tuple[str, ...]: ...


__all__ = [
    "PATH_DEFAULT",
    "PATH_LINEAR",
    "PATH_POINT",
    "CalibrationPort",
    "CalibrationReferencePort",
    "CalibrationRollbackPort",
    "FrameLifecyclePort",
    "FrameView",
    "ProvidingSkill",
    "GraspPlanner",
    "GraspVerificationPort",
    "MotionFacade",
    "MetricReadinessPort",
    "NodeSkill",
    "PolicyPort",
    "ReachCheck",
    "ScenePort",
    "SurveyPort",
    "SkillContext",
    "VisionPort",
    "WorkcellView",
]
