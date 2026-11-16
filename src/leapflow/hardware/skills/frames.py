# Copyright (c) Alibaba, Inc. and its affiliates.
"""Calibrated workcell frames and requirement checks, resolved from versioned records.

``FrameGraph`` snapshots every frame a workcell declares (and its workspace
box and table plane) from the ``CalibrationStore``, plus which non-frame
records the skills ask for are installed; ``unmet_requirements`` explains why
a skill's ``requires`` tokens do not hold in one snapshot.

Every declared frame gets a typed ``FrameDiagnosis`` (``frame_status.py``):
missing, measured against a superseded dependency, carrying evidence from an
older rule set, unproven, less accurate than the workcell requires, contradicted
by an independent check, or valid. Only valid frames resolve; the diagnosis is
what lets a refusal name the cheapest remedy instead of always "calibrate again".

A frame's record proves itself through the provenance validators
(``provenance.py``). A record that cannot, or whose accuracy is unknown, can
still be vouched for by an independent check of that exact version: a passing
``<frame>.verify`` record bound to it. A failed ``<frame>.attestation`` newer
than any passing check marks the frame as drifted.

A token may name one of the skill's arguments in braces (``calib:{camera}``):
``bind_requirements`` resolves it against the arguments a caller passed, and
``requirement_bindings`` reports, for a listing, which values of it the
workcell can meet -- so one skill serves whichever camera is ready.
"""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from leapflow.hardware.skills.frame_status import (
    Accuracy,
    FrameCandidate,
    FrameDiagnosis,
    FrameStatus,
)
from leapflow.hardware.skills.geometry import Plane, Pose, Transform, WorkspaceBox
from leapflow.hardware.skills.protocol import (
    REQUIRE_CALIBRATION,
    REQUIRE_CANDIDATE,
    REQUIRE_MODEL,
    REQUIRE_RECORD,
    REQUIRE_ROLE,
    REQUIRE_SHARE,
    SHARE_FRAMES,
)
from leapflow.hardware.skills.provenance import ACHIEVED_ACCURACY, judge
from leapflow.hardware.skills.workcell import DEPENDS_ON, WORKSPACE_FRAME, WorkcellSpec

logger = logging.getLogger(__name__)

_VERSION_TOLERANCE = 1e-6
VERIFY_SUFFIX = ".verify"  # an independent, held-out check of one frame version
ATTESTATION_SUFFIX = ".attestation"  # a check of one frame version that did not pass

Hint = Callable[[str], str]


def verify_procedure(frame: str) -> str:
    """The record procedure that vouches for one version of *frame*."""
    return f"{frame}{VERIFY_SUFFIX}"


def attestation_procedure(frame: str) -> str:
    """The record procedure a failed independent check of *frame* leaves behind."""
    return f"{frame}{ATTESTATION_SUFFIX}"


# ── Frames ───────────────────────────────────────────────────────────────────


class FrameGraph:
    """Calibrated frames of one workcell, snapshotted at construction. Satisfies ``FrameView``.

    A snapshot, not a live view: a plan and its execution must agree on the
    geometry, so the runtime takes one snapshot per plan and compares versions
    at execute time instead of letting frames shift under a running skill.
    """

    def __init__(
        self,
        spec: WorkcellSpec,
        transforms: Mapping[str, Transform],
        workspace: WorkspaceBox | None,
        versions: Mapping[str, float],
        stale: Mapping[str, str] | None = None,
        table: Plane | None = None,
        parameters: Mapping[str, Mapping[str, Any]] | None = None,
        records: Mapping[str, float] | None = None,
        diagnoses: Mapping[str, FrameDiagnosis] | None = None,
        record_problems: Mapping[str, str] | None = None,
    ) -> None:
        self._spec = spec
        self._transforms = dict(transforms)
        self._workspace = workspace
        self._versions = dict(versions)
        self._table = table
        self._parameters = {name: dict(value) for name, value in (parameters or {}).items()}
        self._records = dict(records or {})
        self._record_problems = dict(record_problems or {})
        self._diagnoses = dict(diagnoses or {})
        for name, reason in (stale or {}).items():
            self._diagnoses.setdefault(
                name, FrameDiagnosis(name, FrameStatus.DEPENDENCY_SUPERSEDED, reason)
            )

    @classmethod
    def snapshot(
        cls,
        spec: WorkcellSpec,
        store: Any | None,
        procedures: Iterable[str] = (),
        candidates: Mapping[str, Sequence[FrameCandidate]] | None = None,
    ) -> "FrameGraph":
        """Resolve and diagnose every declared frame and the workspace from *store*.

        *procedures* names non-frame records (``record:`` requirements) whose
        latest version is noted in ``records``. They never join ``versions``:
        those are the geometry a plan and its evidence bind to. *candidates*
        are measured-but-uninstalled estimates by frame, reported with each
        frame's diagnosis so a refusal can offer to promote one.
        """
        records: dict[str, Any] = {}
        if store is not None:
            for name in (*spec.frames, WORKSPACE_FRAME):
                procedure = _procedure(spec, name)
                record = store.latest(spec.workcell_id, procedure) if procedure else None
                if record is not None:
                    records[name] = record
        superseded = _stale_records(records)
        resolved = _Resolution(spec, store, candidates or {})
        for name in (*spec.frames, WORKSPACE_FRAME):
            if name == WORKSPACE_FRAME and not spec.workspace_procedure:
                continue
            resolved.add(name, records.get(name), superseded.get(name, ""))
        resolved.drop_orphaned_candidates()
        installed, problems = _installed_records(spec, store, procedures, resolved.versions)
        return cls(
            spec,
            resolved.transforms,
            resolved.workspace,
            resolved.versions,
            None,
            resolved.table,
            resolved.parameters,
            installed,
            resolved.diagnoses,
            problems,
        )

    def transform(self, name: str) -> Transform | None:
        return self._transforms.get(name)

    def parent(self, name: str) -> str:
        decl = self._spec.frames.get(name)
        return decl.parent if decl is not None else ""

    def procedure(self, name: str) -> str:
        return _procedure(self._spec, name)

    @property
    def workspace(self) -> WorkspaceBox | None:
        return self._workspace

    @property
    def table(self) -> Plane | None:
        return self._table

    @property
    def versions(self) -> Mapping[str, float]:
        return dict(self._versions)

    @property
    def stale(self) -> Mapping[str, str]:
        """Frames that have a record which does not hold, with the reason why."""
        return {
            name: diagnosis.reason
            for name, diagnosis in self._diagnoses.items()
            if diagnosis.status not in (FrameStatus.VALID, FrameStatus.MISSING)
        }

    def diagnosis(self, name: str) -> FrameDiagnosis:
        """Why *name* does or does not hold in this snapshot."""
        found = self._diagnoses.get(name)
        if found is not None:
            return found
        if name in self._versions:
            return FrameDiagnosis(name, FrameStatus.VALID, version=self._versions[name])
        return FrameDiagnosis(name, FrameStatus.MISSING, required=self.required_accuracy(name))

    @property
    def diagnoses(self) -> Mapping[str, FrameDiagnosis]:
        """Every declared frame's diagnosis (and the workspace's, when declared)."""
        names = [*self._spec.frames]
        if self._spec.workspace_procedure:
            names.append(WORKSPACE_FRAME)
        return {name: self.diagnosis(name) for name in dict.fromkeys([*names, *self._diagnoses])}

    def required_accuracy(self, name: str) -> Accuracy | None:
        """The accuracy the workcell requires of *name*; None for the workspace."""
        return _required_accuracy(self._spec, name)

    def candidates(self, name: str) -> tuple[FrameCandidate, ...]:
        """Uninstalled estimates of *name* this snapshot was given, newest first."""
        return self.diagnosis(name).candidates

    def parameters(self, name: str) -> Mapping[str, Any]:
        """The parameters recorded with a resolved frame (e.g. the intrinsics a
        hand-eye calibration was measured with); empty when unresolved."""
        return dict(self._parameters.get(name) or {})

    @property
    def records(self) -> Mapping[str, float]:
        """Installed non-frame records by procedure, as their latest version."""
        return dict(self._records)

    @property
    def record_problems(self) -> Mapping[str, str]:
        """Installed non-frame records measured against a frame version that no longer holds."""
        return dict(self._record_problems)


class _Resolution:
    """Accumulates one snapshot's resolved geometry and diagnoses, frame by frame."""

    def __init__(
        self,
        spec: WorkcellSpec,
        store: Any | None,
        candidates: Mapping[str, Sequence[FrameCandidate]],
    ) -> None:
        self.spec, self.store, self.candidates = spec, store, candidates
        self.transforms: dict[str, Transform] = {}
        self.versions: dict[str, float] = {}
        self.parameters: dict[str, Mapping[str, Any]] = {}
        self.diagnoses: dict[str, FrameDiagnosis] = {}
        self.workspace: WorkspaceBox | None = None
        self.table: Plane | None = None

    def add(self, name: str, record: Any | None, superseded: str) -> None:
        offered = tuple(self.candidates.get(name, ()))
        required = _required_accuracy(self.spec, name)
        if record is None:
            self._note(name, FrameStatus.MISSING, required=required, candidates=offered)
            return
        version = float(record.recorded_at)
        promotion = (record.parameters or {}).get("promotion")
        if isinstance(promotion, Mapping):  # an installed candidate is no longer on offer
            offered = tuple(c for c in offered if c.digest != promotion.get("candidate_digest"))
        if superseded:
            self._note(name, FrameStatus.DEPENDENCY_SUPERSEDED, superseded, version, required, offered)
            return
        if name == WORKSPACE_FRAME:
            self._add_workspace(record, version)
            return
        transform = _record_transform(record)
        if transform is None:
            self._note(name, FrameStatus.UNPROVEN, "the record is malformed", version, required, offered)
            return
        verdict = judge(record, mounted=bool(self.spec.role(name)))
        check, drift = self._independent_check(name, version)
        achieved = _check_accuracy(check) if check is not None else verdict.achieved
        status, reason = verdict.status, verdict.reason
        if drift is not None:
            status = FrameStatus.DRIFT_DETECTED
            reason = str((drift.parameters or {}).get("reason") or "an independent check disagreed")
        elif status is not FrameStatus.VALID and check is not None:
            status, reason = FrameStatus.VALID, ""  # an independent check vouches for this version
        if status is FrameStatus.VALID and achieved and required and not achieved.within(required):
            status = FrameStatus.INSUFFICIENT_ACCURACY
            reason = f"achieved {achieved.describe()}, required {required.describe()}"
        diagnosis = FrameDiagnosis(
            name,
            status,
            reason,
            version,
            verdict.method,
            achieved,
            required,
            offered,
            check.procedure_id if check is not None else "",
        )
        self.diagnoses[name] = diagnosis
        if diagnosis.ok:
            self.transforms[name] = transform
            self.versions[name] = version
            self.parameters[name] = dict(record.parameters or {})

    def _add_workspace(self, record: Any, version: float) -> None:
        workspace = _record_box(record)
        if workspace is None:
            self._note(WORKSPACE_FRAME, FrameStatus.UNPROVEN, "the record is malformed", version)
            return
        self.workspace, self.table = workspace, _record_table(record)
        self.versions[WORKSPACE_FRAME] = version
        self.parameters[WORKSPACE_FRAME] = dict(record.parameters or {})
        self._note(WORKSPACE_FRAME, FrameStatus.VALID, version=version)

    def _independent_check(self, name: str, version: float) -> tuple[Any | None, Any | None]:
        """The newest verdict on this exact version: a passing check, or a failed one."""
        if self.store is None:
            return None, None
        passed = _bound(self.store.latest(self.spec.workcell_id, verify_procedure(name)), name, version)
        if passed is not None and (
            (passed.parameters or {}).get("training_disjoint") is not True
            or (passed.parameters or {}).get("passed") is False
        ):
            passed = None
        failed = _bound(
            self.store.latest(self.spec.workcell_id, attestation_procedure(name)), name, version
        )
        if failed is not None and (failed.parameters or {}).get("passed") is not False:
            failed = None
        if failed is not None and passed is not None:
            if float(passed.recorded_at) >= float(failed.recorded_at):
                failed = None
            else:
                passed = None
        return passed, failed

    def drop_orphaned_candidates(self) -> None:
        """Forget candidates solved through a frame version that no longer holds.

        Run once every frame is resolved, since a candidate may depend on a frame
        declared after the one it estimates.
        """
        for name, diagnosis in list(self.diagnoses.items()):
            kept = tuple(c for c in diagnosis.candidates if self._holds(c.dependency))
            if kept != diagnosis.candidates:
                self.diagnoses[name] = replace(diagnosis, candidates=kept)

    def _holds(self, dependency: Mapping[str, float]) -> bool:
        for frame, version in dependency.items():
            current = self.versions.get(str(frame))
            try:
                if current is None or abs(float(current) - float(version)) > _VERSION_TOLERANCE:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def _note(
        self,
        name: str,
        status: FrameStatus,
        reason: str = "",
        version: float | None = None,
        required: Accuracy | None = None,
        candidates: tuple[FrameCandidate, ...] = (),
    ) -> None:
        self.diagnoses[name] = FrameDiagnosis(
            name, status, reason, version, required=required, candidates=candidates
        )


def _bound(record: Any | None, name: str, version: float) -> Any | None:
    """*record* when its ``depends_on`` names exactly this version of *name*."""
    if record is None:
        return None
    depends = (record.parameters or {}).get(DEPENDS_ON)
    if not isinstance(depends, Mapping) or name not in depends:
        return None
    try:
        return record if abs(float(depends[name]) - version) <= _VERSION_TOLERANCE else None
    except (TypeError, ValueError):
        return None


def _check_accuracy(record: Any) -> Accuracy | None:
    """The accuracy an independent check established: its stated bound, else its p95."""
    parameters = record.parameters or {}
    stated = Accuracy.from_mapping(parameters.get(ACHIEVED_ACCURACY))
    if stated is not None:
        return stated
    return Accuracy.from_mapping({"translation_mm": parameters.get("p95_mm")})


def _required_accuracy(spec: WorkcellSpec, name: str) -> Accuracy | None:
    if name == WORKSPACE_FRAME:
        return None
    decl = spec.frames.get(name)
    declared = Accuracy.from_mapping(decl.accuracy) if decl is not None else None
    if declared is not None:
        return declared
    return Accuracy.from_mapping({"translation_mm": spec.calibration.get("max_verify_p95_mm")})


def _installed_records(
    spec: WorkcellSpec, store: Any | None, procedures: Iterable[str], versions: Mapping[str, float]
) -> tuple[dict[str, float], dict[str, str]]:
    """Latest version of each named record, and why any no longer matches the frames."""
    installed: dict[str, float] = {}
    problems: dict[str, str] = {}
    if store is None:
        return installed, problems
    for procedure in procedures:
        record = store.latest(spec.workcell_id, procedure)
        if record is None:
            continue
        installed[procedure] = float(record.recorded_at)
        depends = (record.parameters or {}).get(DEPENDS_ON)
        if not isinstance(depends, Mapping):
            continue
        for frame, version in depends.items():
            current = versions.get(str(frame))
            if current is None:
                problems[procedure] = f"measured with a {frame} calibration that no longer holds"
                break
            try:
                moved = abs(float(current) - float(version)) > _VERSION_TOLERANCE
            except (TypeError, ValueError):
                moved = True
            if moved:
                problems[procedure] = f"measured before {frame} was recalibrated"
                break
    return installed, problems


def _procedure(spec: WorkcellSpec, name: str) -> str:
    if name == WORKSPACE_FRAME:
        return spec.workspace_procedure
    decl = spec.frames.get(name)
    return decl.procedure if decl is not None else ""


def _stale_records(records: Mapping[str, Any]) -> dict[str, str]:
    """Records whose ``depends_on`` versions are no longer current, transitively."""
    stale: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for name, record in records.items():
            if name in stale:
                continue
            reason = _dependency_problem(record, records, stale)
            if reason:
                stale[name] = reason
                changed = True
    return stale


def _dependency_problem(record: Any, records: Mapping[str, Any], stale: Mapping[str, str]) -> str:
    depends = (record.parameters or {}).get(DEPENDS_ON)
    if not isinstance(depends, Mapping):
        return ""
    for frame, version in depends.items():
        current = records.get(str(frame))
        if current is None or str(frame) in stale:
            return f"measured with a {frame} calibration that no longer holds"
        try:
            superseded = abs(float(current.recorded_at) - float(version)) > _VERSION_TOLERANCE
        except (TypeError, ValueError):
            superseded = True
        if superseded:
            return f"measured before {frame} was recalibrated"
    return ""


def _record_transform(record: Any) -> Transform | None:
    if record is None:
        return None
    try:
        if record.matrix:
            return Transform.from_matrix(record.matrix)
        if record.pose:
            return Transform.from_pose(Pose.from_mapping(record.pose))
    except (TypeError, ValueError, IndexError) as exc:
        logger.warning(
            "calibration %s/%s is malformed and ignored: %s",
            record.device_id,
            record.procedure_id,
            exc,
        )
    return None


def _record_box(record: Any) -> WorkspaceBox | None:
    if record is None:
        return None
    try:
        return WorkspaceBox.from_mapping(record.parameters or {})
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "workspace calibration %s/%s is malformed and ignored: %s",
            record.device_id,
            record.procedure_id,
            exc,
        )
        return None


def _record_table(record: Any) -> Plane | None:
    """The table plane ``calib.workspace`` measured, when its floor came from the table."""
    floor = (record.parameters or {}).get("floor")
    if not isinstance(floor, Mapping):
        return None
    try:
        if floor.get("method") == "table_z":
            return Plane.horizontal(float(floor["table_z"]))
        if floor.get("method") == "table_points":
            normal = tuple(float(v) for v in floor["normal"])
            if len(normal) != 3:
                raise ValueError("a table normal needs three components")
            return Plane(normal, float(floor["offset_mm"]))  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "table plane in %s/%s is malformed and ignored: %s",
            record.device_id,
            record.procedure_id,
            exc,
        )
    return None


# ── Requirements ─────────────────────────────────────────────────────────────


def unmet_requirements(
    requires: Iterable[str],
    spec: WorkcellSpec,
    frames: FrameGraph,
    hint: Hint | None = None,
) -> list[str]:
    """Human-readable reasons *requires* is not met in this workcell; empty when met.

    *hint* maps an unmet token to its remedy ("run calib.promote frame=camera");
    without one, a frame's remedy names the procedure that records it.
    """
    reasons: list[str] = []
    versions = frames.versions
    for token in requires:
        if token.startswith(REQUIRE_ROLE):
            role = token[len(REQUIRE_ROLE) :]
            if not spec.role(role):
                reasons.append(f"workcell {spec.workcell_id!r} binds no {role}")
        elif token.startswith(REQUIRE_CALIBRATION):
            frame = token[len(REQUIRE_CALIBRATION) :]
            if frame not in versions:
                remedy = hint(token) if hint else ""
                reasons.append(frame_refusal(frames.diagnosis(frame), remedy or _default_remedy(spec, frame)))
        elif token.startswith(REQUIRE_CANDIDATE):
            frame = token[len(REQUIRE_CANDIDATE) :]
            if not frames.candidates(frame):
                remedy = hint(token) if hint else ""
                reasons.append(
                    f"workcell {spec.workcell_id!r} holds no admissible {frame} candidate"
                    + (f" ({remedy})" if remedy else "")
                )
        elif token.startswith(REQUIRE_MODEL):
            name = token[len(REQUIRE_MODEL) :]
            if not spec.model(name):
                reasons.append(
                    f"workcell {spec.workcell_id!r} binds no {name!r} model (declare it under "
                    "'models')"
                )
        elif token.startswith(REQUIRE_SHARE):
            what = token[len(REQUIRE_SHARE) :]
            if what != SHARE_FRAMES:
                reasons.append(f"unknown requirement {token!r}")
            elif not spec.share_frames:
                reasons.append(
                    f"workcell {spec.workcell_id!r} does not share camera frames (the operator "
                    "declares share_frames: true)"
                )
        elif token.startswith(REQUIRE_RECORD):
            reason = _record_refusal(token, spec, frames, hint)
            if reason:
                reasons.append(reason)
        else:
            reasons.append(f"unknown requirement {token!r}")
    return reasons


def frame_refusal(diagnosis: FrameDiagnosis, remedy: str) -> str:
    """One sentence naming why a frame does not hold and what restores it."""
    frame, reason = diagnosis.frame, diagnosis.reason
    status = diagnosis.status
    if status is FrameStatus.MISSING:
        return f"{frame} is not calibrated ({remedy})"
    if status is FrameStatus.DEPENDENCY_SUPERSEDED:
        what = f"{frame} calibration is stale ({reason})"
    elif status is FrameStatus.LEGACY_EVIDENCE:
        what = f"{frame} calibration predates the current evidence rules ({reason})"
    elif status is FrameStatus.INSUFFICIENT_ACCURACY:
        what = f"{frame} calibration is not accurate enough ({reason})"
    elif status is FrameStatus.DRIFT_DETECTED:
        what = f"{frame} calibration no longer matches an independent check ({reason})"
    else:
        what = f"{frame} calibration is unproven ({reason})"
    return f"{what}; {remedy}"


def _default_remedy(spec: WorkcellSpec, frame: str) -> str:
    procedure = _procedure(spec, frame)
    return f"record {procedure}" if procedure else f"declare {frame} under 'frames'"


def _record_refusal(token: str, spec: WorkcellSpec, frames: FrameGraph, hint: Hint | None) -> str:
    procedure = token[len(REQUIRE_RECORD) :]
    remedy = hint(token) if hint else ""
    suffix = f"; {remedy}" if remedy else ""
    if procedure not in frames.records:
        return f"workcell {spec.workcell_id!r} has no {procedure} record installed{suffix}"
    problem = frames.record_problems.get(procedure)
    if problem:
        return f"the {procedure} record is stale ({problem}){suffix}"
    return ""


def required_records(requires: Iterable[str]) -> tuple[str, ...]:
    """The procedures *requires* names as ``record:`` tokens, in first-seen order."""
    return tuple(
        dict.fromkeys(
            token[len(REQUIRE_RECORD) :] for token in requires if token.startswith(REQUIRE_RECORD)
        )
    )


_ARGUMENT = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


def requirement_arguments(requires: Iterable[str]) -> tuple[str, ...]:
    """The argument names *requires* binds, in first-seen order."""
    names: dict[str, None] = {}
    for token in requires:
        names.update((name, None) for name in _ARGUMENT.findall(token))
    return tuple(names)


def bind_requirements(requires: Iterable[str], args: Mapping[str, Any]) -> tuple[str, ...]:
    """*requires* with each ``{argument}`` replaced by its value in validated *args*.

    An argument the caller left out and the schema gives no default for stays
    unbound, so the token names nothing the workcell has and is reported unmet.
    """
    return tuple(
        _ARGUMENT.sub(lambda m: str(args.get(m.group(1), m.group(0))), token) for token in requires
    )


def requirement_bindings(
    requires: Sequence[str],
    args_schema: Mapping[str, Any],
    spec: WorkcellSpec,
    frames: FrameGraph,
    hint: Hint | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    """Why no binding of *requires* holds (empty when one does), and the values that do.

    A bound argument's candidates are the workcell roles its schema ``pattern``
    accepts, or its default when it declares no pattern. Without bound arguments
    this is ``unmet_requirements`` with no bindings.
    """
    names = requirement_arguments(requires)
    if not names:
        return unmet_requirements(requires, spec, frames, hint), {}
    choices = [argument_candidates(args_schema, name, spec) for name in names]
    met: dict[str, list[str]] = {name: [] for name in names}
    refusals: list[str] = []
    combos = list(itertools.product(*choices))
    for combo in combos:
        binding = dict(zip(names, combo))
        reasons = unmet_requirements(bind_requirements(requires, binding), spec, frames, hint)
        if not reasons:
            for name, value in binding.items():
                if value not in met[name]:
                    met[name].append(value)
        elif len(combos) == 1:
            refusals.extend(reasons)
        else:
            label = ", ".join(f"{name}={value}" for name, value in binding.items())
            refusals.append(f"{label}: {'; '.join(reasons)}")
    if not combos:
        refusals = [f"workcell {spec.workcell_id!r} binds no role {name} accepts" for name in names]
    if any(met.values()):
        return [], {name: values for name, values in met.items() if values}
    return refusals, {}


def argument_candidates(args_schema: Mapping[str, Any], name: str, spec: WorkcellSpec) -> list[str]:
    """The workcell values argument *name* may bind: roles its pattern accepts, else its default."""
    schema = (args_schema.get("properties") or {}).get(name) or {}
    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        return sorted(role for role in spec.roles if re.fullmatch(pattern, role))
    default = schema.get("default")
    return [str(default)] if default is not None else []


__all__ = [
    "ATTESTATION_SUFFIX",
    "VERIFY_SUFFIX",
    "FrameGraph",
    "argument_candidates",
    "attestation_procedure",
    "bind_requirements",
    "frame_refusal",
    "requirement_arguments",
    "requirement_bindings",
    "required_records",
    "unmet_requirements",
    "verify_procedure",
]
