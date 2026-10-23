# Copyright (c) Alibaba, Inc. and its affiliates.
"""Calibrated workcell frames and requirement checks, resolved from versioned records.

``FrameGraph`` snapshots every frame a workcell declares (and its workspace
box and table plane) from the ``CalibrationStore``; ``unmet_requirements``
explains why a skill's ``requires`` tokens do not hold in one snapshot.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

from leapflow.hardware.skills.geometry import Plane, Pose, Transform, WorkspaceBox
from leapflow.hardware.skills.protocol import (
    REQUIRE_CALIBRATION,
    REQUIRE_MODEL,
    REQUIRE_ROLE,
    REQUIRE_SHARE,
    SHARE_FRAMES,
)
from leapflow.hardware.skills.workcell import DEPENDS_ON, WORKSPACE_FRAME, WorkcellSpec

logger = logging.getLogger(__name__)

_VERSION_TOLERANCE = 1e-6


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
    ) -> None:
        self._spec = spec
        self._transforms = dict(transforms)
        self._workspace = workspace
        self._versions = dict(versions)
        self._stale = dict(stale or {})
        self._table = table
        self._parameters = {name: dict(value) for name, value in (parameters or {}).items()}

    @classmethod
    def snapshot(cls, spec: WorkcellSpec, store: Any | None) -> "FrameGraph":
        """Resolve every declared frame and the workspace from *store*."""
        records: dict[str, Any] = {}
        if store is not None:
            for name in (*spec.frames, WORKSPACE_FRAME):
                procedure = _procedure(spec, name)
                record = store.latest(spec.workcell_id, procedure) if procedure else None
                if record is not None:
                    records[name] = record
        stale = _stale_records(records)
        transforms: dict[str, Transform] = {}
        versions: dict[str, float] = {}
        parameters: dict[str, Mapping[str, Any]] = {}
        workspace: WorkspaceBox | None = None
        table: Plane | None = None
        for name, record in records.items():
            if name in stale:
                continue
            if name == WORKSPACE_FRAME:
                workspace = _record_box(record)
                resolved = workspace is not None
                table = _record_table(record) if resolved else None
            else:
                transform = _record_transform(record)
                resolved = transform is not None
                if transform is not None:
                    transforms[name] = transform
            if resolved:
                versions[name] = float(record.recorded_at)
                parameters[name] = dict(record.parameters or {})
        return cls(spec, transforms, workspace, versions, stale, table, parameters)

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
        """Frames whose latest record was measured against a superseded dependency."""
        return dict(self._stale)

    def parameters(self, name: str) -> Mapping[str, Any]:
        """The parameters recorded with a resolved frame (e.g. the intrinsics a
        hand-eye calibration was measured with); empty when unresolved."""
        return dict(self._parameters.get(name) or {})


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
    requires: Iterable[str], spec: WorkcellSpec, frames: FrameGraph
) -> list[str]:
    """Human-readable reasons *requires* is not met in this workcell; empty when met."""
    reasons: list[str] = []
    versions = frames.versions
    stale = frames.stale
    for token in requires:
        if token.startswith(REQUIRE_ROLE):
            role = token[len(REQUIRE_ROLE) :]
            if not spec.role(role):
                reasons.append(f"workcell {spec.workcell_id!r} binds no {role}")
        elif token.startswith(REQUIRE_CALIBRATION):
            frame = token[len(REQUIRE_CALIBRATION) :]
            if frame in stale:
                reasons.append(
                    f"{frame} calibration is stale ({stale[frame]}); "
                    f"run calib.{_calib_skill(frame)} again"
                )
            elif frame not in versions:
                reasons.append(f"{frame} is not calibrated (run calib.{_calib_skill(frame)})")
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
        else:
            reasons.append(f"unknown requirement {token!r}")
    return reasons


def _calib_skill(frame: str) -> str:
    return "handeye" if frame.split(".", 1)[0] == "camera" else frame


__all__ = ["FrameGraph", "unmet_requirements"]
