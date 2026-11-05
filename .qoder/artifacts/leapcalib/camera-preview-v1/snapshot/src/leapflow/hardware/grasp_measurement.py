# Copyright (c) Alibaba, Inc. and its affiliates.
"""Qualified contact measurements from metric observations and independent metrology.

This is a node-local adapter, not a classifier over caller-supplied holding flags.
Effort qualification is limited to stationary observations at characterized poses
and widths. Moving effort and legacy current conventions are never admitted.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from leapflow.hardware.calibration_qualification import CalibrationPackage, admit_package
from leapflow.hardware.grasp_evidence import EvidenceIdentity, GraspMeasurement
from leapflow.hardware.skills.geometry import Plane, Pose, Transform


def snapshot(data: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(data, allow_nan=False, sort_keys=True)
    if len(encoded.encode()) > 1024 * 1024:
        raise ValueError("grasp qualification exceeds byte budget")
    return json.loads(encoded)


def digest(data: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def number(value: Any, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError("finite measured numeric value required")
    return float(value)


def vector(value: Sequence[Any]) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("three measured coordinates required")
    return tuple(number(v) for v in value)


@dataclass(frozen=True)
class EffortReference:
    """Independently labeled, disjoint train/holdout measurements in N.

    Each observation carries encoder width, flange pose, decoded effort, and an
    independent force measurement. Nearest characterized width/pose admission
    avoids extrapolating the force model across arbitrary loads or orientations.
    """

    data: Mapping[str, Any]
    empty_max_n: float
    contact_min_n: float
    contact_max_n: float
    uncertainty_n: float

    @classmethod
    def admit(cls, raw: Mapping[str, Any], *, limits: Mapping[str, float]) -> EffortReference:
        data = snapshot(raw)
        if (
            data.get("units") != "N"
            or data.get("condition") != "stationary"
            or not isinstance(data.get("independent_reference"), str)
            or not data["independent_reference"].strip()
            or data.get("physical") is not True
        ):
            raise ValueError("independent physical stationary N-unit force reference required")
        number(data["command_force_n"], positive=True)
        uncertainty = number(data["uncertainty_n"], positive=True)
        if uncertainty > limits["effort_uncertainty_n"]:
            raise ValueError("force uncertainty exceeds declared acceptance")
        training, holdout = data["training"], data["holdout"]
        if (
            not isinstance(training, list)
            or not isinstance(holdout, list)
            or len(training) > 1000
            or len(holdout) > 1000
        ):
            raise ValueError("bounded effort observations required")
        ids = []
        groups: dict[str, list[float]] = {"empty": [], "contact": []}
        for split, rows in (("training", training), ("holdout", holdout)):
            counts = {"empty": 0, "contact": 0}
            for row in rows:
                label = row["label"]
                if label not in counts or row.get("stationary") is not True:
                    raise ValueError("stationary independently labeled force rows required")
                identifier = row["id"]
                if not isinstance(identifier, str) or not identifier.strip():
                    raise ValueError("force observation identifier required")
                ids.append(identifier)
                effort, force, width = (
                    number(row[k]) for k in ("effort_n", "reference_force_n", "width_mm")
                )
                Pose.from_mapping(row["flange"])
                if min(effort, force, width) < 0:
                    raise ValueError("nonnegative physical force and width required")
                if abs(effort - force) + uncertainty > limits["effort_error_n"]:
                    raise ValueError("decoded effort differs from independent force reference")
                if (label == "empty" and force > uncertainty) or (
                    label == "contact" and force <= 2 * uncertainty
                ):
                    raise ValueError("independent force does not establish the row label")
                counts[label] += 1
                if split == "training":
                    groups[label].append(effort)
            if min(counts.values()) < 3:
                raise ValueError("three training and three held-out rows per force class required")
        if len(set(ids)) != len(ids):
            raise ValueError("force training/holdout identifiers must be disjoint and unique")
        empty_max = max(groups["empty"]) + uncertainty
        contact_min = min(groups["contact"]) - uncertainty
        if empty_max >= contact_min:
            raise ValueError("unloaded/contact force distributions overlap")
        reference = cls(
            data, empty_max, contact_min, max(groups["contact"]) + uncertainty, uncertainty
        )
        for row in holdout:
            predicted = reference.classify(
                number(row["effort_n"]),
                number(row["width_mm"]),
                Pose.from_mapping(row["flange"]),
                limits=limits,
            )
            if predicted != row["label"]:
                raise ValueError("independent held-out effort verification failed")
        return reference

    def classify(
        self, effort_n: float, width_mm: float, flange: Pose, *, limits: Mapping[str, float]
    ) -> str:
        effort_n, width_mm = number(effort_n), number(width_mm)
        label = (
            "empty"
            if effort_n + self.uncertainty_n <= self.empty_max_n
            else "contact"
            if (
                effort_n - self.uncertainty_n >= self.contact_min_n
                and effort_n + self.uncertainty_n <= self.contact_max_n
            )
            else "unknown"
        )
        characterized = any(
            row["label"] == label
            and abs(width_mm - row["width_mm"]) <= limits["effort_width_scope_mm"]
            and flange.distance_mm(Pose.from_mapping(row["flange"]))
            <= limits["effort_pose_scope_mm"]
            and flange.rotation_to_deg(Pose.from_mapping(row["flange"]))
            <= limits["effort_pose_scope_deg"]
            for row in self.data["training"]
        )
        return label if characterized else "unknown"


@dataclass(frozen=True)
class GraspQualification:
    data: Mapping[str, Any]
    package: CalibrationPackage
    effort: EffortReference

    @classmethod
    def admit(
        cls,
        raw: Mapping[str, Any],
        *,
        device_id: str,
        recorded_at: float,
        limits: Mapping[str, float],
    ) -> GraspQualification:
        data = snapshot(raw)
        package = admit_package(data["calibration"], device_id=device_id, recorded_at=recorded_at)
        effort = EffortReference.admit(data["effort"], limits=limits)
        bounds = {
            "camera_to_base": limits["max_relative_error_mm"],
            "flange_tcp": limits["max_relative_error_mm"],
            "table_plane": limits["support_mm"],
            "inner_pad_midpoint": limits["pad_midpoint_mm"],
            "jaw_stroke": limits["jaw_stroke_error_mm"],
            "fingertip_plane": limits["contact_geometry_mm"],
        }
        for certificate in package.certificates:
            if (
                certificate.rms_mm + certificate.uncertainty_mm > bounds[certificate.component]
                or certificate.p95_mm + certificate.uncertainty_mm > bounds[certificate.component]
            ):
                raise ValueError("physical calibration errors exceed workcell acceptance")
        for key in (
            "profile_digest",
            "camera_device",
            "arm_declaration_digest",
            "scene_id",
            "geometry_configuration_digest",
            "depth_configuration_digest",
        ):
            if not isinstance(data.get(key), str) or not data[key].strip():
                raise ValueError(f"qualified {key} required")
        obj = data["object_reference"]
        if (
            obj.get("physical") is not True
            or not obj.get("independent_reference")
            or not obj.get("measurement_id")
        ):
            raise ValueError("independent physical object metrology required")
        if any(v <= 0 for v in vector(obj["dimensions_mm"])):
            raise ValueError("positive measured object dimensions required")
        if not 0 < number(obj["uncertainty_mm"]) <= limits["max_uncertainty_mm"]:
            raise ValueError("object measurement uncertainty is unqualified")
        jaw = next(
            r.parameters["geometry"] for r in package.records if r.procedure_id == "jaw_stroke"
        )
        if number(jaw["scale"], positive=True) <= 0 or number(jaw["min_width_mm"]) < 0:
            raise ValueError("qualified jaw stroke required")
        if number(jaw["max_width_mm"]) <= jaw["min_width_mm"]:
            raise ValueError("qualified jaw width range required")
        number(jaw["offset_mm"])
        midpoint = next(
            r.parameters["geometry"]
            for r in package.records
            if r.procedure_id == "inner_pad_midpoint"
        )
        vector(midpoint["tool_mm"])
        fingertip = next(
            r.parameters["geometry"] for r in package.records if r.procedure_id == "fingertip_plane"
        )
        number(fingertip["tool_z_mm"])
        return cls(data, package, effort)

    @property
    def digest(self) -> str:
        return digest(self.data)

    def geometry(self, component: str) -> Mapping[str, Any]:
        return next(
            r.parameters["geometry"] for r in self.package.records if r.procedure_id == component
        )

    def measure(
        self,
        geometry: Mapping[str, Any],
        *,
        width_mm: float,
        effort_n: float,
        flange: Pose,
        tcp: Pose,
        mount: Transform,
        table: Plane,
        synchronized: bool,
        limits: Mapping[str, float],
    ) -> tuple[GraspMeasurement, float | None]:
        """Produce flags from independent metric geometry; never consume hold flags."""
        width_mm, effort_n = number(width_mm), number(effort_n)
        if (
            geometry["profile_digest"] != self.data["profile_digest"]
            or geometry["scene_id"] != self.data["scene_id"]
            or geometry.get("configuration_digest") != self.data["geometry_configuration_digest"]
        ):
            raise ValueError("camera profile or scene differs from qualification")
        pads = geometry.get("pad_depth", [])
        if (
            len(pads) != 2
            or not all(p.get("qualified") is True for p in pads)
            or geometry.get("independent_jaws") is not True
        ):
            raise ValueError("two independently depth-qualified pads required")
        mount_id = self.package.certificates[0].mount_digest
        for depth in (*pads, geometry["depth"]):
            if (
                depth["calibration_digest"] != self.package.digest
                or depth["mount_digest"] != mount_id
                or depth["profile_digest"] != self.data["profile_digest"]
            ):
                raise ValueError("depth qualification scope differs from physical calibration")
        frames = pads[0]["frame_ids"]
        if (
            not frames
            or frames != pads[1]["frame_ids"]
            or (geometry.get("object_detected") and frames != geometry["depth"]["frame_ids"])
        ):
            raise ValueError("pads did not use the same source frames")
        identity = EvidenceIdentity(
            geometry["session_id"],
            geometry["scene_id"],
            self.digest,
            number(geometry["observed_mono"]),
            (frames[-1],),
        )
        camera_base = Transform.from_pose(flange).compose(mount)
        tool_camera = Transform.from_pose(tcp).inverse().compose(camera_base)
        points = geometry.get("jaw_points_camera_mm", [])
        if len(points) != 2:
            raise ValueError("observed pad endpoints required")
        jaws_tool = sorted((tool_camera.apply(vector(p)) for p in points), key=lambda p: p[0])
        gap = jaws_tool[1][0] - jaws_tool[0][0]
        midpoint_tool = tuple((a + b) / 2 for a, b in zip(*jaws_tool))
        jaws_base = Transform.from_pose(tcp).apply(midpoint_tool)
        uncertainty = sum(number(p["uncertainty_mm"]) for p in pads) + max(
            c.uncertainty_mm for c in self.package.certificates
        )
        if uncertainty > limits["max_uncertainty_mm"]:
            raise ValueError("combined measurement uncertainty exceeds acceptance")
        tip_z = number(self.geometry("fingertip_plane")["tool_z_mm"])
        if any(p[2] + uncertainty > tip_z + limits["contact_geometry_mm"] for p in jaws_tool):
            raise ValueError(
                "observed pads extend past the independently qualified fingertip plane"
            )
        expected_midpoint = vector(self.geometry("inner_pad_midpoint")["tool_mm"])
        if math.dist(midpoint_tool, expected_midpoint) + uncertainty > limits["pad_midpoint_mm"]:
            raise ValueError("observed midpoint differs from independently qualified tool geometry")
        jaw = self.geometry("jaw_stroke")
        in_stroke = jaw["min_width_mm"] <= width_mm <= jaw["max_width_mm"]
        width_consistent = (
            in_stroke
            and abs(gap - (jaw["scale"] * width_mm + jaw["offset_mm"])) + uncertainty
            <= limits["jaw_stroke_error_mm"]
        )
        effort_class = self.effort.classify(effort_n, width_mm, flange, limits=limits)
        empty = (
            synchronized
            and width_consistent
            and effort_class == "empty"
            and gap + uncertainty <= limits["empty_gap_mm"]
        )
        proposals = geometry.get("proposals", [])
        valid = [
            p for p in proposals if p.get("observable") is True and p.get("depth_qualified") is True
        ]
        if not geometry.get("object_detected"):
            return GraspMeasurement(
                identity,
                jaws_base,
                jaws_base,
                width_mm,
                uncertainty,
                False,
                synchronized,
                True,
                False,
                width_consistent,
                False,
                empty,
                False,
            ), None
        if len(valid) != 1 or len(proposals) != 1 or geometry["depth"].get("qualified") is not True:
            raise ValueError("a unique observable depth-qualified cuboid is required")
        fit = valid[0]
        dims = vector(geometry["dimensions_mm"][fit["hypothesis_id"]])
        measured = vector(self.data["object_reference"]["dimensions_mm"])
        if any(
            abs(a - b) > self.data["object_reference"]["uncertainty_mm"]
            for a, b in zip(dims, measured)
        ):
            raise ValueError("cuboid dimensions differ from independent object measurement")
        if (
            number(fit["surface_rms_mm"]) + uncertainty > limits["contact_geometry_mm"]
            or number(fit["contour_rms_px"]) > limits["contact_contour_px"]
        ):
            raise ValueError("contact geometry fit is unqualified")
        object_camera = Transform.from_pose(
            Pose(*vector(fit["center_camera_mm"]), *vector(fit["rpy_deg"]))
        )
        object_tool = tool_camera.compose(object_camera)
        # A face normal must align with the closing axis; no rotated-box projection
        # can substitute for observing opposing pad contacts on physical faces.
        columns = [tuple(object_tool.rotation[i][j] for i in range(3)) for j in range(3)]
        axis = max(range(3), key=lambda j: abs(columns[j][0]))
        if math.degrees(math.acos(min(1.0, abs(columns[axis][0])))) > limits["contact_axis_deg"]:
            raise ValueError("cuboid faces are not aligned with the qualified closing axis")
        inverse = object_tool.inverse()
        local_pads = [inverse.apply(p) for p in jaws_tool]
        engagement = True
        for p in local_pads:
            engagement &= (
                abs(abs(p[axis]) - dims[axis] / 2) + uncertainty <= limits["contact_geometry_mm"]
            )
            engagement &= all(abs(p[j]) + uncertainty < dims[j] / 2 for j in range(3) if j != axis)
        engagement &= local_pads[0][axis] * local_pads[1][axis] < 0
        center_base = camera_base.apply(vector(fit["center_camera_mm"]))
        object_base = camera_base.compose(object_camera)
        extent_normal = sum(
            abs(sum(table.normal[i] * object_base.rotation[i][j] for i in range(3))) * dims[j] / 2
            for j in range(3)
        )
        bottom_distance = table.above(center_base) - extent_normal
        support = abs(bottom_distance) + uncertainty <= limits["support_mm"]
        alignment = object_tool.translation[0] - midpoint_tool[0]
        return GraspMeasurement(
            identity,
            center_base,
            jaws_base,
            width_mm,
            uncertainty,
            True,
            synchronized,
            True,
            bool(engagement),
            width_consistent,
            effort_class == "contact",
            False,
            support,
        ), alignment
