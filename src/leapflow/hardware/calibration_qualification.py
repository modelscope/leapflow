# Copyright (c) Alibaba, Inc. and its affiliates.
"""Versioned, held-out physical calibration certificates, never training residuals."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

from leapflow.hardware.calibration_store import CalibrationRecord

COMPONENTS = frozenset({"camera_to_base", "flange_tcp", "fingertip_plane",
                        "inner_pad_midpoint", "jaw_stroke", "table_plane"})


@dataclass(frozen=True)
class QualificationLimits:
    min_training: int
    min_holdout: int
    max_rms_mm: float
    max_p95_mm: float
    max_uncertainty_mm: float

    def __post_init__(self) -> None:
        if (any(type(v) is not int or v < 1 for v in (self.min_training, self.min_holdout)) or any(
                not math.isfinite(v) or v <= 0 for v in
                (self.max_rms_mm, self.max_p95_mm, self.max_uncertainty_mm))):
            raise ValueError("invalid held-out qualification limits")


@dataclass(frozen=True)
class CalibrationQualification:
    component: str
    version: str
    mount_digest: str
    units: str
    orientation_convention: str
    accepted_ids: tuple[str, ...]
    rejected_ids: tuple[str, ...]
    holdout_ids: tuple[str, ...]
    holdout_errors_mm: tuple[float, ...]
    uncertainty_mm: float
    independent_reference: str
    observable: bool
    limits: QualificationLimits
    geometry_digest: str = ""

    def __post_init__(self) -> None:
        if self.component not in COMPONENTS or not self.version or not self.mount_digest:
            raise ValueError("calibration needs a known component, version and mount")
        if type(self.observable) is not bool:
            raise ValueError("observability must be independently established as a boolean")
        for key in ("accepted_ids", "rejected_ids", "holdout_ids", "holdout_errors_mm"):
            values = getattr(self, key)
            if not isinstance(values, (list, tuple)) or len(values) > 10000:
                raise ValueError("bounded observation arrays required")
            object.__setattr__(self, key, tuple(values))
        groups = (set(self.accepted_ids), set(self.rejected_ids), set(self.holdout_ids))
        if (any(len(a) != len(b) for a, b in zip(groups, (self.accepted_ids, self.rejected_ids, self.holdout_ids)))
                or any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2)))):
            raise ValueError("accepted, rejected and held-out observations must be disjoint and unique")
        if any(not isinstance(i, str) or not i.strip() for ids in
               (self.accepted_ids, self.rejected_ids, self.holdout_ids) for i in ids):
            raise ValueError("observations need nonempty evidence identifiers")
        if len(self.holdout_ids) != len(self.holdout_errors_mm):
            raise ValueError("every held-out observation needs its physical error")
        if (any(not math.isfinite(v) or v < 0 for v in self.holdout_errors_mm)
                or not math.isfinite(self.uncertainty_mm) or self.uncertainty_mm < 0):
            raise ValueError("physical errors and uncertainty must be finite and nonnegative")

    @property
    def rms_mm(self) -> float | None:
        return (math.sqrt(sum(v * v for v in self.holdout_errors_mm) / len(self.holdout_errors_mm))
                if self.holdout_errors_mm else None)

    @property
    def p95_mm(self) -> float | None:
        errors = sorted(self.holdout_errors_mm)
        return errors[max(0, math.ceil(len(errors) * 0.95) - 1)] if errors else None

    @property
    def qualified(self) -> bool:
        return bool(
            self.units == "mm" and self.orientation_convention == "Rz@Ry@Rx;degrees"
            and self.independent_reference and self.observable
            and len(self.geometry_digest) == 64
            and all(c in "0123456789abcdef" for c in self.geometry_digest)
            and len(self.accepted_ids) >= self.limits.min_training
            and len(self.holdout_ids) >= self.limits.min_holdout
            and self.rms_mm is not None and self.p95_mm is not None
            and self.rms_mm + self.uncertainty_mm <= self.limits.max_rms_mm
            and self.p95_mm + self.uncertainty_mm <= self.limits.max_p95_mm
            and self.uncertainty_mm <= self.limits.max_uncertainty_mm
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def bind_geometry(self, parameters: Mapping, matrix: Sequence[Sequence[float]] | None = None) -> CalibrationQualification:
        """Bind the certificate to the exact independently validated geometry."""
        return replace(self, geometry_digest=geometry_digest(parameters, matrix))

    def record(self, *, device_id: str, recorded_at: float,
               parameters: Mapping | None = None,
               matrix: Sequence[Sequence[float]] | None = None) -> CalibrationRecord:
        """Use the existing versioned store; recording is not physical qualification."""
        if geometry_digest(parameters or {}, matrix) != self.geometry_digest:
            raise ValueError("geometry differs from the qualified certificate")
        return CalibrationRecord(
            device_id=device_id, procedure_id=self.component, recorded_at=recorded_at,
            parameters={"geometry": dict(parameters or {}), "qualification": asdict(self),
                        "qualified": self.qualified, "digest": self.digest}, matrix=matrix,
        )


def geometry_digest(parameters: Mapping, matrix: Sequence[Sequence[float]] | None = None) -> str:
    """Canonical finite geometry, including its transform, is part of qualification."""
    if not parameters and matrix is None:
        raise ValueError("calibration geometry is required")
    payload = {"parameters": dict(parameters), "matrix": matrix}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class CalibrationPackage:
    """Admission result binding certificates to the geometry actually consumed."""

    certificates: tuple[CalibrationQualification, ...]
    records: tuple[CalibrationRecord, ...]

    @property
    def digest(self) -> str:
        return package_digest(self.certificates)

    def require_geometry(self, component: str, parameters: Mapping,
                         matrix: Sequence[Sequence[float]] | None = None) -> None:
        certificate = next((c for c in self.certificates if c.component == component), None)
        if certificate is None or geometry_digest(parameters, matrix) != certificate.geometry_digest:
            raise ValueError(f"active {component} geometry differs from qualified package")


def admit_package(data: Mapping, *, device_id: str, recorded_at: float,
                  max_bytes: int = 1024 * 1024) -> CalibrationPackage:
    """Validate bounded package content before storing or using any component."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("positive package byte budget required")
    if not device_id or not math.isfinite(recorded_at):
        raise ValueError("device and finite recording time required")
    # Snapshot caller-owned mappings, reject nonfinite JSON, and bound admission.
    encoded = json.dumps(data, allow_nan=False)
    if len(encoded.encode()) > max_bytes:
        raise ValueError("calibration package exceeds byte budget")
    data = json.loads(encoded)
    rows = data.get("components") if isinstance(data, Mapping) else None
    if not isinstance(rows, list) or len(rows) != len(COMPONENTS):
        raise ValueError("all six separate calibration components are required")
    certificates, records = [], []
    for row in rows:
        fields = dict(row["qualification"])
        fields["limits"] = QualificationLimits(**fields["limits"])
        for key in ("accepted_ids", "rejected_ids", "holdout_ids", "holdout_errors_mm"):
            fields[key] = tuple(fields[key])
        certificate = CalibrationQualification(**fields)
        records.append(certificate.record(device_id=device_id, recorded_at=recorded_at,
                                          parameters=row["geometry"], matrix=row.get("matrix")))
        certificates.append(certificate)
    package_digest(certificates)
    return CalibrationPackage(tuple(certificates), tuple(records))


def tool_offset_observable(poses_deg: Sequence[Sequence[float]], *,
                           independent_offset: bool, min_orientation_span_deg: float) -> bool:
    """Fixed-orientation translations cannot identify an unknown 3-D tool offset.

    Diverse orientations must constrain every offset direction; a single rotation
    axis leaves its axial component unobservable. A metrology-fixed offset instead
    permits certification limited to the tested orientations.
    """
    if not math.isfinite(min_orientation_span_deg) or min_orientation_span_deg <= 0:
        raise ValueError("positive orientation diversity required")
    if not poses_deg or any(len(p) != 3 or any(not math.isfinite(v) for v in p) for p in poses_deg):
        return False
    if independent_offset:
        return True
    from leapflow.hardware.rgbd_geometry import _rotation
    from leapflow.hardware.rgbd_quality import _numpy
    np = _numpy()
    rotations = [_rotation(np, p) for p in poses_deg]
    differences = np.concatenate([r - rotations[0] for r in rotations[1:]], axis=0) if len(rotations) > 1 else np.zeros((3, 3))
    singular = np.linalg.svd(differences, compute_uv=False)
    return bool(singular[-1] >= 2 * math.sin(math.radians(min_orientation_span_deg) / 2))


def package_digest(certificates: Sequence[CalibrationQualification]) -> str:
    """Require separately versioned components from the same qualified mount."""
    if {c.component for c in certificates} != COMPONENTS or len(certificates) != len(COMPONENTS):
        raise ValueError("all six separate calibration components are required")
    if not all(c.qualified for c in certificates) or len({c.mount_digest for c in certificates}) != 1:
        raise ValueError("unqualified or mixed-mount calibration package")
    return hashlib.sha256(json.dumps(sorted(c.digest for c in certificates)).encode()).hexdigest()
