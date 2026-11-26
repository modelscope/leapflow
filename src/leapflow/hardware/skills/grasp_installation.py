# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bounded stored-evidence inspection and conservative installation preparation.

No capture, mutation, or new storage owner: the existing installation skill writes
prepared parameters through its ordinary consented RunContext.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping

from leapflow.hardware.attestation import declaration_digest
from leapflow.hardware.calibration_qualification import (
    COMPONENTS,
    CalibrationQualification,
    QualificationLimits,
    geometry_digest,
)
from leapflow.hardware.grasp_measurement import (
    EffortReference,
    calibration_error_bounds,
    digest,
    number,
    snapshot,
)
from leapflow.hardware.skills.geometry import Pose, Transform
from leapflow.security.redact import redact_sensitive_text

PROCEDURE = "grasp.qualification"
GEOMETRY_FRAMES = {
    "camera.wrist": "camera_to_base",
    "tcp": "flange_tcp",
    "workspace": "table_plane",
}
_METADATA = frozenset({"installation_scope", "reused_from"})
_PROFILE_KEYS = (
    "color",
    "depth",
    "infrared",
    "align_depth_to_color",
    "enable_infrared",
    "depth_scale",
)
_PRIVATE_KEYS = frozenset(
    {
        "password",
        "token",
        "secret",
        "credentials",
        "credential",
        "api_key",
        "access_token",
        "transport",
        "raw",
        "pixels",
        "image",
        "images",
        "frames",
        "private_frame",
        "private_frames",
        "path",
        "file_path",
        "directory",
        "frame_bytes",
        "data_base64",
    }
)


def installation_scope(
    ctx: Any, registry: Any, dependencies: Mapping[str, Any], model_basis: Mapping[str, Any]
) -> dict[str, Any]:
    """Stable physical/configuration identity; no session clocks or operational settings."""
    spec, frames = ctx.workcell, ctx.frames
    roles = set(dependencies) | {"arm", "gripper", "camera.wrist"}
    cameras = {}
    for role in sorted(roles):
        device = registry.context(spec.role(role)) if spec.role(role) else None
        if device is None or not role.startswith("camera"):
            continue
        config = device.transport.config
        cameras[role] = {
            "mount_digest": config.get("mount_digest", ""),
            "profile_configuration_digest": digest({k: config.get(k) for k in _PROFILE_KEYS}),
            "geometry_configuration_digest": digest(config.get("geometry_measurement") or {}),
            "depth_configuration_digest": digest(config.get("measurement") or {}),
        }
    arm = registry.context(spec.role("arm"))
    return {
        "workcell": spec.workcell_id,
        "roles": {role: spec.role(role) for role in sorted(roles)},
        "frame_bindings": {
            name: {"parent": frames.parent(name), "procedure": frames.procedure(name)}
            for name in sorted(dependencies)
        },
        "cameras": cameras,
        "arm_declaration_digest": declaration_digest(arm) if arm else "",
        "gripper": dict(spec.gripper_spec),
        "physical_model": {k: v for k, v in model_basis.items() if k != "workcell_digest"},
    }


def _payload(data: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k not in _METADATA}


def _content(record: Any) -> str:
    matrix = (
        (
            Transform.from_matrix(record.matrix)
            if record.matrix
            else Transform.from_pose(Pose.from_mapping(record.pose))
        ).to_matrix()
        if (record.matrix or record.pose)
        else None
    )
    return geometry_digest(record.parameters, matrix)


def _rebind(ctx: Any, registry: Any, source: Any, scope: Mapping[str, Any]) -> dict[str, Any]:
    data = snapshot(source.parameters)
    dependencies = data.get("depends_on")
    if not isinstance(dependencies, Mapping) or not set(GEOMETRY_FRAMES) <= set(dependencies):
        raise ValueError(
            "stored qualification lacks required geometry dependencies; inspect grasp.evidence"
        )
    if "installation_scope" in data and data["installation_scope"] != scope:
        raise ValueError(
            "stored installation device/mount/profile/configuration/model scope changed; measure or verify the affected identity"
        )
    changed = any(
        ctx.frames.versions.get(name) != version for name, version in dependencies.items()
    )
    if changed:
        if data.get("installation_scope") != scope:
            raise ValueError(
                "stored installation scope is missing or changed; verify identity/measure the affected component"
            )
        if any(not camera["mount_digest"] for camera in scope["cameras"].values()):
            raise ValueError(
                "changed-version reuse needs a declared mounting identity; independent verification required"
            )
    store = registry.calibration_store
    for name, version in dependencies.items():
        current_version = ctx.frames.versions.get(name)
        if current_version is None:
            raise ValueError(f"{name} no longer holds; verify this dependency before reuse")
        if changed:
            old = store.get_version(
                ctx.workcell.workcell_id, ctx.frames.procedure(name), number(version)
            )
            current = store.get_version(
                ctx.workcell.workcell_id, ctx.frames.procedure(name), current_version
            )
            if old is None or current is None:
                raise ValueError(
                    f"{name} historical record is missing; reuse cannot establish equivalence"
                )
            if (
                _content(old) != _content(current)
                or old.pose != current.pose
                or old.matrix != current.matrix
            ):
                raise ValueError(
                    f"{name} geometry/provenance changed; new component evidence is required"
                )
        failed = store.latest(ctx.workcell.workcell_id, f"{name}.attestation")
        if (
            failed is not None
            and failed.recorded_at > source.recorded_at
            and failed.parameters.get("passed") is False
        ):
            diagnosis = ctx.frames.diagnosis(name)
            passed = (
                store.latest(ctx.workcell.workcell_id, diagnosis.verified_by)
                if diagnosis.verified_by
                else None
            )
            if passed is None or passed.recorded_at <= failed.recorded_at:
                raise ValueError(
                    f"{name} has contradictory drift evidence; independent verification required"
                )
    if changed:
        data["depends_on"] = {name: ctx.frames.versions[name] for name in dependencies}
        data["reused_from"] = {
            "recorded_at": source.recorded_at,
            "digest": digest(source.parameters),
        }
    return data


def prepare_installation(
    ctx: Any,
    registry: Any,
    args: Mapping[str, Any],
    model_basis: Mapping[str, Any],
    admit: Callable[..., Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Propose an exact artifact and source before approval; execute calls this again."""
    if ("qualification" in args) == ("reuse_from" in args):
        raise ValueError(
            "supply exactly one of qualification or reuse_from; inspect grasp.evidence for stored versions"
        )
    store = registry.calibration_store
    if store is None:
        raise ValueError("calibration storage is unavailable")
    source = None
    if "reuse_from" in args:
        source = store.get_version(
            ctx.workcell.workcell_id, PROCEDURE, number(args["reuse_from"], positive=True)
        )
        if source is None:
            raise ValueError(
                "the exact stored qualification version is missing; inspect grasp.evidence"
            )
        data = snapshot(source.parameters)
    else:
        data = snapshot(args["qualification"])
    scope = installation_scope(ctx, registry, data.get("depends_on", {}), model_basis)
    if source is not None:
        data = _rebind(ctx, registry, source, scope)
    elif "installation_scope" in data and data["installation_scope"] != scope:
        raise ValueError(
            "supplied installation scope differs; provide original measurements without installation metadata"
        )
    data["installation_scope"] = scope
    proposed = admit(data)
    current = store.latest(ctx.workcell.workcell_id, PROCEDURE)
    no_op = None
    if current is not None:
        try:
            identical = digest(_payload(current.parameters)) == digest(_payload(data))
            accepted = admit(current.parameters) if identical else None
        except (KeyError, TypeError, ValueError):
            pass  # invalid evidence is never a successful no-op
        else:
            if accepted is not None and current.parameters.get("installation_scope", scope) == scope:
                no_op, proposed = current.recorded_at, accepted
    return {
        **proposed,
        "no_op": no_op,
        "source": {"recorded_at": source.recorded_at, "digest": digest(source.parameters)}
        if source
        else None,
        "installed_version": current.recorded_at if current else None,
    }


def _safe(value: Any, depth: int = 0) -> Any:
    """Disclosure-only redaction; never change an internal measurement artifact."""
    if depth > 32:
        raise ValueError("evidence exceeds structural disclosure depth")
    if isinstance(value, Mapping):
        return {
            k: _safe(v, depth + 1) for k, v in value.items() if str(k).lower() not in _PRIVATE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_safe(v, depth + 1) for v in value]
    return redact_sensitive_text(value, force=True) if isinstance(value, str) else value


def _component(row: Mapping[str, Any], ctx: Any) -> Mapping[str, Any]:
    if not isinstance(row.get("geometry"), Mapping):
        raise ValueError("component geometry object is missing")
    fields = dict(row["qualification"])
    fields["limits"] = QualificationLimits(**fields["limits"])
    certificate = CalibrationQualification(**fields)
    certificate.record(
        device_id=ctx.workcell.workcell_id,
        recorded_at=1,
        parameters=row["geometry"],
        matrix=row.get("matrix"),
    )
    if not certificate.qualified:
        raise ValueError("certificate does not meet its declared held-out bounds")
    bound = calibration_error_bounds(ctx.workcell.grasp_verify)[certificate.component]
    if max(certificate.rms_mm, certificate.p95_mm) + certificate.uncertainty_mm > bound:
        raise ValueError(f"physical calibration errors exceed workcell acceptance ({bound} mm)")
    frame = next(
        (name for name, component in GEOMETRY_FRAMES.items() if component == certificate.component),
        None,
    )
    if frame:
        transform = ctx.frames.transform(frame)
        active = geometry_digest(
            ctx.frames.parameters(frame), transform.to_matrix() if transform else None
        )
        if active != certificate.geometry_digest:
            raise ValueError(f"active {frame} geometry/provenance differs or is unavailable")
    return {
        "status": "qualified_certificate",
        "digest": certificate.digest,
        "p95_mm": certificate.p95_mm,
        "uncertainty_mm": certificate.uncertainty_mm,
        "evidence_attribution": "external_metrology",
        "locally_traceable": False,
        "independent_reference": certificate.independent_reference,
        "accepted_count": len(certificate.accepted_ids),
        "holdout_count": len(certificate.holdout_ids),
    }


def _component_report(data: Mapping[str, Any], ctx: Any, report: dict[str, Any]) -> None:
    calibration = data.get("calibration")
    rows = calibration.get("components", []) if isinstance(calibration, Mapping) else []
    if not isinstance(rows, list) or len(rows) != len(COMPONENTS):
        report["problems"].append("all six separate calibration components are required")
    if not isinstance(rows, list):
        return
    seen: set[str] = set()
    for row in rows[: len(COMPONENTS)]:
        fields = row.get("qualification") if isinstance(row, Mapping) else None
        name = fields.get("component") if isinstance(fields, Mapping) else None
        if not isinstance(name, str) or name not in COMPONENTS:
            report["problems"].append("unknown or malformed component")
            continue
        try:
            if name in seen:
                raise ValueError("duplicate component certificate")
            report["components"][name] = _component(row, ctx)
        except (KeyError, TypeError, ValueError) as error:
            report["components"][name] = {"status": "invalid", "reason": str(error)}
        seen.add(name)


def _export_package(data: Mapping[str, Any], report: dict[str, Any], max_bytes: int) -> None:
    # Only domain package fields are disclosed. Arbitrary caller extensions
    # (e.g. raw sensor frames) stay private; filtered exports are not restamped.
    public = _safe(
        {
            k: v
            for k, v in data.items()
            if k
            in {
                "calibration",
                "effort",
                "camera_device",
                "arm_declaration_digest",
                "profile_digest",
                "scene_id",
                "geometry_configuration_digest",
                "depth_configuration_digest",
                "object_reference",
                "depends_on",
                "installation_scope",
                "reused_from",
            }
        }
    )
    if len(json.dumps(public, allow_nan=False, sort_keys=True).encode()) <= max_bytes:
        report["package"] = public
        report["package_redacted"] = public != data
    else:
        report["package_omitted"] = (
            "package exceeds disclosure byte budget; use the stored version for reuse"
        )


def inspect_evidence(
    ctx: Any,
    registry: Any,
    admit: Callable[..., Mapping[str, Any]],
    *,
    include_package: bool,
    max_bytes: int,
    model_basis: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Read stored metadata while calibration is blocked; success proves no physical outcome."""
    store = registry.calibration_store
    record = store.latest(ctx.workcell.workcell_id, PROCEDURE) if store else None
    report: dict[str, Any] = {
        "metadata_only": True,
        "execution_ready": False,
        "measured_task_outcome": "unverified",
        "installed_version": record.recorded_at if record else None,
        "components": {component: {"status": "missing"} for component in sorted(COMPONENTS)},
        "force": {"status": "missing"},
        "problems": [],
        "package_admissible": False,
        "attribution_note": "External metrology is supported; identifiers/hashes prove attribution and integrity, not physical truth or local acquisition.",
        "frame_diagnoses": {name: ctx.frames.diagnosis(name).to_dict() for name in GEOMETRY_FRAMES},
    }
    if record is None:
        report["problems"].append(
            "grasp.qualification is missing; supply independent measurements, not a repeated camera fit"
        )
        return report
    try:
        data = snapshot(record.parameters)
    except (TypeError, ValueError, RecursionError) as error:
        report["problems"].append(str(error))
        return report
    report["qualification_digest"] = digest(data)
    report["dependencies"] = data.get("depends_on", {})
    dependencies = report["dependencies"]
    report["dependency_status"] = (
        {
            name: {
                "declared_version": version,
                "current_version": ctx.frames.versions.get(name),
                "status": "matched" if ctx.frames.versions.get(name) == version else "mismatched",
                "diagnosis": ctx.frames.diagnosis(name).to_dict(),
            }
            for name, version in dependencies.items()
        }
        if isinstance(dependencies, Mapping)
        else {"status": "malformed"}
    )
    report["missing_fields"] = [
        key
        for key in (
            "camera_device",
            "arm_declaration_digest",
            "profile_digest",
            "scene_id",
            "geometry_configuration_digest",
            "depth_configuration_digest",
            "object_reference",
            "depends_on",
        )
        if not data.get(key)
    ]
    report["scope"] = {
        k: data.get(k) for k in ("scene_id", "profile_digest", "camera_device", "object_reference")
    }
    report["installation_scope"] = data.get("installation_scope")
    _component_report(data, ctx, report)
    try:
        effort = EffortReference.admit(data["effort"], limits=ctx.workcell.grasp_verify)
        report["force"] = {
            "status": "qualified_stationary_reference",
            "condition": "stationary",
            "independent_reference": effort.data["independent_reference"],
            "locally_traceable": False,
        }
    except (KeyError, TypeError, ValueError) as error:
        report["force"] = {"status": "invalid", "reason": str(error)}
    try:
        admit(data)
        if data.get("installation_scope") is not None and data[
            "installation_scope"
        ] != installation_scope(ctx, registry, dependencies, model_basis):
            raise ValueError("stored physical-model installation scope changed")
        report["package_admissible"] = True
    except (KeyError, TypeError, ValueError) as error:
        report["package_admissible"] = False
        report["problems"].append(str(error))
    report["next_step"] = (
        "calib.grasp_evidence reuse_from=<installed_version> can preview validated reuse; missing evidence needs measurement"
    )
    if include_package:
        _export_package(data, report, max_bytes)
    return _safe(report)
