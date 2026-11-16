# Copyright (c) Alibaba, Inc. and its affiliates.
"""Owner-indexed raw value lineage, separate from feature and physical qualification.

This gate cannot certify tracking association, an independent witness, task pose,
clearance, or a structural parameter. It verifies every primitive source, including
unused and rejected rows, against private unexpired captures. A signed historical
receipt is evidence of acquisition approval, never a new execution permit.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from leapcalib.contracts.identity import digest_of
from leapcalib.contracts.motion import MotionObservation, finite_vector
from leapcalib.contracts.observations import JointObservation, RgbdObservation, Stamp
from leapcalib.contracts.raw_task import RawTaskTransition, identities
from leapcalib.contracts.units import canonical_number
from leapcalib.validation.bundle import EvidenceBundle
from leapflow.hardware.calibration.acquisition import SettledObservation
from leapflow.hardware.lhp.permit import ExecutionPermit, MAX_PERMIT_TTL_S, PERMIT_VERSION

SCHEMA = "leapflow.indexed_source_manifest.v1"
MAX_SOURCES = 4096


@runtime_checkable
class IndexedCaptureReader(Protocol):
    def read(self, identity: str, *, workcell_id: str) -> dict[str, Any]: ...


def approval_provenance(
    receipt: Mapping[str, Any],
    *,
    trusted_keys: Mapping[str, Ed25519PublicKey],
    node_id: str,
) -> dict[str, str]:
    """Verify an indexed historical signature without touching nonce or expiry state.

    The existing gate already consumed authority when it created this private
    receipt. Expiry/pre-boot/replay checks belong to execution, not later evidence
    evaluation. This function does not accept, consume, renew or issue authority.
    """
    identities((node_id,))
    if not isinstance(receipt, Mapping) or set(receipt) != {"mode", "action_digest", "permit"}:
        raise ValueError("exact indexed admission receipt required")
    data = receipt["permit"]
    if receipt["mode"] != "permit_required" or not isinstance(data, Mapping):
        raise ValueError("legacy acquisition has no independently signed approval provenance")
    if set(data) != set(ExecutionPermit.__dataclass_fields__):
        raise ValueError("exact historical permit fields required")
    if (
        type(data["v"]) is not int
        or data["v"] != PERMIT_VERSION
        or type(data["max_uses"]) is not int
        or data["max_uses"] != 1
    ):
        raise ValueError("historical acquisition requires a versioned one-shot permit")
    for name in (
        "issuer",
        "key_id",
        "audience",
        "session_id",
        "workspace",
        "approval_id",
        "action_digest",
        "nonce",
        "signature",
    ):
        identities((data[name],))
    issued, expires = canonical_number(data["issued_at"]), canonical_number(data["expires_at"])
    if not 0 < expires - issued <= MAX_PERMIT_TTL_S or issued <= 0:
        raise ValueError("invalid signed acquisition permit interval")
    if data["audience"] != node_id or data["action_digest"] != receipt["action_digest"]:
        raise ValueError("acquisition approval audience or plan digest differs")
    key = trusted_keys.get(data["key_id"])
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("acquisition signing key is not trusted by this owner")
    permit = ExecutionPermit(**data)
    try:
        key.verify(bytes.fromhex(permit.signature), permit.signable_payload())
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("indexed acquisition signature is invalid") from exc
    return {
        "receipt_digest": digest_of(receipt),
        "action_digest": permit.action_digest,
        "approval_digest": digest_of({"approval_id": permit.approval_id}),
        "client_digest": digest_of(
            {"session_id": permit.session_id, "workspace": permit.workspace}
        ),
        "key_id": permit.key_id,
        "node_id": permit.audience,
    }


@dataclass(frozen=True)
class IndexedSourceManifest:
    """Immutable redacted raw-value lineage, not a partial physical certificate."""

    evidence_digest: str
    topology_digest: str
    node_session_id: str
    rows_json: str

    def __post_init__(self) -> None:
        for identity in (self.evidence_digest, self.topology_digest, self.node_session_id):
            identities((identity,))
        if not isinstance(self.rows_json, str) or len(self.rows_json.encode()) > 2 * 1024 * 1024:
            raise ValueError("bounded immutable indexed source rows required")
        rows = json.loads(self.rows_json)
        if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_SOURCES:
            raise ValueError("bounded indexed source row count required")
        acquisitions = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "source_id",
                "source_digest",
                "acquisition_digest",
                "observation_digest",
                "plan_digest",
                "run_id",
                "approval",
                "acquisition_window_s",
            }:
                raise ValueError("exact indexed source manifest fields required")
            for field in (
                "source_id",
                "source_digest",
                "acquisition_digest",
                "observation_digest",
                "plan_digest",
                "run_id",
            ):
                identities((row[field],))
            approval = row["approval"]
            if not isinstance(approval, dict) or set(approval) != {
                "receipt_digest",
                "action_digest",
                "approval_digest",
                "client_digest",
                "key_id",
                "node_id",
            }:
                raise ValueError("exact redacted approval provenance required")
            for identity in approval.values():
                identities((identity,))
            lower, upper = finite_vector(row["acquisition_window_s"], 2)
            if not 0 < lower < upper:
                raise ValueError("increasing bounded acquisition window required")
            acquisition = {k: v for k, v in row.items() if k not in ("source_id", "source_digest")}
            previous = acquisitions.setdefault(row["acquisition_digest"], acquisition)
            if previous != acquisition:
                raise ValueError("conflicting immutable acquisition lineage")
        identities(tuple(r["source_id"] for r in rows), maximum=MAX_SOURCES)
        digest_of({"sources": rows})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "evidence_digest": self.evidence_digest,
            "topology_digest": self.topology_digest,
            "node_session_id": self.node_session_id,
            "sources": json.loads(self.rows_json),
            "scope": "indexed_raw_values",
            "feature_association": "not_established",
            "physical_admission": "not_established",
        }

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())


def _camera(source: RgbdObservation, capture: SettledObservation) -> None:
    roi = capture.roi
    stamp = Stamp(
        roi.stamp.seconds,
        domain=roi.stamp.domain,
        semantics=roi.stamp.semantics,
        uncertainty_s=roi.stamp.uncertainty_s + roi.exposure_uncertainty_s,
    )
    if (
        source.observation_id != roi.frame_id
        or source.stream != roi.profile
        or source.stamp != stamp
        or source.provenance != roi.provenance
        or source.provenance != "metric_rgbd"
    ):
        raise ValueError("RGB-D source differs from its exact indexed frame/profile/clock")
    if not isinstance(source.samples, tuple) or len(source.samples) > 2048:
        raise ValueError("bounded raw RGB-D samples required")
    seen = set()
    for sample in source.samples:
        u, v = sample.u_px, sample.v_px
        if (
            not u.is_integer()
            or not v.is_integer()
            or not roi.x <= u < roi.x + roi.width
            or not roi.y <= v < roi.y + roi.height
            or (u, v) in seen
        ):
            raise ValueError("raw RGB-D samples require distinct integer pixels inside indexed ROI")
        seen.add((u, v))
        offset = 4 * ((int(v) - roi.y) * roi.width + int(u) - roi.x)
        measured = struct.unpack_from("<f", roi.depth_m_f32le, offset)[0]
        if sample.depth_m != measured:
            raise ValueError("RGB-D sample depth differs from private lossless metric bytes")
        if sample.link_id:
            raise ValueError("raw pixels cannot establish rigid carried-link identity")


def _joint(source: JointObservation, capture: SettledObservation) -> None:
    matches = [f for f in capture.feedback if f.source_id == source.observation_id]
    if len(matches) != 1:
        raise ValueError("raw feedback source identity is not in the indexed cohort")
    sample = matches[0]
    if (
        source.model.digest != sample.model_digest
        or source.stamp != sample.stamp
        or source.q_raw_rad != sample.joints_rad
        or source.still is not True
    ):
        raise ValueError("raw feedback values/model/clock/stillness differ from indexed cohort")


def _plan_binding(
    content: Mapping[str, Any], capture: SettledObservation, bundle: EvidenceBundle
) -> None:
    """Qualification requires sealed source identities; legacy records remain readable."""
    binding = content["binding"]
    collection = binding["plan"]["basis"].get("collection")
    first, roi = capture.feedback[0], capture.roi
    expected = {
        "model_digest": first.model_digest,
        "declaration_digest": first.declaration_digest,
        "joint_ids": list(first.joint_ids),
        "node_session_id": first.node_session_id,
        "camera_device_id": roi.profile.device_id,
        "mount_digest": roi.mount_digest,
        "camera_session_id": roi.session_id,
        "source_profile_digest": roi.source_profile_digest,
        "metric_profile_digest": roi.profile.digest,
        "region": [roi.x, roi.y, roi.width, roi.height],
    }
    if (
        not isinstance(collection, Mapping)
        or any(collection.get(k) != v for k, v in expected.items())
        or not isinstance(collection.get("profile"), Mapping)
        or collection["profile"].get("settled") != content["profile"]
    ):
        raise ValueError(
            "acquisition lacks matching sealed source identities and uncertainty policy"
        )
    if (
        first.model_digest != bundle.topology.model.digest
        or first.device_id != bundle.topology.device.device_id
        or roi.profile not in bundle.topology.streams
    ):
        raise ValueError("indexed acquisition differs from evidence topology")
    if not bundle.topology.mounts or roi.mount_digest not in {
        m.digest for m in bundle.topology.mounts
    }:
        raise ValueError("indexed camera mount is absent from evidence topology")
    if binding["workcell_id"] != bundle.topology.workcell_id or binding[
        "node_session_id"
    ] != bundle.metadata.get("node_session_id"):
        raise ValueError("indexed acquisition belongs to a different workcell/node-session clock")


def indexed_source_manifest(
    store: IndexedCaptureReader,
    bundle: EvidenceBundle,
    acquisition_ids: Mapping[str, str],
    *,
    trusted_keys: Mapping[str, Ed25519PublicKey],
    node_id: str,
) -> IndexedSourceManifest:
    """Recheck all raw primitive rows through the registry store, off the owner loop.

    The map uses actual frame/feedback IDs, never caller-created aliases. Derived
    factors must be recomputed by their method-specific consumer after this gate;
    neither a boolean nor a matching candidate digest is physical evidence.
    """
    if (
        not isinstance(store, IndexedCaptureReader)
        or not isinstance(bundle, EvidenceBundle)
        or not isinstance(acquisition_ids, Mapping)
        or not isinstance(trusted_keys, Mapping)
    ):
        raise ValueError("typed evidence and bounded indexed source map required")
    sources = [o for o in bundle.observations if isinstance(o, (RgbdObservation, JointObservation))]
    if not 1 <= len(sources) <= MAX_SOURCES or set(acquisition_ids) != {
        s.observation_id for s in sources
    }:
        raise ValueError(
            "every raw source, including rejected/unused rows, requires exact indexed lineage"
        )
    identities(tuple(acquisition_ids), maximum=MAX_SOURCES)
    for identity in acquisition_ids.values():
        identities((identity,))
    session = bundle.metadata.get("node_session_id")
    identities((session,))
    contents: dict[str, tuple[Mapping[str, Any], SettledObservation, dict[str, str]]] = {}
    rows = []
    for source in sorted(sources, key=lambda s: s.observation_id):
        identity = acquisition_ids[source.observation_id]
        if identity not in contents:
            content = store.read(identity, workcell_id=bundle.topology.workcell_id)
            capture = SettledObservation.from_mapping(content["observation"])
            _plan_binding(content, capture, bundle)
            approval = approval_provenance(
                content["binding"]["admission"], trusted_keys=trusted_keys, node_id=node_id
            )
            contents[identity] = (content, capture, approval)
        content, capture, approval = contents[identity]
        if isinstance(source, RgbdObservation):
            _camera(source, capture)
        else:
            _joint(source, capture)
        rows.append(
            {
                "source_id": source.observation_id,
                "source_digest": source.digest,
                "acquisition_digest": identity,
                "observation_digest": capture.digest,
                "plan_digest": content["binding"]["plan"]["digest"],
                "run_id": content["binding"]["run_id"],
                "approval": approval,
            }
        )
    # Include acquisition windows in provenance; a later qualifier must count
    # these complete windows rather than the convenient endpoint stamps alone.
    windows = {
        identity: [
            capture.feedback[0].first_receipt_s - capture.feedback[0].stamp.uncertainty_s,
            capture.feedback[-1].last_receipt_s + capture.feedback[-1].stamp.uncertainty_s,
        ]
        for identity, (_, capture, _) in contents.items()
    }
    if any(not math.isfinite(v) for window in windows.values() for v in window):
        raise ValueError("finite indexed acquisition windows required")
    for row in rows:
        row["acquisition_window_s"] = windows[row["acquisition_digest"]]
    return IndexedSourceManifest(
        bundle.digest,
        bundle.topology.digest,
        session,
        json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False),
    )


def _factor_sources(factor: MotionObservation | RawTaskTransition) -> tuple[str, ...]:
    if isinstance(factor, MotionObservation):
        return factor.source_ids
    return (
        *factor.first.frame_ids,
        *factor.first.feedback_ids,
        *factor.second.frame_ids,
        *factor.second.feedback_ids,
    )


def _indexed_windows(
    bundle: EvidenceBundle,
    manifest: IndexedSourceManifest,
) -> dict[str, tuple[float, float, frozenset[str]]]:
    if (
        not isinstance(bundle, EvidenceBundle)
        or not isinstance(manifest, IndexedSourceManifest)
        or (manifest.evidence_digest, manifest.topology_digest, manifest.node_session_id)
        != (bundle.digest, bundle.topology.digest, bundle.metadata.get("node_session_id"))
    ):
        raise ValueError("indexed manifest differs from exact evidence/topology/node session")
    rows = {row["source_id"]: row for row in manifest.to_dict()["sources"]}
    sources = {
        o.observation_id: o
        for o in bundle.observations
        if isinstance(o, (RgbdObservation, JointObservation))
    }
    if set(rows) != set(sources) or any(
        rows[i]["source_digest"] != source.digest for i, source in sources.items()
    ):
        raise ValueError("indexed manifest source values differ from evidence")
    factors = {o.observation_id: o for o in bundle.observations}
    windows = {}
    for identity in (*bundle.accepted_ids, *bundle.rejected_ids, *bundle.holdout_ids):
        factor = factors[identity]
        if not isinstance(factor, (MotionObservation, RawTaskTransition)):
            raise ValueError("indexed independence requires method-specific split factors")
        bindings = [rows[i] for i in _factor_sources(factor)]
        windows[identity] = (
            min(r["acquisition_window_s"][0] for r in bindings),
            max(r["acquisition_window_s"][1] for r in bindings),
            frozenset(r["acquisition_digest"] for r in bindings),
        )
    return windows


def independent_indexed_holdouts(
    bundle: EvidenceBundle,
    manifest: IndexedSourceManifest,
    *,
    baseline_bundle: EvidenceBundle | None = None,
    baseline_manifest: IndexedSourceManifest | None = None,
    separation_s: float = 0.3,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Exclude complete physical acquisition windows, not aliased endpoint records.

    Use only owner-recomputed manifests. This independence check alone establishes
    neither spatial excitation nor witness/measurement validity. Invalid branches
    remain explicit and also block adjacent holdouts from being counted twice.
    """
    separation_s = canonical_number(separation_s)
    if not 0.3 <= separation_s <= 120:
        raise ValueError("bounded indexed temporal separation cannot be weakened")
    windows = _indexed_windows(bundle, manifest)
    blocked = [windows[i] for i in (*bundle.accepted_ids, *bundle.rejected_ids)]
    if (baseline_bundle is None) != (baseline_manifest is None):
        raise ValueError("baseline evidence and indexed lineage must be paired")
    if baseline_bundle is not None:
        if (
            baseline_manifest.node_session_id != manifest.node_session_id
            or baseline_manifest.topology_digest != manifest.topology_digest
        ):
            raise ValueError("candidate/baseline indexed clocks or topology differ")
        baseline_windows = _indexed_windows(baseline_bundle, baseline_manifest)
        left = {r["source_id"]: r for r in manifest.to_dict()["sources"]}
        right = {r["source_id"]: r for r in baseline_manifest.to_dict()["sources"]}
        if any(left[i] != right[i] for i in left.keys() & right.keys()):
            raise ValueError("candidate/baseline shared source lineage conflicts")
        by_acquisition = {}
        for row in (*left.values(), *right.values()):
            value = {k: v for k, v in row.items() if k not in ("source_id", "source_digest")}
            if by_acquisition.setdefault(row["acquisition_digest"], value) != value:
                raise ValueError("candidate/baseline shared acquisition lineage conflicts")
        blocked.extend(
            baseline_windows[i]
            for i in (*baseline_bundle.accepted_ids, *baseline_bundle.rejected_ids)
        )
    accepted, exclusions = [], []
    for identity in sorted(bundle.holdout_ids, key=lambda i: (*windows[i][:2], i)):
        lower, upper, acquisitions = windows[identity]
        overlap = any(
            acquisitions & prior[2]
            or not (upper + separation_s <= prior[0] or lower >= prior[1] + separation_s)
            for prior in blocked
        )
        if overlap:
            exclusions.append((identity, "shared or correlated indexed acquisition window"))
        else:
            accepted.append(identity)
        blocked.append(windows[identity])
    return tuple(accepted), tuple(exclusions)
