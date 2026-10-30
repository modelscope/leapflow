# Copyright (c) Alibaba, Inc. and its affiliates.
"""Indexed disjoint scene geometry checks, not physical static-witness admission.

Raw depth, feedback and acquisition signatures are reopened through the owner.
Optical replay uses only those indexed images through the owner compute worker.
Interval peak rates, static-background setup and clearance remain separate gates.
Neither a verdict nor construction grants motion, response qualification,
global geometry or activation authority.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from leapcalib.contracts.identity import digest_of
from leapcalib.contracts.motion import finite_vector
from leapcalib.contracts.observations import JointObservation, RgbdObservation
from leapcalib.contracts.raw_task import identities
from leapcalib.contracts.tracking import SceneConfig, TrackingConfig
from leapcalib.estimation.tracking import compare_scene_supports
from leapcalib.validation.bundle import EvidenceBundle
from leapflow.hardware.calibration.acquisition import SettledObservation
from leapflow.hardware.calibration.factors import _CaptureSnapshot
from leapflow.hardware.calibration.generations import _json
from leapflow.hardware.calibration.provenance import (
    IndexedCaptureReader,
    IndexedSourceManifest,
    indexed_source_manifest,
)
from leapflow.hardware.calibration.reference import (
    _tracking_domain,
    validate_metric_tracking_result,
)
from leapflow.hardware.calibration.worker import ComputeWorker
from leapflow.hardware.skills.protocol import FAILURE_CHECK_FAILED, SkillError

if TYPE_CHECKING:
    from leapflow.hardware.calibration.evidence import CalibrationEvidenceStore


@dataclass(frozen=True)
class IndexedSceneCheck:
    """Immutable redacted geometry result; a physical witness is never implied."""

    manifest: IndexedSourceManifest
    frame_ids: tuple[str, str]
    tracking: TrackingConfig
    comparison_json: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_ids", tuple(self.frame_ids))
        identities(self.frame_ids, maximum=2)
        if (
            len(self.frame_ids) != 2
            or not isinstance(self.manifest, IndexedSourceManifest)
            or not isinstance(self.tracking, TrackingConfig)
        ):
            raise ValueError("typed indexed scene sources, frames and tracking policy required")
        if not isinstance(self.comparison_json, str) or len(self.comparison_json.encode()) > 16384:
            raise ValueError("bounded immutable scene comparison required")
        value = json.loads(self.comparison_json)
        if not isinstance(value, dict) or value.get("verdict") not in (
            "consistent",
            "relative_motion",
            "ambiguous",
        ):
            raise ValueError("typed scene comparison verdict required")
        if any(
            value.get(key) != "not_established"
            for key in (
                "physical_admission",
                "independent_static_background",
                "interval_peak_rates",
            )
        ) or any(value.get(key) is not False for key in ("witness_valid", "executed")):
            raise ValueError("indexed scene computation cannot promote physical authority")
        sources = {row["source_id"] for row in self.manifest.to_dict()["sources"]}
        if not set(self.frame_ids) <= sources:
            raise ValueError("scene frames lack indexed acquisition lineage")
        digest_of(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "leapflow.indexed_scene_check.v1",
            "source_manifest": self.manifest.to_dict(),
            "frame_ids": list(self.frame_ids),
            "tracking_policy": vars(self.tracking),
            "comparison": json.loads(self.comparison_json),
            "scope": "disjoint_indexed_metric_geometry",
            "feature_association": "not_established",
            "independent_static_background": "not_established",
            "interval_peak_rates": "not_established",
            "clearance": "not_established",
            "physical_admission": "not_established",
            "qualified_consumers": [],
            "witness_valid": False,
            "executed": False,
        }

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())


def _support_ids(tracks: Sequence[Mapping[str, Any]], frame_id: str, limit: int) -> tuple[str, ...]:
    for row in tracks:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"feature_id", "frame_id", "pixel"}
            or row["frame_id"] != frame_id
        ):
            raise ValueError("scene support must bind the exact preceding frame and pixel")
        finite_vector(row["pixel"], 2)
    return identities(tuple(row["feature_id"] for row in tracks), maximum=limit)


@dataclass(frozen=True)
class IndexedSceneReplay:
    """Owner-produced optical tracking; no physical surface identity or authority."""

    geometry: IndexedSceneCheck
    run_id: str
    input_digest: str
    result_json: str
    _result_digest: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for identity in (self.run_id, self.input_digest):
            identities((identity,))
        if not isinstance(self.geometry, IndexedSceneCheck):
            raise ValueError("typed indexed scene geometry required")
        if (
            not isinstance(self.result_json, str)
            or len(self.result_json.encode()) > 2 * 1024 * 1024
        ):
            raise ValueError("bounded immutable optical replay required")
        result = json.loads(self.result_json)
        if (
            not isinstance(result, dict)
            or set(result) != {"schema", "task_tracking", "background_tracking", "comparison"}
            or result["schema"] != "leapcalib.scene_consistency.v1"
            or result["comparison"] != json.loads(self.geometry.comparison_json)
        ):
            raise ValueError("optical replay differs from owner-checked scene geometry")
        for group in ("task_tracking", "background_tracking"):
            row = result[group]
            if (
                not isinstance(row, dict)
                or (row.get("first_frame_id"), row.get("second_frame_id"))
                != self.geometry.frame_ids
                or digest_of(row) != result["comparison"].get(group + "_digest")
                or any(
                    row.get(key) is not False
                    for key in (
                        "witness_valid",
                        "rigid_link_identity_established",
                        "hardware_accuracy_certified",
                        "executed",
                    )
                )
            ):
                raise ValueError("optical replay has foreign sources or physical authority")
        object.__setattr__(self, "_result_digest", digest_of(result))

    def tracking_result(self) -> dict[str, Any]:
        """Decode an immutable node-internal copy off-loop for subsequent tracking."""
        return json.loads(self.result_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "leapflow.indexed_scene_replay.v1",
            "geometry": self.geometry.to_dict(),
            "run_id": self.run_id,
            "input_digest": self.input_digest,
            "result_digest": self._result_digest,
            "feature_association": "owner_worker_optical_flow",
            "physical_surface_identity": "not_established",
            "independent_static_background": "not_established",
            "interval_peak_rates": "not_established",
            "clearance": "not_established",
            "physical_admission": "not_established",
            "qualified_consumers": [],
            "witness_valid": False,
            "executed": False,
        }

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())


def prepare_indexed_scene(
    store: IndexedCaptureReader,
    bundle: EvidenceBundle,
    acquisition_ids: Mapping[str, str],
    frame_ids: Sequence[str],
    *,
    task_tracks: Sequence[Mapping[str, Any]],
    background_tracks: Sequence[Mapping[str, Any]],
    tracking: TrackingConfig,
    scene_config: SceneConfig,
    trusted_keys: Mapping[str, Ed25519PublicKey],
    node_id: str,
) -> tuple[_CaptureSnapshot, IndexedSourceManifest, tuple[SettledObservation, SettledObservation]]:
    """Validate all primitive lineage before any optical computation, off the owner loop."""
    if (
        not isinstance(store, IndexedCaptureReader)
        or not isinstance(bundle, EvidenceBundle)
        or not isinstance(tracking, TrackingConfig)
        or not isinstance(scene_config, SceneConfig)
    ):
        raise ValueError("typed scene reader, raw evidence and frozen policies required")
    if any(not isinstance(value, Mapping) for value in (acquisition_ids, trusted_keys)):
        raise ValueError("typed indexed scene source/key maps required")
    identities(frame_ids, maximum=2)
    if (
        len(frame_ids) != 2
        or len(bundle.observations) > 258
        or any(not isinstance(o, (RgbdObservation, JointObservation)) for o in bundle.observations)
    ):
        raise ValueError("scene check requires only two bounded complete primitive capture windows")
    identities(tuple(acquisition_ids), maximum=258)
    for identity in acquisition_ids.values():
        identities((identity,))
    frames = {o.observation_id for o in bundle.observations if isinstance(o, RgbdObservation)}
    if frames != set(frame_ids) or len(set(acquisition_ids.values())) != 2:
        raise ValueError("scene frame selection differs from the two indexed capture windows")
    if (
        not isinstance(task_tracks, (list, tuple))
        or not isinstance(background_tracks, (list, tuple))
        or not tracking.min_tracks <= len(task_tracks)
        or not tracking.min_tracks <= len(background_tracks)
        or len(task_tracks) + len(background_tracks) > tracking.max_tracks
    ):
        raise ValueError("two frozen scene groups must share one bounded feature budget")
    task_ids = _support_ids(task_tracks, frame_ids[0], tracking.max_tracks)
    background_ids = _support_ids(background_tracks, frame_ids[0], tracking.max_tracks)
    if set(task_ids) & set(background_ids):
        raise ValueError("scene support groups cannot share feature identities")
    snapshot = _CaptureSnapshot(store)
    manifest = indexed_source_manifest(
        snapshot, bundle, acquisition_ids, trusted_keys=trusted_keys, node_id=node_id
    )
    first, second = (
        snapshot.captures[(acquisition_ids[i], bundle.topology.workcell_id)] for i in frame_ids
    )
    required = {
        s for c in (first, second) for s in (c.roi.frame_id, *(f.source_id for f in c.feedback))
    }
    if set(acquisition_ids) != required:
        raise ValueError("scene check must bind every feedback sample in both complete windows")
    _tracking_domain(first, second, tracking)
    return snapshot, manifest, (first, second)


def check_indexed_scene(
    store: IndexedCaptureReader,
    bundle: EvidenceBundle,
    acquisition_ids: Mapping[str, str],
    frame_ids: Sequence[str],
    result: Mapping[str, Any],
    *,
    task_tracks: Sequence[Mapping[str, Any]],
    background_tracks: Sequence[Mapping[str, Any]],
    tracking: TrackingConfig,
    scene_config: SceneConfig,
    trusted_keys: Mapping[str, Ed25519PublicKey],
    node_id: str,
) -> IndexedSceneCheck:
    """Reopen exactly two complete raw windows and check both frozen supports off-loop.

    Numerical comparison is recomputed from owner-checked raw depth/geometry; the
    worker's verdict, uncertainty, digest and authority flags are never trusted.
    The source pixels remain unauthenticated optical association, explicitly
    separate from this bounded geometry gate. No numerical tracker is run on
    the owner loop and no historical permit is consumed, renewed or reissued.
    """
    if (
        not isinstance(result, Mapping)
        or set(result) != {"schema", "task_tracking", "background_tracking", "comparison"}
        or result["schema"] != "leapcalib.scene_consistency.v1"
    ):
        raise ValueError("exact scene worker result schema required")
    _, manifest, captures = prepare_indexed_scene(
        store,
        bundle,
        acquisition_ids,
        frame_ids,
        task_tracks=task_tracks,
        background_tracks=background_tracks,
        tracking=tracking,
        scene_config=scene_config,
        trusted_keys=trusted_keys,
        node_id=node_id,
    )
    groups = (("task_tracking", task_tracks), ("background_tracking", background_tracks))
    support_ids = {
        key: _support_ids(rows, frame_ids[0], tracking.max_tracks) for key, rows in groups
    }
    for key, previous in groups:
        validate_metric_tracking_result(
            result[key], *captures, tracking, set(support_ids[key]), previous_tracks=previous
        )
    comparison = compare_scene_supports(
        captures[0].roi,
        result["task_tracking"],
        result["background_tracking"],
        task_tracks=task_tracks,
        background_tracks=background_tracks,
        config=tracking,
        scene_config=scene_config,
    )
    if not isinstance(result["comparison"], Mapping) or digest_of(
        result["comparison"]
    ) != digest_of(comparison):
        raise ValueError(
            "scene comparison differs from owner-recomputed indexed geometry and policy"
        )
    return IndexedSceneCheck(manifest, tuple(frame_ids), tracking, _json(comparison))


async def replay_indexed_scene(
    store: CalibrationEvidenceStore,
    bundle: EvidenceBundle,
    acquisition_ids: Mapping[str, str],
    frame_ids: Sequence[str],
    *,
    task_tracks: Sequence[Mapping[str, Any]],
    background_tracks: Sequence[Mapping[str, Any]],
    tracking: TrackingConfig,
    scene_config: SceneConfig,
    worker: ComputeWorker,
    run_id: str,
    trusted_keys: Mapping[str, Ed25519PublicKey],
    node_id: str,
) -> IndexedSceneReplay:
    """Compute optical association using private indexed bytes, never caller results.

    Freeze all input before suspension. The existing store tracks this workflow
    and drains its off-loop reads, while cancellation reaches the supervised
    worker exactly once. Neither output nor construction grants motion authority.
    """
    if store._closing:
        raise SkillError(FAILURE_CHECK_FAILED, "acquisition store is shutting down")
    if (
        not isinstance(bundle, EvidenceBundle)
        or not isinstance(worker, ComputeWorker)
        or not isinstance(tracking, TrackingConfig)
        or not isinstance(scene_config, SceneConfig)
        or any(not isinstance(m, Mapping) for m in (acquisition_ids, trusted_keys))
    ):
        raise ValueError("typed indexed scene evidence, policies and owner compute worker required")
    for identity in (run_id, node_id):
        identities((identity,))
    frozen = EvidenceBundle.from_mapping(json.loads(_json(bundle.to_dict())))
    data = json.loads(
        _json(
            {
                "sources": acquisition_ids,
                "frame_ids": frame_ids,
                "task_tracks": task_tracks,
                "background_tracks": background_tracks,
            }
        )
    )
    keys = dict(trusted_keys)
    options = {
        "task_tracks": data["task_tracks"],
        "background_tracks": data["background_tracks"],
        "tracking": tracking,
        "scene_config": scene_config,
        "trusted_keys": keys,
        "node_id": node_id,
    }

    def prepare() -> tuple[_CaptureSnapshot, dict[str, Any], str]:
        snapshot, _, captures = prepare_indexed_scene(
            store, frozen, data["sources"], data["frame_ids"], **options
        )
        payload = {
            "first": captures[0].roi.to_dict(),
            "second": captures[1].roi.to_dict(),
            "task_tracks": data["task_tracks"],
            "background_tracks": data["background_tracks"],
            "config": vars(tracking),
            "scene_config": vars(scene_config),
        }
        return snapshot, payload, digest_of(payload)

    async def operation() -> IndexedSceneReplay:
        snapshot, payload, input_digest = await store._validated_read(
            prepare, name="calibration-indexed-scene-preparation"
        )
        if store._closing:
            raise SkillError(FAILURE_CHECK_FAILED, "acquisition store is shutting down")
        computed = await worker.submit(
            "track_scene",
            payload,
            workcell_id=frozen.topology.workcell_id,
            run_id=run_id,
            model_digest=frozen.topology.model.digest,
        )
        if store._closing or asyncio.current_task().cancelling():
            raise asyncio.CancelledError

        def finish() -> IndexedSceneReplay:
            if not isinstance(computed, Mapping) or set(computed) != {"scene"}:
                raise ValueError("exact scene worker response required")
            result_json = _json(computed["scene"])
            checked = check_indexed_scene(
                snapshot,
                frozen,
                data["sources"],
                data["frame_ids"],
                json.loads(result_json),
                **options,
            )
            return IndexedSceneReplay(checked, run_id, input_digest, result_json)

        return await store._validated_read(finish, name="calibration-indexed-optical-validation")

    return await store._tracked_compute(operation)
