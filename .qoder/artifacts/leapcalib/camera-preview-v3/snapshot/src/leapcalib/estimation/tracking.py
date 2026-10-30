# Copyright (c) Alibaba, Inc. and its affiliates.
"""Lazy target-free patch tracking and robust metric consensus, with no hardware APIs.

Persistent correspondence IDs support relative repeatability, not absolute geometry.
No static-target witness or carried-link identity is inferred by this computation.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from leapcalib.contracts.failures import (
    CalibrationFailure,
    FAILURE_INSUFFICIENT_EVIDENCE,
    FAILURE_WORKER_UNAVAILABLE,
)
from leapcalib.contracts.identity import StreamProfile, digest_of
from leapcalib.contracts.motion import PointCorrespondence
from leapcalib.contracts.tracking import MetricRoi, TRACK_SCHEMA, SceneConfig, TrackingConfig
from leapcalib.estimation.numeric import SolveConfig, numpy
from leapcalib.estimation.registration import register_points


def _insufficient(reason: str) -> None:
    raise CalibrationFailure(FAILURE_INSUFFICIENT_EVIDENCE, reason)


def _deproject(profile: StreamProfile, u: float, v: float, z: float) -> tuple[float, float, float]:
    x, y = (u - profile.ppx_px) / profile.fx_px, (v - profile.ppy_px) / profile.fy_px
    if any(profile.distortion_coeffs):
        if profile.distortion_model not in ("brown_conrady", "inverse_brown_conrady"):
            _insufficient("unsupported distortion; exact metric deprojection is unavailable")
        k = (*profile.distortion_coeffs, 0, 0, 0, 0, 0)[:5]
        ux, uy = x, y
        for _ in range(10):
            r2 = ux * ux + uy * uy
            radial = 1 + k[0] * r2 + k[1] * r2**2 + k[4] * r2**3
            if abs(radial) < 1e-12:
                _insufficient("singular distortion deprojection")
            qx, qy = (
                (ux * radial, uy * radial)
                if profile.distortion_model == "inverse_brown_conrady"
                else (ux, uy)
            )
            dx = 2 * k[2] * qx * qy + k[3] * (r2 + 2 * qx * qx)
            dy = 2 * k[3] * qx * qy + k[2] * (r2 + 2 * qy * qy)
            ux, uy = (x - dx) / radial, (y - dy) / radial
        x, y = ux, uy
    return (x * z, y * z, z)


def _arrays(roi: MetricRoi, cv2: Any, np: Any) -> tuple[Any, Any]:
    color = np.frombuffer(roi.color_bgr8, dtype=np.uint8).reshape(roi.height, roi.width, 3)
    depth = np.frombuffer(roi.depth_m_f32le, dtype="<f4").reshape(roi.height, roi.width)
    if not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError("metric depth ROI contains nonfinite/negative values")
    return cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), depth


def _descriptor(gray: Any, point: Any, cv2: Any, np: Any) -> Any:
    patch = (
        cv2.getRectSubPix(gray, (7, 7), (float(point[0]), float(point[1])))
        .astype(np.float32)
        .reshape(-1)
    )
    patch -= patch.mean()
    norm = float(np.linalg.norm(patch))
    return patch / norm if norm > 1 else np.zeros_like(patch)


def _spread(points: Any, roi: MetricRoi, config: TrackingConfig, np: Any) -> None:
    if len(points) < config.min_tracks:
        _insufficient("fewer than the declared minimum persistent metric tracks")
    extent = np.ptp(points, axis=0)
    if (
        extent[0] < roi.width * config.min_spread_fraction
        or extent[1] < roi.height * config.min_spread_fraction
    ):
        _insufficient("tracks are not distributed in both image dimensions")
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    if singular[-1] < config.min_distance_px:
        _insufficient("collinear image support cannot establish unambiguous yaw")


def _seed(
    gray: Any, roi: MetricRoi, config: TrackingConfig, cv2: Any, np: Any
) -> tuple[list[str], Any]:
    # Equal corner budgets over a grid prevent one textured corner owning the fit.
    selected = []
    for row in range(3):
        for column in range(4):
            mask = np.zeros(gray.shape, np.uint8)
            y0, y1 = row * roi.height // 3, (row + 1) * roi.height // 3
            x0, x1 = column * roi.width // 4, (column + 1) * roi.width // 4
            mask[max(4, y0) : min(roi.height - 4, y1), max(4, x0) : min(roi.width - 4, x1)] = 255
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=max(1, config.max_tracks // 12),
                qualityLevel=config.quality_level,
                minDistance=config.min_distance_px,
                mask=mask,
                blockSize=7,
            )
            if corners is not None:
                selected.extend(corners.reshape(-1, 2))
    if len(selected) < config.min_tracks:
        _insufficient("ordinary scene patch has insufficient distributed texture")
    points = np.asarray(selected[: config.max_tracks], dtype=np.float32)
    descriptors = np.asarray([_descriptor(gray, p, cv2, np) for p in points])
    correlation = descriptors @ descriptors.T
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    ambiguous = (
        (correlation > config.ambiguity_correlation_max) & (distances > config.min_distance_px * 2)
    ).any(axis=1)
    valid = (~ambiguous) & (np.linalg.norm(descriptors, axis=1) > 0.5)
    points = points[valid]
    _spread(points, roi, config, np)
    ids = [
        digest_of(
            {
                "frame": roi.frame_id,
                "roi": [roi.x, roi.y, roi.width, roi.height],
                "pixel": [float(p[0]), float(p[1])],
            }
        )
        for p in points
    ]
    return ids, points


def _continue(
    tracks: Sequence[Mapping[str, Any]], first: MetricRoi, config: TrackingConfig, np: Any
) -> tuple[list[str], Any]:
    if (
        not isinstance(tracks, (list, tuple))
        or not config.min_tracks <= len(tracks) <= config.max_tracks
    ):
        raise ValueError("bounded persistent track list required")
    ids, points = [], []
    for track in tracks:
        if (
            not isinstance(track, Mapping)
            or set(track) != {"feature_id", "frame_id", "pixel"}
            or track["frame_id"] != first.frame_id
        ):
            raise ValueError("persistent tracks must bind the exact preceding source frame")
        identity, pixel = track["feature_id"], track["pixel"]
        if (
            not isinstance(identity, str)
            or not identity
            or len(identity) > 256
            or not isinstance(pixel, (list, tuple))
            or len(pixel) != 2
        ):
            raise ValueError("invalid persistent feature identity/pixel")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in pixel
        ):
            raise ValueError("finite tracked pixels required")
        ids.append(identity)
        points.append([pixel[0] - first.x, pixel[1] - first.y])
    if len(set(ids)) != len(ids):
        raise ValueError("persistent feature IDs must be distinct")
    return ids, np.asarray(points, dtype=np.float32)


def _depth(
    depth: Any, point: Any, roi: MetricRoi, config: TrackingConfig, np: Any
) -> tuple[float, float] | None:
    u, v = int(round(float(point[0]))), int(round(float(point[1])))
    values = depth[
        max(0, v - 2) : min(roi.height, v + 3), max(0, u - 2) : min(roi.width, u + 3)
    ].reshape(-1)
    valid = values > max(0, roi.profile.valid_min_m)
    if roi.profile.valid_max_m:
        valid &= values <= roi.profile.valid_max_m
    values = values[valid]
    if len(values) < 13:
        return None
    z = float(np.median(values))
    mad = float(np.median(np.abs(values - z)))
    if mad > config.max_depth_mad_m:
        return None
    return z, max(config.depth_floor_m, 1.4826 * mad)


def track_metric(
    first: MetricRoi,
    second: MetricRoi,
    *,
    tracks: Sequence[Mapping[str, Any]] = (),
    config: TrackingConfig = TrackingConfig(),
) -> dict[str, Any]:
    """Relative first→second scene registration; loss/ambiguity never reacquires silently."""
    if first.profile != second.profile or (
        first.mount_digest,
        first.session_id,
        first.source_profile_digest,
        first.provenance,
    ) != (second.mount_digest, second.session_id, second.source_profile_digest, second.provenance):
        _insufficient("camera profile/mount/session/provenance changed")
    if (first.x, first.y, first.width, first.height) != (
        second.x,
        second.y,
        second.width,
        second.height,
    ):
        _insufficient("ROI domain changed between persistent frames")
    gap = second.stamp.seconds - first.stamp.seconds
    if first.frame_id == second.frame_id or not 0 < gap <= config.max_frame_gap_s:
        _insufficient("tracking requires increasing fresh frame receipts")
    if len(first.source_stamps) != len(second.source_stamps) or any(
        a.stream != b.stream
        or a.timestamp_domain != b.timestamp_domain
        or b.frame_number <= a.frame_number
        or b.timestamp_ms <= a.timestamp_ms
        for a, b in zip(first.source_stamps, second.source_stamps)
    ):
        _insufficient("native stream clock/sequence changed or repeated")
    try:
        import cv2
    except ImportError as exc:
        raise CalibrationFailure(
            FAILURE_WORKER_UNAVAILABLE, "metric patch tracking requires optional OpenCV"
        ) from exc
    np = numpy()
    a, za = _arrays(first, cv2, np)
    b, zb = _arrays(second, cv2, np)
    ids, p = _continue(tracks, first, config, np) if tracks else _seed(a, first, config, cv2, np)
    _spread(p, first, config, np)
    if ((p < 4).any(axis=1) | (p[:, 0] >= first.width - 4) | (p[:, 1] >= first.height - 4)).any():
        _insufficient("persistent patches left their bounded source ROI")
    q, forward, _ = cv2.calcOpticalFlowPyrLK(
        a, b, p.reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=3
    )
    if q is None:
        _insufficient("tracker lost all features")
    back, backward, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, winSize=(21, 21), maxLevel=3)
    if back is None:
        _insufficient("backward tracking lost all features")
    q, back = q.reshape(-1, 2), back.reshape(-1, 2)
    rows, correspondences, rejected, depth_uncertainties = [], [], [], []
    for index, identity in enumerate(ids):
        pa, pb = p[index], q[index]
        valid = (
            forward[index, 0]
            and backward[index, 0]
            and np.isfinite(pb).all()
            and float(np.linalg.norm(back[index] - pa)) <= config.fb_gate_px
            and 4 <= pb[0] < second.width - 4
            and 4 <= pb[1] < second.height - 4
        )
        if valid:
            valid = (
                float(_descriptor(a, pa, cv2, np) @ _descriptor(b, pb, cv2, np))
                >= config.patch_correlation_min
            )
        da, db = (
            (_depth(za, pa, first, config, np), _depth(zb, pb, second, config, np))
            if valid
            else (None, None)
        )
        if da is None or db is None:
            rejected.append(identity)
            continue
        pixel_a = (float(pa[0]) + first.x, float(pa[1]) + first.y)
        pixel_b = (float(pb[0]) + second.x, float(pb[1]) + second.y)
        xyz_a = _deproject(first.profile, *pixel_a, da[0])
        xyz_b = _deproject(second.profile, *pixel_b, db[0])
        correspondences.append(PointCorrespondence(identity, xyz_a, xyz_b))
        rows.append(
            {
                "feature_id": identity,
                "first_pixel": pixel_a,
                "second_pixel": pixel_b,
                "first_m": xyz_a,
                "second_m": xyz_b,
                "depth_uncertainty_m": max(da[1], db[1]),
            }
        )
        depth_uncertainties.append(max(da[1], db[1]))
    if len(rows) < config.min_tracks:
        _insufficient("insufficient forward/backward, patch-consistent, valid-depth tracks")
    registration = register_points(
        correspondences,
        config=SolveConfig(
            min_points=config.min_tracks,
            max_points=config.max_tracks,
            point_gate_m=config.point_gate_m,
            min_inlier_ratio=config.min_inlier_ratio,
        ),
    )
    accepted = set(registration.accepted_features)
    rows = [r for r in rows if r["feature_id"] in accepted]
    _spread(
        np.asarray(
            [[r["second_pixel"][0] - second.x, r["second_pixel"][1] - second.y] for r in rows]
        ),
        second,
        config,
        np,
    )
    rotation = registration.motion.rotation
    yaw = math.atan2(rotation[1][0], rotation[0][0])
    if abs(yaw) > config.max_yaw_rad:
        _insufficient("scene yaw exceeded the declared tracking branch")
    radius = max(
        float(
            np.linalg.norm(
                np.asarray(r["first_m"])[:2] - np.mean([v["first_m"] for v in rows], axis=0)[:2]
            )
        )
        for r in rows
    )
    floor = max(max(depth_uncertainties), registration.max_error_m)
    return {
        "schema": TRACK_SCHEMA,
        "first_frame_id": first.frame_id,
        "second_frame_id": second.frame_id,
        "first_digest": first.digest,
        "second_digest": second.digest,
        "profile_digest": first.profile.digest,
        "mount_digest": first.mount_digest,
        "session_id": first.session_id,
        "source_profile_digest": first.source_profile_digest,
        "provenance": first.provenance,
        "registration": registration.to_dict(),
        "tracks": rows,
        "persistent_tracks": [
            {"feature_id": r["feature_id"], "frame_id": second.frame_id, "pixel": r["second_pixel"]}
            for r in rows
        ],
        "rejected_features": sorted(set(rejected) | set(registration.rejected_features)),
        "delta_m_rad": [*registration.motion.translation_m, yaw],
        "uncertainty_m_rad": [floor, floor, floor, floor / max(radius, config.depth_floor_m)],
        "timing_uncertainty_s": first.stamp.uncertainty_s
        + second.stamp.uncertainty_s
        + first.exposure_uncertainty_s
        + second.exposure_uncertainty_s,
        "witness_valid": False,
        "rigid_link_identity_established": False,
        "hardware_accuracy_certified": False,
        "executed": False,
    }


def _scene_separate(first: Any, second: Any, minimum: float, np: Any) -> None:
    # Separate local LK windows. Shared image/pyramid/timing errors are still
    # common-mode; disjoint patches are not independent physical sessions.
    distance = np.max(np.abs(first[:, None, :] - second[None, :, :]), axis=2)
    if np.any(distance < minimum):
        _insufficient("task/background local tracking windows overlap or switched support")


def _scene_bounds(
    result: Mapping[str, Any], roi: MetricRoi, config: TrackingConfig, np: Any
) -> tuple[float, float, Any]:
    rows = result["tracks"]
    first = np.asarray([r["first_m"] for r in rows])
    second = np.asarray([r["second_m"] for r in rows])
    spread = min(
        float(np.linalg.svd(points - points.mean(axis=0), compute_uv=False)[1])
        / math.sqrt(len(points))
        for points in (first, second)
    )
    point_floor = max(
        r["depth_uncertainty_m"] * (1 + (abs(p[0]) + abs(p[1])) / p[2])
        + math.sqrt(2) * config.fb_gate_px * p[2] / min(roi.profile.fx_px, roi.profile.fy_px)
        for r in rows
        for p in (r["first_m"], r["second_m"])
    )
    floor = 2 * point_floor + result["registration"]["max_error_m"]
    # In-plane support constrains full rigid orientation, not just favorable yaw.
    angle = 2 * math.asin(min(1.0, floor / max(spread, 1e-12)))
    return floor, angle, first.mean(axis=0)


def track_scene(
    first: MetricRoi,
    second: MetricRoi,
    *,
    task_tracks: Sequence[Mapping[str, Any]],
    background_tracks: Sequence[Mapping[str, Any]],
    config: TrackingConfig = TrackingConfig(),
    scene_config: SceneConfig = SceneConfig(),
) -> dict[str, Any]:
    """Compare separately tracked scene supports without asserting either is static.

    Frozen task/background supports are never seeded or silently reacquired here.
    Common camera motion cancels at the measured task centroid. Relative movement
    is distinct from mount drift; an ambiguous/noisy result never counts as valid.
    A consistent result alone cannot authenticate a physical background, carried
    link, global frame, exposure-rate bound, or clearance and grants no authority.
    """
    if not isinstance(config, TrackingConfig) or not isinstance(scene_config, SceneConfig):
        raise ValueError("typed frozen tracking and scene policies required")
    np = numpy()
    task_ids, task_pixels = _continue(task_tracks, first, config, np)
    background_ids, background_pixels = _continue(background_tracks, first, config, np)
    if (
        set(task_ids) & set(background_ids)
        or len(task_ids) + len(background_ids) > config.max_tracks
    ):
        raise ValueError(
            "task/background supports must be disjoint within one total feature budget"
        )
    if first.profile.distortion_model != "none":
        _insufficient("scene support uncertainty requires an undistorted exact profile")
    _scene_separate(task_pixels, background_pixels, scene_config.min_separation_px, np)
    task = track_metric(first, second, tracks=task_tracks, config=config)
    background = track_metric(first, second, tracks=background_tracks, config=config)
    comparison = compare_scene_supports(
        first,
        task,
        background,
        task_tracks=task_tracks,
        background_tracks=background_tracks,
        config=config,
        scene_config=scene_config,
    )
    return {
        "schema": "leapcalib.scene_consistency.v1",
        "task_tracking": task,
        "background_tracking": background,
        "comparison": comparison,
    }


def compare_scene_supports(
    first: MetricRoi,
    task: Mapping[str, Any],
    background: Mapping[str, Any],
    *,
    task_tracks: Sequence[Mapping[str, Any]],
    background_tracks: Sequence[Mapping[str, Any]],
    config: TrackingConfig,
    scene_config: SceneConfig,
) -> dict[str, Any]:
    """Recompute numerical comparison from checked metric support, never authenticate it.

    Owners must first check both tracking results against indexed source captures.
    This pure calculation deliberately does not establish patch association,
    exposure-time rates, a physically static background, clearance or authority.
    """
    if (
        not isinstance(first, MetricRoi)
        or not isinstance(config, TrackingConfig)
        or not isinstance(scene_config, SceneConfig)
    ):
        raise ValueError("typed metric ROI and frozen comparison policies required")
    np = numpy()
    task_ids, task_pixels = _continue(task_tracks, first, config, np)
    background_ids, background_pixels = _continue(background_tracks, first, config, np)
    if (
        set(task_ids) & set(background_ids)
        or len(task_ids) + len(background_ids) > config.max_tracks
    ):
        raise ValueError(
            "task/background supports must be disjoint within one total feature budget"
        )
    if first.profile.distortion_model != "none":
        _insufficient("scene support uncertainty requires an undistorted exact profile")
    _scene_separate(task_pixels, background_pixels, scene_config.min_separation_px, np)
    if task["rejected_features"] or background["rejected_features"]:
        _insufficient("task/background frozen support lost, occluded or failed rigid consensus")
    if {row["feature_id"] for row in task["tracks"]} != set(task_ids) or {
        row["feature_id"] for row in background["tracks"]
    } != set(background_ids):
        raise ValueError("scene comparison cannot switch or drop frozen support")
    _scene_separate(
        np.asarray([r["second_pixel"] for r in task["tracks"]]),
        np.asarray([r["second_pixel"] for r in background["tracks"]]),
        scene_config.min_separation_px,
        np,
    )
    from leapcalib.geometry.se3 import Transform

    task_motion = Transform.from_mapping(task["registration"]["motion"])
    background_motion = Transform.from_mapping(background["registration"]["motion"])
    task_floor, task_angle, centre = _scene_bounds(task, first, config, np)
    background_floor, background_angle, background_centre = _scene_bounds(
        background, first, config, np
    )
    distance = math.dist(task_motion.apply(centre), background_motion.apply(centre))
    rotation = task_motion.angle_to_rad(background_motion)
    # Shared depth/pixel/timing floors are not averaged down by feature count.
    floor = (
        task_floor
        + background_floor
        + 2 * math.sin(background_angle / 2) * float(np.linalg.norm(centre - background_centre))
    )
    angle_floor = min(math.pi, task_angle + background_angle)
    errors, uncertainty = (distance, rotation), (floor, angle_floor)
    limits = (scene_config.max_translation_m, scene_config.max_rotation_rad)
    confidence = scene_config.confidence_multiplier
    verdict = "ambiguous"
    if all(e + confidence * u <= limit for e, u, limit in zip(errors, uncertainty, limits)):
        verdict = "consistent"
    elif any(e - confidence * u > limit for e, u, limit in zip(errors, uncertainty, limits)):
        verdict = "relative_motion"
    return {
        "verdict": verdict,
        "error_m_rad": list(errors),
        "uncertainty_m_rad": list(uncertainty),
        "policy": vars(scene_config),
        "task_tracking_digest": digest_of(task),
        "background_tracking_digest": digest_of(background),
        "support_digest": digest_of(
            {"task": list(task_tracks), "background": list(background_tracks)}
        ),
        "provenance": first.provenance,
        "coordinates": "relative_at_task_centroid;full_SO3_rad",
        "independent_static_background": "not_established",
        "interval_peak_rates": "not_established",
        "physical_admission": "not_established",
        "witness_valid": False,
        "executed": False,
    }
