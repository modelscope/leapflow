# Copyright (c) Alibaba, Inc. and its affiliates.
"""Rigid cuboid and inner-jaw geometry from metric surfaces and RGB landmarks.

Resting-face alternatives are returned verbatim, never silently selected. RGB
seeds supply orientation hypotheses; the measured surface constrains range.
Missing object depth is not replaced with table or background pixels.
"""
from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

from leapflow.hardware.rgbd import RgbdFrame
from leapflow.hardware.rgbd_quality import DepthEvidence, _numpy


@dataclass(frozen=True)
class CuboidSeed:
    hypothesis_id: str
    dimensions_mm: tuple[float, float, float]
    center_camera_mm: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]


@dataclass(frozen=True)
class CuboidFit:
    hypothesis_id: str
    center_camera_mm: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]
    surface_rms_mm: float
    contour_rms_px: float
    observable: bool
    depth_qualified: bool
    frame_ids: tuple[str, ...]


def _rotation(np: Any, angles: Any) -> Any:
    x, y, z = np.radians(angles)
    cx, cy, cz, sx, sy, sz = np.cos(x), np.cos(y), np.cos(z), np.sin(x), np.sin(y), np.sin(z)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def fit_cuboid_surfaces(frame: RgbdFrame, mask: Any, contour_px: Sequence[Sequence[float]],
                       seeds: Sequence[CuboidSeed], *, evidence: DepthEvidence,
                       surface_sigma_mm: float, contour_sigma_px: float,
                       max_iterations: int = 20, max_points: int = 512) -> tuple[CuboidFit, ...]:
    """Jointly refine bounded RGB pose seeds using measured visible surfaces.

    Rank of the combined residual Jacobian is reported, not assumed. The result
    is a proposal, not a clearance or calibration certificate.
    """
    np = _numpy()
    if (not all(math.isfinite(v) and v > 0 for v in (surface_sigma_mm, contour_sigma_px))
            or not 1 <= max_iterations <= 100 or not 8 <= max_points <= 4096):
        raise ValueError("invalid cuboid-fit budget or uncertainty")
    if not evidence.qualified or not seeds:
        return ()
    image_mask = np.asarray(mask, dtype=bool)
    if image_mask.shape != (frame.height, frame.width):
        raise ValueError("object mask must match the metric frame")
    source_id = f"{frame.metadata.get('session_id', '')}:{frame.metadata.get('color_source', {}).get('frame_number')}:{frame.metadata.get('depth_source', {}).get('frame_number')}"
    if (source_id not in evidence.frame_ids
            or evidence.profile_digest != frame.metadata.get("profile_digest")
            or evidence.calibration_digest != frame.metadata.get("calibration_digest")
            or evidence.mount_digest != frame.metadata.get("mount_digest")
            or evidence.mask_digest != hashlib.sha256(image_mask.tobytes()).hexdigest()):
        raise ValueError("cuboid depth evidence does not identify this frame and mask")
    if len(seeds) > 12:
        raise ValueError("resting-face hypothesis budget exceeded")
    depth = np.asarray(frame.depth_units()).reshape(image_mask.shape) * frame.depth_scale_m * 1000.0
    v, u = np.nonzero(image_mask & (depth > 0))
    if len(v) < 8:
        return ()
    step = max(1, math.ceil(len(v) / max_points))
    points = np.array([frame.intrinsics.deproject(float(x), float(y), float(z))
                       for x, y, z in zip(u[::step], v[::step], depth[v[::step], u[::step]])])
    contour = np.asarray(contour_px, dtype=float)
    if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 4 or not np.isfinite(contour).all():
        raise ValueError("independent RGB contour required")
    contour = contour[::max(1, math.ceil(len(contour) / 256))]
    contour_count = len(contour) + 64
    fits = []
    for seed in seeds:
        dimensions = np.asarray(seed.dimensions_mm, dtype=float)
        params = np.array((*seed.center_camera_mm, *seed.rpy_deg), dtype=float)
        if not np.isfinite(params).all() or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError("finite cuboid seed and positive independent dimension prior required")
        half = dimensions / 2
        vertices = np.array([[x, y, z] for x in (-half[0], half[0])
                             for y in (-half[1], half[1]) for z in (-half[2], half[2])])

        def residual(p: Any) -> Any:
            rotation = _rotation(np, p[3:])
            local = (points - p[:3]) @ rotation
            delta = np.abs(local) - half
            # Signed distance to the finite box, not its infinite face planes.
            surface = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(np.max(delta, axis=1), 0)
            projected = vertices @ rotation.T + p[:3]
            if np.any(projected[:, 2] <= 0):
                return np.full(len(points) + contour_count, 1e6)
            pixels = np.array([frame.intrinsics.project(point) for point in projected])
            hull = _convex_hull(np, pixels)
            edge_error = _distance_to_edges(np, contour, hull)
            silhouette = _perimeter_samples(np, hull, 64)
            reverse_error = np.sqrt(np.min(np.sum((silhouette[:, None] - contour[None, :]) ** 2, axis=2), axis=1))
            return np.concatenate((surface / surface_sigma_mm,
                                   edge_error / contour_sigma_px, reverse_error / contour_sigma_px))

        jacobian = np.zeros((len(points) + contour_count, 6))
        for _ in range(max_iterations):
            base = residual(params)
            for column in range(6):
                shifted = params.copy()
                shifted[column] += 0.01
                jacobian[:, column] = (residual(shifted) - base) / 0.01
            change = np.linalg.solve(jacobian.T @ jacobian + np.eye(6) * 1e-3, -jacobian.T @ base)
            change[:3] = np.clip(change[:3], -5, 5)
            change[3:] = np.clip(change[3:], -2, 2)
            proposal = params + change
            if np.linalg.norm(residual(proposal)) >= np.linalg.norm(base):
                break
            params = proposal
            if np.linalg.norm(change) < 1e-4:
                break
        result = residual(params)
        fits.append(CuboidFit(seed.hypothesis_id, tuple(float(v) for v in params[:3]),
                              tuple(float(v) for v in params[3:]),
                              float(np.sqrt(np.mean(result[:-contour_count] ** 2)) * surface_sigma_mm),
                              float(np.sqrt(np.mean(result[-contour_count:] ** 2)) * contour_sigma_px),
                              bool(np.linalg.matrix_rank(jacobian) == 6),
                              evidence.qualified, evidence.frame_ids))
    return tuple(fits)


def _convex_hull(np: Any, pixels: Any) -> Any:
    ordered = sorted(set(tuple(p) for p in pixels))
    def cross(a: Any, b: Any, c: Any) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    halves = []
    for points in (ordered, ordered[::-1]):
        half = []
        for point in points:
            while len(half) >= 2 and cross(half[-2], half[-1], point) <= 0:
                half.pop()
            half.append(point)
        halves.append(half[:-1])
    return np.asarray(halves[0] + halves[1])


def _distance_to_edges(np: Any, points: Any, hull: Any) -> Any:
    edges = np.roll(hull, -1, axis=0) - hull
    offset = points[:, None] - hull[None, :]
    fractions = np.clip(np.sum(offset * edges, axis=2) / np.maximum(np.sum(edges ** 2, axis=1), 1e-12), 0, 1)
    return np.sqrt(np.min(np.sum((offset - fractions[:, :, None] * edges) ** 2, axis=2), axis=1))


def _perimeter_samples(np: Any, hull: Any, count: int) -> Any:
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.linalg.norm(edges, axis=1)
    starts = np.concatenate(([0.0], np.cumsum(lengths)))
    locations = np.arange(count) * starts[-1] / count
    index = np.minimum(np.searchsorted(starts[1:], locations, side="right"), len(hull) - 1)
    fractions = (locations - starts[index]) / np.maximum(lengths[index], 1e-12)
    return hull[index] + edges[index] * fractions[:, None]


def inner_jaw_points(frame: RgbdFrame, landmark_pixels: Sequence[Sequence[float]], *,
                     independent: bool, radius_px: int, min_samples: int,
                     support_masks: Sequence[Any] | None = None,
                     max_dispersion_mm: float = 3.0) -> tuple[tuple[float, float, float], ...] | None:
    """Deproject both independently detected pad landmarks; no tool-only fallback."""
    np = _numpy()
    if not independent or len(landmark_pixels) != 2 or support_masks is None or len(support_masks) != 2:
        return None
    if not 0 <= radius_px <= 20 or min_samples < 1 or not math.isfinite(max_dispersion_mm) or max_dispersion_mm <= 0:
        raise ValueError("invalid landmark support budget")
    points = []
    for pixel, mask in zip(landmark_pixels, support_masks):
        if len(pixel) != 2 or not all(math.isfinite(v) for v in pixel):
            return None
        u, v = (int(round(value)) for value in pixel)
        if not 0 <= u < frame.width or not 0 <= v < frame.height:
            return None
        support = np.asarray(mask, dtype=bool)
        if support.shape != (frame.height, frame.width) or not support[v, u] or frame.depth_mm(u, v) <= 0:
            return None
        center_depth = frame.depth_mm(u, v)
        samples = [frame.depth_mm(x, y)
                   for y in range(max(0, v - radius_px), min(frame.height, v + radius_px + 1))
                   for x in range(max(0, u - radius_px), min(frame.width, u + radius_px + 1))
                   if support[y, x] and frame.depth_mm(x, y) > 0]
        if len(samples) < min_samples or any(abs(value - center_depth) > max_dispersion_mm for value in samples):
            return None
        points.append(frame.intrinsics.deproject(float(u), float(v), float(np.median(samples))))
    return tuple(tuple(float(v) for v in point) for point in points)


def inner_jaw_midpoint(frame: RgbdFrame, landmark_pixels: Sequence[Sequence[float]], **kwargs: Any) -> tuple[float, float, float] | None:
    """Compatibility surface: the mean of the same two checked pad observations."""
    points = inner_jaw_points(frame, landmark_pixels, **kwargs)
    return tuple((a + b) / 2 for a, b in zip(*points)) if points is not None else None
