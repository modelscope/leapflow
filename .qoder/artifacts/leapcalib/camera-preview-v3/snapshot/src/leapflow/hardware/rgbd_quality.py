# Copyright (c) Alibaba, Inc. and its affiliates.
"""Node-local metric depth qualification. No preview colors or imputed depth.

Masks describe a supported visible face, not the whole hidden cuboid. A missing
measurement uncertainty or mount identity prevents physical qualification even
when image residuals are small. Only bounded scalar evidence crosses HCP.
"""

from __future__ import annotations

import copy
import json
import math
import hashlib
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from leapflow.hardware.rgbd import RgbdFrame
from leapflow.hardware.transport import TransportError


@dataclass(frozen=True)
class DepthLimits:
    min_samples: int
    min_coverage: float
    max_temporal_p95_mm: float
    max_surface_rms_mm: float
    min_background_separation_mm: float
    max_sync_skew_ms: float
    max_age_s: float
    min_frames: int
    erosion_px: int

    def __post_init__(self) -> None:
        positive = (
            self.max_temporal_p95_mm,
            self.max_surface_rms_mm,
            self.min_background_separation_mm,
            self.max_sync_skew_ms,
            self.max_age_s,
        )
        if (
            any(type(v) is not int for v in (self.min_samples, self.min_frames, self.erosion_px))
            or self.min_samples < 3
            or not 2 <= self.min_frames <= 50
            or not 0 <= self.erosion_px <= 20
            or not 0 < self.min_coverage <= 1
            or any(not math.isfinite(v) or v <= 0 for v in positive)
        ):
            raise ValueError("invalid depth qualification limits")


@dataclass(frozen=True)
class DepthEvidence:
    qualified: bool
    reasons: tuple[str, ...]
    native_valid: int
    interior_valid: int
    interior_coverage: float
    distinct_frames: int
    median_range_mm: float | None
    temporal_p95_mm: float | None
    surface_rms_mm: float | None
    background_separation_mm: float | None
    uncertainty_mm: float | None
    frame_ids: tuple[str, ...]
    calibration_digest: str
    mount_digest: str
    mask_digest: str = ""
    profile_digest: str = ""
    observation_times: tuple[float, ...] = ()
    configuration_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class RgbdProcessor(Protocol):
    """Injected node-local processor; output must contain bounded measurements."""

    def measure(self, frame: RgbdFrame) -> DepthEvidence: ...


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as error:
        raise TransportError(
            "metric depth requires numpy on the node", failure_code="depth_processor_unavailable"
        ) from error
    return numpy


def _frame_identity(frame: RgbdFrame, limits: DepthLimits, *, now: float) -> tuple[str, list[str]]:
    metadata, reasons = frame.metadata, []
    color, depth = metadata.get("color_source", {}), metadata.get("depth_source", {})
    numbers = (color.get("frame_number"), depth.get("frame_number"))
    stamps = (color.get("timestamp_ms"), depth.get("timestamp_ms"))
    domains = (color.get("timestamp_domain"), depth.get("timestamp_domain"))
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in numbers):
        reasons.append("source_frame_missing")
    if (
        any(
            isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t)
            for t in stamps
        )
        or not domains[0]
        or domains[0] != domains[1]
    ):
        reasons.append("source_clock_unqualified")
    elif abs(stamps[0] - stamps[1]) > limits.max_sync_skew_ms:
        reasons.append("rgb_depth_skew")
    if metadata.get("aligned_to_color") is not True:
        reasons.append("depth_unaligned")
    if not 0 <= now - frame.monotonic_at <= limits.max_age_s:
        reasons.append("frame_stale")
    for field in ("session_id", "profile_digest", "calibration_digest", "mount_digest"):
        if not metadata.get(field):
            reasons.append(field + "_missing")
    return f"{metadata.get('session_id', '')}:{numbers[0]}:{numbers[1]}", reasons


def qualify_depth(
    frames: Sequence[RgbdFrame],
    mask: Any,
    background_mask: Any,
    *,
    limits: DepthLimits,
    now: float,
    calibration_digest: str,
    mount_digest: str,
    uncertainty_mm: float | None,
) -> DepthEvidence:
    """Measure original/eroded support, per-pixel temporal dispersion and plane fit.

    Zero depth stays missing. Foreground and background masks must be disjoint;
    duplicate source frames and pixelwise holes never contribute confidence.
    """
    np = _numpy()
    if not frames:
        raise ValueError("at least one RGB-D frame is required")
    shape = (frames[0].height, frames[0].width)
    mask, background_mask = np.asarray(mask, dtype=bool), np.asarray(background_mask, dtype=bool)
    if mask.shape != shape or background_mask.shape != shape or np.any(mask & background_mask):
        raise ValueError("object/background masks must be disjoint and frame-sized")
    interior = mask.copy()
    for _ in range(limits.erosion_px):
        padded = np.pad(interior, 1, constant_values=False)
        interior = np.logical_and.reduce(
            [padded[y : y + shape[0], x : x + shape[1]] for y in range(3) for x in range(3)]
        )
    if len(frames) > 50:
        raise ValueError("depth temporal evidence exceeds bounded 50-frame budget")
    reasons, ids, ranges, accepted_frames = [], [], [], []
    provenance = None
    last_stamps, last_numbers = None, None
    for frame in frames:
        identity, failures = _frame_identity(frame, limits, now=now)
        reasons.extend(failures)
        key = (
            frame.metadata.get("session_id"),
            frame.metadata.get("profile_digest"),
            frame.intrinsics.to_dict(),
            frame.metadata.get("calibration_digest"),
            frame.metadata.get("mount_digest"),
            frame.depth_scale_m,
        )
        if provenance is not None and key != provenance:
            reasons.append("profile_or_calibration_changed")
        provenance = key
        if (
            frame.metadata.get("calibration_digest") != calibration_digest
            or frame.metadata.get("mount_digest") != mount_digest
        ):
            reasons.append("calibration_or_mount_mismatch")
        stamps = tuple(
            frame.metadata.get(k, {}).get("timestamp_ms") for k in ("color_source", "depth_source")
        )
        if (
            last_stamps is not None
            and all(isinstance(t, (int, float)) for t in (*stamps, *last_stamps))
            and any(a <= b for a, b in zip(stamps, last_stamps))
        ):
            reasons.append("source_clock_not_advancing")
        last_stamps = stamps
        numbers = tuple(
            frame.metadata.get(k, {}).get("frame_number") for k in ("color_source", "depth_source")
        )
        if (
            last_numbers is not None
            and all(isinstance(n, int) for n in (*numbers, *last_numbers))
            and any(a <= b for a, b in zip(numbers, last_numbers))
        ):
            reasons.append("source_frame_not_advancing")
        last_numbers = numbers
        if identity in ids:
            reasons.append("duplicate_frame")
            continue
        ids.append(identity)
        if (frame.height, frame.width) != shape:
            reasons.append("profile_changed")
            continue
        ranges.append(
            np.asarray(frame.depth_units(), dtype=float).reshape(shape)
            * frame.depth_scale_m
            * 1000.0
        )
        accepted_frames.append(frame)
    depth = ranges[-1]
    valid = np.isfinite(depth) & (depth > 0)
    support = interior & valid
    native_valid, count = int(np.count_nonzero(mask & valid)), int(np.count_nonzero(support))
    coverage = count / max(1, int(np.count_nonzero(interior)))
    median = float(np.median(depth[support])) if count else None
    temporal = None
    if len(ranges) >= limits.min_frames:
        stack = np.stack(ranges)
        common = interior & np.all(np.isfinite(stack) & (stack > 0), axis=0)
        if np.count_nonzero(common) >= limits.min_samples:
            samples = stack[:, common]
            temporal = float(np.percentile(np.abs(samples - np.median(samples, axis=0)), 95))
    rms = None
    if count >= 3:
        v, u = np.nonzero(support)
        points = np.array(
            [
                accepted_frames[-1].intrinsics.deproject(float(x), float(y), float(z))
                for x, y, z in zip(u, v, depth[support])
            ]
        )
        centered = points - np.mean(points, axis=0)
        _, singular, axes = np.linalg.svd(centered, full_matrices=False)
        if singular[1] > 1e-6:
            rms = float(np.sqrt(np.mean((centered @ axes[-1]) ** 2)))
    back = depth[background_mask & valid]
    separation = float(np.median(back) - median) if back.size and median is not None else None
    if count < limits.min_samples or coverage < limits.min_coverage:
        reasons.append("interior_depth_insufficient")
    if temporal is None or temporal > limits.max_temporal_p95_mm:
        reasons.append("temporal_depth_unqualified")
    if rms is None or rms > limits.max_surface_rms_mm:
        reasons.append("surface_fit_unqualified")
    if separation is None or separation < limits.min_background_separation_mm:
        reasons.append("foreground_background_unresolved")
    if uncertainty_mm is None or not math.isfinite(uncertainty_mm) or uncertainty_mm < 0:
        reasons.append("independent_uncertainty_missing")
        uncertainty_mm = None
    return DepthEvidence(
        not reasons,
        tuple(dict.fromkeys(reasons)),
        native_valid,
        count,
        coverage,
        len(ids),
        median,
        temporal,
        rms,
        separation,
        uncertainty_mm,
        tuple(ids),
        calibration_digest,
        mount_digest,
        hashlib.sha256(mask.tobytes()).hexdigest(),
        str(accepted_frames[-1].metadata.get("profile_digest", "")),
        tuple(f.monotonic_at for f in accepted_frames),
    )


class RegionDepthProcessor:
    """Configured static-region quality tool for camera-only qualification reads."""

    def __init__(
        self, config: Mapping[str, Any], *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.clock = clock
        self.configuration_digest = hashlib.sha256(
            json.dumps(
                copy.deepcopy(dict(config)),
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self.max_window_bytes = config.get("max_window_bytes", 64 * 1024 * 1024)
        if type(self.max_window_bytes) is not int or self.max_window_bytes < 1:
            raise ValueError("positive integer RGB-D window byte budget required")
        self.limits = DepthLimits(**config["limits"])
        self.roi = tuple(config["roi"])
        self.background_roi = tuple(config["background_roi"])
        if any(
            len(roi) != 4 or any(type(v) is not int for v in roi)
            for roi in (self.roi, self.background_roi)
        ):
            raise ValueError("integer measurement ROI required")
        self.calibration_digest = config["calibration_digest"]
        self.mount_digest = config["mount_digest"]
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.calibration_digest, self.mount_digest)
        ):
            raise ValueError("nonempty calibration and mount digests required")
        self.uncertainty_mm = config.get("uncertainty_mm")
        self._frames: list[RgbdFrame] = []

    def measure(self, frame: RgbdFrame) -> DepthEvidence:
        np = _numpy()
        masks = []
        for roi in (self.roi, self.background_roi):
            if len(roi) != 4 or not (
                0 <= roi[0] < roi[2] <= frame.width and 0 <= roi[1] < roi[3] <= frame.height
            ):
                raise ValueError("measurement ROI outside camera profile")
            mask = np.zeros((frame.height, frame.width), dtype=bool)
            mask[roi[1] : roi[3], roi[0] : roi[2]] = True
            masks.append(mask)
        frame_bytes = len(frame.color) + len(frame.depth)
        if frame_bytes > self.max_window_bytes:
            raise ValueError("one RGB-D frame exceeds the measurement window byte budget")
        key = tuple(
            frame.metadata.get(k)
            for k in ("session_id", "profile_digest", "mount_digest", "calibration_digest")
        )
        if self._frames and key != tuple(
            self._frames[-1].metadata.get(k)
            for k in ("session_id", "profile_digest", "mount_digest", "calibration_digest")
        ):
            self._frames.clear()
        self._frames.append(frame)
        self._frames = self._frames[-self.limits.min_frames :]
        while sum(len(f.color) + len(f.depth) for f in self._frames) > self.max_window_bytes:
            self._frames.pop(0)
        measured = qualify_depth(
            self._frames,
            *masks,
            limits=self.limits,
            now=self.clock(),
            calibration_digest=self.calibration_digest,
            mount_digest=self.mount_digest,
            uncertainty_mm=self.uncertainty_mm,
        )
        return replace(measured, configuration_digest=self.configuration_digest)
