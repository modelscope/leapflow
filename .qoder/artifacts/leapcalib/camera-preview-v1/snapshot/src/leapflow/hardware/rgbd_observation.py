# Copyright (c) Alibaba, Inc. and its affiliates.
"""Camera-only RGB contour, metric cuboid, and independently observed pad tooling.

Color selectors are declared segmentation priors, not semantic classifiers or
physical certificates. Ambiguous/missing pad detections stay unqualified. All
resting-face fits are returned. This processor never opens an arm transport.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from leapflow.hardware.rgbd import RgbdFrame
from leapflow.hardware.rgbd_geometry import (
    CuboidFit,
    CuboidSeed,
    fit_cuboid_surfaces,
    inner_jaw_points,
)
from leapflow.hardware.rgbd_quality import DepthEvidence, DepthLimits, _numpy, qualify_depth
from leapflow.hardware.transport import TransportError


@dataclass(frozen=True)
class GeometryEvidence:
    depth: DepthEvidence
    proposals: tuple[CuboidFit, ...]
    jaw_midpoint_camera_mm: tuple[float, float, float] | None
    independent_jaws: bool
    session_id: str
    scene_id: str
    profile_digest: str
    observed_mono: float
    qualified: bool = False  # Geometry proposals never certify clearance or a hold.
    jaw_points_camera_mm: tuple[tuple[float, float, float], ...] = ()
    pad_depth: tuple[DepthEvidence, ...] = ()
    object_detected: bool = False
    dimensions_mm: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)
    source: Mapping[str, Any] = field(default_factory=dict)
    configuration_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class GeometryProcessor(Protocol):
    def measure(self, frame: RgbdFrame) -> GeometryEvidence: ...


class ColorGeometryProcessor:
    """Configured RGB selectors feeding lossless metric qualification tools.

    Each selector has an image ROI, lower/upper BGR bounds, and area bounds.
    Jaw selectors refer to visible inner-pad surfaces, not the flange projection.
    Independent holdout qualification of those landmarks remains mandatory.
    """

    def __init__(
        self, config: Mapping[str, Any], *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.config = copy.deepcopy(dict(config))
        self.clock = clock
        self.max_window_bytes = config.get("max_window_bytes", 64 * 1024 * 1024)
        if (
            isinstance(self.max_window_bytes, bool)
            or not isinstance(self.max_window_bytes, int)
            or self.max_window_bytes <= 0
        ):
            raise ValueError("positive integer RGB-D window byte budget required")
        self.limits = DepthLimits(**config["limits"])
        self._frames: list[RgbdFrame] = []
        for key in ("calibration_digest", "mount_digest"):
            value = self.config.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("nonempty calibration and mount digests required")
        self.seeds = tuple(CuboidSeed(**seed) for seed in config["cuboid_seeds"])
        if not 1 <= len(self.seeds) <= 12 or len(config["jaws"]) != 2:
            raise ValueError("bounded cuboid hypotheses and two independent pad selectors required")
        for key in ("surface_sigma_mm", "contour_sigma_px", "jaw_dispersion_mm"):
            value = config[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("finite positive geometry uncertainty required")
        for key in ("jaw_radius_px", "jaw_min_samples"):
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("positive integer jaw sampling limits required")
        for selector in (config["object"], *config["jaws"]):
            lo, hi = selector["min_area_px"], selector["max_area_px"]
            if (
                any(isinstance(v, bool) or not isinstance(v, int) for v in (lo, hi))
                or not 1 <= lo <= hi
            ):
                raise ValueError("positive integer selector area bounds required")
            bounds = (*selector["lower_bgr"], *selector["upper_bgr"])
            if len(bounds) != 6 or any(
                isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255 for v in bounds
            ):
                raise ValueError("integer BGR selector bounds required")

    def measure(self, frame: RgbdFrame) -> GeometryEvidence:
        frame_bytes = len(frame.color) + len(frame.depth)
        if frame_bytes > self.max_window_bytes:
            raise ValueError("one RGB-D frame exceeds the measurement window byte budget")
        np = _numpy()
        try:
            import cv2
        except ImportError as error:
            raise TransportError(
                "contour processing requires node-local OpenCV",
                failure_code="geometry_processor_unavailable",
            ) from error
        rgb = np.frombuffer(frame.color, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        mask, contour, object_valid = self._segment(np, cv2, rgb, self.config["object"])
        background = self._roi(np, frame.width, frame.height, self.config["background_roi"])
        pads = [self._segment(np, cv2, rgb, selector) for selector in self.config["jaws"]]
        jaw_valid = all(row[2] for row in pads) and not np.any(pads[0][0] & pads[1][0])
        pixels = []
        for support, _, valid in pads:
            if valid:
                v, u = np.nonzero(support)
                # The observed inner-pad region centroid; not a tool/FK-only point.
                pixels.append((float(np.mean(u)), float(np.mean(v))))
        key = tuple(
            frame.metadata.get(k)
            for k in ("session_id", "profile_digest", "calibration_digest", "mount_digest")
        )
        if self._frames and key != tuple(
            self._frames[-1].metadata.get(k)
            for k in ("session_id", "profile_digest", "calibration_digest", "mount_digest")
        ):
            self._frames.clear()
        self._frames.append(frame)
        self._frames = self._frames[-self.limits.min_frames :]
        while sum(len(f.color) + len(f.depth) for f in self._frames) > self.max_window_bytes:
            self._frames.pop(0)
        depth = qualify_depth(
            self._frames,
            mask,
            background,
            limits=self.limits,
            now=self.clock(),
            calibration_digest=str(self.config["calibration_digest"]),
            mount_digest=str(self.config["mount_digest"]),
            uncertainty_mm=self.config.get("uncertainty_mm"),
        )
        proposals = (
            fit_cuboid_surfaces(
                frame,
                mask,
                contour,
                self.seeds,
                evidence=depth,
                surface_sigma_mm=self.config["surface_sigma_mm"],
                contour_sigma_px=self.config["contour_sigma_px"],
            )
            if object_valid
            else ()
        )
        # Each pad has its own planar depth qualification. Object absence cannot
        # supply this evidence, and disjoint pads need not share a fitted plane.
        pad_depth = tuple(
            qualify_depth(
                self._frames,
                pad[0],
                background,
                limits=self.limits,
                now=self.clock(),
                calibration_digest=str(self.config["calibration_digest"]),
                mount_digest=str(self.config["mount_digest"]),
                uncertainty_mm=self.config.get("uncertainty_mm"),
            )
            for pad in pads
        )
        points = (
            inner_jaw_points(
                frame,
                pixels,
                independent=jaw_valid,
                support_masks=[p[0] for p in pads],
                radius_px=int(self.config["jaw_radius_px"]),
                min_samples=int(self.config["jaw_min_samples"]),
                max_dispersion_mm=self.config["jaw_dispersion_mm"],
            )
            if all(p.qualified for p in pad_depth)
            else None
        )
        midpoint = tuple((a + b) / 2 for a, b in zip(*points)) if points else None
        return GeometryEvidence(
            depth,
            proposals,
            midpoint,
            points is not None,
            str(frame.metadata.get("session_id", "")),
            str(self.config.get("scene_id", "")),
            str(frame.metadata.get("profile_digest", "")),
            frame.monotonic_at,
            jaw_points_camera_mm=points or (),
            pad_depth=pad_depth,
            object_detected=object_valid,
            dimensions_mm={s.hypothesis_id: s.dimensions_mm for s in self.seeds},
            source={
                k: copy.deepcopy(frame.metadata.get(k, {}))
                for k in ("color_source", "depth_source")
            },
            configuration_digest=hashlib.sha256(
                json.dumps(
                    self.config, sort_keys=True, allow_nan=False, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        )

    @staticmethod
    def _roi(np: Any, width: int, height: int, roi: Any) -> Any:
        if (
            len(roi) != 4
            or any(isinstance(v, bool) or not isinstance(v, int) for v in roi)
            or not (0 <= roi[0] < roi[2] <= width and 0 <= roi[1] < roi[3] <= height)
        ):
            raise ValueError("selector ROI outside active camera profile")
        mask = np.zeros((height, width), dtype=bool)
        mask[int(roi[1]) : int(roi[3]), int(roi[0]) : int(roi[2])] = True
        return mask

    @classmethod
    def _segment(
        cls, np: Any, cv2: Any, image: Any, selector: Mapping[str, Any]
    ) -> tuple[Any, Any, bool]:
        roi = cls._roi(np, image.shape[1], image.shape[0], selector["roi"])
        lower, upper = np.asarray(selector["lower_bgr"]), np.asarray(selector["upper_bgr"])
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or np.any(lower < 0)
            or np.any(upper > 255)
            or np.any(lower > upper)
        ):
            raise ValueError("invalid declared BGR selector")
        selected = roi & np.all((image >= lower) & (image <= upper), axis=2)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            selected.astype(np.uint8), connectivity=8
        )
        candidates = [
            i
            for i in range(1, count)
            if selector["min_area_px"] <= stats[i, cv2.CC_STAT_AREA] <= selector["max_area_px"]
        ]
        if len(candidates) != 1:
            return np.zeros_like(roi), (), False
        mask = labels == candidates[0]
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contour = contours[0].reshape(-1, 2) if len(contours) == 1 else ()
        return mask, contour, len(contour) >= 4
