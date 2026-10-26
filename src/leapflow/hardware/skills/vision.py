# Copyright (c) Alibaba, Inc. and its affiliates.
"""Node-local measurement of the workcell's calibration target (a ChArUco board).

``calib.handeye`` and ``calib.verify`` observe the board a workcell declares
under ``calibration_target``; this module turns one fresh RGB-D frame into the
board's pose in the camera frame. OpenCV with ``cv2.aruco`` and numpy are
imported lazily inside the detection thread, so the node runtime imports without
them and a node that lacks them answers ``skill_unavailable`` only when a
calibration actually needs the camera.

Lens distortion is handled once and model-agnostically: every detected corner
is undistorted to normalized image coordinates with the stream's own
``CameraIntrinsics.deproject`` and the pose is solved for an ideal pinhole
(K = I, no distortion), so every model ``CameraIntrinsics`` inverts works
without translating it into OpenCV's coefficient conventions.

The depth image is sampled where the board centre is seen, giving the same
deprojected point perception would compute there; ``calib.verify`` compares it
with the arm's prediction so the accuracy it reports is the one ``locate`` gets.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping

from leapflow.hardware.rgbd import FrameSource, RgbdFrame
from leapflow.hardware.skills.geometry import Transform, Vec3
from leapflow.hardware.skills.protocol import (
    FAILURE_DEVICE_FAULT,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_TARGET_NOT_FOUND,
    CloudObservation,
    SkillError,
    TargetObservation,
)
from leapflow.hardware.skills.workcell import (
    ROLE_CAMERA,
    WorkcellSpec,
)
from leapflow.hardware.transport import TransportError

logger = logging.getLogger(__name__)

TARGET_CHARUCO = "charuco"
DEFAULT_DICTIONARY = "DICT_5X5_100"
DEPTH_WINDOW_PX = 2  # half-width: the centre depth is the median of a 5x5 window
CLOUD_POINT_LIMIT = 20000  # sampled depth pixels per cloud, evenly strided
_REQUIRED_FIELDS = ("squares_x", "squares_y", "square_mm", "marker_mm")
_TARGET_FIELDS = frozenset({*_REQUIRED_FIELDS, "type", "dictionary", "legacy", "min_corners"})
_OPENCV_HINT = "install opencv-contrib-python-headless (cv2.aruco) and numpy on the node"


@dataclass(frozen=True)
class CharucoTarget:
    """A ChArUco board: *squares_x* by *squares_y* squares, markers in the white ones.

    Board coordinates follow OpenCV: the origin at the outer corner of the first
    square, x along the *squares_x* direction, z into the board. ``legacy`` selects
    the pre-4.6 OpenCV layout some printed boards use.
    """

    squares_x: int
    squares_y: int
    square_mm: float
    marker_mm: float
    dictionary: str = DEFAULT_DICTIONARY
    legacy: bool = False
    min_corners: int = 6

    def __post_init__(self) -> None:
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("a ChArUco board needs at least 3 x 3 squares")
        if not (math.isfinite(self.square_mm) and self.square_mm > 0):
            raise ValueError("square_mm must be a positive length")
        if not 0 < self.marker_mm < self.square_mm:
            raise ValueError("marker_mm must be positive and smaller than square_mm")
        if not self.dictionary.startswith("DICT_"):
            raise ValueError(f"dictionary {self.dictionary!r} is not an ArUco DICT_* name")
        inner = (self.squares_x - 1) * (self.squares_y - 1)
        if not 4 <= self.min_corners <= inner:
            raise ValueError(f"min_corners must be in [4, {inner}] for this board")

    @property
    def center(self) -> Vec3:
        """The board's middle in its own frame (mm)."""
        return (self.squares_x * self.square_mm / 2.0, self.squares_y * self.square_mm / 2.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": TARGET_CHARUCO,
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_mm": self.square_mm,
            "marker_mm": self.marker_mm,
            "dictionary": self.dictionary,
            "legacy": self.legacy,
            "min_corners": self.min_corners,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CharucoTarget":
        """Parse a workcell's ``calibration_target``; raises ``ValueError`` when unusable."""
        kind = str(data.get("type") or TARGET_CHARUCO)
        if kind != TARGET_CHARUCO:
            raise ValueError(f"target type {kind!r} is not supported (use {TARGET_CHARUCO!r})")
        unknown = sorted(set(map(str, data)) - _TARGET_FIELDS)
        if unknown:
            raise ValueError(f"unknown target fields: {', '.join(unknown)}")
        legacy = data.get("legacy", False)
        if not isinstance(legacy, bool):
            raise ValueError("legacy must be true or false")
        missing = [name for name in _REQUIRED_FIELDS if name not in data]
        if missing:
            raise ValueError(f"the target needs {', '.join(missing)}")
        return cls(
            squares_x=_integer(data["squares_x"], "squares_x"),
            squares_y=_integer(data["squares_y"], "squares_y"),
            square_mm=_number(data["square_mm"], "square_mm"),
            marker_mm=_number(data["marker_mm"], "marker_mm"),
            dictionary=str(data.get("dictionary") or DEFAULT_DICTIONARY),
            legacy=legacy,
            min_corners=_integer(data.get("min_corners", 6), "min_corners"),
        )


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _integer(value: Any, name: str) -> int:
    number = _number(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be a whole number")
    return int(number)


# ── Detection ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BoardDetection:
    """The board's pose in one frame and how well its corners reproject."""

    transform: Transform  # T_camera_board, mm
    corners: int
    reprojection_px: float


def detect_board(frame: RgbdFrame, target: CharucoTarget) -> BoardDetection | None:
    """Find *target* in *frame*; None when too few corners (or only one row) are seen.

    Blocking (run it in a thread). Raises ``ImportError`` without OpenCV's aruco
    module or numpy, and ``ValueError`` for a dictionary OpenCV does not know.
    """
    import cv2
    import numpy as np

    aruco = getattr(cv2, "aruco", None)
    if aruco is None or not hasattr(aruco, "CharucoDetector"):
        raise ImportError("this OpenCV build has no cv2.aruco.CharucoDetector")
    code = getattr(aruco, target.dictionary, None)
    if not isinstance(code, int):
        raise ValueError(f"OpenCV knows no ArUco dictionary {target.dictionary!r}")
    board = aruco.CharucoBoard(
        (target.squares_x, target.squares_y),
        target.square_mm,
        target.marker_mm,
        aruco.getPredefinedDictionary(code),
    )
    if target.legacy:
        board.setLegacyPattern(True)
    image = np.frombuffer(frame.color, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    try:
        corners, ids, _, _ = aruco.CharucoDetector(board).detectBoard(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )
    except cv2.error as exc:
        logger.warning("ChArUco detection failed on frame %s: %s", frame.sequence, exc)
        return None
    if ids is None or len(ids) < target.min_corners:
        return None
    # The detector's adaptive window can include nearby ArUco ink and bias a
    # chess corner. Refine locally, bounded by the target's declared white margin.
    pixels = corners.reshape(-1, 2)
    distances = np.linalg.norm(pixels[:, None, :] - pixels[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    margin_px = float(distances.min()) * (target.square_mm - target.marker_mm) / (2 * target.square_mm)
    if not math.isfinite(margin_px) or margin_px <= 1.0:
        return None  # too little white space to refine independently of marker ink
    window = max(1, min(3, int(margin_px) - 1))
    try:
        corners = cv2.cornerSubPix(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), corners, (window, window), (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
    except cv2.error as exc:
        logger.warning("ChArUco refinement failed on frame %s: %s", frame.sequence, exc)
        return None
    pixels = corners.reshape(-1, 2).astype(np.float64).tolist()
    points = board.getChessboardCorners()[ids.reshape(-1)].astype(np.float64)
    if np.ptp(points[:, 0]) == 0.0 or np.ptp(points[:, 1]) == 0.0:
        return None  # one row or column of corners: the board's tilt is unobservable
    intrinsics = frame.intrinsics
    normalized = np.array(
        [intrinsics.deproject(u, v, 1.0)[:2] for u, v in pixels], dtype=np.float64
    )
    identity = np.eye(3)
    ok, rvec, tvec = cv2.solvePnP(points, normalized, identity, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(points, normalized, identity, None, rvec, tvec)
    matrix = cv2.Rodrigues(rvec)[0].tolist()
    transform = Transform(
        tuple(tuple(float(v) for v in row) for row in matrix),  # type: ignore[arg-type]
        tuple(float(v) for v in tvec.reshape(-1)),  # type: ignore[arg-type]
    )
    errors = []
    for point, pixel in zip(points.tolist(), pixels):
        camera_point = transform.apply(point)
        if camera_point[2] <= 0.0:
            return None  # a pose behind the camera is a mirrored, wrong solution
        errors.append(math.dist(intrinsics.project(camera_point), pixel))
    rms = math.sqrt(sum(e * e for e in errors) / len(errors))
    return BoardDetection(transform=transform, corners=len(pixels), reprojection_px=rms)


def depth_point(frame: RgbdFrame, point: Vec3, *, window: int = DEPTH_WINDOW_PX) -> Vec3 | None:
    """The depth image's point where camera-frame *point* is seen, or None without depth.

    The depth is the median of the valid measurements in a small window around
    the pixel, so one missing or flying pixel does not decide the result.
    """
    if point[2] <= 0.0:
        return None
    intrinsics = frame.intrinsics
    u, v = intrinsics.project(point)
    cu, cv = round(u), round(v)
    values = sorted(
        frame.depth_mm(x, y)
        for x in range(max(0, cu - window), min(frame.width, cu + window + 1))
        for y in range(max(0, cv - window), min(frame.height, cv + window + 1))
        if frame.depth_mm(x, y) > 0.0
    )
    if not values:
        return None
    return intrinsics.deproject(u, v, values[len(values) // 2])


def cloud_points(
    frame: RgbdFrame,
    *,
    limit: int = CLOUD_POINT_LIMIT,
    near_mm: float | None = None,
    far_mm: float | None = None,
) -> tuple[Vec3, ...]:
    """The frame's valid depth pixels, evenly strided, in the camera frame (mm).

    Deprojection goes through ``CameraIntrinsics.deproject`` pixel by pixel: that
    is the same inverse perception uses, so whatever lens model the stream
    declares is honoured instead of assumed to be a pinhole. Sampling is by row
    order across the image that survives the filter, which keeps the arm's
    coverage even when its links cross several depth bands.

    ``near_mm``/``far_mm`` cut *before* the stride, and that order is the whole
    point of the window: a room-sized frame is mostly table, floor, and backdrop,
    so striding first spends most of the budget on surfaces no calibration uses
    and leaves the arm too sparse for a 3 mm occupancy grid -- a port that did
    exactly that moved the measured mount 14 mm.
    """
    import numpy as np

    depth = np.frombuffer(frame.depth, dtype=np.uint16).reshape(frame.height, frame.width)
    mask = depth > 0
    if near_mm is not None or far_mm is not None:
        millimetres = depth.astype(np.float32) * (frame.depth_scale_m * 1000.0)
        if near_mm is not None:
            mask &= millimetres > float(near_mm)
        if far_mm is not None:
            mask &= millimetres < float(far_mm)
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return ()
    stride = max(1, math.ceil(len(rows) / max(1, int(limit))))
    intrinsics, scale = frame.intrinsics, frame.depth_scale_m * 1000.0
    sampled = range(0, len(rows), stride)
    return tuple(
        intrinsics.deproject(
            int(columns[index]), int(rows[index]), float(depth[rows[index], columns[index]]) * scale
        )
        for index in sampled
    )


# ── Port ─────────────────────────────────────────────────────────────────────


class WorkcellVision:
    """Calibration measurements for one skill context. Satisfies ``VisionPort``."""

    def __init__(
        self, *, registry: Any, camera: str, target: CharucoTarget | None = None
    ) -> None:
        self._registry = registry
        self._camera = camera
        self._target = target

    @classmethod
    def for_workcell(cls, registry: Any, spec: WorkcellSpec, *, role: str = ROLE_CAMERA) -> "WorkcellVision":
        """The port for *spec*; raises ``skill_unavailable`` without a camera.

        A declared ``calibration_target`` is parsed here, so a malformed board is
        refused at admission time rather than halfway through a calibration. A
        workcell may declare none -- a cloud calibration needs no board -- and
        ``observe_target`` then answers for itself.
        """
        if role.split(".", 1)[0] != ROLE_CAMERA:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, f"{role!r} is not a camera role")
        camera = spec.role(role)
        if not camera:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE, f"workcell {spec.workcell_id!r} binds no {role}"
            )
        target = None
        if spec.calibration_target:
            try:
                target = CharucoTarget.from_mapping(spec.calibration_target)
            except ValueError as exc:
                raise SkillError(FAILURE_SKILL_UNAVAILABLE, f"calibration_target: {exc}") from exc
        return cls(registry=registry, camera=camera, target=target)

    @property
    def target(self) -> CharucoTarget:
        if self._target is None:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                f"workcell {self._camera!r} declares no calibration_target (the ChArUco "
                "board: squares_x, squares_y, square_mm, marker_mm)",
            )
        return self._target

    async def observe_target(self) -> TargetObservation:
        target = self.target
        frame = await self._capture()
        try:
            found = await asyncio.to_thread(detect_board, frame, target)
        except ImportError as exc:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE, f"target detection needs OpenCV ({exc}); {_OPENCV_HINT}"
            ) from exc
        except ValueError as exc:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, f"calibration_target: {exc}") from exc
        if found is None:
            raise SkillError(
                FAILURE_TARGET_NOT_FOUND,
                f"the calibration board is not visible to {self._camera!r} (need at least "
                f"{target.min_corners} ChArUco corners over two rows); check it is in "
                "view, lit evenly, and matches calibration_target",
            )
        center = target.center
        return TargetObservation(
            transform=found.transform,
            center=center,
            corners=found.corners,
            reprojection_px=found.reprojection_px,
            intrinsics=frame.intrinsics.to_dict(),
            observed_at=frame.observed_at,
            depth_point=depth_point(frame, found.transform.apply(center)),
        )

    async def observe_cloud(
        self, *, near_mm: float | None = None, far_mm: float | None = None
    ) -> CloudObservation:
        frame = await self._capture()
        points = await asyncio.to_thread(
            cloud_points, frame, near_mm=near_mm, far_mm=far_mm
        )
        if not points:
            raise SkillError(
                FAILURE_DEVICE_FAULT,
                f"camera {self._camera!r} returned a frame with no valid depth; check the "
                "depth stream and the lens is not covered",
            )
        return CloudObservation(
            points=points, intrinsics=frame.intrinsics.to_dict(), observed_at=frame.observed_at
        )

    async def _capture(self) -> RgbdFrame:
        # The transport is resolved under the device lock, not before it: a preview
        # release waiting on the same lock may otherwise close the camera between the
        # lookup and the capture.
        async with self._registry.device_io(self._camera):
            try:
                source = await self._registry.transport(self._camera)
            except TransportError as exc:
                raise SkillError(
                    FAILURE_SKILL_UNAVAILABLE, f"{self._camera!r} is unavailable: {exc}"
                ) from exc
            if not isinstance(source, FrameSource):
                raise SkillError(
                    FAILURE_SKILL_UNAVAILABLE, f"{self._camera!r} is not an RGB-D camera"
                )
            try:
                return await source.capture_rgbd()
            except TransportError as exc:
                raise SkillError(
                    FAILURE_DEVICE_FAULT, f"camera {self._camera!r} capture failed: {exc}"
                ) from exc


__all__ = [
    "CLOUD_POINT_LIMIT",
    "DEFAULT_DICTIONARY",
    "TARGET_CHARUCO",
    "BoardDetection",
    "CharucoTarget",
    "WorkcellVision",
    "cloud_points",
    "depth_point",
    "detect_board",
]
