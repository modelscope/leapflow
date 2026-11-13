# Copyright (c) Alibaba, Inc. and its affiliates.
"""The table one camera view sees around an object, for differential heights.

A camera chain and the arm chain can agree in x and y yet disagree in z by a
shared offset (a tool length or mount height measured differently) and by a
small tilt (a hand-eye rotation a few degrees off). An object's height read
against the table the *same view* sees -- its own top minus its own table -- is
free of both; ``table_plane`` measures that table. Offset alone is not enough:
under a 5 deg tilt the table at one side of a 100 mm box already stands 8 mm
over the offset plane, passes the height floor as "object", and drags the
footprint -- the centre moved with how loosely the box was drawn. For the same
reason ``convex_mask`` gives a remeasurement the silhouette an object casts in
the view rather than that silhouette's bounding box, which for a turned object
admits as much table as a loosely drawn box.
"""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

from leapflow.hardware.model_contract import MaskRle
from leapflow.hardware.rgbd import RgbdFrame
from leapflow.hardware.skills.geometry import Plane, Transform, Vec3, dot

if TYPE_CHECKING:
    from leapflow.hardware.skills.perception import PerceptionSettings

_TABLE_BIN_MM = 2.0  # height histogram bin that finds the table around a box
_MASK_SAMPLES = 4000  # foreground pixels enough to bound a mask
_FIT_ROUNDS = 4  # trim-and-refit rounds of the ring's plane
_FIT_MAD_K = 2.5  # robust deviations a ring reading may sit off the fitted plane
_MAD_SCALE = 1.4826  # MAD to standard deviation for normal noise


def base_points(
    pixels: Iterable[tuple[int, int]],
    frame: RgbdFrame,
    camera_to_base: Transform,
    settings: "PerceptionSettings",
) -> Iterator[Vec3]:
    """Base-frame points of *pixels* with a depth reading in the trusted range."""
    depth = frame.depth_units()
    scale = frame.depth_scale_m * 1000.0
    intrinsics = frame.intrinsics
    width = frame.width
    for u, v in pixels:
        z = depth[v * width + u] * scale
        if z and settings.min_depth_mm <= z <= settings.max_depth_mm:
            yield camera_to_base.apply(intrinsics.deproject(u, v, z))


def table_plane(
    box: Sequence[float],
    frame: RgbdFrame,
    camera_to_base: Transform,
    table: Plane,
    settings: "PerceptionSettings",
) -> Plane | None:
    """The table around pixel *box* as this view sees it, offset and tilt.

    The ring ``table_ring_px`` wide outside the box is the table the object
    stands on, plus whatever else stands there. The plane is fitted to the ring
    and refitted on the readings near it, so what stands on the table drops out
    as it sits above. A fit tilted more than ``max_table_tilt_deg`` from *table*
    is not trusted as a tilt, and the ring's densest height is used as a pure
    offset instead. ``None`` when the ring holds too few readings within
    ``max_table_offset_mm`` of *table* to say.
    """
    points = _ring_points(box, frame, camera_to_base, table, settings)
    if len(points) < settings.min_points:
        return None
    fitted = _fit_plane(points, table, settings)
    if fitted is not None:
        return fitted
    return Plane(table.normal, table.offset + _densest_offset(points, table))


def _ring_points(
    box: Sequence[float],
    frame: RgbdFrame,
    camera_to_base: Transform,
    table: Plane,
    settings: "PerceptionSettings",
) -> list[Vec3]:
    x0, y0, x1, y1 = box
    ring = settings.table_ring_px
    u0, v0 = max(0, int(x0 - ring)), max(0, int(y0 - ring))
    u1 = min(frame.width - 1, math.ceil(x1 + ring))
    v1 = min(frame.height - 1, math.ceil(y1 + ring))
    step = max(1, math.ceil(math.sqrt((u1 - u0 + 1) * (v1 - v0 + 1) / settings.max_points)))
    pixels = (
        (u, v)
        for v in range(v0, v1 + 1, step)
        for u in range(u0, u1 + 1, step)
        if not (x0 <= u < x1 and y0 <= v < y1)
    )
    limit = settings.max_table_offset_mm
    return [
        p for p in base_points(pixels, frame, camera_to_base, settings)
        if abs(table.above(p)) <= limit
    ]


def _densest_offset(points: Sequence[Vec3], table: Plane) -> float:
    """The ring's table height over *table*: everything on it is higher."""
    heights = [table.above(p) for p in points]
    bins: dict[int, int] = {}
    for h in heights:
        key = math.floor(h / _TABLE_BIN_MM)
        bins[key] = bins.get(key, 0) + 1
    densest = (max(bins, key=lambda key: (bins[key], -key)) + 0.5) * _TABLE_BIN_MM
    return statistics.median(h for h in heights if abs(h - densest) <= 1.5 * _TABLE_BIN_MM)


def _fit_plane(
    points: Sequence[Vec3], table: Plane, settings: "PerceptionSettings"
) -> Plane | None:
    """A trimmed least-squares plane through *points*, or ``None`` when untrustworthy."""
    kept = list(points)
    plane: Plane | None = None
    for _ in range(_FIT_ROUNDS):
        if len(kept) < settings.min_points:
            return None
        plane = _least_squares(kept)
        if plane is None:
            return None
        residuals = [plane.above(p) for p in points]
        middle = statistics.median(abs(r) for r in residuals)
        bound = max(1.5 * _TABLE_BIN_MM, _FIT_MAD_K * _MAD_SCALE * middle)
        kept = [p for p, r in zip(points, residuals) if abs(r) <= bound]
    if plane is None:
        return None
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, dot(plane.normal, table.normal)))))
    return plane if tilt <= settings.max_table_tilt_deg else None


def _least_squares(points: Sequence[Vec3]) -> Plane | None:
    """The plane ``z = a x + b y + c`` closest to *points* in z; ``None`` if degenerate."""
    n = float(len(points))
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    mz = sum(p[2] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    syy = sum((p[1] - my) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    sxz = sum((p[0] - mx) * (p[2] - mz) for p in points)
    syz = sum((p[1] - my) * (p[2] - mz) for p in points)
    det = sxx * syy - sxy * sxy
    if det <= 1e-9 * max(1.0, sxx * syy):
        return None
    a = (sxz * syy - syz * sxy) / det
    b = (syz * sxx - sxz * sxy) / det
    length = math.sqrt(a * a + b * b + 1.0)
    normal = (-a / length, -b / length, 1.0 / length)
    try:
        return Plane(normal, (mz - a * mx - b * my) / length)
    except ValueError:
        return None


def convex_mask(
    pixels: Sequence[tuple[float, float]], width: int, height: int
) -> MaskRle | None:
    """The mask of the convex hull of *pixels* in a *width* x *height* image.

    A pixel is foreground when its centre lies inside the hull; ``None`` when no
    pixel centre does (a hull off the frame or thinner than a pixel).
    """
    hull = _hull(pixels)
    if len(hull) < 3:
        return None
    counts: list[int] = []
    background = 0
    for u in range(width):
        span = _column_span(hull, u + 0.5)
        low = high = -1
        if span is not None:
            low = max(0, math.ceil(span[0] - 0.5))
            high = min(height - 1, math.floor(span[1] - 0.5))
        if high < low or span is None:
            background += height
            continue
        counts += [background + low, high - low + 1]
        background = height - 1 - high
    if not counts:
        return None
    return MaskRle(height=height, width=width, counts=(*counts, background))


def _hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """The convex hull of *points*, counter-clockwise (monotone chain)."""
    ordered = sorted(set((float(u), float(v)) for u, v in points))
    if len(ordered) < 3:
        return ordered

    def turn(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    chains: list[list[tuple[float, float]]] = []
    for sequence in (ordered, ordered[::-1]):
        chain: list[tuple[float, float]] = []
        for point in sequence:
            while len(chain) >= 2 and turn(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        chains.append(chain[:-1])
    return chains[0] + chains[1]


def _column_span(
    hull: Sequence[tuple[float, float]], x: float
) -> tuple[float, float] | None:
    """Where the vertical line at *x* crosses *hull*, lowest and highest v."""
    crossings = []
    for (u0, v0), (u1, v1) in zip(hull, (*hull[1:], hull[0])):
        if u0 != u1 and min(u0, u1) <= x <= max(u0, u1):
            crossings.append(v0 + (x - u0) * (v1 - v0) / (u1 - u0))
    return (min(crossings), max(crossings)) if crossings else None


def mask_box(mask: MaskRle | None) -> tuple[float, float, float, float]:
    """The pixel bounds of *mask*'s foreground; raises ``ValueError`` when it is empty."""
    if mask is None:
        raise ValueError("a mask detection carries no mask")
    pixels = list(mask.foreground(stride=max(1, math.ceil(mask.area / _MASK_SAMPLES))))
    if not pixels:
        raise ValueError("the mask is empty")
    us, vs = [u for u, _ in pixels], [v for _, v in pixels]
    return (min(us), min(vs), max(us) + 1, max(vs) + 1)


__all__ = ["base_points", "convex_mask", "mask_box", "table_plane"]
