# Copyright (c) Alibaba, Inc. and its affiliates.
"""What changed in one camera's view since it was last shown -- cheap, in pixel space.

Every shared frame leaves a :class:`FrameSignature`: a coarse grid of near-surface
depths and mean brightness, sampled from a few thousand pixels. The next frame
of the same camera is compared cell by cell, so S2 is told *that* and *where*
the scene changed (something arrived, rose, or left; the light changed) without
being sent the earlier image again.

A grid only compares like with like: when the camera itself moved (a wrist
camera rides the arm) every cell changes, so a moved camera is reported as not
comparable instead of as a changed scene. Object-level change is pose-free and
lives with the scene view (an object's remembered surface checked against the
new depth). Nothing here knows a camera model: any aligned RGB-D frame works.

Sensor noise is measured, not assumed: each cell carries its depth coverage and
the spread of its depth samples, so a cell whose depth flickers (dark or distant
surfaces, beyond a sensor's sweet spot) must change by more than its own spread,
and gains or loses depth only when its coverage clearly crosses the threshold.

Pure Python: sampling a 640x480 frame costs a few milliseconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from leapflow.hardware.rgbd import RgbdFrame
from leapflow.hardware.skills.geometry import Transform, norm, rotation_distance_deg, sub

GRID_COLS = 8
GRID_ROWS = 6
_SAMPLES = 6000
_MIN_CELL_COVERAGE = 0.3  # a cell with less valid depth has no depth of its own
# Depth "vanished" or "appeared" only across this gap, so coverage hovering
# around the threshold (sparse, noisy depth) is not reported as a change.
_NO_DEPTH_COVERAGE = 0.1
# A cell's depth is its near surface, not its median: an object covering a fifth
# of a cell (a cup on a table) still sets it, while a few speckles do not.
_CELL_DEPTH_QUANTILE = 0.2
# The spread around that quantile is the cell's own noise floor.
_CELL_SPREAD_QUANTILE = 0.4
_DARK_LUMA = 40.0
_BRIGHT_LUMA = 215.0
_SPARSE_DEPTH = 0.3
_REGIONS = (
    ("top-left", "top", "top-right"),
    ("left", "center", "right"),
    ("bottom-left", "bottom", "bottom-right"),
)


@dataclass(frozen=True)
class FrameStats:
    """Whole-frame facts worth saying in words: depth coverage, range, and exposure."""

    depth_coverage: float
    near_mm: float | None
    median_mm: float | None
    luma: float

    @property
    def flags(self) -> tuple[str, ...]:
        found = []
        if self.luma < _DARK_LUMA:
            found.append("dark")
        elif self.luma > _BRIGHT_LUMA:
            found.append("overexposed")
        if self.depth_coverage < _SPARSE_DEPTH:
            found.append("sparse_depth")
        return tuple(found)

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth_coverage": round(self.depth_coverage, 3),
            "near_mm": None if self.near_mm is None else round(self.near_mm),
            "median_mm": None if self.median_mm is None else round(self.median_mm),
            "luma": round(self.luma, 1),
            "flags": list(self.flags),
        }


@dataclass(frozen=True)
class FrameSignature:
    """A camera's view reduced to a grid: near-surface depth (mm, 0 = none), mean
    luma, depth coverage, and depth spread (mm) per cell.

    *pose* is the camera's base-frame mount at capture, when calibrated; it is
    what tells a changed scene from a moved camera.
    """

    camera: str
    frame_id: str
    observed_at: float
    width: int
    height: int
    depth_mm: tuple[float, ...]
    luma: tuple[float, ...]
    pose: Transform | None = None
    cols: int = GRID_COLS
    rows: int = GRID_ROWS
    coverage: tuple[float, ...] = ()
    spread_mm: tuple[float, ...] = ()

    def coverage_of(self, cell: int) -> float:
        if self.coverage:
            return self.coverage[cell]
        return 1.0 if self.depth_mm[cell] else 0.0

    def spread_of(self, cell: int) -> float:
        return self.spread_mm[cell] if self.spread_mm else 0.0

    def cell_of(self, u: float, v: float) -> int:
        col = min(self.cols - 1, max(0, int(u * self.cols / self.width)))
        row = min(self.rows - 1, max(0, int(v * self.rows / self.height)))
        return row * self.cols + col

    def cells_in_box(self, box: Sequence[float]) -> tuple[int, ...]:
        """Cells a full-frame pixel box ``(x0, y0, x1, y1)`` touches."""
        first, last = self.cell_of(box[0], box[1]), self.cell_of(box[2] - 1e-6, box[3] - 1e-6)
        return tuple(
            row * self.cols + col
            for row in range(first // self.cols, last // self.cols + 1)
            for col in range(first % self.cols, last % self.cols + 1)
        )

    def region(self, cell: int) -> str:
        row, col = divmod(cell, self.cols)
        return _REGIONS[min(2, int(3 * (row + 0.5) / self.rows))][min(2, int(3 * (col + 0.5) / self.cols))]


@dataclass(frozen=True)
class ChangeReport:
    """How one camera's view differs from its previous signature."""

    camera: str
    since_frame: str
    age_s: float
    comparable: bool
    changed_cells: tuple[int, ...] = ()
    cells: int = GRID_COLS * GRID_ROWS
    closer: int = 0
    farther: int = 0
    depth_gained: int = 0
    depth_lost: int = 0
    brighter: int = 0
    darker: int = 0
    max_depth_change_mm: float = 0.0
    regions: tuple[str, ...] = ()
    camera_moved_mm: float | None = None
    camera_turned_deg: float | None = None
    note: str = ""

    @property
    def changed_fraction(self) -> float:
        return len(self.changed_cells) / self.cells if self.cells else 0.0

    def summary(self) -> str:
        since = f"since {self.since_frame} ({self.age_s:.0f} s ago)"
        if not self.comparable:
            return f"{self.camera}: {self.note or 'not comparable'} {since}"
        if not self.changed_cells:
            text = f"{self.camera}: no visible change {since}"
        else:
            parts = []
            if self.closer:
                parts.append(f"{self.closer} closer (something arrived or rose)")
            if self.farther:
                parts.append(f"{self.farther} farther (something left or sank)")
            if self.depth_gained or self.depth_lost:
                parts.append(f"depth appeared in {self.depth_gained}, vanished in {self.depth_lost}")
            if self.brighter or self.darker:
                parts.append(f"{self.brighter} brighter, {self.darker} darker")
            text = (
                f"{self.camera}: {self.changed_fraction:.0%} of the view changed {since}, "
                f"at {', '.join(self.regions)}; cells {'; '.join(parts)}"
            )
        return f"{text}; {self.note}" if self.note else text

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "since_frame": self.since_frame,
            "age_s": round(self.age_s, 1),
            "comparable": self.comparable,
            "changed_fraction": round(self.changed_fraction, 3),
            "changed_cells": len(self.changed_cells),
            "regions": list(self.regions),
            "summary": self.summary(),
        }
        if self.changed_cells:
            data.update(
                closer=self.closer, farther=self.farther, depth_gained=self.depth_gained,
                depth_lost=self.depth_lost, brighter=self.brighter, darker=self.darker,
                max_depth_change_mm=round(self.max_depth_change_mm),
            )
        if self.camera_moved_mm is not None:
            data["camera_moved_mm"] = round(self.camera_moved_mm, 1)
            data["camera_turned_deg"] = round(self.camera_turned_deg or 0.0, 1)
        return data


def observe(
    frame: RgbdFrame,
    *,
    camera: str,
    frame_id: str,
    pose: Transform | None,
    cols: int = GRID_COLS,
    rows: int = GRID_ROWS,
    samples: int = _SAMPLES,
) -> tuple[FrameSignature, FrameStats]:
    """Sample *frame* on a regular lattice into its signature and whole-frame stats."""
    width, height = frame.width, frame.height
    step = max(1, int(math.sqrt(width * height / max(1, samples))))
    depth, color = frame.depth_units(), frame.color
    scale = frame.depth_scale_m * 1000.0
    cell_depths: list[list[float]] = [[] for _ in range(cols * rows)]
    cell_luma = [0.0] * (cols * rows)
    cell_count = [0] * (cols * rows)
    valid: list[float] = []
    luma_sum, count = 0.0, 0
    for v in range(step // 2, height, step):
        row = (v * rows // height) * cols
        base = v * width
        for u in range(step // 2, width, step):
            index = base + u
            cell = row + u * cols // width
            b, g, r = color[3 * index], color[3 * index + 1], color[3 * index + 2]
            luma = 0.114 * b + 0.587 * g + 0.299 * r
            luma_sum += luma
            count += 1
            cell_luma[cell] += luma
            cell_count[cell] += 1
            z = depth[index] * scale
            if z:
                cell_depths[cell].append(z)
                valid.append(z)
    coverage = tuple(
        len(values) / cell_count[i] if cell_count[i] else 0.0
        for i, values in enumerate(cell_depths)
    )
    nears, spreads = [], []
    for i, values in enumerate(cell_depths):
        if coverage[i] < _MIN_CELL_COVERAGE:
            nears.append(0.0)
            spreads.append(0.0)
            continue
        values.sort()
        near = _quantile(values, _CELL_DEPTH_QUANTILE)
        nears.append(near)
        spreads.append(_quantile(values, _CELL_SPREAD_QUANTILE) - near)
    lumas = tuple(cell_luma[i] / cell_count[i] if cell_count[i] else 0.0 for i in range(cols * rows))
    valid.sort()
    stats = FrameStats(
        depth_coverage=len(valid) / count if count else 0.0,
        near_mm=valid[int(0.05 * (len(valid) - 1))] if valid else None,
        median_mm=valid[len(valid) // 2] if valid else None,
        luma=luma_sum / count if count else 0.0,
    )
    signature = FrameSignature(
        camera=camera, frame_id=frame_id, observed_at=frame.observed_at, width=width,
        height=height, depth_mm=tuple(nears), luma=lumas, pose=pose, cols=cols, rows=rows,
        coverage=coverage, spread_mm=tuple(spreads),
    )
    return signature, stats


def compare(
    before: FrameSignature,
    after: FrameSignature,
    *,
    depth_mm: float = 15.0,
    depth_ratio: float = 0.03,
    luma: float = 25.0,
    pose_mm: float = 5.0,
    pose_deg: float = 2.0,
) -> ChangeReport:
    """Cell-by-cell change from *before* to *after* (the same camera).

    A cell changed when its depth moved more than ``max(depth_mm, depth_ratio *
    depth, its own spread)``, clearly gained or lost depth, or its brightness
    moved more than *luma*. The spread is the smaller of the two frames': noise
    shows in both, while something arriving widens only the new one.
    """
    age = max(0.0, after.observed_at - before.observed_at)
    head = {"camera": after.camera, "since_frame": before.frame_id, "age_s": age}
    if (before.width, before.height, before.cols, before.rows) != (
        after.width, after.height, after.cols, after.rows
    ):
        return ChangeReport(**head, comparable=False, note="the stream resolution changed")
    moved = turned = None
    note = ""
    if before.pose is not None and after.pose is not None:
        moved = norm(sub(after.pose.translation, before.pose.translation))
        turned = rotation_distance_deg(before.pose.rotation, after.pose.rotation)
        if moved > pose_mm or turned > pose_deg:
            return ChangeReport(
                **head, comparable=False, camera_moved_mm=moved, camera_turned_deg=turned,
                note=(
                    f"the camera moved {moved:.0f} mm / {turned:.0f} deg, so pixels are not "
                    "comparable; judge objects, not pixels"
                ),
            )
    elif before.pose is None or after.pose is None:
        note = "camera pose unknown, so a change may be camera motion"
    changed: list[int] = []
    counts = dict(closer=0, farther=0, depth_gained=0, depth_lost=0, brighter=0, darker=0)
    largest = 0.0
    for cell, (d0, d1, l0, l1) in enumerate(
        zip(before.depth_mm, after.depth_mm, before.luma, after.luma)
    ):
        hit = False
        if d0 and d1:
            delta = d1 - d0
            noise = min(before.spread_of(cell), after.spread_of(cell))
            if abs(delta) > max(depth_mm, depth_ratio * d0, noise):
                counts["closer" if delta < 0 else "farther"] += 1
                largest = max(largest, abs(delta))
                hit = True
        elif d1 and before.coverage_of(cell) < _NO_DEPTH_COVERAGE:
            counts["depth_gained"] += 1
            hit = True
        elif d0 and after.coverage_of(cell) < _NO_DEPTH_COVERAGE:
            counts["depth_lost"] += 1
            hit = True
        if abs(l1 - l0) > luma:
            counts["brighter" if l1 > l0 else "darker"] += 1
            hit = True
        if hit:
            changed.append(cell)
    regions = tuple(dict.fromkeys(after.region(cell) for cell in changed))
    if not changed:
        note = ""  # an unknown pose only matters when something changed
    return ChangeReport(
        **head, comparable=True, changed_cells=tuple(changed), cells=len(after.depth_mm),
        max_depth_change_mm=largest, regions=regions, camera_moved_mm=moved,
        camera_turned_deg=turned, note=note, **counts,
    )


def _quantile(ordered: Sequence[float], q: float) -> float:
    """The *q* quantile of already sorted *ordered*."""
    return ordered[int(q * (len(ordered) - 1))]


__all__ = ["ChangeReport", "FrameSignature", "FrameStats", "compare", "observe"]
