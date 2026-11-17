# Copyright (c) Alibaba, Inc. and its affiliates.
"""Declared capsule visibility: a prediction, never collision-free authority."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from leapflow.hardware.skills.geometry import Vec3, dot


@dataclass(frozen=True)
class Capsule:
    start: Vec3
    end: Vec3
    radius_mm: float

    def blocks(self, camera: Vec3, target: Vec3) -> bool:
        return segment_distance(camera, target, self.start, self.end) < self.radius_mm


def segment_distance(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> float:
    """Distance of two finite segments, including degenerate/parallel segments."""
    u = tuple(y-x for x, y in zip(a, b))
    v = tuple(y-x for x, y in zip(c, d))
    w = tuple(x-y for x, y in zip(a, c))
    aa, bb, cc, dd, ee = dot(u, u), dot(u, v), dot(v, v), dot(u, w), dot(v, w)
    if aa < 1e-12 and cc < 1e-12:
        return math.dist(a, c)
    if aa < 1e-12:
        s, t = 0., min(1., max(0., ee/cc))
    elif cc < 1e-12:
        s, t = min(1., max(0., -dd/aa)), 0.
    else:
        denominator = aa*cc-bb*bb
        s = min(1., max(0., (bb*ee-cc*dd)/denominator)) if denominator > 1e-12 else 0.
        t = (bb*s+ee)/cc
        if t < 0:
            t, s = 0., min(1., max(0., -dd/aa))
        elif t > 1:
            t, s = 1., min(1., max(0., (bb-dd)/aa))
    return math.sqrt(sum((wi+s*ui-t*vi)**2 for wi, ui, vi in zip(w, u, v)))


def capsules(origins: Sequence[Vec3], radii: Sequence[float]) -> tuple[Capsule, ...]:
    if len(radii) != len(origins)-1 or any(not math.isfinite(r) or r <= 0 for r in radii):
        raise ValueError('body_radii_mm must declare one positive radius per kinematic link')
    return tuple(Capsule(a, b, float(r)) for a, b, r in zip(origins, origins[1:], radii))
