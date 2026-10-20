# Copyright (c) Alibaba, Inc. and its affiliates.
"""Forward kinematics from a declared Denavit-Hartenberg table.

A policy that streams joint targets must be checked where its *tool* will go,
but the firmware only reports the flange pose of where the arm *is*.
``DhChain`` closes that gap: the device declaration carries its DH table under
``kinematics.dh`` and this module turns a joint vector into a flange pose in
the arm base frame.

A declared table is a claim, not evidence. Callers compare
``forward(current joints)`` against the flange the firmware reports
(``agreement``) and refuse to predict anything when the two disagree, so a
wrong or stale table fails closed instead of steering safety checks.

Declaration shape (lengths mm, angles deg)::

    dh:
      convention: modified   # Craig: RotX(alpha) TransX(a) RotZ(q + theta_offset) TransZ(d)
      links:                 # one per joint, in joint-channel order
        - {a: 0, alpha: 0, d: 123, theta_offset: 0}

``convention: standard`` reads each link as RotZ(q + theta_offset) TransZ(d)
TransX(a) RotX(alpha).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from leapflow.hardware.skills.geometry import Pose, Transform, rpy_to_matrix

CONVENTION_MODIFIED = "modified"
CONVENTION_STANDARD = "standard"
_CONVENTIONS = (CONVENTION_MODIFIED, CONVENTION_STANDARD)


def _rot_x(deg: float) -> Transform:
    return Transform(rpy_to_matrix(deg, 0.0, 0.0))


def _rot_z(deg: float) -> Transform:
    return Transform(rpy_to_matrix(0.0, 0.0, deg))


def _number(data: Mapping[str, Any], key: str, where: str, default: float | None) -> float:
    raw = data.get(key, default)
    if raw is None or isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{where} needs a numeric {key!r}")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{where} {key!r} must be finite")
    return value


@dataclass(frozen=True)
class DhLink:
    """One revolute link: lengths in mm, angles in degrees."""

    a: float
    alpha: float
    d: float
    theta_offset: float = 0.0

    def transform(self, q_deg: float, convention: str) -> Transform:
        theta = q_deg + self.theta_offset
        offset_a = Transform(translation=(self.a, 0.0, 0.0))
        offset_d = Transform(translation=(0.0, 0.0, self.d))
        if convention == CONVENTION_MODIFIED:
            return _rot_x(self.alpha).compose(offset_a).compose(_rot_z(theta)).compose(offset_d)
        return _rot_z(theta).compose(offset_d).compose(offset_a).compose(_rot_x(self.alpha))


@dataclass(frozen=True)
class DhChain:
    """A serial chain of revolute links from the arm base to the flange."""

    links: tuple[DhLink, ...]
    convention: str = CONVENTION_MODIFIED

    @property
    def dof(self) -> int:
        return len(self.links)

    def forward(self, joints_deg: Sequence[float]) -> Pose:
        """The flange pose in the base frame for *joints_deg* (joint-channel order)."""
        return self.transform(joints_deg).to_pose()

    def transform(self, joints_deg: Sequence[float]) -> Transform:
        """``T_base_flange`` for *joints_deg*, without an Euler round trip."""
        if len(joints_deg) != self.dof:
            raise ValueError(f"the chain has {self.dof} joints, got {len(joints_deg)} values")
        pose = Transform()
        for link, q in zip(self.links, joints_deg):
            pose = pose.compose(link.transform(float(q), self.convention))
        return pose

    def agreement(self, joints_deg: Sequence[float], flange: Pose) -> tuple[float, float]:
        """(mm, deg) between the predicted and the reported flange pose."""
        predicted = self.forward(joints_deg)
        return predicted.distance_mm(flange), predicted.rotation_to_deg(flange)

    def to_dict(self) -> dict[str, Any]:
        """The ``kinematics.dh`` block ``from_mapping`` reads back."""
        return {
            "convention": self.convention,
            "links": [
                {"a": k.a, "alpha": k.alpha, "d": k.d, "theta_offset": k.theta_offset}
                for k in self.links
            ],
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DhChain":
        """Parse a ``kinematics.dh`` block; raises ``ValueError`` on a bad table."""
        convention = str(data.get("convention") or CONVENTION_MODIFIED).strip().lower()
        if convention not in _CONVENTIONS:
            raise ValueError(f"dh convention must be one of {', '.join(_CONVENTIONS)}")
        raw_links = data.get("links")
        if not isinstance(raw_links, Sequence) or isinstance(raw_links, str) or not raw_links:
            raise ValueError("dh needs a non-empty 'links' list")
        links: list[DhLink] = []
        for index, raw in enumerate(raw_links, start=1):
            if not isinstance(raw, Mapping):
                raise ValueError(f"dh link {index} must be a mapping")
            where = f"dh link {index}"
            links.append(
                DhLink(
                    a=_number(raw, "a", where, None),
                    alpha=_number(raw, "alpha", where, None),
                    d=_number(raw, "d", where, None),
                    theta_offset=_number(raw, "theta_offset", where, 0.0),
                )
            )
        return cls(links=tuple(links), convention=convention)


__all__ = ["CONVENTION_MODIFIED", "CONVENTION_STANDARD", "DhChain", "DhLink"]
