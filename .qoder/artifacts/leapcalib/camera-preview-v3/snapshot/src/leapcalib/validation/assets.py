# Copyright (c) Alibaba, Inc. and its affiliates.
"""Which methods the available robot geometry actually permits.

Target-free calibration leans on the robot being its own reference: depth samples
are aligned to the visible surfaces of its own links. That only works if those
surfaces exist as data. A declared Denavit-Hartenberg table is *not* such data --
it places frames along a chain and says nothing about the shape of the metal
between them. Running point-to-plane alignment needs a surface model; a DH table
cannot supply one, however complete and vendor-documented it is.

The tempting shortcut is to approximate the links with boxes and cylinders and
proceed. The result fits, reports a small residual, and is biased by however far
the primitives are from the real castings -- an error that looks like calibration
rather than like a missing input. So this module answers one question honestly:
given the declared assets, which methods may run at all?

When no surface model is available the answer is not "fail". Scene-motion
estimation and task-local adaptation need no robot surface model, and they stay
available. What is refused is the specific claim that depth was matched against
known geometry when the geometry was invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from leapcalib.contracts.failures import (
    FAILURE_GEOMETRY_ASSET_MISSING,
    CalibrationFailure,
)
from leapcalib.contracts.identity import PARAMETER_SOURCES, SOURCE_DECLARED

ASSET_SURFACE_MESH = "surface_mesh"
"""A triangulated visible surface of a link. The only asset point-to-plane ICP may use."""
ASSET_PRIMITIVE_HULL = "primitive_hull"
"""Boxes/cylinders approximating a link. Good for coarse initialisation, biased for ICP."""
ASSET_DH_TABLE = "dh_table"
"""Frame placement along the chain. Carries no surface, so it fits nothing visually."""
ASSET_POINT_LANDMARKS = "point_landmarks"
"""Discrete native features with known positions on a link."""
ASSET_KINDS = frozenset(
    {ASSET_SURFACE_MESH, ASSET_PRIMITIVE_HULL, ASSET_DH_TABLE, ASSET_POINT_LANDMARKS}
)

SURFACE_ASSETS = frozenset({ASSET_SURFACE_MESH})
"""Assets that describe a visible surface. Deliberately a one-element set."""

METHOD_CAD_ICP = "cad_icp"
"""Depth-to-native-geometry point-to-plane alignment. Needs verified surface meshes."""
METHOD_LANDMARK_PNP = "landmark_pnp"
"""Pose from known native landmarks. Needs landmark assets, not full surfaces."""
METHOD_SCENE_MOTION = "scene_motion"
"""Hand-eye from relative camera motion over a static scene. Needs no robot surface model."""
METHOD_TASK_LOCAL = "task_local"
"""Bounded feature-to-joint response learning. Needs no robot geometry at all."""
METHODS = frozenset({METHOD_CAD_ICP, METHOD_LANDMARK_PNP, METHOD_SCENE_MOTION, METHOD_TASK_LOCAL})

# What each method requires of the declared assets. A method absent from this
# table requires nothing, which is why it must be added deliberately.
_REQUIREMENTS: dict[str, frozenset[str]] = {
    METHOD_CAD_ICP: SURFACE_ASSETS,
    METHOD_LANDMARK_PNP: frozenset({ASSET_POINT_LANDMARKS}),
}


@dataclass(frozen=True)
class GeometryAsset:
    """One piece of declared geometry for one link, and whether it was verified.

    ``verified`` means somebody checked this asset against the physical part --
    not that a file parsed. An unverified vendor mesh is still a prior: it may be
    for a different revision, a different gripper, or in the wrong units.
    """

    link_id: str
    kind: str
    digest: str
    source: str = SOURCE_DECLARED
    verified: bool = False
    units: str = "m"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.link_id.strip():
            raise ValueError("a geometry asset needs a link id")
        if self.kind not in ASSET_KINDS:
            raise ValueError(f"asset kind must be one of {sorted(ASSET_KINDS)}")
        if not self.digest.strip():
            raise ValueError("a geometry asset needs a content digest")
        if self.source not in PARAMETER_SOURCES:
            raise ValueError(f"asset source must be one of {sorted(PARAMETER_SOURCES)}")
        if type(self.verified) is not bool:
            raise ValueError("verification must be an explicit boolean")
        if self.units != "m":
            raise ValueError("geometry assets are declared in metres")

    @property
    def describes_surface(self) -> bool:
        """Whether this asset can carry a surface residual. A DH table cannot."""
        return self.kind in SURFACE_ASSETS

    def to_dict(self) -> dict[str, Any]:
        return {
            "link_id": self.link_id,
            "kind": self.kind,
            "digest": self.digest,
            "source": self.source,
            "verified": self.verified,
            "units": self.units,
            "notes": self.notes,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GeometryAsset":
        return cls(
            link_id=str(data["link_id"]),
            kind=str(data["kind"]),
            digest=str(data["digest"]),
            source=str(data.get("source") or SOURCE_DECLARED),
            verified=data.get("verified", False),
            units=str(data.get("units") or "m"),
            notes=str(data.get("notes") or ""),
        )


@dataclass(frozen=True)
class GeometryAvailability:
    """Which methods the declared assets permit, and why the others are refused."""

    required_links: tuple[str, ...]
    assets: tuple[GeometryAsset, ...]
    permitted: tuple[str, ...]
    refusals: Mapping[str, str] = field(default_factory=dict)

    def permits(self, method: str) -> bool:
        return method in self.permitted

    def require(self, method: str) -> None:
        """Raise unless *method* may run against the declared assets."""
        if method in self.permitted:
            return
        reason = self.refusals.get(method, "method is not declared")
        raise CalibrationFailure(FAILURE_GEOMETRY_ASSET_MISSING, reason)

    @property
    def surface_links(self) -> tuple[str, ...]:
        """Required links that have a verified surface model."""
        return tuple(
            sorted(
                {
                    a.link_id
                    for a in self.assets
                    if a.describes_surface and a.verified and a.link_id in self.required_links
                }
            )
        )

    @property
    def missing_surface_links(self) -> tuple[str, ...]:
        """Required links with no verified surface model."""
        have = set(self.surface_links)
        return tuple(sorted(link for link in self.required_links if link not in have))

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_links": list(self.required_links),
            "assets": [a.to_dict() for a in self.assets],
            "permitted": list(self.permitted),
            "refusals": dict(self.refusals),
            "surface_links": list(self.surface_links),
            "missing_surface_links": list(self.missing_surface_links),
        }


def availability(
    assets: Sequence[GeometryAsset],
    *,
    required_links: Sequence[str],
    methods: Sequence[str] = tuple(sorted(METHODS)),
) -> GeometryAvailability:
    """Decide which of *methods* the declared *assets* support for *required_links*.

    A method whose requirement is met for *every* required link is permitted.
    Partial coverage is refused with the missing links named: a solve over a
    subset of links is a different method with a different observability story,
    and should be requested as one rather than falling out of missing data.
    """
    unknown = set(methods) - METHODS
    if unknown:
        raise ValueError(f"unknown calibration methods {sorted(unknown)}")
    links = tuple(dict.fromkeys(required_links))
    assets = tuple(assets)
    permitted: list[str] = []
    refusals: dict[str, str] = {}

    for method in methods:
        required_kinds = _REQUIREMENTS.get(method)
        if not required_kinds:
            permitted.append(method)
            continue
        if not links:
            refusals[method] = f"{method} needs named links to match against"
            continue
        missing = []
        unverified = []
        for link in links:
            candidates = [a for a in assets if a.link_id == link and a.kind in required_kinds]
            if not candidates:
                missing.append(link)
            elif not any(a.verified for a in candidates):
                unverified.append(link)
        if missing:
            kinds = "/".join(sorted(required_kinds))
            refusals[method] = (
                f"{method} needs a {kinds} for every required link; missing {missing}"
            )
        elif unverified:
            refusals[method] = (
                f"{method} needs verified geometry; {unverified} are declared but unverified"
            )
        else:
            permitted.append(method)

    return GeometryAvailability(
        required_links=links,
        assets=assets,
        permitted=tuple(sorted(permitted)),
        refusals=refusals,
    )


__all__ = [
    "ASSET_DH_TABLE",
    "ASSET_KINDS",
    "ASSET_POINT_LANDMARKS",
    "ASSET_PRIMITIVE_HULL",
    "ASSET_SURFACE_MESH",
    "METHODS",
    "METHOD_CAD_ICP",
    "METHOD_LANDMARK_PNP",
    "METHOD_SCENE_MOTION",
    "METHOD_TASK_LOCAL",
    "SURFACE_ASSETS",
    "GeometryAsset",
    "GeometryAvailability",
    "availability",
]
