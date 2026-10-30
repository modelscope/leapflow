# Copyright (c) Alibaba, Inc. and its affiliates.
"""Topology identity: what a calibration result is a result *of*.

An estimate is only meaningful against the hardware that produced it. Re-seat a
camera, swap jaws, change a stream from 848x480 to 1280x720, or point the node at
a different firmware profile, and a transform that was right is now quietly wrong
-- numerically unchanged, physically invalid. So every bundle and every estimate
carries the identity of its topology, and a mismatch is detectable by comparison
rather than by remembering.

Each identity hashes to a digest over its canonical fields, and
:meth:`TopologyIdentity.differs_from` names *which* component moved. That
distinction is what lets invalidation stay proportionate: a new stream profile
invalidates the intrinsics-dependent work, while a remount invalidates the
extrinsics that were measured against the old seat.

A digest is an identity, not a qualification. Two bundles sharing a topology
digest were taken against the same declared hardware; neither is thereby accurate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, dataclass, field
from typing import Any, Mapping, Sequence

from leapcalib.contracts.units import canonical_number

SOURCE_FACTORY = "factory"
"""Shipped by the device (librealsense intrinsics, depth scale). A prior, not a measurement."""
SOURCE_DECLARED = "declared"
"""Written by a human into a declaration (mount pose, CAD offset). A prior."""
SOURCE_ESTIMATED = "estimated"
"""Produced by a LeapCalib solve against observations, with its own uncertainty."""
SOURCE_VENDOR_DOCUMENT = "vendor_document"
"""Published in a vendor manual or SDK preset (the Piper MDH table)."""
PARAMETER_SOURCES = frozenset(
    {SOURCE_FACTORY, SOURCE_DECLARED, SOURCE_ESTIMATED, SOURCE_VENDOR_DOCUMENT}
)

ROLE_FIXED = "fixed"
"""A camera bolted to the cell, observing the robot (the D435if role here)."""
ROLE_WRIST = "wrist"
"""A camera riding the flange, observing the scene (the D405 role here)."""
CAMERA_ROLES = frozenset({ROLE_FIXED, ROLE_WRIST})

# The references a "tool pose" can mean. They are different points on the same
# gripper and conflating any two of them is a silent offset of millimetres to
# centimetres, so a tool identity names which one it is about.
TOOL_CAMERA_CENTRE = "camera_centre"
TOOL_FLANGE = "flange"
TOOL_FINGERTIP_PLANE = "fingertip_plane"
TOOL_INNER_PAD_MIDPOINT = "inner_pad_midpoint"
TOOL_GRASP_REFERENCE = "grasp_reference"
TOOL_REFERENCES = frozenset(
    {
        TOOL_CAMERA_CENTRE,
        TOOL_FLANGE,
        TOOL_FINGERTIP_PLANE,
        TOOL_INNER_PAD_MIDPOINT,
        TOOL_GRASP_REFERENCE,
    }
)


def canonical_payload(value: Any) -> Any:
    """*value* with every number canonical, ready to hash or serialise.

    Mappings are left to ``json.dumps(sort_keys=True)`` for ordering; sequences
    keep theirs, because a joint vector's order is meaning.
    """
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return canonical_number(value)
    if isinstance(value, Mapping):
        return {str(k): canonical_payload(v) for k, v in value.items()}
    if isinstance(value, Sequence):
        return [canonical_payload(v) for v in value]
    raise ValueError(f"{type(value).__name__} is not canonicalisable for a digest")


def digest_of(payload: Mapping[str, Any]) -> str:
    """A stable sha256 over *payload*; nonfinite values are refused, not hashed."""
    encoded = json.dumps(
        canonical_payload(payload), sort_keys=True, allow_nan=False, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _require_id(value: str, what: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{what} needs a bounded nonempty identifier")
    return value


def _text(value: Any) -> None:
    if not isinstance(value, str) or len(value) > 4096:
        raise ValueError("identity fields must be bounded typed strings")


def _fields(data: Mapping[str, Any], cls: Any) -> None:
    fields = cls.__dataclass_fields__
    required = {
        name
        for name, value in fields.items()
        if value.default is MISSING and value.default_factory is MISSING
    }
    if not isinstance(data, Mapping) or set(data) - set(fields) or not required <= set(data):
        raise ValueError("missing or unknown identity fields")


def _array(value: Any, maximum: int) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or len(value) > maximum:
        raise ValueError("bounded identity array required before decoding")
    return value


@dataclass(frozen=True)
class DeviceIdentity:
    """The controlled device a calibration is about, as its declaration names it."""

    device_id: str
    vendor: str = ""
    model: str = ""
    firmware_profile: str = ""

    def __post_init__(self) -> None:
        _require_id(self.device_id, "device")
        for value in (self.vendor, self.model, self.firmware_profile):
            _text(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "vendor": self.vendor,
            "model": self.model,
            "firmware_profile": self.firmware_profile,
        }

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())


@dataclass(frozen=True)
class StreamProfile:
    """One image stream as it was actually configured, with its depth convention.

    Intrinsics are in pixels and the depth scale is metres per stored depth unit,
    mirroring what the RealSense SDK reports for the *running* profile.
    ``source`` records that these are factory priors: the P0 contract keeps them
    comparable, and only a solve with its own evidence may later supersede them.

    ``valid_min_m`` / ``valid_max_m`` are the range over which the device's depth
    is usable. Zero depth means "no measurement" throughout LeapCalib, exactly as
    in ``RgbdFrame``; it is never a point at the origin.
    """

    device_id: str
    stream: str
    width: int
    height: int
    fx_px: float
    fy_px: float
    ppx_px: float
    ppy_px: float
    depth_scale_m: float
    distortion_model: str = "none"
    distortion_coeffs: tuple[float, ...] = ()
    valid_min_m: float = 0.0
    valid_max_m: float = 0.0
    role: str = ROLE_FIXED
    source: str = SOURCE_FACTORY

    def __post_init__(self) -> None:
        _require_id(self.device_id, "stream profile device")
        _require_id(self.stream, "stream profile name")
        if any(
            type(size) is not int or not 1 <= size <= 16384 for size in (self.width, self.height)
        ):
            raise ValueError("a stream profile needs a bounded positive integer image size")
        for value in (self.distortion_model, self.role, self.source):
            _text(value)
        for name in ("fx_px", "fy_px", "ppx_px", "ppy_px", "depth_scale_m"):
            object.__setattr__(self, name, canonical_number(getattr(self, name)))
        if self.fx_px <= 0 or self.fy_px <= 0:
            raise ValueError("a stream profile needs positive focal lengths")
        if self.depth_scale_m <= 0:
            raise ValueError("depth scale must be positive metres per depth unit")
        object.__setattr__(
            self,
            "distortion_coeffs",
            tuple(canonical_number(c) for c in _array(self.distortion_coeffs, 32)),
        )
        object.__setattr__(self, "valid_min_m", canonical_number(self.valid_min_m))
        object.__setattr__(self, "valid_max_m", canonical_number(self.valid_max_m))
        if self.valid_min_m < 0 or self.valid_max_m < 0:
            raise ValueError("a depth validity range cannot be negative")
        if self.valid_max_m and self.valid_max_m <= self.valid_min_m:
            raise ValueError("a depth validity range must be increasing")
        if self.role not in CAMERA_ROLES:
            raise ValueError(f"camera role must be one of {sorted(CAMERA_ROLES)}")
        if self.source not in PARAMETER_SOURCES:
            raise ValueError(f"profile source must be one of {sorted(PARAMETER_SOURCES)}")

    def contains_pixel(self, u: float, v: float) -> bool:
        return 0.0 <= u <= self.width - 1 and 0.0 <= v <= self.height - 1

    def depth_in_range(self, depth_m: float) -> bool:
        """Whether *depth_m* is a usable measurement for this stream (0 is not)."""
        if depth_m <= 0:
            return False
        if depth_m < self.valid_min_m:
            return False
        return not (self.valid_max_m and depth_m > self.valid_max_m)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "stream": self.stream,
            "width": self.width,
            "height": self.height,
            "fx_px": self.fx_px,
            "fy_px": self.fy_px,
            "ppx_px": self.ppx_px,
            "ppy_px": self.ppy_px,
            "depth_scale_m": self.depth_scale_m,
            "distortion_model": self.distortion_model,
            "distortion_coeffs": list(self.distortion_coeffs),
            "valid_min_m": self.valid_min_m,
            "valid_max_m": self.valid_max_m,
            "role": self.role,
            "source": self.source,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StreamProfile":
        _fields(data, cls)
        coefficients = _array(data.get("distortion_coeffs", ()), 32)
        return cls(
            device_id=data["device_id"],
            stream=data["stream"],
            width=data["width"],
            height=data["height"],
            fx_px=data["fx_px"],
            fy_px=data["fy_px"],
            ppx_px=data["ppx_px"],
            ppy_px=data["ppy_px"],
            depth_scale_m=data["depth_scale_m"],
            distortion_model=data.get("distortion_model", "none"),
            distortion_coeffs=tuple(coefficients),
            valid_min_m=data.get("valid_min_m", 0.0),
            valid_max_m=data.get("valid_max_m", 0.0),
            role=data.get("role", ROLE_FIXED),
            source=data.get("source", SOURCE_FACTORY),
        )

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())


@dataclass(frozen=True)
class MountIdentity:
    """Where a camera is seated, and on which authority.

    ``seated_at_s`` is a wall-clock second recording when the mount was last
    disturbed. It exists so a remount is a fact in the record rather than an
    inference from degraded residuals.
    """

    mount_id: str
    parent_frame: str
    source: str = SOURCE_DECLARED
    seated_at_s: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        _require_id(self.mount_id, "mount")
        _require_id(self.parent_frame, "mount parent frame")
        _text(self.notes)
        _text(self.source)
        if self.source not in PARAMETER_SOURCES:
            raise ValueError(f"mount source must be one of {sorted(PARAMETER_SOURCES)}")
        object.__setattr__(self, "seated_at_s", canonical_number(self.seated_at_s))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mount_id": self.mount_id,
            "parent_frame": self.parent_frame,
            "source": self.source,
            "seated_at_s": self.seated_at_s,
            "notes": self.notes,
        }

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())


@dataclass(frozen=True)
class ToolIdentity:
    """The fitted tool and which point on it a pose refers to."""

    tool_id: str
    reference: str = TOOL_FLANGE
    jaw_stroke_m: float = 0.0
    source: str = SOURCE_DECLARED

    def __post_init__(self) -> None:
        _require_id(self.tool_id, "tool")
        _text(self.reference)
        _text(self.source)
        if self.reference not in TOOL_REFERENCES:
            raise ValueError(f"tool reference must be one of {sorted(TOOL_REFERENCES)}")
        object.__setattr__(self, "jaw_stroke_m", canonical_number(self.jaw_stroke_m))
        if self.jaw_stroke_m < 0:
            raise ValueError("a jaw stroke cannot be negative")
        if self.source not in PARAMETER_SOURCES:
            raise ValueError(f"tool source must be one of {sorted(PARAMETER_SOURCES)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "reference": self.reference,
            "jaw_stroke_m": self.jaw_stroke_m,
            "source": self.source,
        }

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())


@dataclass(frozen=True)
class ModelIdentity:
    """The nominal kinematic model an estimate was formed against.

    ``chain_digest`` is a digest of the declared chain in SI units, computed by
    whoever adapts the declaration -- LeapCalib does not read LeapFlow YAML. It
    pins the geometry that joint corrections are *relative to*: the same offsets
    against a different MDH table are a different physical claim.
    """

    model_id: str
    dof: int
    chain_digest: str
    joint_unit: str = "rad"
    dh_convention: str = "modified"
    source: str = SOURCE_VENDOR_DOCUMENT

    def __post_init__(self) -> None:
        _require_id(self.model_id, "kinematic model")
        _require_id(self.chain_digest, "kinematic chain digest")
        if type(self.dof) is not int or not 1 <= self.dof <= 128:
            raise ValueError("a kinematic model needs a bounded positive integer dof")
        for value in (self.joint_unit, self.dh_convention, self.source):
            _text(value)
        if self.joint_unit != "rad":
            raise ValueError("LeapCalib joint angles are radians")
        if self.source not in PARAMETER_SOURCES:
            raise ValueError(f"model source must be one of {sorted(PARAMETER_SOURCES)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "dof": self.dof,
            "chain_digest": self.chain_digest,
            "joint_unit": self.joint_unit,
            "dh_convention": self.dh_convention,
            "source": self.source,
        }

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())


@dataclass(frozen=True)
class TopologyIdentity:
    """Every identity an estimate depends on, in one comparable record."""

    workcell_id: str
    device: DeviceIdentity
    model: ModelIdentity
    streams: tuple[StreamProfile, ...] = ()
    mounts: tuple[MountIdentity, ...] = ()
    tool: ToolIdentity | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.workcell_id, "workcell")
        if not isinstance(self.device, DeviceIdentity) or not isinstance(self.model, ModelIdentity):
            raise ValueError("typed topology device and nominal model required")
        if self.tool is not None and not isinstance(self.tool, ToolIdentity):
            raise ValueError("typed optional topology tool required")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("topology metadata must be a mapping")
        object.__setattr__(self, "streams", tuple(_array(self.streams, 32)))
        object.__setattr__(self, "mounts", tuple(_array(self.mounts, 64)))
        if any(not isinstance(s, StreamProfile) for s in self.streams) or any(
            not isinstance(m, MountIdentity) for m in self.mounts
        ):
            raise ValueError("typed topology streams and mounts required")
        for name, values in (("stream", self.streams), ("mount", self.mounts)):
            keys = [s.digest for s in values]
            if len(set(keys)) != len(keys):
                raise ValueError(f"duplicate {name} identities in one topology")

    @property
    def components(self) -> dict[str, str]:
        """Digest per named component, which is what comparison and invalidation use."""
        parts = {
            "workcell": digest_of({"workcell_id": self.workcell_id}),
            "device": self.device.digest,
            "model": self.model.digest,
            "tool": self.tool.digest if self.tool else "",
        }
        for stream in self.streams:
            parts[f"stream:{stream.device_id}:{stream.stream}"] = stream.digest
        for mount in self.mounts:
            parts[f"mount:{mount.mount_id}"] = mount.digest
        return parts

    @property
    def digest(self) -> str:
        return digest_of(self.components)

    def differs_from(self, other: "TopologyIdentity") -> tuple[str, ...]:
        """The component names that are not identical between the two topologies.

        An empty tuple means the same declared hardware. A non-empty one names
        what to invalidate instead of forcing a full recalibration.
        """
        mine, theirs = self.components, other.components
        return tuple(
            sorted(
                name
                for name in set(mine) | set(theirs)
                if mine.get(name, "") != theirs.get(name, "")
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workcell_id": self.workcell_id,
            "device": self.device.to_dict(),
            "model": self.model.to_dict(),
            "streams": [s.to_dict() for s in self.streams],
            "mounts": [m.to_dict() for m in self.mounts],
            "tool": self.tool.to_dict() if self.tool else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TopologyIdentity":
        _fields(data, cls)
        streams = _array(data.get("streams", ()), 32)
        mounts = _array(data.get("mounts", ()), 64)
        tool = data.get("tool")
        _fields(data["device"], DeviceIdentity)
        _fields(data["model"], ModelIdentity)
        for mount in mounts:
            _fields(mount, MountIdentity)
        if tool is not None:
            _fields(tool, ToolIdentity)
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("topology metadata must be a mapping")
        return cls(
            workcell_id=data["workcell_id"],
            device=DeviceIdentity(**data["device"]),
            model=ModelIdentity(**data["model"]),
            streams=tuple(StreamProfile.from_mapping(s) for s in streams),
            mounts=tuple(MountIdentity(**m) for m in mounts),
            tool=ToolIdentity(**tool) if tool is not None else None,
            metadata=dict(metadata),
        )


__all__ = [
    "CAMERA_ROLES",
    "PARAMETER_SOURCES",
    "ROLE_FIXED",
    "ROLE_WRIST",
    "SOURCE_DECLARED",
    "SOURCE_ESTIMATED",
    "SOURCE_FACTORY",
    "SOURCE_VENDOR_DOCUMENT",
    "TOOL_CAMERA_CENTRE",
    "TOOL_FINGERTIP_PLANE",
    "TOOL_FLANGE",
    "TOOL_GRASP_REFERENCE",
    "TOOL_INNER_PAD_MIDPOINT",
    "TOOL_REFERENCES",
    "DeviceIdentity",
    "ModelIdentity",
    "MountIdentity",
    "StreamProfile",
    "ToolIdentity",
    "TopologyIdentity",
    "canonical_payload",
    "digest_of",
]
