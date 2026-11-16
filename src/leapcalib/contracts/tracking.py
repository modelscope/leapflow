# Copyright (c) Alibaba, Inc. and its affiliates.
"""Import-light lossless metric ROI contracts for the private compute worker."""

from __future__ import annotations

import base64
import binascii
import math
from dataclasses import dataclass
from typing import Any, Mapping

from leapcalib.contracts.identity import StreamProfile, digest_of
from leapcalib.contracts.observations import CLOCK_NODE_MONOTONIC, STAMP_RECEIPT, Stamp
from leapcalib.contracts.units import canonical_number

ROI_SCHEMA = "leapcalib.metric_roi.v1"
TRACK_SCHEMA = "leapcalib.tracking.v1"


def _id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("bounded tracking identities required")
    return value


def _buffer(value: Any, size: int) -> bytes:
    # Check encoded length BEFORE decoding/allocating the lossless buffer.
    if not isinstance(value, str) or len(value) != 4 * ((size + 2) // 3):
        raise ValueError("lossless ROI buffer length mismatch")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("ROI is not canonical base64") from exc
    if len(decoded) != size or base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError("lossless ROI byte count/encoding mismatch")
    return decoded


@dataclass(frozen=True)
class SourceStamp:
    """Native stream clock provenance, never comparable to node receipt seconds."""

    stream: str
    frame_number: int
    timestamp_ms: float
    timestamp_domain: str

    def __post_init__(self) -> None:
        if (
            self.stream not in ("color", "depth")
            or type(self.frame_number) is not int
            or self.frame_number <= 0
        ):
            raise ValueError("native stream and increasing source frame identity required")
        _id(self.timestamp_domain)
        number = canonical_number(self.timestamp_ms)
        if number < 0:
            raise ValueError("nonnegative native source timestamp required")
        object.__setattr__(self, "timestamp_ms", number)


@dataclass(frozen=True)
class MetricRoi:
    """BGR8 and IEEE754 float32-LE metric depth; no preview JPEG or native endianness."""

    frame_id: str
    profile: StreamProfile
    mount_digest: str
    session_id: str
    source_profile_digest: str
    stamp: Stamp
    exposure_uncertainty_s: float
    x: int
    y: int
    width: int
    height: int
    color_bgr8: bytes
    depth_m_f32le: bytes
    provenance: str
    source_stamps: tuple[SourceStamp, ...] = ()

    def __post_init__(self) -> None:
        for name in ("frame_id", "mount_digest", "session_id", "source_profile_digest"):
            _id(getattr(self, name))
        if not isinstance(self.profile, StreamProfile) or not isinstance(self.stamp, Stamp):
            raise ValueError("typed exact profile and receipt stamp required")
        if self.stamp.domain != CLOCK_NODE_MONOTONIC or self.stamp.semantics != STAMP_RECEIPT:
            raise ValueError("tracking uses node receipt time, never an inferred exposure clock")
        if self.provenance not in ("metric_rgbd", "synthetic"):
            raise ValueError("tracking requires metric RGB-D or explicitly synthetic test input")
        object.__setattr__(self, "source_stamps", tuple(self.source_stamps))
        if any(not isinstance(s, SourceStamp) for s in self.source_stamps):
            raise ValueError("typed native stream clock provenance required")
        if self.provenance == "metric_rgbd" or self.source_stamps:
            if tuple(s.stream for s in self.source_stamps) != ("color", "depth"):
                raise ValueError("metric input needs exact color and depth source clock provenance")
        uncertainty = canonical_number(self.exposure_uncertainty_s)
        if uncertainty <= 0 or self.stamp.uncertainty_s <= 0:
            raise ValueError("nonzero receipt/exposure timing floors are required")
        object.__setattr__(self, "exposure_uncertainty_s", uncertainty)
        if any(type(v) is not int for v in (self.x, self.y, self.width, self.height)):
            raise ValueError("integer ROI coordinates required")
        if not 8 <= self.width <= 640 or not 8 <= self.height <= 480:
            raise ValueError("lossless ROI exceeds 640x480 compute budget")
        if (
            self.x < 0
            or self.y < 0
            or self.x + self.width > self.profile.width
            or self.y + self.height > self.profile.height
        ):
            raise ValueError("ROI lies outside exact metric profile")
        pixels = self.width * self.height
        if not isinstance(self.color_bgr8, bytes) or len(self.color_bgr8) != pixels * 3:
            raise ValueError("BGR8 ROI byte count mismatch")
        if not isinstance(self.depth_m_f32le, bytes) or len(self.depth_m_f32le) != pixels * 4:
            raise ValueError("metric float32-LE ROI byte count mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ROI_SCHEMA,
            **vars(self),
            "profile": self.profile.to_dict(),
            "stamp": self.stamp.to_dict(),
            "source_stamps": [vars(s) for s in self.source_stamps],
            "color_bgr8": base64.b64encode(self.color_bgr8).decode("ascii"),
            "depth_m_f32le": base64.b64encode(self.depth_m_f32le).decode("ascii"),
        }

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, max_width: int = 640, max_height: int = 480
    ) -> MetricRoi:
        if set(data) != {"schema", *cls.__dataclass_fields__} or data["schema"] != ROI_SCHEMA:
            raise ValueError("metric ROI schema/fields mismatch")
        width, height = data["width"], data["height"]
        if (
            type(width) is not int
            or type(height) is not int
            or not 8 <= width <= max_width
            or not 8 <= height <= max_height
        ):
            raise ValueError("ROI dimensions exceed budget before buffer allocation")
        profile = data["profile"]
        if not isinstance(profile, Mapping) or set(profile) != set(
            StreamProfile.__dataclass_fields__
        ):
            raise ValueError("exact stream profile fields required")
        if type(profile["width"]) is not int or type(profile["height"]) is not int:
            raise ValueError("integer exact-profile dimensions required")
        stamp = data["stamp"]
        if not isinstance(stamp, Mapping) or set(stamp) != set(Stamp.__dataclass_fields__):
            raise ValueError("exact receipt stamp fields required")
        sources = data["source_stamps"]
        if (
            not isinstance(sources, (list, tuple))
            or len(sources) > 2
            or any(
                not isinstance(s, Mapping) or set(s) != set(SourceStamp.__dataclass_fields__)
                for s in sources
            )
        ):
            raise ValueError("bounded exact native source clock fields required")
        values = {k: v for k, v in data.items() if k != "schema"}
        values.update(
            profile=StreamProfile.from_mapping(profile),
            stamp=Stamp.from_mapping(stamp),
            source_stamps=tuple(SourceStamp(**s) for s in sources),
            color_bgr8=_buffer(data["color_bgr8"], width * height * 3),
            depth_m_f32le=_buffer(data["depth_m_f32le"], width * height * 4),
        )
        return cls(**values)


@dataclass(frozen=True)
class TrackingConfig:
    """Numerical support/refusal bounds, not a physical accuracy certificate."""

    min_tracks: int = 12
    max_tracks: int = 256
    min_spread_fraction: float = 0.2
    min_distance_px: float = 8.0
    quality_level: float = 0.02
    fb_gate_px: float = 0.5
    patch_correlation_min: float = 0.9
    ambiguity_correlation_max: float = 0.98
    depth_floor_m: float = 0.0005
    max_depth_mad_m: float = 0.002
    point_gate_m: float = 0.003
    min_inlier_ratio: float = 0.8
    max_yaw_rad: float = math.radians(5)
    max_frame_gap_s: float = 2.0

    def __post_init__(self) -> None:
        if (
            type(self.min_tracks) is not int
            or type(self.max_tracks) is not int
            or not 12 <= self.min_tracks <= self.max_tracks <= 2048
        ):
            raise ValueError("tracking needs 12..2048 bounded persistent features")
        for name in set(self.__dataclass_fields__) - {"min_tracks", "max_tracks"}:
            number = canonical_number(getattr(self, name))
            if number <= 0:
                raise ValueError("positive tracking thresholds/floors required")
            object.__setattr__(self, name, number)
        if (
            not self.min_spread_fraction <= 1
            or not self.min_inlier_ratio <= 1
            or not self.quality_level < 1
        ):
            raise ValueError("invalid tracking fractions")
        if not 0 < self.patch_correlation_min < self.ambiguity_correlation_max < 1:
            raise ValueError("patch acceptance/ambiguity thresholds must be ordered")


@dataclass(frozen=True)
class SceneConfig:
    """Disjoint support comparison policy, not proof a background is physically static."""

    max_translation_m: float = 0.003
    max_rotation_rad: float = math.radians(2)
    confidence_multiplier: float = 2.0
    min_separation_px: float = 21.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, canonical_number(getattr(self, name)))
        if not 0 < self.max_translation_m <= 0.003 or not 0 < self.max_rotation_rad <= math.radians(
            2
        ):
            raise ValueError("scene comparison cannot widen study motion limits")
        if not 1 <= self.confidence_multiplier <= 10 or not 21 <= self.min_separation_px <= 640:
            raise ValueError(
                "scene comparison needs conservative floors and disjoint local LK windows"
            )
