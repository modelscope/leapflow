# Copyright (c) Alibaba, Inc. and its affiliates.
"""Typed frame diagnoses: why a calibrated frame does or does not hold.

A frame is either usable or it is not, but "not usable" has several causes and
each one has a different remedy: a frame nobody measured needs a measurement, a
frame measured under older evidence rules needs only an independent check, and
a frame that is merely less accurate than the task needs can be tightened
without starting over. Collapsing them into one "stale" string is what made a
calibrated, untouched camera look as if it had to be calibrated again.

Pure data and stdlib only, so builtin skills may import it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class FrameStatus(str, Enum):
    """Why one frame does (``VALID``) or does not hold in a snapshot."""

    VALID = "valid"
    MISSING = "missing"
    DEPENDENCY_SUPERSEDED = "dependency_superseded"
    LEGACY_EVIDENCE = "legacy_evidence"
    UNPROVEN = "unproven"
    INSUFFICIENT_ACCURACY = "insufficient_accuracy"
    DRIFT_DETECTED = "drift_detected"


FRAME_STATUSES = frozenset(status.value for status in FrameStatus)
INVALID_FRAME_STATUSES = frozenset(FRAME_STATUSES - {FrameStatus.VALID.value})


@dataclass(frozen=True)
class Accuracy:
    """A frame's positional (mm) and, optionally, angular (deg) error bound."""

    translation_mm: float
    rotation_deg: float | None = None

    def within(self, required: "Accuracy") -> bool:
        """True when this bound is no worse than *required* on every axis it names."""
        if self.translation_mm > required.translation_mm:
            return False
        if required.rotation_deg is None or self.rotation_deg is None:
            return True
        return self.rotation_deg <= required.rotation_deg

    def plus(self, other: "Accuracy | None") -> "Accuracy":
        """Errors stacked along a chain (a conservative, additive bound)."""
        if other is None:
            return self
        rotation = (
            None
            if self.rotation_deg is None and other.rotation_deg is None
            else (self.rotation_deg or 0.0) + (other.rotation_deg or 0.0)
        )
        return Accuracy(self.translation_mm + other.translation_mm, rotation)

    def describe(self) -> str:
        text = f"{self.translation_mm:.1f} mm"
        if self.rotation_deg is not None:
            text += f" / {self.rotation_deg:.2f} deg"
        return text

    def to_dict(self) -> dict[str, float]:
        out = {"translation_mm": round(self.translation_mm, 3)}
        if self.rotation_deg is not None:
            out["rotation_deg"] = round(self.rotation_deg, 4)
        return out

    @classmethod
    def from_mapping(cls, value: Any) -> "Accuracy | None":
        """Parse ``{"translation_mm": .., "rotation_deg": ..}``; ``None`` when unusable."""
        if not isinstance(value, Mapping):
            return None
        translation = _positive(value.get("translation_mm"))
        if translation is None:
            return None
        return cls(translation, _positive(value.get("rotation_deg")))


@dataclass(frozen=True)
class FrameCandidate:
    """A measured-but-not-installed estimate of one frame, awaiting promotion."""

    digest: str
    frame: str
    method: str
    matrix: tuple[tuple[float, ...], ...]
    achieved: Accuracy | None
    created_at: float
    dependency: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "frame": self.frame,
            "method": self.method,
            "achieved": self.achieved.to_dict() if self.achieved else None,
            "created_at": self.created_at,
            "dependency": dict(self.dependency),
        }


@dataclass(frozen=True)
class FrameDiagnosis:
    """One frame's status in a snapshot, with the evidence behind it."""

    frame: str
    status: FrameStatus
    reason: str = ""
    version: float | None = None
    method: str = ""
    achieved: Accuracy | None = None
    required: Accuracy | None = None
    candidates: tuple[FrameCandidate, ...] = ()
    verified_by: str = ""

    @property
    def ok(self) -> bool:
        return self.status is FrameStatus.VALID

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"frame": self.frame, "status": self.status.value}
        if self.reason:
            out["reason"] = self.reason
        if self.version is not None:
            out["version"] = self.version
        if self.method:
            out["method"] = self.method
        if self.achieved is not None:
            out["achieved"] = self.achieved.to_dict()
        if self.required is not None:
            out["required"] = self.required.to_dict()
        if self.verified_by:
            out["verified_by"] = self.verified_by
        if self.candidates:
            out["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return out


def _positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


__all__ = [
    "FRAME_STATUSES",
    "INVALID_FRAME_STATUSES",
    "Accuracy",
    "FrameCandidate",
    "FrameDiagnosis",
    "FrameStatus",
]
