# Copyright (c) Alibaba, Inc. and its affiliates.
"""SI canon: metres, radians, seconds, and wire fields that name their own unit.

LeapFlow's hardware layer speaks the units its firmware and declarations speak --
millimetres for ``ee.x``, degrees for ``joint.j1.position``, millimetres again for
a calibration certificate (``units: "mm"``,
``orientation_convention: "Rz@Ry@Rx;degrees"``). LeapCalib's numerics speak
metres and radians, because a solver that mixes a 300 mm link with a 0.02 rad
offset in one parameter vector conditions its own Jacobian wrongly.

Both conventions are correct in their own layer, so the conversion is explicit
and lives at the boundary (``leapflow.hardware.calibration.adapter``) rather than
being inferred. Two rules make that boundary auditable:

* A quantity crossing a wire carries its unit. :class:`Quantity` carries it as a
  field; a plain numeric field carries it in its *name* (``depth_m``,
  ``latency_s``, ``offset_rad``). Nothing guesses a unit from magnitude -- 0.3
  is a plausible metre count and a plausible millimetre count, and a wrong guess
  is a 1000x geometry error that no residual check would attribute to parsing.
* A value that reaches a digest is canonical. ``-0.0`` and ``0.0`` compare equal
  but serialise differently, which would give one geometry two digests; NaN and
  infinity are refused outright rather than hashed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

UNIT_METRE = "m"
UNIT_RADIAN = "rad"
UNIT_SECOND = "s"
UNIT_MILLIMETRE = "mm"
UNIT_DEGREE = "deg"
UNIT_MILLISECOND = "ms"
UNIT_PIXEL = "px"

SI_UNITS = frozenset({UNIT_METRE, UNIT_RADIAN, UNIT_SECOND})
"""The only units LeapCalib stores, solves in, or reports."""

# unit -> (factor onto its SI unit, that SI unit). Pixels are deliberately absent:
# a pixel is not convertible without intrinsics, so a pixel quantity stays in its
# own field rather than being silently metricated here.
_TO_SI: dict[str, tuple[float, str]] = {
    UNIT_METRE: (1.0, UNIT_METRE),
    UNIT_MILLIMETRE: (1e-3, UNIT_METRE),
    UNIT_RADIAN: (1.0, UNIT_RADIAN),
    UNIT_DEGREE: (math.pi / 180.0, UNIT_RADIAN),
    UNIT_SECOND: (1.0, UNIT_SECOND),
    UNIT_MILLISECOND: (1e-3, UNIT_SECOND),
}

KNOWN_UNITS = frozenset(_TO_SI) | {UNIT_PIXEL}

# Numeric wire fields end with one of these, so a reader never has to guess.
# Checked by the schema invariant test, not at every construction: the point is
# that the schema cannot grow an unnamed numeric field unnoticed.
UNIT_SUFFIXES: dict[str, str] = {
    "_m": UNIT_METRE,
    "_rad": UNIT_RADIAN,
    "_s": UNIT_SECOND,
    "_mm": UNIT_MILLIMETRE,
    "_deg": UNIT_DEGREE,
    "_ms": UNIT_MILLISECOND,
    "_px": UNIT_PIXEL,
}

# A field whose entire name is a unit word already names its unit, so it needs no
# suffix. Matched exactly rather than as a suffix on purpose: "milliseconds" ends
# with "seconds", and reading that as seconds would be a 1000x error dressed up
# as a passing convention check.
UNIT_WORDS: dict[str, str] = {
    "seconds": UNIT_SECOND,
    "metres": UNIT_METRE,
    "radians": UNIT_RADIAN,
}


def mm_to_m(value: float) -> float:
    """Millimetres (LeapFlow hardware, vendor manuals) to metres (LeapCalib)."""
    return canonical_number(value) * 1e-3


def m_to_mm(value: float) -> float:
    """Metres back to the millimetres a LeapFlow certificate or channel expects."""
    return canonical_number(value) * 1e3


def deg_to_rad(value: float) -> float:
    """Degrees (joint channels, DH declarations) to radians (LeapCalib)."""
    return math.radians(canonical_number(value))


def rad_to_deg(value: float) -> float:
    """Radians back to the degrees a LeapFlow joint channel expects."""
    return math.degrees(canonical_number(value))


def to_si(value: float, unit: str) -> tuple[float, str]:
    """*value* expressed in the SI unit of its dimension, with that unit named.

    Raises ``ValueError`` for an unknown unit and for pixels, which need
    intrinsics rather than a factor.
    """
    try:
        factor, si_unit = _TO_SI[unit]
    except KeyError:
        raise ValueError(f"unit {unit!r} has no declared SI conversion") from None
    return (canonical_number(value) * factor, si_unit)


def canonical_number(value: float) -> float:
    """*value* as a finite float with a single zero representation.

    Refuses NaN and infinity: a nonfinite number in a parameter vector, an
    uncertainty, or a digest payload is a defect to surface, not a value to
    propagate into geometry that safety checks will later consume.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("values must be finite")
    return 0.0 if number == 0.0 else number


def unit_of_field(name: str) -> str | None:
    """The unit *name* declares through its suffix or its whole name, else ``None``.

    Used by the schema invariant test to hold the "name your unit" rule; a
    dimensionless field (a ratio, a count, an identifier) legitimately returns
    ``None`` and is listed explicitly there.
    """
    word = UNIT_WORDS.get(name)
    if word is not None:
        return word
    for suffix, unit in UNIT_SUFFIXES.items():
        if name.endswith(suffix):
            return unit
    return None


@dataclass(frozen=True)
class Quantity:
    """One scalar and the unit it is in, for a field whose unit can vary.

    A fixed-unit field should name its unit instead (``depth_m``); this type is
    for values whose unit is part of the data -- a declared tolerance quoted in
    whatever unit its source used, for instance.
    """

    value: float
    unit: str

    def __post_init__(self) -> None:
        if self.unit not in KNOWN_UNITS:
            raise ValueError(f"unknown unit {self.unit!r}")
        object.__setattr__(self, "value", canonical_number(self.value))

    @property
    def si(self) -> "Quantity":
        """The same quantity in its SI unit; pixels have no SI form and raise."""
        value, unit = to_si(self.value, self.unit)
        return Quantity(value, unit)

    def to_dict(self) -> dict[str, object]:
        return {"value": self.value, "unit": self.unit}


__all__ = [
    "KNOWN_UNITS",
    "SI_UNITS",
    "UNIT_DEGREE",
    "UNIT_METRE",
    "UNIT_MILLIMETRE",
    "UNIT_MILLISECOND",
    "UNIT_PIXEL",
    "UNIT_RADIAN",
    "UNIT_SECOND",
    "UNIT_SUFFIXES",
    "UNIT_WORDS",
    "Quantity",
    "canonical_number",
    "deg_to_rad",
    "m_to_mm",
    "mm_to_m",
    "rad_to_deg",
    "to_si",
    "unit_of_field",
]
