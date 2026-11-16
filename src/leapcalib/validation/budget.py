# Copyright (c) Alibaba, Inc. and its affiliates.
"""The error budget: what accuracy a task needs, and where the error will come from.

A calibration is not "good" or "bad" in the abstract; it either fits a task's
margin or it does not. Writing the budget down first is what keeps a later result
from being graded against whatever it happened to achieve.

Two contributions dominate and both are easy to under-count:

**Observation age.** A pose observed while the arm is moving describes where the
arm *was*. To first order the error is ``(v + r * omega) * tau``: at 20 mm/s with
a 100 ms old observation that is already 2 mm of translation alone, before any
camera, tool or model error. The remedy is lower speed or fresher data; there is
no correction for not knowing when the sample was taken.

**Lever arms.** Angular error becomes positional error in proportion to distance.
One degree at 130 mm is about 2.3 mm -- which is why pixel-level visual agreement
cannot certify a millimetre-level grasp, and why orientation terms need the lever
arm they act through stated alongside them.

Combination is explicit. Summing independent terms linearly is pessimistic;
combining them in quadrature requires them to actually be independent, and
systematic terms never are. :data:`COMBINE_WORST_CASE` is the default because
claiming independence is a claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from leapcalib.contracts.failures import FAILURE_BUDGET_EXCEEDED, CalibrationFailure
from leapcalib.contracts.units import canonical_number

TERM_RANDOM = "random"
"""Varies sample to sample (sensor noise, repeatability). May combine in quadrature."""
TERM_SYSTEMATIC = "systematic"
"""A bias that repeats (model error, depth offset, timing). Always adds linearly."""
TERM_KINDS = frozenset({TERM_RANDOM, TERM_SYSTEMATIC})

COMBINE_WORST_CASE = "worst_case"
"""Every term adds linearly. The default: it assumes nothing about correlation."""
COMBINE_INDEPENDENT = "independent"
"""Random terms combine in quadrature, systematic terms still add linearly."""
COMBINATIONS = frozenset({COMBINE_WORST_CASE, COMBINE_INDEPENDENT})


def delay_error_m(
    *, speed_m_s: float, rotation_rate_rad_s: float, lever_arm_m: float, latency_s: float
) -> float:
    """First-order position error from acting on an observation of age *latency_s*.

    ``(v + r * omega) * tau``. The rotational part needs the lever arm because a
    wrist rotating at a given rate displaces a fingertip in proportion to its
    distance from the axis.
    """
    speed = canonical_number(speed_m_s)
    rate = canonical_number(rotation_rate_rad_s)
    lever = canonical_number(lever_arm_m)
    latency = canonical_number(latency_s)
    if min(speed, rate, lever, latency) < 0:
        raise ValueError("speeds, lever arms and latencies are nonnegative magnitudes")
    return (speed + lever * rate) * latency


def lever_arm_error_m(*, angle_error_rad: float, lever_arm_m: float) -> float:
    """Position error an orientation error produces at a distance.

    Uses ``r * theta`` rather than the exact chord ``2 * r * sin(theta / 2)``:
    the arc is the larger of the two, so the budget errs toward refusing a
    marginal task rather than accepting one.
    """
    angle = abs(canonical_number(angle_error_rad))
    lever = canonical_number(lever_arm_m)
    if lever < 0:
        raise ValueError("a lever arm is a nonnegative distance")
    return lever * angle


@dataclass(frozen=True)
class ErrorTerm:
    """One named contribution to a position error, in metres."""

    name: str
    value_m: float
    kind: str = TERM_SYSTEMATIC
    basis: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("an error term needs a name")
        object.__setattr__(self, "value_m", canonical_number(self.value_m))
        if self.value_m < 0:
            raise ValueError("an error magnitude cannot be negative")
        if self.kind not in TERM_KINDS:
            raise ValueError(f"term kind must be one of {sorted(TERM_KINDS)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value_m": self.value_m,
            "kind": self.kind,
            "basis": self.basis,
        }


@dataclass(frozen=True)
class TaskTolerance:
    """What a task can actually absorb, and where that number came from.

    ``basis`` is required: a tolerance with no stated origin is a wish. It should
    name the physical margin it was derived from -- jaw opening minus object
    width, approach clearance, insertion chamfer.
    """

    task_id: str
    allowed_m: float
    allowed_rad: float = 0.0
    basis: str = ""

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("a tolerance needs a task id")
        object.__setattr__(self, "allowed_m", canonical_number(self.allowed_m))
        object.__setattr__(self, "allowed_rad", canonical_number(self.allowed_rad))
        if self.allowed_m <= 0:
            raise ValueError("an allowed position error must be positive")
        if self.allowed_rad < 0:
            raise ValueError("an allowed orientation error cannot be negative")
        if not self.basis.strip():
            raise ValueError("a tolerance must state the physical margin it came from")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "allowed_m": self.allowed_m,
            "allowed_rad": self.allowed_rad,
            "basis": self.basis,
        }


@dataclass(frozen=True)
class BudgetVerdict:
    """Whether a budget fits a tolerance, and which term decides it."""

    fits: bool
    total_m: float
    allowed_m: float
    dominant_term: str
    combination: str

    @property
    def headroom_m(self) -> float:
        """Metres of margin left; negative when the budget overruns."""
        return self.allowed_m - self.total_m

    def to_dict(self) -> dict[str, Any]:
        return {
            "fits": self.fits,
            "total_m": self.total_m,
            "allowed_m": self.allowed_m,
            "headroom_m": self.headroom_m,
            "dominant_term": self.dominant_term,
            "combination": self.combination,
        }


@dataclass(frozen=True)
class ErrorBudget:
    """Named error terms and the rule for adding them up."""

    terms: tuple[ErrorTerm, ...]
    combination: str = COMBINE_WORST_CASE

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        names = [t.name for t in self.terms]
        if len(set(names)) != len(names):
            raise ValueError("error terms must be uniquely named")
        if self.combination not in COMBINATIONS:
            raise ValueError(f"combination must be one of {sorted(COMBINATIONS)}")

    @property
    def total_m(self) -> float:
        """The combined error under the declared combination rule."""
        if self.combination == COMBINE_WORST_CASE:
            return sum(t.value_m for t in self.terms)
        random = math.sqrt(sum(t.value_m**2 for t in self.terms if t.kind == TERM_RANDOM))
        systematic = sum(t.value_m for t in self.terms if t.kind == TERM_SYSTEMATIC)
        return random + systematic

    @property
    def dominant_term(self) -> str:
        """The largest single contribution, which is where effort pays off."""
        return max(self.terms, key=lambda t: t.value_m).name if self.terms else ""

    def with_term(self, term: ErrorTerm) -> "ErrorBudget":
        return ErrorBudget(terms=self.terms + (term,), combination=self.combination)

    def check(self, tolerance: TaskTolerance) -> BudgetVerdict:
        """Compare the budget against *tolerance* without raising."""
        total = self.total_m
        return BudgetVerdict(
            fits=total <= tolerance.allowed_m,
            total_m=total,
            allowed_m=tolerance.allowed_m,
            dominant_term=self.dominant_term,
            combination=self.combination,
        )

    def require_fit(self, tolerance: TaskTolerance) -> BudgetVerdict:
        """Like :meth:`check`, but refuses a task whose budget does not close."""
        verdict = self.check(tolerance)
        if not verdict.fits:
            raise CalibrationFailure(
                FAILURE_BUDGET_EXCEEDED,
                f"{tolerance.task_id} allows {tolerance.allowed_m * 1e3:.2f} mm but the "
                f"budget totals {verdict.total_m * 1e3:.2f} mm "
                f"(largest term: {verdict.dominant_term})",
            )
        return verdict

    def to_dict(self) -> dict[str, Any]:
        return {
            "terms": [t.to_dict() for t in self.terms],
            "combination": self.combination,
            "total_m": self.total_m,
            "dominant_term": self.dominant_term,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ErrorBudget":
        return cls(
            terms=tuple(
                ErrorTerm(
                    name=str(t["name"]),
                    value_m=float(t["value_m"]),
                    kind=str(t.get("kind") or TERM_SYSTEMATIC),
                    basis=str(t.get("basis") or ""),
                )
                for t in data.get("terms") or ()
            ),
            combination=str(data.get("combination") or COMBINE_WORST_CASE),
        )


def observation_delay_budget(
    *,
    speed_m_s: float,
    rotation_rate_rad_s: float,
    lever_arm_m: float,
    latency_s: float,
    orientation_error_rad: float = 0.0,
    extra_terms: Sequence[ErrorTerm] = (),
) -> ErrorBudget:
    """The budget for acting on a delayed observation, as named terms.

    Latency and orientation error are systematic: repeating the motion repeats
    them, so no amount of averaging removes either.
    """
    terms = [
        ErrorTerm(
            name="observation_delay",
            value_m=delay_error_m(
                speed_m_s=speed_m_s,
                rotation_rate_rad_s=rotation_rate_rad_s,
                lever_arm_m=lever_arm_m,
                latency_s=latency_s,
            ),
            kind=TERM_SYSTEMATIC,
            basis=f"(v + r*omega) * tau with tau={latency_s:g} s",
        )
    ]
    if orientation_error_rad:
        terms.append(
            ErrorTerm(
                name="orientation_lever_arm",
                value_m=lever_arm_error_m(
                    angle_error_rad=orientation_error_rad, lever_arm_m=lever_arm_m
                ),
                kind=TERM_SYSTEMATIC,
                basis=f"r*theta at r={lever_arm_m:g} m",
            )
        )
    return ErrorBudget(terms=tuple(terms) + tuple(extra_terms))


__all__ = [
    "COMBINATIONS",
    "COMBINE_INDEPENDENT",
    "COMBINE_WORST_CASE",
    "TERM_KINDS",
    "TERM_RANDOM",
    "TERM_SYSTEMATIC",
    "BudgetVerdict",
    "ErrorBudget",
    "ErrorTerm",
    "TaskTolerance",
    "delay_error_m",
    "lever_arm_error_m",
    "observation_delay_budget",
]
