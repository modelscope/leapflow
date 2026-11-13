# Copyright (c) Alibaba, Inc. and its affiliates.
"""Plan-time reach: every pose target of a program solved for joints before approval.

The workspace box, the tool cone and the flange envelopes say where the tool
may be; they cannot say whether the arm can take it there. A lift that stays
inside the box can still need a wrist past its limit, and the firmware only
says so once the move is part of a running program -- after the grasp, with an
object in the jaws. ``ReachTracker`` answers at plan time from the arm's
declared kinematics:

- **Seeded.** IK starts from the joints the arm reports and follows the
  program target by target, so each solution is the configuration the
  previous one continues into, not an arbitrary branch.
- **Inside the limits.** A solution stays inside the declared joint limits
  (channel envelopes and ``kinematics.joint_limits``, whichever is narrower)
  shrunk by ``reach_joint_margin_deg``: a firmware refuses a target its own
  IK puts on a limit before it moves.
- **Along the path.** A linear move is sampled every ``reach_step_mm``, and a
  sample whose joints jump more than ``reach_joint_step_deg`` from the one
  before is a configuration switch the straight line cannot ride. A point move
  needs a solution at its end only; the firmware chooses its joint path.
- **Actionable.** An unreachable move names how far along it the arm gets,
  the farthest reachable target and the joint that binds, so the next plan
  can be written to what the arm can do instead of guessed again.
- **Unverified, labelled.** An arm without a DH table, or one whose table
  disagrees with the flange it reports, cannot be predicted. Reach is then
  reported unverified rather than refused: this is a feasibility check, the
  safety checks stay where they are, and the firmware still has the last word.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from leapflow.hardware.skills.adapt.ik import solve_ik
from leapflow.hardware.skills.adapt.spec import IkSettings
from leapflow.hardware.skills.geometry import Pose, Transform, interpolate_rotation
from leapflow.hardware.skills.kinematics import KinematicModel
from leapflow.hardware.skills.protocol import FAILURE_UNREACHABLE, SkillError

Bounds = tuple[tuple[float | None, float | None], ...]

# Bisection rounds that refine where along a move the arm stops reaching it:
# a 10 mm sample spacing resolves to well under a millimetre.
_REFINE_ROUNDS = 6


@dataclass(frozen=True)
class ReachSettings:
    """Sampling and margins of the reach check, from the workcell's ``limits``."""

    step_mm: float
    joint_step_deg: float
    margin_deg: float

    @classmethod
    def from_limits(cls, limits: Mapping[str, float]) -> "ReachSettings":
        return cls(
            step_mm=float(limits["reach_step_mm"]),
            joint_step_deg=float(limits["reach_joint_step_deg"]),
            margin_deg=float(limits["reach_joint_margin_deg"]),
        )


def joint_bounds(arm: Any, names: Sequence[str], margin_deg: float) -> Bounds:
    """Per joint (deg): the narrower of envelope and kinematics limit, less *margin_deg*.

    An unbounded side stays ``None``. A joint whose margined interval is empty
    is pinned to its midpoint, so IK reports it unreachable instead of crashing.
    """
    declared = getattr(getattr(arm, "kinematics", None), "joint_limits", ()) or ()
    out: list[tuple[float | None, float | None]] = []
    for index, name in enumerate(names):
        envelope = arm.channel(name).envelope
        low = envelope.min_value if envelope.declared else None
        high = envelope.max_value if envelope.declared else None
        if index < len(declared):
            low = _narrower(low, math.degrees(declared[index][0]), max)
            high = _narrower(high, math.degrees(declared[index][1]), min)
        low = None if low is None else float(low) + margin_deg
        high = None if high is None else float(high) - margin_deg
        if low is not None and high is not None and low > high:
            low = high = (low + high) / 2.0
        out.append((low, high))
    return tuple(out)


def _narrower(current: float | None, other: float, pick: Any) -> float:
    return other if current is None else pick(float(current), other)


class ReachTracker:
    """Follows a program's pose targets with IK from the arm's reported joints."""

    def __init__(
        self,
        *,
        tcp: Pose,
        chain: KinematicModel | None = None,
        names: Sequence[str] = (),
        joints: Mapping[str, float] | None = None,
        bounds: Bounds = (),
        flange_tcp: Transform | None = None,
        settings: ReachSettings | None = None,
        unverified: str = "",
    ) -> None:
        self._tcp = tcp
        self._chain = chain
        self._names = tuple(names)
        self._bounds = tuple(bounds)
        self._tcp_flange = (flange_tcp or Transform()).inverse()
        self._settings = settings
        self._detail = unverified
        self._q: tuple[float, ...] | None = None
        if chain is not None and joints is not None and settings is not None and not unverified:
            self._q = tuple(float(joints[name]) for name in self._names)

    @classmethod
    def unverified(cls, tcp: Pose, reason: str) -> "ReachTracker":
        """A tracker that checks nothing and says why."""
        return cls(tcp=tcp, unverified=reason or "the arm's kinematics cannot predict reach")

    @property
    def verified(self) -> bool:
        return self._q is not None

    def to_dict(self) -> dict[str, Any]:
        return {"verified": self.verified, "detail": self._detail}

    def move(self, target: Pose, *, linear: bool, label: str) -> None:
        """Follow a move to *target*; raise ``unreachable`` naming how far it gets."""
        origin, seed = self._tcp, self._q
        self._tcp = target
        if seed is None:
            return
        if linear:
            self._q = self._linear(origin, target, seed, label)
            return
        solved = self._solve(target, seed)
        # A distant seed may fail to converge where a solution exists, so a miss
        # walks the straight line to tell "unreachable" from "not found".
        self._q = solved if solved is not None else self._walk(origin, target, seed, label)

    def joints_moved(self, joints: Mapping[str, float], tcp: Pose) -> None:
        """Continue from a joint-space target (missing joints keep their prediction)."""
        self._tcp = tcp
        if self._q is not None:
            self._q = tuple(
                float(joints.get(name, value)) for name, value in zip(self._names, self._q)
            )

    # ── Solving ──

    def _solve(self, pose: Pose, seed: Sequence[float]) -> tuple[float, ...] | None:
        assert self._chain is not None
        flange = Transform.from_pose(pose).compose(self._tcp_flange)
        try:
            return solve_ik(self._chain, flange, seed, self._bounds, IkSettings())
        except ValueError:
            return None

    def _count(self, origin: Pose, target: Pose) -> int:
        settings = self._require_settings()
        by_length = origin.distance_mm(target) / settings.step_mm
        by_turn = origin.rotation_to_deg(target) / settings.joint_step_deg
        return max(1, math.ceil(max(by_length, by_turn)))

    def _require_settings(self) -> ReachSettings:
        assert self._settings is not None
        return self._settings

    def _linear(
        self, origin: Pose, target: Pose, seed: tuple[float, ...], label: str
    ) -> tuple[float, ...]:
        count = self._count(origin, target)
        q = seed
        for index in range(1, count + 1):
            solved = self._solve(_between(origin, target, index / count), q)
            if solved is None or self._jump(q, solved) > self._require_settings().joint_step_deg:
                self._refuse(origin, target, (index - 1) / count, index / count, q, label,
                             branch=solved is not None)
            q = solved  # type: ignore[assignment]
        return q

    def _walk(
        self, origin: Pose, target: Pose, seed: tuple[float, ...], label: str
    ) -> tuple[float, ...]:
        count = self._count(origin, target)
        q = seed
        for index in range(1, count + 1):
            solved = self._solve(_between(origin, target, index / count), q)
            if solved is None:
                self._refuse(origin, target, (index - 1) / count, index / count, q, label,
                             branch=False)
            q = solved  # type: ignore[assignment]
        return q

    @staticmethod
    def _jump(a: Sequence[float], b: Sequence[float]) -> float:
        return max(abs(x - y) for x, y in zip(a, b))

    # ── Refusal ──

    def _refuse(
        self,
        origin: Pose,
        target: Pose,
        good: float,
        bad: float,
        q: tuple[float, ...],
        label: str,
        *,
        branch: bool,
    ) -> None:
        if not branch:
            for _ in range(_REFINE_ROUNDS):
                middle = (good + bad) / 2.0
                solved = self._solve(_between(origin, target, middle), q)
                if solved is None:
                    bad = middle
                else:
                    good, q = middle, solved
        length = origin.distance_mm(target)
        farthest = _between(origin, target, good)
        name, value, limit = self._binding(q)
        where = f"({farthest.x:.0f}, {farthest.y:.0f}, {farthest.z:.0f}) mm"
        if branch:
            cause = (
                f"the straight line switches arm configuration after {good * length:.0f} of "
                f"{length:.0f} mm (near {where}); use path point or split the move"
            )
        else:
            cause = (
                f"the arm reaches only {good * length:.0f} of {length:.0f} mm of this move; "
                f"the farthest reachable target is {where}"
            )
            if name:
                cause += f", where {name} is at {value:.1f} deg against its {limit:.1f} deg limit"
        raise SkillError(
            FAILURE_UNREACHABLE,
            f"{label}: {cause}",
            result={
                "reach": {
                    "requested_mm": round(length, 1),
                    "reachable_mm": round(good * length, 1),
                    "farthest": farthest.to_dict(1),
                    "joint": name,
                    "cause": "configuration_switch" if branch else "joint_limit",
                }
            },
        )

    def _binding(self, q: Sequence[float]) -> tuple[str, float, float]:
        """The joint closest to its (margined) limit, its angle and that limit."""
        best: tuple[float, str, float, float] | None = None
        for name, value, (low, high) in zip(self._names, q, self._bounds):
            for limit in (low, high):
                if limit is None:
                    continue
                gap = abs(value - limit)
                if best is None or gap < best[0]:
                    best = (gap, name, value, limit)
        if best is None:
            return "", 0.0, 0.0
        return best[1], best[2], best[3]


def _between(origin: Pose, target: Pose, fraction: float) -> Pose:
    """The pose *fraction* of the straight line from *origin* to *target*."""
    position = tuple(a + (b - a) * fraction for a, b in zip(origin.position, target.position))
    rotation = interpolate_rotation(origin.rotation, target.rotation, fraction)
    return Transform(rotation, position).to_pose()


__all__ = ["ReachSettings", "ReachTracker", "joint_bounds"]
