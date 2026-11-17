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
- **Along the path.** A linear move is sampled every ``reach_step_mm`` (and
  every ``reach_joint_step_deg`` of tool rotation). A sample whose joints jump
  more than ``reach_joint_step_deg`` from the one before is bisected: a
  continuous motion shrinks its jump with the interval, a configuration switch
  does not, and only the latter is a straight line the arm cannot ride. A
  point move needs a solution at its end only; the firmware chooses its joint
  path.
- **Actionable.** An unreachable move names how far along it the arm gets,
  the farthest reachable target and the joint that binds, so the next plan
  can be written to what the arm can do instead of guessed again.
- **Every problem at once.** After a refusal the tracker continues from the
  farthest pose it reached, so a planner can collect each unreachable step of
  a program in one pass instead of discovering them one plan at a time.
- **Unverified, labelled.** An arm without a DH table, or one whose table
  disagrees with the flange it reports, cannot be predicted. Reach is then
  reported unverified rather than refused: this is a feasibility check, the
  safety checks stay where they are, and the firmware still has the last word.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from leapflow.hardware.skills.planning import PlanningCheckpoint

from leapflow.hardware.skills.adapt.ik import solve_ik
from leapflow.hardware.skills.adapt.spec import IkSettings
from leapflow.hardware.skills.geometry import Pose, Transform, interpolate_rotation
from leapflow.hardware.skills.kinematics import KinematicModel
from leapflow.hardware.skills.protocol import FAILURE_UNREACHABLE, SkillError

Bounds = tuple[tuple[float | None, float | None], ...]

# Bisection rounds that refine where along a move the arm stops reaching it:
# a 10 mm sample spacing resolves to well under a millimetre.
_REFINE_ROUNDS = 6
# Bisection rounds that tell a configuration switch from a fast but continuous
# joint motion: a sample's jump has to survive 16-fold subdivision to count.
_CONTINUITY_ROUNDS = 4


@dataclass(frozen=True)
class ReachSettings:
    """Sampling and margins of the reach check, from the workcell's ``limits``."""

    step_mm: float
    joint_step_deg: float
    margin_deg: float
    continuity_rounds: int = _CONTINUITY_ROUNDS
    refine_rounds: int = _REFINE_ROUNDS

    @classmethod
    def from_limits(cls, limits: Mapping[str, float]) -> "ReachSettings":
        return cls(
            step_mm=float(limits["reach_step_mm"]),
            joint_step_deg=float(limits["reach_joint_step_deg"]),
            margin_deg=float(limits["reach_joint_margin_deg"]),
            continuity_rounds=int(limits.get("reach_continuity_rounds", _CONTINUITY_ROUNDS)),
            refine_rounds=int(limits.get("reach_refine_rounds", _REFINE_ROUNDS)),
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


@dataclass(frozen=True)
class Trace:
    """How far a followed move got: the joints at ``reached`` (fraction 0..1).

    ``failed`` means the move stops short of its end; ``branch`` that it stops
    because the next pose needs another arm configuration, not a joint limit.
    """

    q: tuple[float, ...]
    reached: float
    failed: bool = False
    branch: bool = False


class LineFollower:
    """Stateless IK along straight tool-space lines, inside fixed joint bounds."""

    def __init__(
        self,
        chain: KinematicModel,
        bounds: Bounds,
        flange_tcp: Transform,
        settings: ReachSettings,
        checkpoint: PlanningCheckpoint | None = None,
        observe: Callable[[Sequence[float]], None] | None = None,
        ik_settings: IkSettings | None = None,
    ) -> None:
        self._ik_settings = ik_settings or IkSettings()
        self._checkpoint = checkpoint
        self._observe = observe
        self.solves = 0
        self._chain = chain
        self._bounds = tuple(bounds)
        self._tcp_flange = flange_tcp.inverse()
        self.settings = settings

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    def solve(self, pose: Pose, seed: Sequence[float]) -> tuple[float, ...] | None:
        """Joints that put the tool at *pose*, continued from *seed*; None if none found."""
        if self._checkpoint is not None:
            self._checkpoint.check()
        self.solves += 1
        flange = Transform.from_pose(pose).compose(self._tcp_flange)
        try:
            return solve_ik(self._chain, flange, seed, self._bounds, self._ik_settings,
                            checkpoint=None if self._checkpoint is None else self._checkpoint.check)
        except ValueError:
            return None

    def follow(
        self, origin: Pose, target: Pose, seed: tuple[float, ...], *, continuous: bool
    ) -> Trace:
        """Walk the straight line *origin* -> *target* from joints *seed*.

        ``continuous`` also refuses a configuration switch between samples (a
        linear move); without it only a missing solution stops the walk.
        """
        count = self._count(origin, target)
        depth = self.settings.continuity_rounds if continuous else -1
        q = seed
        for index in range(1, count + 1):
            step = self._advance(origin, target, (index - 1) / count, index / count, q, depth)
            if step.failed:
                if step.branch:
                    return step
                return self._refine(origin, target, step, index / count, continuous)
            q = step.q
        return Trace(q, 1.0)

    def _count(self, origin: Pose, target: Pose) -> int:
        by_length = origin.distance_mm(target) / self.settings.step_mm
        by_turn = origin.rotation_to_deg(target) / self.settings.joint_step_deg
        return max(1, math.ceil(max(by_length, by_turn)))

    def _advance(
        self, origin: Pose, target: Pose, low: float, high: float, q: tuple[float, ...], depth: int
    ) -> Trace:
        """From joints *q* at fraction *low* to *high*, bisecting a large joint jump."""
        solved = self.solve(_between(origin, target, high), q)
        if solved is None:
            return Trace(q, low, failed=True)
        if depth < 0 or _jump(q, solved) <= self.settings.joint_step_deg:
            if self._observe is not None:
                self._observe(solved)
            return Trace(solved, high)
        if depth == 0:
            return Trace(q, low, failed=True, branch=True)
        middle = (low + high) / 2.0
        first = self._advance(origin, target, low, middle, q, depth - 1)
        if first.failed:
            return first
        return self._advance(origin, target, middle, high, first.q, depth - 1)

    def _refine(
        self, origin: Pose, target: Pose, stopped: Trace, bad: float, continuous: bool
    ) -> Trace:
        """Bisect between the last reached fraction and the first missed one."""
        good, q = stopped.reached, stopped.q
        for _ in range(self.settings.refine_rounds):
            middle = (good + bad) / 2.0
            step = self._advance(
                origin,
                target,
                good,
                middle,
                q,
                self.settings.continuity_rounds if continuous else -1,
            )
            if step.failed:
                bad = middle
            else:
                good, q = middle, step.q
        return Trace(q, good, failed=True)


def _jump(a: Sequence[float], b: Sequence[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


class ReachTracker:
    """Follows a program's pose targets with IK from the arm's reported joints.

    A refused move leaves the tracker *detached* at the farthest pose it
    reached: the next move is judged by whether its end is reachable from
    there, and re-attaches the tracker once it is, so one planning pass can
    report every step that needs rewriting.
    """

    def __init__(
        self,
        *,
        tcp: Pose,
        chain: KinematicModel | None = None,
        names: Sequence[str] = (),
        joints: Mapping[str, float] | None = None,
        bounds: Bounds = (),
        metric_bounds: Bounds | None = None,
        flange_tcp: Transform | None = None,
        settings: ReachSettings | None = None,
        unverified: str = "",
        checkpoint: PlanningCheckpoint | None = None,
    ) -> None:
        self._checkpoint = checkpoint
        self._bounds = tuple(bounds if metric_bounds is None else metric_bounds)
        self._joint_margin: float | None = None
        self._joint_travel = 0.0
        self._observed_q: tuple[float, ...] | None = None
        self._tcp = tcp
        self._names = tuple(names)
        self._detail = unverified
        self._line: LineFollower | None = None
        self._q: tuple[float, ...] | None = None
        self._anchor: Pose | None = None  # where a refused move stopped
        if chain is not None and joints is not None and settings is not None and not unverified:
            self._line = LineFollower(chain, bounds, flange_tcp or Transform(), settings,
                                      checkpoint, self.observe_joints)
            self._q = tuple(float(joints[name]) for name in self._names)
            self.observe_joints(self._q)

    @classmethod
    def unverified(cls, tcp: Pose, reason: str) -> "ReachTracker":
        """A tracker that checks nothing and says why."""
        return cls(tcp=tcp, unverified=reason or "the arm's kinematics cannot predict reach")

    @property
    def verified(self) -> bool:
        return self._q is not None

    def to_dict(self) -> dict[str, Any]:
        return {"verified": self.verified, "detail": self._detail}

    def metrics(self) -> dict[str, float | None]:
        complete_bounds = bool(self._bounds) and all(
            low is not None and high is not None and high > low for low, high in self._bounds
        )
        return {"joint_margin": self._joint_margin if self.verified and complete_bounds else None,
                "joint_travel": self._joint_travel if self.verified and complete_bounds else None}

    @property
    def ik_calls(self) -> int:
        return 0 if self._line is None else self._line.solves

    def observe_joints(self, joints: Sequence[float]) -> None:
        q = tuple(joints)
        previous = self._observed_q
        for index, (value, (low, high)) in enumerate(zip(q, self._bounds)):
            if low is None or high is None or high <= low:
                continue
            margin = min(value - low, high - value) / (high - low)
            self._joint_margin = margin if self._joint_margin is None else min(self._joint_margin, margin)
            if previous is not None:
                self._joint_travel += abs(value - previous[index]) / (high - low)
        self._observed_q = q

    def move(self, target: Pose, *, linear: bool, label: str) -> None:
        """Follow a move to *target*; raise ``unreachable`` naming how far it gets."""
        if self._checkpoint is not None:
            self._checkpoint.check()
        origin, seed, anchor = self._tcp, self._q, self._anchor
        self._tcp = target
        if seed is None or self._line is None:
            return
        if anchor is not None or not linear:
            solved = self._line.solve(target, seed)
            if solved is not None:
                self._attach(solved)
                return
        if anchor is not None:
            # The previous step stopped short: only the end of this one can be
            # judged, walked from where the arm actually got to.
            origin = anchor
        # A distant seed may fail to converge where a solution exists, so a
        # missed point move walks the straight line to tell "unreachable" from
        # "not found"; a linear move has to ride that line anyway.
        trace = self._line.follow(origin, target, seed, continuous=linear and anchor is None)
        if not trace.failed:
            self._attach(trace.q)
            return
        self._q, self._anchor = trace.q, _between(origin, target, trace.reached)
        self._refuse(origin, target, trace, label, resumed=anchor is not None)

    @property
    def joints(self) -> Mapping[str, float] | None:
        """Predicted joint endpoint after a successful check; None if unverified."""
        return None if self._q is None else dict(zip(self._names, self._q))

    def joints_moved(self, joints: Mapping[str, float], tcp: Pose) -> None:
        """Continue from a joint-space target (missing joints keep their prediction)."""
        self._tcp = tcp
        if self._q is not None:
            self._attach(
                tuple(float(joints.get(name, value)) for name, value in zip(self._names, self._q))
            )

    def _attach(self, q: tuple[float, ...]) -> None:
        self.observe_joints(q)
        self._q, self._anchor = q, None

    # ── Refusal ──

    def _refuse(
        self, origin: Pose, target: Pose, trace: Trace, label: str, *, resumed: bool
    ) -> None:
        good, q = trace.reached, trace.q
        length, turn = origin.distance_mm(target), origin.rotation_to_deg(target)
        progress = _progress(good, length, turn)
        farthest = _between(origin, target, good)
        name, value, limit = self._binding(q)
        where = f"({farthest.x:.0f}, {farthest.y:.0f}, {farthest.z:.0f}) mm"
        start = "from where the previous step stopped, " if resumed else ""
        reach: dict[str, Any] = {
            "requested_mm": round(length, 1),
            "reachable_mm": round(good * length, 1),
            "requested_deg": round(turn, 1),
            "reachable_deg": round(good * turn, 1),
            "farthest": farthest.to_dict(1),
            "joint": name,
            "cause": "configuration_switch" if trace.branch else "joint_limit",
        }
        if trace.branch:
            cause = (
                f"the straight line switches arm configuration after {progress} "
                f"(near {where}); use path point or split the move"
            )
        else:
            cause = (
                f"{start}the arm reaches only {progress} of this move; "
                f"the farthest reachable target is {where}"
            )
            if name:
                cause += f", where {name} is at {value:.1f} deg against its {limit:.1f} deg limit"
            if (
                origin.rotation_to_deg(target) <= 1e-6
                and math.hypot(origin.x - target.x, origin.y - target.y) <= 1e-6
                and target.z > origin.z
            ):
                reach["z_max_mm"] = round(farthest.z, 1)
        raise SkillError(FAILURE_UNREACHABLE, f"{label}: {cause}", result={"reach": reach})

    def _binding(self, q: Sequence[float]) -> tuple[str, float, float]:
        """The joint closest to its (margined) limit, its angle and that limit."""
        bounds = self._line.bounds if self._line is not None else ()
        best: tuple[float, str, float, float] | None = None
        for name, value, (low, high) in zip(self._names, q, bounds):
            for limit in (low, high):
                if limit is None:
                    continue
                gap = abs(value - limit)
                if best is None or gap < best[0]:
                    best = (gap, name, value, limit)
        if best is None:
            return "", 0.0, 0.0
        return best[1], best[2], best[3]


def _progress(fraction: float, length: float, turn: float) -> str:
    """ "X of Y mm", "X of Y deg" or both, for whichever the move actually does."""
    parts = []
    if length >= 0.5:
        parts.append(f"{fraction * length:.0f} of {length:.0f} mm")
    if turn >= 0.5:
        parts.append(f"{fraction * turn:.0f} of {turn:.0f} deg")
    return " and ".join(parts) or "none"


def _between(origin: Pose, target: Pose, fraction: float) -> Pose:
    """The pose *fraction* of the straight line from *origin* to *target*."""
    position = tuple(a + (b - a) * fraction for a, b in zip(origin.position, target.position))
    rotation = interpolate_rotation(origin.rotation, target.rotation, fraction)
    return Transform(rotation, position).to_pose()


__all__ = ["LineFollower", "ReachSettings", "ReachTracker", "Trace", "joint_bounds"]
