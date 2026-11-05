# Copyright (c) Alibaba, Inc. and its affiliates.
"""Capture groups: one observation from several cameras, soft-synchronised.

A workcell with an overhead camera and a wrist camera asks one question per step
-- "what does the cell look like *now*?" -- and two independent streams answer it
at two different instants. Without a trigger cable the best available guarantee
is a *timestamp* match: capture every member against one shared freshness bound,
measure how far apart the answers landed, and re-capture the laggards a bounded
number of times when the spread exceeds a tolerance.

The group is vendor-neutral. A member is any async callable taking the shared
monotonic bound and returning a frame with a ``monotonic_at`` stamp, so a
RealSense hub, a UVC grabber, or a ROS image subscription join the same group.
Hardware-synchronised rigs report ``SYNC_HARDWARE`` from their producer and need
no re-capture; this module is the honest fallback, and says so in its result
(``sync = "timestamp"``, ``skew_ms``, ``within_tolerance``) rather than implying
the frames were exposed together.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from leapflow.hardware.vision.frames import SYNC_TIMESTAMP, accepts_keyword

GroupMember = Callable[[float], Awaitable[Any]]
"""Capture one frame newer than the given monotonic instant (this process's clock)."""

DEFAULT_SYNC_TOLERANCE_MS = 50.0
DEFAULT_SYNC_ATTEMPTS = 3


@dataclass(frozen=True)
class GroupCapture:
    """The frames one group capture produced, with its synchronisation evidence.

    ``errors`` holds members that produced nothing; a member whose *re-capture*
    failed keeps its first frame and is not listed. ``skew_ms`` spans only the
    members that produced a frame.
    """

    frames: Mapping[str, Any]
    errors: Mapping[str, Exception]
    stamps: Mapping[str, float]
    skew_ms: float
    tolerance_ms: float
    attempts: int
    sync: str = SYNC_TIMESTAMP
    bound: float = 0.0
    recaptured: tuple[str, ...] = field(default=())

    @property
    def within_tolerance(self) -> bool:
        return self.skew_ms <= self.tolerance_ms

    def summary(self) -> dict[str, Any]:
        """JSON-safe synchronisation evidence, without the frames themselves."""
        return {
            "sync": self.sync,
            "skew_ms": round(self.skew_ms, 3),
            "tolerance_ms": self.tolerance_ms,
            "within_tolerance": self.within_tolerance,
            "attempts": self.attempts,
            "members": sorted(self.frames),
            "failed": sorted(self.errors),
            "recaptured": list(self.recaptured),
        }


class CaptureGroup:
    """Capture every member against one bound, then pull laggards into tolerance."""

    def __init__(
        self,
        members: Mapping[str, GroupMember],
        *,
        tolerance_ms: float = DEFAULT_SYNC_TOLERANCE_MS,
        attempts: int = DEFAULT_SYNC_ATTEMPTS,
    ) -> None:
        if not members:
            raise ValueError("a capture group needs at least one member")
        self._members = dict(members)
        self._tolerance_ms = max(0.0, float(tolerance_ms))
        self._attempts = max(1, int(attempts))

    async def capture(self, *, after_monotonic: float | None = None) -> GroupCapture:
        """One soft-synchronised observation.

        Every member runs concurrently, so a slow camera costs its own latency, not
        the sum. A member failure is collected, not raised: the caller decides
        whether a partial observation is usable. Cancellation still propagates.
        """
        bound = time.monotonic() if after_monotonic is None else float(after_monotonic)
        frames, errors = await self._gather({name: bound for name in self._members})
        tolerance_s = self._tolerance_ms / 1000.0
        attempts = 1
        recaptured: list[str] = []
        while attempts < self._attempts:
            stamps = {name: _stamp(frame) for name, frame in frames.items()}
            if len(stamps) < 2:
                break
            newest = max(stamps.values())
            laggards = {
                name: max(stamp, newest - tolerance_s)
                for name, stamp in stamps.items()
                if newest - stamp > tolerance_s
            }
            if not laggards:
                break
            attempts += 1
            fresh, _ = await self._gather(laggards)  # a failed re-capture keeps its first frame
            frames.update(fresh)
            recaptured.extend(name for name in fresh if name not in recaptured)
        stamps = {name: _stamp(frame) for name, frame in frames.items()}
        skew_ms = (max(stamps.values()) - min(stamps.values())) * 1000.0 if stamps else 0.0
        return GroupCapture(
            frames=frames,
            errors=errors,
            stamps=stamps,
            skew_ms=skew_ms,
            tolerance_ms=self._tolerance_ms,
            attempts=attempts,
            bound=bound,
            recaptured=tuple(recaptured),
        )

    async def _gather(
        self, bounds: Mapping[str, float]
    ) -> tuple[dict[str, Any], dict[str, Exception]]:
        names = list(bounds)
        results = await asyncio.gather(
            *(self._members[name](bounds[name]) for name in names), return_exceptions=True
        )
        frames: dict[str, Any] = {}
        errors: dict[str, Exception] = {}
        for name, result in zip(names, results):
            if isinstance(result, Exception):
                errors[name] = result
            elif isinstance(result, BaseException):
                raise result  # cancellation and interpreter exits are never a member failure
            else:
                frames[name] = result
        return frames, errors


def bounded_capture(capture: Callable[..., Awaitable[Any]]) -> GroupMember:
    """Adapt a ``capture(*, after_monotonic=...)`` method into a group member.

    A producer predating freshness bounds is called without one and answers with
    its own notion of "now"; the group's skew measurement still reports honestly
    how far apart the answers landed.
    """
    if accepts_keyword(capture, "after_monotonic"):
        return lambda after: capture(after_monotonic=after)
    return lambda _after: capture()


def _stamp(frame: Any) -> float:
    return float(getattr(frame, "monotonic_at", 0.0) or 0.0)


__all__ = [
    "DEFAULT_SYNC_ATTEMPTS",
    "DEFAULT_SYNC_TOLERANCE_MS",
    "CaptureGroup",
    "GroupCapture",
    "GroupMember",
    "bounded_capture",
]
