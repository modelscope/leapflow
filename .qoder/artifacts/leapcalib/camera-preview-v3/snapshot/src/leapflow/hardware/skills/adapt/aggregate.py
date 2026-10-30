# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tick-level stages: time-indexed chunk aggregation, interpolation and smoothing.

A chunking policy (SmolVLA, ACT, EdgeVLA) answers one observation with a short
sequence of future actions spaced ``step_s`` apart, the first aligned with the
moment the observation was taken. Inference takes time, so by the time a chunk
arrives part of it already lies in the past. ``ChunkAggregator`` therefore
indexes actions by *time*, not by position: ``sample(now)`` reads the live
chunks at ``now`` and silently skips whatever latency consumed.

How overlapping chunks combine is declared (``policies.<name>.aggregate``):

- ``crossfade`` (default): a fresh chunk replaces the one being executed,
  blended over ``blend_s`` so the commanded trajectory stays continuous.
- ``temporal_ensemble`` (ACT): every live chunk votes for ``now``, the oldest
  weighted most -- ``exp(-decay * i)`` for the i-th oldest.
- ``latest``: the fresh chunk takes over at once.

Within a chunk, ``linear`` interpolates between rows and ``hold`` keeps the
row whose slot ``now`` falls in. A chunk is live until one step past its last
row; sampling beyond that returns ``None`` -- starvation is the caller's
decision (hold, then escalate), never an extrapolation.

Pure and clock-free: every method takes the time it should reason about, so
the same code runs under the node clock and under a test's fake clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from leapflow.hardware.skills.adapt.spec import (
    AGGREGATE_CROSSFADE,
    AGGREGATE_ENSEMBLE,
    AGGREGATE_LATEST,
    INTERPOLATION_HOLD,
    INTERPOLATION_LINEAR,
)
from leapflow.hardware.skills.geometry import wrap_deg

Action = tuple[float, ...]
_MAX_ENSEMBLE = 32  # live chunks an ensemble keeps; older ones have long ended anyway


def _mix(a: Action, b: Action, weight: float, cyclic: Sequence[bool]) -> Action:
    """``a`` moved *weight* of the way to ``b``; shortest turn on cyclic channels."""
    return tuple(
        x + weight * (wrap_deg(y - x) if turn else y - x) for x, y, turn in zip(a, b, cyclic)
    )


@dataclass(frozen=True)
class _Chunk:
    start: float
    step_s: float
    actions: tuple[Action, ...]

    @property
    def end(self) -> float:
        """The last action is held for one step, then the chunk is exhausted."""
        return self.start + self.step_s * len(self.actions)

    def at(self, now: float, cyclic: Sequence[bool], interpolation: str) -> Action:
        position = max(0.0, (now - self.start) / self.step_s)
        index = min(int(position), len(self.actions) - 1)
        if interpolation == INTERPOLATION_HOLD or index + 1 >= len(self.actions):
            return self.actions[index]
        return _mix(self.actions[index], self.actions[index + 1], position - index, cyclic)

    def remaining(self, now: float) -> tuple[Action, ...]:
        """The rows whose slot has not started yet."""
        first = max(0, math.ceil((now - self.start) / self.step_s - 1e-9))
        return self.actions[first:]


class ChunkAggregator:
    """Turns asynchronously arriving action chunks into one continuous command stream."""

    def __init__(
        self,
        width: int,
        *,
        cyclic: Sequence[bool] = (),
        mode: str = AGGREGATE_CROSSFADE,
        blend_s: float = 0.1,
        decay: float = 0.01,
        interpolation: str = INTERPOLATION_LINEAR,
    ) -> None:
        if width < 1:
            raise ValueError("an action needs at least one channel")
        if cyclic and len(cyclic) != width:
            raise ValueError(f"cyclic mask has {len(cyclic)} entries for {width} channels")
        if blend_s < 0.0 or not math.isfinite(blend_s):
            raise ValueError("blend_s must be a finite, non-negative duration")
        if decay < 0.0 or not math.isfinite(decay):
            raise ValueError("decay must be finite and non-negative")
        if mode not in (AGGREGATE_CROSSFADE, AGGREGATE_ENSEMBLE, AGGREGATE_LATEST):
            raise ValueError(f"unknown aggregation {mode!r}")
        if interpolation not in (INTERPOLATION_LINEAR, INTERPOLATION_HOLD):
            raise ValueError(f"unknown interpolation {interpolation!r}")
        self._width = width
        self._cyclic = tuple(bool(c) for c in cyclic) or (False,) * width
        self._mode = mode
        self._blend_s = float(blend_s)
        self._decay = float(decay)
        self._interpolation = interpolation
        self._live: list[_Chunk] = []
        self._previous: _Chunk | None = None
        self._blend_start = 0.0
        self.chunks = 0

    @property
    def width(self) -> int:
        return self._width

    @property
    def mode(self) -> str:
        return self._mode

    def offer(
        self, actions: Sequence[Sequence[float]], *, observed_at: float, step_s: float, now: float
    ) -> None:
        """Adopt a chunk whose first action belongs to the instant *observed_at*.

        Raises ``ValueError`` for an empty chunk, a wrong width, a non-finite value
        or a non-positive step: a malformed answer must never become a command.
        """
        if not actions:
            raise ValueError("the policy returned an empty chunk")
        if step_s <= 0.0 or not math.isfinite(step_s):
            raise ValueError("the chunk step must be a positive duration")
        rows: list[Action] = []
        for row in actions:
            if len(row) != self._width:
                raise ValueError(f"an action has {len(row)} values, expected {self._width}")
            values = tuple(float(v) for v in row)
            if not all(math.isfinite(v) for v in values):
                raise ValueError("an action contains a non-finite value")
            rows.append(values)
        chunk = _Chunk(start=float(observed_at), step_s=float(step_s), actions=tuple(rows))
        current = self._live[-1] if self._live else None
        if self._mode == AGGREGATE_ENSEMBLE:
            self._live = [c for c in self._live if now < c.end][-(_MAX_ENSEMBLE - 1) :]
            self._live.append(chunk)
        else:
            live = current is not None and now < current.end
            blending = self._mode == AGGREGATE_CROSSFADE and live
            self._previous, self._blend_start = (current, float(now)) if blending else (None, 0.0)
            self._live = [chunk]
        self.chunks += 1

    def sample(self, now: float) -> Action | None:
        """The action to command at *now*, or ``None`` when the latest chunk is exhausted."""
        current = self._live[-1] if self._live else None
        if current is None or now >= current.end:
            return None
        if self._mode == AGGREGATE_ENSEMBLE:
            return self._ensemble(now)
        value = current.at(now, self._cyclic, self._interpolation)
        previous = self._previous
        if previous is None:
            return value
        progress = (now - self._blend_start) / self._blend_s if self._blend_s > 0.0 else 1.0
        if progress >= 1.0 or now >= previous.end:
            self._previous = None
            return value
        old = previous.at(now, self._cyclic, self._interpolation)
        return _mix(old, value, max(0.0, progress), self._cyclic)

    def _ensemble(self, now: float) -> Action:
        live = [c for c in self._live if c.start <= now < c.end] or [self._live[-1]]
        samples = [c.at(now, self._cyclic, self._interpolation) for c in live]
        weights = [math.exp(-self._decay * i) for i in range(len(samples))]
        total = sum(weights)
        reference = samples[-1]
        values: list[float] = []
        for k, turn in enumerate(self._cyclic):
            # Cyclic channels average their offsets from the newest sample, so a
            # wrap between two votes cannot drag the mean through the far side.
            offsets = [
                wrap_deg(s[k] - reference[k]) if turn else s[k] - reference[k] for s in samples
            ]
            values.append(reference[k] + sum(w * o for w, o in zip(weights, offsets)) / total)
        return tuple(values)

    def pending(self, now: float) -> tuple[Action, ...]:
        """The latest chunk's rows not yet started at *now* (what RTC sends back)."""
        current = self._live[-1] if self._live else None
        return () if current is None else current.remaining(now)

    def remaining_s(self, now: float) -> float:
        """Seconds of commands left in the latest chunk (0 when none)."""
        current = self._live[-1] if self._live else None
        return 0.0 if current is None else max(0.0, current.end - now)

    def duration_s(self) -> float:
        """Span of the latest chunk (0 when none)."""
        current = self._live[-1] if self._live else None
        return 0.0 if current is None else current.end - current.start


class LowPass:
    """First-order low-pass on the command stream; ``cutoff_hz <= 0`` passes through.

    ``alpha = 1 - exp(-2 pi fc dt)`` keeps the filter's time constant independent
    of the tick rate. Cyclic channels filter the shortest turn.
    """

    def __init__(self, cutoff_hz: float, cyclic: Sequence[bool]) -> None:
        self._cutoff = max(0.0, float(cutoff_hz))
        self._cyclic = tuple(cyclic)
        self._state: Action | None = None

    def reset(self, value: Sequence[float] | None = None) -> None:
        self._state = None if value is None else tuple(float(v) for v in value)

    def __call__(self, value: Action, dt: float) -> Action:
        if self._cutoff <= 0.0 or self._state is None or dt <= 0.0:
            self._state = value
            return value
        alpha = 1.0 - math.exp(-2.0 * math.pi * self._cutoff * dt)
        cyclic = self._cyclic or (False,) * len(value)
        self._state = _mix(self._state, value, alpha, cyclic)
        return self._state


__all__ = ["Action", "ChunkAggregator", "LowPass"]
