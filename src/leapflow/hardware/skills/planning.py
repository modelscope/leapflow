# Copyright (c) Alibaba, Inc. and its affiliates.
"""Pure planning contracts, resource policy, and explicit lexicographic selection.

Jobs are constructed by admitted node plugins, never deserialized from the wire.
Only immutable geometry travels to workers; approval and execution stay with the owner.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Protocol, Sequence, TYPE_CHECKING, runtime_checkable

from leapflow.hardware.skills.protocol import (
    FAILURE_INVALID_ARGS, FAILURE_STOPPED, FAILURE_TIMEOUT, FAILURE_UNREACHABLE,
    SkillError, SkillOutcome, SkillPlan,
)

if TYPE_CHECKING:
    from leapflow.hardware.skills.ports import SkillContext

METRICS = {
    "joint_margin": "max", "joint_travel": "min", "tcp_distance_mm": "min",
    "rotation_deg": "min", "wait_s": "min",
}
DEFAULT_ORDER = ("joint_margin", "joint_travel", "tcp_distance_mm")


def encoded(value: Any) -> str:
    """Canonical JSON for exact within-request deduplication and detached values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class PlanningSettings:
    workers: int = 0
    max_candidates: int = 8
    max_jobs: int = 16
    timeout_s: float = 30.0
    max_bytes: int = 256 * 1024
    cancel_grace_s: float = 0.5

    def __post_init__(self) -> None:
        for name, low, high in (("workers", 0, 32), ("max_candidates", 1, 32),
                                ("max_jobs", 1, 128), ("max_bytes", 1024, 512 * 1024)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"planning {name} must be an integer in [{low}, {high}]")
        for name, high in (("timeout_s", 600), ("cancel_grace_s", 5)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= high:
                raise ValueError(f"planning {name} must be finite in (0, {high}]")

    @property
    def worker_count(self) -> int:
        available = getattr(os, "process_cpu_count", os.cpu_count)() or 1
        return min(self.workers or 2, max(1, available - 1), self.max_jobs)

    def to_dict(self) -> dict[str, Any]:
        return {**vars(self), "workers": self.worker_count}

    def check_size(self, value: Any) -> None:
        try:
            size = len(encoded(value).encode("utf-8"))
        except (ValueError, TypeError, RecursionError) as exc:
            raise SkillError(FAILURE_INVALID_ARGS, "planning payload must be finite JSON") from exc
        if size > self.max_bytes:
            raise SkillError(FAILURE_INVALID_ARGS, f"planning payload exceeds {self.max_bytes} bytes")


@runtime_checkable
class PlanningCheckpoint(Protocol):
    def check(self) -> None: ...


@dataclass(frozen=True)
class Deadline:
    expires: float = math.inf
    cancel: Any = None

    def check(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise SkillError(FAILURE_STOPPED, "planning cancelled; nothing executed")
        if time.monotonic() >= self.expires:
            raise SkillError(FAILURE_TIMEOUT, "planning deadline exceeded; nothing executed")


@dataclass(frozen=True)
class FrozenJoints(Mapping[str, float]):
    """Pickleable scalar joint snapshot with no mutable mapping reachable by jobs."""
    entries: tuple[tuple[str, float], ...]

    @classmethod
    def capture(cls, values: Mapping[str, float]) -> FrozenJoints:
        return cls(tuple((name, float(value)) for name, value in values.items()))

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, name: str) -> float:
        for key, value in self.entries:
            if key == name:
                return value
        raise KeyError(name)


@dataclass(frozen=True)
class Evaluation:
    value: SkillPlan | SkillOutcome
    metrics: Mapping[str, float | None] = field(default_factory=dict)
    verified: bool = False
    evidence: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class PlanningJob(Protocol):
    """An admitted plugin's pickleable pure compute job, with no owner references."""
    def evaluate(self, checkpoint: PlanningCheckpoint) -> Evaluation: ...


@runtime_checkable
class CalculationPort(Protocol):
    async def evaluate(self, jobs: Sequence[PlanningJob], *, deadline: float) -> tuple[Evaluation, ...]: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class PreparedPlanning:
    jobs: tuple[PlanningJob | SkillOutcome, ...]
    objects: tuple[Any, ...] = ()
    grasps: tuple[Any, ...] = ()
    held: Any = None
    state_reads: int = 0

    def check_scene(self, ctx: SkillContext) -> None:
        try:
            if ctx.scene.held != self.held:
                raise SkillError("plan_stale", "held object changed during planning")
            for obj in self.objects:
                if ctx.scene.object(obj.object_id) != obj:
                    raise SkillError("plan_stale", "referenced object changed during planning")
            for grasp in self.grasps:
                if ctx.scene.grasp(grasp.grasp_id) != grasp:
                    raise SkillError("plan_stale", "referenced grasp changed during planning")
        except SkillError as error:
            raise SkillError("plan_stale", error.message) from error


@runtime_checkable
class PreparedPlanningSkill(Protocol):
    async def prepare(self, ctx: SkillContext, args: Sequence[Mapping[str, Any]]) -> PreparedPlanning: ...


@dataclass(frozen=True)
class Selection:
    order: tuple[tuple[str, str], ...] = tuple((name, METRICS[name]) for name in DEFAULT_ORDER)
    require_verified: bool = True

    @classmethod
    def from_mapping(cls, value: Any) -> Selection:
        if value is None:
            return cls()
        if not isinstance(value, Mapping) or set(value) - {"order", "require_verified"}:
            raise SkillError(FAILURE_INVALID_ARGS, "selection accepts order and require_verified")
        verified = value.get("require_verified", True)
        if type(verified) is not bool:
            raise SkillError(FAILURE_INVALID_ARGS, "require_verified must be boolean")
        raw = value.get("order", list(DEFAULT_ORDER))
        if not isinstance(raw, (list, tuple)) or not 1 <= len(raw) <= len(METRICS):
            raise SkillError(FAILURE_INVALID_ARGS, "selection.order must be a bounded metric list")
        order = []
        for entry in raw:
            name = entry if isinstance(entry, str) else entry.get("metric") if isinstance(entry, Mapping) else None
            if not isinstance(name, str) or name not in METRICS:
                raise SkillError(FAILURE_INVALID_ARGS, f"unknown selection metric {name!r}")
            direction = METRICS[name] if isinstance(entry, str) else entry.get("direction", METRICS[name])
            if not isinstance(direction, str) or direction not in {"min", "max"} or name in [n for n, _ in order]:
                raise SkillError(FAILURE_INVALID_ARGS, "selection metrics must be unique with min/max direction")
            if isinstance(entry, Mapping) and set(entry) - {"metric", "direction"}:
                raise SkillError(FAILURE_INVALID_ARGS, "metric order accepts metric and direction")
            order.append((name, direction))
        return cls(tuple(order), verified)

    def choose(self, results: Sequence[Evaluation]) -> tuple[int | None, dict[str, Any]]:
        usable = [i for i, r in enumerate(results) if isinstance(r.value, SkillPlan)
                  and (r.verified or not self.require_verified)]
        active = [(n, d) for n, d in self.order if any(results[i].metrics.get(n) is not None for i in usable)]

        def key(index: int) -> tuple[Any, ...]:
            result = results[index]
            score: list[Any] = [not result.verified]
            for name, direction in active:
                value = result.metrics.get(name)
                if value is not None and not math.isfinite(value):
                    raise SkillError(FAILURE_INVALID_ARGS, f"nonfinite planning metric {name}")
                score.extend((value is None, 0 if value is None else value * (-1 if direction == "max" else 1)))
            return (*score, index)

        chosen = min(usable, key=key) if usable else None
        return chosen, {
            "order": [{"metric": n, "direction": d} for n, d in active],
            "require_verified": self.require_verified,
            "omitted_metrics": [n for n, _ in self.order if n not in dict(active)],
            "omission_reason": "No eligible candidate has a measured value for omitted metrics.",
            "metric_scope": "declared-model samples; point transits are endpoint estimates, not firmware trajectories",
        }


def candidates(value: Any, settings: PlanningSettings) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    settings.check_size(value)
    if not isinstance(value, list) or not 1 <= len(value) <= settings.max_candidates:
        raise SkillError(FAILURE_INVALID_ARGS, f"candidates needs 1-{settings.max_candidates} entries")
    result = []
    ids: set[str] = set()
    for entry in value:
        if not isinstance(entry, Mapping) or set(entry) != {"id", "args"}:
            raise SkillError(FAILURE_INVALID_ARGS, "each candidate needs exactly id and args")
        name = entry["id"]
        if not isinstance(name, str) or not name.strip() or len(name) > 64 or name in ids:
            raise SkillError(FAILURE_INVALID_ARGS, "candidate IDs must be unique nonempty strings of at most 64 characters")
        if not isinstance(entry["args"], Mapping):
            raise SkillError(FAILURE_INVALID_ARGS, f"candidate {name}: args must be an object")
        ids.add(name)
        result.append((name, entry["args"]))
    return tuple(result)


def no_selection() -> SkillOutcome:
    return SkillOutcome.failed(FAILURE_UNREACHABLE, "no candidate satisfies feasibility and verification; nothing executed")
