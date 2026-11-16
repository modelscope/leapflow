# Copyright (c) Alibaba, Inc. and its affiliates.
"""Workcell readiness: what would unblock a skill, solved from declared provisions.

A refusal that says only "camera calibration is stale; run calib.handeye again"
makes the operator repeat the most expensive procedure even when a cheaper one
would do -- vouching for the existing measurement, or installing an estimate a
calibration already produced. This module answers "what is the shortest safe
way from here to ready?" without a table of skill names: each skill declares
the requirement tokens it establishes (``SkillSpec.provides``, optionally per
workcell through ``ProvidingSkill``), each frame carries a typed diagnosis, and
the solver chains providers whose own requirements are met or can be met in
turn.

Options are ranked by what they cost the operator: no-motion options first,
then fewer steps, then fewer arguments a human still has to supply. A
requirement no skill can provide (a role, a model, a sharing flag) is a
declaration the operator edits; it is named, never guessed at.
"""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from leapflow.hardware.skills.frame_status import FrameStatus
from leapflow.hardware.skills.frames import (
    FrameGraph,
    argument_candidates,
    bind_requirements,
    requirement_arguments,
    unmet_requirements,
)
from leapflow.hardware.skills.ports import NodeSkill, ProvidingSkill
from leapflow.hardware.skills.protocol import (
    REQUIRE_CALIBRATION,
    REQUIRE_CANDIDATE,
    SkillSpec,
)
from leapflow.hardware.skills.workcell import WorkcellSpec

logger = logging.getLogger(__name__)

MAX_OPTIONS = 3
MAX_DEPTH = 3
MAX_BINDINGS = 16
_ARGUMENT = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
_VALUE = r"[A-Za-z0-9_.\-]+"


@dataclass(frozen=True)
class ReadinessStep:
    """One skill run that establishes one requirement token."""

    skill_id: str
    args: Mapping[str, Any]
    addresses: str
    needs_args: tuple[str, ...] = ()
    motion: bool = False
    consent: str = ""
    expected: str = ""  # the frame status the run is predicted to leave, when not valid

    @property
    def key(self) -> tuple[str, tuple[tuple[str, str], ...]]:
        return self.skill_id, tuple(sorted((k, str(v)) for k, v in self.args.items()))

    def describe(self) -> str:
        text = "run " + " ".join([self.skill_id, *(f"{k}={v}" for k, v in self.args.items())])
        if self.needs_args:
            text += f" (needs {', '.join(self.needs_args)})"
        return text

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "skill_id": self.skill_id,
            "args": dict(self.args),
            "addresses": self.addresses,
            "motion": self.motion,
        }
        if self.needs_args:
            out["needs_args"] = list(self.needs_args)
        if self.consent:
            out["consent"] = self.consent
        if self.expected:
            out["expected_status"] = self.expected
        return out


@dataclass(frozen=True)
class ReadinessOption:
    """An ordered sequence of runs that, together, meets one requirement."""

    steps: tuple[ReadinessStep, ...]
    rank: int = 0

    @property
    def motion(self) -> bool:
        return any(step.motion for step in self.steps)

    def describe(self) -> str:
        return ", then ".join(step.describe() for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.describe(),
            "motion": self.motion,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class Blocker:
    """One unmet requirement of a skill, why, and what would meet it."""

    token: str
    reason: str
    status: str = ""
    options: tuple[ReadinessOption, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"token": self.token, "reason": self.reason}
        if self.status:
            out["status"] = self.status
        if self.options:
            out["options"] = [option.to_dict() for option in self.options]
        return out


@dataclass(frozen=True)
class Readiness:
    """Whether a skill can run now, and the combined shortest path when it cannot."""

    skill_id: str
    blockers: tuple[Blocker, ...]
    next_steps: tuple[ReadinessStep, ...]

    @property
    def ready(self) -> bool:
        return not self.blockers

    def compact(self) -> list[str]:
        """The next steps as one line each, for a catalog listing."""
        return [step.describe() for step in self.next_steps]

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "ready": self.ready,
            "blockers": [blocker.to_dict() for blocker in self.blockers],
            "next_steps": [step.to_dict() for step in self.next_steps],
        }


@dataclass(frozen=True)
class _Provision:
    index: int
    skill: NodeSkill
    template: str
    binding: Mapping[str, str]  # argument values a concrete per-workcell provision implies


class WorkcellReadiness:
    """Solves requirement tokens against one workcell snapshot and one skill catalog."""

    def __init__(
        self,
        skills: Iterable[NodeSkill],
        spec: WorkcellSpec,
        frames: FrameGraph,
        *,
        max_options: int = MAX_OPTIONS,
        max_depth: int = MAX_DEPTH,
    ) -> None:
        self._spec, self._frames = spec, frames
        self._max_options, self._max_depth = max_options, max_depth
        self._provisions = _provisions(tuple(skills), spec, frames)
        self._memo: dict[tuple[str, str, frozenset[str], int], tuple[ReadinessOption, ...]] = {}

    # ── Public API ───────────────────────────────────────────────────────────

    def hint(self, token: str) -> str:
        """The best remedy for one unmet token, as a sentence fragment; empty if none."""
        options = self.options(token)
        return options[0].describe() if options else ""

    def explain(self, skill: SkillSpec, args: Mapping[str, Any] | None = None) -> Readiness:
        """Every unmet requirement of *skill* under *args*, and the steps that meet them."""
        bound = self._bind_free(skill, {**_defaults(skill), **(args or {})})
        blockers: list[Blocker] = []
        for token in dict.fromkeys(bind_requirements(skill.requires, bound)):
            reasons = unmet_requirements((token,), self._spec, self._frames, self.hint)
            if not reasons:
                continue
            options = self.options(token, exclude=frozenset({skill.skill_id}))
            blockers.append(Blocker(token, reasons[0], self._status(token), options))
        steps: dict[Any, ReadinessStep] = {}
        for blocker in blockers:
            if blocker.options:
                for step in blocker.options[0].steps:
                    steps.setdefault(step.key, step)
        return Readiness(skill.skill_id, tuple(blockers), tuple(steps.values()))

    def options(
        self,
        token: str,
        *,
        status: str = "",
        exclude: frozenset[str] = frozenset(),
        depth: int = 0,
    ) -> tuple[ReadinessOption, ...]:
        """Ranked ways to meet *token*; *status* overrides the frame's current status."""
        if depth > self._max_depth:
            return ()
        status = status or self._status(token)
        key = (token, status, exclude, depth)
        if key not in self._memo:
            self._memo[key] = ()  # a cycle back to this token finds no option
            self._memo[key] = self._solve(token, status, exclude, depth)
        return self._memo[key]

    # ── Solving ──────────────────────────────────────────────────────────────

    def _solve(
        self, token: str, status: str, exclude: frozenset[str], depth: int
    ) -> tuple[ReadinessOption, ...]:
        found: dict[Any, ReadinessOption] = {}
        for provision in self._provisions:
            spec = provision.skill.spec
            if spec.skill_id in exclude:
                continue
            bound = _match(provision.template, token)
            if bound is None or not _admits(spec, {**provision.binding, **bound}):
                continue
            bound = {**provision.binding, **bound}
            if token.startswith(REQUIRE_CALIBRATION) and spec.provides_when and status not in spec.provides_when:
                continue
            for option in self._via(provision, bound, token, exclude | {spec.skill_id}, depth):
                found.setdefault(tuple(step.key for step in option.steps), option)
        ranked = sorted(found.values(), key=_rank)
        return tuple(ranked[: self._max_options])

    def _via(
        self,
        provision: _Provision,
        bound: Mapping[str, str],
        token: str,
        exclude: frozenset[str],
        depth: int,
    ) -> list[ReadinessOption]:
        spec = provision.skill.spec
        defaults = _defaults(spec)
        free = [
            name
            for name in requirement_arguments(spec.requires)
            if name not in bound and name not in defaults
        ]
        taken = set(bound.values())
        choices = [
            [value for value in argument_candidates(spec.args_schema, name, self._spec) if value not in taken]
            for name in free
        ]
        options: list[ReadinessOption] = []
        for combo in itertools.islice(itertools.product(*choices), MAX_BINDINGS):
            chosen = {**bound, **dict(zip(free, combo))}
            option = self._option(provision, chosen, {**defaults, **chosen}, token, exclude, depth)
            if option is not None:
                options.append(option)
        return options

    def _option(
        self,
        provision: _Provision,
        chosen: Mapping[str, str],
        args: Mapping[str, Any],
        token: str,
        exclude: frozenset[str],
        depth: int,
    ) -> ReadinessOption | None:
        spec = provision.skill.spec
        before: dict[Any, ReadinessStep] = {}
        for requirement in bind_requirements(spec.requires, args):
            if not unmet_requirements((requirement,), self._spec, self._frames):
                continue
            sub = self.options(requirement, exclude=exclude, depth=depth + 1)
            if not sub:
                return None
            for step in sub[0].steps:
                before.setdefault(step.key, step)
        expected = self._predicted(spec, args, token)
        step = ReadinessStep(
            spec.skill_id,
            {name: value for name, value in chosen.items()},
            token,
            needs_args=tuple(
                name for name in (spec.args_schema.get("required") or ()) if name not in args
            ),
            motion=spec.uses_motion,
            consent=spec.consent if spec.is_actuating else "",
            expected=expected,
        )
        after: tuple[ReadinessStep, ...] = ()
        if expected:
            follow = self.options(token, status=expected, exclude=exclude, depth=depth + 1)
            after = follow[0].steps if follow else ()
        return ReadinessOption((*before.values(), step, *after), provision.index)

    def _predicted(self, spec: SkillSpec, args: Mapping[str, Any], token: str) -> str:
        """The status a candidate promotion will leave, when it is not ``valid``."""
        if not token.startswith(REQUIRE_CALIBRATION):
            return ""
        frame = token[len(REQUIRE_CALIBRATION) :]
        if f"{REQUIRE_CANDIDATE}{frame}" not in bind_requirements(spec.requires, args):
            return ""
        candidates = self._frames.candidates(frame)
        required = self._frames.required_accuracy(frame)
        if not candidates or required is None:
            return ""
        achieved = candidates[0].achieved
        if achieved is None or achieved.within(required):
            return ""
        return FrameStatus.INSUFFICIENT_ACCURACY.value

    def _bind_free(self, skill: SkillSpec, args: dict[str, Any]) -> dict[str, Any]:
        """Bind arguments the caller left open to the first value that meets the most."""
        free = [name for name in requirement_arguments(skill.requires) if name not in args]
        if not free:
            return args
        choices = [argument_candidates(skill.args_schema, name, self._spec) for name in free]
        best, fewest = args, None
        for combo in itertools.islice(itertools.product(*choices), MAX_BINDINGS):
            trial = {**args, **dict(zip(free, combo))}
            unmet = len(unmet_requirements(bind_requirements(skill.requires, trial), self._spec, self._frames))
            if fewest is None or unmet < fewest:
                best, fewest = trial, unmet
        return best

    def _status(self, token: str) -> str:
        if token.startswith(REQUIRE_CALIBRATION):
            return self._frames.diagnosis(token[len(REQUIRE_CALIBRATION) :]).status.value
        return ""


def _provisions(
    skills: Sequence[NodeSkill], spec: WorkcellSpec, frames: FrameGraph
) -> list[_Provision]:
    found: list[_Provision] = []
    for index, skill in enumerate(skills):
        templates = tuple(skill.spec.provides)
        tokens: Sequence[str] = templates
        if isinstance(skill, ProvidingSkill):
            try:
                tokens = tuple(skill.provides_in(spec, frames))
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "%s could not state its provisions: %s", skill.spec.skill_id, exc, exc_info=True
                )
                tokens = ()
        for token in dict.fromkeys(tokens):
            matches = (_match(template, token) for template in templates)
            binding = next((match for match in matches if match), {})
            found.append(_Provision(index, skill, token, binding))
    return found


def _match(template: str, token: str) -> dict[str, str] | None:
    """The argument values that make *template* name *token*; None when it cannot."""
    parts = _ARGUMENT.split(template)
    seen: set[str] = set()
    pattern = []
    for position, part in enumerate(parts):
        if position % 2 == 0:
            pattern.append(re.escape(part))
        elif part in seen:
            pattern.append(f"(?P={part})")
        else:
            seen.add(part)
            pattern.append(f"(?P<{part}>{_VALUE})")
    found = re.fullmatch("".join(pattern), token)
    return dict(found.groupdict()) if found else None


def _admits(spec: SkillSpec, bound: Mapping[str, str]) -> bool:
    """Whether each bound value is one the skill's schema accepts."""
    properties = spec.args_schema.get("properties") or {}
    for name, value in bound.items():
        schema = properties.get(name)
        if not isinstance(schema, Mapping):
            return False
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and not re.fullmatch(pattern, value):
            return False
        enum = schema.get("enum")
        if isinstance(enum, list) and value not in map(str, enum):
            return False
    return True


def _defaults(spec: SkillSpec) -> dict[str, Any]:
    properties = spec.args_schema.get("properties") or {}
    return {
        name: schema["default"]
        for name, schema in properties.items()
        if isinstance(schema, Mapping) and "default" in schema
    }


def _rank(option: ReadinessOption) -> tuple[int, int, int, int]:
    return (
        int(option.motion),
        len(option.steps),
        sum(len(step.needs_args) for step in option.steps),
        option.rank,
    )


__all__ = [
    "Blocker",
    "Readiness",
    "ReadinessOption",
    "ReadinessStep",
    "WorkcellReadiness",
]
