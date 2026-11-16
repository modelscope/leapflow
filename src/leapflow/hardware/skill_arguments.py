# Copyright (c) Alibaba, Inc. and its affiliates.
"""The two levels of an ``hw_skill_run`` call, and the repair between them.

A run request has an envelope the host reads (``workcell``, ``skill``,
``dry_run``, ``intent``, ``wait``, ``timeout_s``) and ``args``, the skill's own
payload, which the node validates against the skill's ``args_schema`` with no
unknown keys allowed. A model that nests an envelope field inside ``args`` used
to get a node refusal for an argument it never meant for the skill -- or, for
``workcell`` and ``skill``, an engine refusal for a "missing" parameter it had
in fact sent.

The repair is deterministic and checked against the skill's declared schema:

- An envelope field inside ``args`` is lifted to the envelope only when the
  skill's ``args_schema`` does not declare it: it cannot be the skill's
  argument, so it can only be the envelope's.
- Lifting never overrides: a field set both beside and inside ``args`` with
  different values is refused, naming both, because either could be meant.
- ``workcell`` and ``skill`` route the request before any schema can be read;
  one the skill also declares as its own argument is refused rather than
  guessed.

Every repair rides on the result (``argument_repair``) and the audit log, so it
reads as a self-repaired call, never as a silent rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from leapflow.hardware.skill_results import FAILURE_INVALID_REQUEST, SkillRefusal, as_mapping
from leapflow.hardware.skill_tool_defs import DEFAULT_RUN_TIMEOUT_S, MAX_RUN_TIMEOUT_S

PAYLOAD_FIELD = "args"
ROUTING_FIELDS: tuple[str, ...] = ("workcell", "skill")
"""Envelope fields the host needs before it can read the skill's schema."""
OPTION_FIELDS: tuple[str, ...] = ("dry_run", "intent", "wait", "timeout_s")
ENVELOPE_FIELDS: tuple[str, ...] = ROUTING_FIELDS + OPTION_FIELDS
_BOOLEAN_FIELDS = frozenset({"dry_run", "wait"})


@dataclass(frozen=True)
class RunRequest:
    """One ``hw_skill_run`` call split into its envelope and the skill's ``args``."""

    workcell: str
    skill: str
    args: dict[str, Any]
    dry_run: bool = False
    wait: bool = True
    intent: str = ""
    timeout_s: float = DEFAULT_RUN_TIMEOUT_S
    lifted: Mapping[str, Any] = field(default_factory=dict)

    def repair_view(self) -> dict[str, Any]:
        """What was moved out of ``args``, for the result S2 reads."""
        names = ", ".join(self.lifted)
        them = "it" if len(self.lifted) == 1 else "them"
        return {
            "lifted": dict(self.lifted),
            "message": (
                f"{names} sat inside args, where only the skill's own arguments go; the "
                f"call ran with {them} as hw_skill_run parameters. Pass {them} beside args "
                "next time."
            ),
        }


def misplaced_fields(args: Any) -> tuple[str, ...]:
    """The envelope fields nested inside *args*."""
    payload = as_mapping(args)
    return tuple(name for name in ENVELOPE_FIELDS if name in payload)


def routing(kwargs: Mapping[str, Any]) -> tuple[str, str]:
    """``(workcell, skill)`` from the envelope, else from inside ``args``."""
    payload = as_mapping(kwargs.get(PAYLOAD_FIELD))
    values = []
    for name in ROUTING_FIELDS:
        value = kwargs.get(name)
        if value in (None, "") and isinstance(payload.get(name), str):
            value = payload[name]
        values.append(str(value or "").strip())
    return values[0], values[1]


def declared_arguments(listing: Mapping[str, Any], skill: str) -> frozenset[str] | None:
    """The argument names *skill* declares in a node ``skill.list`` reply, or
    ``None`` when the reply does not offer it."""
    for cell in listing.get("workcells") or ():
        for entry in as_mapping(cell).get("skills") or ():
            entry = as_mapping(entry)
            if entry.get("skill_id") == skill:
                schema = as_mapping(entry.get("args_schema"))
                return frozenset(str(name) for name in as_mapping(schema.get("properties")))
    return None


def split_request(kwargs: Mapping[str, Any], declared: frozenset[str] | None) -> RunRequest:
    """The request with envelope fields lifted out of ``args`` where unambiguous.

    *declared* is the skill's argument names; ``None`` (the skill's schema is
    unknown) lifts only the routing fields the request cannot proceed without,
    and leaves the rest for the node to refuse by name.
    """
    raw = kwargs.get(PAYLOAD_FIELD)
    if raw is not None and not isinstance(raw, Mapping):
        raise SkillRefusal(FAILURE_INVALID_REQUEST, "args must be an object")
    args = dict(raw or {})
    envelope = {name: kwargs.get(name) for name in ENVELOPE_FIELDS}
    lifted: dict[str, Any] = {}
    for name in misplaced_fields(args):
        if declared is not None and name in declared:
            if name in ROUTING_FIELDS and _unset(envelope[name]):
                raise SkillRefusal(
                    FAILURE_INVALID_REQUEST,
                    f"{name} is required beside args; args.{name} is the skill's own argument",
                )
            continue  # the skill's own argument
        if declared is None and name not in ROUTING_FIELDS:
            continue
        value = args.pop(name)
        if not _unset(envelope[name]) and not _same(name, envelope[name], value):
            raise SkillRefusal(
                FAILURE_INVALID_REQUEST,
                f"{name} is set both beside args ({envelope[name]!r}) and inside it "
                f"({value!r}); it is an hw_skill_run parameter, so set it once, beside args",
            )
        if _unset(envelope[name]):
            if name in _BOOLEAN_FIELDS and not isinstance(value, bool):
                raise SkillRefusal(FAILURE_INVALID_REQUEST, f"{name} must be true or false")
            envelope[name] = value
        lifted[name] = value
    return RunRequest(
        workcell=str(envelope["workcell"] or "").strip(),
        skill=str(envelope["skill"] or "").strip(),
        args=args,
        dry_run=bool(envelope["dry_run"]),
        wait=True if envelope["wait"] is None else bool(envelope["wait"]),
        intent=str(envelope["intent"] or "").strip(),
        timeout_s=_timeout(envelope["timeout_s"]),
        lifted=lifted,
    )


def _timeout(raw: Any) -> float:
    if raw is None:
        return DEFAULT_RUN_TIMEOUT_S
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise SkillRefusal(FAILURE_INVALID_REQUEST, "timeout_s must be a number") from exc
    if not 0 < value <= MAX_RUN_TIMEOUT_S:
        raise SkillRefusal(FAILURE_INVALID_REQUEST, f"timeout_s must be in (0, {MAX_RUN_TIMEOUT_S:g}]")
    return value


def _unset(value: Any) -> bool:
    return value is None or value == ""


def _same(name: str, outer: Any, inner: Any) -> bool:
    """Whether two values of one envelope field mean the same thing.

    A workcell may be qualified (``node/cell``) on one side and bare on the
    other; both name the same cell when the cell parts agree.
    """
    if outer == inner:
        return True
    if name == "workcell" and isinstance(outer, str) and isinstance(inner, str):
        a, b = outer.strip(), inner.strip()
        if "/" in a and "/" in b:
            return a == b
        return a.rpartition("/")[2] == b.rpartition("/")[2]
    return False


__all__ = [
    "ENVELOPE_FIELDS",
    "OPTION_FIELDS",
    "PAYLOAD_FIELD",
    "ROUTING_FIELDS",
    "RunRequest",
    "declared_arguments",
    "misplaced_fields",
    "routing",
    "split_request",
]
