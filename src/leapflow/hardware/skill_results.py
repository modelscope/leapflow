# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shape node-skill replies into the tool results S2 reasons over.

Kept apart from ``skill_tools.py`` (the governed run path) because this is the
product surface the LLM reads -- failure codes, next steps, compact views --
and it changes with prompting experience rather than with the wire protocol.

Every failed or refused result carries ``failure_code``, ``side_effect_state``
and an actionable ``next_step``. When the effect is not NONE the result is
flagged ``effect_uncertain`` and the next step forbids replaying the plan.
"""

from __future__ import annotations

from typing import Any, Mapping

from leapflow.hardware.skills.protocol import (
    FAILURE_ARM_NOT_READY,
    FAILURE_CALIBRATION_MISSING,
    FAILURE_CALIBRATION_STALE,
    FAILURE_CHECK_FAILED,
    FAILURE_DEVICE_BUSY,
    FAILURE_ESTOPPED,
    FAILURE_GUARD_FAILED,
    FAILURE_INTERNAL_DEFECT,
    FAILURE_INVALID_ARGS,
    FAILURE_LEASE_LOST,
    FAILURE_MODEL_UNAVAILABLE,
    FAILURE_OUT_OF_WORKSPACE,
    FAILURE_PLAN_STALE,
    FAILURE_PLAN_UNKNOWN,
    FAILURE_SHIELD_VIOLATION,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_STOPPED,
    FAILURE_TARGET_NOT_FOUND,
    FAILURE_TIMEOUT,
    FAILURE_UNKNOWN_SKILL,
    FAILURE_UNKNOWN_WORKCELL,
    FAILURE_UNREACHABLE,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_UNKNOWN,
    STATUS_FAILED,
    STATUS_SUCCEEDED,
)

# Host-side failure codes (the node's own codes live in ``skills.protocol``).
FAILURE_NODE_UNREACHABLE = "node_unreachable"
FAILURE_SKILL_UNSUPPORTED = "skill_unsupported"
FAILURE_UNKNOWN_NODE = "unknown_node"
FAILURE_NO_SKILL_NODES = "no_skill_nodes"
FAILURE_AMBIGUOUS_WORKCELL = "ambiguous_workcell"
FAILURE_PLAN_UNVERIFIED = "plan_unverified"
FAILURE_WRITES_DISABLED = "writes_disabled"
FAILURE_PERMIT_ISSUER_UNAVAILABLE = "permit_issuer_unavailable"
FAILURE_APPROVAL_DENIED = "approval_denied"
FAILURE_EXECUTE_UNCONFIRMED = "execute_unconfirmed"
FAILURE_STATUS_UNAVAILABLE = "status_unavailable"

_RECHECK = "Check hw_skill_status and the arm state before anything else"
_NO_REPLAY = "The arm may have moved: never repeat this plan; a retry is a new plan."

_NEXT_STEPS: dict[str, str] = {
    FAILURE_CALIBRATION_MISSING: (
        "Run the calib.* skill this skill requires on the same workcell (hw_skill_list "
        "shows each skill's requirements), then plan again."
    ),
    FAILURE_TARGET_NOT_FOUND: (
        "The id expired or the object is gone. Locate again and use the new ids, or ask the "
        "user to point the object out."
    ),
    FAILURE_MODEL_UNAVAILABLE: (
        "The node's perception model did not answer. Ask the user to check that the model "
        "server on the node is running, then plan again."
    ),
    FAILURE_UNREACHABLE: (
        "The target is outside what the arm can safely reach. Tell the user the workcell's "
        "reachable range (hw_skill_list) or choose another target."
    ),
    FAILURE_PLAN_STALE: "Plan again with hw_skill_run; the new plan needs a new approval.",
    FAILURE_GUARD_FAILED: (
        "A step's expectation did not hold (result.step names it; result.guard what was "
        "expected). Read the arm and locate again, then write a new program from where the "
        "arm is now; never repeat the same plan."
    ),
    FAILURE_SHIELD_VIOLATION: f"The run stopped partway. {_RECHECK}.",
    FAILURE_LEASE_LOST: (
        "The run was cut off and the arm halted. Wait for the user to confirm the workcell "
        "is safe before running anything."
    ),
    FAILURE_DEVICE_BUSY: (
        "Another run or control session holds the arm. Wait for it (hw_skill_status) or "
        "stop it (hw_skill_stop)."
    ),
    FAILURE_SKILL_UNAVAILABLE: "A requirement is unmet; hw_skill_list shows the reason.",
    FAILURE_INVALID_ARGS: "Fix the arguments against the skill's args_schema (hw_skill_list).",
    FAILURE_CHECK_FAILED: (
        "A calibration or bring-up check measured out of tolerance (see result.problems). "
        "Tell the user what it found, have them fix it, then run the same calib.* skill again."
    ),
    FAILURE_UNKNOWN_SKILL: "Choose a workcell and a skill listed by hw_skill_list.",
    FAILURE_ARM_NOT_READY: "The arm is not enabled or not ready; ask the user to enable it.",
    FAILURE_STOPPED: f"The run was stopped. {_RECHECK}.",
    FAILURE_APPROVAL_DENIED: "The user declined this run; do not retry it without asking.",
    FAILURE_WRITES_DISABLED: (
        "The node is observe-only. Writes need the node operator to change its write mode."
    ),
    FAILURE_PERMIT_ISSUER_UNAVAILABLE: (
        "Run `leap hw permit-key` on the host and install the printed public key on the "
        "node (`--permit-issuer-key`), then try again."
    ),
    FAILURE_NODE_UNREACHABLE: "Check the node with hw_fleet_status, then try again.",
    FAILURE_SKILL_UNSUPPORTED: "This node runs no skill runtime; upgrade or configure it.",
    FAILURE_AMBIGUOUS_WORKCELL: "Qualify the workcell as <node>/<workcell> (hw_skill_list).",
    FAILURE_PLAN_UNVERIFIED: (
        "The node's plan failed host verification and was not shown for approval. Report "
        "this to the user; the node may be running a mismatched version."
    ),
    FAILURE_EXECUTE_UNCONFIRMED: (
        "The node did not confirm the start, so the arm may be moving. Call hw_skill_status; "
        "use hw_skill_stop or hw_estop if anything is unexpected. Do not run it again."
    ),
    FAILURE_STATUS_UNAVAILABLE: (
        "The run's state could not be read. Call hw_skill_status again; if the node stays "
        "silent its lease lapses and it halts the arm."
    ),
    FAILURE_INTERNAL_DEFECT: (
        "The node's skill code failed (see error). Report it to the user; do not retry."
    ),
}
_ALIASES: dict[str, str] = {
    FAILURE_CALIBRATION_STALE: FAILURE_CALIBRATION_MISSING,
    FAILURE_OUT_OF_WORKSPACE: FAILURE_UNREACHABLE,
    FAILURE_PLAN_UNKNOWN: FAILURE_PLAN_STALE,
    FAILURE_TIMEOUT: FAILURE_SHIELD_VIOLATION,
    FAILURE_ESTOPPED: FAILURE_LEASE_LOST,
    FAILURE_UNKNOWN_WORKCELL: FAILURE_UNKNOWN_SKILL,
    FAILURE_UNKNOWN_NODE: FAILURE_UNKNOWN_SKILL,
    FAILURE_NO_SKILL_NODES: FAILURE_NODE_UNREACHABLE,
}
# Codes whose own next step already says what to do about the movement.
_SELF_EXPLAINING = frozenset({FAILURE_GUARD_FAILED, FAILURE_EXECUTE_UNCONFIRMED})


class SkillRefusal(Exception):
    """A typed host-side refusal, turned into a tool result at the tool boundary."""

    def __init__(
        self, failure_code: str, message: str, *, side_effect_state: str = SIDE_EFFECT_NONE
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code
        self.message = message
        self.side_effect_state = side_effect_state


def unreachable(node_id: str, exc: BaseException) -> SkillRefusal:
    """A node that did not answer before anything moved."""
    return SkillRefusal(FAILURE_NODE_UNREACHABLE, f"node {node_id!r}: {exc}")


def next_step(failure_code: str | None, side_effect_state: str) -> str:
    """The S2-facing next step for a failed or refused skill run."""
    code = _ALIASES.get(str(failure_code or ""), str(failure_code or ""))
    step = _NEXT_STEPS.get(code, "")
    if side_effect_state != SIDE_EFFECT_NONE and code not in _SELF_EXPLAINING:
        return f"{step} {_NO_REPLAY}" if step else f"{_RECHECK}. {_NO_REPLAY}"
    return step or "Nothing moved. Adjust the request and try again."


# ── Reply access ────────────────────────────────────────────────────────────


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def runs_of(reply: Any) -> list[Mapping[str, Any]]:
    """The run dicts of a ``skill.status`` / ``skill.stop`` reply."""
    return [run for run in as_mapping(reply).get("runs") or () if isinstance(run, Mapping)]


# ── Views ───────────────────────────────────────────────────────────────────


def unavailable_entry(node_id: str, refusal: SkillRefusal) -> dict[str, Any]:
    return {"node_id": node_id, "failure_code": refusal.failure_code, "error": refusal.message}


def workcell_view(node_id: str, cell: Mapping[str, Any]) -> dict[str, Any]:
    """A compact, node-qualified workcell card; frame internals stay on the node."""
    skills = []
    for skill in cell.get("skills") or ():
        if not isinstance(skill, Mapping):
            continue
        card = {
            key: skill.get(key)
            for key in ("skill_id", "version", "effect", "card", "args_schema", "consent")
        }
        card["available"] = bool(skill.get("available"))
        if not card["available"]:
            card["reason"] = skill.get("reason")
        skills.append(card)
    return {
        "workcell": f"{node_id}/{cell.get('workcell') or ''}",
        "roles": cell.get("roles"),
        "limits": cell.get("limits"),
        "gripper_spec": cell.get("gripper_spec"),
        "calibration": cell.get("calibration"),
        "calibration_stale": cell.get("calibration_stale") or {},
        "skills": skills,
    }


def plan_view(plan: Mapping[str, Any], consent: str) -> dict[str, Any]:
    """What the approver is shown, and what the digest covers."""
    # Preserve args and basis as well as targets: offline approvals must bind
    # the complete cached node plan, not an abbreviated display of it.
    return {**dict(plan), "consent": consent}


def plan_mismatch(plan: Mapping[str, Any], workcell: str, skill: str) -> str:
    """Why a self-consistent plan is still not the one requested, or ``""``."""
    if plan.get("workcell") != workcell or plan.get("skill_id") != skill:
        return f"the node planned {plan.get('skill_id')!r} on {plan.get('workcell')!r}"
    limits = as_mapping(plan.get("limits"))
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in limits.values()):
        return "the plan limits are not numeric"
    return ""


def run_view(node_id: str, run: Mapping[str, Any]) -> dict[str, Any]:
    """One run as S2 sees it: still moving, or its outcome."""
    view: dict[str, Any] = {
        "ok": True,
        "workcell": f"{node_id}/{run.get('workcell') or ''}",
        "run_id": run.get("run_id"),
        "skill_id": run.get("skill_id"),
        "version": run.get("version"),
        "plan_digest": run.get("plan_digest"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "summary": run.get("summary"),
        "status": run.get("status"),
        "elapsed_s": run.get("elapsed_s"),
    }
    if run.get("stop_reason"):
        view["stop_reason"] = run["stop_reason"]
    outcome = as_mapping(run.get("outcome"))
    if outcome:
        view.update(outcome_fields(outcome))
    else:
        view["side_effect_state"] = SIDE_EFFECT_UNKNOWN  # still moving
    return view


# ── Results ─────────────────────────────────────────────────────────────────


def outcome_fields(outcome: Mapping[str, Any]) -> dict[str, Any]:
    """The tool-result fields of a node ``SkillOutcome`` dict."""
    status = str(outcome.get("status") or STATUS_FAILED)
    side_effect = str(outcome.get("side_effect_state") or SIDE_EFFECT_UNKNOWN)
    fields: dict[str, Any] = {
        "ok": status == STATUS_SUCCEEDED,
        "status": status,
        "failure_code": outcome.get("failure_code"),
        "side_effect_state": side_effect,
        "message": outcome.get("message") or "",
        "result": outcome.get("result") or {},
    }
    if outcome.get("evidence"):
        fields["evidence"] = outcome["evidence"]
    if status != STATUS_SUCCEEDED:
        # ``error`` and ``failure_class`` are the engine's neutral failure
        # contract: the message is the diagnosis, and a node-side defect is
        # declared as one so no recovery loop retries a bug.
        fields["error"] = fields["message"]
        if outcome.get("failure_code") == FAILURE_INTERNAL_DEFECT:
            fields["failure_class"] = FAILURE_INTERNAL_DEFECT
        fields["next_step"] = next_step(outcome.get("failure_code"), side_effect)
        if side_effect != SIDE_EFFECT_NONE:
            fields["effect_uncertain"] = True
    return fields


def outcome_result(workcell: str, skill: str, outcome: Mapping[str, Any]) -> dict[str, Any]:
    return {"workcell": workcell, "skill": skill, **outcome_fields(outcome)}


def refusal_result(
    failure_code: str,
    message: str,
    *,
    side_effect_state: str = SIDE_EFFECT_NONE,
    **extra: Any,
) -> dict[str, Any]:
    """A host-side refusal or uncertainty, shaped like a failed outcome."""
    result = {
        "ok": False,
        **extra,
        "status": STATUS_FAILED,
        "failure_code": failure_code,
        "side_effect_state": side_effect_state,
        "error": message,
        "next_step": next_step(failure_code, side_effect_state),
    }
    if side_effect_state != SIDE_EFFECT_NONE:
        result["effect_uncertain"] = True
    return result


# The engine's tool-result image channel (``engine.tools.tool_attachments``):
# images ride beside a result as content parts, never inside its text.
ATTACHMENTS_FIELD = "_attachments"


def share_frame(result: dict[str, Any], *, consent: bool) -> str:
    """Move a READ outcome's ``result.image`` into the image channel.

    Returns the label of the frame shared (``""`` when none was). A host that did
    not consent drops an image the node should not have sent, so base64 never
    lands in the model's text either way.
    """
    payload = result.get("result")
    if not isinstance(payload, Mapping) or "image" not in payload:
        return ""
    payload = dict(payload)
    image = payload.pop("image")
    result["result"] = payload
    if not consent or not isinstance(image, Mapping):
        return ""
    label = f"frame {payload.get('frame')}, {payload.get('width')}x{payload.get('height')}"
    result[ATTACHMENTS_FIELD] = [
        {"mime": str(image.get("mime") or ""), "data": str(image.get("data") or ""), "label": label}
    ]
    return label


def audit_outcome(outcome: Mapping[str, Any]) -> str:
    return (
        f"{outcome.get('status')} failure={outcome.get('failure_code')} "
        f"side_effect={outcome.get('side_effect_state')}"
    )


__all__ = [
    "ATTACHMENTS_FIELD",
    "FAILURE_AMBIGUOUS_WORKCELL",
    "FAILURE_APPROVAL_DENIED",
    "FAILURE_EXECUTE_UNCONFIRMED",
    "FAILURE_NODE_UNREACHABLE",
    "FAILURE_NO_SKILL_NODES",
    "FAILURE_PERMIT_ISSUER_UNAVAILABLE",
    "FAILURE_PLAN_UNVERIFIED",
    "FAILURE_SKILL_UNSUPPORTED",
    "FAILURE_STATUS_UNAVAILABLE",
    "FAILURE_UNKNOWN_NODE",
    "FAILURE_WRITES_DISABLED",
    "SkillRefusal",
    "as_mapping",
    "audit_outcome",
    "next_step",
    "outcome_fields",
    "outcome_result",
    "plan_mismatch",
    "plan_view",
    "refusal_result",
    "run_view",
    "runs_of",
    "share_frame",
    "unavailable_entry",
    "unreachable",
    "workcell_view",
]
