"""Shared permission-failure predicates for agent and TUI recovery flows."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

PERMISSION_FAILURE_CLASSES = frozenset({"authorization", "scope_denied"})
PERMISSION_FAILURE_CODES = frozenset({"access_denied", "missing_scope", "platform_degraded"})

READINESS_FAILURE_CLASS = "device_not_ready"
"""Failure class marking a command refused because its device is not ready.

A readiness failure is a *feasibility* verdict, not an authorization one: the
caller holds every permission, but the device has not reached the declared state
(homed, initialized, calibrated) that its declaration requires before the write
can be attempted. It is kept distinct from the permission classes above so it is
never mistaken for a scope problem, while still being a hard stop that blocks
approval -- feasibility precedes consent.
"""

_OPERATOR_SYMBOLS: dict[str, str] = {
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
}


def is_permission_failure_payload(payload: Mapping[str, Any] | None) -> bool:
    """Return whether a tool-result payload represents an unresolved permission failure."""
    if not payload or payload.get("ok", True) is not False:
        return False
    failure_class = str(payload.get("failure_class") or "")
    failure_code = str(payload.get("failure_code") or "")
    return failure_class in PERMISSION_FAILURE_CLASSES or failure_code in PERMISSION_FAILURE_CODES


def is_permission_hard_stop_payload(payload: Mapping[str, Any] | None) -> bool:
    """Return whether a failed tool result must stop the current agent turn.

    Permission blockers are external platform boundary conditions, not normal
    retryable tool errors. The agent should surface deterministic recovery
    guidance immediately instead of giving the LLM another chance to retry,
    paraphrase, or invent permission scopes.

    The ``blocks_approval=True`` path is a **global** hard-stop signal: any tool
    result carrying it terminates the turn regardless of whether the failure is
    a platform permission issue.  Hardware readiness failures (IC-1) use this to
    prevent the engine from requesting a follow-up LLM turn after a fail-closed
    interlock refusal, ensuring the deterministic repair instruction reaches the
    user without an intervening model hallucination.
    """
    if not payload or payload.get("ok", True) is not False:
        return False
    if is_permission_failure_payload(payload):
        return True
    if bool(payload.get("blocks_approval")):
        return True
    recoverability = str(payload.get("recoverability") or "")
    retryable = bool(payload.get("retryable", True))
    return recoverability == "admin_required" and not retryable


def _readiness_requirement_phrase(requirement: Mapping[str, Any]) -> str:
    """Render one unmet readiness precondition as an actionable clause.

    A declared precondition names the source channel and the comparison it must
    satisfy, so the caller knows exactly what to bring true. One named on the
    channel but missing from the device declaration is reported as such rather
    than paraphrased into a comparison it does not have -- "cannot be checked"
    and "not satisfied" carry the same weight, and the repair differs.
    """
    name = str(requirement.get("interlock_id") or "(unnamed)")
    description = str(requirement.get("description") or "").strip()
    if not requirement.get("declared", True):
        phrase = f"'{name}' is required by the channel but is not declared on the device"
        return f"{phrase} ({description})" if description else phrase
    channel_id = str(requirement.get("channel_id") or "")
    operator = _OPERATOR_SYMBOLS.get(str(requirement.get("operator") or "eq"), "==")
    value = requirement.get("value")
    phrase = f"'{name}' requires {channel_id} {operator} {value!r}"
    return f"{phrase} ({description})" if description else phrase


def readiness_repair_message(
    device_id: str,
    channel_id: str,
    unmet: Sequence[Mapping[str, Any]],
) -> str:
    """Build the executable repair instruction for an unmet device-readiness state.

    The message names every unsatisfied precondition and the one action that
    resolves them -- bring the device to its declared ready state (its
    initialization / homing / calibration routine), confirm each precondition by
    reading the source channel back, then re-issue the same command. It states
    plainly that no approval was requested, because prompting for a command that
    cannot yet succeed teaches people to click through prompts.
    """
    target = f"{device_id}.{channel_id}" if channel_id else device_id
    clauses = "; ".join(_readiness_requirement_phrase(item) for item in unmet)
    return (
        f"{target} is not ready to command: {clauses}. Bring {device_id} to its "
        "declared ready state -- run its initialization / homing / calibration "
        "routine so every precondition above holds, confirm it by reading the "
        "source channel back, then re-issue the same command. No approval was "
        "requested because the command cannot succeed until the device is ready."
    )


def build_readiness_failure(
    *,
    device_id: str,
    channel_id: str,
    unmet: Sequence[Mapping[str, Any]],
    failure_code: str = "not_ready",
) -> dict[str, Any]:
    """Build a deterministic hard-stop for a command refused on device readiness.

    This is the single authority the engine and TUI both consult, so a device
    that has not reached its declared ready state is reported identically
    everywhere. ``blocks_approval`` makes it a hard stop under
    ``is_permission_hard_stop_payload`` without misclassifying it as a permission
    failure; ``retryable`` is True because the identical command becomes feasible
    once the readiness preconditions hold. The caller owns any transport-specific
    fields (such as the side-effect verdict), since nothing physical was touched.
    """
    return {
        "ok": False,
        "device_id": device_id,
        "channel_id": channel_id,
        "failure_code": failure_code,
        "failure_class": READINESS_FAILURE_CLASS,
        "blocks_approval": True,
        "retryable": True,
        "recoverability": "ready_state_required",
        "error": readiness_repair_message(device_id, channel_id, unmet),
        "repair": {
            "kind": "device_readiness",
            "device_id": device_id,
            "channel_id": channel_id,
            "unmet": [dict(item) for item in unmet],
        },
    }
