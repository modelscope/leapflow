"""Shared permission-failure predicates for agent and TUI recovery flows."""
from __future__ import annotations

from typing import Any, Mapping

PERMISSION_FAILURE_CLASSES = frozenset({"authorization", "scope_denied"})
PERMISSION_FAILURE_CODES = frozenset({"access_denied", "missing_scope", "platform_degraded"})


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
