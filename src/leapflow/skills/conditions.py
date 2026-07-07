"""Pre/postcondition verification for skill execution.

Evaluates declarative condition strings against the runtime environment
before and after skill execution. Unknown conditions pass through silently
to avoid hard rules that limit generalization.
"""

from __future__ import annotations

import logging
import operator
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from leapflow.domain.platform import PlatformManifest, capability_from_str
from leapflow.platform.capabilities import EnvironmentProbe
from leapflow.platform.protocol import HostRpc
from leapflow.skills.registry import Skill, SkillResult

logger = logging.getLogger(__name__)


@dataclass
class ConditionResult:
    """Outcome of a condition check."""

    passed: bool
    reason: str = ""
    failed_conditions: List[str] = field(default_factory=list)


class ConditionChecker:
    """Evaluates skill preconditions and postconditions against runtime state.

    Condition DSL (strings in skill.preconditions / skill.postconditions):
    - "connected"              → platform connection is live
    - "capability:<cap>"       → PlatformManifest has capability
    - "file_exists:<param>"    → named param resolves to existing path
    - "app_running:<bundle>"   → application is active
    - "result_ok"              → postcondition: result.ok is True

    Unknown format → passes silently (graceful degradation).
    """

    def __init__(self, probe: EnvironmentProbe, rpc: HostRpc) -> None:
        self._probe = probe
        self._rpc = rpc

    async def check_preconditions(
        self, skill: Skill, context: Dict[str, Any]
    ) -> ConditionResult:
        """Evaluate all preconditions. Returns on first failure for fast feedback."""
        if not skill.preconditions:
            return ConditionResult(passed=True)

        state = await self._probe.probe()
        failed: List[str] = []

        for cond in skill.preconditions:
            if not await self._evaluate_precondition(cond, context, state):
                failed.append(cond)

        if failed:
            return ConditionResult(
                passed=False,
                reason=f"unmet: {', '.join(failed)}",
                failed_conditions=failed,
            )
        return ConditionResult(passed=True)

    async def verify_postconditions(
        self, skill: Skill, result: SkillResult, context: Dict[str, Any]
    ) -> ConditionResult:
        """Validate postconditions after execution."""
        if not skill.postconditions:
            return ConditionResult(passed=True)

        failed: List[str] = []

        for cond in skill.postconditions:
            if not await self._evaluate_postcondition(cond, result, context):
                failed.append(cond)

        if failed:
            return ConditionResult(
                passed=False,
                reason=f"postcondition unmet: {', '.join(failed)}",
                failed_conditions=failed,
            )
        return ConditionResult(passed=True)

    async def _evaluate_precondition(
        self, cond: str, context: Dict[str, Any], state: Any
    ) -> bool:
        """Evaluate a single precondition string."""
        if cond == "connected":
            return state.connected

        if cond.startswith("capability:"):
            cap_str = cond.split(":", 1)[1]
            cap = capability_from_str(cap_str)
            return cap is not None and state.manifest.supports(cap)

        if cond.startswith("permission:"):
            perm = cond.split(":", 1)[1]
            return state.permissions.get(perm, True)

        if cond.startswith("file_exists:"):
            param_name = cond.split(":", 1)[1]
            path = context.get(param_name)
            if not path:
                return True  # param not provided → skip check
            return await self._check_file_exists(str(path))

        if cond.startswith("app_running:"):
            bundle_id = cond.split(":", 1)[1]
            return await self._check_app_running(bundle_id)

        if cond.startswith("env:"):
            env_var = cond.split(":", 1)[1]
            return bool(os.environ.get(env_var))

        if cond.startswith("param:"):
            return _eval_param_expr(cond[6:], context)

        # Unknown condition → pass through
        return True

    async def _evaluate_postcondition(
        self, cond: str, result: SkillResult, context: Dict[str, Any]
    ) -> bool:
        """Evaluate a single postcondition string."""
        if cond == "result_ok":
            return result.ok

        if cond.startswith("file_exists:"):
            param_name = cond.split(":", 1)[1]
            path = context.get(param_name)
            if not path:
                return True
            return await self._check_file_exists(str(path))

        if cond.startswith("result_has:"):
            key = cond.split(":", 1)[1]
            if isinstance(result.output, dict):
                return key in result.output
            return hasattr(result.output, key)

        # Unknown → pass through
        return True

    async def _check_file_exists(self, path: str) -> bool:
        try:
            result = await self._rpc.call("file.list", {"path": path})
            return bool(result)
        except Exception:
            return False

    async def _check_app_running(self, bundle_id: str) -> bool:
        try:
            result = await self._rpc.call("app.list", {})
            if isinstance(result, list):
                return any(
                    app.get("bundle_id") == bundle_id for app in result
                    if isinstance(app, dict)
                )
            return True  # can't verify → pass
        except Exception:
            return True  # RPC fail → graceful pass


# ── Safe parameter expression evaluator ──

_PARAM_EXPR_RE = re.compile(
    r"^(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)$"
)

_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}


def _eval_param_expr(expr: str, context: Dict[str, Any]) -> bool:
    """Safely evaluate a simple parameter comparison expression.

    Supports: "count > 0", "name != ''", "retries <= 3"
    No eval() — uses regex parse + operator dispatch.
    """
    m = _PARAM_EXPR_RE.match(expr.strip())
    if not m:
        return True  # unparseable → graceful pass

    param_name, op_str, raw_value = m.group(1), m.group(2), m.group(3).strip()
    left = context.get(param_name)
    if left is None:
        return True  # missing param → skip

    right: Any = raw_value
    if raw_value in ("''", '""'):
        right = ""
    elif raw_value.lower() in ("true", "false"):
        right = raw_value.lower() == "true"
    else:
        try:
            right = type(left)(raw_value)
        except (ValueError, TypeError):
            right = raw_value

    try:
        return _OPS[op_str](left, right)
    except (TypeError, KeyError):
        return True
