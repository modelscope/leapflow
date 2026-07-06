"""Human-in-the-loop confirmation for skill execution.

Implements the graduation mechanism: skills progress from STEP → CONFIRM →
NOTIFY → AUTO as they mature through successful executions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Protocol, runtime_checkable

from leapflow.skills.registry import Skill

if TYPE_CHECKING:
    from leapflow.storage.skill_library import SkillLibraryStore

logger = logging.getLogger(__name__)

_DESTRUCTIVE_ACTIONS = frozenset({
    "file.delete", "batch_delete", "shell.exec",
    "file.rename", "batch_rename",
})

StepExecutor = Callable[[int, str], Awaitable[Dict[str, Any]]]
StepCallback = Callable[[int, int, str], None]


class ConfirmLevel(Enum):
    AUTO = "auto"
    NOTIFY = "notify"
    CONFIRM = "confirm"
    STEP = "step"


@dataclass(frozen=True)
class StepResult:
    step_idx: int
    total: int
    description: str
    status: str  # "pending" | "completed" | "skipped"
    output: Any = None


@runtime_checkable
class IOProvider(Protocol):
    """Decoupled I/O for confirmation prompts. Supports CLI, API, and testing."""

    async def prompt(self, message: str) -> str: ...
    async def display(self, message: str) -> None: ...


class ConfirmationHandler:
    """Determines and manages confirmation levels for skill execution."""

    def __init__(self, *, skill_store: Optional["SkillLibraryStore"] = None) -> None:
        self._skill_store = skill_store
        self._step_callback: Optional[StepCallback] = None

    def set_on_step(self, callback: Optional[StepCallback]) -> None:
        """Register a callback invoked at the start of each step in step-through mode.

        Signature: ``callback(step_idx, total_steps, step_description)``.
        """
        self._step_callback = callback

    def determine_level(
        self,
        skill: Skill,
        *,
        override: Optional[ConfirmLevel] = None,
    ) -> ConfirmLevel:
        if override is not None:
            return override

        from leapflow.domain.skill_types import SkillTier

        meta = skill.metadata
        tier = meta.tier

        if tier <= SkillTier.DRAFT:
            return ConfirmLevel.STEP

        if self._has_destructive_ops(skill):
            return ConfirmLevel.CONFIRM

        if tier >= SkillTier.VERIFIED:
            if self._has_recent_regression(skill):
                return ConfirmLevel.CONFIRM
            return ConfirmLevel.AUTO

        return ConfirmLevel.NOTIFY

    def _has_recent_regression(self, skill: Skill) -> bool:
        """Check if any of the last 5 executions had a 'regressed' verdict."""
        if not self._skill_store:
            return False
        try:
            stored = self._skill_store.load_skill_by_title(skill.name)
            if not stored:
                return False
            executions = self._skill_store.load_executions(stored.skill_id, limit=5)
            return any(e.verdict == "regressed" for e in executions)
        except Exception:
            return False

    async def request_confirmation(
        self,
        skill: Skill,
        params: Dict[str, Any],
        level: ConfirmLevel,
        io: IOProvider,
    ) -> str:
        """Returns 'yes', 'no', or 'step'."""
        if level == ConfirmLevel.AUTO:
            return "yes"

        if level == ConfirmLevel.NOTIFY:
            plan = self._format_plan(skill, params)
            await io.display(plan)
            return "yes"

        plan = self._format_plan(skill, params)

        if level == ConfirmLevel.CONFIRM:
            await io.display(plan)
            response = await io.prompt("Execute? (yes/no/step) ")
            return self._normalize_response(response)

        if level == ConfirmLevel.STEP:
            await io.display(plan)
            response = await io.prompt("Step-by-step execution. Start? (yes/no) ")
            normalized = self._normalize_response(response)
            return "step" if normalized == "yes" else normalized

        return "no"

    async def step_through(
        self,
        skill: Skill,
        steps: List[str],
        io: IOProvider,
        *,
        executor: Optional[StepExecutor] = None,
    ) -> List[StepResult]:
        results: List[StepResult] = []
        for i, step_desc in enumerate(steps):
            if self._step_callback:
                try:
                    self._step_callback(i, len(steps), step_desc)
                except Exception as e:
                    logger.debug("confirmation.step_callback_error error=%s", e)
            await io.display(
                f"\nStep {i + 1}/{len(steps)}: {step_desc}"
            )
            response = await io.prompt("  Continue? (yes/skip/stop) ")
            decision = self._normalize_response(response)

            if decision == "stop":
                results.append(StepResult(i, len(steps), step_desc, "skipped"))
                break
            if decision == "skip":
                results.append(StepResult(i, len(steps), step_desc, "skipped"))
                continue

            output = None
            status = "completed"
            if executor:
                try:
                    output = await executor(i, step_desc)
                except Exception as e:
                    status = "failed"
                    output = str(e)
                    await io.display(f"  Step failed: {e}")

            results.append(StepResult(i, len(steps), step_desc, status, output))
            if status == "failed":
                response = await io.prompt("  Continue anyway? (yes/no) ")
                if self._normalize_response(response) == "no":
                    break

        return results

    def _format_plan(self, skill: Skill, params: Dict[str, Any]) -> str:
        lines = [f"Skill: {skill.name}"]
        lines.append(f"  Description: {skill.description}")
        lines.append(f"  Version: v{skill.metadata.version} ({skill.metadata.tier.name})")
        lines.append(f"  Confidence: {skill.metadata.confidence:.0%}")
        if params:
            lines.append("  Parameters:")
            for k, v in params.items():
                lines.append(f"    {k} = {v}")
        if skill.preconditions:
            lines.append(f"  Preconditions: {skill.preconditions}")
        return "\n".join(lines)

    @staticmethod
    def _has_destructive_ops(skill: Skill) -> bool:
        for trigger in skill.triggers:
            for action in _DESTRUCTIVE_ACTIONS:
                if action in trigger.lower():
                    return True
        desc = skill.description.lower()
        return any(kw in desc for kw in ("delete", "remove", "drop", "shell"))

    @staticmethod
    def _normalize_response(response: str) -> str:
        r = response.strip().lower()
        if r in ("y", "yes", "确认", "继续", "好", "next"):
            return "yes"
        if r in ("n", "no", "取消", "不"):
            return "no"
        if r in ("step", "逐步", "s"):
            return "step"
        if r in ("skip", "跳过"):
            return "skip"
        if r in ("stop", "中止", "停"):
            return "stop"
        return "yes"


@dataclass(frozen=True)
class RiskAssessment:
    """Result of dangerous operation detection."""
    action: str
    severity: float  # 0.0-1.0
    risks: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    requires_double_confirmation: bool = False


class DangerousOperationDetector:
    """Detects high-risk operations that require elevated confirmation.
    
    Identifies:
    - Batch operations exceeding threshold
    - Irreversible operations (delete, format, overwrite)
    - Operations touching sensitive paths
    - Operations with broad scope (wildcards, recursive)
    """

    def __init__(
        self,
        *,
        batch_threshold: int = 10,
        sensitive_paths: frozenset[str] = frozenset({
            "/etc", "/usr", "/System", "/bin", "/sbin",
            "~/.ssh", "~/.config", "~/.gnupg",
        }),
        irreversible_actions: frozenset[str] = frozenset({
            "file.delete", "batch_delete", "shell.exec",
            "file.overwrite", "disk.format", "git.force_push",
        }),
    ) -> None:
        self._batch_threshold = batch_threshold
        self._sensitive_paths = sensitive_paths
        self._irreversible_actions = irreversible_actions

    def assess_risk(self, action: str, params: Dict[str, Any]) -> RiskAssessment:
        """Assess the risk level of an operation."""
        risks: List[str] = []
        severity = 0.0
        
        # Check irreversible
        if action in self._irreversible_actions:
            risks.append("irreversible_operation")
            severity = max(severity, 0.8)
        
        # Check batch size
        batch_size = params.get("batch_size", params.get("count", 1))
        if isinstance(batch_size, int) and batch_size > self._batch_threshold:
            risks.append(f"batch_operation ({batch_size} items)")
            severity = max(severity, 0.6)
        
        # Check sensitive paths
        target_path = params.get("path", params.get("target", ""))
        if isinstance(target_path, str) and target_path:
            for sensitive in self._sensitive_paths:
                expanded = sensitive.replace("~", str(Path.home())) if "~" in sensitive else sensitive
                if expanded == "/":
                    # Root "/" only matches exact target "/"
                    if target_path == "/":
                        risks.append(f"sensitive_path ({sensitive})")
                        severity = max(severity, 0.9)
                        break
                elif target_path.startswith(expanded):
                    risks.append(f"sensitive_path ({sensitive})")
                    severity = max(severity, 0.9)
                    break
        
        # Check wildcards / recursive
        if any(params.get(k) for k in ("recursive", "wildcard", "glob")):
            risks.append("broad_scope (recursive/wildcard)")
            severity = max(severity, 0.5)
        
        return RiskAssessment(
            action=action,
            severity=severity,
            risks=risks,
            requires_confirmation=severity >= 0.5,
            requires_double_confirmation=severity >= 0.8,
        )
