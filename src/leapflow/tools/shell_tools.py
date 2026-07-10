"""Shell command execution with timeout, safety, and output redaction.

All handlers follow the ToolBridge convention: receive params dict, return result dict.
Safety layers:
1. Hardline block: always-blocked destructive patterns (rm -rf /, fork bomb, etc.)
2. Dangerous detection: patterns requiring user confirmation (sudo, chmod, etc.)
3. Output redaction: secrets stripped from stdout/stderr before returning to LLM
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
from typing import Any, Dict, FrozenSet, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_MAX_STDOUT = 10_000
_MAX_STDERR = 5_000
_DEFAULT_TIMEOUT = 30.0


@runtime_checkable
class CommandApprovalGate(Protocol):
    """Protocol for command approval (injectable, no hardcoded behavior)."""

    async def check(self, command: str) -> bool:
        """Return True if the command is approved, False to block."""
        ...


@runtime_checkable
class ActionApprovalEvaluator(Protocol):
    """Protocol for structured action approval evaluators."""

    async def evaluate(self, action: Any) -> Any:
        """Return an approval result for a structured action."""
        ...


# Hardline blocks: NEVER bypassed regardless of approval
_HARDLINE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+.*-[^\s]*r[^\s]*f|\brm\s+.*-[^\s]*f[^\s]*r", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\s+/\s*$", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bdd\s+.*of=/dev/", re.IGNORECASE),
    re.compile(r":()\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),  # fork bomb
    re.compile(r"\b>\.?/dev/[sh]d[a-z]", re.IGNORECASE),
    re.compile(r"\bshutdown\b|\breboot\b|\bhalt\b|\bpoweroff\b", re.IGNORECASE),
]

# Dangerous patterns: blocked by default, approvable via CommandApprovalGate
_DANGEROUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\bchmod\s+[0-7]*7[0-7]*\b", re.IGNORECASE),
    re.compile(r"\bchown\b", re.IGNORECASE),
    re.compile(r"\bcurl\b.*\|\s*(ba)?sh", re.IGNORECASE),
    re.compile(r"\bwget\b.*\|\s*(ba)?sh", re.IGNORECASE),
    re.compile(r"\b(?:python[23]?|perl|ruby|node|bash|sh|zsh|ksh)\s+<<", re.IGNORECASE),
    re.compile(r"\b(pip|npm|brew)\s+install\b", re.IGNORECASE),
    re.compile(r"\bgit\s+push\s+.*--force\b", re.IGNORECASE),
    re.compile(r"\brm\s+-r\b", re.IGNORECASE),
    re.compile(r"\bkill\s+-9\b", re.IGNORECASE),
    re.compile(r"\biptables\b|\bnft\b", re.IGNORECASE),
    re.compile(r"\bsystemctl\s+(stop|disable|mask)\b", re.IGNORECASE),
]

# CWD paths that should never be used for shell execution
_BLOCKED_CWD_PREFIXES: FrozenSet[str] = frozenset({
    "/System", "/usr", "/bin", "/sbin", "/var/root",
})

# Module-level approval gate (injected by orchestrator; None = auto-deny dangerous)
_approval_gate: CommandApprovalGate | None = None


def set_approval_gate(gate: CommandApprovalGate | None) -> None:
    """Install a command approval gate for dangerous-command review."""
    global _approval_gate
    _approval_gate = gate


def _is_hardline_blocked(command: str) -> bool:
    return any(p.search(command) for p in _HARDLINE_PATTERNS)


def _is_dangerous(command: str) -> bool:
    return any(p.search(command) for p in _DANGEROUS_PATTERNS)


def _is_cwd_blocked(cwd: str | None) -> bool:
    if not cwd:
        return False
    resolved = os.path.realpath(os.path.expanduser(cwd))
    return any(resolved.startswith(prefix) for prefix in _BLOCKED_CWD_PREFIXES)


async def _approve_command(command: str, cwd: str | None) -> tuple[bool, str]:
    if _approval_gate is None:
        return False, "Dangerous command blocked (no approval gate configured)"
    try:
        if isinstance(_approval_gate, ActionApprovalEvaluator):
            from leapflow.security.actions import ActionDescriptor

            result = await _approval_gate.evaluate(ActionDescriptor.shell(command, cwd=cwd))
            if getattr(result, "approved", False):
                return True, ""
            message = str(getattr(result, "denial_message", "") or "Dangerous command requires approval (denied)")
            return False, message
        approved = await _approval_gate.check(command)
        return approved, "" if approved else "Dangerous command requires approval (denied)"
    except Exception:
        logger.debug("shell approval check failed", exc_info=True)
        return False, "Dangerous command requires approval (denied)"


async def shell_run(params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a shell command with timeout protection and safety layers."""
    command = params.get("command", "")
    cwd = params.get("cwd") or None
    timeout = min(float(params.get("timeout", _DEFAULT_TIMEOUT)), 120.0)

    if not command:
        return {"ok": False, "error": "Missing required parameter: command"}

    if _is_hardline_blocked(command):
        return {"ok": False, "error": "Command blocked by safety policy (destructive pattern detected)"}

    if _is_cwd_blocked(cwd):
        return {"ok": False, "error": f"Working directory blocked by safety policy: {cwd}"}

    if _is_dangerous(command):
        approved, message = await _approve_command(command, cwd)
        if not approved:
            return {"ok": False, "error": message}

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            start_new_session=True,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
        stdout_text = stdout_bytes.decode(errors="replace")[:_MAX_STDOUT]
        stderr_text = stderr_bytes.decode(errors="replace")[:_MAX_STDERR]

        # Redact secrets from output before returning to LLM
        try:
            from leapflow.security.redact import redact_sensitive_text
            stdout_text = redact_sensitive_text(stdout_text)
            stderr_text = redact_sensitive_text(stderr_text)
        except ImportError:
            pass

        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": stdout_text,
            "stderr": stderr_text,
        }
    except asyncio.TimeoutError:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # type: ignore[possibly-undefined]
        except (ProcessLookupError, OSError):
            try:
                proc.kill()  # type: ignore[possibly-undefined]
            except ProcessLookupError:
                pass
        return {"ok": False, "error": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
