"""Shell command execution with timeout and safety.

All handlers follow the ToolBridge convention: receive params dict, return result dict.
Safety: commands execute with configurable timeout; output is truncated to prevent OOM.
Dangerous commands are blocked by a configurable blacklist.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
from typing import Any, Dict


# Hard limits to prevent resource exhaustion
_MAX_STDOUT = 10_000
_MAX_STDERR = 5_000
_DEFAULT_TIMEOUT = 30.0

# Safety: block obviously destructive commands (configurable via override)
_DANGEROUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+.*-[^\s]*r[^\s]*f|\brm\s+.*-[^\s]*f[^\s]*r", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\s+/\s*$", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bdd\s+.*of=/dev/", re.IGNORECASE),
    re.compile(r":(){ :\|:& };:"),  # fork bomb
    re.compile(r"\b>\.?/dev/[sh]d[a-z]", re.IGNORECASE),
]


def _is_dangerous(command: str) -> bool:
    """Check command against the dangerous-patterns blacklist."""
    for pat in _DANGEROUS_PATTERNS:
        if pat.search(command):
            return True
    return False


async def shell_run(params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a shell command with timeout protection."""
    command = params.get("command", "")
    cwd = params.get("cwd") or None
    timeout = min(float(params.get("timeout", _DEFAULT_TIMEOUT)), 120.0)

    if not command:
        return {"ok": False, "error": "Missing required parameter: command"}

    if _is_dangerous(command):
        return {"ok": False, "error": "Command blocked by safety policy (destructive pattern detected)"}

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            start_new_session=True,  # Create new process group for clean kill
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": stdout.decode(errors="replace")[:_MAX_STDOUT],
            "stderr": stderr.decode(errors="replace")[:_MAX_STDERR],
        }
    except asyncio.TimeoutError:
        # Kill entire process group (shell + children)
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
