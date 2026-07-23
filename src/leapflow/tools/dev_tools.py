"""Structured developer verification tools: test_run and lint_check.

Both are thin, structured wrappers over ``shell_run``: they auto-detect (or take
a configured / explicit) command, execute it through ``shell_run`` — inheriting
its hardline/danger/approval/timeout/redaction governance — and parse the output
into structured pass/fail (tests) or issue (lint) fields.

Semantics: ``ok`` means the *runner executed*, not that tests passed / lint was
clean. A failing test suite returns ``ok=True`` with ``success=False`` so it is
informative feedback, never a side-effecting tool error that would halt a batch.
``ok=False`` only when the runner could not run (missing command, blocked,
timed out). Classified read_only for the loop; the real execution safety lives
in the underlying ``shell_run`` gate.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from leapflow.tools.shell_tools import shell_run

logger = logging.getLogger(__name__)

_MAX_OUTPUT = 6000

# Configured command overrides (empty => auto-detect). Injected from settings via
# set_dev_commands(), mirroring the approval-gate injection pattern.
_CONFIGURED: Dict[str, str] = {"test": "", "lint": ""}


def set_dev_commands(*, test_command: str = "", lint_command: str = "") -> None:
    """Register configured test/lint command overrides (empty => auto-detect)."""
    _CONFIGURED["test"] = str(test_command or "").strip()
    _CONFIGURED["lint"] = str(lint_command or "").strip()


def _search_int(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else 0


def _detect_test_command(cwd: Path) -> str:
    if (cwd / "pyproject.toml").exists() or (cwd / "pytest.ini").exists() or (cwd / "setup.cfg").exists() or (cwd / "tests").is_dir():
        return "python -m pytest -q"
    if (cwd / "package.json").exists():
        return "npm test"
    if (cwd / "go.mod").exists():
        return "go test ./..."
    if (cwd / "Cargo.toml").exists():
        return "cargo test"
    return ""


def _detect_lint_command(cwd: Path) -> str:
    if (cwd / "ruff.toml").exists() or (cwd / ".ruff.toml").exists() or (cwd / "pyproject.toml").exists():
        return "ruff check ."
    if (cwd / "package.json").exists():
        return "npx --no-install eslint ."
    if (cwd / "go.mod").exists():
        return "go vet ./..."
    if (cwd / "Cargo.toml").exists():
        return "cargo clippy"
    return ""


def _framework_for(command: str) -> str:
    lowered = command.lower()
    for marker, name in (("pytest", "pytest"), ("ruff", "ruff"), ("eslint", "eslint"),
                         ("go test", "go"), ("go vet", "go"), ("cargo test", "cargo"),
                         ("cargo clippy", "cargo"), ("npm test", "npm")):
        if marker in lowered:
            return name
    return "generic"


def _parse_pytest(text: str, returncode: Optional[int]) -> Dict[str, Any]:
    passed = _search_int(r"(\d+) passed", text)
    failed = _search_int(r"(\d+) failed", text)
    errors = _search_int(r"(\d+) errors?", text)
    skipped = _search_int(r"(\d+) skipped", text)
    failures: List[str] = [ln.strip() for ln in text.splitlines() if ln.startswith("FAILED ")][:50]
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "failures": failures,
        "success": returncode == 0 and failed == 0 and errors == 0,
    }


def _runner_failed(result: Dict[str, Any]) -> bool:
    # shell_run returns a returncode only when the process actually executed;
    # missing/blocked/timed-out runs come back as {"ok": False, "error": ...}.
    return "returncode" not in result


async def test_run(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run the project's test suite and return structured pass/fail results."""
    cwd = Path(str(params.get("cwd") or ".")).expanduser().resolve()
    if not cwd.is_dir():
        return {"ok": False, "error": f"Working directory not found: {cwd}", "failure_code": "path_not_found"}
    command = str(params.get("command") or "").strip() or _CONFIGURED["test"] or _detect_test_command(cwd)
    if not command:
        return {
            "ok": False,
            "error": "Could not detect a test command; pass command= or set tools.test_command.",
            "failure_code": "no_test_command",
            "cwd": str(cwd),
        }

    result = await shell_run({"command": command, "cwd": str(cwd), "timeout": params.get("timeout", 120)})
    if _runner_failed(result):
        return {"ok": False, "error": result.get("error", "test runner failed to execute"),
                "command": command, "cwd": str(cwd), "failure_code": "runner_error"}

    framework = _framework_for(command)
    stdout = str(result.get("stdout", ""))
    stderr = str(result.get("stderr", ""))
    returncode = result.get("returncode")
    payload: Dict[str, Any] = {
        "ok": True,
        "tool": "test_run",
        "command": command,
        "cwd": str(cwd),
        "framework": framework,
        "returncode": returncode,
        "stdout": stdout[:_MAX_OUTPUT],
        "stderr": stderr[:_MAX_OUTPUT],
    }
    if framework == "pytest":
        payload.update(_parse_pytest(stdout + "\n" + stderr, returncode))
    else:
        payload["success"] = returncode == 0
    return payload


async def lint_check(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run the project's linter and return a structured clean/issue result."""
    cwd = Path(str(params.get("cwd") or ".")).expanduser().resolve()
    if not cwd.is_dir():
        return {"ok": False, "error": f"Working directory not found: {cwd}", "failure_code": "path_not_found"}
    command = str(params.get("command") or "").strip() or _CONFIGURED["lint"] or _detect_lint_command(cwd)
    if not command:
        return {
            "ok": False,
            "error": "Could not detect a lint command; pass command= or set tools.lint_command.",
            "failure_code": "no_lint_command",
            "cwd": str(cwd),
        }

    result = await shell_run({"command": command, "cwd": str(cwd), "timeout": params.get("timeout", 120)})
    if _runner_failed(result):
        return {"ok": False, "error": result.get("error", "linter failed to execute"),
                "command": command, "cwd": str(cwd), "failure_code": "runner_error"}

    stdout = str(result.get("stdout", ""))
    stderr = str(result.get("stderr", ""))
    returncode = result.get("returncode")
    combined = stdout + "\n" + stderr
    # ruff prints "Found N errors"; otherwise fall back to counting diagnostic lines.
    issue_count = _search_int(r"Found (\d+) error", combined)
    if issue_count == 0 and returncode != 0:
        issue_count = sum(1 for ln in combined.splitlines() if re.search(r":\d+:\d+:", ln))
    return {
        "ok": True,
        "tool": "lint_check",
        "command": command,
        "cwd": str(cwd),
        "framework": _framework_for(command),
        "returncode": returncode,
        "clean": returncode == 0,
        "issue_count": issue_count,
        "stdout": stdout[:_MAX_OUTPUT],
        "stderr": stderr[:_MAX_OUTPUT],
    }
