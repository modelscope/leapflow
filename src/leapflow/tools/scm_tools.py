"""Typed source-control tools.

The SCM tool intentionally models git operations as structured actions instead
of asking the model to synthesize raw shell commands. This keeps ref semantics
explicit: pulling from ``origin/main`` does not imply pushing to ``origin/main``.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Sequence

from leapflow.security.redact import redact_sensitive_text

_MAX_OUTPUT_CHARS = 10_000
_DEFAULT_TIMEOUT_S = 120.0
_ALLOWED_ACTIONS = frozenset({"status", "pull", "push", "pull_then_push"})
_QUERY_ACTIONS = frozenset({"diff", "log", "status", "branch", "show"})
_LOG_FORMAT = "%h%x1f%an%x1f%ad%x1f%s"  # unit-separator delimited, safe for parsing


@dataclass(frozen=True)
class GitCommandResult:
    """Result from one git command invocation."""

    returncode: int
    stdout: str
    stderr: str


GitRunner = Callable[[Sequence[str], Path, float], Awaitable[GitCommandResult]]


def _clip_output(value: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    text = redact_sensitive_text(str(value or ""), force=True)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _workspace(cwd: Any) -> Path:
    raw = str(cwd or ".").strip() or "."
    return Path(raw).expanduser().resolve()


def _safe_ref(value: Any, *, field: str) -> str:
    ref = str(value or "").strip()
    if not ref:
        return ""
    if ref.startswith("-") or any(ch.isspace() for ch in ref) or ".." in ref:
        raise ValueError(f"Invalid git ref for {field}: {ref}")
    return ref


async def _run_git(args: Sequence[str], cwd: Path, timeout_s: float) -> GitCommandResult:
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        proc.kill()
        await proc.communicate()
        return GitCommandResult(returncode=124, stdout="", stderr=f"git command timed out after {timeout_s:.0f}s")
    return GitCommandResult(
        returncode=proc.returncode,
        stdout=_clip_output(stdout.decode("utf-8", errors="replace")),
        stderr=_clip_output(stderr.decode("utf-8", errors="replace")),
    )


def _step_payload(step: str, args: Sequence[str], result: GitCommandResult) -> Dict[str, Any]:
    return {
        "step": step,
        "command": "git " + " ".join(args),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "ok": result.returncode == 0,
    }


def _failure_payload(
    *,
    action: str,
    cwd: Path,
    step: str,
    args: Sequence[str],
    result: GitCommandResult,
    completed_steps: list[Dict[str, Any]],
    current_branch: str = "",
) -> Dict[str, Any]:
    error = result.stderr.strip() or result.stdout.strip() or f"git {step} failed with exit code {result.returncode}"
    return {
        "ok": False,
        "tool": "scm_sync",
        "scm": "git",
        "action": action,
        "cwd": str(cwd),
        "current_branch": current_branch,
        "failed_step": step,
        "failure_code": f"git_{step}_failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": error,
        "command": "git " + " ".join(args),
        "completed_steps": completed_steps,
        "retryable": False,
    }


async def _current_branch(cwd: Path, timeout_s: float, runner: GitRunner) -> tuple[str, Dict[str, Any] | None]:
    args = ("branch", "--show-current")
    result = await runner(args, cwd, timeout_s)
    if result.returncode != 0:
        return "", _failure_payload(
            action="resolve_current_branch",
            cwd=cwd,
            step="branch",
            args=args,
            result=result,
            completed_steps=[],
        )
    branch = result.stdout.strip()
    if not branch:
        detached = GitCommandResult(returncode=1, stdout=result.stdout, stderr="Current HEAD is detached; explicit push_ref is required.")
        return "", _failure_payload(
            action="resolve_current_branch",
            cwd=cwd,
            step="branch",
            args=args,
            result=detached,
            completed_steps=[],
        )
    return branch, None


async def scm_sync(params: Dict[str, Any], runner: GitRunner | None = None) -> Dict[str, Any]:
    """Run a structured git sync action.

    Parameters
    ----------
    action:
        ``status``, ``pull``, ``push``, or ``pull_then_push``.
    remote:
        Git remote used for pull/push. Defaults to ``origin``.
    pull_ref:
        Ref to pull from. For the common request "pull origin main then push",
        this is ``main``.
    push_ref:
        Ref to push. Defaults to the literal current local branch, not
        ``pull_ref``. Use ``current_branch`` or omit it to keep that behavior.
    """
    action = str(params.get("action") or "pull_then_push").strip().lower()
    if action not in _ALLOWED_ACTIONS:
        return {"ok": False, "error": f"Unsupported SCM action: {action}", "failure_code": "unsupported_scm_action"}

    cwd = _workspace(params.get("cwd"))
    if not cwd.exists():
        return {"ok": False, "error": f"Working directory does not exist: {cwd}", "failure_code": "path_not_found", "cwd": str(cwd)}
    if not cwd.is_dir():
        return {"ok": False, "error": f"Working directory is not a directory: {cwd}", "failure_code": "path_not_directory", "cwd": str(cwd)}

    timeout_s = min(float(params.get("timeout") or _DEFAULT_TIMEOUT_S), _DEFAULT_TIMEOUT_S)
    run = runner or _run_git
    remote = _safe_ref(params.get("remote") or "origin", field="remote")
    pull_ref = _safe_ref(params.get("pull_ref") or "", field="pull_ref")
    push_ref = _safe_ref(params.get("push_ref") or "current_branch", field="push_ref")
    completed_steps: list[Dict[str, Any]] = []

    current_branch = ""
    if action in {"push", "pull_then_push"} and push_ref in {"", "current", "current_branch"}:
        current_branch, branch_failure = await _current_branch(cwd, timeout_s, run)
        if branch_failure is not None:
            branch_failure["action"] = action
            return branch_failure
        push_ref = current_branch

    if action == "status":
        args = ("status", "--short", "--branch")
        result = await run(args, cwd, timeout_s)
        return _failure_payload(action=action, cwd=cwd, step="status", args=args, result=result, completed_steps=[]) if result.returncode != 0 else {
            "ok": True,
            "tool": "scm_sync",
            "scm": "git",
            "action": action,
            "cwd": str(cwd),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "completed_steps": [_step_payload("status", args, result)],
        }

    if action in {"pull", "pull_then_push"}:
        pull_args = ("pull", remote, pull_ref) if pull_ref else ("pull", remote)
        pull_result = await run(pull_args, cwd, timeout_s)
        if pull_result.returncode != 0:
            return _failure_payload(
                action=action,
                cwd=cwd,
                step="pull",
                args=pull_args,
                result=pull_result,
                completed_steps=completed_steps,
                current_branch=current_branch,
            )
        completed_steps.append(_step_payload("pull", pull_args, pull_result))

    if action in {"push", "pull_then_push"}:
        push_args = ("push", remote, push_ref)
        push_result = await run(push_args, cwd, timeout_s)
        if push_result.returncode != 0:
            return _failure_payload(
                action=action,
                cwd=cwd,
                step="push",
                args=push_args,
                result=push_result,
                completed_steps=completed_steps,
                current_branch=current_branch,
            )
        completed_steps.append(_step_payload("push", push_args, push_result))

    return {
        "ok": True,
        "tool": "scm_sync",
        "scm": "git",
        "action": action,
        "cwd": str(cwd),
        "remote": remote,
        "pull_ref": pull_ref,
        "push_ref": push_ref,
        "current_branch": current_branch,
        "completed": True,
        "completed_steps": completed_steps,
        "stdout": "\n".join(step.get("stdout", "") for step in completed_steps if step.get("stdout")),
        "stderr": "\n".join(step.get("stderr", "") for step in completed_steps if step.get("stderr")),
    }


def _parse_log_entries(stdout: str) -> list[Dict[str, Any]]:
    """Parse unit-separator delimited git log lines into structured entries."""
    entries: list[Dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\x1f")
        if len(parts) >= 4:
            entries.append({"hash": parts[0], "author": parts[1], "date": parts[2], "subject": parts[3]})
    return entries


def _parse_branches(stdout: str) -> tuple[str, list[str]]:
    """Parse ``git branch -a`` output into (current, all-branches)."""
    current = ""
    branches: list[str] = []
    for line in stdout.splitlines():
        name = line.strip()
        if not name:
            continue
        if name.startswith("* "):
            name = name[2:].strip()
            current = name
        branches.append(name)
    return current, branches


async def git_query(params: Dict[str, Any], runner: GitRunner | None = None) -> Dict[str, Any]:
    """Read-only structured git inspection: diff, log, status, branch, show.

    Prefer this over shell_run for reading repository state — output is clipped,
    secret-redacted, and (for log/branch) parsed into structured fields. Never
    mutates the repo; use scm_sync for pull/push.
    """
    action = str(params.get("action") or "status").strip().lower()
    if action not in _QUERY_ACTIONS:
        return {"ok": False, "error": f"Unsupported git query action: {action}", "failure_code": "unsupported_git_query"}

    cwd = _workspace(params.get("cwd"))
    if not cwd.exists() or not cwd.is_dir():
        return {"ok": False, "error": f"Working directory not found: {cwd}", "failure_code": "path_not_found", "cwd": str(cwd)}

    timeout_s = min(float(params.get("timeout") or _DEFAULT_TIMEOUT_S), _DEFAULT_TIMEOUT_S)
    run = runner or _run_git
    ref = _safe_ref(params.get("ref") or "", field="ref")
    path = str(params.get("path") or "").strip()
    if path.startswith("-"):
        return {"ok": False, "error": f"Invalid path: {path}", "failure_code": "invalid_path"}

    if action == "diff":
        args: list[str] = ["diff", "--no-color"]
        if params.get("staged"):
            args.append("--staged")
        if ref:
            args.append(ref)
    elif action == "log":
        try:
            max_count = max(1, min(int(params.get("max_count", 20)), 200))
        except (TypeError, ValueError):
            max_count = 20
        args = ["log", f"-n{max_count}", f"--pretty=format:{_LOG_FORMAT}", "--date=short", "--no-color"]
        if params.get("stat"):
            args.append("--stat")
        if ref:
            args.append(ref)
    elif action == "branch":
        args = ["branch", "-a", "--no-color"]
    elif action == "show":
        args = ["show", "--stat", "--no-color", ref or "HEAD"]
    else:  # status
        args = ["status", "--short", "--branch"]
    if path and action in {"diff", "log"}:
        args += ["--", path]

    result = await run(tuple(args), cwd, timeout_s)
    if result.returncode != 0:
        return _failure_payload(action=action, cwd=cwd, step=action, args=args, result=result, completed_steps=[])

    payload: Dict[str, Any] = {
        "ok": True,
        "tool": "git_query",
        "scm": "git",
        "action": action,
        "cwd": str(cwd),
        "command": "git " + " ".join(args),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if action == "log":
        payload["entries"] = _parse_log_entries(result.stdout)
        payload["entry_count"] = len(payload["entries"])
    elif action == "branch":
        current, branches = _parse_branches(result.stdout)
        payload["current_branch"] = current
        payload["branches"] = branches
    return payload
