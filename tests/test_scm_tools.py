from __future__ import annotations

import pytest

from leapflow.tools.scm_tools import GitCommandResult, scm_sync, git_query, git_write


@pytest.mark.asyncio
async def test_scm_sync_pull_then_push_defaults_to_current_branch(tmp_path) -> None:
    commands: list[tuple[str, ...]] = []

    async def fake_runner(args, cwd, timeout_s):
        commands.append(tuple(args))
        if tuple(args) == ("branch", "--show-current"):
            return GitCommandResult(returncode=0, stdout="feature/refactor\n", stderr="")
        return GitCommandResult(returncode=0, stdout="ok", stderr="")

    result = await scm_sync(
        {
            "action": "pull_then_push",
            "cwd": str(tmp_path),
            "remote": "origin",
            "pull_ref": "main",
        },
        runner=fake_runner,
    )

    assert result["ok"] is True
    assert result["push_ref"] == "feature/refactor"
    assert commands == [
        ("branch", "--show-current"),
        ("pull", "origin", "main"),
        ("push", "origin", "feature/refactor"),
    ]


@pytest.mark.asyncio
async def test_scm_sync_pull_failure_does_not_push(tmp_path) -> None:
    commands: list[tuple[str, ...]] = []

    async def fake_runner(args, cwd, timeout_s):
        commands.append(tuple(args))
        if tuple(args) == ("branch", "--show-current"):
            return GitCommandResult(returncode=0, stdout="feature/refactor\n", stderr="")
        if tuple(args) == ("pull", "origin", "main"):
            return GitCommandResult(returncode=1, stdout="", stderr="fatal: refusing to merge")
        raise AssertionError(f"unexpected git command: {args}")

    result = await scm_sync(
        {
            "action": "pull_then_push",
            "cwd": str(tmp_path),
            "remote": "origin",
            "pull_ref": "main",
        },
        runner=fake_runner,
    )

    assert result["ok"] is False
    assert result["failed_step"] == "pull"
    assert result["error"] == "fatal: refusing to merge"
    assert commands == [
        ("branch", "--show-current"),
        ("pull", "origin", "main"),
    ]


@pytest.mark.asyncio
async def test_scm_sync_reports_missing_working_directory(tmp_path) -> None:
    missing = tmp_path / "missing"

    result = await scm_sync({"action": "status", "cwd": str(missing)})

    assert result["ok"] is False
    assert result["failure_code"] == "path_not_found"
    assert str(missing) in result["error"]


@pytest.mark.asyncio
async def test_git_query_diff_builds_readonly_args(tmp_path) -> None:
    commands: list[tuple[str, ...]] = []

    async def fake_runner(args, cwd, timeout_s):
        commands.append(tuple(args))
        return GitCommandResult(returncode=0, stdout="diff --git a/x b/x\n+new\n", stderr="")

    result = await git_query(
        {"action": "diff", "cwd": str(tmp_path), "staged": True, "path": "src/x.py"},
        runner=fake_runner,
    )

    assert result["ok"] is True and result["tool"] == "git_query"
    assert commands == [("diff", "--no-color", "--staged", "--", "src/x.py")]
    assert "+new" in result["stdout"]


@pytest.mark.asyncio
async def test_git_query_log_parses_structured_entries(tmp_path) -> None:
    async def fake_runner(args, cwd, timeout_s):
        out = "abc123\x1fAlice\x1f2026-01-01\x1fInitial\ndef456\x1fBob\x1f2026-01-02\x1fFix bug"
        return GitCommandResult(returncode=0, stdout=out, stderr="")

    result = await git_query({"action": "log", "cwd": str(tmp_path), "max_count": 5}, runner=fake_runner)

    assert result["ok"] is True and result["entry_count"] == 2
    assert result["entries"][0] == {"hash": "abc123", "author": "Alice", "date": "2026-01-01", "subject": "Initial"}
    assert result["entries"][1]["subject"] == "Fix bug"


@pytest.mark.asyncio
async def test_git_query_branch_parses_current(tmp_path) -> None:
    async def fake_runner(args, cwd, timeout_s):
        return GitCommandResult(returncode=0, stdout="* main\n  feature/x\n  remotes/origin/main\n", stderr="")

    result = await git_query({"action": "branch", "cwd": str(tmp_path)}, runner=fake_runner)

    assert result["ok"] is True and result["current_branch"] == "main"
    assert "feature/x" in result["branches"] and "main" in result["branches"]


@pytest.mark.asyncio
async def test_git_query_unsupported_action(tmp_path) -> None:
    result = await git_query({"action": "commit", "cwd": str(tmp_path)})
    assert result["ok"] is False and result["failure_code"] == "unsupported_git_query"


@pytest.mark.asyncio
async def test_git_query_reports_failure(tmp_path) -> None:
    async def fake_runner(args, cwd, timeout_s):
        return GitCommandResult(returncode=128, stdout="", stderr="fatal: not a git repository")

    result = await git_query({"action": "status", "cwd": str(tmp_path)}, runner=fake_runner)

    assert result["ok"] is False and result["failed_step"] == "status"
    assert "not a git repository" in result["error"]


@pytest.mark.asyncio
async def test_git_write_commit_stages_then_commits(tmp_path) -> None:
    commands: list[tuple[str, ...]] = []

    async def fake_runner(args, cwd, timeout_s):
        commands.append(tuple(args))
        return GitCommandResult(returncode=0, stdout="ok", stderr="")

    result = await git_write({"action": "commit", "cwd": str(tmp_path), "message": "feat: x"}, runner=fake_runner)

    assert result["ok"] is True and result["tool"] == "git_write"
    assert commands == [("add", "-A"), ("commit", "-m", "feat: x")]


@pytest.mark.asyncio
async def test_git_write_commit_requires_message(tmp_path) -> None:
    result = await git_write({"action": "commit", "cwd": str(tmp_path)})
    assert result["ok"] is False and result["failure_code"] == "missing_message"


@pytest.mark.asyncio
async def test_git_write_branch_creates_and_switches(tmp_path) -> None:
    commands: list[tuple[str, ...]] = []

    async def fake_runner(args, cwd, timeout_s):
        commands.append(tuple(args))
        return GitCommandResult(returncode=0, stdout="", stderr="")

    result = await git_write({"action": "branch", "cwd": str(tmp_path), "name": "feature/x"}, runner=fake_runner)

    assert result["ok"] is True and commands == [("checkout", "-b", "feature/x")]


@pytest.mark.asyncio
async def test_git_write_checkout_switches(tmp_path) -> None:
    commands: list[tuple[str, ...]] = []

    async def fake_runner(args, cwd, timeout_s):
        commands.append(tuple(args))
        return GitCommandResult(returncode=0, stdout="", stderr="")

    result = await git_write({"action": "checkout", "cwd": str(tmp_path), "ref": "main"}, runner=fake_runner)

    assert result["ok"] is True and commands == [("checkout", "main")]


@pytest.mark.asyncio
async def test_git_write_unsupported_action(tmp_path) -> None:
    result = await git_write({"action": "rebase", "cwd": str(tmp_path)})
    assert result["ok"] is False and result["failure_code"] == "unsupported_git_write"
