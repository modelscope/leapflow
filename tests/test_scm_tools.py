from __future__ import annotations

import pytest

from leapflow.tools.scm_tools import GitCommandResult, scm_sync


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
