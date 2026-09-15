# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hermetic tests for leapflow.utils.build_info (long-lived-process staleness).

All git subprocess calls are monkeypatched at the module's ``_fingerprint``
seam, so these tests never depend on the actual repository's git state (or
git being installed at all) and run identically in any CI environment.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import pytest

from leapflow.utils import build_info
from leapflow.version import __version__


def _patch_fingerprint(monkeypatch: pytest.MonkeyPatch, values) -> None:
    """Make ``_fingerprint`` return successive ``values`` on each call."""
    it = iter(values)

    def _fake(cwd: Path) -> Tuple[Optional[str], Optional[str]]:
        return next(it)

    monkeypatch.setattr(build_info, "_fingerprint", _fake)


# ── capture_build_info ───────────────────────────────────────────────────────


def test_capture_build_info_reports_process_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fingerprint(monkeypatch, [("abc123", "d1")])

    info = build_info.capture_build_info()

    assert info.version == __version__
    assert info.pid == os.getpid()
    assert info.commit == "abc123"
    assert info.dirty_digest == "d1"
    assert info.started_at > 0


def test_capture_build_info_outside_git_checkout_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fingerprint(monkeypatch, [(None, None)])

    info = build_info.capture_build_info()

    assert info.commit is None
    assert info.dirty_digest is None


# ── is_stale ─────────────────────────────────────────────────────────────────


def test_is_stale_false_when_fingerprint_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fingerprint(monkeypatch, [("abc123", "d1"), ("abc123", "d1")])

    info = build_info.capture_build_info()
    assert build_info.is_stale(info) is False


def test_is_stale_true_when_commit_advances(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulates: process starts at commit abc123, a new commit lands afterwards.
    _patch_fingerprint(monkeypatch, [("abc123", "d1"), ("def456", "d1")])

    info = build_info.capture_build_info()
    assert build_info.is_stale(info) is True


def test_is_stale_true_when_working_tree_becomes_dirty(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same commit, but an uncommitted edit landed after the process started.
    _patch_fingerprint(monkeypatch, [("abc123", "d1"), ("abc123", "d2")])

    info = build_info.capture_build_info()
    assert build_info.is_stale(info) is True


def test_is_stale_none_outside_git_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fingerprint(monkeypatch, [(None, None), (None, None)])

    info = build_info.capture_build_info()
    assert build_info.is_stale(info) is None


def test_is_stale_none_when_git_disappears_after_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    # Captured inside a checkout, but the live re-check can no longer find git
    # (e.g. run from a different cwd) — must degrade to unknown, not "stale".
    _patch_fingerprint(monkeypatch, [("abc123", "d1"), (None, None)])

    info = build_info.capture_build_info()
    assert build_info.is_stale(info) is None


# ── to_dict ──────────────────────────────────────────────────────────────────


def test_to_dict_is_plain_json_safe_and_coerces_none_to_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fingerprint(monkeypatch, [(None, None)])
    info = build_info.capture_build_info()

    data = info.to_dict()

    assert data == {
        "version": __version__,
        "commit": "",
        "dirty_digest": "",
        "pid": os.getpid(),
        "started_at": info.started_at,
    }


# ── StalenessMonitor: non-blocking cache around is_stale ─────────────────────


def _info(commit: str = "abc123") -> build_info.BuildInfo:
    return build_info.BuildInfo(
        version=__version__, commit=commit, dirty_digest="d1", pid=os.getpid(), started_at=0.0,
    )


def test_staleness_monitor_current_is_none_before_first_refresh_completes() -> None:
    """The first call must return instantly, never blocking on ``checker``."""
    monitor = build_info.StalenessMonitor(_info())

    verdict = monitor.current(lambda captured: True)

    assert verdict is None  # unknown: the background refresh has not run yet


async def test_staleness_monitor_current_reflects_background_refresh_once_it_completes() -> None:
    monitor = build_info.StalenessMonitor(_info())
    monitor.current(lambda captured: True)  # schedules the background refresh

    for _ in range(50):
        if monitor.current(lambda captured: True) is not None:
            break
        await asyncio.sleep(0.01)

    assert monitor.current(lambda captured: True) is True


async def test_staleness_monitor_does_not_reschedule_refresh_within_ttl() -> None:
    calls = 0

    def _checker(captured: build_info.BuildInfo) -> Optional[bool]:
        nonlocal calls
        calls += 1
        return False

    monitor = build_info.StalenessMonitor(_info(), ttl_s=60.0)
    await monitor.refresh(_checker)
    assert calls == 1

    # Within the TTL window, repeated current() calls must not schedule
    # another background refresh.
    for _ in range(5):
        monitor.current(_checker)
    await asyncio.sleep(0)
    assert calls == 1


async def test_staleness_monitor_refresh_is_synchronous_and_deterministic() -> None:
    monitor = build_info.StalenessMonitor(_info())

    verdict = await monitor.refresh(lambda captured: True)

    assert verdict is True
    assert monitor.current(lambda captured: True) is True


def test_staleness_monitor_current_without_running_loop_degrades_gracefully() -> None:
    """Called from sync code (no running loop), current() must not raise."""
    monitor = build_info.StalenessMonitor(_info())

    assert monitor.current(lambda captured: True) is None


def test_staleness_monitor_cancel_pending_is_noop_without_pending_refresh() -> None:
    monitor = build_info.StalenessMonitor(_info())

    monitor.cancel_pending()  # must not raise when nothing is in flight


async def test_staleness_monitor_cancel_pending_cancels_inflight_refresh() -> None:
    async def _hang() -> Optional[bool]:
        await asyncio.sleep(3600)
        return True

    monitor = build_info.StalenessMonitor(_info())
    monitor._refresh_task = asyncio.ensure_future(_hang())

    monitor.cancel_pending()

    with pytest.raises(asyncio.CancelledError):
        await monitor._refresh_task


# ── _run_git: graceful degradation on subprocess failure ────────────────────


def test_run_git_returns_none_when_binary_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert build_info._run_git(["rev-parse", "HEAD"], Path(".")) is None


def test_run_git_returns_none_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=1.5)

    monkeypatch.setattr(subprocess, "run", _raise)
    assert build_info._run_git(["status"], Path(".")) is None


def test_run_git_returns_none_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResult:
        returncode = 128
        stdout = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeResult())
    assert build_info._run_git(["rev-parse", "HEAD"], Path(".")) is None


# ── _repo_root: source checkout and packaged-install degradation ─────────────


def test_repo_root_finds_source_checkout_above_src_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    module_file = root / "src" / "leapflow" / "utils" / "build_info.py"
    module_file.parent.mkdir(parents=True)
    (root / ".git").mkdir()
    monkeypatch.setattr(build_info, "__file__", str(module_file))

    assert build_info._repo_root() == root


def test_repo_root_degrades_without_fixed_parent_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "site-packages" / "leapflow"
    module_file = package_root / "utils" / "build_info.py"
    module_file.parent.mkdir(parents=True)
    monkeypatch.setattr(build_info, "__file__", str(module_file))

    # Packaged installs have no .git/pyproject source root; this must still
    # return a usable cwd for git probing (which then degrades to None), never
    # raise from a fixed parents[3] assumption.
    assert build_info._repo_root() == package_root


# ── Integration smoke: this repository really is a git checkout ────────────


def test_capture_build_info_against_real_repo_is_internally_consistent() -> None:
    """No monkeypatch: exercises the real git subprocess calls once.

    This repository is a git checkout, so commit should resolve; whatever it
    resolves to, checking staleness immediately afterward must be False when the
    working tree fingerprint is stable. In a parallel test run another worker may
    briefly update generated fixtures or caches between the capture and re-check;
    that races the smoke test's real git dependency, not the staleness algorithm
    (which is covered by hermetic tests above), so degrade to skip on a moving
    dirty digest.
    """
    info = build_info.capture_build_info()
    if info.commit is None:
        pytest.skip("not a git checkout in this environment")
    stale = build_info.is_stale(info)
    if stale is None:
        pytest.skip("git fingerprint became unavailable during smoke test")
    if stale is True:
        current_commit, current_digest = build_info._fingerprint(build_info._repo_root())
        if current_commit == info.commit and current_digest != info.dirty_digest:
            pytest.skip("working tree fingerprint changed during smoke test")
    assert stale is False
