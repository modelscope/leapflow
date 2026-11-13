# Copyright (c) Alibaba, Inc. and its affiliates.
"""Board server discovery, attribution and stop (``leap board stop|status``).

The launcher spawns the board server detached, so a server it fails to retire keeps
its port forever. Identity verification used to need ``psutil``; without it nothing
was ever retired and every new board revision took the next port (seven servers on
8765-8771 on the dev Mac). These tests pin both the fix and the cleanup command.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from leapflow.dashboard import instances, launcher
from leapflow.dashboard.instances import BoardServerProcess, StopResult


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(runtime_dir=tmp_path, dashboard_bind="127.0.0.1", dashboard_port=8765)


# ── argv identity ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "argv",
    [
        ["/opt/env/bin/python3.13", "-m", "leapflow", "board", "--serve", "--token", "t", "--port", "8765"],
        ["/usr/local/bin/leap", "board", "--serve"],
        ["/opt/env/bin/python3.13", "/opt/env/bin/leap", "board", "--serve"],
    ],
)
def test_board_server_argv_shapes_are_recognised(argv: list[str]) -> None:
    assert instances.is_board_server_argv(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["grep", "leapflow", "board", "--serve"],
        ["vim", "notes", "leapflow", "board", "--serve"],
        ["python", "-m", "leapflow", "board", "stop"],
        ["python", "-m", "leapflow", "--serve", "board"],
        ["python", "-m", "leapflow", "daemon", "serve", "--internal"],
        ["python", "-m", "leapcalib.cli", "serve"],
    ],
)
def test_lookalike_argv_is_not_a_board_server(argv: list[str]) -> None:
    assert not instances.is_board_server_argv(argv)


# ── Real process: retirement must not depend on psutil ──────────────────────


@pytest.fixture()
def fake_board_process() -> Iterator[subprocess.Popen[bytes]]:
    """A sleeping process whose argv has the exact board-server shape."""
    proc = subprocess.Popen(  # noqa: S603 - fixed argv
        [sys.executable, "-c", "import time; time.sleep(60)", "-m", "leapflow", "board", "--serve",
         "--port", "0", "--bind", "127.0.0.1"],
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not instances.pid_is_board_server(proc.pid):
        time.sleep(0.05)
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5.0)


def _running(proc: subprocess.Popen[bytes], timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        time.sleep(0.05)
    return proc.poll() is None


@pytest.fixture()
def no_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "psutil", None)  # import psutil -> ImportError


def test_a_stale_server_is_retired_without_psutil(
    tmp_path: Path, fake_board_process: subprocess.Popen[bytes], no_psutil: None,
) -> None:
    settings = _settings(tmp_path)
    pid = fake_board_process.pid
    launcher.write_state(settings, {"port": 1, "bind": "127.0.0.1", "token": "t", "pid": pid})

    launcher._retire_stale_server(settings)

    assert not _running(fake_board_process), "stale board server was abandoned"
    assert launcher.load_state(settings) is None


def test_discovery_lists_an_unrecorded_server_and_stop_scopes_it(
    tmp_path: Path, fake_board_process: subprocess.Popen[bytes], no_psutil: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    pid = fake_board_process.pid
    # The real listing runs, but only our fake is visible: ``everything=True`` below must
    # never reach the developer's own board servers (an unscoped first version did).
    real_listing = instances._list_processes
    monkeypatch.setattr(
        instances, "_list_processes", lambda: [item for item in real_listing() if item[0] == pid]
    )

    found = {server.pid: server for server in instances.discover_board_servers()}
    assert pid in found and found[pid].owner == ""  # not answering -> unattributable

    results, left = launcher.stop_servers(settings)
    assert pid not in {r.pid for r in results}, "an unattributed server must survive a scoped stop"
    assert pid in {s.pid for s in left}

    results, _ = launcher.stop_servers(settings, everything=True)
    assert StopResult(pid, 0, True, "stopped") in results
    assert not _running(fake_board_process)


# ── stop_servers selection ──────────────────────────────────────────────────


def test_stop_servers_stops_only_this_profiles_servers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    owner = launcher.server_owner(settings)
    launcher.write_state(settings, {"port": 8771, "bind": "127.0.0.1", "token": "t", "pid": 20})
    servers = [
        BoardServerProcess(10, "127.0.0.1", 8770, "a", owner=owner),  # ours, lost track of
        BoardServerProcess(20, "127.0.0.1", 8771, "b"),               # recorded, legacy build
        BoardServerProcess(30, "127.0.0.1", 8772, "c", owner="/other/runtime/dashboard.json"),
        BoardServerProcess(40, "127.0.0.1", 8773, "d"),               # unattributable
    ]
    monkeypatch.setattr(instances, "discover_board_servers", lambda: list(servers))
    stopped: list[int] = []

    def _stop(pid: int, *, port: int = 0) -> StopResult:
        stopped.append(pid)
        return StopResult(pid, port, True, "stopped")

    monkeypatch.setattr(instances, "stop_process", _stop)

    results, left = launcher.stop_servers(settings)

    assert stopped == [10, 20]
    assert [s.pid for s in left] == [30, 40]
    assert launcher.load_state(settings) is None, "the recorded server is gone, so is its state"

    stopped.clear()
    launcher.stop_servers(settings, everything=True)
    assert stopped == [10, 20, 30, 40]


def test_state_survives_when_the_recorded_server_could_not_be_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    launcher.write_state(settings, {"port": 8771, "bind": "127.0.0.1", "token": "t", "pid": 20})
    monkeypatch.setattr(
        instances, "discover_board_servers", lambda: [BoardServerProcess(20, "127.0.0.1", 8771, "t")]
    )
    monkeypatch.setattr(
        instances, "stop_process", lambda pid, *, port=0: StopResult(pid, port, False, "cannot signal")
    )

    results, _ = launcher.stop_servers(settings)

    assert results == [StopResult(20, 8771, False, "cannot signal")]
    assert launcher.load_state(settings) is not None


def test_a_shutting_down_server_keeps_its_successors_state(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    launcher.write_state(settings, {"port": 8766, "token": "new", "pid": 222})

    launcher.clear_state(settings, pid=111)  # the predecessor exiting
    assert launcher.load_state(settings)["pid"] == 222

    launcher.clear_state(settings, pid=222)
    assert launcher.load_state(settings) is None


# ── Server self-report and CLI ──────────────────────────────────────────────


async def test_server_info_reports_its_owner() -> None:
    from leapflow.dashboard.server import DashboardServer

    server = DashboardServer(client=SimpleNamespace(), token="t", owner="/p/runtime/dashboard.json")
    try:
        assert (await server._server_info())["owner"] == "/p/runtime/dashboard.json"
    finally:
        server._build_staleness.cancel_pending()


def test_board_stop_needs_no_web_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    from leapflow.cli.commands import dashboard as command

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(command, "load_config", lambda: SimpleNamespace())
    monkeypatch.setattr(command.launcher, "aiohttp_available", lambda: False)
    monkeypatch.setattr(
        command.launcher, "stop_servers",
        lambda settings, *, everything=False: calls.append({"everything": everything}) or ([], []),
    )

    assert command.cmd_dashboard(argparse.Namespace(template="stop", all=True, serve=False)) == 0
    assert calls == [{"everything": True}]
