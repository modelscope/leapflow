# Copyright (c) Alibaba, Inc. and its affiliates.
"""Local node stop: OS boundaries, profile isolation, and real CLI/process wiring."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from leapflow.cli.commands import hardware
from leapflow.hardware.lhp import node_lifecycle as lifecycle

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX node lifecycle")


@pytest.mark.parametrize("argv,expected", [
    (["/env/bin/python3.13", "/env/bin/leap", "hw", "serve", "--node-id", "edge"], True),
    (["python", "-u", "-m", "leapflow", "hw", "serve"], True),
    (["python", "-m", "leapflow.hardware.lhp", "--node-id=edge"], True),
    (["leap", "hw", "stop"], False),
    (["leap", "hw", "serve", "--help"], False),
    (["sh", "-c", "leap hw serve"], False),
    (["python", "fake.py", "leap", "hw", "serve"], False),
    (["python", "-c", "print('example')", "-m", "leapflow.hardware.lhp"], False),
    (["python", "-c", "from multiprocessing.spawn import spawn_main"], False),
    ([], False),
])
def test_exact_serve_command(argv, expected):
    assert lifecycle.is_serve_argv(argv) is expected


@pytest.fixture
def world(monkeypatch, tmp_path):
    """Fake only process filesystem, systemctl, ps, and signal delivery."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    state = SimpleNamespace(runtime=runtime, processes={}, signals=[], commands=[],
                            unit_active=True, unit_result="success", unit_pid=123,
                            accept_stop=True, terminate=True)
    read_text, read_bytes = Path.read_text, Path.read_bytes
    exists, stat, iterdir = Path.exists, Path.stat, Path.iterdir

    def process(path):
        parts = path.parts
        if len(parts) > 2 and parts[1] == "proc" and parts[2].isdigit():
            return state.processes.get(int(parts[2]))
        return None

    def text(path, *args, **kwargs):
        entry = process(path)
        if str(path).startswith("/proc/"):
            if entry is None or not entry["alive"]:
                raise FileNotFoundError(str(path))
            if path.name == "stat":
                fields = ["0"] * 20
                fields[0], fields[19] = "S", entry["start"]
                return "123 (python) " + " ".join(fields)
            if path.name == "cgroup":
                return "0::" + entry.get("cgroup", "/session.scope")
        return read_text(path, *args, **kwargs)

    def bytes_(path, *args, **kwargs):
        entry = process(path)
        if entry is not None and entry["alive"]:
            if path.name == "cmdline":
                return b"\0".join(os.fsencode(arg) for arg in entry["argv"]) + b"\0"
            if path.name == "environ":
                return b"\0".join(f"{key}={value}".encode() for key, value in entry["env"].items())
        return read_bytes(path, *args, **kwargs)

    def exists_(path):
        return True if str(path) in ("/proc/self/stat", "/proc/self/cgroup", "/proc/self") else exists(path)

    def stat_(path, *args, **kwargs):
        entry = process(path)
        return SimpleNamespace(st_uid=entry.get("uid", os.getuid())) if entry else stat(path, *args, **kwargs)

    def iter_(path):
        if str(path) == "/proc":
            return iter(Path(f"/proc/{pid}") for pid in state.processes)
        return iterdir(path)

    def deliver(pid, signum):
        state.signals.append((pid, signum))
        if state.terminate:
            affected = {pid}
            for child, entry in state.processes.items():
                if entry.get("parent", 0) in affected:
                    affected.add(child)
            for target in affected:
                if not state.processes[target].get("retain", False):
                    state.processes[target]["alive"] = False

    def run(argv, **kwargs):
        state.commands.append(argv)
        if argv[:2] == ["ps", "-axo"]:
            output = "\n".join(f"{pid} {entry.get('parent', 0)}" for pid, entry in state.processes.items() if entry["alive"])
            return subprocess.CompletedProcess(argv, 0, output, "")
        if argv[:2] == ["systemctl", "--user"]:
            if "stop" in argv:
                if not state.accept_stop:
                    return subprocess.CompletedProcess(argv, 1, "", "refused")
                state.unit_active = False
                deliver(123, signal.SIGTERM)
                return subprocess.CompletedProcess(argv, 0, "", "")
            output = f"MainPID={state.unit_pid if state.unit_active else 0}\nActiveState={'active' if state.unit_active else 'inactive'}\nResult={state.unit_result}"
            return subprocess.CompletedProcess(argv, 0, output, "")
        raise AssertionError(f"unexpected OS call: {argv}")

    monkeypatch.setattr(Path, "read_text", text)
    monkeypatch.setattr(Path, "read_bytes", bytes_)
    monkeypatch.setattr(Path, "exists", exists_)
    monkeypatch.setattr(Path, "stat", stat_)
    monkeypatch.setattr(Path, "iterdir", iter_)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(os, "kill", deliver)
    monkeypatch.setattr(os, "pidfd_open", lambda pid, flags: pid, raising=False)
    monkeypatch.setattr(signal, "pidfd_send_signal", deliver, raising=False)
    monkeypatch.setattr(os, "close", lambda descriptor: None)

    def add(pid=123, node_id="edge", **kwargs):
        state.processes[pid] = {
            "start": "7", "alive": True, "argv": ["python", "-m", "leapflow", "hw", "serve", "--node-id", node_id],
            "env": {"LEAPFLOW_RUNTIME_DIR": str(runtime)}, **kwargs,
        }
        return lifecycle.NodeProcess(pid=pid, node_id=node_id, start="7")

    state.add = add
    return state


def test_direct_stop_waits_for_workers(world):
    node = world.add()
    world.add(pid=124, parent=123)
    result = lifecycle.stop_node(node, timeout_s=0.1)
    assert result.ok and result.status == "stopped" and not result.survivors
    assert world.signals == [(123, signal.SIGTERM)]
    assert result.to_dict()["hardware_state_verified"] is False
    assert not result.cleanup_complete
    assert any("unconfirmed" in line for line in result.consequences)


@pytest.mark.parametrize("suffix", ["", "/workers"])
def test_systemd_stop_does_not_signal_child_directly(world, suffix):
    node = world.add(cgroup="/user.slice/user@501.service/app.slice/custom-node.service" + suffix)
    result = lifecycle.stop_node(node, timeout_s=0.1)
    assert result.ok and result.via == "systemd:custom-node.service"
    assert ["systemctl", "--user", "--no-block", "stop", "--", "custom-node.service"] in world.commands
    assert not world.unit_active


@pytest.mark.parametrize("cause", ["refused", "wrong_main_pid", "system_scope"])
def test_supervisor_refusal_never_falls_back_to_kill(world, cause):
    cgroup = "/system.slice/custom-node.service" if cause == "system_scope" else "/user.slice/user@501.service/app.slice/custom-node.service"
    node = world.add(cgroup=cgroup)
    if cause == "refused":
        world.accept_stop = False
    elif cause == "wrong_main_pid":
        world.unit_pid = 456
    result = lifecycle.stop_node(node, timeout_s=0.1)
    assert not result.ok and result.status == "refused"
    assert not world.signals


@pytest.mark.parametrize("unit_result", ["timeout", "signal", "core-dump", "exit-code"])
def test_systemd_abnormal_exit_is_not_reported_as_graceful(world, unit_result):
    node = world.add(cgroup="/user.slice/user@501.service/app.slice/custom-node.service")
    world.unit_result = unit_result
    result = lifecycle.stop_node(node, timeout_s=0.1)
    assert not result.ok and result.forced is (unit_result != "exit-code")
    assert result.status == "stopped_uncleanly"
    assert any("unconfirmed" in line for line in result.consequences)


def test_timeout_never_escalates_to_sigkill(world, monkeypatch, capsys):
    world.add()
    world.terminate = False
    monkeypatch.setattr(hardware, "load_config", lambda: SimpleNamespace(runtime_dir=world.runtime))
    args = argparse.Namespace(hw_action="stop", json=True, node_id="edge", timeout=0.01)
    assert hardware.cmd_hardware(args) == 1
    result = json.loads(capsys.readouterr().out)
    assert not result["ok"] and result["status"] == "timeout"
    assert world.signals == [(123, signal.SIGTERM)]
    assert result["failure"]["side_effect_state"] == "unknown"


def test_changed_identity_and_unrelated_commands_not_signalled(world):
    node = world.add(start="8")
    assert lifecycle.stop_node(node).status == "not_running"
    node = world.add(argv=["python", "other.py"])
    assert lifecycle.stop_node(node).status == "refused"
    assert not world.signals


def test_discovery_is_profile_and_user_scoped(world):
    world.add()
    world.add(pid=124, env={"LEAPFLOW_RUNTIME_DIR": str(world.runtime.parent / "other")})
    world.add(pid=125, uid=os.getuid() + 1)
    world.add(pid=126, node_id="different")
    assert [node.pid for node in lifecycle.discover_nodes(world.runtime, "edge")] == [123]


def test_corrupt_stale_and_unrelated_records(world):
    (world.runtime / "lhp-node-bad.json").write_text("[]")
    stale = world.runtime / "lhp-node-edge-999.json"
    stale.write_text(json.dumps({"version": 1, "pid": 999, "node_id": "edge", "start": "7"}))
    world.add(argv=["python", "other.py"])
    unrelated = world.runtime / "lhp-node-edge-123.json"
    unrelated.write_text(json.dumps({"version": 1, "pid": 123, "node_id": "edge", "start": "7"}))
    assert lifecycle.discover_nodes(world.runtime) == []
    assert not stale.exists() and unrelated.exists()


def test_cli_selection_and_json_without_initializing_hardware(world, monkeypatch, capsys):
    monkeypatch.setattr(hardware, "load_config", lambda: SimpleNamespace(runtime_dir=world.runtime))
    monkeypatch.delenv(lifecycle.NODE_ID_ENV, raising=False)
    world.add()
    world.add(pid=124, node_id="other")
    args = argparse.Namespace(hw_action="stop", json=True, node_id="", timeout=0.1)
    assert hardware.cmd_hardware(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "ambiguous_node"
    assert not world.signals
    args.node_id = "edge"
    assert hardware.cmd_hardware(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["node"]["node_id"] == "edge"
    assert payload["consequences"] and payload["failure"] is None
    assert world.processes[124]["alive"]


def test_human_output_displays_hardware_warnings_without_ansi(world, monkeypatch, capsys):
    world.add()
    monkeypatch.setattr(hardware, "load_config", lambda: SimpleNamespace(runtime_dir=world.runtime))
    args = argparse.Namespace(hw_action="stop", json=False, node_id="edge", timeout=0.1)
    assert hardware.cmd_hardware(args) == 0
    output = capsys.readouterr().out
    assert "NOT power-off" in output and "Piper:" in output
    assert "NOT verified" in output and "monitoring" in output
    assert "\033[" not in output


def test_parent_exit_with_worker_survivor_is_not_success(world):
    node = world.add()
    world.add(pid=124, parent=123, retain=True)
    result = lifecycle.stop_node(node, timeout_s=0.01)
    assert not result.ok and result.status == "timeout" and result.survivors == (124,)


@pytest.mark.parametrize("complete", [True, False])
def test_cleanup_receipt_is_not_physical_verification(world, complete):
    node = world.add()
    path = world.runtime / "lhp-node-edge-123.json"
    path.write_text(json.dumps({"version": 1, "pid": 123, "start": "7", "node_id": "edge", "cleanup_complete": complete}))
    node = lifecycle.NodeProcess(pid=node.pid, node_id=node.node_id, start=node.start, record_path=path)
    result = lifecycle.stop_node(node, timeout_s=0.1)
    assert result.ok is complete and result.cleanup_complete is complete
    assert result.status == ("stopped" if complete else "stopped_uncleanly")
    assert not result.to_dict()["hardware_state_verified"] and not path.exists()


def test_parent_exit_during_signal_delivery_still_checks_workers(world, monkeypatch):
    node = world.add()
    world.add(pid=124, parent=123, retain=True)

    def vanished(pid, signum):
        world.processes[pid]["alive"] = False
        raise ProcessLookupError(pid)

    monkeypatch.setattr(signal, "pidfd_send_signal", vanished)
    result = lifecycle.stop_node(node, timeout_s=0.01)
    assert not result.ok and result.survivors == (124,)
    assert result.status == "timeout"


def test_unreadable_supervisor_fails_closed(world, monkeypatch):
    node = world.add()
    read = Path.read_text

    def unreadable(path, *args, **kwargs):
        if str(path) == "/proc/123/cgroup":
            raise PermissionError("supervisor unavailable")
        return read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    result = lifecycle.stop_node(node, timeout_s=0.01)
    assert not result.ok and result.status == "refused"
    assert not world.signals


def test_worker_pid_reuse_is_not_counted_as_a_survivor(world, monkeypatch):
    node = world.add()
    world.add(pid=124, parent=123)

    def exit_and_recycle(pid, signum):
        world.signals.append((pid, signum))
        world.processes[pid]["alive"] = False
        world.processes[124]["start"] = "8"

    monkeypatch.setattr(signal, "pidfd_send_signal", exit_and_recycle)
    result = lifecycle.stop_node(node, timeout_s=0.01)
    assert result.ok and not result.survivors
    assert world.signals == [(123, signal.SIGTERM)]


@pytest.mark.component
def test_records_are_private_atomic_and_ignore_predictable_temp_symlinks(tmp_path):
    victim = tmp_path / "unrelated"
    victim.write_text("preserve")
    legacy_temp = tmp_path / f"lhp-node-edge-{os.getpid()}.tmp"
    legacy_temp.symlink_to(victim)
    path = lifecycle.write_record(tmp_path, node_id="edge", endpoint="tcp://127.0.0.1:1234")
    assert path is not None
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())["cleanup_complete"] is False
    lifecycle.finish_record(path, complete=True)
    assert json.loads(path.read_text())["cleanup_complete"] is True
    assert path.stat().st_mode & 0o777 == 0o600
    assert victim.read_text() == "preserve"


@pytest.mark.parametrize("timeout", [-1, 0, float("nan"), float("inf")])
def test_invalid_timeout_has_no_side_effects(world, capsys, timeout):
    args = argparse.Namespace(hw_action="stop", json=True, node_id="", timeout=timeout)
    assert hardware.cmd_hardware(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "invalid_args"
    assert not world.commands and not world.signals


def test_cli_not_running_is_idempotent(world, monkeypatch, capsys):
    monkeypatch.setattr(hardware, "load_config", lambda: SimpleNamespace(runtime_dir=world.runtime))
    assert hardware.cmd_hardware(argparse.Namespace(hw_action="stop", json=True, node_id="", timeout=1)) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "not_running"
    assert not world.signals


@pytest.mark.component
@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["server", "skills"])
async def test_serve_shutdown_failure_still_closes_remaining_owners(tmp_path, monkeypatch, failure):
    from leapflow.hardware.lhp.runtime import serve_node
    from leapflow.hardware.lhp.server import LhpNodeServer
    from leapflow.hardware.registry import HardwareRegistry
    from leapflow.hardware.skills.runtime import SkillRuntime

    for key, value in {
        "LEAPFLOW_DATA_DIR": str(tmp_path), "LEAPFLOW_PROFILE": "cleanup_test",
        "LEAPFLOW_RUNTIME_DIR": str(tmp_path / "runtime"),
        "LEAPFLOW_HARDWARE_PROVIDERS": "yaml", "LEAPFLOW_HARDWARE_DEVICES_DIR": str(tmp_path / "devices"),
        "LEAPFLOW_WORKSPACE_ROOT": str(tmp_path),
    }.items():
        monkeypatch.setenv(key, value)
    callbacks, closed = [], []
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler",
                        lambda sig, callback: callbacks.append(callback))
    start = LhpNodeServer.start

    async def start_then_stop(self, **kwargs):
        result = await start(self, **kwargs)
        callbacks[0]()
        return result

    monkeypatch.setattr(LhpNodeServer, "start", start_then_stop)

    def track_close(name, closer):
        async def close(self):
            await closer(self)
            closed.append(name)
            if name == failure:
                raise OSError("injected cleanup failure")
        return close

    monkeypatch.setattr(LhpNodeServer, "stop", track_close("server", LhpNodeServer.stop))
    monkeypatch.setattr(SkillRuntime, "close", track_close("skills", SkillRuntime.close))
    monkeypatch.setattr(HardwareRegistry, "close_all", track_close("registry", HardwareRegistry.close_all))
    with pytest.raises(OSError, match="injected cleanup failure"):
        await serve_node(node_id="cleanup_test", port=0)
    assert closed == ["server", "skills", "registry"]
    records = list((tmp_path / "runtime").glob("lhp-node-*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["cleanup_complete"] is False


@pytest.mark.parametrize("entry", [("leapflow", "hw", "serve"), ("leapflow.hardware.lhp",)])
@pytest.mark.component
@pytest.mark.asyncio
async def test_real_cli_serve_then_stop_and_repeat(tmp_path, entry):
    """No mocked wiring: actual empty node, profile records, CLI parser and SIGTERM."""
    env = {**os.environ, "LEAPFLOW_DATA_DIR": str(tmp_path), "LEAPFLOW_PROFILE": "stop_test",
           "LEAPFLOW_RUNTIME_DIR": str(tmp_path / "runtime"), "LEAPFLOW_FLEET_NODES": "[]",
           "LEAPFLOW_HARDWARE_PROVIDERS": "yaml", "LEAPFLOW_HARDWARE_DEVICES_DIR": str(tmp_path / "devices"),
           "LEAPFLOW_WORKSPACE_ROOT": str(tmp_path)}
    for key in ("LEAPFLOW_LHP_NODE_ID", "LEAPFLOW_LHP_TOKEN", "LEAPFLOW_LHP_PERMIT_ISSUER_KEYS"):
        env.pop(key, None)
    server = await asyncio.create_subprocess_exec(
        sys.executable, "-m", *entry, "--node-id", "stop_test", "--port", "0",
        env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        line = await asyncio.wait_for(server.stdout.readline(), 30)
        assert line, (await server.stderr.read()).decode()
        ready = json.loads(line)
        assert ready["event"] == "lhp_node_ready" and ready["device_count"] == 0
        for expected in ("stopped", "not_running"):
            command = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "leapflow", "hw", "stop", "--node-id", "stop_test", "--json",
                env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(command.communicate(), 30)
            assert command.returncode == 0, stderr.decode()
            result = json.loads(stdout)
            assert result["status"] == expected
            if expected == "stopped":
                assert result["cleanup_complete"] and not result["hardware_state_verified"]
        assert await asyncio.wait_for(server.wait(), 5) == 0
        assert not list((tmp_path / "runtime").glob("lhp-node-*.json"))
    finally:
        if server.returncode is None:
            server.terminate()
            await asyncio.wait_for(server.wait(), 10)
