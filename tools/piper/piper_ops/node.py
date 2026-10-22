"""Exact-process local node lifecycle; never launches a second CAN owner."""
from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import stat
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from leapflow.engine.recovery.failure_envelope import SideEffectState
from leapflow.hardware.lhp.client import SocketLhpClient

from .config import Settings
from .evidence import Evidence, Refusal
from .motion import digest_plan


def argv(settings: Settings) -> list[str]:
    return [str(settings.node_python), str(settings.node_leap), "hw", "serve", "--node-id",
            settings.node_id, "--host", settings.node_host, "--port", str(settings.node_port)]


def processes(settings: Settings) -> list[int]:
    signature = "^" + r"\s+".join(re.escape(part) for part in argv(settings)) + "$"
    result = subprocess.run(["pgrep", "-f", signature], capture_output=True, text=True,
                            timeout=settings.request_timeout_s, check=False)
    if result.returncode not in (0, 1):
        raise Refusal("exact node process inspection failed")
    return [int(value) for value in result.stdout.split()]


def process_start(pid: int) -> str:
    """Linux process creation identity; a PID alone is not a safe stop target."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError) as exc:
        raise Refusal("node process creation identity unavailable; no service action") from exc


def preview(settings: Settings, operation: str) -> dict[str, Any]:
    if operation not in ("start", "restart", "status"):
        raise Refusal("unknown node operation")
    found = processes(settings)
    if len(found) > 1:
        raise Refusal("multiple exact node owners found; inspect without starting another")
    return {"version": 1, "operation": operation, "settings": asdict(settings),
            "argv": argv(settings), "pids": found,
            "process_starts": {str(pid): process_start(pid) for pid in found},
            "log": str(settings.root / "data/logs/leap-serve.log"),
            "impact": "Node restart disconnects peripherals and may halt active hardware"}


def _environment(settings: Settings) -> dict[str, str]:
    public = settings.issuer_key.with_suffix(".pub")
    if not settings.node_leap.is_file() or not settings.node_python.is_file() or not public.is_file():
        raise Refusal("node executable or existing trusted public issuer configuration is unavailable")
    token = settings.token_file.read_text().strip()
    if not token:
        raise Refusal("node token is empty")
    environment = dict(os.environ)
    environment["LEAPFLOW_LHP_TOKEN"] = token
    environment["LEAPFLOW_LHP_PERMIT_ISSUER_KEYS"] = str(public)
    return environment


@contextmanager
def lifecycle_lock(settings: Settings) -> Iterator[int]:
    """Serialize cooperating service operations; never remove a lock inode."""
    import fcntl

    settings.root.mkdir(parents=True, exist_ok=True)
    with (settings.root / ".piper-node.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Refusal("another node lifecycle operation is active; inspect before continuing") from exc
        yield lock.fileno()


_PIDFD_TERMINATE = """
import json, os, signal, sys
from pathlib import Path
fd, attempted, ok = None, False, False
try:
    pid, expected_start, expected_argv = int(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
    fd = os.pidfd_open(pid, 0)
    actual_start = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    actual_argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\\0')[:-1]
    if actual_start != expected_start or actual_argv != [os.fsencode(a) for a in expected_argv]:
        raise ValueError('process identity changed')
    attempted = True
    signal.pidfd_send_signal(fd, signal.SIGTERM)
    ok = True
except (OSError, ValueError, AttributeError, IndexError, TypeError):
    pass
finally:
    if fd is not None:
        os.close(fd)
print(json.dumps({'ok': ok, 'signal_attempted': attempted}))
sys.exit(0 if ok else 1)
"""


def terminate_with_interpreter(settings: Settings, pid: int, start: str, evidence: Evidence) -> None:
    """Opt-in platform bridge; the helper binds identity before sending via pidfd."""
    interpreter = settings.node_control_python
    if interpreter is None:
        raise Refusal("safe exact-process termination requires Linux pidfd support; "
                      "optionally configure a trusted node_control_python")
    interpreter = interpreter.resolve(strict=True)
    info = interpreter.stat()
    if info.st_uid != 0 or info.st_mode & 0o022 or not stat.S_ISREG(info.st_mode):
        raise Refusal("node control interpreter must be a root-owned, non-writable regular executable")
    if pid not in processes(settings) or process_start(pid) != start:
        raise Refusal("node process creation identity changed; no termination")
    evidence.record("node_term_attempt", pid=pid, process_start=start, interpreter=str(interpreter))
    try:
        result = subprocess.run([str(interpreter), "-I", "-c", _PIDFD_TERMINATE,
                                 str(pid), start, json.dumps(argv(settings))],
                                capture_output=True, text=True, timeout=settings.request_timeout_s,
                                check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Refusal("node termination delivery unconfirmed; inspect before continuing",
                      state=SideEffectState.UNKNOWN) from exc
    try:
        report = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise Refusal("node termination result unconfirmed; inspect before continuing",
                      state=SideEffectState.UNKNOWN) from exc
    if (result.returncode != 0 or not isinstance(report, dict) or report.get("ok") is not True
            or report.get("signal_attempted") is not True):
        unsent = isinstance(report, dict) and report.get("signal_attempted") is False
        raise Refusal("node termination refused or unconfirmed; inspect before continuing",
                      state=SideEffectState.NONE if unsent else SideEffectState.UNKNOWN)


def terminate_exact(settings: Settings, pid: int, start: str, evidence: Evidence) -> None:
    """Signal the approved process through a stable kernel handle, never a reused PID."""
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        return terminate_with_interpreter(settings, pid, start, evidence)
    try:
        descriptor = os.pidfd_open(pid, 0)
    except ProcessLookupError as exc:
        raise Refusal("approved node process exited before termination") from exc
    try:
        if pid not in processes(settings) or process_start(pid) != start:
            raise Refusal("node process creation identity changed; no termination")
        evidence.record("node_term_attempt", pid=pid, process_start=start)
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
    finally:
        os.close(descriptor)


async def _ready(settings: Settings) -> bool:
    client = SocketLhpClient(settings.endpoint, token=settings.token_file.read_text().strip(),
                             timeout_s=settings.request_timeout_s,
                             connect_timeout_s=settings.request_timeout_s)
    try:
        hello = await client.connect()
        listing = await client.request("list")
        if hello.get("node_id") != settings.node_id or hello.get("write_mode") != "permit_required":
            raise Refusal("node readiness identity or authority mismatch")
        devices = listing.get("devices", [])
        if listing.get("node_id") != settings.node_id or sum(
            isinstance(device, dict) and device.get("device_id") == settings.device_id for device in devices) != 1:
            raise Refusal("node readiness declaration missing or duplicated")
        return True
    finally:
        await asyncio.wait_for(client.close(), settings.request_timeout_s)


def ready(settings: Settings) -> bool:
    """Hello/list readiness only; does not certify peripheral health or send a write."""
    try:
        return asyncio.run(asyncio.wait_for(_ready(settings), settings.request_timeout_s))
    except (OSError, RuntimeError, TimeoutError):
        return False


def execute(settings: Settings, plan: dict[str, Any], evidence: Evidence) -> dict[str, Any]:
    with lifecycle_lock(settings):
        return _execute_locked(settings, plan, evidence)


def _execute_locked(settings: Settings, plan: dict[str, Any], evidence: Evidence) -> dict[str, Any]:
    operation = plan["operation"]
    if operation not in ("start", "restart"):
        raise Refusal("only start/restart may execute a node plan")
    if digest_plan(preview(settings, operation)) != digest_plan(plan):
        raise Refusal("node process/settings changed since approval; no service action")
    environment = _environment(settings)
    found = processes(settings)
    if operation == "start" and found:
        if not ready(settings):
            raise Refusal("existing node is not LHP-ready; no service action")
        return {"ok": True, "status": "already_running", "pids": found, "lhp_ready": True}
    for pid in found:
        terminate_exact(settings, pid, plan["process_starts"][str(pid)], evidence)
    deadline = time.monotonic() + settings.move_timeout_s
    while processes(settings) and time.monotonic() < deadline:
        time.sleep(settings.poll_s)
    if processes(settings):
        raise Refusal("existing node did not stop; refusing a second CAN owner")
    log = settings.root / "data/logs/leap-serve.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    evidence.record("node_start_attempt", argv=argv(settings), log=str(log))
    with log.open("ab") as output:
        child = subprocess.Popen(argv(settings), cwd=settings.node_workspace, env=environment,
                                 stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                 start_new_session=True)
    deadline = time.monotonic() + settings.request_timeout_s
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise Refusal("node exited during start; inspect the local service log")
        running = processes(settings)
        if running == [child.pid] and ready(settings):
            evidence.record("node_ready", pid=child.pid, process_start=process_start(child.pid))
            return {"ok": True, "status": "running", "pids": running, "log": str(log), "lhp_ready": True}
        if len(running) > 1:
            raise Refusal("multiple owners detected after start; inspect immediately")
        time.sleep(settings.poll_s)
    raise Refusal("node start could not be verified; do not blindly repeat")
