# Copyright (c) Alibaba, Inc. and its affiliates.
"""Find and stop the LHP node served on this machine -- ``leap hw stop``.

``leap hw serve`` is a foreground process that shuts down gracefully on
SIGINT/SIGTERM (``runtime.py``). This module is its counterpart: it locates the
serving process, asks whoever owns it to stop it, and waits for the exit, so
the caller learns whether the graceful shutdown actually ran.

Discovery has two sources:

- **Node records.** ``serve_node`` writes ``<runtime_dir>/lhp-node-<id>-<pid>.json``
  (pid, process start identity, endpoint) once it is listening, and records
  cleanup completion on exit. Stale records are deleted during discovery.
- **Process scan** (Linux ``/proc`` only). Legacy nodes without records are
  matched by executable, command line, user, profile, and runtime directory.

Ownership decides the stop path. A node running inside a systemd ``.service``
unit is stopped by systemd: signalling the child directly would leave the
supervisor to judge the exit, and a unit with ``Restart=always`` would start a
rival owner of the CAN bus and cameras. An unsupervised node receives SIGTERM,
after revalidating its command line and creation identity, using a pidfd
where available. No broad process-name kill or automatic SIGKILL is used.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shlex
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from leapflow.hardware.lhp.protocol import LHP_DEFAULT_PORT, is_valid_device_id

logger = logging.getLogger(__name__)

NODE_ID_ENV = "LEAPFLOW_LHP_NODE_ID"
RECORD_PREFIX = "lhp-node-"
RECORD_VERSION = 1

SOURCE_RECORD = "record"
SOURCE_SCAN = "process_scan"

STATUS_STOPPED = "stopped"
STATUS_NOT_RUNNING = "not_running"
STATUS_TIMEOUT = "timeout"
STATUS_REFUSED = "refused"

SCOPE_USER = "user"
SCOPE_SYSTEM = "system"

# Above the serve shutdown budget (per-actuator halt 2 s, skill unwind grace,
# planning-worker join) and the shipped unit's TimeoutStopSec=20.
DEFAULT_STOP_TIMEOUT_S = 30.0
_SUBPROCESS_TIMEOUT_S = 10.0
_POLL_S = 0.2

_UNIT_RE = re.compile(r"[^/]+\.service")


@dataclass(frozen=True)
class NodeProcess:
    """One serving node process, identified by pid *and* creation identity."""

    pid: int
    node_id: str
    start: str
    endpoint: str = ""
    argv: tuple[str, ...] = ()
    source: str = SOURCE_RECORD
    record_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "node_id": self.node_id,
            "endpoint": self.endpoint,
            "source": self.source,
            "record_path": str(self.record_path) if self.record_path else "",
        }


@dataclass(frozen=True)
class Supervisor:
    """The systemd unit that owns a node process."""

    scope: str
    unit: str


@dataclass(frozen=True)
class StopResult:
    """Outcome of one stop attempt; ``consequences`` describe the hardware after it."""

    ok: bool
    status: str
    node: NodeProcess | None = None
    via: str = ""
    forced: bool = False
    detail: str = ""
    survivors: tuple[int, ...] = ()
    consequences: tuple[str, ...] = field(default_factory=tuple)
    cleanup_complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "node": self.node.to_dict() if self.node is not None else None,
            "via": self.via,
            "forced": self.forced,
            "detail": self.detail,
            "survivors": list(self.survivors),
            "consequences": list(self.consequences),
            "cleanup_complete": self.cleanup_complete,
            "hardware_state_verified": False,
        }


# ── Process identity ─────────────────────────────────────────────────────


def process_start(pid: int) -> str | None:
    """Creation identity of a live process, or None when it is gone (or a zombie).

    A pid alone is not a safe target: it is recycled. Linux reads the start
    tick from ``/proc``; elsewhere ``ps`` reports the start time.
    """
    if pid <= 1:
        return None
    stat_path = Path(f"/proc/{pid}/stat")
    if Path("/proc/self/stat").exists():
        try:
            fields = stat_path.read_text().rsplit(")", 1)[1].split()
            return None if fields[0] == "Z" else fields[19]
        except FileNotFoundError:
            return None
        except (OSError, IndexError) as exc:
            raise OSError(f"Cannot inspect process {pid}; no stop confirmation") from exc
    try:
        completed = subprocess.run(
            ["ps", "-o", "stat=,lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OSError("Cannot inspect process identity with ps") from exc
    values = completed.stdout.strip().split(maxsplit=1)
    if not values or values[0].startswith("Z"):
        return None
    if completed.returncode != 0 or len(values) != 2:
        raise OSError("Process identity query failed")
    return values[1]


def is_alive(node: NodeProcess) -> bool:
    return process_start(node.pid) == node.start


def child_processes(pid: int) -> tuple[tuple[int, str], ...]:
    """Snapshot descendants, including workers spawned by non-main threads."""
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid="], capture_output=True, text=True,
        timeout=_SUBPROCESS_TIMEOUT_S, check=False,
    )
    if result.returncode != 0:
        raise OSError("Cannot inspect the node's worker processes")
    parents: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        values = line.split()
        if len(values) == 2 and all(value.isdigit() for value in values):
            child, parent = map(int, values)
            parents.setdefault(parent, []).append(child)
    children: dict[int, str] = {}
    pending, seen = [pid], {pid}
    while pending:
        for child in parents.get(pending.pop(), ()):
            if child in seen:
                continue
            seen.add(child)
            start = process_start(child)
            if start is not None:
                children[child] = start
                pending.append(child)
    return tuple(children.items())


def supervisor_of(pid: int) -> Supervisor | None:
    """The systemd ``.service`` unit whose cgroup holds *pid*, if any.

    A terminal-started node lives in a session ``.scope`` and has no
    supervisor; ``user@<uid>.service`` is the user manager itself, not a unit
    that owns the node.
    """
    try:
        lines = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
    except FileNotFoundError:
        if Path("/proc/self/cgroup").exists() and process_start(pid) is not None:
            raise OSError("Cannot determine the node's supervisor; no signal sent")
        return None
    for line in lines:
        path = line.split(":", 2)[-1]
        for component in reversed(path.split("/")):
            if _UNIT_RE.fullmatch(component) and not component.startswith("user@"):
                scope = SCOPE_USER if "/user@" in path else SCOPE_SYSTEM
                return Supervisor(scope=scope, unit=component)
    return None


# ── Node records (written by serve_node) ─────────────────────────────────


def _replace_record(path: Path, payload: dict[str, Any]) -> None:
    """Publish a private receipt atomically, without following a preexisting temp symlink."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=path.stem + "-",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_record(runtime_dir: Path, *, node_id: str, endpoint: str) -> Path | None:
    """Record this process as the server of *node_id* without storing credentials."""
    if not is_valid_device_id(node_id):
        raise ValueError("Invalid node id for lifecycle record")
    pid = os.getpid()
    start = process_start(pid)
    if start is None:
        logger.warning("Node record skipped: process start identity unavailable")
        return None
    path = Path(runtime_dir) / f"{RECORD_PREFIX}{node_id}-{pid}.json"
    payload = {
        "version": RECORD_VERSION,
        "node_id": node_id,
        "pid": pid,
        "start": start,
        "endpoint": endpoint,
        "started_at": time.time(),
        "cleanup_complete": False,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _replace_record(path, payload)
    except OSError:
        logger.warning("Node record %s could not be written; `leap hw stop` falls back to a process scan",
                       path, exc_info=True)
        return None
    return path


def finish_record(path: Path | None, *, complete: bool) -> None:
    """Leave a cleanup receipt; it does not certify any hardware's physical state."""
    if path is None:
        return
    try:
        data = json.loads(path.read_text())
        data["cleanup_complete"] = complete
        _replace_record(path, data)
    except (OSError, ValueError, TypeError):
        logger.warning("Cannot record node cleanup outcome", exc_info=True)


def remove_record(path: Path | None) -> None:
    """Remove a stale node record, leaving live records untouched by discovery."""
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Node record %s could not be removed", path, exc_info=True)


def _recorded(runtime_dir: Path) -> Iterator[NodeProcess]:
    for path in sorted(Path(runtime_dir).glob(f"{RECORD_PREFIX}*.json")):
        try:
            if path.is_symlink() or path.stat().st_uid != os.getuid():
                continue
            data = json.loads(path.read_text())
            if not isinstance(data, dict) or data.get("version") != RECORD_VERSION:
                continue
            if type(data.get("pid")) is not int or data["pid"] <= 1:
                continue
            if not is_valid_device_id(str(data.get("node_id", ""))):
                continue
            node = NodeProcess(
                pid=int(data["pid"]), node_id=str(data["node_id"]), start=str(data["start"]),
                endpoint=str(data.get("endpoint") or ""), source=SOURCE_RECORD, record_path=path,
            )
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("Ignoring unreadable node record %s", path)
            continue
        if not is_alive(node):
            remove_record(path)
        elif _valid_target(node):
            yield node


# ── Process scan (Linux) ─────────────────────────────────────────────────


def is_serve_argv(argv: Sequence[str]) -> bool:
    """True for ``leap hw serve ...`` and ``python -m leapflow.hardware.lhp ...``."""
    if not argv:
        return False
    parts = list(argv)
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(parts[0]).name):
        parts.pop(0)
        while parts and parts[0] in ("-I", "-u", "-B", "-s", "-S", "-E"):
            parts.pop(0)
        if parts[:2] == ["-m", "leapflow.hardware.lhp"]:
            return "--help" not in parts and "-h" not in parts
        if parts[:2] == ["-m", "leapflow"]:
            parts = parts[2:]
        elif parts and Path(parts[0]).name == "leap":
            parts = parts[1:]
        else:
            return False
    elif Path(parts[0]).name == "leap":
        parts = parts[1:]
    else:
        return False
    if parts[:1] == ["--mock-host"]:
        parts.pop(0)
    return parts[:2] == ["hw", "serve"] and "--help" not in parts and "-h" not in parts


def _flag(argv: Sequence[str], name: str) -> str:
    for index, part in enumerate(argv):
        if part == name and index + 1 < len(argv):
            return argv[index + 1]
        if part.startswith(name + "="):
            return part.split("=", 1)[1]
    return ""


def _endpoint_from_argv(argv: Sequence[str]) -> str:
    unix_path = _flag(argv, "--unix") or _flag(argv, "--unix-path")
    if unix_path:
        return f"unix://{unix_path}"
    host = _flag(argv, "--host") or "127.0.0.1"
    return f"tcp://{host}:{_flag(argv, '--port') or LHP_DEFAULT_PORT}"


def _process_environment(pid: int) -> dict[str, str]:
    # Only identity/path fields; credentials are never retained or printed.
    names = {NODE_ID_ENV, "LEAPFLOW_PROFILE", "LEAPFLOW_DATA_DIR", "LEAPFLOW_RUNTIME_DIR"}
    entries = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    result = {}
    for entry in entries:
        name, _, value = entry.partition(b"=")
        key = name.decode(errors="replace")
        if key in names:
            result[key] = value.decode(errors="replace").strip()
    return result


def _process_argv(pid: int) -> tuple[str, ...]:
    if Path("/proc/self/stat").exists():
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        if Path(f"/proc/{pid}").stat().st_uid != os.getuid():
            return ()
        return tuple(os.fsdecode(part) for part in raw.split(b"\0") if part)
    result = subprocess.run(
        ["ps", "-ww", "-o", "uid=,command=", "-p", str(pid)], capture_output=True,
        text=True, timeout=_SUBPROCESS_TIMEOUT_S, check=False,
    )
    fields = result.stdout.strip().split(maxsplit=1)
    if result.returncode or len(fields) != 2 or fields[0] != str(os.getuid()):
        return ()
    return tuple(shlex.split(fields[1]))


def _valid_target(node: NodeProcess) -> bool:
    if node.pid == os.getpid() or not is_alive(node):
        return False
    argv = _process_argv(node.pid)
    if not is_serve_argv(argv):
        return False
    return not node.argv or node.argv == argv


def _runtime_from_environment(env: dict[str, str]) -> Path:
    root = Path(env.get("LEAPFLOW_DATA_DIR") or "~/.leapflow").expanduser()
    profile = env.get("LEAPFLOW_PROFILE") or "default"
    from leapflow.layout import build_layout

    return Path(env.get("LEAPFLOW_RUNTIME_DIR") or build_layout(root).profile(profile).runtime_dir).expanduser()


def _scanned(runtime_dir: Path, proc_root: Path = Path("/proc")) -> Iterator[NodeProcess]:
    if not (proc_root / "self").exists():
        return
    uid, me = os.getuid(), os.getpid()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        argv = tuple(part.decode(errors="replace") for part in raw.split(b"\0") if part)
        if not is_serve_argv(argv):
            continue
        pid = int(entry.name)
        start = process_start(pid)
        if start is None:
            continue
        try:
            env = _process_environment(pid)
        except OSError:
            continue  # Unknown profile is not authority to stop a process.
        if _runtime_from_environment(env).resolve() != Path(runtime_dir).resolve():
            continue
        yield NodeProcess(
            pid=pid,
            node_id=_flag(argv, "--node-id") or env.get(NODE_ID_ENV, ""),
            start=start,
            endpoint=_endpoint_from_argv(argv),
            argv=argv,
            source=SOURCE_SCAN,
        )


def discover_nodes(runtime_dir: Path, node_id: str = "") -> list[NodeProcess]:
    """Live nodes on this machine, optionally only *node_id*; records win over the scan."""
    nodes = {node.pid: node for node in _recorded(runtime_dir)}
    for node in _scanned(runtime_dir):
        nodes.setdefault(node.pid, node)
    return sorted(
        (node for node in nodes.values() if not node_id or node.node_id == node_id),
        key=lambda node: node.pid,
    )


# ── Stop ─────────────────────────────────────────────────────────────────


def _send(node: NodeProcess, signum: int) -> None:
    """Signal *node* only while it is still the same process; raises ProcessLookupError if not."""
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        descriptor = os.pidfd_open(node.pid, 0)
        try:
            if not _valid_target(node):
                raise ProcessLookupError(node.pid)
            signal.pidfd_send_signal(descriptor, signum)
        finally:
            os.close(descriptor)
        return
    if not _valid_target(node):
        raise ProcessLookupError(node.pid)
    os.kill(node.pid, signum)


def _systemctl(*args: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", *args], capture_output=True, text=True,
        timeout=_SUBPROCESS_TIMEOUT_S, check=False,
    )
    if result.returncode:
        raise OSError("systemctl failed; inspect the owning unit and user manager")
    return result.stdout.strip()


def _unit_state(owner: Supervisor) -> dict[str, str]:
    output = _systemctl("show", "--property=MainPID,ActiveState,Result", "--", owner.unit)
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def _cleanup_receipt(node: NodeProcess) -> bool | None:
    if node.record_path is None:
        return None
    try:
        data = json.loads(node.record_path.read_text())
        if data.get("pid") == node.pid and data.get("start") == node.start:
            result = data.get("cleanup_complete")
            return result if type(result) is bool else None
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _wait_stopped(
    node: NodeProcess, children: tuple[tuple[int, str], ...], owner: Supervisor | None,
    deadline: float,
) -> tuple[bool, tuple[int, ...], dict[str, str]]:
    tracked = dict(children)
    while True:
        alive = is_alive(node)
        if alive:
            tracked.update(child_processes(node.pid))
        survivors = tuple(pid for pid, start in tracked.items() if process_start(pid) == start)
        state = _unit_state(owner) if owner is not None else {}
        unit_stopped = owner is None or (
            state.get("MainPID") == "0" and state.get("ActiveState") in ("inactive", "failed")
        )
        if not alive and not survivors and unit_stopped:
            return True, (), state
        if time.monotonic() >= deadline:
            return False, survivors, state
        time.sleep(min(_POLL_S, max(0.0, deadline - time.monotonic())))


def stop_node(node: NodeProcess, *, timeout_s: float = DEFAULT_STOP_TIMEOUT_S) -> StopResult:
    """Request graceful shutdown, then verify owner and worker exit; never force-kill."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        return StopResult(ok=False, status=STATUS_REFUSED, detail="--timeout must be positive and finite")
    via = ""
    deadline = time.monotonic() + timeout_s
    try:
        if not is_alive(node):
            return StopResult(ok=True, status=STATUS_NOT_RUNNING, node=node,
                              detail="The node had already exited; no signal sent.")
        if not _valid_target(node):
            return StopResult(ok=False, status=STATUS_REFUSED, node=node,
                              detail="Process identity no longer matches hw serve; no signal sent.")
        owner = supervisor_of(node.pid)
        unit = owner.unit if owner is not None else ""
        if owner is not None and owner.scope != SCOPE_USER:
            return StopResult(ok=False, status=STATUS_REFUSED, node=node,
                              detail=f"System unit {unit} owns this node; ask its administrator to stop it.")
        children = child_processes(node.pid)
        if owner is not None:
            state = _unit_state(owner)
            if state.get("MainPID") != str(node.pid) or not _valid_target(node):
                return StopResult(ok=False, status=STATUS_REFUSED, node=node,
                                  detail="The node is not the unit's main process; refusing to stop unrelated services.")
            via = f"systemd:{unit}"
            _systemctl("--no-block", "stop", "--", unit)
        else:
            via = "SIGTERM"
            try:
                _send(node, signal.SIGTERM)
            except ProcessLookupError:
                if is_alive(node):
                    raise
                # Parent exit raced delivery: still check its captured workers.
        logger.info("Requested LHP node stop: node=%s pid=%d via=%s", node.node_id, node.pid, via)
        exited, survivors, state = _wait_stopped(node, children, owner, deadline)
        forced = state.get("Result") in ("timeout", "signal", "core-dump")
        if not exited:
            return StopResult(
                ok=False, status=STATUS_TIMEOUT, node=node, via=via, forced=forced,
                detail="Node, workers, or supervisor still active. Check the node log and use a longer --timeout; no SIGKILL sent.",
                survivors=survivors, consequences=("Shutdown is incomplete; hardware state is unknown.",),
            )
        receipt = _cleanup_receipt(node)
        complete = receipt is True
        unclean = receipt is False or state.get("Result", "success") != "success"
        if node.record_path is not None:
            remove_record(node.record_path)
        return StopResult(
            ok=not unclean, status="stopped_uncleanly" if unclean else STATUS_STOPPED, node=node,
            via=via, forced=forced, cleanup_complete=complete,
            detail="Node and observed workers exited." if not unclean else "Node cleanup or supervisor reported an unclean stop; inspect hardware and logs.",
            consequences=stop_consequences(forced=forced, supervisor=unit, complete=complete),
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.error("LHP node stop could not be confirmed", exc_info=True)
        return StopResult(ok=False, status=STATUS_REFUSED, node=node, via=via,
                          detail=f"Stop refused or unconfirmed: {type(exc).__name__}. Inspect the node log and supervisor.")


def stop_consequences(*, forced: bool, supervisor: str = "", complete: bool = False) -> tuple[str, ...]:
    """Expected shutdown effects, explicitly separate from physical verification."""
    lines = [
        "Graceful cleanup completed." if complete and not forced else
        "Graceful cleanup is unconfirmed; inspect the node log. A process exit does not prove a safe hardware stop.",
        "Shutdown requests cancellation of runs and leases, attempts declared actuator halts, then releases camera previews/streams and device links.",
        "Piper: halt attempts a powered hold-in-place; close only disconnects CAN. No automatic disable, reset, parking move, or gripper-open command is sent.",
        "Service stop is NOT power-off. An enabled arm may remain energized and the gripper may retain its last command; posture, torque, and object retention after disconnect are NOT verified. Other devices follow their driver/firmware semantics.",
        "Node-side monitoring and lease deadman are no longer running; host remote control/estop through this node is unavailable. Keep hardware attended and use the device's physical safety procedures if necessary.",
        "LHP connections close; host fleet marks devices unreachable/offline according to its configured heartbeat. The host daemon/TUI and unrelated services are not stopped.",
    ]
    if forced:
        lines.insert(0, "WARNING: supervisor forced or abnormal termination; cleanup and hardware halt cannot be assumed. Motion may continue.")
    if supervisor:
        lines.append(f"User unit {supervisor} was stopped explicitly, preventing its restart policy from restarting this run. This does not disable boot activation; resume with `systemctl --user start {supervisor}`.")
    else:
        lines.append("Resume with the same `leap hw serve ...` command and profile; an external supervisor, if any, may have its own restart policy.")
    return tuple(lines)


__all__ = [
    "DEFAULT_STOP_TIMEOUT_S",
    "NodeProcess",
    "StopResult",
    "Supervisor",
    "discover_nodes",
    "is_serve_argv",
    "process_start",
    "remove_record",
    "finish_record",
    "stop_consequences",
    "stop_node",
    "supervisor_of",
    "write_record",
]
